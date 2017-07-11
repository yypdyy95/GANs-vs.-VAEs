import image_parser as parse
# uncomment this for use on ssh
import matplotlib
matplotlib.use("ps")
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Input, merge, Embedding
import keras.backend as K
from keras.constraints import maxnorm
from keras.layers.merge import multiply
from keras.optimizers import Adam, RMSprop
from keras.utils.generic_utils import Progbar
from networks import *
import utilities as util
from os.path import isfile
K.set_image_data_format('channels_first') # using theano dimension ordering
import tensorflow as tf
from keras.datasets import cifar10, mnist
from scipy.signal import medfilt

'''
Hyperparameters:
'''
'''
general parameters:
'''
dataset = 'mnist'           # cats, dogs or celeb dataset
load = False                # load saved model
image_dim  = 28             # image.shape = (3,image_dim, image_dim)

'''
Network parameters
'''
g_dropout_rate = 0.         # dropout rate for generator
d_dropout_rate = 0.         # dropout rate for discriminator
g_filters = 64             # number of filters of first layer of generator
d_filters = 128             # number of filters of  of discriminator
filtersize = 4              # size of Conv filters
dilation_rate = 1           # dilation factor for dilated Connvolutions
d_batchnorm = False          # use BatchNormalization in discriminator
out_dim = 1                 # output dimension for discriminator network
d_l2_regularisation = 0*5e-3  # l2 kernel_regularizer for discriminator
g_l2_regularisation = 0*5e-3  # l2 kernel_regularizer for generator
deconv = True               # using strided Conv2DTranspose for Upsampling, else UpSampling2D
opt_d = RMSprop(5e-5)#Adam(lr = 1e-4, beta_1 = 0.5)
opt_g = RMSprop(5e-5)#Adam(lr = 2e-4, beta_1 = 0.5)
                            # optimizers

'''
Training parameters
'''

pretrain_eps = 0            # number of pretraining epochs for discriminator
num_of_imgs = 50000         # number of images used for training
batch_size = 256             # batch size for training
iterations = 1000           # number of iterations of training process
soft_labels = True          # using labels in range around 0/1 insteadof binary labels -> improves stability

one_sided_sl = True        # using label smoothing only for real samples
disc_train_eps = 5          # how often each network will be
gen_train_epochs = 1        # trained in one epoch
add_noise = False            # add noise to images, with decreasing magnitude
mu = 0                      # mean for noise input to generator
sigma = 0.5                 # stddev for noise input

'''
plotting parameters
'''
refresh_interval = 1        # interval for refreshing of plots
save_images = True          # save generated images after every 50th iteration
img_folder = "/output/"    # "C:/Users/Philip/OneDrive/Dokumente/gan_imgs/"
                            # folder where images will be saved in
plot_weights = False        # plot weights of some conv layers during training
clamp_lower, clamp_upper = -0.5, 0.5

def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

'''
deconv_generator:
    fully convolutional gereator network, similar to DCGAN-Architecture
 arguments:
    filters - number of filters for conv layers
'''

def get_deconv_generator(filters = 1024, filtersize = 5,regularisation = 0, dropout_rate =0.5, dilation_rate = 1 ,image_dim = 32 ):

    generator = Sequential()
    reg =  l2(regularisation)

    '''
	Project and Reshape: "just a matrix multiplication" according to DCGAN paper
	'''

    generator.add(  Dense(7*7*filters,  input_shape = [100]))
    generator.add(  BatchNormalization())
    generator.add(  Activation('relu'))
    generator.add(  Reshape((filters, 7 , 7 )))
    generator.add(Conv2DTranspose(int(filters/2),
    						  kernel_size=(filtersize, filtersize),
    						  strides=(2, 2),
    						  dilation_rate = dilation_rate,
    						  padding='same',
    						  kernel_regularizer = reg))
    generator.add(BatchNormalization())
    generator.add(  Activation('relu'))
    generator.add(  Dropout(dropout_rate))

    generator.add(Conv2DTranspose(int(filters/4),
    						  kernel_size=(filtersize, filtersize),
    						  strides=(2, 2),
    						  dilation_rate = dilation_rate,
    						  padding='same',
    						  kernel_regularizer = reg))

    generator.add(  BatchNormalization())
    generator.add(  Activation('relu'))
    generator.add(  Dropout(dropout_rate))


    generator.add(Conv2DTranspose(1,
    						kernel_size=(filtersize, filtersize),
    						strides = (1,1),
    						dilation_rate = dilation_rate,
    						padding='same',
                            activation = 'tanh',
    						kernel_regularizer = reg))


    print(generator.output_shape)
    latent_vector = Input(shape=(100, ))
    image_class = Input(shape= (1,))

    class_embedding = Embedding(10,100)(image_class)
    class_embedding = Flatten()(class_embedding)
    h = multiply([latent_vector, class_embedding])

    fake_im = generator(h)

    model = Model(input = [latent_vector, image_class], output = fake_im)

    return model

def get_discriminator(input_dim = 28, depth = 1,  filters = 256,filtersize = 5, regularisation = 0, dropout_rate = 0.5, dilation_rate = 1,batch_norm = False, out_dim = 2):

    reg = l2(regularisation)
    kernel_constraint=maxnorm(0.1)
    discriminator = Sequential()
    discriminator.add(  Conv2D(int(filters/8),
    					(filtersize, filtersize),
    					strides = (2,2),
    					padding='same',
    					dilation_rate = dilation_rate,
    					input_shape = (1,input_dim,input_dim),
    					kernel_regularizer = reg))
    # 16x 16
    if batch_norm:
    	discriminator.add(  BatchNormalization())
    discriminator.add(  LeakyReLU())
    discriminator.add(  Dropout(dropout_rate))

    discriminator.add(  Conv2D(int(filters/4),
    					(filtersize, filtersize),
    					strides = (2,2),
    					dilation_rate = dilation_rate,
    					padding='same',
             # kernel_constraint = kernel_constraint,
    					kernel_regularizer = reg))
    # 8 x 8

    if batch_norm:
    	discriminator.add(  BatchNormalization())
    discriminator.add(  LeakyReLU())
    discriminator.add(  Dropout(dropout_rate))

    discriminator.add(  Conv2D(int(filters/2),
    					(filtersize, filtersize),
    					strides = (2,2),
    					dilation_rate = dilation_rate,
    					padding='same',

             # kernel_constraint = kernel_constraint,
    					kernel_regularizer = reg))
    # 4 x 4
    if batch_norm:
    	discriminator.add(  BatchNormalization())
    discriminator.add(  LeakyReLU())
    discriminator.add(  Dropout(dropout_rate))
    '''
    discriminator.add(  Conv2D(filters,
    					(filtersize, filtersize),
    					strides = (2,2),
    					padding='same',
    					dilation_rate = dilation_rate,
    					kernel_regularizer = reg))
    # 2 x 2
    if batch_norm:
    	discriminator.add(  BatchNormalization())
    discriminator.add(  LeakyReLU())
    discriminator.add(  Dropout(dropout_rate))
    '''
    discriminator.add(  Flatten())

    '''
    determine if image is real, and in which class it belongs
    '''
    img = Input(shape = (1, image_dim, image_dim))

    features = discriminator(img)

    fake = Dense(1)(features)
    aux = Dense(10, activation='softmax')(features)

    model = Model(input = img, output = [fake, aux])

    return model


(images_train, y_train), (images_test, y_test) = mnist.load_data()
print(images_train.shape)
print(y_train.shape)
images_train = ((images_train - 127.5 ) / 127.5)[:num_of_imgs].reshape(-1,1,28,28)
y_train = y_train[:num_of_imgs].reshape(-1,1)


loss = wasserstein
acc = binary_accuracy_

disc_name =util.get_model_name(discriminator = True, filters = d_filters, dropout_rate = d_dropout_rate,dilation_rate = dilation_rate, batch_norm = d_batchnorm, out_dim = out_dim, filtersize = filtersize)
gen_name = util.get_model_name(discriminator = False, deconv = deconv, filters = g_filters, dropout_rate = g_dropout_rate, dilation_rate = dilation_rate,out_dim = out_dim, filtersize = filtersize)


if load and isfile("./example_networks/" + gen_name):
    generator = load_model("./example_networks/" + gen_name , custom_objects = {'binary_accuracy_':binary_accuracy_, 'wasserstein':wasserstein})
    discriminator = load_model("./example_networks/"+ disc_name, custom_objects = {'binary_accuracy_':binary_accuracy_, 'wasserstein':wasserstein})
    print("loaded networks")
else:
    if load and not isfile("./networks/" + gen_name):
        print("couldn't find model. New model will be generated.")

    discriminator = get_discriminator(input_dim = image_dim, filters = d_filters, filtersize = filtersize ,regularisation = d_l2_regularisation, dropout_rate = d_dropout_rate, batch_norm = True, out_dim = out_dim, dilation_rate = dilation_rate)
    discriminator.compile(loss=[loss, 'categorical_crossentropy'],metrics=[acc, 'categorical_accuracy'], optimizer=opt_d)

    if deconv:
        generator = get_deconv_generator(filters = g_filters, image_dim = image_dim, filtersize = filtersize, regularisation = g_l2_regularisation, dropout_rate = g_dropout_rate, dilation_rate=dilation_rate)
    else:
        generator = get_upSampling_generator(image_dim = image_dim, filters = g_filters, regularisation = g_l2_regularisation, dropout_rate = g_dropout_rate)

'''
create generative adversarial network:
discriminator stacked on generator, which gets 100 random numbers as input
'''

latent = Input(shape=(100, ))
image_class = Input(shape=(1,), dtype='int32')

H = generator([latent, image_class])
fake, aux = discriminator(H)
for l in discriminator.layers:
    l.trainable = False
GAN = Model([latent, image_class], [fake, aux])

GAN.compile(loss= [loss, 'categorical_crossentropy'], metrics=[acc, 'categorical_accuracy'], optimizer =opt_g)
print("\n\n GENERATOR: \n\n")
generator.summary()
print("\n\n DISCRIMINATOR: \n\n")
discriminator.summary()
print(discriminator.metrics_names)

for l in discriminator.layers:
    l.trainable = True

'''
initialize matplotlib windows
'''

fig1, ax1 = plt.subplots(2,1)
ax1 = ax1.reshape(-1)
fig2, ax2 = plt.subplots(4,5)
ax2 = ax2.reshape(-1)
fig2.set_size_inches(10,9)
fig1.set_size_inches(10,8)
plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.05, hspace=0.05)
if plot_weights:
    fig3, ax3 = plt.subplots(2,1)
    ax3 = ax3.reshape(-1)
    fig3.set_size_inches(7,8)
    fig3.tight_layout()


'''
fuzzy labeling: use labels somewhere in the range of 1 or 0 instead
of strict labeling
'''


'''
##################################################################
                        Training
##################################################################
'''
if dataset == 'cifar':
    class_names = {'0':"airplane", '1':"car", '2':"bird", '3':"cat", '4':"deer", '5':"dog", '6':"frog", '7':"horse", '8':"ship", '9':"truck"}
losses = {"d":[],"d_c": [] ,  "g":[], "g_c":[] }
accuracies = {"d":[],"d_c": [] ,  "g":[], "g_c":[] }

plot_noise = np.random.normal(mu, sigma, size=[batch_size, 100])
plot_labels = range(batch_size)
plot_labels = np.mod(plot_labels, 10)

for it in range(iterations):

    print("\niteration ", it+1, " of " , iterations)

    '''
    first train discriminator:
    	- create fake images with generator, concatenate with real images
    	- fit discriminator on those samples
    '''
    batches = int(num_of_imgs/batch_size)

    progress_bar = Progbar(target=batches)

    p = np.random.permutation(num_of_imgs)
    images_train = images_train[p]
    y_train = y_train[p]

    for i in range(batches):
        progress_bar.update(i)

        noise = np.random.normal(mu, sigma, size=[batch_size, 100])
        sampled_labels = np.random.randint(0, 10, (batch_size,1))
        images_fake = generator.predict([noise,sampled_labels])

        images_batch = images_train[i*batch_size:(i+1)*batch_size,:,:,:]
        label_batch = y_train[i * batch_size:(i+ 1) * batch_size]

        if add_noise:
            im_noise = np.exp( - it/10) * np.random.normal(0,0.3,size = images_batch.shape )
            images_batch += im_noise

        lbls = np.concatenate((label_batch, sampled_labels), axis=0)
        aux_y = np.zeros((2*batch_size, 10))

        for i in range (2*batch_size):
            aux_y[i, lbls[i]] = 1
        for i in range(disc_train_eps):
            d_loss1 = discriminator.train_on_batch(images_batch,[-np.ones(batch_size), aux_y[:batch_size]])
            d_loss2 = discriminator.train_on_batch(images_fake, [np.ones(batch_size), aux_y[batch_size:]])
        w_est = d_loss1[1] + d_loss2[1]
        d_loss = 0.5 * (np.array(d_loss1) + np.array(d_loss2))
        '''
            weight_clipping
        '''
        for l in discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, clamp_lower, clamp_upper) for w in weights]
            l.set_weights(weights)


        noise_tr = np.random.normal(mu, sigma, size=[batch_size, 100])
        for i in range(gen_train_epochs):
            g_loss = GAN.train_on_batch([noise_tr,sampled_labels], [-np.ones(batch_size), aux_y[batch_size:]])

        losses["g"].append(g_loss[1])
        losses["g_c"].append(g_loss[2])
        accuracies["g_c"].append(g_loss[6])
        losses["d"].append(w_est)
        losses["d_c"].append(d_loss[2])
        accuracies["d_c"].append(d_loss[6])

    #pred = discriminator.predict(images_batch)
    # uncomment the following to check if training process runs corectly:
    #print(np.max(GAN.layers[2].layers[1].get_weights()[0] - discriminator.layers[1].get_weights()[0]))
    #print(np.max(GAN.layers[1].layers[1].get_weights()[0] - generator.layers[1].get_weights()[0]))
    if np.mod(it+1, refresh_interval) == 0:

        if plot_weights:

            ax3[0].cla()
            ax3[0].set_title("discriminator weights")
            d_dists = util.get_weight_distributions(discriminator, bins = 100)
            for i in range(d_dists.shape[0]):
            	ax3[0].plot(d_dists[i][1][:-1], d_dists[i][0], label = "layer " + str(i) )
            ax3[0].legend()

            ax3[1].cla()
            ax3[1].set_title("generator weights")
            g_dists = util.get_weight_distributions(generator, bins = 100)
            for i in range(g_dists.shape[0]):
            	ax3[1].plot(g_dists[i][1][:-1], g_dists[i][0], label = "layer " + str(i) )
            ax3[1].legend()
            #ax3[1].plot(g_dist[1][:-1], g_dist[0])

        losses_plot_g = np.array(losses['g'])
        losses_plot_d = np.array(losses['d'])

        losses_plot_g_c = np.array(losses['g_c'])
        losses_plot_d_c = np.array(losses['d_c'])
        acc_plot_g_c = np.array(accuracies['g_c'])
        acc_plot_d_c = np.array(accuracies['d_c'])
        images_plot = generator.predict([plot_noise,plot_labels])

        images_plot = images_plot.reshape(-1, 28,28)
        images_plot = (images_plot + 1) * 127
        for i in range(20):
            ax2[i].cla()
            if dataset == 'cifar':
                ax2[i].set_title(class_names[ str(plot_labels[i])])
            ax2[i].imshow(images_plot[i].astype(np.uint8), cmap = 'Greys' )
            ax2[i].axis('off')

        ax1[0].cla()
        ax1[0].plot(medfilt(-losses_plot_d, 101), label='wasserstein estimate')
        ax1[0].legend()


        ax1[1].cla()
        ax1[1].plot(losses_plot_d_c, label='discriminitive categorical loss')
        ax1[1].plot(losses_plot_g_c, label='generative categorical loss')
        ax1[1].legend()

        if save_images:
            fig2.savefig(img_folder + dataset + str(it) + ".png")
            fig1.savefig(img_folder + "hist.png")
        plt.pause(0.0000001)

	# save model every 1000 iterations
    if np.mod(it+1, 25) == 0:
    	discriminator.save("/output/"+ disc_name)
    	generator.save("/output/"+ gen_name)

plt.show()
