import image_parser as parse
# uncomment this for use on ssh
import matplotlib
matplotlib.use("ps")
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Input, merge, Embedding
from keras import backend as K
from keras.layers.merge import multiply
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from networks import *
import utilities as util
from os.path import isfile
K.set_image_data_format('channels_first') # using theano dimension ordering
import tensorflow as tf
from keras.datasets import cifar10


'''
Hyperparameters:
'''
'''
general parameters:
'''
dataset = 'cifar'           # cats, dogs or celeb dataset
load = True                 # load saved model
image_dim  = 32             # image.shape = (3,image_dim, image_dim)

'''
Network parameters
'''
g_dropout_rate = 0.3        # dropout rate for generator
d_dropout_rate = 0.3        # dropout rate for discriminator
g_filters = 256             # number of filters of first layer of generator
d_filters = 256             # number of filters of  of discriminator
filtersize = 4              # size of Conv filters
dilation_rate = 1           # dilation factor for dilated Connvolutions
d_batchnorm = True          # use BatchNormalization in discriminator
out_dim = 1                 # output dimension for discriminator network
d_l2_regularisation = 5e-3  # l2 kernel_regularizer for discriminator
g_l2_regularisation = 5e-3  # l2 kernel_regularizer for generator
deconv = True               # using strided Conv2DTranspose for Upsampling, else UpSampling2D
opt_d = Adam(lr = 2e-4, beta_1 = 0.5)
opt_g = Adam(lr = 2e-4, beta_1 = 0.5)
                            # optimizers

'''
Training parameters
'''

pretrain_eps = 0            # number of pretraining epochs for discriminator
num_of_imgs = 32768         # number of images used for training
batch_size = 128            # batch size for training
iterations = 2500           # number of iterations of training process
soft_labels = True          # using labels in range around 0/1 insteadof binary labels -> improves stability

one_sided_sl = True        # using label smoothing only for real samples
disc_train_eps = 1          # how often each network will be
gen_train_epochs = 1        # trained in one epoch
add_noise = False            # add noise to images, with decreasing magnitude
mu = 0                      # mean for noise input to generator
sigma = 0.5                 # stddev for noise input

'''
plotting parameters
'''
refresh_interval = 5        # interval for refreshing of plots
save_images = True          # save generated images after every 50th iteration
img_folder = "./output/"    # "C:/Users/Philip/OneDrive/Dokumente/gan_imgs/"
                            # folder where images will be saved in
plot_weights = False        # plot weights of some conv layers during training


'''
deconv_generator:
    fully convolutional gereator network, similar to DCGAN-Architecture
 arguments:
    filters - number of filters for conv layers
'''

def get_deconv_generator(filters = 1024, filtersize = 5,regularisation = 1e-2, dropout_rate =0.5, dilation_rate = 1 ,image_dim = 32 ):

    generator = Sequential()
    reg =  l2(regularisation)

    '''
	Project and Reshape: "just a matrix multiplication" according to DCGAN paper
	'''

    generator.add(  Dense(4*4*filters,  input_shape = [100]))
    generator.add(  BatchNormalization())
    generator.add(  Activation('relu'))
    generator.add(  Reshape((filters, 4 , 4 )))
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

    generator.add(BatchNormalization())
    #generator.add(  LeakyReLU())
    generator.add(  Activation('relu'))

    generator.add(  Dropout(dropout_rate))

    generator.add(Conv2DTranspose(3,
    						kernel_size=(filtersize, filtersize),
    						strides = (2,2),
    						dilation_rate = dilation_rate,
    						padding='same',
    						activation='tanh',
    						kernel_regularizer = reg))

    latent_vector = Input(shape=(100, ))
    image_class = Input(shape= (1,))

    class_embedding = Embedding(10,100)(image_class)
    class_embedding = Flatten()(class_embedding)
    h = multiply([latent_vector, class_embedding])

    fake_im = generator(h)

    model = Model(input = [latent_vector, image_class], output = fake_im)

    return model#Model(input = [latent, image_class], output = fake_im)

def get_discriminator(input_dim = 32, depth = 1,  filters = 256,filtersize = 5, regularisation = 1e-4, dropout_rate = 0.5, dilation_rate = 1,batch_norm = False, out_dim = 2):

    reg = l2(regularisation)

    discriminator = Sequential()
    discriminator.add(  Conv2D(int(filters/8),
    					(filtersize, filtersize),
    					strides = (2,2),
    					padding='same',
    					dilation_rate = dilation_rate,
    					input_shape = (3,input_dim,input_dim),
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
    					kernel_regularizer = reg))
    # 4 x 4
    if batch_norm:
    	discriminator.add(  BatchNormalization())
    discriminator.add(  LeakyReLU())
    discriminator.add(  Dropout(dropout_rate))

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
    discriminator.add(  Flatten())

    '''
    determine if image is real, and in which class it belongs
    '''

    img = Input(shape = (3, 32, 32))
    features = discriminator(img)

    fake = Dense(1, activation='sigmoid')(features)
    aux = Dense(10, activation='softmax')(features)

    model = Model(input = img, output = [fake, aux])

    return model


(images_train, y_train), (images_test, y_test) = cifar10.load_data()
images_train = ((images_train - 127.5 ) / 127.5)[:num_of_imgs]
y_train = y_train[:num_of_imgs]

'''
for i in range(10):
    in_class = np.sum(y_train == i)
    print("in class ", i , " samples: ", in_class)
'''

if out_dim == 1:
    loss = 'binary_crossentropy'
    acc = binary_accuracy_

else:
    loss = 'categorical_crossentropy'
    acc = 'categorical_accuracy'


disc_name ="cifar10" + util.get_model_name(dataset = dataset, discriminator = True, filters = d_filters, dropout_rate = d_dropout_rate,dilation_rate = dilation_rate, batch_norm = d_batchnorm, out_dim = out_dim, filtersize = filtersize)
gen_name = "cifar10" + util.get_model_name(dataset = dataset, discriminator = False, deconv = deconv, filters = g_filters, dropout_rate = g_dropout_rate, dilation_rate = dilation_rate,out_dim = out_dim, filtersize = filtersize)


if load and isfile("./networks/" + gen_name):
    generator = load_model("./networks/" + gen_name , custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})
    discriminator = load_model("./networks/"+ disc_name, custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})

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


for l in discriminator.layers:
    l.trainable = True

'''
initialize matplotlib windows
'''

fig1, ax1 = plt.subplots(2,2)
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

if soft_labels:
    one = np.random.uniform(0.7,1.0, size = batch_size)
    if one_sided_sl:
        zero = np.zeros(batch_size)
    else:
        zero = np.random.uniform(0, 0.3, size = batch_size)
else:
    one = np.ones(batch_size)
    zero = np.zeros(batch_size)

real_2d = np.zeros((batch_size,2))
real_2d[:,0] = one
real_2d[:,1] = zero

fake_2d = np.zeros((batch_size,2))
fake_2d[:,0] = zero
fake_2d[:,1] = one

if out_dim == 1:
    y_r = one
    y_f = zero
    y_g = np.concatenate([y_r,y_r])

if out_dim == 2:
    y_r = real_2d
    y_f = fake_2d
    y_g = np.concatenate([y_r, y_r])

'''
##################################################################
                        Training
##################################################################
'''
class_names = {'0':"airplane", '1':"car", '2':"bird", '3':"cat", '4':"deer", '5':"dog", '6':"frog", '7':"horse", '8':"ship", '9':"truck"}
losses = {"d":[],"d_c": [] ,  "g":[], "g_c":[] }
accuracies = {"d":[],"d_c": [] ,  "g":[], "g_c":[] }

plot_noise = np.random.normal(mu, sigma, size=[batch_size, 100])
plot_labels = range(batch_size)
plot_labels = np.mod(plot_labels, 10)

for it in range(iterations):

    print("iteration ", it+1, " of " , iterations)

    '''
    first train discriminator:
    	- create fake images with generator, concatenate with real images
    	- fit discriminator on those samples
    '''
    batches = int(num_of_imgs/batch_size)

    progress_bar = Progbar(target=batches)
    '''
    shuffle images and labels the same way:
    '''
    p = np.random.permutation(num_of_imgs)
    images_train = images_train[p]
    y_train = y_train[p]
    #np.random.shuffle(images_train)

    for i in range(batches):
        progress_bar.update(i)

        noise = np.random.normal(mu, sigma, size=[batch_size, 100])
        sampled_labels = np.random.randint(0, 10, (1*batch_size,1))
        images_fake = generator.predict([noise,sampled_labels])

        images_batch = images_train[i*batch_size:(i+1)*batch_size,:,:,:]
        label_batch = y_train[i * batch_size:(i+ 1) * batch_size]

        if add_noise:
            im_noise = np.exp( - it/10) * np.random.normal(0,0.3,size = images_batch.shape )
            images_batch += im_noise

        lbls = np.concatenate((label_batch, sampled_labels[:batch_size]), axis=0)
        aux_y = np.zeros((2*batch_size, 10))

        for i in range (2*batch_size):
            aux_y[i, lbls[i]] = 1

        for i in range(disc_train_eps):
            d_loss1 = discriminator.train_on_batch(images_batch,[y_r, aux_y[:batch_size]])
            d_loss2 = discriminator.train_on_batch(images_fake, [y_f, aux_y[batch_size:]])

        d_loss = (np.array(d_loss1)+np.array(d_loss2))* 0.5

        noise_tr = np.random.normal(mu, sigma, size=[batch_size, 100])
        for i in range(gen_train_epochs):
            g_loss = GAN.train_on_batch([noise_tr,sampled_labels], [y_g, aux_y])
        losses["g"].append(g_loss[0])
        accuracies["g"].append(g_loss[2])
        losses["g_c"].append(g_loss[1])
        accuracies["g_c"].append(g_loss[3])
        losses["d"].append(d_loss[0])
        accuracies["d"].append(d_loss[2])
        losses["d_c"].append(d_loss[1])
        accuracies["d_c"].append(d_loss[3])

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
        acc_plot_g = np.array(accuracies['g'])
        acc_plot_d = np.array(accuracies['d'])

        losses_plot_g_c = np.array(losses['g_c'])
        losses_plot_d_c = np.array(losses['d_c'])
        acc_plot_g_c = np.array(accuracies['g_c'])
        acc_plot_d_c = np.array(accuracies['d_c'])
        images_plot = generator.predict([plot_noise,plot_labels])

        images_plot = np.swapaxes(images_plot, 1,2)
        images_plot = (np.swapaxes(images_plot, 3,2) + 1) * 127.5

        for i in range(20):
            ax2[i].cla()
            ax2[i].set_title(class_names[ str(plot_labels[i])])
            ax2[i].imshow(images_plot[i].astype(np.uint8) )
            ax2[i].axis('off')
        ax1[0].cla()
        ax1[0].plot(acc_plot_d, label='discriminitive accuracy')
        ax1[0].plot(acc_plot_g, label='generative accuracy')
        ax1[0].legend()

        ax1[1].cla()
        ax1[1].plot(losses_plot_d, label='discriminitive loss')
        ax1[1].plot(losses_plot_g, label='generative loss')
        ax1[1].legend()

        ax1[2].cla()
        ax1[2].plot(acc_plot_d_c, label='discriminitive categorical accuracy')
        ax1[2].plot(acc_plot_g_c, label='generative categorical accuracy')
        ax1[2].legend()

        ax1[3].cla()
        ax1[3].plot(losses_plot_d_c, label='discriminitive categorical loss')
        ax1[3].plot(losses_plot_g_c, label='generative categorical loss')
        ax1[3].legend()
        if save_images:
            fig2.savefig(img_folder + dataset + str(it) + ".png")
            fig1.savefig(img_folder + "hist.png")
        plt.pause(0.0000001)

	# save model every 1000 iterations
    if np.mod(it+1, 25) == 0:
    	discriminator.save("./output/"+"cifar10" + disc_name)
    	generator.save("./output/"+ "cifar10"+gen_name)

plt.show()
