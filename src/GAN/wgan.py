import image_parser as parse
# uncomment this for use on ssh
import matplotlib
matplotlib.use("ps")
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Input, merge, Embedding
import keras.backend as K
from keras.layers.merge import multiply
from keras.optimizers import Adam, RMSprop
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
dataset = 'cats'           # cats, dogs or celeb dataset
load = True                 # load saved model
image_dim  = 64             # image.shape = (3,image_dim, image_dim)

'''
Network parameters
'''
g_dropout_rate = 0.3        # dropout rate for generator
d_dropout_rate = 0.3        # dropout rate for discriminator
g_filters = 128             # number of filters of first layer of generator
d_filters = 128             # number of filters of  of discriminator
filtersize = 4              # size of Conv filters
dilation_rate = 1           # dilation factor for dilated Connvolutions
d_batchnorm = False         # use BatchNormalization in discriminator
# according to improved WGAN paper, BatchNorm shouldn't be applied for WGANs
out_dim = 1                 # output dimension for discriminator network
d_l2_regularisation = 5e-3  # l2 kernel_regularizer for discriminator
g_l2_regularisation = 5e-3  # l2 kernel_regularizer for generator
deconv = True               # using strided Conv2DTranspose for Upsampling, else UpSampling2D
opt_d = RMSprop(5e-5)#Adam(lr = 1e-4, beta_1 = 0.5)
opt_g = RMSprop(5e-5)#Adam(lr = 2e-4, beta_1 = 0.5)
                            # optimizers
clamp_weights = False
clamp_lower, clamp_upper = -0.5, 0.5

'''
Training parameters
'''

pretrain_eps = 0            # number of pretraining epochs for discriminator
num_of_imgs = 12450          # number of images used for training
batch_size = 128            # batch size for training
iterations = 1000           # number of iterations of training process

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


images_train = parse.load_data_set(dataset,image_dim = image_dim)
images_train = ((images_train - 127.5 ) / 127.5)[:num_of_imgs]

loss = wasserstein
acc = binary_accuracy_

disc_name = dataset + "w_" + util.get_model_name(discriminator = True, filters = d_filters, dropout_rate = d_dropout_rate,dilation_rate = dilation_rate, batch_norm = d_batchnorm, out_dim = out_dim, filtersize = filtersize)
gen_name = dataset +"w_" + util.get_model_name(discriminator = False, deconv = deconv, filters = g_filters, dropout_rate = g_dropout_rate, dilation_rate = dilation_rate,out_dim = out_dim, filtersize = filtersize)


if load and isfile("./networks/" + gen_name):
    generator = load_model("./networks/" + gen_name , custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})
    discriminator = load_model("./networks/"+ disc_name, custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})

else:
    if load and not isfile("./networks/" + gen_name):
        print("couldn't find model. New model will be generated.")

    discriminator = get_discriminator(input_dim = image_dim, filters = d_filters, filtersize = filtersize ,regularisation = d_l2_regularisation, dropout_rate = d_dropout_rate, batch_norm = True, out_dim = out_dim, dilation_rate = dilation_rate, wasserstein = True)
    discriminator.compile(loss=loss, optimizer=opt_d)

    if deconv:
        generator = get_deconv_generator(filters = g_filters, image_dim = image_dim, filtersize = filtersize, regularisation = g_l2_regularisation, dropout_rate = g_dropout_rate, dilation_rate=dilation_rate)
    else:
        generator = get_upSampling_generator(image_dim = image_dim, filters = g_filters, regularisation = g_l2_regularisation, dropout_rate = g_dropout_rate)

'''
create generative adversarial network:
discriminator stacked on generator, which gets 100 random numbers as input
'''
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
for l in discriminator.layers:
    l.trainable = False
GAN = Model(gan_input, gan_V)

GAN.compile(loss= loss, optimizer =opt_g)
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

fig1, ax1 = plt.subplots(1,1)

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
##################################################################
                        Training
##################################################################
'''
losses = {"d":[], "g":[] }

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


    for i in range(batches):
        progress_bar.update(i)

        noise = np.random.normal(mu, sigma, size=[batch_size, 100])

        images_fake = generator.predict(noise)

        images_batch = images_train[i*batch_size:(i+1)*batch_size,:,:,:]

        if add_noise:
            im_noise = np.exp( - it/10) * np.random.normal(0,0.3,size = images_batch.shape )
            images_batch += im_noise

        for i in range(disc_train_eps):
            d_loss1 = discriminator.train_on_batch(images_batch,-np.ones(batch_size))
            d_loss2 = discriminator.train_on_batch(images_fake, np.ones(batch_size))

        '''
            weight_clipping
        '''
        if clamp_weights:
            for l in discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, clamp_lower, clamp_upper) for w in weights]
                l.set_weights(weights)

        d_loss = 0.5 * (d_loss1 + d_loss2)#(np.array(d_loss1)+np.array(d_loss2))* 0.5

        noise_tr = np.random.normal(mu, sigma, size=[batch_size, 100])
        for i in range(gen_train_epochs):
            g_loss = GAN.train_on_batch(noise_tr, -np.ones(batch_size))

        losses["g"].append(g_loss)

        losses["d"].append(d_loss)

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

        losses_plot_g = np.array(losses['g'])
        losses_plot_d = np.array(losses['d'])
        images_plot = generator.predict(plot_noise)
        images_plot = np.swapaxes(images_plot, 1,2)
        images_plot = (np.swapaxes(images_plot, 3,2) + 1) * 127.5

        for i in range(20):
            ax2[i].cla()
            ax2[i].imshow(images_plot[i].astype(np.uint8) )
            ax2[i].axis('off')

        ax1.cla()
        ax1.plot(losses_plot_d, label='discriminitive loss')
        ax1.plot(losses_plot_g, label='generative loss')
        ax1.legend()

        if save_images:
            fig2.savefig(img_folder + dataset + str(it) + ".png")
            fig1.savefig(img_folder + "hist.png")
        plt.pause(0.0000001)

	# save model every 1000 iterations
    #if np.mod(it+1, 25) == 0:
    #	discriminator.save("/output/"+"cifar10" + disc_name)
    #	generator.save("/output/"+ "cifar10"+gen_name)

plt.show()
