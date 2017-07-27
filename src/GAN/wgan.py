import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import image_parser as parse
# uncomment this for use on ssh
#import matplotlib
#matplotlib.use("ps")
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
import tensorflow as tf
from keras.datasets import cifar10
import argparse
from scipy.signal import medfilt

'''
Hyperparameters:
'''


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, help='cifar10 | celeb | char74k | cats | dogs ')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_dim', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--num_of_imgs', type=int, default=100000)
parser.add_argument('--channels', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--g_filters', type=int, default=64)
parser.add_argument('--d_filters', type=int, default=64)
parser.add_argument('--filtersize', type=int, default=4)
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr_D', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lr_G', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=3, help='number of D iters per each G iter, default = 3')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not in Discriminator')
parser.add_argument('--d_dropout', type = float, default = 0., help='dropout rate for discriminator')
parser.add_argument('--g_dropout', type = float, default = 0., help='dropout rate for generator')
parser.add_argument('--deconv', type=bool, default=True)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--output_folder',  default="./output/")
opt = parser.parse_args()


'''
general parameters:
'''

d_batchnorm = False         # use BatchNormalization in discriminator
# according to improved WGAN paper, BatchNorm shouldn't be applied for WGANs

'''
Training parameters
'''


add_noise = False           # add noise to images, with decreasing magnitude
mu = 0                      # mean for noise input to generator
sigma = 0.5                 # stddev for noise input

'''
plotting parameters
'''
refresh_interval = 5        # interval for refreshing of plots
save_images = True          # save generated images after every 50th iteration

plot_weights = False        # plot weights of some conv layers during training




opt_d = RMSprop(opt.lr_D)       #Adam(lr = 1e-4, beta_1 = 0.5)
opt_g = RMSprop(opt.lr_G)       #Adam(lr = 2e-4, beta_1 = 0.5)
                            # optimizers


if opt.dataset == 'cifar10':
  (images_train, classes) , (_,_) = cifar10.load_data()
  opt.image_dim = 32
else:
  images_train , classes = parse.load_data_set_classes(opt.dataset,image_dim = opt.image_dim)[:opt.num_of_imgs]

images_train = ((images_train - 127.5 ) / 127.5)
if images_train.shape[0] < opt.num_of_imgs:
  opt.num_of_imgs = images_train.shape[0]

loss = wasserstein

disc_name = opt.dataset + "w_" + util.get_model_name(discriminator = True, filters = opt.d_filters, dropout_rate = opt.d_dropout, batch_norm = d_batchnorm,  filtersize = opt.filtersize)
gen_name = opt.dataset +"w_" + util.get_model_name(discriminator = False, deconv = opt.deconv, filters = opt.g_filters, dropout_rate = opt.g_dropout, filtersize = opt.filtersize)


if opt.load_model and isfile("./networks/" + gen_name):
    generator = load_model("./networks/" + gen_name , custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})
    discriminator = load_model("./networks/"+ disc_name, custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})

else:
    if opt.load_model and not isfile("./networks/" + gen_name):
        print("couldn't find model. New model will be generated.")

    discriminator = get_discriminator(input_dim = opt.image_dim, filters = opt.d_filters, filtersize = opt.filtersize ,regularisation = 0., dropout_rate = opt.d_dropout, batch_norm = True,  wasserstein = True)
    discriminator.compile(loss=loss, optimizer=opt_d)

    if opt.deconv:
        generator = get_deconv_generator(filters = opt.g_filters, image_dim = opt.image_dim, filtersize = opt.filtersize, regularisation = 0., dropout_rate = opt.g_dropout)
    else:
        generator = get_upSampling_generator(image_dim = opt.image_dim, filters = opt.g_filters, regularisation = 0., dropout_rate = opt.g_dropout)

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
plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0., hspace=0.)
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

plot_noise = np.random.normal(mu, sigma, size=[opt.batch_size, 100])

for it in range(opt.epochs):

    print("\niteration ", it+1, " of " , opt.epochs)

    '''
    first train discriminator:
    	- create fake images with generator, concatenate with real images
    	- fit discriminator on those samples
    '''

    batches = int(opt.num_of_imgs/opt.batch_size)

    progress_bar = Progbar(target=batches)

    p = np.random.permutation(opt.num_of_imgs)
    images_train = images_train[p]


    for i in range(batches):
        progress_bar.update(i)

        noise = np.random.normal(mu, sigma, size=[opt.batch_size, 100])

        images_fake = generator.predict(noise)

        images_batch = images_train[i*opt.batch_size:(i+1)*opt.batch_size,:,:,:]

        if add_noise:
            im_noise = np.exp( - it/10) * np.random.normal(0,0.3,size = images_batch.shape )
            images_batch += im_noise

        for i in range(opt.Diters):
            d_loss1 = discriminator.train_on_batch(images_batch,-np.ones(opt.batch_size))
            d_loss2 = discriminator.train_on_batch(images_fake, np.ones(opt.batch_size))

        '''
            weight_clipping
        '''

        for l in discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, opt.clamp_lower, opt.clamp_upper) for w in weights]
            l.set_weights(weights)

        d_loss = (d_loss1 + d_loss2)
        progress_bar.add(d_loss)
        noise_tr = np.random.normal(mu, sigma, size=[opt.batch_size, 100])

        g_loss = GAN.train_on_batch(noise_tr, -np.ones(opt.batch_size))

        losses["g"].append(g_loss)
        losses["d"].append(d_loss)

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

        images_plot = (images_plot + 1) * 127.5

        for i in range(20):
            ax2[i].cla()
            ax2[i].imshow(images_plot[i].astype(np.uint8) )
            ax2[i].axis('off')

        ax1.cla()
        ax1.plot(-medfilt(losses_plot_d,101), label='wasserstein estimate')
        ax1.legend()

        if save_images:
            fig2.savefig(opt.output_folder + opt.dataset + str(it) + ".png")
            fig1.savefig(opt.output_folder + "hist.png")
        plt.pause(0.0000001)

	# save model every 1000 opt.epochs
    if np.mod(it+1, 25) == 0:
    	discriminator.save("./output/" + disc_name)
    	generator.save("./output/" + gen_name)

plt.show()
