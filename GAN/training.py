import image_parser as parse
# uncomment this for use on ssh
#import matplotlib
#matplotlib.use("ps")
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Input
from keras import backend as K
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from networks import *
import utilities as util
from os.path import isfile
K.set_image_data_format('channels_first') # using theano dimension ordering
import tensorflow as tf

'''
Hyperparameters:
'''
'''
general parameters:
'''
dataset = 'celeb'           # cats, dogs or celeb dataset
load = False                # load saved model
image_dim  = 64             # image.shape = (3,image_dim, image_dim)

'''
Network parameters
'''
g_dropout_rate = 0.5        # dropout rate for generator
d_dropout_rate = 0.3        # dropout rate for discriminator
g_filters = 512             # number of filters of first layer of generator
d_filters = 512             # number of filters of  of discriminator
filtersize = 5              # size of Conv filters
dilation_rate = 1           # dilation factor for dilated Connvolutions
d_batchnorm = True          # use BatchNormalization in discriminator
out_dim = 1                 # output dimension for discriminator network
d_l2_regularisation = 5e-4  # l2 kernel_regularizer for discriminator
g_l2_regularisation = 5e-4  # l2 kernel_regularizer for generator
deconv = True               # using strided Conv2DTranspose for Upsampling, else UpSampling2D
opt_d = Adam(lr = 2e-5, beta_1 = 0.6)
opt_g = Adam(lr = 2e-4, beta_1 = 0.6)
                            # optimizers

'''
Training parameters
'''
pretrain_eps = 0            # number of pretraining epochs for discriminator
num_of_imgs = 1024          # number of images used for training
batch_size = 64             # batch size for training
iterations = 2000           # number of iterations of training process
soft_labels = True          # using labels in range around 0/1 insteadof binary labels -> improves stability
one_sided_sl = False        # using label smoothing only for real samples
disc_train_eps = 1          # how often each network will be
gen_train_epochs = 1        # trained in one epoch
add_noise = True            # add noise to images, with decreasing magnitude
mu = 0                      # mean for noise input to generator
sigma = 0.5                 # stddev for noise input

'''
plotting parameters
'''
refresh_interval = 15       # interval for refreshing of plots
save_images = True          # save generated images after every 50th iteration
img_folder = "./imgs/"      # "C:/Users/Philip/OneDrive/Dokumente/gan_imgs/"
                            # folder where images will be saved in
plot_weights = False        # plot weights of some conv layers during training


assert (dataset == 'cats' or dataset == 'celeb' or dataset == "dogs" ), "invalid dataset name, only cats,dogs and celeb are allowed"

images_train = parse.load_data_set(name = dataset, maxitems = num_of_imgs ,image_dim = image_dim)[:num_of_imgs]
images_train = (images_train - 127.5 ) / 127.5


if out_dim == 1:
    loss = 'binary_crossentropy'
    acc = binary_accuracy_

else:
    loss = 'categorical_crossentropy'
    acc = 'categorical_accuracy'


disc_name = util.get_model_name(discriminator = True, filters = d_filters, dropout_rate = d_dropout_rate,dilation_rate = dilation_rate, batch_norm = d_batchnorm, out_dim = out_dim, filtersize = filtersize)
gen_name = util.get_model_name(discriminator = False, deconv = deconv, filters = g_filters, dropout_rate = g_dropout_rate, dilation_rate = dilation_rate,out_dim = out_dim, filtersize = filtersize)


if load and isfile("./networks/" + gen_name):
    generator = load_model("./networks/" + gen_name , custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})
    discriminator = load_model("./networks/"+ disc_name, custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})

else:
    if load and not isfile("./networks/" + gen_name):
        print("couldn't find model. New model will be generated.")

    discriminator = get_discriminator(input_dim = image_dim, filters = d_filters, filtersize = filtersize ,regularisation = d_l2_regularisation, dropout_rate = d_dropout_rate, batch_norm = True, out_dim = out_dim, dilation_rate = dilation_rate)
    discriminator.compile(loss=loss,metrics=[acc, mean_pred], optimizer=opt_d)

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

GAN.compile(loss= loss, metrics=[acc, mean_pred], optimizer =opt_g)

generator.summary()
discriminator.summary()
GAN.summary()

for l in discriminator.layers:
    l.trainable = True

'''
initialize matplotlib windows
'''

fig1, ax1 = plt.subplots(2,1)
ax1 = ax1.reshape(-1)
fig2, ax2 = plt.subplots(4,4)
ax2 = ax2.reshape(-1)
fig2.set_size_inches(7,8)
fig1.set_size_inches(7,8)
plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.05, hspace=0.01)
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
train_disc:
    only training discriminator - may be beneficial for stability in training process
arguments:
    epochs: number of epochs for training
'''
def train_disc(epochs):
    if epochs == 0:
        return
    noise = np.random.normal(0.5,0.2 ,size=[num_of_imgs, 100])

    images_fake = generator.predict(noise)
    if soft_labels:
        y_real = np.random.uniform(0.7, 1.0, size = (num_of_imgs, out_dim))
        y_fake = np.random.normal(0., 0.3, size = (num_of_imgs, out_dim))
        if one_sided_sl:
            y_fake = np.zeros((num_of_imgs, out_dim))
    else:
        y_real = np.ones((num_of_imgs,out_dim))
        y_fake = np.zeros((num_of_imgs, out_dim))

    if out_dim == 2:
        y_real[:,1] = y_real[:,1] - 1
        y_fake[:,0] = y_fake[:,0] - 1
    else:
        y_fake = y_fake - 1
    x = np.concatenate([images_train, images_fake])
    y = np.concatenate([y_real, y_fake])
    init_w = discriminator.layers[0].get_weights()[0]
    discriminator.fit(x,y,epochs = epochs, batch_size = 32)
    print(np.max(discriminator.layers[0].get_weights()[0] - init_w) )

'''
##################################################################
                        Training
##################################################################
'''

train_disc(pretrain_eps)

losses = {"d":[], "g":[]}
accuracies = {"d":[], "g":[]}

for it in range(iterations):

    print("iteration ", it+1, " of " , iterations)

    '''
    first train discriminator:
    	- create fake images with generator, concatenate with real images
    	- fit discriminator on those samples
    '''
    batches = int(num_of_imgs/batch_size)

    progress_bar = Progbar(target=batches)
    np.random.shuffle(images_train)

    for i in range(batches):
        progress_bar.update(i)
        noise = np.random.normal(mu, sigma, size=[batch_size, 100])
        images_fake = generator.predict(noise)

        images_batch = images_train[i*batch_size:(i+1)*batch_size,:,:,:]
        if add_noise:
            im_noise = np.exp( - it/10) * np.random.normal(0,0.3,size = images_batch.shape )
            images_batch += im_noise

        for i in range(disc_train_eps):
            d_loss1 = discriminator.train_on_batch(images_batch,y_r)
            d_loss2 = discriminator.train_on_batch(images_fake, y_f)

        d_loss = (np.array(d_loss1)+np.array(d_loss2))* 0.5

        noise_tr = np.random.normal(mu, sigma, size=[2*batch_size, 100])
        for i in range(gen_train_epochs):
            g_loss = GAN.train_on_batch(noise_tr, y_g)
        losses["g"].append(g_loss[0])
        accuracies["g"].append(g_loss[1])
        losses["d"].append(d_loss[0])
        accuracies["d"].append(d_loss[1])

    print("\n gan accuracy: \t " , g_loss[1])
    print(" disc accuracy:\t ", d_loss[1])
    print(" disc mean: \t" , d_loss[2])
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

    	images_fake = np.swapaxes(images_fake, 1,2)
    	images_fake = (np.swapaxes(images_fake, 3,2) + 1) * 127.5

    	for i in range(16):
    		ax2[i].cla()
    		ax2[i].imshow(images_fake[i].astype(np.uint8) )
    		ax2[i].axis('off')
    	ax1[0].cla()
    	ax1[0].plot(acc_plot_d, label='discriminitive accuracy')
    	ax1[0].plot(acc_plot_g, label='generative accuracy')
    	ax1[0].legend()

    	ax1[1].cla()
    	ax1[1].plot(losses_plot_d, label='discriminitive loss')
    	ax1[1].plot(losses_plot_g, label='generative loss')
    	ax1[1].legend()
    	if save_images:
    		fig2.savefig(img_folder + dataset + str(it) + ".png")
    	plt.pause(0.0000001)

	# save model every 1000 iterations
    if np.mod(it+1, 10) == 0:
    	#GAN.save("./networks/gan_cats.h5")
    	discriminator.save("./networks/" + disc_name)
    	generator.save("./networks/"+ gen_name)

plt.show()
