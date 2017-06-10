import image_parser as parse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Input
from keras import backend as K
from keras.optimizers import Adam
import networks

K.set_image_data_format('channels_first') # using theano dimension ordering

'''
Hyperparameters:
'''

pretrain_disc = True        # pretrain discriminator?
pretrain_eps = 5            # number of pretraining epochs
G_dropout_rate = 0.7        # dropout rate for generator
D_dropout_rate = 0.7        # dropout rate for discriminator
g_filters = 512             # number of filters of first layer of generator
d_filters = 512             # number of filters of  of generator
image_dim  = 64             # image.shape = (3,image_dim, image_dim)
num_of_imgs = 512           # number of images used for training
out_dim = 1                 # output dimension for discriminator network
batch_size = 16             # batch size for training
iterations = 50000          # number of iterations of training process
l2_regularisation_d = 1e-2  # l2 kernel_regularizer for discriminator
l2_regularisation_g = 1e-8  # l2 kernel_regularizer for generator
deconv = False              # using strided Conv2DTranspose for Upsampling, else UpSampling2D
soft_labels = True          # using labels in range around 0/1 insteadof binary labels -> improves stability
save_images = True          # save generated images after every 50th iteration
img_folder = "C:/Users/Philip/OneDrive/Dokumente/gan_imgs/"
                            # folder where images will be saved in

images_train = parse.cats_load_data_set(image_dim = image_dim)[:num_of_imgs]
images_train = images_train /255.0


'''
binary_accuracy_:
    Keras function doesn't work for soft labels
    -> also round y_true
'''
def binary_accuracy_(y_true, y_pred):
    return K.mean(K.equal( K.round(y_true) , K.round(y_pred) ), axis=-1)

'''
mean and min of predictions, just to keep track of training process
'''

def mean_pred(y_true, y_pred):
    return K.mean(y_pred, axis = 0)

def min_pred(y_true, y_pred):
    return K.min(y_pred, axis = 0)

if out_dim == 1:
    loss = 'binary_crossentropy'
    acc = binary_accuracy_
else:
    loss = 'categorical_crossentropy'
    acc = 'categorical_accuracy'

'''
 normalize color chanels:
 may be better for training process, but gave some plotting issues with matplotlib
 -> maybe use cv2
'''
#images_train[:,0] = images_train[:,0] - np.mean(images_train[:,0])
#images_train[:,1] = images_train[:,1] - np.mean(images_train[:,1])
#images_train[:,2] = images_train[:,2] - np.mean(images_train[:,2])

discriminator = networks.get_discriminator(input_dim = image_dim, filters = 256 ,regularisation = l2_regularisation_d, dropout_rate = D_dropout_rate, out_dim = out_dim)
discriminator.compile(loss=loss,metrics=[acc, mean_pred], optimizer=Adam(lr = 1e-3, beta_1 = 0.5))

if deconv:
    generator = networks.get_deconv_generator(filters = g_filters, image_dim = image_dim, regularisation = l2_regularisation_g, dropout_rate = G_dropout_rate)
else:
    generator = networks.get_upSampling_generator(image_dim = image_dim, filters = 256, regularisation = l2_regularisation_g, dropout_rate = G_dropout_rate)

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

GAN.compile(loss= loss, metrics=[acc, mean_pred,  min_pred], optimizer = Adam(lr = 1e-5, beta_1 = 0.5))

generator.summary()
discriminator.summary()
GAN.summary()

for l in discriminator.layers:
    l.trainable = True


fig1, ax1 = plt.subplots(2,1)
ax1 = ax1.reshape(-1)
fig2, ax2 = plt.subplots(4,4)
ax2 = ax2.reshape(-1)
fig2.set_size_inches(7,8)
fig1.set_size_inches(7,8)


'''
fuzzy labeling: use labels somewhere in the range of 1 or 0 instead
of strict labeling
'''
if soft_labels:
    one = np.random.uniform(0.7,1.0, size = batch_size)
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

    noise = np.random.normal(0.5,0.2 ,size=[num_of_imgs, 100])

    images_fake = generator.predict(noise)
    if soft_labels:
        y_real = np.random.uniform(0.7, 1.0, size = (num_of_imgs, out_dim))
        y_fake = np.random.normal(0., 0.3, size = (num_of_imgs, out_dim))
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
    discriminator.fit(x,y,epochs = epochs, batch_size = 32)


if pretrain_disc:
    train_disc(pretrain_eps)

'''
##################################################################
                        Training
##################################################################
'''


losses = {"d":[], "g":[]}
accuracies = {"d":[], "g":[]}


for it in range(iterations):

    if np.mod(it,5) == 0:
        print("iteration ", it)

    '''
    first train discriminator:
        - create fake images with generator, concatenate with real images
        - fit discriminator on those samples
    '''

    noise = np.random.normal(0.5, 0.2, size=[batch_size, 100])
    images_fake = generator.predict(noise)
    images_batch = images_train[np.random.randint(0,images_train.shape[0],size=batch_size),:,:,:]

    im_noise = np.exp(-it) * np.random.normal(0,0.1,size = images_batch.shape )
    images_batch += im_noise
    d_loss1 = discriminator.train_on_batch(images_batch,y_r)
    d_loss2 = discriminator.train_on_batch(images_fake, y_f)
    d_loss1 = discriminator.train_on_batch(images_batch,y_r)
    d_loss2 = discriminator.train_on_batch(images_fake, y_f)
    #d_loss2 = discriminator.train_on_batch(images_fake, y_f)
    d_loss = (np.array(d_loss1)+np.array(d_loss2))* 0.5

    losses["d"].append(d_loss[0])
    accuracies["d"].append(d_loss[1])

    noise_tr = np.random.normal(0.5, 0.2, size=[2*batch_size, 100])
    g_loss = GAN.train_on_batch(noise_tr, y_g)
    if np.mod(it,10) == 0:
        #print("Gan Predictions: \t", np.max(GAN.predict(noise_tr)), np.min(GAN.predict(noise_tr)))
        print(g_loss)
        print("disc on fake: ", d_loss2[2])
    pred = discriminator.predict(images_batch)
    # uncomment the following to check if training process runs corectly:
    #print("disc Predictions: (1)\t", np.min(pred[:,0]), np.max(pred[:,0]))
    #print(np.max(GAN.layers[2].layers[1].get_weights()[0] - discriminator.layers[1].get_weights()[0]))
    #print(np.max(GAN.layers[1].layers[1].get_weights()[0] - generator.layers[1].get_weights()[0]))
    losses["g"].append(g_loss[0])#.history['loss'])
    accuracies["g"].append(g_loss[1])#.history['acc'])

    if np.mod(it+1,50) == 0:
        #train_disc(1)
        losses_plot_g = np.array(losses['g'])
        losses_plot_d = np.array(losses['d'])
        acc_plot_g = np.array(accuracies['g'])
        acc_plot_d = np.array(accuracies['d'])

        images_fake = np.swapaxes(images_fake, 1,2)
        images_fake = np.swapaxes(images_fake, 3,2)*255

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
            fig2.savefig(img_folder + "iteration" + str(it) + ".png")
        plt.pause(0.0000001)

    # save model every 1000 iterations
    if np.mod(it, 1000) == 0:
        GAN.save("./networks/gan_cats.h5")
        discriminator.save("./networks/disc_cats.h5")
        generator.save("./networks/generator_cats.h5")

plt.show()
