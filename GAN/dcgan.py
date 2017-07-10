""" Code based on: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout, BatchNormalization
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

from keras.datasets import cifar10, mnist


import argparse
import sys

class DCGAN_model(object):
    def __init__(self, img_rows=32, img_cols=32, channel=3, upsample = False):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 32 #64
        dropout = 0.4
        # In: 32 x 32 x 3, depth = 1
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 4, strides=2, padding='same',
            input_shape=input_shape))
        self.D.add(LeakyReLU(alpha=0.2))
        #self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 4, strides=2, padding='same'))
        self.D.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=0.2))
        #self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 4, strides=2, padding='same'))
        self.D.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=0.2))
        #self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 4, strides=2, padding='same'))
        self.D.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=0.2))
        #self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        # TODO: Replace fc layer
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))

        self.D.summary()
        return self.D

    def generator(self):
        # TODO: solution for mnist, end result has to be 28x28x1, am besten allgemeine dimensionen
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim =2
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization())
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        #self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(Conv2DTranspose(int(depth/2), 4, strides=2, padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/4), 4, strides=2, padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 4, strides=2, padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(self.channel, 4, strides=2, padding='same'))
        self.G.add(Activation('tanh'))
        self.G.summary()
        return self.G

    def generator_upsample(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = int(self.img_rows/4.0)
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(self.channel, 5, padding='same'))
        self.G.add(Activation('sigmoid'))

        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.AM = Sequential()
        if upsample == True:
            self.AM.add(self.generator_upsample())
        else:
            self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class DCGAN_trainer(object):
    def __init__(self, img_rows=32, img_cols=32, channel=3, dataset="CIFAR", upsample = False):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.dataset = dataset
        self.upsample = upsample

        if self.dataset == "CIFAR":
            print("Training on CIFAR-10 dataset")
            (self.x_train, _), (_, _) = cifar10.load_data()
            print(self.x_train.shape, np.amax(self.x_train))
            self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, self.channel).astype(np.float32)
            self.x_train/=255.0
        elif self.dataset == "MNIST":
            print("Training on MNIST dataset")
            (self.x_train, _), (_, _) = mnist.load_data()
            print(self.x_train.shape, np.amax(self.x_train))
            # shuffle array with hope for better training :)
            #np.random.shuffle(self.x_train)
            self.x_train = self.x_train.reshape(-1, self.img_rows,\
                self.img_cols, 1).astype(np.float32)
            self.x_train/=255.0
        else:
            print("Unknown dataset")

        self.DCGAN_model = DCGAN_model(img_rows=self.img_rows, img_cols=self.img_cols, channel=self.channel, upsample = self.upsample)
        self.discriminator =  self.DCGAN_model.discriminator_model()
        self.adversarial = self.DCGAN_model.adversarial_model()
        if upsample == True:
            self.generator = self.DCGAN_model.generator_upsample()
        else:
            self.generator = self.DCGAN_model.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=100):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        """Plot generated/training images

        # Arguments
            fake: plot generated images if True

        # Returns

        """
        filename = 'cifar.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "cifar_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, self.channel])
            #image *=255
            #plt.imshow(image.astype(np.uint8))
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbose", action = "store_true", help =  \
        "writes more info")
    parser.add_argument("-t", "--test", dest = "testing", action = "store_true", help = \
        "run network on test dataset")
    parser.add_argument("-s", "--save", dest = "model_name",
        help = "save model with given name")
    parser.add_argument("-p", "--plot", dest = "visualize", action = "store_true",
        help = "Plot loss and accuracy and weights of first layer")
    parser.add_argument("-d", "--dataset", dest = "dataset", help = "choose training dataset")
    parser.add_argument("--upsample", dest = "upsample", action = "store_true", help = "if True, uses UpSample2D Layers additional to transpose convolutions.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()

    verbose = args.verbose
    testing = args.testing
    model_name = args.model_name
    visualize = args.visualize
    dataset = args.dataset
    upsample = args.upsample

    if dataset == "MNIST":
        rows = 28
        cols = 28
        channel = 1
        # temporary solution until generator fixed
        upsampling == True
    elif dataset == "CIFAR":
        rows = 32
        cols = 32
        channel = 3
    else:
        print("Please provide either \"MNIST\" or \"CIFAR\" as dataset")

    dcgan = DCGAN_trainer(rows, cols, channel, dataset = dataset, upsample=upsample)
    dcgan.train(train_steps=10000, batch_size=256, save_interval=100)
    dcgan.plot_images(fake=True)
    dcgan.plot_images(fake=False, save2file=True)
