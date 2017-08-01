<<<<<<< HEAD
"""
Implementation of a VAE using Keras and tensorflow.
"""

=======
'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
>>>>>>> 1b6ca73bab0ace3d50f625a63d2857d7198cd34a
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from image_parser import load_data_set_classes, convert_image_cl, load_char74k

#improves the output of keras on Windows
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist, cifar10
from keras.models import load_model
import os

#set the desired data set
dataset = "mnist"

"""
    setup the hyperparameters of the model for the specified data set
"""
def setup_hyperparameters():
    global img_rows, img_cols, img_chns, filters, num_conv, latent_dim, intermediate_dim, epsilon_std, epochs, batch_size
    if dataset == "celeba":
        # input image dimensions
        img_rows, img_cols, img_chns = 64, 64, 3
        # number of convolutional filters to use
        filters = 64
        # convolution kernel size
        num_conv = 3
        #number of latent variables
        latent_dim = 300
        intermediate_dim = 128
        epsilon_std = 1.0
        #number of epochs and batch size for the training of the VAE
        epochs = 100
        batch_size = 100

    elif dataset == "char74k":
        # input image dimensions
        img_rows, img_cols, img_chns = 64, 64, 3
        # number of convolutional filters to use
        filters = 64
        # convolution kernel size
        num_conv = 3
        #number of latent variables
        latent_dim = 300
        intermediate_dim = 128
        epsilon_std = 1.0
        #number of epochs and batch size for the training of the VAE
        epochs = 100
        batch_size = 100

    elif dataset == "cifar10":
        # input image dimensions
        img_rows, img_cols, img_chns = 128, 128, 3
        # number of convolutional filters to use
        filters = 64
        # convolution kernel size
        num_conv = 3
        #number of latent variables
        latent_dim = 300
        intermediate_dim = 128
        epsilon_std = 1.0
        #number of epochs and batch size for the training of the VAE
        epochs = 25
        batch_size = 100

    elif dataset == "mnist":
        # input image dimensions
        img_rows, img_cols, img_chns = 28, 28, 1
        # number of convolutional filters to use
        filters = 64
        # convolution kernel size
        num_conv = 3
        #number of latent variables
        latent_dim = 2
        intermediate_dim = 128
        epsilon_std = 1.0
        #number of epochs and batch size for the training of the VAE
        epochs = 50
        batch_size = 100

setup_hyperparameters()

if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

"""
    loads the images of the data set
"""
def load_data():
    global x_train, y_train, x_test, y_test

    from sklearn.model_selection import train_test_split
    # train the VAE on MNIST digits
    """

    """

    if dataset == "celeba":
        x_data, y_data = load_data_set_classes("celeba", ["Male", "Blond_Hair", "Smiling", "Wearing_Hat"], maxitems=50000, image_dim=img_rows)
        y_data[y_data == -1] = 0
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

    elif dataset == "char74k":
        x_data, y_data = load_char74k(image_dim=img_rows)
        x_train, x_test, y_train, y_test = train_test_split(x_data[:50000], y_data[:50000, 0]-1, test_size=0.1)

    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)
load_data()

"""
    builds, trains and saves the model
"""
def build_train():
    x = Input(batch_shape=(batch_size,) + original_img_size)
    conv_1 = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu')(x)
    conv_2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_2)
    conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)


    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])


    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(filters * img_rows//2 * img_cols//2, activation='relu')

    if K.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, img_rows//2, img_cols//2)
    else:
        output_shape = (batch_size, img_rows//2, img_cols//2, filters)

    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters, num_conv, padding='same', strides=1, activation='relu')

    decoder_deconv_3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
    decoder_mean_squash = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid')

    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean_squash):
            x = K.flatten(x)
            x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
            xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean_squash = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean_squash)
            self.add_loss(loss, inputs=inputs)
            # We don't use this output.
            return x

    y = CustomVariationalLayer()([x, x_decoded_mean_squash])
    vae = Model(inputs=[x], outputs=[y])
    vae.compile(optimizer='rmsprop', loss=None)
    vae.summary()

    vae.fit([x_train,], shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=([x_test], x_test), verbose=2)
    vae.save("./output/vae_uncon_{0}_{1}.h5".format(dataset, latent_dim))

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    encoder.save("./output/encoder_uncon_{0}_{1}.h5".format(dataset, latent_dim))

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _hid_decoded = decoder_hid(decoder_input)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)
    _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
    _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
    _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
    _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
    generator = Model(decoder_input, _x_decoded_mean_squash)

    generator.save("./output/generator_uncon_{0}_{1}.h5".format(dataset, latent_dim))

    return vae, generator

if not os.path.isfile("./output/generator_uncon_{0}_{1}.h5".format(dataset, latent_dim)):
    _, generator = build_train()
else:
    print("loading model...")
    generator = load_model("./output/generator_uncon_{0}_{1}.h5".format(dataset, latent_dim))
    vae = load_model("./output/generator_uncon_{0}_{1}.h5".format(dataset, latent_dim))

#save some samples results now
nb_rows = 50
nb_cols = 50

# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, nb_cols))
grid_y = norm.ppf(np.linspace(0.05, 0.95, nb_rows))

figure = np.zeros((nb_rows * img_rows, nb_cols * img_cols, img_chns))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        img = x_decoded[0].reshape(img_rows, img_cols, img_chns)
        figure[i * img_cols: (i + 1) * img_cols,
               j * img_rows: (j + 1) * img_rows] = img

if img_chns == 1:
    figure = figure[:, :, 0]

import scipy.misc
scipy.misc.imsave('./output/result_mnist_uncond.jpg', figure)
