"""
Implementation of a conditioned VAE using Keras and tensorflow.
"""

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
from keras.layers.merge import Concatenate
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import OneHotEncoder

import scipy.misc

dataset = "celeba"

#indicates whether the model shall be generated or loaded.
#only False is currently supported.
keep_training = False

"""
    setup the hyperparameters of the model for the specified data set
"""
def setup_hyperparameters():
    global img_rows, img_cols, img_chns, filters, num_conv, latent_dim, intermediate_dim, epsilon_std, epochs, batch_size, nb_conditional_parameters
    if dataset == "celeba":
        # input image dimensions
        img_rows, img_cols, img_chns = 64, 64, 3
        # number of convolutional filters to use
        filters = 64
        # convolution kernel size
        num_conv = 3
        #number of latent variables
        latent_dim = 300
        intermediate_dim = 100
        epsilon_std = 2.0
        #number of epochs and batch size for the training of the VAE
        epochs = 400
        batch_size = 100

        nb_conditional_parameters = 4
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
        epochs = 200
        batch_size = 100

        nb_conditional_parameters = 62
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

        nb_conditional_parameters = 10
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
        epochs = 5
        batch_size = 100

        nb_conditional_parameters = 10
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
        encoder = OneHotEncoder(nb_conditional_parameters)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).A
        y_test = encoder.transform(y_test.reshape(-1, 1)).A

    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        encoder = OneHotEncoder(nb_conditional_parameters)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).A
        y_test = encoder.transform(y_test.reshape(-1, 1)).A

    elif dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        encoder = OneHotEncoder(nb_conditional_parameters)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).A
        y_test = encoder.transform(y_test.reshape(-1, 1)).A


    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)
load_data()

"""
    builds, trains and saves the model
"""
def build_train():
    x = Input(batch_shape=(batch_size,) + original_img_size, name="image_input")
    conditional = Input(batch_shape=(batch_size, nb_conditional_parameters), name="conditional_input")
    conv_1 = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu')(x)
    conv_2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_2)
    conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim, name="z_mean")(hidden)
    z_log_var = Dense(latent_dim, name="z_log_var")(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    #merge the latend variables (z) and the conditional input
    merged = Concatenate()([z, conditional])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu', name="decoder_hid")
    decoder_upsample = Dense(filters * img_rows//2 * img_cols//2, activation='relu', name="decoder_upsample")

    if K.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, img_rows//2, img_cols//2)
    else:
        output_shape = (batch_size, img_rows//2, img_cols//2, filters)

    decoder_reshape = Reshape(output_shape[1:], name="decoder_reshape")
    decoder_deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu', name="decoder_deconv_1")
    decoder_deconv_2 = Conv2DTranspose(filters, num_conv, padding='same', strides=1, activation='relu', name="decoder_deconv_2")

    decoder_deconv_3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name="decoder_deconv_3_upsamp")
    decoder_mean_squash = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid', name="decoder_mean_squash")

    hid_decoded = decoder_hid(merged)
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
    vae = Model(inputs=[x, conditional], outputs=[y])
    vae.compile(optimizer='rmsprop', loss=None)
    vae.summary()

    vae.fit([x_train, y_train], shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=([x_test, y_test], x_test), verbose=2)
    vae.save("./output/vae_{0}_{1}_{2}.h5".format(dataset, latent_dim, epochs))

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    encoder.save("./output/encoder_{0}_{1}_{2}.h5".format(dataset, latent_dim, epochs))

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    decoder_conditional = Input(shape=(nb_conditional_parameters,), name="decoder_conditional")
    merged_decoder_input = Concatenate()([decoder_input, decoder_conditional])
    decoder_hid_decoded = decoder_hid(merged_decoder_input)
    decoder_up_decoded = decoder_upsample(decoder_hid_decoded)
    decoder_reshape_decoded = decoder_reshape(decoder_up_decoded)
    decoder_deconv_1_decoded = decoder_deconv_1(decoder_reshape_decoded)
    decoder_deconv_2_decoded = decoder_deconv_2(decoder_deconv_1_decoded)
    decoder_x_decoded_relu = decoder_deconv_3_upsamp(decoder_deconv_2_decoded)
    decoder_x_decoded_mean_squash = decoder_mean_squash(decoder_x_decoded_relu)
    generator = Model([decoder_input, decoder_conditional], decoder_x_decoded_mean_squash)

    generator.save("./output/generator_{0}_{1}_{2}.h5".format(dataset, latent_dim, epochs))

    return vae, generator

if not os.path.isfile("./output/generator_{0}_{1}_{2}.h5".format(dataset, latent_dim, epochs)):
    _, generator = build_train()
else:
    print("loading model...")
    generator = load_model("./output/generator_{0}_{1}_{2}.h5".format(dataset, latent_dim, epochs))
    #vae = load_model("./output/vae_{0}_{1}.h5".format(dataset, latent_dim))
    encoder = load_model("./output/encoder_{0}_{1}_{2}.h5".format(dataset, latent_dim, epochs))
    print("models loaded")

    if keep_training:
        print("keep_training is not supported at the moment - sorry!")
        """
        print("starting training...")
        vae.fit([x_train, y_train], shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=([x_test, y_test], x_test), verbose=2)
        vae.save("./output/vae_{0}_{1}.h5".format(dataset, latent_dim))

        x = vae.get_layer("image_input")
        z_mean = vae.get_layer("z_mean")
        encoder = Model(x, z_mean)
        encoder.save("./output/encoder_{0}_{1}.h5".format(dataset, latent_dim))

        decoder_input = Input(shape=(latent_dim,), name="decoder_input")
        decoder_conditional = Input(shape=(nb_conditional_parameters,), name="decoder_conditional")
        merged_decoder_input = Concatenate()([decoder_input, decoder_conditional])

        decoder_hid_decoded = vae.get_layer("decoder_hid")(merged_decoder_input)
        decoder_up_decoded = vae.get_layer("decoder_upsample")(decoder_hid_decoded)
        decoder_reshape_decoded = vae.get_layer("decoder_reshape")(decoder_up_decoded)
        decoder_deconv_1_decoded = vae.get_layer("decoder_deconv_1")(decoder_reshape_decoded)
        decoder_deconv_2_decoded = vae.get_layer("decoder_deconv_2")(decoder_deconv_1_decoded)
        decoder_x_decoded_relu = vae.get_layer("decoder_deconv_3_upsamp")(decoder_deconv_2_decoded)
        decoder_x_decoded_mean_squash = vae.get_layer("decoder_mean_squash")(decoder_x_decoded_relu)
        generator = Model([decoder_input, decoder_conditional], decoder_x_decoded_mean_squash)

        generator.save("./output/generator_{0}_{1}.h5".format(dataset, latent_dim))
        print("training finished")
        """

#now save some sampled results
nb_rows = 15
nb_cols = 15

#randomly sample the latent variable space
grid = np.array([norm.ppf(np.random.rand(latent_dim)) for i in range(nb_rows*nb_cols) ] )

for j in [1]:
    for n, ind in enumerate([[], [0], [1], [2], [3], [0, 1], [0, 1, 2]]):
        figure = np.zeros((img_rows * nb_rows, img_cols * nb_cols, img_chns))
        for i, x in enumerate(grid):
            z_sample = np.array([x])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
            conditional_sample = np.zeros((batch_size, nb_conditional_parameters))
            conditional_sample[:, ind] = j
            x_decoded = generator.predict([z_sample, conditional_sample], batch_size=batch_size)
            img = x_decoded[0].reshape(img_rows, img_cols, img_chns)
            figure[i%nb_rows * img_rows: (i%nb_rows + 1) * img_rows, i//nb_cols * img_cols: (i//nb_cols + 1) * img_cols] = img

        if img_chns == 1:
            figure = figure[:, :, 0]

        scipy.misc.imsave('./output/result_{0}_{1}.jpg'.format(n, j), figure)
