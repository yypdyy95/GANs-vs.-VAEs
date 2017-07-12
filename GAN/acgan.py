import numpy as np

#import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Embedding
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout, BatchNormalization, Input
from keras.layers.merge import multiply
from keras.optimizers import Adam, RMSprop
from keras.utils.generic_utils import Progbar

import matplotlib
matplotlib.use("ps")
import matplotlib.pyplot as plt
from PIL import Image

import pickle

from keras.datasets import cifar10, mnist

import argparse
import sys

np.random.seed(1337)


def build_generator(latent_size = 100, depth = 64, dropout_rate = 0.0, batch_norm = False, img_size = 28):
    """build generator model

    # Arguments
        latent_size: number of latent variables = noise from which img are gen.
        depth: depth of data before 1st UpSampling
        dropout_rate: applied dropout_rate
        batch_norm: apply Batch Normalization after each layer
        img_size: number of pixel in one row/col

    # Returns
        gen: keras Model instance of generator

    """
    # TODO: Add support for CIFAR-10
    gen = Sequential()
    # yield that generated image has same number of rows as test images
    dim = int(img_size/4.0)

    gen.add(Dense(dim*dim*depth, input_dim=latent_size))
    if batch_norm:
        gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))
    gen.add(Reshape((dim, dim, depth)))
    gen.add(Dropout(dropout_rate))

    gen.add(UpSampling2D())
    gen.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    if batch_norm:
        gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))

    gen.add(UpSampling2D())
    gen.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    if batch_norm:
        gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))

    gen.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    if batch_norm:
        gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))

    gen.add(Conv2DTranspose(channel, 5, padding='same'))
    gen.add(Activation('tanh'))


    latent_vector = Input(shape=(latent_size, ))
    img_class = Input(shape= (1,), dtype = 'int32')

    class_embedding = Embedding(10, latent_size)(img_class)
    class_embedding = Flatten()(class_embedding)
    h = multiply([latent_vector, class_embedding])

    fake_im = gen(h)
    gen = Model(inputs = [latent_vector, img_class], outputs = fake_im)

    gen.summary()
    return gen

def build_discriminator(depth = 32, dropout_rate = 0.0, batch_norm = False, img_size = 28, lReLU_alpha = 0.3):
    """build discriminator Model

    # Arguments
        depth: depth of data before 1st UpSampling
        dropout_rate: applied dropout_rate
        batch_norm: apply Batch Normalization after each layer
        img_size: number of pixel in one row/col

    # Returns

    """

    dis = Sequential()
    input_shape = (img_size, img_size, channel)
    dis.add(Conv2D(depth*1, 4, strides=2, padding='same',
        input_shape=input_shape))
    dis.add(LeakyReLU(alpha=lReLU_alpha))
    dis.add(Dropout(dropout_rate))

    dis.add(Conv2D(depth*2, 4, strides=2, padding='same'))
    if batch_norm:
        dis.add(BatchNormalization())
    dis.add(LeakyReLU(alpha=lReLU_alpha))
    dis.add(Dropout(dropout_rate))

    dis.add(Conv2D(depth*4, 4, strides=2, padding='same'))
    if batch_norm:
        dis.add(BatchNormalization())
    dis.add(LeakyReLU(alpha=lReLU_alpha))
    dis.add(Dropout(dropout_rate))

    dis.add(Conv2D(depth*8, 4, strides=2, padding='same'))
    if batch_norm:
        dis.add(BatchNormalization())
    dis.add(LeakyReLU(alpha=lReLU_alpha))
    dis.add(Dropout(dropout_rate))

    # Out: 1-dim probability
    # TODO: Replace fc layer
    dis.add(Flatten())

    img = Input(shape=(28, 28, 1))
    features = dis(img)

    fake = Dense(1, activation='sigmoid')(features)
    aux = Dense(10, activation='softmax')(features)

    dis = Model(inputs = img, outputs = [fake, aux])

    dis.summary()
    return dis


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", dest="verbose", action = "store_true",
        help = "writes more info")
    parser.add_argument("-t", "--test", dest = "testing", action = "store_true",
        help = "run network on test dataset")
    parser.add_argument("-s", "--save", dest = "model_name",
        help = "save model with given name")
    parser.add_argument("-p", "--plot", dest = "visualize", action = "store_true",
        help = "Plot loss and accuracy and weights of first layer")
    parser.add_argument("-d", "--dataset", dest = "dataset",
        help = "choose training dataset")
    parser.add_argument("--upsample", dest = "upsample", action = "store_true",
        help = "if True, uses UpSample2D Layers additional to transpose convolutions.")

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


    """ Hyperparameter """
    # model
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # training
    batch_size = 256
    epochs = 50
    noise_mu = 0
    noise_sigma = 0.5

    # plotting
    plot_same_noise = True

    """ Data Preprocessing """

    if dataset == "MNIST":
        print("Training on MNIST dataset")
        img_size = 28
        channel = 1
        (x_train, y_train), (_, _) = mnist.load_data()
        x_train = x_train.reshape(-1, img_size,
            img_size, 1).astype(np.float32)
        x_train = (x_train-127.5)/127.5
        num_train = x_train.shape[0]
        print("Input_shape = {}, max_value = {}, min_value = {}".format(x_train.shape,
            np.amax(x_train), np.amin(x_train)))

    elif dataset == "CIFAR":
        print("Training on CIFAR-10 dataset")
        img_size = 32
        channel = 3
        (x_train, y_train), (_, _) = cifar10.load_data()
        x_train = x_train.reshape(-1, img_size,
            img_size, channel).astype(np.float32)
        x_train/=255.0
        num_train = x_train.shape[0]
        print("Input_shape = {}, max_value = {}".format(x_train.shape,
            np.amax(x_train)))
    else:
        print("Please provide either \"MNIST\" or \"CIFAR\" as dataset")


    """ Build Model """

    # build discriminator
    discriminator = build_discriminator(img_size = img_size, depth=32,
        dropout_rate = 0.3)
    discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

    # build generator
    generator = build_generator(latent_size= latent_size, depth = 32,
        img_size = img_size, dropout_rate = 0.3)
    generator.compile(optimizer=Adam(lr= adam_lr, beta_1 = adam_beta_1),
        loss = 'binary_crossentropy')

    latent_vector = Input(shape=(latent_size, ))
    img_class = Input(shape=(1,), dtype = 'int32')

    # generate fake image
    fake = generator([latent_vector, img_class])

    # train only generator in combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent_vector, img_class], [fake, aux])

    combined.compile(optimizer=Adam(lr = adam_lr, beta_1=adam_beta_1),
        loss = ['binary_crossentropy', 'sparse_categorical_crossentropy'])

    """ initialize matplotlib windows """

    if visualize:
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(10, 6)

    """ Train Model """

    train_hist = {'gen':[], 'dis':[], 'gen_batch':[], 'dis_batch':[]}

    # generate latent vector for plotting after each epoch:
    if plot_same_noise:
        noise_to_plot = np.random.uniform(-1, 1, (100, latent_size))

    for epoch in range(epochs):
        print("Epoch {} of {}".format(epoch+1, epochs))

        batches = int(x_train.shape[0]/batch_size)
        progress_bar = Progbar(target=batches)

        # shuffle data and labels
        p = np.random.permutation(num_train)
        x_train = x_train[p]
        y_train = y_train[p]

        epoch_gen_loss = []
        epoch_dis_loss = []

        for i in range(batches):

            progress_bar.update(i)

            #generate batch of noise and labels
            # TODO: maybe uniform
            noise = np.random.normal(noise_mu, noise_sigma, size=[batch_size,
                latent_size])
            sampled_labels = np.random.randint(0, 10, batch_size)

            # generate batch of fake images corresponding to sampled labels
            imgs_fake = generator.predict([noise, sampled_labels.reshape((-1, 1))])

            # batch of real imgs
            imgs_batch = x_train[i * batch_size:(i+1)*batch_size,:,:,:]
            label_batch = y_train[i * batch_size:(i+1)*batch_size]

            # combine fake and real imgs
            X = np.concatenate((imgs_batch, imgs_fake))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # train discriminator
            # TODO: save loss, acc
            epoch_dis_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # noise for generator training: 2xbatch_size since dis trained on
            # fake and real imgs
            noise = np.random.normal(noise_mu, noise_sigma, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size)

            # when training gen, want dis to say all images are real:
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch([noise,
                sampled_labels.reshape((-1,1))], [trick, sampled_labels]))

        # calc mean loss over epoch
        discriminator_train_loss = np.mean(np.array(epoch_dis_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        train_hist['gen'].append(generator_train_loss)
        train_hist['dis'].append(discriminator_train_loss)
        train_hist['gen_batch'].extend(epoch_gen_loss)
        train_hist['dis_batch'].extend(epoch_dis_loss)

        # TODO: Add to txt file
        np.savetxt("/output/gen_loss.txt", generator_train_loss)
        np.savetxt("/output/dis_loss.txt", discriminator_train_loss)

        # plot loss of all batches:
        if visualize:
            ax1.cla()
            print(generator.metrics_names, discriminator.metrics_names)
            print("shape train_hist['dis_batch']: {}".format(np.array(train_hist['dis_batch']).shape))
            ax1.plot(np.array(train_hist['dis_batch'])[:,0], label="dis_loss")
            ax1.plot(np.array(train_hist['gen_batch'])[:, 0], label="gen_loss")
            ax1.legend()
            fig1.savefig("/output/train_hist_epoch_{0:03d}.png".format(epoch))
            plt.show()

        print('\n{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_hist['gen'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_hist['dis'][-1]))



        # save weights every epoch
        if epoch >= 20 and epoch%5 == 0:
            generator.save_weights(
                '/output/params_generator_epoch_{0:03d}.hdf5'.format(epoch))
            discriminator.save_weights(
                '/output/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch))


        # generate some digits to display
        if plot_same_noise:
            noise = noise_to_plot
        else:
            noise = np.random.uniform(-1, 1, (100, latent_size))

        sampled_labels = np.array([
            [i] * 10 for i in range(10)]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            '/output/plot_epoch_{0:03d}_generated.png'.format(epoch))

    # save models
    generator.save("/output/gen_aux_mnist.hdf5")
    discriminator.save("/output/dis_aux_mnist.hdf5")

    with open('train_hist.txt', 'wb') as file:
        file.write(pickle.dumps(train_hist))

    """pickle.dump({'train': train_hist, 'test': test_history},
                open('acgan-history.pkl', 'wb'))
    """
