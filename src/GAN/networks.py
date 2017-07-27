from keras.models import Sequential, load_model, Model
from keras.layers import Input, concatenate, Dense, Reshape, MaxPooling2D,Conv2D,  Dropout, UpSampling2D, Activation, Flatten, AveragePooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.regularizers import l2

from keras.optimizers import Adam

from keras import backend as K
import math
'''
binary_accuracy_:
    Keras function doesn't work for soft labels
    -> also round y_true
'''

def binary_accuracy_(y_true, y_pred):
    return K.mean(K.equal( K.round(y_true) , K.round(y_pred) ), axis=-1)

'''
wasserstein loss
'''
def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

'''
mean and min of predictions, just to keep track of training process
'''

def mean_pred(y_true, y_pred):
    return K.mean(y_pred, axis = 0)

def min_pred(y_true, y_pred):
    return K.min(y_pred, axis = 0)



def get_upSampling_generator(image_dim = 64 , filtersize = 5, dropout_rate = 0.5, regularisation = 1e-4, filters = 256):

    reg = l2(regularisation)
    generator = Sequential()

    generator.add(  Dense(4*4*filters,  input_shape = [100]))
    generator.add(  BatchNormalization())
    #generator.add(  Activation('relu'))
    generator.add(  LeakyReLU())
    generator.add(  Dropout(dropout_rate))
    generator.add(  Reshape((4 , 4, filters )))
    
    for i in int(np.math.log(image_dim/4, 2)):
      generator.add(  Conv2D( int(filters/np.pow(2,(i+1))) , (filtersize, filtersize), padding = 'same',kernel_regularizer =reg)  )
      generator.add(  BatchNormalization())
      generator.add(  LeakyReLU())
      #generator.add(  Activation('relu'))
      generator.add(  Dropout(dropout_rate))
      generator.add( UpSampling2D())
    

    generator.add(  Conv2D( 3 , (3,3), padding = 'same', kernel_regularizer = reg))
    generator.add(  Activation('tanh'))

    return generator

'''
deconv_generator:
    fully convolutional gereator network, similar to DCGAN-Architecture
 arguments:
    filters - number of filters for conv layers
'''

def get_deconv_generator(filters = 1024, filtersize = 5,regularisation = 1e-2, dropout_rate =0.5, image_dim = 128 ):

    generator = Sequential()
    reg =  l2(regularisation)

    '''
    Project and Reshape: "just a matrix multiplication" according to DCGAN paper
      except in their code this is not the case... use the version from their code here
    '''
    generator.add(  Dense(4*4*filters,  input_shape = [100]))
    generator.add(  BatchNormalization())
    generator.add(  Activation('relu'))
    generator.add(  Reshape(( 4 , 4, filters )))
    
    for i in range(int(math.log(image_dim/6,2))):
      generator.add(  Conv2DTranspose(int(filters/(pow(2,i+1))),
                      kernel_size=(filtersize, filtersize),
                      strides=(2, 2),
                      padding='same',
                      kernel_regularizer = reg))
      generator.add(  BatchNormalization())
      #generator.add(  LeakyReLU())
      generator.add(  Activation('relu'))

      generator.add(  Dropout(dropout_rate))
    
    generator.add(Conv2DTranspose(3,
                kernel_size=(filtersize, filtersize),
                strides = (2,2),
                padding='same',
                activation='tanh',
                kernel_regularizer = reg))

    return generator


def get_discriminator(input_dim = 128,  filters = 256,filtersize = 5, regularisation = 1e-4, dropout_rate = 0.5, batch_norm = False, wasserstein = False):

    reg = l2(regularisation)
    discriminator = Sequential()
    n_layers = int(math.log(input_dim / 2, 2))
    for i in range(n_layers):
      
      discriminator.add(  Conv2D(int(filters * math.pow(2,(i-n_layers+1))),
                (filtersize, filtersize),
                strides = (2,2),
                padding='same',
                input_shape = (input_dim,input_dim,3),
                kernel_regularizer = reg))
      #output_shape = (64,64)


      if batch_norm:
        discriminator.add(  BatchNormalization())
      discriminator.add(  LeakyReLU())
      discriminator.add(  Dropout(dropout_rate))

    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    
    if not wasserstein:  
      discriminator.add(  Activation("sigmoid"))
    
    return discriminator
