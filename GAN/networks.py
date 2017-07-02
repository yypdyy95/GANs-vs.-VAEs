from keras.models import Sequential, load_model, Model
from keras.layers import Input, concatenate, Dense, Reshape, MaxPooling2D,Conv2D,  Dropout, UpSampling2D, Activation, Flatten, AveragePooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.regularizers import l2

from keras.optimizers import Adam

from keras import backend as K

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



def get_upSampling_generator(image_dim = 128 , filtersize = 5, dropout_rate = 0.5, regularisation = 1e-4, filters = 256):

    reg = l2(regularisation)
    generator = Sequential()

    generator.add(  Dense(4*4*filters,  input_shape = [100]))
    generator.add(  BatchNormalization())
    #generator.add(  Activation('relu'))
    generator.add(  LeakyReLU())
    generator.add(  Dropout(dropout_rate))
    generator.add(  Reshape((filters, 4 , 4 )))
    # 1,4,4
    generator.add(  Conv2D( int(filters/2) , (filtersize, filtersize), padding = 'same',kernel_regularizer =reg)  )
    generator.add(  BatchNormalization())
    generator.add(  LeakyReLU())
    #generator.add(  Activation('relu'))
    generator.add(  Dropout(dropout_rate))
    generator.add( UpSampling2D())
    #filters,8,8
    generator.add(  Conv2D( int(filters/4) , (filtersize, filtersize), padding = 'same',kernel_regularizer =reg)  )
    generator.add(  BatchNormalization())
    generator.add(  LeakyReLU())
    #generator.add(  Activation('relu'))

    generator.add(  Dropout(dropout_rate))

    generator.add(  UpSampling2D())
    generator.add(  Conv2D( int(filters/8) , (filtersize, filtersize), padding = 'same',kernel_regularizer =reg)  )
    generator.add(  BatchNormalization())
    #generator.add(  Activation('relu'))
    generator.add(  LeakyReLU())
    generator.add(  Dropout(dropout_rate))

    generator.add(  UpSampling2D())
    generator.add(  Conv2D( int(filters/16) , (filtersize, filtersize), padding = 'same',kernel_regularizer =reg))

    # dim = 16*16
    generator.add(  Conv2D( int(filters/32) , (filtersize, filtersize), padding = 'same',kernel_regularizer = reg))
    generator.add(  BatchNormalization())
    generator.add(  LeakyReLU())
    #generator.add(  Activation('relu'))
    generator.add(  Dropout(dropout_rate))
    generator.add(  UpSampling2D())

    if image_dim == 128:
        # dim = 16*16
        generator.add(  Conv2D( int(filters/16) , (filtersize, filtersize), padding = 'same',kernel_regularizer = reg))
        generator.add(  BatchNormalization())
        #generator.add(  LeakyReLU())
        generator.add(  Activation('relu'))
        generator.add(  Dropout(dropout_rate))
        generator.add(  UpSampling2D())


    generator.add(  Conv2D( 3 , (3,3), padding = 'same', kernel_regularizer = reg))
    generator.add(  Activation('tanh'))

    return generator

'''
deconv_generator:
    fully convolutional gereator network, similar to DCGAN-Architecture
 arguments:
    filters - number of filters for conv layers
'''

def get_deconv_generator(filters = 1024, filtersize = 5,regularisation = 1e-2, dropout_rate =0.5, dilation_rate = 1 ,image_dim = 128 ):

	generator = Sequential()
	reg =  l2(regularisation)

	'''
	Project and Reshape: "just a matrix multiplication" according to DCGAN paper
	'''
	generator.add(  Dense(4*4*filters,  input_shape = [100]))
	#generator.add(  BatchNormalization())
	#generator.add(  Activation('relu'))
	#generator.add(  LeakyReLU())
	#generator.add(  Dropout(dropout_rate))
	generator.add(  Reshape((filters, 4 , 4 )))
	#generator.add(BatchNormalization())
	# output_shape = (filters, 4, 4)

	generator.add(Conv2DTranspose(int(filters/2),
							  kernel_size=(filtersize, filtersize),
							  strides=(2, 2),
							  dilation_rate = dilation_rate,
							  padding='same',
							  kernel_regularizer = reg))
	generator.add(BatchNormalization())
	#generator.add(  LeakyReLU())
	generator.add(  Activation('relu'))

	generator.add(  Dropout(dropout_rate))

	#generator.add(BatchNormalization())

	# output_shape = (filters/2,8,8 )

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

	#generator.add(BatchNormalization())
	# output_shape = (filters/2,16,16 )
	generator.add(Conv2DTranspose(int(filters/8),
							kernel_size=(filtersize, filtersize),
							strides=(2, 2),
							dilation_rate = dilation_rate,
							padding='same',
							kernel_regularizer = reg))
	generator.add(BatchNormalization())
	generator.add(  Activation('relu'))

	#generator.add(  LeakyReLU())
	generator.add(  Dropout(dropout_rate))


	# output_shape = (filters/4,32,32 )

	if image_dim == 128:
		generator.add(Conv2DTranspose(int(filters/16),
								kernel_size=(filtersize, filtersize),
								strides=(2, 2),
								padding='same',
								dilation_rate = dilation_rate,
								kernel_regularizer = reg))
		#generator.add(BatchNormalization())
		generator.add(  Activation('relu'))

		#generator.add(  LeakyReLU())
		generator.add(BatchNormalization())

	generator.add(Conv2DTranspose(3,
							kernel_size=(filtersize, filtersize),
							strides = (2,2),
							dilation_rate = dilation_rate,
							padding='same',
							activation='tanh',
							kernel_regularizer = reg))
	#print(generator.output_shape)
	return generator


def get_discriminator(input_dim = 128, depth = 1,  filters = 256,filtersize = 5, regularisation = 1e-4, dropout_rate = 0.5, dilation_rate = 1,batch_norm = False, out_dim = 2):

    reg = l2(regularisation)

    discriminator = Sequential()
    discriminator.add(  Conv2D(int(filters/8),
    					(filtersize, filtersize),
    					strides = (2,2),
    					padding='same',
    					dilation_rate = dilation_rate,
    					input_shape = (3,input_dim,input_dim),
    					kernel_regularizer = reg))
    #output_shape = (64,64)
    if depth == 2:
        discriminator.add(  Conv2D(int(filters/8),
    						(filtersize, filtersize),
    						padding='same',
    						dilation_rate = dilation_rate,
    						input_shape = (3,input_dim,input_dim),
    						kernel_regularizer = reg))
    	#output_shape = (64,64)


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
    #output_shape = (32,32)

    if batch_norm:
    	discriminator.add(  BatchNormalization())
    discriminator.add(  LeakyReLU())
    discriminator.add(  Dropout(dropout_rate))

    if depth == 2:
        discriminator.add(  Conv2D(int(filters/4),
    						(filtersize, filtersize),
    						padding='same',
    						dilation_rate = dilation_rate,
    						input_shape = (3,input_dim,input_dim),
    						kernel_regularizer = reg))

    discriminator.add(  Conv2D(int(filters/2),
    					(filtersize, filtersize),
    					strides = (2,2),
    					dilation_rate = dilation_rate,
    					padding='same',
    					kernel_regularizer = reg))

    if depth == 2:
        discriminator.add(  Conv2D(int(filters/4),
    						(filtersize, filtersize),
    						padding='same',
    						dilation_rate = dilation_rate,
    						input_shape = (3,input_dim,input_dim),
    						kernel_regularizer = reg))

    #output_shape = (16,16)
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
    if depth == 2:
        discriminator.add(  Conv2D(int(filters),
    						(filtersize, filtersize),
    						padding='same',
    						dilation_rate = dilation_rate,
    						input_shape = (3,input_dim,input_dim),
    						kernel_regularizer = reg))
    #output_shape = (16,16)
    if batch_norm:
    	discriminator.add(  BatchNormalization())
    discriminator.add(  LeakyReLU())
    discriminator.add(  Dropout(dropout_rate))

    #discriminator.add(Flatten())
    #discriminator.add(Dense(100 ,activation = 'relu', kernel_regularizer = reg))
    '''
    if input_dim == 128:
    	discriminator.add(  Conv2D(filters, (5, 5), strides = (2,2),padding='same',kernel_regularizer = reg))
    	#output_shape = (4,4)
    	discriminator.add(  BatchNormalization())
    	discriminator.add(  LeakyReLU())
    	discriminator.add(  Dropout(dropout_rate))
    	discriminator.add(  Conv2D(filters, (5, 5), padding='same',kernel_regularizer = reg))
    	discriminator.add(  BatchNormalization())
    	discriminator.add(  LeakyReLU())
    	discriminator.add(  Dropout(dropout_rate))
    discriminator.add(Conv2D(out_dim,(4,4), padding = 'valid', activation = 'sigmoid'))
    '''
    #discriminator.add(  Conv2D(out_dim, (4, 4), padding='valid',kernel_regularizer = reg))

    discriminator.add(Flatten())
    if out_dim ==2:
    	discriminator.add(  Dense(0))
    	discriminator.add(  Activation("softmax"))
    else:
        discriminator.add(  Dense(1))
        discriminator.add(  Activation("sigmoid"))#Dense(1, activation = "sigmoid"))
    return discriminator
