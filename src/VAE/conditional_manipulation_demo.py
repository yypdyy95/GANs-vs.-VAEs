"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from scipy.stats import norm
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from matplotlib.widgets import CheckButtons
import PIL

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

fig , ax = plt.subplots(1,2)
fig.set_size_inches(12,6)
fig.tight_layout()

for x in ax:
    x.set_axis_off()

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        #xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        #kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return metrics.binary_crossentropy(x, x_decoded_mean_squash)#K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


male, blond_hair, smiling, wearing_hat = False, False, False, False

model = load_model('./output/generator_celeba_300_200.h5')
#vae = load_model('./output/vae_celeba_300_200.h5', custom_objects={"img_rows": 64, "img_cols":64, "batch_size":100, "latent_dim": 300, "epsilon_std": 1.0, "CustomVariationalLayer": CustomVariationalLayer})
encoder = load_model('./output/encoder_celeba_300_200.h5')

image_imgobj = None
prediction_imgobj = None
def load_image():
    global image_imgobj, images, img
    global male, blond_hair, smiling, wearing_hat

    def encode_image():
        global encoded_images
        encoded_images = encoder.predict(images, batch_size=100)

    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()
    img_data = Image.open(image_path).resize((64,64), PIL.Image.ANTIALIAS)

    img = np.array(list(img_data.getdata())).reshape((64, 64, 3))
    img = img.astype('float32') / 255.
    images = np.tile(img, 100).reshape(100, 64, 64, 3)
    images = np.empty((100, 64, 64, 3))
    for i in range(100):
        images[i] = img.copy()

    print(images.shape)

    if image_imgobj is None:
        image_imgobj = ax[0].imshow(images[0])
    else:
        image_imgobj.set_data(images[0])

    encode_image()
    male, blond_hair, smiling, wearing_hat = False, False, False, False
    update_prediction()



def update_prediction():
    global prediction_imgobj

    conditional_sample = np.zeros((100, 4))
    if male:
        conditional_sample[:, 0] = 1
    if blond_hair:
        conditional_sample[:, 1] = 1
    if smiling:
        conditional_sample[:, 2] = 1
    if wearing_hat:
        conditional_sample[:, 3] = 1

    prediction_img = model.predict([encoded_images, conditional_sample], batch_size=100)[0]
    #prediction_img = vae.predict([images, conditional_sample], batch_size=100)[0]

    if prediction_imgobj is None:
        prediction_imgobj = ax[1].imshow(prediction_img)
    else:
        prediction_imgobj.set_data(prediction_img)
    plt.draw()

load_image()

checkboxes_ax = plt.axes([0.01, 0.01, 0.1, 0.15])
checkboxes = CheckButtons(checkboxes_ax, ("Male", "Blond Hair", "Smiling", "Wearing Hat"), (male, blond_hair, smiling, wearing_hat))

def checkboxes_onclick(label):
    global male, blond_hair, smiling, wearing_hat

    if label == 'Male':
        male = not male
    elif label == 'Blond Hair':
        blond_hair = not blond_hair
    elif label == 'Smiling':
        smiling = not smiling
    elif label == 'Wearing Hat':
        wearing_hat = not wearing_hat
    update_prediction()
    plt.draw()
checkboxes.on_clicked(checkboxes_onclick)


plt.draw()
plt.show()
