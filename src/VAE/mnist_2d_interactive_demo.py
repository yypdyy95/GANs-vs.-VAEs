import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from scipy.stats import norm
from keras.datasets import mnist, cifar10

#improves the output of keras on Windows
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

class SnaptoCursor(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, ax, prediction_imgobj):
        self.x = np.arange(10)
        self.y = np.arange(10)

        self.prediction_imgobj = prediction_imgobj

    def mouse_move(self, event):
        if not event.inaxes:
            return

        if event.button == 1:
            self.onclick(event)

    def onclick(self, event):
        x, y = event.xdata, event.ydata
        print(x)

        z_sample = np.array([[y, x]])
        fake_im = model.predict([z_sample])[0]

        self.prediction_imgobj.set_data(fake_im.reshape((28, 28)))

        plt.draw()

fig , ax = plt.subplots(1,2)
fig.set_size_inches(12,6)
fig.tight_layout()

for x in ax:
    x.set_axis_off()

X, Y = np.meshgrid(np.linspace(-5, 5, 400),np.linspace(-5, 5, 400))

latent_data = norm.pdf(X)*norm.pdf(Y)

model = load_model('./output/generator_uncon_mnist_2.h5')

encoder = load_model('./output/encoder_uncon_mnist_2.h5')


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
encoded_images = encoder.predict(x_test.reshape(-1, 28, 28, 1), batch_size=100)
s = ax[0].scatter(encoded_images[:, 0], encoded_images[:, 1], c=y_test)
cbaxes = fig.add_axes([0.02, 0.1, 0.02, 0.8])
fig.colorbar(s, orientation='vertical', cax = cbaxes)

z_sample = np.array([[0.0, 0.0]])
fake_im = model.predict([z_sample])[0]
prediciton_imgobj = ax[1].imshow(fake_im.reshape((28, 28)))

ax[0].set_title("Latent Space")
ax[1].set_title("Prediction")

cursor = SnaptoCursor(ax[0], prediciton_imgobj)
plt.connect('motion_notify_event', cursor.mouse_move)
plt.connect('button_press_event', cursor.onclick)

plt.draw()
plt.show()
