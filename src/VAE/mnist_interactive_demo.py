import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from scipy.stats import norm

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

    def __init__(self, ax, noise_imgobj, prediction_imgobj):
        self.x = np.arange(10)
        self.y = np.arange(10)

        self.noise_imgobj = noise_imgobj
        self.prediction_imgobj = prediction_imgobj

        self.current_class = 0

    def scroll(self, event):
        if event.button == "up":
            self.current_class = (self.current_class + 1) % 10
        elif event.button == "down":
            self.current_class = (self.current_class - 1) % 10

        self.onclick(event)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        if event.button == 1:
            self.onclick(event)

    def onclick(self, event):
        noise = self.noise_imgobj.get_array()

        x, y = event.xdata / 100 - 0.5, event.ydata / 100 - 0.5
        print(x)

        z_sample = np.array([[y, x]])
        conditional_sample = np.zeros((100, 10))
        conditional_sample[:, self.current_class] = 1
        fake_im = model.predict([z_sample, conditional_sample])[0]

        self.prediction_imgobj.set_data(fake_im.reshape((28, 28)))

        plt.draw()

fig , ax = plt.subplots(1,2)
fig.set_size_inches(12,6)
fig.tight_layout()

for x in ax:
    x.set_axis_off()

X, Y = np.meshgrid(np.linspace(-2, 2, 100),np.linspace(-2, 2, 100))

latent_data = norm.pdf(X)*norm.pdf(Y)
latent_imgobj = ax[0].imshow(latent_data)

model = load_model('./output/generator_mnist_2.h5')

z_sample = np.array([[0.0, 0.0]])
conditional_sample = np.zeros((100, 10))
conditional_sample[:, 0] = 1
fake_im = model.predict([z_sample, conditional_sample])[0]
prediciton_imgobj = ax[1].imshow(fake_im.reshape((28, 28)))

ax[0].set_title("Latent Space")
ax[1].set_title("Prediction")

cursor = SnaptoCursor(ax[0], latent_imgobj, prediciton_imgobj)
plt.connect('motion_notify_event', cursor.mouse_move)
plt.connect('scroll_event', cursor.scroll)
plt.connect('button_press_event', cursor.onclick)

plt.draw()
plt.show()
