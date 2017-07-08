import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import utilities as util
from networks import *

class SnaptoCursor(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, ax, noise_imgobj, prediction_imgobj):
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = np.arange(10)
        self.y = np.arange(10)

        self.alphas = np.arange(0, 256, 15)/255.0
        self.values = np.linspace(-2, 2, 17)
        self.current_alpha = 0

        self.noise_imgobj = noise_imgobj
        self.prediction_imgobj = prediction_imgobj

    def scroll(self, event):
        if event.button == "up":
            self.current_alpha = (self.current_alpha + 1) % len(self.alphas)
        elif event.button == "down":
            self.current_alpha = (self.current_alpha - 1) % len(self.alphas)

        color = (self.alphas[self.current_alpha], self.alphas[self.current_alpha], self.alphas[self.current_alpha])
        self.lx.set_color(color)
        self.ly.set_color(color)

        plt.draw()

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata-0.49, event.ydata-0.49

        indx = np.searchsorted(self.x, x)
        indy = np.searchsorted(self.y, y)

        if indx >= len(self.x) or indy >= len(self.y):
            return

        x = self.x[indx]
        y = self.y[indy]
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        plt.draw()

        if event.button == 1:
            self.onclick(event)

    def onclick(self, event):
        noise = self.noise_imgobj.get_array()

        x, y = event.xdata-0.49, event.ydata-0.49
        indx = np.searchsorted(self.x, x)
        indy = np.searchsorted(self.y, y)

        noise[indy, indx] = self.values[self.current_alpha]

        self.noise_imgobj.set_data(noise)

        fake_ims = model.predict([noise.reshape(1, 100)])

        fake_ims = fake_ims.transpose((0,2,3,1))
        fake_im = ((fake_ims+1)*127.5).astype(np.uint8)[0]

        self.prediction_imgobj.set_data(fake_im)

        plt.draw()

fig , ax = plt.subplots(1,2)
fig.set_size_inches(12,6)
fig.tight_layout()

model = load_model('./example_networks/gen_de_f512_fs5_o1_d0.5_dil1.h5', custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})

noise = np.clip(np.random.normal(size = (1,100)), -2, 2)

fake_im = model.predict(noise)[0]
fake_im = np.swapaxes(fake_im, 0,1)
fake_im = ((np.swapaxes(fake_im, 2,1) + 1) * 127.5).astype(np.uint8)

noise_imgobj = ax[0].imshow(np.reshape(noise,(10,10)), cmap = 'Greys_r')
prediciton_imgobj = ax[1].imshow(fake_im)

ax[0].set_title("Input Noise")
ax[1].set_title("Prediction")

cursor = SnaptoCursor(ax[0], noise_imgobj, prediciton_imgobj)
plt.connect('motion_notify_event', cursor.mouse_move)
plt.connect('scroll_event', cursor.scroll)
plt.connect('button_press_event', cursor.onclick)

plt.draw()
plt.show()