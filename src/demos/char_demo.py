import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
sys.path.append('./..')
sys.path.append('./../GAN/')
from keras.models import load_model
#import utilities as util
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

        self.ax = ax

    def scroll(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == "up":
            self.current_alpha = (self.current_alpha + 1) % len(self.alphas)
        elif event.button == "down":
            self.current_alpha = (self.current_alpha - 1) % len(self.alphas)

        color = (self.alphas[self.current_alpha], self.alphas[self.current_alpha], self.alphas[self.current_alpha])
        self.lx.set_color(color)
        self.ly.set_color(color)

        plt.draw()

    def mouse_move(self, event):
        if event.inaxes != self.ax:
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
        if event.inaxes != self.ax:
            return

        noise = self.noise_imgobj.get_array()

        x, y = event.xdata-0.49, event.ydata-0.49
        indx = np.searchsorted(self.x, x)
        indy = np.searchsorted(self.y, y)
        if indx <17:
          noise[indy, indx] = self.values[self.current_alpha]

        self.noise_imgobj.set_data(noise)

        currentclass = int(bnextclass.label.get_text())
        fake_ims = model.predict([noise.reshape(1, 100), np.array([currentclass])])

        fake_ims = fake_ims.transpose((0,2,3,1))
        fake_im = ((fake_ims+1)*127.5).astype(np.uint8)[0]

        self.prediction_imgobj.set_data(fake_im)

        plt.draw()

def shuffle(event):
    noise = np.clip(np.random.normal(size = (1,100)), -2, 2)
    noise_imgobj.set_data(noise.reshape(10, 10))

    currentclass = int(bnextclass.label.get_text())
    fake_ims = model.predict([noise.reshape(1, 100), np.array([currentclass])])

    fake_ims = fake_ims.transpose((0,2,3,1))
    fake_im = ((fake_ims+1)*127.5).astype(np.uint8)[0]

    prediction_imgobj.set_data(fake_im)

    plt.draw()


def nextclass(event):
    noise = noise_imgobj.get_array()

    currentclass = int(bnextclass.label.get_text())
    currentclass = (currentclass+1) % 64
    bnextclass.label.set_text(currentclass)

    fake_ims = model.predict([noise.reshape(1, 100), np.array([currentclass])])

    fake_ims = fake_ims.transpose((0,2,3,1))
    fake_im = ((fake_ims+1)*127.5).astype(np.uint8)[0]

    prediction_imgobj.set_data(fake_im)


    currentclassname = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B", 12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H", 18:"I", 19:"J", 20:"K", 21:"L", 22:"M", 23:"N", 24:"O", 25:"P", 26:"Q", 27:"R", 28:"S", 29:"T",30:"U", 31:"V", 32:"W", 33:"X", 34:"Y", 35:"Z", 36:"a", 37:"b", 38:"c", 39:"d", 40:"e", 41:"f", 42:"g", 43:"h", 44:"i", 45:"j", 46:"k", 47:"l", 48:"m", 49:"n", 50:"o", 51:"p", 52:"q", 53:"r", 54:"s", 55:"t", 56:"u", 57:"v", 58:"w", 59:"x", 60:"y", 61:"z" }[currentclass]

    ax[1].set_title("Prediction ({0})".format(currentclassname))
    plt.draw()

fig , ax = plt.subplots(1,2)
fig.set_size_inches(12,6)
fig.tight_layout()

model = load_model('./output/models/char/char_gen.h5', custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})

noise = np.clip(np.random.normal(size = (1,100)), -2, 2)

fake_im = model.predict([noise, np.array([0])])[0]
fake_im = np.swapaxes(fake_im, 0,1)
fake_im = ((np.swapaxes(fake_im, 2,1) + 1) * 127.5).astype(np.uint8)

noise_imgobj = ax[0].imshow(np.reshape(noise,(10,10)), cmap = 'Greys_r')
prediction_imgobj = ax[1].imshow(fake_im)

ax[0].set_title("Input Noise")
ax[1].set_title("Prediction (0)")

cursor = SnaptoCursor(ax[0], noise_imgobj, prediction_imgobj)
plt.connect('motion_notify_event', cursor.mouse_move)
plt.connect('scroll_event', cursor.scroll)
plt.connect('button_press_event', cursor.onclick)

from matplotlib.widgets import Button
axnextclass = plt.axes([0.495, 0.51, 0.030, 0.05])
bnextclass = Button(axnextclass, 0)
bnextclass.on_clicked(nextclass)

axshuffle = plt.axes([0.495, 0.45, 0.030, 0.05])
bshuffle = Button(axshuffle, '↻')
bshuffle.on_clicked(shuffle)


ax[0].set_axis_off()
ax[1].set_axis_off()

plt.draw()
plt.show()