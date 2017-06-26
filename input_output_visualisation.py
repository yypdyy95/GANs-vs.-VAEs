import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import utilities as util
from networks import *

fig , ax = plt.subplots(1,2)
fig.set_size_inches(12,8)

'''
conv_layers = []
for layer in model.layers:
    if "conv" in layer.get_config()['name']:
        conv_layers.append(layer)
#conv_layers[:] = model.layers["onv" in model.layers[:].get_config()['name']]
util.visualize_weights(conv_layers[0])
'''

#model = load_model('./networks/gen_de_f1024_fs5_o1_d0.5_dil1.h5', custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})
model = load_model('./example_networks/gen_de_f512_fs5_o1_d0.5_dil1.h5', custom_objects = {'binary_accuracy_':binary_accuracy_, 'mean_pred':mean_pred, 'min_pred':min_pred})

noise_0 = np.random.normal(size = (1,100))
noise_dif = np.random.normal(size = (1,100))
noise = np.zeros((10,100))

def onclick(event):
    global noise_0
    global noise
    col  = int(event.xdata)
    line = int(event.ydata)
    steps = 10

    noise_dif = np.random.normal(size = (1,100))

    dt = np.linspace(0.0, 1.0, num = steps)
    for i in range(steps):
        noise[i] = noise_0 * np.sin((1-dt[i])) + np.sin(dt[i])*noise_dif
    fake_ims = model.predict(noise)

    fake_ims = fake_ims.transpose((0,2,3,1))
    fake_ims = ((fake_ims+1)*127.5).astype(np.uint8)
    for i in range(steps):
        
        ax[0].cla()
        ax[1].cla()
        ax[0].imshow(np.reshape(noise[i],(10,10)), cmap = 'Greys_r')
        ax[1].imshow(fake_ims[i])
        plt.pause(1e-1)


    noise_0 = noise_dif
    plt.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
fake_im = model.predict(noise_0)[0]
fake_im = np.swapaxes(fake_im, 0,1)
fake_im = ((np.swapaxes(fake_im, 2,1) + 1) * 127.5).astype(np.uint8)
ax[0].imshow(np.reshape(noise_0,(10,10)), cmap = 'Greys_r')
ax[1].imshow(fake_im)

plt.draw()
plt.show()
