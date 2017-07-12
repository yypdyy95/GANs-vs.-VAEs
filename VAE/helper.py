import PIL
from PIL import Image
import numpy as np
import pickle as pickle
import os, sys

def convert_image_cf(img):
    return np.swapaxes(np.swapaxes(np.array(img), 2, 1), 1, 0)

def convert_image_cl(img):
    return np.swapaxes(np.swapaxes(np.array(img), 0, 1), 1, 2)

def celeba_load_data_set(maxitems=25000):
    import dill
    def resize_image(img, width, height):
        #wpercent = (width/float(img.size[0]))
        hsize = height#48# int((float(img.size[1])*float(wpercent)))
        return img.resize((width,hsize), PIL.Image.ANTIALIAS)

    def internal_celeba_load_data_set(maxitems=25000):
        import glob
        from scipy import misc

        filenames = glob.glob("./dvsc_train/*.jpg")
        images = np.empty((maxitems, 3, 96, 128), dtype=np.uint8)

        for i in range(maxitems):
            img = Image.open(filenames[i])
            img = resize_image(img, 128, 96)
            img = convert_image_cf(img)
            images[i] = img

        return images

    if os.path.isfile("celeba.pkl"):
        with open('celeba.pkl', 'rb') as f:
            data = dill.load(f)
    else:
        data = internal_celeba_load_data_set(maxitems)
        with open('celeba.pkl', 'wb') as f:
            dill.dump(data, f)

    return data
