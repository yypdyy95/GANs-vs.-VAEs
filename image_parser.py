import PIL
from PIL import Image
import os
import numpy as np
'''
convert to channels_last format
'''
def convert_image_cf(img):
    return np.swapaxes(np.swapaxes(np.array(img), 2, 1), 1, 0)
'''
convert to channels_last format
'''
def convert_image_cl(img):
    return np.swapaxes(np.swapaxes(np.array(img), 0, 1), 1, 2)

'''
load_data_set:
arguments:
    - name:      one of 'cats', 'dogs' or 'celeb'
    - maxitems:  number of images to be loaded
    - image_dim: image dimesion = (3, image_dim, image_dim)
'''

def load_data_set(name, maxitems=12500, image_dim = 64):
    import dill
    def resize_image(img, width, height):

        hsize = height
        return img.resize((width,hsize), PIL.Image.ANTIALIAS)

    def internal_load_data(maxitems=10000):
        import glob
        from scipy import misc

        filenames = glob.glob("./../datasets/"+ name +"/*.jpg")
        images = np.empty((maxitems, 3, image_dim, image_dim), dtype=np.uint8)

        for i in range(maxitems):
            img = Image.open(filenames[i])

            img = resize_image(img, image_dim, image_dim)
            img = convert_image_cf(img)
            images[i] = img

        return images

    if (os.path.isfile("./../datasets/" + name + ".pkl")):
        with open("./../datasets/" + name + '.pkl', 'rb') as f:
            data = dill.load(f)
    else:
        data = internal_load_data(maxitems)
        with open("./../datasets/" + name + '.pkl', 'wb') as f:
            dill.dump(data, f)

    return data

def load_data_set_classes(name, class_name, maxitems=12500, image_dim = 64, cf=False):
    import dill
    import pandas

    def resize_image(img, width, height):
        hsize = height
        return img.resize((width,hsize), PIL.Image.ANTIALIAS)

    def internal_load_data(maxitems=10000):
        import glob
        from scipy import misc

        csv = pandas.read_csv("./../datasets/celeba/annotations.csv", header=0, delimiter=' ')

        filenames = glob.glob("./../datasets/"+ name + "/img_align_celeba/*.jpg")

        if cf:
            images = np.empty((maxitems, 3, image_dim, image_dim), dtype=np.uint8)
        else:
            images = np.empty((maxitems, image_dim, image_dim, 3), dtype=np.uint8)
        classes = np.empty((maxitems, 1))

        for i in range(maxitems):
            img = Image.open(filenames[i])

            img = resize_image(img, image_dim, image_dim)
            if cf:
                img = convert_image_cf(img)
            images[i] = img
            classes[i, 0] = csv[class_name][i]

        return images, classes

    if (os.path.isfile("./../datasets/" + name + ".pkl")):
        with open("./../datasets/" + name + '.pkl', 'rb') as f:
            data = dill.load(f)
        with open("./../datasets/" + name + '.classes.pkl', 'rb') as f:
            classes = dill.load(f)
    else:
        data, classes = internal_load_data(maxitems)
        with open("./../datasets/" + name + '.pkl', 'wb') as f:
            dill.dump(data, f)
        with open("./../datasets/" + name + '.classes.pkl', 'wb') as f:
            dill.dump(classes, f)

    return data, classes
