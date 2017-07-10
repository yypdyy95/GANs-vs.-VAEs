import PIL
from PIL import Image
import os
import numpy as np
import gzip
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

def resize_image(img, width, height):
    hsize = height
    return img.resize((width,hsize), PIL.Image.ANTIALIAS)

'''
load_data_set:
arguments:
    - name:      one of 'cats', 'dogs' or 'celeb'
    - maxitems:  number of images to be loaded
    - image_dim: image dimesion = (3, image_dim, image_dim)
'''

def load_data_set(name, maxitems=12500, image_dim = 64):
    import dill

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

def load_data_set_classes(name, class_names, maxitems=12500, image_dim = 64, cf=False, header=0, delimiter=' '):
    import dill
    import pandas

    def internal_load_data(maxitems=10000):
        import glob
        from scipy import misc

        csv = pandas.read_csv("./../datasets/" + name + "/annotations.csv", header=header, delimiter=delimiter)

        filenames = glob.glob("./../datasets/"+ name + "/images/*.jpg")

        if cf:
            images = np.empty((maxitems, 3, image_dim, image_dim), dtype=np.uint8)
        else:
            images = np.empty((maxitems, image_dim, image_dim, 3), dtype=np.uint8)
        classes = np.empty((maxitems, len(class_names)))

        for i in range(maxitems):
            img = Image.open(filenames[i])

            img = resize_image(img, image_dim, image_dim)
            if cf:
                img = convert_image_cf(img)
            images[i] = img
            for n, class_name in enumerate(class_names):
                classes[i, n] = csv.loc[csv['name'] == os.path.basename(filenames[i])][class_name]

        return images, classes

    if (os.path.isfile("./../datasets/" + name + ".pkl")):
        with gzip.GzipFile("./../datasets/" + name + '.pkl', 'rb') as f:
            data = dill.load(f)
        with gzip.GzipFile("./../datasets/" + name + '.classes.pkl', 'rb') as f:
            classes = dill.load(f)
    else:
        data, classes = internal_load_data(maxitems)
        with gzip.GzipFile("./../datasets/" + name + '.pkl', 'wb') as f:
            dill.dump(data, f)
        with gzip.GzipFile("./../datasets/" + name + '.classes.pkl', 'wb') as f:
            dill.dump(classes, f)

    return data, classes


def load_char74k(image_dim = 64, cf=False):
    import dill

    def internal_load_data():
        path = "./../datasets/char74k/images"

        #count all images
        import os
        nb_files = sum([len(files) for r, d, files in os.walk(path)])

        if cf:
            images = np.empty((nb_files, 3, image_dim, image_dim), dtype=np.uint8)
        else:
            images = np.empty((nb_files, image_dim, image_dim, 3), dtype=np.uint8)
        y = np.empty((nb_files, 2))

        def read_images(image_type, name):
            nonlocal index
            for r, d, files in os.walk(path + "/" + name):
                for f in files:
                    img = Image.open(r + "/" + f)

                    img = resize_image(img, image_dim, image_dim)
                    if cf:
                        img = convert_image_cf(img)

                    if img.mode != "L":
                        images[index] = img
                    else:
                        images[index] = img.convert(mode="RGB")

                    y[index, 0] = int(r[-3:])
                    y[index, 1] = image_type

                    index += 1

        index = 0
        read_images(0, "Fnt")
        read_images(1, "Hnd")
        read_images(2, "Img")

        return images, y

    if (os.path.isfile("./../datasets/char74k.pkl")):
        with gzip.GzipFile("./../datasets/char74k.pkl", 'rb') as f:
            images = dill.load(f)
        with gzip.GzipFile("./../datasets/char74k.classes.pkl", 'rb') as f:
            y = dill.load(f)
    else:
        images, y = internal_load_data()
        with gzip.GzipFile("./../datasets/char74k.pkl", 'wb') as f:
            dill.dump(images, f)
        with gzip.GzipFile("./../datasets/char74k.classes.pkl", 'wb') as f:
            dill.dump(y, f)

    return images, y
