import PIL
from PIL import Image
import os
import numpy as np
import gzip

"""
    convert to channels_last format
"""
def convert_image_cf(img):
    return np.swapaxes(np.swapaxes(np.array(img), 2, 1), 1, 0)

"""
    convert to channels_last format
"""
def convert_image_cl(img):
    return np.swapaxes(np.swapaxes(np.array(img), 0, 1), 1, 2)

"""
    resize the image
"""
def resize_image(img, width, height):
    hsize = height
    return img.resize((width,hsize), PIL.Image.ANTIALIAS)

"""
    load the data set from path with the specified name
arguments:
    - name:      one of 'cats', 'dogs' or 'celeb'
    - maxitems:  number of images to be loaded
    - image_dim: image dimesion = (3, image_dim, image_dim)
    - path:      relative or absolute path to the datasets directory in which the raw and final data is saved/will be saved
"""
def load_data_set(name, maxitems=12500, image_dim = 64, path = "./../../datasets/"):
    import dill

    def internal_load_data(maxitems=10000):
        import glob
        from scipy import misc

        filenames = glob.glob(os.path.join(path, name +"/*.jpg"))
        images = np.empty((maxitems, 3, image_dim, image_dim), dtype=np.uint8)

        for i in range(maxitems):
            img = Image.open(filenames[i])

            img = resize_image(img, image_dim, image_dim)
            img = convert_image_cf(img)
            images[i] = img

        return images

    if (os.path.isfile(os.path.join(path, name + ".pkl"))):
        with open(os.path.isfile(os.path.join(path, name + ".pkl")), 'rb') as f:
            data = dill.load(f)
    else:
        data = internal_load_data(maxitems)
        with open(os.path.join(path, name + '.pkl'), 'wb') as f:
            dill.dump(data, f)

    return data

"""
    load the data set and the class labels specified in a csv file
    arguments:
        - name:         one of 'cats', 'dogs' or 'celeb'
        - maxitems:     number of images to be loaded
        - image_dim:    image dimesion = (3, image_dim, image_dim)
        - path:         relative or absolute path to the datasets directory in which the raw and final data is saved/will be saved
        - class_names:  listof class names which will be loaded from the annotations.csv file
"""
def load_data_set_classes(name, class_names, maxitems=12500, image_dim = 64, cf=False, path="./../datasets/", header=0, delimiter=' '):
    import dill
    import pandas

    def internal_load_data(maxitems=10000):
        import glob
        from scipy import misc

        csv = pandas.read_csv(os.path.join(path, name + "/annotations.csv"), header=header, delimiter=delimiter)

        filenames = glob.glob(os.path.join(path, name + "/images/*.jpg"))

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

            if i % 100 == 0:
                print(i)

        return images, classes

    if (os.path.isfile(os.path.join(path, name + ".pkl"))):
        with gzip.GzipFile(os.path.join(path, name + '.pkl)', 'rb') as f:
            data = dill.load(f)
        with gzip.GzipFile(os.path.join(path, name + '.classes.pkl'), 'rb') as f:
            classes = dill.load(f)
    else:
        data, classes = internal_load_data(maxitems)
        with gzip.GzipFile(os.path.join(path, name + '.pkl'), 'wb') as f:
            dill.dump(data, f)
        with gzip.GzipFile(os.path.join(path, name + '.classes.pkl'), 'wb') as f:
            dill.dump(classes, f)

    return data, classes

"""
    load the char74k data set
    arguments:
        - image_dim: image dimesion = (3, image_dim, image_dim)
        - path:      relative or absolute path to the datasets directory in which the raw and final data is saved/will be saved
"""
def load_char74k(image_dim = 64, cf=False, path="./../../datasets/"):
    import dill

    path += "/char74k/"

    def internal_load_data():
        image_path = path + "images"

        #count all images
        import os
        nb_files = sum([len(files) for r, d, files in os.walk(image_path)])

        if cf:
            images = np.empty((nb_files, 3, image_dim, image_dim), dtype=np.uint8)
        else:
            images = np.empty((nb_files, image_dim, image_dim, 3), dtype=np.uint8)
        y = np.empty((nb_files, 2))

        def read_images(image_type, name):
            nonlocal index
            for r, d, files in os.walk(image_path + "/" + name):
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

    if (os.path.isfile(path + "char74k.pkl")):
        with gzip.GzipFile(path + "char74k.pkl", 'rb') as f:
            images = dill.load(f)
        with gzip.GzipFile(path + "char74k.classes.pkl", 'rb') as f:
            y = dill.load(f)
    else:
        images, y = internal_load_data()
        with gzip.GzipFile(path + "char74k.pkl", 'wb') as f:
            dill.dump(images, f)
        with gzip.GzipFile(path + "char74k.classes.pkl", 'wb') as f:
            dill.dump(y, f)

    return images, y
