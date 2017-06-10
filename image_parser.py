import PIL
from PIL import Image
import os
import numpy as np
def convert_image_cf(img):
    return np.swapaxes(np.swapaxes(np.array(img), 2, 1), 1, 0)

def convert_image_cl(img):
    return np.swapaxes(np.swapaxes(np.array(img), 0, 1), 1, 2)

def cats_load_data_set(maxitems=2500, image_dim = 64):
    import dill
    def resize_image(img, width, height):
        #wpercent = (width/float(img.size[0]))
        hsize = height#48# int((float(img.size[1])*float(wpercent)))
        return img.resize((width,hsize), PIL.Image.ANTIALIAS)

    def internal_cats_load_data(maxitems=10000):
        import glob
        from scipy import misc

        filenames = glob.glob("./dvsc_train/cat*.jpg")
        images = np.empty((maxitems, 3, image_dim, image_dim), dtype=np.uint8)

        for i in range(maxitems):
            img = resize_image(img, image_dim, image_dim)
            img = Image.open(filenames[i])
            img = convert_image_cf(img)
            images[i] = img

        return images

    if (os.path.isfile("cats.pkl")):
        with open('cats.pkl', 'rb') as f:
            data = dill.load(f)
    else:
        data = internal_cats_load_data(maxitems)
        with open('cats.pkl', 'wb') as f:
            dill.dump(data, f)

    return data
