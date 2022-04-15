
from PIL import Image

import os, numpy as np

def resize_img(im_t):
    im = im_t.copy()
    (x, y) = im.size  # read image size
    x, y = y, x
    if x >= y*3/4:
        w = round(x*4/3)
        out = np.array(im)
        out = np.pad(out, ((0,0), (0,w-y)), 'constant', constant_values = (0,0))
    else:
        w = y
        h = round(w * 3/4)
        out = np.array(im)
        out = np.pad(out, ((0,h-x), (0,0)), 'constant', constant_values = (0,0))
    im = Image.fromarray(out)
    out = im.resize((640, 480))  # resize image
    return out


def resize_img2(im_t):
    im = im_t.copy()
    (x, y) = im.size  # read image size
    x, y = y, x
    if x >= y*3/4:
        w = round(x*4/3)
        out = np.array(im)
        out = np.pad(out, ((0,0), (0,w-y), (0,0)), 'constant', constant_values = (0,0))
    else:
        w = y
        h = round(w * 3/4)
        out = np.array(im)
        out = np.pad(out, ((0,h-x), (0,0), (0,0)), 'constant', constant_values = (0,0))
    im = Image.fromarray(out)
    out = im.resize((640, 480))  # resize image
    return out


fs = os.listdir('images')
for f in fs:
    im = Image.open(os.path.join('images', f))
    im = resize_img2(im, 1)
    im.save('processed/images/'+f)
fs = os.listdir('maps')
for f in fs:
    im = Image.open(os.path.join('maps', f))
    im = resize_img(im, 1)
    im.save('processed/maps/'+f)

