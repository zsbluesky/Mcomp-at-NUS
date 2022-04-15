# This is the evaluation code to output prediction using our saliency model.
#
import argparse
import os
import pathlib as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
from skimage import filters
import skimage.io as sio
import skimage
import SSETM


random.seed(10)

parser = argparse.ArgumentParser(description='Evaluation on MIT1003')

parser.add_argument('--image_model_path', type=pl.Path, default='mit_mse/model5.pth.tar',
                    help='the path of the pre-trained model based on ImageNet')
parser.add_argument('--img_path', type=pl.Path, default='../mitdata/val/images',
                    help='the folder of mit1003 data')
parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')
parser.add_argument('--size', default=(480, 640), type=tuple,
                    help='resize the input image, (640,480) is from the training data, SALICON.')

args = parser.parse_args()


def normalize(x):
    x -= x.min()
    x /= x.max()

def post_process(pred, h, w, pad):
    pred = skimage.transform.resize(pred, (h, w))
    if pad[0] == 0 and pad[1] > 0:
        pred = pred[ :, :-pad[1]]
    elif pad[0] == 1 and pad[1] > 0:
        pred = pred[ :-pad[1], :]
    pred = filters.gaussian(pred, 5)
    normalize(pred)
    pred = (pred * 255).astype(np.uint8)
    return pred

def resize_img(im_t):
    im = im_t.copy()
    (x, y) = im.size  # read image size
    x, y = y, x
    if x >= y*3/4:
        h = x
        w = round(x*4/3)
        out = np.array(im)
        out = np.pad(out, ((0,0), (0,w-y), (0,0)), 'constant', constant_values = (0,0))
        pad = [0, w-y]
    else:
        w = y
        h = round(w * 3/4)
        out = np.array(im)
        out = np.pad(out, ((0,h-x), (0,0), (0,0)), 'constant', constant_values = (0,0))
        pad = [1, h-x]
    im = Image.fromarray(out)
    w, h = im.size
    out = im.resize((640, 480))  # resize image
    return out, h, w, pad

def main():
    global args

    preprocess = transforms.Compose([
	    transforms.ToTensor(),
    ])

    output = '../mittest'
    os.makedirs(output, exist_ok=True)
    img_model = SSETM.model_test(args.image_model_path).cuda().eval()
    lst = os.listdir(args.img_path)
    for i in lst:
        image_path = os.path.join(args.img_path, i)
        pil_img = Image.open(image_path).convert('RGB')
        pil_img, h, w, pad = resize_img(pil_img, 0)
        processed = preprocess(pil_img).unsqueeze(0).cuda()
        with torch.no_grad():
            img_feat, _, _, _, _ = img_model([processed, processed])
            pred = img_feat
        pred = pred.squeeze().detach().cpu().numpy()
        pred = post_process(pred, h, w, pad)
        pred_path = i.split('.')[0] + ".png"
        pred_path = os.path.join(output, pred_path)
        sio.imsave(pred_path, pred)

if __name__ == '__main__':
    main()
