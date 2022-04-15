# This is the evaluation code for salicon.
#
import argparse
import os
import pathlib as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
import numpy as np
from skimage import filters
import skimage.io as sio
import SSETM

random.seed(100)
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--image_model_path', type=pl.Path, default='SSETM_mse/model27500.pth.tar',
                    help='the path of the saved model')
parser.add_argument('--img_path', type=pl.Path, default='../salicon/images/val',
                    help='the folder of salicon data')
parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')
parser.add_argument('--size', default=(480, 640), type=tuple,
                    help='image size of SALICON.')
parser.add_argument('--save_segmentation', action="store_true", default=True,
                    help='apply MSE as a loss function')

args = parser.parse_args()

def normalize(x):
    x -= x.min()
    x /= x.max()

def post_process(pred):
    pred = filters.gaussian(pred, 5)
    normalize(pred)
    pred = (pred * 255).astype(np.uint8)
    return pred

# for segmentation visualization.
colors = [[random.randint(0,255), random.randint(0,255), random.randint(0,255)] for i in range(21)]

def test():
    global args

    preprocess = transforms.Compose([
        transforms.Resize(args.size),
	    transforms.ToTensor(),
    ])

    output = '../salicon_output'
    os.makedirs(output, exist_ok=True)
    model = SSETM.model_test(args.image_model_path).cuda().eval()
    lst = os.listdir(args.img_path)
    for i in lst:
        image_path = os.path.join(args.img_path, i)
        pil_img = Image.open(image_path).convert('RGB')
        processed = preprocess(pil_img).unsqueeze(0).cuda()

        with torch.no_grad():
            pred, _, _, _, y = model([processed, processed])

        pred = pred.squeeze().detach().cpu().numpy()
        pred = post_process(pred)

        pred_path = i.split('.')[0] + ".png"
        pred_path = os.path.join(output, pred_path)
        sio.imsave(pred_path, pred)

        if args.save_segmentation:
            y = y.data.max(1)[1].cpu().numpy()[:, :, :][0]
            r = y.copy()
            g = y.copy()
            b = y.copy()
            for l in range(0, 21):
                r[y == l] = colors[l][ 0]
                g[y == l] = colors[l][ 1]
                b[y == l] = colors[l][ 2]
            y = np.concatenate((np.expand_dims(b,axis=-1),np.expand_dims(g,axis=-1),
                                  np.expand_dims(r,axis=-1)),axis=-1)
            pred_path = i.split('.')[0]+'_seg.png'
            pred_path = os.path.join(output, pred_path)
            cv2.imwrite(pred_path, y)

if __name__ == '__main__':
    test()
