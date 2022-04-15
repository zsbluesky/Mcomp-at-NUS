# This is the code to train SSETM for saliency prediction.
# The method is demonstrated in the dissertation, Semantic Segmentation Enhanced Transformer Model for Human Prediction (SSETM).
#
import argparse
from distutils.version import LooseVersion
from itertools import cycle
import os.path as osp
import pathlib as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import voc
import SaliconLoader
import SSETM

parser = argparse.ArgumentParser(description='SSETM training')
parser.add_argument('--data_folder', type=pl.Path, default='../salicon',
                    help='the folder of salicon data')
parser.add_argument('--output_folder', type=str, default='SSETM',
                    help='the folder used to save the trained model')
parser.add_argument('--model_path', default='backbone/R50+ViT-B_16.npz', type=pl.Path,
                    help='the path of the pre-trained hybrid transformer model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=4, type=int, metavar='N',
                    help='number of total epochs (of PASCAL VOC) to run')
parser.add_argument('--decay_epoch', default=(2,), type=int, metavar='N',
                    help='epochs when lr decays')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size for saliency prediction (default is 8!!)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='losses printing frequency')
parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')

args = parser.parse_args()
global_step = 0 # record the total iteration steps

def main():
    global args

    model = SSETM.model(args.model_path).cuda()
    # if use multiple gpus:
    model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Data loading code
    # 1. dataloader for saliency prediction
    train_loader = torch.utils.data.DataLoader(
        SaliconLoader.ImageList(args.data_folder, transforms.Compose([
            transforms.ToTensor(),
        ]),
        train=True,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    # 2. dataloader for semantic segmentation, batch size is fixed to 1, because of the
    # format of PASCAL annotation.
    root = osp.expanduser('..')
    cuda = True
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    seg_loader = torch.utils.data.DataLoader(
        voc.SBDClassSeg(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)

    # output folder
    args.output_folder = args.output_folder + "_mse"
    args.output_folder = pl.Path(args.output_folder)
    if not args.output_folder.is_dir():
        args.output_folder.mkdir()

    # Loss for saliency prediction
    criterion = nn.MSELoss().cuda()

    # start training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch, seg_loader)

    # save the final model
    final = {'state_dict' : model.state_dict()}
    save_path = args.output_folder / ("model.pth.tar")
    save_model(final, save_path)

def save_model(state, path):
    torch.save(state, path)

# loss for semantic segmentation
def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def train(train_loader, model, criterion, optimizer, epoch, seg_loader):
    global global_step
    global args

    model.train()
    loss_saliency = 0
    loss_segmentation = 0

    # iterations, where total steps of one epoch of PASCAL VOC is much more than saliency prediction.
    for i, ((input, fixmap, smap), (img, seg)) in enumerate(zip(cycle(train_loader), seg_loader)):
        global_step += 1

        # input and GT for saliency prediction.
        input = input.cuda()
        fixmap = fixmap.cuda()
        smap = smap.cuda()
        # img: input for semantic segmentation, seg: GT.
        img = img.cuda()
        seg = seg.cuda()

        # need to modify ".repeat(3, 1, 1, 1)" according to the number of gpus used.
        de1, de2, de3, de4, pre = model([input, img.repeat(3, 1, 1, 1)])

        # multiple supervisions for saliency prediction and CE loss for segmentation.
        loss_sal = criterion(de1, smap)+criterion(de2, smap)/2+criterion(de3, smap)/4+criterion(de4, smap)/8
        loss_seg = cross_entropy2d(pre[0:1], seg)
        loss = loss_sal + 0.1*loss_seg
        loss_saliency += loss_sal.item()
        loss_segmentation += loss_seg.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(seg_loader)), end='')
            print("loss1: {}\t".format(loss_saliency/args.print_freq), end='')
            print("loss2: {}".format(loss_segmentation/args.print_freq))
            loss_saliency = 0
            loss_segmentation = 0

        # if use default batch size 8, the number of iterations of saliency training is about 1250 (10000//8),
        # we save the model after every 11 epochs running of salicon.
        if global_step % (1250*11) == 0:
            state = {'state_dict' : model.state_dict()}
            save_path = args.output_folder / ("model{0}.pth.tar".format(global_step))
            save_model(state, save_path)


# loss recording
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# lr decay
def adjust_learning_rate(optimizer, epoch):
    for i in range(len(args.decay_epoch)):
        if epoch == args.decay_epoch[i]:
            factor = i+1
            lr = args.lr*(0.1**factor)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == '__main__':
    main()
