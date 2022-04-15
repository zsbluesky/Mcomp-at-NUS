# This is the code to train on MIT1003 dataset.
#
import argparse
import os
import pathlib as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import SaliconLoader_mit
import SSETM

parser = argparse.ArgumentParser(description='Training on MIT1003')
parser.add_argument('--data_folder', type=pl.Path, default='../mitdata',
                    help='the folder of mit1003 data')
parser.add_argument('--output_folder', type=str, default='mit',
                    help='the folder used to save the trained model')
parser.add_argument('--model_path', default='SSETM_mse/model27500.pth.tar', type=pl.Path,
                    help='the path of the pre-trained model')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')

args = parser.parse_args()

def main():
    global args

    model = SSETM.model_test(args.model_path).cuda()
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        SaliconLoader_mit.ImageList(args.data_folder, transforms.Compose([
            transforms.ToTensor(),
        ]),
        train=True,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    args.output_folder = args.output_folder + "_mse"
    args.output_folder = pl.Path(args.output_folder)
    if not args.output_folder.is_dir():
        args.output_folder.mkdir()

    criterion = nn.MSELoss().cuda()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        state = {'state_dict' : model.state_dict(),}
        save_path = args.output_folder / ("model{0}.pth.tar".format(epoch+1))
        save_model(state, save_path)

def save_model(state, path):
    torch.save(state, path)

def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()
    for i, (input, smap) in enumerate(train_loader):
        input = input.cuda()
        smap = smap.cuda()

        # during finetuning, the input for segmentation is not useful.
        de1, de2, de3, de4, _ = model([input, input])

        loss = criterion(de1, smap)+criterion(de2, smap)/2+criterion(de3, smap)/4+criterion(de4, smap)/8

        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})'.format(
                   epoch, i, len(train_loader),
                   loss=losses))

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

def adjust_learning_rate(optimizer, epoch):
    factor = 0
    if epoch >= 3:
        factor = 1
    lr = args.lr*(0.1**factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
