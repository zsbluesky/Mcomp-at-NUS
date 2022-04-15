import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import scipy.ndimage.filters as filters
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from networks.vit_seg_modeling import *
from networks.vit_seg_configs import *

class MAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(MAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# used for gradients decay
class g_decay(torch.autograd.Function):
    def __init__(self):
        super(g_decay, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class G_decay(nn.Module):
    def __init__(self, lambda_=0.):
        super(G_decay, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return g_decay.apply(x, self.lambda_)

class SSETM(nn.Module):
    def __init__(self):
        super(SSETM, self).__init__()

        self.g_decay = G_decay(-0.1)
        self.relu = nn.ReLU(inplace=True)

        # define pretrained transformer encoder
        self.config = get_r50_b16_config()
        self.config.n_skip = 3
        self.config.patches.grid = (
        int(480 / 16), int(640 / 16))
        self.transformer = Transformer(self.config, (480, 640), True)

        self.out_channels = 64

        self.output0 = self._make_output(64, readout=self.out_channels)
        self.output1 = self._make_output(256, readout=self.out_channels)
        self.output2 = self._make_output(512, readout=self.out_channels)
        self.output3 = self._make_output(768, readout=self.out_channels)
        
        self.state_transfer = nn.Conv2d(768, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.fusion23 = self._make_fusion(128, readout=self.out_channels)
        self.fusion12 = self._make_fusion(128+64, readout=self.out_channels)
        self.fusion01 = self._make_fusion(128+64, readout=self.out_channels)

        self.combined1 = self._make_output(self.out_channels, sigmoid=True)  # use sigmoid for activation in the last layer.
        self.combined2 = self._make_output(self.out_channels, sigmoid=True)
        self.combined3 = self._make_output(self.out_channels, sigmoid=True)
        self.combined4 = self._make_output(self.out_channels, sigmoid=True)

        self.fc = nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.mam = MAM(64, ratio=8)
        self.drop = nn.Dropout2d()
        self.score = nn.Conv2d(768, 21, 1)
        self.up = nn.ConvTranspose2d(
            21, 21, 32, stride=16, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            from scipy import ndimage
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if True:
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                gsnewh = 30
                gsneww = 40
                print('load_pretrained: grid-size from %s to %s %s' % (gs_old, gsnewh, gsneww))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gsnewh / gs_old, gsneww / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gsnewh * gsneww, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

    def _make_output(self, planes, readout=1, sigmoid=False):
        layers = [
            nn.Conv2d(planes, readout, kernel_size=3, padding=1),
            nn.BatchNorm2d(readout),
        ]
        if sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_fusion(self, planes, readout=1, sigmoid=False):
        layers = [
            nn.Conv2d(planes, readout, kernel_size=3, padding=1),
            nn.BatchNorm2d(readout),
        ]
        if sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        img = x[1]
        hs, ws = img.size(2), img.size(3)
        img = F.interpolate(img, (480, 480))

        x = x[0]
        h1, w1 = x.size(2), x.size(3)

        x, attn_weights, features = self.transformer(x)
        features = features[::-1]
        out0, out1, out2 = features
        out3 = x
        B, n_patch, hidden = out3.size()
        h, w = 30, 40
        out3 = out3.permute(0, 2, 1)
        out3 = out3.contiguous().view(B, hidden, h, w)

        img, attn_weights, features = self.transformer(img)
        B, n_patch, hidden = img.size()
        h, w = 30, 40
        img = img.permute(0, 2, 1)
        img = img.contiguous().view(B, hidden, h, w)

        img = self.fc(self.g_decay(img))
        img = self.drop(self.relu(img))

        co = self.fc(out3)
        co = self.drop(self.relu(co))
        co = self.state_transfer(co)

        out01 = self.output0(out0)
        out11 = self.output1(out1)
        out21 = self.output2(out2)
        out31 = self.output3(out3)

        att = self.mam(co)
        out0 = out01*att
        out1 = out11*att
        out2 = out21*att
        out3 = out31*att

        out3_ = F.interpolate(out3, (out2.size(2), out2.size(3)))
        fusion23 = torch.cat([out2, out3_], dim=1)
        fusion23 = self.fusion23(fusion23)
        fusion23_ = F.interpolate(fusion23, (out1.size(2), out1.size(3)))
        out3_ = F.interpolate(out3, (out1.size(2), out1.size(3)))
        fusion12 = torch.cat([fusion23_, out1, out3_], dim=1)
        fusion12 = self.fusion12(fusion12)
        fusion12_ = F.interpolate(fusion12, (out0.size(2), out0.size(3)))
        out3_ = F.interpolate(out3, (out0.size(2), out0.size(3)))
        fusion01 = torch.cat([fusion12_, out0, out3_], dim=1)
        fusion01 = self.fusion01(fusion01)

        x1 = self.combined1(fusion01)
        x1 = F.interpolate(x1, (h1, w1))
        x2 = self.combined2(fusion12)
        x2 = F.interpolate(x2, (h1, w1))
        x3 = self.combined3(fusion23)
        x3 = F.interpolate(x3, (h1, w1))
        x4 = self.combined4(out3)
        x4 = F.interpolate(x4, (h1, w1))

        img = self.score(img)
        img = self.up(img)
        img = F.interpolate(img, (hs, ws))
        return x1, x2, x3, x4, img


def model(model_path):
    model = SSETM()
    if model_path is None:
        print ("Training from scratch.")
    else:
        model.load_from(weights=np.load(model_path))
        print ("Model loaded", model_path)
    return model

def model_test(model_path):
    model = SSETM()
    if model_path is None:
        print ("Not saved!")
    else:
        model = SSETM()
        model_state = model.state_dict()
        loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
        if "state_dict" in loaded_model:
            loaded_model = loaded_model['state_dict']

        # if use multiple gpus:
        pretrained = {k[7:]:v for k, v in loaded_model.items() if k[7:] in model_state}

        # if only use one gpu:
        if len(pretrained) == 0:
            pretrained = {k:v for k, v in loaded_model.items() if k in model_state}

        # check error
        for k in pretrained.keys():
            if k not in model_state.keys():
                print(k)

        model_state.update(pretrained)
        model.load_state_dict(model_state)
        print ("Model loaded", model_path)
    return model
