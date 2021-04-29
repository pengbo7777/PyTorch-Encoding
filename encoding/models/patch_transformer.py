import torch
from .backbone import resnet50s, resnet101s, resnet152s

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..nn import Encoding, View, Normalize
from .backbone import resnet50s, resnet101s, resnet152s, swin_tiny_patch4_window7_224
from .backbone.swin_transformer import *


class Trans_patch_net(nn.Module):
    def __init__(self, nclass, backbone='swin_transformer'):
        super(Trans_patch_net, self).__init__()
        self.backbone = backbone
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'swin_transformer':
            self.pretrained = swin_tiny_patch4_window7_224(pretrained=True, dilated=False)

        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        n_codes1 = 32
        # n_codes2 = 16
        # n_codes3 = 32
        # n_codes4 = 64

        # 添加vlad模块
        self.dim = 128
        self.alpha = 1.0

        self.head1 = nn.Sequential(
            nn.Conv2d(768, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes1),
            View(-1, 128 * n_codes1),
            Normalize(),
            nn.Linear(128 * n_codes1, nclass),
        )

        self.classifier = nn.Sequential(
            View(-1, 128 * n_codes1),
            Normalize(),
            nn.Linear(128 * n_codes1, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, nclass),
        )
        self.head = nn.Linear(self.pretrained.num_features, nclass) if nclass > 0 else nn.Identity()

    def slide_tensor(self, x):
        tensors = []
        len = 5
        for i in range(0, x.shape[2] - len + 1):
            for j in range(0, x.shape[3] - len + 1):
                # b[i, j] = (a[i - 1, j - 1] + a[i - 1, j] + a[i - 1, j + 1] + a[i, j - 1] + a[i, j] + a[i, j + 1] + a[
                #     i + 1, j - 1] + a[i + 1, j] + a[i + 1, j + 1]) / 9.0
                slide = x[:, :, i:i + len, j:j + len]
                # b[i, j] = (a[i - 1, j - 1] + a[i - 1, j] + a[i - 1, j + 1] + a[i, j - 1] + a[i, j] + a[i, j + 1] + a[
                #     i + 1, j - 1] + a[i + 1, j] + a[i + 1, j + 1]) / 9.0
                tensors.append(slide)
        # tensors = torch.stack(tensors, 1)
        return tensors

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))

        x = self.pretrained.forward_features(x)  # b, 49, 768 ->b,2048, 11, 11
        x = rearrange(x, 'b (w h) d -> b d w h', w=7)

        xs = self.slide_tensor(x)
        x_vs = []
        for patch in xs:
            x_v = self.head1(patch)
            x_v = nn.functional.log_softmax(x_v)

            x_vs.append(x_v)
        x_vs = torch.stack(x_vs, 1)

        # 通道注意力
        # x_vs = self.se(x_vs)

        x = torch.sum(x_vs, 1)
        x = x/len(x_vs)
        # x = self.head1(x)
        return x

def get_patch_transformer(nclass, backbone):
    net = Trans_patch_net(nclass)
    return net
