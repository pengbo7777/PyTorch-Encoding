import torch
from .backbone import resnet50s, resnet101s, resnet152s

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn

from ..nn import Encoding, View, Normalize
from .backbone import resnet50s, resnet101s, resnet152s

__all__ = ['getseten']


class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Net(nn.Module):
    def __init__(self, nclass, backbone='resnet50'):
        super(Net, self).__init__()
        self.backbone = backbone
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152s(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        n_codes1 = 8
        n_codes2 = 16
        n_codes3 = 32
        n_codes4 = 64

        # 添加vlad模块
        self.dim = 128
        self.alpha = 1.0

        self.head1 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes1),
            View(-1, 128 * n_codes1),
            Normalize(),
            nn.Linear(128 * n_codes1, 512),
        )

        self.head2 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes2),
            View(-1, 128 * n_codes2),
            Normalize(),
            nn.Linear(128 * n_codes2, 512),
        )

        self.head3 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes3),
            View(-1, 128 * n_codes3),
            Normalize(),
            nn.Linear(128 * n_codes3, 512),
        )

        self.head4 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes4),
            View(-1, 128 * n_codes4),
            Normalize(),
            nn.Linear(128 * n_codes4, 512),
        )

        self.classifier = nn.Linear(1024, nclass)

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

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)

        # x1 = self.head1(x)
        # x2 = self.head2(x)
        x3 = self.head3(x)
        x4 = self.head4(x)

        x = torch.cat([x3, x4], 1)
        return self.classifier(x)


class Net_sum(nn.Module):
    def __init__(self, nclass, backbone='resnet50'):
        super(Net_sum, self).__init__()
        self.backbone = backbone
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152s(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        n_codes1 = 8
        n_codes2 = 16
        n_codes3 = 32
        n_codes4 = 64

        # 添加vlad模块
        self.dim = 128
        self.alpha = 1.0

        self.head1 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes1),
            View(-1, 128 * n_codes1),
            Normalize(),
            nn.Linear(128 * n_codes1, 512),
        )

        self.head2 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes2),
            View(-1, 128 * n_codes2),
            Normalize(),
            nn.Linear(128 * n_codes2, 512),
        )

        self.head3 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes3),
            View(-1, 128 * n_codes3),
            Normalize(),
            nn.Linear(128 * n_codes3, 512),
        )

        self.head4 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes4),
            View(-1, 128 * n_codes4),
            Normalize(),
            nn.Linear(128 * n_codes4, 512),
        )

        self.classifier = nn.Linear(1024, nclass)

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

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)

        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        x4 = self.head4(x)

        x = x1 + x2 + x3 + x4
        return self.classifier(x)


class Net_patch():
    def __init__(self, nclass, backbone='resnet50'):
        super(Net_patch, self).__init__()
        self.backbone = backbone
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152s(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        n_codes1 = 64
        n_codes2 = 16
        n_codes3 = 32
        n_codes4 = 64

        # 添加vlad模块
        self.dim = 128
        self.alpha = 1.0

        self.head1 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes1),
            View(-1, 128 * n_codes1),
            Normalize(),
            # nn.Linear(128 * n_codes1, 512),
        )

        # self.head2 = nn.Sequential(
        #     nn.Conv2d(2048, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     Encoding(D=128, K=n_codes2),
        #     View(-1, 128 * n_codes2),
        #     Normalize(),
        #     nn.Linear(128 * n_codes2, 512),
        # )
        #
        # self.head3 = nn.Sequential(
        #     nn.Conv2d(2048, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     Encoding(D=128, K=n_codes3),
        #     View(-1, 128 * n_codes3),
        #     Normalize(),
        #     nn.Linear(128 * n_codes3, 512),
        # )
        #
        # self.head4 = nn.Sequential(
        #     nn.Conv2d(2048, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     Encoding(D=128, K=n_codes4),
        #     View(-1, 128 * n_codes4),
        #     Normalize(),
        #     nn.Linear(128 * n_codes4, 512),
        # )

        self.classifier = nn.Linear(128 * n_codes1, nclass)

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

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)

        patch1 = x[:, :, :2, :2]
        patch2 = x[:, :, 2:4, :2]
        patch3 = x[:, :, 4:7, :3]

        patch4 = x[:, :, :2, 2:4]
        patch5 = x[:, :, 2:4, 2:4]

        patch6 = x[:, :, :3, 4:7]
        patch7 = x[:, :, 4:7, 4:7]

        x1 = self.head1(patch1)
        x2 = self.head1(patch2)
        x3 = self.head1(patch3)
        x4 = self.head1(patch4)
        x5 = self.head1(patch5)
        x6 = self.head1(patch6)
        x7 = self.head1(patch7)

        x = x1 + x2 + x3 + x4 + x5 + x6 + x7
        return self.classifier(x)


def getseten(nclass, backbone):
    # net = Net(nclass, backbone)
    # net = Net_sum(nclass, backbone)
    net = Net_patch(nclass,backbone)
    return net
