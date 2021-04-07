import torch
from .backbone import resnet50s, resnet101s, resnet152s

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn

from ..nn import Encoding, View, Normalize
from .backbone import resnet50s, resnet101s, resnet152s

__all__ = ['getseten', 'get_att']


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


class Net_patch(nn.Module):
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

        n_codes1 = 32
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
            # View(-1, 128 * n_codes1),
            # Normalize(),
            # nn.Linear(128 * n_codes1, 512),
        )

        self.se = SELayer(6)

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

        # self.classifier = nn.Linear(128 * 64, nclass)
        self.classifier = nn.Sequential(
            View(-1, 128 * n_codes1),
            Normalize(),
            nn.Linear(128 * n_codes1, nclass),
        )

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
        print(patch1.shape)
        x1 = self.head1(patch1)
        x2 = self.head1(patch2)
        x3 = self.head1(patch3)
        x4 = self.head1(patch4)
        x5 = self.head1(patch5)
        x6 = self.head1(patch6)
        x7 = self.head1(patch7)
        print(x1.shape)
        x8 = torch.stack([x1, x2, x3, x4, x6, x7], 1)
        # x8 = self.se(x8)
        print(x8.shape)
        x8 = torch.sum(x8, 1)
        print(x8.shape)
        # x8 = 0.1 * x1 + 0.1 * x2 + 0.4 * x6 + 0.4 * x7
        # x8 = torch.add(x6, x7)

        x = self.classifier(x8)
        print(x.shape)
        return x


class Att_patch_net(nn.Module):
    def __init__(self, nclass, backbone='resnet50'):
        super(Att_patch_net, self).__init__()
        self.backbone = backbone
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152s(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        n_codes1 = 32
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

        self.se = SELayer(6)

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

        # self.classifier = nn.Linear(128 * 64, nclass)
        self.classifier = nn.Sequential(
            # View(-1, 128 * n_codes1),
            # Normalize(),
            nn.Linear(128 * n_codes1, nclass),
        )
    def slide_tensor(self,x):
        tensors = []
        len = 5
        for i in range(0, x.shape[2] - len):
            for j in range(1, x.shape[3] - len):
                # b[i, j] = (a[i - 1, j - 1] + a[i - 1, j] + a[i - 1, j + 1] + a[i, j - 1] + a[i, j] + a[i, j + 1] + a[
                #     i + 1, j - 1] + a[i + 1, j] + a[i + 1, j + 1]) / 9.0
                slide = x[:, :, i:i+len,j:j+len]
                # b[i, j] = (a[i - 1, j - 1] + a[i - 1, j] + a[i - 1, j + 1] + a[i, j - 1] + a[i, j] + a[i, j + 1] + a[
                #     i + 1, j - 1] + a[i + 1, j] + a[i + 1, j + 1]) / 9.0
                tensors.append(slide)
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

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)

        # patch1 = x[:, :, :2, :2]
        # patch2 = x[:, :, 2:4, :2]
        # patch3 = x[:, :, 4:7, :3]
        #
        # patch4 = x[:, :, :2, 2:4]
        # patch5 = x[:, :, 2:4, 2:4]
        #
        # patch6 = x[:, :, :3, 4:7]
        # patch7 = x[:, :, 4:7, 4:7]
        xs = self.slide_tensor(x)


        patch1 = x[:, :, :5, :5]
        patch2 = x[:, :, 1:6, :5]
        patch3 = x[:, :, 2:7, :5]

        # patch4 = x[:, :, :5, 2:4]
        # patch5 = x[:, :, 2:4, 2:4]

        patch6 = x[:, :, :5, 1:6]
        patch7 = x[:, :, :5, 2:7]

        x1 = self.head1(patch1)
        x2 = self.head1(patch2)
        x3 = self.head1(patch3)
        # x4 = self.head1(patch4)
        # x5 = self.head1(patch5)
        x6 = self.head1(patch6)
        x7 = self.head1(patch7)
        # x8 = 0.1*x1 + 0.1*x2 + 0.4*x6 + 0.4*x7
        # x8 = torch.add(x6, x7)
        x8 = torch.stack([x1, x2, x3, x6, x7], 1)
        x8 = self.se(x8)
        x8 = torch.sum(x8, 1)
        x = self.classifier(x8)
        return x


def getseten(nclass, backbone):
    # net = Net(nclass, backbone)
    # net = Net_sum(nclass, backbone);:
    net = Net_patch(nclass, backbone)
    return net


def get_att(nclass, backbone):
    # net = Net(nclass, backbone)
    # net = Net_sum(nclass, backbone);:
    net = Att_patch_net(nclass, backbone)
    return net


def test():
    net = Net_patch(nclass=3)
    # print(net)
    x = Variable(torch.randn(2, 3, 224, 224))
    # x = Variable(torch.randn(64, 128))
    # x2 = Variable(torch.randn(64, 128))

    # x3 = x + 0.2* x2

    # print(x3.shape)

    y = net(x)
    print(y.shape)
    # print(y.shape)
    # params = net.parameters()
    # sum = 0
    # for param in params:
    #     sum += param.nelement()
    # print('Total params:', sum)


if __name__ == "__main__":
    test()
