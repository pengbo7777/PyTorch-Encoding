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
from .patch_transformer import *

__all__ = ['getseten', 'get_att']

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


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
            # Normalize
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
        # print(patch1.shape)
        x1 = self.head1(patch1)
        x2 = self.head1(patch2)
        x3 = self.head1(patch3)
        x4 = self.head1(patch4)
        x5 = self.head1(patch5)
        x6 = self.head1(patch6)
        x7 = self.head1(patch7)
        # print(x1.shape)
        x8 = torch.stack([x1, x2, x3, x4, x6, x7], 1)
        # x8 = self.se(x8)
        # print(x8.shape)
        x8 = torch.sum(x8, 1)
        # print(x8.shape)
        # x8 = 0.1 * x1 + 0.1 * x2 + 0.4 * x6 + 0.4 * x7
        # x8 = torch.add(x6, x7)

        x = self.classifier(x8)
        # print(x.shape)
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
        x_vs = []
        for patch in xs:
            x_v = self.head1(patch)
            x_vs.append(x_v)
        # vlad patch set
        x_vs = torch.stack(x_vs, 1)

        # 通道注意力
        # x_vs = self.se(x_vs)

        x8 = torch.sum(x_vs, 1)
        x = self.classifier(x8)
        return x


class Tsen_net(nn.Module):
    def __init__(self, nclass, backbone='resnet50'):
        super(Tsen_net, self).__init__()
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

        # self.head1 = nn.Sequential(
        #     nn.Conv2d(2048, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     Encoding(D=128, K=n_codes1),
        #     View(-1, 128 * n_codes1),
        #     Normalize(),
        #     # nn.Linear(128 * n_codes1, 512),
        # )

        self.head1 = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes1),
            # View(-1, 128 * n_codes1),
            # Normalize(),
            # nn.Linear(128 * n_codes, nclass),
        )

        self.se = SELayer(6)
        self.dic_encoder = Transformer(128, 1, 1, 2048, 0.2)
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
        # self.classifier = nn.Sequential(
        #     # View(-1, 128 * n_codes1),
        #     # Normalize(),
        #     nn.Linear(128 * n_codes1, nclass),
        # )

        self.classifier = nn.Sequential(
            View(-1, 128 * n_codes1),
            Normalize(),
            nn.Linear(128 * n_codes1, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, nclass),

        )

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

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        x = self.head1(x)
        x = self.dic_encoder(x)
        x = self.classifier(x)
        return x


class swinTrans_encoder_net(nn.Module):
    def __init__(self, nclass, backbone='swin_transformer'):
        super(swinTrans_encoder_net, self).__init__()
        self.backbone = backbone
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'swin_transformer':
            self.pretrained = swin_tiny_patch4_window7_224(pretrained=True, dilated=False)

        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        n_codes1 = 64
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
            # View(-1, 128 * n_codes1),
            # Normalize(),
            # nn.Linear(128 * n_codes, nclass),
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
        x = self.head1(x)
        return x


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.resize = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            Rearrange('b (h p) d  -> b h p d', p=patch_size),
            # nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # print(img.shape)
        x = self.to_patch_embedding(img)
        # print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.shape)
        # print("self.pos_embedding[:, :(n + 1)].shape")
        # print(self.pos_embedding[:, :(n + 1)].shape)
        # print(self.pos_embedding.shape)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # print(x.shape)
        x = self.transformer(x)
        # print(x.shape)
        # print(x[:, 0].shape)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # print(x.shape)
        # x2 = torch.randn(2, 256, 1024)
        # x2 = self.resize(x2)
        # print(x2.shape)
        return self.mlp_head(x)


# if __name__ == '__main__':
#     v = ViT(
#         image_size=256,
#         patch_size=16,
#         num_classes=1000,
#         dim=1024,
#         depth=6,
#         heads=8,
#         mlp_dim=2048,
#         dropout=0.1,
#         emb_dropout=0.1
#     )
#
#     img = torch.randn(2, 3, 256, 256)
#     # mask = torch.ones(1, 8, 8).bool()  # optional mask, designating which patch to attend to
#
#     preds = v(img)  # (1, 1000)
#     print(preds.shape)
#     # x= torch.randn(2, 256, 1024)
#     # Rearrange('b n d  -> b p1 p2 d', p1=16, p2=16)


def getseten(nclass, backbone):
    # net = Net(nclass, backbone)
    # net = Net_sum(nclass, backbone);
    # net = Net_patch(nclass, backbone)
    # net = Tsen_net(nclass, backbone)
    net = ViT(
        image_size=256,
        patch_size=16,
        num_classes=nclass,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    return net

def get_att(nclass, backbone):
    # net = Net(nclass, backbone)
    # net = Net_sum(nclass, backbone);:
    # net = Tsen_net(nclass, backbone)
    # net = swinTrans_encoder_net(nclass)
    net = Trans_patch_net(nclass)
    return net


def test():
    net = Net_patch(nclass=3)
    # print(net)
    x = Variable(torch.randn(2, 3, 224, 224))
    x2 = Variable(torch.randn(2, 49, 78))

    # x = Variable(torch.randn(64, 128))
    # x2 = Variable(torch.randn(64, 128))

    # x3 = x + 0.2* x2

    # print(x3.shape)
    from einops import rearrange
    x2 = rearrange(x2, 'b,(w,h),d -> b d w h', h=7)
    print(x2.shape)

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
