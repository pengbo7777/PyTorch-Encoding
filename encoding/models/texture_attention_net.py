import torch
from .backbone import resnet50s, resnet101s, resnet152s

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
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
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152s(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        n_codes = 8

        # 添加vlad模块
        self.num_clusters = [8, 16, 32]
        self.dim = 128
        self.alpha = 1.0
        self.normalize_input = True
        self.conv_d1 = nn.Conv2d(self.dim, self.num_clusters[0], kernel_size=(1, 1), bias=True)
        self.conv_d2 = nn.Conv2d(self.dim, self.num_clusters[1], kernel_size=(1, 1), bias=True)
        self.conv_d3 = nn.Conv2d(self.dim, self.num_clusters[2], kernel_size=(1, 1), bias=True)

        self.conv_1_1 = nn.Conv2d(2048, self.dim, kernel_size=(1, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(2048)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU()

        self.centroids1 = nn.Parameter(torch.rand(self.num_clusters[0], self.dim))
        self.centroids2 = nn.Parameter(torch.rand(self.num_clusters[1], self.dim))
        self.centroids3 = nn.Parameter(torch.rand(self.num_clusters[2], self.dim))

        # self._init_params()

        self.vlad_liner1 = nn.Linear(self.dim * self.num_clusters[0], 64)
        self.vlad_liner2 = nn.Linear(self.dim * self.num_clusters[1], 64)
        self.vlad_liner3 = nn.Linear(self.dim * self.num_clusters[2], 64)
        # self.resNest = ResNest.resnest50(pretrained=True)
        self.queeze_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_se = nn.Sequential(
            nn.Linear(3, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1, 3, bias=False),
            nn.Sigmoid()
        )
        self.avgPool = nn.AvgPool2d(7)
        self.poolLiner = nn.Linear(2048, 64)
        self.poolBatchNorm1d = nn.BatchNorm1d(64)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(64 * 64),

            nn.Linear(64 * 64, nclass))
        self.se = SELayer(3)

    # def _init_params(self):
    #     self.conv.weight = nn.Parameter(
    #         (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
    #     )
    #     self.conv.bias = nn.Parameter(
    #         - self.alpha * self.centroids.norm(dim=1)
    #     )

    def vlad_layer(self, x):
        # soft-assignment
        # [20, 2048, 7, 7]
        # print("x初始shape：", x.shape)
        N, C = x.shape[:2]

        soft_assign1 = self.conv_d1(x)
        soft_assign2 = self.conv_d2(x)
        soft_assign3 = self.conv_d3(x)
        # print(soft_assign.shape)
        soft_assign1 = soft_assign1.view(N, self.num_clusters[0], -1)
        soft_assign1 = F.softmax(soft_assign1, dim=1)
        soft_assign2 = soft_assign2.view(N, self.num_clusters[1], -1)
        soft_assign2 = F.softmax(soft_assign2, dim=1)
        soft_assign3 = soft_assign3.view(N, self.num_clusters[2], -1)
        soft_assign3 = F.softmax(soft_assign3, dim=1)
        # print(soft_assign1.shape)
        # print(soft_assign2.shape)
        # print(soft_assign3.shape)
        x_flatten = x.view(N, C, -1)  # C=2048
        residual1 = x_flatten.expand(self.num_clusters[0], -1, -1, -1).permute(1, 0, 2, 3) - \
                    self.centroids1.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual2 = x_flatten.expand(self.num_clusters[1], -1, -1, -1).permute(1, 0, 2, 3) - \
                    self.centroids2.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual3 = x_flatten.expand(self.num_clusters[2], -1, -1, -1).permute(1, 0, 2, 3) - \
                    self.centroids3.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

        residual1 *= soft_assign1.unsqueeze(2)
        residual2 *= soft_assign2.unsqueeze(2)
        residual3 *= soft_assign3.unsqueeze(2)

        vlad1 = residual1.sum(dim=-1)
        # print(vlad1.shape)
        # _, b1, c1 = vlad1.size()
        # print("b1,c1")
        # print(b1, c1)
        vlad1_s = self.queeze_pool(vlad1)
        # print(vlad1_s)

        vlad2 = residual2.sum(dim=-1)
        # print(vlad2.shape)
        # _, b2, c2 = vlad2.size()
        vlad2_s = self.queeze_pool(vlad2)

        vlad3 = residual3.sum(dim=-1)
        # print(vlad3.shape)
        # _, b3, c3 = vlad3.size()
        vlad3_s = self.queeze_pool(vlad3)

        c = torch.stack([vlad1_s, vlad2_s, vlad3_s], 1).squeeze(-1).squeeze(-1)
        # print("c.shape")
        # print(c.shape)
        c = self.fc_se(c)
        vlad2 = F.normalize(vlad2, p=2, dim=2)  # intra-normalization    #32*128
        vlad2 = vlad2.view(x.size(0), -1)  # flatten
        vlad1 = F.normalize(vlad1, p=2, dim=2)  # intra-normalization    #8*128
        vlad1 = vlad1.view(x.size(0), -1)  # flatten
        vlad3 = F.normalize(vlad3, p=2, dim=2)  # intra-normalization
        vlad3 = vlad3.view(x.size(0), -1)  # flatten

        x_vlad1 = F.normalize(vlad1, p=2, dim=1)  # L2
        x_vlad2 = F.normalize(vlad2, p=2, dim=1)  # L2 normalizenormalize
        x_vlad3 = F.normalize(vlad3, p=2, dim=1)  # L2 normalize

        x_vlad1 = self.vlad_liner1(x_vlad1)
        x_vlad2 = self.vlad_liner2(x_vlad2)
        x_vlad3 = self.vlad_liner3(x_vlad3)
        # print("x_vlad1.shape")
        # print(x_vlad1.shape)

        x_vlad = torch.stack([x_vlad1, x_vlad2, x_vlad3], 1)
        # print("x_vlad.shape")
        # print(x_vlad.shape)
        # print(c)
        c = c.unsqueeze(2)
        # print(c)
        # print(x_vlad)

        x_vlad = x_vlad*c.expand_as(x_vlad)
        # print("x_vlad.shape")
        # print(x_vlad.shape)
        x_vlad = torch.sum(x_vlad, dim=1)
        # print(x_vlad.shape)
        return x_vlad

        # print("x1.shape")
        # print(x1.shape)
        # x * y.expand_as(x)
        # y = self.fc(y).view(b, c, 1, 1)

        # self.fc_se(c)
        #
        # self.se()
        # # ---end
        # # print("x1：")
        # # print(x1.shape)
        # # x2 = self.pool(x)
        # x1 = self.vlad_liner(x1)

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



        x = self.resnest50.conv1(x)
        x = self.resnest50.bn1(x)
        x = self.resnest50.relu(x)
        x = self.resnest50.maxpool(x)

        x = self.resnest50.layer1(x)
        x = self.resnest50.layer2(x)
        x = self.resnest50.layer3(x)
        x_source = self.resnest50.layer4(x)

        # vlad
        if self.normalize_input:
            x = F.normalize(x_source, p=2, dim=1)  # across descriptor dim

        # 这里先做一个降维
        x = self.conv_1_1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x1 = self.vlad_layer(x)

        x2 = self.avgPool(x_source)
        x2 = x2.view(-1, 2048)
        x2 = self.poolLiner(x2)
        x2 = self.poolBatchNorm1d(x2)
        x1 = x1.unsqueeze(1).expand(x1.size(0), x2.size(1), x1.size(-1))
        x = x1 * x2.unsqueeze(-1)
        x = x.view(-1, x1.size(-1) * x2.size(1))
        x = self.fc(x)
        # else:
        #     x = self.pretrained(x)
        # return x, vlad
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


def test():
    net = Net(nclass=3)
    print(net)
    x = Variable(torch.randn(2, 3, 352, 352))
    y = net(x)
    print(y.shape)
    # params = net.parameters()
    # sum = 0
    # for param in params:
    #     sum += param.nelement()
    # print('Total params:', sum)

def getseten(nclass,backbone):
    net = Net(nclass,backbone)
    return net
if __name__ == "__main__":
    test()
    # m = nn.AdaptiveMaxPool2d(1)
    # input = torch.randn(8, 128)
    # input2 = torch.randn(8, 128)
    # input3 = torch.randn(8, 128)
    # b = torch.stack([input, input2, input3], 0)
    # print(b.shape)
    # print(input)
    # output = m(input)
    # print(output.shape)
    # print(output)

    # m = nn.Linear(3, 1, bias=False)
    # input = torch.randn(2, 3)
    # output = m(input)
    # print(output.shape)
