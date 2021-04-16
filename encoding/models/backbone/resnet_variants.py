
import torch
from .resnet import ResNet, Bottleneck
from ..model_store import get_model_file
from ..swin_transformer import *

__all__ = ['resnet50s', 'resnet101s', 'resnet152s',
           'resnet50d', 'swin_tiny_patch4_window7_224']

# pspnet version of ResNet
def resnet50s(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-50 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['deep_stem'] = True
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50s', root=root)), strict=False)
    return model

def resnet101s(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-101 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['deep_stem'] = True
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet101s', root=root)), strict=False)
    return model

def resnet152s(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-152 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['deep_stem'] = True
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet152s', root=root)), strict=False)
    return model

# ResNet-D
def resnet50d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32,
                   avg_down=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50d', root=root)), strict=False)
    return model

def swin_tiny_patch4_window7_224(pretrained=True, **kwargs):

    model = SwinTransformer()
    if pretrained:
        path = '/workspace/experiments/swin_tiny_patch4_window7_224.pth'
        # path = 'D:\pengbo\code\swin_tiny_patch4_window7_224.pth'
        model.load_state_dict(torch.load(path))
    return model

if __name__ == '__main__':
    model = swin_tiny_patch4_window7_224(True)
    print(model)
