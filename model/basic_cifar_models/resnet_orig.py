import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basic_cifar_models.utils import register_model, unpack_feature, pack_feature

# Credit to https://github.com/akamaster/pytorch_resnet_cifar10

__all__ = ['_resnet20', "ResNet_CIFAR"]


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = \
                    LambdaLayer(
                        lambda x: F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4], "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x, with_feature=True):
        f_list, x_last = unpack_feature(x)

        x = F.relu(self.bn1(self.conv1(x_last)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(x_last)
        x = F.relu(x)

        return pack_feature(f_list, x, with_feature)


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_filters=(16, 16, 32, 64), option='A'):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = num_filters[0]

        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.layer1 = self._make_layer(block, num_filters[1], num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, num_filters[2], num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, num_filters[3], num_blocks[2], stride=2, option=option)
        self.linear = nn.Linear(num_filters[3], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, with_feature=False):
        x = F.relu(self.bn1(self.conv1(x)))

        # x = pack_feature(*unpack_feature(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        f_list, x = unpack_feature(x)

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)

        f_list.append(x)

        x = self.linear(x)

        return pack_feature(f_list, x, with_feature)


@register_model
def resnet20_pretrain(pretrained=False, **kwargs):
    net = ResNet_CIFAR(BasicBlock, num_blocks=[3, 3, 3], **kwargs)
    if pretrained:
        script_dir = "/home/kailu/.cache/torch/checkpoints"
        state_dict = torch.load(script_dir + '/state_dicts/resnet_orig.pt', map_location='cpu')
        net.load_state_dict(state_dict)
    return net


def _resnet20(expansion=1.0, **kwargs):
    return ResNet_CIFAR(BasicBlock, num_blocks=[3, 3, 3], num_filters=[int(i * expansion) for i in [16, 16, 32, 64]],
                        **kwargs)


@register_model
def resnet32(**kwargs):
    return ResNet_CIFAR(BasicBlock, num_blocks=[5, 5, 5], num_filters=[16, 16, 32, 64], **kwargs)


@register_model
def resnet110(**kwargs):
    return ResNet_CIFAR(BasicBlock, num_blocks=[18, 18, 18], num_filters=[16, 16, 32, 64], **kwargs)


@register_model
def wrn40x4(**kwargs):
    return ResNet_CIFAR(BasicBlock, num_blocks=[6, 6, 6], num_filters=[16, 16 * 4, 32 * 4, 64 * 4], option='B', **kwargs)


@register_model
def resnet20(**kwargs):
    return _resnet20(expansion=1, **kwargs)


@register_model
def resnet20x1(**kwargs):
    return _resnet20(expansion=1, **kwargs)


@register_model
def resnet20x1_5(**kwargs):
    return _resnet20(expansion=1.5, **kwargs)


@register_model
def resnet20x2(**kwargs):
    return _resnet20(expansion=2, **kwargs)


@register_model
def resnet20x2_5(**kwargs):
    return _resnet20(expansion=2.5, **kwargs)


@register_model
def resnet20x3(**kwargs):
    return _resnet20(expansion=3, **kwargs)


@register_model
def resnet20x3_5(**kwargs):
    return _resnet20(expansion=3.5, **kwargs)


@register_model
def resnet20x4(**kwargs):
    return _resnet20(expansion=4, **kwargs)


@register_model
def resnet20x4_5(**kwargs):
    return _resnet20(expansion=4.5, **kwargs)


@register_model
def resnet20x5(**kwargs):
    return _resnet20(expansion=5, **kwargs)


@register_model
def resnet20x5_5(**kwargs):
    return _resnet20(expansion=5.5, **kwargs)


@register_model
def resnet20x6(**kwargs):
    return _resnet20(expansion=6, **kwargs)


@register_model
def resnet20x6_5(**kwargs):
    return _resnet20(expansion=6.5, **kwargs)


@register_model
def resnet20x7(**kwargs):
    return _resnet20(expansion=7, **kwargs)


@register_model
def resnet20x7_5(**kwargs):
    return _resnet20(expansion=7.5, **kwargs)


@register_model
def resnet20x8(**kwargs):
    return _resnet20(expansion=8, **kwargs)
