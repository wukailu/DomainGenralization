from torch import nn
import torch

from model.basic_cifar_models import register_model, unpack_feature, pack_feature


@register_model
def test_model1_f1():
    model = TestModel1_f1()
    return model


@register_model
def test_model1_f2():
    model = TestModel1_f2()
    return model


class TestModel1_f1(nn.Module):
    def forward(self, x, with_feature=False):
        f_list, x = unpack_feature(x)
        x = 5 * x
        return pack_feature(f_list, x, with_feature)


class TestModel1_f2(nn.Module):
    def forward(self, x, with_feature=False):
        f_list, x = unpack_feature(x)
        x = 10 * x.abs() + 5 + torch.randn_like(x)
        return pack_feature(f_list, x, with_feature)

