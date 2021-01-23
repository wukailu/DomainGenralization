from typing import Any

import numpy as np
import torch
from torch import Tensor


def all_sum(tensor):
    import torch.distributed as dist
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def loss_fn_kd(outputs, teacher_outputs, temperature=4):
    import torch.nn.functional as F
    from torch import nn
    t = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / t, dim=1), F.softmax(teacher_outputs / t, dim=1)) * t * t
    return KD_loss


# class Unnormalize:
#     """Converts an image tensor that was previously Normalize'd
#     back to an image with pixels in the range [0, 1]."""
#
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, tensor):
#         mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
#         std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
#         return torch.clamp(tensor * std + mean, 0., 1.)


class SoftCrossEntropy(torch.nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.sum = 0

    def forward(self, inputs: Tensor, target: Tensor):
        batchsize = inputs.size(0)

        cnt = 0
        self.sum = torch.zeros(1).to(target.device)
        prob = torch.softmax(inputs, dim=1)
        for i in range(batchsize):
            if prob[i][target[i]] < 0.8:
                self.sum += torch.nn.functional.cross_entropy(inputs[i].unsqueeze(0), target[i].unsqueeze(0))
                cnt += 1
        if cnt > 0:
            self.sum = self.sum / cnt
        return self.sum


class Meter:
    def update(self, *input: Any):
        raise NotImplementedError()

    def log_metric(self) -> dict:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


def np_to_tensor_img(results: np.ndarray):
    from torchvision.transforms import ToTensor
    import PIL.Image
    import io
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    plt.figure()

    plt.matshow(results)
    plt.colorbar()

    plt.title("relation map")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image