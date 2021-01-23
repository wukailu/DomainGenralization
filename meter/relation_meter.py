from meter.utils import Meter
import torch
import numpy as np


class MeanMeterWithKey(Meter):
    """
    This Meter automatically track input's mean.
    Example.
        update({"your_key": value_tensor})
    """
    def __init__(self, phase: str, workers):
        self.phase = phase
        self.workers = workers
        self.records = {}

    def update(self, results: dict):
        for key, val in results.items():
            if not isinstance(val, torch.Tensor):
                raise NotImplementedError("value in dict must be tensors")
            val = val.detach()
            if self.workers > 1:
                from torch.distributed import all_reduce
                all_reduce(val)
                val = val / self.workers
            if key not in self.records:
                self.records[key] = []
            self.records[key].append(val.cpu().numpy())

    def log_metric(self) -> dict:
        ret = {}
        for key in self.records:
            ret[self.phase + "/" + key] = np.mean(self.records[key], axis=0)
        return ret

    def reset(self):
        self.records = {}
