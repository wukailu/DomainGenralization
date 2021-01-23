import numpy as np
import torch
from torch import Tensor
from meter.utils import all_sum
from meter.utils import Meter


class ClassificationMeter(Meter):
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, phase: str, workers, criterion, num_class):
        self._loss = []
        self._acc = []
        self._confidence = []

        self.phase = phase
        self._workers = workers
        self._criterion = criterion

    def update(self, targets: Tensor, logits: Tensor, loss: Tensor):
        with torch.no_grad():
            targets = targets.data.cuda()
            logits = logits.data.cuda()
            loss = loss.data.cuda()

            predicts = torch.softmax(logits, dim=1).clamp(1e-6, 1 - 1e-6)
            acc = (torch.max(predicts, dim=1)[1] == targets).float().mean()
            max_val = torch.max(predicts, dim=1)[0]
            confidence = (torch.min(max_val, - max_val + 1)).mean()

            if self._workers != 1:
                self._loss.append(all_sum(loss).item() / self._workers)
                self._acc.append(all_sum(acc).item() / self._workers)
                self._confidence.append(all_sum(confidence).item() / self._workers)
            else:
                self._loss.append(loss.item())
                self._acc.append(acc.item())
                self._confidence.append(confidence.item())

    def log_metric(self) -> dict:
        loss = np.nanmean(self._loss)
        acc = np.nanmean(self._acc)
        confidence = 1 / np.nanmean(self._confidence)

        ret = {"acc": acc, "loss": loss, "confidence": confidence}

        return {self.phase + "/" + key: value for key, value in ret.items()}

    def reset(self):
        self._loss = []
        self._acc = []
        self._confidence = []


class MultiClassificationMeter(Meter):
    def __init__(self, phase: str, workers, criterion, num_class, model_num):
        self.num = model_num
        self.meters = [ClassificationMeter(phase, workers, criterion, num_class) for i in range(model_num)]

    def update(self, targets: Tensor, logits: Tensor, loss: Tensor):
        with torch.no_grad():
            for i in range(self.num):
                self.meters[i].update(targets, logits[i], loss)

    def log_metric(self) -> dict:
        ret = {f"meter {i}" + "/" + key: value for i in range(self.num) for key, value in
               self.meters[i].log_metric().items()}
        return ret

    def reset(self):
        for i in range(self.num):
            self.meters[i].reset()
