from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl
from datasets.dataProvider import DataProvider
from meter.utils import Meter
from model import get_classifier

__all__ = ["CIFAR_LightningModule", "CIFAR_Module", "CIFAR_Test_Module"]


class CIFAR_LightningModule(pl.LightningModule, ABC):
    def __init__(self, hparams):  # must name after hparams or there will be plenty of bugs
        super().__init__()
        self.hparams = hparams  # params must be save in self.hparams to make sure checkpoint contain hparams for reconstruction
        self.complete_hparams()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataProvider = DataProvider(params=hparams)
        self.steps_per_epoch = len(self.train_dataloader().dataset) // self.hparams["batch_size"] // self.hparams[
            "gpus"]
        self.train_meter = self.get_meter("train")
        self.val_meter = self.get_meter("validation")
        self.test_meter = self.get_meter("test")
        self.need_to_learn = True
        self.train_results = {}
        self.val_results = {}
        self.test_results = {}

    def get_meter(self, phase: str) -> Meter:
        from meter.classification_meter import ClassificationMeter as Meter
        workers = 1 if phase == "test" or self.hparams["backend"] != "ddp" else self.hparams["gpus"]
        return Meter(phase=phase, workers=workers, criterion=self.criterion, num_class=10)

    def get_parameters_generator(self):
        return self.parameters

    def complete_hparams(self):
        default_list = {
            'optimizer': 'SGD',
            'lr_scheduler': 'OneCycLR',
            'max_lr': 0.1,
            'weight_decay': 5e-4,
            'step_decay': 0.1,
            'batch_size': 128,
        }
        self.hparams = {**default_list, **self.hparams}

    def choose_optimizer(self):
        gen = self.get_parameters_generator()
        if len(list(gen())) == 0:
            self.need_to_learn = False
            return None

        params = gen()
        from torch.optim import SGD, Adam
        if self.hparams['optimizer'] == 'SGD':
            optimizer = SGD(params, lr=self.hparams["max_lr"],
                            weight_decay=self.hparams["weight_decay"],
                            momentum=0.9, nesterov=True)
        elif self.hparams['optimizer'] == 'Adam':
            optimizer = Adam(params, lr=self.hparams['max_lr'],
                             weight_decay=self.hparams['weight_decay'])
        else:
            assert False, "optimizer not implemented"
        return optimizer

    def choose_scheduler(self, optimizer):
        if optimizer is None:
            return None

        from torch.optim import lr_scheduler
        if self.hparams['lr_scheduler'] == 'ExpLR':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        elif self.hparams['lr_scheduler'] == 'CosLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20 * self.steps_per_epoch + 1, eta_min=0)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        elif self.hparams['lr_scheduler'] == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        elif self.hparams['lr_scheduler'] == 'OneCycLR':
            # + 1 to avoid over flow in steps() when there's totally 800 steps specified and 801 steps called
            # there will be such errors.
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams["max_lr"],
                                                steps_per_epoch=self.steps_per_epoch + 1,
                                                epochs=self.hparams["num_epochs"])
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        elif self.hparams['lr_scheduler'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70, 140, 190], gamma=0.1)
        elif self.hparams['lr_scheduler'] == 'MultiStepLR_CRD':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)
        elif self.hparams['lr_scheduler'] == 'MultiStepLR_NN':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 140, 150], gamma=0.1)
        elif self.hparams['lr_scheduler'] == 'MultiStepLR_NN_50':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
        elif self.hparams['lr_scheduler'] == 'MultiStepLR_NN_70_Adam':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 65], gamma=0.1)
        else:
            return None
        return scheduler

    def configure_optimizers(self):
        optimizer = self.choose_optimizer()
        scheduler = self.choose_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataProvider.train_dl

    def val_dataloader(self):
        return self.dataProvider.val_dl

    def test_dataloader(self):
        return self.dataProvider.test_dl

    @abstractmethod
    def forward(self, x):
        return x

    def step(self, meter, batch):
        images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        meter.update(labels, predictions.detach(), loss.detach())
        acc = (torch.max(predictions, dim=1)[1] == labels).float().mean()
        return {'loss': loss, 'progress_bar': {'acc': acc}}

    def epoch_ends(self, outputs, meter):
        log_ret = meter.log_metric()
        meter.reset()
        ret = {key.split('/')[-1]: values for key, values in log_ret.items()}

        append_log = {}
        if "log" in outputs[0]:
            logs = {key: [out[key] for out in outputs] for key in outputs[0].keys()}
            for key, data_list in logs.items():
                try:
                    data = torch.mean(data_list)
                    append_log[meter.phase + "_log/" + key] = data
                except TypeError:
                    pass

        return {'loss': ret['loss'], 'log': {**log_ret, **append_log}}

    def training_step(self, batch, batch_idx):
        return self.step(self.train_meter, batch)

    def training_epoch_end(self, outputs):
        ret = self.epoch_ends(outputs, self.train_meter)
        if 'save_result' in ret:
            self.train_results = ret['save_result']
        return ret

    def validation_step(self, batch, batch_idx):
        return self.step(self.val_meter, batch)

    def validation_epoch_end(self, outputs):
        ret = self.epoch_ends(outputs, self.val_meter)
        ret["acc"] = torch.tensor(ret["log"]["validation/acc"])
        if 'save_result' in ret:
            self.val_results = ret['save_result']
        return ret

    def test_step(self, batch, batch_nb):
        return self.step(self.test_meter, batch)

    def test_epoch_end(self, outputs):
        ret = self.epoch_ends(outputs, self.test_meter)
        if 'save_result' in ret:
            self.test_results = ret['save_result']
        return ret


class CIFAR_Module(CIFAR_LightningModule):
    def __init__(self, hparams):  # must name after hparams or there will be plenty of bugs
        super().__init__(hparams)
        self.model = get_classifier(hparams["backbone"], hparams["dataset"])

    def forward(self, images):
        return self.model(images)


class CIFAR_Test_Module(pl.LightningModule, ABC):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.dataProvider = DataProvider(params=hparams)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_meter = self.get_meter("test")

    def get_meter(self, phase: str):
        from meter.classification_meter import ClassificationMeter as Meter
        workers = 1 if phase == "test" or self.hparams["backend"] != "ddp" else self.hparams["gpus"]
        return Meter(phase=phase, workers=workers, criterion=self.criterion, num_class=10)

    def test_dataloader(self):
        return self.dataProvider.test_dl

    @abstractmethod
    def forward(self, x):
        return x

    def step(self, meter, batch):
        images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        meter.update(labels, predictions.detach(), loss.detach())
        acc = (torch.max(predictions, dim=1)[1] == labels).float().mean()
        return {'loss': loss, 'progress_bar': {'acc': acc}}

    def epoch_ends(self, outputs, meter):
        log_ret = meter.log_metric()
        meter.reset()
        ret = {key.split('/')[-1]: values for key, values in log_ret.items()}

        append_log = {}
        if "log" in outputs[0]:
            logs = {key: [out[key] for out in outputs] for key in outputs[0].keys()}
            for key, data_list in logs.items():
                try:
                    data = torch.mean(data_list)
                    append_log[meter.phase + "_log/" + key] = data
                except TypeError:
                    pass
        return {'loss': ret['loss'], 'log': {**log_ret, **append_log}}

    def test_step(self, batch, batch_nb):
        return self.step(self.test_meter, batch)

    def test_epoch_end(self, outputs):
        return self.epoch_ends(outputs, self.test_meter)
