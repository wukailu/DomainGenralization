import torch

from model.cifar_lightning_model import CIFAR_LightningModule
from model import get_classifier, load_models
from model.basic_cifar_models import ResNet
import copy
from torch import nn
from RepDistiller import models as Rep_models


class Traditional_Ensemble(CIFAR_LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.models = load_models(hparams)
        self.num = len(self.models)
        self.modeles.cuda()

    def forward(self, images):
        predictions = torch.sum(torch.stack([model(images) for model in self.models]), dim=0)
        return predictions

    def configure_optimizers(self):
        return None, None


class Proposed_Ensemble(CIFAR_LightningModule):
    models: nn.ModuleList

    def __init__(self, hparams):
        super().__init__(hparams)
        from model.utils import model_init
        from model.utils import freeze
        self.num = hparams["model_num"]
        self.models = load_models(hparams)
        assert len(self.models) == self.num
        if not hparams["train_teachers"]:
            freeze(self.models)
        self.models.cuda()

        self.large_gens = nn.ModuleList([self._make_large_generator(size * 4, size)
                                         for size in [64, 128, 256, 512]])
        self.small_gens = nn.ModuleList(
            [nn.ModuleList([copy.copy(self._make_small_generator(size, size * 4)) for i in range(self.num)])
             for size in [64, 128, 256, 512]])
        if not hparams["train_gens"]:
            freeze(self.large_gens)
            freeze(self.small_gens)
        else:
            model_init(self.large_gens)
            model_init(self.small_gens)

        self.student = get_classifier(hparams["backbone"], hparams["dataset"])
        if not hparams["train_student"]:
            freeze(self.student)
        else:
            model_init(self.student)

        from model.feature_similarity_measurement import cka_loss

        self.feature_loss = cka_loss()

    def _make_large_generator(self, input_channel, output_channel):
        return nn.Identity()

    def _make_small_generator(self, input_channel, output_channel):
        return nn.Identity()

    def get_parameters_generator(self):
        from model.utils import get_trainable_params
        return get_trainable_params(self)

    def subblock_step(self, model: torch.nn.Module, step: int, x: torch.Tensor):
        if isinstance(model, ResNet):
            assert 0 <= step <= 4
            if step == 0:
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)

                x = model.layer1(x)
            elif step == 1:
                x = model.layer2(x)
            elif step == 2:
                x = model.layer3(x)
            elif step == 3:
                x = model.layer4(x)
            else:
                x = model.avgpool(x)
                x = x.reshape(x.size(0), -1)
                x = model.fc(x)
        elif isinstance(model, Rep_models.resnetv2.ResNet):
            import torch.nn.functional as F
            assert 0 <= step <= 4
            if step == 0:
                x = F.relu(model.bn1(model.conv1(x)))
                x, f1_pre = model.layer1(x)
            elif step == 1:
                x, f2_pre = model.layer2(x)
            elif step == 2:
                x, f3_pre = model.layer3(x)
            elif step == 3:
                x, f4_pre = model.layer4(x)
            else:
                x = model.avgpool(x)
                x = x.view(x.size(0), -1)
                x = model.linear(x)
        else:
            raise NotImplementedError(f"{type(model)} is not implemented!")
        return x

    def forward(self, images: torch.Tensor):
        xs = [images.clone() for i in range(self.num)]
        x_st = images
        middles = [torch.zeros(1)] * 4
        middle_st = [torch.zeros(1)] * 4
        for i in range(4):
            for j in range(self.num):
                xs[j] = self.subblock_step(self.models[j], i, xs[j])
            middles[i] = self.large_gens[i](torch.cat(xs, dim=1))
            for j in range(self.num):
                xs[j] = self.small_gens[i][j](middles[i])

            x_st = self.subblock_step(self.student, i, x_st)
            middle_st[i] = x_st

        for j in range(self.num):
            xs[j] = self.subblock_step(self.models[j], 4, xs[j])
        x_st = self.subblock_step(self.student, 4, x_st)

        return xs, x_st, middles, middle_st

    def step(self, meter, batch):
        from meter.utils import loss_fn_kd
        images, labels = batch
        xs, x_st, middles, middle_st = self.forward(images)
        batchsize = middle_st[0].size(0)

        loss_class1 = torch.mean(torch.stack([self.criterion(xs[i], labels) for i in range(self.num)]))
        loss_class2 = self.criterion(x_st, labels)
        loss_kd_logits = loss_fn_kd(x_st, torch.mean(torch.stack(xs), dim=0))
        weights = torch.tensor(self.hparams["weights"]).to(loss_class1.device)

        # loss_cos = -torch.mean(torch.stack([weights[i] * torch.cosine_similarity(
        #     middle_st[i].view((batchsize, -1)), middles[i].detach().view(batchsize, -1)) for i in range(4)])) + 1
        loss_cos = 0

        loss_kd_feature = torch.mean(weights * torch.stack([self.feature_loss(middle_st[i], middles[i]) for i in range(4)]))
        # loss_kd_feature = torch.tensor(weights) * torch.tensor([self.feature_criterion(middle_st[i], middles[i]) for i in range(4)])/4
        # loss_kd_feature = torch.mean(torch.stack([
        #     weights[i] * torch.nn.functional.mse_loss(middle_st[i], middles[i]) ** 0.5 for i in range(4)]))

        if self.hparams["train_student"]:
            loss = (loss_class1 + loss_class2) / 2 * (1 - self.hparams["weight_kd"]) + (
                    loss_kd_feature * self.hparams["weight_feature"] + loss_kd_logits * (
                    1 - self.hparams["weight_feature"])) * self.hparams["weight_kd"]
        else:
            loss = loss_class1

        if meter.phase == "test":
            import pdb
            pdb.set_trace()

        with torch.no_grad():
            if not self.hparams["train_student"]:
                predictions = xs
                acc = torch.zeros(self.num)
                for i in range(self.num):
                    meter.update(labels, predictions[i].detach(), loss.detach())
                    acc[i] = (torch.max(predictions[i], dim=1)[1] == labels).float().mean()
                acc = torch.mean(acc)
            else:
                predictions = x_st
                meter.update(labels, predictions.detach(), loss.detach())
                acc = (torch.max(predictions, dim=1)[1] == labels).float().mean()
        return {'loss': loss,
                'progress_bar': {'acc': acc, 'loss1': loss_class1, 'loss2': loss_class2, 'logits': loss_kd_logits,
                                 'feature': loss_kd_feature, "cos": loss_cos},
                'log': {'loss1': loss_class1, 'loss2': loss_class2, 'logits': loss_kd_logits, 'feature': loss_kd_feature,
                        "cos": loss_cos}}


class Proposed_Metric(CIFAR_LightningModule):
    models: nn.ModuleList

    def __init__(self, hparams):
        super().__init__(hparams)
        from model.utils import model_init
        from model.utils import freeze
        self.models = nn.ModuleList(
            [get_classifier(hparams["classifiers"][i], hparams["dataset"]) for i in range(self.num)])
        for idx, model in enumerate(self.models):
            checkpoint = torch.load(hparams["pretrain_paths"][idx], map_location='cpu')
            model.load_state_dict({key[6:]: value for key, value in checkpoint["state_dict"].items()})
            freeze(model)
            model.cuda()

        self.channels = [32, 64, 128, 256, 512]
        assert 1 <= hparams["pos"] <= 5
        self.gen0 = self._make_generator(self.channels[hparams["pos"]], hparams["gen_level"])
        self.gen1 = self._make_generator(self.channels[hparams["pos"]], hparams["gen_level"])
        model_init(self.gen0)
        model_init(self.gen1)

    def _make_generator(self, channel, level):
        if level == 0:
            return nn.Identity()
        elif level == 1:
            return nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1, stride=1)
            )
        else:
            raise NotImplementedError(f"level {level} is not implemented")

    def choose_optimizer(self):
        from torch.optim import SGD, Adam
        from model.utils import get_trainable_params
        params_to_update = get_trainable_params(self)
        if self.hparams['optimizer'] == 'SGD':
            optimizer = SGD(params_to_update, lr=self.hparams["max_lr"], weight_decay=self.hparams["weight_decay"],
                            momentum=0.9, nesterov=True)
        elif self.hparams['optimizer'] == 'Adam':
            optimizer = Adam(params_to_update, lr=self.hparams['max_lr'], weight_decay=self.hparams['weight_decay'])
        else:
            assert False, "optimizer not implemented"
        return optimizer

    def subblock_step(self, model: torch.nn.Module, step: int, x: torch.Tensor):
        if isinstance(model, ResNet):
            assert 0 <= step <= 4
            if step == 0:
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
            elif step == 1:
                x = model.layer1(x)
            elif step == 2:
                x = model.layer2(x)
            elif step == 3:
                x = model.layer3(x)
            elif step == 4:
                x = model.layer4(x)
            else:
                x = model.avgpool(x)
                x = x.reshape(x.size(0), -1)
                x = model.fc(x)
        else:
            raise NotImplementedError("Other networks is not implemented!")
        return x

    def subblock_interval(self, model: torch.nn.Module, start: int, end: int, x: torch.Tensor):
        for i in range(start, end):
            x = self.subblock_step(model, i, x)
        return x

    def forward(self, images: torch.Tensor):
        x0 = images
        x1 = images.clone()
        x0 = self.subblock_interval(self.models[0], 0, self.hparams["pos"], x0)
        x1 = self.subblock_interval(self.models[1], 0, self.hparams["pos"], x1)
        x0 = self.gen0(x0)
        x1 = self.gen1(x1)
        x0 = self.subblock_interval(self.models[1], self.hparams["pos"], 6, x0)
        x1 = self.subblock_interval(self.models[0], self.hparams["pos"], 6, x1)

        return x0, x1

    def step(self, meter, batch):
        images, labels = batch
        x0, x1 = self.forward(images)

        loss = (self.criterion(x0, labels) + self.criterion(x1, labels)) / 2

        with torch.no_grad():
            predictions = [x0.detach(), x1.detach()]
            acc = torch.zeros(2)
            meter.update(labels, predictions, loss.detach())
            for i in range(2):
                acc[i] = (torch.max(predictions[i], dim=1)[1] == labels).float().mean()

        return {'loss': loss,
                'progress_bar': {'acc1->2': acc[0], 'acc2->1': acc[1]}}

    def get_meter(self, phase: str):
        from meter.classification_meter import MultiClassificationMeter as Meter
        workers = 1 if phase == "test" or self.hparams["backend"] != "ddp" else self.hparams["gpus"]
        return Meter(phase=phase, workers=workers, criterion=self.criterion, num_class=10, model_num=2)
