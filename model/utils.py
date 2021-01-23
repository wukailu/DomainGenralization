import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from torch.utils.tensorboard.summary import hparams
from typing import List
from datasets import query_dataset


class Partial_Detach(nn.Module):
    def __init__(self, alpha=0):
        """
        When alpha = 0, it's completely detach, when alpha = 1, it's identity
        :param alpha:
        """
        super().__init__()
        self.alpha = 0

    def forward(self, inputs: torch.Tensor):
        if self.alpha == 0:
            return inputs.detach()
        elif self.alpha == 1:
            return inputs
        else:
            return inputs * self.alpha + inputs.detach() * (1 - self.alpha)


class Flatten(nn.Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor):
        return x.flatten(start_dim=self.start_dim)


def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def get_trainable_params(model):
    # print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print("\t", repr(name))
            params_to_update.append(param)
    return params_to_update


def model_init(model: torch.nn.Module):
    from torch import nn
    for name, ch in model.named_children():
        print(f"{name} is initialized")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)) and m.weight.requires_grad:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MyTensorBoardLogger(TensorBoardLogger):

    def __init__(self, *args, **kwargs):
        super(MyTensorBoardLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, *args, **kwargs):
        pass

    @rank_zero_only
    def log_hyperparams_metrics(self, params: dict, metrics: dict) -> None:
        params = self._convert_params(params)
        exp, ssi, sei = hparams(params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)
        # some alternative should be added
        self.tags.update(params)


def get_classifier(classifier: str, dataset: str) -> torch.nn.Module:
    dataset_type = query_dataset(dataset)
    num_classes = dataset_type.num_classes
    if classifier.endswith("_imagenet"):
        from model.imagenet_models import model_dict
        return model_dict[classifier[:-9]](num_classes=num_classes)
    else:
        from model.basic_cifar_models import model_dict
        return model_dict[classifier](num_classes=num_classes)


def load_models(path_list: List[str], dataset: str) -> nn.ModuleList:
    num = len(path_list)
    models: nn.ModuleList = nn.ModuleList([])
    for idx in range(num):
        if path_list[idx].startswith("predefined_"):
            from model.basic_cifar_models import model_dict
            model = model_dict[path_list[idx][len("predefined_"):]]()
        else:
            checkpoint = torch.load(path_list[idx], map_location='cpu')
            if isinstance(checkpoint, dict):

                # If it's a lightning model
                last_param = checkpoint['hyper_parameters']
                if 'dataset' in hparams and dataset != 'concat':
                    if last_param.dataset != dataset:
                        print(
                            f"WARNING!!!!!!! Model trained on {last_param.dataset} will run on {hparams['dataset']}!!!!!!!")
                    assert query_dataset(last_param.dataset).num_classes == query_dataset(
                        hparams['dataset']).num_classes

                model = get_classifier(last_param.backbone, last_param.dataset)
                model.load_state_dict({key[6:]: value for key, value in checkpoint["state_dict"].items()})
            elif isinstance(checkpoint, nn.Module):
                # Maybe it's just a torch.save(model) and torch.load(model)
                model = checkpoint
            else:
                assert False, "checkpoint type not correct!"
        models.append(model)
    return models


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


def print_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of params: {pytorch_total_params:,}')
    return pytorch_total_params
