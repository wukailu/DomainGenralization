from torch.utils.data import Dataset
from datasets.utils import FullDatasetBase, CIFAR10PATH
from torchvision.datasets import cifar
from torchvision import transforms


class CIFAR10(FullDatasetBase):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    img_shape = (3, 32, 32)
    num_classes = 10
    name = "cifar10"

    def gen_train_transforms(self):
        test_transforms, _ = self.gen_test_transforms()
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            test_transforms
        ]), _

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        return cifar.CIFAR10(root=CIFAR10PATH, train=True, download=True,
                             transform=transform, target_transform=target_transform)

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        return cifar.CIFAR10(root=CIFAR10PATH, train=False, download=True,
                             transform=transform, target_transform=target_transform)

    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        return cifar.CIFAR10(root=CIFAR10PATH, train=False, download=True,
                             transform=transform, target_transform=target_transform)

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(Cifar|cifar|CIFAR)([-_])*10$", name)
