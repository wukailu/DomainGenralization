from os import path
from torch.utils.data import Dataset
from datasets.utils import FullDatasetBase, IMAGENET100PATH
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ImageNet100(FullDatasetBase):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_shape = (3, 224, 224)
    num_classes = 100
    name = "imagenet100"

    def gen_train_transforms(self):
        base_transforms, _ = self.gen_base_transforms()
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            base_transforms
        ])
        return train_transforms, _

    def gen_test_transforms(self):
        base_transforms, _ = self.gen_base_transforms()
        test_transforms = transforms.Compose([
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            base_transforms
        ])
        return test_transforms, _

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        return ImageFolder(root=path.join(IMAGENET100PATH, 'ImageNet100/train'),
                           transform=transform, target_transform=target_transform)

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        return ImageFolder(root=path.join(IMAGENET100PATH, 'ImageNet100/val'),
                           transform=transform, target_transform=target_transform)

    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        return ImageFolder(root=path.join(IMAGENET100PATH, 'ImageNet100/val'),
                           transform=transform, target_transform=target_transform)

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(imagenet|ImageNet|Imagenet)([-_])*100$", name)
