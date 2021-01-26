from os import path
from typing import List

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.utils import FullDatasetBase, PACSPATH


class SinglePACS(Dataset):
    def __init__(self, subDatasetName, split, transform):
        self.filename = path.join(PACSPATH, 'kfold', subDatasetName + '_' + split + '.hdf5')
        self.transform = transform
        self.split = split
        domain_data = h5py.File(self.filename, 'r')
        self.pacs_imgs = domain_data.get('images')
        self.pacs_labels = np.array(domain_data.get('labels')) - 1  # Convert labels in the range(1,7) into (0,6)
        print('Domain ', subDatasetName)
        print('Image: ', self.pacs_imgs.shape, ' Labels: ', self.pacs_labels.shape,
              ' Out Classes: ', len(np.unique(self.pacs_labels)))
        unique, counts = np.unique(self.pacs_labels, return_counts=True)
        self.num_class = np.amax(unique) + 1
        self.max_class_size = np.amax(counts)

    def __len__(self):
        return len(self.pacs_imgs)

    def __getitem__(self, index):
        curr_img = Image.fromarray(self.pacs_imgs[index, :, :, :].astype('uint8'), 'RGB')
        img = self.transform(curr_img)
        labels = torch.eye(self.num_class)[self.pacs_labels[index]]

        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        return img, labels, index


class DomainConcat(Dataset):
    def __init__(self, dataset_list: List[SinglePACS]):
        self.dataset_list = dataset_list
        self.len = sum([len(d) for d in dataset_list])
        self.domain_num = len(dataset_list)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        for idx, d in enumerate(self.dataset_list):
            if item < len(d):
                img, labels, index = d[item]
                domain = torch.eye(self.domain_num)[idx]
                return img, labels, domain.long(), index
            else:
                item -= len(d)
        raise IndexError()


class PACS(FullDatasetBase):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_shape = (3, 224, 224)
    num_classes = 10
    name = "cifar10"

    def __init__(self, source_domain_list, target_domain, **kwargs):
        self.source_name_list = source_domain_list
        self.target_name = target_domain
        super().__init__(**kwargs)

    def gen_train_transforms(self):
        base_transforms, _ = self.gen_base_transforms()
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            base_transforms
        ]), _

    def gen_test_transforms(self):
        base_transforms, _ = self.gen_base_transforms()
        return transforms.Compose([
            transforms.Resize((224, 224)),
            base_transforms
        ]), _

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        return DomainConcat([SinglePACS(name, "train", transform) for name in self.source_name_list])

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        return DomainConcat([SinglePACS(name, "val", transform) for name in self.source_name_list])

    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        return DomainConcat([SinglePACS(self.target_name, 'test', transform)])

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(PACS|pacs)", name)


if __name__ == "__main__":
    d = PACS(["art_painting", "cartoon", "photo"], "sketch")
    for i in range(10):
        print(d[i])
