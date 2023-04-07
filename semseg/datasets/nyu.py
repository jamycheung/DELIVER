import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation

class NYU(Dataset):
    """
    num_classes: 40
    """
    CLASSES = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

    PALETTE = None

    def __init__(self, root: str = 'data/NYUDepthv2', split: str = 'train', transform = None, modals = ['img', 'depth'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = self._get_file_names(split)
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])
        rgb = os.path.join(*[self.root, 'RGB', item_name+'.jpg'])
        x1 = os.path.join(*[self.root, 'HHA', item_name+'.jpg'])
        lbl_path = os.path.join(*[self.root, 'Label', item_name+'.png'])

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        if 'depth' in self.modals:
            sample['depth'] = self._open_img(x1)
        if 'lidar' in self.modals:
            raise NotImplementedError()
        if 'event' in self.modals:
            raise NotImplementedError()
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        label[label==255] = 0
        label -= 1
        sample['mask'] = label
        
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(self.root, 'test.txt') if split_name == 'val' else os.path.join(self.root, 'train.txt')
        file_names = []
        with open(source) as f:
            files = f.readlines()
        for item in files:
            file_name = item.strip()
            if ' ' in file_name:
                file_name = file_name.split(' ')[0]
            file_names.append(file_name)
        return file_names


if __name__ == '__main__':
    traintransform = get_train_augmentation((480, 640), seg_fill=255)

    trainset = NYU(transform=traintransform, split='val')
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        print(torch.unique(lbl))
