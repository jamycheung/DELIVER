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

class DELIVER(Dataset):
    """
    num_classes: 25
    """
    CLASSES = ["Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk", "Vegetation", 
                "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GroundRail", 
                "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler", "Bus", "Truck"]

    PALETTE = torch.tensor([[70, 70, 70],
            [100, 40, 40],
            [55, 90, 80],
            [220, 20, 60],
            [153, 153, 153],
            [157, 234, 50],
            [128, 64, 128],
            [244, 35, 232],
            [107, 142, 35],
            [0, 0, 142],
            [102, 102, 156],
            [220, 220, 0],
            [70, 130, 180],
            [81, 0, 81],
            [150, 100, 100],
            [230, 150, 140],
            [180, 165, 180],
            [250, 170, 30],
            [110, 190, 160],
            [170, 120, 50],
            [45, 60, 150],
            [145, 170, 100],
            [  0,  0, 230], 
            [  0, 60, 100],
            [  0,  0, 70],
            ])
    
    def __init__(self, root: str = 'data/DELIVER', split: str = 'train', transform = None, modals = ['img'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        # --- split as case
        if case is not None:
            assert case in ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'], "Case name not available."
            _temp_files = [f for f in self.files if case in f]
            self.files = _temp_files
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} {case} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        rgb = str(self.files[index])
        x1 = rgb.replace('/img', '/hha').replace('_rgb', '_depth')
        x2 = rgb.replace('/img', '/lidar').replace('_rgb', '_lidar')
        x3 = rgb.replace('/img', '/event').replace('_rgb', '_event')
        lbl_path = rgb.replace('/img', '/semantic').replace('_rgb', '_semantic')

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        H, W = sample['img'].shape[1:]
        if 'depth' in self.modals:
            sample['depth'] = self._open_img(x1)
        if 'lidar' in self.modals:
            sample['lidar'] = self._open_img(x2)
        if 'event' in self.modals:
            eimg = self._open_img(x3)
            sample['event'] = TF.resize(eimg, (H, W), TF.InterpolationMode.NEAREST)
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


if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    for case in cases:

        trainset = DELIVER(transform=traintransform, split='val', case=case)
        trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

        for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))