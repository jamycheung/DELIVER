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

class UrbanLF(Dataset):
    """
    num_classes: 14
    """
    CLASSES = ['bike','building','fence','others','person','pole','road','sidewalk','traffic sign','vegetation','vehicle','bridge','rider','sky']

    PALETTE = [[168,198,168],[198,0,0],[202,154,198],[0,0,0],[100,198,198],[198,100,0],[52,42,198],[154,52,192],[198,0,168],[0,198,0],[198,186,90],[108,107,161],[156,200,26],[158,179,202]]

    def __init__(self, root: str = 'data/UrBanLF/Syn', split: str = 'train', transform = None, modals = ['img', '5_1', '5_2', '5_3', '5_4', '5_6', '5_7', '5_8', '5_9'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = sorted(glob.glob(os.path.join(*[root, split, '*', '5_5.png'])))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])
        rgb = item_name
        rgb_dir_name = os.path.dirname(rgb)
        lf_names = []
        lf_paths = []
        for i in range(1, 10):
            for j in range(1, 10):
                lf_name = '{}_{}'.format(i, j)
                if lf_name != '5_5':
                    if lf_name in self.modals:
                        lf_names.append(lf_name)
                        lf_paths.append(os.path.join(rgb_dir_name, lf_name+'.png'))
        if 'real' in self.root:
            lbl_path = item_name.replace('5_5', 'label')
        elif 'Syn' in self.root:
            lbl_path = item_name.replace('5_5.png', '5_5_label.npy')
        else:
            raise NotImplemented
        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        if len(self.modals) > 1:
            for i, lf_name in enumerate(lf_names):
                assert lf_name in lf_paths[i], "Not matched."
                sample[lf_name] = self._open_img(lf_paths[i])
        
        if 'real' in self.root:
            label = io.read_image(lbl_path)
            label = self.encode(label.numpy())
        elif 'Syn' in self.root:
            label = np.load(lbl_path)
            label[label==255] = 0
            label -= 1
            label = torch.tensor(label[None,...])
        else:
            raise NotImplemented
        sample['mask'] = label
        
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = label.long().squeeze(0)
        sample_list = [sample['img']]
        sample_list += [sample[k] for k in lf_names]
        return sample_list, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        label = label.transpose(1,2,0) # C, H, W -> H, W, C
        label_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.int16)
        for ii, lb in enumerate(self.PALETTE):
            label_mask[np.where(np.all(label == lb, axis=-1))[:2]] = ii
        label_mask = label_mask[None,...].astype(int)
        return torch.from_numpy(label_mask)


if __name__ == '__main__':
    traintransform = get_train_augmentation((432, 623), seg_fill=255)

    trainset = UrbanLF(transform=traintransform, modals=['img', '1_2'])
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        print(torch.unique(lbl))
