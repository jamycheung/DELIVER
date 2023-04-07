import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from torchvision import transforms
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation
import cv2
import random
from PIL import Image, ImageOps, ImageFilter

class MCubeS(Dataset):
    """
    num_classes: 20
    """
    CLASSES = ['asphalt','concrete','metal','road_marking','fabric','glass','plaster','plastic','rubber','sand',
    'gravel','ceramic','cobblestone','brick','grass','wood','leaf','water','human','sky',]

    PALETTE = torch.tensor([[ 44, 160,  44],
                [ 31, 119, 180],
                [255, 127,  14],
                [214,  39,  40],
                [140,  86,  75],
                [127, 127, 127],
                [188, 189,  34],
                [255, 152, 150],
                [ 23, 190, 207],
                [174, 199, 232],
                [196, 156, 148],
                [197, 176, 213],
                [247, 182, 210],
                [199, 199, 199],
                [219, 219, 141],
                [158, 218, 229],
                [ 57,  59, 121],
                [107, 110, 207],
                [156, 158, 222],
                [ 99, 121,  57]])

    def __init__(self, root: str = 'data/MCubeS/multimodal_dataset', split: str = 'train', transform = None, modals = ['image', 'aolp', 'dolp', 'nir'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.split = split
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self._left_offset = 192
        	
        self.img_h = 1024
        self.img_w = 1224
        max_dim = max(self.img_h, self.img_w)
        u_vec = (np.arange(self.img_w)-self.img_w/2)/max_dim*2
        v_vec = (np.arange(self.img_h)-self.img_h/2)/max_dim*2
        self.u_map, self.v_map = np.meshgrid(u_vec, v_vec)
        self.u_map = self.u_map[:,:self._left_offset]

        self.base_size = 512
        self.crop_size = 512
        self.files = self._get_file_names(split)
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])
        rgb = os.path.join(*[self.root, 'polL_color', item_name+'.png'])
        x1 = os.path.join(*[self.root, 'polL_aolp_sin', item_name+'.npy'])
        x1_1 = os.path.join(*[self.root, 'polL_aolp_cos', item_name+'.npy'])
        x2 = os.path.join(*[self.root, 'polL_dolp', item_name+'.npy'])
        x3 = os.path.join(*[self.root, 'NIR_warped', item_name+'.png'])
        lbl_path = os.path.join(*[self.root, 'GT', item_name+'.png'])
        nir_mask = os.path.join(*[self.root, 'NIR_warped_mask', item_name+'.png'])
        _mask = os.path.join(*[self.root, 'SS', item_name+'.png'])

        _img = cv2.imread(rgb,-1)[:,:,::-1]
        _img = _img.astype(np.float32)/65535 if _img.dtype==np.uint16 else _img.astype(np.float32)/255
        _target = cv2.imread(lbl_path,-1)
        _mask = cv2.imread(_mask,-1)
        _aolp_sin = np.load(x1)
        _aolp_cos = np.load(x1_1)
        _aolp = np.stack([_aolp_sin, _aolp_cos, _aolp_sin], axis=2) # H x W x 3
        dolp = np.load(x2)
        _dolp = np.stack([dolp, dolp, dolp], axis=2) # H x W x 3
        nir  = cv2.imread(x3,-1)
        nir = nir.astype(np.float32)/65535 if nir.dtype==np.uint16 else nir.astype(np.float32)/255
        _nir = np.stack([nir, nir, nir], axis=2) # H x W x 3

        _nir_mask = cv2.imread(nir_mask,0)

        _img, _target, _aolp, _dolp, _nir, _nir_mask, _mask = _img[:,self._left_offset:], _target[:,self._left_offset:], \
               _aolp[:,self._left_offset:], _dolp[:,self._left_offset:], \
               _nir[:,self._left_offset:], _nir_mask[:,self._left_offset:], _mask[:,self._left_offset:]
        sample = {'image': _img, 'label': _target, 'aolp': _aolp, 'dolp': _dolp, 'nir': _nir, 'nir_mask': _nir_mask, 'u_map': self.u_map, 'v_map': self.v_map, 'mask':_mask}

        if self.split == "train":
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample = self.transform_val(sample)
        else:
            raise NotImplementedError()
        label = sample['label'].long()
        sample = [sample[k] for k in self.modals]
        return sample, label

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            FixScaleCrop(crop_size=1024),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(self.root, 'list_folder/test.txt') if split_name == 'val' else os.path.join(self.root, 'list_folder/train.txt')
        file_names = []
        with open(source) as f:
            files = f.readlines()
        for item in files:
            file_name = item.strip()
            if ' ' in file_name:
                # --- KITTI-360
                file_name = file_name.split(' ')[0]
            file_names.append(file_name)
        return file_names

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img -= self.mean
        img /= self.std

        nir = sample['nir']
        nir = np.array(nir).astype(np.float32)
        # nir /= 255

        return {'image': img,
                'label': mask,
                'aolp' : sample['aolp'], 
                'dolp' : sample['dolp'], 
                'nir'  : nir, 
                'nir_mask': sample['nir_mask'],
                'u_map': sample['u_map'],
                'v_map': sample['v_map'],
                'mask':sample['mask']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        aolp = sample['aolp']
        dolp = sample['dolp']
        nir  = sample['nir']
        nir_mask  = sample['nir_mask']
        SS=sample['mask']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        aolp = np.array(aolp).astype(np.float32).transpose((2, 0, 1))
        dolp = np.array(dolp).astype(np.float32).transpose((2, 0, 1))
        SS = np.array(SS).astype(np.float32)
        nir = np.array(nir).astype(np.float32).transpose((2, 0, 1))
        nir_mask = np.array(nir_mask).astype(np.float32)
        
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        aolp = torch.from_numpy(aolp).float()
        dolp = torch.from_numpy(dolp).float()
        SS = torch.from_numpy(SS).float()
        nir = torch.from_numpy(nir).float()
        nir_mask = torch.from_numpy(nir_mask).float()

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map = torch.from_numpy(u_map.astype(np.float32)).float()
        v_map = torch.from_numpy(v_map.astype(np.float32)).float()

        return {'image': img,
                'label': mask,
                'aolp' : aolp,
                'dolp' : dolp,
                'nir'  : nir,
                'nir_mask'  : nir_mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask':SS}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        aolp = sample['aolp']
        dolp = sample['dolp']
        nir  = sample['nir']
        nir_mask  = sample['nir_mask']
        u_map = sample['u_map']
        v_map = sample['v_map']
        SS=sample['mask']
        if random.random() < 0.5:
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # nir = nir.transpose(Image.FLIP_LEFT_RIGHT)

            img = img[:,::-1]
            mask = mask[:,::-1]
            nir = nir[:,::-1]
            nir_mask = nir_mask[:,::-1]
            aolp  = aolp[:,::-1]
            dolp  = dolp[:,::-1]
            SS  = SS[:,::-1]
            u_map = u_map[:,::-1]

        return {'image': img,
                'label': mask,
                'aolp' : aolp,
                'dolp' : dolp,
                'nir'  : nir,
                'nir_mask'  : nir_mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask':SS}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        nir  = sample['nir']
        if random.random() < 0.5:
            radius = random.random()
            # img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            # nir = nir.filter(ImageFilter.GaussianBlur(radius=radius))
            img = cv2.GaussianBlur(img, (0,0), radius)
            nir = cv2.GaussianBlur(nir, (0,0), radius)

        return {'image': img,
                'label': mask,
                'aolp' : sample['aolp'], 
                'dolp' : sample['dolp'], 
                'nir'  : nir, 
                'nir_mask': sample['nir_mask'],
                'u_map': sample['u_map'],
                'v_map': sample['v_map'],
                'mask':sample['mask']}

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        aolp = sample['aolp']
        dolp = sample['dolp']
        nir = sample['nir']
        nir_mask = sample['nir_mask']
        SS=sample['mask']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # w, h = img.size
        h, w = img.shape[:2]
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            
        # random crop crop_size
        # w, h = img.size
        h, w = img.shape[:2]

        # x1 = random.randint(0, w - self.crop_size)
        # y1 = random.randint(0, h - self.crop_size)
        x1 = random.randint(0, max(0, ow - self.crop_size))
        y1 = random.randint(0, max(0, oh - self.crop_size))

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map    = cv2.resize(u_map,(ow,oh))
        v_map    = cv2.resize(v_map,(ow,oh))
        aolp     = cv2.resize(aolp ,(ow,oh))
        dolp     = cv2.resize(dolp ,(ow,oh))
        SS     = cv2.resize(SS ,(ow,oh))
        img      = cv2.resize(img  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
        mask     = cv2.resize(mask ,(ow,oh), interpolation=cv2.INTER_NEAREST)
        nir      = cv2.resize(nir  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
        nir_mask = cv2.resize(nir_mask  ,(ow,oh), interpolation=cv2.INTER_NEAREST)
        if short_size < self.crop_size:
            u_map_ = np.zeros((oh+padh,ow+padw))
            u_map_[:oh,:ow] = u_map
            u_map = u_map_
            v_map_ = np.zeros((oh+padh,ow+padw))
            v_map_[:oh,:ow] = v_map
            v_map = v_map_
            aolp_ = np.zeros((oh+padh,ow+padw,3))
            aolp_[:oh,:ow] = aolp
            aolp = aolp_
            dolp_ = np.zeros((oh+padh,ow+padw,3))
            dolp_[:oh,:ow] = dolp
            dolp = dolp_

            img_ = np.zeros((oh+padh,ow+padw,3))
            img_[:oh,:ow] = img
            img = img_
            SS_ = np.zeros((oh+padh,ow+padw))
            SS_[:oh,:ow] = SS
            SS = SS_
            mask_ = np.full((oh+padh,ow+padw),self.fill)
            mask_[:oh,:ow] = mask
            mask = mask_
            nir_ = np.zeros((oh+padh,ow+padw,3))
            nir_[:oh,:ow] = nir
            nir = nir_
            nir_mask_ = np.zeros((oh+padh,ow+padw))
            nir_mask_[:oh,:ow] = nir_mask
            nir_mask = nir_mask_

        u_map = u_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        v_map = v_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        aolp  =  aolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
        dolp  =  dolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
        img   =   img[y1:y1+self.crop_size, x1:x1+self.crop_size]
        mask  =  mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        nir   =   nir[y1:y1+self.crop_size, x1:x1+self.crop_size]
        SS   =   SS[y1:y1+self.crop_size, x1:x1+self.crop_size]
        nir_mask = nir_mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        return {'image': img,
                'label': mask,
                'aolp' : aolp,
                'dolp' : dolp,
                'nir'  : nir,
                'nir_mask'  : nir_mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask':SS}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        aolp = sample['aolp']
        dolp = sample['dolp']
        nir = sample['nir']
        nir_mask = sample['nir_mask']
        SS = sample['mask']

        # w, h = img.size
        h, w = img.shape[:2]

        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # nir = nir.resize((ow, oh), Image.BILINEAR)

        # center crop
        # w, h = img.size
        # h, w = img.shape[:2]
        x1 = int(round((ow - self.crop_size) / 2.))
        y1 = int(round((oh - self.crop_size) / 2.))
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # nir = nir.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map = cv2.resize(u_map,(ow,oh))
        v_map = cv2.resize(v_map,(ow,oh))
        aolp  = cv2.resize(aolp ,(ow,oh))
        dolp  = cv2.resize(dolp ,(ow,oh))
        SS  = cv2.resize(SS ,(ow,oh))
        img   = cv2.resize(img  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask ,(ow,oh), interpolation=cv2.INTER_NEAREST)
        nir   = cv2.resize(nir  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
        nir_mask = cv2.resize(nir_mask,(ow,oh), interpolation=cv2.INTER_NEAREST)
        u_map = u_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        v_map = v_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        aolp  =  aolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
        dolp  =  dolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
        img   =   img[y1:y1+self.crop_size, x1:x1+self.crop_size]
        mask  =  mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        SS  =  SS[y1:y1+self.crop_size, x1:x1+self.crop_size]
        nir   =   nir[y1:y1+self.crop_size, x1:x1+self.crop_size]
        nir_mask = nir_mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        return {'image': img,
                'label': mask,
                'aolp' : aolp,
                'dolp' : dolp,
                'nir'  : nir,
                'nir_mask'  : nir_mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask':SS}


if __name__ == '__main__':
    traintransform = get_train_augmentation((1024, 1224), seg_fill=255)

    trainset = MCubeS(transform=traintransform, split='val')
    trainloader = DataLoader(trainset, batch_size=1, num_workers=0, drop_last=False, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        print(torch.unique(lbl))