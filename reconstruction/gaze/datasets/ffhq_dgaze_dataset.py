
import glob
from pathlib import Path
import pandas as pd
from common.bbox import BBox
from copy import deepcopy

import cv2
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import common.utils as cu

from torch.utils.data import ConcatDataset
import albumentations as A



class FFHQDGazeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir: str = '/raid/datasets/ffhq/images1024x1024',
                 anno_file: str = '/raid/datasets/ffhq/valerija_annotations/ffhq-anp-direct-gaze-anno.pkl',
                 resolution: int = 256,
                 ):

        self._root_dir      = root_dir
        self._anno_file     = anno_file
        self._resolution    = resolution
        self._rng           = np.random.RandomState(0)
        self._categories    = {1:[10,20]}

        # Filter annotations to only include categories listed categories
        annotations = pickle.load(open(self._anno_file, 'rb'))['data']
        self._img_files = []
        self._labels    = []
        for img_file, anno in annotations.items():
            if 'category' in anno:
                if anno['category'] in self._categories[1]:
                    self._img_files.append(str(Path(self._root_dir)/img_file))
                    self._labels.append(1)

        transform_list = \
                [
                A.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
                A.ISONoise(p=0.1),
                A.MedianBlur(blur_limit=(1, 7), p=0.1),
                A.GaussianBlur(blur_limit=(1, 7), p=0.1),
                A.MotionBlur(blur_limit=(5, 13), p=0.1),
                A.ImageCompression(quality_lower=10, quality_upper=90, p=0.05),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, interpolation=cv2.INTER_LINEAR,
                                   border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.6),]
        self.transform = A.ReplayCompose(transform_list)

        print(f'Loaded {len(self._img_files)} images for FFHQ Gaze dataset with')

    def __len__(self):
        return len(self._img_files)

    def __getitem__(self, idx):
        img_file           = self._img_files[idx]
        label              = self._labels[idx]

        # Get Image
        img                = cv2.imread(img_file)
        img                = img[:, :, ::-1]  # BGR -> RGB
        img                 = cv2.resize(img, (self._resolution, self._resolution))
        img                 = self.transform(image=img)['image']
        img                 = img.astype(np.float32) / 255

        img                = np.ascontiguousarray(np.moveaxis(img, 2, 0))  # HWC -> CHW
        img                = (img - 0.5)*2

        return {
            'image': img,
            'path': img_file,
            'label': label,
        }


if __name__ == '__main__':
    dataset = FFHQDGazeDataset()
    item    = dataset[0]
    print(item['image'].shape[0])