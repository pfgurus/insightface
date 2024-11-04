import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import glob
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

from torch.utils.data import ConcatDataset
class GazeImageDataset(torch.utils.data.Dataset):
    '''
    Direct and Non-Direct Gaze Image Dataset
    '''
    def __init__(self,
                 pos_root_dir: str = '/home/pc3/param/datasets/LEAPFramesSubset/generated_0.0.3',
                 pos_patterns: list = ('**/*gen-edit.png',),
                 neg_root_dir: str = '/home/pc3/param/datasets/LEAPFramesSubset/generated_0.0.3',
                 neg_patterns: list = ('**/*orig.png',),
                 resolution: int   = 256):

        self._pos_root_dir  = pos_root_dir
        self._neg_root_dir  = neg_root_dir
        self._pos_patterns  = pos_patterns
        self._neg_patterns  = neg_patterns

        self._resolution    = resolution
        self._augmentation  = None
        self._rng           = np.random.RandomState(0)

        self._img_files = []
        self._labels    = []

        for pattern in self._pos_patterns:
            img_files   = glob.glob(str(Path(self._pos_root_dir)/pattern), recursive=True)
            self._img_files.extend(img_files)
            self._labels.extend([1]*len(img_files))

        for pattern in self._neg_patterns:
            img_files   = glob.glob(str(Path(self._neg_root_dir)/pattern), recursive=True)
            self._img_files.extend(img_files)
            self._labels.extend([0]*len(img_files))

        print(f'Loaded {len(self._img_files)} images,'
              f'{np.sum(self._labels)} positive samples and {len(self._img_files) - np.sum(self._labels)} negative samples')
    def __len__(self):
        return len(self._img_files)

    def __getitem__(self, idx):
        img_file           = self._img_files[idx]
        label              = self._labels[idx]

        # Get Image
        img                = cv2.imread(img_file)
        img                = img.astype(np.float32) / 255
        img                = img[:, :, ::-1]  # BGR -> RGB
        if self._augmentation is not None:
            aug            = self._augmentation.make_transform(self._rng)
            img            = aug.transform_image(img)
        else:
            img            = cv2.resize(img, (self._resolution, self._resolution))

        img                = np.ascontiguousarray(np.moveaxis(img, 2, 0))  # HWC -> CHW
        img                = (img - 0.5)*2

        return {
            'image': img,
            'path': img_file,
            'label': label,
        }


if __name__ == '__main__':
    dataset = GazeImageDataset()
    item    = dataset[0]
    print(f'{item["image"].shape}')


