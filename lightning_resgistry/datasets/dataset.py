import os
import pickle as pkl
import random
from sklearn.model_selection import train_test_split

import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset
import nibabel as nib

from .builder import DATASETS, build_data_augmenter
from .augment import Augment3D


@DATASETS.register_module()
class ADDataset(Dataset):
    def __init__(self,
                 data_root=None,
                 split=None,
                 random_seed=42,
                 augment=False,
                 augmenter=None,
                 test_size=0.2,
                 val_size=0.25,
                 ):
        self.data_dir = data_root
        self.split = split
        self.random_seed = random_seed
        self.augment = augment
        self.test_size = test_size
        self.val_size = val_size
        self.check_files()

        if self.augment:
            self.augmenter = build_data_augmenter(augmenter)

    def check_files(self):
        # 加载文件列表
        # 获取目录下所有文件名
        files = os.listdir(self.data_dir)

        # 只保留特定类型的文件，比如 .mat
        files = [f for f in files if f.endswith(".mat")]


        # 转成完整路径
        file_list = [os.path.join(self.data_dir, f) for f in files]

        # 固定随机种子并划分数据集
        train_val, test = train_test_split(
            file_list, test_size=self.test_size, random_state=self.random_seed)
        train, val = train_test_split(
            train_val, test_size=self.val_size, random_state=self.random_seed)

        if self.split == 'train':
            self.path_list = train
        elif self.split == 'val':
            self.path_list = val
        elif self.split == 'test':
            self.path_list = test

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        filename = os.path.splitext(os.path.basename(path))[0]
        if path.endswith(".mat"):
            img = sio.loadmat(path)
            img = img['data']
        else:
            img = nib.load(path).get_fdata()
            img = img[None, :, :, :]

        img = torch.from_numpy(img).float()

        # 数据增强
        if self.augment:
            img = self.augmenter(img)
            img = img[0:1, :, :, :]
        # Extract the first character of the filename and convert it to an integer
        label = int(filename[0])
        label = torch.tensor(label, dtype=torch.long)

        return img, label
