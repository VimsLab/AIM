"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains the data loader for the CUB200 data set.
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data

class CUB200(data.Dataset):
    """ This class contains functions to load the training set and the test set
    of the pre-processed CUB200 dataset in the dataset directory.
    """
    def __init__(self,
                 data_root='',
                 partition='',
                 transform=None):
        """
        Args:
            data_root: str, root directory where the CUB200 dataset is stored
            partition: str, train or test partition of data
            transform: data augmentations to be applied before loading the data
        """
        self.transform = transform
        self.partition = partition
        self.data = []
        for category_index, category in enumerate(sorted(os.listdir(osp.join(data_root, partition)))):
            category_root = osp.join(osp.join(data_root, partition), category)
            for filename in os.listdir(category_root):
                if filename.endswith('.jpg'):
                    self.data.append((osp.join(category_root, filename), category_index))

    def __getitem__(self, i):
        path, target = self.data[i]
        image = Image.open(path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(target, dtype=torch.long)
        return image, target

    def __len__(self):
        return len(self.data)
