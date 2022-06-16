"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file tests a task network along with AIM.
This file mostly contains boilerplate code and we do not add code comments to
this file for the ease of exposition.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.autograd import Variable
from config import get_config
from data import CUB200
from models import Classifier
from loss import DirectionalConsistency

def test_model(model, dataloader, dataset_size):
    model.eval()   # Set model to evaluate mode
    running_corrects = 0
    print('Testing on {0} images...'.format(dataset_size*10))

    for inputs, labels in dataloader:
        inputs = Variable(inputs).to(device)
        bs, ncrops, c, h, w = inputs.shape
        labels = Variable(labels).to(device)
        inputs = inputs.reshape(-1, c, h, w)

        with torch.set_grad_enabled(False):
            temp_outputs_task, _, _ = model(inputs)
            outputs_task = temp_outputs_task.view(bs, ncrops, -1).mean(1)
            _, preds = torch.max(outputs_task, 1)

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / dataset_size

    print('Test Acc: {:.4f}'.format(epoch_acc))

if __name__ == '__main__':
    cfg = get_config(path_config='config/config.yaml')

    # Device setup
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device.CUDA_VISIBLE_DEVICES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random seed setup
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(False)

    # Data setup
    dataset_name = 'CUB200'
    print('Dataset name: ', dataset_name)
    cfg_data = cfg.data[dataset_name]
    augmentations = cfg_data.augmentations.test
    print('Data Augmentations: ')
    for augmentation in augmentations:
        print('\t', augmentation)
    data_transforms = transforms.Compose([
        transforms.Resize(augmentations.Resize.size),
        transforms.TenCrop(augmentations.TenCrop.size),
        transforms.Lambda(lambda crops: \
            torch.stack([transforms.ToTensor()(crop)
                         for crop in crops])),
        transforms.Lambda(lambda crops: \
            torch.stack([
                transforms.Normalize(augmentations.Normalize.mean,
                                     augmentations.Normalize.std)(crop)
                for crop in crops])),
    ])
    dataset = CUB200(data_root=cfg_data.data_root,
                     partition='test',
                     transform=data_transforms)
    # Batch size is set manually to avoid OOM due to 10 crop testing
    dataloader_ft = DataLoader(dataset=dataset,
                               batch_size=16,
                               shuffle=False,
                               num_workers=cfg_data.num_workers)
    print('Number of samples in testing set : ', len(dataset))

    # Model setup
    model_ft = Classifier(cfg.model)
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    # Load best weights
    model_ft.load_state_dict(torch.load('./ckpt_root/' + 'best.pkl'))

    #Test
    test_model(model=model_ft,
               dataloader=dataloader_ft,
               dataset_size=len(dataset))
