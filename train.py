"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file trains a task network along with AIM.
This file mostly contains boilerplate code and we do not add code comments to
this file for the ease of exposition.
"""
import os
import copy
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

def train_model(model,
                dataloader,
                criterion_task,
                criterion_direction,
                optimizer, scheduler,
                num_epochs,
                dataset_sizes):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_task, embeddings_x, embeddings_y = model(inputs)
                    _, preds = torch.max(outputs_task, 1)
                    loss_task = criterion_task(outputs_task, labels)
                    loss_direct = criterion_direction(embeddings_x, embeddings_y)
                    loss = loss_task + loss_direct

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            model_wts = copy.deepcopy(model.state_dict())
            torch.save(model_wts, 'ckpt_root' + '/{0}.pkl'.format(epoch))

            # deep copy the model
            if phase == 'test' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'ckpt_root' + '/best.pkl')

if __name__ == '__main__':
    cfg = get_config(path_config='config/config.yaml')

    # Device setup
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device.CUDA_VISIBLE_DEVICES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data setup
    dataset_name = 'CUB200'
    print('Dataset name: ', dataset_name)
    cfg_data = cfg.data[dataset_name]

    augmentations = cfg_data.augmentations.train
    print('Data Augmentations: ')
    for augmentation in augmentations:
        print('\t', augmentation)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=augmentations.Resize.size),
            transforms.RandomCrop(size=augmentations.RandomCrop.size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=augmentations.ColorJitter.contrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=augmentations.Normalize.mean,
                                 std=augmentations.Normalize.std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(augmentations.Resize.size),
            transforms.CenterCrop(augmentations.RandomCrop.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=augmentations.Normalize.mean,
                                 std=augmentations.Normalize.std)
        ])
    }
    dataset = {partition: CUB200(data_root=cfg_data.data_root,
                                 partition=partition,
                                 transform=data_transforms[partition])
               for partition in ['train', 'test']}
    dataloader_ft = {partition: DataLoader(dataset=dataset[partition],
                                           batch_size=cfg_data.batch_size,
                                           shuffle=cfg_data.shuffle,
                                           num_workers=cfg_data.num_workers)
                     for partition in ['train', 'test']}
    dataset_sizes = {partition: len(dataset[partition])
                     for partition in ['train', 'test']}
    print('Number of samples in training set : ', dataset_sizes['train'])
    print('Number of samples in testing set : ', dataset_sizes['test'])

    # Model setup
    model_ft = Classifier(cfg.model)
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    # Optimizer setup
    criterion_task_ft = nn.CrossEntropyLoss()
    criterion_direction_ft = DirectionalConsistency()
    optimizer_ft = torch.optim.SGD(params=model_ft.parameters(),
                                   lr=cfg.optimizer.SGD.lr,
                                   momentum=cfg.optimizer.SGD.momentum,
                                   weight_decay=cfg.optimizer.SGD.weight_decay)
    lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                       step_size=cfg.optimizer.scheduler.step_size,
                                       gamma=cfg.optimizer.scheduler.gamma)
    # Train
    train_model(model=model_ft,
                dataloader=dataloader_ft,
                criterion_task=criterion_task_ft,
                criterion_direction=criterion_direction_ft,
                optimizer=optimizer_ft,
                scheduler=lr_scheduler,
                num_epochs=cfg.optimizer.num_epochs,
                dataset_sizes=dataset_sizes)
