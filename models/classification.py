"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains the Classifier class that coordinates between the Attention
Module, the Spatial Warper, and the Task Network.
"""
import torch.nn as nn
from .task_network import resnet18
from .warper import SpatialWarper
from .attention import AttentionModule

class Classifier(nn.Module):
    """
    This class coordinates between the Attention Module, the Spatial Warper, and
    the Task Network to classify images.
    """
    def __init__(self, cfg_model):
        """
        Args:
            cfg_model: dict, AIM and model configurations
        """
        super(Classifier, self).__init__()

        self.attention_X = AttentionModule(cfg_model.Attention.AttentionModule)
        self.attention_Y = AttentionModule(cfg_model.Attention.AttentionModule)
        self.warp = SpatialWarper(cfg_model.Warper.SpatialWarper)
        self.task_network = resnet18(pretrained=cfg_model.TaskNetwork.ResNet.pretrained)

    def forward(self, img):
        """
        Args:
            - img: (?, num_channels, height, width), input image(s)

        Returns:
            - output_task: predictions from the task network

            - Sx: (?, num_verts), edge sensitivities along the X-axis

            - Sy: (?, num_verts), edge sensitivities along the Y-axis
        """
        Sx = self.attention_X(img)
        Sy = self.attention_Y(img)
        img_aug = self.warp(img, Sx, Sy)

        output_task = self.task_network(img_aug)

        return output_task, Sx, Sy
