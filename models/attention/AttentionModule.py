"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains the Attention Module described in section 4.2 of the paper .
"""
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv import SAGEConv
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import image2graph, sqrt

class AttentionModule(nn.Module):
    """
    The attention module infers edge sensitivities along each independent
    direction in the input data. Once inferred, they are constrained between
    zero and one through min-max normalization.
    """
    def __init__(self, cfg_AttentionModule):
        """
        Args:
        cfg_AttentionModule: dict,  contains the following attributes:
            - sample_mode: str, mode to downsample the image

            - sample_params: dict, parameters related to sample_mode

            - num_verts_out: int, number of vertices/pixels in the final
              attention map output by the AttentionModule

            - num_verts_out: int, number of vertices/pixels in the final
              attention map output by the AttentionModule

            - in_channels: list, number (in order) of input channels for each
              GCN

            - out_channels: list, number (in order) of channels output by each
              GCN

            - bias: bool, If set to True, the GCNs will learn an additive bias

            - use_mask: bool, If set to True, sets the edge sensitivity at the
              border of the attention map(s) to its minimum value

            - mask_size: int, the size of per side mask borders

            - norm: str, the method to normalize attention map(s) between zero
              and one
        """

        super(AttentionModule, self).__init__()
        sample_mode = cfg_AttentionModule.sample_mode.Interpolate
        norm = cfg_AttentionModule.norm

        assert sample_mode.name in ['F.interpolate']
        assert norm in ['Min-Max']

        if sample_mode.name == 'F.interpolate':
            self.sample_params = sample_mode.sample_params
            self.num_verts_in = self.sample_params.size**2
            cwd = os.path.realpath(__file__)
            cwd = cwd[:cwd.rindex(os.sep)]
            if not osp.exists(osp.join(cwd, 'verts.pt')):
                verts, edges = image2graph(num_verts=self.num_verts_in)
                torch.save(verts, osp.join(cwd, 'verts.pt'))
                torch.save(edges, osp.join(cwd, 'edges.pt'))

            self.verts = torch.load(osp.join(cwd, 'verts.pt'))
            self.edges = torch.load(osp.join(cwd, 'edges.pt'))


        self.in_channels = cfg_AttentionModule.in_channels
        self.out_channels = cfg_AttentionModule.out_channels
        self.num_verts_out = cfg_AttentionModule.num_verts_out

        self.use_mask = cfg_AttentionModule.use_mask
        if self.use_mask:
            self.mask = torch.zeros(sqrt(self.num_verts_in),
                                    sqrt(self.num_verts_in)).float()
            mask_size = cfg_AttentionModule.mask_size
            self.mask[mask_size:-mask_size, mask_size:-mask_size] = 1.0

        self.bias = cfg_AttentionModule.bias
        self.conv_1 = SAGEConv(in_channels=self.in_channels[0],
                               out_channels=self.out_channels[0],
                               bias=self.bias)
        self.skip_1 = SAGEConv(in_channels=self.out_channels[0],
                               out_channels=1,
                               bias=self.bias)

        self.conv_2 = SAGEConv(in_channels=self.in_channels[1],
                               out_channels=self.out_channels[1],
                               bias=self.bias)
        self.skip_2 = SAGEConv(in_channels=self.out_channels[1],
                               out_channels=1,
                               bias=self.bias)

        self.conv_3 = SAGEConv(in_channels=self.in_channels[2],
                               out_channels=self.out_channels[2],
                               bias=self.bias)

        self.avgpool = nn.AdaptiveAvgPool2d((sqrt(self.num_verts_out),
                                             sqrt(self.num_verts_out)))

        self.norm = norm

    def forward(self, img):
        """
        Args:
            - img: (?, num_channels, height, width), input image(s)

        Returns:
            - norm_feature: (?, 1, sqrt(num_verts_out), sqrt(num_verts_out)),
              attention inferred from input image(s) in an independent direction
              in space
        """
        img = self.downsample_img(img)

        num_img, _, _, _ = img.shape
        device = img.device

        verts = Variable(self.verts.to(device), requires_grad=False)
        edges = Variable(self.edges.to(device), requires_grad=False)
        mask = Variable(self.mask.to(device), requires_grad=True).unsqueeze(0)

        # Design of AIM's Attention Module. Refer to Figure 4 in the paper
        img = img.reshape(num_img, self.in_channels[0], -1).permute(0, 2, 1)
        verts = torch.stack([verts]*num_img, 0)

        edges = torch.stack([edges]*num_img, 0).reshape(-1, 2).permute(1, 0)

        feature = img[torch.arange(num_img)[:, None], verts]
        feature = feature.reshape(-1, self.in_channels[0])

        feature = self.conv_1(x=feature, edge_index=edges)
        skip_1 = self.skip_1(feature, edge_index=edges)

        feature = self.conv_2(x=feature, edge_index=edges)
        feature = feature + skip_1
        skip_2 = self.skip_2(feature, edge_index=edges)

        feature = self.conv_3(x=feature, edge_index=edges)
        feature = feature + skip_2

        # Refer to Section 4.4 in the paper
        if self.use_mask:
            feature = feature.reshape(num_img, -1, 1).squeeze(2)
            feature_min = torch.min(feature, 1)[0]
            feature_min = torch.stack([feature_min]*sqrt(self.num_verts_in), 1)
            feature_min = torch.stack([feature_min]*sqrt(self.num_verts_in), 2)
            feature = feature.reshape(num_img,
                                      sqrt(self.num_verts_in),
                                      sqrt(self.num_verts_in))
            feature = feature * mask + feature_min * (1-mask)

        feature = self.avgpool(feature.unsqueeze(1))
        feature = feature.squeeze(1).reshape(num_img, -1)

        # Normalize normalize attention map(s) between zero and one
        norm_feature = self.normalization(feature)

        return norm_feature

    def downsample_img(self, img):
        """
        Utility function to downsample image(s)
        Args:
            - img: (?, num_channels, height, width), input image(s)

        Returns:
            - d_img: (?,
                      num_channels,
                      self.sample_params['size'],
                      self.sample_params['size']), downsampled image(s)
        """
        d_img = F.interpolate(input=img,
                              size=(self.sample_params['size'],
                                    self.sample_params['size']),
                              mode=self.sample_params['mode'],
                              align_corners=self.sample_params['align_corners'])
        return d_img

    def normalization(self, feature):
        """
        Utility function to normalize attention map(s) between zero and one
        Args:
            - feature: (?, 1, num_verts), the average pooled output of the final
              GCN layer

        Returns:
            - norm_feature: (?, num_verts), attention map(s) normalized
              between zero and one
        """
        if self.norm == 'Min-Max':
            norm_feature = (feature - feature.min(1, True)[0]) \
            /(feature.max(1, True)[0] - feature.min(1, True)[0])

        return norm_feature
