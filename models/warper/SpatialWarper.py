"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains the Spatial Warper described in section 4.1 of the paper .
"""
import os
import os.path as osp
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch.autograd import Variable
from torch.nn import Upsample
from torch.nn.functional import grid_sample
from .utils import image2graph, sqrt

class SpatialWarper(nn.Module):
    """
    The Spatial Warper minimizes deformation in the overall shape of the
    graph while deforming each edge locally. For images, the Spatial Warper
    increases the spatial converage of the pixels in the original image by
    shrinking the edges with higher edge sensitivity.
    """
    def __init__(self, cfg_SpatialWarper):
        super(SpatialWarper, self).__init__()
        """
        Args:
        cfg_SpatialWarper: dict,  contains the following attributes:
            - delta: float, deformation factor
            - inv_eps: int, a very large number
            - num_verts: (num_verts,) number of vertices in the graph
            - task_img_size: int, size of the image to the task network
        """
        cwd = os.path.realpath(__file__)
        cwd = cwd[:cwd.rindex(os.sep)]
        if not osp.exists(osp.join(cwd, 'verts.pt')):
            verts, edges = image2graph(num_verts=1024)
            torch.save(verts, osp.join(cwd, 'verts.pt'))
            torch.save(edges, osp.join(cwd, 'edges.pt'))

        self.verts = torch.load(osp.join(cwd, 'verts.pt'))
        self.edges = torch.load(osp.join(cwd, 'edges.pt'))

        self.delta = cfg_SpatialWarper.delta
        self.num_verts = cfg_SpatialWarper.num_verts
        self.task_img_size = cfg_SpatialWarper.task_img_size

    def AB(self, verts, edges, S, dim=0):
        """
        Generate the matrices A and B mentioned in equation 5 of the paper.
        Args:
            - verts: (num_verts, 2), vertices of the graph
            - edges: (?, 2, ), undirected edges of the graph
            - S: (?, num_verts, num_verts), edge sensitivity along an
              independent direction in space

        Return:
            - A: (?, num_verts, num_verts), matrix A mentioned in equation 5
            - B: (?, num_verts, 1), vector B mentioned in equation 5
        """
        num_img = S.shape[0]
        num_verts = verts.shape[0]

        verts_adjacency = to_dense_adj(edges[0, :, :]).squeeze(0)
        verts_stacked = torch.stack([verts[:, dim]]*num_verts, 0)
        denom = (verts_stacked.T - verts_stacked + 1e2)**2

        a = verts_adjacency * (-2/denom)
        a = a + torch.diag(torch.sum(a, 1) * -1)

        A = torch.stack([a]*num_img, 0)
        verts = torch.stack([verts]*num_img, 0)

        verts_stacked = torch.stack([verts[:, :, dim]]*num_verts, 1)
        denom = (verts_stacked.permute(0, 2, 1) - verts_stacked + 1e2)**2
        B = (2 * S * (verts_stacked.permute(0, 2, 1) - verts_stacked))/denom
        B = torch.sum(B, 2)

        return A, B.unsqueeze(2)

    def forward(self, img, verts_saliency_x, verts_saliency_y):
        """
        Args:
            - img: (?, num_channels, height, width), input image(s)
            - verts_saliency_x: (?, num_verts),
              output of the AttentionModule along the X-axis
            - verts_saliency_y: (?, num_verts),
              output of the AttentionModule along the Y-axis

        Returns:
            - img_out: (?, num_channels, task_img_size, task_img_size), warped
              images
        """
        device = img.device
        num_img = img.shape[0]
        num_verts = verts_saliency_x.shape[1]

        verts = self.verts.to(device)
        edges = Variable(self.edges.to(device), requires_grad=False)
        edges = torch.stack([edges]*num_img, 0)

        #X
        edges_saliency_x = verts_saliency_x[torch.arange(num_img)[:, None, None],
                                            edges]
        edges_saliency_x = torch.sum(edges_saliency_x, 2)/2
        deform_saliency_x = self.delta * edges_saliency_x + (1 - edges_saliency_x)

        #Y
        edges_saliency_y = verts_saliency_y[torch.arange(num_img)[:, None, None],
                                            edges]
        edges_saliency_y = torch.sum(edges_saliency_y, 2)/2
        # Refer equation 2 in the paper
        deform_saliency_y = self.delta * edges_saliency_y + (1 - edges_saliency_y)

        edges = edges.permute(0, 2, 1)

        Sx = torch.zeros((num_img, num_verts, num_verts), requires_grad=True).to(device)
        Sy = torch.zeros((num_img, num_verts, num_verts), requires_grad=True).to(device)

        with torch.no_grad():
            for idx in range(num_img):
                sx = to_dense_adj(edges[idx, :, :],
                                  edge_attr=deform_saliency_x[idx, :]).squeeze(0)
                Sx[idx] = sx

                sy = to_dense_adj(edges[idx, :, :],
                                  edge_attr=deform_saliency_y[idx, :]).squeeze(0)
                Sy[idx] = sy

        # Generate matrices A and B mentioned in equation 5
        Ax, Bx = self.AB(verts, edges, Sx, 0)
        Ay, By = self.AB(verts, edges, Sy, 1)

        # Solve for X
        verts_x = torch.linalg.solve(Ax, Bx)
        verts_y = torch.linalg.solve(Ay, By)

        # Normalize to unit sphere as mentioned in section 4.1.1 of the paper
        verts = torch.cat([verts_y, verts_x], 2)
        verts = verts - verts.mean(1, True)
        scale = (verts.abs().max(1, True)[0]).max(2, True)[0]
        verts = verts / scale

        verts_x = verts[:, :, 1]
        verts_y = verts[:, :, 0]
        verts_x = verts_x.reshape(-1, sqrt(self.num_verts), sqrt(self.num_verts))
        verts_y = verts_y.reshape(-1, sqrt(self.num_verts), sqrt(self.num_verts))
        grid = torch.stack([verts_x, verts_y], 1)
        grid = torch.clamp(grid, min=-1, max=1)
        grid = Upsample(size=(self.task_img_size, self.task_img_size),
                        mode='bilinear',
                        align_corners=True)(grid)
        grid = grid.permute(0, 2, 3, 1)
        img_out = grid_sample(img, (grid))
        return img_out
