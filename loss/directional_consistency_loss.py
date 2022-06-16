"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains the directional consistency loss. Refer to section 4.3 in
the paper for more details.
"""
import torch.nn as nn
import torch.nn.functional as F

class DirectionalConsistency(nn.Module):
    """
    This loss enforces the attention module’s embeddings (edge sensitivities) to
    be consistent along each independent direction in space. Note that edge
    sensitivity is inferred by averaging features of the vertices at their
    endpoints, as mentioned in section 4.4 of the paper. Thus, we consider the
    attention at the vertices as edge sensitivities to reduce computation.
    """
    def __init__(self, dim: int = 1, eps: float = 1e-8, weight=0.1):
        """
        Args:
            - dim: int, Dimension to compute cosine similarity. Default: 1
            - eps: float, Small value to avoid division by zero.
            - weight: float, weight for the loss function
        """
        super(DirectionalConsistency, self).__init__()
        self.weight = weight
        self.dim = dim
        self.eps = eps

    def forward(self, Sx, Sy):
        """
        Args:
            - Sx: (? num_verts), edge sensitivities along the X-axis
            - Sy: (? num_verts), edge sensitivities along the Y-axis.
        Returns:
            - directional_consistency_loss: (?)
        """
        # Refer equation 6 in the paper
        cosine_similarity = F.cosine_similarity(x1=Sx,
                                                x2=Sy,
                                                dim=self.dim,
                                                eps=self.eps)
        directional_consistency_loss = 1 - self.weight*cosine_similarity.mean(0)
        return directional_consistency_loss
