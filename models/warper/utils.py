"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes" published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains utility functions used by the Spatial Warper.
"""
import math
import numpy as np
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def sqrt(x):
    """
    Utility function to convert the output of the math.sqrt() function to an
    integer. This function only works when the input number is a perfect square.

    Args:
        - x: int, input number

    Returns:
        - sqrt_x: int, square root of the input number
    """
    sqrt_x = int(math.sqrt(x))
    assert (math.sqrt(x) - sqrt_x) == 0

    return sqrt_x

def image2graph(num_verts):
    """
    This function creates a graph representation for square images.
    Given the number of pixels in an attention map, this function returns
    the locations of vertices and the topology of the graph. The vertices
    represent the pixel locations. The topology of this graph is represented
    by its edges. As mentioned in section 4.1.1 in the paper, unique edges
    exist between horizontal and vertical nodes/vertex neighbors. Note that
    the graphs for the Attention Module and the Spatial Warper are different.

    Args:
        - num_verts: int, number of verts output by the attention map.
    """
    print('The Spatial Warper is creating a graph representation of images..')

    # verts_coords are the pixel locations in 2D space, such that the leftmost
    # pixel values is [-1, -1], the rightmost pixel values is [1, 1], and the
    # location of the pixel at the center of the image is [0, 0]
    verts_coords = []

    # verts_2D and edges create the topology of the graph
    verts_2D = []
    edges = []

    num_edges = 0

    # Location of vertices in 2D space
    for y in range(sqrt(num_verts)):
        for x in range(sqrt(num_verts)):
            verts_coords.append([x/(sqrt(num_verts)-1.0), y/(sqrt(num_verts)-1.0)])
            verts_2D.insert(num_verts, [y, x])
    verts_2D = np.asarray(verts_2D)
    verts_coords = np.asarray(verts_coords)

    # Find the vertex neighbors for all vertices
    for vi in range(num_verts):
        # Find the vertex neighbors for a vertex vi at 2D location [y, x]
        y, x = verts_2D[vi]
        for n in range(-1, 2):
            for m in range(-1, 2):
                # Only select vi's horizontal and vertical vertex neighbors
                if m + n == 1 or m + n == -1:
                    # Check neighbors are not out of the image bounds
                    if x+m >= 0 and y+n >= 0:
                        if x+m < sqrt(num_verts) and y+n < sqrt(num_verts):
                            # Find a vertex neighbor for vertex vi
                            condition = (verts_2D == [y+n, x+m]).all(axis=1)
                            vj = np.argwhere(condition)[0][0]
                            # No self-loops
                            if vj != vi:
                                # Create an edge between vi and vj
                                edges.insert(num_edges, [vi, vj])
                                num_edges += 1
    edges = np.asarray(edges)
    edges = torch.from_numpy(edges).long()
    verts_coords = torch.from_numpy(verts_coords).float()

    return verts_coords, edges
