"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains utility functions used by the Attention Module.
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
    This function creates a graph representation of images. Given the number of
    vertices/pixels in an attention map, this function returns the flattened/1D
    locations of vertices and the topology of the graph. The vertices represent
    the pixel locations. The topology of this graph is represented by its edges.
    As mentioned in section 4.1.1 in the paper, unique edges exist between
    horizontal and vertical nodes/vertex neighbors. Note that the graphs for the
    Attention Module and the Spatial Warper are different.

    Args:
        - num_verts: int, number of verts in the attention map

    Returns:
        - verts_1D: (num_verts), flattened/1D locations of the graph's vertices

        - edges: (num_edges, 2), edges between the graph's vertices
    """
    print('The Attention Module is creating a graph representation of images..')
    # verts_1D represents the flattened pixel locations
    # For example, for a pixel at location [0, 0] in the 2D space, its
    # equivalent flattened vertex location is 0. For pixel at location [0, 1] in
    # the 2D space, the flattened vertex location is 1.
    verts_1D = np.arange(num_verts)

    # verts_2D and edges create the topology of the graph
    verts_2D = []
    edges = []

    num_edges = 0

    # Location of vertices in 2D space
    for y in range(sqrt(num_verts)):
        for x in range(sqrt(num_verts)):
            verts_2D.append([y, x])
    verts_2D = np.asarray(verts_2D)

    assert verts_2D.shape[0] == num_verts

    # Find the vertex neighbors for all vertices
    for vi in verts_1D:
        # Find the vertex neighbors for a vertex vi at 2D location [y, x]
        y, x = verts_2D[vi]
        for n in range(-1, 2):
            for m in range(-1, 2):
                # Only select vi's horizontal and vertical vertex neighbors
                if m + n == 1 or m + n == -1:
                    # Check that neighbor are not out of the image bounds
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
    verts_1D = torch.from_numpy(verts_1D).long()

    # Remove duplicate edges
    data = Data(pos=verts_1D,
                num_nodes=verts_1D.shape[0],
                edge_index=edges.permute(1, 0))
    graph = to_networkx(data, to_undirected=True)
    edges = torch.LongTensor(list(graph.edges)).contiguous()

    print('This graph contains {0} vertices'.format(verts_1D.shape[0]))
    print('This graph contains {0} edges'.format(edges.shape[0]))

    return verts_1D, edges
