"""
Code for the paper "AIM: An Auto-Augmenter for Images and Meshes," published in
the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains an utility function to avoid hardcoding paths and training/
testing related configurations.
"""
import os
import os.path as osp
import yaml
from easydict import EasyDict

def get_config(path_config='config/config.yaml'):
    """
    This utility function provides device, data, model, and optimizer
    configurations to the train and test files.

    Args:
        - path_config: str, config.yaml path relative to the train/test.py files

    Returns:
        - cfg: dict, device, data, model, and optimizer configurations
    """
    with open(path_config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    return cfg
