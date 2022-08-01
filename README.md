# AIM
Official PyTorch implementation for [AIM: an Auto-Augmenter for Images and Meshes](https://openaccess.thecvf.com/content/CVPR2022/papers/Singh_AIM_An_Auto-Augmenter_for_Images_and_Meshes_CVPR_2022_paper.pdf). The code has been implemented and tested on the Ubuntu operating system only.

![Alt text](docs/Overview.png?raw=true)

## Install CUDA Toolkit and cuDNN
Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and the [cuDNN library](https://developer.nvidia.com/rdp/cudnn-archive) matching the version of your Ubuntu operating system. Installation of the [Anaconda Python Distribution](https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh) is required as well. We recommend installing CUDA10.2.

## Download the data set
[CUB200 dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/)
The datasets should be place in datasets/CUB_200_2011
Training data should be in datasets/CUB_200_2011/train/<categories>.
Test set should be in datasets/CUB_200_2011/test/<categories>.

## Train
```
python train.py
```

## Test
```
python test.py
```

## Citation
If you found this work helpful for your research, please consider citing us.
```
@inproceedings{singh2022aim,
  title={AIM: An Auto-Augmenter for Images and Meshes},
  author={Singh, Vinit Veerendraveer and Kambhamettu, Chandra},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={722--731},
  year={2022}
}
```
