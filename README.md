# U_Net_Cell_Neurite
 ## DHM images semantic segmentation using U-Net
U-Net Cell and U-Net Neurite are deep learning models for identifying neurite structure and cell body in Digital Holographic Microscopy(DHM) phase images. The U-Net is a convolutional network architecture for fast and precise segmentation of images. 

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

U-Net-Cell and U-Net-Neurite architecture are based on [DeepNeurite](https://github.com/khCygnal/DeepNeurite) U-Net model strcucture.

![alt text](U-net(Neurite+Cell).png "Logo Title Text 1")

## Overview

### UNet_Cell
In this folder all the
### UNet_Neurite
In this repository all notebooks and image pipeline that are used for U-Net are included.
These pipelines are
  
1. `Image_decomposition_composition.ipynb` is a jupyter notebook that splits input image into small patches with `patchify` function. This notebook will be used inside `U-DHM` or `U-PDHM` pipeline to splits input image into suitable size for models. Because, these models were trained on images with the size of `[128,128]`. Therefore, the prediction result of these models would be more accurate if the input image has the same size as the training image. Hence this pipeline decomposes the input image, which has the size of `[750,750]`, into many patches with the size of `[128,128]`. In this case, U-PDHM or U-DHM model predicts neuronal processes of smaller images, and then we merge all images with `unpatchify` function, which gives a single image that has the same size as the input image.

## Dependencies
Python 3.7, Tensorflow-gpu 1.15.0, Keras 2.3.1

* For Mac M1/M2 user please follow this guidline: [How to Install TensorFlow GPU for Mac M1/M2 with Conda](https://www.youtube.com/watch?v=5DgWvU0p2bk)
