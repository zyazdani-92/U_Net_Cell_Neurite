# U_Net_Cell_Neurite
 ## DHM images semantic segmentation using U-Net
U-Net Cell and U-Net Neurite are deep learning models for identifying neurite structure and cell body in Digital Holographic Microscopy(DHM) phase images. The U-Net is a convolutional network architecture for fast and precise segmentation of images. 

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

U-Net-Cell and U-Net-Neurite architecture are based on [DeepNeurite](https://github.com/khCygnal/DeepNeurite) U-Net model strcucture.

![alt text](U-net(Neurite+Cell).png "Logo Title Text 1")

## Overview

### UNet_Cell

A deep learning algorithm for cell body segmentation from DHM and P-DHM phase images. 


### UNet_Neurite

A deep learning algorithm for neuronal processes segmentation from DHM and P-DHM phase images. 

## Dependencies
Python 3.7, Tensorflow-gpu 1.15.0, Keras 2.3.1

## Installation

1. Clone this repository
2. Create a virtual environment

```
conda create -n UDHM python=3.7
conda activate UDHM
```
3. Install dependencies
```
conda install --file requirements.txt
```
* For Mac M1/M2 users please follow this guideline: [How to Install TensorFlow GPU for Mac M1/M2 with Conda](https://www.youtube.com/watch?v=5DgWvU0p2bk).

4. Navigate to _UNet_Cell_ folder

6. run main.py 

```
python main.py
```

5. Use `DHM_Cell.hdf5` model and *Testing_Pipelines* to do cell body segmentation on test images.

7. For neurite structure segmentation, repeat steps 4-5 once again about _UNet_Neurite_ folder. 

8. Combine trained models results, the output is statisfactory.

<img src="prediction results.png" width="925"/> 
