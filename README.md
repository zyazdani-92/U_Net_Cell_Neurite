# U_Net_Cell_Neurite
 ## DHM images semantic segmentation using U-Net
U-Net Cell body master and U-Net Neurite master are deep learning models for identifying neurite and cell body in Digital Holographic Microscopy(DHM) phase images. The two U-Net models are a convolutional 2D network architecture for fast and precise segmentation of images. 

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

U-Net Cell body master and U-Net Neurite master architecture are based an implementation of the [U-Net model for Keras](https://github.com/pietz/unet-keras).
![alt text](U-Net.svg "Logo Title Text 1")

## Overview

### U-Net Cell body 

A deep learning algorithm for cell body segmentation from DHM phase images. 


### U-Net Neurite 

A deep learning algorithm for neuronal processes segmentation from DHM phase images. 

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
* For Mac M1/M2 users please follow this guideline: [How to Install TensorFlow GPU for Mac M1/M2 with Conda](https://www.youtube.com/watch?v=w2qlou7n7MA).

4. Navigate to _U-Net-training-pipeline_ folder

5. Download the training and validation sets from _SINUS_ and place them in the "data" folder within the "U-Net-Cell-body-master" folder.

6. To train the `DHM_cell.hdf5` and save the model's weights, run the ```Unet_cell_body_model.ipynb ``` Jupyter notebook.

7. For neurite segmentation, repeat steps 4-6 once again for the "UNet_Neurite" folder.

8. Use the `DHM_Cell.hdf5`  and 'DHM_Neurite.hdf5' models with the "Img2mask_pipeline_update23May.ipynb" notebook to perform cell body and neurite segmentation on the image in the "Test_img" folder.

9. Combine trained models results, the output is statisfactory.

<img src="masks_roc.svg" width="1200"/> 
