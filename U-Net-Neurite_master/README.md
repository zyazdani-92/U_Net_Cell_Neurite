
# UNet_Neurit

 * **Mask_DHM:**
    1. _/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet-neurite-master/data/My DHM images for U-Net-Neurite /DHM_img_: 5X DHM images with the size of `[800,800]`.
    2.  _/Volumes/DATA/DHM\ 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet-neurite-master/data/My DHM images for U-Net-Neurite /Neurite_lbl_: Manual anotation images with [Labkit (UI)](https://github.com/juglab/labkit-ui).
    
     
***
* **Prediction samples:**  Output results of `DHM_Neurite.hdf5` model.

***

* **Test images:**  DHM images for testing `DHM_Neurite.hdf5` model.

***


* **Testing Pipeline:**
     1. Using [EMPatches](https://pypi.org/project/empatches/) algorithm to create automatically patches with the size of `[128,128]`. Overlaps between patches can be set, larger value gives more patches. Indices of patches saves and will be used to combine all patches after prediction with differnt overlapping modes, `mode = ['avg','max','min','overwrite']`, which gives a single image that has the same size as the input image.
***


* **Training dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet-neurite-master/data/train/), Contains two files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUnet_Cellbody.ipynb` for training U-Net.
  
***



 * **Validation dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet-neurite-master/data/val) Contains two zip files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUnet_Cellbody.ipynb` as validation data for U-Net.
