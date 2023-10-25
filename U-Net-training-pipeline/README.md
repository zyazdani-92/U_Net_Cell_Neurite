# U-Net_cell_body

 * **Cellpose and manual segmentation and main DHM images :**
    1. _/Volumes/DATA/DHM\ 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/img/Cellpose_: Images segmented automatically by [Cellpose](https://github.com/mouseland/cellpose).
    2.  _/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/img/Cell_lbl_: Manual anotation images.
    3. _/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/img/Update_training_set_:  DHM 5X images.

    
***

* **Training dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/data/train/), Contains two files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUNet.ipynb` for training U-Net.
  
***


 * **Validation dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/data/val) Contains two zip files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUNet.ipynb` as validation data for U-Net.



# U-Net_Neurit

 * **Mask_DHM:**
    1. _/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet-neurite-master/data/My DHM images for U-Net-Neurite /DHM_img_: 5X DHM images with the size of `[800,800]`.
    2.  _/Volumes/DATA/DHM\ 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet-neurite-master/data/My DHM images for U-Net-Neurite /Neurite_lbl_: Manual anotation images with [Labkit (UI)](https://github.com/juglab/labkit-ui).
    
     
***




* **Training dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet-neurite-master/data/train/), Contains two files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUNet.ipynb` for training U-Net.
  
***



 * **Validation dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet-neurite-master/data/val) Contains two zip files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUNet.ipynb` as validation data for U-Net.
