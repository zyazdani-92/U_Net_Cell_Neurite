# UNet_Cell

 * **Cellpose and manual segmentation and main DHM images :**
    1. _/Volumes/DATA/DHM\ 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/img/Cellpose_: Images segmented automatically by [Cellpose](https://github.com/mouseland/cellpose).
    2.  _/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/img/Cell_lbl_: Manual anotation images.
    3. _/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/img/Update_training_set_:  DHM 5X images.

    
***

* **Training dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/data/train/), Contains two files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUnet_Cellbody.ipynb` for training U-Net.
  
***


 * **Validation dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/data/val) Contains two zip files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUnet_Cellbody.ipynb` as validation data for U-Net.


