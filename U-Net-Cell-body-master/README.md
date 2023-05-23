# UNet_Cell

 * **Cellpose_Manual:**
    1. _Cellpose.zip_: Images segmented automatically by [Cellpose](https://github.com/mouseland/cellpose).
    2.  _Manual.zip_: Manual anotation images.
    3. _main_images.zip_: Real P-DHM and DHM 5X images.
    4. `Image_Augmentation.ipynb`: Pipeline for expanding images for training.
    
***
* **Prediction_Samples:**  Output results of `DHM_Cell.hdf5` model.

***

* **Test images:**  DHM and P-DHM images for testing `DHM_Cell.hdf5` model.

***


* **Testing_Pipelines:**
    1. Using [EMPatches](https://pypi.org/project/empatches/) algorithm to create automatically patches with the size of `[128,128]`. Overlaps between patches can be set, larger value gives more patches. Indices of patches saves and will be used to combine all patches after prediction with differnt overlapping modes, `mode = ['avg','max','min','overwrite']`, which gives a single image that has the same size as the input image.

***


* **Training dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/data/train/), Contains two files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUnet_Cellbody.ipynb` for training U-Net.
  
***


 * **Validation dataset** :(/Volumes/DATA/DHM 1082/yaza3022/Different ages phase signal/U-Net pipelines/unet_cell_master/data/val) Contains two zip files (_img_ and _lbl_) of phase images  with the size of `[128,128]`, used in `trainUnet_Cellbody.ipynb` as validation data for U-Net.


