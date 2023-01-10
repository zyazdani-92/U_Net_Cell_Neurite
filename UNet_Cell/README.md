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
    1. `Test_UNet_Cell_patchify.ipynb`: Using [patchify](https://pypi.org/project/patchify/) module to create patches with the size of `[128,128]` from phase image as input to the `DHM_Cell.hdf5`. The model outputs (prediction rsults) are with the size of `[128,128]`. By adopting `unpatchify` function all patches merge together to reach the actual size of input image. It is suggested to use the below pipeline for patch processing.
    2. `Test_UNet_empatches.ipynb`: Using [EMPatches](https://pypi.org/project/empatches/) algorithm to create automatically patches with the size of `[128,128]`. Overlaps between patches can be set, larger value gives more patches. Indices of patches saves and will be used to combine all patches after prediction with differnt overlapping modes, `mode = ['avg','max','min','overwrite']`, which gives a single image that has the same size as the input image.

***


* **Train_img_128** : Contains two zip files (_img_aug.zip_ and _mask_aug.zip_) of phase images  with the size of `[128,128]`, used in `main.py` for training U-Net.
  
***


 * **Val_img_128** : Contains two zip files (_img_aug.zip_ and _mask_aug.zip_) of phase images  with the size of `[128,128]`, used in `main.py` as validation data for U-Net.


