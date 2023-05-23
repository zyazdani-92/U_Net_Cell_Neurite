
# UNet_Neurit

 * **Mask_PDHM_DHM:**
    1. _DHM_: 5X DHM images with the size of `[799,799]`.
    2.  _Mask_: Manual anotation images with [Labkit (UI)](https://github.com/juglab/labkit-ui).
    3. _PDHM_: 5X PDHM images with the size of `[799,799]`.
    4. `Image_expansion_pipeline.ipynb`: Pipeline for expanding images for training.
    
     
***
* **Prediction_Samples:**  Output results of `DHM_Neurite.hdf5` model.

***

* **Test images:**  DHM and P-DHM images for testing `DHM_Neurite.hdf5` model.

***


* **Testing_Pipelines:**
    1. `Test U-DHM on DHM image.ipynb`
    2. `Test U-DHM on PDHM image.ipynb` 
    3. `Test U-PDHM on DHM image.ipynb`
    4. `Test U-PDHM on PDHM image.ipynb`
    
    
    *Using [patchify](https://pypi.org/project/patchify/) module to create patches with the size of `[128,128]` from phase image as input to the `DHM_Neurite.hdf5`. The model outputs (prediction rsults) are with the size of `[128,128]`. By adopting `unpatchify` function all patches merge together to reach the actual size of input image. It is suggested to use the below pipeline for patch processing.
    ~~Add pipeline here to use EMPatches algorithm to create automatically patches with the size of [128,128]~~ 

     * _If PDHM images is used to train U-Net, model called U-PDHM._
     
     
     * _If DHM images is used to train U-Net, model called U-DHM._
***


* **Train_img_128** : Contains three zip files (_DHM_train.zip_, _PDHM_train.zip_ and _Mask_train.zip_) of phase images  with the size of `[128,128]`, used in `main.py` for training U-Net.
  
***


 * **Val_img_128** : Contains two zip files (_DHM_train.zip_, _PDHM_train.zip_ and _Mask_train.zip_) of phase images  with the size of `[128,128]`, used in `main.py` as validation data for U-Net.
