# U-Net_DHM_PDHM

* Cellpose_Manual Contains:
1. **Cellpose.zip**: Images segmented automatically by [Cellpose](https://github.com/mouseland/cellpose).
2.  **Manual.zip**: Manual anotation images.
3.  **main_images.zip**: Real P-DHM and DHM 5X images.

* Test images : _tif_ DHM and P-DHM images for testing `DHM_Cell.hdf5` in * Testing_Pipelines

In this repository all notebooks and image pipeline that are used for U-Net are included.
These pipelines are
  
1. `Image_decomposition_composition.ipynb` is a jupyter notebook that splits input image into small patches with `patchify` function. This notebook will be used inside `U-DHM` or `U-PDHM` pipeline to splits input image into suitable size for models. Because, these models were trained on images with the size of `[128,128]`. Therefore, the prediction result of these models would be more accurate if the input image has the same size as the training image. Hence this pipeline decomposes the input image, which has the size of `[750,750]`, into many patches with the size of `[128,128]`. In this case, U-PDHM or U-DHM model predicts neuronal processes of smaller images, and then we merge all images with `unpatchify` function, which gives a single image that has the same size as the input image.



