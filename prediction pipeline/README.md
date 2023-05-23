
* **




* **Prediction_Pipelines:**
    1. ```Img2mask_pipeline_update23May.ipynb```: Using [EMPatches](https://pypi.org/project/empatches/) algorithm to create automatically patches with the size of `[128,128]`. Overlaps between patches can be set, larger value gives more patches. Indices of patches saves and will be used to combine all patches after prediction with differnt overlapping modes, `mode = ['avg','max','min','overwrite']`, which gives a single image that has the same size as the input image.

***

