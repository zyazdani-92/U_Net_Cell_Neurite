#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
@author: Zahra Yazdani
"""

from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from scipy import ndimage as ndi
from skimage import (color, feature, filters, measure, morphology, segmentation, util)
from skimage import exposure
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
import os
from graphinference import *
import json



neurit_opt_thr = 0.59 #Max F_score  'Use your threshold that you got from ROC AUC'
cell_opt_thr =  0.3 #Max F_score     'Use your threshold that you got from ROC AUC'
def Cell_Mask(cell_merged_img):
    # Binarized cell prediction
    cell_bin = (cell_merged_img>cell_opt_thr).astype(int)
    cell_bin = cell_bin.reshape(cell_bin.shape[:-1])
    #opening dark region in cell_bin
    cc = cell_bin.copy()
    footprint = disk(2)   # adjust based on the better cell sepration result!
    cell = opening(cc, footprint)
    cell = cell.astype('uint8')
    return cell


def Neurite_Mask(neurite_merged_img):
    # Binarized neurite prediction
    neurite_bin = (neurite_merged_img>neurit_opt_thr).astype(int)
    neurite_bin = neurite_bin.reshape(neurite_bin.shape[:-1])
    #closing dark region in neurite_bin
    nn=0
    nn = neurite_bin.copy()
    footprint = disk(1)
    neurite = closing(nn , footprint)
    return neurite

def water_shed(Cell):
    distance = ndi.distance_transform_edt(Cell) 
    local_max_coords = feature.peak_local_max(distance, min_distance=4) # change min_distance to separate more cells
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segmented_cells = segmentation.watershed(-distance, markers, mask=Cell)
    return segmented_cells

def img_read(test_path,file_list):
    DHM_img = io.imread(os.path.join(test_path,file_list))
    return DHM_img


def binary_masks(cell_map, neurite_map): 
    # Segment the neurites
    neurite_mask = Neurite_Mask(neurite_map)>0
    Cell = Cell_Mask(cell_map)
    # Segment the cells using watershed segmentation
    cell_mask = water_shed(Cell)
    cell_num = cell_mask.max()
    cell_mask1 = cell_mask>0
    return cell_mask, neurite_mask, cell_num



def convert_pixel2_to_um2(pixel_measurement):
    # calculate the measurement in micrometer
    scale_factor = 5.86 #um
    magnification = 5
    micrometer_measurement2 = pixel_measurement*(( scale_factor / magnification) ** 2)
    return micrometer_measurement2

def convert_pixel_to_um(pixel_measurement):
    # calculate the measurement in micrometer
    scale_factor = 5.86 #um
    magnification = 5 #5X objective
    micrometer_measurement = pixel_measurement*(( scale_factor / magnification))
    return micrometer_measurement


# In[ ]:




