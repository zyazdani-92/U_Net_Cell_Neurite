#!/usr/bin/env python
# coding: utf-8

'''Mask Processing Pipeline
Writer: Zahra Yazdani
'''

from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from scipy import ndimage as ndi
from skimage import (color, feature, filters, measure, morphology, segmentation, util)
from data_128 import *
from keras.models import load_model
from empatches import EMPatches
import numpy as np
from matplotlib import pyplot as plt
import os

neurit_opt_thr = 0.14141414141414144 # Max balanced accuracy
cell_opt_thr =  0.6565656565656566 #Max f-score

def Cell_Mask(cell_merged_img):
    # Binarized cell prediction
    cell_bin = (cell_merged_img>cell_opt_thr).astype(int)
    cell_bin = cell_bin.reshape(cell_bin.shape[:-1])
    #opening dark region in cell_bin
    cc = cell_bin.copy()
    footprint = disk(2)
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

#computes the distance from non-zero (i.e. non-background) 
                                                  #points to the nearest zero (i.e. background) point.
def wtr_shed(Cell):
    distance = ndi.distance_transform_edt(Cell) 
    local_max_coords = feature.peak_local_max(distance, min_distance=4) # change min_distance to separate more cells
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segmented_cells = segmentation.watershed(-distance, markers, mask=Cell)
    return segmented_cells


def plot_comparison(original, filtered, title1,title2):
    font2 = {'family':'serif','color':'black','size':20}
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 20), sharex=True,
                                   sharey=True)
    
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title(title1,fontdict = font2)
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title2,fontdict = font2)
    ax2.axis('off')



def predict_and_segment(test_path, cell_model_path, neurite_model_path, patchsize=128, overlap=0.1):
    # Load the models
    cell_model = load_model(cell_model_path)
    neurite_model = load_model(neurite_model_path)
    
    # Read the image
    file_list = [f for f in os.listdir(test_path) if f[-3:]=="tif"][0]
    img = img_read(test_path, file_list)
    
    # Extract patches from the image
    emp = EMPatches()
    img_patches, indices = emp.extract_patches(img, patchsize=patchsize, overlap=overlap)
    num_test_images = len(img_patches)
    
    # Generate the test data for the cell model
    test_gene = testGenerators(img_patches)
    
    # Predict cell probabilities
    c_results = cell_model.predict_generator(test_gene, num_test_images, verbose=1)
    
    # Merge the cell patches
    cell_merged_img = emp.merge_patches(c_results, indices, mode='avg')
    
    # Generate the test data for the neurite model
    test_gene = testGenerators(img_patches)
    
    # Predict neurite probabilities
    n_results = neurite_model.predict_generator(test_gene, num_test_images, verbose=1)
    
    # Merge the neurite patches
    neurite_merged_img = emp.merge_patches(n_results, indices, mode='min')
    
    # Segment the neurites
    neurite_mask = Neurite_Mask(neurite_merged_img)>0
    
    Cell = Cell_Mask(cell_merged_img)
    # Segment the cells using watershed segmentation
    cell_lbl = wtr_shed(Cell)
    cell_mask = cell_lbl>0
    
    color_labels = color.label2rgb(cell_lbl, img, alpha=0.4, bg_label=0)

    
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].matshow(img, cmap = 'Greys_r')
    axs[0].axis('off')
    axs[0].set_title('DHM image')
#     axs[1].imshow(color_labels, cmap = 'Greys')
#     axs[1].axis('off')
#     axs[1].set_title(' Number of segmented cells = %i'%((cell_lbl.max())))
    axs[1].matshow(cell_mask, cmap = 'Greys_r')
    axs[1].axis('off')
    axs[1].set_title('Cell body')
    axs[2].matshow(neurite_mask, cmap = 'Greys_r')
    axs[2].axis('off')
    axs[2].set_title('Neurites')
    axs[3].matshow(cell_mask+neurite_mask, cmap = 'Greys_r')
    axs[3].axis('off')
    axs[3].set_title('Neuronal network')
    plt.savefig('prediction results.png')
    plt.show()
    
    return cell_mask, neurite_mask

