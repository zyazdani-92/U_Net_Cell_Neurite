'''Mask processing pipeline
Writer: Zahra Yazdani
'''

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
import numpy as np
from scipy import ndimage as ndi

from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)


neurit_opt_thr =  0.14141414141414144 # Max balanced accuracy
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


