
#Import libraries
import numpy as np
from skimage.io import imread
from skimage.exposure import adjust_gamma
from aicssegmentation.core.pre_processing_utils import edge_preserving_smoothing_3d
from aicssegmentation.core.vessel import filament_2d_wrapper
from skimage.morphology import binary_closing, remove_small_objects, binary_opening, ball, binary_dilation
from aicssegmentation.core.utils import hole_filling
from tifffile import imwrite
from skimage.exposure import rescale_intensity
from skimage.feature import canny
from skimage.measure import label
from skimage.morphology import skeletonize

#Define parameters
minmax_norm_param=[9500]
seg_param=[[1.2, 0.005]]
bincl_param=np.ones((1,4,4))
hole_min_param=1
hole_max_param=40
smallobj_param=200

#Read in image
img = imread("input_images/control_male4_transformed_cropped.tif")

#Apply normalisation
img_cp = np.copy(img)
img_cp[img_cp > minmax_norm_param[0]] = True
img_cp[img_cp != True] = False
img[img > minmax_norm_param[0]] = img.min()

img_adj = adjust_gamma(img, 0.4)
img_adj[img_cp == 1] = img_adj.max()

imwrite("processed_images_transformed/control_male4/control_male4_adjusted.tif", img_adj)

#Apply smoothing
img_smooth = edge_preserving_smoothing_3d(img_adj, numberOfIterations=10)
imwrite("processed_images_transformed/control_male4/img_smooth.tif", img_smooth)

#Segmentation
img_seg = filament_2d_wrapper(img_smooth, seg_param)

#Post-segmentation editing
img_bincl = binary_closing(img_seg, bincl_param)
img_noholes = hole_filling(img_bincl, hole_min=hole_min_param, hole_max=hole_max_param, fill_2d=True)
img_fin = remove_small_objects(img_noholes>0, min_size=smallobj_param)

#Create a mask "bar" to remove the edge effect resulting from transformation
#Credit to idea: https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image

img = imread("processed_images_transformed/control_male4/control_male4_adjusted.tif")

img_canny = img.copy()
for i in range(img.shape[0]):
    cannied = canny(img[i], sigma=1.6)
    img_canny[i] = cannied

dilated = binary_dilation(img_canny, np.ones((1,2,2)))   

def get_largest_cc(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

largest_cc = dilated.copy()
for i in range(dilated.shape[0]):
    if True in dilated[i]:
        largest = get_largest_cc(dilated[i])
        largest_cc[i] = largest
        
bar_mask = binary_dilation(largest_cc, np.ones((1,8,12)))

empty = np.empty(img_fin.shape, dtype="bool")
segmentation_noedge = np.where(bar_mask == 1, empty, img_fin)
segmentation_noedge = remove_small_objects(segmentation_noedge>0, min_size=smallobj_param)
imwrite("processed_images_transformed/control_male4/control_male4_segmentation_frompipeline.tif", segmentation_noedge)


#CREATE MASK

#Load image
img = imread("input_images/control_male4_transformed_cropped.tif")

#Rescale (increase brightness)
img_rescaled = rescale_intensity(img, in_range=(0, 357))

#Binarise at appropriate threshold
mask = img_rescaled > 55000

#Morphological operations to smooth mask
footprint = ball(3)
mask_opened = binary_opening(mask, footprint)
mask_closed = binary_closing(mask_opened, footprint)
footprint2 = np.ones((2,10,10))
mask_closed2 = binary_closing(mask_closed, footprint2)

#Fill in any small holes
mask_noholes = hole_filling(mask_closed2, hole_min=1, hole_max=1700, fill_2d=True)

#Disregard border effect
mask_noedge = np.where(bar_mask == 1, empty, mask_noholes)

#Save mask
imwrite("processed_images_transformed/control_male4/control_male4_mask.tif", mask_noedge)


#CREATE INITIAL SKELETON

#Restrict segmentation to mask
empty = np.empty(mask_noholes.shape, dtype="bool")
segm_masked = np.where(mask_noholes==1, img_fin, empty)

#Skeletonise masked segmentation
skel = skeletonize(segm_masked)

#Cast to binary to save space and save
skel = skel.astype("bool")
imwrite("processed_images_transformed/control_male4/control_male4_skel_raw.tif", skel)
segm_masked = segm_masked.astype("bool")
imwrite("processed_images_transformed/control_male4/control_male4_segmentation.tif", segm_masked)


#CREATE NEW SKELETON BASED ON UPDATED SEGMENTATION

#Load
segm_fixed = imread("processed_images_transformed/control_male4/control_male4_segmentation.tif")

skel = skeletonize(segm_fixed)
skel = skel.astype("bool")

imwrite("processed_images_transformed/control_male4/control_male4_skeleton.tif", skel)
