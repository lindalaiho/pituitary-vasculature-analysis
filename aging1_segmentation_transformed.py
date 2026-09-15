
#Import libraries
import numpy as np
from skimage.io import imread
from skimage.exposure import adjust_gamma
from aicssegmentation.core.pre_processing_utils import edge_preserving_smoothing_3d
from aicssegmentation.core.vessel import filament_2d_wrapper
from skimage.morphology import binary_closing, remove_small_objects, binary_opening, ball
from aicssegmentation.core.utils import hole_filling
from tifffile import imwrite
from skimage.exposure import rescale_intensity
from skimage.morphology import skeletonize

#Define parameters
minmax_norm_param=[20000]
seg_param=[[1.4, 0.011]]
bincl_param=np.ones((1,3,3))
hole_min_param=1
hole_max_param=50
smallobj_param=200

#Read in image
img = imread("input_images/aging1_transformed_cropped.tif")

#Apply normalisation
img_cp = np.copy(img)
img_cp[img_cp > minmax_norm_param[0]] = True
img_cp[img_cp != True] = False
img[img > minmax_norm_param[0]] = img.min()
img_adj = adjust_gamma(img, 0.6)
img_adj[img_cp == 1] = img_adj.max()

#Apply smoothing
img_smooth = edge_preserving_smoothing_3d(img_adj, numberOfIterations=10)

#Segmentation
img_seg = filament_2d_wrapper(img_smooth, seg_param)

#Post-segmentation editing
img_bincl = binary_closing(img_seg, bincl_param)
img_noholes = hole_filling(img_bincl, hole_min=hole_min_param, hole_max=hole_max_param, fill_2d=True)
img_fin = remove_small_objects(img_noholes>0, min_size=smallobj_param)

#Save final segmented image
imwrite("processed_images_transformed/aging1/aging1_segmentation_frompipeline.tif", img_fin)



#CREATE MASK

#Load image
img = imread("input_images/aging1_transformed_cropped.tif")

#Rescale (increase brightness)
img_rescaled = rescale_intensity(img, in_range=(0, 1858))

#Binarise at appropriate threshold
mask = img_rescaled > 41000

#Morphological operations to smooth mask
footprint = ball(3)
mask_opened = binary_opening(mask, footprint)
mask_closed = binary_closing(mask_opened, footprint)
footprint2 = np.ones((2,10,10))
mask_closed2 = binary_closing(mask_closed, footprint2)

#Fill in any small holes
mask_noholes = hole_filling(mask_closed2, hole_min=1, hole_max=3000, fill_2d=True)

#Save mask
imwrite("processed_images_transformed/aging1/aging1_mask.tif", mask_noholes)


#CREATE INITIAL SKELETON

#Restrict segmentation to mask
empty = np.empty(mask_noholes.shape, dtype="bool")
segm_masked = np.where(mask_noholes==1, img_fin, empty)

#Skeletonise masked segmentation
skel = skeletonize(segm_masked)

#Save
imwrite("processed_images_transformed/aging1/aging1_skel_raw.tif", skel)
imwrite("processed_images_transformed/aging1/aging1_segmentation.tif", segm_masked)


#CREATE NEW SKELETON BASED ON UPDATED SEGMENTATION

#Load
segm_fixed = imread("processed_images_transformed/aging1/aging1_segmentation.tif")

skel = skeletonize(segm_fixed)

imwrite("processed_images_transformed/aging1/aging1_skeleton.tif", skel)
