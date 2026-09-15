
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
minmax_norm_param=[7000]
seg_param=[[1.3, 0.004]]
bincl_param=np.ones((1,4,4))
hole_min_param=1
hole_max_param=40
smallobj_param=200

#Read in image
img = imread("input_images/stress4_transformed_cropped.tif")

#Split
img1 = img[:136]
img2 = img[136:]

#Apply normalisation
img1_cp = np.copy(img1)
img1_cp[img1_cp > 9000] = True
img1_cp[img1_cp != True] = False
img1[img1 > 9000] = img1.min()

img1_adj = adjust_gamma(img1, 0.8)
img1_adj[img1_cp == 1] = img1_adj.max()

#Apply normalisation
img2_cp = np.copy(img2)
img2_cp[img2_cp > 7000] = True
img2_cp[img2_cp != True] = False
img2[img2 > 7000] = img2.min()

img2_adj = adjust_gamma(img2, 0.6)
img2_adj[img2_cp == 1] = img2_adj.max()

#Combine
img_adj = np.concatenate((img1_adj, img2_adj), axis=0)

#Apply smoothing
img_smooth = edge_preserving_smoothing_3d(img_adj, numberOfIterations=20)
imwrite("processed_images_transformed/stress4/img_smooth.tif", img_smooth)

#Segmentation
img_seg = filament_2d_wrapper(img_smooth, seg_param)

#Post-segmentation editing
img_bincl = binary_closing(img_seg, bincl_param)
img_noholes = hole_filling(img_bincl, hole_min=hole_min_param, hole_max=hole_max_param, fill_2d=True)
img_fin = remove_small_objects(img_noholes>0, min_size=smallobj_param)

#Save final segmented image
imwrite("processed_images_transformed/stress4/stress4_segmentation_frompipeline.tif", img_fin)





#CREATE MASK

#Load image
img = imread("input_images/stress4_transformed_cropped.tif")

#Rescale (increase brightness)
img_rescaled = rescale_intensity(img, in_range=(34, 420)) 

#Binarise at appropriate threshold
mask = img_rescaled > 55000

#Morphological operations to smooth mask
footprint = ball(3)
mask_opened = binary_opening(mask, footprint)
mask_closed = binary_closing(mask_opened, footprint)
footprint2 = np.ones((1,6,6))
mask_closed2 = binary_closing(mask_closed, footprint2)

#Fill in any small holes
mask_noholes = hole_filling(mask_closed2, hole_min=1, hole_max=1000, fill_2d=True)

#Save mask
imwrite("processed_images_transformed/stress4/stress4_mask.tif", mask_noholes)


#CREATE INITIAL SKELETON

#Restrict segmentation to mask
empty = np.empty(mask_noholes.shape, dtype="bool")
segm_masked = np.where(mask_noholes==1, img_fin, empty)

#Skeletonise masked segmentation
skel = skeletonize(segm_masked)

#Cast to binary to save space and save
skel = skel.astype("bool")
imwrite("processed_images_transformed/stress4/stress4_skel_raw.tif", skel)
segm_masked = segm_masked.astype("bool")
imwrite("processed_images_transformed/stress4/stress4_segmentation.tif", segm_masked)


#CREATE NEW SKELETON BASED ON UPDATED SEGMENTATION

#Load
segm_fixed = imread("processed_images_transformed/stress4/stress4_segmentation.tif")

skel = skeletonize(segm_fixed)
skel = skel.astype("bool")

imwrite("processed_images_transformed/stress4/stress4_skeleton.tif", skel)

