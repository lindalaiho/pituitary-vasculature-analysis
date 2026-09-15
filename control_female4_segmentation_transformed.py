
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

#Read in image
img = imread("input_images/control_female4_transformed_cropped.tif")

#Divide image into reasonable parts given the differeces in z direction
img1 = img[:330]
img2 = img[330:550]
img3 = img[550:705]
img4 = img[705:]

#Apply normalisation
img1_cp = np.copy(img1)
img1_cp[img1_cp > 1600] = True
img1_cp[img1_cp != True] = False
img1[img1 > 1600] = img1.min()

img_adj1 = adjust_gamma(img1, 0.55)
img_adj1[img1_cp == 1] = img_adj1.max()

imwrite("processed_images_transformed/control_female4/img1.tif", img_adj1)

#Apply normalisation
img2_cp = np.copy(img2)
img2_cp[img2_cp > 1800] = True
img2_cp[img2_cp != True] = False
img2[img2 > 1800] = img2.min()

img_adj2 = adjust_gamma(img2, 0.55)
img_adj2[img2_cp == 1] = img_adj2.max()

imwrite("processed_images_transformed/control_female4/img2.tif", img_adj2)


#Apply normalisation
img3_cp = np.copy(img3)
img3_cp[img3_cp > 2400] = True
img3_cp[img3_cp != True] = False
img3[img3 > 2400] = img3.min()

img_adj3 = adjust_gamma(img3, 0.45)
img_adj3[img3_cp == 1] = img_adj3.max()

imwrite("processed_images_transformed/control_female4/img3.tif", img_adj3)


#Apply normalisation
img4_cp = np.copy(img4)
img4_cp[img4_cp > 3200] = True
img4_cp[img4_cp != True] = False
img4[img4 > 3200] = img4.min()

img_adj4 = adjust_gamma(img4, 0.45)
img_adj4[img4_cp == 1] = img_adj4.max()

imwrite("processed_images_transformed/control_female4/img4.tif", img_adj4)

#Join 3 and 4 at this point because the rest of the parameters work for both
img_adj3 = np.concatenate((img_adj3, img_adj4), axis=0)

#Apply smoothing
img_smooth1 = edge_preserving_smoothing_3d(img_adj1, numberOfIterations=20)
imwrite("processed_images_transformed/control_female4/img_smooth1.tif", img_smooth1)
img_smooth2 = edge_preserving_smoothing_3d(img_adj2, numberOfIterations=50)
imwrite("processed_images_transformed/control_female4/img_smooth2.tif", img_smooth2)
img_smooth3 = edge_preserving_smoothing_3d(img_adj3, numberOfIterations=40)
imwrite("processed_images_transformed/control_female4/img_smooth3.tif", img_smooth3)


#Segmentation
img_seg1 = filament_2d_wrapper(img_smooth1, [[1.8, 0.02]])
imwrite("processed_images_transformed/control_female4/seg1.tif", img_seg1)

img_seg2 = filament_2d_wrapper(img_smooth2, [[1.8, 0.008]])
imwrite("processed_images_transformed/control_female4/seg2.tif", img_seg2)

img_seg3 = filament_2d_wrapper(img_smooth3, [[1.8, 0.02]])
imwrite("processed_images_transformed/control_female4/seg3.tif", img_seg3)


#Join together
img_smooth = np.concatenate((img_smooth1, img_smooth2, img_smooth3), axis=0)
img_seg = np.concatenate((img_seg1, img_seg2, img_seg3), axis=0)
imwrite("processed_images_transformed/control_female4/img_smooth.tif", img_smooth)
imwrite("processed_images_transformed/control_female4/seg.tif", img_seg)


#Post-segmentation editing
img_bincl = binary_closing(img_seg, np.ones((1,5,5)))
img_noholes = hole_filling(img_bincl, hole_min=1, hole_max=100, fill_2d=True)
img_fin = remove_small_objects(img_noholes>0, min_size=300)

#Save final segmented image
imwrite("processed_images_transformed/control_female4/control_female4_segmentation_frompipeline.tif", img_fin)



#CREATE MASK

#Load image
img = imread("input_images/control_female4_transformed_cropped.tif")

#Rescale (increase brightness)
img_rescaled = rescale_intensity(img, in_range=(34, 420)) 

#Binarise at appropriate threshold
mask = img_rescaled > 52000

#Morphological operations to smooth mask
footprint = ball(3)
mask_opened = binary_opening(mask, footprint)
mask_closed = binary_closing(mask_opened, footprint)
footprint2 = np.ones((1,6,6))
mask_closed2 = binary_closing(mask_closed, footprint2)

#Fill in any small holes
mask_noholes = hole_filling(mask_closed2, hole_min=1, hole_max=1000, fill_2d=True)

#Save mask
imwrite("processed_images_transformed/control_female4/control_female4_mask.tif", mask_noholes)


#CREATE INITIAL SKELETON

#Restrict segmentation to mask
empty = np.empty(mask_noholes.shape, dtype="bool")
segm_masked = np.where(mask_noholes==1, img_fin, empty)

#Skeletonise masked segmentation
skel = skeletonize(segm_masked)

#Cast to binary to save space and save
skel = skel.astype("bool")
imwrite("processed_images_transformed/control_female4/control_female4_skel_raw.tif", skel)
segm_masked = segm_masked.astype("bool")
imwrite("processed_images_transformed/control_female4/control_female4_segmentation.tif", segm_masked)


#CREATE NEW SKELETON BASED ON UPDATED SEGMENTATION

#Load
segm_fixed = imread("processed_images_transformed/control_female4/control_female4_segmentation.tif")

skel = skeletonize(segm_fixed)
skel = skel.astype("bool")

imwrite("processed_images_transformed/control_female4/control_female4_skeleton.tif", skel)

