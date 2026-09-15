
#Import libraries
from skimage.io import imread
import numpy as np
import skan
import pandas as pd
import scipy.ndimage as ndi
from itertools import islice
import math
import os
import json

#Code to calculate:
    #Vascular proportion
    #Branching point density
    #End point density
    #Tortuosity
    #Radius
    #Length
    #Angle
    #Distance to nearest vessel

#Load segmentation, mask and skeleton
segm = imread("processed_images_transformed/stress3/stress3_segmentation.tif")
mask = imread("processed_images_transformed/stress3/stress3_mask.tif")
skel = imread("processed_images_transformed/stress3/stress3_skeleton.tif")

#Get proportion of vasculature
ap_vol = np.count_nonzero(mask)
segm_vol = np.count_nonzero(segm)
vessel_perc = segm_vol / ap_vol * 100

#Get skan skeleton
skan_skel = skan.Skeleton(skel)
skan_skel_summary = skan.summarize(skan_skel)

#Prune very short branches (likely to be artifacts)
def remove_small_branches(
        skeleton: skan.Skeleton,
        summary: pd.DataFrame,
        min_branch_dist: float = 50,
        branch_type_0: bool = True,
        branch_type_1: bool = True,
        branch_type_2: bool = False,
        branch_type_3: bool = False,
):
    too_short = (summary['branch-distance'] < min_branch_dist)

    # get the branches that are of the type to cut
    #    branch types can be selected as follows:
    #     0 = endpoint-to-endpoint (isolated branch)
    #     1 = junction-to-endpoint
    #     2 = junction-to-junction
    #     3 = isolated cycle
    types_to_prune = []
    if branch_type_0:
        types_to_prune.append(0)
    if branch_type_1:
        types_to_prune.append(1)
    if branch_type_2:
        types_to_prune.append(2)
    if branch_type_3:
        types_to_prune.append(3)
    wrong_type = summary['branch-type'].isin(types_to_prune)

    # Pruning is implemented in https://github.com/jni/skan/pull/117
    # pass in a list of branch ids to get a new skeleton with those branches
    # removed.
    to_cut = too_short & wrong_type
    pruned = skeleton.prune_paths(np.flatnonzero(to_cut))

    summary_pruned = skan.summarize(pruned)
    summary_pruned['index'] = np.arange(summary_pruned.shape[0]) + 1

    return pruned, summary_pruned

skan_skel_pruned = remove_small_branches(skan_skel, summary=skan_skel_summary, min_branch_dist=3)

#Replace old skeleton with new
skan_skel = skan_skel_pruned[0]
skan_skel_summary = skan_skel_pruned[1] 

#Define all vertices
endpt_coord = skan_skel.coordinates[skan_skel.degrees == 1]
bp_coord = skan_skel.coordinates[skan_skel.degrees > 2]

#Count number of each vertex type
endpt_n = len(endpt_coord)
bp_n = len(bp_coord)

#Density of vertex points
endpt_dens = endpt_n / ap_vol
bp_dens = bp_n / ap_vol

#Get branch lengths
branch_length = skan_skel.path_lengths()

#Tortuosity
tortuosity = skan_skel_summary['branch-distance'] / skan_skel_summary['euclidean-distance'] - 1

#Get radius
dst = ndi.distance_transform_edt(segm)
all_paths = [
        skan_skel.path_coordinates(i)
        for i in range(skan_skel.n_paths)
        ]
for i in range(len(all_paths)):
    all_paths[i] = all_paths[i].astype(int)
        
dst_per_path = []
for kordinater in all_paths:
    for kordinat in kordinater:
        dst_val = dst[kordinat[0], kordinat[1], kordinat[2]]
        dst_per_path.append(dst_val)
    
length_to_split = []
for i in all_paths:
    length_to_split.append(len(i))

inp = iter(dst_per_path)
dst_per_path_split = [list(islice(inp, elem))
    for elem in length_to_split]

mean_radius_per_segm = []
for lst in dst_per_path_split:
    mean_radius_per_segm.append(np.mean(lst))
    

#Get angles
def get_angle(segment):
    
    #Define start and end coordinates of the segment
    A1 = segment["image-coord-src-0"]
    A2 = segment["image-coord-src-1"]
    A3 = segment["image-coord-src-2"]
    
    B1 = segment["image-coord-dst-0"]
    B2 = segment["image-coord-dst-1"]
    B3 = segment["image-coord-dst-2"]
    
    D1 = skel.shape[0]
    
    #Calculate angle
    dot_product = (B1-A1)*(D1-A1)
    magnitude1 = math.sqrt((B1-A1)**2 + (B2-A2)**2 + (B3-A3)**2)
    magnitude2 = D1-A1
    
    angle_rad = math.acos(dot_product/(magnitude1*magnitude2))
    angle_deg = angle_rad * (180/math.pi)
    
    return angle_deg

angles = skan_skel_summary.apply(get_angle, axis=1)

#Add all per vessel segment metrics into df and save
rad_series = pd.Series(mean_radius_per_segm)
branch_series = pd.Series(branch_length)
segm_n = pd.Series(np.arange(skan_skel.n_paths))
angle_series = pd.Series(angles)
segment_metrics = pd.concat([rad_series, tortuosity, branch_series, segm_n, angle_series], axis = 1)
segment_metrics = segment_metrics.rename({0: 'Radius', 1: 'Tortuosity_index', 2: 'Branch_length', 3: 'Segment_number', 4: "Angle"}, axis=1)
#Create directory to save into
os.makedirs("outs/geometry_transformed/stress3")
#Save
segment_metrics.to_csv("outs/geometry_transformed/stress3/stress3_segment_metrics.csv", index=False)
    
#Add all per pituitary metrics into df and save   
volume_metrics = {'Vascular_percentage': vessel_perc,
                'Endpoint_number': endpt_n,
                'Endpoint_density_per_um': endpt_dens,
                'Branching_point_number': bp_n,
                'Branching_point_density_per_um': bp_dens}
volume_metrics = pd.DataFrame(volume_metrics, index =[0])
volume_metrics.to_csv("outs/geometry_transformed/stress3/stress3_volume_metrics.csv", index=False)

#Convert to boolean to save space
segm = segm.astype("bool")
#Get distances from each non-vessel pixel to nearest vessel (currently vessel is True so invert array)
dst = ndi.distance_transform_edt(~segm)
#Remove values outside the mask (artifacts)
empty = np.empty(segm.shape, dtype="bool")
dst_mod = np.where(mask == 0, empty, dst)
#Extract into 1D array
vals = dst_mod.flatten()
vals = vals.astype("float32").tolist()
#Remove zeroes
filtered = [x for x in vals if x !=0]
#Write results into JSON file
with open("outs/geometry_transformed/stress3/stress3_distance_to_nearest_vessel.json", "w") as f:
    json.dump(filtered, f)

