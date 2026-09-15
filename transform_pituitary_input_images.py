#Aim: transform all pituitaries to a transverse view with the posterior lobe on top and the portal vasculature leading to the right



#Load libraries
from skimage.io import imread
import numpy as np
import scipy.ndimage as ndi
from tifffile import imwrite


#Optional code to view images
#import napari
#transverse_angles = (0, 0, 90)
#coronal_angles = (180, 90, 0)
#sagittal_angles = (0, 0, 180)
#img = imread("path/to/image")
#viewer = napari.Viewer(ndisplay=3)
#viewer.add_image(img)
#viewer.camera.angles = transverse_angles


#Define rotation matrices along each axis and combine
def rotation_matrix_x(alpha_degree):
    alpha_radian = np.deg2rad(alpha_degree)
    
    rotation_alpha = [
        [1, 0, 0],
        [0, np.cos(alpha_radian), -np.sin(alpha_radian)],
        [0, np.sin(alpha_radian), np.cos(alpha_radian)]
    ]
    
    return np.array(rotation_alpha)

def rotation_matrix_z(gamma_degree):
    gamma_radian = np.deg2rad(gamma_degree)

    rotation_gamma = [
        [np.cos(gamma_radian), -np.sin(gamma_radian), 0],
        [np.sin(gamma_radian), np.cos(gamma_radian), 0],
        [0, 0, 1]
    ]
    
    return np.array(rotation_gamma)
    

def rotation_matrix_y(beta_degree):
    beta_radian = np.deg2rad(beta_degree)

    rotation_beta = [
        [np.cos(beta_radian), 0, np.sin(beta_radian)],
        [0, 1, 0],
        [-np.sin(beta_radian), 0, np.cos(beta_radian)],
    ]

    return np.array(rotation_beta)

def get_rotation_matrix(alpha, beta, gamma):
    return rotation_matrix_x(alpha) @ rotation_matrix_y(beta) @ rotation_matrix_z(gamma)


#Function to apply transformation based on https://github.com/tlambert03/napari-manual-transforms/blob/main/src/napari_manual_transforms/_util.py
def transform_array_3d(
    ary: np.ndarray, matrix, offset, reshape=True):

    if reshape:
        # Compute transformed input bounds
        iz, iy, ix = ary.shape
        out_bounds = matrix @ [
            [0, 0, 0, 0, iz, iz, iz, iz],
            [0, 0, iy, iy, 0, 0, iy, iy],
            [0, ix, 0, ix, 0, ix, 0, ix],
        ]
        # Compute the shape of the transformed input
        out_shape = (out_bounds.ptp(axis=1) + 0.5).astype(int)

        # make new larger input array.
        # FIXME: there has to be a more efficient way using output_shape and
        # offset, but so far I've failed...
        growth = np.array(out_shape) - np.array(ary.shape) 
        half_g = np.abs(growth) // 2
        ary = np.pad(ary, tuple((int(i),) for i in half_g))

    else:
        half_g = np.zeros((3,))

    M = np.eye(4)
    M[:3, :3] = matrix
    T = np.eye(4)
    T[:3, -1] = np.array(offset) + half_g
    matrix = T @ M @ np.linalg.inv(T)

    data = ndi.affine_transform(
        ary, np.linalg.inv(matrix), order=1, prefilter=False
    )
    return (data, -half_g)



#CONTROL MALE1

#Load image
img = imread("input_images/control_male1.tif")

#Define pixel size parameters
spacing_um = (4, 0.755112, 0.755112)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Apply transformation
img_transformed, _ = transform_array_3d(img, scaling_matrix, offset)

#Save transformed image
imwrite("input_images/control_male1_transformed.tif", img_transformed)



#CONTROL MALE2

#Load image
img = imread("input_images/control_male2.tif")

#Define pixel size parameters
spacing_um = (4, 0.755112, 0.755112)

#First rotate by 180 degrees
img = img[::-1, :, ::-1]

#Define rotation (order=z,y,x)
angles = (2, 29, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/control_male2_transformed.tif", img_transformed)






#CONTROL MALE3

#Load image
img = imread("input_images/control_male3.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#First rotate by 180 degrees
img = img[:, :, ::-1]

#Define rotation (order=z,y,x)
angles = (4, 3, -2)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/control_male3_transformed.tif", img_transformed)



#CONTROL MALE4

#Load image
img = imread("input_images/control_male4.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#First rotate by 180 degrees
img = img[::-1, :, ::-1]

#Define rotation (order=z,y,x)
angles = (2, 37, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/control_male4_transformed.tif", img_transformed)



#CONTROL MALE5

#Load image
img = imread("input_images/control_male5.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#Define rotation (order=z,y,x)
angles = (-1, 0, 1)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/control_male5_transformed.tif", img_transformed)



#CONTROL FEMALE1

#Load image
img = imread("input_images/control_female1.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#First rotate by 180 degrees
img = img[::-1, :, :]

#Define rotation (order=z,y,x)
angles = (-6, -40, -7)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/control_female1_transformed.tif", img_transformed)



#CONTROL FEMALE2

#Load image
img = imread("input_images/control_female2.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#First rotate by 180 degrees
img = img[::-1, :, ::-1]

#Define rotation (order=z,y,x)
angles = (5, -2, 2)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/control_female2_transformed.tif", img_transformed)



#CONTROL FEMALE3

#Load image
img = imread("input_images/control_female3.tif")

#Define pixel size parameters
spacing_um = (4, 0.975, 0.975)

#First rotate by 180 degrees
img = img[:, :, ::-1]

#Define rotation (order=z,y,x)
angles = (0, -27, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/control_female3_transformed.tif", img_transformed)



#CONTROL FEMALE4

#Load image
img = imread("input_images/control_female4.tif")

#Define pixel size parameters
spacing_um = (4, 0.975, 0.975)

#First rotate by 180 degrees
img = img[::-1, ::-1, :]

#Define rotation (order=z,y,x)
angles = (6, 24, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/control_female4_transformed.tif", img_transformed)



#LACTATING1

#Load image
img = imread("input_images/lactating1.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#First rotate by 180 degrees
img = img[:, :, ::-1]

#Define rotation (order=z,y,x)
angles = (0, 0, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/lactating1_transformed.tif", img_transformed)



#LACTATING2

#Load image
img = imread("input_images/lactating2.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#First rotate by 180 degrees as required
img = img[:, ::-1, ::-1]

#Define rotation (order=z,y,x)
angles = (-2, -6, -5)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/lactating2_transformed.tif", img_transformed)



#LACTATING3

#Load image
img = imread("input_images/lactating3.tif")

#Define pixel size parameters
spacing_um = (4, 0.975, 0.975)

#First rotate by 180 degrees as required
img = img[:, ::-1, ::-1]

#Define rotation (order=z,y,x)
angles = (-14, 0, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/lactating3_transformed.tif", img_transformed)



#STRESS1

#Load image
img = imread("input_images/stress1.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#Define rotation (order=z,y,x)
angles = (-3, 13, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/stress1_transformed.tif", img_transformed)



#STRESS2

#Load image
img = imread("input_images/stress2.tif")

#Define pixel size parameters
spacing_um = (4, 0.604089, 0.604089)

#Define rotation (order=z,y,x)
angles = (0, -4, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/stress2_transformed.tif", img_transformed)


#STRESS 3

#Load image
img = imread("input_images/stress3.tif")

#Define pixel size parameters
spacing_um = (4, 0.975, 0.975)

#First rotate by 180 degrees as required
img = img[:, ::-1, :]

#Define rotation (order=z,y,x)
angles = (0, -10, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/stress3_transformed.tif", img_transformed)



#STRESS 4

#Load image
img = imread("input_images/stress4.tif")

#Define pixel size parameters
spacing_um = (4, 0.975, 0.975)

#Define rotation (order=z,y,x)
angles = (-12, 0, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/stress4_transformed.tif", img_transformed)



#AGING1

#Load image
img = imread("input_images/aging1.tif")

#Define pixel size parameters
spacing_um = (4, 0.755112, 0.755112)

#Define rotation (order=z,y,x)
angles = (-2, -7, 4)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/aging1_transformed.tif", img_transformed)



#AGING2

#Load image
img = imread("input_images/aging2.tif")

#Define pixel size parameters
spacing_um = (4, 0.755112, 0.755112)

#Define rotation (order=z,y,x)
angles = (-18, 0, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/aging2_transformed.tif", img_transformed)



#AGING3

#Load image
img = imread("input_images/aging3.tif")

#Define pixel size parameters
spacing_um = (4, 0.975, 0.975)

#First rotate by 180 degrees as required
img = img[:, :, ::-1]

#Define rotation (order=z,y,x)
angles = (14, -15, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/aging3_transformed.tif", img_transformed)



#AGING4

#Load image
img = imread("input_images/aging4.tif")

#Define pixel size parameters
spacing_um = (4, 0.975, 0.975)

#First rotate by 180 degrees as required
img = img[:, ::-1, ::-1]

#Define rotation (order=z,y,x)
angles = (-12, 0, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/aging4_transformed.tif", img_transformed)



#AGING5

#Load image
img = imread("input_images/aging5.tif")

#Define pixel size parameters
spacing_um = (4, 0.975, 0.975)

#First rotate by 180 degrees as required
img = img[:, :, ::-1]

#Define rotation (order=z,y,x)
angles = (-19, 0, 0)

alpha = angles[0]
beta = angles[1]
gamma = angles[2]

#Create rotation matrix
rotation_matrix = get_rotation_matrix(alpha, beta, gamma)

#Define offset as centre point
offs1 = img.shape[0] / 2
offs2 = img.shape[1] / 2
offs3 = img.shape[2] / 2
offset = (offs1, offs2, offs3)

#Define scaling
x_scaling = spacing_um[2]
y_scaling = spacing_um[1]
z_scaling = spacing_um[0]

#Create scaling matrix
scaling_matrix = np.array([[z_scaling, 0, 0],
                           [0, y_scaling, 0],
                           [0, 0, x_scaling]])

#Combine matrices - note order of matrix multiplication!
transformation_matrix = rotation_matrix @ scaling_matrix

#Apply transformation
img_transformed, _ = transform_array_3d(img, transformation_matrix, offset)

#Save transformed image
imwrite("input_images/aging5_transformed.tif", img_transformed)

