#Aim: crop all images to exclude unnecessary background AND to crop vertically where the posterior lobe appears



#Load libraries
from skimage.io import imread
from tifffile import imwrite



#CONTROL MALE1

#Load image
img = imread("input_images/control_male1_transformed.tif")

#Crop
img = img[14:799, 623:1575, 506:1994]

#Save cropped image
imwrite("input_images/control_male1_transformed_cropped.tif", img)



#CONTROL MALE2

#Load image
img = imread("input_images/control_male2_transformed.tif")

#Crop
img = img[167:1130, 579:1896, 178:1432]

#Save cropped image
imwrite("input_images/control_male2_transformed_cropped.tif", img)



#CONTROL MALE3

#Load image
img = imread("input_images/control_male3_transformed.tif")

#Crop
img = img[65:523, 1061:2196, 928:1990]

#Save cropped image
imwrite("input_images/control_male3_transformed_cropped.tif", img)



#CONTROL MALE4

#Load image
img = imread("input_images/control_male4_transformed.tif")

#Crop
img = img[202:671, 1242:1958, 1067:1978]

#Save cropped image
imwrite("input_images/control_male4_transformed_cropped.tif", img)



#CONTROL MALE5

#Load image
img = imread("input_images/control_male5_transformed.tif")

#Crop
img = img[263:673, 1262:2042, 1037:1981]

#Save cropped image
imwrite("input_images/control_male5_transformed_cropped.tif", img)



#CONTROL FEMALE1

#Load image
img = imread("input_images/control_female1_transformed.tif")

#Crop
img = img[343:1019, 1007:1836, 600:1858]

#Save cropped image
imwrite("input_images/control_female1_transformed_cropped.tif", img)



#CONTROL FEMALE2

#Load image
img = imread("input_images/control_female2_transformed.tif")

#Crop
img = img[209:678, 1050:1851, 1000:1981]

#Save cropped image
imwrite("input_images/control_female2_transformed_cropped.tif", img)



#CONTROL FEMALE3

#Load image
img = imread("input_images/control_female3_transformed.tif")

#Crop
img = img[489:1314, 164:1203, 596:1963]

#Save cropped image
imwrite("input_images/control_female3_transformed_cropped.tif", img)



#CONTROL FEMALE4

#Load image
img = imread("input_images/control_female4_transformed.tif")

#Crop
img = img[275:1175:, 197:1261, 588:2163]

#Save cropped image
imwrite("input_images/control_female4_transformed_cropped.tif", img)



#LACTATING 1

#Load image
img = imread("input_images/lactating1_transformed.tif")

#Crop
img = img[17:663, 1113:1916, 956:2108]

#Save cropped image
imwrite("input_images/lactating1_transformed_cropped.tif", img)



#LACTATING 2

#Load image
img = imread("input_images/lactating2_transformed.tif")

#Crop
img = img[215:894, 1049:2007, 873:2133]

#Save cropped image
imwrite("input_images/lactating2_transformed_cropped.tif", img)



#LACTATING 3

#Load image
img = imread("input_images/lactating3_transformed.tif")

#Crop
img = img[309:939, 282:1240, 581:1775]

#Save cropped image
imwrite("input_images/lactating3_transformed_cropped.tif", img)



#STRESS 1

#Load image
img = imread("input_images/stress1_transformed.tif")

#Crop
img = img[380:1060, 997:1998, 806:1896]

#Save cropped image
imwrite("input_images/stress1_transformed_cropped.tif", img)



#STRESS 2

#Load image
img = imread("input_images/stress2_transformed.tif")

#Crop
img = img[310:608, 1198:1812, 1222:1928]

#Save cropped image
imwrite("input_images/stress2_transformed_cropped.tif", img)



#STRESS 3

#Load image
img = imread("input_images/stress3_transformed.tif")

#Crop
img = img[342:1097, 110:1070, 480:1829]

#Save cropped image
imwrite("input_images/stress3_transformed_cropped.tif", img)



#STRESS 4

#Load image
img = imread("input_images/stress4_transformed.tif")

#Crop
img = img[148:636, 341:1275, 601:1697]

#Save cropped image
imwrite("input_images/stress4_transformed_cropped.tif", img)



#AGING 1

#Load image
img = imread("input_images/aging1_transformed.tif")

#Crop
img = img[197:1058, 740:1765, 666:1984]

#Save cropped image
imwrite("input_images/aging1_transformed_cropped.tif", img)



#AGING 2

#Load image
img = imread("input_images/aging2_transformed.tif")

#Crop
img = img[17:713, 564:1572, 449:1808]

#Save cropped image
imwrite("input_images/aging2_transformed_cropped.tif", img)



#AGING 3

#Load image
img = imread("input_images/aging3_transformed.tif")

#Crop
img = img[432:980, 329:1149, 983:1965]

#Save cropped image
imwrite("input_images/aging3_transformed_cropped.tif", img)



#AGING 4

#Load image
img = imread("input_images/aging4_transformed.tif")

#Crop
img = img[97:, 472:1362, 799:1640]

#Save cropped image
imwrite("input_images/aging4_transformed_cropped.tif", img)



#AGING 5

#Load image
img = imread("input_images/aging5_transformed.tif")

#Crop
img = img[189:779, 530:1491, 732:1845]

#Save cropped image
imwrite("input_images/aging5_transformed_cropped.tif", img)
