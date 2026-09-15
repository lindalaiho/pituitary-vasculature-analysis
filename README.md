### Code to segment and derive metrics from the vasculature in the anterior pituitary lobe processed by iDISCO+

Pipeline steps:
1. Transform images to be in the same orientation: transform_pituitary_input_images.py
2. Crop the images to cover the same region of the anterior lobe: crop_transformed_pituitary_input_images.py
3. Segment the vasculature, generate a mask of the anterior lobe, and obtain a skeleton of the vasculature for each sample: samplename_segmentation_transformed.py
4. Calculate properties of the vasculature for each sample: samplename_geometry_transformed.py
