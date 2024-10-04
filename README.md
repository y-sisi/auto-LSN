# Refining Deep Learning Segmentation Maps with a Local Thresholding Approach: Application to Liver Surface Nodularity Quantification in CT



Official implementation of "Refining Deep Learning Segmentation Maps with a Local Thresholding Approach: Application to Liver Surface Nodularity Quantification in CT", accepted paper at MICCAI workshop CaPTion 2024.

**Authors**: *Sisi Yang* $^{*[1,2,3,4]}$, *Alexandre Bône* $^{[1]}$, *Thomas Decaens* $^{[5]}$, *Joan Alexis Glaunes* $^{[2,3]}$.  

[1] Guerbet Research, Villepinte, France  
[2] MAP5, Paris, France  
[3] Université Paris Cité, Paris, France 
[4] Hôpital Cochin, APHP, Paris, France
[5] Centre Hospitalier Universitaire Grenoble-Alpes, Grenoble, France

$*$ Corresponding author

This paper introduces a fully automated algorithm for the quantification of liver surface nodularity, using a refinement method on deep learning segmentation with a local thresholding approach.


# Codes

This repo contains the official codes for auto-LSN. 

## Requirements

### Image data
To run properly the code, you will have to provide 'INPUT_path' and 'OUTPUT_path' directly in the script.

All the images must be stored in the `INPUT_path` path and each patient must have two files : one portal venous phase CT scans after the windowing process, in Nifty format, and the liver segmentation in nifty format.
The filenames are as follow:
- `XXXXXXXX__XX001__VEN__windowed.nii.gz`: the portal venous phase CT-scans 
- `XXXXXXXX__XX001__VEN__liver.nii.gz`: the liver segmentation

'patient_id' in our code is filename[12:15], `001` in this example. Please correctly adjust your filename before running the code.
