import os
from os.path import join,isdir
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.signal import savgol_filter
from skimage import measure
from skimage import morphology
from scipy.interpolate import interp1d
from multiprocess import Pool
import tqdm
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
import glob
import pickle


########################Functions##########################
def load_data(path_to_scan, path_to_liver_segmentation): 

    image = nib.load(path_to_scan)

    header = image.header
    voxel_size = header.get_zooms()
    voxel_size = voxel_size[0]

    liver_segmentation = nib.load(path_to_liver_segmentation)
    liver_segmentation = liver_segmentation.get_fdata()

    image = image.get_fdata()
    return image, liver_segmentation
# =============================================
# =============================================


def _select_longest_contour(contours): 
    return sorted(contours, reverse=True, key=(lambda x: len(x)))[0]

def find_contour(slice_liver): 
    contours = measure.find_contours(slice_liver)
    contour = _select_longest_contour(contours)
    return contour

def _get_label_and_props_descending_order(seg): 
    label = measure.label(seg, background=0)
    props = measure.regionprops(label)
    props = sorted(props, key=(lambda x: x.area))[::-1]  
    label_permuted = np.zeros(label.shape, dtype=int)
    for k, prop in enumerate(props): 
        label_permuted[label == prop.label] = k + 1
    props = measure.regionprops(label_permuted)
    return label_permuted, props

def keep_largest_connected_components(seg, n_max=2): 
    label, props = _get_label_and_props_descending_order(seg)
    segs = []
    for k, prop in enumerate(props): 
        segs.append(1 * (label == (k+1)))
        if (k+1) == n_max: 
            break
    return segs
    
# =============================================
# =============================================

def _orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def _hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and _orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and _orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def _rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    U,L = _hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]
        
        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1
        
        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1

def _diameter(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    diam, pair = max([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q)) for p,q in _rotatingCalipers(Points)])
    return diam, pair



##########################
##########################

def select_top_left_component(component1, component2, mask1, mask2): 
    centroid1 = np.mean(component1, axis=0)
    centroid2 = np.mean(component2, axis=0)
    if centroid1[0] + centroid1[1] <= centroid2[0] + centroid2[1]: 
        return component1, mask1
    else: 
        return component2, mask2
    


##########################
##########################

def split_contour(contour): 
    extremity1, extremity2 = _diameter(list(map(list, list(contour))))[1]
    # Project extremities onto contour
    extremity1_projected_idx = np.argmin(np.sum((contour - np.array(extremity1)) ** 2, axis=1))
    extremity2_projected_idx = np.argmin(np.sum((contour - np.array(extremity2)) ** 2, axis=1))
    # Enforce order 1 < 2
    if extremity2_projected_idx < extremity1_projected_idx: 
        extremity1_projected_idx_tmp = extremity1_projected_idx
        extremity1_projected_idx = extremity2_projected_idx
        extremity2_projected_idx = extremity1_projected_idx_tmp
    # Extract components
    contour = list(contour)
    component1 = np.array((contour + contour)[extremity1_projected_idx:extremity2_projected_idx])
    component2 = np.array((contour + contour)[extremity2_projected_idx:extremity1_projected_idx+len(contour)])
   
    return component1, component2

def select_top_left_component(component1, component2): 
    centroid1 = np.mean(component1, axis=0)
    centroid2 = np.mean(component2, axis=0)
    if centroid1[0] + centroid1[1] <= centroid2[0] + centroid2[1]: 
        return component1
    else: 
        return component2






##########################
##########################

def exclude_points_close_to_anchors(contours, exclusion_anchors, resolution_xy, RADIUS_EXCLUSION_NEIGHBOURHOOD_CURV_IN_MM): 
    distances = np.min(cdist(contours, exclusion_anchors, metric='euclidean'), axis=1)
  
    indexes_to_include = distances * resolution_xy > RADIUS_EXCLUSION_NEIGHBOURHOOD_CURV_IN_MM
    return indexes_to_include
    
# =============================================
# =============================================



def compute_contrast_disk_batched(slice_scan, liver_border_dilated, liver_border_eroded, contour_liver, R, spacing, path_out=None):
    contour_liver_int = np.round(contour_liver).astype(int)
    x_contour_liver = contour_liver_int[:, 0]
    y_contour_liver = contour_liver_int[:, 1]
    imsize = slice_scan.shape[0]  
    idx = np.array([imsize * (px + rx) + (py + ry) for (px, py) in zip(x_contour_liver, y_contour_liver) for rx in np.arange(-R, R+1) for ry in np.arange(-R, R+1)])
    disk = morphology.disk(R)

    liver_border_dilated_batch = liver_border_dilated.ravel()[idx].reshape(-1, 2 * R + 1, 2 * R + 1)
    liver_border_eroded_batch = liver_border_eroded.ravel()[idx].reshape(-1, 2 * R + 1, 2 * R + 1)
    slice_scan_batch = slice_scan.ravel()[idx].reshape(-1, 2 * R + 1, 2 * R + 1)

    liver_border_dilated_mask_batch = (liver_border_dilated_batch + disk) == 2
    liver_border_eroded_mask_batch = (liver_border_eroded_batch + disk) == 2

    contrast_external = np.sum(slice_scan_batch * liver_border_dilated_mask_batch, axis=(1, 2)) / np.sum(liver_border_dilated_mask_batch, axis=(1, 2))
    contrast_internal = np.sum(slice_scan_batch * liver_border_eroded_mask_batch, axis=(1, 2)) / np.sum(liver_border_eroded_mask_batch, axis=(1, 2))
    contrast = contrast_external - contrast_internal

    return contrast


##########################
##########################


def cut_contours(contours_liver,SAVGOL_DEGREE, SAVGOL_WINDOW_LENGTH_IN_MM, resolution_xy): 


    if True or SAVGOL_DEGREE == 0:
        contours_cut = []
        SAVGOL_WINDOW_LENGTH_IN_VOXEL = int(SAVGOL_WINDOW_LENGTH_IN_MM / resolution_xy) // 2
        contours_cut.append(contours_liver[SAVGOL_WINDOW_LENGTH_IN_VOXEL:-SAVGOL_WINDOW_LENGTH_IN_VOXEL])
        return contours_cut
    else:
        return contours_liver




def smooth_contours(contours_liver,SAVGOL_DEGREE, SAVGOL_WINDOW_LENGTH_IN_MM, resolution_xy): 
    savgol_window_length_in_voxels = int(np.round(SAVGOL_WINDOW_LENGTH_IN_MM / resolution_xy))

    try:
        contours_smooth = [
            np.stack([savgol_filter(contours_liver[:, 0], savgol_window_length_in_voxels, SAVGOL_DEGREE), 
                      savgol_filter(contours_liver[:, 1], savgol_window_length_in_voxels, SAVGOL_DEGREE)], axis=1) 
            ]
    except ValueError as error:

        contours_smooth = [
            np.stack([savgol_filter(contours_liver[:, 0], savgol_window_length_in_voxels + 1, SAVGOL_DEGREE), 
                      savgol_filter(contours_liver[:, 1], savgol_window_length_in_voxels + 1, SAVGOL_DEGREE)], axis=1) 
            ]
    return contours_smooth


##########################
##########################
    
def upsample_outer_contours(contours, UPSAMPLE_RATIO): 
    contours = np.stack(
        [interp1d(np.arange(len(contours)), contours[:, 0])(np.linspace(0, len(contours)-1, (len(contours)-1)*UPSAMPLE_RATIO+1)), 
         interp1d(np.arange(len(contours)), contours[:, 1])(np.linspace(0, len(contours)-1, (len(contours)-1)*UPSAMPLE_RATIO+1))], axis=1) 
         
    return contours

##########################
##########################


def binarize_outer_contours(contour_liver, slice_liver):
    contour_binary = np.zeros(slice_liver.shape, dtype='uint8')
    contour_binary[np.round(contour_liver[:, 0]).astype('int'), np.round(contour_liver[:, 1]).astype('int')] = 1

    return contour_binary    


##########################
##########################

def compute_projections_outer_contours_distances(contours_liver, contours_smooth):
 
    distance_matrix = cdist(contours_liver, contours_smooth, metric='euclidean')

    distances_in_voxel = list(np.min(distance_matrix, axis=1))
    voxel_to_projection_distances = (contours_liver, contours_smooth[np.argmin(distance_matrix, axis=1)], np.array(distances_in_voxel))

    return voxel_to_projection_distances



def plot_liver_dilated_eroded (slice_scan, slice_liver, liver_dilated, liver_eroded, path_out=None): 
    
    liver_border_dilated = liver_dilated - slice_liver

    liver_border_eroded = slice_liver - liver_eroded


    return liver_border_dilated, liver_border_eroded


def intersection_contour_without_index(mask, contour):
    contour_int = (contour + 0.5).astype('int')
    contour_masked = contour[mask[contour_int[:, 0], contour_int[:, 1]] > 0]
    return contour_masked

INPUT_path = "/sharedrive/2023_LiverNodularity/"

OUTPUT_path = '/sharedrive/2023_LiverNodularity/DELIVER_clean/results/miccai_test'

image_folder = INPUT_path

if not isdir(OUTPUT_path): 
    os.mkdir(OUTPUT_path)

root_path = '/sharedrive/2023_LiverNodularity/'
image_folder = os.path.join(root_path, "2_datasets", "dataset_windowed")


image_list_files = [elem for elem in os.listdir(image_folder) if elem.endswith('VEN__windowed.nii.gz')]
liver_segmentation_list_files = [elem for elem in os.listdir(image_folder) if elem.endswith('VEN__liver.nii.gz')]

image_list_files.sort()
liver_segmentation_list_files.sort()
image_list = [os.path.join(image_folder, elem) for elem in image_list_files]
liver_segmentation_list = [os.path.join(image_folder, elem) for elem in liver_segmentation_list_files]

print("image_list_files",image_list_files)
print("image_list",image_list)


##############hyperparameters##################
SAVGOL_WINDOW_LENGTH_IN_MM = 47
SAVGOL_DEGREE = 2
UPSAMPLE_RATIO = 10
PERCENTAGE_CENTRAL_SLICES = 0.7
RADIUS_EXCLUSION_NEIGHBOURHOOD_CONTRAST_IN_MM = 8
CONTRAST_THRESHOLD = 60   #Δmin
R_DISK_IN_MM = 5 #R_DISK_IN_MM = R_contrast
WIDTH_IN_MM = R_DISK_IN_MM #WIDTH_IN_MM = R_shell
R_MASK_IN_MM = 2.5


def compute_lsn_for_slice_contrast(args):
    z_i, slice_liver, slice_scan, spacing, resolution_xy, savgol_window_length_in_voxels, r_disk_in_voxels, r_mask_in_voxels, patient_id = args
    r_disk_in_voxels = int(np.round(R_DISK_IN_MM / spacing[0]))
    r_mask_in_voxels = int(np.round(R_MASK_IN_MM / spacing[0]))
    slice_livers = keep_largest_connected_components(slice_liver)
    outer_contours_cv_list_nearest = [] 
    
    contours_binary_cv_list_nearest = []
    contours_smooth_cv_list_nearest = []
    output__distances_in_mm_cv_filtered = []

    output__voxel_projection_pairs_cv_filtered = ([],[])

    for index_liver, slice_liver in enumerate(slice_livers): 

        ###
        ### 1.1. Compute green zone (dilation)
        ###
        DILATION_WIDTH_IN_MM = WIDTH_IN_MM 
        dilation_width_in_voxels = int(np.round(DILATION_WIDTH_IN_MM / spacing[0]))
        liver_dilated = morphology.binary_dilation(slice_liver, footprint=morphology.square(dilation_width_in_voxels))
        liver_external = (liver_dilated - slice_liver).astype('uint8')
        ###
        ### 1.2. Compute yellow zone (erosion)
        ###
        EROSION_WIDTH_IN_MM = DILATION_WIDTH_IN_MM
        erosion_width_in_voxels = int(np.round(EROSION_WIDTH_IN_MM / spacing[0]))
        liver_eroded = morphology.binary_erosion(slice_liver, footprint=morphology.square(erosion_width_in_voxels))
        liver_internal = (slice_liver - liver_eroded).astype('uint8')


        ### 1.3. Plot eroded and dilated liver
        ###

        liver_border_dilated, liver_border_eroded = plot_liver_dilated_eroded(slice_scan, slice_liver, liver_dilated, liver_eroded,path_out=None)    

        ### 1.4. detect contours and select only outer contours
        ###
        contours_liver = measure.find_contours(slice_liver) 
        contour_liver = contours_liver[0]
        component1, component2 = split_contour(contour_liver)
        outer_contour_i_without_selection = select_top_left_component(component1, component2)        
        outer_contour_i_without_selection = cut_contours(outer_contour_i_without_selection,SAVGOL_DEGREE,SAVGOL_WINDOW_LENGTH_IN_MM,resolution_xy)

        ### 1.5: select only outer_contours long enough
        ###
        if not (len(np.array(outer_contour_i_without_selection[0])) > savgol_window_length_in_voxels):
            continue

        outer_contour_i = np.array(outer_contour_i_without_selection[0])
        outer_contour_i = outer_contour_i
        
        ###
        ### 1.7: compute disk and contrast on selected outer_contour_i 
        ###
        if not(outer_contour_i.shape[0] > 0):
            continue

        contrast = compute_contrast_disk_batched(slice_scan, liver_border_dilated, liver_border_eroded, outer_contour_i,r_disk_in_voxels, spacing, path_out= None)
        C = np.array(contrast)

        #### 1.8: keep only well contrasted regions of outer_contour_i and at a distance R_CONTRAST of bad contrast points#########
        exclusion_anchors_contrast = [point for contrast_point, point in zip(contrast, outer_contour_i) if abs(contrast_point) < CONTRAST_THRESHOLD]
        exclusion_anchors_contrast = np.array(exclusion_anchors_contrast)
        if exclusion_anchors_contrast.shape[0] > 0:
            points_contrast_to_include = exclude_points_close_to_anchors(outer_contour_i, exclusion_anchors_contrast, resolution_xy, RADIUS_EXCLUSION_NEIGHBOURHOOD_CONTRAST_IN_MM)
            outer_contour_i_contrast = outer_contour_i[points_contrast_to_include]
        else:
            outer_contour_i_contrast = outer_contour_i
            
        ### 1.9: to create a region of interest around the well contrasted outer_contours to select contours to keep
        mask = np.zeros_like(slice_scan)
        for point in outer_contour_i_contrast:
            x_outer_contour = point[0]
            y_outer_contour = point[1]
            x_int = int(np.round(x_outer_contour))
            y_int = int(np.round(y_outer_contour))
            mask[x_int-r_mask_in_voxels:x_int+r_mask_in_voxels+1, y_int-r_mask_in_voxels:y_int+r_mask_in_voxels+1] += morphology.disk(r_mask_in_voxels)
        mask_contrasted = (mask > 0).astype('bool')


        #
        ## 1.10: to find all contours of liver inside the region of interest (dilated and eroded liver) from slice_scan with Chan Vese 
        ########apply mask to find all contours inside the region of interest (on region of dilated and eroded liver)

        if not(slice_scan[mask>0].any()):
            continue

        cv = slice_scan > threshold_otsu(slice_scan[mask>0])
        slice_scan_cv = cv

        liver_contours_cv = measure.find_contours(slice_scan_cv,mask=mask_contrasted)
        liver_contours_cv = [liver_contour_cv for liver_contour_cv in liver_contours_cv if len(liver_contour_cv) > savgol_window_length_in_voxels]
        ###1.11: to find the nearest contour inside of a well contrasted mask
        #######################################
        ########FOR EACH MASK, 
        ######## FIND ALL CONTOURS INSIDE THE MASK, AND CHOOSE THE NERAEST CONTOUR: KEEP THE CONTOUR WITH THE MINIMUM DISTANCE TO THE CONTOUR OF THE AUTOMATIC SEGMENTATION
        ########RESULT =  ONE REAL CONTOUR AND ONE SMOOTH CONTOUR PER MASK
        ########################################
        label = measure.label(mask_contrasted)
        props = measure.regionprops(label)
        for k, _ in enumerate(props):
            mask = (label == k + 1).astype('uint8')

            distances_near_segmentation_inside_mask = []

            for index_contour, liver_contour_cv in enumerate(liver_contours_cv):  


                #################to find the nearest contour inside a mask ########################
                contour_inside_mask = intersection_contour_without_index(mask, liver_contour_cv)
                if len(contour_inside_mask) > 0:
                    dist = np.mean(np.min(cdist(contour_inside_mask, outer_contour_i, metric='euclidean'), axis=1))
                    distances_near_segmentation_inside_mask.append(dist)
                else:
                    distances_near_segmentation_inside_mask.append(1000000000000000000000000000) ###to avoid empty distances for contours outside of the mask
                    
            if len(distances_near_segmentation_inside_mask)>0:

                idx_min = np.argmin(distances_near_segmentation_inside_mask)

                outer_contour_cv_nearest_inside_mask = intersection_contour_without_index(mask, liver_contours_cv[idx_min])

                contour_smooth_cv = smooth_contours(liver_contours_cv[idx_min], SAVGOL_DEGREE, SAVGOL_WINDOW_LENGTH_IN_MM, resolution_xy)
                contour_smooth_cv = np.array(contour_smooth_cv[0])
                contour_smooth_upsampled_cv = upsample_outer_contours(contour_smooth_cv, UPSAMPLE_RATIO)
                voxel_projection_distances_cv = compute_projections_outer_contours_distances(liver_contours_cv[idx_min], contour_smooth_upsampled_cv) 
                contour_smooth_cv_nearest_inside_mask = intersection_contour_without_index(mask, voxel_projection_distances_cv[1])

                if outer_contour_cv_nearest_inside_mask.shape[0] > 0 and contour_smooth_cv_nearest_inside_mask.shape[0] > 0:
                    outer_contours_cv_list_nearest.append(outer_contour_cv_nearest_inside_mask)
                    contour_binary_cv_filtered = binarize_outer_contours(outer_contour_cv_nearest_inside_mask, slice_liver) 
                    contours_binary_cv_list_nearest.append(contour_binary_cv_filtered)
                    contours_smooth_cv_list_nearest.append(contour_smooth_cv_nearest_inside_mask)

    outer_contours_cv_list_nearest_masked_long = [outer_contour_cv_list_nearest_masked
                                                  for outer_contour_cv_list_nearest_masked in outer_contours_cv_list_nearest 
                                                  if len(outer_contour_cv_list_nearest_masked) > savgol_window_length_in_voxels]
    contours_smooth_cv_list_nearest_masked_long = [contour_smooth_cv_list_nearest_masked
                                                   for contour_smooth_cv_list_nearest_masked in contours_smooth_cv_list_nearest
                                                   if len(contour_smooth_cv_list_nearest_masked) > savgol_window_length_in_voxels]
    


    if len(outer_contours_cv_list_nearest_masked_long) > 0 and len(contours_smooth_cv_list_nearest_masked_long) > 0:

        for index_contour, (outer_contour_cv_list_nearest_masked_long, contour_smooth_cv_list_nearest_masked_long) in enumerate(zip (outer_contours_cv_list_nearest_masked_long,contours_smooth_cv_list_nearest_masked_long)):

            contour_binary = binarize_outer_contours(outer_contour_cv_list_nearest_masked_long, slice_liver)
            voxel_projection_distances_cv_filtered_inside_mask_all = compute_projections_outer_contours_distances(outer_contour_cv_list_nearest_masked_long, 
                                                                                                                  contour_smooth_cv_list_nearest_masked_long)
            output__voxel_projection_pairs_cv_filtered =  (output__voxel_projection_pairs_cv_filtered[0] + voxel_projection_distances_cv_filtered_inside_mask_all[0].tolist(),
                                                    output__voxel_projection_pairs_cv_filtered[1] + voxel_projection_distances_cv_filtered_inside_mask_all[1].tolist())
            distances_in_voxel_cv_filtered = voxel_projection_distances_cv_filtered_inside_mask_all[2]

            distances_in_mm_cv_filtered = resolution_xy * np.array(distances_in_voxel_cv_filtered)
            output__distances_in_mm_cv_filtered += list(distances_in_mm_cv_filtered)




    return z_i, output__distances_in_mm_cv_filtered, output__voxel_projection_pairs_cv_filtered, contours_smooth_cv_list_nearest_masked_long 


def compute_LSN(image_list,liver_segmentation_list, save_path,
                PERCENTAGE_CENTRAL_SLICES,
                SAVGOL_DEGREE,SAVGOL_WINDOW_LENGTH_IN_MM,UPSAMPLE_RATIO, 
                CONTRAST_THRESHOLD, RADIUS_EXCLUSION_NEIGHBOURHOOD_CONTRAST_IN_MM, 
                R_DISK_IN_MM, R_MASK_IN_MM,WIDTH_IN_MM):

    for path_nifti_image, path_nifti_liver_segmentation in zip(image_list, liver_segmentation_list):

        filename = os.path.basename(path_nifti_image)
        
        patient_id = filename[12:15] 

        #################################
        spacing = nib.load(path_nifti_image).header.get_zooms()
        resolution_xy = 0.5 * (spacing[0] + spacing[1])


        ###
        ### 0. Hyperparameters
        ###

        savgol_window_length_in_voxels = int(np.round(SAVGOL_WINDOW_LENGTH_IN_MM / resolution_xy))
        r_disk_in_voxels = int(np.round(R_DISK_IN_MM/ spacing[0]))
        r_mask_in_voxels = int(np.round(R_MASK_IN_MM / spacing[0]))
        
        ### 1. Load
        ###
        scan, liver = load_data(path_nifti_image, path_nifti_liver_segmentation)

        liver_z = np.sum(liver, axis=(0, 1))
        z_min = np.argmax(liver_z > 0)
        z_max = liver.shape[2] - 1 - np.argmax(liver_z[::-1] > 0)
        z_mid = round(np.mean([z_min, z_max]))
    
        z_min_ = round(z_mid - (z_max - z_min) * PERCENTAGE_CENTRAL_SLICES * 0.5)
        z_max_ = round(z_mid + (z_max - z_min) * PERCENTAGE_CENTRAL_SLICES * 0.5)



        args = list(range(z_min_, z_max_, 1))

        args = [(
            z_i, 
            np.transpose(liver[:, :, z_i])[::-1, ::-1], 
            np.transpose(scan[:, :, z_i])[::-1, ::-1], 
            spacing, 
            resolution_xy, 
            savgol_window_length_in_voxels,
            r_disk_in_voxels, 
            r_mask_in_voxels,
            patient_id
        ) for z_i in args]

        RESULTS = {
            'patient_id': patient_id,
            'path_to_volume': path_nifti_image, 
            'path_to_liver': path_nifti_liver_segmentation, 
            'results_per_slice': {}
        }
        
        
        with Pool(os.cpu_count() - 1) as pool:
            with tqdm.tqdm(total=len(args)) as pbar:
                 print('Computing LSN for %s' % patient_id)
                 for z_i, distances_in_mm_cv_filtered, voxel_projection_pairs_cv_filtered_inside_mask_all, contours_smooth in pool.imap_unordered(compute_lsn_for_slice_contrast, args):
                     pbar.update()
                     RESULTS['results_per_slice'][z_i] = {
                         'slice_id': z_i, 
                        'distances_in_mm': distances_in_mm_cv_filtered, 
                        'voxel_projection_pairs': voxel_projection_pairs_cv_filtered_inside_mask_all,
                         'contours_smooth': contours_smooth
                    }


        filename_id = 'results_LSN_score_for_%s_contrast_%s_exclusion_%s_R_DISK_IN_MM_%s_R_MASK_IN_MM_%s_savgol_%s_degree%d.pkl'% (patient_id,
                CONTRAST_THRESHOLD,RADIUS_EXCLUSION_NEIGHBOURHOOD_CONTRAST_IN_MM,
                R_DISK_IN_MM,R_MASK_IN_MM,SAVGOL_WINDOW_LENGTH_IN_MM, SAVGOL_DEGREE)
  

        with open(join(save_path, filename_id), 'wb') as f:
            pickle.dump(RESULTS, f)




compute_LSN(image_list[5:7], liver_segmentation_list[5:7], save_path,
            PERCENTAGE_CENTRAL_SLICES,
            SAVGOL_DEGREE,SAVGOL_WINDOW_LENGTH_IN_MM,UPSAMPLE_RATIO, 
            CONTRAST_THRESHOLD, RADIUS_EXCLUSION_NEIGHBOURHOOD_CONTRAST_IN_MM, 
            R_DISK_IN_MM, R_MASK_IN_MM,WIDTH_IN_MM)



    
results_list_path = glob.glob(os.path.join(save_path, f'results_LSN_score_for_*_contrast_{CONTRAST_THRESHOLD}_exclusion_{RADIUS_EXCLUSION_NEIGHBOURHOOD_CONTRAST_IN_MM}_R_DISK_IN_MM_{R_DISK_IN_MM}_R_MASK_IN_MM_{R_MASK_IN_MM}_savgol_{SAVGOL_WINDOW_LENGTH_IN_MM}_degree{SAVGOL_DEGREE}.pkl'))
results_list = [os.path.basename(x) for x in results_list_path]
list_patient_id = [x[22:25] for x in results_list]
df = pd.DataFrame({'patient': list_patient_id})


for result in results_list:
    patient_id = result[22:25]
    results = pickle.load(open(join(save_path, result), 'rb'))
    map_slice_to_lsn_results = results['results_per_slice']
    slice_ids = sorted(list(map_slice_to_lsn_results.keys()))
    lsn_distribution = []
    for value in map_slice_to_lsn_results.values(): 
        lsn_distribution += value['distances_in_mm']
    if len(lsn_distribution) > 0:
        df.loc[df["patient"]== patient_id,"auto-LSN"] = np.median(lsn_distribution)*10

df.to_excel(join(save_path, 'df_contrast_%d_exclusion_%d_R_DISK_IN_MM_%d_R_MASK_IN_MM_%s_savgol_%ddegree_%d.xlsx'% (
        CONTRAST_THRESHOLD,RADIUS_EXCLUSION_NEIGHBOURHOOD_CONTRAST_IN_MM,
        R_DISK_IN_MM,R_MASK_IN_MM,SAVGOL_WINDOW_LENGTH_IN_MM,SAVGOL_DEGREE)), float_format='%.4f', index=False)
