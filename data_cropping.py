########################################################################################################################
#                                                  IMPORTS                                                             #
########################################################################################################################
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import math
from nilearn.image import resample_img


########################################################################################################################
#                                                   DATA                                                               #
########################################################################################################################

# Paths
data_path = '/home/imag2/Desktop/BabySynthSeg-main/Data/FeTA_2022_train/mri'   # Path to the MRI data
segmentation_path = '/home/imag2/Desktop/BabySynthSeg-main/Data/FeTA_2022_train/segmentation'   # Path to the segmentations
results_path_2D = '/home/imag2/Desktop/Segmentation-Mounir/BabySynthSeg copie/Cropped Data/Feta_2022_train/2D'   # Path to save the cropped data and segmentations for the 2D models
results_path_3D = '/home/imag2/Desktop/Segmentation-Mounir/BabySynthSeg copie/Cropped Data/Feta_2022_train/3D'   # Path to save the cropped data and segmentations for the 3D models

# Find all the files in the data and segmentation folders which end with .nii.gz
data_files = [f for f in os.listdir(data_path) if f.endswith('.nii.gz')] 


# Load the data
data = []

for file in data_files:
    data.append((nib.as_closest_canonical(nib.load(os.path.join(data_path, file))).get_fdata(), nib.as_closest_canonical(nib.load(os.path.join(data_path, file))).affine))

print('Data length : ', len(data))


########################################################################################################################
#                                                  CROPPING                                                            #
########################################################################################################################

def MRI_cropping(mri, segmentation=None, filter_size=3, threshold=200, padding=False, dim=(256, 256, 256), normalization=True):

    # Gaussian mask median application in order to make noise disappear
    filtered_mri = ndimage.median_filter(mri[0], size=filter_size)

    # Retrieve the min intensity value in the whole image
    # Should correspond to the value of black pixel - used for padding 
    min_value = np.min(mri[0])

    # Check where the pixel intensity is higher than threshold in the filtered MRI
    intensity_indexes = np.where(filtered_mri >= threshold)

    # Retrieve the intensity values from the filtered MRI data using the intensity_indexes
    # filtered_intensity_values = filtered_mri[intensity_indexes]

    # Print the intensity values
    # print("Intensity values in the filtered MRI corresponding to indices above threshold:")
    # print(filtered_intensity_values)

    # Retrieve the intensity values from the original MRI data
    intensity_values = mri[0][intensity_indexes]

    # Retrieve the min and max intensity values
    min_intensity = np.min(intensity_values)
    max_intensity = np.max(intensity_values)

    print('min_intensity :', min_intensity)
    print('max_intensity :', max_intensity)

    # Retrieve the min and max indices for each dimension
    min_indices = [np.min(index) for index in intensity_indexes]
    max_indices = [np.max(index) for index in intensity_indexes]

    # Cropping the MRI 
    cropped_mri2 = mri[0][min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, :]  # on the 2 first dimensions
    cropped_mri3 = mri[0][min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, min_indices[2]:max_indices[2]+1]  # on the 3 dimensions

    # Append the dimensions of the cropped MRI3 to the list
    #list_cropped_mri_dim.append(cropped_mri3.shape)

    # If the segmentation is given, crop it data based on min and max indices
    if segmentation is not None:
        cropped_seg2 = segmentation[min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, :]  # on the 2 first dimensions
        cropped_seg3 = segmentation[min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, min_indices[2]:max_indices[2]+1]  # on the 3 dimensions
    else: 
        cropped_seg2 = None
        cropped_seg3 = None


    # Normalize the MRI data
    if normalization:
        # Normalize the MRI data
        cropped_mri2 = (cropped_mri2 - np.min(cropped_mri2)) / (np.max(cropped_mri2) - np.min(cropped_mri2))
        cropped_mri3 = (cropped_mri3 - np.min(cropped_mri3)) / (np.max(cropped_mri3) - np.min(cropped_mri3))


    # Padding
    if padding: 
        c_max2 = max(cropped_mri2.shape[0:2])
        pad_x2 = abs(c_max2 - cropped_mri2.shape[0])
        pad_y2 = abs(c_max2 - cropped_mri2.shape[1])

        c_max3 = max(cropped_mri3.shape)
        pad_x3 = abs(c_max3 - cropped_mri3.shape[0])
        pad_y3 = abs(c_max3 - cropped_mri3.shape[1])
        pad_z3 = abs(c_max3 - cropped_mri3.shape[2])
        print('mri pad_x3 :', pad_x3, 'pad_y3 :', pad_y3, 'pad_z3 :', pad_z3)

        cropped_padded_mri2 = np.pad(cropped_mri2, ((math.floor(pad_x2/2), math.ceil(pad_x2/2)), (math.floor(pad_y2/2), math.ceil(pad_y2/2)), (0, 0)), mode='constant', constant_values=min_value)
        cropped_padded_mri3 = np.pad(cropped_mri3, ((math.floor(pad_x3/2), math.ceil(pad_x3/2)), (math.floor(pad_y3/2), math.ceil(pad_y3/2)), (math.floor(pad_z3/2), math.ceil(pad_z3/2))), mode='constant', constant_values=min_value)

        if cropped_seg2 is not None:
            cropped_padded_seg2 = np.pad(cropped_seg2, ((math.floor(pad_x2/2), math.ceil(pad_x2/2)), (math.floor(pad_y2/2), math.ceil(pad_y2/2)), (0, 0)), mode='constant', constant_values=0)
        if cropped_seg3 is not None:
            cropped_padded_seg3 = np.pad(cropped_seg3, ((math.floor(pad_x3/2), math.ceil(pad_x3/2)), (math.floor(pad_y3/2), math.ceil(pad_y3/2)), (math.floor(pad_z3/2), math.ceil(pad_z3/2))), mode='constant', constant_values=0)


        # Resample
        # Calculate zoom factors for each dimension
        zoom_x2 = dim[0] / cropped_padded_mri2.shape[0]
        zoom_y2 = dim[1] / cropped_padded_mri2.shape[1]
        zoom_z2 = 1.0  # No resampling in the third dimension

        zoom_x3 = dim[0] / cropped_padded_mri3.shape[0]
        zoom_y3 = dim[1] / cropped_padded_mri3.shape[1]
        zoom_z3 = dim[2] / cropped_padded_mri3.shape[2]

        # Zoom the images
        cropped_padded_resampled_mri2 = ndimage.zoom(cropped_padded_mri2, (zoom_x2, zoom_y2, zoom_z2), order=1)
        cropped_padded_resampled_mri3 = ndimage.zoom(cropped_padded_mri3, (zoom_x3, zoom_y3, zoom_z3), order=1)

        if cropped_padded_seg2 is not None:
            cropped_padded_resampled_seg2 = ndimage.zoom(cropped_padded_seg2, (zoom_x2, zoom_y2, zoom_z2), order=0)
        else :
            cropped_padded_resampled_seg2 = None
        if cropped_padded_seg3 is not None:
            cropped_padded_resampled_seg3 = ndimage.zoom(cropped_padded_seg3, (zoom_x3, zoom_y3, zoom_z3), order=0)
        else :
            cropped_padded_resampled_seg3 = None

        return cropped_padded_resampled_mri2, cropped_padded_resampled_seg2, cropped_padded_resampled_mri3, cropped_padded_resampled_seg3, mri[1]


########################################################################################################################
#                                                   MAIN                                                               #
########################################################################################################################

# Parameters
filter_size = 5
threshold = 200
padding = True 
dim = (256, 256, 256)
normalization = True


# Loop over all the data
for i in tqdm(range(len(data))):
    print('Data shape before cropping:', data[i][0].shape)

    # Retrieve the name of the mri file
    file_number = data_files[i][-10:-7]   # To modify depending on the file names

    # If the mri has a corresponding segmentation in the segmentation folder
    segmentation_file = os.path.join(segmentation_path, f'segmentation_sub-{file_number}.nii.gz')  # segmentation_path + file_number + '_T2_Segmentation-label' + '.nii.gz'   # To modify depending on the file names
    if os.path.exists(segmentation_file):
        segmentation = nib.as_closest_canonical(nib.load(segmentation_file))
        segmentation_out = resample_img(segmentation, data[i][1], interpolation='nearest')
        print('Segmentation found for file:', file_number)
    else :
        segmentation = None
        print("\nSEGMENTATION FILE: ",segmentation_file)


    # Crop the data
    cropped_padded_resampled_mri2, cropped_padded_resampled_seg2, cropped_padded_resampled_mri3, cropped_padded_resampled_seg3, affine_mri = MRI_cropping(data[i], segmentation_out.get_fdata(), filter_size, threshold, padding, dim, normalization)


    # Save the results
    
    # Create the folder if it does not exist
    if not os.path.exists(results_path_2D):
        os.makedirs(results_path_2D)
    if not os.path.exists(results_path_3D):
        os.makedirs(results_path_3D)
    
    # Save the cropped data and segmentation
    nib.save(nib.Nifti1Image(cropped_padded_resampled_mri2, affine=affine_mri), os.path.join(results_path_2D, file_number + '_mri.nii.gz'))
    nib.save(nib.Nifti1Image(cropped_padded_resampled_mri3, affine=affine_mri), os.path.join(results_path_3D, file_number + '_mri.nii.gz'))
    
    nib.save(nib.Nifti1Image(cropped_padded_resampled_seg2, affine=affine_mri), os.path.join(results_path_2D, file_number + '_segmentation.nii.gz'))
    nib.save(nib.Nifti1Image(cropped_padded_resampled_seg3, affine=affine_mri), os.path.join(results_path_3D, file_number + '_segmentation.nii.gz'))