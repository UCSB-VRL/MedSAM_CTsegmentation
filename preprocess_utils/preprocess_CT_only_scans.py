import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, default_collate
from einops import repeat
from icecream import ic
from PIL import Image
import sys
import csv
import nibabel as nib
import cv2

def preprocess_CT_only_scans(dir):
    output_size=[512, 512]
    
    # Get the files list from the directory
    sample_list = [f for f in os.listdir(dir) if 'Mask' not in f]
    num_data = len(sample_list)
    if num_data ==0:
        print("No relevant file, exiting")
        exit(0) 
    print( "Number of files loading is", num_data)
    print("sample list", sample_list)

    for num, sample in enumerate(sample_list):
        folder = 'preprocess_'+sample.rstrip('.nii.gz')
        scan_save_dir = dir+f'/{folder}/imgs'
        affine_save_dir = dir+f'{folder}/affine'

        image_path = os.path.join(dir, sample_list[num])
        print("image path", image_path) 

        # Check the file extension to determine how to read the file
        if image_path.endswith(('.nii', '.nii.gz')):
            image, affine = read_nifti(image_path)

        # load each of them and get 3 stack of slices_ currently 3 cont.
        # Resize the image and label to the output size
        x, y, num_slices = image.shape
        # Compute resize factors
        x_factor = output_size[0] / x
        y_factor = output_size[1] / y
        
        if x != output_size[0] or y != output_size[1]:
            image = zoom(image, (x_factor, y_factor, 1), order=3)

        x, y, num_slices = image.shape
        for i in range(0, num_slices):
            image_RGB = np.repeat(image[..., i:i + 1], 3, axis=-1)
            image_RGB = image_RGB.astype(np.float32).transpose(2, 0, 1)

            # Update the affine matrix
            resized_affine = affine
            resized_affine[0, 0] *= x_factor  # Scale x-axis
            resized_affine[1, 1] *= y_factor  # Scale y-axis

            # Create the save directory if it doesn't exist
            os.makedirs(scan_save_dir, exist_ok=True)
            os.makedirs(affine_save_dir, exist_ok = True)

            # Define the save path
            scan_save_path = os.path.join(scan_save_dir, os.path.basename(image_path).replace('.nii.gz', f'_{i}.npy'))
            affine_save_path = os.path.join(affine_save_dir, os.path.basename(image_path).replace('.nii.gz', f'_{i}.npy'))

            # Save the NumPy array to a file
            np.save(scan_save_path, image_RGB)
            print(f"Saved f'{i}' as {scan_save_path}")
            np.save(affine_save_path, resized_affine)
            print(f"Saved f'{i}' as {affine_save_path}")

def read_nifti(nifti_path, num_class=4):

    print("nifti path")
    nifti_img = nib.load(nifti_path)
    image = nifti_img.get_fdata()
    #print("IMAGE SHAPE DATA AT TIME OF LOADING", image.shape)
    img_affine = nifti_img.affine
    #voxel_coords =  # Homogeneous coordinates
    #img_world_coords = np.dot(img_affine, 

    # Do skull stripping here. 
    # The annotated segmentation assigns skull also as 0 label which can be confusing for the model
    # Solution is to do skull strip and then do the post processing and then pass to the model.

    # Min-max normalization to ensure values are between 0 and 1
    for z in range(image.shape[2]):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):

                if image[x,y,z]>80: image[x,y,z]=80
                if image[x,y,z]<0: image[x,y,z]=-0
        
    # Normalize.
    # image+=100
    image=image/80
    
    return image, img_affine


# This ensures that the main function is called when the script is run directly
if __name__ == "__main__":
    dir = "/data/home/umang/Vader_data/data/CTScans/Scans_org/Synthrad_CT"
    preprocess_CT_only_scans(dir)