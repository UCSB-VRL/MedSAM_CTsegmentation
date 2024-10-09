import os
import random
import h5py
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

def main():
    
    low_res = [256, 256]
    output_size=[512, 512]
    
    # Get the files list from the directory
    data_files = '/data/home/umang/Vader_data/data/CTScans/Scans_org/Initial_Scans'
    gt_file_path = '/data/home/umang/Vader_data/data/CTScans/Segmentation'
    data_file_w_annot = []
    for file in os.listdir(data_files):
        if 'Final_'+file in os.listdir(gt_file_path):
            data_file_w_annot.append(file)
    print("Data file w annot", data_file_w_annot)

    #sample_list = [f for f in os.listdir(data_files) if 'Mask' not in f]
    sample_list = data_file_w_annot
    num_data = len(sample_list)

    train_ratio=0.75
    # Check if there are any relevant files
    num_data = len(sample_list)
    if num_data == 0:
        print("No relevant files, exiting")
        exit(0)

    # Shuffle the list to ensure randomness
    random.shuffle(sample_list)

    # Calculate the number of training and validation samples
    num_train = int(num_data * train_ratio)
    num_val = num_data - num_train  # Ensures that the remaining files go to validation

    # Split the list
    train_files = sample_list[:num_train]
    val_files = sample_list[num_train:]
    if num_data ==0:
        print("No relevant file, exiting")
        exit(0) 
    print( "Number of files loading is", num_data)
    print("sample list", sample_list)

    for num, sample in enumerate(train_files):
        folder = 'preprocessed_RGB/train_scans/'
        scan_save_dir = f'/data/home/umang/Vader_data/data/CTscans_npy_format/{folder}/imgs'
        seg_save_dir = f'/data/home/umang/Vader_data/data/CTscans_npy_format/{folder}/gts'
        lowres_save_dir = f'/data/home/umang/Vader_data/data/CTscans_npy_format/{folder}/lowres_Segmentation'
        affine_save_dir = f'/data/home/umang/Vader_data/data/CTscans_npy_format/{folder}/affine'

        image_path = os.path.join(data_files, train_files[num])
        gt_path = os.path.join('/data/home/umang/Vader_data/data/CTScans/Segmentation', 'Final_'+train_files[num])

        # Check the file extension to determine how to read the file
        if image_path.endswith(('.nii', '.nii.gz')) and 'Mask' not in image_path:
            image, label, affine = read_nifti(image_path, gt_path, num_class=7)

        # load each of them and get 3 stack of slices_ currently 3 cont.
        # Resize the image and label to the output size
        x, y, num_slices = image.shape
        # Compute resize factors
        x_factor = output_size[0] / x
        y_factor = output_size[1] / y
        
        if x != output_size[0] or y != output_size[1]:
            image = zoom(image, (x_factor, y_factor, 1), order=3)
            label = zoom(label, (x_factor, y_factor, 1), order=0)

        """ REPEATED NUMBER OF SLICES """
    
        x, y, num_slices = image.shape
        for i in range(0, num_slices):
            image_RGB = np.repeat(image[..., i:i + 1], 3, axis=-1)
            label_RGB = np.repeat(label[..., i:i + 1], 3, axis=-1)
            low_res_RGB = zoom(label_RGB, (low_res[0] / label_RGB.shape[0], low_res[1] / label_RGB.shape[1], 1), order=0)
            image_RGB = image_RGB.astype(np.float32).transpose(2, 0, 1)
            label_RGB = label_RGB.astype(np.float32).transpose(2, 0, 1)
            low_res_RGB = low_res_RGB.astype(np.float32).transpose(2, 0, 1)
            print("low res RGB shape", low_res_RGB.shape)
            #print("image stacks shape", image_stacks.shape)
        
            # Update the affine matrix
            resized_affine = affine
            resized_affine[0, 0] *= x_factor  # Scale x-axis
            resized_affine[1, 1] *= y_factor  # Scale y-axis

            # Create the save directory if it doesn't exist
            os.makedirs(scan_save_dir, exist_ok=True)
            os.makedirs(seg_save_dir, exist_ok=True)
            os.makedirs(lowres_save_dir, exist_ok=True)
            os.makedirs(affine_save_dir, exist_ok = True)

            # Define the save path
            scan_save_path = os.path.join(scan_save_dir, os.path.basename(image_path).replace('.nii.gz', f'_{i}.npy'))
            seg_save_path = os.path.join(seg_save_dir, os.path.basename(gt_path).replace('.nii.gz', f'_{i}.npy'))
            lowreslabel_save_path = os.path.join(lowres_save_dir, os.path.basename(gt_path).replace('.nii.gz', f'_{i}.npy'))
            affine_save_path = os.path.join(affine_save_dir, os.path.basename(image_path).replace('.nii.gz', f'_{i}.npy'))

            # Save the NumPy array to a file
            np.save(scan_save_path, image_RGB)
            print(f"Saved f'{i}' as {scan_save_path}")
            # Save the NumPy array to a file
            np.save(seg_save_path, label_RGB)
            print(f"Saved f'{i}' as {seg_save_path}")
            # Save the NumPy array to a file
            np.save(lowreslabel_save_path, low_res_RGB)
            print(f"Saved f'{i}' as {lowreslabel_save_path}")
            np.save(affine_save_path, resized_affine)
            print(f"Saved f'{i}' as {affine_save_path}")

            #print("Unique labels for this scan", np.unique(label_RGB[1,:,:]))


    for num, sample in enumerate(val_files):
        #folder = 'preprocessed_RGB_new/train_scans/'+sample.rstrip('.nii.gz')
        
        folder = 'preprocessed_RGB/val_scans/'
        scan_save_dir = f'/data/home/umang/Vader_data/data/CTscans_npy_format/{folder}/imgs'
        seg_save_dir = f'/data/home/umang/Vader_data/data/CTscans_npy_format/{folder}/gts'
        lowres_save_dir = f'/data/home/umang/Vader_data/data/CTscans_npy_format/{folder}/lowres_Segmentation'
        affine_save_dir = f'/data/home/umang/Vader_data/data/CTscans_npy_format/{folder}/affine'

        image_path = os.path.join(data_files, val_files[num])
        gt_path = os.path.join('/data/home/umang/Vader_data/data/CTScans/Segmentation', 'Final_'+val_files[num])

        # Check the file extension to determine how to read the file
        if image_path.endswith(('.nii', '.nii.gz')) and 'Mask' not in image_path:
            image, label, affine = read_nifti(image_path, gt_path, num_class = 7)

        # load each of them and get 3 stack of slices_ currently 3 cont.
        # Resize the image and label to the output size
        x, y, num_slices = image.shape
        # Compute resize factors
        x_factor = output_size[0] / x
        y_factor = output_size[1] / y
        
        if x != output_size[0] or y != output_size[1]:
            image = zoom(image, (x_factor, y_factor, 1), order=3)
            label = zoom(label, (x_factor, y_factor, 1), order=0)

        """ REPEATED NUMBER OF SLICES """
    
        x, y, num_slices = image.shape
        for i in range(0, num_slices):
            image_RGB = np.repeat(image[..., i:i + 1], 3, axis=-1)
            label_RGB = np.repeat(label[..., i:i + 1], 3, axis=-1)
            low_res_RGB = zoom(label_RGB, (low_res[0] / label_RGB.shape[0], low_res[1] / label_RGB.shape[1], 1), order=0)
            image_RGB = image_RGB.astype(np.float32).transpose(2, 0, 1)
            label_RGB = label_RGB.astype(np.float32).transpose(2, 0, 1)
            low_res_RGB = low_res_RGB.astype(np.float32).transpose(2, 0, 1)
            print("low res RGB shape", low_res_RGB.shape)
            #print("image stacks shape", image_stacks.shape)
        
            # Update the affine matrix
            resized_affine = affine
            resized_affine[0, 0] *= x_factor  # Scale x-axis
            resized_affine[1, 1] *= y_factor  # Scale y-axis

            # Create the save directory if it doesn't exist
            os.makedirs(scan_save_dir, exist_ok=True)
            os.makedirs(seg_save_dir, exist_ok=True)
            os.makedirs(lowres_save_dir, exist_ok=True)
            os.makedirs(affine_save_dir, exist_ok = True)

            # Define the save path
            scan_save_path = os.path.join(scan_save_dir, os.path.basename(image_path).replace('.nii.gz', f'_{i}.npy'))
            seg_save_path = os.path.join(seg_save_dir, os.path.basename(gt_path).replace('.nii.gz', f'_{i}.npy'))
            lowreslabel_save_path = os.path.join(lowres_save_dir, os.path.basename(gt_path).replace('.nii.gz', f'_{i}.npy'))
            affine_save_path = os.path.join(affine_save_dir, os.path.basename(image_path).replace('.nii.gz', f'_{i}.npy'))

            # Save the NumPy array to a file
            np.save(scan_save_path, image_RGB)
            print(f"Saved f'{i}' as {scan_save_path}")
            # Save the NumPy array to a file
            np.save(seg_save_path, label_RGB)
            print(f"Saved f'{i}' as {seg_save_path}")
            # Save the NumPy array to a file
            np.save(lowreslabel_save_path, low_res_RGB)
            print(f"Saved f'{i}' as {lowreslabel_save_path}")
            np.save(affine_save_path, resized_affine)
            print(f"Saved f'{i}' as {affine_save_path}")

def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    label_path = image_path.replace('/Images', '/Masks')
    label = Image.open(label_path)
    image, label = np.array(image) / 255.0, np.uint8(np.array(label) > 0)
    return image, label, affine

def read_nifti(nifti_path, gtPath, num_class=7):

    nifti_img = nib.load(nifti_path)
    image = nifti_img.get_fdata()
    #print("IMAGE SHAPE DATA AT TIME OF LOADING", image.shape)
    img_affine = nifti_img.affine
    #voxel_coords =  # Homogeneous coordinates
    #img_world_coords = np.dot(img_affine, 

    annotation = nib.load(gtPath).get_fdata()
    annot_affine = nib.load(gtPath).affine
    
    # Do skull stripping here. 
    # The annotated segmentation assigns skull also as 0 label which can be confusing for the model
    # Solution is to do skull strip and then do the post processing and then pass to the model.

    # Min-max normalization to ensure values are between 0 and 1
    for z in range(image.shape[2]):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):

                #if annotation[x,y,z]==3 or annotation[x,y,z]==6:
                #    annotation[x,y,z]=1
            
                if num_class <=5:
                    if annotation[x,y,z]==6:
                        annotation[x,y,z]=1

                    if annotation[x,y,z]==5:
                        annotation[x,y,z]=2

                if num_class == 3:
                    if annotation[x,y,z]==4:
                        annotation[x,y,z]=3

                if image[x,y,z]>80: image[x,y,z]=80
                if image[x,y,z]<0: image[x,y,z]=-0
        
    # Normalize.
    # image+=100
    image=image/80
    
    label = annotation
    # Combine image and masks to have consistent dimensions
    #image_data = np.stack([image] * 3, axis=-1)
    #label = np.stack([annotation] * 3, axis=-1)
    #print("image shape", image.shape)
    #print("label shape", label.shape)
    
    if np.allclose(np.array(img_affine), np.array(annot_affine), atol= 1e-4) is False:
        print("Affines not equal, exiting")
        print("Img affine", img_affine)
        print("annot affine", annot_affine)
        exit(0)
    return image, label, img_affine


# This ensures that the main function is called when the script is run directly
if __name__ == "__main__":
    main()