import os
import numpy as np
import nibabel as nib
import csv

def process_scan(seg_path):
    # Load the image
    segimg = nib.load(seg_path)
    segarray = segimg.get_fdata()
    affine = segimg.affine

    # Calculate voxel volume
    vol_per_vox = np.abs(affine[0, 0] * affine[1, 1] * affine[2, 2])
    num_class= len(np.unique(segarray))
    print("num class", num_class)
    # Calculate volumes
    # Initialize volumes for different classes
    
    ventricle=0
    white_matter=0
    subarachnoid =0
    other=0
    for class_index in range(num_class):  # Adjust the range as necessary for your data
        if class_index == 1 or class_index==6:
            ventricle += np.sum(segarray == class_index) * vol_per_vox
        elif class_index == 2 or class_index ==5:
            white_matter += np.sum(segarray == class_index) * vol_per_vox
        elif class_index == 3:
            subarachnoid += np.sum(segarray == class_index) * vol_per_vox
        else:
            other+= np.sum(segarray == class_index) * vol_per_vox
    
    return ventricle, subarachnoid, white_matter, other
    
def main():
    # Data collection
    data = []

    seg_dir = '/data/home/umang/Vader_umang/Unet_org/UNet_Outputs_new_256'
    seg_names = [f for f in os.listdir(seg_dir) if f.endswith('1.nii.gz')]
    print(seg_names)
    for seg_name in seg_names:
        seg_path = os.path.join(seg_dir, seg_name)
        print("Processing:", seg_name)
        
        ventricle, subarachnoid, white_matter, other = process_scan(seg_path)
        
        # Skip invalid scans
        if white_matter <= 5e5 or ventricle <= 2 or subarachnoid <= 2:
            print('Skipping invalid scan.')
            print("white matter", white_matter)
            print("ventricle", ventricle)
            print("subarachnoid", subarachnoid)
            continue
        
        whole_brain = ventricle + subarachnoid + white_matter + other
        imname_short = os.path.splitext(seg_name)[0]
        
        data.append([imname_short, ventricle, subarachnoid, white_matter, other, whole_brain])

    # Sort the data by the first column (scan names)
    data.sort(key=lambda x: x[0])

    # Write to CSV file
    output_file = '/data/home/umang/Vader_umang/Seg_models/MedSAM/files_store/volumes_Unet_new_256.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scan','Vent','Sub','White','Other','Whole'])
        writer.writerows(data)

if __name__ == "__main__":
    main()
