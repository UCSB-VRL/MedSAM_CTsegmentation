import nibabel as nib
import numpy as np
import os
import gzip

def load_nifti(file_path):
    """
    Load NIfTI file and return as numpy array.
    
    Args:
        file_path (str): Path to the NIfTI file.
        
    Returns:
        np.ndarray: Data from NIfTI file.
    """
    nifti_img = nib.load(file_path)  # nibabel handles both .nii and .nii.gz files
    return nifti_img.get_fdata()

def dice_score(pred, target, num_classes):
    """
    Compute Dice Score for each class.
    
    Args:
        pred (np.ndarray): Predicted segmentation array with shape (H, W, D).
        target (np.ndarray): Ground truth segmentation array with shape (H, W, D).
        num_classes (int): Number of classes in segmentation.
    
    Returns:
        dict: Dice scores for each class.
    """
    dice_scores = {}
    
    # Ensure predictions are in one-hot format
    pred_one_hot = np.zeros((num_classes, *pred.shape), dtype=np.float32)
    target_one_hot = np.zeros((num_classes, *target.shape), dtype=np.float32)

    for i in range(num_classes):
        pred_one_hot[i] = (pred == i).astype(np.float32)
        target_one_hot[i] = (target == i).astype(np.float32)
    
    # Calculate Dice score for each class
    for i in range(num_classes):
        intersection = np.sum(pred_one_hot[i] * target_one_hot[i])
        union = np.sum(pred_one_hot[i]) + np.sum(target_one_hot[i])
        if union==0: dice_scores[i]=float("nan") 
        else: 
            dice_scores[i] = (2. * intersection) / (union)
    
    return dice_scores

def average_dice_score(dice_scores):
    """
    Compute the average Dice score from class-wise Dice scores.
    
    Args:
        dice_scores (dict): Dictionary with Dice scores for each class.
    
    Returns:
        float: Average Dice score.
    """
    return np.mean(list(dice_scores.values()))

def process_scans(pred_dir, gt_dir, num_classes):
    """
    Process multiple scans and compute Dice scores.
    
    Args:
        pred_dir (str): Directory containing predicted segmentation NIfTI files.
        gt_dir (str): Directory containing ground truth NIfTI files.
        num_classes (int): Number of classes in segmentation.
        
    Returns:
        dict: Average Dice scores for each class across all scans.
    """
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
    all_dice_scores = {i: [] for i in range(num_classes)}
    
    for pred_file in pred_files:
        # Generate the corresponding ground truth file name
        gt_file = 'Final_' + pred_file.replace('Segmentation_', '')
        #gt_file = 'Final_' + pred_file.replace('.segmented1', '')
        gt_file_path = os.path.join(gt_dir, gt_file)
        
        # Construct full path for the predicted file
        pred_file_path = os.path.join(pred_dir, pred_file)
        
        if not os.path.exists(gt_file_path):
            print(f"Ground truth file {gt_file_path} not found. Skipping.")
            continue
        
        # Load NIfTI files
        pred = load_nifti(pred_file_path)
        target = load_nifti(gt_file_path)
        
        # Min-max normalization to ensure values are between 0 and 1
        for z in range(target.shape[2]):
            for x in range(target.shape[0]):
                for y in range(target.shape[1]):

                    #if target[x,y,z]==3 or target[x,y,z]==6:
                    #    target[x,y,z]=1
                
                    if num_classes <=5:
                        if target[x,y,z]==6:
                            target[x,y,z]=1

                        if target[x,y,z]==5:
                            target[x,y,z]=2

                    if num_classes == 3:
                        if target[x,y,z]==4:
                            target[x,y,z]=3

        # Ensure data is in the same shape and type
        assert pred.shape == target.shape, "Predicted and ground truth shapes do not match"
        
        dice_scores = dice_score(pred, target, num_classes)
        avg_dice = average_dice_score(dice_scores)
        
        print(f"Processing scan pair: {pred_file_path} and {gt_file_path}")
        print("Dice Scores for each class:")
        for cls, score in dice_scores.items():
            print(f"Class {cls}: {score:.4f}")
            all_dice_scores[cls].append(score)
        
        print(f"Average Dice Score for this pair: {avg_dice:.4f}")
    
    # Compute average Dice score for each class across all scans
    average_dice_scores = {cls: np.mean(scores) for cls, scores in all_dice_scores.items()}
    
    print("\nAverage Dice Scores across all scans:")
    for cls, score in average_dice_scores.items():
        print(f"Class {cls}: {score:.4f}")
    
    return average_dice_scores



# Example Usage
if __name__ == "__main__":
        
    # TEST IT ON THE TEST DATSET FOR ALL.
    #pred_dir = "/data/home/umang/Vader_umang/Seg_models/MedSAM/inference_test_set/7class_no_prompt_embed_medsam"
    pred_dir = "/data/home/umang/Vader_umang/Seg_models/MedSAM/Inference_scans_unseen/all_class_MedSAM_NO_SKULLSTRIP_no_prompt_embed/val_set/"
    gt_dir = '/data/home/umang/Vader_data/data/CTScans/Segmentation/'
    
    num_classes = 7  # Adjust according to your dataset
    average_dice_scores = process_scans(pred_dir, gt_dir, num_classes)
