import csv
import os
import time
from CSFseg import segVent
import nibabel as nib

def diceScore7slice(imgName, initial, gtPath, maxs):
    # gtPath=os.path.join(gtPath,'Final_{}.nii.gz'.format(imgName))
    # gtPath=os.path.join(gtPath,'{}ct.segmented1.nii.gz'.format(imgName.split('.')[0][: -2]))
    final = nib.load(gtPath).get_fdata()
    correct=0
    total=0
    TP=[0]*7
    FP=[0]*7
    FN=[0]*7
    
    # Dice score based on 7 slice.
    for i in range(initial.shape[0]):
        for j in range(initial.shape[1]):
            # 7 slice dice score. 
        
            for k in range(maxs-3, maxs+4, 1):
                if final[i,j,k]==0 and initial[i,j,k]==0: continue
                total+=1
                if initial[i,j,k]==final[i,j,k]:
                    TP[int(final[i,j,k])]+=1

                    correct+=1

                else:
                    FN[int(final[i,j,k])]+=1
                    FP[int(initial[i,j,k])]+=1

    return correct, total, TP, FP, FN


def imageList(dataPath):
    fileName=[]
    fileList=[]

    if os.path.isdir(dataPath):
        fileList+=[d for d in os.listdir(dataPath) if '.nii.gz' in d]
        for temp in fileList:
            fileName+=[temp.split('.nii.gz')[0]]
    else:
        raise ValueError('Invalid data path input')
    
    fileList, fileName = zip(*sorted(zip(fileList, fileName)))
    return fileList, fileName


def DiceScoreCalc(final_pred):
    fileList, fileName = imageList(os.path.join("", final_pred))

    f_part = len("Segmentation_")
    l_part = len("")

    with open('CSFmax.csv', mode='w', newline='') as csf_file, \
            open('dicescores_7slice.csv', mode='w', newline='') as dice_file:

        # Create CSV writer objects
        csf_writer = csv.writer(csf_file)
        dice_writer = csv.writer(dice_file)

        # Write headers
        csf_writer.writerow(['Filename', 'Max Position', 'Max Area', 'Total Vent Volume 7 Slice', 'Total Vent Voxels 7 Slice'])
        dice_writer.writerow(['Filename', 'Accuracy'] + [f'Class {i} Dice Score' for i in range(1, 5)])
        
        print("fileName", fileName)
        t1 = time.time()
        print("----------Calculating Dice Score----------")
        for i in range(len(fileName)):
            out_name = fileName[i]
            outputPath = os.path.join("", final_pred)
            imageName= out_name[f_part:]
            segName= "Final_"+ imageName+ ".nii.gz"
            result_name = out_name+".nii.gz"
            gtPath = os.path.join("/data/home/umang/Vader_data/data/CTScans/Segmentation", segName)

            #print("outName, segName, imageName", out_name, segName, imageName)
            print('-------------------',imageName,'----------------------')
            
            total_ventvol, total_ventvoxels, maxArea, maxPos, finalimg =segVent(imageName, outputPath, result_name)

            # Write to CSFmax.csv
            csf_writer.writerow([fileName[i], maxPos, maxArea, total_ventvol, total_ventvoxels])
            print("Writing to CSF calc file")

            """
            correct, total, TP, FP, FN = diceScore7slice(imageName, finalimg, gtPath, maxPos[2])
            print("correct, total, TP, FP, FN", correct, total, TP, FP, FN)

            # Calculate accuracy
            accuracy = correct / total * 100

            # Collect dice scores for all classes
            dice_scores = []
            for j in range(1, 5):
                if TP[j] + FP[j] + FN[j] == 0: 
                    dice_scores.append('N/A')
                else:
                    dice_score = 2 * TP[j] / (2 * TP[j] + FP[j] + FN[j])
                    dice_scores.append(dice_score)
                print(f'Dice score for class {j}: {dice_scores[-1]}')

            # Write accuracy and dice scores to dicescore_wholepipeline.csv
            dice_writer.writerow([fileName[i], accuracy] + dice_scores)
            print(f'Correct point: {correct}/{total}, Accuracy: {accuracy:.2f}%')
            """
            
    t2 = time.time()
    print("Time for dice score calculation", t2-t1)

if __name__== "__main__":
    # data_collect("")
    SAVE_PATH = '/data/home/umang/Vader_umang/Seg_models/MedSAM/Inference_scans_unseen/script_Testing'
    DiceScoreCalc(SAVE_PATH)