#!/bin/bash
export CUDA_VISIBLE_DEVICES="7" 
# Define the paths and parameters
TEST_DIR='/data/home/umang/Vader_data/data/CTScans/all_scans_npy_format/'
SCANS_TEST_SAVE_PATH='/data/home/umang/Vader_umang/Seg_models/MedSAM/Inference_scans/medsam_no_prompt_emeddings_finetuned_Inference_Script/'
ORG_DATA_DIR='/data/home/umang/Vader_data/data/CTScans/Scans_org/Initial_Scans'
TASK_NAME='SAM-ViT-B'
MODEL_TYPE='vit_b'
SAM_CHECKPOINT='/data/home/umang/Vader_umang/Seg_models/MedSAM/sam_vit_b_01ec64.pth'
LOAD_PRETRAIN=True
TRAINED_MODEL_PATH=''
DEVICE='cuda:0'
WORK_DIR='./work_dir'
NUM_EPOCHS=1
BATCH_SIZE=1
NUM_WORKERS=0
WEIGHT_DECAY=0.01
LR=0.0001
USE_WANDB=False
USE_AMP=False
RESUME=''
NUM_CLASSES=4
IMG_SIZE=512
INCLUDE_BG=True
DICE_PARAM=0.8
TRAIN_SPLIT_RATIO=0.75

# Run the Python script with the specified arguments
python inference.py \
    -i "$TEST_DIR" \
    --org_data_dir "$ORG_DATA_DIR" \
    --scans_test_save_path "$SCANS_TEST_SAVE_PATH" \
    -task_name "$TASK_NAME" \
    -model_type "$MODEL_TYPE" \
    -sam_checkpoint "$SAM_CHECKPOINT" \
    $(if [ "$LOAD_PRETRAIN" = true ]; then echo "--load_pretrain"; fi) \
    -trained_model_path "$TRAINED_MODEL_PATH" \
    -device "$DEVICE" \
    -work_dir "$WORK_DIR" \
    -num_epochs "$NUM_EPOCHS" \
    -batch_size "$BATCH_SIZE" \
    -num_workers "$NUM_WORKERS" \
    -weight_decay "$WEIGHT_DECAY" \
    -lr "$LR" \
    $(if [ "$USE_WANDB" = true ]; then echo "--use_wandb"; fi) \
    $(if [ "$USE_AMP" = true ]; then echo "--use_amp"; fi) \
    --resume "$RESUME" \
    --num_classes "$NUM_CLASSES" \
    --img_size "$IMG_SIZE" \
    $(if [ "$INCLUDE_BG" = true ]; then echo "--include_bg"; fi) \
    --dice_param "$DICE_PARAM" \
    --train_split_ratio "$TRAIN_SPLIT_RATIO"
