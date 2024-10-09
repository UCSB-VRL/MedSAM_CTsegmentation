#!/bin/bash

# Define paths and parameters
TEST_DIR='/data/home/umang/Vader_data/data/CTScans/test_scans_split_npy_format'
ORG_DATA_DIR='/data/home/umang/Vader_data/data/CTScans/Scans_org/Initial_Scans'
SCANS_TEST_SAVE_PATH='/data/home/umang/Vader_umang/Seg_models/MedSAM/inference_test_set/script_TESTING/'
TASK_NAME='MedSAM-ViT-B'
MODEL_TYPE='vit_b'
SAM_CHECKPOINT='/data/home/umang/Vader_umang/Seg_models/MedSAM/medsam_vit_b.pth'
LOAD_PRETRAIN=True
TRAINED_MODEL_PATH='/data/home/umang/Vader_umang/Seg_models/MedSAM/checkpoint_dir/MEDSAM_finetune_CT/MedSAM_finetune_CT-20240802-2212/MedSAM_finetune_CT_model_best.pth'
DEVICE='cuda:7'
WORK_DIR='./work_dir'
NUM_EPOCHS=1
BATCH_SIZE=1
NUM_WORKERS=0
WEIGHT_DECAY=0.01
LR=0.0001
RESUME=''
NUM_CLASSES=4
IMG_SIZE=512
DICE_PARAM=0.8
TRAIN_SPLIT_RATIO=0.75

# Run the Python script with the specified arguments
python inference_test_on_val_set.py \
    -i "$TEST_DIR" \
    --org_data_dir "$ORG_DATA_DIR" \
    --scans_test_save_path "$SCANS_TEST_SAVE_PATH" \
    -task_name "$TASK_NAME" \
    -model_type "$MODEL_TYPE" \
    -sam_checkpoint "$SAM_CHECKPOINT" \
    --load_pretrain "$LOAD_PRETRAIN" \
    -trained_model_path "$TRAINED_MODEL_PATH" \
    -device "$DEVICE" \
    -work_dir "$WORK_DIR" \
    -num_epochs "$NUM_EPOCHS" \
    -batch_size "$BATCH_SIZE" \
    -num_workers "$NUM_WORKERS" \
    -weight_decay "$WEIGHT_DECAY" \
    -lr "$LR" \
    --resume "$RESUME" \
    --device "$DEVICE" \
    --num_classes "$NUM_CLASSES" \
    --img_size "$IMG_SIZE" \
    --dice_param "$DICE_PARAM" \
    --train_split_ratio "$TRAIN_SPLIT_RATIO"
