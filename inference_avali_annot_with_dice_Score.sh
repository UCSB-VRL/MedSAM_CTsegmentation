#!/bin/bash

# Environment setup
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=6

# Path to the Python script
PYTHON_SCRIPT='inference_avali_annot_with_dice_Score.py'

# Arguments
TEST_DIR="/data/home/umang/Vader_data/data/CTScans/test_scans_split_npy_format/"
SCANS_TEST_SAVE_PATH="/data/home/umang/Vader_umang/Seg_models/MedSAM/Inference_scans/temp_test/"
ORG_DATA_DIR="/data/home/umang/Vader_umang/Seg_models/data/CTScans/Scans_org/Scans_org"
TASK_NAME="MedSAM-ViT-B"
MODEL_TYPE="vit_b"
SAM_CHECKPOINT="/data/home/umang/Vader_umang/Seg_models/MedSAM/medsam_vit_b.pth"
LOAD_PRETRAIN=True
TRAINED_MODEL_PATH="/data/home/umang/Vader_umang/Seg_models/MedSAM/checkpoint_dir/MEDSAM_finetune_CT/MedSAM_finetune_CT-20240802-2212/MedSAM_finetune_CT_model_best.pth"
DEVICE="cuda:7"
WORK_DIR="./work_dir"
NUM_EPOCHS=1000
BATCH_SIZE=1
NUM_WORKERS=0
WEIGHT_DECAY=0.01
LR=0.0001
USE_WANDB=False
NUM_CLASSES=4
IMG_SIZE=512
INCLUDE_BG=False
DICE_PARAM=0.8
TRAIN_SPLIT_RATIO=0.75

# Execute the Python script with arguments
python3 $PYTHON_SCRIPT \
    --test_dir $TEST_DIR \
    --scans_test_save_path $SCANS_TEST_SAVE_PATH \
    --org_data_dir $ORG_DATA_DIR \
    -task_name $TASK_NAME \
    -model_type $MODEL_TYPE \
    -sam_checkpoint $SAM_CHECKPOINT \
    --load_pretrain $LOAD_PRETRAIN \
    -trained_model_path $TRAINED_MODEL_PATH \
    -device $DEVICE \
    -work_dir $WORK_DIR \
    -num_epochs $NUM_EPOCHS \
    -batch_size $BATCH_SIZE \
    -num_workers $NUM_WORKERS \
    -weight_decay $WEIGHT_DECAY \
    -lr $LR \
    -use_wandb $USE_WANDB \
    --device $DEVICE \
    --num_classes $NUM_CLASSES \
    --img_size $IMG_SIZE \
    --include_bg $INCLUDE_BG \
    --dice_param $DICE_PARAM \
    --train_split_ratio $TRAIN_SPLIT_RATIO
