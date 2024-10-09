#!/bin/bash
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="6, 7" 

# Define default values for arguments
TR_NPY_PATH="/data/home/umang/Vader_data/data/CTscans_npy_format/preprocessed_RGB/train_scans"
TASK_NAME="MEDSAM_finetune_CT_NO_SKULLSTRIP_repeated_img_embeddings_no_prompt_7classes"
MODEL_TYPE="vit_b"
CHECKPOINT="/data/home/umang/Vader_umang/Seg_models/MedSAM/medsam_vit_b.pth"
DEVICE=1
LOAD_PRETRAIN=False
PRETRAIN_MODEL_PATH=""
WORK_DIR="./work_dir"
CHECKPOINT_DIR="/data/home/umang/Vader_umang/Seg_models/MedSAM/checkpoint_dir/7class_MEDSAM_finetune_CT_repeated_img_embeddings_no_prompt"
NUM_EPOCHS=200
BATCH_SIZE=20
NUM_WORKERS=8
WEIGHT_DECAY=0.01
LR=0.0001
USE_WANDB=True
INCLUDE_BG=True
RESUME=""
NUM_CLASSES=6
IMG_SIZE=512
DICE_PARAM=0.8
TRAIN_SPLIT_RATIO=0.75

# Run the Python script with the arguments
python train_one_gpu.py \
    --tr_npy_path "$TR_NPY_PATH" \
    -task_name "$TASK_NAME" \
    -model_type "$MODEL_TYPE" \
    -checkpoint "$CHECKPOINT" \
    -device "$DEVICE" \
    --load_pretrain "$LOAD_PRETRAIN" \
    -pretrain_model_path "$PRETRAIN_MODEL_PATH" \
    -work_dir "$WORK_DIR" \
    -checkpoint_dir "$CHECKPOINT_DIR" \
    -num_epochs "$NUM_EPOCHS" \
    -batch_size "$BATCH_SIZE" \
    -num_workers "$NUM_WORKERS" \
    -weight_decay "$WEIGHT_DECAY" \
    -lr "$LR" \
    -use_wandb "$USE_WANDB" \
    --resume "$RESUME" \
    --num_classes "$NUM_CLASSES" \
    --img_size "$IMG_SIZE"\
    --dice_param "$DICE_PARAM" \
    --train_split_ratio "$TRAIN_SPLIT_RATIO"\
    --include_bg "$INCLUDE_BG"
