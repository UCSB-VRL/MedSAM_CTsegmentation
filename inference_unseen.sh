#!/bin/bash
export CUDA_VISIBLE_DEVICES="6" 
# Define paths and parameters
#!/bin/bash

# Load configuration from JSON file
CONFIG_FILE="config.json"
CONFIG=$(python -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
for key, value in config.items():
    if isinstance(value, str):
        print(f'export {key}=\"{value}\"')
    else:
        print(f'export {key}={value}')
")

# Evaluate the configuration to set environment variables
eval "$CONFIG"

# Run the Python script with the specified arguments
python inference_unseen.py \
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
