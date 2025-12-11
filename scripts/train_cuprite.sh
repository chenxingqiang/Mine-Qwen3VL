#!/bin/bash
#
# Training script for Qwen3-VL fine-tuning on Cuprite hyperspectral data
#
# Usage:
#   ./train_cuprite.sh
#
# Prerequisites:
#   1. Prepare dataset: python scripts/prepare_cuprite_data.py
#   2. Ensure Qwen3-VL model is downloaded or accessible
#

set -e

# =============================================================================
# Configuration
# =============================================================================

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_VL_DIR="${PROJECT_ROOT}/Qwen3-VL"
FINETUNE_DIR="${QWEN_VL_DIR}/qwen-vl-finetune"
DATASET_DIR="${PROJECT_ROOT}/data/cuprite_dataset"

# Model configuration
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
# For local model: MODEL_NAME="/path/to/local/Qwen3-VL-8B-Instruct"

# Training configuration
OUTPUT_DIR="${PROJECT_ROOT}/output/cuprite_finetune"
RUN_NAME="cuprite-mineral-v1"

# Distributed training
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# DeepSpeed configuration
DEEPSPEED_CONFIG="${FINETUNE_DIR}/scripts/zero3.json"

# Hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
NUM_EPOCHS=5
WARMUP_RATIO=0.03

# LoRA configuration
LORA_ENABLE=True
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Image configuration
MAX_PIXELS=50176  # 224 * 224
MIN_PIXELS=784    # 28 * 28

# Module tuning
TUNE_VISION=False
TUNE_MLP=True
TUNE_LLM=True

# =============================================================================
# Setup
# =============================================================================

echo "=============================================="
echo "Qwen3-VL Fine-tuning on Cuprite Dataset"
echo "=============================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Dataset dir: ${DATASET_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${NPROC_PER_NODE}"
echo ""

# Check if dataset exists
if [ ! -f "${DATASET_DIR}/train.json" ]; then
    echo "Error: Training data not found at ${DATASET_DIR}/train.json"
    echo "Please run: python scripts/prepare_cuprite_data.py first"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Setup dataset configuration for Qwen-VL-Finetune
# We need to add our dataset to the data/__init__.py or use a custom config

# Create a temporary dataset config
DATASET_CONFIG="${OUTPUT_DIR}/dataset_config.py"
cat > "${DATASET_CONFIG}" << EOF
# Auto-generated dataset configuration for Cuprite
CUPRITE_TRAIN = {
    "annotation_path": "${DATASET_DIR}/train.json",
    "data_path": "${DATASET_DIR}",
}

CUPRITE_VAL = {
    "annotation_path": "${DATASET_DIR}/val.json",
    "data_path": "${DATASET_DIR}",
}
EOF

echo "Dataset configuration saved to ${DATASET_CONFIG}"

# =============================================================================
# Training
# =============================================================================

echo ""
echo "Starting training..."
echo ""

cd "${FINETUNE_DIR}"

# Build training arguments
ARGS="
    --deepspeed ${DEEPSPEED_CONFIG}
    --model_name_or_path ${MODEL_NAME}
    --dataset_use cuprite_train
    --data_flatten True
    --tune_mm_vision ${TUNE_VISION}
    --tune_mm_mlp ${TUNE_MLP}
    --tune_mm_llm ${TUNE_LLM}
    --bf16
    --lora_enable ${LORA_ENABLE}
    --lora_r ${LORA_R}
    --lora_alpha ${LORA_ALPHA}
    --lora_dropout ${LORA_DROPOUT}
    --output_dir ${OUTPUT_DIR}
    --num_train_epochs ${NUM_EPOCHS}
    --per_device_train_batch_size ${BATCH_SIZE}
    --per_device_eval_batch_size $((BATCH_SIZE * 2))
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS}
    --max_pixels ${MAX_PIXELS}
    --min_pixels ${MIN_PIXELS}
    --eval_strategy no
    --save_strategy steps
    --save_steps 500
    --save_total_limit 3
    --learning_rate ${LEARNING_RATE}
    --weight_decay 0.01
    --warmup_ratio ${WARMUP_RATIO}
    --max_grad_norm 1.0
    --lr_scheduler_type cosine
    --logging_steps 10
    --model_max_length 4096
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name ${RUN_NAME}
    --report_to tensorboard
"

# Note: Before running, you need to register the dataset in qwenvl/data/__init__.py
# Add the following to data_dict:
#
# CUPRITE_TRAIN = {
#     "annotation_path": "/path/to/cuprite_dataset/train.json",
#     "data_path": "/path/to/cuprite_dataset",
# }
#
# data_dict = {
#     "cuprite_train": CUPRITE_TRAIN,
#     ...
# }

echo "Training arguments:"
echo "${ARGS}"
echo ""

# Launch training
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    qwenvl/train/train_qwen.py ${ARGS}

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To use the fine-tuned model:"
echo "  from transformers import AutoModelForImageTextToText, AutoProcessor"
echo "  from peft import PeftModel"
echo ""
echo "  base_model = AutoModelForImageTextToText.from_pretrained('${MODEL_NAME}')"
echo "  model = PeftModel.from_pretrained(base_model, '${OUTPUT_DIR}')"
echo ""

