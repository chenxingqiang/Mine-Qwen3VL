#!/bin/bash
#
# Single GPU training script for Qwen3-VL fine-tuning on Cuprite dataset
# 
# Usage:
#   ./train_cuprite_single_gpu.sh
#
# Prerequisites:
#   1. Generate pseudo-labels: python scripts/generate_pseudo_labels.py
#   2. Convert data format: python scripts/convert_to_qwenvl_format.py
#   3. Install dependencies: pip install transformers peft deepspeed flash-attn
#

set -e

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_VL_DIR="${PROJECT_ROOT}/Qwen3-VL"
FINETUNE_DIR="${QWEN_VL_DIR}/qwen-vl-finetune"
DATASET_DIR="${PROJECT_ROOT}/data/cuprite_dataset"
OUTPUT_DIR="${PROJECT_ROOT}/output/cuprite_finetune"

# Model - use Hugging Face model ID or local path
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

# Training parameters (optimized for single GPU)
LEARNING_RATE=1e-5
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
NUM_EPOCHS=3
WARMUP_RATIO=0.1

# LoRA configuration
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Image configuration (224x224 tiles)
MAX_PIXELS=50176  # 224 * 224
MIN_PIXELS=784    # 28 * 28

# =============================================================================
# Validation
# =============================================================================

echo "=============================================="
echo "Qwen3-VL Cuprite Training (Single GPU)"
echo "=============================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Dataset: ${DATASET_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL_NAME}"
echo ""

# Check dataset
if [ ! -f "${DATASET_DIR}/train_qwenvl.json" ]; then
    echo "Error: Training data not found!"
    echo "Please run:"
    echo "  python scripts/generate_pseudo_labels.py"
    echo "  python scripts/convert_to_qwenvl_format.py"
    exit 1
fi

# Count training samples
TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('${DATASET_DIR}/train_qwenvl.json'))))")
echo "Training samples: ${TRAIN_COUNT}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Training
# =============================================================================

cd "${FINETUNE_DIR}"

echo "Starting training..."
echo ""

python qwenvl/train/train_qwen.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_use cuprite_train \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --lora_enable True \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels ${MAX_PIXELS} \
    --min_pixels ${MIN_PIXELS} \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 2 \
    --learning_rate ${LEARNING_RATE} \
    --mm_projector_lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name cuprite-mineral-single \
    --report_to tensorboard

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Model saved to: ${OUTPUT_DIR}"

