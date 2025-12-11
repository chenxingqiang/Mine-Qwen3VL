# Mine-Qwen3VL: Hyperspectral Mineral Exploration with Qwen3-VL

A fine-tuning pipeline for adapting Qwen3-VL to hyperspectral remote sensing mineral exploration tasks, using the Cuprite AVIRIS dataset as validation.

## Project Overview

This project enables Qwen3-VL (8B) to understand hyperspectral imagery for copper mineralization detection and alteration mineral identification.

### Key Features

- **Hyperspectral to RGB Conversion**: Band selection optimized for alteration minerals
- **Multi-task Fine-tuning**: Binary classification, mineral identification, and detailed analysis
- **LoRA + Projector Training**: Memory-efficient fine-tuning strategy
- **Cuprite Dataset Integration**: Classic USGS-annotated hyperspectral benchmark

---

## Architecture

```
Hyperspectral Data (AVIRIS 224 bands)
        ↓ Band Selection & Preprocessing
Pseudo-color RGB Image (3 channels)
        ↓ Tiling (224×224)
Qwen3-VL Vision Input
        ↓ LoRA + Projector Fine-tuning
Mineralization Prediction / Mineral Identification
```

---

## Dataset: Cuprite AVIRIS

| Attribute | Value |
|-----------|-------|
| Source | AVIRIS (Airborne Imaging Spectrometer) |
| Bands | 224 (188 after water absorption removal) |
| Spectral Range | 400-2500 nm |
| Spatial Resolution | ~20 m |
| Key Minerals | Alunite, Kaolinite, Muscovite, Montmorillonite, Calcite, Chlorite |
| Ground Truth | USGS Mineral Classification Map |

### Data Sources

- USGS Spectroscopy Lab: https://crustal.usgs.gov/speclab/
- AVIRIS Data Portal: https://aviris.jpl.nasa.gov/

---

## Preprocessing Pipeline

### 1. Band Combination Strategies

| Strategy | R (nm) | G (nm) | B (nm) | Purpose |
|----------|--------|--------|--------|---------|
| **Clay Minerals** (Primary) | 2200 | 2100 | 1650 | Highlight Al-OH features |
| Iron Oxide | 860 | 660 | 480 | Highlight iron mineralization |
| Natural Color | 660 | 550 | 470 | Reference visualization |

### 2. Preprocessing Parameters

```python
PREPROCESSING_CONFIG = {
    # Water absorption bands to remove (band indices)
    "water_absorption_bands": [
        list(range(104, 113)),   # 1350-1450 nm
        list(range(148, 167)),   # 1790-1990 nm
    ],
    
    # Noisy bands to remove
    "noisy_bands": list(range(0, 3)) + list(range(220, 224)),
    
    # Normalization method
    "normalization": "min_max",
    
    # Output bit depth
    "output_dtype": "uint8",
}
```

### 3. Tiling Configuration

```python
TILE_CONFIG = {
    "tile_size": 224,           # pixels
    "stride": 112,              # 50% overlap
    "min_valid_ratio": 0.8,     # minimum valid pixel ratio
    "output_format": "PNG",
}
```

---

## Mineral Classification

### Mineral Classes

| ID | Mineral | Chinese | Copper-Related |
|----|---------|---------|----------------|
| 0 | Background | 背景 | No |
| 1 | Alunite | 明矾石 | Yes |
| 2 | Kaolinite | 高岭石 | Yes |
| 3 | Muscovite | 白云母/绢云母 | Yes |
| 4 | Montmorillonite | 蒙脱石 | Yes |
| 5 | Buddingtonite | 铵长石 | No |
| 6 | Calcite | 方解石 | No |
| 7 | Chlorite | 绿泥石 | Yes |
| 8 | Epidote | 绿帘石 | Yes |

### Alteration Zone Criteria

A tile is classified as a copper alteration zone if copper-related minerals exceed **10%** coverage.

---

## Task Design

### Task 1: Binary Classification (Mineralization Detection)

```json
{
    "from": "human",
    "value": "<image>\nDoes this hyperspectral image show copper-related alteration features?"
}
```

### Task 2: Mineral Identification (Multi-label)

```json
{
    "from": "human",
    "value": "<image>\nIdentify the main alteration minerals in this hyperspectral image."
}
```

### Task 3: Detailed Analysis (Open VQA)

```json
{
    "from": "human",
    "value": "<image>\nProvide a detailed analysis of the mineralization features and exploration significance."
}
```

---

## Data Format

### Directory Structure

```
cuprite_dataset/
├── images/
│   ├── clay/                    # Clay mineral band combination
│   │   ├── tile_0000_0000.png
│   │   └── ...
│   ├── iron/                    # Iron oxide band combination
│   └── natural/                 # Natural color
├── train.json                   # Training annotations
├── val.json                     # Validation annotations
└── metadata.json                # Dataset metadata
```

### Annotation Format (Qwen3-VL Compatible)

```json
[
    {
        "image": "images/clay/tile_0001_0002.png",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nDoes this region show copper-related alteration?"
            },
            {
                "from": "gpt",
                "value": "Yes, this area shows significant alteration features. Muscovite and Kaolinite detected, covering approximately 35%, indicating phyllic-argillic alteration with copper mineralization potential."
            }
        ]
    }
]
```

---

## Training Configuration

### Model Selection

- **Model**: `Qwen/Qwen3-VL-8B-Instruct`
- **Strategy**: LoRA + Vision Projector Fine-tuning

### Trainable Components

| Component | Status |
|-----------|--------|
| Vision Encoder (ViT) | ❄️ Frozen |
| Vision Projector (Merger) | 🔥 Trainable |
| LLM Attention (LoRA) | 🔥 Trainable |
| LLM FFN | ❄️ Frozen |

### Training Parameters

```bash
# Key parameters
--model_name_or_path Qwen/Qwen3-VL-8B-Instruct
--tune_mm_vision False
--tune_mm_mlp True
--tune_mm_llm True
--lora_enable True
--lora_r 16
--lora_alpha 32
--learning_rate 1e-5
--per_device_train_batch_size 4
--gradient_accumulation_steps 4
--max_pixels 50176
--min_pixels 784
--num_train_epochs 5
```

### Hardware Requirements

| GPU | Batch Size | Gradient Accumulation | Estimated Memory |
|-----|------------|----------------------|------------------|
| A100 40G | 4 | 4 | ~35 GB |
| RTX 4090 | 2 | 8 | ~22 GB |
| H20 | 4 | 4 | ~35 GB |

---

## Project Structure

```
Mine-Qwen3VL/
├── Qwen3-VL/                    # Official Qwen3-VL code
├── data/
│   ├── raw/                     # Raw Cuprite data
│   ├── processed/               # Preprocessed data
│   └── cuprite_dataset/         # Final dataset
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── hyperspectral_io.py  # Hyperspectral I/O
│   │   ├── band_selection.py    # Band selection & combination
│   │   ├── tiling.py            # Image tiling
│   │   ├── normalization.py     # Normalization
│   │   └── spectral_analysis.py # Mineral detection via spectral features
│   ├── annotation/
│   │   ├── __init__.py
│   │   ├── mineral_analysis.py  # Mineral annotation analysis
│   │   └── json_generator.py    # JSON format generation
│   └── config.py                # Configuration
├── scripts/
│   ├── download_cuprite_data.py       # Download Cuprite dataset
│   ├── generate_pseudo_labels.py      # Spectral-based pseudo-label generation
│   ├── convert_to_qwenvl_format.py    # Convert to Qwen3-VL format
│   ├── train_cuprite.sh               # Multi-GPU training script
│   ├── train_cuprite_single_gpu.sh    # Single-GPU training script
│   └── inference_cuprite.py           # Inference and evaluation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_annotation.py
│   └── test_dataset.py
├── target.md                    # Project goals
└── README.md                    # This file
```

---

## Development Phases

### Phase 1: Data Preprocessing ✅
- [x] Hyperspectral data loading module
- [x] Water absorption band removal
- [x] Band combination implementation
- [x] Image tiling with overlap
- [x] Normalization and export

### Phase 2: Annotation Generation ✅
- [x] USGS ground truth parsing
- [x] Mineral statistics per tile
- [x] Multi-task prompt generation
- [x] JSON format validation

### Phase 3: Training Pipeline ✅
- [x] Dataset configuration in Qwen3-VL
- [x] Training script customization
- [x] Validation evaluation metrics

### Phase 4: Verification ✅
- [x] Synthetic data generation
- [x] End-to-end pipeline testing
- [x] Output format validation

### Phase 5: Real Data Processing ✅
- [x] Download real Cuprite AVIRIS data
- [x] Spectral analysis module for mineral detection
- [x] Automatic pseudo-label generation
- [x] Training data generation (JSONL format)

### Phase 6: Training (Ready)
- [x] Configure Qwen3-VL dataset loader
- [x] Create training scripts (single/multi-GPU)
- [x] Create inference script
- [ ] Fine-tune with LoRA + Projector (requires GPU)
- [ ] Evaluation on test set

---

## Quick Start

### 1. Generate Synthetic Data for Testing

```bash
python scripts/generate_synthetic_data.py
```

### 2. Run Pipeline Verification

```bash
python scripts/verify_pipeline.py --generate
```

### 3. Download and Process Real Cuprite Data

```bash
# Download Cuprite AVIRIS data (~215 MB)
python scripts/download_cuprite_data.py

# Generate pseudo-labels via spectral analysis
python scripts/generate_pseudo_labels.py
```

### 4. Fine-tune Qwen3-VL

```bash
# Install training dependencies
pip install -r requirements_training.txt

# Single GPU training (recommended for testing)
./scripts/train_cuprite_single_gpu.sh

# Multi-GPU training with DeepSpeed
./scripts/train_cuprite.sh
```

### 5. Run Inference

```bash
# Test on a single image
python scripts/inference_cuprite.py --image data/cuprite_dataset/labeled_tiles/tile_0005.png

# Evaluate on test set
python scripts/inference_cuprite.py --test

# Interactive mode
python scripts/inference_cuprite.py
```

---

## Training Setup ✅

### Dependencies

```bash
pip install -r requirements_training.txt
```

### Dataset Configuration

The Cuprite dataset is registered in `Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py`:

```python
data_dict = {
    "cuprite_train": CUPRITE_TRAIN,  # 28 samples
    "cuprite_val": CUPRITE_VAL,      # 3 samples
    "cuprite_test": CUPRITE_TEST,    # 5 samples
}
```

### Training Commands

| Script | Description | GPU Memory |
|--------|-------------|------------|
| `train_cuprite_single_gpu.sh` | Single GPU, LoRA | ~20 GB |
| `train_cuprite.sh` | Multi-GPU, DeepSpeed ZeRO-3 | ~35 GB per GPU |

### Key Training Parameters

```bash
--tune_mm_vision False      # Freeze ViT
--tune_mm_mlp True          # Train projector
--tune_mm_llm True          # Train LLM with LoRA
--lora_r 8                  # LoRA rank
--lora_alpha 16             # LoRA alpha
--learning_rate 1e-5        # Base LR
--mm_projector_lr 1e-4      # Projector LR
--max_pixels 50176          # 224x224 images
```

---

## Real Data Processing ✅

### Pseudo-Label Generation via Spectral Analysis

Successfully processed real Cuprite AVIRIS data using spectral analysis for automatic mineral identification.

#### Spectral Features Used

| Mineral | Key Absorption (nm) | Detection Method |
|---------|---------------------|------------------|
| Kaolinite | 2160, 2200 | Doublet absorption depth |
| Alunite | 1480, 2170 | Al-OH absorption |
| Muscovite | 2200 | Strong Al-OH absorption |
| Chlorite | 2250, 2350 | Fe-OH/Mg-OH absorption |
| Iron Oxide | 860/660 ratio | NIR/Red reflectance ratio |

#### Processing Results

| Metric | Value |
|--------|-------|
| Input Data | 972×614×224 (int16) |
| After Bad Band Removal | 972×614×182 |
| Generated Tiles | 12 (224×224 pixels) |
| Training Samples | 36 (3 tasks per tile) |
| Train/Val/Test Split | 28/3/5 |

#### Mineral Classification Results

```
Iron Oxide: 99.1% (desert oxidation zone)
Muscovite:  0.9% (phyllic alteration)
Kaolinite:  <0.1% (argillic alteration)
```

#### Output Files

```
data/cuprite_dataset/
├── labeled_tiles/
│   ├── label_map.png          # Mineral classification map
│   ├── clay_index.png         # 2200nm absorption depth
│   ├── tile_0000.png ... tile_0011.png
│   └── tiles_metadata.json
├── train_real.json            # Training data
├── val_real.json              # Validation data
└── test_real.json             # Test data
```

#### Scripts

- `scripts/generate_pseudo_labels.py` - Generate pseudo-labels from spectral analysis
- `src/preprocessing/spectral_analysis.py` - Spectral feature extraction module

---

## Verification Results

Pipeline verification completed with **7/7 steps passed**:

| Step | Status | Details |
|------|--------|---------|
| Data Loading | ✅ PASS | 224 bands, 350×350 pixels |
| Preprocessing | ✅ PASS | 189 bands after removal |
| Band Combination | ✅ PASS | 3 RGB composites generated |
| Tiling | ✅ PASS | 4 tiles (224×224) |
| Annotation | ✅ PASS | 4 items, multi-task |
| Dataset Split | ✅ PASS | train=3, val=1 |
| Format Validation | ✅ PASS | Qwen3-VL compatible |

### Sample Output

```json
{
  "image": "tiles/tile_0001_0000.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n分析该区域是否具有铜矿化潜力？"
    },
    {
      "from": "gpt",
      "value": "是，该区域存在明显的蚀变特征。检测到Alunite、Kaolinite、Muscovite、Chlorite，铜矿相关矿物占比约98.4%，属于绢英岩化-泥化带，具有铜矿化潜力。"
    }
  ]
}
```

---

## Test Cases (TDD)

| ID | Test | Expected Result |
|----|------|-----------------|
| T001 | Hyperspectral data loading | Load 224 bands, correct shape |
| T002 | Water band removal | ~188 bands remaining |
| T003 | Band combination | 3-channel RGB, values 0-255 |
| T004 | Image tiling | 224×224 PNG files, expected count |
| T005 | Mineral statistics | Match USGS ground truth |
| T006 | JSON format validation | Qwen3-VL compatible format |
| T007 | Dataset loading | DataProcessor loads successfully |
| T008 | Prompt/Answer integrity | No empty values, reasonable length |

---

## References

1. Qwen3-VL Technical Report: https://arxiv.org/pdf/2511.21631
2. USGS Cuprite Dataset Documentation
3. AVIRIS Data User's Guide

---

## License

This project is for research purposes. Qwen3-VL is subject to the Qwen license.

---

## Contact

For questions about this project, please open an issue in this repository.

