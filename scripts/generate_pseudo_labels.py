#!/usr/bin/env python3
"""
Generate pseudo-labels for Cuprite hyperspectral data using spectral analysis.

This script:
1. Loads real Cuprite AVIRIS data
2. Applies spectral analysis to identify alteration minerals
3. Generates a pseudo-label map
4. Creates training tiles with labels
5. Outputs Qwen3-VL format training data
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.hyperspectral_io import read_envi
from src.preprocessing.normalization import remove_bad_bands
from src.preprocessing.band_selection import create_rgb_composite
from src.preprocessing.tiling import tile_image, save_tiles, TileInfo
from src.preprocessing.spectral_analysis import (
    generate_mineral_map,
    generate_spectral_indices,
)
from src.config import get_config

# Load config
config = get_config()

# Mineral classes mapping
MINERAL_CLASSES = {k: v['name'] for k, v in config.minerals.mineral_classes.items()}

# Prompt templates
PROMPT_TEMPLATES = {
    'BINARY': config.tasks.binary_prompts_en[0],
    'MINERAL_IDENTIFICATION': config.tasks.mineral_prompts_en[0],
    'ANALYSIS': "Provide a detailed analysis of the mineral alteration features visible in this hyperspectral image, including mineral types, distribution, and geological significance.",
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cuprite_aviris"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cuprite_dataset" / "labeled_tiles"
TILE_SIZE = 224
TILE_OVERLAP = 32
MIN_VALID_RATIO = 0.7

# Detection thresholds (tuned for Cuprite)
DETECTION_THRESHOLDS = {
    'kaolinite': 0.015,
    'alunite': 0.015,
    'muscovite': 0.02,
    'chlorite': 0.015,
    'iron_oxide': 1.15
}


def find_cuprite_data(data_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find Cuprite ENVI data and header files.
    
    Returns:
        Tuple of (data_path, header_path)
    """
    # Look for header file
    for hdr_path in data_dir.rglob("*.hdr"):
        if "cuprite" in hdr_path.name.lower():
            # Find corresponding data file
            # Data file is the header file name without .hdr
            data_path = hdr_path.with_suffix('')
            if data_path.exists():
                return data_path, hdr_path
    return None, None


def load_cuprite_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess Cuprite data."""
    logger.info("Loading Cuprite hyperspectral data...")
    
    data_path, hdr_path = find_cuprite_data(data_dir)
    if data_path is None or hdr_path is None:
        raise FileNotFoundError(f"No ENVI data found in {data_dir}")
    
    logger.info(f"Found data: {data_path}")
    logger.info(f"Found header: {hdr_path}")
    
    # Read data
    hs_data = read_envi(data_path, hdr_path)
    data = hs_data.to_hwb()
    wavelengths = hs_data.wavelengths
    
    logger.info(f"Data shape: {data.shape}")
    
    # Convert wavelengths from micrometers to nanometers if needed
    if wavelengths.max() < 10:  # Likely in micrometers
        wavelengths = wavelengths * 1000
        logger.info("Converted wavelengths from micrometers to nanometers")
    
    logger.info(f"Wavelength range: {wavelengths.min():.0f} - {wavelengths.max():.0f} nm")
    
    # Define bad bands: water absorption and noisy bands
    # Water absorption: 1350-1450nm (bands ~104-113), 1790-1990nm (bands ~148-167)
    # Noisy: first 3 and last 4 bands
    n_bands = data.shape[-1]
    bad_band_indices = []
    
    # Add noisy edge bands
    bad_band_indices.extend(range(0, 3))  # First 3
    bad_band_indices.extend(range(n_bands - 4, n_bands))  # Last 4
    
    # Add water absorption bands by wavelength
    for i, wl in enumerate(wavelengths):
        if 1340 < wl < 1460:  # Water absorption region 1
            bad_band_indices.append(i)
        if 1780 < wl < 2000:  # Water absorption region 2
            bad_band_indices.append(i)
    
    bad_band_indices = sorted(set(bad_band_indices))
    logger.info(f"Removing {len(bad_band_indices)} bad bands")
    
    # Remove bad bands
    data_clean, wl_clean = remove_bad_bands(data, bad_band_indices, wavelengths)
    logger.info(f"After bad band removal: {data_clean.shape}")
    
    return data_clean, wl_clean


def generate_pseudo_labels(
    data: np.ndarray,
    wavelengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate pseudo-labels using spectral analysis.
    
    Returns:
        label_map: Mineral class labels (H, W)
        confidence_map: Classification confidence (H, W)
        indices: Spectral indices dict
    """
    logger.info("Generating pseudo-labels via spectral analysis...")
    
    # Generate mineral map
    label_map, confidence_map = generate_mineral_map(
        data, wavelengths, DETECTION_THRESHOLDS
    )
    
    # Generate spectral indices
    indices = generate_spectral_indices(data, wavelengths)
    
    return label_map, confidence_map, indices


def analyze_tile_content(
    label_tile: np.ndarray,
    confidence_tile: np.ndarray,
    min_confidence: float = 0.01
) -> Dict:
    """Analyze mineral content of a tile."""
    h, w = label_tile.shape
    total_pixels = h * w
    
    # Count minerals
    minerals = {}
    for class_id, class_name in MINERAL_CLASSES.items():
        mask = (label_tile == class_id) & (confidence_tile > min_confidence)
        count = np.sum(mask)
        if count > 0:
            minerals[class_name] = {
                'count': int(count),
                'percentage': float(count / total_pixels * 100),
                'avg_confidence': float(np.mean(confidence_tile[mask]))
            }
    
    # Determine dominant mineral
    if minerals:
        dominant = max(minerals.items(), key=lambda x: x[1]['count'])
        dominant_mineral = dominant[0]
        dominant_pct = dominant[1]['percentage']
    else:
        dominant_mineral = "Background"
        dominant_pct = 100.0
    
    # Determine if tile has alteration
    alteration_minerals = ['Alunite', 'Kaolinite', 'Muscovite', 'Chlorite']
    alteration_pct = sum(
        m['percentage'] for name, m in minerals.items() 
        if name in alteration_minerals
    )
    has_alteration = alteration_pct > 5  # >5% alteration minerals
    
    return {
        'minerals': minerals,
        'dominant_mineral': dominant_mineral,
        'dominant_percentage': dominant_pct,
        'has_alteration': has_alteration,
        'alteration_percentage': alteration_pct
    }


def generate_description(analysis: Dict) -> str:
    """Generate natural language description of tile."""
    if not analysis['has_alteration']:
        return "This image shows background terrain with no significant hydrothermal alteration signatures detected."
    
    minerals = analysis['minerals']
    alt_minerals = [
        (name, data) for name, data in minerals.items()
        if name not in ['Background', 'Iron Oxide']
    ]
    
    if not alt_minerals:
        if 'Iron Oxide' in minerals:
            return (
                f"This image shows weak iron oxide staining ({minerals['Iron Oxide']['percentage']:.1f}% coverage) "
                "indicating possible surface weathering or oxidation zone."
            )
        return "This image shows background terrain with minimal alteration signatures."
    
    # Sort by abundance
    alt_minerals.sort(key=lambda x: -x[1]['percentage'])
    
    # Build description
    desc_parts = []
    for mineral, data in alt_minerals[:3]:  # Top 3 minerals
        pct = data['percentage']
        conf = data['avg_confidence']
        if pct > 20:
            intensity = "extensive"
        elif pct > 10:
            intensity = "moderate"
        else:
            intensity = "minor"
        desc_parts.append(f"{intensity} {mineral} ({pct:.1f}%)")
    
    mineral_str = ", ".join(desc_parts)
    
    desc = (
        f"This image shows hydrothermal alteration with {mineral_str}. "
        f"Total alteration coverage: {analysis['alteration_percentage']:.1f}%. "
    )
    
    # Add interpretation
    if 'Alunite' in minerals and minerals['Alunite']['percentage'] > 5:
        desc += "The presence of alunite indicates advanced argillic alteration. "
    if 'Kaolinite' in minerals and minerals['Kaolinite']['percentage'] > 5:
        desc += "Kaolinite suggests argillic alteration conditions. "
    if 'Muscovite' in minerals and minerals['Muscovite']['percentage'] > 5:
        desc += "Muscovite/sericite indicates phyllic alteration, commonly associated with porphyry systems. "
    
    return desc.strip()


def create_training_entry(
    image_path: str,
    analysis: Dict,
    task_type: str = 'analysis'
) -> Dict:
    """Create a Qwen3-VL training entry."""
    if task_type == 'binary':
        # Binary classification: alteration or not
        question = PROMPT_TEMPLATES['BINARY']
        if analysis['has_alteration']:
            answer = "Yes, this image shows hydrothermal alteration signatures."
        else:
            answer = "No, this image does not show significant hydrothermal alteration."
    
    elif task_type == 'mineral':
        # Mineral identification
        question = PROMPT_TEMPLATES['MINERAL_IDENTIFICATION']
        if analysis['minerals']:
            minerals_list = [
                f"{name} ({data['percentage']:.1f}%)"
                for name, data in analysis['minerals'].items()
                if name != 'Background'
            ]
            if minerals_list:
                answer = f"Detected minerals: {', '.join(minerals_list)}."
            else:
                answer = "No alteration minerals detected in this image."
        else:
            answer = "No alteration minerals detected in this image."
    
    elif task_type == 'analysis':
        # Detailed analysis
        question = PROMPT_TEMPLATES['ANALYSIS']
        answer = generate_description(analysis)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return {
        "conversations": [
            {
                "from": "user",
                "value": f"<image>\n{question}"
            },
            {
                "from": "assistant",
                "value": answer
            }
        ],
        "images": [image_path],
        "metadata": {
            "task_type": task_type,
            "has_alteration": analysis['has_alteration'],
            "dominant_mineral": analysis['dominant_mineral'],
            "alteration_percentage": analysis['alteration_percentage']
        }
    }


def main():
    """Main pipeline."""
    logger.info("=" * 60)
    logger.info("Cuprite Pseudo-Label Generation Pipeline")
    logger.info("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data, wavelengths = load_cuprite_data(DATA_DIR)
    
    # Generate pseudo-labels
    label_map, confidence_map, indices = generate_pseudo_labels(data, wavelengths)
    
    # Save label map visualization
    logger.info("Saving label map visualization...")
    
    # Create colored label visualization
    h, w = label_map.shape
    label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color mapping
    colors = {
        0: [50, 50, 50],      # Background - dark gray
        1: [255, 0, 0],       # Alunite - red
        2: [0, 255, 0],       # Kaolinite - green
        3: [0, 0, 255],       # Muscovite - blue
        7: [0, 255, 255],     # Chlorite - cyan
        8: [255, 165, 0],     # Iron Oxide - orange
    }
    
    for class_id, color in colors.items():
        mask = label_map == class_id
        label_rgb[mask] = color
    
    label_viz = Image.fromarray(label_rgb)
    label_viz.save(OUTPUT_DIR / "label_map.png")
    logger.info(f"Saved: {OUTPUT_DIR / 'label_map.png'}")
    
    # Save clay index visualization
    clay_viz = (indices['clay'] * 255).astype(np.uint8)
    Image.fromarray(clay_viz).save(OUTPUT_DIR / "clay_index.png")
    logger.info(f"Saved: {OUTPUT_DIR / 'clay_index.png'}")
    
    # Generate RGB composite for tiles
    logger.info("Generating RGB composite...")
    
    # Clay minerals band combination: R=2200nm, G=2100nm, B=1650nm
    from src.preprocessing.spectral_analysis import wavelength_to_index
    r_idx = wavelength_to_index(2200, wavelengths)
    g_idx = wavelength_to_index(2100, wavelengths)
    b_idx = wavelength_to_index(1650, wavelengths)
    
    logger.info(f"RGB bands: R={r_idx} ({wavelengths[r_idx]:.0f}nm), "
                f"G={g_idx} ({wavelengths[g_idx]:.0f}nm), "
                f"B={b_idx} ({wavelengths[b_idx]:.0f}nm)")
    
    rgb_data = create_rgb_composite(
        data, r_idx, g_idx, b_idx,
        normalize=True, percentile_clip=(2, 98)
    )
    
    # Generate tiles
    stride = TILE_SIZE - TILE_OVERLAP
    logger.info(f"Generating tiles (size={TILE_SIZE}, stride={stride})...")
    tiles = list(tile_image(
        rgb_data, 
        tile_size=TILE_SIZE, 
        stride=stride, 
        min_valid_ratio=MIN_VALID_RATIO
    ))
    logger.info(f"Generated {len(tiles)} valid tiles")
    
    # Process each tile
    training_data = []
    tile_info_list = []
    
    for idx, (tile, info) in enumerate(tiles):
        # Extract corresponding label tile
        label_tile = label_map[info.y_start:info.y_end, info.x_start:info.x_end]
        conf_tile = confidence_map[info.y_start:info.y_end, info.x_start:info.x_end]
        
        # Analyze content
        analysis = analyze_tile_content(label_tile, conf_tile)
        
        # Save tile image
        tile_filename = f"tile_{idx:04d}.png"
        tile_path = OUTPUT_DIR / tile_filename
        
        # Convert to proper format and save
        if tile.max() <= 1.0:
            tile_uint8 = (tile * 255).astype(np.uint8)
        else:
            tile_uint8 = tile.astype(np.uint8)
        Image.fromarray(tile_uint8, mode='RGB').save(tile_path)
        
        # Relative path for training data
        rel_path = f"labeled_tiles/{tile_filename}"
        
        # Create training entries (multiple task types)
        for task_type in ['binary', 'mineral', 'analysis']:
            entry = create_training_entry(rel_path, analysis, task_type)
            training_data.append(entry)
        
        # Store tile info
        tile_info_list.append({
            'filename': tile_filename,
            'row': info.row,
            'col': info.col,
            'y_start': info.y_start,
            'x_start': info.x_start,
            'valid_ratio': info.valid_ratio,
            'analysis': analysis
        })
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(tiles)} tiles...")
    
    # Save training data
    logger.info("Saving training data...")
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(training_data)
    
    # Split dataset
    n_total = len(training_data)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_data = training_data[:n_train]
    val_data = training_data[n_train:n_train+n_val]
    test_data = training_data[n_train+n_val:]
    
    # Save splits
    dataset_dir = PROJECT_ROOT / "data" / "cuprite_dataset"
    
    with open(dataset_dir / "train_real.json", 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(dataset_dir / "val_real.json", 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(dataset_dir / "test_real.json", 'w') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Save tile metadata
    with open(OUTPUT_DIR / "tiles_metadata.json", 'w') as f:
        json.dump(tile_info_list, f, indent=2)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Total tiles: {len(tiles)}")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - {dataset_dir / 'train_real.json'}")
    logger.info(f"  - {dataset_dir / 'val_real.json'}")
    logger.info(f"  - {dataset_dir / 'test_real.json'}")
    logger.info(f"  - {OUTPUT_DIR / 'label_map.png'}")
    logger.info(f"  - {OUTPUT_DIR / 'clay_index.png'}")
    logger.info(f"  - {OUTPUT_DIR / 'tiles_metadata.json'}")
    
    # Statistics
    alteration_tiles = sum(1 for t in tile_info_list if t['analysis']['has_alteration'])
    logger.info("")
    logger.info("Statistics:")
    logger.info(f"  Tiles with alteration: {alteration_tiles}/{len(tiles)} ({100*alteration_tiles/len(tiles):.1f}%)")
    
    # Mineral distribution across all tiles
    mineral_counts = {}
    for t in tile_info_list:
        for mineral in t['analysis']['minerals']:
            if mineral not in mineral_counts:
                mineral_counts[mineral] = 0
            mineral_counts[mineral] += 1
    
    logger.info("  Tiles containing each mineral:")
    for mineral, count in sorted(mineral_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {mineral}: {count} tiles")


if __name__ == "__main__":
    main()

