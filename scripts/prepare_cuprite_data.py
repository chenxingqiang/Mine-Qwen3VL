#!/usr/bin/env python3
"""
Main script for preparing Cuprite hyperspectral data for Qwen3-VL fine-tuning.

This script performs the complete data preparation pipeline:
1. Load hyperspectral data and ground truth
2. Remove bad bands and normalize
3. Apply band combinations to create RGB images
4. Tile images with overlap
5. Analyze mineral content for each tile
6. Generate Qwen3-VL compatible JSON dataset

Usage:
    python prepare_cuprite_data.py --input_dir /path/to/cuprite --output_dir /path/to/output
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_config, Config
from preprocessing import (
    read_envi,
    read_geotiff,
    get_wavelengths,
    apply_band_combination,
    normalize_percentile,
    remove_bad_bands,
    to_uint8,
    save_tiles,
    tile_image,
)
from annotation import (
    load_ground_truth,
    analyze_tile_minerals,
    generate_dataset,
    save_dataset,
    split_dataset,
    validate_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Cuprite hyperspectral data for Qwen3-VL fine-tuning"
    )
    
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing Cuprite data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: data/cuprite_dataset)"
    )
    
    parser.add_argument(
        "--hyperspectral_file",
        type=str,
        default=None,
        help="Hyperspectral data file name (default: auto-detect)"
    )
    
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        default=None,
        help="Ground truth file name (default: auto-detect)"
    )
    
    parser.add_argument(
        "--band_combination",
        type=str,
        default="clay_minerals",
        choices=["clay_minerals", "iron_oxide", "natural_color", "all"],
        help="Band combination to use"
    )
    
    parser.add_argument(
        "--tile_size",
        type=int,
        default=224,
        help="Tile size in pixels"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Stride for tiling (overlap = tile_size - stride)"
    )
    
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.8,
        help="Minimum valid pixel ratio for tiles"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run without saving files"
    )
    
    return parser.parse_args()


def find_data_files(input_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Auto-detect data files in input directory.
    
    Args:
        input_dir: Input directory
        
    Returns:
        Dictionary with file paths
    """
    files = {
        "hyperspectral": None,
        "header": None,
        "ground_truth": None,
    }
    
    # Look for hyperspectral data
    for pattern in ["*.img", "*.IMG", "*reflectance*", "*radiance*"]:
        matches = list(input_dir.glob(pattern))
        if matches:
            files["hyperspectral"] = matches[0]
            break
    
    # Look for header file
    if files["hyperspectral"]:
        hdr_path = files["hyperspectral"].with_suffix(".hdr")
        if hdr_path.exists():
            files["header"] = hdr_path
    
    # Look for ground truth
    for pattern in ["*mineral*", "*class*", "*label*", "*.tif", "*.TIF"]:
        matches = list(input_dir.glob(pattern))
        for match in matches:
            if match.suffix.lower() in [".tif", ".tiff", ".img", ".png"]:
                files["ground_truth"] = match
                break
        if files["ground_truth"]:
            break
    
    return files


def load_hyperspectral_data(
    file_path: Path,
    config: Config
) -> tuple:
    """
    Load and preprocess hyperspectral data.
    
    Args:
        file_path: Path to hyperspectral file
        config: Configuration object
        
    Returns:
        Tuple of (data, wavelengths)
    """
    logger.info(f"Loading hyperspectral data from {file_path}")
    
    # Detect format and load
    suffix = file_path.suffix.lower()
    if suffix in [".img", ".dat", ".raw"]:
        hs_data = read_envi(file_path)
    elif suffix in [".tif", ".tiff"]:
        hs_data = read_geotiff(file_path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    
    data = hs_data.to_hwb()  # Ensure (H, W, B) format
    wavelengths = hs_data.wavelengths
    
    if wavelengths is None:
        logger.warning("Wavelengths not found in metadata, generating default values")
        wavelengths = get_wavelengths(
            data.shape[-1],
            config.bands.wavelength_start,
            config.bands.wavelength_end
        )
    
    logger.info(f"Loaded data: shape={data.shape}, dtype={data.dtype}")
    logger.info(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    
    return data, wavelengths


def preprocess_data(
    data: np.ndarray,
    wavelengths: np.ndarray,
    config: Config
) -> tuple:
    """
    Preprocess hyperspectral data.
    
    Args:
        data: Raw hyperspectral data
        wavelengths: Wavelength array
        config: Configuration object
        
    Returns:
        Tuple of (cleaned_data, cleaned_wavelengths)
    """
    logger.info("Preprocessing hyperspectral data...")
    
    # Remove bad bands
    bad_bands = config.bands.water_absorption_bands + config.bands.noisy_bands
    bad_bands = sorted(set(bad_bands))
    
    data, wavelengths = remove_bad_bands(data, bad_bands, wavelengths)
    logger.info(f"After bad band removal: {data.shape[-1]} bands")
    
    return data, wavelengths


def create_rgb_images(
    data: np.ndarray,
    wavelengths: np.ndarray,
    config: Config,
    band_combination: str = "all"
) -> Dict[str, np.ndarray]:
    """
    Create RGB images using band combinations.
    
    Args:
        data: Preprocessed hyperspectral data
        wavelengths: Wavelength array
        config: Configuration object
        band_combination: Which combination(s) to use
        
    Returns:
        Dictionary of {combination_name: rgb_array}
    """
    logger.info("Creating RGB composites...")
    
    combinations = config.bands.band_combinations
    
    if band_combination != "all":
        combinations = {band_combination: combinations[band_combination]}
    
    rgb_images = {}
    
    for name, combo in combinations.items():
        try:
            rgb = apply_band_combination(
                data, wavelengths, combo,
                normalize=True,
                percentile_clip=(2, 98)
            )
            rgb_images[name] = rgb
            logger.info(f"Created {name} composite: shape={rgb.shape}")
        except Exception as e:
            logger.warning(f"Failed to create {name} composite: {e}")
    
    return rgb_images


def process_tiles(
    rgb_image: np.ndarray,
    ground_truth: np.ndarray,
    output_dir: Path,
    config: Config,
    tile_size: int = 224,
    stride: int = 112,
    min_valid_ratio: float = 0.8,
    dry_run: bool = False
) -> List[Dict[str, Any]]:
    """
    Tile images and analyze mineral content.
    
    Args:
        rgb_image: RGB composite image
        ground_truth: Ground truth label image
        output_dir: Output directory for tiles
        config: Configuration object
        tile_size: Tile size
        stride: Stride for tiling
        min_valid_ratio: Minimum valid pixel ratio
        dry_run: If True, don't save files
        
    Returns:
        List of tile information with mineral statistics
    """
    logger.info("Processing tiles...")
    
    # Convert to uint8
    rgb_uint8 = to_uint8(rgb_image)
    
    if dry_run:
        # Just count tiles
        tile_gen = tile_image(rgb_uint8, tile_size, stride, min_valid_ratio)
        tile_infos = []
        for tile, info in tile_gen:
            # Extract corresponding ground truth tile
            gt_tile = ground_truth[info.y_start:info.y_end, info.x_start:info.x_end]
            
            # Analyze minerals
            stats = analyze_tile_minerals(gt_tile, config.minerals)
            
            tile_infos.append({
                "filename": info.filename,
                "row": info.row,
                "col": info.col,
                "y_start": info.y_start,
                "y_end": info.y_end,
                "x_start": info.x_start,
                "x_end": info.x_end,
                "valid_ratio": info.valid_ratio,
                "stats": stats,
            })
        
        logger.info(f"Dry run: would generate {len(tile_infos)} tiles")
        return tile_infos
    
    # Save tiles
    saved_tiles = save_tiles(
        rgb_uint8,
        output_dir,
        tile_size=tile_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        save_metadata=True
    )
    
    # Analyze mineral content for each tile
    tile_infos = []
    for tile_info in saved_tiles:
        # Extract corresponding ground truth tile
        gt_tile = ground_truth[
            tile_info.y_start:tile_info.y_end,
            tile_info.x_start:tile_info.x_end
        ]
        
        # Analyze minerals
        stats = analyze_tile_minerals(gt_tile, config.minerals)
        
        tile_infos.append({
            "filename": tile_info.filename,
            "row": tile_info.row,
            "col": tile_info.col,
            "y_start": tile_info.y_start,
            "y_end": tile_info.y_end,
            "x_start": tile_info.x_start,
            "x_end": tile_info.x_end,
            "valid_ratio": tile_info.valid_ratio,
            "stats": stats,
        })
    
    return tile_infos


def main():
    """Main execution function."""
    args = parse_args()
    config = get_config()
    
    # Set random seed
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    config.split.random_seed = args.seed
    
    # Setup paths
    input_dir = args.input_dir
    output_dir = args.output_dir or config.paths.dataset_dir
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find data files
    data_files = find_data_files(input_dir)
    
    if args.hyperspectral_file:
        data_files["hyperspectral"] = input_dir / args.hyperspectral_file
    if args.ground_truth_file:
        data_files["ground_truth"] = input_dir / args.ground_truth_file
    
    logger.info(f"Data files: {data_files}")
    
    # Validate files exist
    if not data_files["hyperspectral"] or not data_files["hyperspectral"].exists():
        logger.error("Hyperspectral data file not found!")
        logger.info("Please specify with --hyperspectral_file or place in input directory")
        sys.exit(1)
    
    if not data_files["ground_truth"] or not data_files["ground_truth"].exists():
        logger.warning("Ground truth file not found! Will generate synthetic labels for testing.")
        ground_truth = None
    else:
        ground_truth, _ = load_ground_truth(data_files["ground_truth"])
    
    # Load hyperspectral data
    hs_data, wavelengths = load_hyperspectral_data(data_files["hyperspectral"], config)
    
    # Preprocess
    hs_data, wavelengths = preprocess_data(hs_data, wavelengths, config)
    
    # Create RGB images
    rgb_images = create_rgb_images(hs_data, wavelengths, config, args.band_combination)
    
    if not rgb_images:
        logger.error("Failed to create any RGB composites!")
        sys.exit(1)
    
    # Generate synthetic ground truth if not available
    if ground_truth is None:
        logger.warning("Generating synthetic ground truth for testing...")
        h, w = hs_data.shape[:2]
        ground_truth = np.random.choice(
            [0, 1, 2, 3, 4, 5],
            size=(h, w),
            p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        ).astype(np.uint8)
    
    # Process each band combination
    all_tile_infos = {}
    
    for combo_name, rgb_image in rgb_images.items():
        logger.info(f"\nProcessing {combo_name} combination...")
        
        combo_output_dir = output_dir / "images" / combo_name
        
        tile_infos = process_tiles(
            rgb_image,
            ground_truth,
            combo_output_dir,
            config,
            tile_size=args.tile_size,
            stride=args.stride,
            min_valid_ratio=args.min_valid_ratio,
            dry_run=args.dry_run
        )
        
        all_tile_infos[combo_name] = tile_infos
        logger.info(f"{combo_name}: {len(tile_infos)} tiles generated")
    
    if args.dry_run:
        logger.info("Dry run complete. No files saved.")
        return
    
    # Generate dataset for primary combination
    primary_combo = args.band_combination if args.band_combination != "all" else config.bands.primary_combination
    tile_infos = all_tile_infos[primary_combo]
    stats_list = [info["stats"] for info in tile_infos]
    
    logger.info(f"\nGenerating dataset from {primary_combo} tiles...")
    
    # Generate conversations
    dataset = generate_dataset(
        tile_infos,
        stats_list,
        image_base_path=f"images/{primary_combo}",
        task_config=config.tasks,
        balanced=True
    )
    
    # Validate
    report = validate_dataset(dataset)
    logger.info(f"Dataset validation: {report['valid_items']}/{report['total_items']} valid")
    if report["errors"]:
        logger.warning(f"Validation errors: {report['errors'][:5]}...")
    
    # Split dataset
    config.split.train_ratio = args.train_ratio
    train_dataset, val_dataset = split_dataset(dataset, config.split)
    
    # Calculate statistics
    train_positive = sum(1 for item in train_dataset if item.metadata and item.metadata.get("is_copper_alteration"))
    val_positive = sum(1 for item in val_dataset if item.metadata and item.metadata.get("is_copper_alteration"))
    
    logger.info(f"Train set: {len(train_dataset)} items ({train_positive} positive)")
    logger.info(f"Val set: {len(val_dataset)} items ({val_positive} positive)")
    
    # Save datasets
    save_dataset(train_dataset, output_dir / "train.json", include_metadata=True)
    save_dataset(val_dataset, output_dir / "val.json", include_metadata=True)
    
    # Save dataset configuration
    dataset_config = {
        "name": "cuprite_hyperspectral",
        "version": "1.0",
        "band_combination": primary_combo,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "min_valid_ratio": args.min_valid_ratio,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "total_tiles": len(tile_infos),
        "train_items": len(train_dataset),
        "val_items": len(val_dataset),
        "train_positive_ratio": train_positive / len(train_dataset) if train_dataset else 0,
        "val_positive_ratio": val_positive / len(val_dataset) if val_dataset else 0,
    }
    
    with open(output_dir / "dataset_config.json", 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    logger.info(f"\nDataset preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Train file: {output_dir / 'train.json'}")
    logger.info(f"Val file: {output_dir / 'val.json'}")


if __name__ == "__main__":
    main()

