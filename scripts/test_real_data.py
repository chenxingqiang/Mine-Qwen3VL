#!/usr/bin/env python3
"""
Test script to verify our pipeline works with real Cuprite AVIRIS data.
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_config
from preprocessing import (
    read_envi,
    apply_band_combination,
    normalize_percentile,
    remove_bad_bands,
    to_uint8,
    save_tiles,
    get_wavelengths,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test real Cuprite data processing."""
    config = get_config()
    
    # Path to real Cuprite data
    data_dir = config.paths.raw_data_dir / "cuprite_aviris"
    img_path = data_dir / "cuprite.95.cal.rtgc.v"
    hdr_path = data_dir / "cuprite.95.cal.rtgc.v.hdr"
    
    if not img_path.exists():
        logger.error(f"Data file not found: {img_path}")
        logger.info("Please run: python scripts/download_cuprite_data.py")
        return 1
    
    logger.info("="*60)
    logger.info("Testing Real Cuprite AVIRIS Data Processing")
    logger.info("="*60)
    
    # Step 1: Load data
    logger.info("\n[Step 1] Loading ENVI data...")
    try:
        hs_data = read_envi(img_path, hdr_path)
        logger.info(f"  ✓ Data loaded successfully!")
        logger.info(f"    Shape: {hs_data.shape}")
        logger.info(f"    Bands: {hs_data.n_bands}")
        logger.info(f"    Height: {hs_data.height}")
        logger.info(f"    Width: {hs_data.width}")
        logger.info(f"    Data type: {hs_data.data.dtype}")
        logger.info(f"    Value range: [{hs_data.data.min()}, {hs_data.data.max()}]")
        
        if hs_data.wavelengths is not None:
            logger.info(f"    Wavelengths: {hs_data.wavelengths[0]:.3f} - {hs_data.wavelengths[-1]:.3f} µm")
    except Exception as e:
        logger.error(f"  ✗ Failed to load data: {e}")
        return 1
    
    # Convert to HWB format
    data = hs_data.to_hwb()
    
    # Convert wavelengths from micrometers to nanometers if needed
    wavelengths = hs_data.wavelengths
    if wavelengths is not None and wavelengths.max() < 10:
        # Wavelengths are in micrometers, convert to nm
        wavelengths = wavelengths * 1000
        logger.info(f"    Wavelengths (nm): {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    
    # Step 2: Remove bad bands
    logger.info("\n[Step 2] Removing bad bands...")
    try:
        bad_bands = config.bands.water_absorption_bands + config.bands.noisy_bands
        bad_bands = [b for b in sorted(set(bad_bands)) if 0 <= b < data.shape[-1]]
        
        cleaned_data, cleaned_wl = remove_bad_bands(data, bad_bands, wavelengths)
        logger.info(f"  ✓ Removed {len(bad_bands)} bad bands")
        logger.info(f"    Remaining bands: {cleaned_data.shape[-1]}")
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        # Continue without removing bands
        cleaned_data = data
        cleaned_wl = wavelengths
    
    # Step 3: Create RGB composite
    logger.info("\n[Step 3] Creating RGB composite...")
    try:
        combo = config.bands.band_combinations["clay_minerals"]
        logger.info(f"    Band combination: R={combo['R']}nm, G={combo['G']}nm, B={combo['B']}nm")
        
        rgb = apply_band_combination(
            cleaned_data, cleaned_wl, combo,
            normalize=True,
            percentile_clip=(2, 98)
        )
        logger.info(f"  ✓ RGB composite created!")
        logger.info(f"    Shape: {rgb.shape}")
        logger.info(f"    Value range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return 1
    
    # Step 4: Save sample tiles
    logger.info("\n[Step 4] Creating sample tiles...")
    try:
        output_dir = config.paths.dataset_dir / "real_cuprite_tiles"
        rgb_uint8 = to_uint8(rgb)
        
        # Use larger stride for faster testing
        tile_infos = save_tiles(
            rgb_uint8,
            output_dir,
            tile_size=224,
            stride=224,  # No overlap for quick test
            min_valid_ratio=0.5,
            save_metadata=True
        )
        logger.info(f"  ✓ Created {len(tile_infos)} tiles")
        logger.info(f"    Output: {output_dir}")
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return 1
    
    # Step 5: Generate sample image for visualization
    logger.info("\n[Step 5] Saving full-scene preview...")
    try:
        from PIL import Image
        
        # Resize for preview
        h, w = rgb.shape[:2]
        scale = min(1.0, 1000 / max(h, w))
        new_h, new_w = int(h * scale), int(w * scale)
        
        preview = to_uint8(rgb)
        img = Image.fromarray(preview, mode='RGB')
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        preview_path = output_dir / "cuprite_preview.png"
        img.save(preview_path)
        logger.info(f"  ✓ Saved preview: {preview_path}")
        logger.info(f"    Size: {new_w} x {new_h}")
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ Real Data Test Complete!")
    logger.info("="*60)
    
    # Summary
    logger.info(f"\nGenerated files:")
    logger.info(f"  - Tiles: {output_dir}")
    logger.info(f"  - Count: {len(tile_infos)} tiles")
    
    # Calculate expected tile count
    expected_h = (hs_data.height - 224) // 224 + 1
    expected_w = (hs_data.width - 224) // 224 + 1
    logger.info(f"\nNote: With 224×224 tiles, stride=224:")
    logger.info(f"  - Expected grid: ~{expected_h} × {expected_w} = ~{expected_h * expected_w} tiles")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


