#!/usr/bin/env python3
"""
Generate synthetic hyperspectral data for pipeline verification.

This script creates realistic-looking synthetic data that mimics
Cuprite AVIRIS data structure for testing the complete pipeline.
"""

import numpy as np
from pathlib import Path
import json
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_hyperspectral(
    height: int = 350,
    width: int = 350,
    n_bands: int = 224,
    wavelength_start: float = 400.0,
    wavelength_end: float = 2500.0,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic hyperspectral data with realistic spectral patterns.
    
    Returns:
        Tuple of (data, wavelengths)
    """
    np.random.seed(seed)
    
    wavelengths = np.linspace(wavelength_start, wavelength_end, n_bands)
    
    # Create base reflectance with spatial variation
    data = np.zeros((height, width, n_bands), dtype=np.float32)
    
    # Generate several spectral endmembers
    endmembers = {}
    
    # Background - typical rock spectrum
    bg_spectrum = 0.3 + 0.1 * np.sin(np.linspace(0, 4*np.pi, n_bands))
    endmembers['background'] = bg_spectrum
    
    # Alunite - characteristic absorption at ~1480nm and ~2170nm
    alunite = 0.4 + 0.15 * np.sin(np.linspace(0, 2*np.pi, n_bands))
    alunite[int(n_bands * 0.5):int(n_bands * 0.55)] *= 0.7  # 1480nm absorption
    alunite[int(n_bands * 0.82):int(n_bands * 0.87)] *= 0.6  # 2170nm absorption
    endmembers['alunite'] = alunite
    
    # Kaolinite - doublet at ~2160nm and ~2200nm
    kaolinite = 0.45 + 0.1 * np.cos(np.linspace(0, 3*np.pi, n_bands))
    kaolinite[int(n_bands * 0.80):int(n_bands * 0.84)] *= 0.65
    kaolinite[int(n_bands * 0.85):int(n_bands * 0.88)] *= 0.7
    endmembers['kaolinite'] = kaolinite
    
    # Muscovite - absorption at ~2200nm
    muscovite = 0.5 + 0.12 * np.sin(np.linspace(0, 2.5*np.pi, n_bands))
    muscovite[int(n_bands * 0.85):int(n_bands * 0.90)] *= 0.55
    endmembers['muscovite'] = muscovite
    
    # Chlorite - broad absorption in SWIR
    chlorite = 0.35 + 0.08 * np.cos(np.linspace(0, 4*np.pi, n_bands))
    chlorite[int(n_bands * 0.75):int(n_bands * 0.95)] *= 0.75
    endmembers['chlorite'] = chlorite
    
    # Create spatial abundance maps
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Create patchy distributions for each mineral
    abundances = {}
    
    # Background everywhere
    abundances['background'] = np.ones((height, width)) * 0.3
    
    # Alunite - upper left region
    abundances['alunite'] = 0.4 * np.exp(-((xx - 0.25)**2 + (yy - 0.25)**2) / 0.1)
    abundances['alunite'] += 0.1 * np.random.rand(height, width)
    
    # Kaolinite - center region
    abundances['kaolinite'] = 0.35 * np.exp(-((xx - 0.5)**2 + (yy - 0.5)**2) / 0.15)
    abundances['kaolinite'] += 0.1 * np.random.rand(height, width)
    
    # Muscovite - right region
    abundances['muscovite'] = 0.45 * np.exp(-((xx - 0.75)**2 + (yy - 0.4)**2) / 0.12)
    abundances['muscovite'] += 0.08 * np.random.rand(height, width)
    
    # Chlorite - lower region
    abundances['chlorite'] = 0.3 * np.exp(-((xx - 0.5)**2 + (yy - 0.8)**2) / 0.1)
    abundances['chlorite'] += 0.05 * np.random.rand(height, width)
    
    # Normalize abundances
    total = sum(abundances.values())
    for mineral in abundances:
        abundances[mineral] /= total
    
    # Generate mixed spectra
    for mineral, spectrum in endmembers.items():
        abundance = abundances[mineral][:, :, np.newaxis]
        data += abundance * spectrum[np.newaxis, np.newaxis, :]
    
    # Add noise
    noise = np.random.randn(height, width, n_bands) * 0.02
    data += noise
    
    # Clip to valid range
    data = np.clip(data, 0, 1)
    
    # Scale to typical reflectance values (0-10000)
    data *= 10000
    
    logger.info(f"Generated synthetic hyperspectral data: shape={data.shape}")
    logger.info(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    
    return data, wavelengths, abundances


def generate_ground_truth(
    abundances: dict,
    height: int,
    width: int,
    threshold: float = 0.15
) -> np.ndarray:
    """
    Generate ground truth labels from abundance maps.
    
    Mineral class mapping:
        0: Background
        1: Alunite
        2: Kaolinite
        3: Muscovite
        7: Chlorite
    """
    labels = np.zeros((height, width), dtype=np.uint8)
    
    mineral_map = {
        'alunite': 1,
        'kaolinite': 2,
        'muscovite': 3,
        'chlorite': 7,
    }
    
    # Assign class based on dominant mineral
    for mineral, class_id in mineral_map.items():
        if mineral in abundances:
            mask = abundances[mineral] > threshold
            labels[mask] = class_id
    
    # Count pixels per class
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Ground truth class distribution:")
    for cls, cnt in zip(unique, counts):
        logger.info(f"  Class {cls}: {cnt} pixels ({100*cnt/(height*width):.1f}%)")
    
    return labels


def save_synthetic_data(
    output_dir: Path,
    data: np.ndarray,
    wavelengths: np.ndarray,
    labels: np.ndarray
):
    """Save synthetic data in ENVI-like format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save hyperspectral data as numpy file (simpler for testing)
    np.save(output_dir / "synthetic_hyperspectral.npy", data)
    np.save(output_dir / "synthetic_wavelengths.npy", wavelengths)
    np.save(output_dir / "synthetic_labels.npy", labels)
    
    # Save metadata
    metadata = {
        "shape": list(data.shape),
        "wavelength_range": [float(wavelengths[0]), float(wavelengths[-1])],
        "n_bands": int(len(wavelengths)),
        "data_type": "synthetic",
        "description": "Synthetic hyperspectral data for pipeline verification"
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved synthetic data to {output_dir}")
    return {
        "hyperspectral": output_dir / "synthetic_hyperspectral.npy",
        "wavelengths": output_dir / "synthetic_wavelengths.npy",
        "labels": output_dir / "synthetic_labels.npy",
        "metadata": output_dir / "metadata.json"
    }


def main():
    """Main function to generate synthetic data."""
    config = get_config()
    
    output_dir = config.paths.raw_data_dir / "synthetic"
    
    logger.info("Generating synthetic hyperspectral data for verification...")
    
    # Generate data
    data, wavelengths, abundances = generate_synthetic_hyperspectral(
        height=350,
        width=350,
        n_bands=224,
        seed=42
    )
    
    # Generate ground truth
    labels = generate_ground_truth(abundances, 350, 350)
    
    # Save
    files = save_synthetic_data(output_dir, data, wavelengths, labels)
    
    logger.info("\n=== Synthetic Data Generation Complete ===")
    logger.info(f"Output directory: {output_dir}")
    for name, path in files.items():
        logger.info(f"  {name}: {path}")
    
    return files


if __name__ == "__main__":
    main()

