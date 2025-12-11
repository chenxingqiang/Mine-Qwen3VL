"""
Band selection and combination module for hyperspectral data.

Provides functions to:
- Find band indices from wavelengths
- Select specific bands
- Create RGB composites from band combinations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def wavelength_to_band_index(
    target_wavelength: float,
    wavelengths: np.ndarray,
    tolerance: float = 20.0
) -> int:
    """
    Find the band index closest to the target wavelength.
    
    Args:
        target_wavelength: Target wavelength in nm
        wavelengths: Array of wavelengths for each band
        tolerance: Maximum allowed difference in nm
        
    Returns:
        Band index (0-based)
        
    Raises:
        ValueError: If no band is within tolerance
    """
    differences = np.abs(wavelengths - target_wavelength)
    min_idx = np.argmin(differences)
    min_diff = differences[min_idx]
    
    if min_diff > tolerance:
        raise ValueError(
            f"No band within {tolerance}nm of target {target_wavelength}nm. "
            f"Closest band at {wavelengths[min_idx]:.1f}nm (diff={min_diff:.1f}nm)"
        )
    
    logger.debug(f"Mapped {target_wavelength}nm to band {min_idx} ({wavelengths[min_idx]:.1f}nm)")
    return int(min_idx)


def select_bands(
    data: np.ndarray,
    band_indices: List[int],
    axis: int = -1
) -> np.ndarray:
    """
    Select specific bands from hyperspectral data.
    
    Args:
        data: Hyperspectral data array
        band_indices: List of band indices to select
        axis: Axis along which bands are stored
        
    Returns:
        Array with selected bands only
    """
    return np.take(data, band_indices, axis=axis)


def create_rgb_composite(
    data: np.ndarray,
    r_band: int,
    g_band: int,
    b_band: int,
    normalize: bool = True,
    percentile_clip: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    Create RGB composite from three bands.
    
    Args:
        data: Hyperspectral data, shape (H, W, B)
        r_band: Band index for red channel
        g_band: Band index for green channel
        b_band: Band index for blue channel
        normalize: Whether to normalize to 0-1 range
        percentile_clip: Percentile values for clipping
        
    Returns:
        RGB image array, shape (H, W, 3), values 0-1 if normalized
    """
    # Extract bands
    r = data[:, :, r_band].astype(np.float32)
    g = data[:, :, g_band].astype(np.float32)
    b = data[:, :, b_band].astype(np.float32)
    
    # Stack to RGB
    rgb = np.stack([r, g, b], axis=-1)
    
    if normalize:
        # Normalize each channel independently
        for i in range(3):
            channel = rgb[:, :, i]
            valid_mask = ~np.isnan(channel) & ~np.isinf(channel)
            
            if np.any(valid_mask):
                p_low, p_high = np.percentile(
                    channel[valid_mask], 
                    percentile_clip
                )
                
                if p_high > p_low:
                    channel = (channel - p_low) / (p_high - p_low)
                    channel = np.clip(channel, 0, 1)
                else:
                    channel = np.zeros_like(channel)
                
                rgb[:, :, i] = channel
    
    return rgb


def apply_band_combination(
    data: np.ndarray,
    wavelengths: np.ndarray,
    combination: Dict[str, float],
    normalize: bool = True,
    percentile_clip: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    Apply a predefined band combination to create RGB composite.
    
    Args:
        data: Hyperspectral data, shape (H, W, B)
        wavelengths: Array of wavelengths for each band
        combination: Dictionary with 'R', 'G', 'B' keys mapping to wavelengths
        normalize: Whether to normalize output
        percentile_clip: Percentile values for clipping
        
    Returns:
        RGB composite image, shape (H, W, 3)
    """
    # Get band indices
    r_band = wavelength_to_band_index(combination['R'], wavelengths)
    g_band = wavelength_to_band_index(combination['G'], wavelengths)
    b_band = wavelength_to_band_index(combination['B'], wavelengths)
    
    logger.info(
        f"Band combination: R={combination['R']}nm (band {r_band}), "
        f"G={combination['G']}nm (band {g_band}), "
        f"B={combination['B']}nm (band {b_band})"
    )
    
    return create_rgb_composite(
        data, r_band, g_band, b_band,
        normalize=normalize,
        percentile_clip=percentile_clip
    )


def get_band_statistics(
    data: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    nodata_value: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate statistics for each band.
    
    Args:
        data: Hyperspectral data, shape (H, W, B)
        wavelengths: Optional wavelength array
        nodata_value: Value to exclude from statistics
        
    Returns:
        Dictionary with min, max, mean, std for each band
    """
    n_bands = data.shape[-1]
    
    stats = {
        'min': np.zeros(n_bands),
        'max': np.zeros(n_bands),
        'mean': np.zeros(n_bands),
        'std': np.zeros(n_bands),
    }
    
    for i in range(n_bands):
        band_data = data[:, :, i].flatten()
        
        if nodata_value is not None:
            mask = band_data != nodata_value
            band_data = band_data[mask]
        
        # Remove nan and inf
        mask = ~np.isnan(band_data) & ~np.isinf(band_data)
        band_data = band_data[mask]
        
        if len(band_data) > 0:
            stats['min'][i] = np.min(band_data)
            stats['max'][i] = np.max(band_data)
            stats['mean'][i] = np.mean(band_data)
            stats['std'][i] = np.std(band_data)
    
    if wavelengths is not None:
        stats['wavelengths'] = wavelengths
    
    return stats


def compute_spectral_indices(
    data: np.ndarray,
    wavelengths: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute common spectral indices for mineral mapping.
    
    Args:
        data: Hyperspectral data, shape (H, W, B)
        wavelengths: Array of wavelengths
        
    Returns:
        Dictionary of spectral index images
    """
    indices = {}
    
    try:
        # Iron oxide ratio (useful for ferric minerals)
        # Ratio of ~860nm to ~660nm
        b860 = wavelength_to_band_index(860, wavelengths)
        b660 = wavelength_to_band_index(660, wavelengths)
        
        r860 = data[:, :, b860].astype(np.float32)
        r660 = data[:, :, b660].astype(np.float32)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            iron_ratio = r860 / r660
            iron_ratio[~np.isfinite(iron_ratio)] = 0
        indices['iron_ratio'] = iron_ratio
        
    except ValueError as e:
        logger.warning(f"Could not compute iron ratio: {e}")
    
    try:
        # Clay mineral index (Al-OH absorption at ~2200nm)
        # Depth relative to ~2100nm and ~2250nm
        b2100 = wavelength_to_band_index(2100, wavelengths)
        b2200 = wavelength_to_band_index(2200, wavelengths)
        b2250 = wavelength_to_band_index(2250, wavelengths)
        
        r2100 = data[:, :, b2100].astype(np.float32)
        r2200 = data[:, :, b2200].astype(np.float32)
        r2250 = data[:, :, b2250].astype(np.float32)
        
        # Continuum interpolation
        continuum = (r2100 + r2250) / 2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            clay_index = 1 - (r2200 / continuum)
            clay_index[~np.isfinite(clay_index)] = 0
        indices['clay_index'] = clay_index
        
    except ValueError as e:
        logger.warning(f"Could not compute clay index: {e}")
    
    return indices


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test with synthetic data
    wavelengths = np.linspace(400, 2500, 224)
    data = np.random.rand(100, 100, 224).astype(np.float32)
    
    # Test wavelength to band index
    print("\n=== Wavelength to Band Index ===")
    for wl in [660, 860, 2200]:
        idx = wavelength_to_band_index(wl, wavelengths)
        print(f"{wl}nm -> band {idx} ({wavelengths[idx]:.1f}nm)")
    
    # Test RGB composite
    print("\n=== RGB Composite ===")
    combination = {'R': 2200, 'G': 2100, 'B': 1650}
    rgb = apply_band_combination(data, wavelengths, combination)
    print(f"RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    
    # Test spectral indices
    print("\n=== Spectral Indices ===")
    indices = compute_spectral_indices(data, wavelengths)
    for name, img in indices.items():
        print(f"{name}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")

