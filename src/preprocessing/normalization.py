"""
Normalization module for hyperspectral data preprocessing.

Provides functions for:
- Min-max normalization
- Percentile-based normalization
- Standard normalization (z-score)
- Bad band removal
- Data type conversion
"""

import numpy as np
from typing import List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def remove_bad_bands(
    data: np.ndarray,
    bad_band_indices: List[int],
    wavelengths: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove bad bands (water absorption, noisy bands) from hyperspectral data.
    
    Args:
        data: Hyperspectral data, shape (H, W, B)
        bad_band_indices: List of band indices to remove
        wavelengths: Optional wavelength array
        
    Returns:
        Tuple of (cleaned_data, cleaned_wavelengths)
    """
    n_bands = data.shape[-1]
    
    # Create mask of bands to keep
    keep_mask = np.ones(n_bands, dtype=bool)
    for idx in bad_band_indices:
        if 0 <= idx < n_bands:
            keep_mask[idx] = False
    
    # Filter data
    cleaned_data = data[:, :, keep_mask]
    
    # Filter wavelengths if provided
    cleaned_wavelengths = None
    if wavelengths is not None:
        cleaned_wavelengths = wavelengths[keep_mask]
    
    removed_count = n_bands - np.sum(keep_mask)
    logger.info(f"Removed {removed_count} bad bands, {cleaned_data.shape[-1]} bands remaining")
    
    return cleaned_data, cleaned_wavelengths


def normalize_minmax(
    data: np.ndarray,
    per_band: bool = True,
    clip_range: Optional[Tuple[float, float]] = None,
    nodata_value: Optional[float] = None
) -> np.ndarray:
    """
    Apply min-max normalization to scale data to [0, 1].
    
    Args:
        data: Input data array
        per_band: If True, normalize each band independently
        clip_range: Optional (min, max) to clip output
        nodata_value: Value to treat as no-data (will be excluded from statistics)
        
    Returns:
        Normalized data in range [0, 1]
    """
    data = data.astype(np.float32)
    
    # Create mask for valid data
    valid_mask = ~np.isnan(data) & ~np.isinf(data)
    if nodata_value is not None:
        valid_mask &= (data != nodata_value)
    
    if per_band and len(data.shape) == 3:
        # Normalize each band independently
        result = np.zeros_like(data)
        for i in range(data.shape[-1]):
            band = data[:, :, i]
            band_mask = valid_mask[:, :, i]
            
            if np.any(band_mask):
                vmin = np.min(band[band_mask])
                vmax = np.max(band[band_mask])
                
                if vmax > vmin:
                    result[:, :, i] = (band - vmin) / (vmax - vmin)
                else:
                    result[:, :, i] = 0.0
            
            # Preserve nodata
            if nodata_value is not None:
                result[:, :, i][~band_mask] = 0.0
    else:
        # Global normalization
        if np.any(valid_mask):
            vmin = np.min(data[valid_mask])
            vmax = np.max(data[valid_mask])
            
            if vmax > vmin:
                result = (data - vmin) / (vmax - vmin)
            else:
                result = np.zeros_like(data)
        else:
            result = np.zeros_like(data)
    
    # Apply clipping
    if clip_range is not None:
        result = np.clip(result, clip_range[0], clip_range[1])
    
    return result


def normalize_percentile(
    data: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    per_band: bool = True,
    nodata_value: Optional[float] = None
) -> np.ndarray:
    """
    Apply percentile-based normalization for robust scaling.
    
    Args:
        data: Input data array
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping
        per_band: If True, normalize each band independently
        nodata_value: Value to treat as no-data
        
    Returns:
        Normalized data in range [0, 1]
    """
    data = data.astype(np.float32)
    
    # Create mask for valid data
    valid_mask = ~np.isnan(data) & ~np.isinf(data)
    if nodata_value is not None:
        valid_mask &= (data != nodata_value)
    
    if per_band and len(data.shape) == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[-1]):
            band = data[:, :, i]
            band_mask = valid_mask[:, :, i]
            
            if np.any(band_mask):
                p_low = np.percentile(band[band_mask], percentile_low)
                p_high = np.percentile(band[band_mask], percentile_high)
                
                if p_high > p_low:
                    normalized = (band - p_low) / (p_high - p_low)
                    result[:, :, i] = np.clip(normalized, 0, 1)
                else:
                    result[:, :, i] = 0.0
    else:
        if np.any(valid_mask):
            p_low = np.percentile(data[valid_mask], percentile_low)
            p_high = np.percentile(data[valid_mask], percentile_high)
            
            if p_high > p_low:
                result = (data - p_low) / (p_high - p_low)
                result = np.clip(result, 0, 1)
            else:
                result = np.zeros_like(data)
        else:
            result = np.zeros_like(data)
    
    return result


def normalize_standard(
    data: np.ndarray,
    per_band: bool = True,
    nodata_value: Optional[float] = None
) -> np.ndarray:
    """
    Apply standard (z-score) normalization.
    
    Args:
        data: Input data array
        per_band: If True, normalize each band independently
        nodata_value: Value to treat as no-data
        
    Returns:
        Standardized data (mean=0, std=1)
    """
    data = data.astype(np.float32)
    
    # Create mask for valid data
    valid_mask = ~np.isnan(data) & ~np.isinf(data)
    if nodata_value is not None:
        valid_mask &= (data != nodata_value)
    
    if per_band and len(data.shape) == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[-1]):
            band = data[:, :, i]
            band_mask = valid_mask[:, :, i]
            
            if np.any(band_mask):
                mean = np.mean(band[band_mask])
                std = np.std(band[band_mask])
                
                if std > 0:
                    result[:, :, i] = (band - mean) / std
                else:
                    result[:, :, i] = 0.0
    else:
        if np.any(valid_mask):
            mean = np.mean(data[valid_mask])
            std = np.std(data[valid_mask])
            
            if std > 0:
                result = (data - mean) / std
            else:
                result = np.zeros_like(data)
        else:
            result = np.zeros_like(data)
    
    return result


def to_uint8(
    data: np.ndarray,
    input_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Convert normalized data to uint8 (0-255).
    
    Args:
        data: Normalized data array
        input_range: Expected input value range
        
    Returns:
        uint8 data array
    """
    vmin, vmax = input_range
    
    # Scale to 0-255
    if vmax > vmin:
        scaled = (data - vmin) / (vmax - vmin) * 255.0
    else:
        scaled = np.zeros_like(data)
    
    # Clip and convert
    scaled = np.clip(scaled, 0, 255)
    
    return scaled.astype(np.uint8)


def apply_nodata_mask(
    data: np.ndarray,
    nodata_value: float,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Replace nodata values with a fill value.
    
    Args:
        data: Input data array
        nodata_value: Value to identify as nodata
        fill_value: Value to replace nodata with
        
    Returns:
        Data with nodata values replaced
    """
    result = data.copy()
    result[data == nodata_value] = fill_value
    return result


def enhance_contrast(
    data: np.ndarray,
    method: str = "linear",
    gamma: float = 1.0
) -> np.ndarray:
    """
    Apply contrast enhancement.
    
    Args:
        data: Normalized data (0-1 range)
        method: Enhancement method ("linear", "gamma", "sigmoid")
        gamma: Gamma value for gamma correction
        
    Returns:
        Contrast-enhanced data
    """
    data = np.clip(data, 0, 1)
    
    if method == "linear":
        return data
    
    elif method == "gamma":
        return np.power(data, gamma)
    
    elif method == "sigmoid":
        # Sigmoid stretch
        mid = 0.5
        gain = 5.0
        return 1.0 / (1.0 + np.exp(-gain * (data - mid)))
    
    else:
        logger.warning(f"Unknown enhancement method: {method}, using linear")
        return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    print("\n=== Testing Normalization ===")
    test_data = np.random.rand(100, 100, 224).astype(np.float32) * 10000  # Simulate reflectance values
    
    # Add some nodata
    test_data[0:10, 0:10, :] = -9999
    
    print(f"Input range: [{test_data.min():.2f}, {test_data.max():.2f}]")
    
    # Test min-max normalization
    normalized = normalize_minmax(test_data, nodata_value=-9999)
    print(f"Min-max normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Test percentile normalization
    percentile_norm = normalize_percentile(test_data, nodata_value=-9999)
    print(f"Percentile normalized range: [{percentile_norm.min():.4f}, {percentile_norm.max():.4f}]")
    
    # Test uint8 conversion
    uint8_data = to_uint8(normalized)
    print(f"uint8 range: [{uint8_data.min()}, {uint8_data.max()}]")
    
    # Test bad band removal
    print("\n=== Testing Bad Band Removal ===")
    bad_bands = list(range(0, 5)) + list(range(100, 110))
    cleaned, _ = remove_bad_bands(test_data, bad_bands)
    print(f"Original bands: {test_data.shape[-1]}, After removal: {cleaned.shape[-1]}")

