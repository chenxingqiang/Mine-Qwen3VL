"""
Spectral analysis module for automatic mineral identification.

Uses spectral absorption features to identify alteration minerals:
- Kaolinite: 2160nm and 2200nm doublet absorption
- Alunite: 1480nm and 2170nm absorption
- Muscovite: 2200nm Al-OH absorption
- Chlorite: 2250nm Fe-OH absorption
- Iron oxides: 860nm/660nm ratio
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def wavelength_to_index(target_wl: float, wavelengths: np.ndarray) -> int:
    """Find closest band index to target wavelength."""
    return int(np.argmin(np.abs(wavelengths - target_wl)))


def calculate_absorption_depth(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    absorption_wl: float,
    shoulder_left_wl: float,
    shoulder_right_wl: float
) -> float:
    """
    Calculate absorption depth at a specific wavelength.
    
    Absorption depth = 1 - (R_absorption / R_continuum)
    where R_continuum is interpolated between shoulder wavelengths.
    """
    # Get band indices
    abs_idx = wavelength_to_index(absorption_wl, wavelengths)
    left_idx = wavelength_to_index(shoulder_left_wl, wavelengths)
    right_idx = wavelength_to_index(shoulder_right_wl, wavelengths)
    
    # Get reflectance values
    r_abs = spectrum[abs_idx]
    r_left = spectrum[left_idx]
    r_right = spectrum[right_idx]
    
    # Interpolate continuum
    wl_abs = wavelengths[abs_idx]
    wl_left = wavelengths[left_idx]
    wl_right = wavelengths[right_idx]
    
    # Linear interpolation
    t = (wl_abs - wl_left) / (wl_right - wl_left + 1e-10)
    r_continuum = r_left + t * (r_right - r_left)
    
    # Absorption depth
    if r_continuum > 0:
        depth = 1.0 - (r_abs / r_continuum)
    else:
        depth = 0.0
    
    return max(0.0, depth)


def detect_kaolinite(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    threshold: float = 0.03
) -> Tuple[bool, float]:
    """
    Detect Kaolinite based on 2160nm and 2200nm doublet absorption.
    
    Returns:
        (detected, confidence)
    """
    # 2160nm absorption
    depth_2160 = calculate_absorption_depth(
        spectrum, wavelengths,
        absorption_wl=2160,
        shoulder_left_wl=2100,
        shoulder_right_wl=2230
    )
    
    # 2200nm absorption
    depth_2200 = calculate_absorption_depth(
        spectrum, wavelengths,
        absorption_wl=2200,
        shoulder_left_wl=2100,
        shoulder_right_wl=2260
    )
    
    # Kaolinite has characteristic doublet
    has_doublet = depth_2160 > threshold and depth_2200 > threshold
    confidence = (depth_2160 + depth_2200) / 2
    
    return has_doublet, confidence


def detect_alunite(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    threshold: float = 0.03
) -> Tuple[bool, float]:
    """
    Detect Alunite based on 1480nm and 2170nm absorption.
    
    Returns:
        (detected, confidence)
    """
    # 1480nm absorption (OH)
    depth_1480 = calculate_absorption_depth(
        spectrum, wavelengths,
        absorption_wl=1480,
        shoulder_left_wl=1400,
        shoulder_right_wl=1550
    )
    
    # 2170nm absorption (Al-OH)
    depth_2170 = calculate_absorption_depth(
        spectrum, wavelengths,
        absorption_wl=2170,
        shoulder_left_wl=2100,
        shoulder_right_wl=2230
    )
    
    # Alunite needs both features
    detected = depth_1480 > threshold * 0.5 and depth_2170 > threshold
    confidence = (depth_1480 + depth_2170) / 2
    
    return detected, confidence


def detect_muscovite(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    threshold: float = 0.04
) -> Tuple[bool, float]:
    """
    Detect Muscovite/Sericite based on strong 2200nm Al-OH absorption.
    
    Returns:
        (detected, confidence)
    """
    # Strong 2200nm absorption
    depth_2200 = calculate_absorption_depth(
        spectrum, wavelengths,
        absorption_wl=2200,
        shoulder_left_wl=2100,
        shoulder_right_wl=2280
    )
    
    # 2350nm absorption (weaker)
    depth_2350 = calculate_absorption_depth(
        spectrum, wavelengths,
        absorption_wl=2350,
        shoulder_left_wl=2280,
        shoulder_right_wl=2400
    )
    
    detected = depth_2200 > threshold
    confidence = depth_2200
    
    return detected, confidence


def detect_chlorite(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    threshold: float = 0.03
) -> Tuple[bool, float]:
    """
    Detect Chlorite based on 2250nm and 2350nm Fe-OH/Mg-OH absorption.
    
    Returns:
        (detected, confidence)
    """
    # 2250nm absorption
    depth_2250 = calculate_absorption_depth(
        spectrum, wavelengths,
        absorption_wl=2250,
        shoulder_left_wl=2180,
        shoulder_right_wl=2320
    )
    
    # 2350nm absorption
    depth_2350 = calculate_absorption_depth(
        spectrum, wavelengths,
        absorption_wl=2350,
        shoulder_left_wl=2280,
        shoulder_right_wl=2420
    )
    
    detected = depth_2250 > threshold and depth_2350 > threshold * 0.8
    confidence = (depth_2250 + depth_2350) / 2
    
    return detected, confidence


def detect_iron_oxide(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    threshold: float = 1.3
) -> Tuple[bool, float]:
    """
    Detect iron oxides (hematite, goethite) based on 860nm/660nm ratio.
    
    Returns:
        (detected, confidence)
    """
    idx_860 = wavelength_to_index(860, wavelengths)
    idx_660 = wavelength_to_index(660, wavelengths)
    
    r_860 = spectrum[idx_860]
    r_660 = spectrum[idx_660]
    
    if r_660 > 0:
        ratio = r_860 / r_660
    else:
        ratio = 1.0
    
    detected = ratio > threshold
    confidence = min(1.0, (ratio - 1.0) / 0.5) if ratio > 1.0 else 0.0
    
    return detected, confidence


def classify_pixel(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None
) -> Tuple[int, str, float]:
    """
    Classify a single pixel based on spectral features.
    
    Returns:
        (class_id, class_name, confidence)
    
    Class IDs:
        0: Background
        1: Alunite
        2: Kaolinite
        3: Muscovite
        7: Chlorite
        8: Iron Oxide
    """
    if thresholds is None:
        thresholds = {
            'kaolinite': 0.02,
            'alunite': 0.02,
            'muscovite': 0.03,
            'chlorite': 0.02,
            'iron_oxide': 1.2
        }
    
    results = {}
    
    # Check each mineral
    detected, conf = detect_kaolinite(spectrum, wavelengths, thresholds['kaolinite'])
    if detected:
        results['Kaolinite'] = (2, conf)
    
    detected, conf = detect_alunite(spectrum, wavelengths, thresholds['alunite'])
    if detected:
        results['Alunite'] = (1, conf)
    
    detected, conf = detect_muscovite(spectrum, wavelengths, thresholds['muscovite'])
    if detected:
        results['Muscovite'] = (3, conf)
    
    detected, conf = detect_chlorite(spectrum, wavelengths, thresholds['chlorite'])
    if detected:
        results['Chlorite'] = (7, conf)
    
    detected, conf = detect_iron_oxide(spectrum, wavelengths, thresholds['iron_oxide'])
    if detected:
        results['Iron Oxide'] = (8, conf)
    
    # Return highest confidence detection
    if results:
        best_mineral = max(results.items(), key=lambda x: x[1][1])
        return best_mineral[1][0], best_mineral[0], best_mineral[1][1]
    
    return 0, 'Background', 0.0


def generate_mineral_map(
    data: np.ndarray,
    wavelengths: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate mineral classification map from hyperspectral data.
    
    Args:
        data: Hyperspectral data (H, W, B)
        wavelengths: Wavelength array in nm
        thresholds: Detection thresholds for each mineral
        
    Returns:
        Tuple of (label_map, confidence_maps)
    """
    h, w = data.shape[:2]
    
    label_map = np.zeros((h, w), dtype=np.uint8)
    confidence_map = np.zeros((h, w), dtype=np.float32)
    
    mineral_counts = {}
    
    logger.info(f"Generating mineral map for {h}x{w} image...")
    
    for i in range(h):
        if i % 100 == 0:
            logger.info(f"  Processing row {i}/{h}...")
        for j in range(w):
            spectrum = data[i, j, :].astype(np.float32)
            
            # Skip if all zeros or invalid
            if np.all(spectrum <= 0) or np.any(np.isnan(spectrum)):
                continue
            
            class_id, class_name, conf = classify_pixel(spectrum, wavelengths, thresholds)
            label_map[i, j] = class_id
            confidence_map[i, j] = conf
            
            if class_name not in mineral_counts:
                mineral_counts[class_name] = 0
            mineral_counts[class_name] += 1
    
    # Log statistics
    total_pixels = h * w
    logger.info("Mineral classification results:")
    for mineral, count in sorted(mineral_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_pixels
        logger.info(f"  {mineral}: {count} pixels ({pct:.1f}%)")
    
    return label_map, confidence_map


def generate_spectral_indices(
    data: np.ndarray,
    wavelengths: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Generate spectral indices for mineral mapping.
    
    Returns dictionary of index images.
    """
    h, w = data.shape[:2]
    indices = {}
    
    # Clay mineral index (2200nm absorption depth)
    logger.info("Computing clay mineral index...")
    clay_index = np.zeros((h, w), dtype=np.float32)
    
    idx_2100 = wavelength_to_index(2100, wavelengths)
    idx_2200 = wavelength_to_index(2200, wavelengths)
    idx_2260 = wavelength_to_index(2260, wavelengths)
    
    r_2100 = data[:, :, idx_2100].astype(np.float32)
    r_2200 = data[:, :, idx_2200].astype(np.float32)
    r_2260 = data[:, :, idx_2260].astype(np.float32)
    
    # Continuum
    continuum = (r_2100 + r_2260) / 2
    
    # Absorption depth
    with np.errstate(divide='ignore', invalid='ignore'):
        clay_index = 1.0 - (r_2200 / continuum)
        clay_index[~np.isfinite(clay_index)] = 0
        clay_index = np.clip(clay_index, 0, 1)
    
    indices['clay'] = clay_index
    
    # Iron ratio
    logger.info("Computing iron oxide ratio...")
    idx_860 = wavelength_to_index(860, wavelengths)
    idx_660 = wavelength_to_index(660, wavelengths)
    
    r_860 = data[:, :, idx_860].astype(np.float32)
    r_660 = data[:, :, idx_660].astype(np.float32)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        iron_ratio = r_860 / r_660
        iron_ratio[~np.isfinite(iron_ratio)] = 1.0
        iron_ratio = np.clip(iron_ratio, 0, 3)
    
    indices['iron'] = iron_ratio
    
    # Alteration index (combined)
    alteration = clay_index * 0.7 + (iron_ratio - 1.0).clip(0, 1) * 0.3
    indices['alteration'] = alteration
    
    return indices


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic spectrum
    wavelengths = np.linspace(400, 2500, 224)
    
    # Create synthetic muscovite spectrum
    spectrum = np.ones(224) * 0.4
    # Add 2200nm absorption
    idx_2200 = wavelength_to_index(2200, wavelengths)
    spectrum[idx_2200-5:idx_2200+5] *= 0.85
    
    class_id, class_name, conf = classify_pixel(spectrum, wavelengths)
    print(f"Classification: {class_name} (class {class_id}), confidence: {conf:.3f}")




