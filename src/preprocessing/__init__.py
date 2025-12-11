"""
Preprocessing module for hyperspectral data.

This module provides functions for:
- Reading and writing hyperspectral data (ENVI, GeoTIFF)
- Band selection and combination
- Image normalization
- Tiling with overlap
"""

from .hyperspectral_io import (
    read_envi,
    read_geotiff,
    write_geotiff,
    get_wavelengths,
    HyperspectralData,
)

from .band_selection import (
    wavelength_to_band_index,
    select_bands,
    create_rgb_composite,
    apply_band_combination,
)

from .tiling import (
    calculate_tile_positions,
    extract_tile,
    tile_image,
    save_tiles,
)

from .normalization import (
    normalize_minmax,
    normalize_percentile,
    normalize_standard,
    to_uint8,
    remove_bad_bands,
)

__all__ = [
    # I/O
    "read_envi",
    "read_geotiff",
    "write_geotiff",
    "get_wavelengths",
    "HyperspectralData",
    # Band selection
    "wavelength_to_band_index",
    "select_bands",
    "create_rgb_composite",
    "apply_band_combination",
    # Tiling
    "calculate_tile_positions",
    "extract_tile",
    "tile_image",
    "save_tiles",
    # Normalization
    "normalize_minmax",
    "normalize_percentile",
    "normalize_standard",
    "to_uint8",
    "remove_bad_bands",
]

