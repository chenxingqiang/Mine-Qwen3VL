"""
Hyperspectral data I/O module.

Supports reading ENVI format (.hdr/.img) and GeoTIFF format.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class HyperspectralData:
    """Container for hyperspectral image data and metadata.
    
    Data is always stored in (height, width, bands) format internally.
    Use to_bhw() to convert to (bands, height, width) if needed.
    """
    
    data: np.ndarray  # Shape: (height, width, bands) - always HWB format
    wavelengths: Optional[np.ndarray] = None  # Wavelength for each band (nm)
    nodata_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    crs: Optional[str] = None  # Coordinate reference system
    transform: Optional[Any] = None  # Affine transform
    is_hwb: bool = True  # Whether data is in (H, W, B) format
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return data shape."""
        return self.data.shape
    
    @property
    def n_bands(self) -> int:
        """Return number of bands."""
        if len(self.data.shape) == 3:
            if self.is_hwb:
                return self.data.shape[2]  # Last dimension
            else:
                return self.data.shape[0]  # First dimension
        return 1
    
    @property
    def height(self) -> int:
        """Return image height."""
        if len(self.data.shape) >= 2:
            if self.is_hwb:
                return self.data.shape[0]
            else:
                return self.data.shape[1]
        return 1
    
    @property
    def width(self) -> int:
        """Return image width."""
        if len(self.data.shape) >= 2:
            if self.is_hwb:
                return self.data.shape[1]
            else:
                return self.data.shape[2]
        return self.data.shape[0]
    
    def to_bhw(self) -> np.ndarray:
        """Convert to (bands, height, width) format."""
        if len(self.data.shape) == 3:
            if self.is_hwb:
                # Currently (H, W, B), transpose to (B, H, W)
                return np.transpose(self.data, (2, 0, 1))
            else:
                # Already (B, H, W)
                return self.data.copy()
        return self.data
    
    def to_hwb(self) -> np.ndarray:
        """Convert to (height, width, bands) format."""
        if len(self.data.shape) == 3:
            if not self.is_hwb:
                # Currently (B, H, W), transpose to (H, W, B)
                return np.transpose(self.data, (1, 2, 0))
            else:
                # Already (H, W, B)
                return self.data.copy()
        return self.data


def parse_envi_header(header_path: Path) -> Dict[str, Any]:
    """
    Parse ENVI header file (.hdr).
    
    Args:
        header_path: Path to .hdr file
        
    Returns:
        Dictionary of header parameters
    """
    metadata = {}
    
    with open(header_path, 'r') as f:
        content = f.read()
    
    # Handle multi-line values enclosed in braces
    import re
    
    # Extract key-value pairs
    lines = content.replace('\r\n', '\n').split('\n')
    current_key = None
    current_value = []
    in_braces = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue
            
        if '=' in line and not in_braces:
            if current_key:
                metadata[current_key] = ' '.join(current_value).strip()
            
            key, value = line.split('=', 1)
            current_key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            if value.startswith('{'):
                if value.endswith('}'):
                    metadata[current_key] = value[1:-1].strip()
                    current_key = None
                    current_value = []
                else:
                    in_braces = True
                    current_value = [value[1:]]
            else:
                metadata[current_key] = value
                current_key = None
                current_value = []
        elif in_braces:
            if line.endswith('}'):
                current_value.append(line[:-1])
                metadata[current_key] = ' '.join(current_value).strip()
                current_key = None
                current_value = []
                in_braces = False
            else:
                current_value.append(line)
    
    # Parse numeric values
    for key in ['samples', 'lines', 'bands', 'header_offset', 'data_type', 'byte_order']:
        if key in metadata:
            try:
                metadata[key] = int(metadata[key])
            except ValueError:
                pass
    
    return metadata


def read_envi(
    img_path: Union[str, Path],
    header_path: Optional[Union[str, Path]] = None
) -> HyperspectralData:
    """
    Read ENVI format hyperspectral image.
    
    Args:
        img_path: Path to .img file
        header_path: Path to .hdr file (optional, will try to find automatically)
        
    Returns:
        HyperspectralData object
    """
    img_path = Path(img_path)
    
    # Find header file
    if header_path is None:
        header_path = img_path.with_suffix('.hdr')
        if not header_path.exists():
            header_path = Path(str(img_path) + '.hdr')
    header_path = Path(header_path)
    
    if not header_path.exists():
        raise FileNotFoundError(f"Header file not found: {header_path}")
    
    # Parse header
    metadata = parse_envi_header(header_path)
    
    samples = metadata.get('samples', 0)  # width
    lines = metadata.get('lines', 0)       # height
    bands = metadata.get('bands', 0)
    data_type = metadata.get('data_type', 4)
    byte_order = metadata.get('byte_order', 0)
    interleave = metadata.get('interleave', 'bsq').lower()
    header_offset = metadata.get('header_offset', 0)
    
    # Map ENVI data type to numpy dtype
    dtype_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    dtype = dtype_map.get(data_type, np.float32)
    
    # Read data
    with open(img_path, 'rb') as f:
        f.seek(header_offset)
        data = np.fromfile(f, dtype=dtype)
    
    # Reshape based on interleave
    if interleave == 'bsq':  # Band Sequential
        data = data.reshape((bands, lines, samples))
        data = np.transpose(data, (1, 2, 0))  # Convert to (H, W, B)
    elif interleave == 'bil':  # Band Interleaved by Line
        data = data.reshape((lines, bands, samples))
        data = np.transpose(data, (0, 2, 1))  # Convert to (H, W, B)
    elif interleave == 'bip':  # Band Interleaved by Pixel
        data = data.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Unknown interleave format: {interleave}")
    
    # Handle byte order
    if byte_order == 1:  # Big-endian
        data = data.byteswap()
    
    # Extract wavelengths
    wavelengths = None
    if 'wavelength' in metadata:
        try:
            wl_str = metadata['wavelength']
            wavelengths = np.array([float(x.strip()) for x in wl_str.split(',')])
        except Exception as e:
            logger.warning(f"Failed to parse wavelengths: {e}")
    
    # Extract nodata value
    nodata = metadata.get('data_ignore_value')
    if nodata is not None:
        try:
            nodata = float(nodata)
        except ValueError:
            nodata = None
    
    logger.info(f"Loaded ENVI image: {data.shape}, dtype={data.dtype}")
    
    return HyperspectralData(
        data=data,
        wavelengths=wavelengths,
        nodata_value=nodata,
        metadata=metadata
    )


def read_geotiff(path: Union[str, Path]) -> HyperspectralData:
    """
    Read GeoTIFF format hyperspectral image.
    
    Requires rasterio library.
    
    Args:
        path: Path to GeoTIFF file
        
    Returns:
        HyperspectralData object
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required to read GeoTIFF files. Install with: pip install rasterio")
    
    path = Path(path)
    
    with rasterio.open(path) as src:
        data = src.read()  # Shape: (bands, height, width)
        data = np.transpose(data, (1, 2, 0))  # Convert to (H, W, B)
        
        nodata = src.nodata
        crs = str(src.crs) if src.crs else None
        transform = src.transform
        
        # Try to get wavelengths from metadata
        wavelengths = None
        if 'wavelength' in src.tags():
            try:
                wl_str = src.tags()['wavelength']
                wavelengths = np.array([float(x.strip()) for x in wl_str.split(',')])
            except Exception:
                pass
    
    logger.info(f"Loaded GeoTIFF image: {data.shape}, dtype={data.dtype}")
    
    return HyperspectralData(
        data=data,
        wavelengths=wavelengths,
        nodata_value=nodata,
        crs=crs,
        transform=transform
    )


def write_geotiff(
    data: np.ndarray,
    path: Union[str, Path],
    crs: Optional[str] = None,
    transform: Optional[Any] = None,
    nodata: Optional[float] = None,
    wavelengths: Optional[np.ndarray] = None
) -> None:
    """
    Write data to GeoTIFF format.
    
    Args:
        data: Image data, shape (H, W, B) or (H, W)
        path: Output path
        crs: Coordinate reference system
        transform: Affine transform
        nodata: No-data value
        wavelengths: Wavelength for each band
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError("rasterio is required to write GeoTIFF files.")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle 2D data
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    
    # Convert to (B, H, W) for rasterio
    data_bhw = np.transpose(data, (2, 0, 1))
    bands, height, width = data_bhw.shape
    
    # Default transform
    if transform is None:
        transform = from_bounds(0, 0, width, height, width, height)
    
    # Build metadata tags
    tags = {}
    if wavelengths is not None:
        tags['wavelength'] = ','.join(str(w) for w in wavelengths)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(data_bhw)
        if tags:
            dst.update_tags(**tags)
    
    logger.info(f"Saved GeoTIFF: {path}")


def get_wavelengths(
    n_bands: int,
    wavelength_start: float = 400.0,
    wavelength_end: float = 2500.0
) -> np.ndarray:
    """
    Generate wavelength array if not available in metadata.
    
    Args:
        n_bands: Number of bands
        wavelength_start: Starting wavelength (nm)
        wavelength_end: Ending wavelength (nm)
        
    Returns:
        Array of wavelengths
    """
    return np.linspace(wavelength_start, wavelength_end, n_bands)


if __name__ == "__main__":
    # Test with sample data
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic test data
    test_data = np.random.rand(100, 100, 224).astype(np.float32)
    wavelengths = get_wavelengths(224)
    
    hs_data = HyperspectralData(
        data=test_data,
        wavelengths=wavelengths
    )
    
    print(f"Shape: {hs_data.shape}")
    print(f"Bands: {hs_data.n_bands}")
    print(f"Height: {hs_data.height}")
    print(f"Width: {hs_data.width}")
    print(f"Wavelengths range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

