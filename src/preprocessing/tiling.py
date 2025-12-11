"""
Image tiling module for creating fixed-size patches from large images.

Supports:
- Sliding window tiling with configurable overlap
- Validity checking (minimum valid pixel ratio)
- Batch tile extraction and saving
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Dict, Any
from dataclasses import dataclass
import json
import logging
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Information about a single tile."""
    row: int
    col: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    valid_ratio: float
    filename: str


def calculate_tile_positions(
    image_height: int,
    image_width: int,
    tile_size: int,
    stride: int
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate tile positions for sliding window extraction.
    
    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        tile_size: Size of each tile (square)
        stride: Step size between tiles
        
    Returns:
        List of (y_start, y_end, x_start, x_end) tuples
    """
    positions = []
    
    y = 0
    while y + tile_size <= image_height:
        x = 0
        while x + tile_size <= image_width:
            positions.append((y, y + tile_size, x, x + tile_size))
            x += stride
        y += stride
    
    # Handle edge cases if image is smaller than tile
    if not positions and image_height > 0 and image_width > 0:
        # Pad to tile_size or use what we have
        positions.append((0, min(tile_size, image_height), 0, min(tile_size, image_width)))
    
    logger.info(f"Calculated {len(positions)} tile positions for {image_height}x{image_width} image")
    return positions


def calculate_valid_ratio(
    tile: np.ndarray,
    nodata_value: Optional[float] = None
) -> float:
    """
    Calculate the ratio of valid (non-nodata, non-nan) pixels in a tile.
    
    Args:
        tile: Tile data array
        nodata_value: Value representing no-data pixels
        
    Returns:
        Ratio of valid pixels (0-1)
    """
    total_pixels = tile.size
    
    if total_pixels == 0:
        return 0.0
    
    # Create validity mask
    valid_mask = ~np.isnan(tile) & ~np.isinf(tile)
    
    if nodata_value is not None:
        valid_mask &= (tile != nodata_value)
    
    valid_count = np.sum(valid_mask)
    return float(valid_count) / float(total_pixels)


def extract_tile(
    data: np.ndarray,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Extract a tile from the image, with optional padding.
    
    Args:
        data: Image data, shape (H, W) or (H, W, C)
        y_start, y_end: Row range
        x_start, x_end: Column range
        pad_value: Value to use for padding if tile extends beyond image
        
    Returns:
        Extracted tile array
    """
    h, w = data.shape[:2]
    tile_h = y_end - y_start
    tile_w = x_end - x_start
    
    # Calculate actual extraction bounds
    actual_y_start = max(0, y_start)
    actual_y_end = min(h, y_end)
    actual_x_start = max(0, x_start)
    actual_x_end = min(w, x_end)
    
    # Extract the available portion
    extracted = data[actual_y_start:actual_y_end, actual_x_start:actual_x_end]
    
    # Check if padding is needed
    needs_padding = (
        actual_y_start != y_start or
        actual_y_end != y_end or
        actual_x_start != x_start or
        actual_x_end != x_end
    )
    
    if needs_padding:
        # Create padded output
        if len(data.shape) == 3:
            tile = np.full((tile_h, tile_w, data.shape[2]), pad_value, dtype=data.dtype)
        else:
            tile = np.full((tile_h, tile_w), pad_value, dtype=data.dtype)
        
        # Calculate where to place extracted data
        paste_y = actual_y_start - y_start
        paste_x = actual_x_start - x_start
        
        tile[paste_y:paste_y + extracted.shape[0],
             paste_x:paste_x + extracted.shape[1]] = extracted
        
        return tile
    
    return extracted.copy()


def tile_image(
    data: np.ndarray,
    tile_size: int = 224,
    stride: int = 112,
    min_valid_ratio: float = 0.8,
    nodata_value: Optional[float] = None
) -> Generator[Tuple[np.ndarray, TileInfo], None, None]:
    """
    Generate tiles from an image using sliding window.
    
    Args:
        data: Image data, shape (H, W) or (H, W, C)
        tile_size: Size of each tile (square)
        stride: Step size between tiles
        min_valid_ratio: Minimum ratio of valid pixels required
        nodata_value: Value representing no-data pixels
        
    Yields:
        Tuples of (tile_data, tile_info)
    """
    h, w = data.shape[:2]
    positions = calculate_tile_positions(h, w, tile_size, stride)
    
    valid_count = 0
    total_count = len(positions)
    
    for idx, (y_start, y_end, x_start, x_end) in enumerate(positions):
        row = idx // ((w - tile_size) // stride + 1) if stride <= w - tile_size else idx
        col = idx % ((w - tile_size) // stride + 1) if stride <= w - tile_size else idx
        
        tile = extract_tile(data, y_start, y_end, x_start, x_end)
        valid_ratio = calculate_valid_ratio(tile, nodata_value)
        
        if valid_ratio >= min_valid_ratio:
            valid_count += 1
            
            tile_info = TileInfo(
                row=row,
                col=col,
                y_start=y_start,
                y_end=y_end,
                x_start=x_start,
                x_end=x_end,
                valid_ratio=valid_ratio,
                filename=f"tile_{row:04d}_{col:04d}.png"
            )
            
            yield tile, tile_info
    
    logger.info(f"Generated {valid_count}/{total_count} valid tiles (min_ratio={min_valid_ratio})")


def save_tiles(
    data: np.ndarray,
    output_dir: Path,
    tile_size: int = 224,
    stride: int = 112,
    min_valid_ratio: float = 0.8,
    nodata_value: Optional[float] = None,
    output_format: str = "PNG",
    save_metadata: bool = True
) -> List[TileInfo]:
    """
    Extract tiles from image and save to disk.
    
    Args:
        data: Image data (H, W, C), values 0-1 or 0-255
        output_dir: Directory to save tiles
        tile_size: Size of each tile
        stride: Step size between tiles
        min_valid_ratio: Minimum valid pixel ratio
        nodata_value: No-data value
        output_format: Output format ("PNG" or "JPEG")
        save_metadata: Whether to save tile metadata JSON
        
    Returns:
        List of TileInfo objects for saved tiles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_tiles = []
    
    for tile, tile_info in tile_image(data, tile_size, stride, min_valid_ratio, nodata_value):
        # Ensure proper format for saving
        if tile.max() <= 1.0:
            tile_uint8 = (tile * 255).astype(np.uint8)
        else:
            tile_uint8 = tile.astype(np.uint8)
        
        # Handle grayscale vs RGB
        if len(tile_uint8.shape) == 2:
            img = Image.fromarray(tile_uint8, mode='L')
        elif tile_uint8.shape[2] == 1:
            img = Image.fromarray(tile_uint8[:, :, 0], mode='L')
        elif tile_uint8.shape[2] == 3:
            img = Image.fromarray(tile_uint8, mode='RGB')
        elif tile_uint8.shape[2] == 4:
            img = Image.fromarray(tile_uint8, mode='RGBA')
        else:
            # For multi-band, save first 3 as RGB
            img = Image.fromarray(tile_uint8[:, :, :3], mode='RGB')
        
        # Save image
        output_path = output_dir / tile_info.filename
        img.save(output_path, format=output_format)
        
        saved_tiles.append(tile_info)
        logger.debug(f"Saved tile: {tile_info.filename}")
    
    # Save metadata
    if save_metadata and saved_tiles:
        metadata = {
            "tile_size": tile_size,
            "stride": stride,
            "min_valid_ratio": min_valid_ratio,
            "total_tiles": len(saved_tiles),
            "tiles": [
                {
                    "filename": t.filename,
                    "row": t.row,
                    "col": t.col,
                    "y_start": t.y_start,
                    "y_end": t.y_end,
                    "x_start": t.x_start,
                    "x_end": t.x_end,
                    "valid_ratio": t.valid_ratio,
                }
                for t in saved_tiles
            ]
        }
        
        metadata_path = output_dir / "tiles_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved tile metadata: {metadata_path}")
    
    logger.info(f"Saved {len(saved_tiles)} tiles to {output_dir}")
    return saved_tiles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    print("\n=== Testing Tile Positions ===")
    positions = calculate_tile_positions(
        image_height=350,
        image_width=350,
        tile_size=224,
        stride=112
    )
    print(f"Number of tiles: {len(positions)}")
    for i, (y1, y2, x1, x2) in enumerate(positions[:5]):
        print(f"Tile {i}: y=[{y1}, {y2}), x=[{x1}, {x2})")
    
    # Test tile extraction
    print("\n=== Testing Tile Extraction ===")
    test_image = np.random.rand(350, 350, 3).astype(np.float32)
    
    tiles = list(tile_image(test_image, tile_size=224, stride=112, min_valid_ratio=0.8))
    print(f"Generated {len(tiles)} valid tiles")
    
    if tiles:
        tile, info = tiles[0]
        print(f"First tile shape: {tile.shape}")
        print(f"First tile info: {info}")

