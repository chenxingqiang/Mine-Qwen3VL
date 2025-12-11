"""
Unit tests for preprocessing modules.

Tests:
- Hyperspectral I/O
- Band selection and combination
- Normalization
- Image tiling
"""

import sys
from pathlib import Path
import numpy as np
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


class TestHyperspectralIO:
    """Tests for hyperspectral_io module."""
    
    def test_hyperspectral_data_class(self):
        """Test HyperspectralData dataclass."""
        from preprocessing.hyperspectral_io import HyperspectralData, get_wavelengths
        
        # Create test data in HWB format (height=100, width=100, bands=224)
        data = np.random.rand(100, 100, 224).astype(np.float32)
        wavelengths = get_wavelengths(224)
        
        hs_data = HyperspectralData(
            data=data,
            wavelengths=wavelengths,
            nodata_value=-9999.0,
            is_hwb=True  # Explicitly specify format
        )
        
        assert hs_data.shape == (100, 100, 224)
        assert hs_data.n_bands == 224
        assert hs_data.height == 100
        assert hs_data.width == 100
        assert len(hs_data.wavelengths) == 224
        
    def test_get_wavelengths(self):
        """Test wavelength generation."""
        from preprocessing.hyperspectral_io import get_wavelengths
        
        wavelengths = get_wavelengths(224, 400.0, 2500.0)
        
        assert len(wavelengths) == 224
        assert wavelengths[0] == pytest.approx(400.0)
        assert wavelengths[-1] == pytest.approx(2500.0)
        
    def test_to_bhw_format(self):
        """Test format conversion to (B, H, W)."""
        from preprocessing.hyperspectral_io import HyperspectralData
        
        # HWB format - (height=100, width=80, bands=224)
        data_hwb = np.random.rand(100, 80, 224).astype(np.float32)
        hs_data = HyperspectralData(data=data_hwb, is_hwb=True)
        
        bhw = hs_data.to_bhw()
        assert bhw.shape == (224, 100, 80)  # (B, H, W)
        
    def test_to_hwb_format(self):
        """Test format conversion to (H, W, B)."""
        from preprocessing.hyperspectral_io import HyperspectralData
        
        # BHW format - (bands=224, height=100, width=80)
        data_bhw = np.random.rand(224, 100, 80).astype(np.float32)
        hs_data = HyperspectralData(data=data_bhw, is_hwb=False)
        
        hwb = hs_data.to_hwb()
        assert hwb.shape == (100, 80, 224)  # (H, W, B)


class TestBandSelection:
    """Tests for band_selection module."""
    
    def test_wavelength_to_band_index(self):
        """Test wavelength to band index mapping."""
        from preprocessing.band_selection import wavelength_to_band_index
        
        wavelengths = np.linspace(400, 2500, 224)
        
        # Test exact match (within tolerance)
        idx = wavelength_to_band_index(660, wavelengths)
        assert 0 <= idx < 224
        assert abs(wavelengths[idx] - 660) < 20  # Within tolerance
        
    def test_wavelength_to_band_index_out_of_range(self):
        """Test that out-of-range wavelength raises error."""
        from preprocessing.band_selection import wavelength_to_band_index
        
        wavelengths = np.linspace(400, 2500, 224)
        
        with pytest.raises(ValueError):
            wavelength_to_band_index(3000, wavelengths, tolerance=20)
            
    def test_select_bands(self):
        """Test band selection."""
        from preprocessing.band_selection import select_bands
        
        data = np.random.rand(100, 100, 224).astype(np.float32)
        selected = select_bands(data, [0, 50, 100])
        
        assert selected.shape == (100, 100, 3)
        np.testing.assert_array_equal(selected[:, :, 0], data[:, :, 0])
        np.testing.assert_array_equal(selected[:, :, 1], data[:, :, 50])
        np.testing.assert_array_equal(selected[:, :, 2], data[:, :, 100])
        
    def test_create_rgb_composite(self):
        """Test RGB composite creation."""
        from preprocessing.band_selection import create_rgb_composite
        
        data = np.random.rand(100, 100, 224).astype(np.float32)
        rgb = create_rgb_composite(data, r_band=50, g_band=100, b_band=150)
        
        assert rgb.shape == (100, 100, 3)
        assert rgb.min() >= 0
        assert rgb.max() <= 1
        
    def test_apply_band_combination(self):
        """Test applying predefined band combination."""
        from preprocessing.band_selection import apply_band_combination
        
        wavelengths = np.linspace(400, 2500, 224)
        data = np.random.rand(100, 100, 224).astype(np.float32)
        
        combination = {'R': 2200, 'G': 2100, 'B': 1650}
        rgb = apply_band_combination(data, wavelengths, combination)
        
        assert rgb.shape == (100, 100, 3)
        assert rgb.min() >= 0
        assert rgb.max() <= 1


class TestNormalization:
    """Tests for normalization module."""
    
    def test_normalize_minmax(self):
        """Test min-max normalization."""
        from preprocessing.normalization import normalize_minmax
        
        data = np.random.rand(100, 100, 10) * 10000  # Scale to 0-10000
        normalized = normalize_minmax(data)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
    def test_normalize_minmax_with_nodata(self):
        """Test min-max normalization with nodata values."""
        from preprocessing.normalization import normalize_minmax
        
        data = np.random.rand(100, 100, 10) * 10000
        data[0:10, 0:10, :] = -9999  # Add nodata
        
        normalized = normalize_minmax(data, nodata_value=-9999)
        
        # Nodata areas should be 0
        assert normalized[5, 5, 0] == 0
        
    def test_normalize_percentile(self):
        """Test percentile normalization."""
        from preprocessing.normalization import normalize_percentile
        
        data = np.random.rand(100, 100, 10) * 10000
        normalized = normalize_percentile(data, percentile_low=2, percentile_high=98)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
    def test_normalize_standard(self):
        """Test standard (z-score) normalization."""
        from preprocessing.normalization import normalize_standard
        
        data = np.random.rand(100, 100, 10) * 10000
        standardized = normalize_standard(data)
        
        # Mean should be close to 0 for each band
        for i in range(10):
            band_mean = np.mean(standardized[:, :, i])
            assert abs(band_mean) < 0.1  # Approximately 0
            
    def test_to_uint8(self):
        """Test conversion to uint8."""
        from preprocessing.normalization import to_uint8
        
        data = np.random.rand(100, 100, 3)  # 0-1 range
        uint8_data = to_uint8(data)
        
        assert uint8_data.dtype == np.uint8
        assert uint8_data.min() >= 0
        assert uint8_data.max() <= 255
        
    def test_remove_bad_bands(self):
        """Test bad band removal."""
        from preprocessing.normalization import remove_bad_bands
        
        data = np.random.rand(100, 100, 224)
        wavelengths = np.linspace(400, 2500, 224)
        
        bad_bands = list(range(0, 5)) + list(range(100, 110))
        cleaned, cleaned_wl = remove_bad_bands(data, bad_bands, wavelengths)
        
        expected_bands = 224 - len(bad_bands)
        assert cleaned.shape[-1] == expected_bands
        assert len(cleaned_wl) == expected_bands


class TestTiling:
    """Tests for tiling module."""
    
    def test_calculate_tile_positions(self):
        """Test tile position calculation."""
        from preprocessing.tiling import calculate_tile_positions
        
        positions = calculate_tile_positions(
            image_height=350,
            image_width=350,
            tile_size=224,
            stride=112
        )
        
        assert len(positions) > 0
        
        # Check first position
        y1, y2, x1, x2 = positions[0]
        assert y1 == 0
        assert y2 == 224
        assert x1 == 0
        assert x2 == 224
        
    def test_calculate_tile_positions_no_overlap(self):
        """Test tile positions with no overlap."""
        from preprocessing.tiling import calculate_tile_positions
        
        positions = calculate_tile_positions(
            image_height=448,
            image_width=448,
            tile_size=224,
            stride=224  # No overlap
        )
        
        assert len(positions) == 4  # 2x2 grid
        
    def test_calculate_valid_ratio(self):
        """Test valid pixel ratio calculation."""
        from preprocessing.tiling import calculate_valid_ratio
        
        # All valid
        tile = np.random.rand(224, 224)
        ratio = calculate_valid_ratio(tile)
        assert ratio == 1.0
        
        # With nodata
        tile_with_nodata = np.random.rand(224, 224)
        tile_with_nodata[0:112, :] = -9999
        ratio = calculate_valid_ratio(tile_with_nodata, nodata_value=-9999)
        assert ratio == pytest.approx(0.5)
        
    def test_extract_tile(self):
        """Test tile extraction."""
        from preprocessing.tiling import extract_tile
        
        data = np.arange(100 * 100 * 3).reshape(100, 100, 3)
        tile = extract_tile(data, 10, 30, 20, 40)
        
        assert tile.shape == (20, 20, 3)
        np.testing.assert_array_equal(tile, data[10:30, 20:40, :])
        
    def test_extract_tile_with_padding(self):
        """Test tile extraction with padding for edge cases."""
        from preprocessing.tiling import extract_tile
        
        data = np.ones((50, 50, 3))
        # Request tile that extends beyond image
        tile = extract_tile(data, 40, 64, 40, 64, pad_value=0)
        
        assert tile.shape == (24, 24, 3)
        # The actual data portion should be 1s, padding should be 0s
        assert tile[0, 0, 0] == 1  # Valid region
        assert tile[10, 10, 0] == 0  # Padded region
        
    def test_tile_image(self):
        """Test full image tiling."""
        from preprocessing.tiling import tile_image
        
        data = np.random.rand(350, 350, 3)
        
        tiles = list(tile_image(data, tile_size=224, stride=112, min_valid_ratio=0.8))
        
        assert len(tiles) > 0
        
        # Check first tile
        tile, info = tiles[0]
        assert tile.shape == (224, 224, 3)
        assert info.row >= 0
        assert info.col >= 0
        assert info.valid_ratio >= 0.8
        
    def test_save_tiles(self):
        """Test saving tiles to disk."""
        from preprocessing.tiling import save_tiles
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(350, 350, 3)
            output_dir = Path(tmpdir)
            
            tile_infos = save_tiles(
                data, output_dir,
                tile_size=224, stride=112,
                min_valid_ratio=0.8,
                save_metadata=True
            )
            
            assert len(tile_infos) > 0
            
            # Check files exist
            for info in tile_infos:
                tile_path = output_dir / info.filename
                assert tile_path.exists()
                
            # Check metadata
            metadata_path = output_dir / "tiles_metadata.json"
            assert metadata_path.exists()
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            assert metadata["total_tiles"] == len(tile_infos)


class TestConfig:
    """Tests for configuration module."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        from config import get_config, Config
        
        config = get_config()
        
        assert isinstance(config, Config)
        assert config.bands.total_bands == 224
        assert config.tiling.tile_size == 224
        assert config.minerals.copper_alteration_threshold == 0.10
        
    def test_band_combinations(self):
        """Test band combination definitions."""
        from config import get_config
        
        config = get_config()
        
        assert "clay_minerals" in config.bands.band_combinations
        assert "iron_oxide" in config.bands.band_combinations
        assert "natural_color" in config.bands.band_combinations
        
        clay = config.bands.band_combinations["clay_minerals"]
        assert clay["R"] == 2200.0
        assert clay["G"] == 2100.0
        assert clay["B"] == 1650.0
        
    def test_mineral_config(self):
        """Test mineral configuration."""
        from config import get_config
        
        config = get_config()
        
        assert len(config.minerals.mineral_classes) > 0
        assert config.minerals.mineral_classes[0]["name"] == "Background"
        
        copper_ids = config.minerals.copper_related_ids
        assert len(copper_ids) > 0
        assert 3 in copper_ids  # Muscovite
        
    def test_training_config(self):
        """Test training configuration."""
        from config import get_config
        
        config = get_config()
        
        assert config.training.model_name == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.training.lora_enable == True
        assert config.training.lora_r == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

