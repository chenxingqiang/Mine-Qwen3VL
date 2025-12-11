# Cuprite Hyperspectral Data

## Data Description

This directory contains the Cuprite AVIRIS hyperspectral dataset, a benchmark
dataset for mineral mapping using hyperspectral remote sensing.

### Dataset Details

- **Location**: Cuprite mining district, Nevada, USA
- **Sensor**: AVIRIS (Airborne Visible/Infrared Imaging Spectrometer)
- **Spectral Range**: 400-2500 nm
- **Spectral Bands**: 224 (some removed due to water absorption)
- **Spatial Resolution**: ~20 meters
- **Image Size**: ~350 x 350 pixels (varies by version)

### Key Minerals Present

1. **Alunite** - Potassium aluminum sulfate hydroxide
2. **Kaolinite** - Clay mineral (aluminum silicate)
3. **Muscovite** - White mica (potassium aluminum silicate)
4. **Calcite** - Calcium carbonate
5. **Buddingtonite** - Ammonium feldspar

### File Formats

- `.img` / `.dat`: Raw hyperspectral data (ENVI format)
- `.hdr`: ENVI header file (metadata)
- `.tif`: GeoTIFF format (georeferenced)

### Usage

```python
from src.preprocessing import read_envi

# Load hyperspectral data
hs_data = read_envi("path/to/cuprite.img")
print(f"Shape: {hs_data.shape}")  # (H, W, Bands)
print(f"Wavelengths: {hs_data.wavelengths}")
```

### References

1. Clark, R.N., et al. (1993). "Imaging spectroscopy: Earth and planetary
   remote sensing with the USGS Tetracorder and expert systems."
   Journal of Geophysical Research.

2. USGS Spectroscopy Lab: https://crustal.usgs.gov/speclab/

### Data Source

Downloaded from Purdue University MultiSpec:
https://engineering.purdue.edu/~biehl/MultiSpec/
