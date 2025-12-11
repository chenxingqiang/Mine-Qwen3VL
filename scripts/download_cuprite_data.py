#!/usr/bin/env python3
"""
Download Cuprite AVIRIS hyperspectral data and USGS ground truth.

Data Sources:
1. AVIRIS Cuprite data from USGS/JPL
2. Ground truth mineral maps from USGS Spectroscopy Lab

This script downloads publicly available Cuprite datasets commonly used
for hyperspectral mineral mapping research.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Data sources
CUPRITE_DATASETS = {
    # NASA JPL 1995 Cuprite AVIRIS data (primary source)
    "nasa_cuprite": {
        "url": "https://popo.jpl.nasa.gov/1995_cuprite_RTGC_rfl_cube/cuprite_1995_L2.zip",
        "filename": "cuprite_1995_L2.zip",
        "description": "NASA JPL 1995 Cuprite AVIRIS L2 reflectance data",
        "size_mb": 50,
        "format": "zip"
    },
    
    # NASA JPL Header file
    "nasa_cuprite_hdr": {
        "url": "https://popo.jpl.nasa.gov/1995_cuprite_RTGC_rfl_cube/cuprite.95.cal.rtgc.v.hdr",
        "filename": "cuprite.95.cal.rtgc.v.hdr",
        "description": "Cuprite ENVI header file",
        "size_mb": 0.01,
        "format": "hdr"
    },
    
    # GIC Mirror - University of the Basque Country
    "gic_cuprite": {
        "url": "https://www.ehu.eus/ccwintco/uploads/e/e3/Cuprite.zip",
        "filename": "Cuprite.zip",
        "description": "Cuprite AVIRIS data (GIC Mirror, ~95 MB)",
        "size_mb": 95,
        "format": "zip"
    },
    
    # MicroImages sample data
    "microimages_cuprite": {
        "url": "https://www.microimages.com/downloads/data/cup97.zip",
        "filename": "cup97.zip",
        "description": "Cuprite 1997 hyperspectral sample data",
        "size_mb": 30,
        "format": "zip"
    },
}

# Alternative mirrors and manual download instructions
MANUAL_DOWNLOAD_SOURCES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Manual Download Instructions                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║ If automatic download fails, you can manually download from:                  ║
║                                                                               ║
║ 1. AVIRIS Cuprite Data (Recommended):                                        ║
║    https://engineering.purdue.edu/~biehl/MultiSpec/documentation.html         ║
║    → Download "Cuprite92AV93.zip"                                             ║
║                                                                               ║
║ 2. AVIRIS Data Portal (Full resolution, large files):                         ║
║    https://aviris.jpl.nasa.gov/data/free_data.html                            ║
║    → Search for "Cuprite" in the archive                                      ║
║                                                                               ║
║ 3. GIC (Grupo de Inteligencia Computacional) Mirror:                          ║
║    http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes  ║
║    → Download "Cuprite" dataset                                               ║
║                                                                               ║
║ 4. USGS Spectroscopy Lab (Ground Truth):                                      ║
║    https://crustal.usgs.gov/speclab/cuprite.html                              ║
║    → Mineral maps and reference spectra                                       ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def get_project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


def download_file(url: str, dest_path: Path, expected_size_mb: int = None) -> bool:
    """
    Download a file with progress indicator.
    
    Args:
        url: URL to download
        dest_path: Destination file path
        expected_size_mb: Expected file size in MB (for progress display)
    
    Returns:
        True if download successful
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading: {url}")
    logger.info(f"Destination: {dest_path}")
    
    if expected_size_mb:
        logger.info(f"Expected size: ~{expected_size_mb} MB")
    
    try:
        # Set up request with headers
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; Python script)'}
        )
        
        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = response.getheader('Content-Length')
            if total_size:
                total_size = int(total_size)
                logger.info(f"File size: {total_size / 1024 / 1024:.1f} MB")
            
            downloaded = 0
            chunk_size = 8192
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size:
                        progress = downloaded / total_size * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        print(f"\r  Progress: [{bar}] {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress bar
            logger.info(f"✓ Downloaded successfully: {downloaded / 1024 / 1024:.1f} MB")
            return True
            
    except urllib.error.URLError as e:
        logger.error(f"✗ Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """
    Extract zip or tar archive.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
    
    Returns:
        True if extraction successful
    """
    logger.info(f"Extracting: {archive_path}")
    logger.info(f"To: {extract_dir}")
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(extract_dir)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        logger.info("✓ Extraction complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        return False


def list_extracted_files(directory: Path, max_files: int = 20):
    """List files in directory."""
    files = list(directory.rglob('*'))
    files = [f for f in files if f.is_file()]
    
    logger.info(f"\nExtracted files ({len(files)} total):")
    for f in files[:max_files]:
        rel_path = f.relative_to(directory)
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {rel_path} ({size_kb:.1f} KB)")
    
    if len(files) > max_files:
        logger.info(f"  ... and {len(files) - max_files} more files")


def download_cuprite_nasa(output_dir: Path) -> Path:
    """
    Download Cuprite data from NASA JPL.
    
    Returns:
        Path to extracted data directory
    """
    extract_dir = output_dir / "cuprite_aviris"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("Downloading Cuprite AVIRIS Data (NASA JPL)")
    logger.info("="*60)
    
    # Try NASA JPL first
    dataset = CUPRITE_DATASETS["nasa_cuprite"]
    archive_path = output_dir / dataset["filename"]
    
    logger.info(f"Source: NASA JPL")
    logger.info(f"Description: {dataset['description']}")
    logger.info("")
    
    if archive_path.exists():
        logger.info(f"Archive already exists: {archive_path}")
        success = True
    else:
        success = download_file(dataset["url"], archive_path, dataset["size_mb"])
    
    # If NASA fails, try GIC mirror
    if not success:
        logger.info("\nNASA source unavailable, trying GIC mirror...")
        dataset = CUPRITE_DATASETS["gic_cuprite"]
        archive_path = output_dir / dataset["filename"]
        
        if archive_path.exists():
            logger.info(f"Archive already exists: {archive_path}")
            success = True
        else:
            success = download_file(dataset["url"], archive_path, dataset["size_mb"])
    
    # If both fail, try MicroImages
    if not success:
        logger.info("\nGIC mirror unavailable, trying MicroImages...")
        dataset = CUPRITE_DATASETS["microimages_cuprite"]
        archive_path = output_dir / dataset["filename"]
        
        if archive_path.exists():
            logger.info(f"Archive already exists: {archive_path}")
            success = True
        else:
            success = download_file(dataset["url"], archive_path, dataset["size_mb"])
    
    if not success:
        logger.error("All download sources failed. See manual download instructions below.")
        print(MANUAL_DOWNLOAD_SOURCES)
        return None
    
    # Extract
    if extract_dir.exists() and len(list(extract_dir.iterdir())) > 1:
        logger.info(f"Data already extracted: {extract_dir}")
    else:
        success = extract_archive(archive_path, extract_dir)
        if not success:
            return None
    
    list_extracted_files(extract_dir)
    
    return extract_dir


def create_data_readme(output_dir: Path):
    """Create README for downloaded data."""
    readme_content = """# Cuprite Hyperspectral Data

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
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"\nCreated data README: {readme_path}")


def main():
    """Main function to download Cuprite data."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Cuprite hyperspectral data for pipeline testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--dataset",
        choices=["multispec", "usgs_library", "all"],
        default="multispec",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Show manual download instructions only"
    )
    
    args = parser.parse_args()
    
    if args.manual:
        print(MANUAL_DOWNLOAD_SOURCES)
        return 0
    
    # Set output directory
    project_root = get_project_root()
    output_dir = args.output_dir or (project_root / "data" / "raw")
    
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║       Cuprite Hyperspectral Data Downloader              ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"\nOutput directory: {output_dir}\n")
    
    # Download selected dataset
    success = False
    
    if args.dataset in ["multispec", "all", "nasa"]:
        data_dir = download_cuprite_nasa(output_dir)
        if data_dir:
            success = True
            create_data_readme(output_dir)
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("✓ Download Complete!")
        logger.info("="*60)
        logger.info(f"\nData location: {output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Run data preparation:")
        logger.info(f"     python scripts/prepare_cuprite_data.py --input_dir {output_dir}/cuprite_aviris")
        logger.info("")
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("✗ Download Failed")
        logger.error("="*60)
        print(MANUAL_DOWNLOAD_SOURCES)
        return 1


if __name__ == "__main__":
    sys.exit(main())

