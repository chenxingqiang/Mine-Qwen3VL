"""
Configuration file for Cuprite hyperspectral data preprocessing pipeline.

This module defines all configurable parameters for:
- Band selection and combination
- Preprocessing (normalization, bad band removal)
- Image tiling
- Mineral classification
- Task prompt templates
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# =============================================================================
# Path Configuration
# =============================================================================
@dataclass
class PathConfig:
    """Path configuration for data directories."""
    
    # Project root
    project_root: Path = Path(__file__).parent.parent
    
    # Data directories
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    dataset_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "cuprite_dataset")
    
    # Qwen3-VL directory
    qwen3vl_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "Qwen3-VL")
    
    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        for attr in ['project_root', 'raw_data_dir', 'processed_data_dir', 'dataset_dir', 'qwen3vl_dir']:
            value = getattr(self, attr)
            if isinstance(value, str):
                setattr(self, attr, Path(value))


# =============================================================================
# Band Configuration
# =============================================================================
@dataclass
class BandConfig:
    """Configuration for hyperspectral band selection and combination."""
    
    # Total number of bands in AVIRIS Cuprite data
    total_bands: int = 224
    
    # Wavelength range (nm)
    wavelength_start: float = 400.0
    wavelength_end: float = 2500.0
    
    # Water absorption bands to remove (band indices, 0-based)
    water_absorption_bands: List[int] = field(default_factory=lambda: 
        list(range(104, 113)) +  # 1350-1450 nm
        list(range(148, 167))    # 1790-1990 nm
    )
    
    # Noisy bands to remove (band indices, 0-based)
    noisy_bands: List[int] = field(default_factory=lambda: 
        list(range(0, 3)) +      # First 3 bands (noise)
        list(range(220, 224))    # Last 4 bands (noise)
    )
    
    # Band combination definitions (wavelength in nm)
    # Format: {"name": {"R": wavelength, "G": wavelength, "B": wavelength}}
    band_combinations: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "clay_minerals": {
            "R": 2200.0,  # Al-OH absorption feature
            "G": 2100.0,  # Reference band
            "B": 1650.0,  # Continuum background
            "description": "Clay mineral alteration enhancement"
        },
        "iron_oxide": {
            "R": 860.0,   # NIR reflectance
            "G": 660.0,   # Red band
            "B": 480.0,   # Blue band
            "description": "Iron oxide mineralization enhancement"
        },
        "natural_color": {
            "R": 660.0,   # Red
            "G": 550.0,   # Green
            "B": 470.0,   # Blue
            "description": "Near natural color"
        }
    })
    
    # Primary band combination to use
    primary_combination: str = "clay_minerals"


# =============================================================================
# Preprocessing Configuration
# =============================================================================
@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Normalization method: "min_max", "standard", "percentile"
    normalization_method: str = "min_max"
    
    # Percentile values for percentile normalization
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    
    # Output data type
    output_dtype: str = "uint8"  # "uint8" (0-255) or "float32" (0-1)
    
    # Handle no-data values
    nodata_value: float = -9999.0
    fill_nodata: bool = True
    fill_value: float = 0.0


# =============================================================================
# Tiling Configuration
# =============================================================================
@dataclass
class TilingConfig:
    """Configuration for image tiling."""
    
    # Tile size in pixels
    tile_size: int = 224
    
    # Stride (step size) for sliding window
    # stride < tile_size creates overlap
    stride: int = 112  # 50% overlap
    
    # Minimum ratio of valid (non-nodata) pixels required
    min_valid_ratio: float = 0.8
    
    # Output format
    output_format: str = "PNG"  # "PNG" or "JPEG"
    
    # Filename pattern (Python format string)
    filename_pattern: str = "tile_{row:04d}_{col:04d}.png"
    
    # Whether to save tile metadata
    save_metadata: bool = True


# =============================================================================
# Mineral Classification Configuration
# =============================================================================
@dataclass
class MineralConfig:
    """Configuration for mineral classification."""
    
    # Mineral class definitions
    # Format: {class_id: {"name": str, "cn": str, "copper_related": bool}}
    mineral_classes: Dict[int, Dict] = field(default_factory=lambda: {
        0: {"name": "Background", "cn": "背景", "copper_related": False},
        1: {"name": "Alunite", "cn": "明矾石", "copper_related": True},
        2: {"name": "Kaolinite", "cn": "高岭石", "copper_related": True},
        3: {"name": "Muscovite", "cn": "白云母/绢云母", "copper_related": True},
        4: {"name": "Montmorillonite", "cn": "蒙脱石", "copper_related": True},
        5: {"name": "Buddingtonite", "cn": "铵长石", "copper_related": False},
        6: {"name": "Calcite", "cn": "方解石", "copper_related": False},
        7: {"name": "Chlorite", "cn": "绿泥石", "copper_related": True},
        8: {"name": "Epidote", "cn": "绿帘石", "copper_related": True},
    })
    
    # Threshold for mineral detection (minimum coverage ratio)
    min_coverage_threshold: float = 0.01  # 1%
    
    # Threshold for copper alteration zone classification
    copper_alteration_threshold: float = 0.10  # 10%
    
    @property
    def copper_related_ids(self) -> List[int]:
        """Get list of copper-related mineral class IDs."""
        return [k for k, v in self.mineral_classes.items() if v.get("copper_related", False)]


# =============================================================================
# Task and Prompt Configuration
# =============================================================================
@dataclass
class TaskConfig:
    """Configuration for training tasks and prompts."""
    
    # Binary classification prompts (Chinese)
    binary_prompts_cn: List[str] = field(default_factory=lambda: [
        "请判断该高光谱图像区域是否存在铜矿相关蚀变特征？",
        "分析该区域是否具有铜矿化潜力？",
        "该区域是否存在热液蚀变迹象？",
        "这幅高光谱影像是否显示铜矿蚀变带？",
    ])
    
    # Binary classification prompts (English)
    binary_prompts_en: List[str] = field(default_factory=lambda: [
        "Does this hyperspectral image show copper-related alteration features?",
        "Analyze whether this area has copper mineralization potential.",
        "Are there signs of hydrothermal alteration in this region?",
    ])
    
    # Mineral identification prompts
    mineral_prompts_cn: List[str] = field(default_factory=lambda: [
        "请识别该高光谱图像中的主要蚀变矿物类型。",
        "分析该区域存在哪些矿物？",
        "该图像中可见哪些蚀变矿物？请列出。",
    ])
    
    mineral_prompts_en: List[str] = field(default_factory=lambda: [
        "Identify the main alteration minerals in this hyperspectral image.",
        "What minerals are present in this area?",
        "List the visible alteration minerals in this image.",
    ])
    
    # Detailed analysis prompts
    analysis_prompts_cn: List[str] = field(default_factory=lambda: [
        "请详细分析该高光谱图像的矿化特征和找矿意义。",
        "从蚀变矿物组合角度，分析该区域的成矿潜力。",
        "结合图像特征，给出该区域的地质解释。",
    ])
    
    # Language preference: "cn", "en", "both"
    language: str = "cn"
    
    # Task weights for balanced sampling
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "binary": 0.4,
        "mineral": 0.4,
        "analysis": 0.2,
    })


# =============================================================================
# Dataset Split Configuration
# =============================================================================
@dataclass
class SplitConfig:
    """Configuration for train/validation split."""
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # Split method: "random", "spatial"
    split_method: str = "spatial"
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Stratify by copper alteration status
    stratify: bool = True


# =============================================================================
# Training Configuration
# =============================================================================
@dataclass
class TrainingConfig:
    """Configuration for Qwen3-VL fine-tuning."""
    
    # Model
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    
    # LoRA configuration
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training parameters
    learning_rate: float = 1e-5
    mm_projector_lr: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    warmup_ratio: float = 0.03
    
    # Image parameters
    max_pixels: int = 50176  # 224 * 224
    min_pixels: int = 784    # 28 * 28
    
    # Tunable modules
    tune_mm_vision: bool = False
    tune_mm_mlp: bool = True
    tune_mm_llm: bool = True
    
    # DeepSpeed config
    deepspeed_config: str = "zero3.json"


# =============================================================================
# Master Configuration
# =============================================================================
@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    bands: BandConfig = field(default_factory=BandConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    minerals: MineralConfig = field(default_factory=MineralConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Default configuration instance
default_config = Config()


def get_config() -> Config:
    """Get the default configuration."""
    return default_config


if __name__ == "__main__":
    # Print configuration for verification
    config = get_config()
    print("=== Path Configuration ===")
    print(f"Project root: {config.paths.project_root}")
    print(f"Raw data dir: {config.paths.raw_data_dir}")
    print(f"Dataset dir: {config.paths.dataset_dir}")
    
    print("\n=== Band Configuration ===")
    print(f"Total bands: {config.bands.total_bands}")
    print(f"Primary combination: {config.bands.primary_combination}")
    print(f"Band combinations: {list(config.bands.band_combinations.keys())}")
    
    print("\n=== Mineral Configuration ===")
    print(f"Copper-related minerals: {config.minerals.copper_related_ids}")
    print(f"Alteration threshold: {config.minerals.copper_alteration_threshold}")
    
    print("\n=== Training Configuration ===")
    print(f"Model: {config.training.model_name}")
    print(f"LoRA r: {config.training.lora_r}")
    print(f"Learning rate: {config.training.learning_rate}")

