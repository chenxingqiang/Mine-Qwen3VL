"""
Mineral analysis module for processing ground truth labels.

Analyzes USGS mineral classification maps to generate
structured annotations for each image tile.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)

# Import config for mineral definitions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MineralConfig, default_config


@dataclass
class MineralStats:
    """Statistics for minerals detected in a tile."""
    
    # Mineral coverage ratios {mineral_name: ratio}
    minerals: Dict[str, float] = field(default_factory=dict)
    
    # Copper-related mineral statistics
    copper_related_minerals: Dict[str, float] = field(default_factory=dict)
    copper_mineral_ratio: float = 0.0
    
    # Classification result
    is_copper_alteration: bool = False
    
    # Dominant mineral
    dominant_mineral: Optional[str] = None
    dominant_ratio: float = 0.0
    
    # Total valid pixel ratio
    valid_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "minerals": self.minerals,
            "copper_related_minerals": self.copper_related_minerals,
            "copper_mineral_ratio": self.copper_mineral_ratio,
            "is_copper_alteration": self.is_copper_alteration,
            "dominant_mineral": self.dominant_mineral,
            "dominant_ratio": self.dominant_ratio,
            "valid_ratio": self.valid_ratio,
        }


def load_ground_truth(
    path: Path,
    format: str = "auto"
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load ground truth mineral classification map.
    
    Args:
        path: Path to ground truth file
        format: File format ("auto", "tif", "npy", "envi")
        
    Returns:
        Tuple of (label_array, metadata)
    """
    path = Path(path)
    
    if format == "auto":
        format = path.suffix.lower().lstrip('.')
    
    metadata = None
    
    if format in ['tif', 'tiff']:
        try:
            import rasterio
            with rasterio.open(path) as src:
                labels = src.read(1)  # Read first band
                metadata = {
                    'crs': str(src.crs) if src.crs else None,
                    'transform': src.transform,
                    'nodata': src.nodata,
                }
        except ImportError:
            from PIL import Image
            img = Image.open(path)
            labels = np.array(img)
    
    elif format == 'npy':
        labels = np.load(path)
    
    elif format in ['img', 'envi']:
        # Use our hyperspectral_io module
        from ..preprocessing.hyperspectral_io import read_envi
        hs_data = read_envi(path)
        labels = hs_data.data[:, :, 0] if len(hs_data.data.shape) == 3 else hs_data.data
    
    elif format == 'png':
        from PIL import Image
        img = Image.open(path)
        labels = np.array(img)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded ground truth: shape={labels.shape}, unique values={np.unique(labels)}")
    return labels, metadata


def analyze_tile_minerals(
    tile_labels: np.ndarray,
    mineral_config: Optional[MineralConfig] = None
) -> MineralStats:
    """
    Analyze mineral distribution in a tile.
    
    Args:
        tile_labels: Label array for the tile
        mineral_config: Mineral configuration (uses default if None)
        
    Returns:
        MineralStats object with analysis results
    """
    if mineral_config is None:
        mineral_config = default_config.minerals
    
    total_pixels = tile_labels.size
    
    if total_pixels == 0:
        return MineralStats()
    
    # Count pixels for each class
    unique, counts = np.unique(tile_labels, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    
    # Calculate ratios for each mineral
    minerals = {}
    copper_related = {}
    
    for class_id, info in mineral_config.mineral_classes.items():
        count = class_counts.get(class_id, 0)
        ratio = count / total_pixels
        
        if ratio >= mineral_config.min_coverage_threshold:
            mineral_name = info['name']
            minerals[mineral_name] = round(ratio, 4)
            
            if info.get('copper_related', False):
                copper_related[mineral_name] = round(ratio, 4)
    
    # Calculate copper-related mineral ratio
    copper_ratio = sum(copper_related.values())
    
    # Classify as alteration zone
    is_copper_alteration = copper_ratio >= mineral_config.copper_alteration_threshold
    
    # Find dominant mineral (excluding background)
    dominant_mineral = None
    dominant_ratio = 0.0
    for mineral, ratio in minerals.items():
        if mineral != "Background" and ratio > dominant_ratio:
            dominant_mineral = mineral
            dominant_ratio = ratio
    
    # Calculate valid pixel ratio (non-background)
    background_count = class_counts.get(0, 0)
    valid_ratio = (total_pixels - background_count) / total_pixels
    
    stats = MineralStats(
        minerals=minerals,
        copper_related_minerals=copper_related,
        copper_mineral_ratio=round(copper_ratio, 4),
        is_copper_alteration=is_copper_alteration,
        dominant_mineral=dominant_mineral,
        dominant_ratio=round(dominant_ratio, 4),
        valid_ratio=round(valid_ratio, 4),
    )
    
    logger.debug(f"Tile analysis: copper_ratio={copper_ratio:.2%}, is_alteration={is_copper_alteration}")
    return stats


def classify_alteration_zone(
    stats: MineralStats,
    threshold: float = 0.10
) -> str:
    """
    Classify the alteration zone type based on mineral assemblage.
    
    Args:
        stats: MineralStats from tile analysis
        threshold: Minimum ratio to consider
        
    Returns:
        Alteration zone type string
    """
    minerals = stats.copper_related_minerals
    
    if not minerals:
        return "Background"
    
    # Check for specific alteration types
    has_alunite = minerals.get("Alunite", 0) > threshold
    has_kaolinite = minerals.get("Kaolinite", 0) > threshold
    has_muscovite = minerals.get("Muscovite", 0) > threshold
    has_chlorite = minerals.get("Chlorite", 0) > threshold
    has_epidote = minerals.get("Epidote", 0) > threshold
    has_montmorillonite = minerals.get("Montmorillonite", 0) > threshold
    
    # Classify based on mineral assemblage
    if has_alunite and has_kaolinite:
        return "Advanced Argillic"  # 高级泥化带
    elif has_muscovite or has_kaolinite:
        return "Phyllic-Argillic"  # 绢英岩化-泥化带
    elif has_chlorite and has_epidote:
        return "Propylitic"  # 青磐岩化带
    elif has_montmorillonite:
        return "Argillic"  # 泥化带
    elif has_chlorite:
        return "Chloritic"  # 绿泥石化带
    else:
        return "Weak Alteration"  # 弱蚀变


def get_mineral_description(
    stats: MineralStats,
    language: str = "cn"
) -> Dict[str, str]:
    """
    Generate natural language descriptions of mineral analysis.
    
    Args:
        stats: MineralStats from tile analysis
        language: "cn" for Chinese, "en" for English
        
    Returns:
        Dictionary with different description types
    """
    descriptions = {}
    
    # Get mineral names
    mineral_names = list(stats.minerals.keys())
    copper_minerals = list(stats.copper_related_minerals.keys())
    alteration_type = classify_alteration_zone(stats)
    
    if language == "cn":
        # Chinese descriptions
        if stats.is_copper_alteration:
            mineral_list = "、".join(copper_minerals) if copper_minerals else "未知矿物"
            
            descriptions["binary"] = (
                f"是，该区域存在明显的蚀变特征。"
                f"检测到{mineral_list}，"
                f"铜矿相关矿物占比约{stats.copper_mineral_ratio:.1%}，"
                f"属于{_translate_alteration_type(alteration_type)}，"
                f"具有铜矿化潜力。"
            )
            
            descriptions["mineral"] = (
                f"该区域检测到以下矿物：{mineral_list}。"
                + (f"其中{stats.dominant_mineral}占比最高，约{stats.dominant_ratio:.1%}。" 
                   if stats.dominant_mineral else "")
            )
            
            descriptions["analysis"] = (
                f"该区域高光谱分析结果如下：\n\n"
                f"1. **蚀变矿物组合**: {mineral_list}\n"
                f"2. **蚀变类型**: {_translate_alteration_type(alteration_type)}\n"
                f"3. **矿化评估**: 铜矿相关矿物覆盖率{stats.copper_mineral_ratio:.1%}，"
                f"{'矿化潜力较高' if stats.copper_mineral_ratio > 0.2 else '具有一定矿化潜力'}\n"
                f"4. **找矿建议**: 建议开展进一步地质调查和化探工作\n\n"
                f"综合判断，该区域为铜矿蚀变带，值得重点关注。"
            )
        else:
            descriptions["binary"] = (
                f"否，该区域未检测到明显的铜矿蚀变特征，"
                f"以背景岩石为主。"
            )
            
            if mineral_names:
                non_bg_minerals = [m for m in mineral_names if m != "Background"]
                if non_bg_minerals:
                    descriptions["mineral"] = f"该区域主要为{_translate_alteration_type(alteration_type)}，检测到{','.join(non_bg_minerals)}。"
                else:
                    descriptions["mineral"] = "该区域未检测到明显的蚀变矿物。"
            else:
                descriptions["mineral"] = "该区域未检测到明显的蚀变矿物。"
            
            descriptions["analysis"] = (
                f"该区域高光谱分析显示，未发现明显的铜矿相关蚀变特征，"
                f"矿化潜力较低。"
            )
    
    else:  # English
        if stats.is_copper_alteration:
            mineral_list = ", ".join(copper_minerals) if copper_minerals else "unknown minerals"
            
            descriptions["binary"] = (
                f"Yes, this area shows significant alteration features. "
                f"Detected {mineral_list}, "
                f"copper-related minerals covering approximately {stats.copper_mineral_ratio:.1%}. "
                f"This is {alteration_type} alteration with copper mineralization potential."
            )
            
            descriptions["mineral"] = (
                f"Detected minerals in this area: {mineral_list}. "
                + (f"{stats.dominant_mineral} is dominant at {stats.dominant_ratio:.1%}."
                   if stats.dominant_mineral else "")
            )
            
            descriptions["analysis"] = (
                f"Hyperspectral analysis results:\n\n"
                f"1. **Mineral Assemblage**: {mineral_list}\n"
                f"2. **Alteration Type**: {alteration_type}\n"
                f"3. **Mineralization Assessment**: Copper-related mineral coverage is {stats.copper_mineral_ratio:.1%}\n"
                f"4. **Exploration Recommendation**: Further geological investigation recommended\n\n"
                f"Overall, this area represents a copper alteration zone worth detailed investigation."
            )
        else:
            descriptions["binary"] = (
                f"No, this area does not show significant copper-related alteration features. "
                f"Dominated by background rock."
            )
            descriptions["mineral"] = "No significant alteration minerals detected in this area."
            descriptions["analysis"] = (
                f"Hyperspectral analysis indicates no significant copper-related alteration features. "
                f"Low mineralization potential."
            )
    
    return descriptions


def _translate_alteration_type(alteration_type: str) -> str:
    """Translate alteration type to Chinese."""
    translations = {
        "Advanced Argillic": "高级泥化带",
        "Phyllic-Argillic": "绢英岩化-泥化带",
        "Propylitic": "青磐岩化带",
        "Argillic": "泥化带",
        "Chloritic": "绿泥石化带",
        "Weak Alteration": "弱蚀变带",
        "Background": "背景区",
    }
    return translations.get(alteration_type, alteration_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test with synthetic labels
    print("\n=== Testing Mineral Analysis ===")
    
    # Create test labels (simulate Cuprite minerals)
    np.random.seed(42)
    test_labels = np.random.choice([0, 1, 2, 3, 4], size=(224, 224), p=[0.5, 0.1, 0.15, 0.15, 0.1])
    
    # Analyze
    stats = analyze_tile_minerals(test_labels)
    print(f"Minerals detected: {stats.minerals}")
    print(f"Copper-related: {stats.copper_related_minerals}")
    print(f"Copper ratio: {stats.copper_mineral_ratio:.2%}")
    print(f"Is alteration zone: {stats.is_copper_alteration}")
    print(f"Dominant mineral: {stats.dominant_mineral} ({stats.dominant_ratio:.2%})")
    
    # Get descriptions
    print("\n=== Testing Descriptions ===")
    descriptions = get_mineral_description(stats, language="cn")
    for key, desc in descriptions.items():
        print(f"\n[{key}]:\n{desc}")

