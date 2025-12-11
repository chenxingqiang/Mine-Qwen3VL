"""
Annotation module for generating Qwen3-VL compatible training data.

This module provides:
- Mineral analysis from ground truth labels
- Multi-task prompt generation
- JSON format output for Qwen3-VL fine-tuning
"""

from .mineral_analysis import (
    load_ground_truth,
    analyze_tile_minerals,
    classify_alteration_zone,
    get_mineral_description,
    MineralStats,
)

from .json_generator import (
    generate_conversation,
    generate_dataset,
    save_dataset,
    split_dataset,
    validate_dataset,
)

__all__ = [
    # Mineral analysis
    "load_ground_truth",
    "analyze_tile_minerals",
    "classify_alteration_zone",
    "get_mineral_description",
    "MineralStats",
    # JSON generation
    "generate_conversation",
    "generate_dataset",
    "save_dataset",
    "split_dataset",
    "validate_dataset",
]

