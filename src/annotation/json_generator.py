"""
JSON generator module for creating Qwen3-VL compatible training data.

Generates multi-turn conversations for:
- Binary classification (mineralization detection)
- Mineral identification (multi-label)
- Detailed analysis (open VQA)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from .mineral_analysis import MineralStats, get_mineral_description

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TaskConfig, SplitConfig, default_config

logger = logging.getLogger(__name__)


@dataclass
class ConversationItem:
    """A single conversation item."""
    image: str
    conversations: List[Dict[str, str]]
    metadata: Optional[Dict[str, Any]] = None


def generate_conversation(
    image_path: str,
    stats: MineralStats,
    task_type: str = "binary",
    task_config: Optional[TaskConfig] = None,
    include_image_tag: bool = True
) -> ConversationItem:
    """
    Generate a conversation for a single tile.
    
    Args:
        image_path: Relative path to the tile image
        stats: MineralStats from mineral analysis
        task_type: "binary", "mineral", or "analysis"
        task_config: Task configuration (uses default if None)
        include_image_tag: Whether to include <image> tag in prompt
        
    Returns:
        ConversationItem object
    """
    if task_config is None:
        task_config = default_config.tasks
    
    # Get language-specific prompts
    if task_config.language in ["cn", "both"]:
        language = "cn"
    else:
        language = "en"
    
    # Select prompt based on task type
    if task_type == "binary":
        prompts = task_config.binary_prompts_cn if language == "cn" else task_config.binary_prompts_en
    elif task_type == "mineral":
        prompts = task_config.mineral_prompts_cn if language == "cn" else task_config.mineral_prompts_en
    elif task_type == "analysis":
        prompts = task_config.analysis_prompts_cn
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Randomly select a prompt
    prompt = random.choice(prompts)
    
    # Add image tag if needed
    if include_image_tag:
        prompt = f"<image>\n{prompt}"
    
    # Get answer from mineral descriptions
    descriptions = get_mineral_description(stats, language=language)
    answer = descriptions.get(task_type, descriptions.get("binary", ""))
    
    # Create conversation
    conversations = [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": answer}
    ]
    
    return ConversationItem(
        image=image_path,
        conversations=conversations,
        metadata={
            "task_type": task_type,
            "is_copper_alteration": stats.is_copper_alteration,
            "copper_mineral_ratio": stats.copper_mineral_ratio,
            "dominant_mineral": stats.dominant_mineral,
        }
    )


def generate_multi_turn_conversation(
    image_path: str,
    stats: MineralStats,
    task_config: Optional[TaskConfig] = None
) -> ConversationItem:
    """
    Generate a multi-turn conversation covering multiple aspects.
    
    Args:
        image_path: Relative path to tile image
        stats: MineralStats from analysis
        task_config: Task configuration
        
    Returns:
        ConversationItem with multi-turn conversation
    """
    if task_config is None:
        task_config = default_config.tasks
    
    language = "cn" if task_config.language in ["cn", "both"] else "en"
    descriptions = get_mineral_description(stats, language=language)
    
    conversations = []
    
    # Turn 1: Binary question
    if language == "cn":
        q1 = random.choice(task_config.binary_prompts_cn)
        conversations.append({"from": "human", "value": f"<image>\n{q1}"})
        conversations.append({"from": "gpt", "value": descriptions["binary"]})
        
        # Turn 2: Follow-up mineral question (if alteration detected)
        if stats.is_copper_alteration:
            q2 = "具体检测到哪些蚀变矿物？"
            conversations.append({"from": "human", "value": q2})
            conversations.append({"from": "gpt", "value": descriptions["mineral"]})
    else:
        q1 = random.choice(task_config.binary_prompts_en)
        conversations.append({"from": "human", "value": f"<image>\n{q1}"})
        conversations.append({"from": "gpt", "value": descriptions["binary"]})
        
        if stats.is_copper_alteration:
            q2 = "What specific alteration minerals were detected?"
            conversations.append({"from": "human", "value": q2})
            conversations.append({"from": "gpt", "value": descriptions["mineral"]})
    
    return ConversationItem(
        image=image_path,
        conversations=conversations,
        metadata={
            "task_type": "multi_turn",
            "is_copper_alteration": stats.is_copper_alteration,
            "copper_mineral_ratio": stats.copper_mineral_ratio,
        }
    )


def generate_dataset(
    tile_infos: List[Dict[str, Any]],
    stats_list: List[MineralStats],
    image_base_path: str = "images/clay",
    task_config: Optional[TaskConfig] = None,
    balanced: bool = True
) -> List[ConversationItem]:
    """
    Generate dataset from tile information and mineral statistics.
    
    Args:
        tile_infos: List of tile information dictionaries
        stats_list: List of MineralStats for each tile
        image_base_path: Base path for images
        task_config: Task configuration
        balanced: Whether to balance task types
        
    Returns:
        List of ConversationItem objects
    """
    if task_config is None:
        task_config = default_config.tasks
    
    if len(tile_infos) != len(stats_list):
        raise ValueError("tile_infos and stats_list must have same length")
    
    dataset = []
    
    # Task weights
    weights = task_config.task_weights
    task_types = list(weights.keys())
    task_probs = [weights[t] for t in task_types]
    
    for tile_info, stats in zip(tile_infos, stats_list):
        filename = tile_info.get("filename", f"tile_{tile_info['row']:04d}_{tile_info['col']:04d}.png")
        image_path = f"{image_base_path}/{filename}"
        
        if balanced:
            # Sample task type based on weights
            task_type = random.choices(task_types, weights=task_probs, k=1)[0]
        else:
            # Generate all task types for each tile
            for task_type in task_types:
                item = generate_conversation(image_path, stats, task_type, task_config)
                dataset.append(item)
            continue
        
        item = generate_conversation(image_path, stats, task_type, task_config)
        dataset.append(item)
    
    logger.info(f"Generated {len(dataset)} conversation items")
    return dataset


def split_dataset(
    dataset: List[ConversationItem],
    split_config: Optional[SplitConfig] = None
) -> Tuple[List[ConversationItem], List[ConversationItem]]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: Full dataset
        split_config: Split configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if split_config is None:
        split_config = default_config.split
    
    # Set random seed for reproducibility
    random.seed(split_config.random_seed)
    
    # Shuffle a copy
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    
    if split_config.stratify:
        # Stratified split by copper alteration status
        positive = [item for item in shuffled if item.metadata and item.metadata.get("is_copper_alteration", False)]
        negative = [item for item in shuffled if not item.metadata or not item.metadata.get("is_copper_alteration", False)]
        
        # Split each group
        pos_split = int(len(positive) * split_config.train_ratio)
        neg_split = int(len(negative) * split_config.train_ratio)
        
        train = positive[:pos_split] + negative[:neg_split]
        val = positive[pos_split:] + negative[neg_split:]
        
        random.shuffle(train)
        random.shuffle(val)
    else:
        # Simple random split
        split_idx = int(len(shuffled) * split_config.train_ratio)
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]
    
    logger.info(f"Split dataset: train={len(train)}, val={len(val)}")
    return train, val


def save_dataset(
    dataset: List[ConversationItem],
    output_path: Path,
    include_metadata: bool = False
) -> None:
    """
    Save dataset to JSON file.
    
    Args:
        dataset: List of ConversationItem objects
        output_path: Output file path
        include_metadata: Whether to include metadata in output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list of dicts
    data = []
    for item in dataset:
        entry = {
            "image": item.image,
            "conversations": item.conversations,
        }
        if include_metadata and item.metadata:
            entry["metadata"] = item.metadata
        data.append(entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(data)} items to {output_path}")


def validate_dataset(
    dataset: List[ConversationItem]
) -> Dict[str, Any]:
    """
    Validate dataset format and content.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        Validation report dictionary
    """
    report = {
        "total_items": len(dataset),
        "valid_items": 0,
        "errors": [],
        "task_distribution": {},
        "alteration_distribution": {"positive": 0, "negative": 0},
    }
    
    for i, item in enumerate(dataset):
        errors = []
        
        # Check image path
        if not item.image:
            errors.append(f"Item {i}: Missing image path")
        
        # Check conversations
        if not item.conversations:
            errors.append(f"Item {i}: Empty conversations")
        else:
            for j, conv in enumerate(item.conversations):
                if "from" not in conv or "value" not in conv:
                    errors.append(f"Item {i}, conv {j}: Missing 'from' or 'value'")
                if conv.get("from") not in ["human", "gpt"]:
                    errors.append(f"Item {i}, conv {j}: Invalid 'from' value: {conv.get('from')}")
                if not conv.get("value"):
                    errors.append(f"Item {i}, conv {j}: Empty 'value'")
        
        # Check for <image> tag in first human message
        if item.conversations:
            first_human = next((c for c in item.conversations if c.get("from") == "human"), None)
            if first_human and "<image>" not in first_human.get("value", ""):
                errors.append(f"Item {i}: Missing <image> tag in first human message")
        
        if errors:
            report["errors"].extend(errors)
        else:
            report["valid_items"] += 1
        
        # Track distributions
        if item.metadata:
            task_type = item.metadata.get("task_type", "unknown")
            report["task_distribution"][task_type] = report["task_distribution"].get(task_type, 0) + 1
            
            if item.metadata.get("is_copper_alteration"):
                report["alteration_distribution"]["positive"] += 1
            else:
                report["alteration_distribution"]["negative"] += 1
    
    report["is_valid"] = len(report["errors"]) == 0
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    print("\n=== Testing JSON Generator ===")
    
    # Create test stats
    from mineral_analysis import MineralStats
    
    stats_positive = MineralStats(
        minerals={"Muscovite": 0.25, "Kaolinite": 0.15, "Background": 0.6},
        copper_related_minerals={"Muscovite": 0.25, "Kaolinite": 0.15},
        copper_mineral_ratio=0.40,
        is_copper_alteration=True,
        dominant_mineral="Muscovite",
        dominant_ratio=0.25,
    )
    
    stats_negative = MineralStats(
        minerals={"Background": 0.95, "Calcite": 0.05},
        copper_related_minerals={},
        copper_mineral_ratio=0.0,
        is_copper_alteration=False,
    )
    
    # Generate conversations
    print("\n--- Binary Task (Positive) ---")
    item = generate_conversation("images/clay/tile_0001.png", stats_positive, "binary")
    print(f"Image: {item.image}")
    for conv in item.conversations:
        print(f"[{conv['from']}]: {conv['value'][:100]}...")
    
    print("\n--- Mineral Task (Positive) ---")
    item = generate_conversation("images/clay/tile_0001.png", stats_positive, "mineral")
    for conv in item.conversations:
        print(f"[{conv['from']}]: {conv['value'][:100]}...")
    
    print("\n--- Binary Task (Negative) ---")
    item = generate_conversation("images/clay/tile_0002.png", stats_negative, "binary")
    for conv in item.conversations:
        print(f"[{conv['from']}]: {conv['value']}")
    
    # Test dataset generation
    print("\n=== Testing Dataset Generation ===")
    tile_infos = [{"row": i, "col": 0, "filename": f"tile_{i:04d}_0000.png"} for i in range(10)]
    stats_list = [stats_positive if i % 2 == 0 else stats_negative for i in range(10)]
    
    dataset = generate_dataset(tile_infos, stats_list)
    print(f"Generated {len(dataset)} items")
    
    # Validate
    report = validate_dataset(dataset)
    print(f"Validation: valid={report['is_valid']}, errors={len(report['errors'])}")
    print(f"Task distribution: {report['task_distribution']}")
    print(f"Alteration distribution: {report['alteration_distribution']}")

