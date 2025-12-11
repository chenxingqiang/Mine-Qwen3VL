#!/usr/bin/env python3
"""
Convert training data to Qwen3-VL compatible format.

Changes:
- "images" -> "image" (and extract from list if single image)
- "from": "user" -> "from": "human"
- "from": "assistant" -> "from": "gpt"
- Remove metadata field (not needed for training)
"""

import json
import sys
from pathlib import Path


def convert_entry(entry: dict) -> dict:
    """Convert a single entry to Qwen3-VL format."""
    converted = {}
    
    # Convert images field
    if "images" in entry:
        images = entry["images"]
        if isinstance(images, list):
            if len(images) == 1:
                converted["image"] = images[0]
            else:
                converted["image"] = images
        else:
            converted["image"] = images
    elif "image" in entry:
        converted["image"] = entry["image"]
    
    # Convert conversations
    if "conversations" in entry:
        converted["conversations"] = []
        for msg in entry["conversations"]:
            new_msg = {}
            # Convert role names
            from_role = msg.get("from", "")
            if from_role == "user":
                new_msg["from"] = "human"
            elif from_role == "assistant":
                new_msg["from"] = "gpt"
            else:
                new_msg["from"] = from_role
            
            new_msg["value"] = msg.get("value", "")
            converted["conversations"].append(new_msg)
    
    # Optionally keep metadata for debugging (commented out for clean format)
    # if "metadata" in entry:
    #     converted["metadata"] = entry["metadata"]
    
    return converted


def convert_file(input_path: Path, output_path: Path):
    """Convert a JSON file to Qwen3-VL format."""
    print(f"Converting: {input_path} -> {output_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    converted_data = [convert_entry(entry) for entry in data]
    
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Converted {len(converted_data)} entries")


def main():
    # Project paths
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "data" / "cuprite_dataset"
    
    # Convert all dataset files
    files_to_convert = [
        ("train_real.json", "train_qwenvl.json"),
        ("val_real.json", "val_qwenvl.json"),
        ("test_real.json", "test_qwenvl.json"),
    ]
    
    for input_name, output_name in files_to_convert:
        input_path = dataset_dir / input_name
        output_path = dataset_dir / output_name
        
        if input_path.exists():
            convert_file(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found, skipping")
    
    print("\nConversion complete!")
    print(f"\nQwen3-VL compatible files saved to: {dataset_dir}")
    
    # Show sample
    print("\n--- Sample entry from train_qwenvl.json ---")
    with open(dataset_dir / "train_qwenvl.json", 'r') as f:
        data = json.load(f)
        if data:
            print(json.dumps(data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

