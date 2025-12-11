#!/usr/bin/env python3
"""
End-to-end pipeline verification script.

This script runs the complete data preparation pipeline
using synthetic data and verifies all outputs.
"""

import sys
from pathlib import Path
import numpy as np
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_config, Config
from preprocessing import (
    get_wavelengths,
    apply_band_combination,
    normalize_percentile,
    remove_bad_bands,
    to_uint8,
    save_tiles,
)
from annotation import (
    analyze_tile_minerals,
    generate_dataset,
    save_dataset,
    split_dataset,
    validate_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineVerifier:
    """Verifies the complete data preparation pipeline."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_config()
        self.results = {
            "steps": {},
            "errors": [],
            "warnings": [],
            "passed": True
        }
        
    def log_step(self, step_name: str, status: str, details: dict = None):
        """Log a step result."""
        self.results["steps"][step_name] = {
            "status": status,
            "details": details or {}
        }
        
        if status == "FAIL":
            self.results["passed"] = False
        
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        logger.info(f"{icon} {step_name}: {status}")
        if details:
            for key, value in details.items():
                logger.info(f"   {key}: {value}")
    
    def verify_data_loading(self, data_dir: Path) -> tuple:
        """Step 1: Verify data loading."""
        step_name = "Data Loading"
        
        try:
            # Load synthetic data
            hs_file = data_dir / "synthetic_hyperspectral.npy"
            wl_file = data_dir / "synthetic_wavelengths.npy"
            labels_file = data_dir / "synthetic_labels.npy"
            
            if not all(f.exists() for f in [hs_file, wl_file, labels_file]):
                self.log_step(step_name, "FAIL", {"error": "Data files not found"})
                return None, None, None
            
            data = np.load(hs_file)
            wavelengths = np.load(wl_file)
            labels = np.load(labels_file)
            
            self.log_step(step_name, "PASS", {
                "data_shape": str(data.shape),
                "wavelengths": f"{wavelengths[0]:.1f}-{wavelengths[-1]:.1f}nm",
                "labels_unique": str(np.unique(labels).tolist())
            })
            
            return data, wavelengths, labels
            
        except Exception as e:
            self.log_step(step_name, "FAIL", {"error": str(e)})
            self.results["errors"].append(f"Data loading: {e}")
            return None, None, None
    
    def verify_preprocessing(self, data: np.ndarray, wavelengths: np.ndarray) -> tuple:
        """Step 2: Verify preprocessing (bad band removal, normalization)."""
        step_name = "Preprocessing"
        
        try:
            # Remove bad bands
            bad_bands = self.config.bands.water_absorption_bands + self.config.bands.noisy_bands
            bad_bands = [b for b in sorted(set(bad_bands)) if 0 <= b < len(wavelengths)]
            
            cleaned_data, cleaned_wl = remove_bad_bands(data, bad_bands, wavelengths)
            
            self.log_step(step_name, "PASS", {
                "original_bands": data.shape[-1],
                "after_removal": cleaned_data.shape[-1],
                "removed": len(bad_bands)
            })
            
            return cleaned_data, cleaned_wl
            
        except Exception as e:
            self.log_step(step_name, "FAIL", {"error": str(e)})
            self.results["errors"].append(f"Preprocessing: {e}")
            return None, None
    
    def verify_band_combination(self, data: np.ndarray, wavelengths: np.ndarray) -> dict:
        """Step 3: Verify RGB band combination."""
        step_name = "Band Combination"
        
        try:
            rgb_images = {}
            
            for combo_name, combo in self.config.bands.band_combinations.items():
                rgb = apply_band_combination(
                    data, wavelengths, combo,
                    normalize=True,
                    percentile_clip=(2, 98)
                )
                rgb_images[combo_name] = rgb
            
            self.log_step(step_name, "PASS", {
                "combinations": list(rgb_images.keys()),
                "rgb_shape": str(list(rgb_images.values())[0].shape),
                "value_range": f"[{list(rgb_images.values())[0].min():.3f}, {list(rgb_images.values())[0].max():.3f}]"
            })
            
            return rgb_images
            
        except Exception as e:
            self.log_step(step_name, "FAIL", {"error": str(e)})
            self.results["errors"].append(f"Band combination: {e}")
            return None
    
    def verify_tiling(self, rgb_image: np.ndarray, labels: np.ndarray, output_dir: Path) -> list:
        """Step 4: Verify image tiling."""
        step_name = "Tiling"
        
        try:
            # Convert to uint8
            rgb_uint8 = to_uint8(rgb_image)
            
            # Save tiles
            tile_dir = output_dir / "tiles"
            tile_infos = save_tiles(
                rgb_uint8,
                tile_dir,
                tile_size=self.config.tiling.tile_size,
                stride=self.config.tiling.stride,
                min_valid_ratio=self.config.tiling.min_valid_ratio,
                save_metadata=True
            )
            
            # Analyze minerals for each tile
            for info in tile_infos:
                y1, y2 = info.y_start, info.y_end
                x1, x2 = info.x_start, info.x_end
                
                # Handle edge cases
                y2 = min(y2, labels.shape[0])
                x2 = min(x2, labels.shape[1])
                
                tile_labels = labels[y1:y2, x1:x2]
                stats = analyze_tile_minerals(tile_labels, self.config.minerals)
                info.stats = stats
            
            # Count alteration zones
            n_positive = sum(1 for info in tile_infos if hasattr(info, 'stats') and info.stats.is_copper_alteration)
            
            self.log_step(step_name, "PASS", {
                "total_tiles": len(tile_infos),
                "positive_tiles": n_positive,
                "tile_size": self.config.tiling.tile_size,
                "stride": self.config.tiling.stride
            })
            
            return tile_infos
            
        except Exception as e:
            self.log_step(step_name, "FAIL", {"error": str(e)})
            self.results["errors"].append(f"Tiling: {e}")
            return None
    
    def verify_annotation(self, tile_infos: list, output_dir: Path) -> list:
        """Step 5: Verify annotation generation."""
        step_name = "Annotation Generation"
        
        try:
            # Prepare data for dataset generation
            tile_dicts = []
            stats_list = []
            
            for info in tile_infos:
                tile_dicts.append({
                    "filename": info.filename,
                    "row": info.row,
                    "col": info.col,
                })
                stats_list.append(info.stats)
            
            # Generate dataset
            dataset = generate_dataset(
                tile_dicts,
                stats_list,
                image_base_path="tiles",
                task_config=self.config.tasks,
                balanced=True
            )
            
            # Validate
            report = validate_dataset(dataset)
            
            self.log_step(step_name, "PASS", {
                "total_items": len(dataset),
                "valid_items": report["valid_items"],
                "task_distribution": str(report.get("task_distribution", {})),
                "validation_errors": len(report["errors"])
            })
            
            return dataset
            
        except Exception as e:
            self.log_step(step_name, "FAIL", {"error": str(e)})
            self.results["errors"].append(f"Annotation: {e}")
            return None
    
    def verify_dataset_split(self, dataset: list, output_dir: Path) -> tuple:
        """Step 6: Verify dataset splitting and saving."""
        step_name = "Dataset Split & Save"
        
        try:
            # Split
            train_dataset, val_dataset = split_dataset(dataset, self.config.split)
            
            # Save
            save_dataset(train_dataset, output_dir / "train.json", include_metadata=True)
            save_dataset(val_dataset, output_dir / "val.json", include_metadata=True)
            
            # Verify saved files
            with open(output_dir / "train.json", encoding='utf-8') as f:
                train_loaded = json.load(f)
            
            with open(output_dir / "val.json", encoding='utf-8') as f:
                val_loaded = json.load(f)
            
            self.log_step(step_name, "PASS", {
                "train_items": len(train_loaded),
                "val_items": len(val_loaded),
                "train_file": str(output_dir / "train.json"),
                "val_file": str(output_dir / "val.json")
            })
            
            return train_dataset, val_dataset
            
        except Exception as e:
            self.log_step(step_name, "FAIL", {"error": str(e)})
            self.results["errors"].append(f"Dataset split: {e}")
            return None, None
    
    def verify_output_format(self, output_dir: Path) -> bool:
        """Step 7: Verify output format is Qwen3-VL compatible."""
        step_name = "Qwen3-VL Format Validation"
        
        try:
            train_file = output_dir / "train.json"
            
            with open(train_file, encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                self.log_step(step_name, "FAIL", {"error": "Empty dataset"})
                return False
            
            # Check format
            sample = data[0]
            
            checks = {
                "has_image_field": "image" in sample,
                "has_conversations": "conversations" in sample,
                "conversations_is_list": isinstance(sample.get("conversations"), list),
                "has_human_turn": any(c.get("from") == "human" for c in sample.get("conversations", [])),
                "has_gpt_turn": any(c.get("from") == "gpt" for c in sample.get("conversations", [])),
            }
            
            # Check first human message has <image> tag
            first_human = next((c for c in sample.get("conversations", []) if c.get("from") == "human"), None)
            checks["has_image_tag"] = first_human and "<image>" in first_human.get("value", "")
            
            all_passed = all(checks.values())
            
            self.log_step(step_name, "PASS" if all_passed else "FAIL", checks)
            
            return all_passed
            
        except Exception as e:
            self.log_step(step_name, "FAIL", {"error": str(e)})
            self.results["errors"].append(f"Format validation: {e}")
            return False
    
    def run(self, data_dir: Path = None, output_dir: Path = None):
        """Run complete pipeline verification."""
        logger.info("=" * 60)
        logger.info("Pipeline Verification Starting")
        logger.info("=" * 60)
        
        if data_dir is None:
            data_dir = self.config.paths.raw_data_dir / "synthetic"
        if output_dir is None:
            output_dir = self.config.paths.dataset_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        data, wavelengths, labels = self.verify_data_loading(data_dir)
        if data is None:
            return self.results
        
        # Step 2: Preprocess
        cleaned_data, cleaned_wl = self.verify_preprocessing(data, wavelengths)
        if cleaned_data is None:
            return self.results
        
        # Step 3: Band combination
        rgb_images = self.verify_band_combination(cleaned_data, cleaned_wl)
        if rgb_images is None:
            return self.results
        
        # Use primary combination
        primary = self.config.bands.primary_combination
        rgb_image = rgb_images[primary]
        
        # Step 4: Tiling
        tile_infos = self.verify_tiling(rgb_image, labels, output_dir)
        if tile_infos is None:
            return self.results
        
        # Step 5: Annotation
        dataset = self.verify_annotation(tile_infos, output_dir)
        if dataset is None:
            return self.results
        
        # Step 6: Split and save
        train, val = self.verify_dataset_split(dataset, output_dir)
        if train is None:
            return self.results
        
        # Step 7: Format validation
        self.verify_output_format(output_dir)
        
        # Summary
        logger.info("=" * 60)
        logger.info("Pipeline Verification Complete")
        logger.info("=" * 60)
        
        passed = sum(1 for s in self.results["steps"].values() if s["status"] == "PASS")
        total = len(self.results["steps"])
        
        if self.results["passed"]:
            logger.info(f"✅ All {passed}/{total} steps passed!")
        else:
            logger.error(f"❌ {total - passed}/{total} steps failed")
            for error in self.results["errors"]:
                logger.error(f"  - {error}")
        
        return self.results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify pipeline with synthetic data")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data first")
    args = parser.parse_args()
    
    config = get_config()
    
    # Generate synthetic data if requested or not present
    data_dir = config.paths.raw_data_dir / "synthetic"
    
    if args.generate or not (data_dir / "synthetic_hyperspectral.npy").exists():
        logger.info("Generating synthetic data...")
        from generate_synthetic_data import main as generate_data
        generate_data()
    
    # Run verification
    verifier = PipelineVerifier(config)
    results = verifier.run(data_dir=data_dir)
    
    # Save verification report
    report_path = config.paths.dataset_dir / "verification_report.json"
    
    # Convert results to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(report_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    
    logger.info(f"\nVerification report saved to: {report_path}")
    
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

