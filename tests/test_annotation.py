"""
Unit tests for annotation modules.

Tests:
- Mineral analysis
- JSON dataset generation
"""

import sys
from pathlib import Path
import numpy as np
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


class TestMineralAnalysis:
    """Tests for mineral_analysis module."""
    
    def test_mineral_stats_class(self):
        """Test MineralStats dataclass."""
        from annotation.mineral_analysis import MineralStats
        
        stats = MineralStats(
            minerals={"Muscovite": 0.25, "Kaolinite": 0.15},
            copper_related_minerals={"Muscovite": 0.25, "Kaolinite": 0.15},
            copper_mineral_ratio=0.40,
            is_copper_alteration=True,
            dominant_mineral="Muscovite",
            dominant_ratio=0.25
        )
        
        assert stats.is_copper_alteration == True
        assert stats.copper_mineral_ratio == 0.40
        assert stats.dominant_mineral == "Muscovite"
        
    def test_mineral_stats_to_dict(self):
        """Test MineralStats serialization."""
        from annotation.mineral_analysis import MineralStats
        
        stats = MineralStats(
            minerals={"Muscovite": 0.25},
            copper_related_minerals={"Muscovite": 0.25},
            copper_mineral_ratio=0.25,
            is_copper_alteration=True
        )
        
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert "minerals" in d
        assert "is_copper_alteration" in d
        
    def test_analyze_tile_minerals_positive(self):
        """Test mineral analysis for positive case (copper alteration)."""
        from annotation.mineral_analysis import analyze_tile_minerals
        
        # Create labels with significant copper-related minerals
        # 1=Alunite, 2=Kaolinite, 3=Muscovite (all copper-related)
        labels = np.random.choice([0, 1, 2, 3], size=(224, 224), p=[0.5, 0.15, 0.2, 0.15])
        
        stats = analyze_tile_minerals(labels)
        
        assert stats.copper_mineral_ratio >= 0.4  # Should be around 50%
        assert stats.is_copper_alteration == True
        assert len(stats.copper_related_minerals) > 0
        
    def test_analyze_tile_minerals_negative(self):
        """Test mineral analysis for negative case (no alteration)."""
        from annotation.mineral_analysis import analyze_tile_minerals
        
        # Create labels mostly background
        labels = np.zeros((224, 224), dtype=np.uint8)
        
        stats = analyze_tile_minerals(labels)
        
        assert stats.copper_mineral_ratio < 0.10
        assert stats.is_copper_alteration == False
        
    def test_analyze_tile_minerals_dominant(self):
        """Test dominant mineral detection."""
        from annotation.mineral_analysis import analyze_tile_minerals
        
        # Create labels with Muscovite (3) as dominant
        labels = np.random.choice([0, 3], size=(224, 224), p=[0.4, 0.6])
        
        stats = analyze_tile_minerals(labels)
        
        assert stats.dominant_mineral == "Muscovite"
        assert stats.dominant_ratio > 0.5
        
    def test_classify_alteration_zone(self):
        """Test alteration zone classification."""
        from annotation.mineral_analysis import MineralStats, classify_alteration_zone
        
        # Phyllic-Argillic (with Muscovite)
        stats = MineralStats(
            minerals={"Muscovite": 0.3, "Kaolinite": 0.2},
            copper_related_minerals={"Muscovite": 0.3, "Kaolinite": 0.2},
            copper_mineral_ratio=0.5,
            is_copper_alteration=True
        )
        
        zone = classify_alteration_zone(stats)
        assert "Phyllic" in zone or "Argillic" in zone
        
    def test_classify_alteration_zone_propylitic(self):
        """Test propylitic alteration classification."""
        from annotation.mineral_analysis import MineralStats, classify_alteration_zone
        
        stats = MineralStats(
            minerals={"Chlorite": 0.25, "Epidote": 0.15},
            copper_related_minerals={"Chlorite": 0.25, "Epidote": 0.15},
            copper_mineral_ratio=0.4,
            is_copper_alteration=True
        )
        
        zone = classify_alteration_zone(stats)
        assert zone == "Propylitic"
        
    def test_get_mineral_description_cn(self):
        """Test Chinese mineral description generation."""
        from annotation.mineral_analysis import MineralStats, get_mineral_description
        
        stats = MineralStats(
            minerals={"Muscovite": 0.25, "Kaolinite": 0.15, "Background": 0.6},
            copper_related_minerals={"Muscovite": 0.25, "Kaolinite": 0.15},
            copper_mineral_ratio=0.40,
            is_copper_alteration=True,
            dominant_mineral="Muscovite",
            dominant_ratio=0.25
        )
        
        descriptions = get_mineral_description(stats, language="cn")
        
        assert "binary" in descriptions
        assert "mineral" in descriptions
        assert "analysis" in descriptions
        
        # Check Chinese content
        assert "是" in descriptions["binary"] or "存在" in descriptions["binary"]
        assert "Muscovite" in descriptions["mineral"] or "白云母" in descriptions["mineral"]
        
    def test_get_mineral_description_en(self):
        """Test English mineral description generation."""
        from annotation.mineral_analysis import MineralStats, get_mineral_description
        
        stats = MineralStats(
            minerals={"Background": 0.95},
            copper_related_minerals={},
            copper_mineral_ratio=0.0,
            is_copper_alteration=False
        )
        
        descriptions = get_mineral_description(stats, language="en")
        
        assert "No" in descriptions["binary"] or "not" in descriptions["binary"].lower()


class TestJSONGenerator:
    """Tests for json_generator module."""
    
    def test_generate_conversation_binary(self):
        """Test binary classification conversation generation."""
        from annotation.json_generator import generate_conversation
        from annotation.mineral_analysis import MineralStats
        
        stats = MineralStats(
            minerals={"Muscovite": 0.3},
            copper_related_minerals={"Muscovite": 0.3},
            copper_mineral_ratio=0.3,
            is_copper_alteration=True
        )
        
        item = generate_conversation("images/test.png", stats, task_type="binary")
        
        assert item.image == "images/test.png"
        assert len(item.conversations) == 2
        assert item.conversations[0]["from"] == "human"
        assert item.conversations[1]["from"] == "gpt"
        assert "<image>" in item.conversations[0]["value"]
        
    def test_generate_conversation_mineral(self):
        """Test mineral identification conversation generation."""
        from annotation.json_generator import generate_conversation
        from annotation.mineral_analysis import MineralStats
        
        stats = MineralStats(
            minerals={"Muscovite": 0.25, "Kaolinite": 0.15},
            copper_related_minerals={"Muscovite": 0.25, "Kaolinite": 0.15},
            copper_mineral_ratio=0.40,
            is_copper_alteration=True,
            dominant_mineral="Muscovite",
            dominant_ratio=0.25
        )
        
        item = generate_conversation("images/test.png", stats, task_type="mineral")
        
        assert len(item.conversations) == 2
        # Answer should mention minerals
        answer = item.conversations[1]["value"]
        assert "Muscovite" in answer or "白云母" in answer or "矿物" in answer
        
    def test_generate_conversation_analysis(self):
        """Test detailed analysis conversation generation."""
        from annotation.json_generator import generate_conversation
        from annotation.mineral_analysis import MineralStats
        
        stats = MineralStats(
            minerals={"Muscovite": 0.3},
            copper_related_minerals={"Muscovite": 0.3},
            copper_mineral_ratio=0.3,
            is_copper_alteration=True
        )
        
        item = generate_conversation("images/test.png", stats, task_type="analysis")
        
        assert len(item.conversations) == 2
        # Analysis should be detailed
        answer = item.conversations[1]["value"]
        assert len(answer) > 100  # Should be a detailed response
        
    def test_generate_dataset(self):
        """Test dataset generation from tile infos."""
        from annotation.json_generator import generate_dataset
        from annotation.mineral_analysis import MineralStats
        
        tile_infos = [
            {"row": 0, "col": 0, "filename": "tile_0000_0000.png"},
            {"row": 0, "col": 1, "filename": "tile_0000_0001.png"},
            {"row": 1, "col": 0, "filename": "tile_0001_0000.png"},
        ]
        
        stats_list = [
            MineralStats(
                minerals={"Muscovite": 0.3},
                copper_related_minerals={"Muscovite": 0.3},
                copper_mineral_ratio=0.3,
                is_copper_alteration=True
            ),
            MineralStats(
                minerals={"Background": 0.9},
                copper_related_minerals={},
                copper_mineral_ratio=0.0,
                is_copper_alteration=False
            ),
            MineralStats(
                minerals={"Kaolinite": 0.2},
                copper_related_minerals={"Kaolinite": 0.2},
                copper_mineral_ratio=0.2,
                is_copper_alteration=True
            ),
        ]
        
        dataset = generate_dataset(tile_infos, stats_list)
        
        assert len(dataset) == 3
        for item in dataset:
            assert item.image is not None
            assert len(item.conversations) >= 2
            
    def test_split_dataset(self):
        """Test dataset splitting."""
        from annotation.json_generator import split_dataset, ConversationItem
        
        # Create mock dataset
        dataset = [
            ConversationItem(
                image=f"img_{i}.png",
                conversations=[{"from": "human", "value": "test"}, {"from": "gpt", "value": "test"}],
                metadata={"is_copper_alteration": i % 2 == 0}
            )
            for i in range(100)
        ]
        
        train, val = split_dataset(dataset)
        
        assert len(train) + len(val) == 100
        assert len(train) == 80  # Default 0.8 ratio
        assert len(val) == 20
        
    def test_split_dataset_stratified(self):
        """Test stratified dataset splitting."""
        from annotation.json_generator import split_dataset, ConversationItem
        from config import SplitConfig
        
        # Create imbalanced dataset
        dataset = []
        for i in range(80):
            dataset.append(ConversationItem(
                image=f"img_{i}.png",
                conversations=[{"from": "human", "value": "test"}, {"from": "gpt", "value": "test"}],
                metadata={"is_copper_alteration": False}
            ))
        for i in range(20):
            dataset.append(ConversationItem(
                image=f"img_pos_{i}.png",
                conversations=[{"from": "human", "value": "test"}, {"from": "gpt", "value": "test"}],
                metadata={"is_copper_alteration": True}
            ))
        
        split_config = SplitConfig(stratify=True)
        train, val = split_dataset(dataset, split_config)
        
        # Both splits should have positive samples
        train_positive = sum(1 for item in train if item.metadata.get("is_copper_alteration"))
        val_positive = sum(1 for item in val if item.metadata.get("is_copper_alteration"))
        
        assert train_positive > 0
        assert val_positive > 0
        
    def test_save_dataset(self):
        """Test saving dataset to JSON."""
        from annotation.json_generator import save_dataset, ConversationItem
        
        dataset = [
            ConversationItem(
                image="images/test1.png",
                conversations=[
                    {"from": "human", "value": "<image>\n测试问题？"},
                    {"from": "gpt", "value": "测试回答。"}
                ],
                metadata={"is_copper_alteration": True}
            ),
            ConversationItem(
                image="images/test2.png",
                conversations=[
                    {"from": "human", "value": "<image>\n另一个问题？"},
                    {"from": "gpt", "value": "另一个回答。"}
                ],
                metadata={"is_copper_alteration": False}
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.json"
            save_dataset(dataset, output_path, include_metadata=True)
            
            assert output_path.exists()
            
            with open(output_path, encoding='utf-8') as f:
                loaded = json.load(f)
                
            assert len(loaded) == 2
            assert loaded[0]["image"] == "images/test1.png"
            assert loaded[0]["metadata"]["is_copper_alteration"] == True
            
    def test_validate_dataset(self):
        """Test dataset validation."""
        from annotation.json_generator import validate_dataset, ConversationItem
        
        # Valid dataset
        valid_dataset = [
            ConversationItem(
                image="images/test.png",
                conversations=[
                    {"from": "human", "value": "<image>\n问题"},
                    {"from": "gpt", "value": "回答"}
                ],
                metadata={"task_type": "binary", "is_copper_alteration": True}
            )
        ]
        
        report = validate_dataset(valid_dataset)
        assert report["is_valid"] == True
        assert report["valid_items"] == 1
        
    def test_validate_dataset_invalid(self):
        """Test validation catches invalid items."""
        from annotation.json_generator import validate_dataset, ConversationItem
        
        # Invalid: missing image tag
        invalid_dataset = [
            ConversationItem(
                image="images/test.png",
                conversations=[
                    {"from": "human", "value": "没有图像标签的问题"},
                    {"from": "gpt", "value": "回答"}
                ]
            )
        ]
        
        report = validate_dataset(invalid_dataset)
        assert len(report["errors"]) > 0


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_tile_to_json_pipeline(self):
        """Test the complete pipeline from tile analysis to JSON generation."""
        from annotation.mineral_analysis import analyze_tile_minerals
        from annotation.json_generator import generate_conversation, validate_dataset
        
        # Simulate a tile with minerals
        labels = np.random.choice([0, 1, 2, 3], size=(224, 224), p=[0.5, 0.15, 0.2, 0.15])
        
        # Analyze
        stats = analyze_tile_minerals(labels)
        
        # Generate conversation
        item = generate_conversation("images/clay/tile_0001.png", stats, task_type="binary")
        
        # Validate
        report = validate_dataset([item])
        
        assert report["is_valid"] == True
        assert item.conversations[0]["from"] == "human"
        assert item.conversations[1]["from"] == "gpt"
        assert len(item.conversations[1]["value"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

