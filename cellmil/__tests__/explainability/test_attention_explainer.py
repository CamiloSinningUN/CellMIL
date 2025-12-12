"""
Comprehensive test suite for the Advanced Attention Explainer.

This test suite validates different model types, configurations, and edge cases.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from torch_geometric.data import Data  # type: ignore
from typing import Any

from cellmil.explainability.explain import Explain
import lightning as pl
from cellmil.interfaces.ExplainerCreatorConfig import (
    ExplainerCreatorConfig,
    ExplainMethod,
    VisualizationMode,
    AttentionAggregation,
)
from cellmil.explainability.core.attention_extractor import (
    AttentionResult,
    AttentionExtractorFactory,
)


# Mock models for testing
class LitCLAM(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        # Return mock attention weights [n_classes, n_instances]
        return torch.rand(2, x.shape[0])  # 2 classes


class LitAttentionDeepMIL(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        # Return mock attention weights [attention_branches, n_instances]
        return torch.rand(2, x.shape[0])  # 2 attention heads


class LitGraphMIL(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def get_attention_weights(self, data: Data) -> dict[str, torch.Tensor]:
        # Return mock GraphMIL attention weights
        num_nodes = data.x.shape[0]  # type: ignore
        num_edges = data.edge_index.shape[1]  # type: ignore

        return {
            "gnn_attention_layer_0": torch.rand(num_edges, 4),  # 4 heads
            "gnn_attention_layer_1": torch.rand(num_edges, 4),
            "pooling_attention": torch.rand(1, num_nodes),
        }


class TestAttentionExtractorFactory:
    """Test the attention extractor factory."""

    def test_create_clam_extractor(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention, output_path=Path("/tmp/test")
        )

        extractor = AttentionExtractorFactory.create_extractor("clam", config)
        assert extractor.__class__.__name__ == "CLAMAttentionExtractor"

    def test_create_attentiondeepmil_extractor(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention, output_path=Path("/tmp/test")
        )

        extractor = AttentionExtractorFactory.create_extractor(
            "attentiondeepmil", config
        )
        assert extractor.__class__.__name__ == "AttentionDeepMILExtractor"

    def test_create_graphmil_extractor(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention, output_path=Path("/tmp/test")
        )

        extractor = AttentionExtractorFactory.create_extractor("graphmil", config)
        assert extractor.__class__.__name__ == "GraphMILAttentionExtractor"

    def test_unsupported_model_type(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention, output_path=Path("/tmp/test")
        )

        with pytest.raises(ValueError, match="Unsupported model type"):
            AttentionExtractorFactory.create_extractor("unsupported", config)


class TestCLAMAttentionExtractor:
    """Test CLAM attention extraction."""

    def test_clam_extraction_all_classes(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=Path("/tmp/test"),
            class_index=None,  # All classes
        )

        extractor = AttentionExtractorFactory.create_extractor("clam", config)
        model = LitCLAM()
        data = torch.randn(10, 256)

        result = extractor.extract(model, data)

        assert isinstance(result, AttentionResult)
        assert result.model_type == "CLAM"
        assert "class_0_attention" in result.attention_weights
        assert "class_1_attention" in result.attention_weights
        assert result.metadata["n_classes"] == 2
        assert result.metadata["n_instances"] == 10

    def test_clam_extraction_specific_class(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=Path("/tmp/test"),
            class_index=0,  # Specific class
        )

        extractor = AttentionExtractorFactory.create_extractor("clam", config)
        model = LitCLAM()
        data = torch.randn(15, 256)

        result = extractor.extract(model, data)

        assert "class_attention" in result.attention_weights
        assert result.attention_weights["class_attention"].shape == (1, 15)
        assert result.metadata["selected_class"] == 0


class TestAttentionDeepMILExtractor:
    """Test AttentionDeepMIL attention extraction."""

    def test_attentiondeepmil_extraction_all_heads(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=Path("/tmp/test"),
            attention_head=None,  # All heads
        )

        extractor = AttentionExtractorFactory.create_extractor(
            "attentiondeepmil", config
        )
        model = LitAttentionDeepMIL()
        data = torch.randn(20, 512)

        result = extractor.extract(model, data)

        assert isinstance(result, AttentionResult)
        assert result.model_type == "AttentionDeepMIL"
        assert "mean_attention" in result.attention_weights
        assert "head_0_attention" in result.attention_weights
        assert "head_1_attention" in result.attention_weights
        assert result.metadata["n_heads"] == 2

    def test_attentiondeepmil_extraction_specific_head(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=Path("/tmp/test"),
            attention_head=1,  # Specific head
        )

        extractor = AttentionExtractorFactory.create_extractor(
            "attentiondeepmil", config
        )
        model = LitAttentionDeepMIL()
        data = torch.randn(12, 512)

        result = extractor.extract(model, data)

        assert "head_attention" in result.attention_weights
        assert result.attention_weights["head_attention"].shape == (1, 12)
        assert result.metadata["selected_head"] == 1


class TestGraphMILAttentionExtractor:
    """Test GraphMIL attention extraction."""

    def test_graphmil_extraction_combined(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=Path("/tmp/test"),
            attention_aggregation=AttentionAggregation.combined,
        )

        extractor = AttentionExtractorFactory.create_extractor("graphmil", config)
        model = LitGraphMIL()

        # Create sample graph data
        x = torch.randn(15, 256)
        edge_index = torch.randint(0, 15, (2, 30))
        data = Data(x=x, edge_index=edge_index)

        result = extractor.extract(model, data)

        assert isinstance(result, AttentionResult)
        assert result.model_type == "GraphMIL"
        assert "gnn_attention_layer_0" in result.attention_weights
        assert "gnn_attention_layer_1" in result.attention_weights
        assert "pooling_attention" in result.attention_weights
        assert result.metadata["gnn_layers"] == 2
        assert result.metadata["has_gnn_attention"]
        assert result.metadata["has_pooling_attention"]

    def test_graphmil_pooling_only(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=Path("/tmp/test"),
            attention_aggregation=AttentionAggregation.pooling_only,
        )

        extractor = AttentionExtractorFactory.create_extractor("graphmil", config)
        model = LitGraphMIL()

        x = torch.randn(10, 256)
        edge_index = torch.randint(0, 10, (2, 20))
        data = Data(x=x, edge_index=edge_index)

        result = extractor.extract(model, data)

        assert "pooling_attention" in result.attention_weights
        assert "gnn_attention_layer_0" not in result.attention_weights
        assert "gnn_attention_layer_1" not in result.attention_weights

    def test_graphmil_gnn_layer_specific(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=Path("/tmp/test"),
            attention_aggregation=AttentionAggregation.gnn_layer,
            gnn_layer_index=0,
        )

        extractor = AttentionExtractorFactory.create_extractor("graphmil", config)
        model = LitGraphMIL()

        x = torch.randn(8, 256)
        edge_index = torch.randint(0, 8, (2, 16))
        data = Data(x=x, edge_index=edge_index)

        result = extractor.extract(model, data)

        assert "selected_gnn_layer" in result.attention_weights
        assert (
            result.attention_weights["selected_gnn_layer"].shape[0] == 16
        )  # num_edges


class TestMainExplainer:
    """Test the main Explain class."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)

    def create_sample_cell_data(self, num_cells: int = 10):
        """Create sample cell data for testing."""
        cell_indices = {i: i for i in range(num_cells)}

        cell_data: dict[str, Any] = {
            "cells": [
                {
                    "cell_id": i,
                    "contour": [
                        [float(i), 0.0],
                        [float(i + 1), 0.0],
                        [float(i + 1), 1.0],
                        [float(i), 1.0],
                        [float(i), 0.0],
                    ],
                }
                for i in range(num_cells)
            ]
        }

        cell_data_path = self.output_path / "test_cells.json"
        with open(cell_data_path, "w") as f:
            json.dump(cell_data, f)

        return cell_indices, cell_data_path

    def test_explain_clam(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=self.output_path,
            visualization_mode=VisualizationMode.heatmap,  # Lightweight for testing
        )

        explainer = Explain(config)
        model = LitCLAM()
        data = torch.randn(10, 256)

        results = explainer.generate_explanation(model=model, data=data)

        assert results["model_type"] == "litclam"
        assert "attention_result" in results
        assert "summary" in results
        assert results["summary"]["num_attention_types"] > 0

    def test_explain_with_spatial_data(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=self.output_path,
            visualization_mode=VisualizationMode.geojson,
        )

        explainer = Explain(config)
        model = LitCLAM()
        data = torch.randn(5, 256)

        cell_indices, cell_data_path = self.create_sample_cell_data(5)

        results = explainer.generate_explanation(
            model=model,
            data=data,
            cell_data_path=cell_data_path,
            cell_indices=cell_indices,
        )

        assert "visualization_files" in results
        # Should have created GeoJSON files
        assert "geojson" in results["visualization_files"]

    def test_explain_graphmil(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=self.output_path,
            visualization_mode=VisualizationMode.graph_plot,
            attention_aggregation=AttentionAggregation.combined,
        )

        explainer = Explain(config)
        model = LitGraphMIL()

        x = torch.randn(8, 256)
        edge_index = torch.randint(0, 8, (2, 12))
        data = Data(x=x, edge_index=edge_index)

        results = explainer.generate_explanation(model=model, data=data)

        assert results["model_type"] == "litgraphmil"
        assert results["summary"]["num_attention_types"] >= 3  # GNN layers + pooling

    def test_missing_cell_data_file(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=self.output_path,
            visualization_mode=VisualizationMode.geojson,
        )

        explainer = Explain(config)

        # Missing cell data should be handled gracefully (no exception)
        # but should result in no GeoJSON files being created
        results = explainer.generate_explanation(
            model=LitCLAM(),
            data=torch.randn(5, 256),
            cell_data_path=Path("/nonexistent/path.json"),
            cell_indices={0: 0, 1: 1},
        )

        # Should still return results but with error handling
        assert "model_type" in results
        assert results["model_type"] == "litclam"


class TestAttentionResult:
    """Test the AttentionResult container class."""

    def test_attention_result_creation(self):
        attention_weights = {
            "test_attention": torch.rand(2, 10),
            "another_attention": torch.rand(1, 15),
        }

        metadata = {"test_key": "test_value"}

        result = AttentionResult(attention_weights, metadata, "TestModel")

        assert result.model_type == "TestModel"
        assert result.get_attention("test_attention") is not None
        assert result.get_attention("nonexistent") is None
        assert "test_attention" in result.get_all_keys()
        assert "another_attention" in result.get_all_keys()

    def test_shape_info(self):
        attention_weights = {
            "attention1": torch.rand(3, 8),
            "attention2": torch.rand(1, 12),
        }

        result = AttentionResult(attention_weights, {}, "TestModel")
        shape_info = result.get_shape_info()

        assert shape_info["attention1"] == (3, 8)
        assert shape_info["attention2"] == (1, 12)


class TestExplainConfig:
    """Test the configuration class."""

    def test_config_creation(self):
        config = ExplainerCreatorConfig(
            method=ExplainMethod.attention,
            output_path=Path("/test/path"),
            visualization_mode=VisualizationMode.all,
            attention_aggregation=AttentionAggregation.combined,
            num_attention_bins=12,
        )

        assert config.method == ExplainMethod.attention
        assert config.output_path == Path("/test/path")
        assert config.visualization_mode == VisualizationMode.all
        assert config.num_attention_bins == 12

    def test_config_validation(self):
        # Test invalid number of bins
        with pytest.raises(ValueError):
            ExplainerCreatorConfig(
                method=ExplainMethod.attention,
                output_path=Path("/test"),
                num_attention_bins=100,  # Too many
            )

        # Test invalid GNN layer index
        with pytest.raises(ValueError):
            ExplainerCreatorConfig(
                method=ExplainMethod.attention,
                output_path=Path("/test"),
                gnn_layer_index=-1,  # Negative
            )


# Integration test
class TestIntegration:
    """Integration tests for the complete explanation pipeline."""

    def test_end_to_end_explanation(self):
        """Test complete explanation pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)

            # Setup
            config = ExplainerCreatorConfig(
                method=ExplainMethod.attention,
                output_path=output_path,
                visualization_mode=VisualizationMode.heatmap,
                num_attention_bins=5,
            )

            explainer = Explain(config)
            model = LitCLAM()
            data = torch.randn(12, 256)

            # Execute
            results = explainer.generate_explanation(model=model, data=data)

            # Verify
            assert (output_path / "attention_weights.json").exists()
            assert (output_path / "explanation_metadata.json").exists()
            assert len(results["visualization_files"]["heatmaps"]) > 0

            # Check that heatmap files were created
            for heatmap_file in results["visualization_files"]["heatmaps"]:
                assert heatmap_file.exists()
                assert heatmap_file.suffix == ".png"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
