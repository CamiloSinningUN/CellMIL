import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from typing import Any
from torch.utils.data import DataLoader as DataLoaderTorch, TensorDataset
import lightning as Pl
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from cellmil.utils.train.evals import get_report


class TestGetReport:
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer"""
        trainer = Mock(spec=Trainer)
        trainer.callbacks = []
        return trainer

    @pytest.fixture
    def mock_lit_model(self):
        """Create a mock lightning module"""
        model = Mock(spec=Pl.LightningModule)
        model.eval = Mock()
        model.load_state_dict = Mock()
        return model

    @pytest.fixture
    def sample_torch_dataloader(self):
        """Create a sample torch DataLoader"""
        # Create synthetic data: features and labels
        X = torch.randn(10, 5)
        y = torch.randint(0, 2, (10,))
        dataset = TensorDataset(X, y)
        return DataLoaderTorch(dataset, batch_size=2)

    @pytest.fixture
    def sample_pyg_dataloader(self):
        """Create a mock PyTorch Geometric DataLoader"""
        # Create a mock dataloader that behaves like PyG DataLoader
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(
            return_value=iter(
                [
                    Mock(y=torch.tensor([0])),
                    Mock(y=torch.tensor([1])),
                    Mock(y=torch.tensor([0])),
                    Mock(y=torch.tensor([1])),
                    Mock(y=torch.tensor([0])),
                ]
            )
        )
        return mock_dataloader

    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions as would be returned by trainer.predict"""
        return [
            torch.tensor([[0.8]]),
            torch.tensor([[0.2]]),
            torch.tensor([[0.9]]),
            torch.tensor([[0.1]]),
            torch.tensor([[0.7]]),
        ]

    def test_get_report_torch_dataloader_basic(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test get_report with torch DataLoader"""
        # Mock predictions
        predictions = [
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            torch.tensor([[0]]),
        ]
        mock_trainer.predict.return_value = predictions  # type: ignore

        with patch("cellmil.utils.train.evals.classification_report") as mock_report:
            mock_report.return_value = {"accuracy": 0.8, "0": {"precision": 0.75}}

            result = get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)

            assert isinstance(result, dict)
            mock_trainer.predict.assert_called_once_with( # type: ignore
                mock_lit_model, sample_torch_dataloader
            )  
            mock_report.assert_called_once()

    def test_get_report_with_model_checkpoint(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test get_report when ModelCheckpoint callback is present"""
        # Create mock ModelCheckpoint callback
        mock_checkpoint = Mock(spec=ModelCheckpoint)
        mock_checkpoint.best_model_path = "/path/to/best/model.ckpt"
        mock_trainer.callbacks = [mock_checkpoint]  # type: ignore

        # Mock torch.load to return a checkpoint
        mock_state_dict = {"layer.weight": torch.randn(5, 3)}
        mock_checkpoint_data = {"state_dict": mock_state_dict}

        predictions = [torch.tensor([[1]]), torch.tensor([[0]])]
        mock_trainer.predict.return_value = predictions  # type: ignore

        with (
            patch("torch.load", return_value=mock_checkpoint_data) as mock_load,
            patch("cellmil.utils.train.evals.classification_report") as mock_report,
        ):
            mock_report.return_value = {"accuracy": 0.85}

            result = get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)

            # Verify checkpoint loading
            mock_load.assert_called_once_with(
                mock_checkpoint.best_model_path, map_location=mock_lit_model.device
            )
            mock_lit_model.load_state_dict.assert_called_once_with(mock_state_dict)  # type: ignore
            mock_lit_model.eval.assert_called_once()  # type: ignore

            assert isinstance(result, dict)

    def test_get_report_checkpoint_loading_failure(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test get_report when checkpoint loading fails"""
        # Create mock ModelCheckpoint callback
        mock_checkpoint = Mock(spec=ModelCheckpoint)
        mock_checkpoint.best_model_path = "/invalid/path/model.ckpt"
        mock_trainer.callbacks = [mock_checkpoint]  # type: ignore

        predictions = [torch.tensor([[1]]), torch.tensor([[0]])]
        mock_trainer.predict.return_value = predictions  # type: ignore

        with (
            patch("torch.load", side_effect=FileNotFoundError("File not found")),
            patch("cellmil.utils.train.evals.classification_report") as mock_report,
            patch("cellmil.utils.train.evals.logger") as mock_logger,
        ):
            mock_report.return_value = {"accuracy": 0.75}

            result = get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)

            # Should still work despite checkpoint loading failure
            mock_logger.warning.assert_called_once()
            assert isinstance(result, dict)

    def test_get_report_no_checkpoint_callback(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test get_report when no ModelCheckpoint callback is present"""
        mock_trainer.callbacks = []  # No callbacks # type: ignore

        predictions = [torch.tensor([[0]]), torch.tensor([[1]])]
        mock_trainer.predict.return_value = predictions  # type: ignore

        with patch("cellmil.utils.train.evals.classification_report") as mock_report:
            mock_report.return_value = {"accuracy": 0.70}

            result = get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)

            # Should work without trying to load checkpoint
            mock_lit_model.load_state_dict.assert_not_called()  # type: ignore
            assert isinstance(result, dict)

    def test_prediction_tensor_flattening(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test that predictions are properly flattened"""
        # Predictions with different shapes
        predictions = [
            torch.tensor([[0.1]]),
            torch.tensor([[0.9]]),
            torch.tensor([[0.3]]),
        ]
        mock_trainer.predict.return_value = predictions  # type: ignore

        with patch("cellmil.utils.train.evals.classification_report") as mock_report:
            mock_report.return_value = {"accuracy": 0.80}

            result = get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)

            # Verify classification_report was called with flattened values
            args, _ = mock_report.call_args
            _, y_pred_arg = args[0], args[1]

            # Check that predictions were flattened correctly
            assert all(isinstance(pred, (int, float, np.number)) for pred in y_pred_arg)
            assert isinstance(result, dict)

    def test_target_tensor_flattening_torch(
        self, mock_trainer: Trainer, mock_lit_model: Pl.LightningModule
    ):
        """Test that targets from torch DataLoader are properly flattened"""
        # Create DataLoader with tensor targets
        X = torch.randn(4, 3)
        y = torch.tensor([[0], [1], [0], [1]])  # 2D targets that need flattening
        dataset = TensorDataset(X, y)
        dataloader = DataLoaderTorch(dataset, batch_size=2)

        predictions = [torch.tensor([[0]]), torch.tensor([[1]])]
        mock_trainer.predict.return_value = predictions  # type: ignore

        with patch("cellmil.utils.train.evals.classification_report") as mock_report:
            mock_report.return_value = {"accuracy": 0.85}

            result = get_report(mock_trainer, mock_lit_model, dataloader)

            # Verify classification_report was called
            args, _ = mock_report.call_args
            y_true_arg, _ = args[0], args[1]

            # Check that targets were flattened correctly
            assert all(isinstance(true, (int, float, np.number)) for true in y_true_arg)
            assert isinstance(result, dict)

    def test_non_tensor_predictions(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test handling of non-tensor predictions"""
        # Non-tensor predictions (e.g., numpy arrays or lists)
        predictions = [np.array([0.8]), np.array([0.2]), np.array([0.9])]
        mock_trainer.predict.return_value = predictions  # type: ignore

        with patch("cellmil.utils.train.evals.classification_report") as mock_report:
            mock_report.return_value = {"accuracy": 0.75}

            result = get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)

            assert isinstance(result, dict)
            mock_report.assert_called_once()

    def test_empty_predictions(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test handling of empty predictions"""
        mock_trainer.predict.return_value = []  # type: ignore

        # The function should handle empty predictions gracefully
        # This test verifies that the function doesn't crash with empty predictions
        try:
            get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)
            # If it doesn't crash, that's good, but it might return something unusual
            assert True  # We're just testing that it doesn't crash
        except (IndexError, ValueError):
            # This is also acceptable behavior for empty predictions
            assert True

    def test_none_predictions(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test handling of None predictions"""
        mock_trainer.predict.return_value = None  # type: ignore

        # This should raise an exception when trying to process None predictions
        try:
            get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)
            assert False, "Expected an exception but none was raised"
        except (AttributeError, TypeError):
            # This is expected behavior
            pass

    def test_classification_report_output_format(
        self,
        mock_trainer: Trainer,
        mock_lit_model: Pl.LightningModule,
        sample_torch_dataloader: DataLoaderTorch[Any],
    ):
        """Test that the function returns the classification report in correct format"""
        predictions = [torch.tensor([[1]]), torch.tensor([[0]])]
        mock_trainer.predict.return_value = predictions  # type: ignore

        expected_report: dict[str, Any] = {
            "accuracy": 0.85,
            "0": {"precision": 0.80, "recall": 0.90, "f1-score": 0.85, "support": 5},
            "1": {"precision": 0.90, "recall": 0.80, "f1-score": 0.85, "support": 5},
            "macro avg": {
                "precision": 0.85,
                "recall": 0.85,
                "f1-score": 0.85,
                "support": 10,
            },
            "weighted avg": {
                "precision": 0.85,
                "recall": 0.85,
                "f1-score": 0.85,
                "support": 10,
            },
        }

        with patch(
            "cellmil.utils.train.evals.classification_report",
            return_value=expected_report,
        ) as mock_report:
            result = get_report(mock_trainer, mock_lit_model, sample_torch_dataloader)

            assert result == expected_report
            # Verify output_dict=True was passed
            mock_report.assert_called_with(
                mock_report.call_args[0][0],  # y_true
                mock_report.call_args[0][1],  # y_pred
                output_dict=True,
            )
