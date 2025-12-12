import pytest
import pandas as pd
import numpy as np
from typing import Any
from unittest.mock import patch
from cellmil.utils.train.dataset import split_dataset


class TestSplitDataset:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing"""
        np.random.seed(42)
        data: dict[str, Any] = {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def balanced_dataframe(self):
        """Create a balanced DataFrame with equal class distribution"""
        data = {
            "feature1": list(range(100)),
            "feature2": list(range(100, 200)),
            "label": [0] * 50 + [1] * 50,  # Perfectly balanced
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def imbalanced_dataframe(self):
        """Create an imbalanced DataFrame"""
        data = {
            "feature1": list(range(100)),
            "feature2": list(range(100, 200)),
            "label": [0] * 80 + [1] * 20,  # 80-20 split
        }
        return pd.DataFrame(data)

    def test_split_dataset_basic_train_val_only(self, sample_dataframe: pd.DataFrame):
        """Test basic train/validation split without test set"""
        result_df = split_dataset(sample_dataframe, train_size=0.8, test=False)

        # Check that SPLIT column was added
        assert "SPLIT" in result_df.columns

        # Check split proportions
        train_count = len(result_df[result_df["SPLIT"] == "train"])
        val_count = len(result_df[result_df["SPLIT"] == "val"])

        # Should be approximately 80/20 split
        total = len(result_df)
        assert abs(train_count / total - 0.8) < 0.05  # Allow 5% tolerance
        assert abs(val_count / total - 0.2) < 0.05

        # Verify no test split exists
        assert len(result_df[result_df["SPLIT"] == "test"]) == 0

    def test_split_dataset_with_test_set(self, sample_dataframe: pd.DataFrame):
        """Test train/validation/test split"""
        result_df = split_dataset(sample_dataframe, train_size=0.6, test=True)

        # Check that SPLIT column was added
        assert "SPLIT" in result_df.columns

        # Check split proportions
        train_count = len(result_df[result_df["SPLIT"] == "train"])
        val_count = len(result_df[result_df["SPLIT"] == "val"])
        test_count = len(result_df[result_df["SPLIT"] == "test"])

        total = len(result_df)

        # With train_size=0.6 and test=True:
        # train = 60%, val = 20%, test = 20%
        assert abs(train_count / total - 0.6) < 0.05
        assert abs(val_count / total - 0.2) < 0.05
        assert abs(test_count / total - 0.2) < 0.05

        # Verify all splits exist
        assert train_count > 0
        assert val_count > 0
        assert test_count > 0

    def test_split_dataset_stratification_enabled(
        self, balanced_dataframe: pd.DataFrame
    ):
        """Test that stratification maintains class balance"""
        result_df = split_dataset(
            balanced_dataframe, train_size=0.8, stratify=True, label="label"
        )

        # Check class distribution in train set
        train_df = result_df[result_df["SPLIT"] == "train"]
        train_label_dist = train_df["label"].value_counts(normalize=True)

        # Check class distribution in validation set
        val_df = result_df[result_df["SPLIT"] == "val"]
        val_label_dist = val_df["label"].value_counts(normalize=True)

        # Both should maintain approximately 50-50 balance
        assert abs(train_label_dist[0] - 0.5) < 0.1
        assert abs(train_label_dist[1] - 0.5) < 0.1
        assert abs(val_label_dist[0] - 0.5) < 0.1
        assert abs(val_label_dist[1] - 0.5) < 0.1

    def test_split_dataset_stratification_disabled(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test splitting without stratification"""
        result_df = split_dataset(
            sample_dataframe, train_size=0.8, stratify=False, label="label"
        )

        # Check that splitting works without stratification
        assert "SPLIT" in result_df.columns
        assert len(result_df[result_df["SPLIT"] == "train"]) > 0
        assert len(result_df[result_df["SPLIT"] == "val"]) > 0

    def test_split_dataset_different_train_sizes(self, sample_dataframe: pd.DataFrame):
        """Test different train_size values"""
        for train_size in [0.5, 0.7, 0.9]:
            result_df = split_dataset(
                sample_dataframe.copy(), train_size=train_size, test=False
            )

            train_count = len(result_df[result_df["SPLIT"] == "train"])
            total = len(result_df)

            assert abs(train_count / total - train_size) < 0.05

    def test_split_dataset_custom_label_column(self, sample_dataframe: pd.DataFrame):
        """Test with custom label column name"""
        # Rename label column
        df_custom = sample_dataframe.copy()
        df_custom = df_custom.rename(columns={"label": "target"})

        result_df = split_dataset(df_custom, train_size=0.8, label="target")

        assert "SPLIT" in result_df.columns
        assert len(result_df[result_df["SPLIT"] == "train"]) > 0
        assert len(result_df[result_df["SPLIT"] == "val"]) > 0

    def test_split_dataset_nonexistent_label_column(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test with non-existent label column"""
        result_df = split_dataset(sample_dataframe, train_size=0.8, label="nonexistent")

        # Should still work but without stratification
        assert "SPLIT" in result_df.columns
        assert len(result_df[result_df["SPLIT"] == "train"]) > 0
        assert len(result_df[result_df["SPLIT"] == "val"]) > 0

    def test_split_dataset_custom_random_state(self, sample_dataframe: pd.DataFrame):
        """Test reproducibility with custom random state"""
        result1 = split_dataset(
            sample_dataframe.copy(), train_size=0.8, random_state=123
        )
        result2 = split_dataset(
            sample_dataframe.copy(), train_size=0.8, random_state=123
        )

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_split_dataset_different_random_states(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test that different random states produce different splits"""
        result1 = split_dataset(
            sample_dataframe.copy(), train_size=0.8, random_state=42
        )
        result2 = split_dataset(
            sample_dataframe.copy(), train_size=0.8, random_state=123
        )

        # Results should be different (very unlikely to be identical)
        train_indices1 = set(result1[result1["SPLIT"] == "train"].index)
        train_indices2 = set(result2[result2["SPLIT"] == "train"].index)

        assert train_indices1 != train_indices2

    def test_split_dataset_invalid_train_size_too_large(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test assertion error for train_size > 1"""
        with pytest.raises(AssertionError, match="train_size must be between 0 and 1"):
            split_dataset(sample_dataframe, train_size=1.5)

    def test_split_dataset_invalid_train_size_zero(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test assertion error for train_size = 0"""
        with pytest.raises(AssertionError, match="train_size must be between 0 and 1"):
            split_dataset(sample_dataframe, train_size=0.0)

    def test_split_dataset_invalid_train_size_negative(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test assertion error for negative train_size"""
        with pytest.raises(AssertionError, match="train_size must be between 0 and 1"):
            split_dataset(sample_dataframe, train_size=-0.1)

    def test_split_dataset_edge_case_train_size_one(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test edge case where train_size = 1.0"""
        # This should raise an error because val_size would be 0
        # which sklearn doesn't allow
        with pytest.raises(Exception):  # Could be InvalidParameterError or similar
            split_dataset(sample_dataframe, train_size=1.0, test=False)

    def test_split_dataset_small_dataframe(self):
        """Test with very small DataFrame"""
        small_df = pd.DataFrame({"feature": [1, 2, 3, 4, 5], "label": [0, 1, 0, 1, 0]})

        result_df = split_dataset(small_df, train_size=0.6)

        assert "SPLIT" in result_df.columns
        assert len(result_df[result_df["SPLIT"] == "train"]) > 0
        assert len(result_df[result_df["SPLIT"] == "val"]) > 0

    def test_split_dataset_preserves_original_data(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test that original data is preserved and only SPLIT column is added"""
        original_columns = set(sample_dataframe.columns)
        result_df = split_dataset(sample_dataframe, train_size=0.8)

        # Check that all original columns are preserved
        for col in original_columns:
            assert col in result_df.columns
            pd.testing.assert_series_equal(
                sample_dataframe[col].sort_index(),  # type: ignore
                result_df[col].sort_index(),  # type: ignore
                check_names=False,
            )

        # Check that only SPLIT column was added
        assert len(result_df.columns) == len(original_columns) + 1
        assert "SPLIT" in result_df.columns

    def test_split_dataset_no_data_loss(self, sample_dataframe: pd.DataFrame):
        """Test that no rows are lost during splitting"""
        result_df = split_dataset(sample_dataframe, train_size=0.8, test=True)

        assert len(result_df) == len(sample_dataframe)

        # Check that all indices are preserved
        assert set(result_df.index) == set(sample_dataframe.index)

    def test_split_dataset_with_existing_split_column(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test behavior when SPLIT column already exists"""
        sample_dataframe["SPLIT"] = "existing"

        result_df = split_dataset(sample_dataframe, train_size=0.8)

        # Should overwrite existing SPLIT column
        assert "train" in result_df["SPLIT"].values  # type: ignore
        assert "val" in result_df["SPLIT"].values  # type: ignore
        assert "existing" not in result_df["SPLIT"].values  # type: ignore

    def test_split_dataset_logging_output(self, sample_dataframe: pd.DataFrame):
        """Test that appropriate logging messages are generated"""
        with patch("cellmil.utils.train.dataset.logger") as mock_logger:
            split_dataset(sample_dataframe, train_size=0.8, test=True)

            # Should log train, validation, and test sizes
            assert mock_logger.info.call_count == 3

            # Check that logged messages contain size information
            call_args = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Test size:" in msg for msg in call_args)
            assert any("Validation size:" in msg for msg in call_args)
            assert any("Train size:" in msg for msg in call_args)

    def test_split_dataset_logging_without_test(self, sample_dataframe: pd.DataFrame):
        """Test logging when test=False"""
        with patch("cellmil.utils.train.dataset.logger") as mock_logger:
            split_dataset(sample_dataframe, train_size=0.8, test=False)

            # Should log validation and train sizes only
            assert mock_logger.info.call_count == 2

            call_args = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Validation size:" in msg for msg in call_args)
            assert any("Train size:" in msg for msg in call_args)

    def test_split_dataset_all_unique_splits(self, sample_dataframe: pd.DataFrame):
        """Test that train, val, and test sets have no overlapping indices"""
        result_df = split_dataset(sample_dataframe, train_size=0.6, test=True)

        train_indices = set(result_df[result_df["SPLIT"] == "train"].index)
        val_indices = set(result_df[result_df["SPLIT"] == "val"].index)
        test_indices = set(result_df[result_df["SPLIT"] == "test"].index)

        # Check no overlap between any splits
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0

        # Check that all indices are covered
        all_split_indices = train_indices | val_indices | test_indices
        assert all_split_indices == set(sample_dataframe.index)

    def test_split_dataset_return_value_is_original_df(
        self, sample_dataframe: pd.DataFrame
    ):
        """Test that the function returns the original DataFrame (modified in place)"""
        original_id = id(sample_dataframe)
        result_df = split_dataset(sample_dataframe, train_size=0.8)

        # Should return the same DataFrame object
        assert id(result_df) == original_id
        assert result_df is sample_dataframe
