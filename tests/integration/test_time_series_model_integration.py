"""Integration tests for D1/D2/Model layer compatibility.

This module tests the complete pipeline from raw data through D1 (MultiSourceTSDataSet),
D2 (TSDataModule), and model layers to ensure seamless integration.
"""

import os
import shutil
import tempfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

# Import our library components
from dsipts.data_structure.time_series_d1 import MultiSourceTSDataSet
from dsipts.data_structure.time_series_d2 import TSDataModule, custom_collate_fn
from dsipts.models.LinearTS import LinearTS


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_time_series_data(temp_data_dir) -> Tuple[str, Dict]:
    """Create sample time series data for testing."""
    # Create sample data with multiple groups and features
    np.random.seed(42)

    data_list = []
    groups = ["A", "B", "C"]
    n_timesteps = 100

    for group in groups:
        # Generate time series with trend and seasonality
        time_idx = np.arange(n_timesteps)
        trend = 0.1 * time_idx
        seasonal = 2 * np.sin(2 * np.pi * time_idx / 12)
        noise = np.random.normal(0, 0.5, n_timesteps)

        # Create multiple features
        feature1 = trend + seasonal + noise
        feature2 = 0.5 * feature1 + np.random.normal(0, 0.3, n_timesteps)
        target = feature1 + 0.3 * feature2 + np.random.normal(0, 0.2, n_timesteps)

        # Add categorical features
        hour = np.random.randint(0, 24, n_timesteps)
        day_of_week = np.random.randint(0, 7, n_timesteps)
        month = np.random.randint(1, 13, n_timesteps)

        # Create DataFrame for this group
        group_data = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=n_timesteps, freq="D"),
                "group": group,
                "target": target,
                "feature1": feature1,
                "feature2": feature2,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
            }
        )

        data_list.append(group_data)

    # Combine all groups
    full_data = pd.concat(data_list, ignore_index=True)

    # Save to CSV
    data_path = os.path.join(temp_data_dir, "test_data.csv")
    full_data.to_csv(data_path, index=False)

    # Return path and metadata
    metadata = {
        "time_col": "time",
        "group_col": "group",
        "target_col": "target",
        "feature_cols": ["feature1", "feature2"],
        "categorical_cols": ["hour", "day_of_week", "month"],
        "groups": groups,
        "n_timesteps": n_timesteps,
    }

    return data_path, metadata


class TestTimeSeriesIntegration:
    """Integration tests for the complete D1/D2/Model pipeline."""

    def test_d1_layer_initialization(self, sample_time_series_data):
        """Test D1 layer (MultiSourceTSDataSet) initialization and basic functionality."""
        data_path, metadata = sample_time_series_data

        # Initialize D1 layer
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=[metadata["target_col"]],
            group_cols=metadata["group_col"],
            feature_cols=metadata["feature_cols"],
            cat_cols=metadata["categorical_cols"],
        )

        # Test basic properties
        assert len(d1) > 0  # Dataset has samples
        assert d1.time_col == metadata["time_col"]
        assert d1.target_cols == [metadata["target_col"]]
        assert d1.group_cols == [metadata["group_col"]]

        # Test data access
        sample_item = d1[0]
        assert isinstance(sample_item, dict)
        assert "y" in sample_item  # target data
        assert "x" in sample_item  # feature data
        assert sample_item["x"].shape[1] == len(metadata["feature_cols"])

    def test_d2_layer_initialization(self, sample_time_series_data):
        """Test D2 layer (TSDataModule) initialization and configuration."""
        data_path, metadata = sample_time_series_data

        # Create D1 layer for D2 initialization
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            group_cols=[metadata["group_col"]],
            time_col=metadata["time_col"],
            target_cols=[metadata["target_col"]],
            feature_cols=metadata["feature_cols"],
            cat_cols=metadata["categorical_cols"],
        )

        # Initialize D2 layer
        d2 = TSDataModule(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=4,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        # Test basic properties
        assert d2.past_len == 10
        assert d2.future_len == 5
        assert d2.batch_size == 4
        assert hasattr(d2, "train_indices")
        assert hasattr(d2, "val_indices")
        assert hasattr(d2, "test_indices")

        # Test data access
        sample_item = d2[0]
        assert isinstance(sample_item, tuple)
        assert len(sample_item) == 2  # (x, y)

        x, y = sample_item
        assert isinstance(x, dict)
        assert isinstance(y, torch.Tensor)

        # Test required keys are present
        required_keys = [
            "x_num_past",
            "x_cat_past",
            "x_num_future",
            "x_cat_future",
            "idx_target",
            "y",
        ]
        for key in required_keys:
            assert key in x, f"Missing required key: {key}"

    def test_d2_dataloader_functionality(self, sample_time_series_data):
        """Test D2 layer DataLoader functionality with custom collate function."""
        data_path, metadata = sample_time_series_data

        # Initialize D1 and D2 layers
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            group_cols=[metadata["group_col"]],
            time_col=metadata["time_col"],
            target_cols=[metadata["target_col"]],
            feature_cols=metadata["feature_cols"],
            cat_cols=metadata["categorical_cols"],
        )

        d2 = TSDataModule(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=4,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        # Setup for training
        d2.setup("fit")

        # Get dataloaders
        train_loader = d2.train_dataloader()
        val_loader = d2.val_dataloader()

        # Test train dataloader
        assert isinstance(train_loader, DataLoader)
        batch = next(iter(train_loader))
        assert isinstance(batch, dict)

        # Test batch structure
        required_keys = [
            "x_num_past",
            "x_cat_past",
            "x_num_future",
            "x_cat_future",
            "idx_target",
            "y",
        ]
        for key in required_keys:
            assert key in batch, f"Missing required key in batch: {key}"
            assert isinstance(batch[key], torch.Tensor), f"Key {key} should be a tensor"

        # Test batch dimensions
        batch_size = batch["x_num_past"].shape[0]
        assert batch_size <= d2.batch_size
        assert batch["x_num_past"].shape[1] == d2.past_len
        assert batch["y"].shape[1] == d2.future_len

        # Test validation dataloader if available
        if val_loader is not None:
            val_batch = next(iter(val_loader))
            assert isinstance(val_batch, dict)
            for key in required_keys:
                assert key in val_batch

    def test_model_integration_with_linear_ts(self, sample_time_series_data):
        """Test integration with LinearTS model from our library."""
        data_path, metadata = sample_time_series_data

        # Initialize D1 and D2 layers
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            group_cols=[metadata["group_col"]],
            time_col=metadata["time_col"],
            target_cols=[metadata["target_col"]],
            feature_cols=metadata["feature_cols"],
            cat_cols=metadata["categorical_cols"],
        )

        d2 = TSDataModule(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=4,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        d2.setup("fit")
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))

        # Get dimensions from batch
        past_channels = batch["x_num_past"].shape[-1]
        future_channels = (
            batch["x_num_future"].shape[-1] if batch["x_num_future"].numel() > 0 else 0
        )
        out_channels = batch["y"].shape[-1]

        # Get categorical dimensions
        cat_dims = []
        if batch["x_cat_past"].numel() > 0:
            n_cat_features = batch["x_cat_past"].shape[-1]
            # Assume each categorical feature has max 25 classes for this test
            cat_dims = [25] * n_cat_features

        # Initialize LinearTS model
        model = LinearTS(
            verbose=False,
            past_steps=d2.past_len,
            future_steps=d2.future_len,
            past_channels=past_channels,
            future_channels=future_channels,
            embs=cat_dims,
            cat_emb_dim=8,
            kernel_size=25,
            sum_emb=True,
            out_channels=out_channels,
            hidden_size=128,
            dropout_rate=0.1,
            kind="linear",
            simple=False,  # avoids seasonal_init bug
        )

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch)

        # Test output shape
        expected_shape = (batch["x_num_past"].shape[0], d2.future_len, out_channels, 1)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Test that output is finite
        assert torch.isfinite(output).all(), "Model output contains non-finite values"

    def test_end_to_end_pipeline(self, sample_time_series_data):
        """Test the complete end-to-end pipeline from data to predictions."""
        data_path, metadata = sample_time_series_data

        # Step 1: Initialize D1 layer
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            group_cols=[metadata["group_col"]],
            time_col=metadata["time_col"],
            target_cols=[metadata["target_col"]],
            feature_cols=metadata["feature_cols"],
            cat_cols=metadata["categorical_cols"],
        )

        # Step 2: Initialize D2 layer
        d2 = TSDataModule(
            d1_dataset=d1,
            past_len=8,
            future_len=3,
            batch_size=6,
            split_config=(0.8, 0.1, 0.1),
            precompute=True,
        )

        # Step 3: Setup and get data
        d2.setup("fit")
        train_loader = d2.train_dataloader()

        # Step 4: Process multiple batches
        predictions = []
        targets = []

        model_initialized = False
        model = None

        for i, batch in enumerate(train_loader):
            if i >= 3:  # Test with 3 batches
                break

            # Initialize model on first batch
            if not model_initialized:
                past_channels = batch["x_num_past"].shape[-1]
                future_channels = (
                    batch["x_num_future"].shape[-1] if batch["x_num_future"].numel() > 0 else 0
                )
                out_channels = batch["y"].shape[-1]

                model = LinearTS(
                    verbose=False,
                    past_steps=d2.past_len,
                    future_steps=d2.future_len,
                    past_channels=past_channels,
                    future_channels=future_channels,
                    embs=[],
                    cat_emb_dim=8,
                    kernel_size=25,
                    sum_emb=True,
                    out_channels=out_channels,
                    hidden_size=64,
                    dropout_rate=0.1,
                    kind="linear",
                    simple=False,
                )
                model.eval()
                model_initialized = True

            # Forward pass
            with torch.no_grad():
                output = model(batch)

            predictions.append(output)
            targets.append(batch["y"])

        # Step 5: Validate results
        assert len(predictions) == 3, "Should have processed 3 batches"
        assert len(targets) == 3, "Should have 3 target batches"

        # Check that all predictions have consistent shapes
        for pred, target in zip(predictions, targets):
            assert (
                pred.shape[:-1] == target.shape
            ), f"Shape mismatch: {pred.shape} vs {target.shape}"
            assert torch.isfinite(pred).all(), "Predictions contain non-finite values"
            assert torch.isfinite(target).all(), "Targets contain non-finite values"

    def test_data_format_compatibility(self, sample_time_series_data):
        """Test that D2 output format is directly compatible with model expectations."""
        data_path, metadata = sample_time_series_data

        # Initialize pipeline
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            group_cols=[metadata["group_col"]],
            time_col=metadata["time_col"],
            target_cols=[metadata["target_col"]],
            feature_cols=metadata["feature_cols"],
            cat_cols=metadata["categorical_cols"],
        )

        d2 = TSDataModule(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=4,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        d2.setup("fit")
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))

        # Test that batch contains all model-expected keys
        model_expected_keys = {
            "x_num_past": torch.Tensor,
            "x_cat_past": torch.Tensor,
            "x_num_future": torch.Tensor,
            "x_cat_future": torch.Tensor,
            "idx_target": torch.Tensor,
            "y": torch.Tensor,
        }

        for key, expected_type in model_expected_keys.items():
            assert key in batch, f"Missing model-expected key: {key}"
            assert isinstance(
                batch[key], expected_type
            ), f"Key {key} has wrong type: {type(batch[key])}"

        # Test backward compatibility keys
        backward_compat_keys = ["past_features", "future_targets"]
        for key in backward_compat_keys:
            assert key in batch, f"Missing backward compatibility key: {key}"

        # Test that no adapter is needed - direct model usage
        past_channels = batch["x_num_past"].shape[-1]
        out_channels = batch["y"].shape[-1]

        # This should work without any data transformation
        simple_model = LinearTS(
            past_steps=d2.past_len,
            future_steps=d2.future_len,
            past_channels=past_channels,
            future_channels=0,
            embs=[],
            cat_emb_dim=8,
            kernel_size=25,
            sum_emb=True,
            out_channels=out_channels,
            hidden_size=32,
            dropout_rate=0.1,
            kind="linear",
            simple=False,
            verbose=False,
        )

        simple_model.eval()
        with torch.no_grad():
            # Direct usage without any adapter - this proves compatibility
            output = simple_model(batch)

        assert output is not None
        assert torch.isfinite(output).all()

    def test_custom_collate_function(self, sample_time_series_data):
        """Test the custom collate function behavior."""
        data_path, metadata = sample_time_series_data

        # Initialize pipeline
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            group_cols=[metadata["group_col"]],
            time_col=metadata["time_col"],
            target_cols=[metadata["target_col"]],
            feature_cols=metadata["feature_cols"],
            cat_cols=metadata["categorical_cols"],
        )

        d2 = TSDataModule(
            d1_dataset=d1,
            past_len=5,
            future_len=3,
            batch_size=1,  # Use batch size 1 to get individual samples
            split_config=(0.9, 0.05, 0.05),
            precompute=True,
        )

        # Get individual samples
        samples = [d2[i] for i in range(min(3, len(d2)))]

        # Test collate function directly
        collated = custom_collate_fn(samples)

        # Test that collate function returns a single dictionary
        assert isinstance(collated, dict), "Collate function should return a dictionary"

        # Test that tensors are properly stacked
        for key, value in collated.items():
            if key not in ["group_id", "past_time", "future_time"]:  # These remain as lists
                assert isinstance(
                    value, torch.Tensor
                ), f"Key {key} should be a tensor after collation"
                assert value.shape[0] == len(samples), f"Batch dimension mismatch for key {key}"

        # Test that list fields remain as lists
        list_fields = ["group_id", "past_time", "future_time"]
        for field in list_fields:
            if field in collated:
                assert isinstance(collated[field], list), f"Field {field} should remain as list"
                assert len(collated[field]) == len(
                    samples
                ), f"List length mismatch for field {field}"
