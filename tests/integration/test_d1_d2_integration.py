"""Integration test for D1 and D2 layers.

This module tests the integration between D1 (MultiSourceTSDataSet) and D2 (EncoderDecoder)
layers without involving models. It verifies that the data pipeline works correctly
with various configurations and edge cases.
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
from dsipts.data_structure.d1_layers import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers import EncoderDecoder


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def create_sample_data(
    temp_dir: str,
    n_groups: int = 3,
    n_timesteps: int = 100,
    n_num_features: int = 2,
    n_cat_features: int = 3,
    n_targets: int = 1,
    include_unknown: bool = True,
    include_static: bool = False,  # Changed to False as per user instruction
    debug: bool = False,
) -> Tuple[str, Dict]:
    """Create sample time series data with various feature types.

    Args:
        temp_dir: Directory to save the data
        n_groups: Number of groups to create
        n_timesteps: Number of time steps per group
        n_num_features: Number of numerical features
        n_cat_features: Number of categorical features
        n_targets: Number of target variables
        include_unknown: Whether to include unknown features
        include_static: Whether to include static features

    Returns:
        Tuple of (data_path, metadata)
    """
    np.random.seed(42)

    data_list = []
    groups = [f"group_{i}" for i in range(n_groups)]

    # Define feature and target column names
    num_cols = [f"num_{i}" for i in range(n_num_features)]
    cat_cols = [f"cat_{i}" for i in range(n_cat_features)]
    target_cols = [f"target_{i}" for i in range(n_targets)]

    # Define known/unknown split (if requested)
    if include_unknown:
        known_num = num_cols[: n_num_features // 2]
        unknown_num = num_cols[n_num_features // 2 :]
        known_cat = cat_cols[: n_cat_features // 2]
        unknown_cat = cat_cols[n_cat_features // 2 :]
    else:
        known_num = num_cols
        unknown_num = []
        known_cat = cat_cols
        unknown_cat = []

    # Create static columns if requested (disabled as per user instruction)
    static_cols = []
    if include_static:
        static_cols = ["static_num", "static_cat"]

    if debug:
        print(f"DEBUG: Creating sample data with {n_groups} groups, {n_timesteps} timesteps")
        print(
            f"DEBUG: Numerical features: {n_num_features}, Categorical features: {n_cat_features}"
        )
        print(f"DEBUG: Static features enabled: {include_static}")

    for group in groups:
        # Generate time series with trend and seasonality
        time_idx = np.arange(n_timesteps)

        # Create DataFrame for this group
        group_data = {
            "time": pd.date_range("2020-01-01", periods=n_timesteps, freq="D"),
            "group_id": group,  # Changed from 'group' to 'group_id' to avoid confusion
        }

        # Add numerical features
        for i, col in enumerate(num_cols):
            # Create features with different patterns
            trend = 0.1 * time_idx * (i + 1)
            seasonal = (i + 1) * np.sin(2 * np.pi * time_idx / (12 + i))
            noise = np.random.normal(0, 0.5, n_timesteps)
            group_data[col] = trend + seasonal + noise

        # Add categorical features
        for i, col in enumerate(cat_cols):
            # Create categorical features with different cardinalities
            cardinality = 3 + i
            group_data[col] = [
                f"val_{np.random.randint(0, cardinality)}" for _ in range(n_timesteps)
            ]

        # Add target variables
        for i, col in enumerate(target_cols):
            # Create targets as functions of features
            base = group_data[num_cols[0]] + 0.5 * group_data[num_cols[-1]]
            group_data[col] = base + np.random.normal(0, 0.2, n_timesteps)

        # Add static features if requested
        if include_static:
            group_data["static_num"] = np.random.normal(float(group[-1]), 1.0)
            group_data["static_cat"] = f"static_val_{np.random.randint(0, 3)}"

        # Convert to DataFrame and add to list
        df = pd.DataFrame(group_data)
        data_list.append(df)

    # Combine all groups
    full_data = pd.concat(data_list, ignore_index=True)

    # Save to CSV
    data_path = os.path.join(temp_dir, "test_data.csv")
    full_data.to_csv(data_path, index=False)

    # Return path and metadata
    metadata = {
        "time_col": "time",
        "group_cols": ["group_id"],  # Updated to match the column name in the CSV
        "target_cols": target_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "known_cols": known_num + known_cat,
        "unknown_cols": unknown_num + unknown_cat,
        "static_cols": static_cols,
        "groups": groups,
        "n_timesteps": n_timesteps,
    }

    if debug:
        print(f"DEBUG: Metadata: {metadata}")

    return data_path, metadata


class TestD1D2Integration:
    """Integration tests for D1 and D2 layers."""

    def test_basic_integration(self, temp_data_dir):
        """Test basic integration between D1 and D2 layers."""
        # Create sample data with debugging enabled
        data_path, metadata = create_sample_data(temp_data_dir, debug=True)

        # Print column names for debugging
        print("DEBUG: Checking CSV file structure")
        df = pd.read_csv(data_path)
        print(f"DEBUG: CSV columns: {df.columns.tolist()}")
        print(f"DEBUG: Group values: {df['group_id'].unique().tolist()}")
        print(f"DEBUG: First few rows:\n{df.head()}")

        # Save the CSV file for inspection
        debug_csv_path = os.path.join(temp_data_dir, "debug_data.csv")
        df.to_csv(debug_csv_path, index=False)
        print(f"DEBUG: Saved debug CSV to {debug_csv_path}")

        # Initialize D1 layer with group_cols as a list
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],  # This is a list ["group_id"]
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )

        print(f"DEBUG: D1 initialized with group_cols={d1.group_cols}")
        print(f"DEBUG: D1 has {len(d1)} groups")
        print(f"DEBUG: D1 group info: {list(d1.group_info.keys())[:3]}...")

        # Initialize D2 layer
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        # Test basic properties
        assert len(d2.train_dataset) > 0, "D2 train dataset should have samples"
        assert hasattr(d2, "train_dataset"), "D2 should have train_dataset"
        assert hasattr(d2, "val_dataset"), "D2 should have val_dataset"
        assert hasattr(d2, "test_dataset"), "D2 should have test_dataset"

        # Test data access
        sample = d2.dataset[0]
        assert isinstance(sample, tuple), "D2 __getitem__ should return a tuple"
        assert len(sample) == 2, "D2 __getitem__ should return a tuple of (x, y)"

        x, y = sample
        assert isinstance(x, dict), "First element of tuple should be a dict"
        assert isinstance(y, torch.Tensor), "Second element of tuple should be a tensor"

        # Check that x contains all expected keys
        expected_keys = [
            "past_features",
            "future_targets",
            "x_num_past",
            "x_cat_past",
            "idx_target",
            "y",
        ]
        for key in expected_keys:
            assert key in x, f"Missing key in output dict: {key}"

        # Test index mappings
        assert "idx_known_num" in x, "Missing idx_known_num in output"
        assert "idx_unknown_num" in x, "Missing idx_unknown_num in output"
        assert "idx_known_cat" in x, "Missing idx_known_cat in output"
        assert "idx_unknown_cat" in x, "Missing idx_unknown_cat in output"
        assert "idx_target" in x, "Missing idx_target in output"

        # Test categorical cardinality
        assert "categorical_cardinality_past" in x, "Missing categorical_cardinality_past in output"

        # Test dataloader
        train_loader = d2.train_dataloader()
        assert isinstance(train_loader, DataLoader), "train_dataloader should return a DataLoader"

        # Get a batch from the dataloader
        batch = next(iter(train_loader))
        assert isinstance(batch, dict), "Batch should be a dict (from custom_collate_fn)"

        # Check batch dimensions
        assert batch["x_num_past"].shape[0] > 0, "Batch should have samples"
        assert batch["x_num_past"].shape[1] == 10, "Past length should be 10"
        assert batch["y"].shape[1] == 5, "Future length should be 5"

    def test_empty_group_cols(self, temp_data_dir):
        """Test integration with empty group_cols."""
        # Create sample data with debugging
        data_path, metadata = create_sample_data(temp_data_dir, debug=True)

        # Print column names for debugging
        print("DEBUG: Checking CSV file structure for empty group test")
        df = pd.read_csv(data_path)
        print(f"DEBUG: CSV columns: {df.columns.tolist()}")

        # Initialize D1 layer with empty group_cols
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=None,  # Empty group_cols
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )

        # Initialize D2 layer
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        # Test basic properties
        assert len(d2.dataset) > 0, "D2 dataset should have samples"

        # Test data access
        sample = d2.dataset[0]
        x, y = sample

        # Check that everything works with empty group_cols
        assert "group_id" in x, "group_id should be present even with empty group_cols"

        # Test dataloader
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))

        # Check batch dimensions
        assert batch["x_num_past"].shape[0] > 0, "Batch should have samples"

    def test_known_unknown_features(self, temp_data_dir):
        """Test integration with known/unknown feature specification."""
        # Create sample data with explicit known/unknown numerical columns
        data_path, metadata = create_sample_data(temp_data_dir, include_unknown=True)

        # Extract known/unknown numerical and categorical columns for clarity
        num_cols = metadata["num_cols"]
        cat_cols = metadata["cat_cols"]

        # For this test, we need to be explicit about which columns are numerical vs categorical
        # and which are known vs unknown
        known_num_cols = [col for col in metadata["known_cols"] if col in num_cols]
        unknown_num_cols = [col for col in metadata["unknown_cols"] if col in num_cols]
        known_cat_cols = [col for col in metadata["known_cols"] if col in cat_cols]
        unknown_cat_cols = [col for col in metadata["unknown_cols"] if col in cat_cols]

        # DEBUG: Print metadata to verify known/unknown splits
        print(f"DEBUG: METADATA: {metadata}")
        print(f"DEBUG: Known numerical columns: {known_num_cols}")
        print(f"DEBUG: Unknown numerical columns: {unknown_num_cols}")
        print(f"DEBUG: Known categorical columns: {known_cat_cols}")
        print(f"DEBUG: Unknown categorical columns: {unknown_cat_cols}")

        # Initialize D1 layer with known/unknown columns
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],
            num_cols=num_cols,  # Explicitly set numerical columns
            cat_cols=cat_cols,  # Explicitly set categorical columns
            known_cols=metadata["known_cols"],
            unknown_cols=metadata["unknown_cols"],
        )

        # Initialize D2 layer
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        # Test data access
        sample = d2.dataset[0]
        x, y = sample
        print("Sample for testing known and unkown features: ", type(sample), end="\n")
        print("Input (x) keys:")
        for key in x:
            print(key, "\n", x[key])
        print("Target (y) shape:", y.shape)

        # Check index mappings
        assert len(x["idx_known_num"]) > 0, "idx_known_num should not be empty"
        assert len(x["idx_unknown_num"]) > 0, "idx_unknown_num should not be empty"

        # Test dataloader
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))

        # Check that index mappings are preserved in batch
        assert "idx_known_num" in batch, "idx_known_num should be in batch"
        assert "idx_unknown_num" in batch, "idx_unknown_num should be in batch"
        assert len(batch["idx_known_num"]) > 0, "idx_known_num should not be empty in batch"

    def test_target_in_decoder(self, temp_data_dir):
        """Test integration with target in decoder option."""
        # Create sample data
        data_path, metadata = create_sample_data(temp_data_dir)

        # Initialize D1 layer
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )

        # Initialize D2 layer with include_target_in_decoder=True
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
            include_target_in_decoder=True,
        )

        # Test data access
        sample = d2.dataset[0]
        x, y = sample

        # Check that decoder_target is present
        assert (
            "decoder_target" in x
        ), "decoder_target should be present when include_target_in_decoder=True"

        # Test dataloader
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))

        # Check that decoder_target is preserved in batch
        assert "decoder_target" in batch, "decoder_target should be in batch"
        assert (
            batch["decoder_target"].shape == batch["y"].shape
        ), "decoder_target should have same shape as y"

    def test_categorical_features(self, temp_data_dir):
        """Test integration with categorical features."""
        # Create sample data with more categorical features
        data_path, metadata = create_sample_data(temp_data_dir, n_cat_features=5)

        # Initialize D1 layer
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )

        # Initialize D2 layer
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        # Test data access
        sample = d2.dataset[0]
        x, y = sample

        # Check categorical features
        assert "x_cat_past" in x, "x_cat_past should be present"
        assert x["x_cat_past"].shape[0] == 10, "x_cat_past should have past_len rows"
        assert x["x_cat_past"].shape[1] == len(
            metadata["cat_cols"]
        ), "x_cat_past should have correct number of columns"

        # Check categorical cardinality
        assert "categorical_cardinality_past" in x, "categorical_cardinality_past should be present"
        assert len(x["categorical_cardinality_past"]) == len(
            metadata["cat_cols"]
        ), "categorical_cardinality_past should match number of categorical columns"

        # Test dataloader
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))

        # Check that categorical features are preserved in batch
        assert "x_cat_past" in batch, "x_cat_past should be in batch"
        assert (
            "categorical_cardinality_past" in batch
        ), "categorical_cardinality_past should be in batch"

    def test_multiple_targets(self, temp_data_dir):
        """Test integration with multiple targets."""
        # Create sample data with multiple targets
        data_path, metadata = create_sample_data(temp_data_dir, n_targets=3)

        # Initialize D1 layer
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )

        # Initialize D2 layer
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )

        # Test data access
        sample = d2.dataset[0]
        x, y = sample

        # Check target dimensions
        assert y.shape[-1] == len(
            metadata["target_cols"]
        ), "Target tensor should have correct number of columns"

        # Test dataloader
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))

        # Check that target dimensions are preserved in batch
        assert batch["y"].shape[-1] == len(
            metadata["target_cols"]
        ), "Target tensor should have correct number of columns in batch"


if __name__ == "__main__":
    # For manual testing
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
