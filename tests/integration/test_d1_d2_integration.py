"""Integration test for D1 and D2 layers.

This module tests the integration between D1 (MultiSourceTSDataSet) and D2 (EncoderDecoder)
layers without involving models. It verifies that the data pipeline works correctly
with various configurations and edge cases.
"""

import logging
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

    def _log_test_start(self, test_name):
        """Helper method to log test start with visual separation."""
        logger = logging.getLogger(__name__)
        logger.info("\n" + "=" * 80)
        logger.info(f"STARTING TEST: {test_name}")
        logger.info("=" * 80)
        return logger

    def _log_test_end(self, logger, test_name):
        """Helper method to log test completion."""
        logger.info("-" * 80)
        logger.info(f"COMPLETED TEST: {test_name}")
        logger.info("=" * 80 + "\n")

    def test_basic_integration(self, temp_data_dir):
        """Test basic integration between D1 and D2 layers."""
        logger = self._log_test_start("test_basic_integration")

        # Create sample data with debugging enabled
        logger.info("\n[1/5] CREATING SAMPLE DATA")
        logger.info("-" * 40)
        data_path, metadata = create_sample_data(temp_data_dir, debug=True)
        logger.info(f"✓ Sample data created at: {data_path}")
        logger.info(f"✓ Metadata keys: {list(metadata.keys())}")

        # Log important metadata in a more readable format
        logger.info("\n[2/5] DATASET METADATA")
        logger.info("-" * 40)
        logger.info(f"Time Column: {metadata['time_col']}")
        logger.info(f"Target Columns: {metadata['target_cols']}")
        logger.info(f"Feature Columns: {metadata.get('num_cols', [])}")
        logger.info(f"Categorical Columns: {metadata.get('cat_cols', [])}")
        logger.info(f"Group Columns: {metadata.get('group_cols', [])}")
        logger.info(f"Number of Groups: {len(metadata.get('groups', []))}")
        logger.info(f"Number of Timesteps: {metadata.get('n_timesteps', 'N/A')}")

        # Print column names for debugging
        logger.info("Checking CSV file structure")
        df = pd.read_csv(data_path)
        logger.info(f"CSV columns: {df.columns.tolist()}")
        logger.info(f"Group values: {df['group_id'].unique().tolist()}")
        logger.debug(f"First few rows:\n{df.head()}")

        # Save the CSV file for inspection
        debug_csv_path = os.path.join(temp_data_dir, "debug_data.csv")
        df.to_csv(debug_csv_path, index=False)
        logger.info(f"Saved debug CSV to {debug_csv_path}")

        # Initialize D1 layer with group_cols as a list
        logger.info("Initializing D1 layer")
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],  # This is a list ["group_id"]
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )
        logger.info(f"D1 initialized with group_cols={d1.group_cols}")
        logger.info(f"D1 has {len(d1)} groups")
        logger.debug(f"D1 group info: {list(d1.group_info.keys())[:3]}...")

        # Initialize D2 layer
        logger.info("Initializing D2 layer")
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )
        logger.info("D2 layer initialized successfully")

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
        logger = self._log_test_start("test_empty_group_cols")

        # Create sample data with debugging
        logger.info("Creating sample data with debugging enabled")
        data_path, metadata = create_sample_data(temp_data_dir, debug=True)
        logger.info(f"Sample data created at {data_path}")

        # Print column names for debugging
        logger.info("Checking CSV file structure for empty group test")
        df = pd.read_csv(data_path)
        logger.info(f"CSV columns: {df.columns.tolist()}")

        # Initialize D1 layer with empty group_cols
        logger.info("Initializing D1 layer with empty group_cols")
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=None,  # Empty group_cols
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )
        logger.info("D1 layer initialized with empty group_cols")

        # Initialize D2 layer
        logger.info("Initializing D2 layer")
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )
        logger.info("D2 layer initialized successfully")

        # Test basic properties
        logger.info("Testing basic properties")
        assert len(d2.dataset) > 0, "D2 dataset should have samples"
        logger.info("Basic properties test passed")

        # Test data access
        logger.info("Testing data access")
        sample = d2.dataset[0]
        x, y = sample
        logger.info("Data access successful")

        # Check that everything works with empty group_cols
        logger.info("Checking group_id presence")
        assert "group_id" in x, "group_id should be present even with empty group_cols"
        logger.info("group_id check passed")

        # Test dataloader
        logger.info("Testing dataloader")
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))
        logger.info("Dataloader test successful")

        # Check batch dimensions
        logger.info("Checking batch dimensions")
        assert batch["x_num_past"].shape[0] > 0, "Batch should have samples"
        logger.info("Batch dimensions check passed")
        logger.info("All assertions passed for empty group_cols test")

    def test_known_unknown_features(self, temp_data_dir):
        """Test integration with known/unknown feature specification."""
        logger = self._log_test_start("test_known_unknown_features")

        # Create sample data with explicit known/unknown numerical columns
        logger.info("Creating sample data with explicit known/unknown columns")
        data_path, metadata = create_sample_data(temp_data_dir, include_unknown=True)
        logger.info(f"Sample data created at {data_path}")

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
        logger.debug(f"METADATA: {metadata}")
        logger.debug(f"Known numerical columns: {known_num_cols}")
        logger.debug(f"Unknown numerical columns: {unknown_num_cols}")
        logger.debug(f"Known categorical columns: {known_cat_cols}")
        logger.debug(f"Unknown categorical columns: {unknown_cat_cols}")

        # Initialize D1 layer with known/unknown columns
        logger.info("Initializing D1 layer with known/unknown columns")
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
        logger.info("D1 layer initialized with known/unknown columns")

        # Initialize D2 layer
        logger.info("Initializing D2 layer")
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )
        logger.info("D2 layer initialized successfully")

        # Test data access
        logger.info("Testing data access")
        sample = d2.dataset[0]
        x, y = sample
        logger.info("Data access successful")
        logger.debug("Sample for testing known and unknown features: %s", type(sample))
        logger.debug("Input (x) keys:")
        for key in x:
            logger.debug("%s: %s", key, x[key])
        logger.debug("Target (y) shape: %s", y.shape)

        # Check index mappings
        logger.info("Checking index mappings")
        assert len(x["idx_known_num"]) > 0, "idx_known_num should not be empty"
        assert len(x["idx_unknown_num"]) > 0, "idx_unknown_num should not be empty"
        logger.info("Index mappings check passed")

        # Test dataloader
        logger.info("Testing dataloader")
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))
        logger.info("Dataloader test successful")

        # Check that index mappings are preserved in batch
        logger.info("Checking index mappings in batch")
        assert "idx_known_num" in batch, "idx_known_num should be in batch"
        assert "idx_unknown_num" in batch, "idx_unknown_num should be in batch"
        assert len(batch["idx_known_num"]) > 0, "idx_known_num should not be empty in batch"
        logger.info("Index mappings in batch check passed")
        logger.info("All assertions passed for known/unknown features test")

    def test_target_in_decoder(self, temp_data_dir):
        """Test integration with target in decoder option."""
        logger = self._log_test_start("test_target_in_decoder")

        # Create sample data
        logger.info("Creating sample data")
        data_path, metadata = create_sample_data(temp_data_dir)
        logger.info(f"Sample data created at {data_path}")

        # Initialize D1 layer
        logger.info("Initializing D1 layer")
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )
        logger.info("D1 layer initialized successfully")

        # Initialize D2 layer with include_target_in_decoder=True
        logger.info("Initializing D2 layer with include_target_in_decoder=True")
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
            include_target_in_decoder=True,
        )
        logger.info("D2 layer initialized with include_target_in_decoder=True")

        # Test data access
        logger.info("Testing data access")
        sample = d2.dataset[0]
        x, y = sample
        logger.info("Data access successful")

        # Check that decoder_target is present
        logger.info("Checking decoder_target presence")
        assert (
            "decoder_target" in x
        ), "decoder_target should be present when include_target_in_decoder=True"
        logger.info("decoder_target presence check passed")

        # Test dataloader
        logger.info("Testing dataloader")
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))
        logger.info("Dataloader test successful")

        # Check that decoder_target is preserved in batch
        logger.info("Checking decoder_target in batch")
        assert "decoder_target" in batch, "decoder_target should be in batch"
        assert (
            batch["decoder_target"].shape == batch["y"].shape
        ), "decoder_target should have same shape as y"
        logger.info("decoder_target in batch check passed")
        logger.info("All assertions passed for target in decoder test")

    def test_categorical_features(self, temp_data_dir):
        """Test integration with categorical features."""
        logger = self._log_test_start("test_categorical_features")

        # Create sample data with more categorical features
        logger.info("Creating sample data with more categorical features")
        data_path, metadata = create_sample_data(temp_data_dir, n_cat_features=5)
        logger.info(f"Sample data created at {data_path}")

        # Initialize D1 layer
        logger.info("Initializing D1 layer")
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )
        logger.info("D1 layer initialized successfully")

        # Initialize D2 layer
        logger.info("Initializing D2 layer")
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )
        logger.info("D2 layer initialized successfully")

        # Test data access
        logger.info("Testing data access")
        sample = d2.dataset[0]
        x, y = sample
        logger.info("Data access successful")

        # Check categorical features
        logger.info("Checking categorical features")
        assert "x_cat_past" in x, "x_cat_past should be present"
        assert x["x_cat_past"].shape[0] == 10, "x_cat_past should have past_len rows"
        assert x["x_cat_past"].shape[1] == len(
            metadata["cat_cols"]
        ), "x_cat_past should have correct number of columns"
        logger.info("Categorical features check passed")

        # Check categorical cardinality
        logger.info("Checking categorical cardinality")
        assert "categorical_cardinality_past" in x, "categorical_cardinality_past should be present"
        assert len(x["categorical_cardinality_past"]) == len(
            metadata["cat_cols"]
        ), "categorical_cardinality_past should match number of categorical columns"
        logger.info("Categorical cardinality check passed")

        # Test dataloader
        logger.info("Testing dataloader")
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))
        logger.info("Dataloader test successful")

        # Check that categorical features are preserved in batch
        logger.info("Checking categorical features in batch")
        assert "x_cat_past" in batch, "x_cat_past should be in batch"
        assert (
            "categorical_cardinality_past" in batch
        ), "categorical_cardinality_past should be in batch"
        logger.info("Categorical features in batch check passed")
        logger.info("All assertions passed for categorical features test")

    def test_multiple_targets(self, temp_data_dir):
        """Test integration with multiple targets."""
        logger = self._log_test_start("test_multiple_targets")

        # Create sample data with multiple targets
        logger.info("Creating sample data with multiple targets")
        data_path, metadata = create_sample_data(temp_data_dir, n_targets=3)
        logger.info(f"Sample data created at {data_path}")

        # Initialize D1 layer
        logger.info("Initializing D1 layer with multiple targets")
        d1 = MultiSourceTSDataSet(
            file_paths=[data_path],
            time_col=metadata["time_col"],
            target_cols=metadata["target_cols"],
            group_cols=metadata["group_cols"],
            num_cols=metadata["num_cols"],
            cat_cols=metadata["cat_cols"],
        )
        logger.info("D1 layer initialized successfully")

        # Initialize D2 layer
        logger.info("Initializing D2 layer")
        d2 = EncoderDecoder(
            d1_dataset=d1,
            past_len=10,
            future_len=5,
            batch_size=32,
            split_config=(0.7, 0.15, 0.15),
            precompute=True,
        )
        logger.info("D2 layer initialized successfully")

        # Test data access
        logger.info("Testing data access")
        sample = d2.dataset[0]
        x, y = sample
        logger.info("Data access successful")

        # Check target dimensions
        logger.info("Checking target dimensions")
        assert y.shape[-1] == len(
            metadata["target_cols"]
        ), "Target tensor should have correct number of columns"
        logger.info("Target dimensions check passed")

        # Test dataloader
        logger.info("Testing dataloader")
        train_loader = d2.train_dataloader()
        batch = next(iter(train_loader))
        logger.info("Dataloader test successful")

        # Check that target dimensions are preserved in batch
        logger.info("Checking target dimensions in batch")
        assert batch["y"].shape[-1] == len(
            metadata["target_cols"]
        ), "Target tensor should have correct number of columns in batch"
        logger.info("All assertions passed for multiple targets test")


if __name__ == "__main__":
    # For manual testing
    import os
    import sys

    import pytest

    # Get the absolute path for the log file
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, "test_d1_d2_integration.log")

    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    print(f"SAVING THE LOGS TO: {log_file}")

    # Clear any existing log handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure logging with both file and console handlers
    file_handler = logging.FileHandler(log_file, mode="w")  # 'w' to overwrite existing file
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the root logger
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info("Starting integration tests with verbose logging")

    # Run the tests
    exit_code = pytest.main(["-v", "--tb=short", __file__])

    logger.info(f"Integration tests completed with exit code: {exit_code}")
    sys.exit(exit_code)
