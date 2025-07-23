import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

# Import from new structure only
from dsipts.data_structure.d1_layers import BaseD1Layer, MultiSourceTSDataSet
from dsipts.data_structure.d1_layers.utils import extend_time_df_test_case

# Use the test function directly
extend_time_df = extend_time_df_test_case


@pytest.fixture
def test_data():
    """Fixture to create test data for D1 layer tests."""
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()

    # Generate two CSV files with different groups
    for file_idx in range(2):
        data = []

        # Generate data for each group
        for group_idx in range(3):
            # Determine which file gets which groups
            if (group_idx % 2 == 0 and file_idx == 0) or (group_idx % 2 == 1 and file_idx == 1):
                # Generate time series for this group
                for t in range(10):
                    row = {
                        "group": f"group_{group_idx}",
                        "time": t,
                        "feature_0": np.sin(t / 10 + group_idx) + np.random.normal(0, 0.1),
                        "feature_1": np.cos(t / 10 + group_idx) + np.random.normal(0, 0.1),
                        "target_0": np.sin(t / 5 + group_idx) + np.random.normal(0, 0.1),
                        "cat_feature": f"cat_{np.random.randint(0, 3)}",
                        "static_feature": float(group_idx) * 10,
                    }
                    data.append(row)

        # Create DataFrame and save to CSV
        if data:  # Only create file if there's data for this file
            df = pd.DataFrame(data)
            # Make sure to save the CSV with all columns
            df.to_csv(os.path.join(temp_dir, f"test_data_{file_idx}.csv"), index=False)
        else:
            # Create an empty file with the correct columns to avoid issues
            empty_df = pd.DataFrame(
                columns=[
                    "group",
                    "time",
                    "feature_0",
                    "feature_1",
                    "target_0",
                    "cat_feature",
                    "static_feature",
                ]
            )
            empty_df.to_csv(os.path.join(temp_dir, f"test_data_{file_idx}.csv"), index=False)

    # Verify the files were created correctly
    for file_idx in range(2):
        file_path = os.path.join(temp_dir, f"test_data_{file_idx}.csv")
        df = pd.read_csv(file_path)
        # Ensure all expected columns are present
        assert "feature_0" in df.columns
        assert "feature_1" in df.columns
        assert "target_0" in df.columns

    # Define common parameters
    params = {
        "file_paths": [os.path.join(temp_dir, f"test_data_{i}.csv") for i in range(2)],
        "group_cols": "group",
        "time_col": "time",
        "feature_cols": ["feature_0", "feature_1"],
        "target_cols": ["target_0"],
        "cat_cols": ["cat_feature"],
        "static_cols": ["static_feature"],
    }

    # Yield the test data parameters
    yield params

    # Clean up temporary files
    shutil.rmtree(temp_dir)


def test_init_memory_efficient_false(test_data):
    """Test initialization with memory_efficient=False."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        memory_efficient=False,
    )

    # Check basic attributes
    assert (
        d1_dataset.group_cols == [test_data["group_cols"]]
        if isinstance(test_data["group_cols"], str)
        else test_data["group_cols"]
    )
    assert d1_dataset.time_col == test_data["time_col"]
    assert d1_dataset.feature_cols == test_data["feature_cols"]
    assert d1_dataset.target_cols == test_data["target_cols"]
    assert d1_dataset.cat_cols == test_data["cat_cols"]
    assert not d1_dataset.memory_efficient

    # Check that basic functionality works
    assert len(d1_dataset) > 0

    # Check that data was preloaded
    assert len(d1_dataset.data_cache) > 0

    # Check that internal structures were created
    assert hasattr(d1_dataset, "group_info")
    assert hasattr(d1_dataset, "cumulative_lengths")
    assert len(d1_dataset.cumulative_lengths) > 1


def test_init_memory_efficient_true(test_data):
    """Test initialization with memory_efficient=True."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        memory_efficient=True,
    )

    # Check that the dataset was initialized correctly
    assert len(d1_dataset.file_paths) == 2
    assert d1_dataset.time_col == "time"
    assert d1_dataset.group_cols == ["group"]
    assert d1_dataset.feature_cols == ["feature_0", "feature_1"]
    assert d1_dataset.target_cols == ["target_0"]
    assert d1_dataset.cat_cols == ["cat_feature"]
    assert d1_dataset.memory_efficient

    # Check that data was not preloaded
    assert len(d1_dataset.data_cache) == 0


def test_getitem(test_data):
    """Test __getitem__ method."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        static_cols=test_data["static_cols"],
        memory_efficient=False,
    )

    # Print dataset info
    print(f"Dataset contains {len(d1_dataset)} groups")

    # Get and inspect first group
    group_data = d1_dataset[0]

    # Print structured group data
    print("\nGroup data contents:")
    for key, value in group_data.items():
        if hasattr(value, "shape"):
            print(f"{key}: {type(value)} with shape {value.shape}")
        else:
            print(f"{key}: {type(value)} | {value}")

    # Verify expected structure (new format)
    assert isinstance(group_data, dict), "Should return a dictionary"
    assert "x" in group_data, "Missing features (x)"
    assert "y" in group_data, "Missing targets (y)"
    assert "group_id" in group_data, "Missing group identifier"
    assert "past_time" in group_data, "Missing past_time"
    assert "future_time" in group_data, "Missing future_time"

    # Check that x and y are tensors
    assert torch.is_tensor(group_data["x"]), "Features should be a tensor"
    assert torch.is_tensor(group_data["y"]), "Targets should be a tensor"

    # Check dimensions
    assert group_data["x"].shape[0] == len(test_data["feature_cols"])
    assert group_data["y"].shape[0] == len(test_data["target_cols"])


def test_known_unknown_cols(test_data):
    """Test specifying known and unknown columns."""
    # Create dataset with custom known/unknown columns
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        known_cols=["feature_0"],  # Only feature_0 is known at prediction time
        unknown_cols=["feature_1", "target_0"],  # feature_1 and target_0 are unknown
    )

    # Check that the columns were correctly categorized
    assert d1_dataset.known_cols == ["feature_0"]
    assert d1_dataset.unknown_cols == ["feature_1", "target_0"]

    # Check that the dataset was initialized correctly
    assert len(d1_dataset) > 0


def test_load_group_data(test_data):
    """Test _load_group_data_on_demand method."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        static_cols=test_data["static_cols"],
        memory_efficient=True,  # Test with memory efficient mode
    )

    # Get data for the first group
    # The file_group_key is a tuple of (file_idx, group_id)
    # We need to find a valid file_group_key from the _group_ids
    file_group_key = d1_dataset._group_ids[0]

    # Get data for this file-group combination
    group_data = d1_dataset._load_group_data_on_demand(file_group_key)

    # Check that the returned data has the expected format
    assert isinstance(group_data, pd.DataFrame)
    assert test_data["time_col"] in group_data.columns
    for col in test_data["feature_cols"]:
        assert col in group_data.columns
    for col in test_data["target_cols"]:
        assert col in group_data.columns

    # Test that data loading works
    # Load data again to test consistency
    group_data2 = d1_dataset._load_group_data_on_demand(file_group_key)

    # Both should be identical
    assert np.array_equal(group_data.values, group_data2.values)


def test_len(test_data):
    """Test __len__ method."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        memory_efficient=False,
    )

    # Check that the length is correct
    # The new implementation returns total samples, not groups
    # So we expect more than 3 (it should be the total number of rows)
    assert len(d1_dataset) > 3


def test_static_cols(test_data):
    """Test handling of static columns."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        static_cols=["static_feature"],
        memory_efficient=False,
    )

    # Check that static columns were set correctly
    assert d1_dataset.static_cols == ["static_feature"]

    # Get data for the first group
    group_data = d1_dataset[0]

    # Check that static columns are included
    assert "static_features" in group_data
    assert group_data["static_features"].shape[0] == 1  # One static feature


def test_data_caching(test_data):
    """Test data caching behavior."""
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        static_cols=test_data["static_cols"],
        memory_efficient=False,
    )

    # First access should load data
    group_data1 = d1_dataset[0]

    # Second access should use cache
    group_data2 = d1_dataset[0]

    # Both should be identical
    assert torch.equal(group_data1["x"], group_data2["x"])
    assert torch.equal(group_data1["y"], group_data2["y"])
    assert group_data1["group_id"] == group_data2["group_id"]

    # Check that cache is working
    assert len(d1_dataset.data_cache) > 0


def test_extend_time_df():
    """Test the extend_time_df_test_case function."""
    from dsipts.data_structure.d1_layers.utils import extend_time_df_test_case

    # Create sample data with gaps
    df = pd.DataFrame({"time": [0, 2, 4], "feature": [1.0, 2.0, 3.0], "group": ["A", "A", "A"]})

    # Extend the time series
    extended_df = extend_time_df_test_case(df)

    # Check that gaps were filled
    assert len(extended_df) == 5  # Should now have rows for t=0,1,2,3,4
    assert list(extended_df["time"].sort_values()) == [0, 1, 2, 3, 4]

    # Check that feature column exists in extended_df
    assert "feature" in extended_df.columns

    # Check that rows for t=1 and t=3 have NaN for feature column
    t1_row = extended_df[extended_df["time"] == 1]
    t3_row = extended_df[extended_df["time"] == 3]
    assert len(t1_row) == 1
    assert len(t3_row) == 1
    assert pd.isna(t1_row["feature"].iloc[0])
    assert pd.isna(t3_row["feature"].iloc[0])


def test_d1_base_class_interface():
    """Test that BaseD1Layer defines the correct interface."""
    # Check that BaseD1Layer is abstract
    with pytest.raises(TypeError):
        BaseD1Layer()

    # Check required methods and properties
    required_methods = ["__len__", "__getitem__"]
    required_properties = [
        "group_cols",
        "target_cols",
        "feature_cols",
        "cat_cols",
        "known_cols",
        "unknown_cols",
    ]

    for method in required_methods:
        assert hasattr(BaseD1Layer, method)

    for prop in required_properties:
        assert hasattr(BaseD1Layer, prop)


def test_backward_compatibility_imports():
    """Test that legacy D1 imports still work."""
    # Test legacy imports from main module
    from dsipts.data_structure import LegacyMultiSourceTSDataSet, MultiSourceTSDataSet

    # These should be importable without errors and should be the same class
    assert LegacyMultiSourceTSDataSet is not None
    assert LegacyMultiSourceTSDataSet is MultiSourceTSDataSet  # Should be an alias


def test_temporal_enrichment_feature(test_data):
    """Test temporal categorical enrichment feature."""
    # Test with temporal enrichment enabled
    d1_dataset = MultiSourceTSDataSet(
        file_paths=test_data["file_paths"],
        group_cols=test_data["group_cols"],
        time_col=test_data["time_col"],
        feature_cols=test_data["feature_cols"],
        target_cols=test_data["target_cols"],
        cat_cols=test_data["cat_cols"],
        memory_efficient=False,
        enrich_cat=["hour", "dow", "month"],
    )

    # Since the test data has integer time column, temporal enrichment will be skipped
    # The cat_cols should remain unchanged
    assert d1_dataset.cat_cols == test_data["cat_cols"]

    # Test that the dataset was initialized correctly
    assert len(d1_dataset) > 0
