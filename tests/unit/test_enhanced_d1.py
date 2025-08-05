#!/usr/bin/env python3
"""
Test script to verify enhanced D1 layer functionality.
"""

import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, "/home/sandeep/DSIPTS_PTF")

from dsipts.data_structure.d1_layers import MultiSourceTSDataSet


def create_test_dataframe():
    """Create a test DataFrame for testing."""
    np.random.seed(42)

    # Create sample time series data
    n_groups = 3
    n_timesteps = 50

    data = []
    for group_id in range(n_groups):
        for t in range(n_timesteps):
            data.append(
                {
                    "group_id": f"group_{group_id}",
                    "time": t,
                    "feature_1": np.random.randn(),
                    "feature_2": np.random.randn(),
                    "cat_feature": np.random.choice(["A", "B", "C"]),
                    "target": np.random.randn(),
                }
            )

    return pd.DataFrame(data)


def test_dataframe_input():
    """Test D1 layer with DataFrame input."""
    print("Testing D1 layer with DataFrame input...")

    # Create test DataFrame
    df = create_test_dataframe()
    print(f"Created test DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Initialize D1 layer with DataFrame
    d1 = MultiSourceTSDataSet(
        dataframes=[df],
        group_cols=["group_id"],
        time_col="time",
        target_cols=["target"],
        cat_cols=["cat_feature"],
        num_cols=["feature_1", "feature_2"],
        memory_efficient=False,
    )

    print(f"D1 dataset length: {len(d1)}")
    print(f"Number of groups: {d1.metadata['n_groups']}")

    # Test new metadata structure
    print("\n=== NEW METADATA STRUCTURE ===")
    print(f"idx_categorical: {d1.metadata['idx_categorical']}")
    print(f"idx_known_future: {d1.metadata['idx_known_future']}")
    print(f"idx_unknown_future: {d1.metadata['idx_unknown_future']}")
    print(f"idx_targets: {d1.metadata['idx_targets']}")
    print(f"n_future_groups: {d1.metadata['n_future_groups']}")

    if "categorical_mappings" in d1.metadata:
        print(f"categorical_mappings: {d1.metadata['categorical_mappings']}")

    # Test data access
    sample = d1[0]
    print(f"\nSample keys: {sample.keys()}")

    print("✅ DataFrame input test passed!")
    return d1


def test_csv_input():
    """Test D1 layer with CSV file input."""
    print("\nTesting D1 layer with CSV file input...")

    # Create test CSV file
    df = create_test_dataframe()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

    try:
        # Initialize D1 layer with CSV file
        d1 = MultiSourceTSDataSet(
            file_paths=[csv_path],
            group_cols=["group_id"],
            time_col="time",
            target_cols=["target"],
            cat_cols=["cat_feature"],
            num_cols=["feature_1", "feature_2"],
            memory_efficient=False,
        )

        print(f"D1 dataset length: {len(d1)}")
        print(f"Number of groups: {d1.metadata['n_groups']}")

        # Test new metadata structure
        print("\n=== NEW METADATA STRUCTURE ===")
        print(f"idx_categorical: {d1.metadata['idx_categorical']}")
        print(f"idx_targets: {d1.metadata['idx_targets']}")
        print(f"n_future_groups: {d1.metadata['n_future_groups']}")

        print("✅ CSV input test passed!")
        return d1

    finally:
        # Clean up
        os.unlink(csv_path)


def main():
    """Run all tests."""
    print("Testing Enhanced D1 Layer Functionality")
    print("=" * 50)

    try:
        # Test DataFrame input
        d1_df = test_dataframe_input()

        # Test CSV input
        d1_csv = test_csv_input()

        print("\n" + "=" * 50)
        print("✅ All tests passed! Enhanced D1 layer is working correctly.")

        # Compare metadata structures
        print("\n=== METADATA COMPARISON ===")
        print("DataFrame metadata keys:", sorted(d1_df.metadata.keys()))
        print("CSV metadata keys:", sorted(d1_csv.metadata.keys()))

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import os
    import sys

    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, "test_enhanced_d1.log")

    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

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

    sys.exit(main())
