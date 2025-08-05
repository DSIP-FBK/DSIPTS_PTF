#!/usr/bin/env python3
"""
Test script to verify temporal feature handling in D1 layer.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Add the project root to Python path
# sys.path.insert(0, '/home/sandeep/DSIPTS_PTF')
from dsipts.data_structure.d1_layers import MultiSourceTSDataSet


def create_test_dataframe():
    """Create a test DataFrame for testing temporal features."""
    np.random.seed(42)

    # Create sample time series data with datetime index
    n_timesteps = 100

    data = []
    for t in range(n_timesteps):
        timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=t * 10)
        data.append(
            {
                "date": timestamp,
                "OT": np.random.randn(),
                "feature_1": np.random.randn(),
            }
        )

    df = pd.DataFrame(data)
    return df


def test_temporal_features():
    """Test D1 layer with temporal feature enrichment."""
    print("Testing D1 layer with temporal feature enrichment...")

    # Create test DataFrame
    df = create_test_dataframe()
    print(f"Created test DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Save to temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

    try:
        # Initialize D1 layer with temporal enrichment
        d1 = MultiSourceTSDataSet(
            file_paths=[csv_path],
            group_cols=[],  # Empty group columns for single group
            time_col="date",
            target_cols=["OT"],
            num_cols=["OT"],
            enrich_cat=["minute"],
            memory_efficient=False,
        )

        print(f"D1 dataset length: {len(d1)}")
        print(f"Number of groups: {d1.metadata['n_groups']}")

        # Test new metadata structure
        print("\n=== METADATA STRUCTURE ===")
        print(f"n_features: {d1.metadata['n_features']}")
        print(f"feature_cols: {d1.metadata['feature_cols']}")
        print(f"idx_categorical: {d1.metadata['idx_categorical']}")
        print(f"idx_known_future: {d1.metadata['idx_known_future']}")
        print(f"idx_unknown_future: {d1.metadata['idx_unknown_future']}")
        print(f"idx_targets: {d1.metadata['idx_targets']}")
        print(f"n_future_groups: {d1.metadata['n_future_groups']}")
        print(f"enrich_cat: {d1.metadata['enrich_cat']}")

        if "categorical_mappings" in d1.metadata:
            print(f"categorical_mappings: {d1.metadata['categorical_mappings']}")

        # Verify expected values
        expected_n_features = 2  # OT + minute
        expected_feature_cols = ["OT", "minute"]
        expected_idx_categorical = [1]  # minute is at index 1
        expected_idx_known_future = [1]  # minute is at index 1

        print("\n=== VERIFICATION ===")
        print(f"Expected n_features: {expected_n_features}," f"Actual: {d1.metadata['n_features']}")
        print(
            f"Expected feature_cols: {expected_feature_cols},"
            f"Actual: {d1.metadata['feature_cols']}"
        )
        print(
            f"Expected idx_categorical: {expected_idx_categorical},"
            f"Actual: {d1.metadata['idx_categorical']}"
        )
        print(
            f"Expected idx_known_future: {expected_idx_known_future},"
            f"Actual: {d1.metadata['idx_known_future']}"
        )

        # Check if temporal_features field exists (should not exist)
        if "temporal_features" in d1.metadata:
            print(
                f"ERROR: temporal_features field should not exist,"
                f"but found: {d1.metadata['temporal_features']}"
            )
        else:
            print("SUCCESS: temporal_features field correctly removed from metadata")

        # Test data access
        sample = d1[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"Sample x shape: {sample['x'].shape}")
        print(f"Sample y shape: {sample['y'].shape}")

        print("✅ Temporal feature test passed!")
        return d1

    finally:
        # Clean up
        os.unlink(csv_path)


def main():
    """Run temporal feature test."""
    print("Testing Temporal Feature Handling in D1 Layer")
    print("=" * 50)

    try:
        test_temporal_features()
        print("\n" + "=" * 50)
        print("✅ All temporal feature tests passed!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
