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
    n_groups = 3

    data = []
    for g in range(n_groups):
        for t in range(n_timesteps):
            timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=t * 10)
            data.append(
                {
                    "date": timestamp,
                    "group_id": f"group_{g}",
                    "OT": np.random.randn(),
                    "OT2": np.random.randn(),  # Second target
                    "num_feature": np.random.randn(),
                    "cat_feature": np.random.choice(["A", "B", "C"]),
                    "cat_feature_2": np.random.choice(["X", "Y", "Z"]),
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
        # Test Case 1: Basic temporal enrichment
        print("\n=== TEST CASE 1: Basic temporal enrichment ===")
        d1 = MultiSourceTSDataSet(
            file_paths=[csv_path],
            group_cols=["group_id"],
            time_col="date",
            target_cols=["OT"],
            num_cols=["OT", "num_feature"],
            cat_cols=["cat_feature", "cat_feature_2"],
            enrich_cat=["minute", "hour"],
            memory_efficient=False,
        )

        # Test Case 2: Multiple targets
        print("\n=== TEST CASE 2: Multiple targets ===")
        d2 = MultiSourceTSDataSet(
            file_paths=[csv_path],
            group_cols=["group_id"],
            time_col="date",
            target_cols=["OT", "OT2"],
            num_cols=["OT", "OT2", "num_feature"],
            cat_cols=["cat_feature"],
            enrich_cat=["dow", "hour"],
            memory_efficient=False,
        )

        # Test Case 3: No temporal enrichment
        print("\n=== TEST CASE 3: No temporal enrichment ===")
        d3 = MultiSourceTSDataSet(
            file_paths=[csv_path],
            group_cols=["group_id"],
            time_col="date",
            target_cols=["OT"],
            num_cols=["OT", "num_feature"],
            cat_cols=["cat_feature"],
            enrich_cat=None,
            memory_efficient=False,
        )

        # Run verifications for each test case
        for i, (name, dataset) in enumerate(
            [("Basic", d1), ("Multi-target", d2), ("No enrichment", d3)], 1
        ):
            print(f"\n=== Verifying {name} ===")
            print(f"Dataset {i} length: {len(dataset)}")
            print(f"Number of groups: {dataset.metadata['n_groups']}")

            # Print metadata
            print("\n=== FULL METADATA ===")
            for key, value in dataset.metadata.items():
                if isinstance(value, (list, dict)) and len(str(value)) > 100:
                    print(f"{key}: {type(value)} (length: {len(value)})")
                else:
                    print(f"{key}: {value}")

            # Test data access
            sample = dataset[0]
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"Sample x shape: {sample['x'].shape}")
            print(f"Sample y shape: {sample['y'].shape}")
            print(f"group_id (integer label): {sample['group_id']}")
            if "group_mapping" in dataset.metadata:
                print(f"group_mapping (int -> label): {dataset.metadata['group_mapping']}")
            print(
                "Note: group_id is the integer encoding; "
                "use group_mapping to recover the original label."
            )
            print(f"✅ {name} test passed!")

        return d1, d2, d3

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
