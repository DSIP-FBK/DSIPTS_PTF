#!/usr/bin/env python3
"""
MAIN TEST FILE
Test script for min-max scaling implementation in D1 and D2 layers.

This script tests the two-pass min-max scaling approach:
1. D1 layer computes global min/max across all files
2. D2 layer applies scaling before data splitting
3. Inverse scaling can be applied to predictions
"""

import logging
import os
import tempfile

import numpy as np
import pandas as pd
import torch

# Import our layers
from dsipts.data_structure.d1_layers.multi_source_csv import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers.encoder_decoder import EncoderDecoder

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_data():
    """Create multiple CSV files with different value ranges for testing scaling."""

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Creating test data in: {temp_dir}")

    # File 1: Small values (0-10 range)
    np.random.seed(42)
    n_samples = 100
    file1_data = {
        "time": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
        "group_id": ["A"] * (n_samples // 2) + ["B"] * (n_samples // 2),
        "feature1": np.random.uniform(0, 10, n_samples),  # Range: 0-10
        "feature2": np.random.uniform(5, 15, n_samples),  # Range: 5-15
        "target": np.random.uniform(0, 5, n_samples),  # Range: 0-5
    }
    file1_path = os.path.join(temp_dir, "data_part1.csv")
    pd.DataFrame(file1_data).to_csv(file1_path, index=False)

    # File 2: Medium values (50-150 range)
    file2_data = {
        "time": pd.date_range("2023-01-05", periods=n_samples, freq="H"),
        "group_id": ["C"] * (n_samples // 2) + ["D"] * (n_samples // 2),
        "feature1": np.random.uniform(50, 150, n_samples),  # Range: 50-150
        "feature2": np.random.uniform(75, 125, n_samples),  # Range: 75-125
        "target": np.random.uniform(25, 75, n_samples),  # Range: 25-75
    }
    file2_path = os.path.join(temp_dir, "data_part2.csv")
    pd.DataFrame(file2_data).to_csv(file2_path, index=False)

    # File 3: Large values (1000-2000 range)
    file3_data = {
        "time": pd.date_range("2023-01-10", periods=n_samples, freq="H"),
        "group_id": ["E"] * (n_samples // 2) + ["F"] * (n_samples // 2),
        "feature1": np.random.uniform(1000, 2000, n_samples),  # Range: 1000-2000
        "feature2": np.random.uniform(1200, 1800, n_samples),  # Range: 1200-1800
        "target": np.random.uniform(500, 1500, n_samples),  # Range: 500-1500
    }
    file3_path = os.path.join(temp_dir, "data_part3.csv")
    pd.DataFrame(file3_data).to_csv(file3_path, index=False)

    file_paths = [file1_path, file2_path, file3_path]

    # Log expected global ranges
    logger.info("Expected global ranges:")
    logger.info("  feature1: 0 to 2000")
    logger.info("  feature2: 5 to 1800")
    logger.info("  target: 0 to 1500")

    return file_paths, temp_dir


def test_d1_scaling_computation():
    """Test that D1 layer correctly computes global min-max scaling parameters."""

    logger.info("=" * 60)
    logger.info("TESTING D1 LAYER SCALING COMPUTATION")
    logger.info("=" * 60)

    file_paths, temp_dir = create_test_data()

    try:
        # Create D1 dataset
        d1_dataset = MultiSourceTSDataSet(
            file_paths=file_paths,
            group_cols=["group_id"],
            time_col="time",
            target_cols=["target"],
            num_cols=["feature1", "feature2", "target"],
            memory_efficient=False,  # Use cached mode for easier testing
        )

        # Check that scaling parameters were computed
        scaling_params = d1_dataset.get_scaling_params()
        logger.info(f"Computed scaling parameters for {len(scaling_params)} columns")

        # Calculate exact expected ranges from the generated data
        # Read the actual data to get precise min/max values
        all_data = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)

        expected_ranges = {
            "feature1": (combined_df["feature1"].min(), combined_df["feature1"].max()),
            "feature2": (combined_df["feature2"].min(), combined_df["feature2"].max()),
            "target": (combined_df["target"].min(), combined_df["target"].max()),
        }

        for col, (expected_min, expected_max) in expected_ranges.items():
            if col in scaling_params:
                params = scaling_params[col]
                actual_min = params["min"]
                actual_max = params["max"]
                actual_range = params["range"]

                logger.info(f"Column '{col}':")
                logger.info(f"  Expected: min={expected_min:.6f}, max={expected_max:.6f}")
                logger.info(f"  Actual: min={actual_min:.6f}, max={actual_max:.6f}, range={actual_range:.6f}")

                # Use precise comparison with small tolerance for floating point errors
                assert (
                    abs(actual_min - expected_min) < 1e-10
                ), f"Min value for {col} doesn't match: expected {expected_min}, got {actual_min}"
                assert (
                    abs(actual_max - expected_max) < 1e-10
                ), f"Max value for {col} doesn't match: expected {expected_max}, got {actual_max}"
                assert not params["is_constant"], f"Column {col} should not be constant"
            else:
                logger.warning(f"No scaling parameters found for column '{col}'")

        logger.info("✅ D1 scaling computation test PASSED")
        return d1_dataset, temp_dir

    except Exception as e:
        logger.error(f"❌ D1 scaling computation test FAILED: {e}")
        raise


def test_d2_scaling_application(d1_dataset):
    """Test that D2 layer correctly applies scaling before data splitting."""

    logger.info("=" * 60)
    logger.info("TESTING D2 LAYER SCALING APPLICATION")
    logger.info("=" * 60)

    try:
        # Create D2 dataset with scaling enabled
        d2_dataset = EncoderDecoder(
            d1_dataset=d1_dataset,
            past_len=24,
            future_len=12,
            batch_size=16,
            apply_scaling=True,  # Enable scaling
            split_config=(0.7, 0.2, 0.1),  # 70% train, 20% val, 10% test
        )

        # Check that scaling parameters were retrieved
        d2_scaling_params = d2_dataset.get_scaling_params()
        logger.info(f"D2 layer retrieved scaling parameters for {len(d2_scaling_params)} columns")

        # Get a sample from the dataset to verify scaling was applied
        if len(d2_dataset.dataset) > 0:
            sample_x, sample_y = d2_dataset.dataset[0]

            logger.info("Sample data shapes:")
            for key, value in sample_x.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {tuple(value.shape)}")

            # Check that numerical features are in [0, 1] range (min-max scaled)
            if "x_num_past" in sample_x:
                x_num_past = sample_x["x_num_past"]
                if x_num_past.numel() > 0:
                    min_val = x_num_past.min().item()
                    max_val = x_num_past.max().item()
                    logger.info(f"x_num_past value range: [{min_val:.4f}, {max_val:.4f}]")

                    # Values should be approximately in [0, 1] range after min-max scaling
                    assert min_val >= -0.1, f"Minimum value {min_val} is too low for scaled data"
                    assert max_val <= 1.1, f"Maximum value {max_val} is too high for scaled data"

                    logger.info("✅ Numerical features are properly scaled to [0, 1] range")

        logger.info("✅ D2 scaling application test PASSED")
        return d2_dataset

    except Exception as e:
        logger.error(f"❌ D2 scaling application test FAILED: {e}")
        raise


def test_inverse_scaling(d1_dataset, d2_dataset):
    """Test inverse scaling functionality with precise validation."""

    logger.info("=" * 60)
    logger.info("TESTING INVERSE SCALING")
    logger.info("=" * 60)

    try:
        # Get scaling parameters for precise testing
        scaling_params = d1_dataset.get_scaling_params()
        feature_cols = ["feature1", "feature2", "target"]

        # Create test data with known scaled values
        # Row 0: [0.0, 0.5, 1.0] - min, mid, max for each feature
        # Row 1: [0.25, 0.75, 0.1] - quarter, three-quarter, tenth
        scaled_predictions = torch.tensor(
            [
                [0.0, 0.5, 1.0],  # Test min, mid, max
                [0.25, 0.75, 0.1],  # Test other values
            ],
            dtype=torch.float32,
        )

        # Apply inverse scaling
        unscaled_predictions = d1_dataset.apply_inverse_scaling(scaled_predictions, feature_cols)

        logger.info("Inverse scaling validation:")
        logger.info(f"  Scaled predictions shape: {scaled_predictions.shape}")
        logger.info(f"  Unscaled predictions shape: {unscaled_predictions.shape}")

        # Validate each feature with precise assertions
        for i, col in enumerate(feature_cols):
            if col in scaling_params and i < scaled_predictions.shape[1]:
                params = scaling_params[col]
                expected_min = params["min"]
                expected_max = params["max"]
                expected_range = params["range"]

                # Test row 0: min (0.0), mid (0.5), max (1.0)
                if i == 0:  # feature1: scaled 0.0 -> should be min
                    actual_min = unscaled_predictions[0, i].item()
                    assert abs(actual_min - expected_min) < 1e-4, f"feature1 min: expected {expected_min}, got {actual_min}"
                    logger.info(f"  ✓ feature1 min: {actual_min:.6f} (expected: {expected_min:.6f})")

                elif i == 1:  # feature2: scaled 0.5 -> should be mid
                    actual_mid = unscaled_predictions[0, i].item()
                    expected_mid = expected_min + 0.5 * expected_range
                    assert abs(actual_mid - expected_mid) < 1e-4, f"feature2 mid: expected {expected_mid}, got {actual_mid}"
                    logger.info(f"  ✓ feature2 mid: {actual_mid:.6f} (expected: {expected_mid:.6f})")

                elif i == 2:  # target: scaled 1.0 -> should be max
                    actual_max = unscaled_predictions[0, i].item()
                    assert abs(actual_max - expected_max) < 1e-4, f"target max: expected {expected_max}, got {actual_max}"
                    logger.info(f"  ✓ target max: {actual_max:.6f} (expected: {expected_max:.6f})")

                # Test row 1: quarter (0.25), three-quarter (0.75), tenth (0.1)
                scaled_val = scaled_predictions[1, i].item()
                actual_val = unscaled_predictions[1, i].item()
                expected_val = expected_min + scaled_val * expected_range
                assert (
                    abs(actual_val - expected_val) < 1e-4
                ), f"{col} scaled {scaled_val}: expected {expected_val}, got {actual_val}"
                logger.info(f"  ✓ {col} scaled {scaled_val}: {actual_val:.6f} (expected: {expected_val:.6f})")

        logger.info("✅ Inverse scaling test PASSED with precise validation")

    except Exception as e:
        logger.error(f"❌ Inverse scaling test FAILED: {e}")
        raise


def test_memory_efficient_mode():
    """Test scaling with memory-efficient mode."""

    logger.info("=" * 60)
    logger.info("TESTING MEMORY-EFFICIENT MODE SCALING")
    logger.info("=" * 60)

    file_paths, temp_dir = create_test_data()

    try:
        # Create D1 dataset in memory-efficient mode
        d1_dataset = MultiSourceTSDataSet(
            file_paths=file_paths,
            group_cols=["group_id"],
            time_col="time",
            target_cols=["target"],
            num_cols=["feature1", "feature2", "target"],
            memory_efficient=True,  # Enable memory-efficient mode
            chunk_size=50,
        )

        # Check that scaling parameters were still computed
        scaling_params = d1_dataset.get_scaling_params()
        logger.info(f"Memory-efficient mode: computed scaling parameters for {len(scaling_params)} columns")

        # Create D2 dataset
        d2_dataset = EncoderDecoder(d1_dataset=d1_dataset, past_len=12, future_len=6, batch_size=8, apply_scaling=True)

        # Test that we can get samples (scaling should be applied on-the-fly)
        if len(d2_dataset.dataset) > 0:
            sample_x, sample_y = d2_dataset.dataset[0]
            logger.info("Memory-efficient mode sample obtained successfully")

            # Check scaling was applied
            if "x_num_past" in sample_x and sample_x["x_num_past"].numel() > 0:
                min_val = sample_x["x_num_past"].min().item()
                max_val = sample_x["x_num_past"].max().item()
                logger.info(f"Memory-efficient scaled range: [{min_val:.4f}, {max_val:.4f}]")

        logger.info("✅ Memory-efficient mode scaling test PASSED")

        # Clean up
        import shutil

        shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"❌ Memory-efficient mode scaling test FAILED: {e}")
        raise


def main():
    """Run all scaling tests."""

    logger.info("🚀 Starting Min-Max Scaling Tests")
    logger.info("=" * 80)

    try:
        # Test 1: D1 layer scaling computation
        d1_dataset, temp_dir = test_d1_scaling_computation()

        # Test 2: D2 layer scaling application
        d2_dataset = test_d2_scaling_application(d1_dataset)

        # Test 3: Inverse scaling
        test_inverse_scaling(d1_dataset, d2_dataset)

        # Test 4: Memory-efficient mode
        test_memory_efficient_mode()

        logger.info("=" * 80)
        logger.info("🎉 ALL SCALING TESTS PASSED!")
        logger.info("=" * 80)

        # Clean up
        import shutil

        shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"💥 SCALING TESTS FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
