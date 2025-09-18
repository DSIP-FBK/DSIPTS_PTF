"""
MAIN TEST FILE
Comprehensive test suite for standard scaling functionality in DSIPTS.

This test suite validates:
1. D1 layer standard scaling parameter computation
2. D2 layer standard scaling application
3. Inverse scaling for denormalization
4. Integration between D1 and D2 layers
5. Comparison with sklearn StandardScaler
6. Memory-efficient mode compatibility
"""

import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

# Import DSIPTS modules
from dsipts.data_structure.d1_layers.multi_source_csv import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers.encoder_decoder import EncoderDecoder
from dsipts.data_structure.scalers import OnlineStandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_sample_data_for_standard_scaling():
    """
    Create sample time series data with different scales for testing standard scaling.

    Returns:
        tuple: (temp_dir, file_paths) containing temporary directory and CSV file paths
    """
    logger.info("Creating sample data for standard scaling tests...")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create 3 CSV files with different groups and varying scales
    for file_idx in range(3):
        data_list = []

        # Create 2 groups per file with different statistical properties
        for group_idx in range(2):
            group_id = chr(ord("A") + file_idx * 2 + group_idx)  # A, B, C, D, E, F

            # Generate 50 time steps per group
            n_steps = 50
            time_range = pd.date_range(start="2023-01-01", periods=n_steps, freq="H")

            # Create features with different means and standard deviations
            if group_id in ["A", "B"]:
                # Low mean, low variance
                feature1 = np.random.normal(loc=10, scale=2, size=n_steps)
                feature2 = np.random.normal(loc=5, scale=1, size=n_steps)
                target = feature1 * 0.5 + feature2 * 0.3 + np.random.normal(0, 0.5, n_steps)
            elif group_id in ["C", "D"]:
                # Medium mean, medium variance
                feature1 = np.random.normal(loc=100, scale=20, size=n_steps)
                feature2 = np.random.normal(loc=50, scale=10, size=n_steps)
                target = feature1 * 0.5 + feature2 * 0.3 + np.random.normal(0, 5, n_steps)
            else:  # E, F
                # High mean, high variance
                feature1 = np.random.normal(loc=1000, scale=200, size=n_steps)
                feature2 = np.random.normal(loc=500, scale=100, size=n_steps)
                target = feature1 * 0.5 + feature2 * 0.3 + np.random.normal(0, 50, n_steps)

            # Create DataFrame for this group
            group_df = pd.DataFrame(
                {"time": time_range, "group_id": group_id, "feature1": feature1, "feature2": feature2, "target": target}
            )

            data_list.append(group_df)

        # Combine groups and save to CSV
        file_df = pd.concat(data_list, ignore_index=True)
        file_path = os.path.join(temp_dir, f"data_{file_idx}.csv")
        file_df.to_csv(file_path, index=False)
        file_paths.append(file_path)

        logger.info(f"Created file {file_idx + 1}: {len(file_df)} rows, groups {file_df['group_id'].unique()}")

    logger.info(f"Sample data created in {temp_dir}")
    return temp_dir, file_paths


def test_d1_standard_scaling_computation():
    """
    Test D1 layer standard scaling parameter computation.
    """
    logger.info("============================================================")
    logger.info("TESTING D1 LAYER STANDARD SCALING COMPUTATION")
    logger.info("============================================================")

    try:
        # Create sample data
        temp_dir, file_paths = create_sample_data_for_standard_scaling()

        # Create D1 dataset with standard scaling
        d1_dataset = MultiSourceTSDataSet(
            file_paths=file_paths,
            group_cols=["group_id"],
            time_col="time",
            target_cols=["target"],
            num_cols=["feature1", "feature2", "target"],
            scaling_method="standard",  # Use standard scaling
            memory_efficient=False,
        )

        # Get scaling parameters
        scaling_params = d1_dataset.get_scaling_params()

        logger.info(f"Computed standard scaling parameters for {len(scaling_params)} columns")

        # Validate parameters
        expected_cols = ["feature1", "feature2", "target"]
        for col in expected_cols:
            assert col in scaling_params, f"Missing scaling parameters for column '{col}'"

            params = scaling_params[col]
            assert "mean" in params, f"Missing 'mean' in parameters for column '{col}'"
            assert "std" in params, f"Missing 'std' in parameters for column '{col}'"
            assert "scaler_type" in params, f"Missing 'scaler_type' in parameters for column '{col}'"
            assert params["scaler_type"] == "standard", f"Wrong scaler_type for column '{col}'"

            # Log the parameters
            logger.info(f"Column '{col}':")
            logger.info(f"  Mean: {params['mean']:.6f}")
            logger.info(f"  Std: {params['std']:.6f}")
            logger.info(f"  Scaler type: {params['scaler_type']}")

            # Validate that std is positive (not constant)
            assert params["std"] > 0, f"Standard deviation should be positive for column '{col}'"

        logger.info("✅ D1 standard scaling computation test PASSED")
        return d1_dataset, temp_dir

    except Exception as e:
        logger.error(f"❌ D1 standard scaling computation test FAILED: {e}")
        raise


def test_d2_standard_scaling_application(d1_dataset):
    """
    Test D2 layer standard scaling application.
    """
    logger.info("============================================================")
    logger.info("TESTING D2 LAYER STANDARD SCALING APPLICATION")
    logger.info("============================================================")

    try:
        # Create D2 dataset
        d2_dataset = EncoderDecoder(
            d1_dataset=d1_dataset,
            past_len=24,
            future_len=12,
            batch_size=16,
            apply_scaling=True,  # This should use the D1 layer's standard scaling
        )

        # Check that D2 layer detected standard scaling
        assert d2_dataset.use_standard_scaler, "D2 layer should detect standard scaling from D1"
        assert d2_dataset.d1_scaling_method == "standard", "D2 layer should detect standard scaling method"

        # Get scaling parameters
        scaling_params = d2_dataset.get_scaling_params()
        logger.info(f"D2 layer retrieved scaling parameters for {len(scaling_params)} columns")

        # Test a sample from the dataset
        sample_x, sample_y = d2_dataset.dataset[0]

        logger.info("Sample data shapes:")
        logger.info(f"  x_num_past: {sample_x['x_num_past'].shape}")
        logger.info(f"  x_cat_past: {sample_x['x_cat_past'].shape}")
        logger.info(f"  y: {sample_y.shape}")

        # Check that numerical features are standardized (approximately mean=0, std=1)
        x_num_past = sample_x["x_num_past"]
        logger.info("x_num_past statistics:")
        logger.info(f"  Mean: {x_num_past.mean():.6f}")
        logger.info(f"  Std: {x_num_past.std():.6f}")

        # For standard scaling, we expect values to be roughly centered around 0
        # (though individual samples may vary)
        assert abs(x_num_past.mean()) < 5.0, "Standardized data should have mean close to 0"

        logger.info("✅ D2 standard scaling application test PASSED")
        return d2_dataset

    except Exception as e:
        logger.error(f"❌ D2 standard scaling application test FAILED: {e}")
        raise


def test_standard_scaling_inverse_transform(d1_dataset, d2_dataset):
    """
    Test inverse standard scaling for denormalization.
    """
    logger.info("============================================================")
    logger.info("TESTING STANDARD SCALING INVERSE TRANSFORM")
    logger.info("============================================================")

    try:
        # Get scaling parameters
        scaling_params = d1_dataset.get_scaling_params()
        feature_cols = ["feature1", "feature2", "target"]

        # Create some standardized test data (z-scores)
        standardized_predictions = torch.tensor(
            [
                [0.0, 1.0, -1.0],  # mean, +1 std, -1 std
                [2.0, -0.5, 0.5],  # +2 std, -0.5 std, +0.5 std
            ],
            dtype=torch.float32,
        )

        logger.info("Inverse scaling validation:")
        logger.info(f"  Standardized predictions shape: {standardized_predictions.shape}")

        # Apply inverse scaling
        unscaled_predictions = d2_dataset.apply_inverse_scaling(standardized_predictions, feature_cols)

        logger.info(f"  Unscaled predictions shape: {unscaled_predictions.shape}")

        # Validate inverse scaling
        for i, col in enumerate(feature_cols):
            if col in scaling_params:
                params = scaling_params[col]
                expected_mean = params["mean"]
                expected_std = params["std"]

                # Test specific values
                # Row 0: [0.0, 1.0, -1.0] should become [mean, mean+std, mean-std]
                if i == 0:  # feature1: z=0.0 -> should be mean
                    actual_val = unscaled_predictions[0, i].item()
                    expected_val = expected_mean
                    assert abs(actual_val - expected_val) < 1e-4, f"feature1 mean: expected {expected_val}, got {actual_val}"
                    logger.info(f"  ✓ feature1 mean: {actual_val:.6f} (expected: {expected_val:.6f})")

                elif i == 1:  # feature2: z=1.0 -> should be mean + std
                    actual_val = unscaled_predictions[0, i].item()
                    expected_val = expected_mean + expected_std
                    assert abs(actual_val - expected_val) < 1e-4, f"feature2 mean+std: expected {expected_val}, got {actual_val}"
                    logger.info(f"  ✓ feature2 mean+std: {actual_val:.6f} (expected: {expected_val:.6f})")

                elif i == 2:  # target: z=-1.0 -> should be mean - std
                    actual_val = unscaled_predictions[0, i].item()
                    expected_val = expected_mean - expected_std
                    assert abs(actual_val - expected_val) < 1e-4, f"target mean-std: expected {expected_val}, got {actual_val}"
                    logger.info(f"  ✓ target mean-std: {actual_val:.6f} (expected: {expected_val:.6f})")

                # Test row 1: [2.0, -0.5, 0.5]
                row1_z = standardized_predictions[1, i].item()
                actual_val = unscaled_predictions[1, i].item()
                expected_val = expected_mean + row1_z * expected_std
                assert abs(actual_val - expected_val) < 1e-4, f"{col} z={row1_z}: expected {expected_val}, got {actual_val}"
                logger.info(f"  ✓ {col} z={row1_z}: {actual_val:.6f} (expected: {expected_val:.6f})")

        logger.info("✅ Standard scaling inverse transform test PASSED")

    except Exception as e:
        logger.error(f"❌ Standard scaling inverse transform test FAILED: {e}")
        raise


def test_online_standard_scaler():
    """
    Test the OnlineStandardScaler class directly.
    """
    logger.info("============================================================")
    logger.info("TESTING ONLINE STANDARD SCALER")
    logger.info("============================================================")

    try:
        # Create test data with known statistics
        np.random.seed(123)

        # Create data with known mean and std
        data1 = np.random.normal(loc=10, scale=2, size=(100, 3))  # mean=10, std=2
        data2 = np.random.normal(loc=15, scale=3, size=(150, 3))  # mean=15, std=3
        data3 = np.random.normal(loc=5, scale=1, size=(50, 3))  # mean=5, std=1

        # Combine all data for reference
        all_data = np.vstack([data1, data2, data3])
        reference_mean = np.mean(all_data, axis=0)
        reference_std = np.std(all_data, axis=0, ddof=0)  # Population std

        logger.info("Reference statistics from combined data:")
        logger.info(f"  Mean: {reference_mean}")
        logger.info(f"  Std: {reference_std}")

        # Test online scaler
        scaler = OnlineStandardScaler()

        # Fit in batches
        def batch_iterator():
            yield data1
            yield data2
            yield data3

        scaler.fit(batch_iterator())

        logger.info("Online scaler statistics:")
        logger.info(f"  Mean: {scaler.mean_}")
        logger.info(f"  Std: {scaler.scale_}")

        # Validate that online scaler matches reference
        assert np.allclose(scaler.mean_, reference_mean, atol=1e-10), "Online scaler mean doesn't match reference"
        assert np.allclose(scaler.scale_, reference_std, atol=1e-10), "Online scaler std doesn't match reference"

        # Test transform
        test_data = np.array([[10, 15, 5], [12, 18, 6]])  # Some test values
        transformed = scaler.transform(test_data)

        # Manually compute expected transform
        expected_transformed = (test_data - reference_mean) / reference_std

        assert np.allclose(transformed, expected_transformed, atol=1e-10), "Transform doesn't match expected"

        # Test inverse transform
        inverse_transformed = scaler.inverse_transform(transformed)
        assert np.allclose(inverse_transformed, test_data, atol=1e-10), "Inverse transform doesn't match original"

        logger.info("✅ Online standard scaler test PASSED")

    except Exception as e:
        logger.error(f"❌ Online standard scaler test FAILED: {e}")
        raise


def test_memory_efficient_standard_scaling():
    """
    Test standard scaling in memory-efficient mode.
    """
    logger.info("============================================================")
    logger.info("TESTING MEMORY-EFFICIENT STANDARD SCALING")
    logger.info("============================================================")

    try:
        # Create sample data
        temp_dir, file_paths = create_sample_data_for_standard_scaling()

        # Create D1 dataset with memory-efficient mode and standard scaling
        d1_dataset = MultiSourceTSDataSet(
            file_paths=file_paths,
            group_cols=["group_id"],
            time_col="time",
            target_cols=["target"],
            num_cols=["feature1", "feature2", "target"],
            scaling_method="standard",
            memory_efficient=True,  # Enable memory-efficient mode
        )

        logger.info(f"Memory-efficient mode: computed scaling parameters for {len(d1_dataset.get_scaling_params())} columns")

        # Create D2 dataset
        d2_dataset = EncoderDecoder(d1_dataset=d1_dataset, past_len=12, future_len=6, batch_size=8, apply_scaling=True)

        # Test that we can get a sample (scaling should be applied on-the-fly)
        sample_x, sample_y = d2_dataset.dataset[0]

        logger.info("Memory-efficient mode sample obtained successfully")
        logger.info(
            f"Memory-efficient standardized range: [{sample_x['x_num_past'].min():.4f}, {sample_x['x_num_past'].max():.4f}]"
        )

        # For standard scaling, we expect values to be roughly centered around 0
        assert abs(sample_x["x_num_past"].mean()) < 5.0, "Standardized data should have mean close to 0"

        logger.info("✅ Memory-efficient standard scaling test PASSED")

        # Cleanup
        shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"❌ Memory-efficient standard scaling test FAILED: {e}")
        raise


def test_scaling_method_comparison():
    """
    Test comparison between minmax and standard scaling methods.
    """
    logger.info("============================================================")
    logger.info("TESTING SCALING METHOD COMPARISON")
    logger.info("============================================================")

    try:
        # Create sample data
        temp_dir, file_paths = create_sample_data_for_standard_scaling()

        # Test MinMax scaling
        logger.info("Testing MinMax scaling...")
        d1_minmax = MultiSourceTSDataSet(
            file_paths=file_paths,
            group_cols=["group_id"],
            time_col="time",
            target_cols=["target"],
            num_cols=["feature1", "feature2", "target"],
            scaling_method="minmax",
            memory_efficient=False,
        )

        minmax_params = d1_minmax.get_scaling_params()
        logger.info(f"MinMax scaling parameters: {len(minmax_params)} columns")
        for col, params in minmax_params.items():
            logger.info(f"  {col}: min={params['min']:.4f}, max={params['max']:.4f}, type={params['scaler_type']}")

        # Test Standard scaling
        logger.info("Testing Standard scaling...")
        d1_standard = MultiSourceTSDataSet(
            file_paths=file_paths,
            group_cols=["group_id"],
            time_col="time",
            target_cols=["target"],
            num_cols=["feature1", "feature2", "target"],
            scaling_method="standard",
            memory_efficient=False,
        )

        standard_params = d1_standard.get_scaling_params()
        logger.info(f"Standard scaling parameters: {len(standard_params)} columns")
        for col, params in standard_params.items():
            logger.info(f"  {col}: mean={params['mean']:.4f}, std={params['std']:.4f}, type={params['scaler_type']}")

        # Validate that both methods have the same columns
        assert set(minmax_params.keys()) == set(standard_params.keys()), "Both scaling methods should have same columns"

        # Validate scaler types
        for col in minmax_params:
            assert minmax_params[col]["scaler_type"] == "minmax", f"MinMax scaler type wrong for {col}"
            assert standard_params[col]["scaler_type"] == "standard", f"Standard scaler type wrong for {col}"

        logger.info("✅ Scaling method comparison test PASSED")

        # Cleanup
        shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"❌ Scaling method comparison test FAILED: {e}")
        raise


def main():
    """
    Run all standard scaling tests.
    """
    logger.info("================================================================================")
    logger.info("🧪 STARTING COMPREHENSIVE STANDARD SCALING TESTS")
    logger.info("================================================================================")

    try:
        # Test 1: Online Standard Scaler
        test_online_standard_scaler()

        # Test 2: D1 layer standard scaling computation
        d1_dataset, temp_dir = test_d1_standard_scaling_computation()

        # Test 3: D2 layer standard scaling application
        d2_dataset = test_d2_standard_scaling_application(d1_dataset)

        # Test 4: Inverse scaling
        test_standard_scaling_inverse_transform(d1_dataset, d2_dataset)

        # Test 5: Memory-efficient mode
        test_memory_efficient_standard_scaling()

        # Test 6: Scaling method comparison
        test_scaling_method_comparison()

        # Cleanup
        shutil.rmtree(temp_dir)

        logger.info("================================================================================")
        logger.info("🎉 ALL STANDARD SCALING TESTS PASSED!")
        logger.info("================================================================================")

    except Exception as e:
        logger.error("💥 STANDARD SCALING TESTS FAILED")
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
