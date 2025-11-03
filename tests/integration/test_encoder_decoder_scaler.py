"""
Tests for scikit-learn scaler integration in EncoderDecoder class.
Tests StandardScaler and MinMaxScaler with the simplified API.
"""

import logging
import tempfile

import numpy as np
import pandas as pd
from tabulate import tabulate

from dsipts.data_structure.d1_layers.multi_source_csv import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers.encoder_decoder import EncoderDecoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data():
    """Create a simple test dataset with known mean and standard deviation."""
    np.random.seed(42)

    # Create 100 rows of data with 3 numeric features and 2 targets
    n_rows = 100
    df = pd.DataFrame(
        {
            "time": pd.date_range(start="2023-01-01", periods=n_rows, freq="h"),
            "feature1": np.random.normal(0, 1, n_rows),
            "feature2": np.random.normal(10, 5, n_rows),
            "feature3": np.random.normal(-5, 2, n_rows),
            "target1": np.random.normal(100, 20, n_rows),
            "target2": np.random.normal(50, 10, n_rows),
        }
    )

    # Save to temporary CSV
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    return temp_file.name, df


def log_data_statistics(df, feature_cols, title="Data Statistics"):
    """Log statistics about the data before/after scaling."""
    stats = []
    for col in feature_cols:
        stats.append(
            {
                "Feature": col,
                "Mean": df[col].mean(),
                "Std": df[col].std(),
                "Min": df[col].min(),
                "Max": df[col].max(),
                "25%": df[col].quantile(0.25),
                "50%": df[col].median(),
                "75%": df[col].quantile(0.75),
            }
        )

    # Create a pretty table
    table = tabulate(
        [{k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in s.items()} for s in stats],
        headers="keys",
        tablefmt="grid",
    )

    logger.info(f"\n{title}:\n{table}")
    return stats


def visualize_data_transformation(original_data, transformed_data, feature_names, title="Data Transformation"):
    """Visualize the transformation of data before and after scaling."""
    # Select a subset of data points to display (first 5 rows)
    n_samples = min(5, len(original_data))

    # Create a table with original and transformed values
    table_data = []
    for i in range(n_samples):
        row = {"Sample": i}
        for j, feature in enumerate(feature_names):
            row[f"{feature} (Original)"] = original_data[i, j]
            row[f"{feature} (Transformed)"] = transformed_data[i, j]
        table_data.append(row)

    # Create a pretty table
    table = tabulate(
        [{k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in row.items()} for row in table_data],
        headers="keys",
        tablefmt="grid",
    )

    logger.info(f"\n{title} - Sample Data Points:\n{table}")


def test_standard_scaler():
    """Test that StandardScaler is applied correctly in EncoderDecoder."""
    logger.info("Testing StandardScaler functionality in EncoderDecoder")

    # Create test data
    csv_path, original_df = create_test_data()

    # Log original data statistics
    feature_cols = ["feature1", "feature2", "feature3"]
    original_stats = log_data_statistics(original_df, feature_cols, "Original Data Statistics (Before Scaling)")

    # Create D1 dataset
    d1_dataset = MultiSourceTSDataSet(
        file_paths=[csv_path],
        time_col="time",
        num_cols=feature_cols,
        target_cols=["target1", "target2"],
    )

    # Create D2 dataset with StandardScaler
    d2_dataset = EncoderDecoder(
        d1_dataset=d1_dataset,
        past_len=24,
        future_len=12,
        batch_size=16,
        scaling_method="standard",
        scale_targets=False,
        split_ratio=(0.7, 0.15, 0.15),
    )

    # Setup to fit scaler and create splits
    d2_dataset.setup(stage="fit")  # Creates train + val
    d2_dataset.setup(stage="test")  # Creates test

    # Check that the scaler is fitted
    assert d2_dataset.is_scaler_fitted, "Scaler should be fitted after setup"

    # Verify scaler parameters match the original data statistics
    # The scaler should have been fitted on the training data only
    feature_means = d2_dataset.feature_scaler.mean_
    feature_stds = np.sqrt(d2_dataset.feature_scaler.var_)

    # Log scaler metadata
    logger.info("\nStandardScaler Metadata:")
    logger.info(f"Feature names: {feature_cols}")
    logger.info(f"Scaler means: {feature_means}")
    logger.info(f"Scaler stds: {feature_stds}")

    # Create a table with scaler parameters
    scaler_params = []
    for i, col in enumerate(feature_cols):
        scaler_params.append(
            {
                "Feature": col,
                "Mean": feature_means[i],
                "Std": feature_stds[i],
                "Scale Factor": 1 / feature_stds[i] if feature_stds[i] > 0 else 0,
            }
        )

    table = tabulate(
        [{k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in p.items()} for p in scaler_params],
        headers="keys",
        tablefmt="grid",
    )
    logger.info(f"\nScaler Parameters:\n{table}")

    # Get a batch from each dataset
    train_sample, _ = d2_dataset.train_dataset[0]
    val_sample, _ = d2_dataset.val_dataset[0]
    test_sample, _ = d2_dataset.test_dataset[0]

    # Check that numeric features are scaled
    train_features = train_sample["x_num_past"].numpy()
    # val_features = val_sample["x_num_past"].numpy()
    # test_features = test_sample["x_num_past"].numpy()

    # Calculate mean and std of the scaled features
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    # Log scaled data statistics
    logger.info("\nScaled Data Statistics:")
    logger.info(f"Scaled train features mean: {train_mean}")
    logger.info(f"Scaled train features std: {train_std}")

    # Create a table comparing original vs scaled statistics
    comparison_table = []
    for i, col in enumerate(feature_cols):
        comparison_table.append(
            {
                "Feature": col,
                "Original Mean": original_stats[i]["Mean"],
                "Scaled Mean": train_mean[i],
                "Original Std": original_stats[i]["Std"],
                "Scaled Std": train_std[i],
                "Original Range": f"{original_stats[i]['Min']:.4f}" f"              to {original_stats[i]['Max']:.4f}",
                "Scaled Range": f"{np.min(train_features[:, i]):.4f}" f"              to {np.max(train_features[:, i]):.4f}",
            }
        )

    table = tabulate(
        [{k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in row.items()} for row in comparison_table],
        headers="keys",
        tablefmt="grid",
    )
    logger.info(f"\nOriginal vs Scaled Statistics:\n{table}")

    # Visualize sample data points before and after transformation
    # Get the first few rows of original data for the features
    original_samples = original_df[feature_cols].values[:5]
    # Get the corresponding scaled values (first 5 time steps of the first sample)
    scaled_samples = train_features[:5, :]

    # Visualize the transformation
    visualize_data_transformation(original_samples, scaled_samples, feature_cols, "StandardScaler Transformation")

    # After StandardScaler, mean should be close to 0 and std close to 1 for non-zero std columns
    # Get indices of columns with non-zero standard deviation in the original data
    non_zero_std_cols = np.where(feature_stds > 0)[0]

    assert np.allclose(
        train_mean[non_zero_std_cols], np.zeros_like(train_mean[non_zero_std_cols]), atol=0.5
    ), "Scaled train features should have mean close to 0"
    assert np.allclose(
        train_std[non_zero_std_cols], np.ones_like(train_std[non_zero_std_cols]), atol=0.5
    ), "Scaled train features should have std close to 1"

    logger.info("StandardScaler test passed!")


def test_custom_scaler():
    """Test that a custom scaler (MinMaxScaler) is applied correctly."""
    logger.info("\n\nTesting custom scaler (MinMaxScaler) functionality in EncoderDecoder")

    # Create test data
    csv_path, original_df = create_test_data()

    # Log original data statistics
    feature_cols = ["feature1", "feature2", "feature3"]
    original_stats = log_data_statistics(original_df, feature_cols, "Original Data Statistics (Before MinMax Scaling)")

    # Create D1 dataset
    d1_dataset = MultiSourceTSDataSet(
        file_paths=[csv_path],
        time_col="time",
        num_cols=feature_cols,
        target_cols=["target1", "target2"],
        enrich_cat=["minute", "hour"],
    )

    # Create D2 dataset with MinMaxScaler
    d2_dataset = EncoderDecoder(
        d1_dataset=d1_dataset,
        past_len=24,
        future_len=12,
        batch_size=16,
        scaling_method="minmax",
        scale_targets=False,
        split_ratio=(0.7, 0.15, 0.15),
    )

    # Setup to fit scaler and create splits
    d2_dataset.setup(stage="fit")  # Creates train + val
    d2_dataset.setup(stage="test")  # Creates test

    # Check that the scaler is fitted
    assert d2_dataset.is_scaler_fitted, "Scaler should be fitted after setup"

    # Get a batch from each dataset
    train_sample, _ = d2_dataset.train_dataset[0]
    val_sample, _ = d2_dataset.val_dataset[0]
    test_sample, _ = d2_dataset.test_dataset[0]

    # Check that numeric features are scaled
    train_features = train_sample["x_num_past"].numpy()
    # val_features = val_sample["x_num_past"].numpy()
    # test_features = test_sample["x_num_past"].numpy()

    # After MinMaxScaler, all values should be between 0 and 1
    assert np.all(train_features >= 0) and np.all(train_features <= 1), "MinMaxScaler should scale features between 0 and 1"

    # Log scaled data statistics
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)
    train_min = np.min(train_features, axis=0)
    train_max = np.max(train_features, axis=0)

    logger.info("\nMinMaxScaled Data Statistics:")
    logger.info(f"Scaled train features mean: {train_mean}")
    logger.info(f"Scaled train features std: {train_std}")
    logger.info(f"Scaled train features min: {train_min}")
    logger.info(f"Scaled train features max: {train_max}")

    # Create a table comparing original vs scaled statistics
    comparison_table = []
    for i, col in enumerate(feature_cols):
        comparison_table.append(
            {
                "Feature": col,
                "Original Min": original_stats[i]["Min"],
                "Scaled Min": train_min[i],
                "Original Max": original_stats[i]["Max"],
                "Scaled Max": train_max[i],
                "Original Range": original_stats[i]["Max"] - original_stats[i]["Min"],
                "Scaled Range": train_max[i] - train_min[i],
            }
        )

    table = tabulate(
        [{k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in row.items()} for row in comparison_table],
        headers="keys",
        tablefmt="grid",
    )
    logger.info(f"\nOriginal vs MinMaxScaled Statistics:\n{table}")

    # Visualize sample data points before and after transformation
    # Get the first few rows of original data for the features
    original_samples = original_df[feature_cols].values[:5]
    # Get the corresponding scaled values (first 5 time steps of the first sample)
    scaled_samples = train_features[:5, :]

    # Visualize the transformation
    visualize_data_transformation(original_samples, scaled_samples, feature_cols, "MinMaxScaler Transformation")

    logger.info("Custom scaler (MinMaxScaler) test passed!")


def test_target_scaling():
    """Test that target scaling works correctly."""
    logger.info("\n\nTesting target scaling functionality in EncoderDecoder")

    # Create test data
    csv_path, original_df = create_test_data()

    # Log original data statistics for targets
    feature_cols = ["feature1", "feature2", "feature3"]
    target_cols = ["target1", "target2"]

    # Log original feature statistics
    # feature_stats = log_data_statistics(
    #     original_df, feature_cols, "Original Feature Statistics (Before Scaling)"
    # )

    # Log original target statistics
    target_stats = log_data_statistics(original_df, target_cols, "Original Target Statistics (Before Scaling)")

    # Create D1 dataset
    d1_dataset = MultiSourceTSDataSet(file_paths=[csv_path], time_col="time", num_cols=feature_cols, target_cols=target_cols)

    # Create D2 dataset with StandardScaler and target scaling
    d2_dataset = EncoderDecoder(
        d1_dataset=d1_dataset,
        past_len=24,
        future_len=12,
        batch_size=16,
        scaling_method="standard",
        scale_targets=True,
        split_ratio=(0.7, 0.15, 0.15),
    )

    # Setup to fit scaler and create splits
    d2_dataset.setup(stage="fit")  # Creates train + val
    d2_dataset.setup(stage="test")  # Creates test

    # Check that both scalers are fitted
    assert d2_dataset.is_scaler_fitted, "Feature scaler should be fitted after setup"
    assert hasattr(d2_dataset, "target_scaler"), "Target scaler should be created"

    # Log feature scaler metadata
    logger.info("\nFeature StandardScaler Metadata:")
    logger.info(f"Feature names: {feature_cols}")
    logger.info(f"Feature scaler means: {d2_dataset.feature_scaler.mean_}")
    logger.info(f"Feature scaler stds: {np.sqrt(d2_dataset.feature_scaler.var_)}")

    # Log target scaler metadata
    logger.info("\nTarget StandardScaler Metadata:")
    logger.info(f"Target names: {target_cols}")
    logger.info(f"Target scaler means: {d2_dataset.target_scaler.mean_}")
    logger.info(f"Target scaler stds: {np.sqrt(d2_dataset.target_scaler.var_)}")

    # Create tables with scaler parameters
    feature_scaler_params = []
    for i, col in enumerate(feature_cols):
        feature_scaler_params.append(
            {
                "Feature": col,
                "Mean": d2_dataset.feature_scaler.mean_[i],
                "Std": np.sqrt(d2_dataset.feature_scaler.var_)[i],
                "Scale Factor": 1 / np.sqrt(d2_dataset.feature_scaler.var_)[i] if d2_dataset.feature_scaler.var_[i] > 0 else 0,
            }
        )

    target_scaler_params = []
    for i, col in enumerate(target_cols):
        target_scaler_params.append(
            {
                "Target": col,
                "Mean": d2_dataset.target_scaler.mean_[i],
                "Std": np.sqrt(d2_dataset.target_scaler.var_)[i],
                "Scale Factor": 1 / np.sqrt(d2_dataset.target_scaler.var_)[i] if d2_dataset.target_scaler.var_[i] > 0 else 0,
            }
        )

    feature_table = tabulate(
        [{k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in p.items()} for p in feature_scaler_params],
        headers="keys",
        tablefmt="grid",
    )
    logger.info(f"\nFeature Scaler Parameters:\n{feature_table}")

    target_table = tabulate(
        [{k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in p.items()} for p in target_scaler_params],
        headers="keys",
        tablefmt="grid",
    )
    logger.info(f"\nTarget Scaler Parameters:\n{target_table}")

    # Get a batch from each dataset
    train_sample, train_target = d2_dataset.train_dataset[0]
    val_sample, val_target = d2_dataset.val_dataset[0]
    test_sample, test_target = d2_dataset.test_dataset[0]

    # Check that targets are scaled
    train_targets = train_target.numpy()
    # val_targets = val_target.numpy()
    # test_targets = test_target.numpy()

    # Calculate mean and std of the scaled targets
    train_target_mean = np.mean(train_targets, axis=0)
    train_target_std = np.std(train_targets, axis=0)

    logger.info(f"\nScaled train targets mean: {train_target_mean}")
    logger.info(f"Scaled train targets std: {train_target_std}")

    # Create a table comparing original vs scaled target statistics
    comparison_table = []
    for i, col in enumerate(target_cols):
        comparison_table.append(
            {
                "Target": col,
                "Original Mean": target_stats[i]["Mean"],
                "Scaled Mean": train_target_mean[i],
                "Original Std": target_stats[i]["Std"],
                "Scaled Std": train_target_std[i],
                "Original Range": f"{target_stats[i]['Min']:.4f} to {target_stats[i]['Max']:.4f}",
                "Scaled Range": f"{np.min(train_targets[:, i]):.4f} to" f"                {np.max(train_targets[:, i]):.4f}",
            }
        )

    table = tabulate(
        [{k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in row.items()} for row in comparison_table],
        headers="keys",
        tablefmt="grid",
    )
    logger.info(f"\nOriginal vs Scaled Target Statistics:\n{table}")

    # Visualize sample data points before and after transformation for targets
    # Get the first few rows of original data for the targets
    original_target_samples = original_df[target_cols].values[:5]
    # Get the corresponding scaled values (first 5 time steps of the first sample)
    scaled_target_samples = train_targets[:5, :]

    # Visualize the transformation
    visualize_data_transformation(
        original_target_samples,
        scaled_target_samples,
        target_cols,
        "Target StandardScaler Transformation",
    )

    # After StandardScaler, mean should be close to 0 and std close to 1 for non-zero std columns
    # We know our test data has non-zero std for all target columns, so we can check all

    assert np.allclose(
        train_target_mean, np.zeros_like(train_target_mean), atol=0.5
    ), "Scaled train targets should have mean close to 0"
    assert np.allclose(
        train_target_std, np.ones_like(train_target_std), atol=0.5
    ), "Scaled train targets should have std close to 1"

    # The targets are returned separately as y, not in the x dictionary
    # Just verify they are scaled (already checked above with train_target_mean and std)

    logger.info("Target scaling test passed!")


def test_no_scaling():
    """Test that data remains unscaled when scaling_method=None."""
    logger.info("\n\nTesting no scaling functionality in EncoderDecoder")

    # Create test data
    csv_path, original_df = create_test_data()

    # Create D1 dataset
    d1_dataset = MultiSourceTSDataSet(
        file_paths=[csv_path],
        time_col="time",
        num_cols=["feature1", "feature2", "feature3"],
        target_cols=["target1", "target2"],
    )

    # Create D2 dataset WITHOUT scaling
    d2_dataset = EncoderDecoder(
        d1_dataset=d1_dataset,
        past_len=24,
        future_len=12,
        batch_size=16,
        scaling_method=None,
        scale_targets=False,
        split_ratio=(0.7, 0.15, 0.15),
    )

    # Setup
    d2_dataset.setup(stage="fit")

    # Check that no scaler is fitted
    assert d2_dataset.feature_scaler is None, "Feature scaler should be None when scaling_method=None"
    assert d2_dataset.target_scaler is None, "Target scaler should be None when scale_targets=False"

    # Get a sample
    train_sample, train_target = d2_dataset.train_dataset[0]
    train_features = train_sample["x_num_past"].numpy()

    # Check that values are NOT scaled (should be in original range)
    # Original data has feature1 ~ N(0, 1), feature2 ~ N(10, 5), feature3 ~ N(-5, 2)
    mean_feature2 = np.mean(train_features[:, 1])
    logger.info(f"\nUnscaled feature2 mean: {mean_feature2:.2f} (should be ~10)")

    # Feature2 should be around 10, not around 0
    assert abs(mean_feature2 - 10) < 5, f"Feature2 should be unscaled (~10), got {mean_feature2}"

    logger.info("No scaling test passed!")


def test_scaled_vs_unscaled_comparison():
    """Compare scaled vs unscaled values side-by-side."""
    logger.info("\n\nTesting scaled vs unscaled comparison")

    # Create test data
    csv_path, original_df = create_test_data()

    # Create D1 dataset
    d1_dataset = MultiSourceTSDataSet(
        file_paths=[csv_path],
        time_col="time",
        num_cols=["feature1", "feature2", "feature3"],
        target_cols=["target1", "target2"],
    )

    # Create D2 WITHOUT scaling
    d2_unscaled = EncoderDecoder(
        d1_dataset=d1_dataset,
        past_len=24,
        future_len=12,
        batch_size=16,
        scaling_method=None,
        scale_targets=False,
        split_ratio=(0.7, 0.15, 0.15),
    )
    d2_unscaled.setup(stage="fit")

    # Create D2 WITH StandardScaler
    d2_scaled = EncoderDecoder(
        d1_dataset=d1_dataset,
        past_len=24,
        future_len=12,
        batch_size=16,
        scaling_method="standard",
        scale_targets=True,
        split_ratio=(0.7, 0.15, 0.15),
    )
    d2_scaled.setup(stage="fit")

    # Get samples from training data
    x_unscaled, y_unscaled = d2_unscaled.train_dataset[0]
    x_scaled, y_scaled = d2_scaled.train_dataset[0]

    # Compare statistics
    unscaled_mean = x_unscaled["x_num_past"].mean().item()
    scaled_mean = x_scaled["x_num_past"].mean().item()

    logger.info(f"\nUnscaled x_num_past mean: {unscaled_mean:.2f}")
    logger.info(f"Scaled x_num_past mean: {scaled_mean:.2f}")

    # Unscaled should NOT be close to 0, scaled should be close to 0
    assert abs(unscaled_mean) > 1.0, f"Unscaled mean should not be ~0, got {unscaled_mean}"
    assert abs(scaled_mean) < 1.0, f"Scaled mean should be ~0, got {scaled_mean}"

    logger.info("Scaled vs unscaled comparison test passed!")


def test_memory_efficient_modes_consistency():
    """Test that scaling is consistent across memory_efficient modes."""
    logger.info("\n\nTesting scaling consistency across memory_efficient modes")

    # Create test data
    csv_path, original_df = create_test_data()

    # Mode 1: memory_efficient=False (pre-transform)
    d1_precompute = MultiSourceTSDataSet(
        file_paths=[csv_path],
        time_col="time",
        num_cols=["feature1", "feature2", "feature3"],
        target_cols=["target1", "target2"],
        memory_efficient=False,
    )

    d2_precompute = EncoderDecoder(
        d1_dataset=d1_precompute,
        past_len=24,
        future_len=12,
        batch_size=16,
        scaling_method="standard",
        scale_targets=True,
        split_ratio=(0.7, 0.15, 0.15),
    )
    d2_precompute.setup(stage="fit")

    # Mode 2: memory_efficient=True (on-the-fly)
    d1_onthefly = MultiSourceTSDataSet(
        file_paths=[csv_path],
        time_col="time",
        num_cols=["feature1", "feature2", "feature3"],
        target_cols=["target1", "target2"],
        memory_efficient=True,
    )

    d2_onthefly = EncoderDecoder(
        d1_dataset=d1_onthefly,
        past_len=24,
        future_len=12,
        batch_size=16,
        scaling_method="standard",
        scale_targets=True,
        split_ratio=(0.7, 0.15, 0.15),
    )
    d2_onthefly.setup(stage="fit")

    # Compare scaler parameters
    scaler_mean_1 = d2_precompute.feature_scaler.mean_
    scaler_mean_2 = d2_onthefly.feature_scaler.mean_

    logger.info(f"\nMode 1 (pre-transform) scaler mean: {scaler_mean_1}")
    logger.info(f"Mode 2 (on-the-fly) scaler mean: {scaler_mean_2}")

    # Scaler parameters should match
    assert np.allclose(scaler_mean_1, scaler_mean_2, rtol=1e-5), "Scaler parameters should match across modes"

    # Get same window from both
    x1, y1 = d2_precompute.train_dataset[0]
    x2, y2 = d2_onthefly.train_dataset[0]

    # Scaled values should match
    assert np.allclose(
        x1["x_num_past"].numpy(), x2["x_num_past"].numpy(), rtol=1e-5, atol=1e-7
    ), "Scaled values should match across memory_efficient modes"

    logger.info("Memory efficient modes consistency test passed!")


if __name__ == "__main__":
    test_standard_scaler()
    test_custom_scaler()
    test_target_scaling()
    test_no_scaling()
    test_scaled_vs_unscaled_comparison()
    test_memory_efficient_modes_consistency()
    logger.info("\n\n" + "=" * 80)
    logger.info("ALL SCALING TESTS PASSED!")
    logger.info("=" * 80)
