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
    )

    # Split data
    train_dataset, val_dataset, test_dataset = d2_dataset.split_data(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, method="temporal"
    )

    # Check that the scaler is fitted
    assert d2_dataset.is_scaler_fitted, "Scaler should be fitted after split_data"

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
    train_sample, _ = train_dataset[0]
    val_sample, _ = val_dataset[0]
    test_sample, _ = test_dataset[0]

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
    )

    # Split data
    train_dataset, val_dataset, test_dataset = d2_dataset.split_data(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, method="temporal"
    )

    # Check that the scaler is fitted
    assert d2_dataset.is_scaler_fitted, "Scaler should be fitted after split_data"

    # Get a batch from each dataset
    train_sample, _ = train_dataset[0]
    val_sample, _ = val_dataset[0]
    test_sample, _ = test_dataset[0]

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
    )

    # Split data
    train_dataset, val_dataset, test_dataset = d2_dataset.split_data(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, method="temporal"
    )

    # Check that both scalers are fitted
    assert d2_dataset.is_scaler_fitted, "Feature scaler should be fitted after split_data"
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
    train_sample, train_target = train_dataset[0]
    val_sample, val_target = val_dataset[0]
    test_sample, test_target = test_dataset[0]

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


if __name__ == "__main__":
    test_standard_scaler()
    test_custom_scaler()
    test_target_scaling()
    logger.info("All tests passed!")
