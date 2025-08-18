"""Integration test for D1 and D2 layers.

This module tests the integration between D1 (MultiSourceTSDataSet) and D2 (EncoderDecoder)
layers without involving models. It verifies that the data pipeline works correctly
with various configurations and edge cases.

This test focuses on the modern interface without backward compatibility,
ensuring that the D1 and D2 layers work together correctly with the latest features.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import our library components
from dsipts.data_structure.d1_layers import MultiSourceTSDataSet


def temp_data_dir():
    """Create a temporary directory for test data."""
    # temp_dir = tempfile.mkdtemp()
    # yield temp_dir
    yield "/home/sandeep/DSIPTS_PTF/tests/integration"
    # shutil.rmtree(temp_dir)


def create_sample_data(
    temp_dir: str,
    n_groups: int = 3,
    n_timesteps: int = 100,
    n_num_features: int = 2,
    n_cat_features: int = 3,
    n_targets: int = 1,
    include_unknown: bool = True,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    test_name_csv: str = "debug_data.csv",
) -> Tuple[str, Dict[str, Any]]:
    """Create sample time series data with various feature types.

    Args:
        temp_dir: Directory to save the data
        n_groups: Number of groups to create
        n_timesteps: Number of time steps per group
        n_num_features: Number of numerical features
        n_cat_features: Number of categorical features
        n_targets: Number of target variables
        include_unknown: Whether to include unknown features
        test_name_csv: Name of the csv files being created

    Returns:
        Tuple of (data_path, metadata)
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Use provided logger or create a new one
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Creating sample data with {n_groups} groups, {n_timesteps} timesteps")
    logger.info(f"Numerical features: {n_num_features}, Categorical features: {n_cat_features}")

    data_list = []
    groups = [f"group_{i}" for i in range(n_groups)]
    logger.debug(f"Group IDs: {groups}")

    # Define feature and target column names
    num_cols = [f"num_{i}" for i in range(n_num_features)]
    cat_cols = [f"cat_{i}" for i in range(n_cat_features)]
    target_cols = [f"target_{i}" for i in range(n_targets)]

    logger.info(f"Numerical columns: {num_cols}")
    logger.info(f"Categorical columns: {cat_cols}")
    logger.info(f"Target columns: {target_cols}")

    # Define known/unknown split (if requested)
    if include_unknown:
        known_num = num_cols[: n_num_features // 2]
        unknown_num = num_cols[n_num_features // 2 :]
        known_cat = cat_cols[: n_cat_features // 2]
        unknown_cat = cat_cols[n_cat_features // 2 :]
        logger.info(f"Known numerical columns: {known_num}")
        logger.info(f"Unknown numerical columns: {unknown_num}")
        logger.info(f"Known categorical columns: {known_cat}")
        logger.info(f"Unknown categorical columns: {unknown_cat}")
    else:
        known_num = num_cols
        unknown_num = []
        known_cat = cat_cols
        unknown_cat = []
        logger.info("All features set as known (no unknown features)")

    for group in groups:
        logger.debug(f"Generating data for group: {group}")
        # Generate time series with trend and seasonality
        time_idx = np.arange(n_timesteps)

        # Create DataFrame for this group
        group_data = {
            "time": pd.date_range("2020-01-01", periods=n_timesteps, freq="D"),
            "group_id": group,  # Use group_id as the standard group column name
        }
        logger.debug(f"Time range: {group_data['time'].min()} to {group_data['time'].max()}")

        # Add numerical features
        logger.debug(f"Adding {len(num_cols)} numerical features for group {group}")
        for i, col in enumerate(num_cols):
            # Create features with different patterns
            trend = 0.1 * time_idx * (i + 1)
            seasonal = (i + 1) * np.sin(2 * np.pi * time_idx / (12 + i))
            noise = np.random.normal(0, 0.5, n_timesteps)
            group_data[col] = trend + seasonal + noise
            logger.debug(
                f"Created numerical feature {col} with range:"
                f" [{group_data[col].min():.2f}, {group_data[col].max():.2f}]"
            )

        # Add categorical features
        logger.debug(f"Adding {len(cat_cols)} categorical features for group {group}")
        for i, col in enumerate(cat_cols):
            # Create categorical features with different cardinalities
            cardinality = 3 + i
            group_data[col] = [
                f"val_{np.random.randint(0, cardinality)}" for _ in range(n_timesteps)
            ]
            unique_vals = set(group_data[col])
            logger.debug(
                f"Created categorical feature {col} with"
                f" {len(unique_vals)} unique values: {unique_vals}"
            )

        # Add target variables
        logger.debug(f"Adding {len(target_cols)} target variables for group {group}")
        for i, col in enumerate(target_cols):
            # Create targets as functions of features
            if i < len(num_cols):
                # Base target on numerical feature with some noise
                base_feature = group_data[num_cols[i]]
                group_data[col] = base_feature * 2 + np.random.normal(0, 1, n_timesteps)
                logger.debug(
                    f"Created target {col} based on {num_cols[i]} with range:"
                    f" [{group_data[col].min():.2f}, {group_data[col].max():.2f}]"
                )
            else:
                # Create synthetic target
                trend = 0.2 * time_idx
                seasonal = 5 * np.sin(2 * np.pi * time_idx / 30)
                noise = np.random.normal(0, 2, n_timesteps)
                group_data[col] = trend + seasonal + noise
                logger.debug(
                    f"Created synthetic target {col} with range:"
                    f" [{group_data[col].min():.2f}, {group_data[col].max():.2f}]"
                )

        # Convert to DataFrame and append to list
        df = pd.DataFrame(group_data)
        logger.debug(f"Created DataFrame for group {group} with shape {df.shape}")
        data_list.append(df)

    # Combine all groups
    combined_df = pd.concat(data_list, ignore_index=True)
    logger.info(f"Combined DataFrame created with shape {combined_df.shape}")

    # Log data statistics
    logger.info("Data statistics:")
    for col in combined_df.columns:
        if col in num_cols + target_cols:
            logger.info(
                f"  {col}: min={combined_df[col].min():.2f},"
                f" max={combined_df[col].max():.2f},"
                f" mean={combined_df[col].mean():.2f}"
            )
        elif col in cat_cols:
            unique_vals = combined_df[col].nunique()
            logger.info(f"  {col}: {unique_vals} unique values")

    # Save to CSV
    data_path = os.path.join(temp_dir, test_name_csv)
    combined_df.to_csv(data_path, index=False)
    logger.info(f"Data saved to CSV: {data_path}")

    # Create metadata
    metadata = {
        "time_col": "time",
        "target_cols": target_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "group_cols": ["group_id"],  # Use group_id as the standard group column name
        "groups": groups,
        "n_timesteps": n_timesteps,
    }

    # Add known/unknown columns to metadata if used
    if include_unknown:
        metadata["known_cols"] = known_num + known_cat
        metadata["unknown_cols"] = unknown_num + unknown_cat
        logger.info(
            f"Added known/unknown columns to metadata:"
            f" {len(metadata['known_cols'])} known,"
            f" {len(metadata['unknown_cols'])} unknown"
        )

    logger.info(f"Metadata created with keys: {list(metadata.keys())}")
    logger.info(f"Metadata looks like this:: {metadata}")
    return data_path, metadata


file_path = "/home/sandeep/DSIPTS_PTF/tests/integration"

# def test_basic_integration(file_path):
"""Test basic integration between D1 and D2 layers."""
logger = logging.getLogger(__name__)
test_name_csv = "test_basic_integration.csv"
logger.info("\n[1/5] CREATING SAMPLE DATA")
logger.info("-" * 40)
data_path, metadata = create_sample_data(
    file_path, debug=True, logger=logger, test_name_csv=test_name_csv
)
logger.info(f"✓ Sample data created at: {data_path}")
logger.info(f"✓ Metadata keys: {list(metadata.keys())}")

# Log important metadata in a more readable format
logger.info("\n[2/5] DATASET METADATA")
logger.info("-" * 40)
logger.info(f"Time Column: {metadata['time_col']}")
logger.info(f"Target Columns: {metadata['target_cols']}")
logger.info(f"Numerical Columns: {metadata.get('num_cols', [])}")
logger.info(f"Categorical Columns: {metadata.get('cat_cols', [])}")
logger.info(f"Group Columns: {metadata.get('group_cols', [])}")
logger.info(f"Number of Groups: {len(metadata.get('groups', []))}")
logger.info(f"Number of Timesteps: {metadata.get('n_timesteps', 'N/A')}")

# Print column names for debugging
logger.info("\n[3/5] CHECKING CSV FILE STRUCTURE")
logger.info("-" * 40)
df = pd.read_csv(data_path)
logger.info(f"CSV columns: {df.columns.tolist()}")
logger.info(f"Group values: {df['group_id'].unique().tolist()}")
logger.info(f"Time range: {pd.to_datetime(df['time']).min()} to {pd.to_datetime(df['time']).max()}")
logger.debug(f"First few rows:\n{df.head()}")

# Save the CSV file for inspection
debug_csv_path = os.path.join(file_path, test_name_csv)
df.to_csv(debug_csv_path, index=False)
logger.info(f"Saved debug CSV to {debug_csv_path}")

# Initialize D1 layer with group_cols as a list
logger.info("\n[4/5] INITIALIZING D1 LAYER")
logger.info("-" * 40)
d1 = MultiSourceTSDataSet(
    file_paths=[data_path],
    time_col=metadata["time_col"],
    target_cols=metadata["target_cols"],
    group_cols=metadata["group_cols"],  # This is a list ["group_id"]
    num_cols=metadata["num_cols"],
    cat_cols=metadata["cat_cols"],
    global_forecasting=False,
    enrich_cat=["hour", "dow"],
)
logger.info(f"D1 initialized with group_cols={d1.group_cols}")
logger.info(f"D1 has {len(d1)} groups")
logger.info(f"D1 total length: {d1.total_length}")

# Log D1 metadata
logger.info("D1 metadata:")
for key, value in d1.metadata.items():
    if isinstance(value, list) and len(value) > 10:
        logger.info(f"  {key}: {value[:5]} ... (truncated, {len(value)} items)")
    else:
        logger.info(f"  {key}: {value}")

# Log categorical encoders
logger.info("D1 categorical encoders:")
for col, encoder in d1.label_encoders.items():
    logger.info(f"  {col}: {len(encoder.classes_)} classes")
