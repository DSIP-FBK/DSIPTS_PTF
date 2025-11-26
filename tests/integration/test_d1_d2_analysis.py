"""
Comprehensive D1 and D2 Layer Analysis Test

This test analyzes D1 and D2 layer behavior under various conditions:
1. Data loading methods: CSV, DataFrame, Parquet, Pickle
2. Memory efficiency: On/Off
3. Global vs Local forecasting
4. Temporal enrichment: None, Hour+DOW, Hour+DOW+Month
5. Scaling: Before/After statistics

Generates detailed logs for analysis.
Usage:
python3 tests/integration/test_d1_d2_analysis.py 2>&1 | tee tests/integration/d1_d2_analysis_output.log
"""

import logging
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dsipts.data_structure.d1_layers.multi_source_csv import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers.encoder_decoder import EncoderDecoder

# Configure detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def print_separator(title, char="=", width=100):
    """Print a formatted separator."""
    logger.info("")
    logger.info(char * width)
    logger.info(f" {title} ".center(width, char))
    logger.info(char * width)


def print_subsection(title, char="-", width=100):
    """Print a formatted subsection."""
    logger.info("")
    logger.info(char * width)
    logger.info(f" {title} ")
    logger.info(char * width)


def create_sample_data(n_groups=3, seq_length=200, with_datetime=True):
    """Create sample time series data for testing."""
    np.random.seed(42)
    data = []

    start_date = datetime(2024, 1, 1)

    for g in range(n_groups):
        for t in range(seq_length):
            if with_datetime:
                timestamp = start_date + timedelta(hours=t)
            else:
                timestamp = t

            data.append(
                {
                    "timestamp": timestamp,
                    "group_id": f"group_{g}",
                    "station": f"station_{g % 2}",  # Additional categorical
                    "temperature": 20 + 10 * np.sin(2 * np.pi * t / 24) + g * 5 + np.random.randn(),
                    "humidity": 60 + 20 * np.cos(2 * np.pi * t / 24) + np.random.randn() * 5,
                    "pressure": 1013 + np.random.randn() * 10,
                    "target": 100 + 50 * np.sin(2 * np.pi * t / 48) + g * 10 + np.random.randn() * 5,
                }
            )

    return pd.DataFrame(data)


def analyze_d1_metadata(d1_dataset, test_name):
    """Analyze and log D1 layer metadata."""
    print_subsection(f"D1 Metadata Analysis: {test_name}")

    # Get metadata if available
    metadata = d1_dataset.metadata if hasattr(d1_dataset, "metadata") else {}

    group_ids = d1_dataset._group_ids if hasattr(d1_dataset, "_group_ids") else []

    logger.info(f"Number of groups: {len(group_ids)}")
    logger.info(f"Group IDs: {group_ids[:5]}{'...' if len(group_ids) > 5 else ''}")
    logger.info(f"Total samples: {len(d1_dataset)}")
    logger.info(f"Memory efficient: {d1_dataset.memory_efficient}")
    logger.info(f"Global forecasting: {d1_dataset.global_forecasting}")

    # Column information
    logger.info(f"Time column: {d1_dataset._time_col}")
    logger.info(f"Group columns: {d1_dataset._group_cols}")
    logger.info(f"Numerical columns: {d1_dataset._num_cols}")
    logger.info(f"Categorical columns: {d1_dataset._cat_cols}")
    logger.info(f"Target columns: {d1_dataset._target_cols}")
    logger.info(f"Past columns: {d1_dataset._past_cols}")
    logger.info(f"Future columns: {d1_dataset._future_cols}")

    # Temporal enrichment
    if d1_dataset._enrich_cat:
        logger.info(f"Temporal enrichment: {d1_dataset._enrich_cat}")
    else:
        logger.info("Temporal enrichment: None")

    # Cardinality information
    if hasattr(d1_dataset, "cardinality"):
        logger.info(f"Categorical cardinalities: {d1_dataset.cardinality}")

    # Show ALL metadata keys and values in organized sections
    if metadata:
        logger.info(f"\n{'='*80}")
        logger.info("COMPLETE METADATA DUMP (ORGANIZED BY CATEGORY):")
        logger.info(f"{'='*80}")

        # Section 1: Basic Info
        logger.info("\n[BASIC INFO]")
        for key in ["n_targets", "n_features", "n_categorical", "n_groups", "total_samples", "n_files", "n_file_groups"]:
            if key in metadata:
                logger.info(f"  {key}: {metadata[key]}")

        # Section 2: Column Information
        logger.info("\n[COLUMN INFORMATION]")
        for key in ["time_col", "target_cols", "feature_cols", "categorical_columns", "enrich_cat"]:
            if key in metadata:
                logger.info(f"  {key}: {metadata[key]}")

        # Section 3: Categorical Details
        logger.info("\n[CATEGORICAL DETAILS]")
        for key in ["cat_cols_list", "cat_cardinalities", "idx_categorical"]:
            if key in metadata:
                logger.info(f"  {key}: {metadata[key]}")

        # Section 4: Past/Future Configuration
        logger.info("\n[PAST/FUTURE CONFIGURATION]")
        for key in ["n_past", "n_future", "past_cols", "future_cols", "original_future_cols", "idx_past", "idx_future"]:
            if key in metadata:
                logger.info(f"  {key}: {metadata[key]}")

        # Section 5: Target Information
        logger.info("\n[TARGET INFORMATION]")
        for key in ["idx_targets"]:
            if key in metadata:
                logger.info(f"  {key}: {metadata[key]}")

        # Section 6: Group Mapping
        logger.info("\n[GROUP MAPPING]")
        for key in ["group_mapping", "reverse_mapping"]:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, dict) and len(value) <= 5:
                    logger.info(f"  {key}: {value}")
                elif isinstance(value, dict):
                    logger.info(f"  {key}: dict with {len(value)} entries")
                    logger.info(f"    Sample (first 3): {dict(list(value.items())[:3])}")
                else:
                    logger.info(f"  {key}: {value}")

        # Section 7: Configuration Flags
        logger.info("\n[CONFIGURATION FLAGS]")
        for key in ["memory_efficient", "global_forecasting"]:
            if key in metadata:
                logger.info(f"  {key}: {metadata[key]}")

        # Section 8: File Paths
        logger.info("\n[FILE PATHS]")
        if "file_paths" in metadata:
            logger.info(f"  file_paths: {metadata['file_paths']}")

        # Section 9: Any remaining keys
        shown_keys = {
            "n_targets",
            "n_features",
            "n_categorical",
            "n_groups",
            "total_samples",
            "n_files",
            "n_file_groups",
            "time_col",
            "target_cols",
            "feature_cols",
            "categorical_columns",
            "enrich_cat",
            "cat_cols_list",
            "cat_cardinalities",
            "idx_categorical",
            "n_past",
            "n_future",
            "past_cols",
            "future_cols",
            "original_future_cols",
            "idx_past",
            "idx_future",
            "idx_targets",
            "group_mapping",
            "reverse_mapping",
            "memory_efficient",
            "global_forecasting",
            "file_paths",
        }
        remaining_keys = set(metadata.keys()) - shown_keys
        if remaining_keys:
            logger.info("\n[OTHER METADATA]")
            for key in sorted(remaining_keys):
                value = metadata[key]
                if isinstance(value, (list, tuple)) and len(value) > 10:
                    logger.info(f"  {key}: {type(value).__name__} with {len(value)} items (first 5: {value[:5]})")
                elif isinstance(value, dict) and len(value) > 10:
                    logger.info(f"  {key}: dict with {len(value)} keys (sample: {dict(list(value.items())[:3])})")
                else:
                    logger.info(f"  {key}: {value}")

        logger.info(f"\n{'='*80}")


def analyze_d1_getitem(d1_dataset, test_name, sample_idx=0):
    print_subsection(f"D1 GetItem Analysis: {test_name} (Sample {sample_idx})")

    sample = d1_dataset[sample_idx]

    logger.info(f"Sample keys: {list(sample.keys())}")

    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE D1 __GETITEM__ OUTPUT:")
    logger.info(f"{'='*80}")

    for key, value in sample.items():
        if hasattr(value, "shape"):
            logger.info(f"  {key}:")
            logger.info(f"    shape: {value.shape}")
            logger.info(f"    dtype: {value.dtype}")
            if value.numel() <= 50:  # Print small tensors
                logger.info(f"    values: {value}")
            else:
                logger.info(f"    values (first 10): {value.flatten()[:10]}")
        elif isinstance(value, (list, tuple)):
            logger.info(f"  {key}:")
            logger.info(f"    type: {type(value).__name__}")
            logger.info(f"    length: {len(value)}")
            if len(value) <= 20:
                logger.info(f"    values: {value}")
            else:
                logger.info(f"    values (first 10): {value[:10]}")
        elif isinstance(value, dict):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info(f"{'='*80}")


def analyze_d2_metadata(d2_dataset, test_name):
    """Analyze and log D2 layer metadata."""
    print_subsection(f"D2 Metadata Analysis: {test_name}")

    logger.info(f"Past length: {d2_dataset.past_len}")
    logger.info(f"Future length: {d2_dataset.future_len}")
    logger.info(f"Step size: {d2_dataset.step_size}")
    logger.info(f"Batch size: {d2_dataset.batch_size}")
    logger.info(f"Number of valid windows: {len(d2_dataset.valid_windows)}")

    # Scaling information
    logger.info(f"Scaling method: {d2_dataset.scaling_method}")
    logger.info(f"Scale targets: {d2_dataset.scale_targets}")
    logger.info(f"Scaler fitted: {d2_dataset.is_scaler_fitted}")

    if d2_dataset.is_scaler_fitted and d2_dataset.feature_scaler:
        logger.info(f"Feature scaler type: {type(d2_dataset.feature_scaler).__name__}")
        if hasattr(d2_dataset.feature_scaler, "mean_"):
            logger.info(f"Feature scaler mean: {d2_dataset.feature_scaler.mean_}")
            logger.info(f"Feature scaler std: {np.sqrt(d2_dataset.feature_scaler.var_)}")

    if d2_dataset.target_scaler:
        logger.info(f"Target scaler type: {type(d2_dataset.target_scaler).__name__}")
        if hasattr(d2_dataset.target_scaler, "mean_"):
            logger.info(f"Target scaler mean: {d2_dataset.target_scaler.mean_}")
            logger.info(f"Target scaler std: {np.sqrt(d2_dataset.target_scaler.var_)}")


def analyze_d2_getitem(d2_dataset, test_name, sample_idx=0):
    """Analyze and log D2 layer __getitem__ output."""
    print_subsection(f"D2 GetItem Analysis: {test_name} (Sample {sample_idx})")

    x, y = d2_dataset.dataset[sample_idx]

    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE D2 __GETITEM__ OUTPUT:")
    logger.info(f"{'='*80}")

    logger.info(f"Input (x) keys: {list(x.keys())}")
    logger.info("\nInput (x) structure:")
    for key, value in x.items():
        if hasattr(value, "shape"):
            logger.info(f"  {key}:")
            logger.info(f"    shape: {value.shape}")
            logger.info(f"    dtype: {value.dtype}")
            if value.numel() <= 50:  # Print small tensors
                logger.info(f"    values:\n{value}")
            else:
                logger.info(f"    values (first 20): {value.flatten()[:20]}")
        elif isinstance(value, (list, tuple)):
            logger.info(f"  {key}:")
            logger.info(f"    type: {type(value).__name__}")
            logger.info(f"    length: {len(value)}")
            if len(value) <= 20:
                logger.info(f"    values: {value}")
            else:
                logger.info(f"    values (first 10): {value[:10]}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("\nTarget (y):")
    logger.info(f"  shape: {y.shape}")
    logger.info(f"  dtype: {y.dtype}")
    if y.numel() <= 50:
        logger.info(f"  values:\n{y}")
    else:
        logger.info(f"  values (first 20): {y.flatten()[:20]}")

    logger.info(f"{'='*80}")


def analyze_scaling_statistics(df, feature_cols, target_cols, test_name):
    """Analyze and log data statistics before scaling."""
    print_subsection(f"Data Statistics (Before Scaling): {test_name}")

    logger.info("Feature Statistics:")
    for col in feature_cols:
        stats = df[col].describe()
        logger.info(f"  {col}:")
        logger.info(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        logger.info(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        logger.info(f"    25%: {stats['25%']:.4f}, 50%: {stats['50%']:.4f}, 75%: {stats['75%']:.4f}")

    logger.info("Target Statistics:")
    for col in target_cols:
        stats = df[col].describe()
        logger.info(f"  {col}:")
        logger.info(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        logger.info(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")


def analyze_scaled_data(d2_dataset, test_name, train_dataset=None):
    """Analyze and log data statistics after scaling."""
    if not d2_dataset.is_scaler_fitted:
        logger.info("Scaler not fitted, skipping scaled data analysis")
        return

    print_subsection(f"Data Statistics (After Scaling): {test_name}")

    # For memory-efficient mode, iterate through all data to get accurate stats
    if d2_dataset.d1_dataset.memory_efficient and train_dataset:
        logger.info("Computing statistics from all training data (memory-efficient mode)...")
        all_features = []
        all_targets = []

        for i in range(len(train_dataset)):
            x, y = train_dataset[i]
            if "x_num_past" in x:
                all_features.append(x["x_num_past"].numpy())
            all_targets.append(y.numpy())

        if all_features:
            features = np.concatenate(all_features, axis=0)
            logger.info("Scaled Feature Statistics (from all training data):")
            logger.info(f"  Total samples: {features.shape[0]}")
            logger.info(f"  Feature shape: {features.shape}")
            logger.info(f"  Mean per feature: {np.mean(features, axis=0)}")
            logger.info(f"  Std per feature: {np.std(features, axis=0)}")
            logger.info(f"  Min per feature: {np.min(features, axis=0)}")
            logger.info(f"  Max per feature: {np.max(features, axis=0)}")

        if d2_dataset.scale_targets and all_targets:
            targets = np.concatenate(all_targets, axis=0)
            logger.info("\nScaled Target Statistics (from all training data):")
            logger.info(f"  Total samples: {targets.shape[0]}")
            logger.info(f"  Mean: {np.mean(targets):.4f}")
            logger.info(f"  Std: {np.std(targets):.4f}")
            logger.info(f"  Min: {np.min(targets):.4f}")
            logger.info(f"  Max: {np.max(targets):.4f}")
    else:
        # Non-memory-efficient mode - get scaled data from train_dataset
        if train_dataset is None:
            logger.warning("No train_dataset provided, cannot show scaled statistics")
            return

        x, y = train_dataset[0]

        if "x_num_past" in x:
            features = x["x_num_past"].numpy()
            logger.info("Scaled Feature Statistics (from train_dataset sample):")
            logger.info(f"  Shape: {features.shape}")
            logger.info(f"  Mean per feature: {np.mean(features, axis=0)}")
            logger.info(f"  Std per feature: {np.std(features, axis=0)}")
            logger.info(f"  Min per feature: {np.min(features, axis=0)}")
            logger.info(f"  Max per feature: {np.max(features, axis=0)}")

        if d2_dataset.scale_targets:
            targets = y.numpy()
            logger.info("\nScaled Target Statistics (from train_dataset sample):")
            logger.info(f"  Shape: {targets.shape}")
            logger.info(f"  Mean: {np.mean(targets):.4f}")
            logger.info(f"  Std: {np.std(targets):.4f}")
            logger.info(f"  Min: {np.min(targets):.4f}")
            logger.info(f"  Max: {np.max(targets):.4f}")


def test_data_loading_methods():
    """Test different data loading methods."""
    print_separator("TEST 1: DATA LOADING METHODS")

    # Create sample data
    df = create_sample_data(n_groups=3, seq_length=200)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Test configurations
    configs = [
        ("CSV - Memory Efficient OFF", "csv", False),
        ("CSV - Memory Efficient ON", "csv", True),
        ("DataFrame - Memory Efficient OFF", "dataframe", False),
        ("DataFrame - Memory Efficient ON (should warn)", "dataframe", True),
        ("Parquet - Memory Efficient OFF", "parquet", False),
        ("Parquet - Memory Efficient ON", "parquet", True),
        ("Pickle - Memory Efficient OFF", "pickle", False),
    ]

    for test_name, file_format, memory_eff in configs:
        print_separator(f"Test: {test_name}", char="-")

        try:
            # Save data in appropriate format
            if file_format == "csv":
                file_path = os.path.join(temp_dir, "data.csv")
                df.to_csv(file_path, index=False)
                d1 = MultiSourceTSDataSet(
                    file_paths=[file_path],
                    time_col="timestamp",
                    group_cols=["group_id"],
                    num_cols=["temperature", "humidity", "pressure"],
                    cat_cols=["station"],
                    target_cols=["target"],
                    memory_efficient=memory_eff,
                )

            elif file_format == "dataframe":
                d1 = MultiSourceTSDataSet(
                    dataframes=[df],
                    time_col="timestamp",
                    group_cols=["group_id"],
                    num_cols=["temperature", "humidity", "pressure"],
                    cat_cols=["station"],
                    target_cols=["target"],
                    memory_efficient=memory_eff,
                )

            elif file_format == "parquet":
                file_path = os.path.join(temp_dir, "data.parquet")
                df.to_parquet(file_path, index=False)
                d1 = MultiSourceTSDataSet(
                    file_paths=[file_path],
                    time_col="timestamp",
                    group_cols=["group_id"],
                    num_cols=["temperature", "humidity", "pressure"],
                    cat_cols=["station"],
                    target_cols=["target"],
                    memory_efficient=memory_eff,
                )

            elif file_format == "pickle":
                file_path = os.path.join(temp_dir, "data.pkl")
                df.to_pickle(file_path)
                d1 = MultiSourceTSDataSet(
                    file_paths=[file_path],
                    time_col="timestamp",
                    group_cols=["group_id"],
                    num_cols=["temperature", "humidity", "pressure"],
                    cat_cols=["station"],
                    target_cols=["target"],
                    memory_efficient=memory_eff,
                )

            # Analyze D1
            analyze_d1_metadata(d1, test_name)
            analyze_d1_getitem(d1, test_name, sample_idx=0)

            logger.info(f"✓ {test_name} completed successfully")

        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


def test_global_vs_local_forecasting():
    """Test global vs local forecasting behavior."""
    print_separator("TEST 2: GLOBAL VS LOCAL FORECASTING")

    df = create_sample_data(n_groups=3, seq_length=200)

    configs = [
        ("Local Forecasting", False),
        ("Global Forecasting", True),
    ]

    for test_name, global_forecasting in configs:
        print_separator(f"Test: {test_name}", char="-")

        # D1 Layer
        d1 = MultiSourceTSDataSet(
            dataframes=[df],
            time_col="timestamp",
            group_cols=["group_id"],
            num_cols=["temperature", "humidity", "pressure"],
            cat_cols=["station"],
            target_cols=["target"],
            global_forecasting=global_forecasting,
        )

        analyze_d1_metadata(d1, test_name)
        analyze_d1_getitem(d1, test_name, sample_idx=0)

        # D2 Layer
        d2 = EncoderDecoder(d1_dataset=d1, past_len=24, future_len=12, batch_size=16)

        analyze_d2_metadata(d2, test_name)
        analyze_d2_getitem(d2, test_name, sample_idx=0)

        logger.info(f"✓ {test_name} completed successfully")


def test_temporal_enrichment():
    """Test temporal enrichment effects."""
    print_separator("TEST 3: TEMPORAL ENRICHMENT")

    df = create_sample_data(n_groups=3, seq_length=200)

    configs = [
        ("No Temporal Enrichment", None),
        ("Hour + DOW", ["hour", "dow"]),
        ("Hour + DOW + Month + Minute", ["hour", "dow", "month", "minute"]),
    ]

    for test_name, enrich_cat in configs:
        print_separator(f"Test: {test_name}", char="-")

        # D1 Layer
        d1 = MultiSourceTSDataSet(
            dataframes=[df],
            time_col="timestamp",
            group_cols=["group_id"],
            num_cols=["temperature", "humidity", "pressure"],
            cat_cols=["station"],
            target_cols=["target"],
            enrich_cat=enrich_cat,
        )

        analyze_d1_metadata(d1, test_name)
        analyze_d1_getitem(d1, test_name, sample_idx=0)

        # D2 Layer
        d2 = EncoderDecoder(d1_dataset=d1, past_len=24, future_len=12, batch_size=16)

        analyze_d2_metadata(d2, test_name)
        analyze_d2_getitem(d2, test_name, sample_idx=0)

        logger.info(f"✓ {test_name} completed successfully")


def test_scaling_effects():
    """Test scaling effects on data."""
    print_separator("TEST 4: SCALING EFFECTS")

    df = create_sample_data(n_groups=3, seq_length=200)

    feature_cols = ["temperature", "humidity", "pressure"]
    target_cols = ["target"]

    configs = [
        ("No Scaling", None, False),
        ("StandardScaler - Features Only", "standard", False),
        ("StandardScaler - Features + Targets", "standard", True),
        ("MinMaxScaler - Features Only", "minmax", False),
        ("MinMaxScaler - Features + Targets", "minmax", True),
    ]

    for test_name, scaling_method, scale_targets in configs:
        print_separator(f"Test: {test_name}", char="-")

        # Analyze original data statistics
        analyze_scaling_statistics(df, feature_cols, target_cols, test_name)

        # D1 Layer - Use memory_efficient=False for scaling tests to see full flow
        d1 = MultiSourceTSDataSet(
            dataframes=[df],
            time_col="timestamp",
            group_cols=["group_id"],
            num_cols=feature_cols,
            cat_cols=["station"],
            target_cols=target_cols,
            memory_efficient=False,  # Disable for clearer scaling analysis
        )

        # D2 Layer with scaling
        d2 = EncoderDecoder(
            d1_dataset=d1, past_len=24, future_len=12, batch_size=16, scaling_method=scaling_method, scale_targets=scale_targets
        )

        # Split to fit scaler
        train_dataset = None
        if scaling_method:
            d2.setup(stage="train")
            train_dataset = d2.train_dataset

        analyze_d2_metadata(d2, test_name)
        analyze_scaled_data(d2, test_name, train_dataset=train_dataset)

        logger.info(f"✓ {test_name} completed successfully")


def test_comprehensive_scenario():
    """Test a comprehensive scenario combining multiple features."""
    print_separator("TEST 5: COMPREHENSIVE SCENARIO")

    test_name = "CSV + Global + Temporal + StandardScaler"
    print_separator(f"Test: {test_name}", char="-")

    # Create data
    df = create_sample_data(n_groups=3, seq_length=200)

    # Save to CSV
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "data.csv")
    df.to_csv(csv_path, index=False)

    # Analyze original data
    analyze_scaling_statistics(df, ["temperature", "humidity", "pressure"], ["target"], test_name)

    # D1 Layer with all features
    d1 = MultiSourceTSDataSet(
        file_paths=[csv_path],
        time_col="timestamp",
        group_cols=["group_id"],
        num_cols=["temperature", "humidity", "pressure"],
        cat_cols=["station"],
        target_cols=["target"],
        enrich_cat=["hour", "dow"],
        global_forecasting=True,
        memory_efficient=False,
    )

    analyze_d1_metadata(d1, test_name)
    analyze_d1_getitem(d1, test_name, sample_idx=0)

    # D2 Layer with scaling
    d2 = EncoderDecoder(d1_dataset=d1, past_len=24, future_len=12, batch_size=16, scaling_method="standard", scale_targets=True)

    # Split and fit scaler
    d2.setup(stage="train")

    analyze_d2_metadata(d2, test_name)
    analyze_d2_getitem(d2, test_name, sample_idx=0)
    analyze_scaled_data(d2, test_name, train_dataset=d2.train_dataset)

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)

    logger.info(f"✓ {test_name} completed successfully")


if __name__ == "__main__":
    print_separator("D1 AND D2 LAYER COMPREHENSIVE ANALYSIS", char="=", width=100)
    logger.info("Starting comprehensive analysis of D1 and D2 layers...")
    logger.info("This test will generate detailed logs for analysis.")
    logger.info("")

    # Run all tests
    test_data_loading_methods()
    test_global_vs_local_forecasting()
    test_temporal_enrichment()
    test_scaling_effects()
    test_comprehensive_scenario()

    print_separator("ANALYSIS COMPLETE", char="=", width=100)
    logger.info("All tests completed successfully!")
    logger.info("Review the logs above for detailed analysis of D1 and D2 layer behavior.")
