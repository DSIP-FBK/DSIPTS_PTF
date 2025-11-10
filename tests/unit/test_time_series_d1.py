"""
Comprehensive Unit Test Suite for D1 Layer (MultiSourceTSDataSet)

Coverage:
- Basic initialization and data loading
- Memory-efficient vs in-memory modes
- Temporal enrichment (hour, dow, month, minute)
- Categorical encoding with ordered lists
- Global vs local forecasting
- Known/unknown columns
- Multi-file and DataFrame inputs
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from dsipts.data_structure.d1_layers import MultiSourceTSDataSet

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def basic_csv_data(temp_dir):
    """Basic CSV with numerical and categorical features."""
    np.random.seed(42)
    data = []
    for g in range(3):
        for t in range(50):
            data.append(
                {
                    "time": t,
                    "group_id": f"group_{g}",
                    "num_0": np.sin(t / 10) + g,
                    "num_1": np.cos(t / 10) + g * 2,
                    "cat_0": f"cat_{np.random.randint(0,3)}",
                    "target_0": np.sin(t / 5) + g,
                }
            )
    df = pd.DataFrame(data)
    path = os.path.join(temp_dir, "data.csv")
    df.to_csv(path, index=False)
    return {
        "file_paths": [path],
        "group_cols": ["group_id"],
        "time_col": "time",
        "target_cols": ["target_0"],
        "num_cols": ["num_0", "num_1"],
        "cat_cols": ["cat_0"],
        "past_cols": ["num_0", "num_1", "cat_0"],  # All features in past
        "future_cols": ["num_0"],  # Only num_0 known in future
    }


@pytest.fixture
def datetime_data(temp_dir):
    """Datetime data for temporal enrichment."""
    np.random.seed(42)
    data = []
    start = datetime(2024, 1, 1)
    for g in range(2):
        for h in range(168):  # 1 week hourly
            data.append(
                {
                    "timestamp": start + timedelta(hours=h),
                    "group_id": f"group_{g}",
                    "temp": 20 + 10 * np.sin(2 * np.pi * h / 24),
                    "weather": np.random.choice(["sunny", "cloudy"]),
                }
            )
    df = pd.DataFrame(data)
    path = os.path.join(temp_dir, "datetime.csv")
    df.to_csv(path, index=False)
    return {
        "file_paths": [path],
        "group_cols": ["group_id"],
        "time_col": "timestamp",
        "target_cols": ["temp"],
        "cat_cols": ["weather"],
    }


class TestD1Basic:
    """Basic functionality tests."""

    def test_csv_init(self, basic_csv_data):
        d1 = MultiSourceTSDataSet(**basic_csv_data)
        assert len(d1) == 3
        assert d1.group_cols == ["group_id"]
        logger.info("✓ CSV init")

    def test_memory_modes(self, basic_csv_data):
        d1_mem = MultiSourceTSDataSet(**basic_csv_data, memory_efficient=False)
        d1_eff = MultiSourceTSDataSet(**basic_csv_data, memory_efficient=True)
        assert len(d1_mem) == len(d1_eff) == 3
        logger.info("✓ Memory modes")

    def test_getitem_structure(self, basic_csv_data):
        d1 = MultiSourceTSDataSet(**basic_csv_data)
        sample = d1[0]
        assert all(k in sample for k in ["x", "y", "group_id", "seq_len"])
        assert sample["x"].shape[0] == sample["seq_len"]
        logger.info(f"✓ Sample: x={sample['x'].shape}, y={sample['y'].shape}")

    def test_explicit_metadata_and_getitem_dump(self, basic_csv_data):
        """Explicitly print all D1 metadata and getitem output for inspection."""
        d1 = MultiSourceTSDataSet(**basic_csv_data, enrich_cat=["hour", "dow"])

        print("\n" + "=" * 100)
        print("D1 METADATA COMPLETE DUMP:")
        print("=" * 100)
        print(d1.metadata)
        print("=" * 100)

        sample = d1[0]
        print("\n" + "=" * 100)
        print("D1 __GETITEM__[0] COMPLETE DUMP:")
        print("=" * 100)
        print(sample)
        print("=" * 100)

        # Also print individual keys for clarity
        print("\nD1 __GETITEM__[0] DETAILED BREAKDOWN:")
        for key, value in sample.items():
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            if hasattr(value, "shape"):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Value:\n{value}")
            else:
                print(f"  Value: {value}")

        logger.info("✓ Explicit metadata and getitem dump completed")


class TestD1CategoricalInfo:
    """Test categorical information as ordered lists."""

    def test_cat_cols_and_cardinalities(self, basic_csv_data):
        d1 = MultiSourceTSDataSet(**basic_csv_data)

        # Check metadata instead of sample (cat_cols removed from getitem per MR comment)
        assert "cat_past_list" in d1.metadata
        assert "cat_past_cardinalities" in d1.metadata
        cat_past_list = d1.metadata["cat_past_list"]
        cat_past_cardinalities = d1.metadata["cat_past_cardinalities"]
        assert len(cat_past_list) == len(cat_past_cardinalities)

        # Verify order preservation
        for col, card in zip(cat_past_list, cat_past_cardinalities):
            if col in d1.label_encoders:
                actual = len(d1.label_encoders[col].classes_)
                assert card == actual
        logger.info(f"✓ Cat info: {cat_past_list} -> {cat_past_cardinalities}")


class TestD1TemporalEnrichment:
    """Test temporal enrichment features."""

    def test_hour_enrichment(self, datetime_data):
        d1 = MultiSourceTSDataSet(**datetime_data, enrich_cat=["hour"])

        assert "hour" in d1.metadata["cat_past_list"]
        assert "hour" in d1.label_encoders
        logger.info(f"✓ Hour enrichment: {len(d1.label_encoders['hour'].classes_)} hours")

    def test_dow_enrichment(self, datetime_data):
        d1 = MultiSourceTSDataSet(**datetime_data, enrich_cat=["dow"])

        assert "dow" in d1.metadata["cat_past_list"]
        assert len(d1.label_encoders["dow"].classes_) == 7
        logger.info("✓ DOW enrichment: 7 days")

    def test_multiple_enrichment(self, datetime_data):
        d1 = MultiSourceTSDataSet(**datetime_data, enrich_cat=["hour", "dow", "month"])

        for feat in ["hour", "dow", "month"]:
            assert feat in d1.metadata["cat_past_list"]
        logger.info("✓ Multiple enrichment")


class TestD1GlobalForecasting:
    """Test global vs local forecasting modes."""

    def test_global_forecasting_true(self, basic_csv_data):
        d1 = MultiSourceTSDataSet(**basic_csv_data, global_forecasting=True)
        assert d1.global_forecasting is True
        assert "group_id" not in d1.cat_cols  # Not auto-added in global mode
        logger.info("✓ Global forecasting")

    def test_global_forecasting_false(self, basic_csv_data):
        d1 = MultiSourceTSDataSet(**basic_csv_data, global_forecasting=False)
        assert d1.global_forecasting is False
        assert "group_id" in d1.cat_cols  # Auto-added in local mode
        logger.info("✓ Local forecasting")


class TestD1PastFuture:
    """Test past/future column handling."""

    def test_past_cols(self, basic_csv_data):
        d1 = MultiSourceTSDataSet(**basic_csv_data)
        assert "num_0" in d1.past_cols
        assert "num_1" in d1.past_cols
        logger.info(f"✓ Past cols: {d1.past_cols}")

    def test_future_cols(self, basic_csv_data):
        d1 = MultiSourceTSDataSet(**basic_csv_data)
        assert "num_0" in d1.future_cols
        assert "num_1" not in d1.future_cols  # num_1 not in future
        logger.info(f"✓ Future cols: {d1.future_cols}")


class TestD1EdgeCases:
    """Test edge cases and error handling."""

    def test_empty_group(self, temp_dir):
        # Create data with potential empty group
        df = pd.DataFrame({"time": [1, 2], "group": ["a", "a"], "val": [1, 2], "target": [1, 2]})
        path = os.path.join(temp_dir, "edge.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path], group_cols=["group"], time_col="time", target_cols=["target"], num_cols=["val"]
        )
        assert len(d1) == 1
        logger.info("✓ Edge case handled")


class TestD1MultiFile:
    """Test multi-file loading scenarios."""

    def test_multi_file_groups(self, temp_dir):
        np.random.seed(42)
        paths = []
        for f in range(3):
            data = [{"time": t, "group": f"g_{f}_{t%2}", "val": t, "target": t} for t in range(30)]
            df = pd.DataFrame(data)
            path = os.path.join(temp_dir, f"file_{f}.csv")
            df.to_csv(path, index=False)
            paths.append(path)

        d1 = MultiSourceTSDataSet(
            file_paths=paths, group_cols=["group"], time_col="time", target_cols=["target"], num_cols=["val"]
        )
        assert len(d1) == 6  # 3 files × 2 groups
        logger.info(f"✓ Multi-file: {len(d1)} groups across {len(paths)} files")


class TestD1CategoricalEdgeCases:
    """Test categorical encoding edge cases."""

    def test_unknown_categories(self, temp_dir):
        # Create training data with known categories
        train_data = [{"time": t, "group": "g1", "cat": f"c_{t%3}", "val": t, "target": t} for t in range(50)]
        df = pd.DataFrame(train_data)
        path = os.path.join(temp_dir, "train.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path], group_cols=["group"], time_col="time", target_cols=["target"], num_cols=["val"], cat_cols=["cat"]
        )

        # Check known categories
        assert "cat" in d1.label_encoders
        known_cats = set(d1.label_encoders["cat"].classes_)
        assert known_cats == {"c_0", "c_1", "c_2"}
        logger.info(f"✓ Known categories: {known_cats}")


class TestD1TemporalEdgeCases:
    """Test temporal enrichment edge cases."""

    def test_all_temporal_features(self, temp_dir):
        np.random.seed(42)
        start = datetime(2024, 1, 1)
        data = [{"timestamp": start + timedelta(hours=h), "group": "g1", "val": h, "target": h} for h in range(100)]
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "temporal.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            group_cols=["group"],
            time_col="timestamp",
            target_cols=["target"],
            num_cols=["val"],
            enrich_cat=["minute", "hour", "dow", "month"],
        )

        for feat in ["minute", "hour", "dow", "month"]:
            assert feat in d1.metadata["cat_past_list"]
        logger.info("✓ All temporal features")

    def test_temporal_cardinalities_validation(self, temp_dir):
        """COMPREHENSIVE: Validate temporal features have correct cardinalities."""
        np.random.seed(42)
        start = datetime(2024, 1, 1)
        data = []
        for h in range(200):
            data.append({"timestamp": start + timedelta(hours=h), "group": "g1", "val": h, "target": h})
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        # Test with hour and dow
        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="timestamp",
            group_cols=["group"],
            num_cols=["val"],
            target_cols=["target"],
            enrich_cat=["hour", "dow"],
        )

        # Check cardinalities in metadata (removed from sample per MR comment)
        assert "cat_past_cardinalities" in d1.metadata, "cat_past_cardinalities missing from metadata"
        cardinalities = d1.metadata["cat_past_cardinalities"]

        # Should have: group (1), hour (24), dow (7)
        assert len(cardinalities) == 3, f"Expected 3 cardinalities, got {len(cardinalities)}: {cardinalities}"

        # STRONG VALIDATION: Check specific values
        assert 24 in cardinalities, f"hour cardinality (24) not found in {cardinalities}"
        assert 7 in cardinalities, f"dow cardinality (7) not found in {cardinalities}"

        logger.info(f"✓ Temporal cardinalities validated: {cardinalities}")

    def test_all_temporal_cardinalities(self, temp_dir):
        """COMPREHENSIVE: Test all temporal feature cardinalities."""
        np.random.seed(42)
        start = datetime(2024, 1, 1)
        data = []
        # Generate data spanning multiple months and with varying minutes
        for d in range(90):  # 90 days to cover multiple months
            for h in range(24):  # 4 times per day
                for m in range(60):  # Different minutes
                    data.append(
                        {
                            "timestamp": start + timedelta(days=d, hours=h, minutes=m),
                            "group": "g1",
                            "val": len(data),
                            "target": len(data),
                        }
                    )
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        # Test with all temporal features
        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="timestamp",
            group_cols=["group"],
            num_cols=["val"],
            target_cols=["target"],
            enrich_cat=["hour", "dow", "month", "minute"],
        )

        cat_cols = d1.metadata["cat_past_list"]
        cardinalities = d1.metadata["cat_past_cardinalities"]

        # STRONG VALIDATION: Check all temporal features present
        assert len(cat_cols) >= 4, f"Expected at least 4 cat_cols, got {len(cat_cols)}"

        # STRONG VALIDATION: Verify specific cardinalities
        expected_cardinalities = {24, 7, 3, 60}  # hour, dow, month, minute
        found_cardinalities = set(cardinalities)

        for expected in expected_cardinalities:
            assert expected in found_cardinalities, f"Expected cardinality {expected} not found in {cardinalities}"

        logger.info(f"✓ All temporal cardinalities correct: {cardinalities}")

    def test_temporal_enrichment_in_metadata(self, temp_dir):
        """COMPREHENSIVE: Validate temporal enrichment updates metadata correctly."""
        np.random.seed(42)
        start = datetime(2024, 1, 1)
        data = []
        for h in range(200):
            data.append(
                {
                    "timestamp": start + timedelta(hours=h),
                    "group": "g1",
                    "station": "A" if h % 2 == 0 else "B",
                    "val": h,
                    "target": h,
                }
            )
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        # Create D1 with temporal enrichment
        # Note: past_cols and future_cols are explicitly provided to show expected behavior
        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="timestamp",
            group_cols=["group"],
            cat_cols=["station"],
            num_cols=["val"],
            target_cols=["target"],
            past_cols=["station"],  # Explicitly include station in past
            future_cols=["station"],  # Explicitly include station in future
            enrich_cat=["hour", "dow"],
        )

        # VALIDATION 1: Check metadata exists
        assert hasattr(d1, "metadata"), "metadata attribute missing"
        metadata = d1.metadata

        # VALIDATION 2: Check cat_past_list includes temporal features
        assert "cat_past_list" in metadata, "cat_past_list missing from metadata"
        cat_past_list = metadata["cat_past_list"]

        assert "hour" in cat_past_list, f"hour not in cat_past_list: {cat_past_list}"
        assert "dow" in cat_past_list, f"dow not in cat_past_list: {cat_past_list}"
        assert "station" in cat_past_list, f"station not in cat_past_list: {cat_past_list}"
        assert "group" in cat_past_list, f"group not in cat_past_list: {cat_past_list}"

        # VALIDATION 3: Check cat_past_cardinalities matches cat_past_list
        assert "cat_past_cardinalities" in metadata, "cat_past_cardinalities missing from metadata"
        cat_past_cardinalities = metadata["cat_past_cardinalities"]

        assert len(cat_past_list) == len(cat_past_cardinalities), (
            f"Length mismatch: cat_past_list={len(cat_past_list)}, " f"cat_past_cardinalities={len(cat_past_cardinalities)}"
        )

        # VALIDATION 4: Check specific cardinalities for temporal features
        hour_idx = cat_past_list.index("hour")
        dow_idx = cat_past_list.index("dow")

        assert cat_past_cardinalities[hour_idx] == 24, f"hour cardinality should be 24, got {cat_past_cardinalities[hour_idx]}"
        assert cat_past_cardinalities[dow_idx] == 7, f"dow cardinality should be 7, got {cat_past_cardinalities[dow_idx]}"

        # VALIDATION 5: Check enrich_cat in metadata
        assert "enrich_cat" in metadata, "enrich_cat missing from metadata"
        assert metadata["enrich_cat"] == ["hour", "dow"], f"enrich_cat mismatch: {metadata['enrich_cat']}"

        # VALIDATION 5b: Check cat_future_list includes temporal features but NOT group
        assert "cat_future_list" in metadata, "cat_future_list missing from metadata"
        cat_future_list = metadata["cat_future_list"]
        assert "hour" in cat_future_list, f"hour not in cat_future_list: {cat_future_list}"
        assert "dow" in cat_future_list, f"dow not in cat_future_list: {cat_future_list}"
        assert "station" in cat_future_list, f"station not in cat_future_list: {cat_future_list}"
        assert "group" not in cat_future_list, f"group should NOT be in cat_future_list: {cat_future_list}"

        # VALIDATION 5c: Check cat_future_cardinalities matches cat_future_list
        assert "cat_future_cardinalities" in metadata, "cat_future_cardinalities missing from metadata"
        cat_future_cardinalities = metadata["cat_future_cardinalities"]
        assert len(cat_future_list) == len(cat_future_cardinalities), (
            f"Length mismatch: cat_future_list={len(cat_future_list)}, "
            f"cat_future_cardinalities={len(cat_future_cardinalities)}"
        )

        # VALIDATION 6: Verify __getitem__ matches metadata
        sample = d1[0]
        assert (
            d1.metadata["cat_past_list"] == cat_past_list
        ), f"getitem cat_cols doesn't match metadata: {sample['cat_cols']} != {cat_past_list}"
        assert d1.metadata["cat_past_cardinalities"] == cat_past_cardinalities, (
            f"getitem cat_past_cardinalities doesn't match metadata: "
            f"{sample['cat_past_cardinalities']} != {cat_past_cardinalities}"
        )

        logger.info("✓ Temporal enrichment in metadata validated")
        logger.info(f"  cat_past_list: {cat_past_list}")
        logger.info(f"  cat_past_cardinalities: {cat_past_cardinalities}")
        logger.info(f"  cat_future_list: {cat_future_list}")
        logger.info(f"  cat_future_cardinalities: {cat_future_cardinalities}")

    def test_temporal_enrichment_metadata_all_features(self, temp_dir):
        """COMPREHENSIVE: Test metadata with all temporal features."""
        np.random.seed(42)
        start = datetime(2024, 1, 1)
        data = []
        for d in range(90):
            for h in range(0, 24, 6):  # Every 6 hours
                for m in [0, 15, 30, 45]:  # 4 different minutes
                    data.append(
                        {
                            "timestamp": start + timedelta(days=d, hours=h, minutes=m),
                            "group": "g1",
                            "val": len(data),
                            "target": len(data),
                        }
                    )
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        # Create D1 with all temporal features
        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="timestamp",
            group_cols=["group"],
            num_cols=["val"],
            target_cols=["target"],
            enrich_cat=["minute", "hour", "dow", "month"],
        )

        metadata = d1.metadata

        # VALIDATION 1: All temporal features in cat_past_list
        cat_past_list = metadata["cat_past_list"]
        for feat in ["minute", "hour", "dow", "month"]:
            assert feat in cat_past_list, f"{feat} not in cat_past_list: {cat_past_list}"

        # VALIDATION 2: Correct cardinalities for each temporal feature
        cat_past_cardinalities = metadata["cat_past_cardinalities"]
        expected_cardinalities = {
            "minute": 4,  # 0, 15, 30, 45
            "hour": 4,  # 0, 6, 12, 18
            "dow": 7,  # 7 days
            "month": 3,  # Jan, Feb, Mar
        }

        for feat, expected_card in expected_cardinalities.items():
            feat_idx = cat_past_list.index(feat)
            actual_card = cat_past_cardinalities[feat_idx]
            assert actual_card == expected_card, f"{feat} cardinality should be {expected_card}, got {actual_card}"

        # VALIDATION 3: Metadata matches getitem
        assert d1.metadata["cat_past_list"] == cat_past_list
        assert d1.metadata["cat_past_cardinalities"] == cat_past_cardinalities

        logger.info("✓ All temporal features in metadata validated")
        logger.info(f"  cat_past_list: {cat_past_list}")
        logger.info(f"  cat_past_cardinalities: {cat_past_cardinalities}")


def test_d1_memory_efficient_override_for_dataframes():
    """Test that memory_efficient is forced to False when using DataFrames."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2025-01-01", periods=500, freq="h"),
            "OT": np.random.randn(500) * 10 + 50,
        }
    )
    df["group"] = 10
    df.loc[df.shape[0] // 2 :, "group"] = 17

    # Request True - should be overridden to False
    d1_true = MultiSourceTSDataSet(
        dataframes=[df],
        group_cols=["group"],
        time_col="date",
        num_cols=["OT"],
        target_cols=["OT"],
        memory_efficient=True,  # Request True
    )

    # Should be overridden to False
    assert d1_true.memory_efficient is False, "memory_efficient should be False for DataFrames"
    assert d1_true.cached_data is not None, "cached_data should exist (not None)"

    # Request False - should stay False
    d1_false = MultiSourceTSDataSet(
        dataframes=[df], group_cols=["group"], time_col="date", num_cols=["OT"], target_cols=["OT"], memory_efficient=False
    )

    assert d1_false.memory_efficient is False, "memory_efficient should be False"
    logger.info("✓ memory_efficient override test passed")


def test_d1_target_in_features_by_default():
    """Test that target is included in features by default."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2025-01-01", periods=100, freq="h"),
            "OT": np.random.randn(100) * 10 + 50,
        }
    )
    df["group"] = 10

    d1 = MultiSourceTSDataSet(
        dataframes=[df], group_cols=["group"], time_col="date", num_cols=["OT"], target_cols=["OT"], global_forecasting=False
    )

    sample = d1[0]

    # Target should be in features (x)
    assert "OT" in d1.metadata["feature_cols"], "OT should be in feature_cols"

    # x should contain the target column
    n_features = sample["x"].shape[1]
    assert n_features >= 1, "x should have at least 1 feature (the target)"

    logger.info("✓ target in features test passed")


def test_d1_sample_structure_strict():
    """Test D1 sample has expected strict structure."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2025-01-01", periods=100, freq="h"),
            "OT": np.random.randn(100) * 10 + 50,
        }
    )
    df["group"] = 10
    df.loc[df.shape[0] // 2 :, "group"] = 17

    d1 = MultiSourceTSDataSet(
        dataframes=[df], group_cols=["group"], time_col="date", num_cols=["OT"], target_cols=["OT"], global_forecasting=False
    )

    sample = d1[0]

    # Check required keys
    required_keys = ["x", "y", "group_id"]
    for key in required_keys:
        assert key in sample, f"Sample should contain '{key}'"

    # Verify tensor types and shapes
    import torch

    assert isinstance(sample["x"], torch.Tensor), "x should be a tensor"
    assert isinstance(sample["y"], torch.Tensor), "y should be a tensor"
    assert sample["x"].dim() == 2, "x should be 2D (timesteps, features)"
    assert sample["y"].dim() == 2, "y should be 2D (timesteps, targets)"

    # Verify group_id
    assert isinstance(sample["group_id"], int), "group_id should be an integer"
    assert sample["group_id"] in [0, 1], "group_id should be 0 or 1"

    logger.info("✓ D1 sample structure strict test passed")


class TestTemporalCardinalityComputation:
    """Test temporal cardinality computation in both memory_efficient modes."""

    def test_temporal_cardinality_memory_efficient_false(self, temp_dir):
        """Test temporal cardinality computation with memory_efficient=False."""
        np.random.seed(42)
        # Create 7 days of hourly data to get all 7 days of week
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=168, freq="h"),
                "group": ["A"] * 84 + ["B"] * 84,
                "station": ["S1", "S2"] * 84,
                "val": np.random.randn(168),
                "target": np.random.randn(168),
            }
        )
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="timestamp",
            group_cols=["group"],
            cat_cols=["station"],
            num_cols=["val"],
            target_cols=["target"],
            past_cols=["station"],
            future_cols=["station"],
            enrich_cat=["hour", "dow"],
            memory_efficient=False,
        )

        # Check temporal cardinalities
        cat_past_list = d1.metadata["cat_past_list"]
        cat_past_cardinalities = d1.metadata["cat_past_cardinalities"]

        assert "hour" in cat_past_list, "hour should be in cat_past_list"
        assert "dow" in cat_past_list, "dow should be in cat_past_list"

        hour_idx = cat_past_list.index("hour")
        dow_idx = cat_past_list.index("dow")

        assert cat_past_cardinalities[hour_idx] == 24, (
            f"hour cardinality should be 24 (memory_efficient=False), " f"got {cat_past_cardinalities[hour_idx]}"
        )
        assert cat_past_cardinalities[dow_idx] == 7, (
            f"dow cardinality should be 7 (memory_efficient=False), " f"got {cat_past_cardinalities[dow_idx]}"
        )

        logger.info("✓ Temporal cardinality computation (memory_efficient=False) passed")

    def test_temporal_cardinality_memory_efficient_true(self, temp_dir):
        """Test temporal cardinality computation with memory_efficient=True."""
        np.random.seed(42)
        # Create 7 days of hourly data to get all 7 days of week
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=168, freq="h"),
                "group": ["A"] * 84 + ["B"] * 84,
                "station": ["S1", "S2"] * 84,
                "val": np.random.randn(168),
                "target": np.random.randn(168),
            }
        )
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="timestamp",
            group_cols=["group"],
            cat_cols=["station"],
            num_cols=["val"],
            target_cols=["target"],
            past_cols=["station"],
            future_cols=["station"],
            enrich_cat=["hour", "dow"],
            memory_efficient=True,
        )

        # Check temporal cardinalities
        cat_past_list = d1.metadata["cat_past_list"]
        cat_past_cardinalities = d1.metadata["cat_past_cardinalities"]

        assert "hour" in cat_past_list, "hour should be in cat_past_list"
        assert "dow" in cat_past_list, "dow should be in cat_past_list"

        hour_idx = cat_past_list.index("hour")
        dow_idx = cat_past_list.index("dow")

        assert cat_past_cardinalities[hour_idx] == 24, (
            f"hour cardinality should be 24 (memory_efficient=True), " f"got {cat_past_cardinalities[hour_idx]}"
        )
        assert cat_past_cardinalities[dow_idx] == 7, (
            f"dow cardinality should be 7 (memory_efficient=True), " f"got {cat_past_cardinalities[dow_idx]}"
        )

        logger.info("✓ Temporal cardinality computation (memory_efficient=True) passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
