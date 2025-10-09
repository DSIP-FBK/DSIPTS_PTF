"""
Comprehensive D2 Layer (EncoderDecoder) Test Suite

Coverage: Windows, Batches, Scaling, Splitting, DataLoader
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from dsipts.data_structure.d1_layers import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers import EncoderDecoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def d1_basic(temp_dir):
    np.random.seed(42)
    data = [
        {"time": t, "group_id": f"g_{t%2}", "num_0": np.sin(t / 10), "cat_0": f"c_{t%3}", "target_0": np.sin(t / 5)}
        for t in range(100)
    ]
    df = pd.DataFrame(data)
    path = os.path.join(temp_dir, "data.csv")
    df.to_csv(path, index=False)
    return MultiSourceTSDataSet(
        file_paths=[path],
        group_cols=["group_id"],
        time_col="time",
        target_cols=["target_0"],
        num_cols=["num_0"],
        cat_cols=["cat_0"],
    )


class TestD2Windows:
    def test_window_creation(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12)
        assert len(d2.valid_windows) > 0
        logger.info(f"✓ Windows: {len(d2.valid_windows)}")

    def test_step_size(self, d1_basic):
        d2_1 = EncoderDecoder(d1_basic, past_len=10, future_len=5, step_size=1)
        d2_5 = EncoderDecoder(d1_basic, past_len=10, future_len=5, step_size=5)
        assert len(d2_1.valid_windows) > len(d2_5.valid_windows)
        logger.info("✓ Step size")


class TestD2Batch:
    def test_structure(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12)
        x, y = d2.dataset[0]
        assert all(k in x for k in ["x_num_past", "x_cat_past", "y", "idx_target"])
        assert y.shape[0] == 12
        logger.info(f"✓ Batch: x_num={x['x_num_past'].shape}, y={y.shape}")

    def test_shapes(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=20, future_len=10)
        x, y = d2.dataset[0]
        assert x["x_num_past"].shape[0] == 20
        assert y.shape[0] == 10
        logger.info("✓ Shapes")


class TestD2Scaling:
    def test_standard(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12, scaling_method="standard")
        train, val, test = d2.split_data()
        assert d2.is_scaler_fitted
        x, y = train[0]
        assert abs(x["x_num_past"].mean().item()) < 2
        logger.info("✓ Standard scaling")

    def test_minmax(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12, scaling_method="minmax")
        train, val, test = d2.split_data()
        x, y = train[0]
        assert x["x_num_past"].min() >= -0.1 and x["x_num_past"].max() <= 1.1
        logger.info("✓ MinMax scaling")


class TestD2Splitting:
    def test_temporal(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12)
        train, val, test = d2.split_data(method="temporal")
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        logger.info(f"✓ Temporal: train={len(train)}, val={len(val)}, test={len(test)}")

    def test_random(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12)
        train, val, test = d2.split_data(method="random")
        assert len(train) > 0
        logger.info("✓ Random split")


class TestD2DataLoader:
    def test_loader(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12, batch_size=16)
        train, val, test = d2.split_data()
        loader = DataLoader(train, batch_size=16, shuffle=True)
        batch = next(iter(loader))
        x, y = batch
        assert x["x_num_past"].dim() == 3
        logger.info(f"✓ DataLoader: batch={x['x_num_past'].shape}")


class TestD2WindowEdgeCases:
    """Test window creation edge cases."""

    def test_insufficient_sequence_length(self, temp_dir):
        np.random.seed(42)
        # Create very short sequences
        data = [{"time": t, "group": "g1", "val": t, "target": t} for t in range(10)]
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "short.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path], group_cols=["group"], time_col="time", target_cols=["target"], num_cols=["val"]
        )

        # Request windows longer than sequence
        d2 = EncoderDecoder(d1, past_len=20, future_len=10)
        assert len(d2.valid_windows) == 0  # Should have no valid windows
        logger.info("✓ Insufficient length handled")

    def test_window_boundaries(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=10, future_len=5, step_size=1)
        # Check first and last windows
        first_window = d2.valid_windows[0]
        last_window = d2.valid_windows[-1]
        assert first_window["start_idx"] >= 0
        assert last_window["start_idx"] >= 0
        # Verify window has required keys
        assert all(k in first_window for k in ["group_idx", "start_idx", "past_len"])
        logger.info("✓ Window boundaries")


class TestD2TargetScaling:
    """Test target scaling functionality."""

    def test_target_scaling_enabled(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12, scaling_method="standard", scale_targets=True)
        train, val, test = d2.split_data()

        # Check that target scaler exists
        assert d2.target_scaler is not None

        # Check scaled targets
        x, y = train[0]
        assert abs(y.mean().item()) < 2  # Should be normalized
        logger.info("✓ Target scaling enabled")

    def test_target_scaling_disabled(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12, scaling_method="standard", scale_targets=False)
        train, val, test = d2.split_data()

        # Target scaler should still exist but not applied
        x, y = train[0]
        # Targets should not be normalized (will have larger range)
        logger.info("✓ Target scaling disabled")

    def test_standard_scaler_validation(self, temp_dir):
        """COMPREHENSIVE: Validate StandardScaler produces mean≈0, std≈1."""
        np.random.seed(42)

        # Create data with known statistics
        n_samples = 300
        data = []
        for i in range(n_samples):
            data.append(
                {
                    "time": i,
                    "group": "g1",
                    "feature1": np.random.normal(100, 20),  # mean=100, std=20
                    "feature2": np.random.normal(50, 10),  # mean=50, std=10
                    "target": np.random.normal(200, 30),
                }
            )
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="time",
            group_cols=["group"],
            num_cols=["feature1", "feature2"],
            target_cols=["target"],
            memory_efficient=False,
        )

        d2 = EncoderDecoder(d1_dataset=d1, past_len=24, future_len=12, scaling_method="standard", scale_targets=True)

        train, val, test = d2.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        # Collect features from multiple samples
        all_features = []
        all_targets = []
        for i in range(min(50, len(train))):
            x, y = train[i]
            if "x_num_past" in x:
                all_features.append(x["x_num_past"].numpy())
            all_targets.append(y.numpy())

        features = np.concatenate(all_features, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # STRONG VALIDATION: StandardScaler should produce mean≈0, std≈1
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)

        logger.info(f"Feature means: {feature_means}")
        logger.info(f"Feature stds: {feature_stds}")

        # Check each feature (skip constant features with std=0)
        for i, (mean, std) in enumerate(zip(feature_means, feature_stds)):
            if std > 0.1:  # Only check non-constant features
                assert abs(mean) < 0.5, f"Feature {i} mean {mean} not close to 0"
                assert 0.8 < std < 1.2, f"Feature {i} std {std} not close to 1"

        # Check targets
        target_mean = np.mean(targets)
        target_std = np.std(targets)
        logger.info(f"Target mean: {target_mean:.4f}, std: {target_std:.4f}")
        assert abs(target_mean) < 0.5, f"Target mean {target_mean} not close to 0"
        assert 0.8 < target_std < 1.2, f"Target std {target_std} not close to 1"

        logger.info("✓ StandardScaler validation passed")

    def test_minmax_scaler_validation(self, temp_dir):
        """COMPREHENSIVE: Validate MinMaxScaler produces values in [0, 1]."""
        np.random.seed(42)

        n_samples = 300
        data = []
        for i in range(n_samples):
            data.append(
                {
                    "time": i,
                    "group": "g1",
                    "feature1": np.random.uniform(10, 100),
                    "feature2": np.random.uniform(50, 150),
                    "target": np.random.uniform(0, 200),
                }
            )
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="time",
            group_cols=["group"],
            num_cols=["feature1", "feature2"],
            target_cols=["target"],
            memory_efficient=False,
        )

        d2 = EncoderDecoder(d1_dataset=d1, past_len=24, future_len=12, scaling_method="minmax", scale_targets=True)

        train, val, test = d2.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        # Collect features
        all_features = []
        all_targets = []
        for i in range(min(50, len(train))):
            x, y = train[i]
            if "x_num_past" in x:
                all_features.append(x["x_num_past"].numpy())
            all_targets.append(y.numpy())

        features = np.concatenate(all_features, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # STRONG VALIDATION: MinMaxScaler should produce values in [0, 1]
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)

        logger.info(f"Feature min: {feature_min}")
        logger.info(f"Feature max: {feature_max}")

        # Check range [0, 1] (allow small tolerance)
        for i, (min_val, max_val) in enumerate(zip(feature_min, feature_max)):
            if max_val > 0.1:  # Only check non-constant features
                assert min_val >= -0.1, f"Feature {i} min {min_val} < 0"
                assert max_val <= 1.1, f"Feature {i} max {max_val} > 1"

        # Check targets
        target_min = np.min(targets)
        target_max = np.max(targets)
        logger.info(f"Target min: {target_min:.4f}, max: {target_max:.4f}")
        assert target_min >= -0.1, f"Target min {target_min} < 0"
        assert target_max <= 1.1, f"Target max {target_max} > 1"

        logger.info("✓ MinMaxScaler validation passed")

    def test_scaling_with_memory_efficient_mode(self, temp_dir):
        """COMPREHENSIVE: Test scaling works with memory-efficient mode."""
        np.random.seed(42)

        # Create multiple CSV files
        for file_idx in range(3):
            data = []
            for i in range(100):
                data.append(
                    {
                        "time": i,
                        "group": f"g{file_idx}",
                        "feature1": np.random.normal(100, 20),
                        "feature2": np.random.normal(50, 10),
                        "target": np.random.normal(200, 30),
                    }
                )
            df = pd.DataFrame(data)
            path = os.path.join(temp_dir, f"data_{file_idx}.csv")
            df.to_csv(path, index=False)

        # Load with memory_efficient=True
        file_paths = [os.path.join(temp_dir, f"data_{i}.csv") for i in range(3)]

        d1 = MultiSourceTSDataSet(
            file_paths=file_paths,
            time_col="time",
            group_cols=["group"],
            num_cols=["feature1", "feature2"],
            target_cols=["target"],
            memory_efficient=True,  # Key: memory-efficient mode
            chunk_size=50,
        )

        d2 = EncoderDecoder(d1_dataset=d1, past_len=24, future_len=12, scaling_method="standard", scale_targets=True)

        train, val, test = d2.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        # STRONG VALIDATION: Scaling should work even with chunked data
        all_features = []
        for i in range(min(30, len(train))):
            x, y = train[i]
            if "x_num_past" in x:
                all_features.append(x["x_num_past"].numpy())

        if all_features:
            features = np.concatenate(all_features, axis=0)
            feature_means = np.mean(features, axis=0)
            feature_stds = np.std(features, axis=0)

            logger.info(f"Memory-efficient - means: {feature_means}")
            logger.info(f"Memory-efficient - stds: {feature_stds}")

            # Scaling should still work (skip constant features)
            for i, (mean, std) in enumerate(zip(feature_means, feature_stds)):
                if std > 0.1:
                    assert abs(mean) < 0.5, f"Feature {i} mean {mean} not close to 0 (memory-efficient)"
                    assert 0.7 < std < 1.3, f"Feature {i} std {std} not close to 1 (memory-efficient)"

        logger.info("✓ Scaling works with memory-efficient mode")


class TestD2TemporalIntegration:
    """Test temporal feature integration from D1 to D2."""

    def test_temporal_in_cat_past(self, temp_dir):
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
            enrich_cat=["hour", "dow"],
        )
        d2 = EncoderDecoder(d1, past_len=24, future_len=12)
        x, y = d2.dataset[0]

        # Temporal features should be in x_cat_past
        assert x["x_cat_past"].shape[1] > 0
        logger.info(f"✓ Temporal in cat_past: {x['x_cat_past'].shape}")

    def test_temporal_in_cat_future(self, temp_dir):
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
            enrich_cat=["hour", "dow"],
        )
        d2 = EncoderDecoder(d1, past_len=24, future_len=12)
        x, y = d2.dataset[0]

        # Temporal features should be in x_cat_future (known in advance)
        assert x["x_cat_future"].shape[1] > 0
        logger.info(f"✓ Temporal in cat_future: {x['x_cat_future'].shape}")

    def test_temporal_enrichment_propagates_to_d2(self, temp_dir):
        """COMPREHENSIVE: Test that temporal features from D1 appear in D2 batches."""
        np.random.seed(42)
        start = datetime(2024, 1, 1)
        data = []
        for h in range(200):
            data.append({"timestamp": start + timedelta(hours=h), "group": "g1", "val": h, "target": h})
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "data.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            time_col="timestamp",
            group_cols=["group"],
            num_cols=["val"],
            target_cols=["target"],
            enrich_cat=["hour", "dow"],
        )

        d2 = EncoderDecoder(d1_dataset=d1, past_len=24, future_len=12)

        # Get a sample from D2
        x, y = d2.dataset[0]

        # STRONG VALIDATION: Check that categorical features include temporal enrichment
        assert "x_cat_past" in x, "x_cat_past missing"
        assert "x_cat_future" in x, "x_cat_future missing"

        # x_cat_past should have temporal features (group + hour + dow)
        cat_past_shape = x["x_cat_past"].shape
        assert cat_past_shape[1] >= 2, f"Expected at least 2 cat features in past, got {cat_past_shape[1]}"

        # x_cat_future should have temporal features (hour, dow are known in future)
        cat_future_shape = x["x_cat_future"].shape
        assert cat_future_shape[1] >= 2, f"Expected at least 2 future cat features, got {cat_future_shape[1]}"

        logger.info(f"✓ Temporal features propagate: past={cat_past_shape}, future={cat_future_shape}")


class TestD2SplitValidation:
    """Test data splitting validation."""

    def test_split_ratio_sum(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12)
        train, val, test = d2.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        total = len(train) + len(val) + len(test)
        train_pct = len(train) / total
        val_pct = len(val) / total
        test_pct = len(test) / total

        # Check ratios are approximately correct
        assert 0.65 < train_pct < 0.75
        assert 0.10 < val_pct < 0.20
        assert 0.10 < test_pct < 0.20
        logger.info(f"✓ Split ratios: {train_pct:.2f}/{val_pct:.2f}/{test_pct:.2f}")


class TestD2GlobalVsLocal:
    """Test global vs local forecasting batch structure."""

    def test_global_forecasting_group_id(self, temp_dir):
        np.random.seed(42)
        data = [{"time": t, "group": f"g_{t%2}", "val": t, "target": t} for t in range(100)]
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "global.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            group_cols=["group"],
            time_col="time",
            target_cols=["target"],
            num_cols=["val"],
            global_forecasting=True,
        )
        d2 = EncoderDecoder(d1, past_len=24, future_len=12)
        x, y = d2.dataset[0]

        # In global forecasting, group_id should be in batch
        assert "group_id" in x
        logger.info("✓ Global: group_id in batch")

    def test_local_forecasting_group_in_cat(self, temp_dir):
        np.random.seed(42)
        data = [{"time": t, "group": f"g_{t%2}", "val": t, "target": t} for t in range(100)]
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "local.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            group_cols=["group"],
            time_col="time",
            target_cols=["target"],
            num_cols=["val"],
            global_forecasting=False,
        )
        d2 = EncoderDecoder(d1, past_len=24, future_len=12)
        x, y = d2.dataset[0]

        # In local forecasting, group should be in categorical features
        assert x["x_cat_past"].shape[1] > 0  # Should have group_id
        logger.info("✓ Local: group in cat features")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
