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
        assert all(k in first_window for k in ["group_idx", "start_idx", "past_len", "future_len"])
        logger.info("✓ Window boundaries")


class TestD2TargetScaling:
    """Test target scaling functionality."""

    def test_target_scaling_enabled(self, d1_basic):
        d2 = EncoderDecoder(d1_basic, past_len=24, future_len=12, scaling_method="standard", scale_targets=True)
        train, val, test = d2.split_data()

        # Check that target scaler exists
        assert d2._scaler.target_scaler is not None

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
