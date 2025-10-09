"""
Comprehensive D1-D2 Integration Test Suite

Tests end-to-end pipeline functionality including:
- Complete D1 → D2 data flow
- Scaling integration and isolation
- Temporal enrichment propagation
- Global vs local forecasting
- Memory efficiency
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch
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
def realistic_weather_data(temp_dir):
    """Create realistic weather dataset."""
    np.random.seed(42)
    start = datetime(2024, 1, 1)
    data = []
    for g in range(3):  # 3 weather stations
        for h in range(24 * 30):  # 30 days hourly
            timestamp = start + timedelta(hours=h)
            data.append(
                {
                    "timestamp": timestamp,
                    "station_id": f"station_{g}",
                    "temperature": 20 + 10 * np.sin(2 * np.pi * h / 24) + g * 5 + np.random.normal(0, 2),
                    "humidity": 60 + 20 * np.cos(2 * np.pi * h / 24) + np.random.normal(0, 5),
                    "pressure": 1013 + np.random.normal(0, 3),
                    "weather": np.random.choice(["sunny", "cloudy", "rainy"]),
                }
            )
    df = pd.DataFrame(data)
    path = os.path.join(temp_dir, "weather.csv")
    df.to_csv(path, index=False)
    return path


class TestD1D2Pipeline:
    """Test complete D1 → D2 pipeline."""

    def test_basic_pipeline(self, realistic_weather_data):
        # D1: Load data
        d1 = MultiSourceTSDataSet(
            file_paths=[realistic_weather_data],
            group_cols=["station_id"],
            time_col="timestamp",
            target_cols=["temperature"],
            num_cols=["humidity", "pressure"],
            cat_cols=["weather"],
            enrich_cat=["hour", "dow"],
        )

        # D2: Create windows
        d2 = EncoderDecoder(d1, past_len=24, future_len=12, batch_size=32)

        # Split data
        train, val, test = d2.split_data(method="temporal")

        # Verify pipeline
        assert len(d1) == 3  # 3 stations
        assert len(d2.valid_windows) > 0
        assert len(train) > 0

        # Get sample
        x, y = train[0]
        # Note: May have 3 features if group_id is included in local forecasting
        assert x["x_num_past"].shape[0] == 24  # 24 timesteps
        assert x["x_num_past"].shape[1] >= 2  # At least 2 numerical features
        assert y.shape == (12, 1)  # 1 target

        logger.info(f"✓ Pipeline: D1({len(d1)} groups) → D2({len(d2.valid_windows)} windows) → Train({len(train)})")

    def test_pipeline_with_scaling(self, realistic_weather_data):
        d1 = MultiSourceTSDataSet(
            file_paths=[realistic_weather_data],
            group_cols=["station_id"],
            time_col="timestamp",
            target_cols=["temperature"],
            num_cols=["humidity", "pressure"],
            enrich_cat=["hour"],
        )

        d2 = EncoderDecoder(d1, past_len=24, future_len=12, scaling_method="standard", scale_targets=True)
        train, val, test = d2.split_data()

        # Verify scaling
        assert d2.is_scaler_fitted
        x, y = train[0]

        # Check normalized
        assert abs(x["x_num_past"].mean().item()) < 2
        assert abs(y.mean().item()) < 2

        logger.info("✓ Pipeline with scaling")

    def test_dataloader_integration(self, realistic_weather_data):
        d1 = MultiSourceTSDataSet(
            file_paths=[realistic_weather_data],
            group_cols=["station_id"],
            time_col="timestamp",
            target_cols=["temperature"],
            num_cols=["humidity", "pressure"],
        )

        d2 = EncoderDecoder(d1, past_len=24, future_len=12, batch_size=32)
        train, val, test = d2.split_data()

        # Create DataLoader
        loader = DataLoader(train, batch_size=32, shuffle=True)
        batch = next(iter(loader))
        x, y = batch

        # Verify batched structure
        assert x["x_num_past"].dim() == 3  # [batch, seq, features]
        assert y.dim() == 3  # [batch, seq, targets]

        logger.info(f"✓ DataLoader: batch_shape={x['x_num_past'].shape}")


class TestD1D2Scaling:
    """Test scaling integration and isolation."""

    def test_scaling_isolation(self, realistic_weather_data):
        # Create D1 without scaling
        d1 = MultiSourceTSDataSet(
            file_paths=[realistic_weather_data],
            group_cols=["station_id"],
            time_col="timestamp",
            target_cols=["temperature"],
            num_cols=["humidity", "pressure"],
        )

        # Get raw D1 sample
        d1_sample = d1[0]
        d1_mean = d1_sample["x"][:, 0].mean().item()

        # Create D2 with scaling
        d2 = EncoderDecoder(d1, past_len=24, future_len=12, scaling_method="standard")
        train, val, test = d2.split_data()

        # Get D1 sample again (should be unchanged)
        d1_sample_after = d1[0]
        d1_mean_after = d1_sample_after["x"][:, 0].mean().item()

        # D1 data should not be affected by D2 scaling
        assert abs(d1_mean - d1_mean_after) < 0.1  # Allow small numerical differences

        # D2 data should be scaled (closer to 0)
        x, y = train[0]
        d2_mean = x["x_num_past"][:, 0].mean().item()
        # D2 should be more normalized (closer to 0 for standard scaling)
        assert abs(d2_mean) < 2  # Standard scaled should be close to 0

        logger.info(f"✓ Scaling isolation: D1_mean={d1_mean:.2f}, D2_mean={d2_mean:.2f}")

    def test_inverse_scaling_accuracy(self, realistic_weather_data):
        d1 = MultiSourceTSDataSet(
            file_paths=[realistic_weather_data],
            group_cols=["station_id"],
            time_col="timestamp",
            target_cols=["temperature"],
            num_cols=["humidity", "pressure"],
        )

        d2 = EncoderDecoder(d1, past_len=24, future_len=12, scaling_method="standard")
        train, val, test = d2.split_data()

        x, y = train[0]
        scaled = x["x_num_past"]

        # Apply inverse scaling
        inverse = d2.apply_inverse_scaling(scaled, data_type="features")

        # Re-scale
        rescaled = d2.feature_scaler.transform(inverse.numpy())
        rescaled_tensor = torch.from_numpy(rescaled).float()

        # Should match original scaled values
        diff = (scaled - rescaled_tensor).abs().max().item()
        assert diff < 0.01

        logger.info(f"✓ Inverse scaling accuracy: max_diff={diff:.6f}")


class TestD1D2MemoryEfficiency:
    """Test memory-efficient mode integration."""

    def test_memory_efficient_d1_with_d2(self, realistic_weather_data):
        d1 = MultiSourceTSDataSet(
            file_paths=[realistic_weather_data],
            group_cols=["station_id"],
            time_col="timestamp",
            target_cols=["temperature"],
            num_cols=["humidity", "pressure"],
            memory_efficient=True,
        )

        d2 = EncoderDecoder(d1, past_len=24, future_len=12)
        train, val, test = d2.split_data()

        # Should work with memory-efficient D1
        x, y = train[0]
        assert x["x_num_past"].shape[0] == 24

        logger.info("✓ Memory-efficient D1 with D2")


class TestD1D2TemporalFlow:
    """Test temporal enrichment flow from D1 to D2."""

    def test_temporal_enrichment_propagation(self, realistic_weather_data):
        d1 = MultiSourceTSDataSet(
            file_paths=[realistic_weather_data],
            group_cols=["station_id"],
            time_col="timestamp",
            target_cols=["temperature"],
            num_cols=["humidity"],
            enrich_cat=["hour", "dow", "month"],
        )

        # Check D1 has temporal features
        d1_sample = d1[0]
        assert "hour" in d1_sample["cat_cols"]
        assert "dow" in d1_sample["cat_cols"]
        assert "month" in d1_sample["cat_cols"]

        # Check D2 propagates temporal features
        d2 = EncoderDecoder(d1, past_len=24, future_len=12)
        x, y = d2.dataset[0]

        # Temporal features should be in both past and future
        assert x["x_cat_past"].shape[1] >= 3  # hour, dow, month (+ possibly group)
        assert x["x_cat_future"].shape[1] >= 3  # Temporal features known in advance

        logger.info(f"✓ Temporal flow: cat_past={x['x_cat_past'].shape}, cat_future={x['x_cat_future'].shape}")


class TestD1D2MultiTarget:
    """Test multiple target columns."""

    def test_multiple_targets(self, temp_dir):
        np.random.seed(42)
        data = [{"time": t, "group": "g1", "val": t, "target_0": t, "target_1": t * 2, "target_2": t * 3} for t in range(100)]
        df = pd.DataFrame(data)
        path = os.path.join(temp_dir, "multi_target.csv")
        df.to_csv(path, index=False)

        d1 = MultiSourceTSDataSet(
            file_paths=[path],
            group_cols=["group"],
            time_col="time",
            target_cols=["target_0", "target_1", "target_2"],
            num_cols=["val"],
        )

        d2 = EncoderDecoder(d1, past_len=24, future_len=12)
        x, y = d2.dataset[0]

        assert y.shape[1] == 3  # 3 targets
        logger.info(f"✓ Multiple targets: y={y.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
