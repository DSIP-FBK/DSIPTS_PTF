#!/usr/bin/env python3
"""
MAIN TEST FILE
Test script for D1/D2 layers with the exact weather dataset configuration.

This script tests the D1/D2 layers with the specific configuration provided:
- global_forecasting=False
- No group_cols specified
- Enriched with 'minute'
"""

import logging
import os

import numpy as np
import pandas as pd
import torch

from dsipts.data_structure.d1_layers.multi_source_csv import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers.encoder_decoder import EncoderDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("WEATHER_CONFIG_TEST")


def create_synthetic_weather_data(output_path: str, seq_length: int = 100):
    """Create synthetic weather data for testing."""
    # Create a dataframe with a datetime column
    data = []

    # Create a base datetime
    base_date = pd.Timestamp("2023-01-01")

    for i in range(seq_length):
        # Create a timestamp with hourly frequency
        timestamp = base_date + pd.Timedelta(hours=i)

        # Generate synthetic weather data
        temperature = 20 + 10 * np.sin(i / 10) + np.random.normal(0, 2)

        data.append(
            {
                "date": timestamp,
                "OT": temperature,  # Outdoor Temperature
            }
        )

    # Create dataframe
    df = pd.DataFrame(data)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Created synthetic weather data with {seq_length} timestamps")
    logger.info(f"Data saved to {output_path}")

    return df


def test_weather_exact_config():
    """Test D1/D2 layers with the exact weather dataset configuration."""
    # Create synthetic weather data
    data_path = os.path.join(os.path.dirname(__file__), "synthetic_weather_exact.csv")
    create_synthetic_weather_data(data_path)

    # D1 configuration with the exact weather dataset configuration
    d1_conf = dict(
        file_paths=[data_path],
        group_cols=[],  # No group columns
        time_col="date",
        target_cols=["OT"],
        cat_cols=None,
        num_cols=["OT"],
        known_cols=None,
        unknown_cols=None,
        enrich_cat=["minute", "hour"],
        weights=None,
        memory_efficient=False,
        chunk_size=10000,
        global_forecasting=False,
    )

    # Initialize D1 layer
    logger.info("Initializing D1 layer with exact weather configuration")
    d1_dataset = MultiSourceTSDataSet(**d1_conf)

    # Check if enriched values are in categorical columns
    logger.info(f"D1 categorical columns: {d1_dataset.cat_cols}")
    assert "minute" in d1_dataset.cat_cols, "minute should be in categorical columns"

    # Initialize D2 layer
    logger.info("Initializing D2 layer")
    d2_dataset = EncoderDecoder(
        d1_dataset=d1_dataset,
        past_len=24,
        future_len=12,
        batch_size=16,
        step_size=1,
        split_method="percentage",
        split_config=(0.7, 0.15, 0.15),
    )

    # Get a sample from D2
    logger.info("Getting a sample from D2 layer")
    sample_idx = 0
    x, y = d2_dataset.dataset[sample_idx]

    # Check for temporal enrichment columns as separate keys
    logger.info(f"Keys in input dictionary: {x.keys()}")
    logger.info(f"x_cat_past shape: {x['x_cat_past'].shape}")

    # Check for temporal enrichment columns as separate keys
    temporal_features = ["hour", "dow", "month", "year", "minute"]
    temporal_keys = [key for key in x.keys() if key in temporal_features]
    logger.info(f"Temporal enrichment keys: {temporal_keys}")

    # Verify each temporal feature is exposed
    for feature in d1_dataset.metadata.get("enrich_cat", []):
        assert feature in x, f"Expected temporal feature {feature} to be exposed directly"
        logger.info(f"Verified temporal feature '{feature}' is exposed directly")

    # Log the keys in the input dictionary
    logger.info(f"Keys in input dictionary: {x.keys()}")

    # Check if x_cat_past exists and contains categorical features
    assert "x_cat_past" in x, "x_cat_past should be in the input dictionary"
    logger.info(f"x_cat_past shape: {tuple(x['x_cat_past'].shape)}")

    # Get the categorical indices from D1 metadata
    # meta = getattr(d1_dataset, "metadata", {}) or {}
    # idx_categorical = list(meta.get("idx_categorical", []))
    # feature_cols = meta.get("feature_cols", [])

    # Log the categorical indices and feature columns
    logger.info(f"Categorical indices: {d1_dataset.metadata.get('idx_categorical')}")
    logger.info(f"Feature columns: {d1_dataset.metadata.get('feature_cols')}")
    logger.info(f"Enriched categorical features: {d1_dataset.metadata.get('enrich_cat')}")
    logger.info(f"Group ID type: {type(x['group_id'])}")
    logger.info(f"Group ID value: {x['group_id']}")

    # Get a batch from the dataloader to check batch structure
    train_loader = d2_dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Log the batch structure
    logger.info("Batch structure from dataloader:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: tensor of shape {tuple(value.shape)}, dtype={value.dtype}")
        else:
            logger.info(f"  {key}: {type(value)}")

    # Verify temporal features in batch structure
    for feature in d1_dataset.metadata.get("enrich_cat", []):
        assert feature in batch, f"Expected temporal feature {feature} to be in batch"
        assert isinstance(batch[feature], torch.Tensor), f"Expected {feature} to be a tensor"
        assert (
            batch[feature].shape[0] == batch["x_num_past"].shape[0]
        ), f"Expected {feature} to have same batch size as x_num_past"
        assert (
            batch[feature].shape[1] == d2_dataset.past_len
        ), f"Expected {feature} to have length equal to past_len"
        logger.info(
            f"Verified temporal feature '{feature}' in batch structure"
            f"with shape {tuple(batch[feature].shape)}"
        )

    # Check if x_cat_past exists in the batch
    assert "x_cat_past" in batch, "x_cat_past should be in the batch"

    logger.info("Weather exact configuration test completed successfully")
    return x, y, batch


if __name__ == "__main__":
    try:
        x, y, batch = test_weather_exact_config()
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
