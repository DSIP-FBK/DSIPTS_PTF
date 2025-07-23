"""
Encoder-Decoder implementation for D2 layer.

This module provides the EncoderDecoder class (formerly TSDataModule) that creates
sliding windows and encoder-decoder structures from D1 layer data.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Sampler

from ..d1_layers.base_d1 import BaseD1Layer

logger = logging.getLogger(__name__)


class TimeSeriesSubset:
    """Minimal subset class for dataset splits."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class EncoderDecoder(pl.LightningDataModule):
    """
    D2 layer for time series data that creates encoder-decoder structures.

    This class (formerly TSDataModule) extends PyTorch Dataset and creates sliding windows
    from D1 layer data, formatting them for encoder-decoder models like those in
    PyTorch Forecasting.
    """

    def __init__(
        self,
        d1_dataset: BaseD1Layer,
        past_len: int,
        future_len: int,
        batch_size: int = 32,
        step_size: int = 1,
        min_valid_length: Optional[int] = None,
        split_method: str = "percentage",
        split_config: Optional[tuple] = None,
        num_workers: int = 0,
        sampler: Optional[Sampler] = None,
        known_cols: Optional[List[str]] = None,
        unknown_cols: Optional[List[str]] = None,
        target_normalizer: Optional[str] = None,
        categorical_encoders: Optional[Dict] = None,
        max_samples_per_group: Optional[int] = None,
        precompute: bool = True,
    ):
        """
        Initialize the EncoderDecoder.

        Args:
            d1_dataset: Any D1 layer implementation (BaseD1Layer subclass)
            past_len: Length of the past sequence (encoder)
            future_len: Length of the future sequence (decoder)
            step_size: Step size for sliding window
            known_cols: Known columns at prediction time (extracted from d1_dataset if None)
            unknown_cols: Unknown columns at prediction time (extracted from d1_dataset if None)
            target_normalizer: Type of target normalization
            categorical_encoders: Categorical encoders
            max_samples_per_group: Maximum samples per group
        """
        super().__init__()

        self.d1_dataset = d1_dataset
        self.past_len = past_len
        self.future_len = future_len
        self.batch_size = batch_size
        self.step_size = step_size
        self.min_valid_length = min_valid_length or past_len
        self.split_method = split_method
        self.split_config = split_config
        self.num_workers = num_workers
        self.sampler = sampler
        self.target_normalizer = target_normalizer
        self.categorical_encoders = categorical_encoders or {}
        self.max_samples_per_group = max_samples_per_group
        self.precompute = precompute

        # Extract column information from D1 dataset
        self.known_cols = known_cols or d1_dataset.known_cols
        self.unknown_cols = unknown_cols or d1_dataset.unknown_cols
        self.group_cols = d1_dataset.group_cols
        self.target_cols = d1_dataset.target_cols
        self.feature_cols = d1_dataset.feature_cols
        self.cat_cols = d1_dataset.cat_cols or []

        # Separate categorical and continuous columns
        all_feature_cols = self.feature_cols + self.target_cols
        self.cat_feature_cols = [col for col in all_feature_cols if col in self.cat_cols]
        self.cont_feature_cols = [col for col in all_feature_cols if col not in self.cat_cols]

        # Build valid windows
        self._build_valid_windows()

        logger.info(f"EncoderDecoder initialized with {len(self.valid_windows)} valid windows")

        # Create datasets if precompute is True
        if precompute:
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None

            if self.split_config:
                train_indices, val_indices, test_indices = self._create_splits(self.split_config)

                self.train_dataset = TimeSeriesSubset(self, train_indices)
                self.val_dataset = TimeSeriesSubset(self, val_indices)
                self.test_dataset = TimeSeriesSubset(self, test_indices)

                logger.info(
                    f"Split statistics: Train: {len(train_indices)}, "
                    f"Validation: {len(val_indices)}, Test: {len(test_indices)}"
                )
            else:
                # Default to all indices as training
                self.train_dataset = TimeSeriesSubset(self, list(range(len(self.valid_windows))))
                self.val_dataset = TimeSeriesSubset(self, [])
                self.test_dataset = TimeSeriesSubset(self, [])

    def _build_valid_windows(self):
        """
        Build valid sliding windows from the D1 dataset.

        This method scans through all data in the D1 dataset and identifies
        valid windows that have sufficient past and future data.
        """
        self.valid_windows = []

        # Group data by group columns
        # Since D1 dataset handles grouping internally, we need to extract group information
        group_data = {}

        # Collect all samples and group them
        for idx in range(len(self.d1_dataset)):
            sample = self.d1_dataset[idx]
            group_id = sample["group_id"]

            if group_id not in group_data:
                group_data[group_id] = []

            group_data[group_id].append((idx, sample))

        # Process each group to find valid windows
        for group_id, group_samples in group_data.items():
            # Sort by time if available
            if "past_time" in group_samples[0][1]:
                group_samples.sort(
                    key=lambda x: x[1]["past_time"] if x[1]["past_time"] is not None else 0
                )

            # Find valid windows in this group
            group_indices = [idx for idx, _ in group_samples]

            # Create sliding windows
            for i in range(
                0, len(group_indices) - self.past_len - self.future_len + 1, self.step_size
            ):
                past_indices = group_indices[i : i + self.past_len]
                future_indices = group_indices[
                    i + self.past_len : i + self.past_len + self.future_len
                ]

                # Validate window (check for sufficient non-NaN values)
                if self._is_valid_window(past_indices, future_indices):
                    self.valid_windows.append(
                        {
                            "group_id": group_id,
                            "past_indices": past_indices,
                            "future_indices": future_indices,
                            "start_idx": i,
                        }
                    )

                    # Limit samples per group if specified
                    if (
                        self.max_samples_per_group
                        and len([w for w in self.valid_windows if w["group_id"] == group_id])
                        >= self.max_samples_per_group
                    ):
                        break

    def _is_valid_window(self, past_indices: List[int], future_indices: List[int]) -> bool:
        """
        Check if a window is valid (has sufficient non-NaN data).

        Args:
            past_indices: Indices for past data
            future_indices: Indices for future data

        Returns:
            True if window is valid
        """
        # For now, assume all windows are valid
        # In a more sophisticated implementation, you might check for NaN values
        return len(past_indices) == self.past_len and len(future_indices) == self.future_len

    def __len__(self):
        """Return the number of valid windows."""
        return len(self.valid_windows)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Get a sample with encoder-decoder structure.

        Args:
            idx: Index of the window to retrieve

        Returns:
            Tuple of (input_dict, target_tensor) where input_dict contains
            encoder-decoder structure compatible with PyTorch Forecasting
        """
        window = self.valid_windows[idx]
        past_indices = window["past_indices"]
        future_indices = window["future_indices"]

        # Get past and future data
        past_samples = [self.d1_dataset[i] for i in past_indices]
        future_samples = [self.d1_dataset[i] for i in future_indices]

        # Extract features and targets
        past_features = torch.stack([sample["x"] for sample in past_samples])
        future_targets = torch.stack([sample["y"] for sample in future_samples])

        # Separate categorical and continuous features
        encoder_cont = past_features  # All features for now
        encoder_cat = torch.zeros(self.past_len, len(self.cat_feature_cols))  # Placeholder

        # Future features (known at prediction time)
        decoder_cont = torch.zeros(self.future_len, len(self.cont_feature_cols))  # Placeholder
        decoder_cat = torch.zeros(self.future_len, len(self.cat_feature_cols))  # Placeholder

        # Create time indices
        encoder_time_idx = torch.arange(self.past_len)
        decoder_time_idx = torch.arange(self.past_len, self.past_len + self.future_len)

        # Create masks (all valid for now)
        encoder_mask = torch.ones(self.past_len, dtype=torch.bool)
        decoder_mask = torch.ones(self.future_len, dtype=torch.bool)

        # Sequence lengths
        encoder_lengths = torch.tensor([self.past_len])
        decoder_lengths = torch.tensor([self.future_len])

        # Target indices (which features are targets)
        target_indices = list(range(len(self.target_cols)))
        idx_target = torch.tensor(target_indices)

        # Static features (placeholder)
        static_categorical_features = torch.zeros(len(self.cat_cols))
        static_continuous_features = torch.zeros(0)  # No static continuous features for now

        # Target scale (no scaling for now)
        target_scale = torch.ones(len(self.target_cols))

        # Build input dictionary with encoder-decoder structure
        x = {
            # Encoder data
            "encoder_cont": encoder_cont,
            "encoder_cat": encoder_cat,
            "encoder_lengths": encoder_lengths,
            "encoder_time_idx": encoder_time_idx,
            "encoder_mask": encoder_mask,
            # Decoder data
            "decoder_cont": decoder_cont,
            "decoder_cat": decoder_cat,
            "decoder_lengths": decoder_lengths,
            "decoder_time_idx": decoder_time_idx,
            "decoder_mask": decoder_mask,
            # Static features
            "static_categorical_features": static_categorical_features,
            "static_continuous_features": static_continuous_features,
            # Target information
            "idx_target": idx_target,
            "target_scale": target_scale,
            # Model-compatible keys
            "x_num_past": encoder_cont,
            "x_cat_past": encoder_cat,
            "x_num_future": decoder_cont,
            "x_cat_future": decoder_cat,
            "y": future_targets,
            # Backward compatibility keys
            "past_features": encoder_cont,
            "future_targets": future_targets,
            # Metadata
            "group_id": window["group_id"],
            "past_time": [past_samples[i]["past_time"] for i in range(len(past_samples))],
            "future_time": [future_samples[i]["future_time"] for i in range(len(future_samples))],
        }

        return x, future_targets

    def _create_splits(self, split_config):
        """
        Create train/validation/test splits based on the specified configuration.

        Args:
            split_config: Configuration for splits:
                        - For 'percentage' method: (train%, val%, test%)
                        - For 'group' method: (train_groups, val_groups, test_groups)

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if self.split_method == "percentage":
            # Percentage-based split
            train_pct, val_pct, test_pct = split_config
            total_samples = len(self)

            # Calculate indices for each split
            train_end = int(total_samples * train_pct)
            val_end = int(total_samples * (train_pct + val_pct))

            train_indices = list(range(0, train_end))
            val_indices = list(range(train_end, val_end))
            test_indices = list(range(val_end, total_samples))

            return train_indices, val_indices, test_indices

        elif self.split_method == "group":
            # Group-based split
            train_groups, val_groups, test_groups = split_config

            # Map group names to indices
            train_indices = []
            val_indices = []
            test_indices = []

            for idx, window in enumerate(self.valid_windows):
                group_id = window["group_id"]

                if group_id in train_groups:
                    train_indices.append(idx)
                elif group_id in val_groups:
                    val_indices.append(idx)
                elif group_id in test_groups:
                    test_indices.append(idx)

            return train_indices, val_indices, test_indices

        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

    def split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        method: str = "temporal",
    ) -> Tuple["TimeSeriesSubset", "TimeSeriesSubset", "TimeSeriesSubset"]:
        """
        Split the dataset into train, validation, and test sets.

        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            method: Split method ('temporal' or 'random')

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        total_samples = len(self)

        if method == "temporal":
            # Temporal split - earlier data for training, later for validation/test
            train_end = int(total_samples * train_ratio)
            val_end = int(total_samples * (train_ratio + val_ratio))

            train_indices = list(range(0, train_end))
            val_indices = list(range(train_end, val_end))
            test_indices = list(range(val_end, total_samples))
        else:
            # Random split
            indices = np.random.permutation(total_samples)
            train_end = int(total_samples * train_ratio)
            val_end = int(total_samples * (train_ratio + val_ratio))

            train_indices = indices[:train_end].tolist()
            val_indices = indices[train_end:val_end].tolist()
            test_indices = indices[val_end:].tolist()

        logger.info(
            f"Split statistics: Train: {len(train_indices)}, "
            f"Validation: {len(val_indices)}, Test: {len(test_indices)}"
        )

        return (
            TimeSeriesSubset(self, train_indices),
            TimeSeriesSubset(self, val_indices),
            TimeSeriesSubset(self, test_indices),
        )

    def setup(self, stage=None):
        """Set up datasets for training, validation, and testing."""
        if self.train_dataset is None and self.split_config:
            train_indices, val_indices, test_indices = self._create_splits(self.split_config)

            self.train_dataset = TimeSeriesSubset(self, train_indices)
            self.val_dataset = TimeSeriesSubset(self, val_indices)
            self.test_dataset = TimeSeriesSubset(self, test_indices)

            logger.info(
                f"Setup completed with split statistics: Train: {len(train_indices)}, "
                f"Validation: {len(val_indices)}, Test: {len(test_indices)}"
            )

    def train_dataloader(self):
        """Return the training dataloader."""
        from .utils import custom_collate_fn

        if self.train_dataset is None:
            # If no explicit split was provided, use all data for training
            self.train_dataset = TimeSeriesSubset(self, list(range(len(self.valid_windows))))

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            sampler=self.sampler,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        from .utils import custom_collate_fn

        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        from .utils import custom_collate_fn

        if self.test_dataset is None or len(self.test_dataset) == 0:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )
