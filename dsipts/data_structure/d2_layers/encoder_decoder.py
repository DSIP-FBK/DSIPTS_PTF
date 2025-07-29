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


class EncoderDecoderDataset:
    """Dataset class that handles windowing logic and encoder-decoder structure creation."""

    def __init__(
        self,
        d1_dataset: BaseD1Layer,
        valid_windows: List[Dict],
        past_len: int,
        future_len: int,
        target_cols: List[str],
        cat_cols: List[str] = None,
        cont_feature_cols: List[str] = None,
        cat_feature_cols: List[str] = None,
    ):
        """Initialize the encoder-decoder dataset.

        Args:
            d1_dataset: The D1 layer dataset
            valid_windows: List of valid window dictionaries
            past_len: Length of past sequence
            future_len: Length of future sequence
            target_cols: Target column names
            cat_cols: Categorical column names
            cont_feature_cols: Continuous feature column names
            cat_feature_cols: Categorical feature column names
        """
        self.d1_dataset = d1_dataset
        self.valid_windows = valid_windows
        self.past_len = past_len
        self.future_len = future_len
        self.target_cols = target_cols or []
        self.cat_cols = cat_cols or []
        self.cont_feature_cols = cont_feature_cols or []
        # Auto-detect categorical feature columns from D1 dataset if not provided
        if cat_feature_cols is None:
            try:
                # Access the cat_cols property from D1 dataset
                cat_cols_from_d1 = d1_dataset.cat_cols
                print(f"DEBUG: D1 cat_cols: {cat_cols_from_d1}")
                self.cat_feature_cols = cat_cols_from_d1 or []
                print(f"DEBUG: Set cat_feature_cols to: {self.cat_feature_cols}")
            except (AttributeError, TypeError) as e:
                print(f"DEBUG: Error accessing cat_cols: {e}")
                self.cat_feature_cols = []
        else:
            self.cat_feature_cols = cat_feature_cols
            print(f"DEBUG: Using provided cat_feature_cols: {self.cat_feature_cols}")

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
            clean batch structure with only necessary keys
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

        # Build clean input dictionary - only include keys when data is present
        x = {}

        # Core LinearTS model keys (always present)
        x["x_num_past"] = past_features  # Past numerical features
        x["y"] = future_targets  # Target values for training
        x["idx_target"] = torch.tensor(list(range(len(self.target_cols))))  # Target indices

        # Add categorical features only if present
        if len(self.cat_feature_cols) > 0:
            # Extract categorical features from past samples
            cat_features = torch.zeros(self.past_len, len(self.cat_feature_cols), dtype=torch.long)
            for i, sample in enumerate(past_samples):
                if "categorical" in sample and sample["categorical"] is not None:
                    # Extract categorical features for this time step
                    cat_data = sample["categorical"]
                    if isinstance(cat_data, np.ndarray):
                        cat_features[i] = torch.from_numpy(cat_data.astype(np.int64))
                    else:
                        cat_features[i] = torch.tensor(cat_data, dtype=torch.long)
            x["x_cat_past"] = cat_features

        # Add future numerical features only if present (known future features)
        if len(self.cont_feature_cols) > 0 and hasattr(self, "_has_future_features"):
            future_cont = torch.zeros(self.future_len, len(self.cont_feature_cols))
            x["x_num_future"] = future_cont

        # Add future categorical features only if present
        if len(self.cat_feature_cols) > 0:
            # Always provide future categorical features if we have categorical features
            # This ensures the LinearTS complex case gets the expected input dimensions
            future_cat = torch.zeros(self.future_len, len(self.cat_feature_cols), dtype=torch.long)
            x["x_cat_future"] = future_cat

        # Add categorical features only if they exist
        if self.cat_feature_cols and len(self.cat_feature_cols) > 0:
            # Only include categorical keys if categorical columns exist
            static_cat = torch.zeros(len(self.cat_feature_cols))
            x["static_categorical_features"] = static_cat

            # Add encoder_cat only if categorical features exist
            x["encoder_cat"] = torch.zeros((self.past_len, len(self.cat_feature_cols)))
            x["x_cat_past"] = x["encoder_cat"]

            # Add decoder_cat only if categorical features exist
            if self.future_len > 0:
                x["decoder_cat"] = torch.zeros((self.future_len, len(self.cat_feature_cols)))
                x["x_cat_future"] = x["decoder_cat"]

        # Backward compatibility keys (kept for existing code)
        x["past_features"] = past_features
        x["future_targets"] = future_targets

        # Essential metadata
        x["group_id"] = window.get("group_id", 0)
        x["time_idx"] = torch.arange(self.past_len)[-1].item()  # Last time index

        # Target tensor (for loss computation)
        y = future_targets

        return x, y


class TimeSeriesSubset:
    """Subset class for dataset splits - delegates to EncoderDecoderDataset.
    Maintains clean separation of concerns, allows for different
    split strategies without modifying the core dataset.
    """

    def __init__(self, dataset: EncoderDecoderDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        """Return the number of samples in this subset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Get a sample with encoder-decoder structure.

        Args:
            idx: Index of the window to retrieve

        Returns:
            Tuple of (input_dict, target_tensor) where input_dict contains
            encoder-decoder structure compatible with PyTorch Forecasting
        """
        # Map subset index to dataset index
        dataset_idx = self.indices[idx]
        return self.dataset[dataset_idx]


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
        target_normalizer: Optional[str] = None,
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
            target_normalizer: Type of target normalization
            max_samples_per_group: Maximum samples per group

        Note:
            known_cols and unknown_cols are automatically inherited from the d1_dataset
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
        self.max_samples_per_group = max_samples_per_group
        self.precompute = precompute

        # Extract column information from D1 dataset
        self.known_cols = d1_dataset.known_cols
        self.unknown_cols = d1_dataset.unknown_cols
        self.group_cols = d1_dataset.group_cols or []
        self.target_cols = d1_dataset.target_cols
        self.feature_cols = d1_dataset.feature_cols

        # Handle potentially None or empty categorical columns
        try:
            self.cat_cols = d1_dataset.cat_cols if d1_dataset.cat_cols else []
        except (AttributeError, TypeError):
            logger.warning("No categorical columns found in D1 dataset or cat_cols is None")
            self.cat_cols = []

        # Separate categorical and continuous columns
        all_feature_cols = self.feature_cols + self.target_cols
        # Categorical feature columns can be either:
        # 1. Feature/target columns that are also in cat_cols, OR
        # 2. Pure categorical columns (cat_cols that are not in feature/target cols)
        self.cat_feature_cols = [col for col in all_feature_cols if col in self.cat_cols] + [
            col for col in self.cat_cols if col not in all_feature_cols
        ]
        self.cont_feature_cols = [col for col in all_feature_cols if col not in self.cat_cols]

        # Build valid windows
        self._build_valid_windows()

        logger.info(f"EncoderDecoder initialized with {len(self.valid_windows)} valid windows")

        # Create the main dataset with windowing logic
        self.dataset = EncoderDecoderDataset(
            d1_dataset=self.d1_dataset,
            valid_windows=self.valid_windows,
            past_len=self.past_len,
            future_len=self.future_len,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            cont_feature_cols=self.cont_feature_cols,
            cat_feature_cols=self.cat_feature_cols,
        )

        # Create datasets if precompute is True
        if precompute:
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None

            if self.split_config:
                train_indices, val_indices, test_indices = self._create_splits(self.split_config)

                self.train_dataset = TimeSeriesSubset(self.dataset, train_indices)
                self.val_dataset = TimeSeriesSubset(self.dataset, val_indices)
                self.test_dataset = TimeSeriesSubset(self.dataset, test_indices)

                logger.info(
                    f"Split statistics: Train: {len(train_indices)}, "
                    f"Validation: {len(val_indices)}, Test: {len(test_indices)}"
                )
            else:
                # Default to all indices as training
                self.train_dataset = TimeSeriesSubset(
                    self.dataset, list(range(len(self.valid_windows)))
                )
                self.val_dataset = TimeSeriesSubset(self.dataset, [])
                self.test_dataset = TimeSeriesSubset(self.dataset, [])

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
            total_samples = len(self.valid_windows)

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
        total_samples = len(self.dataset)

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
            TimeSeriesSubset(self.dataset, train_indices),
            TimeSeriesSubset(self.dataset, val_indices),
            TimeSeriesSubset(self.dataset, test_indices),
        )

    def setup(self, stage=None):
        """Set up datasets for training, validation, and testing."""
        if self.train_dataset is None and self.split_config:
            train_indices, val_indices, test_indices = self._create_splits(self.split_config)

            self.train_dataset = TimeSeriesSubset(self.dataset, train_indices)
            self.val_dataset = TimeSeriesSubset(self.dataset, val_indices)
            self.test_dataset = TimeSeriesSubset(self.dataset, test_indices)

            logger.info(
                f"Setup completed with split statistics: Train: {len(train_indices)}, "
                f"Validation: {len(val_indices)}, Test: {len(test_indices)}"
            )

    def train_dataloader(self):
        """Return the training dataloader."""
        from .utils import custom_collate_fn

        if self.train_dataset is None:
            # If no explicit split was provided, use all data for training
            self.train_dataset = TimeSeriesSubset(
                self.dataset, list(range(len(self.valid_windows)))
            )

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
