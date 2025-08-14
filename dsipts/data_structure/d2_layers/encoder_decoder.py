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
from torch.utils.data import DataLoader, Dataset, Sampler

from ..d1_layers.base_d1 import BaseD1Layer

logger = logging.getLogger(__name__)


class EncoderDecoderDataset(Dataset):
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
        include_target_in_decoder: bool = False,
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
            include_target_in_decoder: If True, includes target in decoder (for select models)
        """
        self.d1_dataset = d1_dataset
        self.valid_windows = valid_windows
        self.past_len = past_len
        self.future_len = future_len
        self.target_cols = target_cols or []
        self.cat_cols = cat_cols or []
        self.cont_feature_cols = cont_feature_cols or []
        self.include_target_in_decoder = include_target_in_decoder
        # Auto-detect categorical feature columns from D1 dataset if not provided
        if cat_feature_cols is None:
            try:
                # Access the cat_cols property from D1 dataset
                cat_cols_from_d1 = d1_dataset.cat_cols
                self.cat_feature_cols = cat_cols_from_d1 or []
            except (AttributeError, TypeError):
                self.cat_feature_cols = []
        else:
            self.cat_feature_cols = cat_feature_cols

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
        group_idx = window["group_idx"]
        start_idx = window["start_idx"]

        # Get the full group data from D1 dataset
        group_sample = self.d1_dataset[group_idx]

        # Extract the window from the group's sequence
        past_end = start_idx + self.past_len
        future_end = past_end + self.future_len

        # Extract past features and future targets from the group's tensors
        # past_features = group_sample["x"][start_idx:past_end]  # [past_len, n_features] #noqa TODOremove this?
        future_targets = group_sample["y"][past_end:future_end]  # [future_len, n_targets]

        # Build clean input dictionary - only include keys when data is present
        x = {}

        # Use D1 metadata as source of truth for indices
        meta: Dict[str, Any] = getattr(self.d1_dataset, "metadata", {}) or {}
        idx_categorical: List[int] = list(meta.get("idx_categorical", []))
        idx_known_future: List[int] = list(meta.get("idx_known_future", []))
        # TODO: remove idx_unknown_feature?
        idx_unknown_future: List[int] = list(meta.get("idx_unknown_future", []))  # noqa
        idx_targets_full: List[int] = list(meta.get("idx_targets", []))

        # Get temporal features from metadata if available
        enrich_cat = meta.get("enrich_cat", [])
        feature_cols = meta.get("feature_cols", [])

        # Ensure all temporal features are treated as categorical
        # This is a safety check in case idx_categorical doesn't include them
        if enrich_cat and feature_cols:
            for temporal_feature in enrich_cat:
                if temporal_feature in feature_cols:
                    feature_idx = feature_cols.index(temporal_feature)
                    if feature_idx not in idx_categorical:
                        idx_categorical.append(feature_idx)
                        logger.info(
                            f"Added temporal feature '{temporal_feature}' (idx: {feature_idx}) to categorical indices"  # noqa
                        )

        # Determine numeric feature indices as complement of categorical
        n_features = int(meta.get("n_features", group_sample["x"].shape[1]))
        all_idx = list(range(n_features))
        idx_num = [i for i in all_idx if i not in idx_categorical]

        # Slice past/future from full X
        X_full = group_sample["x"]
        X_past = X_full[start_idx:past_end]
        X_future = X_full[past_end:future_end]

        # Split numeric and categorical tensors with correct dtypes
        x_num_past = (
            X_past[:, idx_num].float()
            if len(idx_num) > 0
            else torch.zeros((self.past_len, 0), dtype=torch.float32)
        )
        x["x_num_past"] = x_num_past

        if len(idx_categorical) > 0:
            x_cat_past = X_past[:, idx_categorical].long()
            x["x_cat_past"] = x_cat_past

        # Known future features (split into num/cat)
        if self.future_len > 0 and len(idx_known_future) > 0:
            future_num_idx = [i for i in idx_known_future if i in idx_num]
            future_cat_idx = [i for i in idx_known_future if i in idx_categorical]

            if len(future_num_idx) > 0:
                x["x_num_future"] = X_future[:, future_num_idx].float()
            if len(future_cat_idx) > 0:
                x["x_cat_future"] = X_future[:, future_cat_idx].long()

        # Targets for decoder (future target values)
        x["y"] = future_targets.float()

        # Map idx_targets (relative to full X) into positions within x_num_past
        num_pos_map = {orig: pos for pos, orig in enumerate(idx_num)}
        mapped_targets = [num_pos_map[i] for i in idx_targets_full if i in num_pos_map]
        if len(mapped_targets) == 0 and len(idx_targets_full) > 0:
            logger.warning("All targets mapped to non-numeric features; idx_target will be empty")
        x["idx_target"] = torch.tensor(mapped_targets, dtype=torch.long)

        # Include target in decoder part if requested (for select models)
        if self.include_target_in_decoder and self.future_len > 0:
            x["decoder_target"] = future_targets.float()

        # Backward compatibility keys (kept for existing code)
        x["future_targets"] = future_targets.float()

        # Debug information (for tracing back to original CSV data)
        # Keep group_id as integer for consistency and debugging
        group_id = window.get("group_id", 0)
        if isinstance(group_id, str):
            # Convert string group_id to integer using group mapping if available
            meta_group_mapping = meta.get("group_mapping", {})
            group_id = meta_group_mapping.get(group_id, 0)
        x["group_id"] = int(group_id)  # Keep for debugging

        # Use actual start index from window for debugging (trace back to original CSV)
        x["time_idx"] = start_idx  # Actual start index in original data for debugging

        # Target tensor (for loss computation)
        y = future_targets

        return x, y


class EncoderDecoderSubset:
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
        split_config: Optional[Tuple] = None,
        num_workers: int = 0,
        sampler: Optional[Sampler] = None,
        target_normalizer: Optional[str] = None,
        max_samples_per_group: Optional[int] = None,
        precompute: bool = True,
        include_target_in_decoder: bool = False,
    ):
        """
        Initialize the EncoderDecoder.

        Args:
            d1_dataset: Any D1 layer implementation (BaseD1Layer subclass)
            past_len: Length of the past sequence (encoder)
            future_len: Length of the future sequence (decoder)
            batch_size: Batch size for dataloaders
            step_size: Step size for sliding window
            min_valid_length: Minimum required length for a valid window
            split_method: Method for splitting data ('percentage' or 'group')
            split_config: Configuration for splits
            num_workers: Number of workers for dataloaders
            sampler: Optional sampler for training dataloader
            target_normalizer: Optional normalizer for targets
            max_samples_per_group: Maximum samples per group
            precompute: Whether to precompute valid windows
            include_target_in_decoder: If True, include target in decoder part (for some models)
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
            target_cols=self.d1_dataset.target_cols,
            cat_cols=getattr(self.d1_dataset, "cat_cols", None),
            cont_feature_cols=getattr(self.d1_dataset, "num_cols", None),
            include_target_in_decoder=include_target_in_decoder,
        )

        # Create datasets if precompute is True
        if precompute:
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None

            if self.split_config:
                train_indices, val_indices, test_indices = self._create_splits(self.split_config)

                self.train_dataset = EncoderDecoderSubset(self.dataset, train_indices)
                self.val_dataset = EncoderDecoderSubset(self.dataset, val_indices)
                self.test_dataset = EncoderDecoderSubset(self.dataset, test_indices)

                logger.info(
                    f"Split statistics: Train: {len(train_indices)}, "
                    f"Validation: {len(val_indices)}, Test: {len(test_indices)}"
                )
            else:
                # Default to all indices as training
                self.train_dataset = EncoderDecoderSubset(
                    self.dataset, list(range(len(self.valid_windows)))
                )
                self.val_dataset = EncoderDecoderSubset(self.dataset, [])
                self.test_dataset = EncoderDecoderSubset(self.dataset, [])

    def _build_valid_windows(self):
        """
        Build valid sliding windows from the D1 dataset.

        The D1 dataset returns all data for a group as a single sample.
        We need to extract individual timesteps from each group to create windows.
        """
        self.valid_windows = []

        # Process each group in the D1 dataset
        for group_idx in range(len(self.d1_dataset)):
            group_sample = self.d1_dataset[group_idx]
            group_id = group_sample.get("group_id", group_idx)
            seq_len = group_sample.get("seq_len", 0)

            logger.debug(f"Processing group {group_id} with sequence length {seq_len}")

            # Create sliding windows within this group's sequence
            max_windows = seq_len - self.past_len - self.future_len + 1
            logger.debug(
                f"Group {group_id}: seq_len={seq_len}, past_len={self.past_len}, future_len={self.future_len}, max_windows={max_windows}"  # noqa
            )

            if max_windows > 0:
                for i in range(0, max_windows, self.step_size):
                    # Create window metadata
                    window = {
                        "group_idx": group_idx,  # Index in D1 dataset
                        "group_id": group_id,  # Group identifier
                        "start_idx": i,  # Start position within group sequence
                        "past_len": self.past_len,
                        "future_len": self.future_len,
                    }

                    self.valid_windows.append(window)
                    logger.debug(
                        f"Added valid window {len(self.valid_windows)} for group {group_id} at position {i}"  # noqa
                    )

                    # Limit samples per group if specified
                    if (
                        self.max_samples_per_group
                        and len([w for w in self.valid_windows if w["group_id"] == group_id])
                        >= self.max_samples_per_group
                    ):
                        break
            else:
                logger.warning(
                    f"Group {group_id} has insufficient data for windows (seq_len={seq_len}, required={self.past_len + self.future_len})"  # noqa
                )

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
    ) -> Tuple["EncoderDecoderSubset", "EncoderDecoderSubset", "EncoderDecoderSubset"]:
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
            EncoderDecoderSubset(self.dataset, train_indices),
            EncoderDecoderSubset(self.dataset, val_indices),
            EncoderDecoderSubset(self.dataset, test_indices),
        )

    def setup(self, stage=None):
        """Set up datasets for training, validation, and testing."""
        if self.train_dataset is None and self.split_config:
            train_indices, val_indices, test_indices = self._create_splits(self.split_config)

            self.train_dataset = EncoderDecoderSubset(self.dataset, train_indices)
            self.val_dataset = EncoderDecoderSubset(self.dataset, val_indices)
            self.test_dataset = EncoderDecoderSubset(self.dataset, test_indices)

            logger.info(
                f"Setup completed with split statistics: Train: {len(train_indices)}, "
                f"Validation: {len(val_indices)}, Test: {len(test_indices)}"
            )

    def train_dataloader(self):
        """Return the training dataloader."""
        from .utils import custom_collate_fn

        if self.train_dataset is None:
            # If no explicit split was provided, use all data for training
            self.train_dataset = EncoderDecoderSubset(
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
