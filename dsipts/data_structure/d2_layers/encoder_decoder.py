"""
Encoder-Decoder implementation for D2 layer.

Provides EncoderDecoder class for creating sliding windows and encoder-decoder
structures from D1 layer data. Handles data scaling as well.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..d1_layers.base_d1 import BaseD1Layer
from ..scalers import Scaler

logger = logging.getLogger(__name__)


class EncoderDecoderDataset(Dataset):
    """Dataset for windowing and encoder-decoder structure creation."""

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
        """Initialize encoder-decoder dataset."""

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
                self.cat_feature_cols = d1_dataset.cat_cols or []
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

        logger.debug(f"Getting item {idx} - window: group_idx={group_idx}, start_idx={start_idx}")
        group_sample = self.d1_dataset[group_idx]

        # Use D1 metadata as source of truth for indices
        meta: Dict[str, Any] = getattr(self.d1_dataset, "metadata", {}) or {}
        d1_metadata = meta  # Make d1_metadata available in this scope
        idx_categorical: List[int] = list(meta.get("idx_categorical", []))
        feature_cols = meta.get("feature_cols", [])
        enrich_cat = meta.get("enrich_cat", [])

        # Temporal enrichment features are  x_cat_past and x_cat_future

        # Log group sample structure
        if idx == 0:  # Only log for the first item to avoid excessive logging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Group sample keys: {list(group_sample.keys())}")
                for key, value in group_sample.items():
                    if isinstance(value, torch.Tensor):
                        logger.debug(f"  {key}: tensor of shape {tuple(value.shape)}")
                    else:
                        logger.debug(f"  {key}: {type(value)}")

        # Apply scaling to group sample if needed
        d1_scaling_method = getattr(self.d1_dataset, "scaling_method", "minmax")
        if hasattr(self.d1_dataset, "scaling_params") and self.d1_dataset.scaling_params:
            # Check if we need to apply scaling on-the-fly
            memory_efficient = getattr(self.d1_dataset, "memory_efficient", False)

            # For memory-efficient mode, always apply scaling on-the-fly
            if memory_efficient:
                feature_cols = meta.get("feature_cols", [])
                if "x" in group_sample and len(feature_cols) > 0:
                    group_sample["x"] = self.d1_dataset._apply_scaling_to_tensor(group_sample["x"], feature_cols)
            # For non-memory-efficient mode with standard scaling, ensure data is scaled
            elif d1_scaling_method == "standard":
                # Check if the data appears to be unscaled (mean far from 0)
                feature_cols = meta.get("feature_cols", [])
                if "x" in group_sample and len(feature_cols) > 0:
                    x_mean = group_sample["x"].mean().item()
                    if abs(x_mean) > 5.0:  # If mean is far from 0, data might not be scaled
                        logger.warning(f"Data appears unscaled (mean={x_mean:.2f}), applying standard scaling on-the-fly")
                        group_sample["x"] = self.d1_dataset._apply_scaling_to_tensor(group_sample["x"], feature_cols)

        # Extract the window from the group's sequence
        past_end = start_idx + self.past_len
        future_end = past_end + self.future_len
        logger.debug(f"Window indices: start={start_idx}, past_end={past_end}, future_end={future_end}")

        # Extract past features and future targets from the group's tensors
        future_targets = group_sample["y"][past_end:future_end]  # [future_len, n_targets]
        logger.debug(f"Extracted future_targets with shape: {tuple(future_targets.shape)}")
        x = {}

        # Get additional metadata for processing
        idx_known_future: List[int] = list(meta.get("idx_known", []))  # Use idx_known from D1

        # For non-global forecasting, automatically include group columns in known future features
        global_forecasting = d1_metadata.get("global_forecasting", True)
        if not global_forecasting:
            group_cols = meta.get("group_cols", [])
            feature_cols = meta.get("feature_cols", [])
            for group_col in group_cols:
                if group_col in feature_cols:
                    group_idx = feature_cols.index(group_col)
                    if group_idx not in idx_known_future:
                        idx_known_future.append(group_idx)
                        logger.debug(
                            f"Auto-added group column '{group_col}' (idx: {group_idx}) to idx_known_future for non-global forecasting"  # noqa
                        )

        idx_targets_full: List[int] = list(meta.get("idx_targets", []))

        # Ensure all temporal features are treated as categorical
        # This is a safety check in case idx_categorical doesn't include them
        if enrich_cat and feature_cols:
            for temporal_feature in enrich_cat:
                if temporal_feature in feature_cols:
                    feature_idx = feature_cols.index(temporal_feature)
                    if feature_idx not in idx_categorical:
                        idx_categorical.append(feature_idx)

        # Determine numeric feature indices as complement of categorical
        n_features = int(meta.get("n_features", group_sample["x"].shape[1]))
        all_idx = list(range(n_features))
        idx_num = [i for i in all_idx if i not in idx_categorical]

        # Slice past/future from full X
        X_full = group_sample["x"]
        X_past = X_full[start_idx:past_end]
        X_future = X_full[past_end:future_end]

        # Split numeric and categorical tensors with correct dtypes
        x_num_past = X_past[:, idx_num].float() if len(idx_num) > 0 else torch.zeros((self.past_len, 0), dtype=torch.float32)
        x["x_num_past"] = x_num_past

        if len(idx_categorical) > 0:
            x_cat_past = X_past[:, idx_categorical].long()
            x["x_cat_past"] = x_cat_past

        # Known future features (split into num/cat)
        if self.future_len > 0:
            # For numeric features
            if len(idx_known_future) > 0:
                future_num_idx = [i for i in idx_known_future if i in idx_num]
                if len(future_num_idx) > 0:
                    x["x_num_future"] = X_future[:, future_num_idx].float()

            # For categorical features - include temporal enrichment + known categorical features
            future_cat_indices = []
            future_cat_names = []

            # Always include temporal enrichment features (they are known in advance)
            if enrich_cat and feature_cols:
                for temporal_feature in enrich_cat:
                    if temporal_feature in feature_cols:
                        feature_idx = feature_cols.index(temporal_feature)
                        if feature_idx in idx_categorical and feature_idx not in future_cat_indices:
                            future_cat_indices.append(feature_idx)
                            future_cat_names.append(temporal_feature)

            # Add known categorical features from idx_known_future
            if len(idx_known_future) > 0:
                # Simple rule: ALL known categorical columns go to x_cat_future
                # This includes user-specified known_cols + auto-added group_cols (for local forecasting) + temporal enrichment
                future_cat_idx = [i for i in idx_known_future if i in idx_categorical]
                for idx in future_cat_idx:
                    if idx not in future_cat_indices and idx < len(feature_cols):
                        feature_name = feature_cols[idx]
                        future_cat_indices.append(idx)
                        future_cat_names.append(feature_name)
                        logger.info(f"Including '{feature_name}' in x_cat_future (known categorical)")

            # Create x_cat_future tensor if we have categorical features
            if len(future_cat_indices) > 0:
                x_cat_future = X_future[:, future_cat_indices].long()
                x["x_cat_future"] = x_cat_future
                logger.info(f"x_cat_future shape: {tuple(x_cat_future.shape)} ({len(future_cat_indices)} features)")
            else:
                logger.info("No categorical features for x_cat_future")

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

        # CORRECTED LOGIC: Handle group_id based on global_forecasting setting
        # When global_forecasting=True: group_id as separate batch key
        # When global_forecasting=False: group_id treated as categorical feature (not separate batch key)
        global_forecasting = d1_metadata.get("global_forecasting", True)

        if global_forecasting:
            # Global forecasting: add group_id as separate batch key
            group_id = window.get("group_id", 0)
            # Handle different group_id types
            if isinstance(group_id, str):
                # Convert string group_id to integer using group mapping if available
                meta_group_mapping = meta.get("group_mapping", {})
                group_id = meta_group_mapping.get(group_id, 0)
                x["group_id"] = int(group_id)
            elif isinstance(group_id, (int, float)):
                x["group_id"] = int(group_id)
            else:
                # For tuple or other types, keep as is
                x["group_id"] = group_id
        # For local forecasting (global_forecasting=False), group_id is NOT added as separate batch key
        # It's already included in x_cat_past and x_cat_future as a categorical feature

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
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Get sample with encoder-decoder structure."""
        dataset_idx = self.indices[idx]
        return self.dataset[dataset_idx]


class EncoderDecoder(pl.LightningDataModule):
    """D2 layer for creating encoder-decoder structures from D1 data."""

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
        scaling_method: str = "standard",
        scale_targets: bool = False,
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
            scaling_method: Method for manual scaling ("standard" or "minmax", default: "standard")
                          Scaling is applied ONLY on training data to prevent data leakage
                          Supports both memory_efficient=True/False modes from D1 layer
            scale_targets: If True, also scale target variables (default: False)
                          Target scaling is also fitted only on training data
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
        self.scale_targets = scale_targets
        self.scaling_method = scaling_method

        # Initialize manual scaling
        self._scaler = Scaler(scaling_method=scaling_method, scale_targets=scale_targets)
        self.is_scaler_fitted = False

        # Extract column information from D1 dataset
        self.known_cols = d1_dataset.known_cols
        self.unknown_cols = d1_dataset.unknown_cols
        self.group_cols = d1_dataset.group_cols or []
        self.target_cols = d1_dataset.target_cols
        self.feature_cols = d1_dataset.feature_cols

        # Handle categorical columns
        try:
            self.cat_cols = d1_dataset.cat_cols if d1_dataset.cat_cols else []
            logger.debug(f"  cat_cols from D1: {len(self.cat_cols)} cols")
        except (AttributeError, TypeError):
            logger.warning("No categorical columns found in D1 dataset or cat_cols is None")
            self.cat_cols = []

        # Separate categorical and continuous columns
        all_feature_cols = self.feature_cols + self.target_cols
        self.cat_feature_cols = [col for col in all_feature_cols if col in self.cat_cols] + [
            col for col in self.cat_cols if col not in all_feature_cols
        ]
        self.cont_feature_cols = [col for col in all_feature_cols if col not in self.cat_cols]

        # Build valid windows
        self._build_valid_windows()
        logger.info(f"Initialized with {len(self.valid_windows)} windows")

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

        # Store reference to D2 layer in dataset for scaling access
        self.dataset.d2_layer = self

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
                self.train_dataset = EncoderDecoderSubset(self.dataset, list(range(len(self.valid_windows))))
                self.val_dataset = EncoderDecoderSubset(self.dataset, [])
                self.test_dataset = EncoderDecoderSubset(self.dataset, [])

    def fit_scaler(self, dataset):
        """
        Fit the scaler on numeric features from the given dataset.

        Args:
            dataset: Training dataset to fit the scaler on
        """
        self._scaler.fit_scaler(dataset)
        self.is_scaler_fitted = self._scaler.is_scaler_fitted

    def transform_with_scaler(self, dataset):
        """
        Transform the numeric features in the dataset using the fitted scaler.
        """
        return self._scaler.transform_with_scaler(dataset)

    def apply_inverse_scaling(self, data, data_type="features"):
        """
        Apply inverse scaling to denormalize predictions using manual scaling parameters.

        """
        return self._scaler.apply_inverse_scaling(data, data_type)

    def _build_valid_windows(self):
        """Build valid sliding windows from the D1 dataset."""
        from .utils import build_valid_windows

        self.valid_windows = build_valid_windows(
            self.d1_dataset, self.past_len, self.future_len, self.step_size, self.max_samples_per_group
        )

    def _is_valid_window(self, past_indices: List[int], future_indices: List[int]) -> bool:
        """Check if a window is valid (has sufficient data)."""
        from .utils import is_valid_window

        return is_valid_window(past_indices, future_indices, self.past_len, self.future_len)

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
        from .utils import create_random_splits, create_temporal_splits

        # Verify ratios sum to 1
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            logger.warning(f"Split ratios sum to {ratio_sum:.3f}, not 1.0")

        # Create splits using utility functions
        if method == "temporal":
            train_indices, val_indices, test_indices = create_temporal_splits(
                self.valid_windows, train_ratio, val_ratio, test_ratio
            )
        else:
            train_indices, val_indices, test_indices = create_random_splits(
                self.valid_windows, train_ratio, val_ratio, test_ratio, seed=42
            )

        logger.info(f"Split complete: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        train_dataset = EncoderDecoderSubset(self.dataset, train_indices)
        val_dataset = EncoderDecoderSubset(self.dataset, val_indices)
        test_dataset = EncoderDecoderSubset(self.dataset, test_indices)

        # Fit scaler on training data only
        self.fit_scaler(train_dataset)

        # Apply scaler to all datasets if fitted
        if self.is_scaler_fitted:
            train_dataset = self.transform_with_scaler(train_dataset)
            val_dataset = self.transform_with_scaler(val_dataset)
            test_dataset = self.transform_with_scaler(test_dataset)

        return (train_dataset, val_dataset, test_dataset)

    def setup(self, stage=None):
        """Set up datasets for training, validation, and testing."""
        if self.train_dataset is None and self.split_config:
            train_indices, val_indices, test_indices = self._create_splits(self.split_config)

            self.train_dataset = EncoderDecoderSubset(self.dataset, train_indices)
            self.val_dataset = EncoderDecoderSubset(self.dataset, val_indices)
            self.test_dataset = EncoderDecoderSubset(self.dataset, test_indices)

            # Fit scaler on training data only
            self.fit_scaler(self.train_dataset)

            # Apply scaler to all datasets if fitted
            if self.is_scaler_fitted:
                self.train_dataset = self.transform_with_scaler(self.train_dataset)
                self.val_dataset = self.transform_with_scaler(self.val_dataset)
                self.test_dataset = self.transform_with_scaler(self.test_dataset)

    def train_dataloader(self):
        """Return the training dataloader."""
        from .utils import custom_collate_fn

        if self.train_dataset is None:
            self.train_dataset = EncoderDecoderSubset(self.dataset, list(range(len(self.valid_windows))))

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.sampler is None else False,  # Don't shuffle when using a sampler
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            sampler=self.sampler,
        )

        return dataloader

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
