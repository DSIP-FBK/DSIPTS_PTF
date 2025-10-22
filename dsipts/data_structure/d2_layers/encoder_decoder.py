"""
Encoder-Decoder implementation for D2 layer.

Provides EncoderDecoder class for creating sliding windows and encoder-decoder
structures from D1 layer data. Handles data scaling as well.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from ..d1_layers.base_d1 import BaseD1Layer
from .utils import custom_collate_fn

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

        # Scaler placeholders - will be set by EncoderDecoder after fitting
        self.feature_scaler = None
        self.target_scaler = None
        self.scale_targets = False

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
        if idx == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Group sample keys: {list(group_sample.keys())}")
                for key, value in group_sample.items():
                    if isinstance(value, torch.Tensor):
                        logger.debug(f"  {key}: tensor of shape {tuple(value.shape)}")
                    else:
                        logger.debug(f"  {key}: {type(value)}")

        # Extract the window from the group's sequence
        past_end = start_idx + self.past_len
        future_end = past_end + self.future_len
        logger.debug(f"Window indices: start={start_idx}, past_end={past_end}, future_end={future_end}")

        # Extract past features and future targets from the group's tensors
        future_targets = group_sample["y"][past_end:future_end]  # [future_len, n_targets]
        logger.debug(f"Extracted future_targets with shape: {tuple(future_targets.shape)}")
        x = {}

        idx_future: List[int] = list(meta.get("idx_future", []))  # Features available in future

        global_forecasting = d1_metadata.get("global_forecasting", True)
        if not global_forecasting:
            group_cols = meta.get("group_cols", [])
            feature_cols = meta.get("feature_cols", [])
            for group_col in group_cols:
                if group_col in feature_cols:
                    group_idx = feature_cols.index(group_col)
                    if group_idx not in idx_future:
                        idx_future.append(group_idx)
                        logger.debug(
                            f"Auto-added group column '{group_col}' (idx: {group_idx}) to idx_future for non-global forecasting"
                        )

        idx_targets_full: List[int] = list(meta.get("idx_targets", []))

        # Ensure all temporal features are treated as categorical
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

        # Future features (split into num/cat)
        if self.future_len > 0:
            # For numeric features
            if len(idx_future) > 0:
                future_num_idx = [i for i in idx_future if i in idx_num]
                if len(future_num_idx) > 0:
                    x["x_num_future"] = X_future[:, future_num_idx].float()

            future_cat_indices = []
            future_cat_names = []
            if enrich_cat and feature_cols:
                for temporal_feature in enrich_cat:
                    if temporal_feature in feature_cols:
                        feature_idx = feature_cols.index(temporal_feature)
                        if feature_idx in idx_categorical and feature_idx not in future_cat_indices:
                            future_cat_indices.append(feature_idx)
                            future_cat_names.append(temporal_feature)

            if len(idx_future) > 0:
                future_cat_idx = [i for i in idx_future if i in idx_categorical]
                for idx in future_cat_idx:
                    if idx not in future_cat_indices and idx < len(feature_cols):
                        feature_name = feature_cols[idx]
                        future_cat_indices.append(idx)
                        future_cat_names.append(feature_name)
                        logger.info(f"Including '{feature_name}' in x_cat_future (future categorical)")

            # Create x_cat_future tensor if we have categorical features
            if len(future_cat_indices) > 0:
                x_cat_future = X_future[:, future_cat_indices].long()
                x["x_cat_future"] = x_cat_future
                logger.debug(f"x_cat_future shape: {tuple(x_cat_future.shape)} ({len(future_cat_indices)} features)")
            else:
                logger.debug("No categorical features for x_cat_future")

        x["y"] = future_targets.float()
        num_pos_map = {orig: pos for pos, orig in enumerate(idx_num)}
        mapped_targets = [num_pos_map[i] for i in idx_targets_full if i in num_pos_map]
        if len(mapped_targets) == 0 and len(idx_targets_full) > 0:
            logger.warning("All targets mapped to non-numeric features; idx_target will be empty")
        x["idx_target"] = torch.tensor(mapped_targets, dtype=torch.long)

        if self.include_target_in_decoder and self.future_len > 0:
            x["decoder_target"] = future_targets.float()

        global_forecasting = d1_metadata.get("global_forecasting", True)

        if global_forecasting:
            group_id = window.get("group_id", 0)
            if isinstance(group_id, str):
                meta_group_mapping = meta.get("group_mapping", {})
                group_id = meta_group_mapping.get(group_id, 0)
                x["group_id"] = int(group_id)
            elif isinstance(group_id, (int, float)):
                x["group_id"] = int(group_id)
            else:
                x["group_id"] = group_id
        x["time_idx"] = start_idx
        y = future_targets

        # Apply scaling on-the-fly if scaler is fitted
        if self.feature_scaler is not None and hasattr(self.feature_scaler, "n_features_in_"):
            # Scale x_num_past
            if "x_num_past" in x and x["x_num_past"].numel() > 0:
                x_num_past_scaled = self.feature_scaler.transform(x["x_num_past"].numpy())
                x["x_num_past"] = torch.from_numpy(x_num_past_scaled).float()

            # Scale x_num_future (if present)
            if "x_num_future" in x and x["x_num_future"].numel() > 0:
                x_num_future_scaled = self.feature_scaler.transform(x["x_num_future"].numpy())
                x["x_num_future"] = torch.from_numpy(x_num_future_scaled).float()

        # Apply target scaling if enabled
        if self.scale_targets and self.target_scaler is not None and hasattr(self.target_scaler, "n_features_in_"):
            y_scaled = self.target_scaler.transform(y.numpy())
            y = torch.from_numpy(y_scaled).float()
            x["y"] = y  # Update y in x dict as well

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
        scaling_method: Optional[str] = None,
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
            split_config: Tuple (train_ratio, val_ratio, test_ratio) for data splitting
            num_workers: Number of workers for dataloaders
            sampler: Optional sampler for training dataloader
            target_normalizer: Optional normalizer for targets
            max_samples_per_group: Maximum samples per group
            precompute: if True, build valid windows in __init__ (default: True)
            include_target_in_decoder: if True include target in decoder part
            scaling_method: Scaling method ("standard" or "minmax" or None)
            scale_targets: if True, also scale target variables (default: False)
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

        # Initialize scikit-learn scalers directly
        if scaling_method == "standard":
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler() if scale_targets else None
        elif scaling_method == "minmax":
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler() if scale_targets else None
        elif scaling_method is None or scaling_method == "none":
            self.feature_scaler = None
            self.target_scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}. Use 'standard', 'minmax', or None.")

        self.is_scaler_fitted = False

        # Memory efficiency mode - read from D1 dataset
        self.memory_efficient = getattr(self.d1_dataset, "memory_efficient", True)
        logger.info(f"Memory efficient mode: {self.memory_efficient}")

        # Extract column information from D1 dataset
        self.past_cols = d1_dataset.past_cols
        self.future_cols = d1_dataset.future_cols
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

        # Initialize dataset splits as None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Auto-setup if split_config provided
        if split_config is not None:
            if len(split_config) != 3:
                raise ValueError(f"split_config must be tuple of 3 values (train, val, test), got {len(split_config)}")
            train_ratio, val_ratio, test_ratio = split_config
            logger.info(f"Auto-running setup() with split_config: {split_config}")
            self.setup(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

    def _fit_scaler_direct_d1(self, train_indices):
        """
        Optimized scaler fitting for memory_efficient=False.
        Directly accesses cached D1 data without going through D2 __getitem__.

        Args:
            train_indices: List of D2 window indices for training
        """
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        import numpy as np

        logger.info(f"Fitting {self.scaling_method} scaler directly on cached D1 data...")

        # Get metadata
        meta = getattr(self.d1_dataset, "metadata", {})
        idx_categorical = list(meta.get("idx_categorical", []))
        feature_cols = meta.get("feature_cols", [])
        n_features_total = len(feature_cols)
        idx_num = [i for i in range(n_features_total) if i not in idx_categorical]

        # Identify unique D1 group indices needed for training windows
        group_indices_needed = sorted(list(set(self.valid_windows[idx]["group_idx"] for idx in train_indices)))

        logger.info(f"  Accessing {len(group_indices_needed)} unique D1 groups for {len(train_indices)} training windows")

        # Extract RAW numerical data directly from D1 cache
        all_features_list = []
        all_targets_list = []

        for group_idx in group_indices_needed:
            # Get D1 data (will use cache if available, otherwise load)
            group_data = self.d1_dataset[group_idx]

            # Extract numeric columns from 'x' tensor
            if idx_num and "x" in group_data:
                numeric_data = group_data["x"][:, idx_num].numpy()
                all_features_list.append(numeric_data)

            # Extract targets if scaling targets
            if self.scale_targets and "y" in group_data:
                all_targets_list.append(group_data["y"].numpy())

        # Fit feature scaler on concatenated raw data
        if all_features_list:
            features_array = np.concatenate(all_features_list, axis=0)
            self.feature_scaler.fit(features_array)
            logger.info(
                f"  Feature scaler fitted on {features_array.shape[0]} total D1 timesteps with {features_array.shape[1]} features"
            )

        # Fit target scaler
        if self.scale_targets and self.target_scaler and all_targets_list:
            targets_array = np.concatenate(all_targets_list, axis=0)
            self.target_scaler.fit(targets_array)
            logger.info(
                f"  Target scaler fitted on {targets_array.shape[0]} total D1 timesteps with {targets_array.shape[1]} targets"
            )

        self.is_scaler_fitted = True

    def _fit_scaler_batched(self, train_indices):
        """
        Optimized scaler fitting using partial_fit with direct D1 cache access (for memory_efficient=True).
        Args:
            train_indices: List of D2 window indices for training
        """
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        logger.info(f"Fitting {self.scaling_method} scaler using partial_fit with direct D1 access (memory-efficient mode)...")

        # Get metadata
        meta = getattr(self.d1_dataset, "metadata", {})
        idx_categorical = list(meta.get("idx_categorical", []))
        feature_cols = meta.get("feature_cols", [])
        n_features_total = len(feature_cols)
        idx_num = [i for i in range(n_features_total) if i not in idx_categorical]

        # Identify unique D1 group indices needed for training windows
        group_indices_needed = sorted(list(set(self.valid_windows[idx]["group_idx"] for idx in train_indices)))

        logger.info(f"  Processing {len(group_indices_needed)} unique D1 groups for {len(train_indices)} training windows")
        logger.info("  Using partial_fit to minimize memory footprint")

        # Process each D1 group separately (memory-efficient)
        for group_num, group_idx in enumerate(group_indices_needed):
            if group_num % max(1, len(group_indices_needed) // 10) == 0:
                logger.info(
                    f"  Processing group {group_num + 1}/{len(group_indices_needed)} ({100 * (group_num + 1) / len(group_indices_needed):.1f}%)"  # noqa
                )

            # Get D1 data (will use cache if available, otherwise load)
            group_data = self.d1_dataset[group_idx]

            # Extract numeric columns
            if idx_num and "x" in group_data:
                numeric_data = group_data["x"][:, idx_num].numpy()

                # Partial fit on this group's data
                if numeric_data.size > 0:
                    self.feature_scaler.partial_fit(numeric_data)

            # Extract targets if scaling targets
            if self.scale_targets and self.target_scaler is not None and "y" in group_data:
                target_data = group_data["y"].numpy()
                if target_data.size > 0:
                    self.target_scaler.partial_fit(target_data)

        logger.info(f"Scaler fitted successfully on {len(group_indices_needed)} groups using partial_fit")
        self.is_scaler_fitted = True

    def fit_scaler(self, train_indices):
        """
        Fit the scaler on training data using direct D1 cache access.
        Delegates to the appropriate method based on memory_efficient mode.

        Args:
            train_indices: List of D2 window indices for training
        """
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        # Both modes now use direct D1 access!
        if not self.memory_efficient:
            # Direct D1 access with single fit() call
            self._fit_scaler_direct_d1(train_indices)
        else:
            # Direct D1 access with partial_fit() for memory efficiency
            self._fit_scaler_batched(train_indices)

        # Attach fitted scalers to dataset
        if self.is_scaler_fitted:
            logger.info("Attaching fitted scalers to dataset for on-the-fly transformation in __getitem__()")
            self.dataset.feature_scaler = self.feature_scaler
            self.dataset.target_scaler = self.target_scaler
            self.dataset.scale_targets = self.scale_targets

    def apply_inverse_scaling(self, data, data_type="features"):
        """
        Apply inverse scaling to denormalize predictions.

        Args:
            data: Data to inverse transform (numpy array or torch tensor)
                  Can be 2D [samples, features] or 3D [batch, time, features]
            data_type: Type of data ('features' or 'targets')

        Returns:
            Inverse transformed data in the same format as input
        """
        if not self.is_scaler_fitted:
            logger.warning("Scaler not fitted, returning original data")
            return data

        # Select appropriate scaler
        if data_type == "targets" and self.target_scaler:
            scaler = self.target_scaler
        elif data_type == "features" and self.feature_scaler:
            scaler = self.feature_scaler
        else:
            logger.warning(f"No scaler available for data_type='{data_type}', returning original data")
            return data

        # Convert to numpy if needed
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data

        # Handle 3D arrays [batch, time, features] by reshaping to 2D
        original_shape = data_np.shape
        if data_np.ndim == 3:
            # Reshape [batch, time, features] -> [batch*time, features]
            batch_size, time_steps, n_features = data_np.shape
            data_np = data_np.reshape(-1, n_features)

        # Apply inverse transform
        data_inverse = scaler.inverse_transform(data_np)

        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            data_inverse = data_inverse.reshape(original_shape)

        # Convert back to tensor if input was tensor
        if is_tensor:
            return torch.from_numpy(data_inverse).float()
        return data_inverse

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

    def _pretransform_splits(self, train_dataset, val_dataset, test_dataset):
        """
        Pre-transform all data when memory_efficient=False.

        This method transforms all samples upfront and caches them, trading memory for speed.
        Inference will be faster since transformation happens only once.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset) with pre-transformed data
        """
        logger.info("Pre-transforming training data...")
        train_transformed = self._pretransform_dataset(train_dataset)
        val_transformed = self._pretransform_dataset(val_dataset) if val_dataset and len(val_dataset) > 0 else val_dataset
        test_transformed = self._pretransform_dataset(test_dataset) if test_dataset and len(test_dataset) > 0 else test_dataset

        logger.info("Pre-transformation complete!")
        return train_transformed, val_transformed, test_transformed

    def _pretransform_dataset_direct(self, indices):
        """
        Pre-transform dataset by directly accessing D1 data

        Args:
            indices: List of window indices to extract
        Returns:
            PreTransformedDataset wrapper with cached transformed data
        """
        import numpy as np

        logger.info(f"Pre-transforming {len(indices)} samples (direct D1 access, no __getitem__)...")

        # Get D1 metadata
        meta = getattr(self.d1_dataset, "metadata", {}) or {}
        idx_categorical = list(meta.get("idx_categorical", []))
        feature_cols = meta.get("feature_cols", [])
        enrich_cat = meta.get("enrich_cat", [])
        idx_future = list(meta.get("idx_future", []))
        idx_targets_full = list(meta.get("idx_targets", []))
        global_forecasting = meta.get("global_forecasting", True)

        # Ensure temporal features are categorical
        if enrich_cat and feature_cols:
            for temporal_feature in enrich_cat:
                if temporal_feature in feature_cols:
                    feature_idx = feature_cols.index(temporal_feature)
                    if feature_idx not in idx_categorical:
                        idx_categorical.append(feature_idx)

        # Determine numeric indices
        n_features = int(meta.get("n_features", 0))
        all_idx = list(range(n_features))
        idx_num = [i for i in all_idx if i not in idx_categorical]

        # ULTRA-FAST EXTRACTION: Use index range instead of individual windows
        logger.info("  Extracting data using index range (ultra-fast)...")

        # Get D1 data once
        group_sample = self.d1_dataset[0]  # Assuming single group
        X_full = group_sample["x"].numpy()  # Convert to numpy once
        y_full = group_sample["y"].numpy()

        # Find the data range we need (minimum start to maximum end across all windows)
        # This handles both contiguous and non-contiguous window indices
        all_starts = [self.valid_windows[idx]["start_idx"] for idx in indices]
        all_ends = [self.valid_windows[idx]["start_idx"] + self.past_len + self.future_len for idx in indices]

        first_start = min(all_starts)
        last_end = min(max(all_ends), len(X_full))

        # Extract the entire data range at once (single slice!)
        X_range = X_full[first_start:last_end]  # Shape: (total_timesteps, n_features)
        y_range = y_full[first_start:last_end]  # Shape: (total_timesteps, n_targets)

        logger.info(f"  Extracted data range [{first_start}:{last_end}] = {last_end - first_start} timesteps")

        # Now extract windows from this range using relative indices
        window_starts = np.array([self.valid_windows[idx]["start_idx"] - first_start for idx in indices])

        # Create index arrays for ALL windows
        past_indices = window_starts[:, None] + np.arange(self.past_len)[None, :]
        future_indices = (window_starts[:, None] + self.past_len) + np.arange(self.future_len)[None, :]

        # Extract ALL windows at once using advanced indexing
        X_past_all = X_range[past_indices]  # (n_windows, past_len, n_features)
        X_future_all = X_range[future_indices]  # (n_windows, future_len, n_features)
        y_future_all = y_range[future_indices]  # (n_windows, future_len, n_targets)

        # Extract numeric features for ALL windows at once
        if len(idx_num) > 0:
            all_x_num_past = X_past_all[:, :, idx_num]

            future_num_idx = [i for i in idx_future if i in idx_num]
            if len(future_num_idx) > 0:
                all_x_num_future = X_future_all[:, :, future_num_idx]
            else:
                all_x_num_future = None
        else:
            all_x_num_past = None
            all_x_num_future = None

        # Store targets and categorical features
        all_y = y_future_all
        all_X_past = X_past_all
        all_X_future = X_future_all

        logger.info(f"  Extracted {len(indices)} windows in single vectorized operation")

        # Transform ALL numeric features in one shot (already vectorized!)
        logger.info("  Applying scaling transformation (single call, fully vectorized)...")

        if self.feature_scaler is not None and hasattr(self.feature_scaler, "n_features_in_"):
            # Transform x_num_past - already a numpy array (n_windows, past_len, n_features)
            if all_x_num_past is not None:
                original_shape = all_x_num_past.shape
                # Reshape to 2D: (n_windows * past_len, n_features)
                reshaped = all_x_num_past.reshape(-1, original_shape[-1])
                # Transform ALL at once - SINGLE CALL!
                transformed = self.feature_scaler.transform(reshaped)
                # Reshape back: (n_windows, past_len, n_features)
                all_x_num_past = transformed.reshape(original_shape)
                logger.info(
                    f" Transformed {len(indices)} windows × {self.past_len} timesteps= {reshaped.shape[0]} values in single call"
                )

            # Transform x_num_future
            if all_x_num_future is not None:
                original_shape = all_x_num_future.shape
                reshaped = all_x_num_future.reshape(-1, original_shape[-1])
                transformed = self.feature_scaler.transform(reshaped)
                all_x_num_future = transformed.reshape(original_shape)

        # Transform targets if enabled
        if self.scale_targets and self.target_scaler is not None and hasattr(self.target_scaler, "n_features_in_"):
            logger.info("  Applying target scaling (single call, fully vectorized)...")
            original_shape = all_y.shape
            reshaped = all_y.reshape(-1, original_shape[-1])
            transformed_y = self.target_scaler.transform(reshaped)
            all_y = transformed_y.reshape(original_shape)

        # Pre-compute future categorical indices once
        future_cat_indices = []
        if len(idx_categorical) > 0:
            if enrich_cat and feature_cols:
                for temporal_feature in enrich_cat:
                    if temporal_feature in feature_cols:
                        feature_idx = feature_cols.index(temporal_feature)
                        if feature_idx in idx_categorical and feature_idx not in future_cat_indices:
                            future_cat_indices.append(feature_idx)

            if len(idx_future) > 0:
                future_cat_idx = [i for i in idx_future if i in idx_categorical]
                for idx in future_cat_idx:
                    if idx not in future_cat_indices:
                        future_cat_indices.append(idx)

        # Pre-compute idx_target mapping once
        num_pos_map = {orig: pos for pos, orig in enumerate(idx_num)}
        mapped_targets = [num_pos_map[i] for i in idx_targets_full if i in num_pos_map]
        idx_target_tensor = torch.tensor(mapped_targets, dtype=torch.long)

        # Return vectorized dataset (creates dicts on-demand in __getitem__)
        logger.info(f"✅ Pre-transformation complete: {len(indices)} samples ready (vectorized storage)")
        from .utils import VectorizedPreTransformedDataset

        return VectorizedPreTransformedDataset(
            all_x_num_past,
            all_x_num_future,
            all_X_past,
            all_X_future,
            all_y,
            indices,
            self.valid_windows,
            idx_categorical,
            future_cat_indices,
            idx_target_tensor,
            global_forecasting,
            meta,
        )

    def _pretransform_dataset(self, dataset):
        """Wrapper that delegates to direct D1 access method."""
        # Extract indices from the dataset subset
        if hasattr(dataset, "indices"):
            indices = dataset.indices
        else:
            indices = list(range(len(dataset)))

        return self._pretransform_dataset_direct(indices)

    def _create_group_temporal_splits(self, train_ratio, val_ratio, test_ratio):
        """
        Create temporal splits within each group.
        Args:
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            test_ratio: Ratio for test data

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        train_windows = []
        val_windows = []
        test_windows = []

        # Group windows by group_id
        from collections import defaultdict

        windows_by_group = defaultdict(list)
        for idx, window in enumerate(self.valid_windows):
            group_id = window["group_id"]
            windows_by_group[group_id].append(idx)

        logger.info(f"Creating group-based temporal splits for {len(windows_by_group)} groups")

        # Split each group temporally
        for group_id, window_indices in windows_by_group.items():
            n_windows = len(window_indices)

            # Calculate split points for this group
            train_end = int(n_windows * train_ratio)
            val_end = int(n_windows * (train_ratio + val_ratio))

            # Assign windows to splits
            train_windows.extend(window_indices[:train_end])
            val_windows.extend(window_indices[train_end:val_end])
            test_windows.extend(window_indices[val_end:])

        logger.debug(f"Group splits: {len(train_windows)} train, {len(val_windows)} val, {len(test_windows)} test windows")

        return train_windows, val_windows, test_windows

    def setup(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Setup method: splits data, fits scaler, and prepares datasets.
        Handles:
        1. Splitting data based on split_method ('percentage' or 'group')
        2. Fitting scaler on training data
        3. Transformation strategy based on memory_efficient:
           - memory_efficient= False: pre-transforms all data upfront (faster inference)
           - memory_efficient=True: transforms on-the-fly in __getitem__ (lower memory)

        Args:
            train_ratio: Ratio of data for training (default: 0.7)
            val_ratio: Ratio of data for validation (default: 0.15)
            test_ratio: Ratio of data for testing (default: 0.15)
        """
        # Skip if already set up
        if self.train_dataset is not None:
            logger.info("Datasets already set up, skipping setup()")
            return

        # Verify ratios sum to 1
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            logger.warning(f"Split ratios sum to {ratio_sum:.3f}, not 1.0")

        # Step 1: Create splits based on split_method
        logger.info(
            f"Creating splits with method='{self.split_method}' (train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

        if self.split_method == "percentage":
            # Global percentage split
            from .utils import create_temporal_splits

            train_indices, val_indices, test_indices = create_temporal_splits(
                self.valid_windows, train_ratio, val_ratio, test_ratio
            )
        elif self.split_method == "group":
            # Group-based temporal split
            train_indices, val_indices, test_indices = self._create_group_temporal_splits(train_ratio, val_ratio, test_ratio)
        else:
            raise ValueError(f"Unknown split_method: {self.split_method}. Use 'percentage' or 'group'.")

        logger.info(f"Split complete: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        # Step 2: Create subset datasets
        train_dataset = EncoderDecoderSubset(self.dataset, train_indices)
        val_dataset = EncoderDecoderSubset(self.dataset, val_indices)
        test_dataset = EncoderDecoderSubset(self.dataset, test_indices)

        # Step 3: Fit scaler and apply transformation strategy
        if self.scaling_method is not None:
            logger.info(f"Fitting {self.scaling_method} scaler on training data...")

            # Both modes now use direct D1 access with train_indices!
            self.fit_scaler(train_indices)

            self.is_scaler_fitted = True

            # Step 4: Apply transformation based on memory_efficient flag
            if not self.memory_efficient:
                # Pre-transform all data for faster inference
                logger.info("Pre-transforming all splits (memory_efficient=False)...")
                train_dataset, val_dataset, test_dataset = self._pretransform_splits(train_dataset, val_dataset, test_dataset)
            else:
                # Attach scalers to dataset for on-the-fly transformation
                logger.info("Attaching scalers for on-the-fly transformation (memory_efficient=True)")
                self.dataset.feature_scaler = self.feature_scaler
                self.dataset.target_scaler = self.target_scaler
                self.dataset.scale_targets = self.scale_targets

        # Step 5: Store datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        logger.info("Setup complete!")

    def train_dataloader(self):
        """Return the training dataloader."""

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

        if self.test_dataset is None or len(self.test_dataset) == 0:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )
