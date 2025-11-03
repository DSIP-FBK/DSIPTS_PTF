"""
Encoder-Decoder implementation for D2 layer.

Provides EncoderDecoder class for creating sliding windows and encoder-decoder
structures from D1 layer data. Handles data scaling as well.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
        use_cache: bool = True,
        cache_size: int = 32000,
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
        self.use_cache = use_cache
        self.cache_size = cache_size

        # Scaler placeholders - will be set by EncoderDecoder after fitting
        self.feature_scaler = None
        self.target_scaler = None

        # Setup LRU cache for transformed windows if enabled
        if self.use_cache:
            self._get_transformed_window = lru_cache(maxsize=cache_size)(self._get_transformed_window_impl)

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

    def _get_transformed_window_impl(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Internal method to get and transform a window (cacheable).
        This is the expensive part that benefits from caching.
        """
        return self._get_window_no_cache(idx)

    def _get_window_no_cache(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Get a sample with encoder-decoder structure (no caching).
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

                # -------------------------------------------------
                # TODO: DELETE THIS ONE‑TIME LOGGING OF TRANSFORMED FEATURE VALUES
                # -------------------------------------------------
                if not getattr(self, "_logged_transform", False):
                    # Show the first 5 rows (or fewer if the tensor is smaller)
                    rows = min(5, x["x_num_past"].shape[0])
                    logger.info(
                        f"[Scaler] First {rows} rows of transformed FEATURES (x_num_past) – raw values:\n"
                        f"{x['x_num_past'][:rows].cpu().numpy()}"
                    )
                    # Mark that we have already logged – prevents repeated output
                    self._logged_transform = True

            # Scale x_num_future (if present)
            if "x_num_future" in x and x["x_num_future"].numel() > 0:
                x_num_future_scaled = self.feature_scaler.transform(x["x_num_future"].numpy())
                x["x_num_future"] = torch.from_numpy(x_num_future_scaled).float()

                # -------------------------------------------------
                # TODO: DELETE THIS ONE‑TIME LOGGING OF TRANSFORMED FEATURE VALUES
                # -------------------------------------------------
                if not getattr(self, "_logged_transform", False):
                    # Show the first 5 rows (or fewer if the tensor is smaller)
                    rows = min(5, x["x_num_future"].shape[0])
                    logger.info(
                        f"[Scaler] First {rows} rows of transformed FEATURES (x_num_future) – raw values:\n"
                        f"{x['x_num_future'][:rows].cpu().numpy()}"
                    )
                    # Mark that we have already logged – prevents repeated output
                    self._logged_transform = True

            # ONE‑TIME LOGGING OF TRANSFORMED FEATURE VALUES
            # -------------------------------------------------
            if not getattr(self, "_logged_transform", False):
                # Show the first 5 rows (or fewer if the tensor is smaller)
                rows = min(5, x["x_num_future"].shape[0])
                logger.info(
                    f"[Scaler] First {rows} rows of transformed FEATURES (x_num_future) – raw values:\n"
                    f"{x['x_num_future'][:rows].cpu().numpy()}"
                )
                # Mark that we have already logged – prevents repeated output
                self._logged_transform = True

        # Apply target scaling if enabled and y contains data
        # We check that the scaler exists, has a `transform` method, and that y is a non‑empty tensor.
        if (
            self.target_scaler is not None
            and hasattr(self.target_scaler, "transform")
            and isinstance(y, torch.Tensor)
            and y.numel() > 0
        ):
            # Transform y using the scaler and convert back to a torch tensor
            y = torch.from_numpy(self.target_scaler.transform(y.numpy())).float()
            x["y"] = y

            # -------------------------------------------------
            # TODO: DELETE THIS ONE‑TIME LOGGING OF TRANSFORMED TARGET VALUES
            # -------------------------------------------------
            if not getattr(self, "_logged_transform", False):
                # Show the first 5 rows (or fewer if the tensor is smaller)
                rows = min(5, y.shape[0])
                logger.info(f"[Scaler] First {rows} rows of transformed TARGET (y) – raw values:\n{y[:rows].cpu().numpy()}")
                # Mark that we have already logged – prevents repeated output
                self._logged_transform = True

        return x, y

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Get a sample with encoder-decoder structure.
        Uses LRU cache if enabled to avoid redundant window extraction and processing.
        """
        if self.use_cache:
            return self._get_transformed_window(idx)
        else:
            return self._get_window_no_cache(idx)


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
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_group_config: Optional[Tuple[List, List, List]] = None,
        num_workers: int = 0,
        sampler: Optional[Sampler] = None,
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
            split_ratio: Tuple of (train_ratio, val_ratio, test_ratio) for splitting data.
                Default: (0.7, 0.15, 0.15)
                - For local forecasting: Applied directly to all windows
                - For global forecasting: Applied based on split_group_config strategy
            split_group_config: Tuple of 3 lists (train_groups, val_groups, test_groups).
                Default: None (pure temporal split, local forecasting)
                - If None: Pure temporal split on all windows
                - If (train, [], []): Temporal split of train groups only
                - If (train, val, test): Hybrid split (ratio on train, 100% for val/test)
            num_workers: Number of workers for dataloaders
            sampler: Optional sampler for training dataloader
            max_samples_per_group: Maximum samples per group
            precompute: if True, build valid windows in __init__ (default: True)
            include_target_in_decoder: if True include target in decoder part
            scaling_method: Scaler for features. Options: 'standard', 'minmax', or None
            scale_targets: If True, scale target variables using same scaler as features
        """
        super().__init__()

        self.d1_dataset = d1_dataset
        self.past_len = past_len
        self.future_len = future_len
        self.batch_size = batch_size
        self.step_size = step_size
        self.min_valid_length = min_valid_length or past_len
        self.split_ratio = split_ratio
        self.split_group_config = split_group_config
        self.num_workers = num_workers
        self.sampler = sampler
        self.max_samples_per_group = max_samples_per_group
        self.precompute = precompute
        self.scaling_method = scaling_method
        self.scale_targets = scale_targets

        # Determine if global forecasting is enabled from D1 dataset
        self.global_forecasting = getattr(self.d1_dataset, "global_forecasting", False)
        logger.info(f"Global forecasting mode: {self.global_forecasting}")

        # State flags for setup() method
        self.splits_created = False  # Track if splits have been created
        self.is_scaler_fitted = False  # Track if scaler has been fitted

        # Initialize scikit-learn scalers (not fitted yet)
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
            use_cache=self.memory_efficient,
            cache_size=32000,
        )

        # Placeholders for split datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Store split indices
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def _fit_scaler_direct_d1(self, train_indices):
        """
        Optimized scaler fitting for memory_efficient=False.

        Args:
            train_indices: List of D2 window indices for training
        """
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        logger.info(f"Fitting {self.scaling_method} scaler on training windows...")

        # Get metadata
        meta = getattr(self.d1_dataset, "metadata", {})
        idx_categorical = list(meta.get("idx_categorical", []))
        feature_cols = meta.get("feature_cols", [])
        n_features_total = len(feature_cols)
        idx_num = [i for i in range(n_features_total) if i not in idx_categorical]

        logger.info(f"  Fitting on {len(train_indices)} training windows")

        # Extract data ONLY from training windows
        all_features_list = []
        all_targets_list = []

        for window_idx in train_indices:
            window = self.valid_windows[window_idx]
            group_idx = window["group_idx"]
            start_idx = window["start_idx"]
            past_len = window["past_len"]
            end_idx = start_idx + past_len
            group_data = self.d1_dataset[group_idx]

            if idx_num and "x" in group_data:
                window_numeric = group_data["x"][start_idx:end_idx, idx_num].numpy()
                if window_numeric.size > 0:
                    all_features_list.append(window_numeric)

            if self.scale_targets and "y" in group_data:
                window_targets = group_data["y"][start_idx:end_idx].numpy()
                if window_targets.size > 0:
                    all_targets_list.append(window_targets)

        # Fit feature scaler on concatenated raw data
        if all_features_list:
            features_array = np.concatenate(all_features_list, axis=0)
            self.feature_scaler.fit(features_array)
            logger.info(
                f"  Feature scaler fitted on {features_array.shape[0]} total D1 timesteps with {features_array.shape[1]} features"
            )

        # Fit target scaler
        if self.target_scaler and all_targets_list:
            targets_array = np.concatenate(all_targets_list, axis=0)
            self.target_scaler.fit(targets_array)
            logger.info(
                f"  Target scaler fitted on {targets_array.shape[0]} total D1 timesteps with {targets_array.shape[1]} targets"
            )

        self.is_scaler_fitted = True

    def _fit_scaler_batched(self, train_indices):
        """
        Optimized scaler fitting using partial_fit on training windows only (for memory_efficient=True).
        Args:
            train_indices: List of D2 window indices for training
        """
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        logger.info(f"Fitting {self.scaling_method} scaler using partial_fit on training windows only...")

        # Get metadata
        meta = getattr(self.d1_dataset, "metadata", {})
        idx_categorical = list(meta.get("idx_categorical", []))
        feature_cols = meta.get("feature_cols", [])
        n_features_total = len(feature_cols)
        idx_num = [i for i in range(n_features_total) if i not in idx_categorical]

        logger.info(f"  Fitting on {len(train_indices)} training windows using partial_fit")

        batch_size = 100
        total_timesteps = 0

        for batch_start in range(0, len(train_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(train_indices))
            batch_indices = train_indices[batch_start:batch_end]

            if batch_start % (batch_size * 10) == 0:
                logger.info(
                    f"  Processing windows {batch_start}/{len(train_indices)} ({100 * batch_start / len(train_indices):.1f}%)"
                )

            # Collect data from this batch of windows
            batch_features = []
            batch_targets = []

            for window_idx in batch_indices:
                window = self.valid_windows[window_idx]
                group_idx = window["group_idx"]
                start_idx = window["start_idx"]
                past_len = window["past_len"]
                end_idx = start_idx + past_len
                group_data = self.d1_dataset[group_idx]
                if idx_num and "x" in group_data:
                    window_numeric = group_data["x"][start_idx:end_idx, idx_num].numpy()
                    if window_numeric.size > 0:
                        batch_features.append(window_numeric)

                if self.scale_targets and "y" in group_data:
                    window_targets = group_data["y"][start_idx:end_idx].numpy()
                    if window_targets.size > 0:
                        batch_targets.append(window_targets)

            if batch_features:
                batch_features_array = np.concatenate(batch_features, axis=0)
                self.feature_scaler.partial_fit(batch_features_array)
                total_timesteps += batch_features_array.shape[0]

            if self.target_scaler and batch_targets:
                batch_targets_array = np.concatenate(batch_targets, axis=0)
                self.target_scaler.partial_fit(batch_targets_array)

        logger.info(f"  Feature scaler fitted on {total_timesteps} total timesteps from {len(train_indices)} training windows")
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
        if not self.memory_efficient:
            self._fit_scaler_direct_d1(train_indices)
        else:
            self._fit_scaler_batched(train_indices)
        if self.is_scaler_fitted:
            logger.info("Attaching fitted scalers to dataset for on-the-fly transformation in __getitem__()")
            self.dataset.feature_scaler = self.feature_scaler
            self.dataset.target_scaler = self.target_scaler

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

        # seleting appropriate scaler
        if data_type == "targets" and self.target_scaler:
            scaler = self.target_scaler
        elif data_type == "features" and self.feature_scaler:
            scaler = self.feature_scaler
        else:
            logger.warning(f"No scaler available for data_type='{data_type}', returning original data")
            return data

        # convert to numpy if needed
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
        """
        logger.info(f"Pre-transforming {len(indices)} samples")

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
            # one-time build of fast lookup structures
            feature_to_idx = {name: i for i, name in enumerate(feature_cols)}
            cat_set = set(idx_categorical)  # existing indices

            for temporal_feature in enrich_cat:
                feature_idx = feature_to_idx.get(temporal_feature)
                if feature_idx is not None and feature_idx not in cat_set:
                    idx_categorical.append(feature_idx)
                    cat_set.add(feature_idx)  # keep set in sync
        """ # TODO: REMOVE COMMENTED CODE If the above optimized code works fine [O(1) complexity]
        if enrich_cat and feature_cols:
            for temporal_feature in enrich_cat:
                if temporal_feature in feature_cols:
                    feature_idx = feature_cols.index(temporal_feature)
                    if feature_idx not in idx_categorical:
                        idx_categorical.append(feature_idx)
        """
        # Determine numeric indices
        n_features = int(meta.get("n_features", 0))
        all_idx = list(range(n_features))
        idx_num = [i for i in all_idx if i not in idx_categorical]

        # ULTRA-FAST EXTRACTION: Group windows by group_idx for efficient extraction
        logger.info("  Extracting data by group (optimized)...")

        # Group windows by their group_idx
        from collections import defaultdict

        windows_by_group = defaultdict(list)
        for idx in indices:
            group_idx = self.valid_windows[idx]["group_idx"]
            windows_by_group[group_idx].append(idx)

        logger.info(f"  Processing {len(windows_by_group)} unique groups for {len(indices)} windows")

        # Extract data for each group separately
        all_x_past_list = []
        all_x_future_list = []
        all_y_future_list = []
        all_window_info = []

        for group_idx, group_window_indices in windows_by_group.items():
            # Get D1 data for this group
            group_data = self.d1_dataset[group_idx]
            X_group = group_data["x"].numpy()
            y_group = group_data["y"].numpy()

            # Extract windows for this group
            for window_idx in group_window_indices:
                start_idx = self.valid_windows[window_idx]["start_idx"]
                end_idx = start_idx + self.past_len + self.future_len

                # Check bounds
                if end_idx > len(X_group):
                    logger.warning(f"  Skipping window {window_idx}: end_idx {end_idx} > group length {len(X_group)}")
                    continue

                # Extract past and future windows
                x_past = X_group[start_idx : start_idx + self.past_len]
                x_future = X_group[start_idx + self.past_len : end_idx]
                y_future = y_group[start_idx + self.past_len : end_idx]

                all_x_past_list.append(x_past)
                all_x_future_list.append(x_future)
                all_y_future_list.append(y_future)
                all_window_info.append(self.valid_windows[window_idx])

        # Stack all windows
        X_past_all = np.stack(all_x_past_list, axis=0)
        X_future_all = np.stack(all_x_future_list, axis=0)
        y_future_all = np.stack(all_y_future_list, axis=0)

        logger.info(f"  Extracted {len(all_x_past_list)} windows successfully")

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
        if self.target_scaler is not None and hasattr(self.target_scaler, "n_features_in_"):
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
            list(range(len(all_window_info))),  # Use sequential indices
            all_window_info,  # Use extracted window info
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

    def _create_splits(self):
        """
        Unified method to create train/val/test splits with flexible group-based logic.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        from collections import defaultdict

        from .utils import create_temporal_splits

        train_ratio, val_ratio, test_ratio = self.split_ratio

        # Verify ratios sum to 1
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            logger.warning(f"Split ratios sum to {ratio_sum:.3f}, not 1.0")

        # If global_forecasting=False (local forecasting), ALWAYS use temporal split
        if not self.global_forecasting:
            logger.info("[D2 split] global_forecasting=False (Local forecasting). Ignoring split_group_config.")
            logger.info("[D2 split] Applying pure temporal split to all windows.")
            train_indices, val_indices, test_indices = create_temporal_splits(
                self.valid_windows, train_ratio, val_ratio, test_ratio
            )
            logger.info(
                f"[D2 split] Temporal split complete: Train={len(train_indices)}, "
                f"Val={len(val_indices)}, Test={len(test_indices)}"
            )
            return train_indices, val_indices, test_indices

        # === Case 1: NO split_group_config provided ===
        # This is a pure TEMPORAL (Local) split
        if self.split_group_config is None:
            logger.info("No split_group_config provided. Applying temporal split to all groups.")
            train_indices, val_indices, test_indices = create_temporal_splits(
                self.valid_windows, train_ratio, val_ratio, test_ratio
            )
            logger.info(
                f"[D2 split] Temporal split complete: Train={len(train_indices)}, "
                f"Val={len(val_indices)}, Test={len(test_indices)}"
            )
            return train_indices, val_indices, test_indices

        # === Case 2: split_group_config ARE provided ===
        # This is a GLOBAL split

        train_groups = self.split_group_config[0]
        val_groups = self.split_group_config[1]
        test_groups = self.split_group_config[2]

        if not train_groups:
            raise ValueError("Global forecasting requires a non-empty 'train' list in split_group_config.")

        # CRITICAL: Validate that groups are mutually exclusive
        train_set = set(train_groups)
        val_set = set(val_groups)
        test_set = set(test_groups)

        # Check for overlaps: to prevent data leakage between train/val/test
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap:
            raise ValueError(
                f"DATA LEAKAGE DETECTED: Groups {train_val_overlap} appear in BOTH train_groups and val_groups. "
                f"This causes validation data to leak into training data. "
                f"Groups must be mutually exclusive across train/val/test splits."
            )
        if train_test_overlap:
            raise ValueError(
                f"DATA LEAKAGE DETECTED: Groups {train_test_overlap} appear in BOTH train_groups and test_groups. "
                f"This causes test data to leak into training data. "
                f"Groups must be mutually exclusive across train/val/test splits."
            )
        if val_test_overlap:
            raise ValueError(
                f"DATA LEAKAGE DETECTED: Groups {val_test_overlap} appear in BOTH val_groups and test_groups. "
                f"This causes test data to leak into validation data. "
                f"Groups must be mutually exclusive across train/val/test splits."
            )

        logger.info("[D2 split] Group validation passed: No overlapping groups detected.")

        # Collate all windows by their group
        windows_by_group = defaultdict(list)
        for idx, window in enumerate(self.valid_windows):
            windows_by_group[window["group_id"]].append(idx)

        # --- Initialize final index lists ---
        train_indices = []
        val_indices = []
        test_indices = []

        # --- Strategy A: "Strict Group Separation" ---
        # If user provides val or test groups, we assume they want strict separation.
        # The split_ratio will *only* be used on the train_groups.
        if val_groups or test_groups:
            logger.info("Hybrid group/temporal split: val/test groups provided.")

            # 1. Add 100% of val_groups to val_indices
            for group_id in val_groups:
                val_indices.extend(windows_by_group[group_id])
                logger.info(f"[D2 split]   Val group '{group_id}': {len(windows_by_group[group_id])} windows -> 100% to val")

            # 2. Add 100% of test_groups to test_indices
            for group_id in test_groups:
                test_indices.extend(windows_by_group[group_id])
                logger.info(f"[D2 split]   Test group '{group_id}': {len(windows_by_group[group_id])} windows -> 100% to test")

            # 3. Split the train_groups using the ratio
            train_group_windows = []
            for group_id in train_groups:
                train_group_windows.extend(windows_by_group[group_id])
                logger.info(f"[D2 split]   Train group '{group_id}': {len(windows_by_group[group_id])} windows")

            # Apply ratio to train_group_windows
            n_train_group = len(train_group_windows)
            train_end = int(n_train_group * train_ratio)
            val_end = int(n_train_group * (train_ratio + val_ratio))

            # Add the split data to the final lists
            train_indices.extend(train_group_windows[:train_end])
            val_indices.extend(train_group_windows[train_end:val_end])
            test_indices.extend(train_group_windows[val_end:])

            logger.info("Hybrid group/temporal split complete.")

        # --- Strategy B: "Temporal Split of Train Groups" ---
        # if user ONLY provides train_groups: they want to create all three sets from that list.
        else:
            logger.info("Only train_groups provided. Splitting them temporally by split_ratio.")
            train_group_windows = []
            for group_id in train_groups:
                train_group_windows.extend(windows_by_group[group_id])
                logger.info(f"[D2 split]   Train group '{group_id}': {len(windows_by_group[group_id])} windows")

            # Apply ratio to create all three sets
            n_train_group = len(train_group_windows)
            train_end = int(n_train_group * train_ratio)
            val_end = int(n_train_group * (train_ratio + val_ratio))

            train_indices = train_group_windows[:train_end]
            val_indices = train_group_windows[train_end:val_end]
            test_indices = train_group_windows[val_end:]
            logger.info("Pure temporal split of train_groups complete.")

        logger.info(f"Final split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        return train_indices, val_indices, test_indices

    def setup(self, stage: str):
        """
        Called once per stage ('fit', 'test', 'predict') per process.

        This method:
        1. Creates train/val/test splits (guarded by splits_created flag)
        2. Fits scaler on training data (guarded by is_scaler_fitted flag)
        3. Transforms data based on memory_efficient mode
        4. Creates stage-specific datasets

        Args:
            stage: Stage of training ('fit', 'test', 'predict', or PyTorch Lightning TrainerFn enum)
        """
        logger.info(f"[D2 setup] Called with stage='{stage}'")

        # Convert TrainerFn enum to string if needed
        stage_str = str(stage).split(".")[-1].lower() if hasattr(stage, "name") else str(stage).lower()

        # 🎯 STAGE 1: Create splits (ONCE, guarded by flag)
        if not self.splits_created:
            logger.info("[D2 setup] Splits not created. Creating train/val/test splits...")

            # Use unified split method
            self.train_indices, self.val_indices, self.test_indices = self._create_splits()

            logger.info(
                f"[D2 setup] Splits created: Train={len(self.train_indices)}, "
                f"Val={len(self.val_indices)}, Test={len(self.test_indices)}"
            )
            self.splits_created = True
        else:
            logger.info("[D2 setup] Splits already created. Skipping split step.")

        # 🎯 STAGE 2: Fit scaler (ONCE, guarded by flag)
        if self.scaling_method is not None and not self.is_scaler_fitted:
            logger.info(f"[D2 setup] Scaler not fitted. Fitting {self.scaling_method} scaler on training data...")

            # Fit scaler on training data only
            self.fit_scaler(self.train_indices)

            logger.info(f"[D2 setup] ✅ Scaler fitted (is_scaler_fitted={self.is_scaler_fitted})")

        elif self.scaling_method is not None and self.is_scaler_fitted:
            logger.info("[D2 setup] Scaler already fitted. Skipping fit step.")
        else:
            logger.info("[D2 setup] No scaling_method specified. Skipping scaler fitting.")

        # 🎯 STAGE 3: Create stage-specific datasets
        if "fit" in stage_str:
            logger.info("[D2 setup] Setting up 'fit' stage (train + val datasets)...")

            # Create subset datasets
            train_dataset = EncoderDecoderSubset(self.dataset, self.train_indices)
            val_dataset = EncoderDecoderSubset(self.dataset, self.val_indices) if self.val_indices else None

            # Apply transformation strategy
            if self.scaling_method is not None:
                if not self.memory_efficient:
                    logger.info("[D2 setup] Pre-transforming train/val data (memory_efficient=False)...")
                    train_dataset = self._pretransform_dataset_direct(self.train_indices)
                    val_dataset = self._pretransform_dataset_direct(self.val_indices) if self.val_indices else None
                else:
                    logger.info("[D2 setup] Attaching scalers for on-the-fly transformation (memory_efficient=True)")
                    self.dataset.feature_scaler = self.feature_scaler
                    self.dataset.target_scaler = self.target_scaler

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            logger.info("[D2 setup] 'fit' datasets created.")

        if "test" in stage_str:
            logger.info("[D2 setup] Setting up 'test' stage...")

            test_dataset = EncoderDecoderSubset(self.dataset, self.test_indices) if self.test_indices else None

            if self.scaling_method is not None and test_dataset:
                if not self.memory_efficient:
                    logger.info("[D2 setup] Pre-transforming test data (memory_efficient=False)...")
                    test_dataset = self._pretransform_dataset_direct(self.test_indices)
                else:
                    logger.info("[D2 setup] Using on-the-fly transformation for test (memory_efficient=True)")
                    # Scalers already attached in 'fit' stage

            self.test_dataset = test_dataset
            logger.info("[D2 setup] 'test' dataset created.")

        if "predict" in stage_str:
            logger.info("[D2 setup] Setting up 'predict' stage...")
            # For predict, typically use test dataset
            if self.test_dataset is None:
                self.test_dataset = EncoderDecoderSubset(self.dataset, self.test_indices) if self.test_indices else None
            logger.info("[D2 setup] 'predict' dataset ready.")

        logger.info(f"[D2 setup] Setup complete for stage='{stage}'!")

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
