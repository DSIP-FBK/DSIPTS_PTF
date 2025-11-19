"""
Encoder-Decoder implementation for D2 layer.

Provides EncoderDecoder class for creating sliding windows and encoder-decoder
structures from D1 layer data. Handles data scaling as well.
"""

import logging
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from ..d1_layers.base_d1 import BaseD1Layer
from .utils import *  # noqa: F401, F403

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

        # --- OPTIMIZATION: Use pre-calculated indices from D1 metadata ---
        meta = getattr(self.d1_dataset, "metadata", {}) or {}

        # Get D1 metadata (source of truth - all indices pre-calculated in D1)
        self.idx_cat_past = list(meta.get("idx_cat_past", []))
        self.idx_cat_future = list(meta.get("idx_cat_future", []))
        idx_targets_full = list(meta.get("idx_targets", []))
        self.global_forecasting = meta.get("global_forecasting", True)

        n_features = int(meta.get("n_features", 0))
        all_idx = set(range(n_features))

        # Calculate numeric indices ONCE (all non-categorical)
        self.idx_num = sorted(list(all_idx - set(self.idx_cat_past)))

        # Calculate future numeric indices ONCE (intersection of numeric and future)
        idx_future_all = set(meta.get("idx_future", []))
        self.idx_num_future = sorted(list(set(self.idx_num) & idx_future_all))

        # Calculate target index mapping ONCE
        num_pos_map = {orig: pos for pos, orig in enumerate(self.idx_num)}
        mapped_targets = [num_pos_map[i] for i in idx_targets_full if i in num_pos_map]
        self.idx_target_tensor = torch.tensor(mapped_targets, dtype=torch.long)

        logger.debug("[EncoderDecoderDataset] Using pre-calculated indices from D1:")
        logger.debug(f"  idx_num: {len(self.idx_num)} features")
        logger.debug(f"  idx_cat_past: {len(self.idx_cat_past)} features")
        logger.debug(f"  idx_cat_future: {len(self.idx_cat_future)} features")
        logger.debug(f"  idx_num_future: {len(self.idx_num_future)} features")
        # --- End of optimization ---

        # Setup LRU cache for transformed windows if enabled
        if self.use_cache:
            self._get_transformed_window = lru_cache(maxsize=cache_size)(self._get_window_no_cache)

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

    def _get_window_no_cache(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Get a sample with encoder-decoder structure (no caching).
        OPTIMIZED: Uses pre-calculated indices from __init__ - no redundant calculations.
        """
        window = self.valid_windows[idx]
        group_idx = window["group_idx"]
        start_idx = window["start_idx"]

        logger.debug(f"Getting item {idx} - window: group_idx={group_idx}, start_idx={start_idx}")
        group_sample = self.d1_dataset[group_idx]

        # --- 1. Slice tensors from D1 data ---
        past_end = start_idx + self.past_len
        future_end = past_end + self.future_len

        X_full = group_sample["x"]
        X_past = X_full[start_idx:past_end]
        X_future = X_full[past_end:future_end]
        y = group_sample["y"][past_end:future_end]  # Future targets

        x = {}  # Output dictionary

        # --- 2. Slice Past (using pre-calculated indices) ---
        x["x_num_past"] = (
            X_past[:, self.idx_num].float() if len(self.idx_num) > 0 else torch.zeros((self.past_len, 0), dtype=torch.float32)
        )

        if len(self.idx_cat_past) > 0:
            x["x_cat_past"] = X_past[:, self.idx_cat_past].long()

        # --- 3. Slice Future (using pre-calculated indices) ---
        if self.future_len > 0:
            if len(self.idx_num_future) > 0:
                x["x_num_future"] = X_future[:, self.idx_num_future].float()

            if len(self.idx_cat_future) > 0:
                x["x_cat_future"] = X_future[:, self.idx_cat_future].long()

        # --- 4. Add targets and metadata ---
        x["y"] = y.float()
        x["idx_target"] = self.idx_target_tensor
        x["time_idx"] = start_idx

        if self.global_forecasting:
            group_id = window.get("group_id", 0)
            if isinstance(group_id, str):
                meta = getattr(self.d1_dataset, "metadata", {}) or {}
                meta_group_mapping = meta.get("group_mapping", {})
                group_id = meta_group_mapping.get(group_id, 0)
            x["group_id"] = int(group_id) if isinstance(group_id, (int, float, str)) else group_id

        if self.include_target_in_decoder and self.future_len > 0:
            x["decoder_target"] = y.float()

        # Apply scaling on-the-fly if scaler is fitted
        # Handle both per-group and global scaling
        if self.feature_scaler is not None:
            # Determine which scaler to use
            if isinstance(self.feature_scaler, dict):
                # Per-group scaling
                scaler = self.feature_scaler.get(group_idx)
                if scaler is None:
                    logger.warning(f"No scaler found for group {group_idx}, skipping scaling")
            else:
                # Global scaling
                scaler = self.feature_scaler if hasattr(self.feature_scaler, "n_features_in_") else None

            if scaler is not None and hasattr(scaler, "n_features_in_"):
                # Scale x_num_past
                if "x_num_past" in x and x["x_num_past"].numel() > 0:
                    x_num_past_scaled = scaler.transform(x["x_num_past"].numpy())
                    x["x_num_past"] = torch.from_numpy(x_num_past_scaled).float()

                # Scale x_num_future (if present)
                if "x_num_future" in x and x["x_num_future"].numel() > 0:
                    x_num_future_scaled = scaler.transform(x["x_num_future"].numpy())
                    x["x_num_future"] = torch.from_numpy(x_num_future_scaled).float()

                # DEBUG: ONE TIME LOGGING OF TRANSFORMED FEATURE VALUES
                # -------------------------------------------------
                if not getattr(self, "_logged_transform", False):
                    # Show the first 5 rows (or fewer if the tensor is smaller)
                    if "x_num_past" in x and x["x_num_past"].numel() > 0:
                        rows = min(5, x["x_num_past"].shape[0])
                        logger.info(
                            f"[Scaler] First {rows} rows of transformed FEATURES (x_num_past) – raw values:\n"
                            f"{x['x_num_past'][:rows].cpu().numpy()}"
                        )
                    if "x_num_future" in x and x["x_num_future"].numel() > 0:
                        rows = min(5, x["x_num_future"].shape[0])
                        logger.info(
                            f"[Scaler] First {rows} rows of transformed FEATURES (x_num_future) – raw values:\n"
                            f"{x['x_num_future'][:rows].cpu().numpy()}"
                        )
                    # Mark that we have already logged – prevents repeated output
                    self._logged_transform = True

        # Apply target scaling if enabled and y contains data
        # Handle both per-group and global target scaling
        if self.target_scaler is not None and isinstance(y, torch.Tensor) and y.numel() > 0:
            # Determine which target scaler to use
            if isinstance(self.target_scaler, dict):
                # Per-group scaling
                target_scaler = self.target_scaler.get(group_idx)
                if target_scaler is None:
                    logger.warning(f"No target scaler found for group {group_idx}, skipping target scaling")
            else:
                # Global scaling
                target_scaler = self.target_scaler if hasattr(self.target_scaler, "transform") else None

            if target_scaler is not None and hasattr(target_scaler, "transform"):
                # Transform y using the scaler and convert back to a torch tensor
                y = torch.from_numpy(target_scaler.transform(y.numpy())).float()
                x["y"] = y
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
        normalize_per_group: bool = False,
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
            normalize_per_group: If True and global_forecasting=False, fit separate scalers per group
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
        self.normalize_per_group = normalize_per_group

        # Determine if global forecasting is enabled from D1 dataset
        self.global_forecasting = getattr(self.d1_dataset, "global_forecasting", False)
        logger.info(f"Global forecasting mode: {self.global_forecasting}")

        # Validate normalize_per_group: only applicable with local forecasting
        if normalize_per_group and self.global_forecasting:
            raise ValueError(
                "normalize_per_group=True is only applicable with local forecasting (global_forecasting=False). "
                "Per-group scaling is not supported for global forecasting mode."
            )

        # State flags for setup() method
        self.splits_created = False  # Track if splits have been created
        self.is_scaler_fitted = False  # Track if scaler has been fitted

        # Initialize scikit-learn scalers (not fitted yet)
        # For normalize_per_group, we'll use dictionaries to store per-group scalers
        # Initialize scalers based on scaling strategy
        # Two types of scalers:
        # 1. Dict of scalers (per-group): {group_idx: StandardScaler(), ...}
        # 2. Single scaler (global): StandardScaler() or MinMaxScaler()
        if normalize_per_group and not self.global_forecasting:
            # Per-group scalers stored as dict[group_id -> scaler]
            # Each group gets its own scaler fitted on that group's training data
            self.feature_scaler = {}  # Dict: {group_idx: StandardScaler()}
            self.target_scaler = {} if scale_targets else None  # Dict: {group_idx: StandardScaler()}
            self.per_group_scaling = True
            logger.info("Per-group scaling enabled (normalize_per_group=True, global_forecasting=False)")
        elif scaling_method == "standard":
            # Global scaling: single scaler for all groups
            self.feature_scaler = StandardScaler()  # Single sklearn scaler object
            self.target_scaler = StandardScaler() if scale_targets else None  # Single sklearn scaler object
            self.per_group_scaling = False
        elif scaling_method == "minmax":
            # Global scaling: single scaler for all groups
            self.feature_scaler = MinMaxScaler()  # Single sklearn scaler object
            self.target_scaler = MinMaxScaler() if scale_targets else None  # Single sklearn scaler object
            self.per_group_scaling = False
        elif scaling_method is None or scaling_method == "none":
            # No scaling
            self.feature_scaler = None
            self.target_scaler = None
            self.per_group_scaling = False
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

        # Handle categorical columns - use D1 metadata directly
        meta = getattr(self.d1_dataset, "metadata", {}) or {}

        try:
            self.cat_cols = d1_dataset.cat_cols if d1_dataset.cat_cols else []
            # Check for categorical features in past/future lists (more accurate than just cat_cols)
            cat_past_list = meta.get("cat_past_list", [])
            cat_future_list = meta.get("cat_future_list", [])
            has_categorical = len(cat_past_list) > 0 or len(cat_future_list) > 0
            logger.debug(f"  cat_cols from D1: {len(self.cat_cols)} cols")
            logger.debug(f"  cat_past_list: {len(cat_past_list)} features, cat_future_list: {len(cat_future_list)} features")
            if not has_categorical:
                logger.debug("  No categorical features in past or future")
        except (AttributeError, TypeError):
            logger.warning("No categorical columns found in D1 dataset or cat_cols is None")
            self.cat_cols = []

        # Use pre-calculated indices from D1 metadata
        self.idx_cat_past = list(meta.get("idx_cat_past", []))
        self.idx_cat_future = list(meta.get("idx_cat_future", []))

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

    def _has_window_gaps(self):
        """
        Simple check: if step_size > window_size, there will be gaps between windows.
        Window size = past_len + future_len

        Returns:
            bool: True if gaps exist (use boolean mask), False if dense (use range extraction)
        """
        window_size = self.past_len + self.future_len
        has_gaps = self.step_size > window_size

        if has_gaps:
            logger.debug(f"Window gaps detected: step_size={self.step_size} > window_size={window_size}")
        else:
            logger.debug(f"Dense windows: step_size={self.step_size} <= window_size={window_size}")

        return has_gaps

    def _extract_with_range(self, group_x_data, group_y_data, windows_in_group, idx_num):
        """
        Extract features and targets using contiguous range extraction (fast path).
        Assumes dense windows (step_size <= window_size).

        IMPORTANT: This operates GROUP-WISE, not window-by-window.
        Extracts ONE contiguous slice covering all windows in the group.

        Args:
            group_x_data: Group feature tensor
            group_y_data: Group target tensor
            windows_in_group: List of windows for this group
            idx_num: Indices of numeric features

        Returns:
            tuple: (numeric_features, targets)
        """
        if not windows_in_group:
            return None, None

        # Find range boundaries
        sorted_windows = sorted(windows_in_group, key=lambda w: w["start_idx"])
        first_start = sorted_windows[0]["start_idx"]
        last_window = sorted_windows[-1]
        last_end = last_window["start_idx"] + last_window["past_len"] + self.future_len

        # Single contiguous slice
        numeric_features = None
        if idx_num and group_x_data is not None:
            numeric_features = group_x_data[first_start:last_end, :][:, idx_num].numpy()

        targets = None
        if self.scale_targets and group_y_data is not None:
            targets = group_y_data[first_start:last_end].numpy()

        return numeric_features, targets

    def _extract_with_mask(self, group_x_data, group_y_data, windows_in_group, group_len, idx_num):
        """
        Extract features and targets using boolean mask (precise but slower).

        IMPORTANT: This operates GROUP-WISE, not window-by-window.
        Creates boolean mask for entire group, then extracts in ONE operation.

        Args:
            group_x_data: Group feature tensor
            group_y_data: Group target tensor
            windows_in_group: List of windows for this group
            group_len: Total length of the group
            idx_num: Indices of numeric features

        Returns:
            tuple: (numeric_features, targets)
        """
        # Create boolean masks
        feature_mask = np.zeros(group_len, dtype=bool)
        target_mask = np.zeros(group_len, dtype=bool)

        for window in windows_in_group:
            start_idx = window["start_idx"]
            past_len = window["past_len"]

            # Paint mask for features (past)
            past_end_idx = start_idx + past_len
            feature_mask[start_idx:past_end_idx] = True

            # Paint mask for targets (future)
            future_end_idx = past_end_idx + self.future_len
            target_mask[past_end_idx:future_end_idx] = True

        # Extract using masks
        numeric_features = None
        if idx_num and group_x_data is not None:
            numeric_features = group_x_data[feature_mask, :][:, idx_num].numpy()

        targets = None
        if self.scale_targets and group_y_data is not None:
            targets = group_y_data[target_mask].numpy()

        return numeric_features, targets

    def _fit_scaler_direct_d1(self, train_indices):
        """
        Hybrid scaler fitting with automatic optimization strategy selection.

        Uses range extraction for dense windows (step_size <= window_size) and
        boolean mask for sparse windows (step_size > window_size).
        This method uses fit() for memory_efficient=False mode.

        Args:
            train_indices: List of D2 window indices for training
        """
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        # Simple check: if step_size > window_size, there are gaps
        has_gaps = self._has_window_gaps()
        method_name = "boolean_mask" if has_gaps else "range_extraction"

        logger.info(f"Fitting {self.scaling_method} scaler on training windows (Hybrid Method: {method_name})...")

        # Get metadata
        meta = getattr(self.d1_dataset, "metadata", {})
        idx_cat_past = list(meta.get("idx_cat_past", []))
        feature_cols = meta.get("feature_cols", [])
        n_features_total = len(feature_cols)
        idx_num = [i for i in range(n_features_total) if i not in idx_cat_past]

        # --- Step 1: Collate all training windows by their D1 group ---
        # Group-wise processing: organize windows by group for efficient extraction
        windows_by_group = defaultdict(list)
        for window_idx in train_indices:
            window = self.valid_windows[window_idx]
            windows_by_group[window["group_idx"]].append(window)

        logger.info(f"  Fitting on {len(train_indices)} windows across {len(windows_by_group)} D1 groups...")

        all_features_list = []
        all_targets_list = []

        # --- Step 2: Iterate per-group with hybrid optimization ---
        if self.per_group_scaling:
            # Per-group scaling: fit separate scaler for each group
            logger.info("  Using per-group scaling (separate scaler per group)")
            for group_idx, windows_in_group in windows_by_group.items():
                # Get the entire raw data tensor for this group from D1 cache
                group_data = self.d1_dataset[group_idx]
                group_x_data = group_data["x"]
                group_y_data = group_data["y"]
                group_len = len(group_x_data)

                # --- Step 3: Extract data using appropriate method ---
                if has_gaps:
                    # Use boolean mask for sparse windows
                    numeric_features, targets = self._extract_with_mask(
                        group_x_data, group_y_data, windows_in_group, group_len, idx_num
                    )
                else:
                    # Use range extraction for dense windows
                    numeric_features, targets = self._extract_with_range(group_x_data, group_y_data, windows_in_group, idx_num)

                # --- Step 4: Fit scaler for this group ---
                if numeric_features is not None and numeric_features.size > 0:
                    # Create and fit scaler for this group
                    if self.scaling_method == "standard":
                        group_scaler = StandardScaler()
                    elif self.scaling_method == "minmax":
                        group_scaler = MinMaxScaler()
                    else:
                        group_scaler = StandardScaler()  # Default

                    group_scaler.fit(numeric_features)
                    self.feature_scaler[group_idx] = group_scaler
                    logger.debug(f"    Group {group_idx}: fitted scaler on {numeric_features.shape[0]} timesteps")

                # Fit target scaler for this group
                if self.target_scaler is not None and targets is not None and targets.size > 0:
                    if self.scaling_method == "standard":
                        group_target_scaler = StandardScaler()
                    elif self.scaling_method == "minmax":
                        group_target_scaler = MinMaxScaler()
                    else:
                        group_target_scaler = StandardScaler()  # Default

                    group_target_scaler.fit(targets)
                    self.target_scaler[group_idx] = group_target_scaler

            logger.info(f"  Per-group scalers fitted for {len(self.feature_scaler)} groups")
        else:
            # Global scaling: fit single scaler on all data
            for group_idx, windows_in_group in windows_by_group.items():
                # Get the entire raw data tensor for this group from D1 cache
                group_data = self.d1_dataset[group_idx]
                group_x_data = group_data["x"]
                group_y_data = group_data["y"]
                group_len = len(group_x_data)

                # --- Step 3: Extract data using appropriate method ---
                if has_gaps:
                    # Use boolean mask for sparse windows
                    numeric_features, targets = self._extract_with_mask(
                        group_x_data, group_y_data, windows_in_group, group_len, idx_num
                    )
                else:
                    # Use range extraction for dense windows
                    numeric_features, targets = self._extract_with_range(group_x_data, group_y_data, windows_in_group, idx_num)

                # --- Step 4: Collect data ---
                if numeric_features is not None and numeric_features.size > 0:
                    all_features_list.append(numeric_features)

                if targets is not None and targets.size > 0:
                    all_targets_list.append(targets)

            # --- Step 5: Fit scaler (outside the loop) ---
            # Fit feature scaler on concatenated raw data
            if all_features_list:
                features_array = np.concatenate(all_features_list, axis=0)
                self.feature_scaler.fit(features_array)
                logger.info(
                    f"  Feature scaler fitted on {features_array.shape[0]} total timesteps"
                    f" with {features_array.shape[1]} features"
                )
            else:
                logger.warning("No numeric features found to fit scaler.")

            # Fit target scaler
            if self.target_scaler and all_targets_list:
                targets_array = np.concatenate(all_targets_list, axis=0)
                self.target_scaler.fit(targets_array)
                logger.info(
                    f"  Target scaler fitted on {targets_array.shape[0]} total timesteps with {targets_array.shape[1]} targets"
                )
            elif self.scale_targets:
                logger.warning("Target scaling was requested, but no target data was found to fit scaler.")

        self.is_scaler_fitted = True

    def _fit_scaler_batched_fallback(self, group_data, group_len, feature_mask, target_mask, idx_num):
        """
        Fallback method for extremely large groups that don't fit in memory.
        Processes the group in chunks and calls partial_fit on each chunk.

        Args:
            group_data: The loaded group data dict from D1 (already loaded)
            group_len: Length of the group data
            feature_mask: Boolean mask for feature extraction
            target_mask: Boolean mask for target extraction
            idx_num: Indices of numeric features

        Returns:
            Number of timesteps processed
        """
        CHUNK_SIZE = 500_000  # Process 500k rows at a time
        total_timesteps = 0

        logger.info(f"    Using chunked processing (chunk_size={CHUNK_SIZE:,} rows)...")

        # Use already-loaded group data (passed as argument)
        group_x_data = group_data["x"]
        group_y_data = group_data["y"]

        # Process features in chunks
        if idx_num and "x" in group_data:
            for start in range(0, group_len, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, group_len)
                chunk_mask = feature_mask[start:end]

                # Skip if no data in this chunk
                if not chunk_mask.any():
                    continue

                # Extract numeric features from this chunk
                chunk_data = group_x_data[start:end]
                numeric_features = chunk_data[chunk_mask, :][:, idx_num].numpy()

                if numeric_features.size > 0:
                    self.feature_scaler.partial_fit(numeric_features)
                    total_timesteps += numeric_features.shape[0]

                # Free memory
                del numeric_features, chunk_data

        # Process targets in chunks
        if self.target_scaler and "y" in group_data:
            for start in range(0, group_len, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, group_len)
                chunk_mask = target_mask[start:end]

                # Skip if no data in this chunk
                if not chunk_mask.any():
                    continue

                # Extract targets from this chunk
                chunk_data = group_y_data[start:end]
                targets = chunk_data[chunk_mask].numpy()

                if targets.size > 0:
                    self.target_scaler.partial_fit(targets)

                # Free memory
                del targets, chunk_data

        return total_timesteps

    def _fit_scaler_batched(self, train_indices):
        """
        Hybrid scaler fitting with automatic optimization strategy selection.

        Uses range extraction for dense windows (step_size <= window_size) and
        boolean mask for sparse windows (step_size > window_size).
        This method uses partial_fit() for memory_efficient=True mode.

        For extremely large groups that don't fit in memory, it automatically falls back
        to chunked processing.

        Args:
            train_indices: List of D2 window indices for training
        """
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        from collections import defaultdict

        # Try to import psutil for memory checking, but don't fail if not available
        try:
            import psutil

            memory_check_available = True
        except ImportError:
            memory_check_available = False
            logger.debug("psutil not available, memory checking disabled")

        # Simple check: if step_size > window_size, there are gaps
        has_gaps = self._has_window_gaps()
        method_name = "boolean_mask" if has_gaps else "range_extraction"

        logger.info(
            f"Fitting {self.scaling_method} scaler using partial_fit on training windows (Hybrid Method: {method_name})..."
        )

        # Get metadata
        meta = getattr(self.d1_dataset, "metadata", {})
        idx_cat_past = list(meta.get("idx_cat_past", []))
        feature_cols = meta.get("feature_cols", [])
        n_features_total = len(feature_cols)
        idx_num = [i for i in range(n_features_total) if i not in idx_cat_past]

        # --- Step 1: Collate all training windows by their D1 group ---
        windows_by_group = defaultdict(list)
        for window_idx in train_indices:
            window = self.valid_windows[window_idx]
            windows_by_group[window["group_idx"]].append(window)

        logger.info(f"  Fitting on {len(train_indices)} windows across {len(windows_by_group)} D1 groups using partial_fit...")

        total_timesteps = 0
        chunked_count = 0

        # --- Step 2: Iterate per-group with hybrid optimization ---
        for i, (group_idx, windows_in_group) in enumerate(windows_by_group.items()):
            logger.info(f"  Processing group {i+1}/{len(windows_by_group)} (Group Index: {group_idx})...")

            # Get group data
            group_data = self.d1_dataset[group_idx]
            group_len = len(group_data["x"])

            # --- Step 3: Memory check (if psutil available) ---
            use_chunked = False
            if memory_check_available:
                mem_avail = psutil.virtual_memory().available
                # Estimate memory needed: rows * features * 8 bytes (float64)
                row_bytes = group_len * n_features_total * 8
                memory_threshold = 0.8 * mem_avail

                if row_bytes > memory_threshold:
                    use_chunked = True
                    chunked_count += 1
                    logger.warning(
                        f"    Group {group_idx} requires ~{row_bytes / (1024**3):.2f} GB "
                        f"(>{memory_threshold / (1024**3):.2f} GB available) - using chunked processing"
                    )

            if use_chunked:
                # --- Step 4a: Chunked fallback for very large groups ---
                # Create boolean mask for chunked processing
                feature_mask = np.zeros(group_len, dtype=bool)
                target_mask = np.zeros(group_len, dtype=bool)

                for window in windows_in_group:
                    start_idx = window["start_idx"]
                    past_len = window["past_len"]
                    past_end_idx = start_idx + past_len
                    feature_mask[start_idx:past_end_idx] = True
                    future_end_idx = past_end_idx + self.future_len
                    target_mask[past_end_idx:future_end_idx] = True

                timesteps = self._fit_scaler_batched_fallback(group_data, group_len, feature_mask, target_mask, idx_num)
                total_timesteps += timesteps
            else:
                # --- Step 4b: Extract data using appropriate method ---
                group_x_data = group_data["x"]
                group_y_data = group_data["y"]

                if has_gaps:
                    # Use boolean mask for sparse windows
                    numeric_features, targets = self._extract_with_mask(
                        group_x_data, group_y_data, windows_in_group, group_len, idx_num
                    )
                else:
                    # Use range extraction for dense windows
                    numeric_features, targets = self._extract_with_range(group_x_data, group_y_data, windows_in_group, idx_num)

                # --- Step 5: Partial fit ---
                if numeric_features is not None and numeric_features.size > 0:
                    self.feature_scaler.partial_fit(numeric_features)
                    total_timesteps += numeric_features.shape[0]
                    del numeric_features

                if targets is not None and targets.size > 0 and self.target_scaler:
                    self.target_scaler.partial_fit(targets)
                    del targets

        # --- Step 6: Finalize ---
        logger.info(f"  Feature scaler fitted on {total_timesteps} total timesteps from {len(train_indices)} training windows")
        self.is_scaler_fitted = True

    def fit_scaler(self, train_indices):
        """Fit scaler on training data"""
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        if not self.memory_efficient:
            self._fit_scaler_direct_d1(train_indices)
        else:
            self._fit_scaler_batched(train_indices)

        # Attach fitted scalers to dataset for on-the-fly transformation
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
        Pre-transform dataset by directly accessing D1 data.
        Uses pre-calculated categorical indices from D1 metadata for optimization.
        """
        # Get D1 metadata
        meta = getattr(self.d1_dataset, "metadata", {}) or {}

        # Refresh indices from D1 metadata (in case they were updated during enrichment)
        idx_cat_past = list(meta.get("idx_cat_past", []))
        self.idx_cat_past = idx_cat_past  # Update self as well
        self.idx_cat_future = list(meta.get("idx_cat_future", []))

        # feature_cols = meta.get("feature_cols", [])
        idx_future = list(meta.get("idx_future", []))
        idx_targets_full = list(meta.get("idx_targets", []))
        global_forecasting = meta.get("global_forecasting", True)

        # Determine numeric indices
        n_features = int(meta.get("n_features", 0))
        all_idx = list(range(n_features))
        idx_num = [i for i in all_idx if i not in idx_cat_past]

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

        # Store full feature arrays and targets
        all_X_past = X_past_all
        all_X_future = X_future_all
        all_y = y_future_all

        # Calculate future numeric indices based on current idx_num
        # (not using pre-calculated from dataset, which may be stale if metadata was updated)
        future_num_idx = [i for i in idx_num if i in idx_future] if len(idx_num) > 0 else []

        # Transform numeric features in-place in the full arrays
        logger.info("  Applying scaling transformation (single call, fully vectorized)...")

        if self.feature_scaler is not None and hasattr(self.feature_scaler, "n_features_in_") and len(idx_num) > 0:
            # Transform numeric features in all_X_past
            n_windows, past_len, n_features = all_X_past.shape
            # Extract numeric features
            x_num_past = all_X_past[:, :, idx_num]
            # Reshape to 2D: (n_windows * past_len, n_num_features)
            reshaped = x_num_past.reshape(-1, len(idx_num))
            # Transform ALL at once - SINGLE CALL!
            transformed = self.feature_scaler.transform(reshaped)
            # Reshape back and update in-place
            all_X_past[:, :, idx_num] = transformed.reshape(n_windows, past_len, len(idx_num))
            logger.info(
                f"  Transformed {len(indices)} windows × {self.past_len} timesteps = {reshaped.shape[0]} values in single call"
            )

            # Transform numeric features in all_X_future if present
            # Only transform features that were in the training set (idx_num)
            if len(future_num_idx) > 0:
                n_windows, future_len, n_features = all_X_future.shape
                x_num_future = all_X_future[:, :, future_num_idx]
                reshaped = x_num_future.reshape(-1, len(future_num_idx))
                # Only transform if future features match the fitted scaler's feature count
                if len(future_num_idx) == self.feature_scaler.n_features_in_:
                    transformed = self.feature_scaler.transform(reshaped)
                    all_X_future[:, :, future_num_idx] = transformed.reshape(n_windows, future_len, len(future_num_idx))
                else:
                    logger.warning(
                        f"  Skipping future feature scaling: future has {len(future_num_idx)} features "
                        f"but scaler expects {self.feature_scaler.n_features_in_} features"
                    )

        # Transform targets if enabled
        if self.target_scaler is not None and hasattr(self.target_scaler, "n_features_in_"):
            logger.info("  Applying target scaling (single call, fully vectorized)...")
            original_shape = all_y.shape
            reshaped = all_y.reshape(-1, original_shape[-1])
            transformed_y = self.target_scaler.transform(reshaped)
            all_y = transformed_y.reshape(original_shape)

        # Use pre-calculated cardinalities from D1 metadata
        cat_past_cardinalities = meta.get("cat_past_cardinalities", [])
        cat_future_cardinalities = meta.get("cat_future_cardinalities", [])

        # Store in metadata
        meta_updated = meta.copy()
        meta_updated["cat_past_cardinalities"] = cat_past_cardinalities
        meta_updated["cat_future_cardinalities"] = cat_future_cardinalities
        meta_updated["idx_cat_past"] = idx_cat_past
        meta_updated["idx_cat_future"] = self.idx_cat_future

        logger.debug("[D2] Using categorical metadata:")
        logger.debug(f"  cat_past_cardinalities: {cat_past_cardinalities}")
        logger.debug(f"  cat_future_cardinalities: {cat_future_cardinalities}")

        # Pre-compute idx_target mapping once
        num_pos_map = {orig: pos for pos, orig in enumerate(idx_num)}
        mapped_targets = [num_pos_map[i] for i in idx_targets_full if i in num_pos_map]
        idx_target_tensor = torch.tensor(mapped_targets, dtype=torch.long)

        # Return vectorized dataset (creates dicts on-demand in __getitem__)
        logger.info(f"✅ Pre-transformation complete: {len(indices)} samples ready (vectorized storage)")
        from .utils import VectorizedPreTransformedDataset

        return VectorizedPreTransformedDataset(
            all_X_past,
            all_X_future,
            all_y,
            list(range(len(all_window_info))),  # Use sequential indices
            all_window_info,  # Use extracted window info
            idx_num,
            idx_cat_past,
            future_num_idx,
            self.idx_cat_future,  # Use pre-calculated from dataset
            idx_target_tensor,
            global_forecasting,
            meta_updated,
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

        # Normalize group IDs: ensure consistency between split_group_config and window group_ids
        sample_group_id = self.valid_windows[0]["group_id"] if len(self.valid_windows) > 0 else None
        windows_use_tuples = isinstance(sample_group_id, tuple)

        def normalize_group_id(gid):
            # If windows use tuples, convert integers to tuples
            if windows_use_tuples:
                if isinstance(gid, int):
                    return (gid,)
                elif isinstance(gid, tuple):
                    return gid
                else:
                    # Wrap other types in tuple
                    return (gid,)
            # If windows use integers/other, extract from tuples if needed
            else:
                if isinstance(gid, tuple) and len(gid) == 1:
                    return gid[0]
                else:
                    return gid

        train_groups = [normalize_group_id(g) for g in train_groups]
        # Handle None values for val_groups and test_groups
        val_groups = [normalize_group_id(g) for g in val_groups] if val_groups is not None else None
        test_groups = [normalize_group_id(g) for g in test_groups] if test_groups is not None else None

        # CRITICAL: Validate that groups are mutually exclusive (skip if None)
        train_set = set(train_groups)
        val_set = set(val_groups) if val_groups is not None else set()
        test_set = set(test_groups) if test_groups is not None else set()

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

        # --- Determine splitting strategy based on val_groups and test_groups ---
        # Strategy A: If val_groups or test_groups have values → Strict group separation
        # Strategy B: If val_groups and test_groups are empty lists [] → Keep val/test empty (train only)
        # Strategy C: If val_groups and test_groups are None (not provided) → Split train groups by ratio

        val_groups_provided = val_groups is not None and len(val_groups) > 0
        test_groups_provided = test_groups is not None and len(test_groups) > 0
        val_groups_empty = val_groups is not None and len(val_groups) == 0
        test_groups_empty = test_groups is not None and len(test_groups) == 0

        # --- Strategy A: "Strict Group Separation" ---
        # If user provides val or test groups with actual values, use strict separation
        if val_groups_provided or test_groups_provided:
            logger.info("Group-based split: val/test groups provided with values.")

            # 1. Add 100% of val_groups to val_indices
            for group_id in val_groups:
                val_indices.extend(windows_by_group[group_id])
                logger.info(f"[D2 split]   Val group '{group_id}': {len(windows_by_group[group_id])} windows -> 100% to val")

            # 2. Add 100% of test_groups to test_indices
            for group_id in test_groups:
                test_indices.extend(windows_by_group[group_id])
                logger.info(f"[D2 split]   Test group '{group_id}': {len(windows_by_group[group_id])} windows -> 100% to test")

            # 3. Add 100% of train_groups to train_indices (NO SPLITTING WITHIN GROUPS)
            for group_id in train_groups:
                train_indices.extend(windows_by_group[group_id])
                logger.info(f"[D2 split]   Train group '{group_id}': {len(windows_by_group[group_id])} windows -> 100% to train")

            logger.info("Strict group separation complete (no groups split across splits).")

        # --- Strategy B: "Train Only" ---
        # If user provides empty lists for val/test, keep only train (no val/test)
        elif val_groups_empty and test_groups_empty:
            logger.info("Train-only split: val_groups=[] and test_groups=[] provided (keeping val/test empty).")

            # Add 100% of train_groups to train_indices only
            for group_id in train_groups:
                train_indices.extend(windows_by_group[group_id])
                logger.info(f"[D2 split]   Train group '{group_id}': {len(windows_by_group[group_id])} windows -> 100% to train")

            # val_indices and test_indices remain empty
            logger.info("Train-only split complete (val and test are empty).")

        # --- Strategy C: "Temporal Split of Train Groups" ---
        # If user ONLY provides train_groups (val/test are None), split train groups by ratio
        else:
            logger.info("Temporal split: Only train_groups provided (val/test are None). Splitting by split_ratio.")
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
            logger.info("Temporal split of train_groups complete.")

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
