"""
Encoder-Decoder implementation for D2 layer.

This module provides the EncoderDecoder class (formerly TSDataModule) that creates
sliding windows and encoder-decoder structures from D1 layer data.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

# Conditional import for MinMaxScaler
try:
    from sklearn.preprocessing import MinMaxScaler

    MINMAX_SCALER_AVAILABLE = True
except ImportError:
    MINMAX_SCALER_AVAILABLE = False
    MinMaxScaler = None

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
        logger.info("Initializing EncoderDecoderDataset with:")
        logger.info(f"  past_len: {past_len}, future_len: {future_len}")
        logger.info(f"  target_cols: {target_cols}")
        logger.info(f"  cat_cols: {cat_cols}")
        logger.info(f"  cont_feature_cols: {cont_feature_cols}")
        logger.info(f"  cat_feature_cols: {cat_feature_cols}")
        logger.info(f"  include_target_in_decoder: {include_target_in_decoder}")
        logger.info(f"  valid_windows: {len(valid_windows)} windows")

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
                self.cat_feature_cols = d1_dataset.cat_cols or []
                logger.debug(f"Auto-detected {len(self.cat_feature_cols)} categorical feature columns from D1")
            except (AttributeError, TypeError):
                self.cat_feature_cols = []
                logger.warning("Could not auto-detect categorical feature columns from D1")
        else:
            self.cat_feature_cols = cat_feature_cols
            logger.debug(f"Using provided categorical feature columns: {self.cat_feature_cols}")

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

        # Get the full group data from D1 dataset
        group_sample = self.d1_dataset[group_idx]

        # Use D1 metadata as source of truth for indices
        meta: Dict[str, Any] = getattr(self.d1_dataset, "metadata", {}) or {}
        d1_metadata = meta  # Make d1_metadata available in this scope
        idx_categorical: List[int] = list(meta.get("idx_categorical", []))
        feature_cols = meta.get("feature_cols", [])
        enrich_cat = meta.get("enrich_cat", [])

        # Note: Temporal enrichment features are now only available through x_cat_past and x_cat_future
        # They should not be exposed as separate keys in the batch structure

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
        # Check if D1 dataset has scaling parameters
        # d2_layer = getattr(self, "d2_layer", None)
        d1_scaling_method = getattr(self.d1_dataset, "scaling_method", "minmax")

        # For standard scaling with memory_efficient=True, we need to apply scaling on-the-fly
        # For minmax scaling or standard scaling with memory_efficient=False, the scaling should already be applied
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

        # Build clean input dictionary - only include keys when data is present
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
        scaler: Optional[Any] = None,
        scale_targets: bool = False,
        apply_scaling: bool = True,
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
            scaler: Optional scaler for numeric features (default: StandardScaler)
            scale_targets: If True, also scale target variables (default: False)
            apply_scaling: If True, apply min-max scaling using D1 layer scaling parameters (default: True)
        """
        super().__init__()

        logger.info("Initializing EncoderDecoder layer")
        logger.debug(f"  past_len: {past_len}, future_len: {future_len}")
        logger.debug(f"  batch_size: {batch_size}, step_size: {step_size}")
        logger.debug(f"  split_method: {split_method}, precompute: {precompute}")

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
        self.apply_scaling = apply_scaling

        # Initialize scaler and determine scaling method
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_scaler_fitted = False
        self.use_minmax_scaler = False
        self.use_standard_scaler = False
        self.d1_scaling_method = getattr(d1_dataset, "scaling_method", "minmax")

        # Check if MinMaxScaler is provided and available
        if MINMAX_SCALER_AVAILABLE and isinstance(self.scaler, MinMaxScaler):
            self.use_minmax_scaler = True
            logger.info("Using sklearn MinMaxScaler for scaling")
        elif apply_scaling:
            # Determine scaling method from D1 layer
            if self.d1_scaling_method == "standard":
                self.use_standard_scaler = True
                logger.info("Using standard scaling implementation from D1 layer")
            else:
                logger.info("Using custom min-max scaling implementation")

        # Get scaling parameters from D1 layer
        self.scaling_params = {}
        if apply_scaling and hasattr(d1_dataset, "get_scaling_params"):
            self.scaling_params = d1_dataset.get_scaling_params()
            if self.scaling_params:
                logger.info(f"Retrieved scaling parameters for {len(self.scaling_params)} columns from D1 layer")
                # Log scaling method from parameters
                first_param = next(iter(self.scaling_params.values()), {})
                scaler_type = first_param.get("scaler_type", "minmax")
                logger.info(f"D1 layer scaling method: {scaler_type}")
            else:
                logger.info("No scaling parameters found in D1 layer")
        elif apply_scaling:
            logger.warning("apply_scaling=True but D1 dataset does not support scaling parameters")

        # Extract column information from D1 dataset
        self.known_cols = d1_dataset.known_cols
        self.unknown_cols = d1_dataset.unknown_cols
        self.group_cols = d1_dataset.group_cols or []
        self.target_cols = d1_dataset.target_cols
        self.feature_cols = d1_dataset.feature_cols

        # Log only essential column information
        logger.debug("Column information from D1 dataset:")
        logger.debug(f"  known_cols: {len(self.known_cols)} cols, group_cols: {len(self.group_cols)} cols")
        logger.debug(f"  target_cols: {self.target_cols}")

        # Handle potentially None or empty categorical columns
        try:
            self.cat_cols = d1_dataset.cat_cols if d1_dataset.cat_cols else []
            logger.debug(f"  cat_cols from D1: {len(self.cat_cols)} cols")
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

        logger.debug("Derived column classifications:")
        logger.debug(
            f"  cat_feature_cols: {len(self.cat_feature_cols)} cols, cont_feature_cols: {len(self.cont_feature_cols)} cols"
        )

        # Build valid windows
        logger.info("Building valid windows from D1 dataset...")
        self._build_valid_windows()

        # Apply scaling to D1 dataset if requested
        if self.apply_scaling:
            if self.use_minmax_scaler:
                logger.info("Fitting and applying sklearn MinMaxScaler to D1 dataset...")
                self._fit_and_apply_sklearn_scaler()
            elif self.use_standard_scaler:
                logger.info("Using standard scaling from D1 dataset (already applied)...")
                # Standard scaling is already applied in D1 layer, no additional action needed
            elif self.scaling_params:
                scaling_method = self.d1_scaling_method
                logger.info(f"Applying {scaling_method} scaling to D1 dataset...")
                self._apply_scaling_to_d1_dataset()
            else:
                logger.warning("Scaling requested but no scaling method available")

        logger.info(f"EncoderDecoder initialized with {len(self.valid_windows)} windows")

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

    def _apply_scaling_to_d1_dataset(self):
        """
        Apply min-max scaling to the D1 dataset using precomputed scaling parameters.

        This method modifies the D1 dataset in-place by applying min-max scaling
        to numerical features based on global min/max values computed during D1 initialization.
        """
        if not self.scaling_params:
            logger.warning("No scaling parameters available for scaling")
            return

        logger.info(f"Applying min-max scaling to {len(self.scaling_params)} numerical columns")

        # Get feature column information from D1 metadata
        d1_metadata = getattr(self.d1_dataset, "metadata", {})
        feature_cols = d1_metadata.get("feature_cols", [])

        if not feature_cols:
            logger.warning("No feature columns found in D1 metadata")
            return

        # Apply scaling to each group in the D1 dataset
        total_groups = len(self.d1_dataset)
        scaled_groups = 0

        for group_idx in range(total_groups):
            try:
                # Get group data
                if hasattr(self.d1_dataset, "cached_data") and self.d1_dataset.cached_data:
                    # Use cached data if available
                    file_group_key = self.d1_dataset._group_ids[group_idx]
                    if file_group_key in self.d1_dataset.cached_data:
                        group_data = self.d1_dataset.cached_data[file_group_key]

                        # Apply scaling to numerical columns
                        scaled_data = self._apply_scaling_to_dataframe(group_data, feature_cols)

                        # Update cached data
                        self.d1_dataset.cached_data[file_group_key] = scaled_data
                        scaled_groups += 1
                else:
                    # For memory-efficient mode, we'll apply scaling on-the-fly during __getitem__
                    # This is handled in the _apply_scaling_to_tensor method
                    pass

            except Exception as e:
                logger.warning(f"Failed to apply scaling to group {group_idx}: {e}")

        if scaled_groups > 0:
            logger.info(f"Applied scaling to {scaled_groups} cached groups")
        else:
            logger.info("Scaling will be applied on-the-fly during data loading (memory-efficient mode)")

    def _apply_scaling_to_dataframe(self, df, feature_cols):
        """
        Apply min-max scaling to a DataFrame.

        Args:
            df: DataFrame to scale
            feature_cols: List of feature column names

        Returns:
            Scaled DataFrame
        """
        scaled_df = df.copy()

        for col in feature_cols:
            if col in self.scaling_params and col in scaled_df.columns:
                params = self.scaling_params[col]

                if not params["is_constant"]:
                    # Apply min-max scaling: (x - min) / (max - min)
                    scaled_df[col] = (scaled_df[col] - params["min"]) / params["range"]
                else:
                    # For constant columns, set to 0
                    scaled_df[col] = 0.0

                logger.debug(f"Applied scaling to column '{col}'")

        return scaled_df

    def _apply_scaling_to_tensor(self, tensor, feature_cols):
        """
        Apply min-max scaling to a tensor.

        Args:
            tensor: PyTorch tensor to scale [seq_len, n_features]
            feature_cols: List of feature column names

        Returns:
            Scaled tensor
        """
        if not self.scaling_params or tensor.numel() == 0:
            return tensor

        scaled_tensor = tensor.clone()

        for i, col in enumerate(feature_cols):
            if col in self.scaling_params and i < tensor.shape[-1]:
                params = self.scaling_params[col]

                if not params["is_constant"]:
                    # Apply min-max scaling: (x - min) / (max - min)
                    scaled_tensor[..., i] = (tensor[..., i] - params["min"]) / params["range"]
                else:
                    # For constant columns, set to 0
                    scaled_tensor[..., i] = 0.0

        return scaled_tensor

    def get_scaling_params(self):
        """
        Get the scaling parameters used by this D2 layer.

        Returns:
            Dictionary containing scaling parameters for each numerical column
        """
        return self.scaling_params

    def apply_inverse_scaling(self, data, columns=None):
        """
        Apply inverse scaling to denormalize predictions (supports both minmax and standard scaling).

        Args:
            data: Data to denormalize (numpy array, pandas DataFrame, or torch tensor)
            columns: List of column names (required if data is numpy array or tensor)

        Returns:
            Denormalized data in the same format as input
        """
        if self.use_minmax_scaler:
            return self.apply_inverse_sklearn_scaling(data, columns)
        elif self.use_standard_scaler or self.d1_scaling_method == "standard":
            # Use D1 layer's inverse scaling (handles both minmax and standard)
            if hasattr(self.d1_dataset, "apply_inverse_scaling"):
                return self.d1_dataset.apply_inverse_scaling(data, columns)
            else:
                logger.warning("D1 dataset does not support inverse scaling")
                return data
        elif hasattr(self.d1_dataset, "apply_inverse_scaling"):
            return self.d1_dataset.apply_inverse_scaling(data, columns)
        else:
            logger.warning("No inverse scaling method available")
            return data

    def _fit_and_apply_sklearn_scaler(self):
        """
        Fit and apply sklearn MinMaxScaler to the D1 dataset.

        This method collects all numerical data, fits the scaler, and applies scaling.
        """
        if not self.use_minmax_scaler:
            logger.warning("sklearn MinMaxScaler not available or not selected")
            return

        logger.info("Fitting sklearn MinMaxScaler on D1 dataset...")

        # Get feature column information from D1 metadata
        d1_metadata = getattr(self.d1_dataset, "metadata", {})
        feature_cols = d1_metadata.get("feature_cols", [])
        num_cols = getattr(self.d1_dataset, "num_cols", [])

        # Filter to only numerical columns (exclude categorical and time columns)
        scalable_cols = []
        time_col = getattr(self.d1_dataset, "_time_col", "time")
        group_cols = getattr(self.d1_dataset, "group_cols", [])

        for col in num_cols:
            if col != time_col and col not in group_cols:
                scalable_cols.append(col)

        if not scalable_cols:
            logger.warning("No scalable numerical columns found for sklearn MinMaxScaler")
            return

        logger.info(f"Collecting data for {len(scalable_cols)} columns: {scalable_cols}")

        # Collect all numerical data for fitting
        all_numerical_data = []
        total_groups = len(self.d1_dataset)

        for group_idx in range(total_groups):
            try:
                group_sample = self.d1_dataset[group_idx]
                if "x" in group_sample:
                    # Extract numerical features based on feature column indices
                    x_data = group_sample["x"]  # [seq_len, n_features]

                    # Get indices of scalable columns in feature_cols
                    scalable_indices = []
                    for col in scalable_cols:
                        if col in feature_cols:
                            scalable_indices.append(feature_cols.index(col))

                    if scalable_indices and x_data.shape[1] > max(scalable_indices):
                        # Extract only the scalable numerical columns
                        numerical_data = x_data[:, scalable_indices].numpy()
                        all_numerical_data.append(numerical_data)

            except Exception as e:
                logger.warning(f"Failed to collect data from group {group_idx}: {e}")

        if not all_numerical_data:
            logger.warning("No numerical data collected for sklearn MinMaxScaler")
            return

        # Combine all data and fit scaler
        combined_data = np.vstack(all_numerical_data)
        logger.info(f"Fitting MinMaxScaler on {combined_data.shape[0]} samples with {combined_data.shape[1]} features")

        self.scaler.fit(combined_data)
        self.is_scaler_fitted = True
        self.scalable_cols = scalable_cols
        self.scalable_indices = scalable_indices

        logger.info("sklearn MinMaxScaler fitted successfully")

        # Apply scaling to cached data if available
        if hasattr(self.d1_dataset, "cached_data") and self.d1_dataset.cached_data:
            self._apply_sklearn_scaling_to_cached_data()

    def _apply_sklearn_scaling_to_cached_data(self):
        """
        Apply sklearn MinMaxScaler to cached data in D1 dataset.
        """
        if not self.is_scaler_fitted:
            logger.warning("Scaler not fitted, cannot apply scaling")
            return

        logger.info("Applying sklearn MinMaxScaler to cached data...")
        scaled_groups = 0

        for file_group_key, group_data in self.d1_dataset.cached_data.items():
            try:
                # Apply scaling to DataFrame
                scaled_data = group_data.copy()

                for i, col in enumerate(self.scalable_cols):
                    if col in scaled_data.columns:
                        # Transform single column
                        col_data = scaled_data[[col]].values
                        scaled_col_data = self.scaler.transform(col_data[:, [i]])
                        scaled_data[col] = scaled_col_data.flatten()

                self.d1_dataset.cached_data[file_group_key] = scaled_data
                scaled_groups += 1

            except Exception as e:
                logger.warning(f"Failed to apply sklearn scaling to group {file_group_key}: {e}")

        logger.info(f"Applied sklearn MinMaxScaler to {scaled_groups} cached groups")

    def apply_inverse_sklearn_scaling(self, data, columns=None):
        """
        Apply inverse sklearn MinMaxScaler to denormalize predictions.

        Args:
            data: Data to denormalize (numpy array, pandas DataFrame, or torch tensor)
            columns: List of column names (required if data is numpy array or tensor)

        Returns:
            Denormalized data in the same format as input
        """
        if not self.is_scaler_fitted:
            logger.warning("Scaler not fitted, cannot apply inverse scaling")
            return data

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            for i, col in enumerate(self.scalable_cols):
                if col in data.columns:
                    col_data = data[[col]].values
                    unscaled_col_data = self.scaler.inverse_transform(col_data[:, [i]])
                    result[col] = unscaled_col_data.flatten()
            return result

        elif isinstance(data, np.ndarray) and columns is not None:
            result = data.copy()
            for i, col in enumerate(columns):
                if col in self.scalable_cols and i < data.shape[1]:
                    scaler_idx = self.scalable_cols.index(col)
                    col_data = data[:, [i]]
                    # Create dummy array with correct shape for inverse_transform
                    dummy_data = np.zeros((col_data.shape[0], len(self.scalable_cols)))
                    dummy_data[:, scaler_idx] = col_data.flatten()
                    unscaled_dummy = self.scaler.inverse_transform(dummy_data)
                    result[:, i] = unscaled_dummy[:, scaler_idx]
            return result

        elif hasattr(data, "clone"):  # PyTorch tensor
            result = data.clone()
            if columns is not None:
                for i, col in enumerate(columns):
                    if col in self.scalable_cols and i < data.shape[-1]:
                        scaler_idx = self.scalable_cols.index(col)
                        col_data = data[..., i].numpy()
                        # Create dummy array for inverse_transform
                        dummy_data = np.zeros((col_data.shape[0], len(self.scalable_cols)))
                        dummy_data[:, scaler_idx] = col_data.flatten()
                        unscaled_dummy = self.scaler.inverse_transform(dummy_data)
                        result[..., i] = torch.from_numpy(unscaled_dummy[:, scaler_idx])
            return result

        else:
            logger.warning("Unsupported data type for sklearn inverse scaling")
            return data

    def _build_valid_windows(self):
        """
        Build valid sliding windows from the D1 dataset.

        The D1 dataset returns all data for a group as a single sample.
        We need to extract individual timesteps from each group to create windows.
        """
        logger.info("Building valid windows from D1 dataset...")
        self.valid_windows = []
        total_groups = len(self.d1_dataset)

        windows_per_group = {}
        insufficient_groups = []

        # Process each group in the D1 dataset
        for group_idx in range(total_groups):
            group_sample = self.d1_dataset[group_idx]
            group_id = group_sample.get("group_id", group_idx)
            seq_len = group_sample.get("seq_len", 0)

            logger.info(f"DEBUG: Processing group {group_id} (seq_len={seq_len})")
            logger.info(f"DEBUG: Group sample keys: {list(group_sample.keys())}")
            logger.info(f"DEBUG: past_len={self.past_len}, future_len={self.future_len}")

            # Create sliding windows within this group's sequence
            max_windows = seq_len - self.past_len - self.future_len + 1
            logger.info(
                f"DEBUG: Group {group_id}: {max_windows} windows possible (seq_len={seq_len}, required={self.past_len + self.future_len})"  # noqa
            )

            if max_windows > 0:
                group_windows = 0
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
                    group_windows += 1

                    # Log every 100 windows to avoid excessive output
                    if len(self.valid_windows) % 100 == 0:
                        logger.debug(
                            f"Added {len(self.valid_windows)} windows so far" f"(current: group {group_id}, position {i})"
                        )

                    # Limit samples per group if specified
                    if (
                        self.max_samples_per_group
                        and len([w for w in self.valid_windows if w["group_id"] == group_id]) >= self.max_samples_per_group
                    ):
                        logger.info(f"Reached max_samples_per_group ({self.max_samples_per_group})" f"for group {group_id}")
                        break

                windows_per_group[group_id] = group_windows
            else:
                insufficient_groups.append(group_id)
                logger.warning(
                    f"Group {group_id} has insufficient data for windows (seq_len={seq_len},"
                    f" required={self.past_len + self.future_len})"
                )

        # Summary statistics
        logger.info(f"Created {len(self.valid_windows)} windows from {len(windows_per_group)} groups")
        if insufficient_groups:
            logger.debug(f"Insufficient groups: {len(insufficient_groups)}")

        # Log distribution of windows per group
        if windows_per_group and logger.isEnabledFor(logging.DEBUG):
            min_windows = min(windows_per_group.values())
            max_windows = max(windows_per_group.values())
            avg_windows = sum(windows_per_group.values()) / len(windows_per_group)
            logger.debug(f"Windows per group - Min: {min_windows}, Max: {max_windows}, Avg: {avg_windows:.1f}")

            # Log groups with most and least windows
            if len(windows_per_group) > 1:
                most_windows_group = max(windows_per_group.items(), key=lambda x: x[1])[0]
                least_windows_group = min(windows_per_group.items(), key=lambda x: x[1])[0]
                # Move detailed window statistics to debug level
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Group distribution - Most: {most_windows_group} ({windows_per_group[most_windows_group]}), "
                        f"Least: {least_windows_group} ({windows_per_group[least_windows_group]})"
                    )
        elif len(self.valid_windows) == 0:
            logger.warning("No valid windows could be created from any group")

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

    def fit_scaler(self, dataset: EncoderDecoderSubset):
        """
        Fit the scaler on numeric features from the given dataset.

        Args:
            dataset: Dataset to fit the scaler on (typically training dataset)
        """
        if not hasattr(self, "scaler") or self.scaler is None:
            logger.info("No scaler provided, skipping scaler fitting")
            return

        logger.info(f"Fitting {type(self.scaler).__name__} on training data")

        # Collect all numeric features from the dataset
        all_numeric_features = []

        for idx in range(len(dataset)):
            x, _ = dataset[idx]
            if "x_num_past" in x and x["x_num_past"].shape[1] > 0:
                # Extract numeric features and reshape to 2D array
                numeric_features = x["x_num_past"].numpy().reshape(-1, x["x_num_past"].shape[1])
                all_numeric_features.append(numeric_features)

        if not all_numeric_features:
            logger.warning("No numeric features found in dataset, cannot fit scaler")
            return

        # Concatenate all features and fit the scaler
        all_features = np.vstack(all_numeric_features)
        logger.info(f"Fitting scaler on {all_features.shape[0]} samples" f" with {all_features.shape[1]} features")

        self.scaler.fit(all_features)
        self.is_scaler_fitted = True
        logger.info("Scaler fitted successfully")

        # If scale_targets is True, fit another scaler for target variables
        if self.scale_targets:
            logger.info("Fitting target scaler")
            self.target_scaler = StandardScaler()

            all_targets = []
            for idx in range(len(dataset)):
                x, y = dataset[idx]
                if isinstance(y, torch.Tensor) and y.numel() > 0:
                    targets = y.numpy().reshape(-1, y.shape[-1])
                    all_targets.append(targets)

            if all_targets:
                all_targets = np.vstack(all_targets)
                logger.info(f"Fitting target scaler on {all_targets.shape[0]}" f" samples with {all_targets.shape[1]} features")
                self.target_scaler.fit(all_targets)
                logger.info("Target scaler fitted successfully")
            else:
                logger.warning("No targets found in dataset, cannot fit target scaler")
                self.scale_targets = False

    def transform_with_scaler(self, dataset: EncoderDecoderSubset) -> EncoderDecoderSubset:
        """
        Transform the numeric features in the dataset using the fitted scaler.

        Args:
            dataset: Dataset to transform

        Returns:
            Transformed dataset
        """
        if not self.is_scaler_fitted:
            logger.warning("Scaler not fitted, returning original dataset")
            return dataset

        logger.info(f"Transforming dataset with {type(self.scaler).__name__}")

        # Create a new dataset with transformed features
        class ScaledEncoderDecoderSubset(EncoderDecoderSubset):
            def __init__(self, original_subset, scaler, target_scaler=None, scale_targets=False):
                self.dataset = original_subset.dataset
                self.indices = original_subset.indices
                self.scaler = scaler
                self.target_scaler = target_scaler
                self.scale_targets = scale_targets

            def __getitem__(self, idx):
                x, y = super().__getitem__(idx)

                # Transform numeric features
                if "x_num_past" in x and x["x_num_past"].shape[1] > 0:
                    # Get original shape for reshaping back after transformation
                    orig_shape = x["x_num_past"].shape

                    # Reshape to 2D for transformation
                    features = x["x_num_past"].numpy().reshape(-1, orig_shape[1])

                    # Transform and reshape back
                    transformed = self.scaler.transform(features)
                    x["x_num_past"] = torch.tensor(transformed.reshape(orig_shape), dtype=torch.float32)

                    # Also update the backward compatibility key if present
                    if "past_features" in x:
                        x["past_features"] = x["x_num_past"]

                # Transform targets if requested
                if self.scale_targets and self.target_scaler is not None:
                    if isinstance(y, torch.Tensor) and y.numel() > 0:
                        orig_shape = y.shape
                        targets = y.numpy().reshape(-1, orig_shape[-1])
                        transformed = self.target_scaler.transform(targets)
                        y = torch.tensor(transformed.reshape(orig_shape), dtype=torch.float32)

                        # Also update the target in x dictionary
                        if "y" in x:
                            x["y"] = y
                        if "future_targets" in x:
                            x["future_targets"] = y

                return x, y

        # Create and return the scaled dataset
        target_scaler = getattr(self, "target_scaler", None) if self.scale_targets else None
        return ScaledEncoderDecoderSubset(dataset, self.scaler, target_scaler, self.scale_targets)

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
        logger.info("========== SPLITTING DATASET ==========")
        logger.info(f"Split method: {method}")
        logger.info(f"Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

        total_samples = len(self.dataset)
        logger.info(f"Total samples available: {total_samples}")

        # Verify ratios sum to 1
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            logger.warning(f"Split ratios sum to {ratio_sum:.3f}, not 1.0. This may cause unexpected behavior.")

        if method == "temporal":
            logger.info("Using temporal split (earlier data for training, later for validation/test)")
            # Temporal split - earlier data for training, later for validation/test
            train_end = int(total_samples * train_ratio)
            val_end = int(total_samples * (train_ratio + val_ratio))

            logger.info(f"Split points: train_end={train_end}, val_end={val_end}")

            train_indices = list(range(0, train_end))
            val_indices = list(range(train_end, val_end))
            test_indices = list(range(val_end, total_samples))

            # Log window information for each split
            if train_indices:
                first_train = self.valid_windows[train_indices[0]]
                last_train = self.valid_windows[train_indices[-1]]
                logger.info(f"Train data spans from window {first_train['start_idx']}" f"to {last_train['start_idx']}")

            if val_indices:
                first_val = self.valid_windows[val_indices[0]]
                last_val = self.valid_windows[val_indices[-1]]
                logger.info(f"Validation data spans from window {first_val['start_idx']}" f"to {last_val['start_idx']}")

            if test_indices:
                first_test = self.valid_windows[test_indices[0]]
                last_test = self.valid_windows[test_indices[-1]]
                logger.info(f"Test data spans from window {first_test['start_idx']}" f" to {last_test['start_idx']}")

        else:
            logger.info("Using random split (shuffled indices)")
            # Random split
            np.random.seed(42)  # For reproducibility
            indices = np.random.permutation(total_samples)
            train_end = int(total_samples * train_ratio)
            val_end = int(total_samples * (train_ratio + val_ratio))

            logger.info(f"Split points after shuffling: train_end={train_end}, val_end={val_end}")

            train_indices = indices[:train_end].tolist()
            val_indices = indices[train_end:val_end].tolist()
            test_indices = indices[val_end:].tolist()

            logger.info(f"Shuffled indices for random split (showing first 5): {train_indices[:5]}...")

        # Log group distribution in each split
        train_groups = set(self.valid_windows[i]["group_id"] for i in train_indices)
        val_groups = set(self.valid_windows[i]["group_id"] for i in val_indices)
        test_groups = set(self.valid_windows[i]["group_id"] for i in test_indices)

        # Check for group overlap
        train_val_overlap = train_groups.intersection(val_groups)
        train_test_overlap = train_groups.intersection(test_groups)
        val_test_overlap = val_groups.intersection(test_groups)

        logger.info("Group distribution in splits:")
        logger.info(f"  Train: {len(train_groups)} unique groups")
        logger.info(f"  Val: {len(val_groups)} unique groups")
        logger.info(f"  Test: {len(test_groups)} unique groups")
        logger.info("Group overlap between splits:")
        logger.info(f"  Train-Val overlap: {len(train_val_overlap)} groups")
        logger.info(f"  Train-Test overlap: {len(train_test_overlap)} groups")
        logger.info(f"  Val-Test overlap: {len(val_test_overlap)} groups")

        logger.info(
            f"Split statistics: Train: {len(train_indices)} samples ({train_ratio:.1%}), "
            f"Validation: {len(val_indices)} samples ({val_ratio:.1%}), "
            f"Test: {len(test_indices)} samples ({test_ratio:.1%})"
        )

        train_dataset = EncoderDecoderSubset(self.dataset, train_indices)
        val_dataset = EncoderDecoderSubset(self.dataset, val_indices)
        test_dataset = EncoderDecoderSubset(self.dataset, test_indices)

        # Fit scaler on training data only
        self.fit_scaler(train_dataset)

        # Apply scaler to all datasets if fitted
        if self.is_scaler_fitted:
            logger.info("Applying fitted scaler to all datasets")
            train_dataset = self.transform_with_scaler(train_dataset)
            val_dataset = self.transform_with_scaler(val_dataset)
            test_dataset = self.transform_with_scaler(test_dataset)

        logger.info("Dataset split complete")
        logger.info("============================================")

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
                logger.info("Applying fitted scaler to all datasets")
                self.train_dataset = self.transform_with_scaler(self.train_dataset)
                self.val_dataset = self.transform_with_scaler(self.val_dataset)
                self.test_dataset = self.transform_with_scaler(self.test_dataset)

            logger.info(
                f"Setup completed with split statistics: Train: {len(train_indices)}, "
                f"Validation: {len(val_indices)}, Test: {len(test_indices)}"
            )

    def train_dataloader(self):
        """Return the training dataloader."""
        from .utils import custom_collate_fn

        logger.info("Creating training dataloader")

        if self.train_dataset is None:
            # If no explicit split was provided, use all data for training
            logger.info("No explicit train split found, using all data for training")
            self.train_dataset = EncoderDecoderSubset(self.dataset, list(range(len(self.valid_windows))))

        logger.info(f"Training dataset size: {len(self.train_dataset)} samples")
        logger.info(f"Batch size: {self.batch_size}, Num workers: {self.num_workers}")
        logger.info(f"Using custom collate function: {custom_collate_fn.__name__}")
        logger.info(f"Shuffle: True, Custom sampler: {self.sampler is not None}")

        if self.sampler is not None:
            logger.info(f"Using custom sampler: {type(self.sampler).__name__}")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.sampler is None else False,  # Don't shuffle when using a sampler
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            sampler=self.sampler,
        )

        logger.info(f"Created training dataloader with {len(dataloader)} batches")
        return dataloader

    def val_dataloader(self):
        """Return the validation dataloader."""
        from .utils import custom_collate_fn

        logger.info("Creating validation dataloader")

        if self.val_dataset is None or len(self.val_dataset) == 0:
            logger.info("No validation dataset available, skipping validation dataloader creation")
            return None

        logger.info(f"Validation dataset size: {len(self.val_dataset)} samples")
        logger.info(f"Batch size: {self.batch_size}, Num workers: {self.num_workers}")
        logger.info(f"Using custom collate function: {custom_collate_fn.__name__}")
        logger.info("Shuffle: False")

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

        logger.info(f"Created validation dataloader with {len(dataloader)} batches")
        return dataloader

    def test_dataloader(self):
        """Return the test dataloader."""
        from .utils import custom_collate_fn

        logger.info("Creating test dataloader")

        if self.test_dataset is None or len(self.test_dataset) == 0:
            logger.info("No test dataset available, skipping test dataloader creation")
            return None

        logger.info(f"Test dataset size: {len(self.test_dataset)} samples")
        logger.info(f"Batch size: {self.batch_size}, Num workers: {self.num_workers}")
        logger.info(f"Using custom collate function: {custom_collate_fn.__name__}")
        logger.info("Shuffle: False")

        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

        logger.info(f"Created test dataloader with {len(dataloader)} batches")
        return dataloader
