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
        # Works for both memory_efficient=True and False modes
        if self.feature_scaler is not None and hasattr(self.feature_scaler, "n_features_in_"):
            # Scale x_num_past
            if "x_num_past" in x and x["x_num_past"].numel() > 0:
                x_num_past_np = x["x_num_past"].numpy()
                x_num_past_scaled = self.feature_scaler.transform(x_num_past_np)
                x["x_num_past"] = torch.from_numpy(x_num_past_scaled).float()

            # Scale x_num_future (if present)
            if "x_num_future" in x and x["x_num_future"].numel() > 0:
                x_num_future_np = x["x_num_future"].numpy()
                x_num_future_scaled = self.feature_scaler.transform(x_num_future_np)
                x["x_num_future"] = torch.from_numpy(x_num_future_scaled).float()

        # Apply target scaling if enabled
        if self.scale_targets and self.target_scaler is not None and hasattr(self.target_scaler, "n_features_in_"):
            y_np = y.numpy()
            y_scaled = self.target_scaler.transform(y_np)
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
            split_config: Configuration for splits
            num_workers: Number of workers for dataloaders
            sampler: Optional sampler for training dataloader
            target_normalizer: Optional normalizer for targets
            max_samples_per_group: Maximum samples per group
            precompute: Whether to precompute valid windows
            include_target_in_decoder: If True, include target in decoder part (for some models)
            scaling_method: Scaling method string ("standard" or "minmax" or None)
                          Scaling workflow:
                          1. User provides scaling_method string
                          2. Scaler fitted on training data after split_data()
                             - If memory_efficient=True: Uses partial_fit on chunks
                             - If memory_efficient=False: Uses fit on all training data
                          3. Transformation:
                             - If memory_efficient=True: Applied on-the-fly in __getitem__()
                             - If memory_efficient=False: Pre-applied to all splits after fitting
            scale_targets: If True, also scale target variables (default: False)
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
        Uses batches for efficiency in both partial_fit and fit modes.

        Args:
            dataset: Training dataset to fit the scaler on
        """
        if self.feature_scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return

        import numpy as np

        # Use DataLoader for efficient batch processing
        # Larger batch size improves performance
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

        use_partial_fit = getattr(self.d1_dataset, "memory_efficient", False)

        if use_partial_fit:
            logger.info(f"Fitting {self.scaling_method} scaler using partial_fit (memory-efficient mode)...")

            for batch in dataloader:
                # Batch is a dictionary with batched tensors

                # Combine past and future numeric features for fitting
                numeric_features_batch = []
                if "x_num_past" in batch and batch["x_num_past"].numel() > 0:
                    numeric_features_batch.append(batch["x_num_past"])
                if "x_num_future" in batch and batch["x_num_future"].numel() > 0:
                    numeric_features_batch.append(batch["x_num_future"])

                # Fit feature scaler on the combined batch
                if numeric_features_batch:
                    # Shape: (batch_size, seq_len, num_features) -> (batch_size * seq_len, num_features)
                    combined_features = torch.cat(numeric_features_batch, dim=1)
                    features_np = combined_features.reshape(-1, combined_features.shape[-1]).numpy()
                    if features_np.size > 0:
                        self.feature_scaler.partial_fit(features_np)

                # Fit target scaler if requested
                if self.scale_targets and self.target_scaler is not None:
                    if "y" in batch and batch["y"].numel() > 0:
                        targets_np = batch["y"].reshape(-1, batch["y"].shape[-1]).numpy()
                        self.target_scaler.partial_fit(targets_np)

            logger.info(f"Scaler fitted successfully on {len(dataset)} samples using partial_fit")

        else:  # Batch mode (fit)
            logger.info(f"Fitting {self.scaling_method} scaler on training data (batch mode)...")

            all_features_list = []
            all_targets_list = []

            for batch in dataloader:
                # Combine past and future numeric features
                numeric_features_batch = []
                if "x_num_past" in batch and batch["x_num_past"].numel() > 0:
                    numeric_features_batch.append(batch["x_num_past"])
                if "x_num_future" in batch and batch["x_num_future"].numel() > 0:
                    numeric_features_batch.append(batch["x_num_future"])

                if numeric_features_batch:
                    combined_features = torch.cat(numeric_features_batch, dim=1)
                    all_features_list.append(combined_features.numpy())

                if self.scale_targets and "y" in batch and batch["y"].numel() > 0:
                    all_targets_list.append(batch["y"].numpy())

            # Fit feature scaler on all collected data
            if all_features_list:
                features_array = np.concatenate(all_features_list, axis=0)
                features_array = features_array.reshape(-1, features_array.shape[-1])
                self.feature_scaler.fit(features_array)
                logger.info(
                    f"Feature scaler fitted on {features_array.shape[0]} total timesteps with {features_array.shape[1]} features"
                )

            # Fit target scaler
            if self.scale_targets and self.target_scaler and all_targets_list:
                targets_array = np.concatenate(all_targets_list, axis=0)
                targets_array = targets_array.reshape(-1, targets_array.shape[-1])
                self.target_scaler.fit(targets_array)
                logger.info(
                    f"Target scaler fitted on {targets_array.shape[0]} total timesteps with {targets_array.shape[1]} targets"
                )

        self.is_scaler_fitted = True

        # Attach fitted scalers to the main dataset for on-the-fly transformation
        # This ensures scaling works even when fit_scaler is called directly (not via split_data)
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

        # Apply inverse transform
        data_inverse = scaler.inverse_transform(data_np)

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

    def _apply_scaling_to_splits(self, train_dataset, val_dataset, test_dataset):
        """
        Fit scaler on training data and attach to dataset for on-the-fly transformation.

        This unified approach works for both memory_efficient=True and False modes.
        Transformation happens on-the-fly in EncoderDecoderDataset.__getitem__().

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Step 1: Fit scaler on training data only (no data leakage)
        self.fit_scaler(train_dataset)

        # Step 2: Attach fitted scalers to the main dataset instance
        # This makes them available to all subsets (train, val, test)
        if self.is_scaler_fitted:
            logger.info("Attaching fitted scalers to dataset for on-the-fly transformation in __getitem__()")
            self.dataset.feature_scaler = self.feature_scaler
            self.dataset.target_scaler = self.target_scaler
            self.dataset.scale_targets = self.scale_targets

        return train_dataset, val_dataset, test_dataset

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

        # Apply scaling using consolidated method
        return self._apply_scaling_to_splits(train_dataset, val_dataset, test_dataset)

    def setup(self, stage=None):
        """Set up datasets for training, validation, and testing (PyTorch Lightning compatibility)."""
        if self.train_dataset is None and self.split_config:
            train_indices, val_indices, test_indices = self._create_splits(self.split_config)

            train_dataset = EncoderDecoderSubset(self.dataset, train_indices)
            val_dataset = EncoderDecoderSubset(self.dataset, val_indices)
            test_dataset = EncoderDecoderSubset(self.dataset, test_indices)

            # Apply scaling using consolidated method
            self.train_dataset, self.val_dataset, self.test_dataset = self._apply_scaling_to_splits(
                train_dataset, val_dataset, test_dataset
            )

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
