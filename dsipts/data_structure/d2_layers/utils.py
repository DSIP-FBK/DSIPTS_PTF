"""Utility functions for D2 layer implementations."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """
    Custom collate function for batching time series data.

    This function handles both tuple and dict input formats and returns
    a single dictionary with batched tensors and lists.

    Args:
        batch: List of samples from the dataset

    Returns:
        Dictionary containing batched data
    """
    # Handle both tuple (x, y) and dict formats
    if isinstance(batch[0], tuple):
        x_list = [item[0] for item in batch]
    else:
        x_list = batch

    # Initialize result dictionary
    result = {}

    # Process each key in the first sample
    for key in x_list[0]:
        if key in ["group_id", "past_time", "future_time"]:
            # Keep as lists for non-tensor data
            result[key] = [sample[key] for sample in x_list]
        else:
            # Stack tensors
            tensor_list = []
            for sample in x_list:
                item = sample[key]
                if isinstance(item, np.ndarray):
                    tensor_list.append(torch.from_numpy(item))
                elif isinstance(item, torch.Tensor):
                    tensor_list.append(item)
                else:
                    # Convert to tensor if it's a scalar or list
                    tensor_list.append(torch.tensor(item))

            # Stack all tensors
            if len(tensor_list) > 0:
                try:
                    result[key] = torch.stack(tensor_list)
                except RuntimeError:
                    # If stacking fails, pad sequences to same length
                    max_len = max(t.shape[0] if len(t.shape) > 0 else 1 for t in tensor_list)
                    padded_tensors = []
                    for t in tensor_list:
                        if len(t.shape) == 0:
                            # Scalar tensor
                            padded_tensors.append(t.unsqueeze(0))
                        elif t.shape[0] < max_len:
                            # Pad to max length
                            pad_size = max_len - t.shape[0]
                            if len(t.shape) == 1:
                                padding = torch.zeros(pad_size)
                            else:
                                padding = torch.zeros(pad_size, *t.shape[1:])
                            padded_tensors.append(torch.cat([t, padding]))
                        else:
                            padded_tensors.append(t)
                    result[key] = torch.stack(padded_tensors)

    return result


def is_valid_window(past_indices: List[int], future_indices: List[int], past_len: int, future_len: int) -> bool:
    """
    Check if a window is valid (has sufficient data).

    Args:
        past_indices: Indices for past data
        future_indices: Indices for future data
        past_len: Required past length
        future_len: Required future length

    Returns:
        True if window is valid
    """
    return len(past_indices) == past_len and len(future_indices) == future_len


def build_valid_windows(
    d1_dataset, past_len: int, future_len: int, step_size: int, max_samples_per_group: int = None
) -> List[Dict[str, Any]]:
    """
    Build valid sliding windows from the D1 dataset.

    Args:
        d1_dataset: D1 layer dataset
        past_len: Length of past sequence
        future_len: Length of future sequence
        step_size: Step size for sliding window
        max_samples_per_group: Maximum samples per group

    Returns:
        List of valid window dictionaries
    """
    valid_windows = []
    total_groups = len(d1_dataset)
    windows_per_group = {}
    insufficient_groups = []

    for group_idx in range(total_groups):
        group_sample = d1_dataset[group_idx]
        group_id = group_sample.get("group_id", group_idx)
        seq_len = group_sample.get("seq_len", 0)

        # Create sliding windows within this group's sequence
        max_windows = seq_len - past_len - future_len + 1

        if max_windows > 0:
            group_windows = 0
            for i in range(0, max_windows, step_size):
                window = {
                    "group_idx": group_idx,
                    "group_id": group_id,
                    "start_idx": i,
                    "past_len": past_len,
                }
                valid_windows.append(window)
                group_windows += 1

                # Limit samples per group if specified
                if (
                    max_samples_per_group is not None
                    and len([w for w in valid_windows if w["group_id"] == group_id]) >= max_samples_per_group
                ):
                    break

            windows_per_group[group_id] = group_windows
        else:
            insufficient_groups.append(group_id)
    logger.info(f"Created {len(valid_windows)} windows from {len(windows_per_group)} groups")
    if insufficient_groups:
        logger.debug(f"Skipped {len(insufficient_groups)} groups with insufficient data")

    return valid_windows


def create_temporal_splits(
    valid_windows: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create temporal train/val/test splits.

    Args:
        valid_windows: List of valid windows
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        test_ratio: Test data ratio

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    n_samples = len(valid_windows)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, n_samples))

    return train_indices, val_indices, test_indices


def create_random_splits(
    valid_windows: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = None
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create random train/val/test splits.

    Args:
        valid_windows: List of valid windows
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        test_ratio: Test data ratio
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    n_samples = len(valid_windows)
    indices = list(range(n_samples))

    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    return train_indices, val_indices, test_indices


class PreTransformedDataset:
    """
    Wrapper for pre-transformed datasets.

    This class stores transformed data in vectorized numpy arrays and creates
    sample dicts on-demand in __getitem__().

    Trade-off: Uses more memory but provides faster setup and inference.
    """

    def __init__(self, transformed_samples: List[Tuple[Dict[str, Any], torch.Tensor]]):
        """
        Initialize with pre-transformed samples.

        Args:
            transformed_samples: List of (x, y) tuples where x is dict and y is tensor
        """
        self.samples = transformed_samples

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Get a pre-transformed sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (x, y) where x is dict and y is tensor
        """
        return self.samples[idx]


class VectorizedPreTransformedDataset:
    """
    Vectorized storage for pre-transformed datasets.
    """

    def __init__(
        self,
        all_X_past,
        all_X_future,
        all_y,
        indices,
        valid_windows,
        idx_num,
        idx_categorical,
        future_num_idx,
        future_cat_indices,
        idx_target_tensor,
        global_forecasting,
        meta,
    ):
        """Store vectorized data."""
        self.all_X_past = all_X_past
        self.all_X_future = all_X_future
        self.all_y = all_y
        self.indices = indices
        self.valid_windows = valid_windows
        self.idx_num = idx_num
        self.idx_categorical = idx_categorical
        self.future_num_idx = future_num_idx
        self.future_cat_indices = future_cat_indices
        self.idx_target_tensor = idx_target_tensor
        self.global_forecasting = global_forecasting
        self.meta = meta

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Create dict on-demand (fast)."""
        window_idx = self.indices[idx]
        window = self.valid_windows[window_idx]
        start_idx = window["start_idx"]

        x = {}

        # Add numeric features by slicing from full arrays
        if len(self.idx_num) > 0:
            x["x_num_past"] = torch.from_numpy(self.all_X_past[idx, :, :][:, self.idx_num]).float()
        else:
            x["x_num_past"] = torch.zeros((self.all_X_past.shape[1], 0), dtype=torch.float32)

        if len(self.future_num_idx) > 0:
            x["x_num_future"] = torch.from_numpy(self.all_X_future[idx, :, self.future_num_idx]).float()

        # Add categorical features by slicing from full arrays
        if len(self.idx_categorical) > 0:
            x["x_cat_past"] = torch.from_numpy(self.all_X_past[idx, :, :][:, self.idx_categorical]).long()

            if len(self.future_cat_indices) > 0:
                x["x_cat_future"] = torch.from_numpy(self.all_X_future[idx, :, :][:, self.future_cat_indices]).long()

        # Add targets
        y = torch.from_numpy(self.all_y[idx]).float()
        x["y"] = y

        # Add idx_target
        x["idx_target"] = self.idx_target_tensor

        # Add group_id if needed
        if self.global_forecasting:
            group_id = window.get("group_id", 0)
            if isinstance(group_id, str):
                meta_group_mapping = self.meta.get("group_mapping", {})
                group_id = meta_group_mapping.get(group_id, 0)
                x["group_id"] = int(group_id)
            elif isinstance(group_id, (int, float)):
                x["group_id"] = int(group_id)
            else:
                x["group_id"] = group_id

        x["time_idx"] = start_idx

        return x, y
