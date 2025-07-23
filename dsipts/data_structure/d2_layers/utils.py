"""Utility functions for D2 layer implementations."""

import numpy as np
import torch


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
