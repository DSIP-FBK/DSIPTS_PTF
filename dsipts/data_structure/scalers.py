"""Scaling utilities for time series data preprocessing using scikit-learn.

This module provides scalers that use scikit-learn's StandardScaler and MinMaxScaler
with partial_fit for incremental learning, suitable for both in-memory and
memory-efficient modes.

Key features:
- Uses scikit-learn's StandardScaler and MinMaxScaler
- Support for incremental learning via partial_fit
- Works for both memory-efficient and in-memory modes
- Standard (z-score) and MinMax (0-1) scaling methods
- Inverse scaling for denormalization
- Transform on-the-fly during __getitem__ calls
"""

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

logger = logging.getLogger(__name__)


class Scaler:
    """
    Scaler using scikit-learn's StandardScaler and MinMaxScaler with partial_fit.
    Supports both memory-efficient and in-memory modes via incremental learning.
    """

    def __init__(self, scaling_method="standard", scale_targets=False):
        """
        Initialize the scaler.

        Args:
            scaling_method: Method for scaling ("standard" or "minmax")
            scale_targets: Whether to scale target variables
        """
        self.scaling_method = scaling_method
        self.scale_targets = scale_targets
        self.is_scaler_fitted = False

        # Initialize sklearn scalers
        if scaling_method == "standard":
            self.feature_scaler = SklearnStandardScaler()
            self.target_scaler = SklearnStandardScaler() if scale_targets else None
        elif scaling_method == "minmax":
            self.feature_scaler = SklearnMinMaxScaler()
            self.target_scaler = SklearnMinMaxScaler() if scale_targets else None
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}. Use 'standard' or 'minmax'.")

    def fit_scaler(self, train_dataset):
        """
        Fit scaling parameters on training data using sklearn's partial_fit.
        Works for both memory-efficient and in-memory modes.

        Args:
            train_dataset: Training dataset subset
        """
        logger.info(f"Fitting {self.scaling_method} scaler using sklearn's partial_fit...")

        # Use partial_fit for incremental learning (works for both modes)
        for i in range(len(train_dataset)):
            sample = train_dataset[i]

            # Unpack sample (tuple format: (x, y))
            if isinstance(sample, tuple):
                x, y = sample
            else:
                # Backward compatibility: sample might be a dict
                x = sample
                y = sample.get("y", None)

            # Fit feature scaler on numerical features
            if "x_num_past" in x and x["x_num_past"].numel() > 0:
                features = x["x_num_past"].numpy()
                self.feature_scaler.partial_fit(features)

            # Fit target scaler if requested
            if self.scale_targets and self.target_scaler is not None and y is not None:
                if y.numel() > 0:
                    targets = y.numpy()
                    self.target_scaler.partial_fit(targets)

        self.is_scaler_fitted = True
        logger.info(f"Scaler fitted successfully on {len(train_dataset)} samples")

    def apply_inverse_scaling(self, data, data_type="features"):
        """
        Apply inverse scaling to denormalize data.

        Args:
            data: Data to denormalize (numpy array, pandas DataFrame, or torch tensor)
            data_type: Type of data ('features' or 'targets')

        Returns:
            Denormalized data in the same format as input
        """
        if not self.is_scaler_fitted:
            logger.warning("Scaler not fitted, cannot apply inverse scaling")
            return data

        # Select appropriate scaler
        scaler = self.feature_scaler if data_type == "features" else self.target_scaler
        if scaler is None:
            logger.warning(f"No scaler available for {data_type}")
            return data

        # Handle different data types
        if isinstance(data, pd.DataFrame):
            result = scaler.inverse_transform(data.values)
            return pd.DataFrame(result, index=data.index, columns=data.columns)
        elif isinstance(data, np.ndarray):
            return scaler.inverse_transform(data)
        elif hasattr(data, "numpy"):  # PyTorch tensor
            result_np = scaler.inverse_transform(data.numpy())
            return torch.from_numpy(result_np).float()
        else:
            logger.warning("Unsupported data type for inverse scaling")
            return data

    def transform_with_scaler(self, dataset):
        """
        Create a dataset wrapper that applies scaling on-the-fly during __getitem__ calls.

        Args:
            dataset: Dataset to transform

        Returns:
            Scaled dataset wrapper
        """
        if not self.is_scaler_fitted:
            logger.warning("Scaler not fitted, returning original dataset")
            return dataset

        logger.info(f"Creating scaled dataset wrapper with {self.scaling_method} scaling")

        # Import here to avoid circular imports
        from .d2_layers.encoder_decoder import EncoderDecoderSubset

        # Create scaled dataset wrapper with on-the-fly transformation
        class ScaledEncoderDecoderSubset(EncoderDecoderSubset):
            """Dataset wrapper that applies scaling during __getitem__ calls."""

            def __init__(self, original_subset, feature_scaler, target_scaler, scale_targets):
                self.dataset = original_subset.dataset
                self.indices = original_subset.indices
                self.feature_scaler = feature_scaler
                self.target_scaler = target_scaler
                self.scale_targets = scale_targets

            def __getitem__(self, idx):
                x, y = super().__getitem__(idx)

                # Scale numerical features
                if "x_num_past" in x and x["x_num_past"].numel() > 0:
                    scaled_features = self.feature_scaler.transform(x["x_num_past"].numpy())
                    x["x_num_past"] = torch.from_numpy(scaled_features).float()

                    # Update backward compatibility key if present
                    if "past_features" in x:
                        x["past_features"] = x["x_num_past"]

                # Scale targets if requested
                if self.scale_targets and self.target_scaler is not None:
                    scaled_targets = self.target_scaler.transform(y.numpy())
                    y = torch.from_numpy(scaled_targets).float()

                    # Update target in x dictionary if present
                    if "y" in x:
                        x["y"] = y

                return x, y

        return ScaledEncoderDecoderSubset(dataset, self.feature_scaler, self.target_scaler, self.scale_targets)
