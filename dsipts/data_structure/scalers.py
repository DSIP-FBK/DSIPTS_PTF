"""Online scaling utilities for time series data preprocessing.

This module provides scalers that can compute statistics incrementally
without loading all data into memory at once, making them suitable for
large datasets and streaming scenarios.

Key features:
- Manual scaling implementations (no sklearn dependencies)
- Support for both memory-efficient and in-memory modes
- Standard (z-score) and MinMax (0-1) scaling methods
- Online/incremental statistics computation for large datasets
- Inverse scaling for denormalization
"""

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class StatsRecorder:
    """
    Online statistics recorder for computing mean and std incrementally.
    Useful for memory-efficient scaling parameter computation.
    """

    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std = m / (m + n) * self.std**2 + n / (m + n) * newstd**2 + m * n / (m + n) ** 2 * (tmp - newmean) ** 2
            self.std = np.sqrt(self.std)

            self.nobservations += n


class MinMaxRecorder:
    """
    Online min/max recorder for computing scaling parameters incrementally.
    Useful for memory-efficient minmax scaling parameter computation.
    """

    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.min_vals = data.min(axis=0)
            self.max_vals = data.max(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            self.min_vals = np.minimum(self.min_vals, data.min(axis=0))
            self.max_vals = np.maximum(self.max_vals, data.max(axis=0))
            self.nobservations += data.shape[0]


class ManualScaler:
    """
    Manual scaling implementation with support for both memory-efficient and in-memory modes.
    Provides standard (z-score) and minmax (0-1) scaling methods without sklearn dependencies.
    """

    def __init__(self, scaling_method="standard", scale_targets=False):
        """
        Initialize the manual scaler.

        Args:
            scaling_method: Method for scaling ("standard" or "minmax")
            scale_targets: Whether to scale target variables
        """
        self.scaling_method = scaling_method
        self.scale_targets = scale_targets
        self.is_scaler_fitted = False
        self.scaling_params = {}

        # Initialize recorders for memory-efficient mode
        self.feature_stats_recorder = None
        self.target_stats_recorder = None
        self.feature_minmax_recorder = None
        self.target_minmax_recorder = None

    def fit_manual_scaler(self, train_dataset, d1_dataset):
        """
        Fit manual scaling parameters on training data only.

        Args:
            train_dataset: Training dataset subset
        """
        logger.info(f"Computing {self.scaling_method} scaling parameters from training data...")

        # Check if D1 dataset is memory efficient
        is_memory_efficient = getattr(d1_dataset, "memory_efficient", False)

        if is_memory_efficient:
            self._fit_manual_scaler_memory_efficient(train_dataset)
        else:
            self._fit_manual_scaler_in_memory(train_dataset)

        self.is_scaler_fitted = True
        logger.info(f"Manual {self.scaling_method} scaling parameters computed successfully")

    def _fit_manual_scaler_in_memory(self, train_dataset):
        """
        Fit scaling parameters when data can be loaded in memory (memory_efficient=False).
        """
        logger.info("Using in-memory approach for scaling parameter computation")

        # Collect all training data
        all_features = []
        all_targets = []

        for i in range(len(train_dataset)):
            sample = train_dataset[i]

            # Extract features and targets
            if "x_num_past" in sample:
                features = sample["x_num_past"].numpy()  # [past_len, n_features]
                all_features.append(features)

            if self.scale_targets and "y" in sample:
                targets = sample["y"].numpy()  # [future_len, n_targets]
                all_targets.append(targets)

        # Combine all data
        if all_features:
            combined_features = np.vstack(all_features)  # [total_samples, n_features]
            self._compute_scaling_params(combined_features, "features")

        if all_targets and self.scale_targets:
            combined_targets = np.vstack(all_targets)  # [total_samples, n_targets]
            self._compute_scaling_params(combined_targets, "targets")

    def _fit_manual_scaler_memory_efficient(self, train_dataset):
        """
        Fit scaling parameters for memory-efficient mode (memory_efficient=True).
        Uses online statistics computation.
        """
        logger.info("Using memory-efficient approach for scaling parameter computation")

        # Initialize recorders
        if self.scaling_method == "standard":
            self.feature_stats_recorder = StatsRecorder()
            if self.scale_targets:
                self.target_stats_recorder = StatsRecorder()
        else:  # minmax
            self.feature_minmax_recorder = MinMaxRecorder()
            if self.scale_targets:
                self.target_minmax_recorder = MinMaxRecorder()

        # Process training data in chunks
        for i in range(len(train_dataset)):
            sample = train_dataset[i]

            # Update feature statistics
            if "x_num_past" in sample:
                features = sample["x_num_past"].numpy()  # [past_len, n_features]

                if self.scaling_method == "standard":
                    self.feature_stats_recorder.update(features)
                else:  # minmax
                    self.feature_minmax_recorder.update(features)

            # Update target statistics
            if self.scale_targets and "y" in sample:
                targets = sample["y"].numpy()  # [future_len, n_targets]

                if self.scaling_method == "standard":
                    self.target_stats_recorder.update(targets)
                else:  # minmax
                    self.target_minmax_recorder.update(targets)

        # extract final parameters
        if self.scaling_method == "standard":
            if self.feature_stats_recorder and self.feature_stats_recorder.nobservations > 0:
                self.scaling_params["features"] = {
                    "mean": self.feature_stats_recorder.mean,
                    "std": self.feature_stats_recorder.std,
                    "method": "standard",
                }

            if self.scale_targets and self.target_stats_recorder and self.target_stats_recorder.nobservations > 0:
                self.scaling_params["targets"] = {
                    "mean": self.target_stats_recorder.mean,
                    "std": self.target_stats_recorder.std,
                    "method": "standard",
                }
        else:  # minmax
            if self.feature_minmax_recorder and self.feature_minmax_recorder.nobservations > 0:
                range_vals = self.feature_minmax_recorder.max_vals - self.feature_minmax_recorder.min_vals
                # Avoid division by zero
                range_vals = np.where(range_vals == 0, 1.0, range_vals)

                self.scaling_params["features"] = {
                    "min": self.feature_minmax_recorder.min_vals,
                    "max": self.feature_minmax_recorder.max_vals,
                    "range": range_vals,
                    "method": "minmax",
                }

            if self.scale_targets and self.target_minmax_recorder and self.target_minmax_recorder.nobservations > 0:
                range_vals = self.target_minmax_recorder.max_vals - self.target_minmax_recorder.min_vals
                # Avoid division by zero
                range_vals = np.where(range_vals == 0, 1.0, range_vals)

                self.scaling_params["targets"] = {
                    "min": self.target_minmax_recorder.min_vals,
                    "max": self.target_minmax_recorder.max_vals,
                    "range": range_vals,
                    "method": "minmax",
                }

    def _compute_scaling_params(self, data, data_type):
        """
        Compute scaling parameters for given data.

        Args:
            data: numpy array of shape [n_samples, n_features]
            data_type: 'features' or 'targets'
        """
        if self.scaling_method == "standard":
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)

            self.scaling_params[data_type] = {"mean": mean, "std": std, "method": "standard"}
        else:  # minmax
            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            range_vals = max_vals - min_vals
            # Avoid division by zero
            range_vals = np.where(range_vals == 0, 1.0, range_vals)

            self.scaling_params[data_type] = {"min": min_vals, "max": max_vals, "range": range_vals, "method": "minmax"}

    def apply_inverse_scaling(self, data, data_type="features"):
        """
        Apply inverse scaling to denormalize predictions using manual scaling parameters.

        Args:
            data: Data to denormalize (numpy array, pandas DataFrame, or torch tensor)
            data_type: Type of data ('features' or 'targets')

        Returns:
            Denormalized data in the same format as input
        """
        if not self.is_scaler_fitted:
            logger.warning("Scaler not fitted, cannot apply inverse scaling")
            return data

        if data_type not in self.scaling_params:
            logger.warning(f"No scaling parameters found for {data_type}")
            return data

        params = self.scaling_params[data_type]

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if params["method"] == "standard":
                result = result * params["std"] + params["mean"]
            else:  # minmax
                result = result * params["range"] + params["min"]
            return result

        elif isinstance(data, np.ndarray):
            if params["method"] == "standard":
                return data * params["std"] + params["mean"]
            else:  # minmax
                return data * params["range"] + params["min"]

        elif hasattr(data, "numpy"):  # PyTorch tensor
            data_np = data.numpy()
            if params["method"] == "standard":
                result_np = data_np * params["std"] + params["mean"]
            else:  # minmax
                result_np = data_np * params["range"] + params["min"]
            return torch.from_numpy(result_np)
        else:
            logger.warning("Unsupported data type for manual inverse scaling")
            return data

    def fit_scaler(self, dataset, d1_dataset):
        """
        Fit the manual scaler on numeric features from the given dataset.

        Args:
            dataset: Dataset to fit the scaler on (typically training dataset)
            d1_dataset: D1 dataset for memory efficiency check
        """
        self.fit_manual_scaler(dataset, d1_dataset)

    def transform_with_scaler(self, dataset):
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

        logger.info(f"Transforming dataset with manual {self.scaling_method} scaling")

        # Import here to avoid circular imports
        from .d2_layers.encoder_decoder import EncoderDecoderSubset

        # Create a new dataset with transformed features
        class ScaledEncoderDecoderSubset(EncoderDecoderSubset):
            def __init__(self, original_subset, scale_targets=False, scaling_params=None):
                self.dataset = original_subset.dataset
                self.indices = original_subset.indices
                self.scale_targets = scale_targets
                self.scaling_params = scaling_params or {}

            def __getitem__(self, idx):
                x, y = super().__getitem__(idx)

                # Transform numeric features using manual scaling
                if "x_num_past" in x and x["x_num_past"].shape[1] > 0:
                    x["x_num_past"] = self._apply_manual_scaling(x["x_num_past"], "features")

                    # Also update the backward compatibility key if present
                    if "past_features" in x:
                        x["past_features"] = x["x_num_past"]

                # Transform targets if requested using manual scaling
                if self.scale_targets:
                    y = self._apply_manual_scaling(y, "targets")

                    # Also update the target in x dictionary
                    if "y" in x:
                        x["y"] = y
                    if "future_targets" in x:
                        x["future_targets"] = y

                return x, y

            def _apply_manual_scaling(self, data, data_type):
                """Apply manual scaling to data."""
                if data_type not in self.scaling_params:
                    return data

                params = self.scaling_params[data_type]
                data_np = data.numpy()

                if params["method"] == "standard":
                    scaled_data = (data_np - params["mean"]) / params["std"]
                else:  # minmax
                    scaled_data = (data_np - params["min"]) / params["range"]

                return torch.tensor(scaled_data, dtype=torch.float32)

        # Create and return the scaled dataset
        return ScaledEncoderDecoderSubset(dataset, self.scale_targets, self.scaling_params)
