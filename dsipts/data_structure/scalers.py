"""
Online scaling utilities for time series data preprocessing.

This module provides scalers that can compute statistics incrementally
without loading all data into memory at once, making them suitable for
large datasets and streaming scenarios.
"""

import logging
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class OnlineStandardScaler:
    """
    Computes the mean and standard deviation in an online fashion, batch by batch.
    This is useful for large datasets that don't fit into memory.

    The scaler performs two main operations:
    1. fit(): Iterates through all batches of data once to compute the final
              mean and variance of the entire dataset.
    2. transform(): Applies the standardization (z-score normalization) using the
                    computed global mean and standard deviation.

    The update formulas are based on Welford's algorithm for online variance
    calculation, extended for batches.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the online standard scaler.

        Args:
            epsilon (float): Small value added to std dev to prevent division by zero
        """
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = 0
        self.epsilon = epsilon
        self.is_fitted_ = False

    def _partial_fit(self, batch: np.ndarray):
        """
        Update the running mean and variance with a single batch of data.

        Args:
            batch (np.ndarray): New batch of data, shape (n_samples, n_features)
        """
        batch = np.atleast_2d(batch)

        # Handle the very first batch to initialize dimensions
        if self.n_samples_seen_ == 0:
            self.n_features_in_ = batch.shape[1]
            self.mean_ = np.zeros(self.n_features_in_)
            self.var_ = np.zeros(self.n_features_in_)

        if batch.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Batch has {batch.shape[1]} features, but scaler was fitted " f"with {self.n_features_in_} features."
            )

        # Get stats for the new batch
        new_count = batch.shape[0]
        new_mean = np.mean(batch, axis=0)
        new_var = np.var(batch, axis=0, ddof=0)  # Population variance

        # Store old stats before updating
        old_count = self.n_samples_seen_
        old_mean = self.mean_.copy()
        old_var = self.var_.copy()

        # Update total count
        self.n_samples_seen_ += new_count

        # If it's not the first batch, update mean and variance
        if old_count > 0:
            # Update mean using weighted average formula
            self.mean_ = (old_count * old_mean + new_count * new_mean) / self.n_samples_seen_

            # Update variance using the formula for combining variances of two groups
            # Formula: Var_combined = (m*Var_m + n*Var_n)/(m+n) + (m*n)/(m+n)^2 * (Mean_m - Mean_n)^2
            term1 = old_count * old_var
            term2 = new_count * new_var
            term3 = (old_count * new_count / self.n_samples_seen_**2) * (old_mean - new_mean) ** 2
            self.var_ = (term1 + term2) / self.n_samples_seen_ + term3
        else:
            # For the first batch, the global stats are just the batch stats
            self.mean_ = new_mean
            self.var_ = new_var

    def fit(self, data_iterator: Iterator[np.ndarray]):
        """
        Compute the final mean and variance by iterating through all data batches.
        This is the "metadata calculation" pass.

        Args:
            data_iterator (Iterator[np.ndarray]): Iterator that yields batches of data

        Returns:
            self: The fitted scaler instance
        """
        logger.info("Fitting OnlineStandardScaler...")
        self.n_samples_seen_ = 0  # Reset for a new fit

        batch_count = 0
        for batch in data_iterator:
            self._partial_fit(batch)
            batch_count += 1

        if self.n_samples_seen_ == 0:
            raise ValueError("No data provided to fit the scaler")

        # Finalize the scale (standard deviation) after seeing all data
        self.scale_ = np.sqrt(self.var_ + self.epsilon)
        self.is_fitted_ = True

        logger.info(
            f"OnlineStandardScaler fitted on {self.n_samples_seen_} samples "
            f"across {batch_count} batches with {self.n_features_in_} features"
        )
        logger.info(f"Final mean: {self.mean_}")
        logger.info(f"Final std: {self.scale_}")

        return self

    def transform(self, batch: np.ndarray) -> np.ndarray:
        """
        Standardize a batch of data using the fitted mean and scale.

        Args:
            batch (np.ndarray): The data to transform

        Returns:
            np.ndarray: The standardized data
        """
        if not self.is_fitted_:
            raise RuntimeError("This scaler instance is not fitted yet. Call 'fit' first.")

        batch = np.atleast_2d(batch)
        if batch.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Batch has {batch.shape[1]} features, but scaler was fitted " f"with {self.n_features_in_} features."
            )

        return (batch - self.mean_) / self.scale_

    def inverse_transform(self, batch: np.ndarray) -> np.ndarray:
        """
        Reverse the standardization transformation.

        Args:
            batch (np.ndarray): The standardized data to inverse transform

        Returns:
            np.ndarray: The original scale data
        """
        if not self.is_fitted_:
            raise RuntimeError("This scaler instance is not fitted yet. Call 'fit' first.")

        batch = np.atleast_2d(batch)
        if batch.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Batch has {batch.shape[1]} features, but scaler was fitted " f"with {self.n_features_in_} features."
            )

        return batch * self.scale_ + self.mean_

    def get_params(self) -> Dict[str, Any]:
        """
        Get the scaling parameters.

        Returns:
            Dictionary containing mean, std, and other parameters
        """
        if not self.is_fitted_:
            return {}

        return {
            "mean": self.mean_.copy(),
            "std": self.scale_.copy(),
            "var": self.var_.copy(),
            "n_samples_seen": self.n_samples_seen_,
            "n_features": self.n_features_in_,
            "epsilon": self.epsilon,
        }

    def fit_transform(self, data_iterator: Iterator[np.ndarray]) -> np.ndarray:
        """
        Fit the scaler on all data and then transform the entire dataset.

        NOTE: This method must store all data in memory to return the transformed
        version, which might defeat the purpose of online scaling for very large datasets.
        It's provided for convenience, similar to scikit-learn's API.

        Args:
            data_iterator (Iterator[np.ndarray]): Iterator that yields batches of data

        Returns:
            np.ndarray: The entire standardized dataset
        """
        # Store all batches in a list during the fit pass
        all_data_list = []

        def fit_and_store_iterator():
            for batch in data_iterator:
                all_data_list.append(batch)
                yield batch

        self.fit(fit_and_store_iterator())

        # Concatenate and transform
        if not all_data_list:
            return np.array([])

        full_dataset = np.vstack(all_data_list)
        return self.transform(full_dataset)


class StandardScalerAdapter:
    """
    Adapter class to integrate OnlineStandardScaler with DSIPTS D1/D2 layers.

    This class provides a consistent interface for standard scaling that works
    with the existing DSIPTS architecture, handling both pandas DataFrames
    and torch tensors.
    """

    def __init__(self, columns: Optional[list] = None, epsilon: float = 1e-8):
        """
        Initialize the standard scaler adapter.

        Args:
            columns: List of column names to scale (None means all numeric columns)
            epsilon: Small value to prevent division by zero
        """
        self.columns = columns
        self.epsilon = epsilon
        self.scaler = OnlineStandardScaler(epsilon=epsilon)
        self.column_indices = None
        self.is_fitted = False

    def fit_dataframe_iterator(self, df_iterator: Iterator[pd.DataFrame]):
        """
        Fit the scaler on an iterator of DataFrames.

        Args:
            df_iterator: Iterator yielding pandas DataFrames
        """

        def array_iterator():
            for df in df_iterator:
                if self.columns is None:
                    # Use all numeric columns
                    numeric_df = df.select_dtypes(include=[np.number])
                    if self.column_indices is None:
                        self.columns = numeric_df.columns.tolist()
                        self.column_indices = [df.columns.get_loc(col) for col in self.columns]
                    yield numeric_df.values
                else:
                    # Use specified columns
                    if self.column_indices is None:
                        self.column_indices = [df.columns.get_loc(col) for col in self.columns]
                    yield df[self.columns].values

        self.scaler.fit(array_iterator())
        self.is_fitted = True
        return self

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a DataFrame using the fitted scaler.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_dataframe_iterator first.")

        result_df = df.copy()
        if self.columns is not None:
            scaled_values = self.scaler.transform(df[self.columns].values)
            result_df[self.columns] = scaled_values

        return result_df

    def inverse_transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform a DataFrame.

        Args:
            df: Scaled DataFrame to inverse transform

        Returns:
            Original scale DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_dataframe_iterator first.")

        result_df = df.copy()
        if self.columns is not None:
            unscaled_values = self.scaler.inverse_transform(df[self.columns].values)
            result_df[self.columns] = unscaled_values

        return result_df

    def transform_tensor(self, tensor: torch.Tensor, columns: list) -> torch.Tensor:
        """
        Transform a torch tensor using the fitted scaler.

        Args:
            tensor: Tensor to transform
            columns: List of column names corresponding to tensor dimensions

        Returns:
            Transformed tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted.")

        # Create a copy of the tensor to modify
        result = tensor.clone()

        # Find indices of columns to scale
        scale_indices = []
        scale_cols = []

        for i, col in enumerate(columns):
            if col in self.columns:
                scale_indices.append(i)
                # Find the index of this column in the scaler's columns
                scaler_col_idx = self.columns.index(col)
                scale_cols.append(scaler_col_idx)

        if not scale_indices:
            return tensor

        # Convert to numpy for processing
        numpy_data = tensor.detach().numpy()

        # For each feature dimension that needs scaling
        for tensor_idx, scaler_idx in zip(scale_indices, scale_cols):
            # Extract the specific feature column
            feature_data = numpy_data[..., tensor_idx]

            # Apply standardization manually using the correct mean and std for this column
            mean = self.scaler.mean_[scaler_idx]
            std = self.scaler.scale_[scaler_idx]

            # Standardize: (x - mean) / std
            scaled_feature = (feature_data - mean) / std

            # Update the result tensor
            result[..., tensor_idx] = torch.from_numpy(scaled_feature)

        return result

    def inverse_transform_tensor(self, tensor: torch.Tensor, columns: list) -> torch.Tensor:
        """
        Inverse transform a torch tensor.

        Args:
            tensor: Scaled tensor to inverse transform
            columns: List of column names corresponding to tensor dimensions

        Returns:
            Original scale tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted.")

        # Create a copy of the tensor to modify
        result = tensor.clone()

        # Find indices of columns to unscale
        scale_indices = []
        scale_cols = []

        for i, col in enumerate(columns):
            if col in self.columns:
                scale_indices.append(i)
                # Find the index of this column in the scaler's columns
                scaler_col_idx = self.columns.index(col)
                scale_cols.append(scaler_col_idx)

        if not scale_indices:
            return tensor

        # Convert to numpy for processing
        numpy_data = tensor.detach().numpy()

        # For each feature dimension that needs unscaling
        for tensor_idx, scaler_idx in zip(scale_indices, scale_cols):
            # Extract the specific feature column
            feature_data = numpy_data[..., tensor_idx]

            # Apply inverse standardization manually using the correct mean and std for this column
            mean = self.scaler.mean_[scaler_idx]
            std = self.scaler.scale_[scaler_idx]

            # Inverse standardize: x * std + mean
            unscaled_feature = feature_data * std + mean

            # Update the result tensor
            result[..., tensor_idx] = torch.from_numpy(unscaled_feature)

        return result

    def get_scaling_params(self) -> Dict[str, Dict[str, float]]:
        """
        Get scaling parameters in a format compatible with DSIPTS.

        Returns:
            Dictionary mapping column names to their scaling parameters
        """
        if not self.is_fitted:
            return {}

        params = self.scaler.get_params()
        result = {}

        for i, col in enumerate(self.columns):
            result[col] = {
                "mean": float(params["mean"][i]),
                "std": float(params["std"][i]),
                "var": float(params["var"][i]),
                "is_constant": float(params["std"][i]) < self.epsilon,
                "scaler_type": "standard",
            }

        return result
