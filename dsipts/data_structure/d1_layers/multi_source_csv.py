"""
Multi-source CSV implementation for D1 layer.

This module provides the MultiSourceTSDataSet class that handles raw data from multiple CSV files
with enhanced features including temporal categorical enrichment and improved logging.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder

from .base_d1 import BaseD1Layer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiSourceTSDataSet(BaseD1Layer):
    """
    Layer 1 (D1) dataset for multi-source time series data.

    This dataset:
    1. Loads time series data from multiple CSV files
    2. Handles categorical encoding and normalization
    3. Efficiently processes data in chunks for memory-efficient operation
    4. Preserves NaN values for D2 layer to handle
    5. Supports temporal categorical enrichment
    6. Provides improved logging for frequency inference

    It does NOT compute validity of windows or create sliding windows - that is
    the responsibility of the D2 layer (TSDataProcessor).
    """

    def __init__(
        self,
        file_paths: List[str],
        group_cols: Optional[Union[str, List[str]]] = None,
        time_col: str = None,
        target_cols: List[str] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        known_cols: Optional[List[str]] = None,
        unknown_cols: Optional[List[str]] = None,
        enrich_cat: Optional[List[str]] = None,
        weights: Optional[str] = None,
        memory_efficient: bool = False,
        chunk_size: int = 10000,
    ):
        """
        Initialize the MultiSourceTSDataSet.

        Args:
            file_paths: List of paths to CSV files containing time series data
            group_cols: Column(s) that identify unique time series groups
            time_col: Column containing time/date information
            target_cols: Columns to use as targets (y)
            cat_cols: Categorical columns that need encoding
            num_cols: Numerical columns (if None, all non-categorical
                columns are treated as numerical)
            known_cols: Known columns at prediction time (feature columns that are known)
            unknown_cols: Unknown columns at prediction time (if None,
                all target_cols are considered unknown)
            enrich_cat: List of temporal categorical variables to create from time column.
                Supported values: ['hour', 'dow', 'month', 'minute']
            weights: Name of weights column
            memory_efficient: Whether to use memory-efficient mode
            chunk_size: Chunk size for processing data (used in memory-efficient
                mode)
        """
        super().__init__()

        # Basic configuration
        self.file_paths = file_paths
        self.time_col = time_col
        self.weights = weights

        # Handle group columns (can be single column or multiple)
        # Handle None, empty list, or NaN values
        if group_cols is None or (isinstance(group_cols, list) and len(group_cols) == 0):
            self._group_cols = []
            logger.info("No group columns provided, using default grouping")
        elif isinstance(group_cols, str):
            self._group_cols = [group_cols]
        else:
            self._group_cols = group_cols

        # Initialize attributes with proper defaults
        self._target_cols = target_cols or []
        self._time_col = time_col
        self._cat_cols = cat_cols or []
        self._num_cols = num_cols or []

        # Handle group columns properly (already set self._group_cols above)

        # Set known and unknown columns with proper handling for None values
        self._known_cols = known_cols or []
        self._unknown_cols = unknown_cols or list(self._target_cols) if self._target_cols else []

        # Infer feature_cols automatically from file headers and other specifications
        self._feature_cols = self._infer_feature_columns()
        self._enrich_cat = enrich_cat
        self.enrich_cat = enrich_cat or []
        self._validate_enrich_cat()

        # Flag to track if temporal features have been added to categorical columns
        self._is_file_read = False

        # If _num_cols not specified, infer from feature_cols and cat_cols
        if not self._num_cols:
            # Get all possible columns (features + targets)
            all_cols = self._feature_cols + self._target_cols
            self._num_cols = [c for c in all_cols if c not in self._cat_cols]
            logger.info(f"Inferred {len(self._num_cols)} numerical columns: {self._num_cols}")

        # Internal state
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        self.max_length = None  # Can be set later if needed for time series regularization

        # Initialize label encoders for categorical columns
        self.label_encoders = {}

        # For compatibility with test code, initialize data attribute
        self.data = None

        # Pre-loaded data cache (only used when memory_efficient=False)
        self.data_cache = {}

        # Process files to build metadata and encoders
        self._process_files()

        # Prepare metadata
        self._prepare_metadata()

        # Preload data if memory_efficient is False
        if not self.memory_efficient:
            self._preload_data()

    def _infer_feature_columns(self) -> List[str]:
        """
        Infer feature columns automatically from file headers and other specifications.

        Logic:
        1. If known_cols is specified, use it as the primary source for feature columns
        2. Otherwise, read the first CSV file to get all available columns
        3. Exclude special columns: time_col, group_cols, target_cols, weights
        4. Include columns from num_cols and cat_cols if specified
        5. Filter out any enriched temporal features that will be added later

        Returns:
            List of inferred feature column names
        """
        # Priority 1: If known_cols is explicitly specified, use it
        if self._known_cols:
            logger.info("Using known_cols as feature_cols")
            return list(self._known_cols)

        # Priority 2: If num_cols or cat_cols are specified, use them (excluding targets)
        if self._num_cols or self._cat_cols:
            logger.info("Inferring feature_cols from num_cols and cat_cols")
            all_specified_cols = list(set(self._num_cols + self._cat_cols))
            feature_cols = [col for col in all_specified_cols if col not in self._target_cols]
            return feature_cols

        # Priority 3: Read from file headers and infer automatically
        logger.info("Inferring feature_cols from file headers")
        try:
            # Read the first few rows of the first file to get column names
            first_file = self.file_paths[0]
            sample_df = pd.read_csv(first_file, nrows=1)
            all_columns = list(sample_df.columns)

            # Define special columns to exclude from features
            special_columns = set()

            # Add time column
            if self.time_col:
                special_columns.add(self.time_col)

            # Add group columns
            if self._group_cols:
                if isinstance(self._group_cols, str):
                    special_columns.add(self._group_cols)
                else:
                    special_columns.update(self._group_cols)

            # Add target columns
            special_columns.update(self._target_cols)

            # Add weights column
            if self.weights:
                special_columns.add(self.weights)

            # Add potential temporal enrichment columns (they will be added later)
            if self.enrich_cat and self.time_col:
                for enrich_option in self.enrich_cat:
                    special_columns.add(enrich_option)  # Simple names like 'hour', 'dow'

            # Filter out special columns to get feature columns
            feature_cols = [col for col in all_columns if col not in special_columns]

            logger.info(
                f"Inferred {len(feature_cols)} feature columns from file headers: {feature_cols}"
            )
            return feature_cols

        except Exception as e:
            logger.error(f"Failed to infer feature columns from file headers: {e}")
            logger.warning("Falling back to empty feature columns list")
            return []

    def _validate_enrich_cat(self):
        """Validate the enrich_cat parameter."""
        valid_enrich_options = ["hour", "dow", "month", "minute"]
        for option in self.enrich_cat:
            if option not in valid_enrich_options:
                raise ValueError(
                    f"Invalid enrich_cat option: {option}. "
                    f"Valid options are: {valid_enrich_options}"
                )

    def _parse_and_enrich_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and enrich a chunk of data.

        Args:
            chunk: DataFrame chunk to process

        Returns:
            Processed DataFrame chunk
        """
        # If time_col is missing, create one as integer range and set as time_col
        if self.time_col not in chunk.columns:
            logger.info(
                f"Time column '{self.time_col}' not found."
                f"Creating 'time' column as integer range."
            )
            chunk["time"] = range(len(chunk))
            self.time_col = "time"

        # Now, handle the time column type
        if not pd.api.types.is_datetime64_any_dtype(chunk[self.time_col]):
            try:
                chunk[self.time_col] = pd.to_datetime(chunk[self.time_col])
                logger.info(f"Converted '{self.time_col}' to datetime.")
            except Exception as e:
                # If conversion fails, check if int/float
                if pd.api.types.is_integer_dtype(
                    chunk[self.time_col]
                ) or pd.api.types.is_float_dtype(chunk[self.time_col]):
                    logger.info(
                        f"Time column '{self.time_col}' is numeric (int/float)."
                        f"Proceeding as numeric time."
                    )
                else:
                    logger.warning(
                        f"Time column '{self.time_col}' cant be converted to datetime."
                        f"Its not numerical, Temporal enrichment skipped! Error: {e}"
                    )

        # Enrich with temporal features if requested and time_col is datetime
        if self.enrich_cat and pd.api.types.is_datetime64_any_dtype(chunk[self.time_col]):
            chunk = self._enrich_temporal_features(chunk)

        # Always sort by time_col
        chunk = chunk.sort_values(by=self.time_col).reset_index(drop=True)

        # Set of mandatory columns (cat, num, target, group, time)
        mandatory_cols = set(self.cat_cols or [])
        mandatory_cols.update(self._num_cols or [])
        mandatory_cols.update(self.target_cols or [])

        # Always include time column
        if self.time_col and self.time_col in chunk.columns:
            mandatory_cols.add(self.time_col)

        # Always include group columns
        if isinstance(self.group_cols, list):
            for col in self.group_cols:
                if col in chunk.columns:
                    mandatory_cols.add(col)
        elif self.group_cols and self.group_cols in chunk.columns:
            mandatory_cols.add(self.group_cols)

        filtered_cols = [col for col in mandatory_cols if col in chunk.columns]
        return chunk[filtered_cols]

    def _enrich_temporal_features(self, dataset):
        """
        Enrich dataset with temporal categorical features.

        Args:
            dataset: DataFrame to enrich

        Returns:
            Dataset enriched with temporal categorical features
        """
        if not self.enrich_cat or self.time_col not in dataset.columns:
            return dataset

        # Ensure time column is datetime (should already be handled by _parse_and_enrich_chunk)
        if not pd.api.types.is_datetime64_any_dtype(dataset[self.time_col]):
            logger.warning(
                f"Time column {self.time_col} is not datetime type for temporal enrichment"
            )
            return dataset

        # Add temporal features directly without mapping dict
        for column in self.enrich_cat:
            if column == "hour":
                dataset[column] = dataset[self.time_col].dt.hour
            elif column == "dow":
                dataset[column] = dataset[self.time_col].dt.dayofweek
            elif column == "month":
                dataset[column] = dataset[self.time_col].dt.month
            elif column == "minute":
                dataset[column] = dataset[self.time_col].dt.minute
            else:
                if column not in dataset.columns:
                    logger.error(
                        f"I can not automatically enrich column {column}. Please contact the developers or add it manually to your dataset."  # noqa: E501
                    )

        # Add temporal categorical features to cat_cols and known_cols only once
        if self.enrich_cat and not self._is_file_read:
            if self._cat_cols is None:
                self._cat_cols = []
            self._cat_cols.extend(self.enrich_cat)

            # Also add to known_cols since these are always known
            if self._known_cols is not None:
                self._known_cols.extend(self.enrich_cat)

            self._is_file_read = True
            logger.info(f"Added temporal categorical features to columns: {self.enrich_cat}")

        return dataset

    @property
    def group_cols(self) -> List[str]:
        """Get the group columns."""
        return self._group_cols

    @property
    def target_cols(self) -> List[str]:
        """Get the target columns."""
        return self._target_cols or []

    @property
    def feature_cols(self) -> List[str]:
        """Get the feature columns."""
        return self._feature_cols or []

    @property
    def cat_cols(self) -> Optional[List[str]]:
        """Get the categorical columns."""
        return self._cat_cols or []

    @property
    def num_cols(self) -> Optional[List[str]]:
        """Get the numerical columns."""
        return self._num_cols or []

    @property
    def known_cols(self) -> Optional[List[str]]:
        """Get the known future columns."""
        return self._known_cols or []

    @property
    def unknown_cols(self) -> Optional[List[str]]:
        """Get the unknown future columns."""
        return self._unknown_cols or []

    def _process_files(self):
        """
        Process each file to extract group information and update encoders.

        This method:
        1. Scans through all CSV files (in chunks if memory_efficient=True)
        2. Identifies unique groups across all files
        3. Updates label encoders for categorical columns
        4. Builds a mapping of where each group's data is located
        5. Calculates the total length of each group
        6. Treats groups as file-specific to handle large files efficiently
        7. Preserves NaN values for valid index computation in D2 layer
        """
        # Initialize data structures
        self.total_length = 0  # Total number of rows across all groups
        self.file_info = []  # Information about each group in each file
        self.group_info = {}  # Maps (file_idx, group_key) to their locations in files
        self.lengths = {}  # Store the length of each group (for compatibility)
        self.file_group_map = []  # Maps global index to (file_idx, group_key) tuples
        self.file_sizes = []  # Store file sizes for memory management

        logger.info("Processing files to build metadata...")
        # Process each file
        for file_idx, file_path in enumerate(self.file_paths):
            logger.info(f"Processing file {file_idx + 1}/{len(self.file_paths)}: {file_path}")

            # Track groups in this file
            file_groups = set()

            # Get file size
            file_size = os.path.getsize(file_path)
            self.file_sizes.append(file_size)

            if self.memory_efficient:
                # Process in chunks for memory efficiency
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    chunk = self._parse_and_enrich_chunk(chunk)
                    self._process_chunk(chunk, file_idx, file_path, file_groups)
            else:
                # Load entire file at once for small files
                chunk = pd.read_csv(file_path)
                chunk = self._parse_and_enrich_chunk(chunk)
                self._process_chunk(chunk, file_idx, file_path, file_groups)

            # Add all groups from this file to the global mapping
            for file_group_key in file_groups:
                self.file_group_map.append(file_group_key)

        # Store unique file-group combinations for iteration
        self._group_ids = list(self.group_info.keys())
        logger.info(f"Found {len(self._group_ids)} unique file-group combinations")

    def _process_chunk(self, chunk, file_idx, file_path, file_groups):
        """
        Process a single chunk of data from a file.

        Args:
            chunk: DataFrame chunk to process
            file_idx: Index of the file being processed
            file_path: Path to the file
            file_groups: Set to track groups in this file
        """
        if len(chunk) == 0:
            return

        # Handle grouping logic based on group_cols
        if not self.group_cols:
            # No group columns - treat all data as a single group
            # Add a dummy column with the same value for all rows
            chunk["_single_group"] = "_global"
            group_col_for_groupby = "_single_group"
            logger.debug("No group columns provided - treating all data as a single group")
        elif isinstance(self.group_cols, list) and len(self.group_cols) > 1:
            # Create a composite key for multi-column groups
            chunk["_composite_group_key"] = chunk[self.group_cols].apply(lambda x: tuple(x), axis=1)
            group_col_for_groupby = "_composite_group_key"
        elif isinstance(self.group_cols, list) and len(self.group_cols) == 1:
            # Single group column in a list - extract the column name
            group_col_for_groupby = self.group_cols[0]
            logger.debug(f"Using single group column in list: {group_col_for_groupby}")
        else:
            # Single group column case (string)
            group_col_for_groupby = self.group_cols

        # Group by the composite key
        for group_key_str, group_data in chunk.groupby(group_col_for_groupby):
            # Store the original values as a tuple for reference
            if not self.group_cols:
                # For single group case, use a placeholder
                original_values = ("_global",)
            elif isinstance(self.group_cols, list):
                if len(self.group_cols) > 0:
                    # For list of group columns, extract values for each column
                    values = []
                    for col in self.group_cols:
                        values.append(group_data[col].iloc[0])
                    original_values = tuple(values)
                else:
                    # Empty list case
                    original_values = ("_global",)
            else:
                # For single group column (string), get the value
                original_values = (group_data[self.group_cols].iloc[0],)

            # Use the composite string as the group key for efficiency
            group_key = (group_key_str,)

            # Create file-specific group identifier
            file_group_key = (file_idx, group_key)
            file_groups.add(file_group_key)

            # Update group info with both composite key and original values
            if file_group_key not in self.group_info:
                self.group_info[file_group_key] = {
                    "file_path": file_path,
                    "file_idx": file_idx,
                    "group_key": group_key,
                    "original_values": original_values,  # Store original column values
                    "group_columns": self.group_cols,  # Store column names
                    "length": 0,
                    "start_idx": self.total_length,
                }

            # Update length
            group_length = len(group_data)
            self.group_info[file_group_key]["length"] += group_length
            self.total_length += group_length

            # Update categorical encoders
            self._update_encoders(group_data)

    def _update_encoders(self, data):
        """
        Update label encoders with new categorical data.

        Args:
            data: DataFrame containing the data to update encoders with
        """
        for col in self.cat_cols:
            if col in data.columns:
                # Get non-null values
                values = data[col].dropna().astype(str)
                if len(values) > 0:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        )
                        # Fit with initial values
                        self.label_encoders[col].fit(values.values.reshape(-1, 1))
                    else:
                        # Update encoder with new values
                        existing_categories = set(self.label_encoders[col].categories_[0])
                        new_values = set(values.unique()) - existing_categories
                        if new_values:
                            # Refit with all values (existing + new)
                            all_values = list(existing_categories) + list(new_values)
                            self.label_encoders[col].fit(np.array(all_values).reshape(-1, 1))

    def _get_categorical_cardinality(self, col):
        """
        Get cardinality (number of unique values) for a categorical column.
        Handles both memory-efficient and non-memory-efficient modes.
        """
        if self.data is not None and col in self.data.columns:
            # Non-memory-efficient mode: data is loaded
            return len(self.data[col].unique())
        else:
            # Memory-efficient mode: need to scan files to get cardinality
            unique_values = set()
            for file_group_key in self._group_ids:
                group_info = self.group_info[file_group_key]
                file_path = group_info["file_path"]

                # Read just the categorical column from the file
                try:
                    df_col = pd.read_csv(file_path, usecols=[col])
                    unique_values.update(df_col[col].dropna().unique())
                except (KeyError, pd.errors.ParserError):
                    # Column doesn't exist in this file, skip
                    continue
            return len(unique_values)

    def _prepare_metadata(self):
        """
        Prepare dataset metadata including dimensions, column info, and statistics.
        """
        # Create a cumulative index mapping for efficient lookup
        self.cumulative_lengths = [0]
        for file_group_key in self._group_ids:
            group_length = self.group_info[file_group_key]["length"]
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + group_length)

        # Store total dataset length
        self.dataset_length = self.cumulative_lengths[-1]

        # Prepare comprehensive metadata dictionary
        self.metadata = {
            # Dataset dimensions
            "n_targets": len(self.target_cols),
            "n_features": len(self.feature_cols),
            "n_categorical": len(self.cat_cols),
            "n_known_future": len(self.known_cols),
            "n_unknown_future": len(self.unknown_cols),
            "target_cols": self.target_cols,
            "feature_cols": self.feature_cols,
            "n_groups": len(self._group_ids),
            # Column types
            "time_col": self.time_col,
            "known_cols": self.known_cols if self.known_cols else [],
            "unknown_cols": self.unknown_cols if self.unknown_cols else [],
            "enrich_cat": self.enrich_cat if self.enrich_cat else [],
            "temporal_features": [f"{self.time_col}_{feat}" for feat in (self.enrich_cat or [])]
            if self.time_col
            else [],
        }

        # Add categorical information to metadata only if categorical columns exist
        if self.cat_cols and len(self.cat_cols) > 0:
            self.metadata["categorical_columns"] = self.cat_cols
            self.metadata["categorical_cardinalities"] = {
                col: len(self.label_encoders[col].categories_[0])
                for col in self.cat_cols
                if col in self.label_encoders
            }

        # Add group information to metadata
        # Always add group information to metadata, even for empty group_cols
        self.metadata["group_cols"] = self.group_cols

        # For empty group_cols, add special metadata indicating global grouping
        if not self.group_cols:
            self.metadata["single_group"] = True
            self.metadata["n_groups"] = 1
            logger.info("Dataset has no group columns - treating as a single global group")
        # Adding group mapping information for composite keys
        elif isinstance(self.group_cols, list) and len(self.group_cols) > 1:
            # mapping from composite key to integer id
            unique_groups = [info["group_key"][0] for info in self.group_info.values()]
            group_to_int = {group: idx for idx, group in enumerate(set(unique_groups))}

            # we create reverse mapping from integer ID to original values
            reverse_mapping = {}
            for file_group_key, info in self.group_info.items():
                group_key = info["group_key"][0]
                if "original_values" in info:
                    reverse_mapping[group_to_int[group_key]] = {
                        "composite_key": group_key,
                        "original_values": dict(zip(self.group_cols, info["original_values"])),
                    }

            # add mappings to metadata
            self.metadata["group_mapping"] = group_to_int
            self.metadata["reverse_mapping"] = reverse_mapping
            self.metadata["n_groups"] = len(group_to_int)

        # Add dataset structure information
        self.metadata["total_samples"] = self.dataset_length
        self.metadata["n_files"] = len(self.file_paths)
        self.metadata["n_file_groups"] = len(self._group_ids)
        self.metadata["memory_efficient"] = self.memory_efficient

        logger.info(f"Dataset prepared with {self.dataset_length} total samples")
        logger.info(
            f"Metadata: {self.metadata.get('n_targets', 0)} targets, "
            f"{self.metadata.get('n_features', 0)} features, "
            f"{self.metadata.get('n_categorical', 0)} categorical, "
            f"{self.metadata.get('n_groups', 0)} groups"
        )

    def _preload_data(self):
        """
        Preload all data into memory for faster access.
        Only used when memory_efficient=False.
        """
        logger.info("Preloading data into memory...")

        for file_group_key in self._group_ids:
            group_info = self.group_info[file_group_key]
            file_path = group_info["file_path"]
            group_key = group_info["group_key"]

            # Load the entire file
            if file_path not in self.data_cache:
                df = pd.read_csv(file_path)
                df = self._enrich_temporal_features(df)
                self.data_cache[file_path] = df

            # Extract group data
            df = self.data_cache[file_path]
            group_data = self._extract_group_data(df, group_key)
            group_data = self._process_group_data(group_data)

            # Store processed group data
            self.data_cache[file_group_key] = group_data

        logger.info("Data preloading completed")

    def _extract_group_data(self, df, group_key):
        """
        Extract data for a specific group from a DataFrame.

        Args:
            df: DataFrame to extract from
            group_key: Group key to extract

        Returns:
            DataFrame containing only data for the specified group
        """
        # Handle empty group_cols case (return all data)
        if not self.group_cols:
            return df

        # Create a mask for the group
        if isinstance(self.group_cols, list):
            if len(self.group_cols) > 1:
                # Multiple group columns
                mask = pd.Series(True, index=df.index)
                for i, col in enumerate(self.group_cols):
                    mask &= df[col] == group_key[i]
            elif len(self.group_cols) == 1:
                # Single group column in a list
                col = self.group_cols[0]
                mask = df[col] == group_key[0]
                logger.debug(f"Filtering on single group column in list: {col}")
            else:
                # Empty list case - return all data
                logger.debug("Empty group_cols list - returning all data")
                return df
        else:
            # Single group column as string
            mask = df[self.group_cols] == group_key[0]

        # Return the filtered data without copying
        return df[mask]

    def _process_group_data(self, group_data):
        """
        Process group data by applying encodings and transformations.

        Args:
            group_data: DataFrame containing group data

        Returns:
            Processed DataFrame
        """
        # Apply categorical encodings
        for col in self.cat_cols:
            if col in group_data.columns and col in self.label_encoders:
                # Handle NaN values
                non_null_mask = group_data[col].notna()
                if non_null_mask.any():
                    # Transform non-null values
                    values_to_transform = (
                        group_data.loc[non_null_mask, col].astype(str).values.reshape(-1, 1)
                    )
                    encoded_values = self.label_encoders[col].transform(values_to_transform)
                    group_data.loc[non_null_mask, col] = encoded_values.flatten()

        return group_data

    def _infer_frequency(self, time_col_data):
        """
        Infer the frequency of the time series data with improved logging.

        Args:
            time_col_data: Time column data

        Returns:
            Inferred frequency
        """
        logger.info("Inferring frequency from time column data...")

        try:
            if pd.api.types.is_datetime64_dtype(time_col_data):
                # For datetime, calculate timedeltas
                time_diffs = time_col_data.diff().dropna()
                # Get the most common difference (mode)
                freq = time_diffs.mode().iloc[0]
                logger.info(f"Inferred frequency for datetime data: {freq}")
            else:
                # For numeric time, calculate differences
                time_diffs = np.diff(time_col_data)
                # Get the most common difference (mode)
                freq = pd.Series(time_diffs).mode().iloc[0]
                logger.info(f"Inferred frequency for numeric data: {freq}")

            return freq

        except Exception as e:
            logger.warning(f"Failed to infer frequency: {e}. Using default frequency of 1.")
            return 1

    def __len__(self) -> int:
        """
        Return the number of groups in the dataset.

        Returns:
            Number of unique groups
        """
        return len(self._group_ids)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get all data for a specific group.

        Args:
            idx: Group index (0 to number_of_groups-1)

        Returns:
            Dictionary containing all data for the specified group
        """
        if idx >= len(self._group_ids):
            raise IndexError(
                f"Group index {idx} out of range. Available groups: {len(self._group_ids)}"
            )

        # Get the file-group key for this group index
        file_group_key = self._group_ids[idx]

        # Get all data for this group
        if self.memory_efficient:
            # Load data on demand
            group_data = self._load_group_data_on_demand(file_group_key)
        else:
            # Use preloaded data
            group_data = self.data_cache[file_group_key]

        # Sort by time if time column exists
        if self.time_col in group_data.columns:
            group_data = group_data.sort_values(by=self.time_col)

        # Get group ID - for composite keys, use integer mapping if available
        group_key = file_group_key[1]
        if (
            isinstance(self.group_cols, list)
            and len(self.group_cols) > 1
            and "group_mapping" in self.metadata
        ):
            # Use the integer mapping for efficiency
            group_id = self.metadata["group_mapping"].get(group_key[0], group_key)
        else:
            group_id = group_key

        # Extract all features and targets for this group efficiently
        if len(group_data) == 0:
            logger.warning(f"Empty group data found for group {group_id}.")  # noqa
            x = torch.empty(0, len(self.feature_cols), dtype=torch.float32)
            y = torch.empty(0, len(self.target_cols), dtype=torch.float32)
            time_indices = []
        else:
            # Separate numerical and categorical features
            num_feature_cols = [col for col in self._feature_cols if col in self._num_cols]

            # Extract numerical features using vectorized operations
            if num_feature_cols:
                num_feature_values = group_data[num_feature_cols].values
                x = torch.tensor(num_feature_values, dtype=torch.float32)
            else:
                # If no numerical features, create empty tensor with correct shape
                x = torch.empty((len(group_data), 0), dtype=torch.float32)

            # extracting targets using vectorized operations
            target_values = group_data[self._target_cols].values
            y = torch.tensor(target_values, dtype=torch.float32)

            # Extract time indices
            time_indices = (
                group_data[self.time_col].tolist() if self.time_col in group_data.columns else []
            )

        # Prepare the group sample
        sample = {
            "x": x,  # All features for this group [seq_len, n_features]
            "y": y,  # All targets for this group [seq_len, n_targets]
            "group_id": group_id,  # Group identifier
            "past_time": time_indices,  # All time indices for this group
            "future_time": time_indices,  # Same as past_time for now
            "seq_len": len(group_data),  # Length of the sequence
        }

        return sample

    def _load_group_data_on_demand(self, file_group_key):
        """
        Load group data on demand for memory-efficient mode.

        Args:
            file_group_key: Tuple of (file_idx, group_key)

        Returns:
            DataFrame containing the group data
        """
        group_info = self.group_info[file_group_key]
        file_path = group_info["file_path"]
        group_key = group_info["group_key"]

        # Load the file (could be cached at file level)
        df = pd.read_csv(file_path)
        df = self._enrich_temporal_features(df)

        # Extract and process group data
        group_data = self._extract_group_data(df, group_key)
        return self._process_group_data(group_data)
