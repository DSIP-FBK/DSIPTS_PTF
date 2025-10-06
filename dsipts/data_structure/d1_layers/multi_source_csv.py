"""
Multi-source CSV implementation for D1 layer.

This module provides the MultiSourceTSDataSet class for loading and preprocessing
time series data from multiple CSV files with support for temporal enrichment
and categorical encoding.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch

from .base_d1 import BaseD1Layer
from .utils import (
    apply_label_encoding,
    enrich_temporal_features,
    extract_group_data,
    get_categorical_cardinality,
    parse_and_enrich_chunk,
    process_group_data,
    update_label_encoders,
    validate_enrich_cat,
)

logger = logging.getLogger(__name__)


class MultiSourceTSDataSet(BaseD1Layer):
    """
    D1 layer dataset for multi-source time series data.

    Handles loading, preprocessing, and categorical encoding of time series data
    from multiple CSV files. Supports temporal enrichment and memory-efficient
    processing for large datasets.

    Features:
    - Multi-file data loading with group-based organization
    - Categorical encoding with label encoders
    - Temporal feature enrichment (hour, day-of-week, etc.)
    - Memory-efficient chunk-based processing
    - Automatic frequency inference
    """

    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        dataframes: Optional[List[pd.DataFrame]] = None,
        group_cols: Optional[Union[str, List[str]]] = None,
        time_col: str = "time",
        target_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        known_cols: Optional[List[str]] = None,
        unknown_cols: Optional[List[str]] = None,
        enrich_cat: Optional[List[str]] = None,
        global_forecasting: bool = False,
        weights: Optional[str] = None,
        memory_efficient: bool = False,
        chunk_size: int = 10000,
    ):
        """
        Initialize the MultiSourceTSDataSet.

        Args:
            file_paths: List of paths to CSV files containing time series data (optional if dataframes provided)
            dataframes: List of pandas DataFrames containing time series data (optional if file_paths provided)
            group_cols: Column(s) that identify unique time series groups
            time_col: Column containing time/date information
            target_cols: Columns to use as targets (y)
            cat_cols: Categorical columns that need encoding
            num_cols: Numerical columns (if None, all non-categorical columns are treated as numerical)
            known_cols: Known columns at prediction time (feature columns that are known)
            unknown_cols: Unknown columns at prediction time (if None, all target_cols are considered unknown)
            enrich_cat: List of temporal categorical variables to create. Supported values: ['hour', 'dow', 'month', 'minute']
            global_forecasting: If True, use global forecasting. If False (default)
                and multiple groups exist, add group as categorical known variable.
            weights: Name of weights column
            memory_efficient: Whether to use memory-efficient mode
            chunk_size: Chunk size for processing data (used in memory-efficient mode)
        """
        super().__init__()

        # Validate input - must provide either file_paths or dataframes
        if not file_paths and not dataframes:
            raise ValueError("Must provide either file_paths or dataframes")
        if file_paths and dataframes:
            raise ValueError("Cannot provide both file_paths and dataframes - choose one")

        # Basic configuration
        self.file_paths = file_paths or []
        self.dataframes = dataframes or []
        self.use_dataframes = bool(dataframes)
        self._time_col = time_col
        self._weights = weights

        # Create pseudo file paths for dataframes for consistent processing
        if self.use_dataframes:
            self.file_paths = [f"dataframe_{i}" for i in range(len(self.dataframes))]

        # Handle group columns (can be single column or multiple)
        if group_cols is None or (isinstance(group_cols, list) and len(group_cols) == 0):
            self._group_cols = []
            if global_forecasting:
                raise ValueError("Global forecasting requires group columns")
        elif isinstance(group_cols, str):
            self._group_cols = [group_cols]
        else:
            self._group_cols = group_cols

        # Initialize attributes with proper defaults
        self._target_cols = target_cols or []
        self._cat_cols = cat_cols or []
        self._num_cols = num_cols or []
        self._enrich_cat = enrich_cat or []
        self.global_forecasting = global_forecasting

        self._known_cols = known_cols or []
        self._unknown_cols = unknown_cols or list(self._target_cols) if self._target_cols else []
        self._original_known_cols = self._known_cols.copy() if self._known_cols else []

        # if global_forecasting =False and multiple groups exist, add group columns to categorical and known variables
        self._apply_global_forecasting_logic()

        # Infer feature_cols automatically from headers and other specifications
        self._feature_cols = self._infer_feature_columns()
        self._validate_enrich_cat()

        # Flag to track if temporal features have been added to categorical columns
        self._is_file_read = False

        # If _num_cols not specified, infer from feature_cols and cat_cols
        if not self._num_cols:
            all_cols = self._feature_cols + self._target_cols
            self._num_cols = [c for c in all_cols if c not in self._cat_cols]

        # Internal state
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size

        # Initialize data structures
        self.group_info = {}
        self.label_encoders = {}
        self.file_group_map = []
        self.cached_data = {} if not memory_efficient else None

        # Process the data (files or dataframes)
        if self.use_dataframes:
            self._process_dataframes()
        else:
            self._process_files()

        self._prepare_metadata()
        if not memory_efficient:
            self._preload_data()

    def _infer_feature_columns(self) -> List[str]:
        """
        Infer feature columns from specified columns or data headers.
        """
        # Use explicitly specified columns (including targets)
        if self._known_cols or self._num_cols or self._cat_cols or self._target_cols:
            feature_cols = list(dict.fromkeys(self._known_cols + self._num_cols + self._cat_cols + self._target_cols))
            if self._enrich_cat:
                feature_cols = list(dict.fromkeys(feature_cols + self._enrich_cat))
            return feature_cols

        # Infer from data headers
        if self.use_dataframes:
            if not self.dataframes:
                raise ValueError("Cannot infer feature columns: no dataframes provided")
            all_columns = self.dataframes[0].columns.tolist()
        else:
            if not self.file_paths:
                raise ValueError("Cannot infer feature columns: no file paths provided")
            try:
                sample_df = pd.read_csv(self.file_paths[0], nrows=1)
                all_columns = sample_df.columns.tolist()
            except Exception as e:
                raise ValueError(f"Could not read file {self.file_paths[0]} to infer columns: {e}")

        # Exclude special columns from features
        special_columns = {self._weights} if self._weights else set()
        feature_cols = [col for col in all_columns if col not in special_columns]

        # Add temporal features if specified
        if self._enrich_cat:
            feature_cols = list(dict.fromkeys(feature_cols + self._enrich_cat))

        return feature_cols

    def _apply_global_forecasting_logic(self):
        """Apply global forecasting logic based on groups and global_forecasting flag."""
        # No action needed if no groups or global forecasting enabled
        if not self._group_cols or self.global_forecasting:
            return

        # For local forecasting, add group columns to categorical and known columns
        for group_col in self._group_cols:
            if group_col not in self._cat_cols:
                self._cat_cols.append(group_col)
            if group_col not in self._known_cols:
                self._known_cols.append(group_col)

    def _validate_enrich_cat(self):
        """Validate the enrich_cat parameter and update categorical and known columns"""
        validate_enrich_cat(self._enrich_cat)

        for option in self._enrich_cat:
            if option not in self._cat_cols:
                self._cat_cols.append(option)
            if option not in self._known_cols:
                self._known_cols.append(option)

    def _parse_and_enrich_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and enrich a chunk of data.
        """
        # If time_col is missing, create one as integer range
        if self._time_col not in chunk.columns:
            chunk["time"] = range(len(chunk))
            self._time_col = "time"

        # Set of mandatory columns (cat, num, target, group, time)
        mandatory_cols = set(self._cat_cols + self._num_cols + self._target_cols)

        # Always include time column
        if self._time_col and self._time_col in chunk.columns:
            mandatory_cols.add(self._time_col)

        # Always include group columns
        if isinstance(self.group_cols, list):
            for col in self.group_cols:
                if col in chunk.columns:
                    mandatory_cols.add(col)
        elif self.group_cols and self.group_cols in chunk.columns:
            mandatory_cols.add(self.group_cols)

        # Use utility function for parsing and enrichment
        chunk = parse_and_enrich_chunk(chunk, self._time_col, self._enrich_cat, list(mandatory_cols))

        # Update encoders after enrichment
        if self._enrich_cat:
            self._update_encoders(chunk)
        return chunk

    def _enrich_temporal_features(self, dataset):
        """
        Enrich dataset with temporal categorical features.

        This method is now a wrapper around the utility function.
        It handles the column management logic specific to this class.
        """
        if not self._enrich_cat or self._time_col not in dataset.columns:
            return dataset

        dataset = enrich_temporal_features(dataset, self._enrich_cat, self._time_col)

        # Handle column management (class-specific logic)
        enriched_features = [col for col in self._enrich_cat if col in dataset.columns]

        # Add temporal categorical features to cat_cols and known_cols only once
        if enriched_features and not self._is_file_read:
            if self._cat_cols is None:
                self._cat_cols = []

            for feature in enriched_features:
                if feature not in self._cat_cols:
                    self._cat_cols.append(feature)
                if self._known_cols is not None and feature not in self._known_cols:
                    self._known_cols.append(feature)

            self._is_file_read = True
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
        """Get the known columns."""
        return self._known_cols or []

    @property
    def unknown_cols(self) -> Optional[List[str]]:
        """Get the unknown columns."""
        return self._unknown_cols or []

    def _process_dataframes(self):
        """
        Process pandas DataFrames to extract group information and update encoders.
        Similar to _process_files but works with in-memory DataFrames.
        """
        # Initialize data structures
        self.total_length = 0
        self.file_info = []
        self.group_info = {}
        self.lengths = {}
        self.file_group_map = []
        self.file_sizes = []

        # Process each DataFrame
        for df_idx, df in enumerate(self.dataframes):
            self.file_sizes.append(len(df))
            file_groups = set()
            self._process_dataframe_chunk(df, df_idx, f"dataframe_{df_idx}", file_groups)
            self.file_group_map.append(file_groups)

        self._group_ids = list(self.group_info.keys())

    def _process_files(self):
        """
        Process each file to extract group information and update encoders.
        Builds a mapping of where each group's data is located and
        calculates the total length of each group.
        """

        self.total_length = 0  # Total number of rows across all groups
        self.file_info = []  # Information about each group in each file
        self.group_info = {}  # Maps (file_idx, group_key) to their locations in files
        self.lengths = {}  # Store the length of each group (for compatibility)
        self.file_group_map = []  # Maps global index to (file_idx, group_key) tuples
        self.file_sizes = []  # Store file sizes for memory management

        logger.info("Processing files to build metadata...")
        for file_idx, file_path in enumerate(self.file_paths):
            logger.info(f"Processing file {file_idx + 1}/{len(self.file_paths)}: {file_path}")
            file_groups = set()

            file_size = os.path.getsize(file_path)  # Get file size
            self.file_sizes.append(file_size)

            if self.memory_efficient:  # Process in chunks for memory efficiency
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    chunk = self._parse_and_enrich_chunk(chunk)
                    self._process_chunk(chunk, file_idx, file_path, file_groups)
            else:  # Load entire file at once for small files
                chunk = pd.read_csv(file_path)
                chunk = self._parse_and_enrich_chunk(chunk)
                self._process_chunk(chunk, file_idx, file_path, file_groups)

            for file_group_key in file_groups:  # Add all groups from this file to the global mapping
                self.file_group_map.append(file_group_key)

        # Store unique file-group combinations for iteration
        self._group_ids = list(self.group_info.keys())

    def _process_dataframe_chunk(self, chunk, df_idx, df_name, file_groups):
        """
        Process a DataFrame chunk.
        """

        if len(chunk) == 0:
            return

        # Handle grouping logic based on group_cols (same as _process_chunk)
        if not self.group_cols:
            # No group columns - treat all data as a single group
            chunk["_single_group"] = "_global"
            group_col_for_groupby = "_single_group"
        elif isinstance(self.group_cols, list) and len(self.group_cols) > 1:
            # Create a composite key for multi-column groups
            chunk["_composite_group_key"] = chunk[self.group_cols].apply(lambda x: tuple(x), axis=1)
            group_col_for_groupby = "_composite_group_key"
        elif isinstance(self.group_cols, list) and len(self.group_cols) == 1:
            # Single group column in a list
            group_col_for_groupby = self.group_cols[0]
        else:
            # Single group column case (string)
            group_col_for_groupby = self.group_cols

        # Group by the composite key
        for group_key, group_data in chunk.groupby(group_col_for_groupby):
            file_group_key = (df_name, group_key)
            file_groups.add(file_group_key)

            # Store group information
            if file_group_key not in self.group_info:
                self.group_info[file_group_key] = {
                    "length": 0,
                    "df_idx": df_idx,
                    "group_key": (group_key,),
                }

                # Store original values for composite keys if needed
                if isinstance(self.group_cols, list) and len(self.group_cols) > 1:
                    # For composite keys, store the original column values
                    first_row = group_data.iloc[0]
                    original_values = [first_row[col] for col in self.group_cols]
                    self.group_info[file_group_key]["original_values"] = original_values
            self.group_info[file_group_key]["length"] += len(group_data)
            self._update_encoders(group_data)

            # Add to file group map
            for _ in range(len(group_data)):
                self.file_group_map.append(file_group_key)

    def _process_chunk(self, chunk, file_idx, file_path, file_groups):
        """
        Process a single chunk of data from a file.
        """
        if len(chunk) == 0:
            return

        # Handle grouping logic based on group_cols
        if not self.group_cols:
            # No group columns - treat all data as a single group
            chunk["_single_group"] = "_global"
            group_col_for_groupby = "_single_group"
        elif isinstance(self.group_cols, list) and len(self.group_cols) > 1:
            # Create a composite key for multi-column groups
            chunk["_composite_group_key"] = chunk[self.group_cols].apply(lambda x: tuple(x), axis=1)
            group_col_for_groupby = "_composite_group_key"
        elif isinstance(self.group_cols, list) and len(self.group_cols) == 1:
            # Single group column in a list
            group_col_for_groupby = self.group_cols[0]
        else:
            group_col_for_groupby = self.group_cols

        for group_key_str, group_data in chunk.groupby(group_col_for_groupby):
            if not self.group_cols:
                original_values = ("_global",)
            elif isinstance(self.group_cols, list):
                if len(self.group_cols) > 0:
                    values = []
                    for col in self.group_cols:
                        values.append(group_data[col].iloc[0])
                    original_values = tuple(values)
                else:
                    original_values = ("_global",)
            else:
                original_values = (group_data[self.group_cols].iloc[0],)

            group_key = (group_key_str,)
            file_group_key = (file_idx, group_key)
            file_groups.add(file_group_key)

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

        This method is now a wrapper around the utility function.
        """
        update_label_encoders(data, self._cat_cols, self.label_encoders, self._group_cols, self._enrich_cat)

    def _apply_label_encoding(self, data):
        """Apply label encoding to categorical columns in the data.

        This method is now a wrapper around the utility function.
        """
        return apply_label_encoding(data, self._cat_cols, self.label_encoders)

    def _get_categorical_cardinality(self, col):
        """
        Get the cardinality (number of unique values) for a categorical column.

        This method is now a wrapper around the utility function.
        """
        return get_categorical_cardinality(col, self.label_encoders, self.data)

    def _prepare_metadata(self):
        """Prepare dataset metadata including dimensions, column info, and statistics."""

        # Create a cumulative index mapping for efficient lookup
        self.cumulative_lengths = [0]
        for file_group_key in self._group_ids:
            group_length = self.group_info[file_group_key]["length"]
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + group_length)

        # Store total dataset length
        self.dataset_length = self.cumulative_lengths[-1]

        # Calculate feature indices
        cat_indices = [self.feature_cols.index(col) for col in (self._cat_cols or []) if col in self.feature_cols]
        known_indices = [self.feature_cols.index(col) for col in (self._known_cols or []) if col in self.feature_cols]
        unknown_indices = [self.feature_cols.index(col) for col in (self._unknown_cols or []) if col in self.feature_cols]
        target_indices = [self.feature_cols.index(col) for col in (self._target_cols or []) if col in self.feature_cols]

        # Prepare comprehensive metadata dictionary
        self.metadata = {
            # Dataset dimensions (counts)
            "n_targets": len(self._target_cols),
            "n_features": len(self.feature_cols),
            "n_categorical": len(self._cat_cols),
            "n_known": len(self.known_cols) if self.known_cols else 0,
            "n_unknown": len(self.unknown_cols) if self.unknown_cols else 0,
            # Column names
            "target_cols": self._target_cols,
            "feature_cols": self.feature_cols,
            # Feature indices
            "idx_categorical": cat_indices,
            "idx_known": known_indices,
            "idx_unknown": unknown_indices,
            "idx_targets": target_indices,
            # Group information
            "n_groups": len(self._group_ids),
            # Column types and temporal information
            "time_col": self._time_col,
            "known_cols": self.known_cols if self.known_cols else [],
            "unknown_cols": self.unknown_cols if self.unknown_cols else [],
            "enrich_cat": self._enrich_cat if self._enrich_cat else [],
        }

        # Add categorical information to metadata only if categorical columns exist
        if self._cat_cols and len(self._cat_cols) > 0:
            self.metadata["categorical_columns"] = self._cat_cols

            # Enhanced categorical cardinality information
            cardinalities = {}
            categorical_mappings = {}

            # Process all categorical columns
            for col in self._cat_cols:
                # Handle group columns
                if col in self.group_cols and hasattr(self, "group_info") and self.group_info:
                    if col in self.label_encoders:
                        n_categories = len(self.label_encoders[col].classes_)
                        group_values = self.label_encoders[col].classes_.tolist()
                    else:
                        group_values = set()
                        for group_key, info in self.group_info.items():
                            if "original_values" in info and info["group_columns"] == self.group_cols:
                                if len(info["original_values"]) == 1:
                                    group_values.add(str(info["original_values"][0]))
                        group_values = sorted(list(group_values))
                        n_categories = len(group_values)

                    cardinalities[col] = n_categories
                    categorical_mappings[col] = {
                        "categories": group_values,
                        "cardinality": n_categories,
                        "feature_index": self.feature_cols.index(col) if col in self.feature_cols else -1,
                    }

                # Handle regular categorical columns (non-group)
                elif col in self.label_encoders:
                    n_categories = len(self.label_encoders[col].classes_)
                    categories = self.label_encoders[col].classes_.tolist()
                    cardinalities[col] = n_categories
                    categorical_mappings[col] = {
                        "categories": categories,
                        "cardinality": n_categories,
                        "feature_index": self.feature_cols.index(col) if col in self.feature_cols else -1,
                    }

            # Update metadata with the dictionaries
            self.metadata["categorical_cardinalities"] = cardinalities
            self.metadata["categorical_mappings"] = categorical_mappings

            # Handle group mapping
            if isinstance(self.group_cols, list) and len(self.group_cols) > 1:
                # mapping from composite key to integer id for multi-column groups
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

            # Add group mapping for single-column group keys
            elif self.group_cols:
                # group_cols is a string or single-item list
                unique_groups = [info["group_key"] for info in self.group_info.values()]
                group_to_int = {group: idx for idx, group in enumerate(sorted(set(unique_groups)))}
                reverse_mapping = {idx: group for group, idx in group_to_int.items()}
                self.metadata["group_mapping"] = group_to_int
                self.metadata["reverse_mapping"] = reverse_mapping
                self.metadata["n_groups"] = len(group_to_int)

        # Add dataset structure information
        self.metadata["total_samples"] = self.dataset_length
        self.metadata["n_files"] = len(self.file_paths)
        self.metadata["n_file_groups"] = len(self._group_ids)
        self.metadata["memory_efficient"] = self.memory_efficient
        self.metadata["file_paths"] = self.file_paths
        self.metadata["global_forecasting"] = self.global_forecasting

    def _update_metadata_after_preload(self):
        """Update metadata after preloading data to include enriched features."""
        if not hasattr(self, "metadata") or not self.metadata:
            return

        # Update categorical cardinalities and mappings for enriched features
        if hasattr(self, "_enrich_cat") and self._enrich_cat:
            if "categorical_cardinalities" not in self.metadata:
                self.metadata["categorical_cardinalities"] = {}
            if "categorical_mappings" not in self.metadata:
                self.metadata["categorical_mappings"] = {}

            cardinalities = self.metadata["categorical_cardinalities"]
            categorical_mappings = self.metadata["categorical_mappings"]

            for feature in self._enrich_cat:
                if feature in self.label_encoders:
                    categories = self.label_encoders[feature].classes_
                    n_categories = len(categories)
                    cardinalities[feature] = n_categories

                    # Also update categorical_mappings
                    categorical_mappings[feature] = {
                        "categories": categories.tolist(),
                        "cardinality": n_categories,
                        "feature_index": self.feature_cols.index(feature) if feature in self.feature_cols else -1,
                    }

            self.metadata["categorical_cardinalities"] = cardinalities
            self.metadata["categorical_mappings"] = categorical_mappings

            known_indices = [self.feature_cols.index(col) for col in self._known_cols if col in self.feature_cols]
            self.metadata["idx_known"] = known_indices
            self.metadata["n_known"] = len(self._known_cols) if self._known_cols else 0

            self.metadata["known_cols"] = self._known_cols.copy() if self._known_cols else []
            self.metadata["original_known_cols"] = self._original_known_cols.copy() if self._original_known_cols else None

    def _preload_data(self):
        """Preload all data into memory for faster access (memory_efficient=False)."""
        self.cached_data = {}

        for file_group_key in self._group_ids:
            if self.use_dataframes:
                df_name, group_key = file_group_key
                group_data = self._load_group_data_from_dataframe(file_group_key)
            else:
                file_path, group_key = file_group_key
                group_data = self._load_group_data(file_group_key)
            self.cached_data[file_group_key] = group_data

        self._update_metadata_after_preload()

    def _load_group_data_from_dataframe(self, file_group_key):
        """
        Load data for a specific group from a DataFrame.
        """
        df_name, group_key = file_group_key
        df_idx = self.group_info[file_group_key]["df_idx"]
        df = self.dataframes[df_idx]

        if not self.group_cols:
            group_data = df.copy()
        elif isinstance(self.group_cols, list) and len(self.group_cols) > 1:
            mask = df[self.group_cols].apply(lambda x: tuple(x), axis=1) == group_key
            group_data = df[mask].copy()
        elif isinstance(self.group_cols, list) and len(self.group_cols) == 1:
            actual_group_value = group_key[0] if isinstance(group_key, tuple) else group_key
            group_data = df[df[self.group_cols[0]] == actual_group_value].copy()
        else:
            group_data = df[df[self.group_cols] == group_key].copy()
        return self._parse_and_enrich_chunk(group_data)

    def _load_group_data(self, file_group_key):
        """
        Load data for a specific group from file.
        """
        file_idx, group_key = file_group_key
        file_path = self.file_paths[file_idx]

        # Load and preprocess entire file (parse time, enrich temporal, update encoders)
        df = pd.read_csv(file_path)

        # Extract and process group data
        df = self._parse_and_enrich_chunk(df)

        # Extract group data without applying encoding
        group_data = self._extract_group_data(df, group_key)
        return group_data

    def _extract_group_data(self, df, group_key):
        """
        Extract data for a specific group from a DataFrame.

        This method handles class-specific group key normalization then uses the utility function.
        """
        # Normalize group_key: unwrap nested tuple like ((a, b),) -> (a, b)
        if isinstance(group_key, tuple) and len(group_key) == 1 and isinstance(group_key[0], tuple):
            group_key = group_key[0]

        # Use utility function for extraction
        return extract_group_data(df, group_key, self.group_cols)

    def _process_group_data(self, group_data):
        """
        Process group data by applying encodings and transformations.

        This method is now a wrapper around the utility function.
        """
        return process_group_data(group_data, self._cat_cols, self.label_encoders)

    def _load_group_data_on_demand(self, file_group_key):
        """
        Load group data on demand for memory-efficient mode.
        """
        group_info = self.group_info[file_group_key]
        file_path = group_info["file_path"]
        group_key = group_info["group_key"]

        # Load the file (could be cached at file level)
        df = pd.read_csv(file_path)
        df = self._parse_and_enrich_chunk(df)

        # Extract group data without applying encoding
        group_data = self._extract_group_data(df, group_key)
        return group_data

    def __len__(self) -> int:
        """
        Return the number of groups in the dataset.
        """
        return len(self._group_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample by index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing the sample data with keys:
            - group_id: Group identifier
            - past_time: Time index for the sample
            - x: Feature tensor (numerical features)
            - x_cat: Categorical feature tensor (if categorical columns exist)
            - y: Target tensor
            - metadata: Additional metadata
        """
        if idx >= len(self._group_ids):
            raise IndexError(f"Group index {idx} out of range. Available groups: {len(self._group_ids)}")

        file_group_key = self._group_ids[idx]

        if not self.memory_efficient and file_group_key in self.cached_data:
            group_data = self.cached_data[file_group_key]
        else:
            if self.use_dataframes:
                group_data = self._load_group_data_from_dataframe(file_group_key)
            else:
                group_data = self._load_group_data(file_group_key)

        # Sort by time if time column exists
        if self._time_col in group_data.columns:
            group_data = group_data.sort_values(by=self._time_col)

        # Get group ID - using integer encoding if mapping exists
        group_key = file_group_key[1]
        if "group_mapping" in self.metadata:
            # For single-col groups, group_key is str, but mapping expects tuple
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_id = self.metadata["group_mapping"].get(group_key, group_key)
        else:
            group_id = group_key

        # Extract all features and targets for this group
        if len(group_data) == 0:
            logger.warning(f"Empty group data found for group {group_id}.")
            x = torch.empty(0, len(self.feature_cols), dtype=torch.float32)
            y = torch.empty(0, len(self._target_cols), dtype=torch.float32)
            time_indices = []
        else:
            # Separate numerical and categorical features
            num_feature_cols = [col for col in self._feature_cols if col in self._num_cols]
            cat_feature_cols = [col for col in self._feature_cols if col in self._cat_cols]

            # Extract numerical features using vectorized operations
            if num_feature_cols:
                num_feature_values = group_data[num_feature_cols].values
                x_num = torch.tensor(num_feature_values, dtype=torch.float32)
            else:
                # If no numerical features, create empty tensor with correct shape
                x_num = torch.empty((len(group_data), 0), dtype=torch.float32)

            # Extract categorical features if they exist
            if cat_feature_cols:
                # Apply label encoding to categorical features only at retrieval time
                encoded_cat_data = group_data[cat_feature_cols].copy()
                for col in cat_feature_cols:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        values_str = group_data[col].astype(str)
                        # Safe mapping: unknown categories -> -1
                        mapping = {cls: i for i, cls in enumerate(le.classes_)}
                        encoded_cat_data[col] = values_str.map(mapping).fillna(-1).astype(int)
                cat_feature_values = encoded_cat_data.values
                x_cat = torch.tensor(cat_feature_values, dtype=torch.long)
            else:
                x_cat = torch.empty((len(group_data), 0), dtype=torch.long)

            # Combine features in the correct order to match feature_cols
            feature_tensors = []
            for col in self.feature_cols:
                if col in num_feature_cols:
                    col_idx = num_feature_cols.index(col)
                    feature_tensors.append(x_num[:, col_idx : col_idx + 1])
                elif col in cat_feature_cols:
                    col_idx = cat_feature_cols.index(col)
                    feature_tensors.append(x_cat[:, col_idx : col_idx + 1].float())
                else:
                    feature_tensors.append(torch.zeros((len(group_data), 1), dtype=torch.float32))

            x = torch.cat(feature_tensors, dim=-1)

            # extracting targets using vectorized operations
            target_values = group_data[self._target_cols].values
            y = torch.tensor(target_values, dtype=torch.float32)

            # Extract time indices
            time_indices = group_data[self._time_col].tolist() if self._time_col in group_data.columns else []

        # Prepare categorical information as ordered lists
        cat_cols_list = []
        cat_cardinalities_list = []

        if hasattr(self, "_cat_cols") and self._cat_cols:
            for col in self._cat_cols:
                if col in self.label_encoders:
                    cat_cols_list.append(col)
                    cat_cardinalities_list.append(len(self.label_encoders[col].classes_))

        # Prepare the group sample
        sample = {
            "x": x,  # All features for this group [seq_len, n_features]
            "y": y,  # All targets for this group [seq_len, n_targets]
            "group_id": group_id,  # Group identifier
            "past_time": time_indices,  # All time indices for this group
            "future_time": time_indices,  # Same as past_time for now
            "seq_len": len(group_data),  # Length of the sequence
            "cat_cols": cat_cols_list,  # Ordered list of categorical column names
            "cat_cardinalities": cat_cardinalities_list,  # Ordered list of cardinalities (same order as cat_cols)
        }

        return sample
