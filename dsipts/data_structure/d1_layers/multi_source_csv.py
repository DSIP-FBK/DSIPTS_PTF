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
        past_cols: Optional[List[str]] = None,
        future_cols: Optional[List[str]] = None,
        enrich_cat: Optional[List[str]] = None,
        global_forecasting: bool = False,
        weights: Optional[str] = None,
        memory_efficient: bool = False,
        chunk_size: int = 10000,
        read_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the MultiSourceTSDataSet.

        Args:
            file_paths: List of paths to CSV files containing time series data (optional if dataframes provided)
            dataframes: List of pandas DataFrames containing time series data (optional if file_paths provided)
            group_cols: Column(s) that identify unique time series groups
            time_col: Column containing time/date information
            target_cols: Columns to use as targets (y)
            cat_cols: Categorical columns
            num_cols: Numerical columns (if None, all non-categorical columns are treated as numerical)
            past_cols: Columns available in past sequence (if None, all features used)
            future_cols: Columns available in future sequence
                (if None, temporal + group features used)
            enrich_cat: List of temporal categorical variables to create. Supported values: ['hour', 'dow', 'month', 'minute']
            global_forecasting:  set as False by default
            weights: Name of weights column
            memory_efficient: Whether to use memory-efficient mode
            chunk_size: Chunk size for processing data (used in memory-efficient mode)
            read_options: Dictionary of options to pass to pandas read functions
                Examples:
                - {'sep': ';'} for semicolon-separated files
                - {'na_values': ['NA', 'NULL', '-9999']} for custom missing
                - {'usecols': ['col1', 'col2']} to read specific columns
                - {'dtype': {'col1': 'int32'}} to specify data types
        """
        super().__init__()

        # Store read options for file reading
        self.read_options = read_options if read_options is not None else {}

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

        self._past_cols = past_cols or []
        self._future_cols = future_cols or []
        self._original_future_cols = self._future_cols.copy() if self._future_cols else []

        # if global_forecasting =False and multiple groups exist, add group columns to categorical and future variables
        self._apply_global_forecasting_logic()

        # Infer feature_cols automatically from headers and other specifications
        self._feature_cols = self._infer_feature_columns()
        self._validate_enrich_cat()

        self._is_file_read = False  # Flag to track if temporal features have been added to cat columns

        if not self._num_cols:  # infer from feature_cols and cat_cols
            all_cols = self._feature_cols + self._target_cols
            self._num_cols = [c for c in all_cols if c not in self._cat_cols]

        # Internal state - handle memory_efficient for DataFrames
        if self.use_dataframes and memory_efficient:
            logger.warning(
                "Memory-efficient mode is not supported when loading from DataFrames. "
                "All data will be processed in-memory. Setting memory_efficient=False."
            )
            self.memory_efficient = False
        else:
            self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size

        # Initialize data structures
        self.group_info = {}
        self.label_encoders = {}
        self.file_group_map = []
        self.cached_data = {} if not self.memory_efficient else None

        # Process the data (unified for files and dataframes)
        self._process_data()

        self._prepare_metadata()
        if not self.memory_efficient:
            self._preload_data()

    def _infer_feature_columns(self) -> List[str]:
        """
        Infer feature columns from specified columns or data headers.
        """
        # Use explicitly specified columns (including targets)
        if self._past_cols or self._num_cols or self._cat_cols or self._target_cols:
            feature_cols = list(dict.fromkeys(self._past_cols + self._num_cols + self._cat_cols + self._target_cols))
            if self._enrich_cat:
                feature_cols = list(dict.fromkeys(feature_cols + self._enrich_cat))
            return feature_cols

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

        # Exclude special columns (weights) from features
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

        # For local forecasting, add group columns to categorical and future columns
        for group_col in self._group_cols:
            if group_col not in self._cat_cols:
                self._cat_cols.append(group_col)
            if group_col not in self._past_cols:
                self._past_cols.append(group_col)

    def _validate_enrich_cat(self):
        """Validate the enrich_cat parameter and update categorical, future and past columns"""
        validate_enrich_cat(self._enrich_cat)
        for option in self._enrich_cat:
            if option not in self._cat_cols:
                self._cat_cols.append(option)
            if option not in self._future_cols:
                self._future_cols.append(option)
            if option not in self._past_cols:
                self._past_cols.append(option)

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
        Wrapper function to enrich dataset with temporal categorical features.
        """
        if not self._enrich_cat or self._time_col not in dataset.columns:
            return dataset

        dataset = enrich_temporal_features(dataset, self._enrich_cat, self._time_col)

        # Handle column management (class-specific logic)
        enriched_features = [col for col in self._enrich_cat if col in dataset.columns]

        # Add temporal categorical features to cat_cols and future_cols only once
        if enriched_features and not self._is_file_read:
            if self._cat_cols is None:
                self._cat_cols = []

            for feature in enriched_features:
                if feature not in self._cat_cols:
                    self._cat_cols.append(feature)
                if self._future_cols is not None and feature not in self._future_cols:
                    self._future_cols.append(feature)

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
    def past_cols(self) -> Optional[List[str]]:
        """Get the past columns."""
        return self._past_cols or []

    @property
    def future_cols(self) -> Optional[List[str]]:
        """Get the future columns."""
        return self._future_cols or []

    def _data_chunk_generator(self):
        """
        Generator that yields data chunks from either files or DataFrames.
        Supports multiple file formats (CSV, Parquet, JSON, etc.) with custom read options.

        Yields:
            tuple: (source_idx, source_name, chunk_dataframe)
        """
        if self.use_dataframes:
            # Yield entire DataFrames
            for idx, df in enumerate(self.dataframes):
                source_name = f"dataframe_{idx}"
                self.file_sizes.append(len(df))
                yield idx, source_name, df
        else:
            # File format reader mapping
            reader_map = {
                ".csv": pd.read_csv,
                ".parquet": pd.read_parquet,
                ".pq": pd.read_parquet,
                ".json": pd.read_json,
                ".jsonl": pd.read_json,
                ".pkl": pd.read_pickle,
                ".pickle": pd.read_pickle,
            }

            # Formats that support chunking
            chunkable_formats = {".csv", ".json", ".jsonl"}

            logger.info("Processing files to build metadata...")
            for idx, file_path in enumerate(self.file_paths):
                logger.info(f"Processing file {idx + 1}/{len(self.file_paths)}: {file_path}")

                # Determine file format
                _, extension = os.path.splitext(file_path)
                extension = extension.lower()
                reader_func = reader_map.get(extension)

                if not reader_func:
                    logger.warning(f"Unsupported file format '{extension}' for {file_path}. Skipping.")
                    continue

                # Store file size
                file_size = os.path.getsize(file_path)
                self.file_sizes.append(file_size)

                try:
                    # Handle chunked reading for supported formats
                    if self.memory_efficient and extension in chunkable_formats:
                        # Combine user options with chunksize
                        options = {**self.read_options, "chunksize": self.chunk_size}

                        # Special handling for JSON/JSONL
                        if extension in {".json", ".jsonl"}:
                            options.setdefault("lines", True)

                        for chunk in reader_func(file_path, **options):
                            chunk = self._parse_and_enrich_chunk(chunk)
                            yield idx, file_path, chunk
                    else:
                        # Non-chunked reading (Parquet, or non-memory-efficient mode)
                        options = self.read_options.copy()
                        options.pop("chunksize", None)  # Remove chunksize if present

                        chunk = reader_func(file_path, **options)
                        chunk = self._parse_and_enrich_chunk(chunk)
                        yield idx, file_path, chunk

                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    raise

    def _process_data(self):
        """
        Unified entry point for processing all data sources (files or DataFrames).
        Uses generator pattern for memory efficiency.
        """
        # Initialize data structures
        self.total_length = 0
        self.file_info = []
        self.group_info = {}
        self.lengths = {}
        self.file_group_map = []
        self.file_sizes = []

        # Track groups per source
        source_groups_map = {}

        # Process all data sources using the generator
        for source_idx, source_name, chunk in self._data_chunk_generator():
            if source_idx not in source_groups_map:
                source_groups_map[source_idx] = set()

            self._process_chunk(chunk, source_idx, source_name, source_groups_map[source_idx])

        # Build file_group_map from source_groups_map
        for source_idx in sorted(source_groups_map.keys()):
            for file_group_key in source_groups_map[source_idx]:
                self.file_group_map.append(file_group_key)

        # Store unique file-group combinations
        self._group_ids = list(self.group_info.keys())

        n_sources = len(self.file_paths if not self.use_dataframes else self.dataframes)
        logger.info(f"Processed {len(self._group_ids)} groups from {n_sources} sources")

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
        A wrapper around utility function to update label encoders with new categorical data.
        """
        update_label_encoders(data, self._cat_cols, self.label_encoders, self._group_cols, self._enrich_cat)

    def _apply_label_encoding(self, data):
        """
        A wrapper around utility function to apply label encoding to categorical columns in the data.
        """
        return apply_label_encoding(data, self._cat_cols, self.label_encoders)

    def _get_categorical_cardinality(self, col):
        """
        A wrapper around utility function to get the cardinality for a categorical column.
        """
        return get_categorical_cardinality(col, self.label_encoders, self.data)

    def _prepare_metadata(self):
        """Prepare dataset metadata including dimensions, column info, and statistics."""

        # Create a cumulative index mapping for efficient lookup
        self.cumulative_lengths = [0]
        for file_group_key in self._group_ids:
            group_length = self.group_info[file_group_key]["length"]
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + group_length)

        self.dataset_length = self.cumulative_lengths[-1]

        # Calculate feature indices
        cat_indices = [self.feature_cols.index(col) for col in (self._cat_cols or []) if col in self.feature_cols]
        past_indices = [self.feature_cols.index(col) for col in (self._past_cols or []) if col in self.feature_cols]
        future_indices = [self.feature_cols.index(col) for col in (self._future_cols or []) if col in self.feature_cols]
        target_indices = [self.feature_cols.index(col) for col in (self._target_cols or []) if col in self.feature_cols]

        # Prepare comprehensive metadata dictionary
        self.metadata = {
            "n_targets": len(self._target_cols),
            "n_features": len(self.feature_cols),
            "n_categorical": len(self._cat_cols),
            "n_past": len(self.past_cols) if self.past_cols else 0,
            "n_future": len(self.future_cols) if self.future_cols else 0,
            # Column names
            "target_cols": self._target_cols,
            "feature_cols": self.feature_cols,
            # Feature indices
            "idx_categorical": cat_indices,
            "idx_past": past_indices,
            "idx_future": future_indices,
            "idx_targets": target_indices,
            # Group information
            "n_groups": len(self._group_ids),
            # Column types and temporal information
            "time_col": self._time_col,
            "past_cols": self.past_cols if self.past_cols else [],
            "future_cols": self.future_cols if self.future_cols else [],
            "enrich_cat": self._enrich_cat if self._enrich_cat else [],
        }

        # Add categorical information to metadata only if categorical columns exist
        if self._cat_cols and len(self._cat_cols) > 0:
            self.metadata["categorical_columns"] = self._cat_cols

            # Build ordered lists of cat columns and their cardinalities
            cat_cols_list = []
            cat_cardinalities = []

            # Process all categorical columns in order
            for col in self._cat_cols:
                # Handle group columns
                if col in self.group_cols and hasattr(self, "group_info") and self.group_info:
                    if col in self.label_encoders:
                        n_categories = len(self.label_encoders[col].classes_)
                    else:
                        group_values = set()
                        for group_key, info in self.group_info.items():
                            if "original_values" in info and info["group_columns"] == self.group_cols:
                                if len(info["original_values"]) == 1:
                                    group_values.add(str(info["original_values"][0]))
                        n_categories = len(sorted(list(group_values)))

                    cat_cols_list.append(col)
                    cat_cardinalities.append(n_categories)

                # Handle regular categorical columns (non-group)
                elif col in self.label_encoders:
                    n_categories = len(self.label_encoders[col].classes_)
                    cat_cols_list.append(col)
                    cat_cardinalities.append(n_categories)

            # Store as ordered lists (matching __getitem__ format)
            self.metadata["cat_cols_list"] = cat_cols_list
            self.metadata["cat_cardinalities"] = cat_cardinalities

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

        if hasattr(self, "_enrich_cat") and self._enrich_cat:
            # Update cat_cols_list and cat_cardinalities to include temporal features
            cat_cols_list = list(self.metadata.get("cat_cols_list", []))
            cat_cardinalities = list(self.metadata.get("cat_cardinalities", []))

            for feature in self._enrich_cat:
                if feature in self.label_encoders and feature not in cat_cols_list:
                    categories = self.label_encoders[feature].classes_
                    n_categories = len(categories)
                    cat_cols_list.append(feature)
                    cat_cardinalities.append(n_categories)

            self.metadata["cat_cols_list"] = cat_cols_list
            self.metadata["cat_cardinalities"] = cat_cardinalities

            past_indices = [self.feature_cols.index(col) for col in self._past_cols if col in self.feature_cols]
            self.metadata["idx_past"] = past_indices
            self.metadata["n_past"] = len(self._past_cols) if self._past_cols else 0

            self.metadata["past_cols"] = self._past_cols.copy() if self._past_cols else []

            future_indices = [self.feature_cols.index(col) for col in self._future_cols if col in self.feature_cols]
            self.metadata["idx_future"] = future_indices
            self.metadata["n_future"] = len(self._future_cols) if self._future_cols else 0

            self.metadata["future_cols"] = self._future_cols.copy() if self._future_cols else []
            self.metadata["original_future_cols"] = self._original_future_cols.copy() if self._original_future_cols else None

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
        df_idx = self.group_info[file_group_key]["file_idx"]  # Use file_idx, not df_idx
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

        # Determine file format and reader
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        reader_map = {
            ".csv": pd.read_csv,
            ".parquet": pd.read_parquet,
            ".pq": pd.read_parquet,
            ".json": pd.read_json,
            ".jsonl": pd.read_json,
            ".pkl": pd.read_pickle,
            ".pickle": pd.read_pickle,
        }

        reader_func = reader_map.get(extension, pd.read_csv)

        # Load file with user-provided options
        options = self.read_options.copy()
        options.pop("chunksize", None)  # Remove chunksize for full read

        df = reader_func(file_path, **options)

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

        # Prepare the group sample
        features = x
        targets = y
        group_id = group_id
        time_indices = time_indices
        seq_len = len(group_data)

        # Build cat_cols and cardinalities dynamically to include temporal enrichment
        cat_cols_list = []
        cat_cardinalities_list = []

        # Get cardinalities from metadata (includes both regular and temporal features)
        all_cardinalities = self.metadata.get("categorical_cardinalities", {})

        # Iterate through current _cat_cols (includes temporal enrichment)
        for col in self._cat_cols:
            if col in all_cardinalities:
                cat_cols_list.append(col)
                cat_cardinalities_list.append(all_cardinalities[col])
            elif col in self.label_encoders:
                # Fallback: get from label encoder if not in metadata
                cat_cols_list.append(col)
                cat_cardinalities_list.append(len(self.label_encoders[col].classes_))

        return {
            "x": features,
            "y": targets,
            "group_id": group_id,
            "past_time": time_indices,  # All time indices for this group
            "future_time": time_indices,  # Same as past_time for now
            "seq_len": seq_len,  # Length of the sequence
            "cat_cols": cat_cols_list,
            "cat_cardinalities": cat_cardinalities_list,
        }
