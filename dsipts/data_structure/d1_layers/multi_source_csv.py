"""
Multi-source CSV implementation for D1 layer.

This module provides the MultiSourceTSDataSet class that handles raw data from multiple CSV files
with enhanced features including temporal categorical enrichment and improved logging.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

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
            file_paths: List of paths to CSV files containing time series data
            (optional if dataframes provided)
            dataframes: List of pandas DataFrames containing time series data
            (optional if file_paths provided)
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
            global_forecasting: If True, use global forecasting. If False (default)
                and multiple groups exist, add group as categorical known variable.
            weights: Name of weights column
            memory_efficient: Whether to use memory-efficient mode
            chunk_size: Chunk size for processing data (used in memory-efficient
                mode)
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
        self.time_col = time_col
        self.weights = weights

        # Create pseudo file paths for dataframes for consistent processing
        if self.use_dataframes:
            self.file_paths = [f"dataframe_{i}" for i in range(len(self.dataframes))]
            logger.info(f"Using {len(self.dataframes)} pandas DataFrames as input")
        else:
            logger.info(f"Using {len(self.file_paths)} file paths as input")

        # Handle group columns (can be single column or multiple)
        # Handle None, empty list, or NaN values
        if group_cols is None or (isinstance(group_cols, list) and len(group_cols) == 0):
            self._group_cols = []
            logger.info("No group columns provided, using default grouping")
            if global_forecasting:
                logger.info("Since there are no groups, cant proceed with global forecasting.")
                raise ValueError("Global forecasting requires group columns")
        elif isinstance(group_cols, str):
            self._group_cols = [group_cols]
            logger.info(f"Group column utilised during init: {group_cols}")
        else:
            self._group_cols = group_cols
            logger.info(f"Group columns utilised during init: {group_cols}")

        # Initialize attributes with proper defaults
        self._target_cols = target_cols or []
        self._time_col = time_col
        self._cat_cols = cat_cols or []
        self._num_cols = num_cols or []
        self._enrich_cat = enrich_cat
        self.enrich_cat = enrich_cat or []
        self.global_forecasting = global_forecasting

        # Handle group columns properly (already set self._group_cols above)

        # Set known and unknown columns with proper handling for None values
        self._known_cols = known_cols or []
        self._unknown_cols = unknown_cols or list(self._target_cols) if self._target_cols else []

        # Handle global forecasting logic: if global_forecasting=False and multiple groups exist,
        # add group columns to categorical and known variables
        self._apply_global_forecasting_logic()

        # Infer feature_cols automatically from headers and other specifications
        self._feature_cols = self._infer_feature_columns()
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
        Infer feature columns from known_cols, num_cols, cat_cols, or headers.
        Priority: known_cols > (num_cols + cat_cols) > headers
        """
        logger = logging.getLogger(__name__)
        logger.info("Inferring feature columns now!")

        # Priority 1: Use known_cols, num_cols and cat_cols and enrich_cat if specified
        # Approach 1: excluding targets from features (needs separate logic for index cal then)
        """
        if self._known_cols or self._num_cols or self._cat_cols:
            feature_cols = list(dict.fromkeys(self._known_cols + self._num_cols + self._cat_cols))
            logger.info(f"Feature cols: known_cols + num_cols + cat_cols: {feature_cols}")
            # Add temporal features if specified
            if self.enrich_cat:
                feature_cols = list(dict.fromkeys(feature_cols + self.enrich_cat))
                logger.info(f"Updated feature columns after adding enrich_cat: {feature_cols}")
            return feature_cols
        """

        # Priority 1: (known_cols, num_cols & cat_cols & target_cols) and (enrich_cat) if specified
        # Approach 2: including targets (automatically handles the index cal logic with crrent code)
        if self._known_cols or self._num_cols or self._cat_cols or self.target_cols:
            feature_cols = list(
                dict.fromkeys(self._known_cols + self._num_cols + self._cat_cols + self.target_cols)
            )
            logger.info(
                f"Feature cols: known_cols + num_cols + cat_cols + target_cols: {feature_cols}"
            )
            # Add temporal features if specified
            if self.enrich_cat:
                feature_cols = list(dict.fromkeys(feature_cols + self.enrich_cat))
                logger.info(f"Updated feature columns after adding enrich_cat: {feature_cols}")
            return feature_cols

        """
        # Priority 2: Use num_cols + cat_cols if specified
        if self._num_cols or self._cat_cols:
            feature_cols = list(set(self._num_cols + self._cat_cols))
            # Include target columns in features (they can be both features and targets)
            feature_cols = list(dict.fromkeys(feature_cols + self._target_cols))
            logger.info(
                f"num_cols + cat_cols + target_cols as feature columns: {feature_cols}"
            )
            # Add temporal features if specified
            if self.enrich_cat:
                feature_cols = list(dict.fromkeys(feature_cols + self.enrich_cat))
                logger.info(
                    f"Updated feature columns after adding enrich_cat: {feature_cols}"
                )
            return feature_cols
        """

        # Priority 3: Infer from headers (files or dataframes)
        # #TODO: needs improvements, how to reach columns when dataframes are provided.
        logger.info("Inferring feature columns from headers...")

        if self.use_dataframes:
            if not self.dataframes:
                raise ValueError("Cannot infer feature columns: no dataframes provided")
            # Get columns from first dataframe
            all_columns = self.dataframes[0].columns.tolist()
        else:
            if not self.file_paths:
                raise ValueError("Cannot infer feature columns: no file paths provided")
            # Read first file to get column names
            first_file = self.file_paths[0]
            try:
                sample_df = pd.read_csv(first_file, nrows=1)
                all_columns = sample_df.columns.tolist()
            except Exception as e:
                raise ValueError(f"Could not read file {first_file} to infer columns: {e}")

        # Exclude special columns from features
        special_columns = set()
        if self._time_col:
            special_columns.add(self._time_col)
        if self.weights:
            special_columns.add(self.weights)

        # Include all columns except special ones
        feature_cols = [col for col in all_columns if col not in special_columns]
        # Add temporal features if specified
        if self.enrich_cat:
            feature_cols = list(dict.fromkeys(feature_cols + self.enrich_cat))

        logger.info(f"Inferred feature columns from headers: {feature_cols}")
        logger.info(f"Excluded special columns: {list(special_columns)}")

        return feature_cols

    def _apply_global_forecasting_logic(self):
        """Apply global forecasting logic based on groups and global_forecasting flag.

        Main flow logic:
        - If groups are not given: local forecasting (no special handling needed)
        - If groups are given and global_forecasting=True: global forecasting (no special handling)
        - If groups are given and global_forecasting=False: local forecasting
          -> Treat groups as categorical known values
          -> Add to categorical and known columns lists
          -> Apply label encoding later in processing
        """
        logger = logging.getLogger(__name__)

        # Case 1: No groups provided - local forecasting by default
        if not self._group_cols:
            logger.info("No group columns provided - using local forecasting")
            return

        # Case 2: Groups provided and global forecasting enabled
        if self.global_forecasting:
            logger.info(f"Global forecasting enabled with group columns {self._group_cols}")
            logger.info("Groups will be used for global model training")
            return

        # Case 3: Groups provided but global forecasting disabled - local forecasting
        logger.info(f"Local forecasting with group columns {self._group_cols}")
        logger.info("Groups will be treated as categorical known values for local forecasting")

        # Add group columns to categorical columns (groups need label encoding)
        for group_col in self._group_cols:
            if group_col not in self._cat_cols:
                self._cat_cols.append(group_col)
                logger.info(
                    f"Added group column '{group_col}' to categorical columns for label encoding"
                )

        # Add group columns to known columns (group identity is known at prediction time)
        for group_col in self._group_cols:
            if group_col not in self._known_cols:
                self._known_cols.append(group_col)
                logger.info(f"Added group column '{group_col}' to known columns")

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
            # Update encoders with enriched features
            self._update_encoders(chunk)

        # Set of mandatory columns (cat, num, target, group, time)
        mandatory_cols = set(self._cat_cols + self._num_cols + self._target_cols)

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

        When enrichment is done, new features are added as categorical known values.
        These need to be updated in categorical and known columns lists and
        label encoding should be applied to these new features.

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
        enriched_features = []
        for column in self.enrich_cat:
            if column == "hour":
                dataset[column] = dataset[self.time_col].dt.hour
                enriched_features.append(column)
            elif column == "dow":
                dataset[column] = dataset[self.time_col].dt.dayofweek
                enriched_features.append(column)
            elif column == "month":
                dataset[column] = dataset[self.time_col].dt.month
                enriched_features.append(column)
            elif column == "minute":
                dataset[column] = dataset[self.time_col].dt.minute
                enriched_features.append(column)
            else:
                if column not in dataset.columns:
                    logger.error(
                        f"I can not automatically enrich column {column}. Please contact the developers or add it manually to your dataset."  # noqa: E501
                    )

        # Add temporal categorical features to cat_cols and known_cols only once
        if enriched_features and not self._is_file_read:
            if self._cat_cols is None:
                self._cat_cols = []

            # Add enriched features to categorical columns (they need label encoding)
            for feature in enriched_features:
                if feature not in self._cat_cols:
                    self._cat_cols.append(feature)

            # Also add to known_cols since temporal features are always known at prediction time
            if self._known_cols is not None:
                for feature in enriched_features:
                    if feature not in self._known_cols:
                        self._known_cols.append(feature)

            self._is_file_read = True
            logger.info(
                f"Added temporal categorical features to categorical columns: {enriched_features}"
            )
            logger.info(
                f"Added temporal categorical features to known columns: {enriched_features}"
            )
            logger.info(
                "Label encoding will be applied to these enriched features during processing"
            )

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

    def _process_dataframes(self):
        """
        Process pandas DataFrames to extract group information and update encoders.

        Similar to _process_files but works with in-memory DataFrames.
        """
        logger.info("Processing DataFrames to build metadata...")

        # Initialize data structures
        self.total_length = 0
        self.file_info = []
        self.group_info = {}
        self.lengths = {}
        self.file_group_map = []
        self.file_sizes = []

        # Process each DataFrame
        for df_idx, df in enumerate(self.dataframes):
            logger.info(f"Processing DataFrame {df_idx + 1}/{len(self.dataframes)}")

            # Store DataFrame size for memory management
            self.file_sizes.append(len(df))

            # Track groups in this DataFrame
            file_groups = set()

            # Process the entire DataFrame (no chunking needed since it's in memory)
            self._process_dataframe_chunk(df, df_idx, f"dataframe_{df_idx}", file_groups)

            # Store file groups for this DataFrame
            self.file_group_map.append(file_groups)

        # Store unique file-group combinations for iteration
        self._group_ids = list(self.group_info.keys())
        logger.info(f"Found {len(self._group_ids)} unique DataFrame-group combinations")

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

    def _process_dataframe_chunk(self, chunk, df_idx, df_name, file_groups):
        """
        Process a DataFrame chunk (similar to _process_chunk but for DataFrames).

        Args:
            chunk: DataFrame to process
            df_idx: Index of the DataFrame being processed
            df_name: Name/identifier for the DataFrame
            file_groups: Set to track groups in this DataFrame
        """
        if len(chunk) == 0:
            return

        # Handle grouping logic based on group_cols (same as _process_chunk)
        if not self.group_cols:
            # No group columns - treat all data as a single group
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

            # Update group length
            self.group_info[file_group_key]["length"] += len(group_data)

            # Update label encoders for categorical columns
            self._update_encoders(group_data)

            # Add to file group map
            for _ in range(len(group_data)):
                self.file_group_map.append(file_group_key)

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
        This method applies label encoding to:
        1. Group columns (when treated as categorical in local forecasting)
        2. Original categorical columns
        3. Enriched temporal categorical features

        The encoders are stored and can be used later for decoding.

        Args:
            data: DataFrame containing the data to update encoders with
        """
        for col in self.cat_cols:
            if col in data.columns:
                # Get non-null values and convert to string for consistent encoding
                values = data[col].dropna().astype(str)
                if len(values) > 0:
                    if col not in self.label_encoders:
                        # Create new encoder for this column
                        self.label_encoders[col] = LabelEncoder()
                        # Set up handling for unknown values
                        if hasattr(self.label_encoders[col], "handle_unknown"):
                            self.label_encoders[col].handle_unknown = "use_encoded_value"
                            self.label_encoders[col].unknown_value = -1

                        # Fit with initial values
                        unique_values = values.unique()
                        self.label_encoders[col].fit(unique_values)

                        # Log encoder creation
                        if col in self._group_cols:
                            logger.info(
                                f"Created label encoder for group column '{col}'"
                                f"with {len(unique_values)} categories"
                            )
                        elif col in (self.enrich_cat or []):
                            logger.info(
                                f"Created label encoder for enriched feature '{col}'"
                                f"with {len(unique_values)} categories"
                            )
                            logger.info(f"the unique values are: {unique_values}")
                        else:
                            logger.info(
                                f"Created label encoder for categorical column '{col}'"
                                f"with {len(unique_values)} categories"
                            )
                    else:
                        # Update existing encoder with new values
                        existing_categories = set(self.label_encoders[col].classes_)
                        new_values = set(values.unique()) - existing_categories
                        if new_values:
                            # Refit with all values (existing + new)
                            all_values = list(existing_categories) + list(new_values)
                            self.label_encoders[col].fit(np.array(all_values))
                            logger.debug(
                                f"Updated label encoder for column '{col}'"
                                f"with {len(new_values)} new categories"
                            )
                # logger.info(f"Updated LE for {col} here is: {self.label_encoders[col].classes_}")

    def _apply_label_encoding(self, data):
        """Apply label encoding to categorical columns in the data.

        This method transforms categorical values to their encoded integer representations
        using the fitted label encoders.

        Args:
            data: DataFrame to apply encoding to

        Returns:
            DataFrame with categorical columns encoded
        """
        data_encoded = data.copy()

        for col in self.cat_cols:
            if col in data_encoded.columns and col in self.label_encoders:
                # Get non-null values
                non_null_mask = data_encoded[col].notna()
                if non_null_mask.any():
                    # Convert to string for consistent encoding
                    values_to_encode = data_encoded.loc[non_null_mask, col].astype(str)

                    try:
                        # Apply label encoding
                        encoded_values = self.label_encoders[col].transform(values_to_encode)
                        data_encoded.loc[non_null_mask, col] = encoded_values

                        # Log encoding application
                        if col in self._group_cols:
                            logger.debug(f"Applied label encoding to group column '{col}'")
                        elif col in (self.enrich_cat or []):
                            logger.debug(f"Applied label encoding to enriched feature '{col}'")
                        else:
                            logger.debug(f"Applied label encoding to categorical column '{col}'")

                    except ValueError as e:
                        logger.warning(f"Could not encode column '{col}': {e}")
                        # Handle unknown categories by assigning -1 or keeping original values
                        if hasattr(self.label_encoders[col], "unknown_value"):
                            data_encoded.loc[non_null_mask, col] = self.label_encoders[
                                col
                            ].unknown_value

        return data_encoded

    def _get_categorical_cardinality(self, col):
        """
        Get the cardinality (number of unique values) for a categorical column.

        Args:
            col: Column name

        Returns:
            Number of unique categories for the column
        """
        if col in self.label_encoders:
            return len(self.label_encoders[col].classes_)
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
        logger = logging.getLogger(__name__)

        # Log dataset configuration
        logger.info("\n" + "=" * 80)
        logger.info("D1 LAYER - DATASET CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Time Column: {self.time_col}")
        logger.info(f"Target Columns: {self.target_cols}")
        logger.info(f"Feature Columns: {self.feature_cols}")
        logger.info(f"Categorical Columns: {self.cat_cols}")
        logger.info(f"Known Future Columns: {self.known_cols}")
        logger.info(f"Unknown Future Columns: {self.unknown_cols}")
        logger.info(f"Group Columns: {self.group_cols}")
        logger.info(f"Memory Efficient Mode: {self.memory_efficient}")
        logger.info("-" * 80 + "\n")

        # Log group information
        if self.group_cols:
            logger.info("GROUP INFORMATION:")
            for group_id, info in self.group_info.items():
                logger.info(f"  - Group {group_id}: {info['length']} samples")
        else:
            logger.info("No group columns - treating as a single global group")

        # Create a cumulative index mapping for efficient lookup
        logger.info("\nBuilding cumulative index mapping...")
        self.cumulative_lengths = [0]
        for file_group_key in self._group_ids:
            group_length = self.group_info[file_group_key]["length"]
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + group_length)

        logger.info(f"Total samples across all groups: {self.cumulative_lengths[-1]}")

        # Store total dataset length
        self.dataset_length = self.cumulative_lengths[-1]

        # Log dataset statistics
        logger.info("\nDATASET STATISTICS:")
        logger.info("-" * 40)
        logger.info(f"Total Samples: {self.dataset_length:,}")
        logger.info(f"Number of Groups: {len(self._group_ids)}")
        logger.info(f"Number of Files: {len(self.file_paths)}")
        logger.info(f"Number of Features: {len(self.feature_cols)}")
        logger.info(f"Values of Features: {self.feature_cols}")
        logger.info(f"Number of Targets: {len(self.target_cols)}")
        logger.info(f"Values of Targets: {self.target_cols}")
        logger.info(f"Number of Categorical Columns: {len(self.cat_cols)}")
        logger.info(f"Values of Categorical Columns: {self.cat_cols}")

        # Log memory usage if data is cached
        if hasattr(self, "cached_data") and self.cached_data:
            total_mb = sum(df.memory_usage(deep=True).sum() for df in self.cached_data.values()) / (
                1024**2
            )
            logger.info(f"Cached Data Memory Usage: {total_mb:.2f} MB")

        logger.info("-" * 80 + "\n")

        # Calculate feature indices for metadata
        logger.info("CALCULATING FEATURE INDICES...")

        # Get indices of different feature types within the feature_cols list
        # NOTE: Preserve the order of the source lists (cat/known/unknown/target)
        # by mapping each column to its index in feature_cols, filtering to existing columns.
        cat_indices = [
            self.feature_cols.index(col)
            for col in (self.cat_cols or [])
            if col in self.feature_cols
        ]
        known_indices = [
            self.feature_cols.index(col)
            for col in (self.known_cols or [])
            if col in self.feature_cols
        ]
        unknown_indices = [
            self.feature_cols.index(col)
            for col in (self.unknown_cols or [])
            if col in self.feature_cols
        ]
        target_indices = [
            self.feature_cols.index(col)
            for col in (self.target_cols or [])
            if col in self.feature_cols
        ]

        logger.info(f"Categorical feature indices: {cat_indices}")
        logger.info(f"Known future feature indices: {known_indices}")
        logger.info(f"Unknown future feature indices: {unknown_indices}")
        logger.info(f"Target feature indices: {target_indices}")

        # Calculate groups per file
        groups_per_file = []
        for file_idx in range(len(self.file_paths)):
            file_groups = [key for key in self._group_ids if key[0] == file_idx]
            groups_per_file.append([key[1] for key in file_groups])  # Extract group IDs

        logger.info(f"Groups per file: {groups_per_file}")

        # Prepare comprehensive metadata dictionary
        logger.info("PREPARING METADATA...")
        self.metadata = {
            # Dataset dimensions (counts)
            "n_targets": len(self.target_cols),
            "n_features": len(self.feature_cols),
            "n_categorical": len(self.cat_cols),
            "n_known_future": len(self.known_cols) if self.known_cols else 0,
            "n_unknown_future": len(self.unknown_cols) if self.unknown_cols else 0,
            # Column names
            "target_cols": self.target_cols,
            "feature_cols": self.feature_cols,
            # Feature indices
            "idx_categorical": cat_indices,
            "idx_known_future": known_indices,
            "idx_unknown_future": unknown_indices,
            "idx_targets": target_indices,
            # Group information
            "n_groups": len(self._group_ids),
            # Column types and temporal information
            "time_col": self.time_col,
            "known_cols": self.known_cols if self.known_cols else [],
            "unknown_cols": self.unknown_cols if self.unknown_cols else [],
            "enrich_cat": self.enrich_cat if self.enrich_cat else [],
        }

        # Add categorical information to metadata only if categorical columns exist
        if self.cat_cols and len(self.cat_cols) > 0:
            logger.info("Processing categorical columns...")
            self.metadata["categorical_columns"] = self.cat_cols

            # Enhanced categorical cardinality information
            cardinalities = {}
            categorical_mappings = {}

            for col in self.cat_cols:
                # Special handling for group columns - use known group metadata
                if col in self.group_cols and hasattr(self, "group_info") and self.group_info:
                    # Extract all unique group values from group_info
                    group_values = set()
                    for group_key, info in self.group_info.items():
                        if "original_values" in info and info["group_columns"] == self.group_cols:
                            # For single group column, extract the value
                            if len(info["original_values"]) == 1:
                                group_values.add(str(info["original_values"][0]))

                    # Ensure we have all group values
                    group_values = sorted(list(group_values))
                    n_categories = len(group_values)
                    cardinalities[col] = n_categories

                    # Update or create label encoder with all group values
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        if hasattr(self.label_encoders[col], "handle_unknown"):
                            self.label_encoders[col].handle_unknown = "use_encoded_value"
                            self.label_encoders[col].unknown_value = -1

                    # Fit encoder with all group values
                    self.label_encoders[col].fit(group_values)

                    # Store the actual category mappings for reference
                    categorical_mappings[col] = {
                        "categories": group_values,
                        "cardinality": n_categories,
                        "feature_index": self.feature_cols.index(col)
                        if col in self.feature_cols
                        else -1,
                    }

                    logger.info(
                        f"  - {col}: {n_categories} categories"
                        f"{group_values[:5]}{'...' if n_categories > 5 else ''}"
                    )
                elif col in self.label_encoders:
                    categories = self.label_encoders[col].classes_
                    n_categories = len(categories)
                    cardinalities[col] = n_categories

                    # Store the actual category mappings for reference
                    categorical_mappings[col] = {
                        "categories": categories.tolist(),
                        "cardinality": n_categories,
                        "feature_index": self.feature_cols.index(col)
                        if col in self.feature_cols
                        else -1,
                    }

                    logger.info(
                        f"  - {col}: {n_categories} categories {categories[:5].tolist()}{'...' if n_categories > 5 else ''}"  # noqa: E501
                    )

            self.metadata["categorical_cardinalities"] = []
            self.metadata["categorical_mappings"] = categorical_mappings
            # Populate simple lists for easier access
            for col_name, mapping in categorical_mappings.items():
                self.metadata["categorical_cardinalities"].append(mapping["cardinality"])

        # Add group information to metadata
        self.metadata["group_cols"] = self.group_cols

        # For empty group_cols, add special metadata indicating global grouping
        if not self.group_cols:
            self.metadata["single_group"] = True
            self.metadata["n_groups"] = 1
            logger.info("No group columns specified - treating as a single global group")
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
        self.metadata["file_paths"] = self.file_paths  # NEW: store file paths for reference

        # Log final metadata summary
        logger.info("\nFINAL METADATA SUMMARY")
        logger.info("=" * 80)
        for key, value in self.metadata.items():
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                logger.info(f"{key}: {type(value)} (length: {len(value)})")
            else:
                logger.info(f"{key}: {value}")

        logger.info("=" * 80)
        logger.info("D1 LAYER INITIALIZATION COMPLETE\n")

    def _preload_data(self):
        """
        Preload all data into memory for faster access.
        Only called when memory_efficient=False.
        """
        logger.info("Preloading data into memory...")
        self.cached_data = {}

        for file_group_key in self._group_ids:
            if self.use_dataframes:
                df_name, group_key = file_group_key
                group_data = self._load_group_data_from_dataframe(file_group_key)
            else:
                file_path, group_key = file_group_key
                group_data = self._load_group_data(file_group_key)
            self.cached_data[file_group_key] = group_data

        logger.info(f"Preloaded {len(self.cached_data)} groups into memory")

    def _load_group_data_from_dataframe(self, file_group_key):
        """
        Load data for a specific group from a DataFrame.

        Args:
            file_group_key: Tuple of (df_name, group_key)

        Returns:
            DataFrame containing the group's data
        """
        df_name, group_key = file_group_key
        df_idx = self.group_info[file_group_key]["df_idx"]
        df = self.dataframes[df_idx]

        # Handle grouping logic (same as file-based approach)
        if not self.group_cols:
            # No group columns - return entire DataFrame
            group_data = df.copy()
        elif isinstance(self.group_cols, list) and len(self.group_cols) > 1:
            # Multi-column grouping
            mask = df[self.group_cols].apply(lambda x: tuple(x), axis=1) == group_key
            group_data = df[mask].copy()
        elif isinstance(self.group_cols, list) and len(self.group_cols) == 1:
            # Single column in list
            # Extract the actual group value from the tuple
            actual_group_value = group_key[0] if isinstance(group_key, tuple) else group_key
            group_data = df[df[self.group_cols[0]] == actual_group_value].copy()
        else:
            # Single column as string
            group_data = df[df[self.group_cols] == group_key].copy()

        # Apply enrichment and processing
        return self._parse_and_enrich_chunk(group_data)

    def _load_group_data(self, file_group_key):
        """
        Load data for a specific group from file.

        Args:
            file_group_key: Tuple of (file_idx, group_key)

        Returns:
            DataFrame containing the group data
        """
        file_idx, group_key = file_group_key
        file_path = self.file_paths[file_idx]

        # Load and preprocess entire file (parse time, enrich temporal, update encoders)
        df = pd.read_csv(file_path)
        """ TODO: Remove this commented code if logic works
        # Handle grouping logic (same as DataFrame approach)
        if not self.group_cols:
            # No group columns - return entire file
            group_data = df.copy()
        elif isinstance(self.group_cols, list) and len(self.group_cols) > 1:
            # Multi-column grouping
            mask = df[self.group_cols].apply(lambda x: tuple(x), axis=1) == group_key
            group_data = df[mask].copy()
        elif isinstance(self.group_cols, list) and len(self.group_cols) == 1:
            # Single column in list
            # Extract the actual group value from the tuple
            actual_group_value = group_key[0] if isinstance(group_key, tuple) else group_key
            group_data = df[df[self.group_cols[0]] == actual_group_value].copy()
        else:
            # Single column as string
            group_data = df[df[self.group_cols] == group_key].copy()

        # Apply enrichment and processing
        return self._parse_and_enrich_chunk(group_data)

        df = self._enrich_temporal_features(df)
        """
        # Extract and process group data
        df = self._parse_and_enrich_chunk(df)

        # Extract group data without applying encoding
        group_data = self._extract_group_data(df, group_key)
        return group_data

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

        # Normalize group_key: unwrap nested tuple like ((a, b),) -> (a, b)
        if isinstance(group_key, tuple) and len(group_key) == 1 and isinstance(group_key[0], tuple):
            group_key = group_key[0]

        # Create a mask for the group
        if isinstance(self.group_cols, list):
            if len(self.group_cols) > 1:
                # Ensure group_key is a tuple of the same length as group_cols
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                if len(group_key) != len(self.group_cols):
                    logger.debug(
                        f"Group key length mismatch: expected {len(self.group_cols)},"
                        f" got {len(group_key)}; using tuple equality filter"
                    )
                mask = df[self.group_cols].apply(lambda x: tuple(x), axis=1) == tuple(group_key)
            elif len(self.group_cols) == 1:
                # Single group column in a list
                col = self.group_cols[0]
                key_val = group_key[0] if isinstance(group_key, tuple) else group_key
                mask = df[col] == key_val
                logger.debug(f"Filtering on single group column in list: {col}")
            else:
                # Empty list case
                logger.debug("Empty group_cols list - returning all data")
                return df
        else:
            # Single group column as string
            key_val = group_key[0] if isinstance(group_key, tuple) else group_key
            mask = df[self.group_cols] == key_val

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
            raise IndexError(
                f"Group index {idx} out of range. Available groups: {len(self._group_ids)}"
            )

        # Get the file-group key for this group index
        file_group_key = self._group_ids[idx]

        # Get all data for this group
        if not self.memory_efficient and file_group_key in self.cached_data:
            # Use cached data
            group_data = self.cached_data[file_group_key]
        else:
            # Load from file or DataFrame
            if self.use_dataframes:
                group_data = self._load_group_data_from_dataframe(file_group_key)
            else:
                group_data = self._load_group_data(file_group_key)

        # Sort by time if time column exists
        if self.time_col in group_data.columns:
            group_data = group_data.sort_values(by=self.time_col)

        # Get group ID - always use integer encoding if mapping exists
        group_key = file_group_key[1]
        if "group_mapping" in self.metadata:
            # For single-col groups, group_key is str, but mapping expects tuple
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_id = self.metadata["group_mapping"].get(group_key, group_key)
        else:
            group_id = group_key

        # Extract all features and targets for this group efficiently
        if len(group_data) == 0:
            logger.warning(f"Empty group data found for group {group_id}.")
            x = torch.empty(0, len(self.feature_cols), dtype=torch.float32)
            y = torch.empty(0, len(self.target_cols), dtype=torch.float32)
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
                # Combine numerical and categorical features
                x = torch.cat([x_num, x_cat.float()], dim=-1)
            else:
                x = x_num

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
        df = self._parse_and_enrich_chunk(df)

        # Extract group data without applying encoding
        group_data = self._extract_group_data(df, group_key)
        return group_data
