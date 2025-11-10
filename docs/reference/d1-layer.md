# D1 Layer Reference

The D1 layer is responsible for ingesting raw data, enriching it, and producing metadata consumed by downstream components. The default implementation, `MultiSourceTSDataSet`, lives in `dsipts/data_structure/d1_layers/multi_source_csv.py`.

## Initialisation Parameters

- **`file_paths`** (`List[Union[str, Path]]`): Paths to CSV files. Each file is treated as a group when `group_cols` is not provided.
- **`dataframes`** (`Optional[List[pd.DataFrame]]`): Alternative to `file_paths` for in-memory data.
- **`time_col`** (`str`): Timestamp column used to order samples. Defaults to `"time"` when available.
- **`target_cols`** (`List[str]`): Columns to forecast.
- **`past_cols` / `future_cols`** (`Optional[List[str]]`): Numeric features available in past and future. When omitted, D1 infers them from the dataset.
- **`cat_cols`** (`Optional[List[str]]`): Explicit categorical columns present in the source data.
- **`group_cols`** (`Optional[List[str]]`): Columns defining independent entities for global forecasting.
- **`enrich_cat`** (`Optional[List[str]]`): Temporal enrichment options. Supported values: `"hour"`, `"dow"`, `"month"`, `"minute"`.
- **`global_forecasting`** (`bool`): When `True`, windows can draw from multiple entities and `group_id` becomes part of the batch.
- **`memory_efficient`** (`bool`): Controls caching behaviour. `False` keeps parsed data in memory; `True` streams from disk.

## Metadata Contract

D1 exposes a metadata dictionary accessed via `d1.metadata`. Key fields include:

- `time_col`, `target_cols`, `num_past_cols`, `num_future_cols`, `cat_past_list`, `cat_future_list`.
- `cat_past_cardinalities`, `cat_future_cardinalities`: Computed from actual data, including enrichment features.
- `group_ids`, `group_mapping`: Track entity identifiers used by D2 and models.
- `global_forecasting`: Propagated to D2 to determine whether to expose `group_id` in batches.
- `original_known_cols`: Captures user intent for `known_cols` so D2 can decide which categorical features are truly known ahead of time.

## Lifecycle Hooks

1. **`__init__`**
   - Parses inputs, validates column selections, and prepares metadata placeholders.
   - Calls `_prepare_metadata()` to populate column lists.
2. **`_preload_data()`** (only when `memory_efficient=False`)
   - Loads each group into memory and stores in `self.cached_data`.
3. **`_update_metadata_after_preload()`**
   - Computes categorical cardinalities and updates metadata with observed statistics.
4. **`__getitem__(group_idx)`**
   - Returns a dictionary containing `x`, `cat`, `time`, and `group_id` arrays for the requested group.

## Best Practices

- Keep `target_cols` included in `past_cols` for autoregressive models requiring historical target values.
- Use `enrich_cat` instead of manually computing temporal features to ensure cardinalities remain consistent.
- For very large datasets, set `memory_efficient=True` and rely on D2's streaming scaler to manage memory usage.
- Avoid mutating `d1.metadata` directly. Instead, pass configuration parameters during initialisation.

Refer to the [data pipeline guide](../user-guide/data-pipeline.md) for an end-to-end walkthrough of how D1 interacts with D2.
