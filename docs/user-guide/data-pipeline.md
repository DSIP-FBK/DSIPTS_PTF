# Data Pipeline

This page explains how DSIPTS-PTF converts raw time series data into model-ready batches using the layered D1 → D2 architecture.

## D1 Layer Responsibilities

D1 layers live in `dsipts/data_structure/d1_layers/`. The default implementation, `MultiSourceTSDataSet`, performs the following steps:

- **Source loading**: Reads CSV files or dataframes supplied through `file_paths` and groups rows using `group_cols`.
- **Temporal enrichment**: Adds derived categorical columns such as `hour`, `dow`, `month`, or `minute` as requested via `enrich_cat`.
- **Metadata curation**: Builds dictionaries describing numerical and categorical columns, target columns, window lengths, and group mappings. Metadata is mapped through `self.metadata` and consumed downstream by D2.
- **Cardinality tracking**: Computes categorical cardinalities, including enrichment features, inside `_compute_all_categorical_cardinalities()` to ensure embedding sizes remain accurate.
- **Caching**: When `memory_efficient=False`, data is cached in memory to speed up downstream processing. With `memory_efficient=True`, chunks are streamed on demand.

Key configuration arguments:

- `time_col`: Timestamp column used to order samples.
- `target_cols`: List of columns to predict.
- `past_cols` / `future_cols`: Control past/future numeric feature availability. Defaults fall back to the detected numerical columns.
- `cat_cols`: Explicit categorical variables present in the source data.
- `global_forecasting`: Toggle between global and local forecasting semantics. Influences how group identifiers are handled.

Inspect metadata after initialisation:

```python
from dsipts.data_structure.d1_layers import MultiSourceTSDataSet

d1 = MultiSourceTSDataSet(file_paths=["data/weather.csv"], enrich_cat=["hour", "dow"])
print(d1.metadata.keys())
```

## D2 Layer Responsibilities

D2 layers in `dsipts/data_structure/d2_layers/` convert D1 outputs into sliding windows. `EncoderDecoder` provides:

- **Window generation**: Builds valid windows according to `past_len`, `future_len`, and optional `skip_step`.
- **Splitting**: Supports temporal splits via `split_ratio` and group-aware splits through `group_lists`. Global forecasting scenarios can route selected groups entirely to validation or test sets.
- **Scaling**: Fits feature and target scalers only on training windows to avoid leakage. Choose `scaling_method` and `target_scaling_method` (`"standard"`, `"minmax"`, or `None`).
- **Memory policies**: When `memory_efficient=False`, `_pretransform_dataset_direct()` precomputes transformed tensors. When `True`, scaling happens lazily in `__getitem__()`.
- **Batch format**: `__getitem__()` returns `(x, y)` where `x` contains keys such as `x_num_past`, `x_cat_past`, `x_cat_future`, `idx_target`, and optionally `group_id` when `global_forecasting=True`.

Example setup:

```python
from dsipts.data_structure.d2_layers import EncoderDecoder

d2 = EncoderDecoder(
    d1_dataset=d1,
    past_len=96,
    future_len=24,
    split_ratio=(0.7, 0.15, 0.15),
    scaling_method="standard",
    target_scaling_method=True,
    memory_efficient=False,
)

d2.setup(stage="fit")
train_loader = d2.train_dataloader(batch_size=64)
```

## Metadata Flow

1. D1 populates metadata fields such as `cat_past_list`, `cat_future_list`, `future_cat_indices`, and scaling statistics placeholders.
2. `EncoderDecoder` reads metadata during construction and updates it after scaler fitting via `_update_metadata_after_preload()`.
3. Downstream models depend on consistent column ordering across metadata fields. Avoid mutating metadata directly; instead, adjust the D1 constructor arguments.

## Debugging Tips

- Use `d2.log_dataset_summary()` (if available) or inspect `d2.valid_windows` to confirm window counts.
- Enable logging by setting the `DSIPTS_LOG_LEVEL` environment variable or adjusting the logging configuration in `pyproject.toml`.
- For temporal enrichment issues, verify that `time_col` is parsed as a datetime and that `enrich_cat` contains supported options.

Continue to [Training Workflow](training.md) for guidance on fitting models with PyTorch Lightning.
