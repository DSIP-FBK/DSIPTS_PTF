# Training Workflow

This section covers model training using PyTorch Lightning with DSIPTS-PTF datasets.

## Lightning Integration

`EncoderDecoder` implements the Lightning `DataModule` pattern:

1. Instantiate D1 and D2 objects.
2. Call `d2.setup(stage="fit")` manually or let Lightning call it during `trainer.fit()`.
3. Use the dataloader accessors to obtain loaders per stage.

```python
import pytorch_lightning as pl
from dsipts.data_structure.d1_layers import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers import EncoderDecoder
from dsipts.models.linear.auto_regression import LinearTS

# D1 setup
weather_csv = "tests/integration/synthetic_weather.csv"
d1 = MultiSourceTSDataSet(
    file_paths=[weather_csv],
    time_col="time",
    target_cols=["target"],
    past_cols=["target", "temperature", "humidity"],
    future_cols=["temperature", "humidity"],
    enrich_cat=["hour", "dow"],
)

# D2 setup
encoder_decoder = EncoderDecoder(
    d1_dataset=d1,
    past_len=96,
    future_len=24,
    scaling_method="standard",
    target_scaling_method=True,
)

model = LinearTS(
    past_steps=96,
    future_steps=24,
    out_channels=1,
    past_channels=len(encoder_decoder.metadata["num_past_cols"]),
    future_channels=len(encoder_decoder.metadata["num_future_cols"]),
)

trainer = pl.Trainer(max_epochs=5, accelerator="auto")
trainer.fit(model=model, datamodule=encoder_decoder)
```

Lightning will call `setup(stage="fit")`, fit the scaler once on the training windows, and reuse it for validation and test stages.

## Memory Modes

- **`memory_efficient=False`**: Pre-transforms all windows after fitting the scaler. Faster iteration at the cost of higher RAM usage.
- **`memory_efficient=True`**: Applies scaling on-the-fly inside `__getitem__()`. Use for very large datasets.

Switch between modes by passing the flag to the `EncoderDecoder` constructor.

## Hyperparameter Tuning

Leverage PyTorch Lightning's `Tuner` utilities to find batch sizes and learning rates without refitting scalers:

```python
from pytorch_lightning.tuner import Tuner

trainer = pl.Trainer(max_epochs=1, accelerator="auto")
tuner = Tuner(trainer)

# Fits scaler only once during the first setup call
tuner.scale_batch_size(model, datamodule=encoder_decoder)
tuner.lr_find(model, datamodule=encoder_decoder)
```

Internally, `EncoderDecoder` guards scaler fitting through `is_scaler_fitted`. Subsequent tuner calls reuse the trained scaler, preventing leakage.

## Logging and Callbacks

- Include `pytorch_lightning.callbacks.ModelCheckpoint` to persist best checkpoints.
- Use Aim or TensorBoard logging through Lightning's logger ecosystem.
- For deterministic runs, enable `trainer = pl.Trainer(deterministic=True)` and set random seeds within your script.

## Common Pitfalls

- Always let Lightning call `setup()`. Avoid fetching dataloaders before `trainer.fit()` unless you manually call `setup(stage="fit")` first.
- Ensure target columns are part of `past_cols` when using autoregressive models that expect past target values.
- For global forecasting, configure `group_lists` to avoid leaking evaluation groups into training.

Next, explore the [Hydra configuration workflow](hydra-configs.md) to manage complex experiment setups.
