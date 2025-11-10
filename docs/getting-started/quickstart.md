# Quickstart Pipeline

This tutorial demonstrates how to ingest data with D1, create model-ready windows with D2, train a model, and evaluate predictions using the DSIPTS-PTF stack.

## 1. Prepare a Dataset

The repository ships with synthetic weather data under `tests/integration/synthetic_weather.csv`.

```python
import pandas as pd
from pathlib import Path

data_path = Path("tests/integration/synthetic_weather.csv")
df = pd.read_csv(data_path)
print(df.head())
```

Ensure the dataframe has a timestamp column (`time`), numerical features, optional categorical columns, and at least one target column.

## 2. Instantiate a D1 Layer

```python
from dsipts.data_structure.d1_layers import MultiSourceTSDataSet

d1 = MultiSourceTSDataSet(
    file_paths=[data_path],
    time_col="time",
    target_cols=["target"],
    past_cols=["temperature", "humidity", "target"],
    future_cols=["temperature", "humidity"],
    cat_cols=["station"],
    enrich_cat=["hour", "dow"],
    global_forecasting=False,
)
```

D1 handles temporal enrichment, metadata extraction, and caching. Metadata is accessible via `d1.metadata`.

## 3. Create a D2 Encoder-Decoder Dataset

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
train_loader = d2.train_dataloader(batch_size=32, shuffle=True)
```

The scaler is fitted on training windows only during `setup(stage="fit")`. When `memory_efficient=False`, transformed windows are cached for fast iteration.

## 4. Train a Model

```python
import pytorch_lightning as pl
from dsipts.models.linear.auto_regression import LinearTS

model = LinearTS(
    past_steps=96,
    future_steps=24,
    out_channels=len(d1.metadata["target_cols"]),
    past_channels=len(d1.metadata["num_past_cols"]),
    future_channels=len(d1.metadata["num_future_cols"]),
)

trainer = pl.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=d2.val_dataloader())
```

Lightning calls `d2.setup(stage="fit")` and `d2.setup(stage="validate")` automatically, reusing the fitted scalers without leakage.

## 5. Evaluate

```python
test_loader = d2.test_dataloader()
trainer.test(model=model, dataloaders=test_loader)
```

For custom inference, iterate over the dataloaders:

```python
batch = next(iter(test_loader))
y_pred = model(batch)
print(y_pred.shape)
```

## Next Steps

- Learn more about [configuring data layers](../user-guide/data-pipeline.md).
- Explore model options in the [reference section](../reference/models.md).
- Reproduce experiments using the Hydra templates in `bash_examples/README.md`.
