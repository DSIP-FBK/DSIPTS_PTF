Basic Usage
===========

Quick Start Example
-------------------

Here's a simple example to get you started with DSIPTS:

.. code-block:: python

   from dsipts.data_structure.d1_layers import MultiSourceTSDataSet
   from dsipts.data_structure.d2_layers import EncoderDecoder
   from dsipts.models import ITransformer
   import pytorch_lightning as pl

   # 1. Load your time series data (D1 Layer)
   d1_dataset = MultiSourceTSDataSet(
       file_paths=['data/weather.csv'],
       target_cols=['temperature'],
       time_col='timestamp',
       enrich_cat=['hour', 'dow']  # Add temporal features
   )

   # 2. Create encoder-decoder structure (D2 Layer)
   d2_dataset = EncoderDecoder(
       d1_dataset=d1_dataset,
       past_len=96,      # 96 timesteps for input
       future_len=24,    # 24 timesteps to predict
       scaling_method='standard'
   )

   # 3. Setup data splits
   d2_dataset.setup(stage='fit')

   # 4. Initialize model
   model = ITransformer(
       out_channels=1,
       past_steps=96,
       future_steps=24,
       d_model=512,
       n_heads=8
   )

   # 5. Train with PyTorch Lightning
   trainer = pl.Trainer(max_epochs=10)
   trainer.fit(model, d2_dataset)

Data Layer Architecture
-----------------------

DSIPTS uses a two-layer data architecture:

**D1 Layer (Data Loading)**
  Handles raw data loading, preprocessing, and temporal enrichment.

**D2 Layer (Windowing & Batching)**
  Creates sliding windows, handles train/val/test splits, and manages scaling.

This separation provides flexibility and prevents data leakage.

Model Training
--------------

All models in DSIPTS are PyTorch Lightning modules, which means you get:

* Automatic GPU/TPU support
* Distributed training
* Checkpointing
* Logging integration
* And more!

See the :doc:`advanced-features` section for more details.
