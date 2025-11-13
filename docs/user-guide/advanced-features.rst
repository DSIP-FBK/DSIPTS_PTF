Advanced Features
=================

Temporal Enrichment
-------------------

Add temporal features automatically:

.. code-block:: python

   d1_dataset = MultiSourceTSDataSet(
       file_paths=['data.csv'],
       target_cols=['value'],
       enrich_cat=['hour', 'dow', 'month', 'minute']
   )

Available temporal features:

* ``hour``: Hour of day (0-23)
* ``dow``: Day of week (0-6)
* ``month``: Month (1-12)
* ``minute``: Minute (0-59)

Global vs Local Forecasting
----------------------------

**Local Forecasting** (default):
  Each time series is treated independently.

**Global Forecasting**:
  Train a single model across multiple time series.

.. code-block:: python

   d1_dataset = MultiSourceTSDataSet(
       file_paths=['group_a.csv', 'group_b.csv'],
       group_cols=['group_id'],
       global_forecasting=True
   )

Custom Data Splits
------------------

Split by groups:

.. code-block:: python

   d2_dataset = EncoderDecoder(
       d1_dataset=d1_dataset,
       split_ratio=(0.7, 0.15, 0.15),
       split_group_config=([0, 1], [2])  # Train on groups 0,1; test on 2
   )

Scaling Strategies
------------------

DSIPTS supports multiple scaling methods:

.. code-block:: python

   d2_dataset = EncoderDecoder(
       d1_dataset=d1_dataset,
       scaling_method='standard',  # or 'minmax'
       target_scaling_method='standard'  # Scale targets separately
   )

Memory Efficiency
-----------------

For large datasets:

.. code-block:: python

   d1_dataset = MultiSourceTSDataSet(
       file_paths=['large_data.csv'],
       memory_efficient=True  # Load data on-demand
   )

   d2_dataset = EncoderDecoder(
       d1_dataset=d1_dataset,
       memory_efficient=True  # Transform on-the-fly
   )

Quantile Forecasting
--------------------

Predict multiple quantiles:

.. code-block:: python

   model = ITransformer(
       out_channels=3,
       past_steps=96,
       future_steps=24,
       quantiles=[0.1, 0.5, 0.9]
   )

Custom Models
-------------

Extend the base model class:

.. code-block:: python

   from dsipts.models.base import BaseModel

   class MyCustomModel(BaseModel):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Your architecture here

       def forward(self, batch):
           # Your forward pass
           pass
