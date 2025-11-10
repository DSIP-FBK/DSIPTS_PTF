# Overview

DSIPTS-PTF (Data-Source Informed Predictive Time Series - PyTorch Framework) provides a modular stack for loading time series data, transforming it into model-ready windows, and training deep learning architectures optimised for forecasting.

## Core Concepts

- **Layered data processing**: D1 layers handle raw ingestion, enrichment, and metadata while D2 layers expose model-ready windows with consistent batch structures.
- **Model zoo**: Production-grade implementations for linear baselines, recurrent networks, transformers, diffusion models and specialised architectures.
- **Configuration-driven workflows**: Hydra + OmegaConf templates under `bash_examples/` and notebooks in `notebooks/` to reproduce experiments quickly.
- **Scalable scaling**: Scaling logic runs in D2 to avoid leakage, with memory-aware options for very large datasets.

## When to Use DSIPTS-PTF

Use the framework when you need reproducible, GPU-ready forecasting experiments with:

- Complex categorical enrichment requirements.
- Global or local forecasting across many entities.
- Quantile prediction, probabilistic outputs, or multiple horizons.
- Automated training utilities such as PyTorch Lightning and tuner-assisted optimisation.

Proceed to the [installation guide](installation.md) to set up the environment, then walk through the [quickstart pipeline](quickstart.md).
