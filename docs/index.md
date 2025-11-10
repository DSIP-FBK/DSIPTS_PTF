# DSIPTS-PTF Documentation

Welcome to the documentation portal for DSIPTS-PTF, a production-ready toolkit for end-to-end time series forecasting research and experimentation. The library provides:

- Unified data layering through `dsipts/data_structure/d1_layers/` and `dsipts/data_structure/d2_layers/` for reproducible preprocessing and batching.
- Optimised model implementations in `dsipts/models/` spanning linear baselines, recurrent architectures, transformers and diffusion-based approaches.
- Configuration-driven experiments via the `bash_examples/` Hydra templates and ready-to-run notebooks under `notebooks/`.

Use the navigation to explore the guides:

- **Getting Started**: Install the package, prepare datasets and run the quickstart pipeline.
- **User Guide**: Learn how D1 and D2 layers cooperate, configure splits, scaling and enrichment, and manage training workflows.
- **Reference**: Dive into layer APIs, supported models and key utilities.
- **Development**: Follow contributor practices, test strategy and release workflow.

If you are migrating from the legacy DSIPTS repository, review the `development/contributing.md` notes for compatibility changes and modernised interfaces.

Ready to begin? Start with the [overview](getting-started/overview.md) to understand the architecture and choose your path through the rest of the documentation.
