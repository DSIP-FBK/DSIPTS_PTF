# Hydra Configuration Workflow

Experiments in DSIPTS-PTF are commonly orchestrated through Hydra and OmegaConf templates stored in `bash_examples/`. This workflow lets you reproduce complex training pipelines with minimal Python code.

## Directory Layout

- `bash_examples/config_<dataset>/architecture/`: Model presets per dataset family.
- `bash_examples/config_<dataset>/stack/`: Stack configurations that compose D1/D2 settings with models.
- `bash_examples/config_<dataset>/compare_*.yaml`: Bundles comparing multiple experiments in a single run.
- `bash_examples/utils.py`: Helper functions for initialising models, datamodules, and callbacks.

## Launching an Experiment

```bash
python bash_examples/train.py \
  --config-dir bash_examples/config_weather \
  --config-name config \
  architecture=itransformer \
  dataset.path=data/weather.csv \
  train_config.max_epochs=20
```

Hydra merges the base config with overrides passed on the command line. Key nodes include:

- `architecture`: Selects the model and its hyperparameters.
- `dataset`: Controls D1 and D2 settings such as window lengths, enrichment, and scaling.
- `train_config`: Houses PyTorch Lightning trainer parameters and logging options.

## Tips for Customisation

- **Override cascading**: Use `+` to append new keys without modifying the base config, e.g. `+optim_config.weight_decay=0.01`.
- **Version control**: Keep custom overrides in separate YAML files and pass `--config-path` or `--config-dir` to point to them.
- **Hydra sweepers**: Integrate Optuna or basic grid search by adding the appropriate Hydra sweeper plugin and defining search spaces.

## Logging and Outputs

By default, Hydra creates a timestamped output directory under `outputs/`. You can control it via `hydra.run.dir=<path>` or set `hydra.output_subdir=null` to reuse the working directory.

Aim logging is configured inside the templates; ensure you run `aim init` once per machine before launching sweeps.

## Debugging Configs

- Enable `HYDRA_FULL_ERROR=1` for verbose stack traces.
- Use `python bash_examples/train.py --cfg job --resolve` to inspect the final merged configuration without running training.
- Validate datamodule parameters by calling the `print_config()` helper defined in some templates or by logging `d2.metadata` after setup.

Refer back to the [training guide](training.md) for code-level integration details.
