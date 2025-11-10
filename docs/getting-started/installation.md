# Installation

## Prerequisites

- Python 3.9 or newer
- pip or conda for dependency management
- CUDA-capable GPU (optional, recommended for training large models)
- Git access to clone the repository and obtain configuration bundles

## Clone the Repository

```bash
git clone https://github.com/DSIP-FBK/DSIPTS_PTF.git
cd DSIPTS_PTF
```

## Create a Virtual Environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

Or with Conda:

```bash
conda create -n dsipts-ptf python=3.10
conda activate dsipts-ptf
```

## Install Dependencies

Install the project with extras required for documentation and testing:

```bash
pip install -r requirements.txt
pip install -e .
```

To enable optional performance tooling and linting used in CI:

```bash
pip install -r docs/requirements.txt  # if available
pip install pre-commit
pre-commit install
```

> The `pyproject.toml` exports the library as `dsipts`. Editable mode keeps the installation synced with local changes.

## Verify the Setup

```bash
python - <<'PY'
import dsipts
print(dsipts.__version__)
PY
```

You can now proceed to the [quickstart pipeline](quickstart.md) to run an end-to-end example.
