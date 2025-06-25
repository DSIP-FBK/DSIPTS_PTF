from setuptools import find_packages, setup
import os

# Define core requirements directly in setup.py to avoid dependency on requirements.txt
core_requirements = [
    "numpy>=1.24.0",
    "aim>=3.29.1",
    "torch>=2.0.0,<2.7.0",
    "scipy>=1.10.0",
    "pytorch-lightning==1.9.4",
    "pandas>=2.0.0",
    "torchmetrics>=0.11.0",
    "lightning_utilities>=0.8.0",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.2",
    "hydra-joblib-launcher>=1.2.0",
    "hydra-optuna-sweeper>=1.2.0",
    "beautifulsoup4==4.13.4",
    "html5lib>=1.1",
    "html-table-parser-python3==0.3.1",
    "sphinx>=7.0.0",
    "sphinx_rtd_theme>=1.0.0",
    "plotly>=5.14.0",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "scikit-learn>=1.2.0",
    "numba>=0.57.0",
    "einops>=0.6.0",
    "matplotlib>=3.7.0",
    "sphinx_pdj_theme>=0.4.0",
    "sphinx_mdinclude>=0.5.0",
    "requests>=2.28.0",
    "starlette>=0.30.0,<0.47.0",
    "pydantic>=1.10.0,<3.0.0",
]

# Try to read from requirements.txt if available, otherwise use the core requirements
try:
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as f:
            requirements = f.read().splitlines()
    else:
        requirements = core_requirements
except Exception:
    # Fallback to core requirements if any issue occurs
    requirements = core_requirements

# Define documentation-specific dependencies
docs_requires = [
    "sphinx>=7.0.0",
    "sphinx_pdj_theme>=0.4.0",
    "sphinx_mdinclude>=0.5.0",
]

setup(
    name="dsipts",
    version="1.1.3",
    author="Andrea Gobbi",
    author_email="agobbi@fbk.eu",
    packages=find_packages(exclude=("tests",)),
    description="Python library for time series forecasting",
    setup_requires=[],
    install_requires=requirements,
    extras_require={
        "docs": docs_requires,
    },
)

'''
"""Custom clean command to tidy up the project root."""
CLEAN_FILES = ['build', 'dist', 'egg-info']

here = os.getcwd()

for dir in os.listdir(here):
    if any( [f in dir for f in CLEAN_FILES] ):
        shutil.rmtree(os.path.join(here,dir))
'''