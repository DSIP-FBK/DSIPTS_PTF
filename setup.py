from setuptools import find_packages, setup
import os

# Define core requirements - only what's needed for the package to function
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
    "scikit-learn>=1.2.0",
    "numba>=0.57.0",
    "einops>=0.6.0",
    "matplotlib>=3.7.0",
    "requests>=2.28.0",
    "starlette>=0.30.0,<0.47.0",
    "pydantic>=1.10.0,<3.0.0",
    "plotly>=5.14.0",
    "beautifulsoup4==4.13.4",
    "html5lib>=1.1",
    "html-table-parser-python3==0.3.1",
]

# Define documentation-specific dependencies
# Include core requirements so autodoc can import modules
docs_requires = [
    "sphinx_mdinclude>=0.5.0",
    "sphinx_rtd_theme>=1.0.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "torch>=2.0.0,<2.7.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.2.0",
]

# Define test-specific dependencies
test_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

# Define dev-specific dependencies (combines docs and test requirements)
dev_requires = docs_requires + test_requires

# Use core_requirements directly instead of reading from requirements.txt
# This avoids issues during build when requirements.txt might not be available
requirements = core_requirements

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
        "test": test_requires,
        "dev": dev_requires,
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
