from setuptools import find_packages, setup
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Define documentation-specific dependencies
docs_requires = [
    "sphinx>=7.0.0",
    "sphinx_rtd_theme>=1.0.0",
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