Installation
============

Requirements
------------

DSIPTS requires Python 3.8 or later.

Install from PyPI
-----------------

.. code-block:: bash

   pip install dsipts

Install from Source
-------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/your-org/DSIPTS_PTF.git
   cd DSIPTS_PTF
   pip install -e .

Dependencies
------------

Core dependencies include:

* PyTorch >= 2.0.0
* PyTorch Lightning == 1.9.4
* NumPy >= 1.24.0
* Pandas >= 2.0.0
* scikit-learn >= 1.2.0

For the complete list, see ``requirements.txt``.

Optional Dependencies
---------------------

For documentation building:

.. code-block:: bash

   pip install -e ".[docs]"

For development:

.. code-block:: bash

   pip install -e ".[dev]"
