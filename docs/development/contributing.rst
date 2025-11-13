Contributing
============

We welcome contributions to DSIPTS! This guide will help you get started.

Development Setup
-----------------

1. Fork and clone the repository:

.. code-block:: bash

   git clone https://github.com/your-username/DSIPTS_PTF.git
   cd DSIPTS_PTF

2. Install development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

3. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Running Tests
-------------

Run all tests:

.. code-block:: bash

   pytest tests/

Run specific test categories:

.. code-block:: bash

   pytest tests/unit/          # Unit tests only
   pytest tests/integration/   # Integration tests only

With coverage:

.. code-block:: bash

   pytest tests/ --cov=dsipts --cov-report=html

Code Style
----------

We use Ruff for linting and formatting:

.. code-block:: bash

   ruff check .
   ruff format .

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Pull Request Process
---------------------

1. Create a new branch for your feature:

.. code-block:: bash

   git checkout -b feature/my-new-feature

2. Make your changes and commit:

.. code-block:: bash

   git add .
   git commit -m "Add my new feature"

3. Push to your fork:

.. code-block:: bash

   git push origin feature/my-new-feature

4. Open a Pull Request on GitHub

Guidelines
----------

* Write clear commit messages
* Add tests for new features
* Update documentation as needed
* Follow the existing code style
* Keep PRs focused on a single feature/fix

Questions?
----------

Feel free to open an issue for questions or discussions!
