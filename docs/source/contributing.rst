Contributing
============

We welcome contributions to SheetWise! This document provides guidelines for contributing.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Run tests and ensure they pass
6. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Khushiyant/sheetwise.git
   cd sheetwise

   # Install dependencies
   pip install -e ".[dev]"

   # Or with uv
   uv pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=sheetwise --cov-report=html

   # Run specific test file
   pytest tests/test_core.py

Code Style
----------

We use:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

.. code-block:: bash

   # Format code
   black src/ tests/

   # Sort imports
   isort src/ tests/

   # Check linting
   flake8 src/ tests/

   # Type check
   mypy src/

Documentation
-------------

When adding new features, please:

1. Add docstrings to all public functions and classes
2. Update the relevant user guide documentation
3. Add examples if appropriate
4. Regenerate API docs:

.. code-block:: bash

   python scripts/generate_docs.py
   cd docs && make html

Submitting Changes
------------------

1. Ensure all tests pass
2. Update CHANGELOG.md
3. Write a clear commit message
4. Push to your fork
5. Submit a pull request with a clear description

Pull Request Guidelines
-----------------------

- Keep changes focused and atomic
- Include tests for new features
- Update documentation
- Follow existing code style
- Reference any related issues

Questions?
----------

Feel free to open an issue on GitHub if you have questions!
