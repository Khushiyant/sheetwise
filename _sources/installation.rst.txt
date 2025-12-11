Installation
============

Requirements
------------

* Python 3.10 or higher
* pip or uv package manager

Install from PyPI
-----------------

The easiest way to install SheetWise is using pip:

.. code-block:: bash

   pip install sheetwise

Install with uv (Recommended)
------------------------------

For faster installation using uv:

.. code-block:: bash

   uv add sheetwise

Development Installation
------------------------

To install SheetWise for development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Khushiyant/sheetwise.git
   cd sheetwise

   # Install with development dependencies using pip
   pip install -e ".[dev]"

   # Or using uv (recommended for faster installation)
   uv pip install -e ".[dev]"

Verify Installation
-------------------

To verify that SheetWise is installed correctly:

.. code-block:: python

   import sheetwise
   print(sheetwise.__version__)

Optional Dependencies
---------------------

For visualization features:

.. code-block:: bash

   pip install sheetwise[viz]

For all features including development tools:

.. code-block:: bash

   pip install sheetwise[dev]
