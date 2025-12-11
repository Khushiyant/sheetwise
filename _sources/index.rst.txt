SheetWise Documentation
=======================

.. image:: https://img.shields.io/pypi/v/sheetwise.svg
   :target: https://pypi.org/project/sheetwise/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Welcome to SheetWise
--------------------

**SheetWise** is a Python package for encoding spreadsheets for Large Language Models, implementing the SpreadsheetLLM research framework from Microsoft Research.

Key Features
~~~~~~~~~~~~

* **Intelligent Compression**: Up to 96% reduction in token usage
* **Auto-Configuration**: Automatically optimizes compression settings
* **Multi-Table Support**: Handles complex spreadsheets with multiple tables
* **Structural Analysis**: Identifies and preserves important elements
* **Format-Aware**: Preserves data type and formatting information
* **Formula Analysis**: Extract and simplify Excel formulas
* **Multi-Sheet Support**: Process entire workbooks
* **Visualization Tools**: Generate visual reports

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install sheetwise

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from sheetwise import SpreadsheetLLM

   # Initialize the framework
   sllm = SpreadsheetLLM()

   # Load your spreadsheet
   df = pd.read_excel("your_spreadsheet.xlsx")

   # Compress and encode for LLM use
   llm_ready_text = sllm.compress_and_encode_for_llm(df)

   # Use this text with ChatGPT/Claude
   print(llm_ready_text)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/compression
   user_guide/encoding
   user_guide/formulas
   user_guide/tables
   user_guide/visualization
   user_guide/cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/compressor
   api/encoders
   api/chain
   api/smart_tables
   api/formula_parser
   api/workbook
   api/visualizer
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
