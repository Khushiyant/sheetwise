Quick Start Guide
=================

This guide will help you get started with SheetWise in just a few minutes.

Basic Example
-------------

Here's a simple example to compress and encode a spreadsheet:

.. code-block:: python

   import pandas as pd
   from sheetwise import SpreadsheetLLM

   # Initialize
   sllm = SpreadsheetLLM()

   # Load your spreadsheet
   df = pd.read_excel("data.xlsx")

   # Compress and encode
   encoded = sllm.compress_and_encode_for_llm(df)
   print(encoded)

Auto-Configuration
------------------

Let SheetWise automatically optimize compression settings:

.. code-block:: python

   from sheetwise import SpreadsheetLLM

   sllm = SpreadsheetLLM()
   df = pd.read_excel("data.xlsx")

   # Auto-configure based on spreadsheet characteristics
   auto_compressed = sllm.compress_with_auto_config(df)

Token Budget Control
--------------------

Ensure your output fits within a token limit:

.. code-block:: python

   from sheetwise import SpreadsheetLLM

   sllm = SpreadsheetLLM()
   df = pd.read_excel("large_file.xlsx")

   # Automatically adjust compression to fit 4000 tokens
   encoded = sllm.encode_to_token_limit(df, max_tokens=4000)

Working with Formulas
---------------------

Extract and analyze Excel formulas:

.. code-block:: python

   from sheetwise import FormulaParser

   parser = FormulaParser()
   formulas = parser.extract_formulas("workbook.xlsx")
   
   # Build dependency graph
   parser.build_dependency_graph()
   
   # Get formula impact
   impact = parser.get_formula_impact("Sheet1!A1")
   print(impact)

Advanced Table Detection
------------------------

Detect and classify tables in your spreadsheet:

.. code-block:: python

   from sheetwise import SmartTableDetector

   detector = SmartTableDetector(header_detection=True)
   tables = detector.detect_tables(df)

   for table in tables:
       print(f"Table type: {table.table_type}")
       print(f"Has headers: {table.has_headers}")
       print(f"Header rows: {table.header_rows}")
       print(f"Header cols: {table.header_cols}")

Multi-Sheet Workbooks
---------------------

Process entire workbooks:

.. code-block:: python

   from sheetwise import WorkbookManager, SpreadsheetLLM

   workbook = WorkbookManager()
   sheets = workbook.load_workbook("workbook.xlsx")
   
   # Detect cross-sheet references
   refs = workbook.detect_cross_sheet_references()
   
   # Compress entire workbook
   sllm = SpreadsheetLLM()
   compressed = workbook.compress_workbook(sllm.compressor)
   encoded = workbook.encode_workbook_for_llm(compressed)

Visualization
-------------

Generate visual reports:

.. code-block:: python

   from sheetwise import CompressionVisualizer, SpreadsheetLLM

   sllm = SpreadsheetLLM()
   visualizer = CompressionVisualizer()
   
   df = pd.read_excel("data.xlsx")
   compressed = sllm.compress_spreadsheet(df)
   
   # Create heatmap
   fig = visualizer.create_data_density_heatmap(df)
   fig.savefig("heatmap.png")
   
   # Generate HTML report
   html = visualizer.generate_html_report(df, compressed)
   with open("report.html", "w") as f:
       f.write(html)

Command Line Interface
----------------------

Use SheetWise from the command line:

.. code-block:: bash

   # Basic usage
   sheetwise input.xlsx -o output.txt

   # With auto-configuration
   sheetwise input.xlsx --auto-config --verbose

   # Run demo
   sheetwise --demo

   # JSON output
   sheetwise input.xlsx --format json

Next Steps
----------

* Learn about :doc:`user_guide/compression` techniques
* Explore the :doc:`api/core` reference
* Check out :doc:`examples` for more use cases
