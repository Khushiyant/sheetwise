Examples
========

This section provides practical examples of using SheetWise.

Basic Examples
--------------

Simple Compression
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sheetwise import SpreadsheetLLM
   import pandas as pd

   sllm = SpreadsheetLLM()
   df = pd.read_excel("data.xlsx")
   result = sllm.compress_and_encode_for_llm(df)
   print(result)

Advanced Examples
-----------------

Financial Statement Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sheetwise import SmartTableDetector, TableType
   import pandas as pd

   # Load financial statement
   df = pd.read_excel("financial_statement.xlsx")
   
   # Detect tables with headers
   detector = SmartTableDetector(header_detection=True)
   tables = detector.detect_tables(df)
   
   # Find pivot-style tables (financial statements)
   for table in tables:
       if table.table_type == TableType.PIVOT_TABLE:
           print(f"Financial table detected!")
           print(f"Column headers: {table.header_rows}")
           print(f"Row headers: {table.header_cols}")

Multi-Sheet Workbook Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sheetwise import WorkbookManager, SpreadsheetLLM

   # Load entire workbook
   workbook = WorkbookManager()
   sheets = workbook.load_workbook("quarterly_report.xlsx")
   
   # Process all sheets
   sllm = SpreadsheetLLM()
   compressed = workbook.compress_workbook(sllm.compressor)
   
   # Generate LLM-ready text
   encoded = workbook.encode_workbook_for_llm(compressed)

For more examples, see the `examples/` directory in the repository.
