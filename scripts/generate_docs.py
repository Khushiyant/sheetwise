"""
Automatic documentation generation script for SheetWise.

This script uses sphinx-apidoc to automatically generate API documentation
from the source code docstrings.
"""

import os
import subprocess
import sys
from pathlib import Path


def generate_api_docs():
    """Generate API documentation using sphinx-apidoc."""
    
    # Get paths
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "src" / "sheetwise"
    docs_api_dir = project_root / "docs" / "source" / "api"
    
    print("Generating API documentation...")
    print(f"   Source: {source_dir}")
    print(f"   Output: {docs_api_dir}")
    
    # Ensure the api directory exists
    docs_api_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old API docs
    for file in docs_api_dir.glob("*.rst"):
        if file.name != "index.rst":
            file.unlink()
            print(f"   Removed old file: {file.name}")
    
    # Run sphinx-apidoc
    cmd = [
        "sphinx-apidoc",
        "-f",  # Force overwrite
        "-e",  # Put each module on its own page
        "-M",  # Put module documentation before submodule
        "-T",  # Don't create table of contents file
        "-o", str(docs_api_dir),  # Output directory
        str(source_dir),  # Source directory
        str(source_dir / "__pycache__"),  # Exclude pattern
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("API documentation generated successfully!")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error generating API docs: {e}")
        if e.stderr:
            print(e.stderr)
        sys.exit(1)
    
    # Create a custom API index
    create_api_index(docs_api_dir)


def create_api_index(api_dir):
    """Create a custom API reference index."""
    
    index_content = """API Reference
=============

This section contains the complete API documentation for SheetWise.

Core Classes
------------

.. toctree::
   :maxdepth: 2

   sheetwise.core
   sheetwise.compressor
   sheetwise.encoders
   sheetwise.chain

Advanced Features
-----------------

.. toctree::
   :maxdepth: 2

   sheetwise.smart_tables
   sheetwise.formula_parser
   sheetwise.workbook
   sheetwise.visualizer

Supporting Modules
------------------

.. toctree::
   :maxdepth: 2

   sheetwise.classifiers
   sheetwise.detectors
   sheetwise.extractors
   sheetwise.data_types
   sheetwise.utils
   sheetwise.cli

Module Index
------------

.. autosummary::
   :toctree: generated
   :recursive:

   sheetwise
"""
    
    index_file = api_dir / "index.rst"
    index_file.write_text(index_content)
    print(f"Created API index at {index_file}")


def generate_user_guide_stubs():
    """Generate stub files for user guide if they don't exist."""
    
    project_root = Path(__file__).parent.parent
    user_guide_dir = project_root / "docs" / "source" / "user_guide"
    user_guide_dir.mkdir(parents=True, exist_ok=True)
    
    guides = {
        "compression.rst": """Compression Guide
================

Learn how to use SheetWise's compression features.

.. automodule:: sheetwise.compressor
   :members:
""",
        "encoding.rst": """Encoding Guide
==============

Learn about different encoding strategies.

.. automodule:: sheetwise.encoders
   :members:
""",
        "formulas.rst": """Formula Analysis
================

Extract and analyze Excel formulas.

.. automodule:: sheetwise.formula_parser
   :members:
""",
        "tables.rst": """Table Detection
===============

Detect and classify tables in spreadsheets.

.. automodule:: sheetwise.smart_tables
   :members:
""",
        "visualization.rst": """Visualization
=============

Generate visual reports and charts.

.. automodule:: sheetwise.visualizer
   :members:
""",
        "cli.rst": """Command Line Interface
======================

Use SheetWise from the command line.

.. automodule:: sheetwise.cli
   :members:
"""
    }
    
    for filename, content in guides.items():
        file_path = user_guide_dir / filename
        if not file_path.exists():
            file_path.write_text(content)
            print(f"Created {filename}")


def main():
    """Main function to generate all documentation."""
    
    print("=" * 60)
    print("SheetWise Documentation Generator")
    print("=" * 60)
    
    generate_api_docs()
    print()
    generate_user_guide_stubs()
    
    print()
    print("=" * 60)
    print("Documentation generation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run: cd docs && make html")
    print("  2. Open: docs/build/html/index.html")


if __name__ == "__main__":
    main()
