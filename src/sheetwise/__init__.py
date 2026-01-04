"""
SpreadsheetLLM: A Python Package for Encoding Spreadsheets for Large Language Models

This package implements the key components from the SpreadsheetLLM research:
- SheetCompressor: Efficient encoding framework with three modules
- Chain of Spreadsheet: Multi-step reasoning approach
- Vanilla encoding methods with cell addresses and formats

Additional features include:
- Formula extraction and analysis
- Multi-sheet workbook support
- Advanced table detection
- Visualization tools

Based on the research paper: "SpreadsheetLLM: Encoding Spreadsheets for Large Language Models"
by Microsoft Research Team
"""

from .chain import ChainOfSpreadsheet
from .compressor import SheetCompressor
from .core import SpreadsheetLLM
from .data_types import CellInfo, TableRegion
from .encoders import VanillaEncoder
from .utils import create_realistic_spreadsheet

# New enhanced modules
from .formula_parser import FormulaParser, FormulaDependencyAnalyzer
from .visualizer import CompressionVisualizer
from .workbook import WorkbookManager
from .loaders import SheetwiseLoader
from .smart_tables import SmartTableDetector, TableType, EnhancedTableRegion

try:
    from importlib.metadata import version
    __version__ = version("sheetwise")
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version
    __version__ = version("sheetwise")
except Exception:
    # Fallback if package not installed
    __version__ = "2.1.0"

__author__ = "Based on Microsoft Research SpreadsheetLLM"

__all__ = [
    # Core components
    "SpreadsheetLLM",
    "SheetCompressor",
    "VanillaEncoder",
    "ChainOfSpreadsheet",
    "CellInfo",
    "TableRegion",
    "create_realistic_spreadsheet",

    # Loaders
    "SheetwiseLoader",
    
    # Formula handling
    "FormulaParser",
    "FormulaDependencyAnalyzer",
    
    # Visualization
    "CompressionVisualizer",
    
    # Multi-sheet support
    "WorkbookManager",
    
    # Enhanced table detection
    "SmartTableDetector",
    "TableType",
    "EnhancedTableRegion",
]
