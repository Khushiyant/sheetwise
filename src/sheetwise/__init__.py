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

from sheetwise.core.core_aggregator import SpreadsheetLLM, ChainOfSpreadsheet, SheetCompressor
from sheetwise.encoding.encoding_aggregator import CellInfo,TableRegion,VanillaEncoder,FormulaParser, FormulaDependencyAnalyzer
from sheetwise.utils.utils_aggregator import create_realistic_spreadsheet,CompressionVisualizer
from sheetwise.workbook.workbook_aggregator import WorkbookManager
from sheetwise.tables.tables_aggregator import SmartTableDetector, TableType, EnhancedTableRegion

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
