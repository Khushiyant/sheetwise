"""Data types and structures used throughout the SpreadsheetLLM package."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CellInfo:
    """Information about a spreadsheet cell"""

    address: str
    value: Any
    data_type: str
    format_string: Optional[str] = None
    row: int = 0
    col: int = 0


@dataclass
class TableRegion:
    """Represents a detected table region in the spreadsheet"""

    top_left: str
    bottom_right: str
    rows: range
    cols: range
    confidence: float = 0.0
