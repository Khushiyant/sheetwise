"""Table detection utilities."""

from typing import List

import pandas as pd

from sheetwise.encoding.data_types import TableRegion


class TableDetector:
    """Detects table regions in spreadsheets"""

    def __init__(self, min_table_size: int = 2):
        self.min_table_size = min_table_size

    def detect_tables(self, df: pd.DataFrame) -> List[TableRegion]:
        """
        Detect table regions in the spreadsheet

        Args:
            df: Input DataFrame

        Returns:
            List of detected table regions
        """
        tables = []

        # Simple heuristic: find rectangular regions with data
        non_empty_cells = []
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                if pd.notna(row[col]) and row[col] != "":
                    non_empty_cells.append((i, j))

        if not non_empty_cells:
            return tables

        # Find bounding box of non-empty cells
        min_row = min(cell[0] for cell in non_empty_cells)
        max_row = max(cell[0] for cell in non_empty_cells)
        min_col = min(cell[1] for cell in non_empty_cells)
        max_col = max(cell[1] for cell in non_empty_cells)

        # Create table region
        top_left = self._to_excel_address(min_row, min_col)
        bottom_right = self._to_excel_address(max_row, max_col)

        table = TableRegion(
            top_left=top_left,
            bottom_right=bottom_right,
            rows=range(min_row, max_row + 1),
            cols=range(min_col, max_col + 1),
            confidence=1.0,
        )

        tables.append(table)
        return tables

    def _to_excel_address(self, row: int, col: int) -> str:
        """Convert row, column indices to Excel address"""
        col_letter = ""
        col_num = col + 1
        while col_num > 0:
            col_num -= 1
            col_letter = chr(col_num % 26 + ord("A")) + col_letter
            col_num //= 26
        return f"{col_letter}{row + 1}"
