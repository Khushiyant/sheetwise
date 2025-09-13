"""Compression modules for SpreadsheetLLM framework."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd

from sheetwise.encoding.classifiers import DataTypeClassifier


class StructuralAnchorExtractor:
    """Implements structural-anchor-based extraction for layout understanding"""

    def __init__(self, k: int = 4):
        """
        Initialize with k parameter controlling neighborhood retention

        Args:
            k: Number of rows/columns to retain around anchor points
        """
        self.k = k

    def find_structural_anchors(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Identify heterogeneous rows and columns that serve as structural anchors

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (anchor_rows, anchor_cols)
        """
        anchor_rows = []
        anchor_cols = []

        # Find heterogeneous rows (rows with significant changes)
        for i in range(len(df)):
            if self._is_heterogeneous_row(df, i):
                anchor_rows.append(i)

        # Find heterogeneous columns
        for j in range(len(df.columns)):
            if self._is_heterogeneous_col(df, j):
                anchor_cols.append(j)

        return anchor_rows, anchor_cols

    def _is_heterogeneous_row(self, df: pd.DataFrame, row_idx: int) -> bool:
        """Check if a row is heterogeneous (contains diverse data types/formats)"""
        if row_idx >= len(df):
            return False

        row_data = df.iloc[row_idx]
        data_types = [DataTypeClassifier.classify_cell_type(val) for val in row_data]
        unique_types = set(data_types)

        # Consider row heterogeneous if it has multiple data types or is at boundary
        return len(unique_types) > 2 or row_idx in [0, len(df) - 1]

    def _is_heterogeneous_col(self, df: pd.DataFrame, col_idx: int) -> bool:
        """Check if a column is heterogeneous"""
        if col_idx >= len(df.columns):
            return False

        col_data = df.iloc[:, col_idx]
        data_types = [DataTypeClassifier.classify_cell_type(val) for val in col_data]
        unique_types = set(data_types)

        return len(unique_types) > 2 or col_idx in [0, len(df.columns) - 1]

    def extract_skeleton(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spreadsheet skeleton by keeping only structurally important rows/columns
        More aggressive compression by removing homogeneous empty regions

        Args:
            df: Input DataFrame

        Returns:
            Compressed DataFrame with structural skeleton
        """
        # Find rows and columns with actual content
        rows_with_content = []
        cols_with_content = []

        for i in range(len(df)):
            row_has_content = any(
                pd.notna(df.iloc[i, j]) and df.iloc[i, j] != ""
                for j in range(len(df.columns))
            )
            if row_has_content:
                rows_with_content.append(i)

        for j in range(len(df.columns)):
            col_has_content = any(
                pd.notna(df.iloc[i, j]) and df.iloc[i, j] != "" for i in range(len(df))
            )
            if col_has_content:
                cols_with_content.append(j)

        # Find structural anchors among content rows/columns
        anchor_rows, anchor_cols = self.find_structural_anchors(df)

        # Combine content-based and anchor-based selection
        important_rows = set(rows_with_content)
        important_cols = set(cols_with_content)

        # Add anchors and their neighborhoods (but only if they have nearby content)
        for anchor in anchor_rows:
            if any(
                abs(anchor - content_row) <= self.k for content_row in rows_with_content
            ):
                for i in range(
                    max(0, anchor - self.k), min(len(df), anchor + self.k + 1)
                ):
                    important_rows.add(i)

        for anchor in anchor_cols:
            if any(
                abs(anchor - content_col) <= self.k for content_col in cols_with_content
            ):
                for j in range(
                    max(0, anchor - self.k), min(len(df.columns), anchor + self.k + 1)
                ):
                    important_cols.add(j)

        # If no content found, keep minimal structure
        if not important_rows:
            important_rows = {0, min(5, len(df) - 1)}
        if not important_cols:
            important_cols = {0, min(5, len(df.columns) - 1)}

        # Extract skeleton
        sorted_rows = sorted(list(important_rows))
        sorted_cols = sorted(list(important_cols))

        skeleton_df = df.iloc[sorted_rows, sorted_cols].copy()

        return skeleton_df


class InvertedIndexTranslator:
    """Implements inverted-index translation for token efficiency"""

    def translate(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Convert spreadsheet to inverted index format
        More efficient by grouping empty cells and deduplicating values

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with cell values as keys and cell addresses as values
        """
        inverted_index = defaultdict(list)
        empty_cells = []

        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                cell_value = row[col]
                cell_addr = self._to_excel_address(i, j)

                if pd.isna(cell_value) or cell_value == "":
                    empty_cells.append(cell_addr)
                else:
                    str_value = str(cell_value).strip()
                    inverted_index[str_value].append(cell_addr)

        # Don't include empty cells in final output (major token savings)
        # Only keep non-empty values
        final_index = {}
        for value, addresses in inverted_index.items():
            if value and value.strip():  # Skip empty/whitespace values
                if len(addresses) > 1:
                    # Try to create ranges for contiguous cells
                    final_index[value] = self._merge_address_ranges(addresses)
                else:
                    final_index[value] = addresses

        return final_index

    def _to_excel_address(self, row: int, col: int) -> str:
        """Convert row, column indices to Excel address (e.g., A1)"""
        col_letter = ""
        col_num = col + 1
        while col_num > 0:
            col_num -= 1
            col_letter = chr(col_num % 26 + ord("A")) + col_letter
            col_num //= 26
        return f"{col_letter}{row + 1}"

    def _merge_address_ranges(self, addresses: List[str]) -> List[str]:
        """Attempt to merge contiguous cell addresses into ranges"""
        if len(addresses) <= 1:
            return addresses
            
        # Parse addresses and sort them
        parsed = []
        for addr in addresses:
            col_match = ""
            row_match = ""
            i = 0
            while i < len(addr) and addr[i].isalpha():
                col_match += addr[i]
                i += 1
            row_match = addr[i:]
            
            # Convert column letters to numbers
            col_num = 0
            for char in col_match:
                col_num = col_num * 26 + (ord(char) - ord('A') + 1)
            
            parsed.append((col_num, int(row_match), addr))
        
        parsed.sort()
        
        # Group contiguous addresses
        ranges = []
        current_range = [parsed[0]]
        
        for i in range(1, len(parsed)):
            prev_col, prev_row, _ = current_range[-1]
            curr_col, curr_row, _ = parsed[i]
            
            # Check if addresses are contiguous
            if (curr_col == prev_col and curr_row == prev_row + 1) or \
               (curr_row == prev_row and curr_col == prev_col + 1):
                current_range.append(parsed[i])
            else:
                # Process current range
                if len(current_range) >= 3:  # Only create ranges for 3+ cells
                    start_addr = current_range[0][2]
                    end_addr = current_range[-1][2]
                    ranges.append(f"{start_addr}:{end_addr}")
                else:
                    ranges.extend([cell[2] for cell in current_range])
                current_range = [parsed[i]]
        
        # Process final range
        if len(current_range) >= 3:
            start_addr = current_range[0][2]
            end_addr = current_range[-1][2]
            ranges.append(f"{start_addr}:{end_addr}")
        else:
            ranges.extend([cell[2] for cell in current_range])
        
        return ranges


class DataFormatAggregator:
    """Implements data-format-aware aggregation for numerical cells"""

    def aggregate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate cells by data format and type

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing aggregated format information
        """
        format_groups = defaultdict(list)

        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                cell_value = row[col]
                if pd.notna(cell_value) and cell_value != "":
                    data_type = DataTypeClassifier.classify_cell_type(cell_value)
                    cell_addr = InvertedIndexTranslator()._to_excel_address(i, j)

                    format_groups[data_type].append(
                        {"address": cell_addr, "value": cell_value, "row": i, "col": j}
                    )

        # Aggregate contiguous regions with same format
        aggregated = {}
        for data_type, cells in format_groups.items():
            if len(cells) > 1:
                # Group cells by proximity
                regions = self._group_contiguous_cells(cells)
                aggregated[data_type] = regions
            else:
                aggregated[data_type] = cells

        return aggregated

    def _group_contiguous_cells(self, cells: List[Dict]) -> List[Dict]:
        """Group contiguous cells with same data type"""
        if len(cells) <= 1:
            return cells
            
        # Sort cells by position
        cells.sort(key=lambda x: (x['row'], x['col']))
        
        groups = []
        current_group = [cells[0]]
        
        for i in range(1, len(cells)):
            prev_cell = current_group[-1]
            curr_cell = cells[i]
            
            # Check if cells are adjacent (same row, next column OR same column, next row)
            is_adjacent = (
                (prev_cell['row'] == curr_cell['row'] and 
                 curr_cell['col'] == prev_cell['col'] + 1) or
                (prev_cell['col'] == curr_cell['col'] and 
                 curr_cell['row'] == prev_cell['row'] + 1)
            )
            
            if is_adjacent:
                current_group.append(curr_cell)
            else:
                # Finalize current group
                if len(current_group) >= 3:
                    # Create range representation
                    start_addr = current_group[0]['address']
                    end_addr = current_group[-1]['address']
                    groups.append({
                        'type': 'range',
                        'start': start_addr,
                        'end': end_addr,
                        'count': len(current_group),
                        'sample_value': current_group[0]['value']
                    })
                else:
                    # Keep individual cells
                    groups.extend(current_group)
                
                current_group = [curr_cell]
        
        # Handle final group
        if len(current_group) >= 3:
            start_addr = current_group[0]['address']
            end_addr = current_group[-1]['address']
            groups.append({
                'type': 'range',
                'start': start_addr,
                'end': end_addr,
                'count': len(current_group),
                'sample_value': current_group[0]['value']
            })
        else:
            groups.extend(current_group)
        
        return groups
