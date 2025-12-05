"""Compression modules for SpreadsheetLLM framework."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .classifiers import DataTypeClassifier


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
        Identify heterogeneous rows and columns that serve as structural anchors.
        Optimized using vectorized operations.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (anchor_rows, anchor_cols)
        """
        if df.empty:
            return [], []

        # create a grid of types for the entire dataframe at once
        type_grid = df.map(DataTypeClassifier.classify_cell_type)

        # Vectorized check for heterogeneity
        # A row/col is heterogeneous if it has > 2 unique types
        row_nunique = type_grid.nunique(axis=1)
        col_nunique = type_grid.nunique(axis=0)

        # Get integer indices where unique count > 2
        # We use .values to ignore index labels (which might be strings) and get raw positions
        anchor_rows = np.where(row_nunique.values > 2)[0].tolist()
        anchor_cols = np.where(col_nunique.values > 2)[0].tolist()

        # Always include boundaries if they aren't already included
        if 0 not in anchor_rows:
            anchor_rows.insert(0, 0)
        if len(df) - 1 not in anchor_rows and len(df) > 0:
            anchor_rows.append(len(df) - 1)

        if 0 not in anchor_cols:
            anchor_cols.insert(0, 0)
        if len(df.columns) - 1 not in anchor_cols and len(df.columns) > 0:
            anchor_cols.append(len(df.columns) - 1)

        return sorted(list(set(anchor_rows))), sorted(list(set(anchor_cols)))

    def extract_skeleton(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spreadsheet skeleton by keeping only structurally important rows/columns.
        More aggressive compression by removing homogeneous empty regions.

        Args:
            df: Input DataFrame

        Returns:
            Compressed DataFrame with structural skeleton
        """
        if df.empty:
            return df

        # Vectorized detection of content
        # Create a boolean mask where True indicates content exists
        content_mask = df.notna() & (df != "")
        
        # Get indices of rows/cols that have ANY content
        rows_with_content = content_mask.any(axis=1).values
        rows_with_content_indices = np.where(rows_with_content)[0]
        
        cols_with_content = content_mask.any(axis=0).values
        cols_with_content_indices = np.where(cols_with_content)[0]

        # Find structural anchors
        anchor_rows, anchor_cols = self.find_structural_anchors(df)

        # Efficiently calculate valid anchors (those near content)
        # Using sets for O(1) lookups and automatic deduplication
        important_rows = set(rows_with_content_indices)
        important_cols = set(cols_with_content_indices)

        # Process Row Anchors
        if len(rows_with_content_indices) > 0:
            # Convert to numpy arrays for broadcasting
            content_row_arr = np.array(rows_with_content_indices)
            
            for anchor in anchor_rows:
                # Vectorized distance check: find min distance to any content row
                min_dist = np.min(np.abs(content_row_arr - anchor))
                
                if min_dist <= self.k:
                    # Add neighborhood
                    start = max(0, anchor - self.k)
                    end = min(len(df), anchor + self.k + 1)
                    important_rows.update(range(start, end))

        # Process Column Anchors
        if len(cols_with_content_indices) > 0:
            content_col_arr = np.array(cols_with_content_indices)
            
            for anchor in anchor_cols:
                min_dist = np.min(np.abs(content_col_arr - anchor))
                
                if min_dist <= self.k:
                    start = max(0, anchor - self.k)
                    end = min(len(df.columns), anchor + self.k + 1)
                    important_cols.update(range(start, end))

        # Fallback for empty/sparse sheets
        if not important_rows:
            important_rows = {0, min(5, len(df) - 1)} if len(df) > 0 else {0}
        if not important_cols:
            important_cols = {0, min(5, len(df.columns) - 1)} if len(df.columns) > 0 else {0}

        # Sort and slice
        sorted_rows = sorted(list(important_rows))
        sorted_cols = sorted(list(important_cols))

        # Using iloc for integer-position based indexing
        skeleton_df = df.iloc[sorted_rows, sorted_cols].copy()

        return skeleton_df


class InvertedIndexTranslator:
    """Implements inverted-index translation for token efficiency"""

    def translate(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Convert spreadsheet to inverted index format.
        Optimized to use DataFrame stacking instead of iteration.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with cell values as keys and cell addresses as values
        """
        if df.empty:
            return {}

        # Resetting index to get simple integer coordinates relative to the current slice
        # This ensures we work with 0-based integer indices regardless of the DF's index labels
        temp_df = df.copy()
        temp_df.index = range(len(temp_df))
        temp_df.columns = range(len(temp_df.columns))
        
        stacked = temp_df.stack()
        mask = (stacked != "") & pd.notna(stacked)
        valid_cells = stacked[mask]

        inverted_index = defaultdict(list)
        
        # Iterate over the valid cells only
        for (row_idx, col_idx), value in valid_cells.items():
            str_value = str(value).strip()
            if str_value:
                cell_addr = self._to_excel_address(row_idx, col_idx)
                inverted_index[str_value].append(cell_addr)

        # Merge ranges for final output
        final_index = {}
        for value, addresses in inverted_index.items():
            if len(addresses) > 1:
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
            
        # Helper to parse A1 to (col, row)
        def parse_addr(addr):
            i = 0
            while i < len(addr) and addr[i].isalpha():
                i += 1
            col_str = addr[:i]
            row_str = addr[i:]
            
            col_num = 0
            for char in col_str:
                col_num = col_num * 26 + (ord(char) - ord('A') + 1)
            return col_num, int(row_str), addr

        parsed = [parse_addr(addr) for addr in addresses]
        # Sort by row then column
        parsed.sort(key=lambda x: (x[1], x[0]))
        
        ranges = []
        current_range = [parsed[0]]
        
        for i in range(1, len(parsed)):
            prev_col, prev_row, _ = current_range[-1]
            curr_col, curr_row, _ = parsed[i]
            
            # Check adjacency (horizontal OR vertical)
            is_horizontal_next = (curr_row == prev_row and curr_col == prev_col + 1)
            is_vertical_next = (curr_col == prev_col and curr_row == prev_row + 1)
            
            if is_horizontal_next or is_vertical_next:
                current_range.append(parsed[i])
            else:
                self._finalize_range(current_range, ranges)
                current_range = [parsed[i]]
        
        self._finalize_range(current_range, ranges)
        return ranges

    def _finalize_range(self, current_range, ranges):
        """Helper to format and append range"""
        if len(current_range) >= 3:
            start_addr = current_range[0][2]
            end_addr = current_range[-1][2]
            ranges.append(f"{start_addr}:{end_addr}")
        else:
            ranges.extend([x[2] for x in current_range])


class DataFormatAggregator:
    """Implements data-format-aware aggregation for numerical cells"""

    def aggregate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate cells by data format and type.
        Vectorized where possible.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing aggregated format information
        """
        if df.empty:
            return {}

        # Create type grid
        type_grid = df.map(DataTypeClassifier.classify_cell_type)
        
        # We need coordinates for the groups
        aggregated = {}
        
        # Group by type
        unique_types = type_grid.stack().unique()
        
        for data_type in unique_types:
            if data_type == "Empty":
                continue
                
            # Get boolean mask for this type
            mask = (type_grid == data_type)
            
            # Get coordinates
            rows, cols = np.where(mask)
            cells = []
            
            # This part is still a loop but over specific cells of one type
            for r, c in zip(rows, cols):
                val = df.iloc[r, c]
                addr = InvertedIndexTranslator()._to_excel_address(r, c)
                cells.append({
                    "address": addr,
                    "value": val,
                    "row": r,
                    "col": c
                })
            
            if len(cells) > 1:
                aggregated[data_type] = self._group_contiguous_cells(cells)
            else:
                aggregated[data_type] = cells

        return aggregated

    def _group_contiguous_cells(self, cells: List[Dict]) -> List[Dict]:
        """Group contiguous cells with same data type"""
        if len(cells) <= 1:
            return cells
            
        # Sort cells by position (row-major)
        cells.sort(key=lambda x: (x['row'], x['col']))
        
        groups = []
        current_group = [cells[0]]
        
        for i in range(1, len(cells)):
            prev_cell = current_group[-1]
            curr_cell = cells[i]
            
            # Check if cells are adjacent (horizontal OR vertical)
            is_horizontal = (prev_cell['row'] == curr_cell['row'] and 
                           curr_cell['col'] == prev_cell['col'] + 1)
            is_vertical = (prev_cell['col'] == curr_cell['col'] and 
                         curr_cell['row'] == prev_cell['row'] + 1)
            
            if is_horizontal or is_vertical:
                current_group.append(curr_cell)
            else:
                self._finalize_group(current_group, groups)
                current_group = [curr_cell]
        
        self._finalize_group(current_group, groups)
        return groups

    def _finalize_group(self, current_group, groups):
        """Helper to format and append group"""
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