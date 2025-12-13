"""Compression modules for SpreadsheetLLM framework (Enhanced)."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set

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
        
        Improvements (v2.5.1):
        - Added 'Transition Detection': Captures boundaries between regions (e.g., Header -> Body)
        - Vectorized implementation
        """
        if df.empty:
            return [], []

        # Create a grid of types
        type_grid = df.map(DataTypeClassifier.classify_cell_type)
        type_values = type_grid.values

        # 1. Internal Heterogeneity (Existing logic)
        # A row/col is heterogeneous if it has > 2 unique types
        row_nunique = type_grid.nunique(axis=1)
        col_nunique = type_grid.nunique(axis=0)

        anchor_rows = set(np.where(row_nunique.values > 2)[0])
        anchor_cols = set(np.where(col_nunique.values > 2)[0])

        # 2. Structural Transitions (New logic)
        # If a row is significantly different from the one below it, it's a boundary.
        # This catches "All-String Header" vs "All-Float Data"
        if len(df) > 1:
            # Compare Row[i] vs Row[i+1]
            # We count how many columns changed type
            row_diffs = (type_values[:-1] != type_values[1:])
            # If > 50% of columns change type, it's a boundary
            change_magnitude = row_diffs.sum(axis=1)
            threshold = max(1, df.shape[1] // 2)
            
            # Get indices where change is significant
            transition_indices = np.where(change_magnitude >= threshold)[0]
            
            # Add i and i+1 (the boundary pair)
            anchor_rows.update(transition_indices)
            anchor_rows.update(transition_indices + 1)

        # 3. Always include boundaries
        anchor_rows.add(0)
        anchor_rows.add(len(df) - 1)
        anchor_cols.add(0)
        anchor_cols.add(len(df.columns) - 1)

        return sorted(list(anchor_rows)), sorted(list(anchor_cols))

    def extract_skeleton(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spreadsheet skeleton by keeping only structurally important rows/columns.
        """
        if df.empty:
            return df

        # Vectorized detection of content
        content_mask = df.notna() & (df != "")
        
        rows_with_content_indices = np.where(content_mask.any(axis=1).values)[0]
        cols_with_content_indices = np.where(content_mask.any(axis=0).values)[0]

        anchor_rows, anchor_cols = self.find_structural_anchors(df)

        important_rows = set(rows_with_content_indices)
        important_cols = set(cols_with_content_indices)

        # Process Anchors with broadcasting
        if len(rows_with_content_indices) > 0:
            content_row_arr = np.array(rows_with_content_indices)
            for anchor in anchor_rows:
                # Find content rows within k distance
                dist = np.abs(content_row_arr - anchor)
                nearby = content_row_arr[dist <= self.k]
                important_rows.update(nearby)

        if len(cols_with_content_indices) > 0:
            content_col_arr = np.array(cols_with_content_indices)
            for anchor in anchor_cols:
                dist = np.abs(content_col_arr - anchor)
                nearby = content_col_arr[dist <= self.k]
                important_cols.update(nearby)

        # Fallback for empty/sparse sheets
        if not important_rows:
            important_rows = {0}
        if not important_cols:
            important_cols = {0}

        # Slicing
        return df.iloc[sorted(list(important_rows)), sorted(list(important_cols))].copy()


class InvertedIndexTranslator:
    """Implements inverted-index translation with 2D block compression"""

    def translate(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        if df.empty:
            return {}

        # Reset index for clean coordinates
        temp_df = df.copy()
        temp_df.index = range(len(temp_df))
        temp_df.columns = range(len(temp_df.columns))
        
        stacked = temp_df.stack()
        mask = (stacked != "") & pd.notna(stacked)
        valid_cells = stacked[mask]

        # Group coordinates by value
        value_groups = defaultdict(list)
        for (row, col), value in valid_cells.items():
            value_groups[str(value).strip()].append((row, col))

        # Convert groups to compressed ranges
        final_index = {}
        for val, coords in value_groups.items():
            if not val: continue
            if len(coords) > 1:
                final_index[val] = self._merge_2d_ranges(coords)
            else:
                r, c = coords[0]
                final_index[val] = [self._to_excel_address(r, c)]

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

    def _merge_2d_ranges(self, coords: List[Tuple[int, int]]) -> List[str]:
        """
        Merges a list of (row, col) coordinates into optimal rectangular ranges.
        Uses a greedy maximal rectangle algorithm.
        """
        # Convert to set for fast lookup
        point_set = set(coords)
        ranges = []
        
        # Sort to ensure deterministic processing (Row-major)
        sorted_coords = sorted(coords)
        
        processed = set()

        for r, c in sorted_coords:
            if (r, c) in processed:
                continue
                
            # Start a new rectangle here
            width = 1
            height = 1
            
            # 1. Expand Right
            while (r, c + width) in point_set and (r, c + width) not in processed:
                width += 1
                
            # 2. Expand Down (must match full width)
            can_expand_down = True
            while can_expand_down:
                next_r = r + height
                # Check if the entire row segment exists below
                row_segment = [(next_r, c + w) for w in range(width)]
                if all(p in point_set and p not in processed for p in row_segment):
                    height += 1
                else:
                    can_expand_down = False
            
            # 3. Mark as processed
            for i in range(height):
                for j in range(width):
                    processed.add((r + i, c + j))
            
            # 4. Generate Address String
            start_addr = self._to_excel_address(r, c)
            if width == 1 and height == 1:
                ranges.append(start_addr)
            else:
                end_addr = self._to_excel_address(r + height - 1, c + width - 1)
                ranges.append(f"{start_addr}:{end_addr}")
                
        return ranges


class DataFormatAggregator:
    """Implements data-format-aware aggregation for numerical cells"""

    def aggregate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate cells by data format and type.
        Uses 2D Block Compression for token efficiency.
        """
        if df.empty:
            return {}

        type_grid = df.map(DataTypeClassifier.classify_cell_type)
        aggregated = {}
        unique_types = type_grid.stack().unique()
        
        for data_type in unique_types:
            if data_type == "Empty":
                continue
                
            mask = (type_grid == data_type)
            rows, cols = np.where(mask)
            
            # Prepare coords for 2D merger
            coords = list(zip(rows, cols))
            
            if len(coords) > 1:
                # Reuse the logic, but we need metadata, not just address strings
                ranges = self._group_2d_blocks(df, coords)
                aggregated[data_type] = ranges
            else:
                # Single cell case
                r, c = coords[0]
                aggregated[data_type] = [{
                    "address": InvertedIndexTranslator()._to_excel_address(r, c),
                    "value": df.iloc[r, c],
                    "row": r, "col": c
                }]

        return aggregated

    def _group_2d_blocks(self, df: pd.DataFrame, coords: List[Tuple[int, int]]) -> List[Dict]:
        """Similar to _merge_2d_ranges but preserves sample values and counts"""
        point_set = set(coords)
        groups = []
        sorted_coords = sorted(coords)
        processed = set()
        translator = InvertedIndexTranslator()

        for r, c in sorted_coords:
            if (r, c) in processed:
                continue
            
            width = 1
            height = 1
            
            # Expand Right
            while (r, c + width) in point_set and (r, c + width) not in processed:
                width += 1
                
            # Expand Down
            while True:
                next_r = r + height
                row_segment = [(next_r, c + w) for w in range(width)]
                if all(p in point_set and p not in processed for p in row_segment):
                    height += 1
                else:
                    break
            
            # Mark processed
            for i in range(height):
                for j in range(width):
                    processed.add((r + i, c + j))
            
            start_addr = translator._to_excel_address(r, c)
            
            if width == 1 and height == 1:
                groups.append({
                    "address": start_addr,
                    "value": df.iloc[r, c],
                    "row": r, "col": c
                })
            else:
                end_addr = translator._to_excel_address(r + height - 1, c + width - 1)
                groups.append({
                    'type': 'range',
                    'start': start_addr,
                    'end': end_addr,
                    'count': width * height,
                    'sample_value': df.iloc[r, c] # Representative value
                })
                
        return groups