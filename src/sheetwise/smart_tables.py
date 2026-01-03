"""Advanced table detection and classification utilities."""

from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass

from .data_types import TableRegion


class TableType(Enum):
    """Types of tables that can be detected."""
    DATA_TABLE = "data_table"  # Regular data table with headers
    PIVOT_TABLE = "pivot_table"  # Pivot table or cross-tabulation
    MATRIX = "matrix"  # Numeric matrix (all numbers)
    FORM = "form"  # Form-like layout (label:value pairs)
    MIXED = "mixed"  # Mixed or complex table
    SPARSE = "sparse"  # Very sparse data structure


@dataclass
class EnhancedTableRegion(TableRegion):
    """Extended table region with additional metadata."""
    table_type: TableType = TableType.DATA_TABLE
    has_headers: bool = False
    header_rows: Optional[List[int]] = None
    header_cols: Optional[List[int]] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.header_rows is None:
            self.header_rows = []
        if self.header_cols is None:
            self.header_cols = []
    
    # Convenience properties to maintain compatibility with old code
    @property
    def start_row(self) -> int:
        return self.rows.start
    
    @property
    def end_row(self) -> int:
        return self.rows.stop - 1
    
    @property
    def start_col(self) -> int:
        return self.cols.start
    
    @property
    def end_col(self) -> int:
        return self.cols.stop - 1


class SmartTableDetector:
    """
    Advanced table detection with enhanced capabilities.
    
    This class provides utilities to:
    1. Detect multiple tables in spreadsheets
    2. Identify table headers and structures
    3. Classify tables by type
    4. Handle complex table layouts
    """
    
    def __init__(self, 
                min_table_size: int = 2, 
                max_empty_ratio: float = 0.7,
                header_detection: bool = True):
        """
        Initialize the detector.
        
        Args:
            min_table_size: Minimum number of rows/columns to consider a table
            max_empty_ratio: Maximum ratio of empty cells allowed in a table
            header_detection: Whether to detect headers
        """
        self.min_table_size = min_table_size
        self.max_empty_ratio = max_empty_ratio
        self.header_detection = header_detection
    
    def detect_tables(self, df: pd.DataFrame) -> List[EnhancedTableRegion]:
        """
        Detect multiple tables in a spreadsheet.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of detected enhanced table regions
        """
        # First pass: Create a binary mask of non-empty cells
        mask = ~df.isna() & (df != '')
        
        # If the whole sheet is mostly empty, treat the entire non-empty part as one table
        if mask.sum().sum() < (df.shape[0] * df.shape[1] * 0.1):
            # Just get the bounding box of all non-empty cells
            non_empty_rows = mask.any(axis=1)
            non_empty_cols = mask.any(axis=0)
            
            if non_empty_rows.sum() >= self.min_table_size and non_empty_cols.sum() >= self.min_table_size:
                start_row = non_empty_rows.idxmax()
                end_row = non_empty_rows[::-1].idxmax()
                start_col = non_empty_cols.idxmax()
                end_col = non_empty_cols[::-1].idxmax()
                
                table_region = EnhancedTableRegion(
                    top_left=f"{chr(65 + start_col)}{start_row + 1}",
                    bottom_right=f"{chr(65 + end_col)}{end_row + 1}",
                    rows=range(start_row, end_row + 1),
                    cols=range(start_col, end_col + 1),
                    table_type=TableType.SPARSE,
                    confidence=0.8
                )
                
                # Detect headers for this sparse table
                if self.header_detection:
                    self._detect_headers(df, table_region)
                
                return [table_region]
            return []
        
        # Use connected components to identify distinct tables
        # (This is a simplified version, a full implementation would use more robust algorithms)
        tables = []
        visited = np.zeros(df.shape, dtype=bool)
        
        # Scan for potential table starts
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if mask.iloc[i, j] and not visited[i, j]:
                    # Try to grow a table from this seed
                    table = self._grow_table(df, mask, visited, i, j)
                    if table:
                        tables.append(table)
        
        # If no tables were found with the connected component approach,
        # fall back to a simple scan for the largest continguous region
        if not tables:
            # Find the largest contiguous block of non-empty cells
            max_density = 0
            best_table = None
            
            for i in range(df.shape[0] - self.min_table_size + 1):
                for j in range(df.shape[1] - self.min_table_size + 1):
                    for height in range(self.min_table_size, df.shape[0] - i + 1):
                        for width in range(self.min_table_size, df.shape[1] - j + 1):
                            window = mask.iloc[i:i+height, j:j+width]
                            non_empty = window.sum().sum()
                            density = non_empty / (height * width)
                            
                            if density > max_density and density > (1 - self.max_empty_ratio):
                                max_density = density
                                best_table = EnhancedTableRegion(
                                    top_left=f"{chr(65 + j)}{i + 1}",
                                    bottom_right=f"{chr(65 + j+width-1)}{i+height}",
                                    rows=range(i, i+height),
                                    cols=range(j, j+width),
                                    table_type=TableType.DATA_TABLE,
                                    confidence=density
                                )
            
            if best_table:
                # Detect headers for this table
                if self.header_detection:
                    self._detect_headers(df, best_table)
                tables.append(best_table)
        
        # For each detected table, identify its type and headers
        for table in tables:
            if self.header_detection:
                self._detect_headers(df, table)
            self._classify_table_type(df, table)
        
        return tables
    
    def _grow_table(self, df: pd.DataFrame, mask: pd.DataFrame, 
                   visited: np.ndarray, start_row: int, start_col: int) -> Optional[EnhancedTableRegion]:
        """
        Grow a table region from a starting cell.
        
        Args:
            df: Input DataFrame
            mask: Binary mask of non-empty cells
            visited: Visited cells mask
            start_row: Starting row index
            start_col: Starting column index
            
        Returns:
            EnhancedTableRegion or None if no valid table
        """
        # Queue for breadth-first traversal
        queue = [(start_row, start_col)]
        visited[start_row, start_col] = True
        
        # Track min/max bounds
        min_row, max_row = start_row, start_row
        min_col, max_col = start_col, start_col
        
        # Count of cells in the table
        cell_count = 1
        
        while queue:
            row, col = queue.pop(0)
            
            # Check all 4 adjacent cells
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if (0 <= new_row < df.shape[0] and 
                    0 <= new_col < df.shape[1] and 
                    not visited[new_row, new_col] and 
                    mask.iloc[new_row, new_col]):
                    
                    # Add to queue
                    queue.append((new_row, new_col))
                    visited[new_row, new_col] = True
                    cell_count += 1
                    
                    # Update bounds
                    min_row = min(min_row, new_row)
                    max_row = max(max_row, new_row)
                    min_col = min(min_col, new_col)
                    max_col = max(max_col, new_col)
        
        # Check if the table is large enough
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        
        if width >= self.min_table_size and height >= self.min_table_size:
            # Check density
            table_size = width * height
            density = cell_count / table_size
            
            if density >= (1 - self.max_empty_ratio):
                # Convert row/col indices to cell references
                top_left = f"{chr(65 + min_col)}{min_row + 1}"
                bottom_right = f"{chr(65 + max_col)}{max_row + 1}"
                
                return EnhancedTableRegion(
                    top_left=top_left,
                    bottom_right=bottom_right,
                    rows=range(min_row, max_row + 1),
                    cols=range(min_col, max_col + 1),
                    confidence=density
                )
        
        return None
    
    def _detect_headers(self, df: pd.DataFrame, table: EnhancedTableRegion) -> None:
        """
        Detect header rows and columns in a table.
        
        Args:
            df: Input DataFrame
            table: Table region to analyze
            
        Updates the table object with header information
        """
        # Extract the table data
        table_data = df.iloc[table.rows.start:table.rows.stop, table.cols.start:table.cols.stop]
        
        # Row headers detection
        header_rows = []
        
        # Check first row
        if table_data.shape[0] > 1:
            first_row = table_data.iloc[0]
            rest_rows = table_data.iloc[1:]
            
            # Header heuristics:
            # 1. More string values in header than in data rows
            # 2. String values in header tend to be shorter than in data rows
            # 3. Data type in columns is consistent below the header
            
            # String ratio in first row
            first_row_string_ratio = sum(isinstance(val, str) for val in first_row if pd.notna(val)) / sum(pd.notna(val) for val in first_row) if sum(pd.notna(val) for val in first_row) > 0 else 0
            
            # String ratio in rest of the table
            rest_string_ratio = sum(isinstance(val, str) for val in rest_rows.values.flatten() if pd.notna(val)) / sum(pd.notna(val) for val in rest_rows.values.flatten()) if sum(pd.notna(val) for val in rest_rows.values.flatten()) > 0 else 0
            
            # If first row has significantly more strings, likely a header
            if first_row_string_ratio > 0.6 and first_row_string_ratio > rest_string_ratio * 1.2:
                header_rows.append(table.start_row)
        
        # Column headers detection (similar approach)
        header_cols = []
        
        if table_data.shape[1] > 1:
            first_col = table_data.iloc[:, 0]
            rest_cols = table_data.iloc[:, 1:]
            
            first_col_string_ratio = sum(isinstance(val, str) for val in first_col if pd.notna(val)) / sum(pd.notna(val) for val in first_col) if sum(pd.notna(val) for val in first_col) > 0 else 0
            rest_col_string_ratio = sum(isinstance(val, str) for val in rest_cols.values.flatten() if pd.notna(val)) / sum(pd.notna(val) for val in rest_cols.values.flatten()) if sum(pd.notna(val) for val in rest_cols.values.flatten()) > 0 else 0
            
            if first_col_string_ratio > 0.6 and first_col_string_ratio > rest_col_string_ratio * 1.2:
                header_cols.append(table.start_col)
        
        # Update the table
        table.header_rows = header_rows
        table.header_cols = header_cols
        table.has_headers = len(header_rows) > 0 or len(header_cols) > 0
    
    def _classify_table_type(self, df: pd.DataFrame, table: EnhancedTableRegion) -> None:
        """
        Classify the type of table.
        
        Args:
            df: Input DataFrame
            table: Table region to classify
            
        Updates the table object with type information
        """
        # Extract the table data
        table_data = df.iloc[table.start_row:table.end_row+1, table.start_col:table.end_col+1]
        
        # Check if mostly numeric (matrix)
        numeric_ratio = table_data.apply(pd.to_numeric, errors='coerce').notna().sum().sum() / (table_data.notna().sum().sum())
        
        # Check if very sparse
        sparsity = 1 - (table_data.notna().sum().sum() / (table_data.shape[0] * table_data.shape[1]))
        
        # Check for pivot table patterns
        has_row_and_col_headers = len(table.header_rows) > 0 and len(table.header_cols) > 0
        numeric_interior = False
        
        if has_row_and_col_headers and table_data.shape[0] > 2 and table_data.shape[1] > 2:
            # Check if the interior cells (excluding headers) are mostly numeric
            interior = table_data.iloc[1:, 1:]
            numeric_interior = interior.apply(pd.to_numeric, errors='coerce').notna().sum().sum() / (interior.notna().sum().sum()) > 0.7
        
        # Check for form-like patterns (key-value pairs)
        key_value_pattern = False
        
        if table_data.shape[1] == 2:
            first_col = table_data.iloc[:, 0]
            second_col = table_data.iloc[:, 1]
            
            # If first column is all strings and second column has values
            if (first_col.apply(lambda x: isinstance(x, str)).mean() > 0.8 and 
                second_col.notna().mean() > 0.5):
                key_value_pattern = True
        
        # Make classification decision
        if sparsity > 0.8:
            table.table_type = TableType.SPARSE
        elif key_value_pattern:
            table.table_type = TableType.FORM
        elif has_row_and_col_headers and numeric_interior:
            table.table_type = TableType.PIVOT_TABLE
        elif numeric_ratio > 0.8:
            table.table_type = TableType.MATRIX
        elif table.has_headers:
            table.table_type = TableType.DATA_TABLE
        else:
            table.table_type = TableType.MIXED
    
    def extract_tables_to_dataframes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract all tables from a spreadsheet into separate dataframes.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping table names to extracted DataFrames
        """
        tables = self.detect_tables(df)
        result = {}
        
        for i, table in enumerate(tables):
            # Extract table data
            table_df = df.iloc[table.start_row:table.end_row+1, table.start_col:table.end_col+1].copy()
            
            # Handle headers if present
            if table.has_headers and table.header_rows:
                # Use first header row as column names
                header_idx = table.header_rows[0] - table.start_row
                if header_idx >= 0 and header_idx < table_df.shape[0]:
                    headers = table_df.iloc[header_idx]
                    table_df.columns = headers
                    # Remove the header row
                    table_df = table_df.iloc[header_idx+1:]
            
            table_name = f"table_{i+1}_{table.table_type.value}"
            result[table_name] = table_df
        
        return result
