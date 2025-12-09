"""Main SpreadsheetLLM class integrating all components (Offline Edition)."""

from typing import Any, Dict, Optional, Union
import logging
import json
import pandas as pd
import numpy as np

# Optional dependency for SQL
try:
    import duckdb
except ImportError:
    duckdb = None

from .chain import ChainOfSpreadsheet
from .compressor import SheetCompressor
from .encoders import VanillaEncoder


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class SpreadsheetLLM:
    """
    Main class integrating all SheetWise components.
    Includes Offline SQL and JSON export capabilities.
    """

    def __init__(self, compression_params: Dict[str, Any] = None, enable_logging: bool = False):
        """
        Initialize SheetWise framework.

        Args:
            compression_params: Parameters for SheetCompressor
            enable_logging: Enable detailed logging for debugging
        """
        self.params = compression_params or {}
        self.compressor = SheetCompressor(**self.params)
        self.vanilla_encoder = VanillaEncoder()
        # Pass compressor to chain, though chain now uses SmartTableDetector internally
        self.chain_processor = ChainOfSpreadsheet(self.compressor)
        
        if enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger('sheetwise')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load_from_file(self, filepath: str) -> pd.DataFrame:
        """Load spreadsheet from file."""
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            return pd.read_excel(filepath)
        elif filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")

    # --- New Offline Features ---

    def query_sql(self, df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
        """
        Run a SQL query directly against the DataFrame.
        
        Args:
            df: The dataframe to query (treated as table 'df' or 'input_data')
            sql_query: Standard SQL query (e.g., "SELECT * FROM input_data WHERE Year > 2020")
            
        Returns:
            Result as a new DataFrame
        """
        if duckdb is None:
            raise ImportError("Please install 'duckdb' for SQL support: pip install duckdb")
        
        # Register the dataframe as a view named 'input_data'
        input_data = df  # noqa: F841 (variable used inside SQL query context)
        return duckdb.sql(sql_query).df()

    def encode_to_json(self, df: pd.DataFrame) -> str:
        """
        Encode compressed spreadsheet data into structured JSON.
        Ideal for piping into other scripts or APIs.
        """
        # Compress first
        compressed = self.compressor.compress(df)
        
        # Construct structured output
        output = {
            "metadata": {
                "original_rows": df.shape[0],
                "original_cols": df.shape[1],
                "compression_ratio": round(compressed['compression_ratio'], 2)
            },
            "data_types": compressed.get("format_aggregation", {}),
            "cell_index": compressed.get("inverted_index", {})
        }
        
        # Use custom NumpyEncoder to handle np.int64, np.float64, etc.
        return json.dumps(output, indent=2, cls=NumpyEncoder)

    # --- Existing Features ---

    def auto_configure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Auto-configure compression parameters."""
        total_cells = df.shape[0] * df.shape[1]
        non_empty = self._count_non_empty_cells(df)
        sparsity = 1 - (non_empty / total_cells) if total_cells > 0 else 0
        
        config = {}
        if sparsity > 0.9: config['k'] = 2
        elif sparsity > 0.7: config['k'] = 3
        else: config['k'] = 5
            
        if sparsity > 0.95: config['use_aggregation'] = False
        if sparsity > 0.5:
            config['use_extraction'] = True
            config['use_translation'] = True
            
        return config

    def compress_and_encode_for_llm(self, df: pd.DataFrame) -> str:
        """Original Markdown encoding (retained for compatibility)."""
        compressed = self.compressor.compress(df)
        return self.encode_compressed_for_llm(compressed)

    def encode_compressed_for_llm(self, compressed_result: Dict[str, Any]) -> str:
        """Generate text representation (Markdown)."""
        lines = []
        lines.append(f"# Data (Compressed {compressed_result['compression_ratio']:.1f}x)")
        lines.append("")

        if "inverted_index" in compressed_result:
            lines.append("## Values:")
            for value, addresses in compressed_result["inverted_index"].items():
                addr_str = ",".join(addresses)
                lines.append(f"{value}|{addr_str}")

        if "format_aggregation" in compressed_result:
            lines.append("\n## Types:")
            for data_type, cells in compressed_result["format_aggregation"].items():
                if len(cells) > 5:
                    lines.append(f"{data_type}: {len(cells)} cells")

        return "\n".join(lines)

    def process_qa_query(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Process QA query using Chain of Spreadsheet."""
        return self.chain_processor.process_query(df, query)

    def _count_non_empty_cells(self, df: pd.DataFrame) -> int:
        return df.map(lambda x: x != "" and pd.notna(x)).sum().sum()
    
    # Kept for backward compatibility
    def encode_vanilla(self, df: pd.DataFrame, include_format: bool = False) -> str:
        return self.vanilla_encoder.encode_to_markdown(df, include_format)
    
    def compress_spreadsheet(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.compressor.compress(df)
    
    def get_encoding_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics."""
        # Simplified stats for offline usage
        return {
            "original_shape": df.shape,
            "non_empty_cells": self._count_non_empty_cells(df),
            "compression_ratio": self.compressor.compress(df)["compression_ratio"]
        }