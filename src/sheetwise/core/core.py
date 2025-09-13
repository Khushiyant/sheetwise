"""Main SpreadsheetLLM class integrating all components."""

from typing import Any, Dict, Optional, Union
import logging

import pandas as pd

from sheetwise.core.chain import ChainOfSpreadsheet
from sheetwise.core.compressor import SheetCompressor
from sheetwise.encoding.encoders import VanillaEncoder, JSONEncoder


class SpreadsheetLLM:
    """
    Main class integrating all SpreadsheetLLM components
    """

    def __init__(self, compression_params: Dict[str, Any] = None, enable_logging: bool = False):
        """
        Initialize SpreadsheetLLM framework

        Args:
            compression_params: Parameters for SheetCompressor
            enable_logging: Enable detailed logging for debugging
        """
        params = compression_params or {}
        self.compressor = SheetCompressor(**params)
        self.vanilla_encoder = VanillaEncoder()
        self.json_encoder = JSONEncoder()
        self.chain_processor = ChainOfSpreadsheet(self.compressor)
        
        # Setup logging if requested
        if enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """Setup logging for operations"""
        self.logger = logging.getLogger('sheetwise')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def auto_configure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Auto-configure compression parameters based on spreadsheet characteristics
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            Optimized compression parameters
        """
        total_cells = df.shape[0] * df.shape[1]
        non_empty = self._count_non_empty_cells(df)
        sparsity = 1 - (non_empty / total_cells) if total_cells > 0 else 0
        
        # Auto-tune parameters
        config = {}
        
        # Adjust k based on sparsity
        if sparsity > 0.9:  # Very sparse
            config['k'] = 2
        elif sparsity > 0.7:  # Moderately sparse  
            config['k'] = 3
        else:  # Dense data
            config['k'] = 5
            
        # Disable aggregation for very sparse data
        if sparsity > 0.95:
            config['use_aggregation'] = False
            
        # Always use extraction and translation for sparse data
        if sparsity > 0.5:
            config['use_extraction'] = True
            config['use_translation'] = True
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Auto-configured for {sparsity:.1%} sparsity: {config}")
            
        return config

    def compress_with_auto_config(self, df: pd.DataFrame) -> str:
        """
        Automatically configure and compress spreadsheet
        
        Args:
            df: Input DataFrame
            
        Returns:
            LLM-ready text with optimal compression
        """
        # Get optimal configuration
        auto_config = self.auto_configure(df)
        
        # Create new compressor with optimal settings
        optimal_compressor = SheetCompressor(**auto_config)
        
        # Compress and encode
        compressed = optimal_compressor.compress(df)
        return self.encode_compressed_for_llm(compressed)

    def load_from_file(self, filepath: str) -> pd.DataFrame:
        """Load spreadsheet from file"""
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            return pd.read_excel(filepath)
        elif filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")

    def encode_vanilla(self, df: pd.DataFrame, include_format: bool = False) -> str:
        """Encode using vanilla method"""
        return self.vanilla_encoder.encode(df, include_format)
    
    def encode_json(self, df: pd.DataFrame) -> str:
        """Encode with json encoding"""
        return self.json_encoder.encode(df)

    def compress_spreadsheet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compress spreadsheet using SheetCompressor"""
        return self.compressor.compress(df)

    def process_qa_query(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Process QA query using Chain of Spreadsheet"""
        return self.chain_processor.process_query(df, query)

    def encode_compressed_for_llm(self, compressed_result: Dict[str, Any]) -> str:
        """
        Generate LLM-ready text from compressed result
        This is the key output that users paste into ChatGPT/Claude

        Args:
            compressed_result: Output from compress_spreadsheet()

        Returns:
            Clean, minimal text representation for LLM consumption
        """
        lines = []

        # Add compression metadata
        lines.append(
            f"# Spreadsheet Data (Compressed {compressed_result['compression_ratio']:.1f}x)"
        )
        lines.append("")

        # Use inverted index if available (most efficient)
        if "inverted_index" in compressed_result:
            lines.append("## Cell Data (value|addresses):")
            for value, addresses in compressed_result["inverted_index"].items():
                addr_str = ",".join(addresses)
                lines.append(f"{value}|{addr_str}")

        # Add format information compactly
        if "format_aggregation" in compressed_result:
            lines.append("\n## Data Types:")
            for data_type, cells in compressed_result["format_aggregation"].items():
                if len(cells) > 5:  # Only show significant type groups
                    lines.append(f"{data_type}: {len(cells)} cells")

        return "\n".join(lines)

    def compress_and_encode_for_llm(self, df: pd.DataFrame) -> str:
        """
        One-step function: compress spreadsheet and return LLM-ready text
        This is the main function users will call

        Args:
            df: Input DataFrame

        Returns:
            Text ready to paste into ChatGPT/Claude
        """
        compressed = self.compress_spreadsheet(df)
        return self.encode_compressed_for_llm(compressed)

    def get_encoding_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the spreadsheet encoding"""
        vanilla_encoding = self.encode_vanilla(df)
        compressed_result = self.compress_spreadsheet(df)
        json_encoding = self.encode_json(df)

        # Count actual tokens recieved via vanilla encoding
        vanilla_tokens = self.vanilla_encoder.estimate_tokens(vanilla_encoding) 

        # Count estimated tokens recieved via json encoding
        json_tokens = self.json_encoder.estimate_tokens(json_encoding)

        # For compressed data, count meaningful entries
        compressed_tokens = 0
        if "inverted_index" in compressed_result:
            compressed_tokens += len(compressed_result["inverted_index"])
        if "format_aggregation" in compressed_result:
            for data_type, cells in compressed_result["format_aggregation"].items():
                compressed_tokens += len(cells)

        # Fallback token count
        if compressed_tokens == 0:
            compressed_tokens = len(str(compressed_result).split())

        return {
            "original_shape": df.shape,
            "compressed_shape": compressed_result["compressed_data"].shape,
            "vanilla_tokens_estimate": vanilla_tokens,
            "json_tokens_estimate":json_tokens,
            "compressed_tokens_estimate": compressed_tokens,
            "compression_ratio": compressed_result["compression_ratio"],
            "token_reduction_ratio": vanilla_tokens / compressed_tokens
            if compressed_tokens > 0
            else 0,
            "sparsity_percentage": self._calculate_sparsity(df),
            "non_empty_cells": self._count_non_empty_cells(df),
        }

    def _calculate_sparsity(self, df: pd.DataFrame) -> float:
        """Calculate percentage of empty cells"""
        total_cells = df.shape[0] * df.shape[1]
        non_empty = self._count_non_empty_cells(df)
        return ((total_cells - non_empty) / total_cells) * 100

    def _count_non_empty_cells(self, df: pd.DataFrame) -> int:
        """Count non-empty cells"""
        return df.map(lambda x: x != "" and pd.notna(x)).sum().sum()
