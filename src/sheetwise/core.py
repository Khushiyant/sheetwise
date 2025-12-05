"""Main SpreadsheetLLM class integrating all components."""

from typing import Any, Dict, Optional, Union
import logging
import math

import pandas as pd

from .chain import ChainOfSpreadsheet
from .compressor import SheetCompressor
from .encoders import VanillaEncoder


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
        self.params = compression_params or {}
        self.compressor = SheetCompressor(**self.params)
        self.vanilla_encoder = VanillaEncoder()
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

    def encode_to_token_limit(self, df: pd.DataFrame, max_tokens: int) -> str:
        """
        Attempt to compress the spreadsheet to fit within a specific token limit.
        Iteratively adjusts compression aggressiveness.

        Args:
            df: Input DataFrame
            max_tokens: The target maximum number of tokens (approx words/0.75)

        Returns:
            Encoded string fitting the limit, or best effort result.
        """
        # Estimated char/token ratio. 4 chars ~= 1 token is a safe rule of thumb
        max_chars = max_tokens * 4
        
        # 1. Try Auto-Config first (balanced approach)
        encoded = self.compress_with_auto_config(df)
        if len(encoded) <= max_chars:
            return encoded
            
        # 2. Iterative Reduction: Reduce 'k' (neighborhood size)
        # We start from k=3 and go down to 0 (just anchors, no context)
        for k in range(3, -1, -1):
            if hasattr(self, 'logger'):
                self.logger.info(f"Output too large ({len(encoded)} chars). Retrying with k={k}")
            
            compressor = SheetCompressor(k=k, use_extraction=True, use_translation=True, use_aggregation=True)
            result = compressor.compress(df)
            encoded = self.encode_compressed_for_llm(result)
            
            if len(encoded) <= max_chars:
                return encoded

        # 3. Last Resort: Aggressive Truncation
        # If even k=0 is too big, we likely have huge text cells. 
        # We can enable a future feature here to truncate cell values.
        if hasattr(self, 'logger'):
            self.logger.warning(f"Could not meet token limit {max_tokens}. Returning best effort ({len(encoded)} chars).")
            
        return encoded

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
        return self.vanilla_encoder.encode_to_markdown(df, include_format)

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

        # Count actual tokens more accurately
        vanilla_tokens = len(vanilla_encoding.split("|"))  # Each cell is a token

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