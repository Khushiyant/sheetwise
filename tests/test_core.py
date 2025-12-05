"""Test the main SpreadsheetLLM class."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from sheetwise import SpreadsheetLLM


class TestSpreadsheetLLM:
    """Test cases for the main SpreadsheetLLM class."""

    def test_initialization(self):
        """Test SpreadsheetLLM initialization."""
        sllm = SpreadsheetLLM()
        assert sllm.compressor is not None
        assert sllm.vanilla_encoder is not None
        assert sllm.chain_processor is not None

    def test_initialization_with_params(self):
        """Test SpreadsheetLLM initialization with custom parameters."""
        params = {"k": 2, "use_extraction": False}
        sllm = SpreadsheetLLM(compression_params=params)
        assert sllm.compressor.k == 2
        assert sllm.compressor.use_extraction is False

    def test_encode_vanilla(self, sample_dataframe):
        """Test vanilla encoding."""
        sllm = SpreadsheetLLM()
        encoded = sllm.encode_vanilla(sample_dataframe)

        assert isinstance(encoded, str)
        assert "A1,Header1" in encoded
        assert "B2,100" in encoded
        assert len(encoded) > 0

    def test_compress_spreadsheet(self, sparse_dataframe):
        """Test spreadsheet compression."""
        sllm = SpreadsheetLLM()
        result = sllm.compress_spreadsheet(sparse_dataframe)

        assert "original_shape" in result
        assert "compressed_data" in result
        assert "compression_ratio" in result
        assert "compression_steps" in result

        assert result["original_shape"] == sparse_dataframe.shape
        assert result["compression_ratio"] >= 1.0

    def test_compress_and_encode_for_llm(self, financial_dataframe):
        """Test the main compression and encoding pipeline."""
        sllm = SpreadsheetLLM()
        encoded = sllm.compress_and_encode_for_llm(financial_dataframe)

        assert isinstance(encoded, str)
        assert "Spreadsheet Data (Compressed" in encoded
        assert len(encoded) > 0

    def test_process_qa_query(self, sample_dataframe):
        """Test QA query processing."""
        sllm = SpreadsheetLLM()
        query = "What is the total revenue?"
        result = sllm.process_qa_query(sample_dataframe, query)

        assert "compression_info" in result
        assert "detected_tables" in result
        assert "query" in result
        assert result["query"] == query

    def test_get_encoding_stats(self, sample_dataframe):
        """Test encoding statistics calculation."""
        sllm = SpreadsheetLLM()
        stats = sllm.get_encoding_stats(sample_dataframe)

        required_keys = [
            "original_shape",
            "compressed_shape",
            "vanilla_tokens_estimate",
            "compressed_tokens_estimate",
            "compression_ratio",
            "token_reduction_ratio",
            "sparsity_percentage",
            "non_empty_cells",
        ]

        for key in required_keys:
            assert key in stats

        assert stats["original_shape"] == sample_dataframe.shape
        assert stats["compression_ratio"] >= 1.0
        assert 0 <= stats["sparsity_percentage"] <= 100

    def test_load_from_file_unsupported_format(self):
        """Test loading from unsupported file format."""
        sllm = SpreadsheetLLM()

        with pytest.raises(ValueError, match="Unsupported file format"):
            sllm.load_from_file("test.txt")

    def test_encode_compressed_for_llm(self, sample_dataframe):
        """Test encoding compressed result for LLM."""
        sllm = SpreadsheetLLM()
        compressed = sllm.compress_spreadsheet(sample_dataframe)
        encoded = sllm.encode_compressed_for_llm(compressed)

        assert isinstance(encoded, str)
        assert "Spreadsheet Data" in encoded
        assert len(encoded) > 0

    def test_encode_to_token_limit_basic(self, sample_dataframe):
        """Test token limiting with a limit large enough to fit easily."""
        sllm = SpreadsheetLLM()
        # Set a very large limit (e.g., 1000 tokens)
        encoded = sllm.encode_to_token_limit(sample_dataframe, max_tokens=1000)
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Should contain basic data
        assert "Header1" in encoded

    def test_encode_to_token_limit_iterative(self, sparse_dataframe):
        """
        Test that the method iteratively reduces 'k' when the output is too large.
        We mock the internal methods to simulate a large output shrinking.
        """
        sllm = SpreadsheetLLM(enable_logging=True)
        
        # Create mock side effects for compress_with_auto_config and encode_compressed_for_llm
        # 1. First call (auto-config) returns a huge string (too big)
        # 2. Second call (k=3) returns a huge string
        # 3. Third call (k=2) returns a small string (fits!)
        
        long_string = "x" * 5000  # > 1000 chars
        short_string = "small result" # < 1000 chars
        
        with patch.object(sllm, 'compress_with_auto_config', return_value=long_string), \
             patch.object(sllm, 'encode_compressed_for_llm', side_effect=[long_string, short_string]):
            
            # We set max_tokens=250 -> max_chars = 1000
            result = sllm.encode_to_token_limit(sparse_dataframe, max_tokens=250)
            
            assert result == short_string