"""Test the main SpreadsheetLLM class (Offline Edition)."""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from sheetwise import SpreadsheetLLM

# Try importing duckdb to skip SQL tests if not installed
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class TestSpreadsheetLLM:
    """Test cases for the main SpreadsheetLLM class."""

    def test_initialization(self):
        """Test SpreadsheetLLM initialization."""
        sllm = SpreadsheetLLM()
        assert sllm.compressor is not None
        assert sllm.vanilla_encoder is not None
        assert sllm.chain_processor is not None

    def test_compress_spreadsheet(self, sparse_dataframe):
        """Test spreadsheet compression."""
        sllm = SpreadsheetLLM()
        result = sllm.compress_spreadsheet(sparse_dataframe)

        assert "original_shape" in result
        assert "compressed_data" in result
        assert "compression_ratio" in result
        # Note: 'compression_steps' is optional depending on implementation, 
        # checking the core keys is sufficient.
        assert result["original_shape"] == sparse_dataframe.shape
        assert result["compression_ratio"] >= 1.0

    def test_compress_and_encode_for_llm(self, financial_dataframe):
        """Test the main compression and encoding pipeline."""
        sllm = SpreadsheetLLM()
        encoded = sllm.compress_and_encode_for_llm(financial_dataframe)

        assert isinstance(encoded, str)
        # Updated assertion to match new offline header format
        assert "# Data (Compressed" in encoded
        assert len(encoded) > 0

    def test_process_qa_query(self, sample_dataframe):
        """Test deterministic QA query processing."""
        sllm = SpreadsheetLLM()
        query = "What is the total revenue?"
        
        # Mock the detector or ensure sample_dataframe has data
        result = sllm.process_qa_query(sample_dataframe, query)

        # Updated assertions for new Offline Chain keys
        assert "query" in result
        assert "matches_found" in result
        assert "top_hits" in result
        assert "processing_stages" in result
        assert result["query"] == query

    def test_get_encoding_stats(self, sample_dataframe):
        """Test encoding statistics calculation."""
        sllm = SpreadsheetLLM()
        stats = sllm.get_encoding_stats(sample_dataframe)

        # Updated keys for simplified offline stats
        required_keys = [
            "original_shape",
            "non_empty_cells",
            "compression_ratio"
        ]

        for key in required_keys:
            assert key in stats

        assert stats["original_shape"] == sample_dataframe.shape
        assert stats["compression_ratio"] >= 1.0

    def test_encode_compressed_for_llm(self, sample_dataframe):
        """Test encoding compressed result for LLM."""
        sllm = SpreadsheetLLM()
        compressed = sllm.compress_spreadsheet(sample_dataframe)
        encoded = sllm.encode_compressed_for_llm(compressed)

        assert isinstance(encoded, str)
        # Updated assertion
        assert "# Data (Compressed" in encoded
        assert len(encoded) > 0

    @pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
    def test_query_sql(self, sample_dataframe):
        """Test SQL querying capability."""
        sllm = SpreadsheetLLM()
        # Create a simple query
        result = sllm.query_sql(sample_dataframe, "SELECT * FROM df LIMIT 1")
        assert not result.empty
        assert result.shape[0] == 1

    def test_encode_to_json(self, sample_dataframe):
        """Test JSON export."""
        sllm = SpreadsheetLLM()
        json_output = sllm.encode_to_json(sample_dataframe)
        
        assert isinstance(json_output, str)
        assert "metadata" in json_output
        assert "cell_index" in json_output
        assert "{" in json_output and "}" in json_output