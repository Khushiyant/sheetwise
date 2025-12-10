"""Test the SheetCompressor class."""

import pandas as pd
import pytest

from sheetwise.compressor import SheetCompressor


class TestSheetCompressor:
    """Test cases for the SheetCompressor class."""

    def test_initialization_default(self):
        """Test default initialization."""
        compressor = SheetCompressor()
        assert compressor.k == 4
        assert compressor.use_extraction is True
        assert compressor.use_translation is True
        assert compressor.use_aggregation is True
        assert compressor.extractor is not None
        assert compressor.translator is not None
        assert compressor.aggregator is not None

    def test_initialization_custom(self):
        """Test custom initialization."""
        compressor = SheetCompressor(
            k=2, use_extraction=False, use_translation=True, use_aggregation=False
        )
        assert compressor.k == 2
        assert compressor.use_extraction is False
        assert compressor.use_translation is True
        assert compressor.use_aggregation is False
        assert compressor.extractor is None
        assert compressor.translator is not None
        assert compressor.aggregator is None

    def test_compress_all_modules(self, sparse_dataframe):
        """Test compression with all modules enabled."""
        compressor = SheetCompressor()
        result = compressor.compress(sparse_dataframe)

        assert "original_shape" in result
        assert "compressed_data" in result
        assert "compression_ratio" in result
        assert "compression_steps" in result

        # Should have all three compression steps
        step_names = [step["step"] for step in result["compression_steps"]]
        assert "structural_extraction" in step_names
        assert "inverted_translation" in step_names
        assert "format_aggregation" in step_names

        assert "inverted_index" in result
        assert "format_aggregation" in result

    def test_compress_extraction_only(self, sparse_dataframe):
        """Test compression with only extraction module."""
        compressor = SheetCompressor(
            use_extraction=True, use_translation=False, use_aggregation=False
        )
        result = compressor.compress(sparse_dataframe)

        step_names = [step["step"] for step in result["compression_steps"]]
        assert "structural_extraction" in step_names
        assert "inverted_translation" not in step_names
        assert "format_aggregation" not in step_names

        assert "inverted_index" not in result
        assert "format_aggregation" not in result

    def test_compress_translation_only(self, sample_dataframe):
        """Test compression with only translation module."""
        compressor = SheetCompressor(
            use_extraction=False, use_translation=True, use_aggregation=False
        )
        result = compressor.compress(sample_dataframe)

        step_names = [step["step"] for step in result["compression_steps"]]
        assert "structural_extraction" not in step_names
        assert "inverted_translation" in step_names
        assert "format_aggregation" not in step_names

        assert "inverted_index" in result
        assert "format_aggregation" not in result

    def test_compress_no_modules(self, sample_dataframe):
        """Test compression with no modules enabled."""
        compressor = SheetCompressor(
            use_extraction=False, use_translation=False, use_aggregation=False
        )
        result = compressor.compress(sample_dataframe)

        assert len(result["compression_steps"]) == 0
        assert "inverted_index" not in result
        assert "format_aggregation" not in result

        # Should return original dataframe
        pd.testing.assert_frame_equal(result["compressed_data"], sample_dataframe)

    def test_compression_ratio_calculation(self, sparse_dataframe):
        """Test compression ratio calculation."""
        compressor = SheetCompressor()
        result = compressor.compress(sparse_dataframe)

        original_cells = sparse_dataframe.shape[0] * sparse_dataframe.shape[1]
        compressed_cells = (
            result["compressed_data"].shape[0] * result["compressed_data"].shape[1]
        )
        expected_ratio = original_cells / compressed_cells

        assert abs(result["compression_ratio"] - expected_ratio) < 0.01

    def test_compress_inplace(self, sample_dataframe):
            """Test compression with inplace=True."""
            compressor = SheetCompressor()
            
            # Create a copy to compare later
            original_copy = sample_dataframe.copy()
            
            # Compress inplace
            result = compressor.compress(sample_dataframe, inplace=True)
            
            assert "compressed_data" in result
            
            # Ensure result is valid
            assert not result["compressed_data"].empty
            
            # Verify the operation didn't crash and produced same results as standard
            standard_result = compressor.compress(original_copy, inplace=False)
            pd.testing.assert_frame_equal(result["compressed_data"], standard_result["compressed_data"])