"""Test the enhanced features of SheetWise."""

import pytest
import pandas as pd
from sheetwise.core.core import SpreadsheetLLM
from sheetwise.utils.utils import create_realistic_spreadsheet


class TestEnhancedFeatures:
    """Test cases for enhanced features."""

    def test_auto_configuration(self):
        """Test auto-configuration feature."""
        sllm = SpreadsheetLLM()
        
        # Test with sparse data
        sparse_df = pd.DataFrame({
            'A': [1, '', '', '', 5],
            'B': ['', 2, '', '', ''],
            'C': ['', '', '', 3, '']
        })
        
        config = sllm.auto_configure(sparse_df)
        
        assert 'k' in config
        assert 'use_extraction' in config
        assert 'use_translation' in config
        assert config['k'] <= 5  # Should be smaller for sparse data
        
    def test_compress_with_auto_config(self):
        """Test auto-config compression."""
        sllm = SpreadsheetLLM()
        df = create_realistic_spreadsheet()
        
        result = sllm.compress_with_auto_config(df)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Spreadsheet Data" in result
    def test_logging_enabled(self):
        """Test logging functionality."""
        sllm = SpreadsheetLLM(enable_logging=True)
        
        # Should have logger attribute
        assert hasattr(sllm, 'logger')
        
        # Test that auto-config generates logs (would need to capture logs in real test)
        df = create_realistic_spreadsheet()
        config = sllm.auto_configure(df)
        
        assert isinstance(config, dict)
        
    def test_enhanced_address_ranges(self):
        """Test the enhanced address range merging."""
        from sheetwise.tables.extractors import InvertedIndexTranslator
        
        translator = InvertedIndexTranslator()
        
        # Test with contiguous addresses
        addresses = ["A1", "A2", "A3", "A4", "A5"]
        merged = translator._merge_address_ranges(addresses)
        
        # Should create a range for 5 contiguous cells
        assert any(":" in addr for addr in merged)
        
    def test_contiguous_cell_grouping(self):
        """Test contiguous cell grouping in data aggregator."""
        from sheetwise.tables.extractors import DataFormatAggregator
        
        aggregator = DataFormatAggregator()
        
        # Test with contiguous cells
        cells = [
            {'address': 'A1', 'value': 1, 'row': 0, 'col': 0},
            {'address': 'A2', 'value': 2, 'row': 1, 'col': 0},
            {'address': 'A3', 'value': 3, 'row': 2, 'col': 0},
        ]
        
        grouped = aggregator._group_contiguous_cells(cells)
        
        # Should group contiguous cells into ranges
        assert len(grouped) > 0
        
        # Check if range was created for 3+ contiguous cells
        has_range = any(isinstance(item, dict) and item.get('type') == 'range' for item in grouped)
        assert has_range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
