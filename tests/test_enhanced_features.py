"""Test the enhanced features of SheetWise (Offline Edition)."""

import pytest
import pandas as pd
import json
from sheetwise import SpreadsheetLLM
from sheetwise.utils import create_realistic_spreadsheet

# Check for optional dependencies
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

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
        # Sparse data implies smaller k
        assert config['k'] <= 5  

    def test_json_export_structure(self):
        """Test that JSON export produces valid, structured data."""
        sllm = SpreadsheetLLM()
        df = create_realistic_spreadsheet()
        
        json_str = sllm.encode_to_json(df)
        data = json.loads(json_str)
        
        assert "metadata" in data
        assert "original_rows" in data["metadata"]
        assert "cell_index" in data
        assert isinstance(data["cell_index"], dict)

    @pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB needed for SQL tests")
    def test_sql_integration(self):
        """Test advanced SQL integration."""
        sllm = SpreadsheetLLM()
        
        # Create a df with specific values
        df = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'Revenue': [100, 200, 300]
        })
        
        # Query it
        result = sllm.query_sql(df, "SELECT Revenue FROM df WHERE Year > 2020")
        
        assert len(result) == 2
        assert 200 in result['Revenue'].values
        assert 300 in result['Revenue'].values
        assert 100 not in result['Revenue'].values

if __name__ == "__main__":
    pytest.main([__file__, "-v"])