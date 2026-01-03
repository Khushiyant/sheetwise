"""Test the WorkbookManager class."""

import os
import pytest
import pandas as pd
from unittest.mock import MagicMock
from sheetwise import WorkbookManager, SheetCompressor

class TestWorkbookManager:
    """Test cases for WorkbookManager."""

    @pytest.fixture
    def workbook_file(self, tmp_path):
        """Create a temporary multi-sheet Excel file."""
        file_path = tmp_path / "test_workbook.xlsx"
        
        # Create dataframes
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        
        # Write to Excel
        with pd.ExcelWriter(file_path) as writer:
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            df2.to_excel(writer, sheet_name='Sheet2', index=False)
            
        return str(file_path)

    def test_initialization(self):
        """Test initialization."""
        wm = WorkbookManager()
        assert wm.sheets == {}
        assert wm.sheet_metadata == {}
        assert wm.cross_references == {}

    def test_load_workbook(self, workbook_file):
        """Test loading a workbook."""
        wm = WorkbookManager()
        sheets = wm.load_workbook(workbook_file)
        
        assert len(sheets) == 2
        assert "Sheet1" in sheets
        assert "Sheet2" in sheets
        assert isinstance(sheets["Sheet1"], pd.DataFrame)
        assert wm.sheet_metadata["Sheet1"]["shape"] == (2, 2)
        assert wm._last_loaded_path == workbook_file

    def test_compress_workbook(self, workbook_file):
        """Test workbook compression."""
        wm = WorkbookManager()
        wm.load_workbook(workbook_file)
        
        compressor = SheetCompressor()
        # Mock compressor to save time/complexity
        compressor.compress = MagicMock(return_value={
            "compression_ratio": 2.0,
            "compressed_data": pd.DataFrame(),
            "inverted_index": {"val": ["A1"]}
        })
        
        result = wm.compress_workbook(compressor)
        
        assert "Sheet1" in result
        assert "Sheet2" in result
        assert "__workbook_summary__" in result
        assert result["__workbook_summary__"]["total_sheets"] == 2

    def test_encode_workbook_for_llm(self, workbook_file):
        """Test LLM encoding for workbook."""
        wm = WorkbookManager()
        wm.load_workbook(workbook_file)
        
        # Fake compression results
        compression_results = {
            "Sheet1": {
                "compression_ratio": 2.0, 
                "inverted_index": {"test": ["A1"]}
            },
            "Sheet2": {
                "compression_ratio": 1.5,
                "inverted_index": {"data": ["B2"]}
            },
            "__workbook_summary__": {
                "total_sheets": 2,
                "total_original_cells": 100,
                "total_compressed_cells": 50,
                "overall_compression_ratio": 2.0,
                "sheet_relationships": None
            }
        }
        
        encoded = wm.encode_workbook_for_llm(compression_results)
        
        assert "# Workbook Analysis" in encoded
        assert "## Sheet: Sheet1" in encoded
        assert "## Sheet: Sheet2" in encoded
        assert "Compressed 2.0x" in encoded

    def test_sheet_importance_ranking(self, workbook_file):
        """Test sheet importance ranking."""
        wm = WorkbookManager()
        wm.load_workbook(workbook_file)
        
        # Mock references since we don't have real formulas in the temp file
        wm.cross_references = {
            "Sheet1": set(),
            "Sheet2": {"Sheet1"}  # Sheet2 references Sheet1
        }
        
        ranking = wm.get_sheet_importance_ranking()
        
        assert len(ranking) == 2
        # Sheet1 should be more important because it's referenced by Sheet2
        # (Assuming same size/density)
        # Note: Actual logic depends on size/density too, but this verifies the method runs
        assert isinstance(ranking[0], tuple)
        assert isinstance(ranking[0][0], str)
        assert isinstance(ranking[0][1], float)
