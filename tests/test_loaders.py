import pytest
import pandas as pd
import json
from sheetwise import SheetwiseLoader

def test_loader_csv(tmp_path, sample_dataframe):
    # Setup
    csv_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    
    # Test
    loader = SheetwiseLoader(csv_path)
    docs = loader.load()
    
    # Verify
    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(csv_path)
    # Check for SheetWise Markdown signature or content
    assert "# Data (Compressed" in docs[0].page_content or "Values:" in docs[0].page_content

def test_loader_excel_multiple_sheets(tmp_path, sample_dataframe, financial_dataframe):
    # Setup
    xlsx_path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        sample_dataframe.to_excel(writer, sheet_name="Sample", index=False)
        financial_dataframe.to_excel(writer, sheet_name="Financial", index=False)
        
    # Test default mode (one doc per sheet)
    loader = SheetwiseLoader(xlsx_path, mode="sheets")
    docs = loader.load()
    
    assert len(docs) == 2
    sheet_names = {d.metadata["sheet_name"] for d in docs}
    assert sheet_names == {"Sample", "Financial"}
    
    # Test single mode
    loader_single = SheetwiseLoader(xlsx_path, mode="single")
    docs_single = loader_single.load()
    
    assert len(docs_single) == 1
    assert "Sample" in docs_single[0].metadata["sheets"]
    assert "Financial" in docs_single[0].metadata["sheets"]
    assert "# Sheet: Sample" in docs_single[0].page_content
    assert "# Sheet: Financial" in docs_single[0].page_content

def test_loader_json_layout(tmp_path, sample_dataframe):
    csv_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    
    loader = SheetwiseLoader(csv_path, layout="json")
    docs = loader.load()
    
    content = json.loads(docs[0].page_content)
    assert "metadata" in content
    assert "cell_index" in content
