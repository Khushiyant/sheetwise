from typing import Any, Dict, Iterator, List, Optional, Union
import json
from pathlib import Path
import pandas as pd

from .core import SpreadsheetLLM
from .workbook import WorkbookManager

class SheetwiseLoader:
    """
    Loader for spreadsheets using SheetWise.
    Compatible with LangChain's BaseLoader interface.
    """

    def __init__(
        self, 
        file_path: Union[str, Path], 
        mode: str = "sheets",
        layout: str = "markdown",
        compression_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the loader.

        Args:
            file_path: Path to the spreadsheet file.
            mode: Loading mode.
                  - "sheets": One document per sheet (default).
                  - "single": All sheets merged into one document.
            layout: Output format.
                    - "markdown": Optimized Markdown string (SpreadsheetLLM style).
                    - "json": Structured JSON string.
            compression_params: Parameters for the SheetCompressor.
        """
        self.file_path = str(file_path)
        self.mode = mode
        self.layout = layout
        self.compression_params = compression_params or {}
        self.sheet_llm = SpreadsheetLLM(compression_params=self.compression_params)
        self.workbook_manager = WorkbookManager()

    def lazy_load(self) -> Iterator["Document"]:
        """A lazy loader for Documents."""
        # Try to import Document from langchain_core, fallback to simple class
        try:
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain.schema import Document
            except ImportError:
                class Document:
                    def __init__(self, page_content: str, metadata: dict):
                        self.page_content = page_content
                        self.metadata = metadata
                    def __repr__(self):
                        return f"Document(page_content='{self.page_content[:20]}...', metadata={self.metadata})"
                    def __eq__(self, other):
                        if not isinstance(other, Document):
                            return False
                        return self.page_content == other.page_content and self.metadata == other.metadata

        # Attempt to load as a workbook (Excel)
        try:
             sheets = self.workbook_manager.load_workbook(self.file_path)
        except Exception:
             # Fallback for CSV/TSV/Legacy XLS or if load_workbook fails
             try:
                 # SpreadsheetLLM's load_from_file handles CSV/TSV and legacy XLS
                 df = self.sheet_llm.load_from_file(self.file_path)
                 sheets = {"Sheet1": df}
             except Exception as e:
                 raise ValueError(f"Could not load file {self.file_path}: {e}")

        
        # Process sheets
        if self.mode == "single":
            # Combine all content
            full_content = []
            combined_metadata = {
                "source": self.file_path, 
                "sheets": list(sheets.keys()),
                "total_sheets": len(sheets)
            }
            
            for sheet_name, df in sheets.items():
                content = self._process_sheet(df)
                if len(sheets) > 1:
                     full_content.append(f"# Sheet: {sheet_name}\n{content}")
                else:
                     full_content.append(content)
            
            yield Document(page_content="\n\n".join(full_content), metadata=combined_metadata)

        else: # "sheets"
            for sheet_name, df in sheets.items():
                content = self._process_sheet(df)
                metadata = {"source": self.file_path, "sheet_name": sheet_name}
                yield Document(page_content=content, metadata=metadata)

    def load(self) -> List["Document"]:
        """Load data into Documents."""
        return list(self.lazy_load())

    def _process_sheet(self, df: pd.DataFrame) -> str:
        if self.layout == "json":
            return self.sheet_llm.encode_to_json(df)
        else:
            # Markdown
            return self.sheet_llm.compress_and_encode_for_llm(df)
