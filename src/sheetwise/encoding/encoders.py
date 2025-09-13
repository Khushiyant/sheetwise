"""Encoding utilities for spreadsheet data."""
from abc import ABC, abstractmethod
import pandas as pd
import json
import re

class Encoder(ABC):
    """Base class for implementing different Encoders"""
    
    @abstractmethod
    def encode():
        pass

    def _to_excel_address(self, row: int, col: int) -> str:
        """Convert row, column indices to Excel address"""
        col_letter = ""
        col_num = col + 1
        while col_num > 0:
            col_num -= 1
            col_letter = chr(col_num % 26 + ord("A")) + col_letter
            col_num //= 26
        return f"{col_letter}{row + 1}"
    
        

class VanillaEncoder(Encoder):
    """Spreadsheet encoding to Markdown-like format with cell addresses and formats"""

    def encode(self, df: pd.DataFrame, include_format: bool = False) -> str:
        """
        Encode spreadsheet

        Args:
            df: Input DataFrame
            include_format: Whether to include format information

        Returns:
            Markdown-style string representation
        """
        lines = []

        for i, row in df.iterrows():
            row_parts = []
            for j, col in enumerate(df.columns):
                cell_value = row[col]
                cell_addr = self._to_excel_address(i, j)

                if pd.isna(cell_value) or cell_value == "":
                    cell_repr = f"{cell_addr}, "
                else:
                    cell_repr = f"{cell_addr},{cell_value}"

                row_parts.append(cell_repr)

            lines.append("|".join(row_parts))

        return "\n".join(lines)
    
    def estimate_tokens(self,encoded_data: str) -> int:
        """Markdown Specific token estimation"""
        return len(encoded_data.split("|")) # Each cell as a token

class JSONEncoder(Encoder):
    """Encoder for JSON format with JSON-specific token estimation"""
    
    def encode(self, df: pd.DataFrame) -> str:
        """Encode DataFrame to JSON format"""
        data = {
            "columns": list(df.columns),
            "data": df.where(pd.notna(df), None).values.tolist(),
            "dimensions": {
                "rows": len(df),
                "columns": len(df.columns)
            }
        }
        return json.dumps(data, indent=2)
    
    def estimate_tokens(self, encoded_data: str) -> int:
        """JSON-specific token estimation"""
    
        # Count structural tokens (each JSON symbol is typically a token)
        structural_tokens = (
            encoded_data.count('{') + encoded_data.count('}') +
            encoded_data.count('[') + encoded_data.count(']') +
            encoded_data.count(':') + encoded_data.count(',') +
            encoded_data.count('"') * 2  # Opening and closing quotes
        )
        
        # Count content tokens (approximate)
        content_str = re.sub(r'[{}[\]":,]', ' ', encoded_data)
        content_tokens = len(content_str.split())
        
        return structural_tokens + content_tokens
        
        
        


    
