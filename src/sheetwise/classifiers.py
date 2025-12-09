"""Data type classification utilities for spreadsheet cells."""

import re
from typing import Any

import pandas as pd


class DataTypeClassifier:
    """Rule-based classifier for identifying data types, PII, and business entities."""

    @staticmethod
    def classify_cell_type(value: Any) -> str:
        """Classify cell value into predefined data types."""
        if pd.isna(value) or value == "" or value is None:
            return "Empty"

        str_value = str(value).strip()

        # Check if value is effectively empty after stripping
        if not str_value:
            return "Empty"

        # --- Business & PII Logic (High Priority) ---

        # Email
        if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", str_value):
            return "PII_Email"

        # US Phone Number (Loose match)
        if re.match(r"^(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}$", str_value):
            return "PII_Phone"

        # SSN (Simple US format)
        if re.match(r"^\d{3}-\d{2}-\d{4}$", str_value):
            return "PII_SSN"

        # IBAN (International Bank Account Number) - Generic structure
        if re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$", str_value):
            return "Biz_IBAN"

        # Ticker Symbol (e.g., $AAPL, AAPL) - Heuristic: All caps, 2-5 chars
        if re.match(r"^\$?[A-Z]{2,5}$", str_value) and not re.match(r"^\d+$", str_value):
            return "Biz_Ticker"

        # --- Standard Types (Lower Priority) ---

        # Year pattern
        if re.match(r"^\d{4}$", str_value) and 1900 <= int(str_value) <= 2100:
            return "Year"

        # Scientific notation
        if re.match(r"^-?\d+\.?\d*[eE][+-]?\d+$", str_value):
            return "Scientific"

        # Integer
        try:
            int(str_value.replace(",", ""))
            if "." not in str_value:
                return "Integer"
        except ValueError:
            pass

        # Float
        try:
            float(str_value.replace(",", ""))
            return "Float"
        except ValueError:
            pass

        # Percentage
        if str_value.endswith("%"):
            return "Percentage"

        # Date patterns
        date_patterns = [
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",
            r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$",
            r"^\d{1,2}-\w{3}-\d{4}$",
        ]
        for pattern in date_patterns:
            if re.match(pattern, str_value):
                return "Date"

        # Time patterns
        if re.match(r"^\d{1,2}:\d{2}(:\d{2})?(\s?(AM|PM))?$", str_value, re.IGNORECASE):
            return "Time"

        # Currency
        currency_symbols = ["$", "€", "£", "¥", "₹"]
        if any(symbol in str_value for symbol in currency_symbols):
            return "Currency"

        return "String"