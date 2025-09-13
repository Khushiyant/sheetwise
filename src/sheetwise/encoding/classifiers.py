"""Data type classification utilities for spreadsheet cells."""

import re
from typing import Any

import pandas as pd


class DataTypeClassifier:
    """Rule-based classifier for identifying data types in spreadsheet cells"""

    @staticmethod
    def classify_cell_type(value: Any) -> str:
        """Classify cell value into predefined data types"""
        if pd.isna(value) or value == "" or value is None:
            return "Empty"

        str_value = str(value).strip()

        # Check if value is effectively empty after stripping
        if not str_value:
            return "Empty"

        # Year pattern
        if re.match(r"^\d{4}$", str_value) and 1900 <= int(str_value) <= 2100:
            return "Year"

        # Scientific notation (check before float to catch scientific notation)
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

        # Email
        if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", str_value):
            return "Email"

        return "Others"
