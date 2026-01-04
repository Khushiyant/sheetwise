"""
Privacy and PII redaction utilities.
"""

import re
from typing import Any, Dict, List, Optional, Union
import pandas as pd

class PIIRedactor:
    """
    Redacts Personally Identifiable Information (PII) from dataframes.
    Uses regex patterns to identify and mask sensitive data.
    """

    # Common Regex Patterns
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        "ipv6": r'([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
    }

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None, mask: str = "[REDACTED]"):
        """
        Initialize the redactor.

        Args:
            custom_patterns: Dictionary of {name: regex_pattern} to add/override.
            mask: String to replace PII with. Default is "[REDACTED]".
        """
        self.patterns = self.PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        self.mask = mask
        # Pre-compile regexes for performance
        self._regex_map = {
            name: re.compile(pattern, flags=re.IGNORECASE) 
            for name, pattern in self.patterns.items()
        }

    def redact(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Redact PII from the dataframe.

        Args:
            df: Input dataframe.
            inplace: Whether to modify the dataframe in place.

        Returns:
            Redacted dataframe.
        """
        if not inplace:
            df = df.copy()

        # Iterate over object/string columns only to save time
        object_cols = df.select_dtypes(include=['object', 'string']).columns
        
        for col in object_cols:
            df[col] = df[col].apply(self._redact_value)
            
        return df

    def _redact_value(self, value: Any) -> Any:
        """Helper to redact a single value."""
        if not isinstance(value, str):
            return value
        
        # Apply all patterns
        redacted_value = value
        for name, regex in self._regex_map.items():
            if regex.search(redacted_value):
                # Replace the match with the mask
                # If we want to preserve the type (e.g. [EMAIL]), we could use name
                # For now, simple masking
                replacement = f"[{name.upper()}]" if self.mask == "[TYPE]" else self.mask
                redacted_value = regex.sub(replacement, redacted_value)
                
        return redacted_value

    def add_pattern(self, name: str, pattern: str):
        """Add a new pattern dynamically."""
        self.patterns[name] = pattern
        self._regex_map[name] = re.compile(pattern, flags=re.IGNORECASE)
