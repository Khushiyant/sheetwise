"""Test the data type classifiers (Offline Edition)."""

import pytest
from sheetwise.classifiers import DataTypeClassifier


class TestDataTypeClassifier:
    """Test cases for the DataTypeClassifier class."""

    def test_classify_empty_values(self):
        """Test classification of empty values."""
        classifier = DataTypeClassifier()
        empty_values = [None, "", " ", float("nan")]
        for value in empty_values:
            result = classifier.classify_cell_type(value)
            assert result == "Empty"

    def test_classify_year(self):
        """Test year classification."""
        classifier = DataTypeClassifier()
        year_values = ["2023", "1999", "2000", "2024"]
        for value in year_values:
            result = classifier.classify_cell_type(value)
            assert result == "Year"

    def test_classify_integer(self):
        """Test integer classification."""
        classifier = DataTypeClassifier()
        integer_values = ["123", "0", "-456", "1,000", "999,999"]
        for value in integer_values:
            result = classifier.classify_cell_type(value)
            assert result == "Integer"

    def test_classify_float(self):
        """Test float classification."""
        classifier = DataTypeClassifier()
        float_values = ["123.45", "0.0", "-456.78", "1,000.50"]
        for value in float_values:
            result = classifier.classify_cell_type(value)
            assert result == "Float"

    def test_classify_pii_email(self):
        """Test PII Email classification."""
        classifier = DataTypeClassifier()
        email_values = [
            "test@example.com",
            "user.name@domain.co.uk",
            "info@company.org",
        ]
        for value in email_values:
            result = classifier.classify_cell_type(value)
            # Offline Edition returns PII_Email, not just Email
            assert result == "PII_Email"

    def test_classify_business_types(self):
        """Test new business logic classifiers."""
        classifier = DataTypeClassifier()
        
        # IBAN
        assert classifier.classify_cell_type("DE89370400440532013000") == "Biz_IBAN"
        
        # Ticker
        assert classifier.classify_cell_type("$AAPL") == "Biz_Ticker"
        assert classifier.classify_cell_type("MSFT") == "Biz_Ticker"
        
        # SSN
        assert classifier.classify_cell_type("123-45-6789") == "PII_SSN"

    def test_classify_default_string(self):
        """Test default string fallback."""
        classifier = DataTypeClassifier()
        other_values = ["Hello World", "Product Name", "ABC123", "Text Data"]
        for value in other_values:
            result = classifier.classify_cell_type(value)
            # Offline Edition defaults to "String", not "Others"
            assert result == "String"