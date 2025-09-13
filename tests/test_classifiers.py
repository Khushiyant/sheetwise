"""Test the data type classifiers."""

import pytest

from sheetwise.encoding.classifiers import DataTypeClassifier


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

        # Test invalid years
        invalid_years = ["1800", "3000", "23", "202"]
        for value in invalid_years:
            result = classifier.classify_cell_type(value)
            assert result != "Year"

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

    def test_classify_percentage(self):
        """Test percentage classification."""
        classifier = DataTypeClassifier()

        percentage_values = ["50%", "100%", "0.5%", "-25%"]
        for value in percentage_values:
            result = classifier.classify_cell_type(value)
            assert result == "Percentage"

    def test_classify_scientific(self):
        """Test scientific notation classification."""
        classifier = DataTypeClassifier()

        scientific_values = ["1.23e10", "4.56E-3", "-7.89e+5", "2E10"]
        for value in scientific_values:
            result = classifier.classify_cell_type(value)
            assert result == "Scientific"

    def test_classify_date(self):
        """Test date classification."""
        classifier = DataTypeClassifier()

        date_values = [
            "2023-01-15",
            "01/15/2023",
            "15/01/2023",
            "1-Jan-2023",
            "15-Dec-2023",
        ]
        for value in date_values:
            result = classifier.classify_cell_type(value)
            assert result == "Date"

    def test_classify_time(self):
        """Test time classification."""
        classifier = DataTypeClassifier()

        time_values = ["12:30", "09:15:30", "3:45 PM", "11:59 AM"]
        for value in time_values:
            result = classifier.classify_cell_type(value)
            assert result == "Time"

    def test_classify_currency(self):
        """Test currency classification."""
        classifier = DataTypeClassifier()

        currency_values = ["$100", "€50", "£25", "¥1000", "₹500"]
        for value in currency_values:
            result = classifier.classify_cell_type(value)
            assert result == "Currency"

    def test_classify_email(self):
        """Test email classification."""
        classifier = DataTypeClassifier()

        email_values = [
            "test@example.com",
            "user.name@domain.co.uk",
            "info@company.org",
        ]
        for value in email_values:
            result = classifier.classify_cell_type(value)
            assert result == "Email"

        # Test invalid emails
        invalid_emails = ["notanemail", "@domain.com", "user@"]
        for value in invalid_emails:
            result = classifier.classify_cell_type(value)
            assert result != "Email"

    def test_classify_others(self):
        """Test classification of other text values."""
        classifier = DataTypeClassifier()

        other_values = ["Hello World", "Product Name", "ABC123", "Text Data"]
        for value in other_values:
            result = classifier.classify_cell_type(value)
            assert result == "Others"
