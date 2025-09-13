"""Test utility functions."""

import pandas as pd
import pytest

from sheetwise.utils.utils import create_realistic_spreadsheet


class TestUtils:
    """Test cases for utility functions."""

    def test_create_realistic_spreadsheet(self):
        """Test creation of realistic spreadsheet."""
        df = create_realistic_spreadsheet()

        # Check basic properties
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 30)

        # Check that it's sparse (mostly empty)
        non_empty_cells = df.map(lambda x: x != "" and pd.notna(x)).sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        sparsity = (total_cells - non_empty_cells) / total_cells

        assert sparsity > 0.5  # Should be more than 50% empty

        # Check for expected content
        assert "Q4 Financial Report" in df.values.flatten()
        assert "Revenue ($M)" in df.values.flatten()
        assert "INVENTORY ANALYSIS" in df.values.flatten()

        # Check for realistic data types
        values = df.values.flatten()
        values = [v for v in values if v != "" and pd.notna(v)]

        # Should contain various data types
        has_text = any(
            isinstance(v, str)
            and not str(v).replace(".", "").replace("-", "").isdigit()
            for v in values
        )
        has_numbers = any(isinstance(v, (int, float)) for v in values)

        assert has_text
        assert has_numbers
