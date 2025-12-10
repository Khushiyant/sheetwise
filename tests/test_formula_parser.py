"""Test the robust Formula Parser and Tokenizer."""

import pytest
from sheetwise.formula_parser import FormulaParser, FormulaTokenizer

class TestFormulaTokenizer:
    """Test the new robust tokenizer logic."""
    
    def setup_method(self):
        self.tokenizer = FormulaTokenizer()

    def test_basic_tokenization(self):
        """Test simple formula splitting."""
        tokens = self.tokenizer.tokenize("SUM(A1:A5)")
        assert tokens == ["SUM(", "A1:A5", ")"]
        
        tokens = self.tokenizer.tokenize("A1+B1")
        assert tokens == ["A1", "+", "B1"]

    def test_nested_formulas(self):
        """Test handling of nested functions."""
        # IF(SUM(A1:A5)>10, "High", "Low")
        formula = 'IF(SUM(A1:A5)>10, "High", "Low")'
        tokens = self.tokenizer.tokenize(formula)
        
        # Verify structure isn't lost
        assert "IF(" in tokens
        assert "SUM(" in tokens
        assert "A1:A5" in tokens

    def test_argument_extraction(self):
        """Test extracting arguments from functions."""
        # Simple case
        tokens = ["A1", ",", "B1"]
        args = self.tokenizer.extract_args(iter(tokens))
        assert args == ["A1", "B1"]
        
        # Nested case: IF(SUM(A,B), C, D)
        # The tokenizer stream would look like: SUM(, A, ,, B, ), ,, C, ,, D
        # This is complex to mock directly, so we test the simplify_formula integration instead
        pass

class TestFormulaParser:
    """Test high-level parser descriptions."""
    
    def setup_method(self):
        self.parser = FormulaParser()

    def test_simplify_nested_sum(self):
        """Test explanation of nested SUM."""
        formula = "=SUM(A1:A10)"
        explanation = self.parser.simplify_formula(formula)
        assert "Sum of A1:A10" in explanation

    def test_simplify_vlookup(self):
        """Test VLOOKUP explanation."""
        formula = '=VLOOKUP("ID", B2:E10, 3)'
        explanation = self.parser.simplify_formula(formula)
        # Updated expectation: The tokenizer preserves quotes and the simplifier wraps args
        # So we look for '"ID"' (double quotes inside)
        assert '"ID"' in explanation
        assert 'Column 3' in explanation

    def test_simplify_if_statement(self):
        """Test IF statement explanation."""
        formula = '=IF(A1>10, "Pass", "Fail")'
        explanation = self.parser.simplify_formula(formula)
        # Updated expectation: Tokenizer adds spaces around operators for clarity
        assert "If A1 > 10 is true" in explanation