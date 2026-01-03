"""Test the CompressionVisualizer class."""

import pytest
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sheetwise import CompressionVisualizer, SpreadsheetLLM

# Use non-interactive backend for testing
matplotlib.use('Agg')

class TestVisualizer:
    """Test cases for CompressionVisualizer."""

    @pytest.fixture
    def visualizer(self):
        return CompressionVisualizer()

    def test_create_data_density_heatmap(self, visualizer, sample_dataframe):
        """Test heatmap creation."""
        fig = visualizer.create_data_density_heatmap(sample_dataframe)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_original_vs_compressed(self, visualizer, sample_dataframe):
        """Test comparison plot."""
        sllm = SpreadsheetLLM()
        compressed = sllm.compress_spreadsheet(sample_dataframe)
        
        fig = visualizer.compare_original_vs_compressed(sample_dataframe, compressed)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_generate_html_report(self, visualizer, sample_dataframe):
        """Test HTML report generation."""
        sllm = SpreadsheetLLM()
        compressed = sllm.compress_spreadsheet(sample_dataframe)
        
        html = visualizer.generate_html_report(sample_dataframe, compressed)
        
        assert isinstance(html, str)
        # Check for deprecation message or new behavior if changed
        assert "Use generate_interactive_report" in html

    def test_generate_interactive_report(self, visualizer, sample_dataframe, tmp_path):
        """Test interactive report file creation."""
        sllm = SpreadsheetLLM()
        compressed = sllm.compress_spreadsheet(sample_dataframe)
        
        report_path = tmp_path / "report.html"
        visualizer.generate_interactive_report(sample_dataframe, compressed, str(report_path))
        
        assert report_path.exists()
        content = report_path.read_text()
        assert "<html" in content
