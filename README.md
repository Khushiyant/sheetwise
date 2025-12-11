# SheetWise
<img width="1000" height="500" alt="Gemini_Generated_Image_caqyklcaqyklcaqy" src="https://github.com/user-attachments/assets/9eba0c7f-6a01-4900-8085-d3bba1a6ca5a" />

A Python package for encoding spreadsheets for Large Language Models, implementing the SpreadsheetLLM research framework.

[![PyPI version](https://img.shields.io/pypi/v/sheetwise.svg)](https://pypi.org/project/sheetwise/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SheetWise is a Python package that implements the key components from Microsoft Research's SpreadsheetLLM paper for efficiently encoding spreadsheets for use with Large Language Models. The package provides:

- **SheetCompressor**: Efficient encoding framework with three compression modules
- **Chain of Spreadsheet**: Multi-step reasoning approach for spreadsheet analysis
- **Vanilla Encoding**: Traditional cell-by-cell encoding methods
- **Token Optimization**: Significant reduction in token usage
- **Formula Analysis**: Extract and simplify Excel formulas
- **Multi-Sheet Support**: Process entire workbooks with cross-sheet references
- **Visualization Tools**: Generate visual reports of compression results

## Key Features

- **Intelligent Compression**: Up to 96% reduction in token usage while preserving semantic information
- **Auto-Configuration**: Automatically optimizes compression settings based on spreadsheet characteristics  
- **Multi-Table Support**: Handles complex spreadsheets with multiple tables and regions
- **Structural Analysis**: Identifies and preserves important structural elements
- **Format-Aware**: Preserves data type and formatting information
- **Enhanced Algorithms**: Improved range detection and contiguous cell grouping
- **Easy Integration**: Simple API for immediate use

### Token Budgeting

Unsure if your spreadsheet fits in the context window? Use `encode_to_token_limit`:

```python
from sheetwise import SpreadsheetLLM
import pandas as pd

df = pd.read_excel("large_file.xlsx")
sllm = SpreadsheetLLM()

# Automatically adjust compression to fit within 4000 tokens
encoded_text = sllm.encode_to_token_limit(df, max_tokens=4000)
```

### Memory Optimization (Large Files)

For very large spreadsheets (100MB+), you can use `inplace=True` to significantly reduce RAM usage during compression.

```python
# Standard Compression (Creates a copy in RAM)
result = compressor.compress(df)

# Memory-Efficient Compression (Modifies DataFrame in-place or minimizes copies)
# Use this when working with datasets close to your RAM limit.
result = compressor.compress(df, inplace=True)
```

## Installation

### Using pip

```bash
pip install sheetwise
```

### Using Poetry

```bash
poetry add sheetwise
```

### Development Installation

```bash
git clone https://github.com/yourusername/sheetwise.git
cd sheetwise
poetry install
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from sheetwise import SpreadsheetLLM

# Initialize the framework
sllm = SpreadsheetLLM()

# Load your spreadsheet
df = pd.read_excel("your_spreadsheet.xlsx")

# Compress and encode for LLM use
llm_ready_text = sllm.compress_and_encode_for_llm(df)

# Copy and paste this text directly into ChatGPT/Claude
print(llm_ready_text)
```

### Advanced Usage

```python
from sheetwise import SpreadsheetLLM, SheetCompressor

# Auto-configuration
sllm = SpreadsheetLLM(enable_logging=True)
auto_compressed = sllm.compress_with_auto_config(df)  # Automatically optimizes settings

# Manual configuration
compressor = SheetCompressor(
    k=2,  # Structural anchor neighborhood size
    use_extraction=True,
    use_translation=True, 
    use_aggregation=True
)

# Compress the spreadsheet
compressed_result = compressor.compress(df)
print(f"Compression ratio: {compressed_result['compression_ratio']:.1f}x")
print(f"Compressed shape: {compressed_result['compressed_df'].shape}")

# Or use with SpreadsheetLLM for full pipeline
sllm = SpreadsheetLLM(compression_params={
    'k': 2,
    'use_extraction': True,
    'use_translation': True, 
    'use_aggregation': True
})

# Get detailed statistics
stats = sllm.get_encoding_stats(df)
print(f"Token reduction: {stats['token_reduction_ratio']:.1f}x")

# Process QA queries
result = sllm.process_qa_query(df, "What was the total revenue in 2023?")
```

### Enhanced Features Usage (v2.0+)

```python
from sheetwise import (
    SpreadsheetLLM, 
    FormulaParser, 
    WorkbookManager, 
    CompressionVisualizer, 
    SmartTableDetector
)

# Formula extraction and analysis
formula_parser = FormulaParser()
formulas = formula_parser.extract_formulas("your_spreadsheet.xlsx")
formula_parser.build_dependency_graph()
impact = formula_parser.get_formula_impact("Sheet1!A1")
formula_text = formula_parser.encode_formulas_for_llm()

# Multi-sheet support
workbook = WorkbookManager()
sheets = workbook.load_workbook("your_workbook.xlsx")
cross_refs = workbook.detect_cross_sheet_references()
sllm = SpreadsheetLLM()
compressed = workbook.compress_workbook(sllm.compressor)
encoded = workbook.encode_workbook_for_llm(compressed)

# Visualization
visualizer = CompressionVisualizer()
df = sllm.load_from_file("your_spreadsheet.xlsx")
compressed_result = sllm.compress_spreadsheet(df)
fig = visualizer.create_data_density_heatmap(df)
fig.savefig("heatmap.png")
html_report = visualizer.generate_html_report(df, compressed_result)

# Advanced table detection
detector = SmartTableDetector()
tables = detector.detect_tables(df)
extracted_tables = detector.extract_tables_to_dataframes(df)
```

### Command Line Interface

```bash
# Basic usage
sheetwise input.xlsx -o output.txt --stats

# Auto-configure compression
sheetwise input.xlsx --auto-config --verbose

# Run demo with sample data
sheetwise --demo --auto-config

# Use vanilla encoding instead of compression
sheetwise input.xlsx --vanilla

# Output in JSON format
sheetwise input.xlsx --format json
```

### Enhanced CLI Features (v2.0+)

```bash
# Extract and analyze formulas
sheetwise your_spreadsheet.xlsx --extract-formulas

# Process all sheets in a workbook
sheetwise your_workbook.xlsx --multi-sheet

# Generate visualizations
sheetwise your_spreadsheet.xlsx --visualize

# Detect and extract tables
sheetwise your_spreadsheet.xlsx --detect-tables

# Generate an HTML report
sheetwise your_spreadsheet.xlsx --format html
```

## Benchmarks & Visualization

SheetWise includes a benchmarking script to evaluate compression, speed, and memory usage across spreadsheets. This helps you understand performance and compare results visually.

### Running Benchmarks

1. Place your sample spreadsheets in `benchmarks/samples/` (supports .xlsx and .csv).
2. Run the benchmark script:

```bash
python scripts/generate_benchmarks.py
```

3. Results and charts will be saved in `benchmarks/results/` and `benchmarks/charts/`.


---

## Core Components

### 1. SheetCompressor

The main compression framework with three modules:

- **Structural Anchor Extraction**: Identifies and preserves structurally important rows/columns
- **Inverted Index Translation**: Creates efficient value-to-location mappings
- **Data Format Aggregation**: Groups cells by data type and format

### 2. Chain of Spreadsheet

Multi-step reasoning approach:

1. **Table Identification**: Automatically detects table regions
2. **Compression**: Applies SheetCompressor to reduce size
3. **Query Processing**: Identifies relevant regions for specific queries

### 3. Enhanced Modules (v2.0+)

- **FormulaParser**: Extracts and analyzes Excel formulas
- **WorkbookManager**: Handles multi-sheet workbooks and cross-references
- **CompressionVisualizer**: Generates visualizations and reports
- **SmartTableDetector**: Advanced table detection and classification

## Examples

### Working with Financial Data

```python
from sheetwise import SpreadsheetLLM
from sheetwise.utils import create_realistic_spreadsheet

# Create sample financial spreadsheet
df = create_realistic_spreadsheet()

sllm = SpreadsheetLLM()

# Analyze the data
stats = sllm.get_encoding_stats(df)
print(f"Original size: {stats['original_shape']}")
print(f"Sparsity: {stats['sparsity_percentage']:.1f}% empty cells")
print(f"Compression: {stats['compression_ratio']:.1f}x smaller")

# Generate LLM-ready output
encoded = sllm.compress_and_encode_for_llm(df)
print("\nReady for LLM:")
print(encoded[:300] + "...")
```

### Visualizing Compression

```python
from sheetwise import SpreadsheetLLM, CompressionVisualizer
import pandas as pd

# Load your data
df = pd.read_excel("complex_spreadsheet.xlsx")

# Compress the data
sllm = SpreadsheetLLM()
compressed_result = sllm.compress_spreadsheet(df)

# Create visualizations
visualizer = CompressionVisualizer()

# Generate heatmap of data density
fig1 = visualizer.create_data_density_heatmap(df)
fig1.savefig("density_heatmap.png")

# Compare original vs compressed
fig2 = visualizer.compare_original_vs_compressed(df, compressed_result)
fig2.savefig("compression_comparison.png")

# Generate HTML report with all visualizations
html_report = visualizer.generate_html_report(df, compressed_result)
with open("compression_report.html", "w") as f:
    f.write(html_report)

# Compare different compression strategies
configs = [
    {"name": "Extraction Only", "use_translation": False, "use_aggregation": False},
    {"name": "Translation Only", "use_extraction": False, "use_aggregation": False}, 
    {"name": "All Modules", "use_extraction": True, "use_translation": True, "use_aggregation": True}
]

for config in configs:
    compressor = SheetCompressor(**{k: v for k, v in config.items() if k != "name"})
    result = compressor.compress(df)
    print(f"{config['name']}: {result['compression_ratio']:.1f}x compression")
```



## Performance

SpreadsheetLLM achieves significant improvements over vanilla encoding:

| Metric | Vanilla | SpreadsheetLLM | Improvement |
|--------|---------|----------------|-------------|
| Token Count | ~25,000 | ~1,200 | **96% reduction** |
| Sparsity Handling | Poor | Excellent | **Removes empty regions** |
| Multi-Table Support | Limited | Native | **Preserves structure** |
| Format Preservation | Basic | Advanced | **Type-aware grouping** |

## API Reference

### SpreadsheetLLM Class

The main interface for the framework.

#### Methods

- `compress_and_encode_for_llm(df)`: One-step compression and encoding
- `compress_spreadsheet(df)`: Apply compression pipeline  
- `encode_vanilla(df)`: Traditional encoding
- `get_encoding_stats(df)`: Detailed compression statistics
- `process_qa_query(df, query)`: Chain of Spreadsheet reasoning
- `load_from_file(filepath)`: Load spreadsheet from file

### SheetCompressor Class

Core compression framework.

#### Parameters

- `k`: Structural anchor neighborhood size (default: 4)
- `use_extraction`: Enable structural extraction (default: True)
- `use_translation`: Enable inverted index translation (default: True)
- `use_aggregation`: Enable format aggregation (default: True)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository

git clone https://github.com/yourusername/sheetwise.git
cd sheetwise

# Install development dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run black src tests
poetry run isort src tests
poetry run flake8 src tests
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/test_core.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SpreadsheetLLM in your research, please cite:

```bibtex
@article{spreadsheetllm2024,
  title={SpreadsheetLLM: Encoding Spreadsheets for Large Language Models},
  author={Microsoft Research Team},
  journal={arXiv preprint},
  year={2024}
}
```


## Support

- [Documentation](https://sheetwise.readthedocs.io)
- [Issue Tracker](https://github.com/yourusername/sheetwise/issues)
- [Discussions](https://github.com/yourusername/sheetwise/discussions)
