# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [2.6.1] - 2025-12-13

### Added
- **Robust File Detection**: Implemented binary signature ("magic number") verification in `load_from_file` to correctly identify file types regardless of extension (fixes crashes on mislabeled files).
- **Legacy Excel Support**: Added `xlrd` dependency to support older `.xls` files.
- **CI/CD Pipeline**: Added `.github/workflows/tests.yml` to automate testing across multiple Python versions.

### Enhanced
- **SQL Security**: Completely rewrote `query_sql` to use isolated DuckDB connections and parameterized queries, preventing SQL injection risks.
- **Compression Algorithms**:
    - **Transition Anchors**: Updated `StructuralAnchorExtractor` to detect data type transitions (e.g., Header -> Body) ensuring context is preserved.
    - **2D Block Merging**: Updated `InvertedIndexTranslator` to merge rectangular regions (e.g., `A1:B2`) instead of just rows, significantly reducing token usage for dense tables.

### Fixed
- **Dependency Handling**: Fixed optional/missing dependency issues for `duckdb` and `xlrd`.
- **Production Stability**: Removed "research-grade" shortcuts in favor of robust error handling and type safety.

## [2.4.0] - 2025-12-09

### Added
- **Offline SQL Interface**: Integrated `duckdb` to allow standard SQL queries directly against spreadsheet data via `query_sql()`.
- **Deterministic Reasoning Engine**: Replaced LLM-based reasoning with a fuzzy keyword search engine using `thefuzz` in `ChainOfSpreadsheet`.
- **Structured JSON Export**: New `encode_to_json()` method that exports compressed data to machine-readable JSON, featuring a custom `NumpyEncoder` for numerical stability.
- **Advanced Data Classifiers**: Expanded `DataTypeClassifier` to detect PII (Emails, SSNs, Phone Numbers) and Business Entities (IBANs, Stock Tickers).

### Changed
- **Core Philosophy Pivot**: Transformed the project into a fully offline tool, removing dependencies on external LLM APIs and context window constraints.
- **Refactored Reasoning**: `ChainOfSpreadsheet` now uses `SmartTableDetector` and relevance scoring instead of LLM prompts.
- **Updated Tests**: Refactored test suite (`test_classifiers.py`, `test_core.py`) to validate deterministic outputs and new offline capabilities.

### Removed
- **LLM-Specific Features**: Removed `encode_to_token_limit` and other token-counting logic associated with LLM context windows.
- **Heuristic Detectors**: Deprecated/removed older heuristic table detectors in favor of `SmartTableDetector`.

### Fixed
- **JSON Serialization**: Implemented a robust `NumpyEncoder` to fix `TypeError` crashes when exporting `int64` or `float64` data types to JSON.

## [2.3.0] - 2025-08-19

### Added
- **Colorful CLI Output**: Enhanced CLI with rich color, tables, and progress bars for a more user-friendly experience (using the `rich` library).
- **Parallel Processing**: Added `--parallel` and `--jobs` options for fast multi-sheet workbook processing.
- **Benchmark & Visualization Script**: New script (`scripts/generate_benchmarks.py`) to benchmark compression, speed, and memory usage, and generate beautiful Seaborn charts for documentation.
- **Benchmarks Section in Docs**: Updated README with instructions and example output for running and visualizing benchmarks.

### Enhanced
- **CLI Usability**: Improved error messages, output formatting, and summary tables for both demo and normal CLI modes.
- **Documentation**: Expanded documentation to cover new CLI features, parallel processing, and benchmarking workflow.

### Fixed
- **CLI Argument Handling**: Improved validation and error reporting for missing or invalid arguments.
- **Visualization Output**: Fixed issues with saving and displaying generated charts.

## [2.0.0] - 2025-08-02

### Added
- **Enhanced Spreadsheet Analysis**: Improved structural analysis for better data representation
- **Expanded Visualization Tools**: More visualization options for compression analysis
- **Optimized Token Usage**: Further token reduction for more efficient processing
- **Multi-Sheet Analysis**: Better handling of workbooks with multiple sheets
- **Smart Table Detection**: Advanced algorithms for detecting and classifying table regions
- **Formula Extraction & Analysis**: Improved extraction and dependency analysis for Excel formulas

### Enhanced
- **Core Architecture**: Refactored core architecture for better maintainability and extensibility
- **Performance Optimizations**: Reduced memory usage and improved processing speed
- **CLI Interface**: Added new command options for enhanced functionality
- **Documentation**: Comprehensive documentation updates with more examples

### Fixed
- **Table Detection Edge Cases**: Fixed issues with complex table structures
- **Compression Ratio Calculation**: More accurate calculation for sparse spreadsheets
- **Memory Leaks**: Addressed potential memory issues with large spreadsheets

## [1.1.0] - 2025-07-30

### Added
- **Auto-Configuration**: New `auto_configure()` method that automatically optimizes compression settings based on spreadsheet characteristics (sparsity, size, data types)
- **Auto-Compress**: New `compress_with_auto_config()` method for one-step automatic optimization and compression
- **Multi-LLM Support**: Provider-specific output formats for ChatGPT, Claude, and Gemini via `encode_for_llm_provider()`
- **Enhanced CLI**: Support for `--auto-config`, `--format json`, `--verbose` flags with demo mode
- **Advanced Logging**: Optional detailed logging for debugging and monitoring compression operations
- **Enhanced Range Detection**: Improved `_merge_address_ranges()` method that creates ranges like `A1:A5` for contiguous cells
- **Contiguous Cell Grouping**: Enhanced `_group_contiguous_cells()` method for better data format aggregation
- **Comprehensive Test Suite**: Added `test_enhanced_features.py` with 6 new test cases
- **CSV Testing Tools**: Added demo scripts and test files for easy CSV testing
- **Format Comparison Tools**: Added utilities to compare different LLM output formats

### Enhanced
- **CLI Interface**: Can now combine `--demo` with `--vanilla`, `--auto-config`, `--format`, and `--stats` options
- **JSON Output**: Proper JSON serialization with numpy type handling
- **Error Handling**: Better error messages and validation
- **Code Coverage**: Increased test coverage to 34 passing tests

### Fixed
- **Range Detection**: Fixed incomplete implementation in address range merging
- **Cell Grouping**: Fixed placeholder implementation in contiguous cell detection
- **JSON Serialization**: Fixed numpy data type serialization issues
- **CLI Argument Handling**: Fixed argument parsing for combined flags

### Performance
- **Token Reduction**: Improved compression ratios, especially for sparse data (up to 5.5x reduction vs vanilla)
- **Format Optimization**: Provider-specific formats reduce token usage by 34-90% compared to general format
- **Auto-tuning**: Automatic parameter optimization based on data characteristics

### Documentation
- **README Updates**: Added documentation for all new features and CLI options
- **Format Guide**: Created comprehensive LLM format comparison documentation
- **Testing Guide**: Added CSV testing documentation and examples
- **API Reference**: Updated with new methods and parameters

## [1.0.0] - 2024-XX-XX

### Added
- Initial release of SheetWise
- Core compression framework with three modules
- Vanilla encoding methods
- Chain of Spreadsheet reasoning
- Basic CLI interface
- Comprehensive test suite