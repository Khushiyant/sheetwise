# CHANGELOG


## v2.6.2 (2025-12-14)

### Bug Fixes

- Add interactive HTML report generation for spreadsheet compression analysis
  ([`0831171`](https://github.com/Khushiyant/sheetwise/commit/0831171e485a519c3efc60fba53472d0e166ac25))

### Chores

- Update CHANGELOG for version 2.6.1 with new features, enhancements, and fixes
  ([`b6717af`](https://github.com/Khushiyant/sheetwise/commit/b6717afdb6d53bd6b57d947e77c8825274261ad5))

### Continuous Integration

- Add release workflow
  ([`3fff10b`](https://github.com/Khushiyant/sheetwise/commit/3fff10bb9c1c379524c6df4c90282b121d56ff5d))


## v2.6.1 (2025-12-13)

### Bug Fixes

- Add tomli dependency for improved TOML file handling in development
  ([`309ace4`](https://github.com/Khushiyant/sheetwise/commit/309ace492762bd766310e764c6829a6572ff6e6e))

- Change data loading method from Excel to CSV for consistency
  ([`fd5bacb`](https://github.com/Khushiyant/sheetwise/commit/fd5bacb5a460b274f8df743fecd44cc5904d6a6d))

- Remove tomli dependency and revert version reading in Sphinx config
  ([`1f425d2`](https://github.com/Khushiyant/sheetwise/commit/1f425d21bb062146b4d86a76f17be3d4f27b34f5))

- Update documentation workflow to install specific dependencies for API generation
  ([`4600f18`](https://github.com/Khushiyant/sheetwise/commit/4600f18bf1b3e426b787571f0655e1cf4be08d69))

- Update links and author information in documentation files
  ([`8a6dd0c`](https://github.com/Khushiyant/sheetwise/commit/8a6dd0c963dc9698a935ea96066b7916c706a691))

- Update PyPI version badge in README.md for accurate display
  ([`58765b7`](https://github.com/Khushiyant/sheetwise/commit/58765b769c00258a5398d37c86406af113661f6c))

- Update Python version in workflow and dynamically read project version from pyproject.toml
  ([`2cc5387`](https://github.com/Khushiyant/sheetwise/commit/2cc538724b2d20b9ec7c1fab30fc78344bd27f56))

- Update Python version link in README.md to correct release
  ([`5985859`](https://github.com/Khushiyant/sheetwise/commit/59858595f2c93060f615bb536ee28fff28d9a555))

- Update release version to 2.5.1 in Sphinx config and improve test workflow
  ([`58c5b52`](https://github.com/Khushiyant/sheetwise/commit/58c5b524d80325045f1322735fc9be0ffdd253d1))

### Chores

- Add comprehensive documentation for SheetWise
  ([`a19187e`](https://github.com/Khushiyant/sheetwise/commit/a19187ebb0015bb7a8c7190bcab8e51929698ad8))

- Created examples for basic and advanced usage in `examples.rst` - Established main documentation
  structure in `index.rst` - Added installation instructions in `installation.rst` - Included
  license information in `license.rst` - Developed a quick start guide in `quickstart.rst` -
  Documented command line interface in `cli.rst` - Added user guides for compression, encoding,
  formulas, tables, and visualization - Updated `pyproject.toml` for documentation dependencies -
  Implemented scripts for building and generating documentation - Enhanced `uv.lock` with new
  dependencies for documentation tools

- Add image to README
  ([`34884d6`](https://github.com/Khushiyant/sheetwise/commit/34884d6ce99003e924c6be174196b43f8900547c))

Added an image to the README for better visualization.

- Add pyproject.toml for project configuration and dependencies
  ([`212b093`](https://github.com/Khushiyant/sheetwise/commit/212b0937fa484a8fc731dde46840f1c8bbfb80b0))

- Initialize Poetry configuration for the 'sheetwise' package. - Specify package metadata including
  name, version, description, authors, and classifiers. - Define dependencies for the project,
  including pandas, numpy, and openpyxl. - Set up development and testing dependencies with pytest,
  black, flake8, and others. - Configure build system to use poetry-core. - Add script entry point
  for command line interface. - Include configuration for black, isort, mypy, pytest, and coverage.

- Change from loop to matric check
  ([`0d7f40b`](https://github.com/Khushiyant/sheetwise/commit/0d7f40b4da8a16aaad07a77a0f8d7d70d71ff8ef))

- Enhance benchmarking script with error handling and auto-sample generation
  ([`f038262`](https://github.com/Khushiyant/sheetwise/commit/f038262a74b5f93ac5bb51dd8f0d6a52a7775c80))

- Enhance index.html with additional meta tags for improved SEO and social sharing
  ([`acb516f`](https://github.com/Khushiyant/sheetwise/commit/acb516fdc146331a3403331dfba50b14dfaa983e))

- Integrated `duckdb` to allow standard SQL queries directly against spreadsheet data via
  `query_sql()`
  ([`8754059`](https://github.com/Khushiyant/sheetwise/commit/87540591b9108313774bbf543266c5dd4ab1b52d))

- Remove obsolete HTML and CSS files from documentation
  ([`2e0f024`](https://github.com/Khushiyant/sheetwise/commit/2e0f02469b965b3aaa5381bd095800c96a627af0))

- Remove Read the Docs configuration file
  ([`f172a05`](https://github.com/Khushiyant/sheetwise/commit/f172a05727d8efcf152a9130fbec347b19454f10))

- Update pyproject.toml for version 2.3.0 and restructure dependencies
  ([`29ddb39`](https://github.com/Khushiyant/sheetwise/commit/29ddb3950cb345d438bb3603dd435ce12c85d09a))

- Update version to 2.1.1 in pyproject.toml and clean up README.md formatting
  ([`5f65b58`](https://github.com/Khushiyant/sheetwise/commit/5f65b580422cd367f1c38793d0661bcc56fdf142))

- Update version to 2.2.0 in pyproject.toml
  ([`9f1a327`](https://github.com/Khushiyant/sheetwise/commit/9f1a3271676e89152f7eab7f0eae28ed3f25d432))

### Documentation

- Add Token Budgeting section with usage example in README
  ([`33557c6`](https://github.com/Khushiyant/sheetwise/commit/33557c6ccb034d18ea03a64f23b728b955a54952))

### Features

- Add advanced table detection and classification utilities
  ([`5c554e9`](https://github.com/Khushiyant/sheetwise/commit/5c554e956b56017e6647ed7be467829255f1861f))

- Introduced SmartTableDetector for detecting and classifying tables in spreadsheets. - Implemented
  EnhancedTableRegion to extend table metadata with types and header information. - Added
  visualization utilities for spreadsheet compression analysis in CompressionVisualizer. - Created
  WorkbookManager for handling multi-sheet workbooks and cross-sheet references. - Developed methods
  for detecting cross-sheet references and generating sheet relationship graphs. - Enhanced tests
  for new features and ensured logging functionality works as expected.

- Add benchmarking and visualization capabilities to README and CLI
  ([`0c74580`](https://github.com/Khushiyant/sheetwise/commit/0c74580a89bb8f430c4804d2780025758c7fb41c))

- Add comprehensive tests for SpreadsheetLLM, SheetCompressor, and utility functions
  ([`82e7e63`](https://github.com/Khushiyant/sheetwise/commit/82e7e63f806f55890461d07471a7017c44bc3605))

- Add initial documentation and styles for SheetWise project
  ([`8380323`](https://github.com/Khushiyant/sheetwise/commit/83803235402e79830241abb85107489c6303f440))

- Add memory optimization for large files and enhance formula parsing with robust tokenizer
  ([`aa7124a`](https://github.com/Khushiyant/sheetwise/commit/aa7124a02f636375374c91209d339ca55dd37b91))

- Add Read the Docs configuration file for documentation build
  ([`dd78ee2`](https://github.com/Khushiyant/sheetwise/commit/dd78ee23f275ca8c22f615be26d0a65fd54ec261))

- Add seaborn and rich as dependencies in pyproject.toml
  ([`62f3186`](https://github.com/Khushiyant/sheetwise/commit/62f3186563b82ae95fa57b139474da36c2da2b23))

- Enhance SheetWise with auto-configuration, multi-LLM support, and improved logging
  ([`aed9f04`](https://github.com/Khushiyant/sheetwise/commit/aed9f04e774f06116d0cfbad862c74891adc73ca))

- Enhance StructuralAnchorExtractor with transition detection and vectorized implementation
  ([`5844c97`](https://github.com/Khushiyant/sheetwise/commit/5844c9798ad126808517795f78bb960b34b439d0))

- Implement robust file loading and SQL querying in SpreadsheetLLM, update version to 2.5.1 and add
  xlrd dependency
  ([`4b09409`](https://github.com/Khushiyant/sheetwise/commit/4b094095c2d4c76d00c8653964e907cc14d78b06))

- Implement SpreadsheetLLM framework with utilities for encoding spreadsheets
  ([`1ba9b7b`](https://github.com/Khushiyant/sheetwise/commit/1ba9b7bfc95d9307dc3d4908be288c9f6bd23d1b))

- Added core components for SpreadsheetLLM including compression, encoding, and table detection. -
  Introduced utility functions to create realistic demo spreadsheets. - Developed a command line
  interface for user interaction with the SpreadsheetLLM package. - Implemented data type
  classification and structural anchor extraction for efficient data handling. - Created a
  comprehensive README with package overview and usage instructions.

- Update version to 1.2.0, add changelog, and improve version retrieval in __init__.py
  ([`e234cae`](https://github.com/Khushiyant/sheetwise/commit/e234cae229ee3825e35673835e909e7abea7f588))

### Refactoring

- Add index.html and styles.css for improved design and functionality
  ([`53bfb04`](https://github.com/Khushiyant/sheetwise/commit/53bfb0437cb44b1632d9c70a7d6f70c065b8ec7a))

- Remove unused modules and classes from SpreadsheetLLM package
  ([`7bb7402`](https://github.com/Khushiyant/sheetwise/commit/7bb74022e85948b766bf1f6e3c0ff3f84298afb0))
