"""Main compression framework combining all modules."""

from typing import Any, Dict, Optional

import pandas as pd

from sheetwise.tables.extractors import (
    DataFormatAggregator,
    InvertedIndexTranslator,
    StructuralAnchorExtractor,
)


class SheetCompressor:
    """
    Main compression framework combining all three modules:
    1. Structural-anchor-based extraction
    2. Inverted-index translation
    3. Data-format-aware aggregation
    """

    def __init__(
        self,
        k: int = 4,
        use_extraction: bool = True,
        use_translation: bool = True,
        use_aggregation: bool = True,
    ):
        """
        Initialize SheetCompressor with module options

        Args:
            k: Parameter for structural anchor extraction
            use_extraction: Whether to use structural anchor extraction
            use_translation: Whether to use inverted index translation
            use_aggregation: Whether to use data format aggregation
        """
        self.k = k
        self.use_extraction = use_extraction
        self.use_translation = use_translation
        self.use_aggregation = use_aggregation

        self.extractor = StructuralAnchorExtractor(k) if use_extraction else None
        self.translator = InvertedIndexTranslator() if use_translation else None
        self.aggregator = DataFormatAggregator() if use_aggregation else None

    def compress(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply compression pipeline to spreadsheet data

        Args:
            df: Input DataFrame

        Returns:
            Compressed representation
        """
        result = {"original_shape": df.shape, "compression_steps": []}

        current_df = df.copy()

        # Step 1: Structural anchor extraction
        if self.use_extraction and self.extractor:
            current_df = self.extractor.extract_skeleton(current_df)
            result["compression_steps"].append(
                {"step": "structural_extraction", "shape_after": current_df.shape}
            )

        # Step 2: Inverted index translation
        if self.use_translation and self.translator:
            inverted_index = self.translator.translate(current_df)
            result["inverted_index"] = inverted_index
            result["compression_steps"].append(
                {"step": "inverted_translation", "unique_values": len(inverted_index)}
            )

        # Step 3: Data format aggregation
        if self.use_aggregation and self.aggregator:
            format_groups = self.aggregator.aggregate(current_df)
            result["format_aggregation"] = format_groups
            result["compression_steps"].append(
                {"step": "format_aggregation", "format_types": len(format_groups)}
            )

        result["compressed_data"] = current_df
        result["compression_ratio"] = (df.shape[0] * df.shape[1]) / (
            current_df.shape[0] * current_df.shape[1]
        )

        return result
