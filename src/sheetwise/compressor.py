"""Main compression framework combining all modules."""

from typing import Any, Dict, Optional
import pandas as pd
import gc

from .extractors import (
    DataFormatAggregator,
    InvertedIndexTranslator,
    StructuralAnchorExtractor,
)


class SheetCompressor:
    """
    Main compression framework combining all three modules.
    Optimized for memory efficiency.
    """

    def __init__(
        self,
        k: int = 4,
        use_extraction: bool = True,
        use_translation: bool = True,
        use_aggregation: bool = True,
    ):
        self.k = k
        self.use_extraction = use_extraction
        self.use_translation = use_translation
        self.use_aggregation = use_aggregation

        self.extractor = StructuralAnchorExtractor(k) if use_extraction else None
        self.translator = InvertedIndexTranslator() if use_translation else None
        self.aggregator = DataFormatAggregator() if use_aggregation else None

    def compress(self, df: pd.DataFrame, inplace: bool = False) -> Dict[str, Any]:
        """
        Apply compression pipeline to spreadsheet data.
        
        Args:
            df: Input DataFrame
            inplace: If True, attempts to minimize memory copies (CAUTION: modifies data flow)
                     Note: Pandas operations often return copies anyway, but this flag
                     prevents the initial full copy.
                     
        Returns:
            Compressed representation
        """
        result = {"original_shape": df.shape, "compression_steps": []}

        # Memory Optimization: Avoid initial copy if requested
        if inplace:
            current_df = df
        else:
            current_df = df.copy()

        # Step 1: Structural anchor extraction
        if self.use_extraction and self.extractor:
            # Extraction reduces rows/cols, so it naturally creates a smaller copy
            new_df = self.extractor.extract_skeleton(current_df)
            
            # Explicitly free memory of intermediate dataframe if we own it
            if not inplace and current_df is not df:
                del current_df
                gc.collect()
                
            current_df = new_df
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
        
        # Safe division for empty dataframes
        orig_size = df.shape[0] * df.shape[1]
        comp_size = current_df.shape[0] * current_df.shape[1]
        
        if comp_size > 0:
            result["compression_ratio"] = orig_size / comp_size
        else:
            result["compression_ratio"] = 0.0

        return result