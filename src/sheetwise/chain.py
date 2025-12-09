"""Chain of Spreadsheet reasoning implementation (Offline Edition)."""

from typing import Any, Dict, List, Optional

import pandas as pd
try:
    from thefuzz import fuzz, process
except ImportError:
    # Fallback if thefuzz is not installed
    fuzz = None

from .compressor import SheetCompressor
from .data_types import TableRegion
from .smart_tables import SmartTableDetector, EnhancedTableRegion


class ChainOfSpreadsheet:
    """
    Implements deterministic 'Chain of Spreadsheet' reasoning.
    Uses fuzzy matching and heuristic scoring instead of LLMs.
    """

    def __init__(self, compressor: SheetCompressor = None):
        self.compressor = compressor or SheetCompressor()
        self.detector = SmartTableDetector()

    def process_query(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Process a query using deterministic search logic.

        Args:
            df: Input DataFrame
            query: Natural language query (e.g., "Revenue in 2023")

        Returns:
            Dictionary containing identified tables and search scores.
        """
        if fuzz is None:
            raise ImportError("Please install 'thefuzz' for offline reasoning: pip install thefuzz")

        # 1. Detect all tables
        tables = self.detector.detect_tables(df)
        
        # 2. Score tables based on relevance to query
        scored_results = []
        
        for table in tables:
            score = self._calculate_relevance(df, table, query)
            if score > 40:  # Threshold for relevance
                # Extract the actual data for the result
                table_data = df.iloc[
                    table.start_row : table.end_row + 1, 
                    table.start_col : table.end_col + 1
                ]
                scored_results.append({
                    "score": score,
                    "table_type": table.table_type.value,
                    "region": f"{table.top_left}:{table.bottom_right}",
                    "data_preview": table_data.head().to_dict(orient='split')
                })

        # Sort by relevance
        scored_results.sort(key=lambda x: x['score'], reverse=True)

        result = {
            "query": query,
            "matches_found": len(scored_results),
            "top_hits": scored_results[:3],  # Return top 3 most relevant tables
            "all_tables_count": len(tables),
            "processing_stages": [
                "smart_table_detection",
                "fuzzy_keyword_scoring",
                "relevance_ranking"
            ],
        }

        return result

    def _calculate_relevance(self, df: pd.DataFrame, table: EnhancedTableRegion, query: str) -> int:
        """
        Calculate a relevance score (0-100) for a table against a query.
        Prioritizes matches in Headers > Matches in Data.
        """
        query_lower = query.lower()
        
        # Extract header text
        headers = []
        if table.header_rows:
            header_row_idx = table.header_rows[0]
            headers = df.iloc[header_row_idx, table.start_col : table.end_col + 1].astype(str).tolist()
        
        # Extract data sample (flattened)
        data_sample = df.iloc[
            table.start_row + 1 : min(table.end_row, table.start_row + 5), 
            table.start_col : table.end_col + 1
        ].astype(str).values.flatten().tolist()

        max_header_score = 0
        if headers:
            # Check best partial match in headers
            best_header = process.extractOne(query, headers, scorer=fuzz.partial_ratio)
            if best_header:
                max_header_score = best_header[1]

        max_data_score = 0
        if data_sample:
             # Check best partial match in data bodies (e.g. searching for a specific ID)
            best_data = process.extractOne(query, data_sample, scorer=fuzz.partial_ratio)
            if best_data:
                max_data_score = best_data[1]

        # Weighted Score: Headers are 2x as important as data values
        # If header matches perfectly (100), score is high.
        final_score = (max_header_score * 0.7) + (max_data_score * 0.3)
        return int(final_score)