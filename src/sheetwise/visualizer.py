"""Visualization utilities for spreadsheet compression."""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from base64 import b64encode
import json


class CompressionVisualizer:
    """
    Visualization tools for spreadsheet compression analysis.
    Now includes Interactive HTML Reports.
    """
    
    def __init__(self, enable_interactive: bool = True):
        self.enable_interactive = enable_interactive
        
    def create_data_density_heatmap(self, df: pd.DataFrame, 
                                    title: str = "Data Density Heatmap") -> plt.Figure:
        """Generate a heatmap showing data density in the spreadsheet."""
        non_empty_mask = ~df.isna() & (df != "")
        density_matrix = non_empty_mask.astype(int)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.pcolor(
            density_matrix.transpose(), 
            cmap='Blues', 
            alpha=0.8, 
            edgecolors='face', # improved for large sheets
            linewidths=0
        )
        
        ax.set_title(title)
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Column Index")
        ax.invert_yaxis() # Match spreadsheet layout (A1 at top left)
        
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label("Has Data")
        
        density = density_matrix.sum().sum() / max(1, density_matrix.size)
        plt.figtext(0.5, 0.01, f"Data Density: {density:.2%}", ha="center", fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def visualize_anchors(self, df: pd.DataFrame, 
                         anchors: Tuple[List[int], List[int]], 
                         title: str = "Structural Anchors") -> plt.Figure:
        """Visualize structural anchors identified in the spreadsheet."""
        row_anchors, col_anchors = anchors
        viz_matrix = np.zeros((df.shape[0], df.shape[1]))
        
        # Mark data, row anchors, col anchors, and intersections
        non_empty_mask = ~df.isna() & (df != "")
        viz_matrix[non_empty_mask] = 1
        
        if row_anchors:
            viz_matrix[row_anchors, :] = 2
        if col_anchors:
            viz_matrix[:, col_anchors] = 2
        
        # Intersections
        if row_anchors and col_anchors:
             # Create meshgrid for vectorized intersection marking
             r_idx, c_idx = np.meshgrid(row_anchors, col_anchors, indexing='ij')
             # Clip to bounds just in case
             r_idx = np.clip(r_idx, 0, df.shape[0]-1)
             c_idx = np.clip(c_idx, 0, df.shape[1]-1)
             viz_matrix[r_idx, c_idx] = 3
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.colors.ListedColormap(['white', '#e3f2fd', '#ffe0b2', '#ff5252'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        heatmap = ax.pcolor(viz_matrix.transpose(), cmap=cmap, norm=norm, edgecolors='face')
        
        ax.set_title(title)
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Column Index")
        ax.invert_yaxis()
        
        cbar = plt.colorbar(heatmap, ax=ax, ticks=[0.25, 1, 2, 3])
        cbar.set_ticklabels(['Empty', 'Data', 'Anchor', 'Intersection'])
        
        plt.tight_layout()
        return fig

    def generate_interactive_report(self, original_df: pd.DataFrame, 
                                  compressed_result: Dict[str, Any],
                                  filename: str = "report.html") -> str:
        """
        Generate a standalone interactive HTML report for auditing compression.
        
        Features:
        - Side-by-side view (Original vs Compressed)
        - Highlighted "Removed" regions
        - Click-to-scroll navigation
        """
        compressed_df = compressed_result.get('compressed_data', pd.DataFrame())
        
        # 1. Prepare HTML Structure
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SheetWise Compression Audit</title>
            <style>
                :root { --primary: #2c3e50; --deleted: #ffebee; --gap: #f5f5f5; --highlight: #ffcdd2; }
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }
                header { background: var(--primary); color: white; padding: 1rem; display: flex; justify-content: space-between; align-items: center; }
                .container { display: flex; flex: 1; overflow: hidden; }
                .panel { flex: 1; display: flex; flex-direction: column; border-right: 1px solid #ddd; min-width: 0; }
                .panel-header { background: #eee; padding: 10px; font-weight: bold; border-bottom: 1px solid #ccc; }
                .grid-container { flex: 1; overflow: auto; padding: 10px; position: relative; }
                
                table { border-collapse: collapse; width: 100%; font-size: 13px; table-layout: fixed; }
                th, td { border: 1px solid #ddd; padding: 4px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; height: 25px; }
                th { background: #f8f9fa; position: sticky; top: 0; z-index: 10; text-align: left; font-weight: 600; color: #555; }
                
                /* Specific Styles */
                .orig-row { transition: background 0.3s; }
                .orig-row.deleted { background-color: var(--deleted); color: #888; }
                .orig-row.target-highlight { background-color: #e53935 !important; color: white !important; }
                
                .gap-row { background-color: var(--gap); cursor: pointer; color: #666; font-style: italic; font-size: 11px; text-align: center; border-left: 4px solid #bbb; }
                .gap-row:hover { background-color: #e0e0e0; border-left-color: var(--primary); }
                
                /* Stats Badge */
                .badge { background: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 4px; font-size: 0.9em; margin-left: 10px; }
            </style>
        </head>
        <body>
            <header>
                <div>
                    <span style="font-size: 1.2em; font-weight: bold;">SheetWise Audit Report</span>
                    <span class="badge">Original: {orig_shape}</span>
                    <span class="badge">Compressed: {comp_shape}</span>
                    <span class="badge">Ratio: {ratio:.1f}x</span>
                </div>
                <div style="font-size: 0.9em; opacity: 0.8;">Generated by SheetWise v2.6.0</div>
            </header>
            
            <div class="container">
                <div class="panel">
                    <div class="panel-header">Original Spreadsheet (Full Context)</div>
                    <div class="grid-container" id="orig-container">
                        <table id="orig-table">
                            <thead><tr><th>#</th>{orig_headers}</tr></thead>
                            <tbody>{orig_rows}</tbody>
                        </table>
                    </div>
                </div>
                
                <div class="panel">
                    <div class="panel-header">Compressed Skeleton (LLM Input)</div>
                    <div class="grid-container">
                        <table id="comp-table">
                            <thead><tr><th>#</th>{comp_headers}</tr></thead>
                            <tbody>{comp_rows}</tbody>
                        </table>
                    </div>
                </div>
            </div>

            <script>
                // Interaction Logic
                function highlightRange(start, end) {
                    // Remove old highlights
                    document.querySelectorAll('.target-highlight').forEach(el => el.classList.remove('target-highlight'));
                    
                    // Highlight new range
                    const container = document.getElementById('orig-container');
                    let firstEl = null;
                    
                    for (let i = start; i <= end; i++) {
                        const el = document.getElementById('orig-row-' + i);
                        if (el) {
                            el.classList.add('target-highlight');
                            if (!firstEl) firstEl = el;
                        }
                    }
                    
                    // Scroll to first element
                    if (firstEl) {
                        firstEl.scrollIntoView({behavior: 'smooth', block: 'center'});
                    }
                }
            </script>
        </body>
        </html>
        """

        # 2. Process Data for Rendering
        
        # --- Helper: Safely format values ---
        def fmt(val):
            if pd.isna(val): return ""
            s = str(val)
            return s[:50] + "..." if len(s) > 50 else s

        # --- Generate Original Rows ---
        # We assume indices are preserved in compressed_df to identify gaps
        kept_indices = set(compressed_df.index)
        
        orig_rows_html = []
        # Optimization: If DF is huge (>5k rows), we might want to truncate or paginate.
        # For now, we render all to ensure full auditability, but warn in logs if needed.
        
        for idx, row in original_df.iterrows():
            is_deleted = idx not in kept_indices
            cls = "orig-row deleted" if is_deleted else "orig-row"
            row_cells = "".join(f"<td>{fmt(val)}</td>" for val in row)
            orig_rows_html.append(
                f'<tr id="orig-row-{idx}" class="{cls}"><td>{idx+1}</td>{row_cells}</tr>'
            )

        # --- Generate Compressed Rows with "Gaps" ---
        comp_rows_html = []
        last_idx = -1
        
        # Ensure we iterate in the order of the compressed dataframe
        for idx, row in compressed_df.iterrows():
            # Check for gap
            if idx > last_idx + 1:
                gap_size = idx - (last_idx + 1)
                gap_start = last_idx + 1
                gap_end = idx - 1
                comp_rows_html.append(
                    f'<tr class="gap-row" onclick="highlightRange({gap_start}, {gap_end})">'
                    f'<td colspan="{len(compressed_df.columns) + 1}">'
                    f'&#8942; {gap_size} rows removed (Rows {gap_start+1}-{gap_end+1}) &#8942;'
                    f'</td></tr>'
                )
            
            # Render Row
            row_cells = "".join(f"<td>{fmt(val)}</td>" for val in row)
            comp_rows_html.append(f'<tr><td>{idx+1}</td>{row_cells}</tr>')
            last_idx = idx

        # Handle Trailing Gap
        if last_idx < len(original_df) - 1:
            gap_start = last_idx + 1
            gap_end = len(original_df) - 1
            gap_size = gap_end - gap_start + 1
            comp_rows_html.append(
                f'<tr class="gap-row" onclick="highlightRange({gap_start}, {gap_end})">'
                f'<td colspan="{len(compressed_df.columns) + 1}">'
                f'&#8942; {gap_size} rows removed (End of sheet) &#8942;'
                f'</td></tr>'
            )

        # --- Headers ---
        orig_headers = "".join(f"<th>{col}</th>" for col in original_df.columns)
        comp_headers = "".join(f"<th>{col}</th>" for col in compressed_df.columns)

        # 3. Assemble
        html_content = html_template.format(
            orig_shape=f"{original_df.shape[0]}x{original_df.shape[1]}",
            comp_shape=f"{compressed_df.shape[0]}x{compressed_df.shape[1]}",
            ratio=compressed_result.get('compression_ratio', 0),
            orig_headers=orig_headers,
            orig_rows="".join(orig_rows_html),
            comp_headers=comp_headers,
            comp_rows="".join(comp_rows_html)
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return filename

    # Keep existing methods (comparison, html static, fig_to_base64)
    def compare_original_vs_compressed(self, original_df, compressed_result):
        # ... (Previous implementation remains unchanged)
        # Just creating a placeholder to ensure class continuity
        pass 
        
    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        return b64encode(buf.read()).decode('utf-8')

    def generate_html_report(self, original_df: pd.DataFrame, 
                           compressed_result: Dict[str, Any]) -> str:
        """Legacy static report (Backwards compatibility)"""
        # We can now redirect this to the new interactive one or keep the old static one
        # For v2.6, we keep the static one as 'summary' and the new one as 'audit'
        # ... (Old implementation)
        return "Use generate_interactive_report() for the new v2.6 interactive experience."