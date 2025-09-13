"""Visualization utilities for spreadsheet compression."""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from base64 import b64encode


class CompressionVisualizer:
    """
    Visualization tools for spreadsheet compression analysis.
    
    This class provides utilities to:
    1. Generate heatmaps showing data density
    2. Visualize structural anchors and compression
    3. Compare original and compressed data
    4. Create shareable visual reports
    """
    
    def __init__(self, enable_interactive: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            enable_interactive: Whether to enable interactive visualizations
        """
        self.enable_interactive = enable_interactive
        
    def create_data_density_heatmap(self, df: pd.DataFrame, 
                                    title: str = "Data Density Heatmap") -> plt.Figure:
        """
        Generate a heatmap showing data density in the spreadsheet.
        
        Args:
            df: The dataframe to visualize
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        # Create a boolean mask of non-empty cells
        non_empty_mask = ~df.isna()
        
        # Convert to numeric for heatmap (1 for values, 0 for NaN)
        density_matrix = non_empty_mask.astype(int)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        heatmap = ax.pcolor(
            density_matrix.transpose(), 
            cmap='Blues', 
            alpha=0.8, 
            edgecolors='gray', 
            linewidths=0.01
        )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Column Index")
        
        # Add color bar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label("Has Data")
        
        # Set grid lines
        ax.grid(True, color='gray', linestyle='-', linewidth=0.25, alpha=0.5)
        
        # Add density stats
        density = density_matrix.sum().sum() / (density_matrix.shape[0] * density_matrix.shape[1])
        plt.figtext(0.5, 0.01, f"Data Density: {density:.2%}", ha="center", fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def visualize_anchors(self, df: pd.DataFrame, 
                         anchors: Tuple[List[int], List[int]], 
                         title: str = "Structural Anchors") -> plt.Figure:
        """
        Visualize structural anchors identified in the spreadsheet.
        
        Args:
            df: Original dataframe
            anchors: Tuple of (row_anchors, col_anchors)
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        row_anchors, col_anchors = anchors
        
        # Create a matrix for visualization
        viz_matrix = np.zeros((df.shape[0], df.shape[1]))
        
        # Mark regular data cells
        non_empty_mask = ~df.isna()
        viz_matrix[non_empty_mask] = 1
        
        # Mark row anchors
        for row in row_anchors:
            viz_matrix[row, :] = 2
            
        # Mark column anchors
        for col in col_anchors:
            viz_matrix[:, col] = 2
            
        # Where they intersect should be even higher
        for row in row_anchors:
            for col in col_anchors:
                viz_matrix[row, col] = 3
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Custom colormap: white for empty, light blue for data, 
        # orange for anchors, red for intersections
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'orange', 'red'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot heatmap
        heatmap = ax.pcolor(
            viz_matrix.transpose(), 
            cmap=cmap,
            norm=norm,
            edgecolors='gray', 
            linewidths=0.01
        )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Column Index")
        
        # Add color bar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label("Cell Type")
        cbar.set_ticks([0.25, 1, 2, 3])
        cbar.set_ticklabels(['Empty', 'Data', 'Anchor', 'Key Intersection'])
        
        # Add summary
        plt.figtext(0.5, 0.01, 
                  f"Identified {len(row_anchors)} row anchors and {len(col_anchors)} column anchors", 
                  ha="center", fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def compare_original_vs_compressed(self, 
                                      original_df: pd.DataFrame, 
                                      compressed_result: Dict[str, Any]) -> plt.Figure:
        """
        Create a visual comparison between original and compressed data.
        
        Args:
            original_df: Original dataframe
            compressed_result: Compression result dictionary
            
        Returns:
            Matplotlib figure with comparison visualization
        """
        # Extract information
        compressed_df = compressed_result.get('compressed_data', pd.DataFrame())
        compression_ratio = compressed_result.get('compression_ratio', 1.0)
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Visualize original data
        original_density = ~original_df.isna()
        ax1.pcolor(
            original_density.transpose(),
            cmap='Blues',
            alpha=0.8,
            edgecolors='gray',
            linewidths=0.01
        )
        ax1.set_title(f"Original Data\n{original_df.shape[0]} × {original_df.shape[1]} cells")
        ax1.set_xlabel("Row")
        ax1.set_ylabel("Column")
        
        # Visualize compressed data (if available)
        if not compressed_df.empty:
            compressed_density = ~compressed_df.isna()
            ax2.pcolor(
                compressed_density.transpose(),
                cmap='Greens',
                alpha=0.8,
                edgecolors='gray',
                linewidths=0.01
            )
            ax2.set_title(f"Compressed Data\n{compressed_df.shape[0]} × {compressed_df.shape[1]} cells")
            ax2.set_xlabel("Row")
            ax2.set_ylabel("Column")
        else:
            ax2.text(0.5, 0.5, "No compressed dataframe available", 
                   ha='center', va='center', transform=ax2.transAxes)
        
        # Add summary information
        plt.figtext(0.5, 0.01, 
                  f"Compression Ratio: {compression_ratio:.1f}x", 
                  ha="center", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_html_report(self, original_df: pd.DataFrame, 
                           compressed_result: Dict[str, Any]) -> str:
        """
        Generate an HTML report with visualizations.
        
        Args:
            original_df: Original dataframe
            compressed_result: Compression result dictionary
            
        Returns:
            HTML string with embedded visualizations
        """
        # Create visualizations
        density_fig = self.create_data_density_heatmap(original_df)
        
        # Get anchors if available
        if 'structural_anchors' in compressed_result:
            anchors_fig = self.visualize_anchors(
                original_df, 
                compressed_result['structural_anchors']
            )
        else:
            # Create dummy figure if no anchors
            anchors_fig = plt.figure(figsize=(10, 8))
            ax = anchors_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No anchor data available", 
                  ha='center', va='center', transform=ax.transAxes)
            
        comparison_fig = self.compare_original_vs_compressed(
            original_df, 
            compressed_result
        )
        
        # Convert figures to base64 for embedding in HTML
        density_img = self._fig_to_base64(density_fig)
        anchors_img = self._fig_to_base64(anchors_fig)
        comparison_img = self._fig_to_base64(comparison_fig)
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spreadsheet Compression Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .viz {{ margin-bottom: 40px; text-align: center; }}
                .viz img {{ max-width: 100%; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                h2 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Spreadsheet Compression Analysis</h1>
                    <p>Generated by SheetWise Visualization Module</p>
                </div>
                
                <div class="stats">
                    <h2>Compression Statistics</h2>
                    <p><strong>Original Size:</strong> {original_df.shape[0]} rows × {original_df.shape[1]} columns = {original_df.shape[0] * original_df.shape[1]} cells</p>
                    <p><strong>Compression Ratio:</strong> {compressed_result.get('compression_ratio', 'N/A')}x</p>
                    <p><strong>Non-empty Cells:</strong> {(~original_df.isna()).sum().sum()}</p>
                    <p><strong>Data Density:</strong> {(~original_df.isna()).sum().sum() / (original_df.shape[0] * original_df.shape[1]):.1%}</p>
                </div>
                
                <div class="viz">
                    <h2>Data Density Visualization</h2>
                    <img src="data:image/png;base64,{density_img}" alt="Data Density Heatmap">
                </div>
                
                <div class="viz">
                    <h2>Structural Anchors</h2>
                    <img src="data:image/png;base64,{anchors_img}" alt="Structural Anchors">
                </div>
                
                <div class="viz">
                    <h2>Original vs. Compressed Comparison</h2>
                    <img src="data:image/png;base64,{comparison_img}" alt="Comparison">
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # Close to prevent memory leaks
        return img_str
        
    def save_visualization_to_file(self, fig: plt.Figure, 
                                 filename: str) -> str:
        """
        Save visualization to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close to prevent memory leaks
        return filename
