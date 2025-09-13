"""Command line interface for SheetWise."""


import argparse
import sys
import os
from pathlib import Path
import json
import pandas as pd

from sheetwise.core.core import SpreadsheetLLM
from sheetwise.utils.visualizer import CompressionVisualizer
from sheetwise.encoding.formula_parser import FormulaParser
from sheetwise.workbook.workbook import WorkbookManager
from sheetwise.tables.smart_tables import SmartTableDetector

# Rich for colorized CLI output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SheetWise: Encode spreadsheets for Large Language Models"
    )

    parser.add_argument(
        "input_file",
        nargs="?",  # Make input_file optional
        help="Path to input spreadsheet file (.xlsx, .xls, or .csv)",
    )

    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")

    parser.add_argument(
        "--compression-ratio", type=float, default=None, help="Target compression ratio"
    )

    parser.add_argument(
        "--vanilla",
        action="store_true",
        help="Use vanilla encoding instead of compression",
    )

    parser.add_argument("--stats", action="store_true", help="Show encoding statistics")

    parser.add_argument("--demo", action="store_true", help="Run demo with sample data")
    
    parser.add_argument("--auto-config", action="store_true", 
                       help="Automatically configure compression parameters")
    
    parser.add_argument("--format", choices=['text', 'json', 'html'], default='text',
                       help="Output format (text, json, or html)")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    # New command-line options for enhanced features
    feature_group = parser.add_argument_group('Enhanced Features')
    
    feature_group.add_argument("--extract-formulas", action="store_true",
                             help="Extract and analyze formulas from the spreadsheet")
    
    feature_group.add_argument("--visualize", action="store_true",
                             help="Generate visualization of spreadsheet compression")
    
    feature_group.add_argument("--multi-sheet", action="store_true",
                             help="Process all sheets in a workbook")
    feature_group.add_argument("--parallel", action="store_true",
                             help="Enable parallel processing for multi-sheet workbooks")
    feature_group.add_argument("--jobs", type=int, default=None,
                             help="Number of parallel workers (default: number of sheets)")
    
    feature_group.add_argument("--detect-tables", action="store_true",
                             help="Detect and extract tables from the spreadsheet")
    
    feature_group.add_argument("--report", action="store_true",
                             help="Generate comprehensive HTML report")

    args = parser.parse_args()

    console = Console()

    # Handle demo mode
    if args.demo:
        from sheetwise.utils.utils import create_realistic_spreadsheet

        console.rule("[bold blue]SheetWise Demo Mode")
        console.print("[bold green]Running SheetWise demo...", style="green")
        df = create_realistic_spreadsheet()
        sllm = SpreadsheetLLM(enable_logging=args.verbose)

        console.print(f"[bold yellow]Created demo spreadsheet:[/] {df.shape}")
        
        # Choose encoding method based on options
        if args.vanilla:
            console.print("[bold cyan]Using vanilla encoding...[/]")
            encoded = sllm.encode_vanilla(df)
            encoding_type = "vanilla"
        elif args.auto_config:
            console.print("[bold cyan]Using auto-configuration...[/]")
            encoded = sllm.compress_with_auto_config(df)
            encoding_type = "auto-compressed"
        else:
            encoded = sllm.compress_and_encode_for_llm(df)
            encoding_type = "compressed"

        # Get stats if requested
        stats = {}
        if args.stats:
            stats = sllm.get_encoding_stats(df)
            table = Table(title=f"Encoding Statistics ({encoding_type})", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim", width=28)
            table.add_column("Value", style="bold")
            for key, value in stats.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.2f}")
                else:
                    table.add_row(key, str(value))
            console.print(table)

        # Handle visualization if requested
        if args.visualize:
            console.print("[bold blue]Generating visualization...[/]")
            visualizer = CompressionVisualizer()
            compressed_result = sllm.compress_spreadsheet(df)
            # Progress bar for visualizations
            for desc, func in track([
                ("Data density heatmap", visualizer.create_data_density_heatmap),
                ("Compression comparison", lambda d: visualizer.compare_original_vs_compressed(d, compressed_result))
            ], description="[green]Creating visualizations..."):
                if desc == "Data density heatmap":
                    fig = func(df)
                    viz_path = "density_heatmap.png"
                    visualizer.save_visualization_to_file(fig, viz_path)
                    console.print(f"[green]Saved visualization to {viz_path}")
                else:
                    fig2 = func(df)
                    viz_path2 = "compression_comparison.png"
                    visualizer.save_visualization_to_file(fig2, viz_path2)
                    console.print(f"[green]Saved comparison visualization to {viz_path2}")

        # Handle table detection if requested
        if args.detect_tables:
            console.print("[bold blue]Detecting tables...[/]")
            detector = SmartTableDetector()
            tables = list(track(detector.detect_tables(df), description="[yellow]Detecting tables..."))
            console.print(f"[bold green]Detected {len(tables)} tables:")
            table = Table(title="Detected Tables", show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim")
            table.add_column("Rows")
            table.add_column("Columns")
            table.add_column("Type")
            table.add_column("Headers")
            for i, t in enumerate(tables):
                table.add_row(
                    str(i+1),
                    f"{t.start_row}-{t.end_row}",
                    f"{t.start_col}-{t.end_col}",
                    t.table_type.value,
                    str(t.has_headers)
                )
            console.print(table)
        
        # Handle output format
        if args.format == "json":
            # Convert numpy types to native Python types for JSON serialization
            json_stats = {}
            if stats:
                for key, value in stats.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        json_stats[key] = value.item()
                    elif isinstance(value, tuple):  # shape tuple
                        json_stats[key] = list(value)
                    else:
                        json_stats[key] = value
            
            output_data = {
                "encoding_type": encoding_type,
                "data_shape": list(df.shape),  # Convert to list for JSON
                "output_length": len(encoded),
                "content": encoded
            }
            
            if args.stats:
                output_data["statistics"] = json_stats
            
            formatted_output = json.dumps(output_data, indent=2)
            print(f"\nJSON Output:")
            print(formatted_output)
        else:
            # Text format (default)
            print(f"\nLLM-ready output ({encoding_type}, {len(encoded)} characters):")
            print(encoded[:500] + "..." if len(encoded) > 500 else encoded)
        
        return

    # Validate input file is provided when not in demo mode
    if not args.input_file:
        console.print(Panel("input_file is required when not using --demo", title="[red]Error", style="bold red"))
        parser.print_help()
        sys.exit(1)

    # Validate input file
    if not Path(args.input_file).exists():
        console.print(Panel(f"Input file '{args.input_file}' not found", title="[red]Error", style="bold red"))
        sys.exit(1)

    try:
        # Initialize SpreadsheetLLM
        sllm = SpreadsheetLLM()

        # Load spreadsheet
        df = sllm.load_from_file(args.input_file)
        console.rule(f"[bold blue]Loaded spreadsheet: {df.shape} ({args.input_file})")

        # Parallel processing for multi-sheet workbooks
        if args.multi_sheet:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            wb_manager = WorkbookManager()
            sheets = wb_manager.load_workbook(args.input_file)
            num_workers = args.jobs or len(sheets)
            console.print(f"[bold cyan]Processing {len(sheets)} sheets with {num_workers} workers...[/]")
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(wb_manager.compress_sheet, sheet, sllm.compressor): name for name, sheet in sheets.items()}
                for future in track(as_completed(futures), total=len(futures), description="[green]Processing sheets..."):
                    name = futures[future]
                    try:
                        compressed = future.result()
                        results.append((name, compressed))
                    except Exception as e:
                        console.print(Panel(f"Error processing sheet {name}: {e}", title="[red]Error", style="bold red"))
            # Show summary table
            table = Table(title="Sheet Compression Results", show_header=True, header_style="bold magenta")
            table.add_column("Sheet Name", style="dim")
            table.add_column("Rows")
            table.add_column("Columns")
            table.add_column("Compression Ratio")
            for name, compressed in results:
                shape = compressed.get('original_shape', (0, 0))
                ratio = compressed.get('compression_ratio', 0)
                table.add_row(name, str(shape[0]), str(shape[1]), f"{ratio:.2f}")
            console.print(table)
            return

        # Generate encoding (single sheet)
        if args.vanilla:
            encoded = sllm.encode_vanilla(df)
            encoding_type = "vanilla"
        else:
            encoded = sllm.compress_and_encode_for_llm(df)
            encoding_type = "compressed"

        # Show statistics if requested
        if args.stats:
            stats = sllm.get_encoding_stats(df)
            table = Table(title=f"Encoding Statistics ({encoding_type})", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim", width=28)
            table.add_column("Value", style="bold")
            for key, value in stats.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.2f}")
                else:
                    table.add_row(key, str(value))
            console.print(table)

        # Output result
        if args.output:
            with open(args.output, "w") as f:
                f.write(encoded)
            console.print(Panel(f"Encoded output written to: {args.output}", title="[green]Success", style="bold green"))
        else:
            console.print(encoded)

    except Exception as e:
        console.print(Panel(f"{e}", title="[red]Error", style="bold red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
