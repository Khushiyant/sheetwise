"""Command line interface for SheetWise."""

import argparse
import sys
import os
from pathlib import Path
import json

import pandas as pd

from . import SpreadsheetLLM, FormulaParser, CompressionVisualizer, WorkbookManager, SmartTableDetector

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
        nargs="?",
        help="Path to input spreadsheet file (.xlsx, .xls, or .csv)",
    )

    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")

    parser.add_argument(
        "--compression-ratio", type=float, default=None, help="Target compression ratio"
    )
    
    # NEW: Token limit argument
    parser.add_argument(
        "--token-limit", type=int, default=None, 
        help="Hard limit on output tokens (automatically increases compression to fit)"
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
    
    # Enhanced features
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
    
    args = parser.parse_args()
    console = Console()

    # Demo Mode Logic
    if args.demo:
        from .utils import create_realistic_spreadsheet
        console.rule("[bold blue]SheetWise Demo Mode")
        console.print("[bold green]Running SheetWise demo...", style="green")
        df = create_realistic_spreadsheet()
        sllm = SpreadsheetLLM(enable_logging=args.verbose)
        console.print(f"[bold yellow]Created demo spreadsheet:[/] {df.shape}")
        
        if args.vanilla:
            console.print("[bold cyan]Using vanilla encoding...[/]")
            encoded = sllm.encode_vanilla(df)
            encoding_type = "vanilla"
        elif args.token_limit:
            console.print(f"[bold cyan]Compressing to fit {args.token_limit} tokens...[/]")
            encoded = sllm.encode_to_token_limit(df, args.token_limit)
            encoding_type = "budget-constrained"
        elif args.auto_config:
            console.print("[bold cyan]Using auto-configuration...[/]")
            encoded = sllm.compress_with_auto_config(df)
            encoding_type = "auto-compressed"
        else:
            encoded = sllm.compress_and_encode_for_llm(df)
            encoding_type = "compressed"

        # ... (Display logic remains similar, see standard output below) ...
        # Simplified output for brevity in this update
        console.print(f"\nLLM-ready output ({encoding_type}, {len(encoded)} characters):")
        console.print(encoded[:500] + "..." if len(encoded) > 500 else encoded)
        return

    # Standard Mode Logic
    if not args.input_file:
        console.print(Panel("input_file is required when not using --demo", title="[red]Error", style="bold red"))
        parser.print_help()
        sys.exit(1)

    if not Path(args.input_file).exists():
        console.print(Panel(f"Input file '{args.input_file}' not found", title="[red]Error", style="bold red"))
        sys.exit(1)

    try:
        sllm = SpreadsheetLLM(enable_logging=args.verbose)
        df = sllm.load_from_file(args.input_file)
        console.rule(f"[bold blue]Loaded spreadsheet: {df.shape} ({args.input_file})")

        # Select Encoding Method
        if args.vanilla:
            encoded = sllm.encode_vanilla(df)
            encoding_type = "vanilla"
        elif args.token_limit:
            console.print(f"[cyan]Optimizing compression for {args.token_limit} token limit...[/]")
            encoded = sllm.encode_to_token_limit(df, args.token_limit)
            encoding_type = "budget-constrained"
        elif args.auto_config:
            console.print("[cyan]Auto-configuring compression...[/]")
            encoded = sllm.compress_with_auto_config(df)
            encoding_type = "auto-compressed"
        else:
            encoded = sllm.compress_and_encode_for_llm(df)
            encoding_type = "compressed"

        # Statistics
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

        # Output
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