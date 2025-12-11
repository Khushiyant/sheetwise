"""
Benchmark and Visualization Script for SheetWise

- Runs compression and analysis on a set of sample spreadsheets
- Collects metrics (compression ratio, time, memory, etc.)
- Generates beautiful charts using seaborn/matplotlib
- Saves charts for documentation/website
"""
import os
import time
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from memory_profiler import memory_usage
except ImportError:
    print("Error: memory_profiler not found. Please run: uv add --dev memory_profiler")
    exit(1)

from sheetwise import SpreadsheetLLM
from sheetwise.utils import create_realistic_spreadsheet

# Directory with sample spreadsheets
SAMPLES_DIR = "benchmarks/samples"
RESULTS_DIR = "benchmarks/results"
CHARTS_DIR = "benchmarks/charts"

# Ensure directories exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Find all .xlsx and .csv files
sample_files = glob.glob(os.path.join(SAMPLES_DIR, "*.xlsx")) + glob.glob(os.path.join(SAMPLES_DIR, "*.csv"))

# --- Auto-generate samples if none exist ---
if not sample_files:
    print(f"No files found in '{SAMPLES_DIR}'. Generating demo samples...")
    
    # 1. Realistic Financial Sheet (from your utils)
    print(" - Generating financial_demo.xlsx...")
    df_real = create_realistic_spreadsheet()
    df_real.to_excel(os.path.join(SAMPLES_DIR, "financial_demo.xlsx"), index=False)
    
    # 2. Dense Random Data (CSV)
    print(" - Generating dense_random.csv...")
    df_dense = pd.DataFrame(np.random.randn(100, 15), columns=[f"Metric_{i}" for i in range(15)])
    df_dense.to_csv(os.path.join(SAMPLES_DIR, "dense_random.csv"), index=False)
    
    # 3. Highly Sparse Sheet
    print(" - Generating sparse_corners.csv...")
    df_sparse = pd.DataFrame("", index=range(500), columns=[f"Col_{i}" for i in range(20)])
    df_sparse.iloc[0, 0] = "TopLeft"
    df_sparse.iloc[250, 10] = "Center"
    df_sparse.iloc[499, 19] = "BottomRight"
    df_sparse.to_csv(os.path.join(SAMPLES_DIR, "sparse_corners.csv"), index=False)
    
    # Refresh file list
    sample_files = glob.glob(os.path.join(SAMPLES_DIR, "*.xlsx")) + glob.glob(os.path.join(SAMPLES_DIR, "*.csv"))

results = []

print(f"\nStarting benchmark on {len(sample_files)} files...")
print("-" * 50)

for file in sample_files:
    print(f"Processing: {os.path.basename(file)}...")
    try:
        if file.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
            
        sllm = SpreadsheetLLM()
        
        # Measure Memory & Time
        start_time = time.time()
        # memory_usage returns (mem_usage, retval)
        mem_usage, _ = memory_usage((sllm.compress_spreadsheet, (df,)), retval=True, max_usage=True)
        elapsed = time.time() - start_time
        
        # Get Compression Stats
        stats = sllm.get_encoding_stats(df)
        
        results.append({
            "file": os.path.basename(file),
            "rows": df.shape[0],
            "cols": df.shape[1],
            "cells": df.shape[0] * df.shape[1],
            "compression_ratio": stats["compression_ratio"],
            "token_reduction": stats["token_reduction_ratio"],
            "sparsity": stats["sparsity_percentage"],
            "time_sec": elapsed,
            "max_mem_mb": mem_usage
        })
    except Exception as e:
        print(f"  Failed to process {file}: {e}")

if not results:
    print("No results generated. Exiting.")
    exit(1)

# Save results as CSV
results_df = pd.DataFrame(results)
results_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# --- Generate Visualizations ---

# Set style
sns.set_theme(style="whitegrid")

# 1. Plot: Compression Ratio vs. Sparsity (Colored by File Size)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=results_df,
    x="sparsity",
    y="compression_ratio",
    size="cells",
    hue="file",
    sizes=(100, 1000),
    palette="viridis",
    alpha=0.7
)
plt.title("Compression Efficiency: Higher Sparsity = Higher Compression")
plt.xlabel("Sparsity (%)")
plt.ylabel("Compression Ratio (x times smaller)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "compression_vs_sparsity.png"))
print("Saved chart: compression_vs_sparsity.png")

# 2. Plot: Processing Time vs. Cells
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=results_df,
    x="cells",
    y="time_sec",
    hue="file",
    s=200,
    palette="deep"
)
plt.title("Performance Scaling")
plt.xlabel("Total Cells (Rows × Cols)")
plt.ylabel("Processing Time (seconds)")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "performance_scaling.png"))
print("Saved chart: performance_scaling.png")

# 3. Plot: Max Memory Usage Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(
    data=results_df.sort_values("max_mem_mb", ascending=False),
    x="file",
    y="max_mem_mb",
    palette="crest"
)
plt.title("Memory Consumption by File")
plt.xlabel("Filename")
plt.ylabel("Peak Memory Usage (MB)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "memory_usage.png"))
print("Saved chart: memory_usage.png")

print(f"\nDone! Charts saved to {CHARTS_DIR}")