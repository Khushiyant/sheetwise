"""Utility functions and demo data creation."""

import numpy as np
import pandas as pd


def create_realistic_spreadsheet():
    """Create a realistic large, sparse spreadsheet with multiple tables"""
    # Create a larger 100x30 DataFrame with very sparse data (simulating real spreadsheets)
    df = pd.DataFrame(index=range(100), columns=[f"Col_{i}" for i in range(30)])

    # Fill with mostly empty data
    df = df.fillna("")

    # Add a financial summary table (rows 5-15, cols 2-8)
    financial_headers = [
        "",
        "",
        "Financial Summary",
        "2020",
        "2021",
        "2022",
        "2023",
        "Total",
        "",
    ]
    financial_data = [
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "Q4 Financial Report", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "Metrics", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "Revenue ($M)", 12.5, 14.5, 16.2, 18.9, 62.1, ""],
        ["", "", "Expenses ($M)", 9.5, 10.8, 11.8, 13.5, 45.6, ""],
        ["", "", "Profit ($M)", 3.0, 3.7, 4.4, 5.4, 16.5, ""],
        ["", "", "Growth %", "15.5%", "23.3%", "18.9%", "22.7%", "", ""],
        ["", "", "Market Share %", "12.3%", "13.1%", "14.5%", "15.8%", "", ""],
        ["", "", "", "", "", "", "", "", ""],
        ["", "", "Notes: All figures in millions USD", "", "", "", "", "", ""],
    ]

    for i in range(min(len(financial_data), 100 - 5)):
        for j in range(min(len(financial_data[i]), 30)):
            if financial_data[i][j] != "":
                df.iloc[i + 5, j] = financial_data[i][j]

    # Add a large products table (rows 25-65, cols 10-20) with many empty rows between
    products = [
        'Laptop Pro 15"',
        "Desktop Workstation",
        "Tablet Ultra",
        "Smartphone 5G",
        "Smartwatch Pro",
        "Wireless Headphones",
        "Bluetooth Speaker",
        "Digital Camera",
        '4K Monitor 27"',
        "Gaming Keyboard",
        "Precision Mouse",
        "HD Webcam",
        "Fast Charger",
        "USB-C Hub",
        "External SSD",
        "Wireless Router",
        'Smart TV 55"',
        "Soundbar Premium",
        "Gaming Console",
        "VR Headset",
    ]

    # Headers for products
    product_headers = [
        "Product Name",
        "SKU",
        "Quantity",
        "Unit Price",
        "Total Value",
        "Status",
        "Category",
        "Supplier",
        "Last Updated",
        "Notes",
    ]

    for j, header in enumerate(product_headers):
        if j + 10 < 30:
            df.iloc[24, j + 10] = header

    # Add products with gaps (simulating real sparse data)
    product_row = 25
    for i, product in enumerate(products):
        if product_row < 95:
            df.iloc[product_row, 10] = product
            df.iloc[product_row, 11] = f"SKU-{1000+i:04d}"
            df.iloc[product_row, 12] = np.random.randint(0, 500)
            df.iloc[product_row, 13] = f"${np.random.randint(50, 2000)}"
            df.iloc[product_row, 14] = f"${np.random.randint(1000, 50000)}"
            df.iloc[product_row, 15] = np.random.choice(
                ["In Stock", "Low Stock", "Out of Stock", "Discontinued"]
            )
            df.iloc[product_row, 16] = np.random.choice(
                ["Electronics", "Accessories", "Computing", "Mobile"]
            )
            df.iloc[product_row, 17] = np.random.choice(
                ["TechCorp", "GlobalSupply", "MegaVendor", "EliteManuf"]
            )
            df.iloc[
                product_row, 18
            ] = f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"

            # Add some gaps between products (realistic sparsity)
            if i % 3 == 0:
                product_row += np.random.randint(2, 4)  # Skip 1-3 rows
            else:
                product_row += 1

    # Add customer demographics table (rows 70-85, cols 5-12)
    regions = [
        "North America",
        "Europe",
        "Asia Pacific",
        "South America",
        "Middle East",
        "Africa",
        "Australia",
        "Nordic Countries",
        "Eastern Europe",
        "Caribbean",
    ]

    # Demographics headers
    demo_headers = [
        "Region",
        "Active Customers",
        "Retention Rate",
        "Avg Age",
        "Avg Annual Spend",
        "Growth Rate",
        "Satisfaction Score",
        "Notes",
    ]

    for j, header in enumerate(demo_headers):
        if j + 5 < 30:
            df.iloc[69, j + 5] = header

    for i, region in enumerate(regions):
        row_idx = 70 + i
        if row_idx < 95:
            df.iloc[row_idx, 5] = region
            df.iloc[row_idx, 6] = f"{np.random.randint(1000, 50000):,}"
            df.iloc[row_idx, 7] = f"{np.random.randint(65, 95)}%"
            df.iloc[row_idx, 8] = np.random.randint(25, 65)
            df.iloc[row_idx, 9] = f"${np.random.randint(500, 5000):,}"
            df.iloc[row_idx, 10] = f"{np.random.randint(-5, 25)}%"
            df.iloc[row_idx, 11] = f"{np.random.uniform(3.5, 5.0):.1f}/5"

    # Add scattered metadata and notes throughout (realistic document structure)
    df.iloc[0, 0] = "CONFIDENTIAL - Q4 2023 Business Analytics Report"
    df.iloc[1, 0] = "Generated: December 31, 2023"
    df.iloc[2, 0] = "Department: Business Intelligence"
    df.iloc[0, 25] = "Page 1 of 3"
    df.iloc[18, 0] = "--- End Financial Section ---"
    df.iloc[22, 10] = "INVENTORY ANALYSIS"
    df.iloc[23, 10] = "Data as of: Dec 30, 2023"
    df.iloc[67, 5] = "=== CUSTOMER DEMOGRAPHICS ==="
    df.iloc[68, 5] = "Source: CRM Database"
    df.iloc[90, 0] = "Report compiled by: Analytics Team"
    df.iloc[95, 0] = "Next review date: Q1 2024"
    df.iloc[99, 0] = "--- END OF REPORT ---"

    # Add some formulas and references (as text, simulating Excel formulas)
    df.iloc[13, 8] = "=SUM(D6:G6)"  # Total revenue formula
    df.iloc[14, 8] = "=SUM(D7:G7)"  # Total expenses formula

    return df
