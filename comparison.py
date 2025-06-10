import pandas as pd
import numpy as np
import base64
from io import BytesIO

def compare_data(df1, df2):
    # Ensure PATTERN column exists
    for df in [df1, df2]:
        if 'PATTERN' not in df.columns:
            df['PATTERN'] = "-"

    # Check if dataframes are empty
    if df1.empty or df2.empty:
        raise ValueError("One or both input dataframes are empty")

    # Get brand names before renaming
    brand_a = df1['Brand'].iloc[0] if not df1.empty else "Unknown_A"
    brand_b = df2['Brand'].iloc[0] if not df2.empty else "Unknown_B"

    # Temporary rename for merging
    df1 = df1.rename(columns={'Brand': 'Brand_A', 'Price': 'Price_A', 'PATTERN': 'PATTERN_A'})
    df2 = df2.rename(columns={'Brand': 'Brand_B', 'Price': 'Price_B', 'PATTERN': 'PATTERN_B'})

    # Merge data
    merged = pd.merge(df1, df2, on="SIZE", how="inner")

    # Check if merge resulted in empty dataframe
    if merged.empty:
        raise ValueError("No matching tire sizes found between the two suppliers")

    # Create dynamic column names
    merged = merged.rename(columns={
        'PATTERN_A': f'PATTERN_{brand_a}',
        'Price_A': f'Price_{brand_a}',
        'PATTERN_B': f'PATTERN_{brand_b}',
        'Price_B': f'Price_{brand_b}'
    })

    # Clean numeric values
    price_cols = [f'Price_{brand_a}', f'Price_{brand_b}']
    merged[price_cols] = merged[price_cols].apply(pd.to_numeric, errors='coerce')
    merged = merged.dropna(subset=price_cols)

    # Check if any data remains after cleaning
    if merged.empty:
        raise ValueError("No valid price data found after cleaning")

    # Determine cheaper brand
    merged['Cheaper_Brand'] = np.select(
        [
            merged[price_cols[0]] < merged[price_cols[1]], 
            merged[price_cols[1]] < merged[price_cols[0]]
        ],
        [brand_a, brand_b],
        default="Same"
    )
    merged['Price_Diff'] = (merged[price_cols[0]] - merged[price_cols[1]]).abs().round(2)

    cheaper_prices = merged[[price_cols[0], price_cols[1]]].min(axis=1)
    # Avoid division by zero
    cheaper_prices = cheaper_prices.replace(0, np.nan)
    merged['Price_Pct_Diff'] = (merged['Price_Diff'] / cheaper_prices).round(4)

    return merged[['SIZE', f'PATTERN_{brand_a}', f'Price_{brand_a}',
                  f'PATTERN_{brand_b}', f'Price_{brand_b}', 
                  'Cheaper_Brand', 'Price_Diff', 'Price_Pct_Diff']]

def compare_data_three_files(df1, df2, df3):
    """Compare data from three suppliers"""
    # Ensure PATTERN column exists
    for df in [df1, df2, df3]:
        if 'PATTERN' not in df.columns:
            df['PATTERN'] = "-"

    # Check if dataframes are empty
    if df1.empty or df2.empty or df3.empty:
        raise ValueError("One or more input dataframes are empty")

    # Get brand names before renaming
    brand_a = df1['Brand'].iloc[0] if not df1.empty else "Unknown_A"
    brand_b = df2['Brand'].iloc[0] if not df2.empty else "Unknown_B"
    brand_c = df3['Brand'].iloc[0] if not df3.empty else "Unknown_C"

    # Create separate dataframes with only the columns we need to avoid merge conflicts
    df1_clean = df1[['SIZE', 'Brand', 'Price', 'PATTERN']].copy()
    df2_clean = df2[['SIZE', 'Brand', 'Price', 'PATTERN']].copy()
    df3_clean = df3[['SIZE', 'Brand', 'Price', 'PATTERN']].copy()
    
    # Rename columns with suffixes for merging
    df1_clean = df1_clean.rename(columns={'Brand': 'Brand_A', 'Price': 'Price_A', 'PATTERN': 'PATTERN_A'})
    df2_clean = df2_clean.rename(columns={'Brand': 'Brand_B', 'Price': 'Price_B', 'PATTERN': 'PATTERN_B'})
    df3_clean = df3_clean.rename(columns={'Brand': 'Brand_C', 'Price': 'Price_C', 'PATTERN': 'PATTERN_C'})

    # Merge data step by step with explicit suffixes
    merged = pd.merge(df1_clean, df2_clean, on="SIZE", how="inner", suffixes=('', ''))
    if merged.empty:
        raise ValueError("No matching tire sizes found between suppliers A and B")
    
    merged = pd.merge(merged, df3_clean, on="SIZE", how="inner", suffixes=('', ''))
    if merged.empty:
        raise ValueError("No matching tire sizes found across all three suppliers")

    # Create dynamic column names
    merged = merged.rename(columns={
        'PATTERN_A': f'PATTERN_{brand_a}',
        'Price_A': f'Price_{brand_a}',
        'PATTERN_B': f'PATTERN_{brand_b}',
        'Price_B': f'Price_{brand_b}',
        'PATTERN_C': f'PATTERN_{brand_c}',
        'Price_C': f'Price_{brand_c}'
    })

    # Clean numeric values
    price_cols = [f'Price_{brand_a}', f'Price_{brand_b}', f'Price_{brand_c}']
    merged[price_cols] = merged[price_cols].apply(pd.to_numeric, errors='coerce')
    merged = merged.dropna(subset=price_cols)

    # Check if any data remains after cleaning
    if merged.empty:
        raise ValueError("No valid price data found after cleaning")

    # Determine cheapest brand
    def get_cheapest_brand(row):
        prices = [row[price_cols[0]], row[price_cols[1]], row[price_cols[2]]]
        brands = [brand_a, brand_b, brand_c]
        min_price = min(prices)
        cheapest_idx = prices.index(min_price)
        return brands[cheapest_idx]

    merged['Cheapest_Brand'] = merged.apply(get_cheapest_brand, axis=1)

    # Calculate price differences from the cheapest
    merged['Min_Price'] = merged[price_cols].min(axis=1)
    merged['Max_Price'] = merged[price_cols].max(axis=1)
    merged['Price_Diff'] = (merged['Max_Price'] - merged['Min_Price']).round(2)
    
    # Avoid division by zero
    merged['Min_Price'] = merged['Min_Price'].replace(0, np.nan)
    merged['Price_Pct_Diff'] = (merged['Price_Diff'] / merged['Min_Price']).round(4)

    # Return the results
    return merged[['SIZE', 
                  f'PATTERN_{brand_a}', f'Price_{brand_a}',
                  f'PATTERN_{brand_b}', f'Price_{brand_b}', 
                  f'PATTERN_{brand_c}', f'Price_{brand_c}',
                  'Cheapest_Brand', 'Price_Diff', 'Price_Pct_Diff']]

def download_comparison_excel(df):
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Comparison')
            
            workbook = writer.book
            worksheet = writer.sheets['Comparison']
            green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})

            # Modified: Exclude Price_Diff from price columns
            # Modified: Exclude both Price_Diff and Price_Pct_Diff
            price_cols = [col for col in df.columns 
                        if col.startswith('Price_') 
                        and col not in ['Price_Diff', 'Price_Pct_Diff']]
            
            # Handle both 2-file and 3-file scenarios
            if len(price_cols) < 2:
                raise ValueError(f"Need at least 2 price columns. Found: {price_cols}")
            
            if 'Price_Pct_Diff' in df.columns:
                pct_col_idx = df.columns.get_loc('Price_Pct_Diff')
                percent_format = workbook.add_format({'num_format': '0.00%'})
                worksheet.set_column(pct_col_idx, pct_col_idx, None, percent_format)

            # Highlight cheapest prices
            for row_num in range(1, len(df)+1):
                if len(price_cols) == 2:
                    # 2-file comparison
                    price_a = df.at[row_num-1, price_cols[0]]
                    price_b = df.at[row_num-1, price_cols[1]]
                    
                    if pd.notna(price_a) and pd.notna(price_b):
                        col_idx = price_cols[0] if price_a < price_b else price_cols[1]
                        worksheet.write(
                            row_num,
                            df.columns.get_loc(col_idx),
                            price_a if price_a < price_b else price_b,
                            green_format
                        )
                elif len(price_cols) == 3:
                    # 3-file comparison
                    prices = [df.at[row_num-1, col] for col in price_cols]
                    if all(pd.notna(price) for price in prices):
                        min_price = min(prices)
                        min_idx = prices.index(min_price)
                        col_name = price_cols[min_idx]
                        worksheet.write(
                            row_num,
                            df.columns.get_loc(col_name),
                            min_price,
                            green_format
                        )

        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="Tire_Comparison.xlsx">📥 Download Excel</a>'
    except Exception as e:
        return f"Error generating Excel: {str(e)}"
