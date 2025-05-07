import pandas as pd
import numpy as np
import base64
from io import BytesIO

def compare_data(df1, df2):
    # Ensure PATTERN column exists
    for df in [df1, df2]:
        if 'PATTERN' not in df.columns:
            df['PATTERN'] = "-"

    # Temporary rename for merging
    df1 = df1.rename(columns={'Brand': 'Brand_A', 'Price': 'Price_A', 'PATTERN': 'PATTERN_A'})
    df2 = df2.rename(columns={'Brand': 'Brand_B', 'Price': 'Price_B', 'PATTERN': 'PATTERN_B'})

    # Merge data
    merged = pd.merge(df1, df2, on="SIZE", how="inner")

    # Get actual brand names
    brand_a = merged['Brand_A'].iloc[0]
    brand_b = merged['Brand_B'].iloc[0]

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

    # Determine cheaper brand
# In compare_data() function, fix the np.select() call:
    merged['Cheaper_Brand'] = np.select(
        [
            merged[price_cols[0]] < merged[price_cols[1]], 
            merged[price_cols[1]] < merged[price_cols[0]]
        ],
        [brand_a, brand_b],
        default="Same"
    )
    merged['Price_Diff'] = (merged[price_cols[0]] - merged[price_cols[1]]).abs().round(2)

    return merged[['SIZE', f'PATTERN_{brand_a}', f'Price_{brand_a}',
                  f'PATTERN_{brand_b}', f'Price_{brand_b}', 
                  'Cheaper_Brand', 'Price_Diff']]

def download_comparison_excel(df):
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Comparison')
            
            workbook = writer.book
            worksheet = writer.sheets['Comparison']
            green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})

            # Modified: Exclude Price_Diff from price columns
            price_cols = [col for col in df.columns 
                         if col.startswith('Price_') and col != 'Price_Diff']
            
            if len(price_cols) != 2:
                raise ValueError(f"Need exactly 2 price columns. Found: {price_cols}")

            # Rest of the function remains the same
            for row_num in range(1, len(df)+1):
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

        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="Tire_Comparison.xlsx">📥 Download Excel</a>'
    except Exception as e:
        return f"Error generating Excel: {str(e)}"
