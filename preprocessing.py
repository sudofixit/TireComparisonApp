import pandas as pd
import re
import streamlit as st

def split_size_index_pattern(desc):
    """
    Improved pattern extraction with better index detection
    Returns: SIZE, INDEX (load/speed), PATTERN
    """
    try:
        desc = str(desc).strip()
        
        # Main regex for common size formats
        size_pattern = re.compile(r"""
            (LT)?\s*  # Optional LT prefix
            (\d{3}/\d{2}[A-Z]?\d{1,2}|  # Standard tire sizes (265/70R17)
            \d{2,3}x\d{1,3}\.\d{1,2}|  # Flotation sizes (35x12.50)
            \d{2}\.\d{2}-?\d{2})       # Numeric patterns (16.9-30)
        """, re.VERBOSE | re.IGNORECASE)
        
        # Try to find size first
        size_match = size_pattern.search(desc)
        if size_match:
            size = size_match.group(2)
            remaining = desc.replace(size_match.group(), "").strip()
        else:
            size = ""
            remaining = desc

        # Extract index (load/speed) from remaining
        index_match = re.search(r"\b(\d{2,3}[A-Z]{1})\b", remaining)
        if index_match:
            index = index_match.group(1)
            pattern = remaining.replace(index, "").strip()
        else:
            index = ""
            pattern = remaining

        return pd.Series([size, index, pattern])

    except:
        return pd.Series(["", "", desc])

def clean_price_column(series):
    """Unchanged from original"""
    cleaned = (
        series.astype(str)
              .str.replace(r"[^\d.,]", "", regex=True)
              .str.replace(",", ".", regex=False)
              .str.extract(r"(\d+\.\d+|\d+)")[0]
    )
    return pd.to_numeric(cleaned, errors="coerce")

def process_uploaded_file(df, size_col, price_col, pattern_col=None):
    """
    Simplified processing flow with clear pattern handling
    """
    result = pd.DataFrame()

    # Basic cleaning
    result["Brand"] = df["Brand"].astype(str).str.strip()
    result["SIZE"] = df[size_col].astype(str).str.strip().str.replace(r"\s+", "", regex=True).str.upper()
    
    # Price cleaning
    df[price_col] = clean_price_column(df[price_col])
    result["Price"] = df[price_col]
    
    # Handle Pattern column or extract from size
    if pattern_col and pattern_col != "None" and pattern_col in df.columns:
        # Simple cleanup if pattern column exists
        result["PATTERN"] = df[pattern_col].astype(str).str.strip()
    else:
        # Split size into components when no pattern column
        extracted = df[size_col].astype(str).apply(split_size_index_pattern)
        result[["SIZE", "INDEX", "PATTERN"]] = extracted
        result["PATTERN"] = result["PATTERN"].str.strip()

    # Final filtering
    result = result[(result["Price"] > 0) & (result["SIZE"].str.strip() != "")]
    result.dropna(subset=["Price", "SIZE"], inplace=True)
    
    return result