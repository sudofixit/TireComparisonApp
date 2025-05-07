import streamlit as st
import pandas as pd
from preprocessing import process_uploaded_file
from comparison import compare_data, download_comparison_excel

st.set_page_config(page_title="Tire Price Comparison", layout="wide")
st.title("🔧 Tire Price Comparison")

# --- 1. File Upload ---
st.header("1. Upload Supplier Files")
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Supplier A", type=["xlsx"], key="file1")
with col2:
    file2 = st.file_uploader("Supplier B", type=["xlsx"], key="file2")

if file1 and file2:
    # --- 2. Data Loading ---
    def load_and_preview(file, label):
        try:
            xl = pd.ExcelFile(file)
            raw = pd.concat([xl.parse(sheet, header=None) for sheet in xl.sheet_names])

            header_keywords = ["price", "size", "pattern", "code", "weight"]
            best_header_row = 0
            max_matches = 0

            for i in range(min(5, len(raw))):
                matches = sum(
                    any(kw in str(cell).lower() for kw in header_keywords)
                    for cell in raw.iloc[i]
                )
                if matches > max_matches:
                    max_matches = matches
                    best_header_row = i

            df = pd.concat([xl.parse(sheet, header=best_header_row) for sheet in xl.sheet_names])
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            st.success(f"✅ {label} Loaded ({len(df)} rows)")
            st.dataframe(df.head(3), height=150)
            return df
        except Exception as e:
            st.error(f"❌ Failed to load {label}: {str(e)}")
            return pd.DataFrame()

    st.header("2. Preview Data")
    df1 = load_and_preview(file1, "Supplier A")
    df2 = load_and_preview(file2, "Supplier B")

    # --- 3. Column Mapping ---
    st.header("3. Map Columns")
    map_col1, map_col2 = st.columns(2)

    def get_mapping(df, label, file_key, col):
        def detect_column(cols, keywords):
            """Find first column containing any keyword (case-insensitive)"""
            cols_lower = [c.lower() for c in cols]
            for keyword in keywords:
                if keyword in cols_lower:
                    return cols[cols_lower.index(keyword)]
                for i, c in enumerate(cols):
                    if keyword in c.lower():
                        return cols[i]
            return None

        def has_combined_pattern(size_col):
            """Check if size column contains pattern data"""
            if not size_col or size_col == "None":
                return False
            try:
                sample = df[size_col].astype(str).str.contains(r'\d+[A-Za-z]{2,}', regex=True)
                return sample.any()
            except:
                return False

        with col:
            cols = df.columns.tolist()
            st.markdown(f"**{label} Column Mapping**")

            # Brand Detection
            if 'Brand' not in cols:
                brand_val = st.text_input("Brand (manually enter)", key=f"manual_brand_{label}")
                if not brand_val:
                    st.warning("Brand name is required.")
                    st.stop()
                df.insert(0, 'Brand', brand_val)
                cols.insert(0, 'Brand')
                brand_col = 'Brand'
            else:
                detected_brand = detect_column(cols, ['brand'])
                brand_col = st.selectbox(
                    "Brand", 
                    cols,
                    index=cols.index(detected_brand) if detected_brand else 0,
                    key=f"brand_{label}"
                )
                df['Brand'] = df[brand_col]

            # Size Detection
            size_keywords = ['size description', 'description', 'size']
            detected_size = detect_column(cols, size_keywords)
            size_col = st.selectbox(
                "Size",
                cols,
                index=cols.index(detected_size) if detected_size else 0,
                key=f"size_{label}"
            )

            # Price Detection
            price_priority = ['price', 'fob', 'cpt', 'cost']
            detected_price = detect_column(cols, price_priority)
            price_col = st.selectbox(
                "Price",
                cols,
                index=cols.index(detected_price) if detected_price else 0,
                key=f"price_{label}"
            )

            # Pattern Handling
            detected_pattern = detect_column(cols, ['pattern'])
            if has_combined_pattern(size_col):
                pattern_col = "None"
            else:
                pattern_options = ["None"] + cols
                pattern_index = pattern_options.index(detected_pattern) if detected_pattern else 0
                pattern_col = st.selectbox(
                    "Pattern (optional)",
                    pattern_options,
                    index=pattern_index,
                    key=f"pattern_{label}"
                )

            # Save mapping
            if st.button(f"💾 Save {label} Mapping", key=f"save_{label}"):
                st.session_state[f"mapping_{file_key}"] = {
                    'brand': brand_col,
                    'size': size_col,
                    'price': price_col,
                    'pattern': pattern_col
                }
                st.success("✅ Mapping saved!")

            if f"mapping_{file_key}" in st.session_state:
                if st.checkbox(f"Use saved mapping for {label}", value=True):
                    saved = st.session_state[f"mapping_{file_key}"]
                    return df, saved['brand'], saved['size'], saved['price'], saved['pattern']

            return df, brand_col, size_col, price_col, pattern_col

    df1, brand1, size1, price1, pattern1 = get_mapping(df1, "Supplier A", file1.name, map_col1)
    df2, brand2, size2, price2, pattern2 = get_mapping(df2, "Supplier B", file2.name, map_col2)

    # --- 4. Comparison ---
    st.header("4. Compare and Export")

    if st.button("▶ Compare Prices"):
        with st.spinner("Processing..."):
            try:
                df1_clean = process_uploaded_file(
                    df1.assign(Brand=df1[brand1]),
                    size1, price1,
                    pattern1 if pattern1 != "None" else None
                )
                df2_clean = process_uploaded_file(
                    df2.assign(Brand=df2[brand2]),
                    size2, price2,
                    pattern2 if pattern2 != "None" else None
                )

                if df1_clean.empty or df2_clean.empty:
                    st.warning("⚠️ No valid data to compare after cleaning.")
                    st.stop()

                result = compare_data(df1_clean, df2_clean)

                # --- Dynamic Column Handling ---
                price_cols = [col for col in result.columns 
                             if col.startswith('Price_') and col != 'Price_Diff']
                
                # --- Highlight Logic ---
                def highlight_cheaper_price(row):
                    styles = [''] * len(result.columns)
                    price_a_idx = result.columns.get_loc(price_cols[0])
                    price_b_idx = result.columns.get_loc(price_cols[1])
                    
                    if row[price_cols[0]] < row[price_cols[1]]:
                        styles[price_a_idx] = 'background-color: lightpink'
                    elif row[price_cols[1]] < row[price_cols[0]]:
                        styles[price_b_idx] = 'background-color: lightpink'
                    return styles

                st.subheader("📊 Comparison Results")
                st.dataframe(
                    result.style
                        .format({col: "{:.2f}" for col in price_cols + ['Price_Diff']})
                        .apply(highlight_cheaper_price, axis=1),
                    height=500
                )

                st.markdown(download_comparison_excel(result), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Comparison failed: {str(e)}")
