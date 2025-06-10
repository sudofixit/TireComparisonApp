import streamlit as st
import pandas as pd
from preprocessing import process_uploaded_file, split_size_index_pattern
from comparison import compare_data, compare_data_three_files, download_comparison_excel

st.set_page_config(page_title="Tire Price Comparison", layout="wide")
st.title("🔧 Tire Price Comparison")

# --- Comparison Type Selection ---
st.header("Select Comparison Type")
comparison_type = st.radio(
    "How many suppliers do you want to compare?",
    ("2 Suppliers", "3 Suppliers"),
    horizontal=True
)

# --- 1. File Upload ---
st.header("1. Upload Supplier Files")
if comparison_type == "2 Suppliers":
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Supplier A", type=["xlsx"], key="file1")
    with col2:
        file2 = st.file_uploader("Supplier B", type=["xlsx"], key="file2")
    file3 = None
else:  # 3 Suppliers
    col1, col2, col3 = st.columns(3)
    with col1:
        file1 = st.file_uploader("Supplier A", type=["xlsx"], key="file1")
    with col2:
        file2 = st.file_uploader("Supplier B", type=["xlsx"], key="file2")
    with col3:
        file3 = st.file_uploader("Supplier C", type=["xlsx"], key="file3")

# Check if required files are uploaded
files_ready = (file1 and file2) if comparison_type == "2 Suppliers" else (file1 and file2 and file3)

if files_ready:
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
    if comparison_type == "3 Suppliers":
        df3 = load_and_preview(file3, "Supplier C")

    # --- 3. Column Mapping ---
    st.header("3. Map Columns")
    if comparison_type == "2 Suppliers":
        map_col1, map_col2 = st.columns(2)
    else:
        map_col1, map_col2, map_col3 = st.columns(3)

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

            # Pattern Detection
            pattern_keywords = ['pattern', 'tread', 'design', 'model']
            detected_pattern = detect_column(cols, pattern_keywords)
            pattern_options = ["None", "Auto-extract from size"] + cols
            if detected_pattern:
                pattern_index = pattern_options.index(detected_pattern) if detected_pattern in pattern_options else 0
            else:
                pattern_index = 0
            
            pattern_col = st.selectbox(
                "Pattern",
                pattern_options,
                index=pattern_index,
                key=f"pattern_{label}",
                help="Select pattern column or choose auto-extraction"
            )

            # Pattern Source Configuration
            if has_combined_pattern(size_col) and pattern_col == "None":
                st.info("🔍 Pattern detected in size column - will auto-extract")
                pattern_source = "Auto-extracted from size"
            elif pattern_col == "None":
                pattern_source = "No patterns"
            elif pattern_col == "Auto-extract from size":
                pattern_source = "Auto-extracted from size"
            else:
                pattern_source = f"Column: {pattern_col}"

            # Pattern Preview
            if st.checkbox(f"🔍 Preview Pattern Mapping for {label}", key=f"preview_pattern_{label}"):
                st.markdown(f"**Pattern Source: {pattern_source}**")
                
                if pattern_col == "None" or pattern_source.startswith("Auto-extracted"):
                    # Show auto-extraction preview
                    sample_sizes = df[size_col].dropna().head(5)
                    st.markdown("**Auto-extraction Preview:**")
                    
                    preview_data = []
                    for size in sample_sizes:
                        try:
                            extracted = split_size_index_pattern(size)
                            preview_data.append({
                                'Original': str(size),
                                'Extracted Size': extracted[0],
                                'Index': extracted[1], 
                                'Pattern': extracted[2]
                            })
                        except:
                            preview_data.append({
                                'Original': str(size),
                                'Extracted Size': str(size),
                                'Index': '',
                                'Pattern': ''
                            })
                    
                    preview_df = pd.DataFrame(preview_data)
                    st.dataframe(preview_df, height=200)
                    
                else:
                    # Show column-based pattern preview
                    sample_data = df[[size_col, pattern_col]].dropna().head(5)
                    sample_data.columns = ['Size', 'Pattern']
                    st.dataframe(sample_data, height=200)

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
    if comparison_type == "3 Suppliers":
        df3, brand3, size3, price3, pattern3 = get_mapping(df3, "Supplier C", file3.name, map_col3)

    # --- Pattern Display Options ---
    st.header("4. Display Options")
    col_display1, col_display2 = st.columns(2)
    
    with col_display1:
        show_patterns = st.checkbox("📋 Show Pattern Columns in Results", value=True, 
                                   help="Include pattern columns in comparison results")
    
    with col_display2:
        pattern_comparison = st.checkbox("🔄 Enable Pattern-based Matching", value=False,
                                       help="Match tires based on both size AND pattern (stricter matching)")

    # --- 5. Final Preview Before Comparison ---
    if st.checkbox("👀 Preview Processed Data", help="See how your data will look after processing"):
        st.subheader("Processed Data Preview")
        
        preview_cols = st.columns(3 if comparison_type == "3 Suppliers" else 2)
        
        suppliers_data = [
            (df1, brand1, size1, price1, pattern1, "Supplier A"),
            (df2, brand2, size2, price2, pattern2, "Supplier B")
        ]
        
        if comparison_type == "3 Suppliers":
            suppliers_data.append((df3, brand3, size3, price3, pattern3, "Supplier C"))
        
        for i, (df, brand, size, price, pattern, name) in enumerate(suppliers_data):
            with preview_cols[i]:
                st.markdown(f"**{name} Preview**")
                try:
                    # Handle pattern parameter for preview
                    pattern_param = None if (pattern == "None" or pattern == "Auto-extract from size") else pattern
                    preview_df = process_uploaded_file(
                        df.assign(Brand=df[brand]),
                        size, price,
                        pattern_param
                    )
                    st.write(f"Rows after processing: {len(preview_df)}")
                    if not preview_df.empty:
                        display_cols = ['SIZE', 'Price']
                        if show_patterns and 'PATTERN' in preview_df.columns:
                            display_cols.append('PATTERN')
                        st.dataframe(preview_df[display_cols].head(3), height=150)
                    else:
                        st.warning("No valid data after processing")
                except Exception as e:
                    st.error(f"Preview error: {str(e)}")

    # --- 6. Comparison ---
    st.header("5. Compare and Export")

    if st.button("▶ Compare Prices"):
        with st.spinner("Processing..."):
            try:
                # Handle pattern column logic for each supplier
                def get_pattern_param(pattern_col):
                    if pattern_col == "None" or pattern_col == "Auto-extract from size":
                        return None
                    else:
                        return pattern_col

                df1_clean = process_uploaded_file(
                    df1.assign(Brand=df1[brand1]),
                    size1, price1,
                    get_pattern_param(pattern1)
                )
                df2_clean = process_uploaded_file(
                    df2.assign(Brand=df2[brand2]),
                    size2, price2,
                    get_pattern_param(pattern2)
                )

                if comparison_type == "3 Suppliers":
                    df3_clean = process_uploaded_file(
                        df3.assign(Brand=df3[brand3]),
                        size3, price3,
                        get_pattern_param(pattern3)
                    )

                    if df1_clean.empty or df2_clean.empty or df3_clean.empty:
                        st.warning("⚠️ No valid data to compare after cleaning.")
                        st.stop()

                    result = compare_data_three_files(df1_clean, df2_clean, df3_clean)
                else:
                    if df1_clean.empty or df2_clean.empty:
                        st.warning("⚠️ No valid data to compare after cleaning.")
                        st.stop()

                    result = compare_data(df1_clean, df2_clean)

                # Filter columns based on display options
                display_result = result.copy()
                if not show_patterns:
                    pattern_cols = [col for col in result.columns if col.startswith('PATTERN_')]
                    display_result = display_result.drop(columns=pattern_cols)

                # --- Dynamic Column Handling ---
                price_cols = [col for col in result.columns 
                             if col.startswith('Price_') and col not in ['Price_Diff', 'Price_Pct_Diff']]
                
                # --- Highlight Logic ---
                def highlight_cheaper_price(row):
                    styles = [''] * len(display_result.columns)
                    if len(price_cols) == 2:
                        if price_cols[0] in display_result.columns and price_cols[1] in display_result.columns:
                            price_a_idx = display_result.columns.get_loc(price_cols[0])
                            price_b_idx = display_result.columns.get_loc(price_cols[1])
                            
                            if row[price_cols[0]] < row[price_cols[1]]:
                                styles[price_a_idx] = 'background-color: lightpink'
                            elif row[price_cols[1]] < row[price_cols[0]]:
                                styles[price_b_idx] = 'background-color: lightpink'
                    elif len(price_cols) == 3:
                        # For 3 suppliers, highlight the cheapest price
                        min_price = min(row[price_cols[0]], row[price_cols[1]], row[price_cols[2]])
                        for col in price_cols:
                            if col in display_result.columns and row[col] == min_price:
                                styles[display_result.columns.get_loc(col)] = 'background-color: lightgreen'
                                break
                    return styles

                st.subheader("📊 Comparison Results")
                
                # Show summary stats
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Total Matches", len(result))
                with col_stats2:
                    avg_diff = result['Price_Diff'].mean() if 'Price_Diff' in result.columns else 0
                    st.metric("Avg Price Difference", f"${avg_diff:.2f}")
                with col_stats3:
                    if 'Price_Pct_Diff' in result.columns:
                        avg_pct = result['Price_Pct_Diff'].mean() * 100
                        st.metric("Avg % Difference", f"{avg_pct:.1f}%")

                # Format and display results
                format_dict = {col: "{:.2f}" for col in price_cols + (['Price_Diff'] if 'Price_Diff' in display_result.columns else [])}
                if 'Price_Pct_Diff' in display_result.columns:
                    format_dict['Price_Pct_Diff'] = "{:.2%}"

                st.dataframe(
                    display_result.style
                        .format(format_dict)
                        .apply(highlight_cheaper_price, axis=1),
                    height=500
                )

                # --- Pattern Analysis Section ---
                if show_patterns and any(col.startswith('PATTERN_') for col in result.columns):
                    st.subheader("🔍 Pattern Analysis")
                    
                    pattern_cols = [col for col in result.columns if col.startswith('PATTERN_')]
                    
                    if len(pattern_cols) >= 2:
                        # Create tabs for different pattern views
                        tab1, tab2, tab3 = st.tabs(["📊 Pattern Distribution", "🔄 Pattern Matching", "📋 Unique Patterns"])
                        
                        with tab1:
                            st.markdown("**Pattern Distribution by Supplier:**")
                            for col in pattern_cols:
                                supplier_name = col.replace('PATTERN_', '')
                                pattern_counts = result[col].value_counts().head(10)
                                
                                if not pattern_counts.empty:
                                    st.markdown(f"**{supplier_name} - Top 10 Patterns:**")
                                    pattern_df = pd.DataFrame({
                                        'Pattern': pattern_counts.index,
                                        'Count': pattern_counts.values,
                                        'Percentage': (pattern_counts.values / len(result) * 100).round(1)
                                    })
                                    st.dataframe(pattern_df, height=200)
                        
                        with tab2:
                            st.markdown("**Pattern Matching Analysis:**")
                            
                            # Check how many exact pattern matches exist
                            if len(pattern_cols) == 2:
                                exact_matches = (result[pattern_cols[0]] == result[pattern_cols[1]]).sum()
                                total_rows = len(result)
                                match_percentage = (exact_matches / total_rows * 100) if total_rows > 0 else 0
                                
                                col_match1, col_match2 = st.columns(2)
                                with col_match1:
                                    st.metric("Exact Pattern Matches", exact_matches)
                                with col_match2:
                                    st.metric("Match Percentage", f"{match_percentage:.1f}%")
                                
                                # Show some examples
                                if exact_matches > 0:
                                    st.markdown("**Examples of Exact Pattern Matches:**")
                                    exact_match_examples = result[result[pattern_cols[0]] == result[pattern_cols[1]]][
                                        ['SIZE'] + pattern_cols
                                    ].head(5)
                                    st.dataframe(exact_match_examples, height=150)
                            
                            elif len(pattern_cols) == 3:
                                # For 3 suppliers, show pattern consistency
                                all_same = ((result[pattern_cols[0]] == result[pattern_cols[1]]) & 
                                          (result[pattern_cols[1]] == result[pattern_cols[2]])).sum()
                                two_same = (((result[pattern_cols[0]] == result[pattern_cols[1]]) |
                                           (result[pattern_cols[1]] == result[pattern_cols[2]]) |
                                           (result[pattern_cols[0]] == result[pattern_cols[2]])) & 
                                          ~((result[pattern_cols[0]] == result[pattern_cols[1]]) & 
                                            (result[pattern_cols[1]] == result[pattern_cols[2]]))).sum()
                                
                                col_m1, col_m2, col_m3 = st.columns(3)
                                with col_m1:
                                    st.metric("All 3 Match", all_same)
                                with col_m2:
                                    st.metric("2 of 3 Match", two_same)
                                with col_m3:
                                    st.metric("All Different", len(result) - all_same - two_same)
                        
                        with tab3:
                            st.markdown("**Unique Patterns Across All Suppliers:**")
                            
                            all_patterns = set()
                            for col in pattern_cols:
                                patterns = result[col].dropna().unique()
                                all_patterns.update(patterns)
                            
                            all_patterns = sorted([p for p in all_patterns if p and p != '-'])
                            
                            if all_patterns:
                                st.write(f"Total unique patterns found: **{len(all_patterns)}**")
                                
                                # Show patterns in a nice format
                                pattern_display = pd.DataFrame({
                                    'Pattern': all_patterns[:20],  # Show first 20
                                    'Index': range(1, min(21, len(all_patterns) + 1))
                                })
                                st.dataframe(pattern_display, height=300)
                                
                                if len(all_patterns) > 20:
                                    st.info(f"Showing first 20 of {len(all_patterns)} total patterns")
                            else:
                                st.warning("No patterns found in the data")

                st.markdown(download_comparison_excel(result), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Comparison failed: {str(e)}")
