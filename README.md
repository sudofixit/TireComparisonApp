# 🚗 TireComparisonApp

TireComparisonApp is a user-friendly web application built with **Streamlit** that allows users to compare tire prices and patterns between two different Excel files. It's designed for businesses or users who want to quickly identify the cheaper brand, price differences, and pattern differences between suppliers.

---

## 🔧 Features

- Upload two Excel files with tire data.
- Automatically detect and clean tire size and pattern formats.
- Compare prices between matching sizes and patterns.
- Highlight cheaper brand in green.
- Calculate price differences.
- Export results to Excel.
- Works with or without brand and pattern columns.

---

## 📂 File Requirements

Each Excel file should contain:
- Tire Size (e.g., `195/60 R16`)
- Pattern (e.g., `SPLM705`, optionally prefixed by speed/load index like `84H`)
- Price (numeric)

Optional columns:
- Brand (if present, column names will be shown as `Price_BrandName`, `Pattern_BrandName`)

---

## 🚀 How to Run Locally

Make sure you have Python 3.7+ installed.

### 1. Clone the repository

git clone https://github.com/sudofixit/TireComparisonApp.git
cd TireComparisonApp

python -m venv venv
venv\Scripts\activate  # On Windows

pip install -r requirements.txt

streamlit run app.py


