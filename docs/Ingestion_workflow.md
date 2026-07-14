# Ingestion Workflow

This project keeps the original Kaggle CSV files in `data/raw/` and loads them
into a local SQLite database for analysis, cleaning, and Phase 3 preprocessing.

1. Keep source files unchanged
   - The 9 Olist CSV files stay in `data/raw/` exactly as downloaded.
   - These files are treated as the immutable source of truth.

2. Run the ingestion script
   - Use `python scripts/import.py` to load the CSV files into SQLite.
   - The script reads every `.csv` file in `data/raw/` with pandas.

3. Create raw tables in SQLite
   - Each file is written to `db/olist.db` as a `raw_<table>` table.
   - Example: `olist_orders_dataset.csv` becomes `raw_orders`.

4. Verify the load
   - Row counts in SQLite should match the row counts in the source CSV files.
   - This confirms the ingestion step did not drop or duplicate data.

5. Clean downstream data
   - `scripts/clean.py` reads from the `raw_<table>` tables.
   - It applies deduplication, type conversion, and null handling.
   - Cleaned results are written to `clean_<table>` tables in the same database.
   - The cleaned CSV outputs are also saved in `data/cleaned/`.

6. Run Phase 3 preprocessing and EDA
   - Use `python scripts/phase3.py` to load the `clean_<table>` tables from `db/olist.db`.
   - The script merges the cleaned tables into one `analysis_table` for downstream analysis.
   - It performs post-merge sanity checks, outlier handling, feature engineering, scaling, and summary statistics.
   - Charts are saved in `outputs/figures/` and the plain-English summary is saved to `outputs/phase3_summary.txt`.

7. Preserve separation of concerns
   - Raw files and `raw_` tables are never modified after ingestion.
   - All transformations happen in the cleaning step, not during ingestion.

#Phase 1


## Commands To Demonstrate Phases 2 and 3

Run these commands from the project root in PowerShell:

```powershell
cd "C:\Users\Moandemah Foundation\Desktop\new school\second semester\CSIT608\olist_project"
.\venv\Scripts\Activate.ps1
python scripts\ingest.py
python scripts\clean.py
python scripts\phase3.py
Get-ChildItem data\raw
Get-ChildItem data\cleaned
Get-ChildItem outputs\figures
```

Use these commands to show evidence that the workflow worked:

```powershell
Get-Content docs\ingestion_workflow.md
Import-Csv data\cleaned\clean_orders.csv | Select-Object -First 10
Import-Csv data\cleaned\clean_customers.csv | Select-Object -First 10
Get-Content data\cleaned\clean_orders.csv -TotalCount 5
Get-Content outputs\phase3_summary.txt
```

Compare raw and cleaned tables side by side:

```powershell
python -c "import sqlite3, pandas as pd; conn = sqlite3.connect('db/olist.db'); print('RAW TABLE:'); print(pd.read_sql_query('SELECT * FROM raw_orders LIMIT 10', conn)); print(); print('CLEANED TABLE:'); print(pd.read_sql_query('SELECT * FROM clean_orders LIMIT 10', conn))"
```

This command shows the first 10 rows from both the `raw_orders` and `clean_orders` tables so you can see what the cleaning process did (deduplication, type conversion, null handling).

To confirm Phase 3 wrote the analysis table, run:

```powershell
python -c "import sqlite3, pandas as pd; conn = sqlite3.connect('db/olist.db'); print(pd.read_sql_query('SELECT COUNT(*) AS rows FROM analysis_table', conn))"
```

## Phase 3 Analysis Commands

Use these commands after the raw CSVs have been ingested and cleaned. The raw CSV files are the source data, but Phase 3 computes the analysis results from the cleaned database tables that were created from them.

Run the full Phase 3 pipeline, which produces missing-data checks, duplicate checks, outlier handling, transformation, scaling, feature engineering, descriptive statistics, distributions, correlation, uncertainty, and pattern discovery:

```powershell
python scripts\phase3.py
```

Inspect missing data from the raw CSV source files:

```powershell
python -c "import pandas as pd; df = pd.read_csv('data/raw/olist_order_reviews_dataset.csv'); print(df.isna().sum().sort_values(ascending=False))"
```

Check duplicates in a raw CSV file:

```powershell
python -c "import pandas as pd; df = pd.read_csv('data/raw/olist_orders_dataset.csv'); print('duplicate rows:', df.duplicated().sum())"
```

Preview outlier candidates from a numeric raw CSV column:

```powershell
python -c "import pandas as pd; df = pd.read_csv('data/raw/olist_order_items_dataset.csv'); q1 = df['price'].quantile(0.25); q3 = df['price'].quantile(0.75); iqr = q3 - q1; lower = q1 - 1.5 * iqr; upper = q3 + 1.5 * iqr; print('outliers:', ((df['price'] < lower) | (df['price'] > upper)).sum()); print('bounds:', lower, upper)"
```

Confirm transformation, scaling, and feature engineering by checking the saved analysis table:

```powershell
python -c "import sqlite3, pandas as pd; conn = sqlite3.connect('db/olist.db'); df = pd.read_sql_query('SELECT delivery_days, delivery_delay_days, was_late, item_total_value, freight_ratio, price_log, freight_value_log, item_total_value_log, price_scaled, freight_value_scaled, delivery_days_scaled FROM analysis_table LIMIT 10', conn); print(df)"
```

Get descriptive statistics from the analysis table:

```powershell
python -c "import sqlite3, pandas as pd; conn = sqlite3.connect('db/olist.db'); df = pd.read_sql_query('SELECT price, freight_value, delivery_days, review_score, item_total_value FROM analysis_table', conn); print(df.describe())"
```

Confirm distributions, correlation, uncertainty, and pattern discovery outputs:

```powershell
Get-ChildItem outputs\figures
Get-Content outputs\phase3_summary.txt
```

The figures folder contains the distribution plots, boxplots, correlation heatmap, and weekday pattern chart. The summary file includes the uncertainty statement and the main pattern-discovery insights.

If you need to show the database file is present, use:

```powershell
Get-ChildItem db


#Phase 5 Commands
#Install plotly


```pip install streamlit plotly
streamlit run scripts/phase5.py

#start streamlit plotly

```.\.venv\Scripts\python.exe -m streamlit run scripts/phase5.py

#phase 6 commands

```how to run phase 6 using streamlit
#command
.\.venv\Scripts\python.exe -m streamlit run scripts/phase6.py
