import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# Run this script with: python scripts/import.py
# It loads every CSV in data/raw/ into the local SQLite database.
RAW_DIR = Path("data/raw")
DB_PATH = Path("db/olist.db")

# Create a SQLite engine that pandas will use to write tables.
engine = create_engine(f"sqlite:///{DB_PATH}")

def ingest_all():
    # Find every CSV file in the raw data folder.
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSVs found in data/raw/")

    log = []
    for file in csv_files:
        # Convert the file name into a database table name.
        table_name = "raw_" + file.stem.replace("olist_", "").replace("_dataset", "")
        
        # Read the CSV and replace the matching table with the latest data.
        df = pd.read_csv(file)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        log.append((file.name, table_name, len(df)))
        print(f"Loaded {file.name} -> {table_name} ({len(df)} rows)")

    return log

if __name__ == "__main__":
    # Execute the full ingest process when the script is run directly.
    ingest_all()
