import sqlite3
import pandas as pd
from pathlib import Path

# Connect to the SQLite database created during ingestion (stdlib sqlite3 —
# no external DB driver required).
DB_PATH = Path("db/olist.db")
DB_PATH.parent.mkdir(exist_ok=True, parents=True)
conn = sqlite3.connect(DB_PATH)

# Write cleaned CSV files to a separate output folder.
CLEAN_DIR = Path("data/cleaned")
CLEAN_DIR.mkdir(exist_ok=True, parents=True)

# Folder holding the original raw CSVs, used as a fallback if a raw_* table
# is missing from the database (e.g. geolocation / category translation
# were never ingested into SQLite).
RAW_CSV_DIR = Path("data/raw")


def _table_exists(table_name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone() is not None


def load_raw(table_name: str, csv_filename: str) -> pd.DataFrame:
    """Load a raw table from SQLite; fall back to the raw CSV if the
    table doesn't exist in the database yet."""
    if _table_exists(table_name):
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    csv_path = RAW_CSV_DIR / csv_filename
    print(f"  (table '{table_name}' not found in DB, reading {csv_path} instead)")
    return pd.read_csv(csv_path)


def drop_duplicates_full(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse any duplicate rows down to a single copy."""
    return df.drop_duplicates(keep="first")


def drop_missing_or_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any row that has a missing OR empty-string cell in ANY column."""
    df = df.copy()
    # Blank/whitespace-only strings count as "empty", not just NaN.
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
        df[c] = df[c].replace("", pd.NA)
    return df.dropna(how="any")


def remove_outliers_iqr(df: pd.DataFrame, cols, multiplier: float = 1.5) -> pd.DataFrame:
    """Drop rows that are outliers (outside Q1 - k*IQR .. Q3 + k*IQR) on
    any of the given numeric columns."""
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask &= df[c].between(low, high)
    return df[mask]


def clean_orders():
    df = load_raw("raw_orders", "olist_orders_dataset.csv")
    before = len(df)

    df = drop_duplicates_full(df)

    date_cols = [c for c in df.columns if "date" in c or "timestamp" in c]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df = drop_missing_or_empty(df)

    valid_statuses = {
        "delivered", "invoiced", "shipped", "processing",
        "unavailable", "canceled", "created", "approved",
    }
    df = df[df["order_status"].isin(valid_statuses)]

    # Outlier removal: implausible delivery durations.
    df["_delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df = remove_outliers_iqr(df, ["_delivery_days"])
    df = df.drop(columns=["_delivery_days"])

    df.to_sql("clean_orders", conn, if_exists="replace", index=False)
    df.to_csv(CLEAN_DIR / "clean_orders.csv", index=False)
    print(f"clean_orders: {before} -> {len(df)} rows")


def clean_customers():
    df = load_raw("raw_customers", "olist_customers_dataset.csv")
    before = len(df)

    df = drop_duplicates_full(df)
    df["customer_zip_code_prefix"] = df["customer_zip_code_prefix"].astype(str).str.zfill(5)
    df = drop_missing_or_empty(df)

    df = df[df["customer_zip_code_prefix"].str.match(r"^\d{5}$")]
    for c in ["customer_city", "customer_state"]:
        df[c] = df[c].str.lower()
    df = df[df["customer_state"].str.match(r"^[a-z]{2}$")]

    df.to_sql("clean_customers", conn, if_exists="replace", index=False)
    df.to_csv(CLEAN_DIR / "clean_customers.csv", index=False)
    print(f"clean_customers: {before} -> {len(df)} rows")


def clean_order_items():
    df = load_raw("raw_order_items", "olist_order_items_dataset.csv")
    before = len(df)

    df = drop_duplicates_full(df)
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
    for c in ["price", "freight_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = drop_missing_or_empty(df)
    df = df[(df["price"] > 0) & (df["freight_value"] >= 0)]

    df = remove_outliers_iqr(df, ["price", "freight_value"])

    df.to_sql("clean_order_items", conn, if_exists="replace", index=False)
    df.to_csv(CLEAN_DIR / "clean_order_items.csv", index=False)
    print(f"clean_order_items: {before} -> {len(df)} rows")


def clean_payments():
    df = load_raw("raw_order_payments", "olist_order_payments_dataset.csv")
    before = len(df)

    df = drop_duplicates_full(df)
    df["payment_type"] = df["payment_type"].str.strip().str.lower()
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce")
    df["payment_installments"] = pd.to_numeric(df["payment_installments"], errors="coerce")

    df = drop_missing_or_empty(df)
    df["payment_installments"] = df["payment_installments"].astype(int)

    df = df[df["payment_type"] != "not_defined"]
    df = df[df["payment_value"] >= 0]
    df = df[df["payment_installments"] >= 0]

    df = remove_outliers_iqr(df, ["payment_value"])

    df.to_sql("clean_payments", conn, if_exists="replace", index=False)
    df.to_csv(CLEAN_DIR / "clean_payments.csv", index=False)
    print(f"clean_payments: {before} -> {len(df)} rows")


def clean_reviews():
    df = load_raw("raw_order_reviews", "olist_order_reviews_dataset.csv")
    before = len(df)

    df = drop_duplicates_full(df)
    for c in ["review_creation_date", "review_answer_timestamp"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")

    # review_id must be unique; keep the most recently answered copy.
    df = df.sort_values("review_answer_timestamp").drop_duplicates(subset=["review_id"], keep="last")

    # Every column (including the free-text comment fields) must be
    # non-missing/non-empty for a row to survive, per the requirement
    # that ANY missing or empty cell disqualifies the row. Note: this
    # removes the large majority of reviews, since most customers leave
    # no written comment (raw data is ~88% blank on comment title and
    # ~59% blank on comment message).
    df = drop_missing_or_empty(df)

    df = df[df["review_score"].between(1, 5)]
    df = df[df["review_answer_timestamp"] >= df["review_creation_date"]]

    df.to_sql("clean_reviews", conn, if_exists="replace", index=False)
    df.to_csv(CLEAN_DIR / "clean_reviews.csv", index=False)
    print(f"clean_reviews: {before} -> {len(df)} rows")


def clean_category_translation():
    df = load_raw("raw_category_translation", "product_category_name_translation.csv")
    before = len(df)

    df = drop_duplicates_full(df)
    df = drop_missing_or_empty(df)
    for c in ["product_category_name", "product_category_name_english"]:
        df[c] = df[c].str.lower()
    df = df.drop_duplicates(subset=["product_category_name"], keep="first")

    df.to_sql("clean_category_translation", conn, if_exists="replace", index=False)
    df.to_csv(CLEAN_DIR / "clean_category_translation.csv", index=False)
    print(f"clean_category_translation: {before} -> {len(df)} rows")
    return df


def clean_products(category_lookup: pd.DataFrame):
    df = load_raw("raw_products", "olist_products_dataset.csv")
    before = len(df)

    df = drop_duplicates_full(df)

    numeric_cols = [
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
        "product_name_lenght", "product_description_lenght", "product_photos_qty",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = drop_missing_or_empty(df)
    df = df.drop_duplicates(subset=["product_id"], keep="first")

    dim_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    for c in dim_cols:
        df = df[df[c] > 0]

    df = remove_outliers_iqr(df, dim_cols)

    df["product_category_name"] = df["product_category_name"].str.lower()
    df = df.merge(
        category_lookup[["product_category_name", "product_category_name_english"]],
        on="product_category_name",
        how="inner",
    )

    df.to_sql("clean_products", conn, if_exists="replace", index=False)
    df.to_csv(CLEAN_DIR / "clean_products.csv", index=False)
    print(f"clean_products: {before} -> {len(df)} rows")


def clean_sellers():
    df = load_raw("raw_sellers", "olist_sellers_dataset.csv")
    before = len(df)

    df = drop_duplicates_full(df)
    df["seller_zip_code_prefix"] = df["seller_zip_code_prefix"].astype(str).str.zfill(5)
    df = drop_missing_or_empty(df)
    df = df.drop_duplicates(subset=["seller_id"], keep="first")

    df = df[df["seller_zip_code_prefix"].str.match(r"^\d{5}$")]
    for c in ["seller_city", "seller_state"]:
        df[c] = df[c].str.lower()
    df = df[df["seller_state"].str.match(r"^[a-z]{2}$")]

    df.to_sql("clean_sellers", conn, if_exists="replace", index=False)
    df.to_csv(CLEAN_DIR / "clean_sellers.csv", index=False)
    print(f"clean_sellers: {before} -> {len(df)} rows")


# NOTE: geolocation is intentionally NOT cleaned/produced by this pipeline.
# The raw geolocation CSV is left untouched.


if __name__ == "__main__":
    clean_orders()
    clean_customers()
    clean_order_items()
    clean_payments()
    clean_reviews()
    category_lookup = clean_category_translation()
    clean_products(category_lookup)
    clean_sellers()
    conn.close()
    print("\nAll tables cleaned (geolocation skipped). CSVs written to data/cleaned/")
