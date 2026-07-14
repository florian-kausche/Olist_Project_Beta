"""
================================================================================
PHASE 3: DATA CLEANING AND EXPLORATORY DATA ANALYSIS — INTERACTIVE DASHBOARD
Olist Brazilian E-commerce Dataset — CSIT 608
================================================================================
WHAT THIS SCRIPT DOES:
  Runs the same Phase 3 pipeline as before (raw-data audit, cleaning,
  merging, outlier treatment, feature engineering, transformation/scaling,
  descriptive statistics, and pattern discovery) but renders every step as
  an INTERACTIVE PAGE IN YOUR BROWSER instead of printing to the console
  and saving static PNGs — same idea as phase5.py's dashboard.

HOW TO RUN:
    pip install streamlit plotly
    streamlit run scripts/phase3.py     # interactive dashboard in the browser
    python scripts/phase3.py            # same pipeline, printed to the terminal
  `streamlit run` opens automatically in your browser at http://localhost:8501.
  `python scripts/phase3.py` runs the identical pipeline but has no browser/
  Streamlit runtime, so it prints all tables/metrics as plain text instead
  (charts are skipped in this mode — a one-line note says so where relevant).

WHERE THE DATA COMES FROM:
    Reads the raw Olist CSVs straight from data/raw/ (no Phase 2 / database
    step required) and does the cleaning + merging itself, cached so it
    only runs once per session — see STEP 1.

COVERAGE (mapped to what Phase 3 asks for):
  - Cleaning & missing data -> tab "🧹 Cleaning"
  - Duplicates              -> tab "🧹 Cleaning"
  - Outliers                -> tab "📈 Outliers"
  - Transformation & scaling-> tab "🔧 Transform & Scale"
  - Feature engineering     -> tab "🏗️ Features"
  - Descriptive statistics  -> tab "📊 Descriptive Stats"
  - Distributions           -> tab "📉 Distributions"
  - Correlation             -> tab "🔗 Correlation"
  - Uncertainty (95% CI)    -> tab "🎯 Uncertainty"
  - Pattern discovery       -> tab "🔍 Patterns"

DESIGN PRINCIPLES APPLIED (same conventions as phase5.py):
  - Perception: bar length / position over angle or 3D — see "CHART CHOICE".
  - Ethics: axes start at zero, nothing hidden — see "ETHICS".
  - Accessibility: colorblind-safe palette, hover tooltips — see "ACCESSIBILITY".
  - Encoding: consistent color meaning across the whole app — see "ENCODING".
================================================================================
"""

# ------------------------------------------------------------------------
# STEP 0: IMPORTS
# ------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import streamlit.runtime as st_runtime

# ------------------------------------------------------------------------
# STEP 0a: DUAL-MODE SUPPORT
# ------------------------------------------------------------------------
# `streamlit run scripts/phase3.py` -> normal interactive dashboard.
# `python scripts/phase3.py`        -> no Streamlit runtime exists, so every
#   st.* call would otherwise be a silent no-op (that's the "missing
#   ScriptRunContext" warning you see). Instead, when there's no runtime we
#   swap `st` for a tiny shim that prints the same content straight to the
#   terminal, so the SAME script body below works in both modes untouched.
RUNNING_IN_STREAMLIT = st_runtime.exists()


class _ConsoleColumn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def metric(self, label, value):
        print(f"  {label}: {value}")


class _ConsoleTab:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print("\n" + "=" * 70)
        print(self.name)
        print("=" * 70)
        return self

    def __exit__(self, *exc):
        return False


class _ConsoleExpander:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        print(f"\n--- {self.label} ---")
        return self

    def __exit__(self, *exc):
        return False


class _ConsoleSpinner:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        if self.text:
            print(self.text)
        return self

    def __exit__(self, *exc):
        return False


class ConsoleUI:
    """Drop-in stand-in for the small slice of the streamlit API this script
    uses. When there's no Streamlit runtime, `st` is rebound to an instance
    of this class so `st.title(...)`, `st.dataframe(...)`, `with st.tabs(...)`
    etc. print plain text/tables to the terminal instead of doing nothing."""

    @staticmethod
    def set_page_config(**kwargs):
        pass

    @staticmethod
    def cache_data(func=None, **kwargs):
        if func is not None:
            return func

        def decorator(f):
            return f

        return decorator

    @staticmethod
    def spinner(text=""):
        return _ConsoleSpinner(text)

    @staticmethod
    def title(text):
        print("\n" + text)
        print("#" * len(text))

    @staticmethod
    def caption(text):
        print(f"({text})")

    @staticmethod
    def columns(n):
        return [_ConsoleColumn() for _ in range(n)]

    @staticmethod
    def metric(label, value):
        print(f"{label}: {value}")

    @staticmethod
    def markdown(text):
        print(text)

    @staticmethod
    def tabs(names):
        return [_ConsoleTab(name) for name in names]

    @staticmethod
    def subheader(text):
        print("\n" + text)
        print("-" * len(text))

    @staticmethod
    def dataframe(data, **kwargs):
        try:
            print(data.to_string())
        except AttributeError:
            print(data)

    @staticmethod
    def expander(label, **kwargs):
        return _ConsoleExpander(label)

    @staticmethod
    def info(text):
        print(f"[INFO] {text}")

    @staticmethod
    def error(text):
        print(f"[ERROR] {text}")

    @staticmethod
    def stop():
        raise SystemExit(0)

    @staticmethod
    def plotly_chart(fig, **kwargs):
        print("[chart not shown in terminal mode — run "
              "`streamlit run scripts/phase3.py` to view it in the browser]")


if not RUNNING_IN_STREAMLIT:
    st = ConsoleUI()


# ------------------------------------------------------------------------
# STEP 0b: PAGE CONFIG — must be the FIRST streamlit command in the script
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Phase 3 — Cleaning & EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------
# STEP 0c: PROJECT PATHS — matches phase5.py's convention
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ACCESSIBILITY: same colorblind-safe palette as phase5.py, so "bad" and
# "good" mean the same thing across every phase of the project.
COLOR_GOOD = "#0072B2"
COLOR_BAD = "#D55E00"
COLOR_NEUTRAL = "#999999"
CATEGORY_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

# Approximate bounding box for Brazil (mainland + islands), used to flag
# geolocation coordinates that are geocoding errors rather than statistical
# spread. IQR alone can't tell "far south Brazil" from "wrong hemisphere".
BRAZIL_LAT_RANGE = (-34.0, 5.5)
BRAZIL_LNG_RANGE = (-74.0, -34.0)

RAW_CSV_FILES = {
    "olist_customers_dataset.csv": "customer_id",
    "olist_geolocation_dataset.csv": None,
    "olist_order_payments_dataset.csv": None,
    "olist_order_items_dataset.csv": None,
    "olist_orders_dataset.csv": "order_id",
    "olist_products_dataset.csv": "product_id",
    "olist_order_reviews_dataset.csv": "review_id",
    "product_category_name_translation.csv": None,
    "olist_sellers_dataset.csv": "seller_id",
}


# ==========================================================================
# STEP 1: THE PIPELINE — audit, clean, merge, treat outliers, engineer,
# transform/scale. Wrapped in @st.cache_data so it runs ONCE per session
# instead of re-doing all of this on every filter click.
# ==========================================================================
def detect_outliers_iqr_series(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        return 0, Q1, Q3
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = int(((series < lower) | (series > upper)).sum())
    return n_out, lower, upper


def excel_col_letter(idx):
    """0-indexed dataframe column position -> Excel-style column letter (A, B, ..., Z, AA, ...)."""
    letters = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def is_id_like_column(col_name):
    """True for identifier/code columns (zip codes, *_id, sequence numbers)
    that are numeric but not a meaningful quantity — IQR on these produces
    statistically meaningless 'outliers' (e.g. flagging zip code 99999 as
    an outlier the same way it would flag an unusually high price)."""
    name = col_name.lower()
    if name == "id" or name.endswith("_id"):
        return True
    if "zip_code" in name or name.endswith("_zip"):
        return True
    return False


def audit_raw_csv_files(max_samples=10):
    """Missing data / duplicates / outliers for every RAW file, before cleaning.
    Returns (summary_df, details_dict). details_dict[fname] holds Excel-style
    cell references (e.g. 'F8') for a sample of the actual missing/duplicate/
    outlier cells so specific rows can be pointed to, not just counts."""
    rows = []
    details = {}
    for fname, pk in RAW_CSV_FILES.items():
        path = RAW_DATA_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        n_rows, n_cols = df.shape
        missing = df.isna().sum()
        missing = missing[missing > 0]
        dup_exact = int(df.duplicated().sum())
        dup_pk = int(df.duplicated(subset=[pk]).sum()) if (pk and pk in df.columns) else None

        file_detail = {"missing": [], "duplicates": [], "outliers": [], "invalid_geo": []}

        # --- missing: cell refs per column (Excel row = df index + 2) ---
        for ci, col in enumerate(df.columns):
            na_idx = df.index[df[col].isna()]
            if len(na_idx) > 0:
                letter = excel_col_letter(ci)
                sample = [f"{letter}{i + 2}" for i in na_idx[:max_samples]]
                file_detail["missing"].append({
                    "column": col, "letter": letter, "count": int(len(na_idx)),
                    "sample_cells": sample,
                    "more": max(0, len(na_idx) - max_samples),
                })

        # --- duplicates: exact rows + primary-key rows ---
        dup_exact_idx = df.index[df.duplicated(keep=False)]
        if len(dup_exact_idx) > 0:
            sample = [f"row {i + 2}" for i in dup_exact_idx[:max_samples]]
            file_detail["duplicates"].append({
                "type": "exact duplicate rows", "count": int(len(dup_exact_idx)),
                "sample_rows": sample, "more": max(0, len(dup_exact_idx) - max_samples),
            })
        if pk and pk in df.columns:
            dup_pk_extra_idx = df.index[df.duplicated(subset=[pk], keep="first")]
            if len(dup_pk_extra_idx) > 0:
                sample = [f"row {i + 2} ({pk}={df.loc[i, pk]})" for i in dup_pk_extra_idx[:max_samples]]
                file_detail["duplicates"].append({
                    "type": f"duplicate primary key '{pk}' (extra rows beyond the first occurrence — "
                            f"what drop_duplicates() would remove)",
                    "count": int(len(dup_pk_extra_idx)),
                    "sample_rows": sample, "more": max(0, len(dup_pk_extra_idx) - max_samples),
                })

        # --- outliers: cell refs + values per numeric column (skip ID/code-like cols) ---
        outlier_cols = 0
        for ci, col in enumerate(df.columns):
            if col not in df.select_dtypes(include=[np.number]).columns:
                continue
            if is_id_like_column(col):
                continue
            series = df[col].dropna()
            if series.nunique() <= 1:
                continue
            n_out, lower, upper = detect_outliers_iqr_series(series)
            if n_out > 0:
                outlier_cols += 1
                letter = excel_col_letter(ci)
                mask = (series < lower) | (series > upper)
                out_idx = series.index[mask]
                sample = [f"{letter}{i + 2}={df.loc[i, col]}" for i in out_idx[:max_samples]]
                file_detail["outliers"].append({
                    "column": col, "letter": letter, "count": int(n_out),
                    "lower_bound": round(float(lower), 2), "upper_bound": round(float(upper), 2),
                    "sample_cells": sample, "more": max(0, n_out - max_samples),
                })

        # --- geographic validity: real coordinate errors, not just IQR spread ---
        lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
        lng_col = next((c for c in df.columns if c.lower().endswith("lng") or "lon" in c.lower()), None)
        if lat_col and lng_col:
            lat_letter = excel_col_letter(list(df.columns).index(lat_col))
            lng_letter = excel_col_letter(list(df.columns).index(lng_col))
            bad_lat = ~df[lat_col].between(*BRAZIL_LAT_RANGE)
            bad_lng = ~df[lng_col].between(*BRAZIL_LNG_RANGE)
            bad_mask = (bad_lat | bad_lng) & df[lat_col].notna() & df[lng_col].notna()
            bad_idx = df.index[bad_mask]
            if len(bad_idx) > 0:
                sample = [
                    f"row {i + 2} ({lat_letter}{i + 2}={df.loc[i, lat_col]:.3f}, "
                    f"{lng_letter}{i + 2}={df.loc[i, lng_col]:.3f})"
                    for i in bad_idx[:max_samples]
                ]
                file_detail["invalid_geo"].append({
                    "lat_col": lat_col, "lng_col": lng_col, "count": int(len(bad_idx)),
                    "sample_rows": sample, "more": max(0, len(bad_idx) - max_samples),
                    "bounds": f"lat in [{BRAZIL_LAT_RANGE[0]}, {BRAZIL_LAT_RANGE[1]}], "
                              f"lng in [{BRAZIL_LNG_RANGE[0]}, {BRAZIL_LNG_RANGE[1]}]",
                })

        details[fname] = file_detail
        invalid_geo_count = file_detail["invalid_geo"][0]["count"] if file_detail["invalid_geo"] else 0
        rows.append({
            "file": fname, "rows": n_rows, "cols": n_cols,
            "columns_with_missing": len(missing), "total_missing_cells": int(missing.sum()),
            "exact_duplicate_rows": dup_exact, "duplicate_primary_key_rows": dup_pk,
            "columns_with_outliers": outlier_cols,
            "invalid_geo_coordinates": invalid_geo_count,
        })
    return pd.DataFrame(rows), details


def clean_tables(tables):
    """Cleans each raw table: dedupe, fill/keep missing values deliberately."""
    clean, notes = {}, []

    df = tables["customers"].drop_duplicates()
    clean["customers"] = df

    before = len(tables["geolocation"])
    df = tables["geolocation"].drop_duplicates()
    clean["geolocation"] = df
    notes.append(("geolocation", "dropped exact duplicate rows", before - len(df)))

    clean["payments"] = tables["payments"].copy()

    df = tables["order_items"].copy()
    if "shipping_limit_date" in df.columns:
        df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
    clean["order_items"] = df

    df = tables["orders"].drop_duplicates()
    for col in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimated_delivery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    clean["orders"] = df
    notes.append(("orders", "kept missing delivery/approval dates as NaT (undelivered orders, not errors)", 0))

    df = tables["products"].drop_duplicates()
    if "product_category_name" in df.columns:
        df["product_category_name"] = df["product_category_name"].fillna("unknown")
    dim_cols = ["product_name_lenght", "product_description_lenght", "product_photos_qty",
                "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    filled = 0
    for col in dim_cols:
        if col in df.columns and df[col].isna().any():
            filled += int(df[col].isna().sum())
            df[col] = df[col].fillna(df[col].median())
    clean["products"] = df
    notes.append(("products", "filled category with 'unknown'; filled missing dimensions with median", filled))

    df = tables["reviews"].copy()
    if "review_creation_date" in df.columns:
        df["review_creation_date"] = pd.to_datetime(df["review_creation_date"], errors="coerce")
    dup_count = int(df.duplicated(subset=["review_id"]).sum())
    df = df.sort_values("review_creation_date").drop_duplicates(subset=["review_id"], keep="last")
    clean["reviews"] = df
    notes.append(("reviews", "kept most recent record per duplicate review_id", dup_count))

    clean["category_translation"] = tables["category_translation"].drop_duplicates()
    clean["sellers"] = tables["sellers"].drop_duplicates()

    return clean, notes


def build_analysis_table(tables):
    df = tables["order_items"].merge(tables["orders"], on="order_id", how="left")
    df = df.merge(tables["customers"], on="customer_id", how="left")
    df = df.merge(tables["products"], on="product_id", how="left")
    df = df.merge(tables["sellers"], on="seller_id", how="left")
    df = df.merge(tables["category_translation"], on="product_category_name", how="left")

    payments_summary = (
        tables["payments"].groupby("order_id", as_index=False).agg(
            total_payment_value=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            payment_types_used=("payment_type", "nunique"),
        )
    )
    df = df.merge(payments_summary, on="order_id", how="left")
    df = df.merge(tables["reviews"][["order_id", "review_score"]], on="order_id", how="left")
    return df


def detect_outliers_iqr(df, column):
    Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    is_outlier = (df[column] < lower) | (df[column] > upper)
    return is_outlier, lower, upper


@st.cache_data
def run_pipeline():
    """Runs the whole Phase 3 pipeline once and returns everything the
    tabs below need to display. Cached so re-running the app (or changing
    a widget) doesn't repeat all this work."""
    raw_audit_df, raw_audit_details = audit_raw_csv_files()

    name_map = {
        "olist_customers_dataset.csv": "customers", "olist_geolocation_dataset.csv": "geolocation",
        "olist_order_payments_dataset.csv": "payments", "olist_order_items_dataset.csv": "order_items",
        "olist_orders_dataset.csv": "orders", "olist_products_dataset.csv": "products",
        "olist_order_reviews_dataset.csv": "reviews",
        "product_category_name_translation.csv": "category_translation",
        "olist_sellers_dataset.csv": "sellers",
    }
    raw_tables = {key: pd.read_csv(RAW_DATA_DIR / fname) for fname, key in name_map.items()}

    tables, cleaning_notes = clean_tables(raw_tables)
    df = build_analysis_table(tables)
    rows_before_merge_dedupe = len(df)

    missing_report = df.isna().sum()
    missing_report = missing_report[missing_report > 0].sort_values(ascending=False)
    post_merge_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()

    outlier_summary = {}
    for col in ["price", "freight_value", "total_payment_value"]:
        if col not in df.columns:
            continue
        is_out, lower, upper = detect_outliers_iqr(df, col)
        n_out = int(is_out.sum())
        outlier_summary[col] = {"count": n_out, "pct": round(n_out / len(df) * 100, 2),
                                 "lower": round(float(lower), 2), "upper": round(float(upper), 2)}
        if n_out > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)

    # --- feature engineering ---
    if {"order_purchase_timestamp", "order_delivered_customer_date"}.issubset(df.columns):
        df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(df.columns):
        df["delivery_delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
        df["was_late"] = df["delivery_delay_days"] > 0
    if {"price", "freight_value"}.issubset(df.columns):
        df["item_total_value"] = df["price"] + df["freight_value"]
    if {"freight_value", "item_total_value"}.issubset(df.columns):
        df["freight_ratio"] = (df["freight_value"] / df["item_total_value"]).round(3)
    if "order_purchase_timestamp" in df.columns:
        df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.day_name()
        df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    if "review_score" in df.columns:
        df["review_sentiment"] = pd.cut(df["review_score"], bins=[0, 2, 3, 5],
                                         labels=["negative", "neutral", "positive"])

    # --- transform + scale ---
    for col in ["price", "freight_value", "item_total_value"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])
    scale_cols = [c for c in ["price", "freight_value", "delivery_days"] if c in df.columns]
    if scale_cols:
        scaler = StandardScaler()
        valid_idx = df[scale_cols].dropna().index
        scaled = scaler.fit_transform(df.loc[valid_idx, scale_cols])
        for i, col in enumerate(scale_cols):
            df.loc[valid_idx, f"{col}_scaled"] = scaled[:, i]

    return {
        "raw_audit_df": raw_audit_df,
        "raw_audit_details": raw_audit_details,
        "cleaning_notes": cleaning_notes,
        "df": df,
        "rows_before_merge_dedupe": rows_before_merge_dedupe,
        "post_merge_dupes": post_merge_dupes,
        "missing_report": missing_report,
        "outlier_summary": outlier_summary,
        "scale_cols": scale_cols,
    }


with st.spinner("Running Phase 3 pipeline (cleaning, merging, feature engineering)..."):
    results = run_pipeline()

df = results["df"]

if df.empty:
    st.error("The analysis table came out empty. Check that data/raw/ contains the 9 Olist CSVs.")
    st.stop()


# ==========================================================================
# STEP 2: TITLE + HEADLINE KPIs
# ==========================================================================
st.title("🧼 Olist Phase 3 — Data Cleaning & Exploratory Data Analysis")
st.caption(
    "Every step below runs live against the raw CSVs in data/raw/ — cleaning, "
    "missing data, duplicates, outliers, transformation, scaling, feature "
    "engineering, descriptive statistics, and pattern discovery."
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Order Items (final)", f"{len(df):,}")
col2.metric("Columns (final)", f"{df.shape[1]}")
col3.metric("Avg Review Score", f"{df['review_score'].mean():.2f} / 5" if "review_score" in df.columns else "N/A")
col4.metric("Avg Delivery Time", f"{df['delivery_days'].mean():.1f} days" if "delivery_days" in df.columns else "N/A")
col5.metric("Post-merge Duplicates Removed", f"{results['post_merge_dupes']:,}")

st.markdown("---")

tab_clean, tab_outliers, tab_transform, tab_features, tab_stats, tab_dist, tab_corr, tab_uncertainty, tab_patterns = st.tabs(
    ["🧹 Cleaning", "📈 Outliers", "🔧 Transform & Scale", "🏗️ Features",
     "📊 Descriptive Stats", "📉 Distributions", "🔗 Correlation", "🎯 Uncertainty", "🔍 Patterns"]
)


# --------------------------------------------------------------------------
# TAB 1: CLEANING — raw audit, missing data, duplicates
# --------------------------------------------------------------------------
with tab_clean:
    st.subheader("Raw File Audit (before cleaning)")
    st.caption(
        "CHART CHOICE: a table, not a chart — this is reference data a reader "
        "scans for specific numbers, not a pattern to spot visually."
    )
    st.dataframe(results["raw_audit_df"], width='stretch')

    st.subheader("Cell-Level Detail: Exactly Which Rows/Columns")
    st.caption(
        "Excel-style cell references (row numbers count the header row, so "
        "row 2 = the first data row). Long lists are sampled to the first "
        "10 hits per column, with a '+N more' count for the rest. Note: "
        "IQR flags outliers purely on spread, so a bounded, discrete scale "
        "like `review_score` (1–5) will show many 'outliers' just because "
        "low scores are rare relative to the 4–5 cluster — that's skew, not "
        "a data-quality problem, and it isn't capped/treated like price is."
    )
    for fname, detail in results["raw_audit_details"].items():
        if not (detail["missing"] or detail["duplicates"] or detail["outliers"] or detail["invalid_geo"]):
            continue
        with st.expander(f"📄 {fname}"):
            if detail["missing"]:
                st.markdown("**Missing values**")
                for m in detail["missing"]:
                    more = f" *(+{m['more']} more)*" if m["more"] else ""
                    st.markdown(
                        f"- Column `{m['column']}` ({m['letter']}): **{m['count']}** missing → "
                        f"{', '.join(m['sample_cells'])}{more}"
                    )
            if detail["duplicates"]:
                st.markdown("**Duplicates**")
                for d in detail["duplicates"]:
                    more = f" *(+{d['more']} more)*" if d["more"] else ""
                    st.markdown(
                        f"- {d['type']}: **{d['count']}** rows → {', '.join(d['sample_rows'])}{more}"
                    )
            if detail["outliers"]:
                st.markdown("**Outliers (IQR method)**")
                for o in detail["outliers"]:
                    more = f" *(+{o['more']} more)*" if o["more"] else ""
                    st.markdown(
                        f"- Column `{o['column']}` ({o['letter']}): bounds "
                        f"[{o['lower_bound']}, {o['upper_bound']}], **{o['count']}** outliers → "
                        f"{', '.join(o['sample_cells'])}{more}"
                    )
            if detail["invalid_geo"]:
                st.markdown("**Invalid Coordinates (outside Brazil's bounding box)**")
                st.caption(
                    "This is a real validity check, not IQR spread — flags points "
                    "that fall outside Brazil entirely (e.g. wrong hemisphere), "
                    "which IQR can miss or over-flag depending on the column's spread."
                )
                for g in detail["invalid_geo"]:
                    more = f" *(+{g['more']} more)*" if g["more"] else ""
                    st.markdown(
                        f"- `{g['lat_col']}` / `{g['lng_col']}`: valid range {g['bounds']} — "
                        f"**{g['count']}** rows fall outside it → {', '.join(g['sample_rows'])}{more}"
                    )

    st.subheader("What Cleaning Did to Each Table")
    for table_name, action, count in results["cleaning_notes"]:
        if count:
            st.markdown(f"- **{table_name}**: {action} — **{count:,}** rows/cells affected")
        else:
            st.markdown(f"- **{table_name}**: {action}")

    st.subheader("Missing Data After Merging")
    st.caption(
        "ETHICS: merging tables can re-introduce missingness (e.g. an order "
        "with no matching review). We report it rather than silently dropping rows."
    )
    if len(results["missing_report"]) > 0:
        missing_df = results["missing_report"].reset_index()
        missing_df.columns = ["column", "missing_count"]
        missing_df["missing_pct"] = (missing_df["missing_count"] / len(df) * 100).round(2)
        fig = px.bar(
            missing_df, x="missing_pct", y="column", orientation="h",
            labels={"missing_pct": "% missing", "column": ""},
            color_discrete_sequence=[COLOR_NEUTRAL],
        )
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No missing values remain in the merged table.")

    st.subheader("Exact Duplicate Rows After Merging")
    st.metric("Duplicates removed post-merge", f"{results['post_merge_dupes']:,}")


# --------------------------------------------------------------------------
# TAB 2: OUTLIERS — IQR method, before/after capping
# --------------------------------------------------------------------------
with tab_outliers:
    st.subheader("Outliers Detected & Capped (IQR method)")
    st.caption(
        "ETHICS: outliers are CAPPED (winsorized) at the IQR bounds, not "
        "deleted — a genuinely expensive order is real data, not an error. "
        "Deleting it would quietly bias the dataset toward cheaper orders. "
        "Scope: only continuous money columns (price, freight_value, "
        "total_payment_value) are capped here — a bounded, discrete rating "
        "like review_score is deliberately left out. IQR would flag low "
        "star ratings as 'outliers' purely because they're rare relative to "
        "the 4-5 cluster, but that's a real, meaningful opinion, not noise "
        "to be squashed."
    )
    if results["outlier_summary"]:
        outlier_df = pd.DataFrame(results["outlier_summary"]).T.reset_index()
        outlier_df.columns = ["column", "count", "pct", "lower_bound", "upper_bound"]
        st.dataframe(outlier_df, width='stretch')

        fig = px.bar(
            outlier_df, x="pct", y="column", orientation="h",
            labels={"pct": "% of rows flagged as outliers", "column": ""},
            color_discrete_sequence=[COLOR_BAD],
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No outliers detected in the checked columns.")

    st.subheader("Boxplots After Capping")
    box_cols = [c for c in ["price", "freight_value", "delivery_days"] if c in df.columns]
    if box_cols:
        fig = go.Figure()
        for col in box_cols:
            fig.add_trace(go.Box(y=df[col], name=col, marker_color=COLOR_NEUTRAL))
        st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# TAB 3: TRANSFORM & SCALE
# --------------------------------------------------------------------------
with tab_transform:
    st.subheader("Log Transformation (right-skewed columns)")
    st.caption(
        "CHART CHOICE: side-by-side histograms — before vs after log1p — "
        "makes the skew reduction visible rather than just asserted."
    )
    log_cols = [c for c in ["price", "freight_value", "item_total_value"] if f"{c}_log" in df.columns]
    for col in log_cols:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col, nbins=40, labels={col: f"{col} (raw)"},
                                color_discrete_sequence=[COLOR_NEUTRAL])
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.histogram(df, x=f"{col}_log", nbins=40, labels={f"{col}_log": f"{col} (log1p)"},
                                color_discrete_sequence=[COLOR_GOOD])
            st.plotly_chart(fig, width='stretch')

    st.subheader("Scaled Columns (StandardScaler)")
    st.caption(
        "Scaling puts numeric columns on a comparable 0-centered scale — "
        "matters for distance-based models (e.g. clustering), not for tree models. "
        "Original columns are kept alongside the scaled ones, nothing is lost."
    )
    scaled_cols = [f"{c}_scaled" for c in results["scale_cols"]]
    st.dataframe(df[results["scale_cols"] + scaled_cols].describe().round(2), width='stretch')


# --------------------------------------------------------------------------
# TAB 4: FEATURE ENGINEERING
# --------------------------------------------------------------------------
with tab_features:
    st.subheader("New Features Created")
    feature_cols = ["delivery_days", "delivery_delay_days", "was_late", "item_total_value",
                     "freight_ratio", "purchase_dayofweek", "purchase_month", "review_sentiment"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    st.markdown(
        "- **delivery_days** — days between purchase and delivery\n"
        "- **delivery_delay_days** — actual delivery vs estimated (positive = late)\n"
        "- **was_late** — True/False flag from delivery_delay_days\n"
        "- **item_total_value** — price + freight_value\n"
        "- **freight_ratio** — freight_value as a share of item_total_value\n"
        "- **purchase_dayofweek / purchase_month** — for seasonality patterns\n"
        "- **review_sentiment** — review_score bucketed into negative/neutral/positive"
    )
    st.subheader("Sample of Engineered Columns")
    st.dataframe(df[feature_cols].head(20), width='stretch')


# --------------------------------------------------------------------------
# TAB 5: DESCRIPTIVE STATISTICS
# --------------------------------------------------------------------------
with tab_stats:
    st.subheader("Summary Statistics")
    numeric_cols = ["price", "freight_value", "delivery_days", "delivery_delay_days",
                     "review_score", "item_total_value", "total_payment_value"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    st.dataframe(df[numeric_cols].describe().round(2), width='stretch')

    if "was_late" in df.columns:
        st.metric("Late Delivery Rate", f"{df['was_late'].mean():.1%}")


# --------------------------------------------------------------------------
# TAB 6: DISTRIBUTIONS
# --------------------------------------------------------------------------
with tab_dist:
    st.subheader("Distributions of Key Numeric Variables")
    st.caption(
        "CHART CHOICE: histograms — shape and spread matter more here than "
        "exact values, which is what histograms are built to show."
    )
    dist_cols = [c for c in ["price", "freight_value", "delivery_days", "review_score"] if c in df.columns]
    cols_per_row = 2
    for i in range(0, len(dist_cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(dist_cols[i:i + cols_per_row]):
            with row_cols[j]:
                fig = px.histogram(df, x=col, nbins=40, labels={col: col},
                                    color_discrete_sequence=[COLOR_GOOD])
                st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# TAB 7: CORRELATION
# --------------------------------------------------------------------------
with tab_corr:
    st.subheader("Correlation Matrix")
    st.caption(
        "ACCESSIBILITY: diverging color scale with numbers printed in each "
        "cell, so the pattern doesn't rely on color alone."
    )
    corr_cols = [c for c in ["price", "freight_value", "delivery_days", "delivery_delay_days", "review_score"] if c in df.columns]
    if corr_cols:
        corr_matrix = df[corr_cols].corr().round(2)
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, width='stretch')

        if "review_score" in corr_matrix.columns:
            corr_with_review = corr_matrix["review_score"].drop("review_score").abs().sort_values(ascending=False)
            if not corr_with_review.empty:
                top_factor = corr_with_review.index[0]
                st.info(
                    f"**'{top_factor}'** has the strongest correlation with review_score "
                    f"({corr_matrix.loc[top_factor, 'review_score']:.2f})."
                )


# --------------------------------------------------------------------------
# TAB 8: UNCERTAINTY — 95% confidence interval
# --------------------------------------------------------------------------
with tab_uncertainty:
    st.subheader("95% Confidence Interval — Average Review Score")
    st.caption(
        "A single average hides how confident we are in it. This range is "
        "where the TRUE average review score is likely to fall, using "
        "mean ± 1.96 × (std / sqrt(n))."
    )
    if "review_score" in df.columns:
        scores = df["review_score"].dropna()
        mean, std, n = scores.mean(), scores.std(), len(scores)
        margin = 1.96 * (std / np.sqrt(n))
        lower, upper = mean - margin, mean + margin

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Review Score", f"{mean:.3f}")
        c2.metric("95% CI Lower", f"{lower:.3f}")
        c3.metric("95% CI Upper", f"{upper:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lower, upper], y=[1, 1], mode="lines", line=dict(color=COLOR_GOOD, width=6)))
        fig.add_trace(go.Scatter(x=[mean], y=[1], mode="markers", marker=dict(color=COLOR_BAD, size=14)))
        fig.update_yaxes(visible=False, range=[0.5, 1.5])
        fig.update_xaxes(title="Review score")
        fig.update_layout(showlegend=False, height=200, title="95% Confidence Interval (dot = mean)")
        st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# TAB 9: PATTERN DISCOVERY
# --------------------------------------------------------------------------
with tab_patterns:
    st.subheader("Average Review Score by Purchase Day of Week")
    st.caption("CHART CHOICE: bar chart across fixed weekday categories, sorted by the calendar not by value.")
    if {"purchase_dayofweek", "review_score"}.issubset(df.columns):
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        avg_by_day = df.groupby("purchase_dayofweek")["review_score"].mean().reindex(order)
        fig = px.bar(x=avg_by_day.index, y=avg_by_day.values,
                     labels={"x": "", "y": "Average review score"},
                     color_discrete_sequence=[COLOR_GOOD])
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Top 10 Product Categories by Order Volume")
    st.caption(
        "CHART CHOICE: horizontal bar, sorted largest-to-smallest — not a "
        "pie chart, since comparing 10 slice angles is unreliable perceptually."
    )
    cat_col = "product_category_name_english" if "product_category_name_english" in df.columns else "product_category_name"
    if cat_col in df.columns:
        top10 = df[cat_col].value_counts().head(10).sort_values()
        fig = px.bar(x=top10.values, y=top10.index, orientation="h",
                     labels={"x": "Number of order items", "y": ""},
                     color_discrete_sequence=[CATEGORY_PALETTE[0]])
        st.plotly_chart(fig, width='stretch')


# ==========================================================================
# STEP 3: FOOTER
# ==========================================================================
st.markdown("---")
st.caption(
    "Data source: Olist Brazilian E-commerce public dataset. "
    f"This view reflects {len(df):,} order items after cleaning, deduplication, "
    "and outlier capping. All monetary values in Brazilian Real (R$)."
)
