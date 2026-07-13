"""
================================================================================
PHASE 5: VISUALISATION, DASHBOARD DESIGN AND INTERACTION
Olist Brazilian E-commerce Dataset — CSIT 608
================================================================================
WHAT THIS SCRIPT DOES:
  Builds an INTERACTIVE DASHBOARD (runs in your web browser) that lets a
  manager explore Performance, Customer, Product, Delivery, and Review
  insights — with filters, not just static charts.

WHY STREAMLIT (a Python dashboard library) instead of Tableau/Power BI:
  Since your project pipeline is already Python (pandas/SQLAlchemy), a
  Python dashboard keeps everything in one language and one repo, and is
  free/open-source. Streamlit turns a plain Python script into a web app
  with almost no HTML/JS/CSS needed — good if you don't know frontend code.
  (If your team prefers Tableau/Power BI instead, the "analysis_table" this
  reads from is exactly what you'd connect either tool to — just point it
  at db/olist.db.)

HOW TO RUN:
    pip install streamlit plotly
    streamlit run scripts/dashboard.py
  This opens automatically in your browser at http://localhost:8501

DESIGN PRINCIPLES APPLIED (mapped to what Phase 5 asks for):
  - Visualisation principles / perception: uses pre-attentive attributes
    (position, length, color) rather than harder-to-read ones (angle in
    pie charts, 3D). See comments marked "PERCEPTION" below.
  - Chart selection: each chart type is chosen to match the QUESTION being
    asked, not picked for looks. See comments marked "CHART CHOICE".
  - Design ethics: axes start at zero where truncating would mislead;
    we never hide unfavorable numbers. See comments marked "ETHICS".
  - Accessibility: colorblind-safe palette, readable font sizes, every
    chart has a text title/caption (screen-reader-friendly), high contrast.
    See comments marked "ACCESSIBILITY".
  - Visual encoding: consistent color meaning across the whole dashboard
    (e.g. "late" is always the same red everywhere). See "ENCODING".
================================================================================
"""

# ------------------------------------------------------------------------
# STEP 0: IMPORTS
# ------------------------------------------------------------------------
# streamlit -> turns this script into an interactive web dashboard
# plotly    -> interactive charts (hover tooltips, zoom) — better than
#              static matplotlib images for a dashboard a manager will click around
# pandas    -> data handling
# sqlalchemy -> reads the analysis_table your Phase 3 script built
# ------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from pathlib import Path

# ------------------------------------------------------------------------
# STEP 0b: PAGE CONFIG — must be the FIRST streamlit command in the script
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist E-commerce Performance Dashboard",
    layout="wide",          # use the full browser width, not a narrow column
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------
# STEP 0c: PROJECT PATHS — matches phase3.py / phase4.py
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db" / "olist.db"

# ------------------------------------------------------------------------
# ACCESSIBILITY: a fixed, colorblind-safe palette used EVERYWHERE.
# This specific palette (from Okabe-Ito) is designed to remain
# distinguishable for the most common forms of color blindness.
# Using the SAME colors for the SAME meaning across every chart also
# reduces cognitive load — a manager learns "red = bad" once, not per chart.
# ------------------------------------------------------------------------
COLOR_GOOD = "#0072B2"      # blue  -> positive / on-time / high satisfaction
COLOR_BAD = "#D55E00"       # orange/red -> negative / late / low satisfaction
COLOR_NEUTRAL = "#999999"   # gray  -> neutral / baseline
CATEGORY_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


# ==========================================================================
# STEP 1: LOAD DATA (cached so it doesn't re-query the DB on every click)
# ==========================================================================
@st.cache_data
def load_data():
    """
    @st.cache_data means: run this function once, remember the result, and
    reuse it instantly on every future interaction (filter change, etc.)
    instead of re-reading the whole database every time. This is what
    makes the dashboard feel fast and "interactive".
    """
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql_table("analysis_table", con=engine)

    # Make sure date columns are real dates, not text, so we can filter by them
    for col in ["order_purchase_timestamp", "order_delivered_customer_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


df = load_data()

if df.empty:
    st.error("analysis_table is empty. Run scripts/phase3.py first to build it.")
    st.stop()  # stops the dashboard here so nothing below tries to run on empty data


# ==========================================================================
# STEP 2: SIDEBAR FILTERS — this is what makes it a DASHBOARD, not a report
# ==========================================================================
# A static report shows one fixed view. A dashboard lets the manager ask
# their OWN questions ("what about just São Paulo?", "what about last
# quarter?") without needing to come back to you for a new chart.
st.sidebar.header("Filters")
st.sidebar.caption("Narrow down the data below — every chart updates automatically.")

# --- Date range filter ---
if "order_purchase_timestamp" in df.columns:
    min_date = df["order_purchase_timestamp"].min()
    max_date = df["order_purchase_timestamp"].max()
    date_range = st.sidebar.date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df["order_purchase_timestamp"] >= pd.Timestamp(start_date))
            & (df["order_purchase_timestamp"] <= pd.Timestamp(end_date))
        ]

# --- Customer state filter (if that column exists from clean_customers) ---
if "customer_state" in df.columns:
    all_states = sorted(df["customer_state"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect(
        "Customer state", options=all_states, default=[]
    )
    if selected_states:  # only filter if the manager actually picked something
        df = df[df["customer_state"].isin(selected_states)]

# --- Product category filter (if that column exists) ---
if "product_category_name" in df.columns:
    all_categories = sorted(df["product_category_name"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Product category", options=all_categories, default=[]
    )
    if selected_categories:
        df = df[df["product_category_name"].isin(selected_categories)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** order items after filters.")

# ETHICS: if filters produce zero rows, say so clearly instead of showing
# empty/misleading charts that look like "zero activity happened".
if df.empty:
    st.warning("No data matches the current filters. Try widening your selection.")
    st.stop()


# ==========================================================================
# STEP 3: TITLE + HEADLINE KPIs
# ==========================================================================
# PERCEPTION: big numbers at the top are read FIRST and FASTEST by anyone
# glancing at a dashboard. Put the metrics a manager cares about most,
# front and center, before any chart.
st.title("📦 Olist E-commerce Performance Dashboard")
st.caption(
    "Explore performance, customer, product, delivery, and review insights. "
    "Use the filters on the left to focus on a date range, state, or category."
)

col1, col2, col3, col4, col5 = st.columns(5)

total_orders = df["order_id"].nunique() if "order_id" in df.columns else len(df)
total_revenue = df["item_total_value"].sum() if "item_total_value" in df.columns else df["price"].sum()
avg_review = df["review_score"].mean() if "review_score" in df.columns else None
avg_delivery = df["delivery_days"].mean() if "delivery_days" in df.columns else None
late_rate = df["was_late"].mean() if "was_late" in df.columns else None

# st.metric shows a big number with an optional delta arrow — ideal for KPIs
col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Total Revenue (R$)", f"{total_revenue:,.0f}")
col3.metric("Avg Review Score", f"{avg_review:.2f} / 5" if avg_review is not None else "N/A")
col4.metric("Avg Delivery Time", f"{avg_delivery:.1f} days" if avg_delivery is not None else "N/A")
col5.metric("Late Delivery Rate", f"{late_rate:.1%}" if late_rate is not None else "N/A")

st.markdown("---")


# ==========================================================================
# STEP 4: TABS — organizes the dashboard by BUSINESS AREA, matching the
# brief's requirement to cover performance, customer, product, delivery,
# and review insights without overwhelming one single page.
# ==========================================================================
tab_performance, tab_customer, tab_product, tab_delivery, tab_review = st.tabs(
    ["📈 Performance", "👥 Customer", "🛍️ Product", "🚚 Delivery", "⭐ Reviews"]
)


# --------------------------------------------------------------------------
# TAB 1: PERFORMANCE — revenue trend over time
# --------------------------------------------------------------------------
with tab_performance:
    st.subheader("Revenue Over Time")
    st.caption(
        "CHART CHOICE: a line chart is the correct choice for showing a trend "
        "over time — position along a line is one of the most accurately "
        "perceived visual encodings (Cleveland & McGill's perception ranking)."
    )

    if "order_purchase_timestamp" in df.columns and "item_total_value" in df.columns:
        monthly_revenue = (
            df.set_index("order_purchase_timestamp")["item_total_value"]
            .resample("ME")   # "ME" = resample by Month End
            .sum()
            .reset_index()
        )
        fig = px.line(
            monthly_revenue, x="order_purchase_timestamp", y="item_total_value",
            markers=True,
            labels={"order_purchase_timestamp": "Month", "item_total_value": "Revenue (R$)"},
        )
        # ETHICS: force the y-axis to start at 0. A line chart with a
        # truncated axis can make a small revenue dip LOOK like a collapse.
        fig.update_yaxes(rangemode="tozero")
        fig.update_traces(line_color=COLOR_GOOD)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Required columns not available for this chart.")

    st.subheader("Orders by Weekday")
    st.caption(
        "CHART CHOICE: a bar chart for comparing discrete categories (days of "
        "the week) — bar LENGTH is easier to compare accurately than, say, "
        "the area of circles."
    )
    if "purchase_dayofweek" in df.columns:
        order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        counts = df["purchase_dayofweek"].value_counts().reindex(order_days).fillna(0)
        fig = px.bar(
            x=counts.index, y=counts.values,
            labels={"x": "Day of week", "y": "Number of orders"},
            color_discrete_sequence=[COLOR_GOOD],
        )
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# TAB 2: CUSTOMER — geography and payment behavior
# --------------------------------------------------------------------------
with tab_customer:
    st.subheader("Orders by Customer State")
    st.caption(
        "CHART CHOICE: horizontal bar chart, sorted largest-to-smallest. "
        "Sorting (rather than alphabetical order) lets a manager instantly "
        "spot the top states without hunting."
    )
    if "customer_state" in df.columns:
        state_counts = df["customer_state"].value_counts().head(15).sort_values()
        fig = px.bar(
            x=state_counts.values, y=state_counts.index, orientation="h",
            labels={"x": "Number of orders", "y": "State"},
            color_discrete_sequence=[COLOR_GOOD],
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("customer_state column not available.")

    st.subheader("Payment Installments Used")
    st.caption(
        "CHART CHOICE: histogram — shows the SHAPE of a numeric distribution "
        "(most customers pay in how many installments?), which a bar chart "
        "of raw counts wouldn't reveal as clearly."
    )
    if "payment_installments_max" in df.columns:
        fig = px.histogram(
            df, x="payment_installments_max", nbins=24,
            labels={"payment_installments_max": "Number of installments"},
            color_discrete_sequence=[COLOR_GOOD],
        )
        st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# TAB 3: PRODUCT — top categories by revenue and by volume
# --------------------------------------------------------------------------
with tab_product:
    st.subheader("Top 10 Product Categories by Revenue")
    st.caption(
        "CHART CHOICE: horizontal bar, not a pie chart. Pie charts rely on "
        "comparing ANGLES/AREAS, which perception research shows people "
        "judge far less accurately than bar LENGTH — especially with more "
        "than 4-5 slices, which quickly becomes unreadable (a design-ethics "
        "concern: a misleading pie chart isn't just ugly, it's inaccurate)."
    )
    if "product_category_name" in df.columns and "item_total_value" in df.columns:
        top_categories = (
            df.groupby("product_category_name")["item_total_value"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )
        fig = px.bar(
            x=top_categories.values, y=top_categories.index, orientation="h",
            labels={"x": "Revenue (R$)", "y": "Product category"},
            color_discrete_sequence=[COLOR_GOOD],
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Required columns not available for this chart.")

    st.subheader("Price Distribution")
    fig = px.box(
        df, y="price", points=False,
        labels={"price": "Price (R$)"},
        color_discrete_sequence=[COLOR_NEUTRAL],
    )
    st.caption(
        "CHART CHOICE: boxplot — shows median, spread, and outliers together "
        "in one compact view, better for skewed price data than a single average."
    )
    st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# TAB 4: DELIVERY — on-time performance
# --------------------------------------------------------------------------
with tab_delivery:
    st.subheader("On-Time vs Late Deliveries")
    st.caption(
        "ENCODING: 'late' is always orange/red and 'on-time' is always blue "
        "across this whole dashboard, so the manager doesn't have to "
        "re-learn the color meaning on every tab."
    )
    if "was_late" in df.columns:
        late_counts = df["was_late"].map({True: "Late", False: "On-time"}).value_counts()
        fig = px.bar(
            x=late_counts.index, y=late_counts.values,
            labels={"x": "", "y": "Number of orders"},
            color=late_counts.index,
            color_discrete_map={"Late": COLOR_BAD, "On-time": COLOR_GOOD},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Delivery Time Distribution")
    if "delivery_days" in df.columns:
        fig = px.histogram(
            df, x="delivery_days", nbins=40,
            labels={"delivery_days": "Delivery time (days)"},
            color_discrete_sequence=[COLOR_GOOD],
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("Average Delivery Time by State")
    st.caption(
        "ACCESSIBILITY: hovering any bar shows the exact number as a tooltip "
        "(Plotly does this by default), so the chart doesn't rely on color "
        "alone to convey information — useful for colorblind users."
    )
    if {"customer_state", "delivery_days"}.issubset(df.columns):
        avg_by_state = (
            df.groupby("customer_state")["delivery_days"].mean().sort_values(ascending=False).head(15)
        )
        fig = px.bar(
            x=avg_by_state.index, y=avg_by_state.values,
            labels={"x": "State", "y": "Avg delivery time (days)"},
            color_discrete_sequence=[COLOR_GOOD],
        )
        st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# TAB 5: REVIEWS — satisfaction breakdown and drivers
# --------------------------------------------------------------------------
with tab_review:
    st.subheader("Review Score Breakdown")
    st.caption(
        "CHART CHOICE: bar chart across the fixed 1-5 scale (not a pie), so "
        "the manager can see the actual shape of satisfaction — e.g. whether "
        "it's bimodal (lots of 1s AND 5s) which a single average would hide."
    )
    if "review_score" in df.columns:
        review_counts = df["review_score"].value_counts().sort_index()
        # Color scale from bad (1) to good (5), consistent with our red/blue meaning
        colors = [COLOR_BAD, "#E69F00", COLOR_NEUTRAL, "#56B4E9", COLOR_GOOD]
        fig = px.bar(
            x=review_counts.index.astype(str), y=review_counts.values,
            labels={"x": "Review score", "y": "Number of reviews"},
            color=review_counts.index.astype(str),
            color_discrete_sequence=colors,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Review Score vs Delivery Delay")
    st.caption(
        "CHART CHOICE: scatter plot to show the RELATIONSHIP between two "
        "numeric variables — position on both axes lets the manager see the "
        "pattern directly, supporting the same finding as the Phase 4 "
        "hypothesis test."
    )
    if {"delivery_delay_days", "review_score"}.issubset(df.columns):
        sample = df.dropna(subset=["delivery_delay_days", "review_score"]).sample(
            min(3000, len(df)), random_state=42
        )  # sample for speed — plotting 100k+ points would be slow and unreadable
        fig = px.scatter(
            sample, x="delivery_delay_days", y="review_score",
            opacity=0.3,
            labels={"delivery_delay_days": "Delivery delay (days, positive=late)", "review_score": "Review score"},
            color_discrete_sequence=[COLOR_GOOD],
        )
        st.plotly_chart(fig, width='stretch')


# ==========================================================================
# STEP 5: FOOTER — data provenance, for transparency (design ethics)
# ==========================================================================
st.markdown("---")
st.caption(
    "ETHICS/TRANSPARENCY: Data source: Olist Brazilian E-commerce public "
    f"dataset. This view reflects {len(df):,} order items after your current "
    "filters. All monetary values in Brazilian Real (R$)."
)
