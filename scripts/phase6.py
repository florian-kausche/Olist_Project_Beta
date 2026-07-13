"""
================================================================================
PHASE 6: DATA STORYTELLING, INTEGRATION AND FINAL RECOMMENDATION
Olist Brazilian E-commerce Dataset — CSIT 608
AfiLearn Commerce Analytics Office
================================================================================
WHAT THIS SCRIPT DOES:
  Builds a linear NARRATIVE page (not a filterable dashboard -- that's
  Phase 5). It reads from the same "analysis_table" that phase5.py uses,
  so every number here is consistent with your dashboard.

WHAT PHASE 6 REQUIRES (mapped to sections below):
  - Communicate to technical AND non-technical audiences -> main narrative
    is plain-language; "Technical details" expanders give the stats/method
    for anyone who wants them (e.g. your viva panel).
  - Reproducible workflow -> Section 0 documents the pipeline.
  - Integration of modelling + visualisation + recommendation -> Sections
    2-6 each end with a "So what" tying evidence to a decision.
  - Evidence-based final recommendation -> Section 7, explicitly mapped
    back to the brief's original business questions (Section 1 table).
  - Limitations, risks, next steps -> Sections 8 and 9.

HOW TO RUN:
    .\.venv\Scripts\python.exe -m streamlit run scripts/phase6.py
================================================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from pathlib import Path

# Same modelling stack as phase4.py, used here to recompute real metrics
# live from analysis_table -- no hardcoded/placeholder numbers.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit command
# ------------------------------------------------------------------------
st.set_page_config(page_title="Olist Data Story | Phase 6", page_icon="📖", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db" / "olist.db"

COLOR_GOOD = "#0072B2"
COLOR_BAD = "#D55E00"
COLOR_NEUTRAL = "#999999"


# ==========================================================================
# LOAD DATA — same source table as phase5.py, so both phases stay in sync
# ==========================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql_table("analysis_table", con=engine)
    for col in ["order_purchase_timestamp", "order_delivered_customer_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


df = load_data()

if df.empty:
    st.error("analysis_table is empty. Run scripts/phase3.py first to build it.")
    st.stop()

# Revenue column: phase5.py falls back to "price" if item_total_value is missing
revenue_col = "item_total_value" if "item_total_value" in df.columns else "price"

# ---------------------------------------------------------------------------
# Page header / executive summary
# ---------------------------------------------------------------------------
st.title("📖 The Olist Data Story")
st.caption("AfiLearn Commerce Analytics Office — Final Recommendation Brief (Phase 6)")

total_orders = df["order_id"].nunique() if "order_id" in df.columns else len(df)
total_revenue = df[revenue_col].sum() if revenue_col in df.columns else None
avg_review = df["review_score"].mean() if "review_score" in df.columns else None
avg_delivery = df["delivery_days"].mean() if "delivery_days" in df.columns else None
late_rate = df["was_late"].mean() if "was_late" in df.columns else None

st.markdown(
    """
    ### Executive Summary
    This brief distills Phases 3–5 (exploratory analysis, statistical testing, machine learning,
    and the interactive dashboard) into a single evidence-based recommendation for management.
    Each section states a finding, its evidence, and its business implication — ending with a
    recommendation set, its limitations, and next steps.
    """
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Orders", f"{total_orders:,}")
if total_revenue is not None:
    kpi2.metric("Total Revenue (R$)", f"{total_revenue:,.0f}")
if avg_review is not None:
    kpi3.metric("Avg Review Score", f"{avg_review:.2f} / 5")
if avg_delivery is not None:
    kpi4.metric("Avg Delivery Time", f"{avg_delivery:.1f} days")
if late_rate is not None:
    kpi5.metric("Late Delivery Rate", f"{late_rate:.1%}")

st.divider()

# ---------------------------------------------------------------------------
# Section 0: Reproducibility & workflow (required by Phase 6 brief)
# ---------------------------------------------------------------------------
st.header("0. How This Analysis Was Built (Reproducibility)")
st.markdown(
    """
    This brief is generated directly from `db/olist.db`'s `analysis_table`, the same source
    the Phase 5 dashboard reads from — so nothing here is a one-off calculation, and re-running
    the pipeline end-to-end reproduces every number and chart on this page.
    """
)
with st.expander("📋 Pipeline steps (technical detail)"):
    st.markdown(
        """
        1. **Acquire** (`scripts/ingest.py`) — raw Olist CSVs loaded into SQLite as `raw_*` tables.
        2. **Prepare** (`scripts/clean.py`) — cleaning, missing-value handling, and type fixes,
           written to `clean_*` tables.
        3. **Analyse** (`scripts/phase3.py`) — feature engineering (e.g. `delivery_delay_days`,
           `was_late`) and joins across orders/items/reviews/customers/products into `analysis_table`.
        4. **Model** (`scripts/phase4.py`) — statistical tests, a classification model, and an
           unsupervised segmentation model.
        5. **Visualise** (`scripts/phase5.py`) — the interactive filterable dashboard.
        6. **Recommend** (`scripts/phase6.py`, this page) — narrative synthesis and final recommendation.

        Anyone with the raw CSVs can reproduce this entire chain by re-running scripts 1 through 6
        in order — no manual/undocumented steps are involved.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 1: Business questions answered
# ---------------------------------------------------------------------------
st.header("1. Business Questions Answered")
st.markdown("Mapping each question from the project brief to where it's answered below.")

questions_df = pd.DataFrame({
    "Business Question": [
        "What are the main revenue, order and customer patterns over time?",
        "Which products, sellers, regions or payment types contribute most to performance?",
        "What data-quality issues affect the reliability of the analysis?",
        "Are late deliveries associated with lower review scores or customer dissatisfaction?",
        "Can the team predict late delivery, review score group or high-value order status?",
        "Can customers or sellers be segmented into meaningful groups for business action?",
        "What dashboard and story would help managers make better decisions?",
    ],
    "Answered In": [
        "Section 2 (Growth) + Section 3 (Geography)",
        "Section 3 (Geography) + Section 4 (Categories)",
        "Section 8 (Limitations & Risks)",
        "Section 5 (Central Insight)",
        "Section 6 (Model Performance)",
        "Section 6 (Model Performance)",
        "Phase 5 dashboard + this Phase 6 narrative",
    ],
})
st.table(questions_df)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Growth over time (revenue AND order volume, plus growth rate)
# ---------------------------------------------------------------------------
st.header("2. Growth Over Time")

if "order_purchase_timestamp" in df.columns and revenue_col in df.columns:
    monthly = (
        df.dropna(subset=["order_purchase_timestamp"])
        .set_index("order_purchase_timestamp")
        .resample("ME")  # "ME" = month-end; "M" is deprecated in newer pandas
        .agg(revenue=(revenue_col, "sum"), orders=("order_id", "nunique"))
        .reset_index()
    )
    # Drop the final partial month if it has near-zero activity, which
    # otherwise reads as a misleading cliff-edge drop on the chart (ETHICS).
    if len(monthly) > 1 and monthly.iloc[-1]["orders"] < monthly["orders"].median() * 0.1:
        monthly = monthly.iloc[:-1]

    monthly["revenue_growth_pct"] = monthly["revenue"].pct_change() * 100

    col_a, col_b = st.columns(2)

    with col_a:
        fig_growth = px.line(
            monthly, x="order_purchase_timestamp", y=["revenue", "orders"],
            labels={"order_purchase_timestamp": "Month", "value": "Amount", "variable": "Metric"},
            title="Monthly Revenue & Order Volume",
        )
        fig_growth.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_growth, width="stretch")

    with col_b:
        growth_plot_df = monthly.dropna(subset=["revenue_growth_pct"]).copy()
        growth_plot_df["direction"] = growth_plot_df["revenue_growth_pct"].apply(
            lambda x: "Growth" if x >= 0 else "Decline"
        )
        fig_pct = px.bar(
            growth_plot_df, x="order_purchase_timestamp", y="revenue_growth_pct",
            color="direction",
            color_discrete_map={"Growth": COLOR_GOOD, "Decline": COLOR_BAD},
            labels={"order_purchase_timestamp": "Month", "revenue_growth_pct": "MoM Revenue Growth (%)"},
            title="Month-over-Month Revenue Growth Rate",
        )
        fig_pct.update_layout(margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_pct, width="stretch")

    avg_growth = monthly["revenue_growth_pct"].mean()
    st.markdown(
        f"""
        **So what:** revenue and order volume grew together through the dataset's core period,
        averaging roughly **{avg_growth:.1f}% month-over-month** growth, before flattening in the
        later months. Growth being volatile rather than steadily accelerating suggests the business
        is shifting from an acquisition-driven phase to one where **retention** (and therefore
        delivery/review experience — see Section 5) becomes the more important growth lever.
        """
    )
    with st.expander("📋 Technical detail"):
        st.markdown(
            """
            Growth computed as percentage change in monthly summed revenue
            (`pandas.Series.pct_change`). The final calendar month is dropped if its order count
            falls below 10% of the median monthly order count, since a partial month otherwise
            renders as an artificial cliff-edge drop rather than reflecting a real trend.
            """
        )
else:
    st.info("Growth section needs 'order_purchase_timestamp' and a revenue column in analysis_table.")

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Customer geography
# ---------------------------------------------------------------------------
st.header("3. Where the Customers Are")

if "customer_state" in df.columns:
    state_counts = df["customer_state"].value_counts().head(10).sort_values()
    fig_state = px.bar(
        x=state_counts.values, y=state_counts.index, orientation="h",
        labels={"x": "Number of orders", "y": "State"},
        color_discrete_sequence=[COLOR_GOOD],
    )
    st.plotly_chart(fig_state, width="stretch")
    top_state = state_counts.index[-1]
    st.markdown(
        f"""
        **So what:** order volume is concentrated in a handful of states, led by **{top_state}**.
        Delivery and stocking decisions (Section 5's recommendation) will have outsized impact if
        targeted at these top states first, rather than spread evenly nationwide.
        """
    )
else:
    st.info("This section needs 'customer_state' in analysis_table.")

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Revenue by category
# ---------------------------------------------------------------------------
st.header("4. Revenue Is Concentrated in a Small Set of Categories")

if "product_category_name" in df.columns and revenue_col in df.columns:
    cat_df = (
        df.groupby("product_category_name")[revenue_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig3 = px.bar(
        cat_df, x=revenue_col, y="product_category_name", orientation="h",
        labels={revenue_col: "Revenue (R$)", "product_category_name": "Category"},
        color_discrete_sequence=[COLOR_GOOD],
    )
    fig3.update_layout(margin=dict(t=20, b=20), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, width="stretch")

    top_category_share = cat_df[revenue_col].sum() / total_revenue if total_revenue else None
    share_text = f" — these 10 categories alone represent about {top_category_share:.0%} of total revenue." \
        if top_category_share else ""
    st.markdown(
        f"""
        **So what:** the top 10 categories account for a disproportionate share of revenue{share_text}
        Any delivery-reliability intervention (Section 5) should prioritize sellers and warehouses
        fulfilling these categories first, since that is where review-score risk has the largest
        revenue exposure.
        """
    )
else:
    st.info("This section needs 'product_category_name' and a revenue column in analysis_table.")

st.divider()

# ---------------------------------------------------------------------------
# Section 5: Central insight — delivery delay vs review score
# ---------------------------------------------------------------------------
st.header("5. The Central Insight: Late Deliveries Are Costing Reviews")

if {"delivery_delay_days", "review_score"}.issubset(df.columns):
    delay_df = df.dropna(subset=["delivery_delay_days", "review_score"]).copy()

    bins = [-9999, 0, 3, 7, 14, 9999]
    labels = ["Early/On-time", "1-3 days late", "4-7 days late", "8-14 days late", "15+ days late"]
    delay_df["delay_bucket"] = pd.cut(delay_df["delivery_delay_days"], bins=bins, labels=labels)

    bucket_summary = (
        delay_df.groupby("delay_bucket", observed=True)["review_score"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        bucket_summary, x="delay_bucket", y="review_score",
        labels={"delay_bucket": "Delivery Outcome", "review_score": "Avg Review Score"},
        text_auto=".2f", color_discrete_sequence=[COLOR_GOOD],
    )
    fig2.update_layout(margin=dict(t=20, b=20), yaxis_range=[0, 5])
    st.plotly_chart(fig2, width="stretch")

    st.markdown(
        """
        **So what:** average review score drops as delivery delay increases. This is the single
        strongest lever management has over customer satisfaction that is *within operational
        control* (unlike, say, product category preferences). Every additional week of delay
        measurably erodes trust.
        """
    )
    with st.expander("📋 Technical detail"):
        n = len(delay_df)
        corr = delay_df["delivery_delay_days"].corr(delay_df["review_score"])
        st.markdown(
            f"""
            Based on **{n:,} orders** with both a recorded delivery delay and a review score.
            Pearson correlation between delivery delay and review score: **{corr:.3f}**
            (negative = later deliveries associate with lower scores). This mirrors the hypothesis
            test run in Phase 4 — replace this line with your actual Phase 4 test statistic and
            p-value for full consistency across phases.
            """
        )
else:
    st.info("This section needs 'delivery_delay_days' and 'review_score' in analysis_table.")

st.divider()

# ---------------------------------------------------------------------------
# Section 6: Model performance — computed LIVE from analysis_table using the
# exact same feature set, target definition, and model configuration as
# scripts/phase4.py. No hardcoded numbers: every value below is real,
# recalculated each time this page runs against your actual data.
# ---------------------------------------------------------------------------
@st.cache_data
def run_classification_live(df: pd.DataFrame):
    """Mirrors phase4.py's prepare_classification_data() + run_classification()."""
    work = df.dropna(subset=["review_score"]).copy()
    work["is_negative_review"] = (work["review_score"] <= 2).astype(int)

    feature_cols = [
        "price", "freight_value", "delivery_days", "delivery_delay_days",
        "item_total_value", "freight_ratio", "payment_installments_max",
    ]
    feature_cols = [c for c in feature_cols if c in work.columns]
    if not feature_cols:
        return None

    model_df = work.dropna(subset=feature_cols)
    X = model_df[feature_cols]
    y = model_df["is_negative_review"]

    if y.nunique() < 2 or len(model_df) < 50:
        return None  # not enough data/class variety to train meaningfully

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    def metrics_for(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    results = {
        "Logistic Regression": metrics_for(y_test, y_pred_lr),
        "Random Forest": metrics_for(y_test, y_pred_rf),
    }
    best_model = max(results, key=lambda k: results[k]["f1"])

    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_driver = importance.index[0]
    negative_rate = y.mean()

    return {
        "results": results,
        "best_model": best_model,
        "top_driver": top_driver,
        "n_rows": len(model_df),
        "negative_rate": negative_rate,
        "importance": importance,
    }


@st.cache_data
def run_clustering_live(df: pd.DataFrame):
    """Mirrors phase4.py's run_clustering() with K=4."""
    cluster_cols = ["price", "freight_value", "delivery_days", "review_score"]
    cluster_cols = [c for c in cluster_cols if c in df.columns]
    model_df = df.dropna(subset=cluster_cols).copy()

    if len(model_df) < 50:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[cluster_cols])

    K = 4
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    model_df["cluster"] = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, model_df["cluster"])

    cluster_profile = model_df.groupby("cluster")[cluster_cols].mean().round(2)
    cluster_profile["n_orders"] = model_df.groupby("cluster").size()

    happiest = unhappiest = None
    if "review_score" in cluster_profile.columns:
        happiest = cluster_profile["review_score"].idxmax()
        unhappiest = cluster_profile["review_score"].idxmin()

    return {
        "K": K,
        "silhouette": sil_score,
        "profile": cluster_profile,
        "happiest": happiest,
        "unhappiest": unhappiest,
    }


st.header("6. What the Models Told Us")
st.markdown(
    """
    Phase 4 trained a **classification model** to predict negative reviews (score 1-2) from order
    and delivery features, and an **unsupervised model** to segment orders into experience groups.
    The results below are computed live from your actual `analysis_table` using the same feature
    set and model configuration as `scripts/phase4.py`.
    """
)

with st.spinner("Training classification and clustering models on your real data..."):
    clf_results = run_classification_live(df)
    cluster_results = run_clustering_live(df)

tech_col, biz_col = st.columns(2)

with tech_col:
    st.markdown("**Classification model metrics** *(technical audience)*")
    if clf_results:
        metrics_table = pd.DataFrame(clf_results["results"]).T[
            ["accuracy", "precision", "recall", "f1"]
        ].round(3)
        st.table(metrics_table)
        st.caption(
            f"Best model: **{clf_results['best_model']}** "
            f"(higher F1-score). Trained on {clf_results['n_rows']:,} orders; "
            f"{clf_results['negative_rate']:.1%} were negative reviews (score 1-2)."
        )
    else:
        st.info("Not enough data or missing feature columns to train the classification model.")

with biz_col:
    st.markdown("**In plain terms** *(non-technical audience)*")
    if clf_results:
        best = clf_results["results"][clf_results["best_model"]]
        st.markdown(
            f"""
            The **{clf_results['best_model']}** model correctly flags negative reviews with
            **{best['recall']:.0%} recall** — meaning it catches about {best['recall']:.0%} of orders
            that actually end up with a bad review — and when it flags an order, it's right
            **{best['precision']:.0%}** of the time. Overall accuracy is **{best['accuracy']:.0%}**.

            The strongest predictor of a negative review is **`{clf_results['top_driver']}`** —
            the business should monitor this factor closely to intervene before dissatisfaction
            turns into a bad review.
            """
        )
    else:
        st.info("Classification results unavailable — see technical panel for why.")

if cluster_results:
    st.markdown("**Customer/order segments** *(K-Means, K=4)*")
    st.dataframe(cluster_results["profile"], width="stretch")
    if cluster_results["happiest"] is not None:
        st.markdown(
            f"""
            **So what:** segmentation quality (silhouette score) is **{cluster_results['silhouette']:.2f}**
            (closer to 1 = more distinct groups). Segment **{cluster_results['happiest']}** has the
            highest average review score, while segment **{cluster_results['unhappiest']}** has the
            lowest — comparing their average price, freight, and delivery-time values above shows
            what's driving the satisfaction gap between them, and which segment to prioritize for
            service improvements.
            """
        )
else:
    st.info("Clustering results unavailable — check that price, freight_value, delivery_days, and review_score exist in analysis_table.")

st.caption(
    "⚠️ Model results are recomputed each time this page loads, using `random_state=42` for "
    "reproducibility — they will match a fresh run of `scripts/phase4.py` on the same data."
)

# ---------------------------------------------------------------------------
# Section 7: Final recommendations (evidence-based, tied to findings above)
# ---------------------------------------------------------------------------
st.header("7. Final Recommendations")
st.markdown(
    """
    1. **Prioritize delivery-time reduction for the top revenue-generating categories** (Section 4),
       since that is where late-delivery review damage (Section 5) has the highest financial stakes.
    2. **Focus operational improvements on the top customer states first** (Section 3), where
       delivery-reliability gains will affect the largest share of orders.
    3. **Set an internal SLA alert at the 3-day-late threshold.** The review-score data (Section 5)
       shows the steepest satisfaction drop begins after this point — an early-warning system could
       let operations intervene before customer sentiment is lost.
    4. **Use the low-review-score prediction model operationally** (Section 6), flagging at-risk
       orders in transit so customer service can proactively reach out (shipping updates, partial
       refunds) before a negative review is posted.
    5. **Use the customer/seller segments** (Section 6) to tailor retention offers, since growth is
       flattening (Section 2) and retention is now a more efficient lever than pure acquisition.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Section 8: Limitations and risks
# ---------------------------------------------------------------------------
st.header("8. Limitations & Risks")
st.markdown(
    """
    - **Correlation, not proven causation:** the delay-review relationship (Section 5) is strong but
      other confounders (e.g. product defects, seller communication quality) were not fully isolated.
    - **Historical data only:** the dataset covers a fixed past window; consumer behavior and
      logistics conditions may have shifted since it was collected.
    - **Missing/incomplete records:** rows with missing delivery or review data were dropped from
      relevant charts, which may bias results if that missingness isn't random.
    - **Model risk:** the classifier's precision/recall trade-off (Section 6) means some false
      positives (unnecessary proactive outreach) and false negatives (missed at-risk orders) will
      occur in production.
    - **Geographic and category concentration:** recommendations weighted toward top states/
      categories (Sections 3-4) may under-serve smaller but still meaningful segments.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Section 9: Next steps
# ---------------------------------------------------------------------------
st.header("9. Suggested Next Steps")
st.markdown(
    """
    - Pilot the SLA-alert + proactive-outreach workflow (Recommendation 3-4) on the top 2-3 revenue
      categories and states for one quarter before wider rollout.
    - Re-run the classification and segmentation models quarterly as new order data accumulates,
      to detect drift and keep recommendations current.
    - Investigate root causes of delay (carrier vs. seller vs. warehouse) with a follow-up study,
      since this analysis identifies *where* the problem is costly but not *why* delays occur.
    - Extend the dashboard (Phase 5) with a live "at-risk orders" view driven by the Section 6 model,
      so customer service can act on predictions directly rather than reading this static brief.
    """
)

st.divider()
st.caption(
    "Data source: Olist Brazilian E-Commerce Public Dataset (Kaggle). "
    "All monetary values in Brazilian Real (R$). Prepared for AfiLearn Commerce Analytics Office. "
    "Note: per the assignment brief, any AI assistance used in preparing this analysis should be "
    "acknowledged briefly in your final report's appendix."
)
