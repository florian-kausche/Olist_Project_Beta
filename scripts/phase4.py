"""
================================================================================
PHASE 4: STATISTICAL ANALYSIS AND MACHINE LEARNING
Olist Brazilian E-commerce Dataset — CSIT 608
================================================================================
WHAT THIS SCRIPT DOES (in order):
  1. Loads the "analysis_table" you built at the end of Phase 3
  2. STATISTICAL ANALYSIS: hypothesis test + confidence interval + A/B-test
     reasoning — "does late delivery actually cause lower review scores?"
  3. SUPERVISED LEARNING (classification): predicts whether an order will
     get a NEGATIVE review, using logistic regression + random forest,
     validated with train/test split and cross-validation
  4. SUPERVISED LEARNING (regression, bonus): predicts delivery_days from
     order features with linear regression
  5. UNSUPERVISED LEARNING: K-Means clustering to segment orders into
     customer-experience groups, plus PCA to visualize the clusters in 2D
  6. Every section ends with a plain-English "what this means for the
     business" interpretation, saved into outputs/phase4_summary.txt

HOW TO RUN:
    python scripts/ml.py                    # prints everything to the terminal
    streamlit run scripts/phase4.py         # same run, plus a browser dashboard
  Both run the identical pipeline. In terminal mode there's no Streamlit
  runtime, so the st.* dashboard calls are silently skipped — you get the
  same results via the print() statements below, with no warning spam.

REQUIRES (install once):
    pip install scikit-learn scipy

WHERE THINGS GO:
    - Charts  -> outputs/figures/
    - Summary -> outputs/phase4_summary.txt
================================================================================
"""

# ------------------------------------------------------------------------
# STEP 0: IMPORTS — grouped by what they're for, so it's clear why each
# library is here even if you've never used it before.
# ------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pathlib import Path
import streamlit.runtime as st_runtime

# ------------------------------------------------------------------------
# STEP 0a: DUAL-MODE SUPPORT
# ------------------------------------------------------------------------
# This script already prints every result via plain print() (see below), so
# it's fully usable with `python scripts/phase4.py` as the docstring says.
# The st.* calls layered alongside those prints are for the optional
# `streamlit run scripts/phase4.py` dashboard view. Problem: outside of
# `streamlit run` there's no ScriptRunContext, so every one of those st.*
# calls prints a "missing ScriptRunContext" warning even though it's
# otherwise harmless. Fix: when there's no Streamlit runtime, swap `st` for
# a silent no-op stand-in — the print() statements already give you the
# full terminal report, so the st.* calls just need to disappear quietly.
RUNNING_IN_STREAMLIT = st_runtime.exists()


class _NullWidget:
    """Absorbs any attribute/method access and any further chaining (e.g.
    col.metric(...)) without doing anything or raising."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class NullUI:
    """Drop-in stand-in for streamlit when there's no runtime. Every method
    call (st.header, st.write, st.columns, st.pyplot, etc.) becomes a no-op
    instead of emitting 'missing ScriptRunContext' warnings."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop

    def columns(self, n):
        return [_NullWidget() for _ in range(n)]


if not RUNNING_IN_STREAMLIT:
    st = NullUI()

# Charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics: hypothesis testing and confidence intervals
from scipy import stats

# Machine learning: splitting data, scaling, and cross-validation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Supervised learning models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Model evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)

# Unsupervised learning: clustering and dimensionality reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------------
# STEP 0b: PROJECT PATHS — edit DB_PATH if your file is named differently
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db" / "olist.db"   # must match phase3.py's DB_PATH exactly
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "phase4_summary.txt"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

# Collects plain-English, business-facing takeaways as we go
insights = []


def log_insight(text):
    """Print an insight now, and remember it for the final summary file."""
    print(f"  -> INSIGHT: {text}")
    st.info(f"**INSIGHT:** {text}")
    insights.append(text)


# ==========================================================================
# STEP 1: LOAD THE ANALYSIS TABLE FROM PHASE 3
# ==========================================================================
def load_analysis_table(engine):
    """
    Loads the single wide table (one row per order item) that Phase 3
    produced, including engineered features like delivery_days,
    was_late, review_sentiment, item_total_value, etc.
    """
    print("\n[STEP 1] Loading analysis_table from the database...")
    st.header("Step 1: Load Analysis Table")

    # Check first whether the table actually exists, so we can give a
    # clear, actionable error instead of a raw pandas traceback.
    from sqlalchemy import inspect
    existing_tables = inspect(engine).get_table_names()

    if "analysis_table" not in existing_tables:
        error_msg = (
            "'analysis_table' was not found in the database.\n\n"
            f"Database file checked: {DB_PATH}\n\n"
            f"Tables that DO exist there: {existing_tables}\n\n"
            "This almost always means Phase 3 (scripts/eda.py) has not been "
            "run successfully yet — it's the script that creates "
            "'analysis_table'. Run:\n"
            "    python scripts/eda.py\n"
            "and confirm it completes all 10 steps without error, then "
            "re-run this script."
        )
        st.error(error_msg)
        raise RuntimeError("\n\n" + error_msg)

    df = pd.read_sql_table("analysis_table", con=engine)
    print(f"  loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write(f"Loaded **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.")
    st.dataframe(df.head())
    return df


# ==========================================================================
# STEP 2: STATISTICAL ANALYSIS — HYPOTHESIS TEST + CONFIDENCE INTERVAL
# ==========================================================================
def hypothesis_test_late_delivery(df):
    """
    BUSINESS QUESTION: "Do customers whose orders arrive LATE really give
    lower review scores, or could the difference we see just be random
    noise?" This is classic A/B-test-style reasoning: Group A = on-time
    orders, Group B = late orders.

    We use an INDEPENDENT SAMPLES T-TEST, which compares the average of a
    numeric variable (review_score) between two separate groups.

    HYPOTHESES (standard statistical setup):
      H0 (null hypothesis)........ average review score is THE SAME for
                                    on-time and late orders (no real effect)
      H1 (alternative hypothesis).. average review score is DIFFERENT
                                    between the two groups

    DECISION RULE: if p-value < 0.05, we reject H0 — i.e. the difference
    is unlikely to be random chance, so we treat it as a real effect.
    """
    print("\n[STEP 2] Hypothesis test: late delivery vs review score...")
    st.header("Step 2: Hypothesis Test — Late Delivery vs Review Score")

    if not {"was_late", "review_score"}.issubset(df.columns):
        print("  required columns missing, skipping this step")
        st.warning("Required columns missing, skipping this step.")
        return

    # Split into two groups and drop missing review scores in each
    on_time_scores = df.loc[df["was_late"] == False, "review_score"].dropna()
    late_scores = df.loc[df["was_late"] == True, "review_score"].dropna()

    # stats.ttest_ind runs the independent t-test.
    # equal_var=False uses "Welch's t-test", which doesn't assume the two
    # groups have identical variance — safer default when group sizes differ.
    t_stat, p_value = stats.ttest_ind(on_time_scores, late_scores, equal_var=False)

    mean_on_time = on_time_scores.mean()
    mean_late = late_scores.mean()

    print(f"  on-time orders: n={len(on_time_scores)}, mean review={mean_on_time:.3f}")
    print(f"  late orders:    n={len(late_scores)}, mean review={mean_late:.3f}")
    print(f"  t-statistic={t_stat:.3f}, p-value={p_value:.6f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("On-time avg review", f"{mean_on_time:.2f}", help=f"n={len(on_time_scores)}")
    col2.metric("Late avg review", f"{mean_late:.2f}", help=f"n={len(late_scores)}")
    col3.metric("p-value", f"{p_value:.6f}")

    # --- 95% confidence interval on the DIFFERENCE between the two means ---
    # This tells us the plausible range for how much review score drops
    # due to being late, not just a single point estimate.
    diff = mean_on_time - mean_late
    se_diff = np.sqrt(
        on_time_scores.var(ddof=1) / len(on_time_scores)
        + late_scores.var(ddof=1) / len(late_scores)
    )
    ci_lower = diff - 1.96 * se_diff
    ci_upper = diff + 1.96 * se_diff
    print(f"  95% CI for the difference in means: ({ci_lower:.3f}, {ci_upper:.3f})")
    st.write(f"95% confidence interval for the difference in means: **({ci_lower:.3f}, {ci_upper:.3f})**")

    # --- Chart: compare the two distributions side by side ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        x=df["was_late"].map({True: "Late", False: "On-time"}),
        y=df["review_score"], ax=ax, palette=["#55A868", "#C44E52"]
    )
    ax.set_title("Review Score: On-time vs Late Delivery")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "hypothesis_test_late_vs_review.png", dpi=150)
    st.pyplot(fig)
    plt.close(fig)

    # --- Business-language interpretation ---
    if p_value < 0.05:
        log_insight(
            f"Late deliveries are STATISTICALLY SIGNIFICANTLY associated with "
            f"lower review scores (on-time avg {mean_on_time:.2f} vs late avg "
            f"{mean_late:.2f}, p={p_value:.4f}). We're 95% confident the true "
            f"gap is between {ci_lower:.2f} and {ci_upper:.2f} points. "
            f"Business implication: investing in on-time delivery is likely to "
            f"directly improve customer satisfaction scores, not just a coincidence."
        )
    else:
        log_insight(
            f"No statistically significant difference in review scores between "
            f"on-time and late orders (p={p_value:.4f}). Business implication: "
            f"delivery lateness alone may not be the main driver of review "
            f"scores — other factors (product quality, price) may matter more."
        )


# ==========================================================================
# STEP 3: SUPERVISED LEARNING — CLASSIFICATION (predict negative review)
# ==========================================================================
def prepare_classification_data(df):
    """
    Builds the feature matrix (X) and target (y) for our classification
    task: "will this order get a NEGATIVE review (score 1 or 2)?"

    X = the input features the model learns from
    y = the thing we're trying to predict (0 = not negative, 1 = negative)
    """
    print("\n[STEP 3a] Preparing data for classification...")
    st.header("Step 3: Classification — Predicting Negative Reviews")
    st.subheader("3a. Preparing the data")

    df = df.dropna(subset=["review_score"]).copy()

    # Target: 1 if review_score is 1 or 2 (negative), else 0
    df["is_negative_review"] = (df["review_score"] <= 2).astype(int)

    feature_cols = [
        "price", "freight_value", "delivery_days", "delivery_delay_days",
        "item_total_value", "freight_ratio", "payment_installments_max",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Drop rows with missing values in the features we're using —
    # most models can't handle NaN directly.
    model_df = df.dropna(subset=feature_cols)

    X = model_df[feature_cols]
    y = model_df["is_negative_review"]

    print(f"  features used: {feature_cols}")
    print(f"  rows available for modelling: {len(X)}")
    print(f"  negative review rate in data: {y.mean():.2%}")

    st.write(f"**Features used:** {', '.join(feature_cols)}")
    col1, col2 = st.columns(2)
    col1.metric("Rows available for modelling", f"{len(X):,}")
    col2.metric("Negative review rate", f"{y.mean():.2%}")

    return X, y, feature_cols


def run_classification(X, y, feature_cols):
    """
    Trains TWO classification models — logistic regression (simple,
    interpretable) and random forest (more flexible, usually more
    accurate) — and compares them using standard classification metrics.
    """
    print("\n[STEP 3b] Training classification models...")
    st.subheader("3b. Training classification models")

    # --- Train/test split: hold back 20% of data the model never sees
    # during training, so we can honestly check how well it generalizes.
    # random_state=42 just makes the split reproducible every time you run it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic regression is sensitive to feature scale, so we standardize
    # (mean=0, std=1). Random forest doesn't need this, but it doesn't hurt it.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Model 1: Logistic Regression ---
    # class_weight="balanced" tells the model to pay extra attention to the
    # minority class (negative reviews are usually rarer than positive ones).
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]  # probability of class 1
    st.markdown("**Logistic Regression results:**")
    results["Logistic Regression"] = evaluate_classifier(y_test, y_pred_lr, y_prob_lr)

    # --- Model 2: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)  # tree models don't need scaled features
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    st.markdown("**Random Forest results:**")
    results["Random Forest"] = evaluate_classifier(y_test, y_pred_rf, y_prob_rf)

    # --- 5-fold cross-validation on the random forest (VALIDATION step) ---
    # Instead of trusting a single train/test split, we split the data into
    # 5 chunks, train on 4 and test on 1, five times, rotating which chunk
    # is held out. This gives a more reliable picture of model performance.
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1")
    print(f"  5-fold cross-validation F1 scores: {np.round(cv_scores, 3)}")
    print(f"  cross-validation F1 mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    st.write(
        f"5-fold cross-validation F1 mean: **{cv_scores.mean():.3f} "
        f"(+/- {cv_scores.std():.3f})** — scores: {np.round(cv_scores, 3).tolist()}"
    )

    # --- Feature importance from random forest — which factors matter most ---
    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  feature importance (random forest):")
    print(importance.round(3))
    st.write("**Feature importance (random forest):**")
    st.dataframe(importance.round(3).rename("importance"))

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_title("What Predicts a Negative Review? (Feature Importance)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "classification_feature_importance.png", dpi=150)
    st.pyplot(fig)
    plt.close(fig)

    # --- ROC curve comparison ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, y_prob in [("Logistic Regression", y_prob_lr), ("Random Forest", y_prob_rf)]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # random-guess baseline
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Predicting Negative Reviews")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "classification_roc_curve.png", dpi=150)
    st.pyplot(fig)
    plt.close(fig)

    # --- Business-language interpretation ---
    best_model = max(results, key=lambda k: results[k]["f1"])
    best_metrics = results[best_model]
    top_driver = importance.index[0]

    log_insight(
        f"The {best_model} model predicts negative reviews with "
        f"{best_metrics['accuracy']:.1%} accuracy and an F1 score of "
        f"{best_metrics['f1']:.2f} (F1 balances catching real negative "
        f"reviews vs. false alarms). The strongest predictor is "
        f"'{top_driver}' — meaning the business should monitor and act on "
        f"this factor early to head off dissatisfied customers before they "
        f"leave a bad review."
    )

    return results, rf, importance


def evaluate_classifier(y_test, y_pred, y_prob):
    """
    Computes the standard set of classification metrics:
      accuracy  = overall % of correct predictions
      precision = of orders we FLAGGED as negative, how many really were?
                  (high precision = few false alarms)
      recall    = of orders that WERE actually negative, how many did we catch?
                  (high recall = few missed problems)
      f1        = balance between precision and recall (single summary number)
      roc_auc   = how well the model ranks negative vs positive reviews
                  overall, across all possible decision thresholds (0.5=random, 1.0=perfect)
    """
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred)

    print(f"    accuracy={metrics['accuracy']:.3f}  precision={metrics['precision']:.3f}  "
          f"recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
    print(f"    confusion matrix:\n{cm}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1", f"{metrics['f1']:.3f}")
    m5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    st.text(f"Confusion matrix:\n{cm}")

    return metrics


# ==========================================================================
# STEP 4: SUPERVISED LEARNING — REGRESSION (predict delivery_days)
# ==========================================================================
def run_regression(df):
    """
    A second, simpler supervised learning example: predicting a NUMBER
    (delivery_days) instead of a category. Useful to show both supervised
    learning types (classification AND regression) in one project.
    """
    print("\n[STEP 4] Training regression model: predicting delivery_days...")
    st.header("Step 4: Regression — Predicting Delivery Days")

    feature_cols = ["price", "freight_value", "item_total_value"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = "delivery_days"

    if target_col not in df.columns or not feature_cols:
        print("  required columns missing, skipping this step")
        st.warning("Required columns missing, skipping this step.")
        return

    model_df = df.dropna(subset=feature_cols + [target_col])
    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE={mae:.2f} days, RMSE={rmse:.2f} days, R2={r2:.3f}")
    r1, r2_col, r3 = st.columns(3)
    r1.metric("MAE", f"{mae:.2f} days")
    r2_col.metric("RMSE", f"{rmse:.2f} days")
    r3.metric("R²", f"{r2:.3f}")

    # R^2 close to 0 means these features barely explain delivery time —
    # a common, honest finding (delivery time depends more on logistics/
    # geography than on order price), which is itself a useful business insight.
    log_insight(
        f"A linear regression predicting delivery time from order value and "
        f"freight cost explains only {r2:.1%} of the variation (R²={r2:.2f}), "
        f"with an average error of about {mae:.1f} days. Business implication: "
        f"delivery speed is driven mainly by logistics/geography, not order "
        f"value — so improving delivery times requires fixing shipping "
        f"operations, not pricing."
    )


# ==========================================================================
# STEP 5: UNSUPERVISED LEARNING — CLUSTERING + DIMENSIONALITY REDUCTION
# ==========================================================================
def run_clustering(df):
    """
    BUSINESS QUESTION: "Are there natural groups of orders/customers with
    different experience profiles (e.g. cheap+fast+happy vs expensive+
    slow+unhappy) that we should treat differently?"

    K-MEANS groups rows into K clusters by minimizing the distance between
    each point and its cluster's center. It's UNSUPERVISED because we never
    tell it the "right answer" — it finds structure on its own.
    """
    print("\n[STEP 5] Clustering orders into customer-experience segments...")
    st.header("Step 5: Clustering — Customer Experience Segments")

    cluster_cols = ["price", "freight_value", "delivery_days", "review_score"]
    cluster_cols = [c for c in cluster_cols if c in df.columns]
    model_df = df.dropna(subset=cluster_cols).copy()

    # Scale features first — K-Means uses distance, so unscaled columns
    # (e.g. price in reais vs delivery_days in single digits) would unfairly
    # dominate the clustering.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[cluster_cols])

    # --- Choose K using the "elbow method": try several K values and plot
    # how much the within-cluster error (inertia) drops. We pick the point
    # where adding more clusters stops helping much (the "elbow").
    inertias = []
    k_range = range(2, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(k_range), inertias, marker="o")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia (within-cluster error)")
    ax.set_title("Elbow Method for Choosing K")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "clustering_elbow.png", dpi=150)
    st.pyplot(fig)
    plt.close(fig)

    # For this project we fix K=4 (a reasonable, interpretable number of
    # customer segments) — adjust based on where your elbow chart bends.
    K = 4
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    model_df["cluster"] = kmeans.fit_predict(X_scaled)

    # Silhouette score: measures how well-separated the clusters are.
    # Ranges -1 to 1; higher = clusters are dense and well-separated.
    sil_score = silhouette_score(X_scaled, model_df["cluster"])
    print(f"  K={K}, silhouette score={sil_score:.3f}")
    st.write(f"K = **{K}**, silhouette score = **{sil_score:.3f}**")

    # --- Describe each cluster in plain terms (the segment "profile") ---
    cluster_profile = model_df.groupby("cluster")[cluster_cols].mean().round(2)
    cluster_profile["n_orders"] = model_df.groupby("cluster").size()
    print("\n  cluster profiles (average values per segment):")
    print(cluster_profile)
    st.write("**Cluster profiles (average values per segment):**")
    st.dataframe(cluster_profile)

    # --- PCA: Dimensionality Reduction ---
    # We have 4 features (4 dimensions) — too many to plot directly. PCA
    # compresses them into 2 "principal components" that capture as much
    # of the original variation as possible, purely so we can SEE the
    # clusters on a 2D scatterplot. It's a visualization tool here, not a
    # new business metric.
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: 2 components explain {explained:.1%} of total variance")
    st.write(f"PCA: 2 components explain **{explained:.1%}** of total variance")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=model_df["cluster"], cmap="tab10", alpha=0.5, s=15
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Order Segments (PCA view, {explained:.0%} variance explained)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "clustering_pca.png", dpi=150)
    st.pyplot(fig)
    plt.close(fig)

    # --- Business-language interpretation: name the most/least happy segment ---
    if "review_score" in cluster_profile.columns:
        happiest = cluster_profile["review_score"].idxmax()
        unhappiest = cluster_profile["review_score"].idxmin()
        log_insight(
            f"Order segmentation (K-Means, K={K}, silhouette={sil_score:.2f}) "
            f"reveals distinct customer-experience groups. Segment {happiest} "
            f"has the highest satisfaction (avg review "
            f"{cluster_profile.loc[happiest, 'review_score']:.2f}) while "
            f"segment {unhappiest} has the lowest "
            f"({cluster_profile.loc[unhappiest, 'review_score']:.2f}). "
            f"Business implication: the unhappiest segment should be "
            f"prioritized for service improvements — check its average "
            f"price/freight/delivery_days values above to identify the cause."
        )

    return cluster_profile


# ==========================================================================
# STEP 6: SAVE SUMMARY
# ==========================================================================
def save_summary():
    """Writes all business-language insights collected during this script
    into a single text file — this becomes your Phase 4 interpretation writeup."""
    print("\n[STEP 6] Saving summary...")
    st.header("Step 6: Summary — All Business Insights")
    with open(SUMMARY_PATH, "w") as f:
        f.write("PHASE 4 — STATISTICAL ANALYSIS & MACHINE LEARNING: BUSINESS INTERPRETATION\n")
        f.write("=" * 75 + "\n\n")
        for i, item in enumerate(insights, start=1):
            f.write(f"{i}. {item}\n\n")
    print(f"  wrote summary to {SUMMARY_PATH}")
    for i, item in enumerate(insights, start=1):
        st.markdown(f"**{i}.** {item}")
    st.success(f"Summary saved to `{SUMMARY_PATH}`")


# ==========================================================================
# MAIN — RUNS ALL STEPS IN ORDER
# ==========================================================================
def main():
    st.set_page_config(page_title="Phase 4: Statistical Analysis & ML", layout="wide")
    st.title("Phase 4: Statistical Analysis and Machine Learning")
    st.caption("Olist Brazilian E-commerce Dataset — CSIT 608")

    print("=" * 70)
    print("PHASE 4: STATISTICAL ANALYSIS AND MACHINE LEARNING")
    print("=" * 70)

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = load_analysis_table(engine)

    # --- Statistical analysis (hypothesis test + confidence interval) ---
    hypothesis_test_late_delivery(df)

    # --- Supervised learning: classification ---
    X, y, feature_cols = prepare_classification_data(df)
    run_classification(X, y, feature_cols)

    # --- Supervised learning: regression (bonus second example) ---
    run_regression(df)

    # --- Unsupervised learning: clustering + PCA ---
    run_clustering(df)

    save_summary()

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE. Check outputs/figures/ and outputs/phase4_summary.txt")
    print("=" * 70)
    st.success("Phase 4 complete. Charts saved to `outputs/figures/`, summary saved to `outputs/phase4_summary.txt`.")


if __name__ == "__main__":
    main()
