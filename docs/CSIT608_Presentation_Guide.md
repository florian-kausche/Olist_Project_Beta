# Final Presentation Script
### AfiLearn Commerce Analytics Office — Brazilian E-Commerce (Olist) Analysis
### CSIT 608, University of Ghana — Team Beta

*Written to be spoken as a team ("we"), and focused on presenting the actual findings and working artefacts — the dashboard, the numbers, the recommendation — rather than narrating the build process step by step. Process detail is kept in the appendix for viva questions, where it belongs.*

---

## Opening

"Good [morning/afternoon]. We are Team Beta, the AfiLearn Commerce Analytics Office. We were asked to understand order performance, customer behaviour, product performance, delivery delays, and customer reviews for Olist — a Brazilian e-commerce marketplace — and turn that into a decision a manager can act on.

Our headline finding: late delivery is costing Olist nearly two full review-score points on average, it is statistically proven, and we can now predict which orders are at risk before the bad review is written. Let us show you the evidence, then the live dashboard, then our recommendation."

---

## 1. The Business Questions We Answered

"We organized our work around six business questions:

1. What are the main revenue, order, and customer patterns over time?
2. Which products, sellers, regions, or payment types contribute most to performance?
3. What data-quality issues affect the reliability of the analysis?
4. Are late deliveries associated with lower review scores or customer dissatisfaction?
5. Can we predict late delivery or a low review score before it happens?
6. Can customers be segmented into meaningful groups for business action?

Every result we're about to show answers one of those six directly."

---

## 2. Growth and Where the Customers Are

*[Show the dashboard's Performance and Customer tabs, or the narrative page's Sections 2–3.]*

"Revenue and order volume both show clear growth over the period, though the growth rate is flattening in the more recent months — which is why retention, not just acquisition, matters going forward. Customers are also geographically concentrated in a small number of states, so delivery and stocking decisions in those states carry outsized impact."

---

## 3. Where Revenue Actually Comes From

*[Show the Product tab / Section 4.]*

"Revenue is concentrated in a small set of product categories. That matters directly for our delivery findings — any delivery-reliability fix should be prioritized for the categories and states that generate the most revenue, not applied evenly everywhere."

---

## 4. The Central Finding: Late Delivery Is Costing Reviews

*[Show the boxplot / Section 5.]*

"We tested it directly: is the gap between on-time and late order review scores real, or could it be random noise? On-time orders — 104,624 of them — average 4.15 out of 5. Late orders — 7,084 of them — average 2.26. That is not a small gap, and it is not chance: our t-statistic was 99.3, with a p-value effectively at zero. We are 95% confident the true difference sits between 1.86 and 1.93 review points.

That single result is the backbone of our recommendation: fixing delivery reliability is the single highest-leverage lever Olist has for improving customer satisfaction."

---

## 5. What Our Models Told Us

*[Show the Section 6 model results.]*

"We didn't stop at proving the relationship — we built a model that predicts it. Our Random Forest classifier predicts whether an order will receive a negative review with 82.9% accuracy and an F1 score of 0.42, validated with 5-fold cross-validation to confirm it generalizes rather than memorizes. The model confirms what the statistics already showed: delivery delay is the dominant factor, responsible for over 80% of the model's predictive weight, ahead of price, freight cost, and payment installments combined.

We also segmented orders into four customer-experience groups using clustering, which surfaced a clearly unhappy segment the business should prioritize for service recovery — and, as a secondary check, we tested whether delivery time itself could be predicted from order value; it could not, meaning delivery speed is a logistics problem to fix operationally, not something that can be solved through pricing."

---

## 6. The Live Dashboard

*[Open the Streamlit dashboard now.]*

"This is what we're handing to Olist's managers — not a static report, but a live tool. It has five tabs: Performance, Customer, Product, Delivery, and Review, with filters for date range, state, and category that update every chart instantly."

*[Click one filter live — narrow to a single state.]*

"That's the dashboard responding in real time. A manager doesn't need us in the room to explore their own question."

---

## 7. Our Recommendation

*[Show Section 7 of the narrative page and read directly from the screen.]*

"Based on everything above, we recommend:

1. Prioritize delivery-time reduction for the top revenue-generating categories, where late-delivery review damage has the highest financial stakes.
2. Focus operational improvements on the top customer states first, where volume impact is largest.
3. Set an internal SLA alert at the three-day-late threshold.
4. Put the negative-review prediction model into operational use, flagging at-risk orders before the customer leaves a bad review.
5. Use the customer segments to target retention offers, since growth is flattening and retention is now the more efficient lever than pure acquisition."

---

## 8. Limitations — Stated Upfront

"Two things we want to be transparent about. First, what we've shown is a strong statistical association between delivery delay and review score — not a controlled experiment, so we describe it as correlation, not proven causation. Second, our prediction model has a precision-recall trade-off: some at-risk orders will still be missed, and some flagged orders won't actually turn negative. Both are normal for this type of analysis, and we've outlined next steps to address them with more time and data."

---

## Closing

"To summarize: we took Olist's raw order, payment, product, and review data, proved statistically that late delivery is driving down customer satisfaction, built a validated model that predicts which orders are at risk, segmented customers by experience, and delivered both a live dashboard and a clear, evidence-based recommendation. We're happy to take questions, or go deeper into any part of the analysis."

---

## Viva / Q&A Reference — Process Detail On Request Only

*Everything below answers "how did you actually do this" — we don't lead with it, but any of us should be able to answer confidently if asked, since the brief states any team member may be questioned on any part of the workflow.*

### Anticipated Questions

**"Why SQLite instead of a database server?"**
"SQLite is a single-file database — no server to install, fully portable, and it's exactly what pandas and SQLAlchemy work with natively. At Olist's actual scale we would migrate to PostgreSQL; the workflow logic wouldn't change."

**"How do you know your model isn't overfitting?"**
"We used a held-out test set the model never saw during training, plus 5-fold cross-validation — training and testing five times on different data slices and averaging the results. Consistent scores across those five folds is what confirms the model generalizes."

**"Why cap outliers instead of deleting them?"**
"Deleting a genuinely expensive order removes real sales data, not bad data. Capping — winsorizing — keeps the row but stops it from distorting averages, which is the more defensible choice when a high value could easily be a real premium transaction rather than an error."

**"What would you do with more time or data?"**
"Extend the dashboard with a live at-risk-orders view driven directly by the classification model, and test the delivery-review relationship with a real controlled experiment rather than relying on observed correlation alone."

**"What data-quality issues did you find?"**
"After merging, the columns with the most missing values were ones tied to orders without a matching review or payment record — not a data-quality error, just an expected gap from the join. We also found and removed exact duplicate rows, and used the IQR method to cap statistical outliers in price, freight value, and total payment value rather than deleting them."

---

## Technical Appendix A — Commands We Ran

**Environment setup:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".venv\Scripts\Activate.ps1"
pip install matplotlib seaborn scikit-learn scipy streamlit plotly sqlalchemy pandas
```

**Stage 1 — Acquire (ingestion):**
```powershell
python scripts/import.py
```
Loads every CSV in `data/raw/` into `db/olist.db` as `raw_*` tables.

**Stage 2 — Prepare (cleaning):**
```powershell
python scripts/clean.py
```
Reads each `raw_*` table, writes cleaned `clean_*` tables plus CSV backups in `data/cleaned/`.

**Stage 3 — Analyse (EDA):**
```powershell
python scripts/phase3.py
```
Merges the seven `clean_*` tables into `analysis_table`, saves charts to `outputs/figures/` and a summary to `outputs/phase3_summary.txt`.

**Stage 4 — Model (statistics & ML):**
```powershell
python scripts/phase4.py
```
Runs the hypothesis test, trains both classifiers and the regression model, runs K-Means + PCA, saves `outputs/phase4_summary.txt`. A Streamlit version (`streamlit run scripts/phase4.py`) renders the same analysis as a browser page.

**Stage 5 — Visualise (dashboard):**
```powershell
streamlit run scripts/phase5.py
```
Launches the live filterable dashboard at `http://localhost:8501`.

**Stage 6 — Recommend (data story):**
```powershell
streamlit run scripts/phase6.py
```
Launches the narrative page, computing model metrics live from `analysis_table` so it can never drift out of sync with the dashboard.

---

## Technical Appendix B — Formulae, With Our Actual Values

### B1. IQR outlier detection (Prepare/Analyse phase)
```
Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 − Q1
Lower bound = Q1 − 1.5 × IQR      Upper bound = Q3 + 1.5 × IQR
```
Applied to `price`, `freight_value`, `total_payment_value`; values outside the bounds are clipped (winsorized), not deleted.

### B2. Log transform
```
price_log = ln(1 + price)
```
`log1p` handles a price of exactly 0 safely.

### B3. Standardization (z-score)
```
z = (x − mean) / standard deviation
```
Applied to `price`, `freight_value`, `delivery_days` before clustering.

### B4. Welch's t-test
```
t = (mean₁ − mean₂) / √(s₁²/n₁ + s₂²/n₂)
```
**Our values:** on-time n = 104,624, mean = 4.153; late n = 7,084, mean = 2.256; t = 99.306; p ≈ 0.000000

### B5. 95% confidence interval for a difference in means
```
SE = √(s₁²/n₁ + s₂²/n₂)      CI = (mean₁ − mean₂) ± 1.96 × SE
```
**Our values:** difference = 1.897, 95% CI = (1.859, 1.934)

### B6. Classification metrics
```
Accuracy = (TP+TN)/(TP+TN+FP+FN)   Precision = TP/(TP+FP)
Recall = TP/(TP+FN)                 F1 = 2×(Precision×Recall)/(Precision+Recall)
```
**Our values:**

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.710 | 0.256 | 0.506 | 0.340 | 0.670 |
| Random Forest | 0.829 | 0.422 | 0.424 | 0.423 | 0.710 |

Random Forest confusion matrix: `[[16764, 1878], [1862, 1370]]`
5-fold CV F1 scores: `[0.397, 0.410, 0.404, 0.402, 0.392]` → mean 0.401 (± 0.006)

Feature importance (sums to 1.0):
```
delivery_delay_days        0.426
delivery_days               0.391
freight_value                0.060
payment_installments_max     0.036
freight_ratio                 0.031
item_total_value              0.029
price                          0.028
```

### B7. Regression metrics
```
MAE = mean(|actual − predicted|)   RMSE = √(mean((actual − predicted)²))   R² = 1 − (SS_res/SS_tot)
```
**Our values:** MAE = 5.93 days, RMSE = 9.26 days, R² = 0.090

### B8. K-Means and silhouette score
```
K-Means minimizes inertia = Σ (distance to cluster center)²
Silhouette = (b − a) / max(a, b)
```
**Our values:** K = 4, clustered on standardized `price`, `freight_value`, `delivery_days`, `review_score`.
*(Exact silhouette score and per-cluster averages print live each run — read off screen during the demo rather than quoted here, since K-Means re-fits on every run.)*

### B9. PCA
Finds new axes (principal components) that are linear combinations of the original features, chosen to capture maximum variance. Used here to compress 4 dimensions to 2 purely for visualizing the clusters; the explained-variance percentage is read live off the chart title.

---

## One-Page Cheat Sheet — Second Screen

- **Pipeline (brief's terminology):** Acquire → Prepare → Analyse → Model → Visualise → Recommend.
- **Headline stat:** on-time avg 4.15 vs late avg 2.26, p≈0, 95% CI (1.86, 1.93).
- **Best model:** Random Forest, 82.9% accuracy, F1 0.42, CV mean 0.401 — delivery_delay_days = strongest predictor.
- **Regression finding:** delivery time R²≈0.09 — logistics problem, not pricing.
- **Clustering:** K=4 segments via K-Means + PCA.
- **Recommendations:** delivery-time reduction, prioritize top states, 3-day SLA alert, operationalize the model, segment-based retention.
- **Limitations, stated upfront:** correlation not causation; precision/recall trade-off.
