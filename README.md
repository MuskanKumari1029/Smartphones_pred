# Smartphone Value Intelligence Engine

A spec-based **price fairness model + market segmentation + recommendation engine**,
built on a dataset of 980 smartphones. Instead of just predicting price, this project
answers three questions a real buyer (or a product/pricing analyst) actually cares about:

1. **Is this phone fairly priced given its specs?**
2. **What natural market segments exist, based on specs alone (not price)?**
3. **Given a budget and a priority (camera / gaming / battery), what should I actually buy?**

---

## 1. Problem Framing

Rather than stopping at "predict price → report R²," this project treats the prediction
as an intermediate step. The **residual** between predicted "fair" price and actual
listed price becomes a **value score** — a business-relevant, explainable output.

## 2. Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `01_clean_and_features.py` | Cleaning (median/mode imputation), feature engineering: text-derived model tier (Pro/Max/Ultra/Lite), brand tier, RAM/storage-per-₹1000, battery-per-screen-inch, pixel density proxy |
| 2 | `02_price_model.py` | Trains & compares Linear Regression, Random Forest, XGBoost (grid-searched) on log(price); evaluates on original ₹ scale |
| 3 | `03_value_score.py` | Computes predicted fair price vs. actual price → value score (%) and a 5-tier value label |
| 4 | `04_explainability.py` | SHAP global importance, beeswarm, and a per-phone waterfall explanation |
| 5 | `05_clustering.py` | KMeans segmentation on specs only (price excluded) → auto-named segments (e.g. "Premium Gaming/Performance Camera-focused") |
| 6 | `06_recommender.py` | Two recommender modes: budget+priority weighted scoring, and cosine-similarity "phones similar to X" |
| — | `app.py` | Streamlit app tying all four pieces together into one interactive tool |

## 3. Model Results

Trained on an 80/20 split, evaluated on original ₹ price scale (after inverse log-transform):

| Model | MAE (₹) | MAPE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 4,180 | 12.33% | 0.930 |
| Random Forest | **1,006** | **3.04%** | **0.9946** |
| XGBoost (tuned via grid search) | 1,029 | 2.88% | 0.9941 |

**Random Forest selected** as the production model (best MAE; XGBoost was competitive but
not better after tuning — a useful honest note for interviews: bigger model ≠ automatically better).

### Top price drivers (mean |SHAP value|)
1. `ram_per_1000rs` (RAM efficiency relative to price)
2. `ram_capacity`
3. `internal_memory`
4. `storage_per_1000rs`
5. `processor_speed`

## 4. Value Score — Sanity-Checked Against Reality

The model flags luxury/collector phones as "overpriced" purely from specs — without ever
being told about brand prestige — which is a strong validation signal:

- **Vertu Signature Touch** (₹6,50,000): model predicts fair price of ₹1,67,848 → **-74% (overpriced)**, correctly identifying luxury brand premium unrelated to specs.
- **Xiaomi Redmi K20 Pro Signature Edition** (₹4,80,000): -35%, same story — a limited-edition collector price, not a spec-driven one.

Meanwhile it surfaces genuine bargains: e.g. entry-level phones from Realme/Nokia/itel priced
well below what their specs would predict.

## 5. Market Segments (unsupervised, specs-only)

KMeans clustering (k selected via elbow + silhouette) on specs — **deliberately excluding
price** — surfaces natural segments, auto-named from cluster spec profiles:
- Budget
- Mid-range Camera-focused
- Premium
- Premium Gaming/Performance Camera-focused
- (one small outlier cluster of extreme-battery devices)

This shows the market naturally organizes around camera and performance axes independent
of price tier, not just "cheap vs. expensive."

## 6. Recommendation Engine — Two Approaches, Deliberately Different

- **Budget + priority mode**: weighted normalized spec score blended with value score.
  (An earlier cosine-similarity-to-"ideal-vector" version was tried first and **rejected**
  after it surfaced a 3GB RAM budget phone as a top "gaming" pick — cosine similarity
  compares *direction*, not magnitude, so it can be fooled by low-spec phones that are
  proportionally balanced. Documenting this failure and the fix is intentional — it shows
  the reasoning, not just the working code.)
- **Similar-phones mode**: classic cosine similarity between two concrete spec vectors —
  the right tool here, since we're comparing a real phone to other real phones, not to an
  arbitrary constructed "ideal."

## 7. Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app has four tools: Value Lookup, Budget Recommender, Similar Phones, and Market Segments.

## 8. What This Project Demonstrates

- Supervised regression with proper model comparison and hyperparameter tuning
- Feature engineering from both numeric specs and free-text fields
- Model explainability (SHAP) at both global and individual-prediction level
- Unsupervised learning (clustering) for market segmentation
- Content-based recommendation system design, including catching and fixing a real
  methodological flaw (cosine similarity misuse)
- End-to-end deployment as an interactive tool

## Suggested Resume Bullets

- Built an end-to-end price-fairness and recommendation engine on 980 smartphones,
  achieving 3% MAPE (R²=0.995) with a tuned Random Forest, and used SHAP to explain
  price drivers at global and per-phone level.
- Designed a residual-based "value score" that automatically flagged luxury/limited-edition
  phones as overpriced (e.g. -74% for a ₹6.5L phone) purely from spec data, without brand
  prestige as an input — validating the model's real-world grounding.
- Built a content-based recommendation engine (cosine similarity + weighted spec scoring)
  and market segmentation (KMeans) on specs alone; identified and corrected a similarity-
  metric flaw that initially misranked low-spec phones for performance-priority queries.
- Deployed the full pipeline as an interactive Streamlit app with four tools: price-fairness
  lookup, budget-based recommendations, similar-phone search, and segment exploration.
