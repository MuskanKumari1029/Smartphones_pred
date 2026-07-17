"""
Smartphone Value Intelligence Engine
Streamlit app: predicted fair price, value score, market segment, and
budget/priority-based recommendations, all built on a spec dataset of 980 phones.

Run locally with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smartphone Value Intelligence Engine", layout="wide")

DATA_DIR = "app_data"

@st.cache_data
def load_data():
    return pd.read_csv(f"{DATA_DIR}/with_segments.csv")

@st.cache_resource
def load_models():
    price_model = joblib.load(f"{DATA_DIR}/best_price_model.pkl")
    feature_cols = joblib.load(f"{DATA_DIR}/feature_columns.pkl")
    rec_scaler = joblib.load(f"{DATA_DIR}/recommender_scaler.pkl")
    rec_features = joblib.load(f"{DATA_DIR}/recommender_features.pkl")
    return price_model, feature_cols, rec_scaler, rec_features

df = load_data()
price_model, feature_cols, rec_scaler, rec_features = load_models()
feat_matrix = rec_scaler.transform(df[rec_features])

PRIORITY_WEIGHTS = {
    'camera':  {'primary_camera_rear': 3, 'primary_camera_front': 2},
    'gaming':  {'processor_speed': 3, 'refresh_rate': 2.5, 'ram_capacity': 2},
    'battery': {'battery_capacity': 3},
    'overall': {}
}

def recommend_by_budget(budget, priority='overall', top_n=8, brand_preference=None):
    candidates = df[df['price'] <= budget].copy()
    if candidates.empty:
        return pd.DataFrame()
    idx = candidates.index
    weights = np.ones(len(rec_features))
    for feat, w in PRIORITY_WEIGHTS.get(priority, {}).items():
        weights[rec_features.index(feat)] = w
    weights = weights / weights.sum()
    candidates['spec_score'] = (feat_matrix[idx] * weights).sum(axis=1)
    candidates['value_score_norm'] = (candidates['value_score_pct'] - df['value_score_pct'].min()) / \
                                      (df['value_score_pct'].max() - df['value_score_pct'].min() + 1e-9)
    candidates['recommend_score'] = 0.65 * candidates['spec_score'] + 0.35 * candidates['value_score_norm']
    if brand_preference and brand_preference != "Any":
        candidates.loc[candidates['brand_name'].str.lower() == brand_preference.lower(), 'recommend_score'] += 0.05
    return candidates.sort_values('recommend_score', ascending=False).head(top_n)

def similar_phones(model_name, top_n=5):
    matches = df[df['model'] == model_name]
    if matches.empty:
        return pd.DataFrame()
    ref_idx = matches.index[0]
    sims = cosine_similarity(feat_matrix[ref_idx].reshape(1, -1), feat_matrix).flatten()
    sim_df = df.copy()
    sim_df['similarity'] = sims
    sim_df = sim_df[sim_df.index != ref_idx]
    return sim_df.sort_values('similarity', ascending=False).head(top_n)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📱 Smartphone Value Intelligence Engine")
st.sidebar.markdown("A spec-based price fairness model + recommender, trained on 980 phones.")
mode = st.sidebar.radio("Choose a tool", ["Value Lookup", "Budget Recommender", "Similar Phones", "Market Segments"])

# ---------------- MODE 1: VALUE LOOKUP ----------------
if mode == "Value Lookup":
    st.header("🔍 Is this phone fairly priced?")
    brand_filter = st.selectbox("Brand", ["All"] + sorted(df['brand_name'].unique().tolist()))
    filtered = df if brand_filter == "All" else df[df['brand_name'] == brand_filter]
    model_choice = st.selectbox("Model", sorted(filtered['model'].unique().tolist()))

    row = df[df['model'] == model_choice].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Listed Price", f"₹{row['price']:,.0f}")
    col2.metric("Model-Predicted Fair Price", f"₹{row['predicted_fair_price']:,.0f}")
    col3.metric("Value Verdict", row['value_label'],
                delta=f"{row['value_score_pct']:.1f}%")

    st.caption(f"Segment: **{row['segment_name']}**  |  Brand tier: **{row['brand_tier']}**  |  Rating: {row['rating']}/100")

    with st.expander("Key specs"):
        spec_cols = ['ram_capacity', 'internal_memory', 'battery_capacity', 'screen_size',
                     'refresh_rate', 'primary_camera_rear', 'primary_camera_front', 'processor_speed']
        st.table(row[spec_cols])

# ---------------- MODE 2: BUDGET RECOMMENDER ----------------
elif mode == "Budget Recommender":
    st.header("🎯 Find the best phone for your budget")
    c1, c2, c3 = st.columns(3)
    budget = c1.slider("Max budget (₹)", 3000, 200000, 20000, step=1000)
    priority = c2.selectbox("Priority", ["overall", "camera", "gaming", "battery"])
    brand_pref = c3.selectbox("Preferred brand (optional)", ["Any"] + sorted(df['brand_name'].unique().tolist()))

    results = recommend_by_budget(budget, priority, top_n=8, brand_preference=brand_pref)
    if results.empty:
        st.warning("No phones found under this budget.")
    else:
        display_cols = ['brand_name', 'model', 'price', 'predicted_fair_price', 'value_label',
                         'segment_name', 'ram_capacity', 'battery_capacity', 'primary_camera_rear', 'refresh_rate']
        st.dataframe(results[display_cols].rename(columns={
            'brand_name': 'Brand', 'model': 'Model', 'price': 'Price (₹)',
            'predicted_fair_price': 'Fair Price (₹)', 'value_label': 'Value',
            'segment_name': 'Segment', 'ram_capacity': 'RAM (GB)',
            'battery_capacity': 'Battery (mAh)', 'primary_camera_rear': 'Rear Cam (MP)',
            'refresh_rate': 'Refresh (Hz)'
        }), use_container_width=True, hide_index=True)

# ---------------- MODE 3: SIMILAR PHONES ----------------
elif mode == "Similar Phones":
    st.header("🔗 Find phones similar to one you like")
    ref_model = st.selectbox("Reference phone", sorted(df['model'].unique().tolist()))
    results = similar_phones(ref_model, top_n=6)
    st.dataframe(results.rename(columns={
        'brand_name': 'Brand', 'model': 'Model', 'price': 'Price (₹)',
        'segment_name': 'Segment', 'similarity': 'Similarity'
    }), use_container_width=True, hide_index=True)

# ---------------- MODE 4: MARKET SEGMENTS ----------------
elif mode == "Market Segments":
    st.header("📊 Market Segments (unsupervised clustering on specs, not price)")
    seg_summary = df.groupby('segment_name').agg(
        count=('model', 'count'),
        avg_price=('price', 'mean'),
        avg_ram=('ram_capacity', 'mean'),
        avg_battery=('battery_capacity', 'mean'),
        avg_camera=('primary_camera_rear', 'mean')
    ).round(0).sort_values('avg_price')
    st.dataframe(seg_summary, use_container_width=True)

    seg_choice = st.selectbox("Explore a segment", seg_summary.index.tolist())
    st.dataframe(df[df['segment_name'] == seg_choice][['brand_name', 'model', 'price', 'value_label']]
                 .sort_values('price').rename(columns={'brand_name': 'Brand', 'model': 'Model',
                                                         'price': 'Price (₹)', 'value_label': 'Value'}),
                 use_container_width=True, hide_index=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with scikit-learn, XGBoost/RandomForest, SHAP, and cosine-similarity content-based filtering.")
