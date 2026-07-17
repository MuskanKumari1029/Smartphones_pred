"""
Step 6: Content-Based Recommendation Engine (v2 - fixed)

Two recommendation modes:
1. recommend_by_budget(budget, priority) -> best phones within budget for a stated priority,
   using a weighted normalized spec score (not raw cosine similarity, which broke down for
   low-magnitude budget phones) combined with the value score from Step 3.
2. similar_phones(model_name) -> classic "phones similar to X" using cosine similarity on
   the full normalized spec vector of a SPECIFIC reference phone. This is where cosine
   similarity is the right tool, since we're comparing two concrete points, not a point to
   an arbitrary "ideal" vector.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv('data/with_segments.csv').reset_index(drop=True)

FEATURES = ['ram_capacity', 'internal_memory', 'battery_capacity', 'screen_size',
            'refresh_rate', 'primary_camera_rear', 'primary_camera_front',
            'processor_speed', 'rating']

scaler = MinMaxScaler()
feat_matrix = scaler.fit_transform(df[FEATURES])  # 0-1 normalized, used by both modes

PRIORITY_WEIGHTS = {
    'camera':  {'primary_camera_rear': 3, 'primary_camera_front': 2},
    'gaming':  {'processor_speed': 3, 'refresh_rate': 2.5, 'ram_capacity': 2},
    'battery': {'battery_capacity': 3},
    'overall': {}
}


def recommend_by_budget(budget, priority='overall', top_n=5, brand_preference=None):
    """
    Weighted normalized spec score (0-1 per feature after MinMax scaling) averaged with
    priority weights, blended with the value score. This directly rewards phones with
    HIGH absolute spec values in the prioritized dimensions, unlike a cosine-similarity-
    to-ideal approach which can be fooled by proportionally-similar-but-low-spec phones.
    """
    candidates = df[df['price'] <= budget].copy()
    if candidates.empty:
        return pd.DataFrame()

    idx = candidates.index
    weights = np.ones(len(FEATURES))
    for feat, w in PRIORITY_WEIGHTS.get(priority, {}).items():
        weights[FEATURES.index(feat)] = w
    weights = weights / weights.sum()  # normalize so score stays in a comparable 0-1 range

    weighted_spec_score = (feat_matrix[idx] * weights).sum(axis=1)
    candidates['spec_score'] = weighted_spec_score

    candidates['value_score_norm'] = (candidates['value_score_pct'] - df['value_score_pct'].min()) / \
                                      (df['value_score_pct'].max() - df['value_score_pct'].min() + 1e-9)

    candidates['recommend_score'] = 0.65 * candidates['spec_score'] + 0.35 * candidates['value_score_norm']

    if brand_preference:
        candidates.loc[candidates['brand_name'].str.lower() == brand_preference.lower(), 'recommend_score'] += 0.05

    result = candidates.sort_values('recommend_score', ascending=False).head(top_n)
    return result[['brand_name', 'model', 'price', 'predicted_fair_price', 'value_label',
                    'segment_name', 'ram_capacity', 'battery_capacity', 'primary_camera_rear',
                    'refresh_rate', 'recommend_score']]


def similar_phones(model_name, top_n=5):
    """Find phones with the most similar overall spec profile to a given phone (by model name)."""
    matches = df[df['model'].str.lower() == model_name.lower()]
    if matches.empty:
        return pd.DataFrame()
    ref_idx = matches.index[0]
    sims = cosine_similarity(feat_matrix[ref_idx].reshape(1, -1), feat_matrix).flatten()
    sim_df = df.copy()
    sim_df['similarity'] = sims
    sim_df = sim_df[sim_df.index != ref_idx]  # exclude itself
    result = sim_df.sort_values('similarity', ascending=False).head(top_n)
    return result[['brand_name', 'model', 'price', 'segment_name', 'similarity']]


if __name__ == '__main__':
    print("=== recommend_by_budget: budget=20000, priority=camera ===")
    print(recommend_by_budget(20000, priority='camera').to_string(index=False))

    print("\n=== recommend_by_budget: budget=15000, priority=gaming ===")
    print(recommend_by_budget(15000, priority='gaming').to_string(index=False))

    print("\n=== recommend_by_budget: budget=60000, priority=overall ===")
    print(recommend_by_budget(60000, priority='overall').to_string(index=False))

    print("\n=== similar_phones: 'OnePlus 11 5G' ===")
    print(similar_phones('OnePlus 11 5G').to_string(index=False))

    joblib.dump(scaler, 'models/recommender_scaler.pkl')
    joblib.dump(FEATURES, 'models/recommender_features.pkl')
    print("\nSaved recommender scaler + feature list to models/")
