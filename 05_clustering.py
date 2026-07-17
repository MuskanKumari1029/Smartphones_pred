"""
Step 5: Market Segmentation via Unsupervised Clustering
Clusters phones by SPECS ONLY (not price) to discover natural market segments.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('data/with_value_scores.csv')

# Spec-only features for clustering (deliberately excluding price)
cluster_features = [
    'ram_capacity', 'internal_memory', 'battery_capacity', 'screen_size',
    'refresh_rate', 'primary_camera_rear', 'primary_camera_front',
    'num_rear_cameras', 'processor_speed', 'total_cameras',
    'pixel_density_proxy', 'rating'
]

X_cluster = df[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Elbow + silhouette to choose k
inertias, sil_scores = [], []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(list(K_range), inertias, marker='o')
ax[0].set_title('Elbow Method'); ax[0].set_xlabel('k'); ax[0].set_ylabel('Inertia')
ax[1].plot(list(K_range), sil_scores, marker='o', color='darkorange')
ax[1].set_title('Silhouette Score'); ax[1].set_xlabel('k'); ax[1].set_ylabel('Score')
plt.tight_layout()
plt.savefig('outputs_plots/clustering_k_selection.png', dpi=150)
plt.close()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"Silhouette scores: {dict(zip(K_range, [round(s,3) for s in sil_scores]))}")
print(f"Chosen k (by silhouette): {best_k}")

# Final clustering with chosen k (using 5 for interpretable segment naming if close to best)
final_k = best_k if best_k >= 3 else 5
km_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

# Profile each cluster to assign human-readable segment names
profile = df.groupby('cluster')[cluster_features + ['price']].mean().round(1)
print("\nCluster profiles (mean specs):")
print(profile.to_string())

# Auto-name segments based on profile characteristics
def name_segment(row, price_median, ram_median, cam_median, refresh_median):
    tags = []
    if row['price'] > price_median * 1.4:
        tags.append('Premium')
    elif row['price'] < price_median * 0.7:
        tags.append('Budget')
    else:
        tags.append('Mid-range')
    if row['refresh_rate'] >= refresh_median * 1.2 and row['ram_capacity'] >= ram_median:
        tags.append('Gaming/Performance')
    if row['primary_camera_rear'] >= cam_median * 1.15:
        tags.append('Camera-focused')
    if row['battery_capacity'] >= df['battery_capacity'].median() * 1.1:
        tags.append('Battery-focused')
    return ' '.join(tags) if tags else 'All-rounder'

price_med = df['price'].median()
ram_med = df['ram_capacity'].median()
cam_med = df['primary_camera_rear'].median()
refresh_med = df['refresh_rate'].median()

segment_names = {}
for c in profile.index:
    segment_names[c] = name_segment(profile.loc[c], price_med, ram_med, cam_med, refresh_med)

df['segment_name'] = df['cluster'].map(segment_names)

print("\nSegment names assigned:")
for c, n in segment_names.items():
    count = (df['cluster'] == c).sum()
    print(f"  Cluster {c}: '{n}'  ({count} phones, avg price rs.{profile.loc[c,'price']:,.0f})")

df.to_csv('data/with_segments.csv', index=False)
joblib.dump(km_final, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/cluster_scaler.pkl')
joblib.dump(cluster_features, 'models/cluster_features.pkl')
joblib.dump(segment_names, 'models/segment_names.pkl')

print("\nSaved: data/with_segments.csv, models/kmeans_model.pkl")
