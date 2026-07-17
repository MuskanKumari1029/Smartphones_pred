"""
Step 4: SHAP Explainability
Global feature importance + example per-phone explanation.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = joblib.load('models/best_price_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')
X_test = pd.read_csv('data/X_test.csv')[feature_cols]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ---- Global feature importance (bar) ----
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False, max_display=15)
plt.title('Global Feature Importance — What Drives Smartphone Price', fontsize=12)
plt.tight_layout()
plt.savefig('outputs_plots/shap_global_importance.png', dpi=150)
plt.close()

# ---- Beeswarm summary (direction of effect) ----
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_test, show=False, max_display=15)
plt.title('SHAP Summary — Direction & Magnitude of Feature Effects', fontsize=12)
plt.tight_layout()
plt.savefig('outputs_plots/shap_beeswarm.png', dpi=150)
plt.close()

# ---- Per-phone explanation example (pick one interesting row) ----
df_full = pd.read_csv('data/with_value_scores.csv')
X_test_idx = X_test.index
sample_row_pos = 0
sample_idx = X_test_idx[sample_row_pos]

phone_info = df_full.loc[sample_idx, ['brand_name', 'model', 'price', 'predicted_fair_price', 'value_label']]
print("Example explained phone:")
print(phone_info)

base_val = explainer.expected_value
if hasattr(base_val, '__len__'):
    base_val = base_val[0]

plt.figure(figsize=(10, 5))
shap.waterfall_plot(shap.Explanation(
    values=shap_values[sample_row_pos],
    base_values=base_val,
    data=X_test.iloc[sample_row_pos],
    feature_names=X_test.columns.tolist()
), max_display=12, show=False)
plt.title(f"Why this price? {phone_info['brand_name']} {phone_info['model']}", fontsize=11)
plt.tight_layout()
plt.savefig('outputs_plots/shap_waterfall_example.png', dpi=150)
plt.close()

# Save mean abs shap importance as a table too (useful for README / resume bullet)
mean_abs_shap = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
mean_abs_shap.to_csv('outputs_plots/shap_feature_importance_table.csv', index=False)
print("\nTop 10 features by mean |SHAP value|:")
print(mean_abs_shap.head(10).to_string(index=False))

print("\nSaved SHAP plots to outputs_plots/")
