"""
Step 3: Value Score = Predicted Fair Price vs Actual Price
Positive value score = underpriced (good value). Negative = overpriced.
"""
import pandas as pd
import numpy as np
import joblib

df_full = pd.read_csv('data/cleaned_features.csv')
df_model_ready = pd.read_csv('data/model_ready.csv')

model = joblib.load('models/best_price_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')

X_full = df_model_ready[feature_cols]
log_pred = model.predict(X_full)
predicted_price = np.expm1(log_pred)

df_full['predicted_fair_price'] = predicted_price.round(0)
df_full['value_score_rs'] = (df_full['predicted_fair_price'] - df_full['price']).round(0)
df_full['value_score_pct'] = ((df_full['predicted_fair_price'] - df_full['price']) / df_full['price'] * 100).round(2)

def label_value(pct):
    if pct >= 15:
        return 'Great Value (underpriced)'
    elif pct >= 5:
        return 'Good Value'
    elif pct >= -5:
        return 'Fairly Priced'
    elif pct >= -15:
        return 'Slightly Overpriced'
    else:
        return 'Overpriced'

df_full['value_label'] = df_full['value_score_pct'].apply(label_value)

df_full.to_csv('data/with_value_scores.csv', index=False)

print("=== Value Label Distribution ===")
print(df_full['value_label'].value_counts())

print("\n=== Top 10 BEST VALUE phones (most underpriced relative to specs) ===")
top_value = df_full.sort_values('value_score_pct', ascending=False)[
    ['brand_name', 'model', 'price', 'predicted_fair_price', 'value_score_pct']
].head(10)
print(top_value.to_string(index=False))

print("\n=== Top 10 MOST OVERPRICED phones ===")
top_overpriced = df_full.sort_values('value_score_pct', ascending=True)[
    ['brand_name', 'model', 'price', 'predicted_fair_price', 'value_score_pct']
].head(10)
print(top_overpriced.to_string(index=False))

top_value.to_csv('outputs_plots/top10_best_value.csv', index=False)
top_overpriced.to_csv('outputs_plots/top10_overpriced.csv', index=False)

print("\nSaved: data/with_value_scores.csv, outputs_plots/top10_best_value.csv, outputs_plots/top10_overpriced.csv")
