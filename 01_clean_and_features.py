"""
Step 1: Cleaning + Feature Engineering
Smartphone Value Intelligence Engine
"""
import pandas as pd
import numpy as np
import re

df = pd.read_csv('data/raw.csv')
print(f"Raw shape: {df.shape}")

# ---------- CLEANING ----------
# Drop rows with no price (target) - none expected, but safe
df = df[df['price'].notna()].copy()

# Fill missing numeric specs with median (per-column, simple and defensible for a portfolio project)
num_cols_to_impute = ['rating', 'num_cores', 'processor_speed', 'battery_capacity',
                       'fast_charging', 'num_front_cameras', 'primary_camera_front']
for col in num_cols_to_impute:
    df[col] = df[col].fillna(df[col].median())

# processor_brand / os: fill with mode
for col in ['processor_brand', 'os']:
    df[col] = df[col].fillna(df[col].mode()[0])

# fast_charging_available already binary flag; if fast_charging missing it was filled above
# extended_upto: NaN genuinely means "no extended memory" -> fill with 0
df['extended_upto'] = df['extended_upto'].fillna(0)

print("Missing values after cleaning:")
print(df.isnull().sum().sum(), "total missing cells remaining")

# ---------- FEATURE ENGINEERING ----------

# 1. Tier tags extracted from model name text (lightweight NLP-style feature extraction)
def extract_tier(model_name):
    name = model_name.lower()
    if any(t in name for t in ['ultra', 'pro max', 'pro+']):
        return 'ultra'
    elif 'pro' in name:
        return 'pro'
    elif 'max' in name:
        return 'max'
    elif any(t in name for t in ['lite', 'se ', ' se']):
        return 'lite'
    elif any(t in name for t in ['neo', 'core', 'y series']):
        return 'entry'
    else:
        return 'standard'

df['model_tier'] = df['model'].apply(extract_tier)

# 2. Brand tier — based on median price per brand (premium / mid / budget)
brand_median_price = df.groupby('brand_name')['price'].median()
def brand_tier_from_price(p):
    if p >= 40000:
        return 'premium'
    elif p >= 18000:
        return 'mid'
    else:
        return 'budget'
brand_tier_map = brand_median_price.apply(brand_tier_from_price)
df['brand_tier'] = df['brand_name'].map(brand_tier_map)

# 3. Spec ratios / density features
df['ram_per_1000rs'] = df['ram_capacity'] / (df['price'] / 1000)
df['storage_per_1000rs'] = df['internal_memory'] / (df['price'] / 1000)
df['battery_per_screen_inch'] = df['battery_capacity'] / df['screen_size']
df['total_cameras'] = df['num_rear_cameras'] + df['num_front_cameras']
df['pixel_density_proxy'] = (df['resolution_width'] * df['resolution_height']) / (df['screen_size'] ** 2)
df['has_fast_charging_flag'] = (df['fast_charging'] >= 30).astype(int)

# 4. Log price (helps regression with the long tail up to 6.5L)
df['log_price'] = np.log1p(df['price'])

# ---------- ENCODE CATEGORICALS FOR MODELING ----------
cat_cols = ['brand_name', 'processor_brand', 'os', 'model_tier', 'brand_tier']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Drop columns not useful as model features
drop_cols = ['model']  # keep model name for display purposes in a separate lookup
model_lookup = df[['model', 'brand_name']].copy()

df_encoded = df_encoded.drop(columns=drop_cols)

df.to_csv('data/cleaned_features.csv', index=False)
df_encoded.to_csv('data/model_ready.csv', index=False)

print(f"\nFinal cleaned shape: {df.shape}")
print(f"Model-ready (encoded) shape: {df_encoded.shape}")
print(f"\nModel tier distribution:\n{df['model_tier'].value_counts()}")
print(f"\nBrand tier distribution:\n{df['brand_tier'].value_counts()}")
print("\nSaved: data/cleaned_features.csv, data/model_ready.csv")
