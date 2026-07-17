"""
Step 2: Price Prediction Model
Compares Linear Regression, Random Forest, XGBoost.
Predicts log(price), evaluates on original price scale.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import joblib
import json

df = pd.read_csv('data/model_ready.csv')

TARGET = 'log_price'
DROP_FOR_X = ['price', 'log_price']
X = df.drop(columns=DROP_FOR_X)
y = df[TARGET]
y_raw = df['price']

X_train, X_test, y_train, y_test, yraw_train, yraw_test = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42
)

results = {}

def evaluate(name, model, X_tr, X_te, log_pred_te):
    price_pred = np.expm1(log_pred_te)
    price_pred = np.clip(price_pred, 1000, None)
    mae = mean_absolute_error(yraw_test, price_pred)
    mape = mean_absolute_percentage_error(yraw_test, price_pred)
    r2 = r2_score(yraw_test, price_pred)
    results[name] = {'MAE': round(mae, 2), 'MAPE': round(mape * 100, 2), 'R2': round(r2, 4)}
    print(f"{name:25s} | MAE: rs.{mae:>9,.0f} | MAPE: {mape*100:5.2f}% | R2: {r2:.4f}")

# ---- Baseline: Linear Regression ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lin = LinearRegression()
lin.fit(X_train_s, y_train)
evaluate("Linear Regression", lin, X_train_s, X_test_s, lin.predict(X_test_s))

# ---- Random Forest ----
rf = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=2,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate("Random Forest", rf, X_train, X_test, rf.predict(X_test))

# ---- XGBoost (tuned via small grid search) ----
xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}
grid = GridSearchCV(xgb_base, param_grid, cv=KFold(5, shuffle=True, random_state=42),
                     scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X_train, y_train)
best_xgb = grid.best_estimator_
print(f"\nBest XGBoost params: {grid.best_params_}")
evaluate("XGBoost (tuned)", best_xgb, X_train, X_test, best_xgb.predict(X_test))

print("\n=== Model Comparison Summary ===")
for k, v in results.items():
    print(k, v)

# Pick best model by MAE
best_name = min(results, key=lambda k: results[k]['MAE'])
print(f"\nBest model: {best_name}")

best_model = {'Linear Regression': lin, 'Random Forest': rf, 'XGBoost (tuned)': best_xgb}[best_name]

# Save artifacts
joblib.dump(best_model, 'models/best_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')  # only used if best model is linear
joblib.dump(list(X.columns), 'models/feature_columns.pkl')
with open('models/model_results.json', 'w') as f:
    json.dump({'results': results, 'best_model': best_name}, f, indent=2)

# Save train/test splits for downstream steps (residuals, SHAP)
X_test.to_csv('data/X_test.csv', index=False)
X_train.to_csv('data/X_train.csv', index=False)
yraw_test.to_csv('data/y_test_price.csv', index=False)
yraw_train.to_csv('data/y_train_price.csv', index=False)

print("\nSaved model, scaler, feature columns, and results to models/")
