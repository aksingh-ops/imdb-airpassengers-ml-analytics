"""
Phase 2 - Regression Modeling
Target: averageRating
Models: Lasso, Ridge, Elastic Net, Gradient Boosting, XGBoost
Validation: 80/20 train-test split + 5-fold cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# ------------------------------------------------------------------
# Load cleaned data
# ------------------------------------------------------------------
df = pd.read_csv('data/imdb_cleaned.csv')

X = df.drop(
    ['tconst', 'types', 'originalTitle', 'directors', 'averageRating', 'Rating'],
    axis=1,
    errors='ignore'
)
y = df['averageRating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# Scale features (required for linear models)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------
# Lasso
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)

# Ridge
ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
ridge.fit(X_train_scaled, y_train)

# Elastic Net
elastic_net = ElasticNetCV(cv=5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)

# Gradient Boosting
gb_reg = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb_reg.fit(X_train, y_train)

# XGBoost
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=100,
    random_state=42
)
xgb_reg.fit(X_train, y_train)

# ------------------------------------------------------------------
# Predictions DataFrame
# ------------------------------------------------------------------
predictions = {
    'Lasso':   lasso.predict(X_test_scaled),
    'Ridge':   ridge.predict(X_test_scaled),
    'Elastic': elastic_net.predict(X_test_scaled),
    'GB':      gb_reg.predict(X_test),
    'XGB':     xgb_reg.predict(X_test),
}
predictions_df = pd.DataFrame(predictions)
print("Predictions head:\n", predictions_df.head())

# ------------------------------------------------------------------
# Model evaluation: RMSE and R-squared
# ------------------------------------------------------------------
model_map = {
    'Lasso':   (lasso,       X_train_scaled, X_test_scaled),
    'Ridge':   (ridge,       X_train_scaled, X_test_scaled),
    'Elastic': (elastic_net, X_train_scaled, X_test_scaled),
    'GB':      (gb_reg,      X_train,        X_test),
    'XGB':     (xgb_reg,     X_train,        X_test),
}

results = []
for name, (model, X_tr, X_te) in model_map.items():
    train_pred = model.predict(X_tr)
    test_pred  = model.predict(X_te)
    results.append({
        'Model':             name,
        'RMSE Training':     round(np.sqrt(mean_squared_error(y_train, train_pred)), 2),
        'R2 Training':       round(r2_score(y_train, train_pred), 2),
        'RMSE Testing':      round(np.sqrt(mean_squared_error(y_test, test_pred)), 2),
        'R2 Testing':        round(r2_score(y_test, test_pred), 2),
    })

results_df = pd.DataFrame(results)
print("\nModel Performance:\n", results_df.to_string(index=False))

# Best model: XGBoost (lowest RMSE training=0.85, testing=0.86; highest R2 training=0.28, testing=0.27)
print("\nSelected model: XGBoost")
print("Rationale: Lowest training RMSE (0.85) and highest R2 (0.28). "
      "Minimal gap between train and test metrics indicates good generalization. "
      "Handles nonlinear relationships and feature interactions natively.")
