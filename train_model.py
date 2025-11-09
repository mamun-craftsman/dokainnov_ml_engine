import pandas as pd
import lightgbm as lgb
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
from datetime import datetime
from lightgbm import early_stopping, log_evaluation

# Config paths
DATA_CSV = "data/train_sales.csv"
MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = MODEL_DIR / "train_log.txt"

# Load data
df = pd.read_csv(DATA_CSV, parse_dates=['sale_date'])

# Fill missing engineered features if needed
optional_features = ['is_festival', 'days_to_next_festival', 'day_of_week', 'is_weekend', 'month', 'day_of_month', 'year']
for f in optional_features:
    if f not in df.columns:
        df[f] = 0

# Feature and target columns
features = [
    'category', 'cost_price', 'unit_price', 'day_of_week', 'is_weekend',
    'month', 'day_of_month', 'year', 'is_festival', 'days_to_next_festival'
]
target = 'quantity'

product_ids = df['product_id'].unique().tolist()
print(f"Found {len(product_ids)} unique products.")

global_rmse = []
log_lines = [f"Training run started: {datetime.now()}\n"]

# Build global dummy categorical columns schema to keep encoding consistent
print("Building global feature categorical schema...")
all_dummies = pd.get_dummies(df[['category']].astype(str), prefix='category')
global_columns = all_dummies.columns.tolist()
print(f"Global category columns: {global_columns}")

for pid in product_ids:
    subdf = df[df['product_id'] == pid]
    if len(subdf) < 40:
        print(f"Skipping product {pid} due to insufficient data: {len(subdf)} rows")
        continue

    # Prepare feature matrix & target
    X = subdf[features].copy()
    y = subdf[target].values

    # One-hot encode and align columns globally
    X = pd.get_dummies(X, columns=['category'])
    for col in global_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[sorted(X.columns)].fillna(0)

    # Split data robustly and reproducibly
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbosity': -1,
        'seed': 42,
    }

    # Check and load previous model for incremental training if possible
    model_path = MODEL_DIR / f"lgbm_product_{pid}.pkl"
    init_model = None
    if model_path.exists():
        try:
            old_model, old_columns = joblib.load(model_path)
            if list(X.columns) == old_columns:
                init_model = old_model
                print(f"Using incremental training for product {pid}")
            else:
                print(f"Feature mismatch for product {pid}, retraining from scratch")
        except Exception:
            print(f"Failed to load previous model for product {pid}, retraining from scratch")

    # Train with early stopping and logging callbacks
    callbacks = [early_stopping(stopping_rounds=15), log_evaluation(period=0)]
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=200,
        callbacks=callbacks,
        init_model=init_model
    )

    # Evaluate and log RMSE
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    global_rmse.append(rmse)
    print(f"✅ Product {pid}: RMSE={rmse:.4f}")

    # Save model with feature columns for alignment in future
    joblib.dump((model, X.columns.tolist()), model_path)

    # Write metadata JSON for tracking
    meta = {
        "product_id": pid,
        "samples": len(subdf),
        "rmse": rmse,
        "best_iteration": model.best_iteration,
        "trained_at": str(datetime.now()),
        "features": X.columns.tolist(),
    }
    with open(MODEL_DIR / f"meta_{pid}.json", "w") as f:
        json.dump(meta, f, indent=2)

    log_lines.append(f"Product {pid}: RMSE={rmse:.4f}, samples={len(subdf)}\n")

# Summarize global RMSE stats
if global_rmse:
    avg_rmse = np.mean(global_rmse)
    print(f"\nAverage RMSE over {len(global_rmse)} products: {avg_rmse:.4f}")
    log_lines.append(f"\nAverage RMSE: {avg_rmse:.4f}\n")

# Write training log
with open(LOG_FILE, "a") as f:
    f.writelines(log_lines)

print("✅ Training completed successfully for all products.")
