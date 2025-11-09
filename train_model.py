import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

DATA_CSV = "data/train_sales.csv"
MODEL_DIR = "model/"
Path(MODEL_DIR).mkdir(exist_ok=True)

df = pd.read_csv(DATA_CSV, parse_dates=['sale_date'])

features = [
    'category', 'cost_price', 'unit_price', 'day_of_week', 'is_weekend',
    'month', 'day_of_month', 'year', 'is_festival', 'days_to_next_festival'
]
target = 'quantity'

product_ids = df['product_id'].unique()

for pid in product_ids:
    subdf = df[df['product_id'] == pid]
    if len(subdf) < 40:
        continue  # Skip insufficient data
    
    X = subdf[features]
    X = pd.get_dummies(X, columns=['category'])
    y = subdf[target]
    
    # Align columns - for safety
    X = X.fillna(0)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 15,
        'learning_rate': 0.07,
        'feature_fraction': 0.7,
        'verbosity': -1
    }
    
    model = lgb.train(params, train_data, valid_sets=[train_data, val_data], num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)
    joblib.dump((model, X.columns), f"{MODEL_DIR}lgbm_product_{pid}.pkl")
    print(f"Trained and saved model for product {pid}")

print("All models trained!")
