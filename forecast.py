import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path

# ========== Force UTF-8 encoding for Windows ==========
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'

# ========== Paths (absolute) ==========
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
INPUT_CSV = BASE_DIR / "data" / "predict_input.csv"
OUTPUT_CSV = BASE_DIR / "data" / "forecast_output.csv"

# ========== Validation ==========
if not INPUT_CSV.exists():
    print(f"[ERROR] Input file not found: {INPUT_CSV}")
    print(f"[INFO] Run generate_forecast_input.py first!")
    sys.exit(1)

if not MODEL_DIR.exists():
    print(f"[ERROR] Models directory not found: {MODEL_DIR}")
    print(f"[INFO] Run train_model.py first!")
    sys.exit(1)

# ========== Load Input ==========
print(f"[INFO] Loading input: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, parse_dates=['sale_date'])
df['forecast_qty'] = 0.0

product_ids = df['product_id'].unique()
print(f"[INFO] Forecasting {len(product_ids)} products over {len(df)} rows...")

# ========== Forecast Loop ==========
success_count = 0
skip_count = 0
error_count = 0

for pid in product_ids:
    model_file = MODEL_DIR / f"lgbm_product_{pid}.pkl"
    
    if not model_file.exists():
        print(f"[SKIP] Product {pid}: No trained model found")
        skip_count += 1
        continue
    
    try:
        # Load model and feature columns
        model, feature_cols = joblib.load(model_file)
        
        # Get product data
        subdf = df[df['product_id'] == pid].copy()
        
        # Prepare features matching training schema
        X = subdf[[
            'category', 'cost_price', 'unit_price', 'day_of_week', 'is_weekend',
            'month', 'day_of_month', 'year', 'is_festival', 'days_to_next_festival'
        ]].copy()
        
        # One-hot encode category
        X = pd.get_dummies(X, columns=['category'])
        
        # Align columns to match trained model
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Ensure correct column order
        X = X[feature_cols].fillna(0)
        
        # Predict using best iteration
        preds = model.predict(X, num_iteration=model.best_iteration)
        
        # Store predictions
        df.loc[df['product_id'] == pid, 'forecast_qty'] = preds
        
        # Log success
        weekly_total = preds.sum()
        print(f"[OK] Product {pid}: {weekly_total:.1f} units (7 days)")
        success_count += 1
        
    except Exception as e:
        print(f"[ERROR] Product {pid}: {str(e)}")
        error_count += 1

# ========== Save Output ==========
output_dir = OUTPUT_CSV.parent
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_CSV, index=False)

# ========== Summary ==========
print(f"\n{'='*60}")
print(f"[SUMMARY] Forecast Complete")
print(f"{'='*60}")
print(f"  Success: {success_count} products")
print(f"  Skipped: {skip_count} products (no model)")
print(f"  Errors:  {error_count} products")
print(f"  Total:   {len(product_ids)} products")
print(f"\n  Output: {OUTPUT_CSV}")
print(f"  Total Forecasted: {df['forecast_qty'].sum():.1f} units")
print(f"{'='*60}\n")

# ========== Sample Output ==========
print("Sample forecast (first 20 rows):")
print(df[['product_id', 'sale_date', 'forecast_qty']].head(20).to_string(index=False))

print(f"\n[DONE] Forecast saved to: {OUTPUT_CSV}")
