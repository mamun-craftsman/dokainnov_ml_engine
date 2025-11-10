import sys
from pathlib import Path
import sqlite3

print("="*70)
print("DIAGNOSTIC CHECK")
print("="*70)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "dokainnov_prototype" / "database" / "dokainnov.db"
FORECAST_CSV = BASE_DIR / "data" / "forecast_output.csv"

print(f"\n1. PATHS:")
print(f"   BASE_DIR: {BASE_DIR}")
print(f"   DB_PATH: {DB_PATH}")
print(f"   DB exists: {DB_PATH.exists()}")
print(f"   CSV: {FORECAST_CSV}")
print(f"   CSV exists: {FORECAST_CSV.exists()}")

if not DB_PATH.exists():
    print("\n[CRITICAL] Database file not found!")
    sys.exit(1)

print(f"\n2. DATABASE CHECK:")
try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"   Tables: {', '.join(tables)}")
    
    if 'product_forecasts' in tables:
        cursor.execute("PRAGMA table_info(product_forecasts)")
        cols = cursor.fetchall()
        print(f"\n   product_forecasts columns:")
        for col in cols:
            print(f"     - {col[1]} ({col[2]})")
        
        cursor.execute("SELECT COUNT(*) FROM product_forecasts")
        count = cursor.fetchone()[0]
        print(f"\n   Current records: {count}")
    else:
        print("\n   [ERROR] product_forecasts table missing!")
    
    cursor.execute("SELECT COUNT(*) FROM products")
    prod_count = cursor.fetchone()[0]
    print(f"   Products in DB: {prod_count}")
    
    cursor.execute("SELECT product_id, product_name FROM products LIMIT 3")
    print(f"\n   Sample products:")
    for row in cursor.fetchall():
        print(f"     - {row[0]}: {row[1]}")
    
    conn.close()
    print("\n   [OK] Database connection successful")
    
except Exception as e:
    print(f"\n   [ERROR] Database error: {e}")
    sys.exit(1)

if FORECAST_CSV.exists():
    print(f"\n3. CSV CHECK:")
    import pandas as pd
    df = pd.read_csv(FORECAST_CSV)
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Products: {df['product_id'].nunique()}")
    print(f"   Sample product IDs: {df['product_id'].unique()[:5].tolist()}")
else:
    print(f"\n3. CSV CHECK:")
    print(f"   [ERROR] Forecast CSV not found!")

print(f"\n4. TRYING INSERT TEST:")
try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    from datetime import datetime
    test_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO product_forecasts 
        (product_id, forecast_date, forecast_qty, expected_profit, reorder_needed, ai_advice)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (999, test_date, 10.5, 100.0, 5.0, "Test advice"))
    
    conn.commit()
    print(f"   [OK] Test insert successful")
    
    cursor.execute("SELECT * FROM product_forecasts WHERE product_id = 999")
    result = cursor.fetchone()
    print(f"   [OK] Test read successful: {result}")
    
    cursor.execute("DELETE FROM product_forecasts WHERE product_id = 999")
    conn.commit()
    print(f"   [OK] Test delete successful")
    
    conn.close()
    
except Exception as e:
    print(f"   [ERROR] Insert test failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n5. API KEY CHECK:")
api_key = "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"
print(f"   Length: {len(api_key)}")
print(f"   Prefix: {api_key[:15]}...")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
