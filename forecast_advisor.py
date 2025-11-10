import os
import json
import re
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from openai import OpenAI

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
FORECAST_CSV = BASE_DIR / "data" / "forecast_output.csv"
DB_PATH = BASE_DIR.parent / "dokainnov_prototype" / "database" / "dokainnov.db"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"

FESTIVALS = {
    "Eid-ul-Fitr 2025": datetime(2025, 3, 30).date(),
    "Eid-ul-Adha 2025": datetime(2025, 6, 7).date(),
    "Durga Puja 2025": datetime(2025, 10, 2).date(),
    "Pohela Boishakh 2025": datetime(2025, 4, 14).date(),
}

def load_external_context():
    context_file = BASE_DIR / "data" / "external_context.txt"
    try:
        if context_file.exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except:
        pass
    return ""

def get_upcoming_festivals(days=30):
    today = datetime.now().date()
    upcoming = []
    for name, date in FESTIVALS.items():
        days_away = (date - today).days
        if 0 < days_away <= days:
            upcoming.append({"name": name, "days_away": days_away, "date": date.strftime("%d %B %Y")})
    return sorted(upcoming, key=lambda x: x['days_away'])

def get_product_context(product_id: int, forecast_df):
    import sqlite3
    
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    
    conn = sqlite3.connect(str(DB_PATH))
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        product_row = cursor.fetchone()
        
        if not product_row:
            raise ValueError(f"Product {product_id} not found")
        
        columns = [desc[0] for desc in cursor.description]
        product = dict(zip(columns, product_row))
        
        cursor.execute("""
            SELECT date(sale_date) as sale_date, SUM(quantity) as qty
            FROM sale_items
            WHERE product_id = ? 
            AND date(sale_date) >= date('now', '-60 days')
            GROUP BY date(sale_date)
            ORDER BY date(sale_date) DESC
        """, (product_id,))
        
        history_rows = cursor.fetchall()
        
    finally:
        conn.close()
    
    total_sales_60d = sum(row[1] for row in history_rows) if history_rows else 0
    
    if history_rows:
        recent_30d = sum(row[1] for row in history_rows[:30])
        older_30d = sum(row[1] for row in history_rows[30:60]) if len(history_rows) > 30 else max(1, recent_30d)
        last_sale_date = datetime.strptime(history_rows[0][0], '%Y-%m-%d').date()
        days_since_sale = (datetime.now().date() - last_sale_date).days
    else:
        recent_30d = 0
        older_30d = 1
        days_since_sale = 999
    
    trend = ((recent_30d - older_30d) / older_30d * 100) if older_30d > 0 else 0
    
    forecast_data = forecast_df[forecast_df['product_id'] == product_id]
    weekly_qty = float(forecast_data['forecast_qty'].sum()) if not forecast_data.empty else 0.0
    
    return {
        'product_id': int(product_id),
        'name': str(product['name']),
        'category': str(product['category']),
        'stock': int(product['current_stock']),
        'cost': float(product['cost_price']),
        'price': float(product['selling_price']),
        'profit_per_unit': float(product['selling_price'] - product['cost_price']),
        'total_sales_60d': int(total_sales_60d),
        'trend_pct': round(float(trend), 1),
        'days_since_sale': int(days_since_sale),
        'forecast_qty': weekly_qty,
    }

def extract_json_from_text(text: str):
    try:
        return json.loads(text.strip())
    except:
        pass
    
    match = re.search(r'``````', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    match = re.search(r'(\{[\s\S]*\})', text, re.DOTALL)
    if match:
        try:
            candidate = match.group(1)
            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
            return json.loads(candidate)
        except:
            pass
    
    raise ValueError("No valid JSON found")

def call_openrouter_advisor(products_context):
    today = datetime.now().strftime("%d %B %Y")
    festivals = get_upcoming_festivals(30)
    
    festival_text = "\n".join([f"- {f['name']}: {f['days_away']} days away" for f in festivals]) if festivals else "- No major festivals in next 30 days"
    
    product_summaries = []
    for ctx in products_context:
        stock_gap = ctx['forecast_qty'] - ctx['stock']
        expected_profit = ctx['forecast_qty'] * ctx['profit_per_unit']
        
        product_summaries.append(f"""Product {ctx['product_id']}: {ctx['name']}
Category: {ctx['category']}
Forecast: {ctx['forecast_qty']:.1f} units (7 days)
Current Stock: {ctx['stock']} units
Need to reorder: {max(0, stock_gap):.1f} units
Cost: Tk{ctx['cost']:.0f} | Sell: Tk{ctx['price']:.0f} | Profit/unit: Tk{ctx['profit_per_unit']:.0f}
Expected profit: Tk{expected_profit:.0f}
Sales trend (60d): {ctx['trend_pct']:+.1f}%
Days since last sale: {ctx['days_since_sale']}""")
    
    external_context = load_external_context()
    
    system_prompt = """You are a practical business advisor for small retail shops in Bangladesh.
Give advice in simple Bangla that even uneducated shopkeepers can understand.
You can override the ML forecast if business logic (festivals, trends, market conditions) demands it.
Use English for numbers (Tk, units, %)."""
    
    user_prompt = f"""Today's date: {today}

UPCOMING FESTIVALS:
{festival_text}

BUSINESS OWNER'S NOTES:
{external_context if external_context else "No additional context provided"}

PRODUCTS TO ANALYZE ({len(products_context)}):

{chr(10).join(product_summaries)}

YOUR TASK:
Give 2-4 sentence practical advice for each product in simple Bangla.

Consider:
- If festival coming, suggest increased stock
- If trend negative, explain why and give solution
- If stock sitting idle, warn about capital freeze
- If profit margin low, suggest skip/reduce
- If no recent sales, suggest discontinue

YOU MUST RESPOND WITH ONLY THIS JSON FORMAT (no other text):
{{
  "products": [
    {{"product_id": 191, "advice": "Your advice in Bangla here"}},
    {{"product_id": 192, "advice": "Your advice in Bangla here"}}
  ],
  "summary": "Overall summary of analysis"
}}"""
    
    try:
        print(f"[API] Calling DeepSeek for {len(products_context)} products...")
        
        completion = client.chat.completions.create(
            extra_headers={"HTTP-Referer": "https://github.com/dokainnov", "X-Title": "Dokainnov"},
            extra_body={},
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        
        response_text = completion.choices[0].message.content
        
        print(f"\n{'='*60}")
        print("API RESPONSE:")
        print(f"{'='*60}")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print(f"{'='*60}\n")
        
        result = extract_json_from_text(response_text)
        print(f"[OK] Parsed {len(result.get('products', []))} recommendations")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_products(product_ids):
    print("\n" + "="*70)
    print("AI BUSINESS ADVISOR")
    print("="*70 + "\n")
    
    if not FORECAST_CSV.exists():
        raise FileNotFoundError(f"Forecast CSV missing: {FORECAST_CSV}")
    
    forecast_df = pd.read_csv(str(FORECAST_CSV))
    forecast_df['product_id'] = forecast_df['product_id'].astype(int)
    print(f"[INFO] Loaded {len(forecast_df)} forecast rows\n")
    
    contexts = []
    for pid in product_ids:
        try:
            ctx = get_product_context(pid, forecast_df)
            contexts.append(ctx)
            print(f"[OK] Product {pid}: {ctx['name']} | Stock: {ctx['stock']} | Forecast: {ctx['forecast_qty']:.1f}")
        except Exception as e:
            print(f"[FAIL] Product {pid}: {e}")
    
    if not contexts:
        raise ValueError("No valid product contexts")
    
    batch_size = 25
    all_results = {"products": [], "summary": ""}
    
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(contexts) + batch_size - 1) // batch_size
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_num}/{total_batches} - {len(batch)} products")
        print(f"{'='*70}\n")
        
        result = call_openrouter_advisor(batch)
        
        if result:
            all_results['products'].extend(result.get('products', []))
            all_results['summary'] += result.get('summary', '') + "\n"
            print(f"\n[OK] Batch {batch_num} complete\n")
        else:
            print(f"\n[WARN] Batch {batch_num} failed\n")
    
    output_file = BASE_DIR / "data" / "ai_recommendations.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("AI ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Products: {len(all_results['products'])}")
    print(f"JSON: {output_file}")
    print(f"{'='*70}\n")
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--products', type=str, required=True)
    args = parser.parse_args()
    
    try:
        pids = [int(p.strip()) for p in args.products.split(',')]
        analyze_products(pids)
        sys.exit(0)
    except Exception as e:
        print(f"\nFAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
