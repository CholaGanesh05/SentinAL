import pandas as pd
import os

RAW_DIR = 'data/raw'
files = ['financial_metrics.csv', 'transaction_network.csv', 'market_news.csv']

print("--- DATA INSPECTION REPORT ---\n")

for f in files:
    path = os.path.join(RAW_DIR, f)
    print(f"FILE: {f}")
    if os.path.exists(path):
        try:
            # Read only first 3 rows to save memory
            df = pd.read_csv(path, nrows=3)
            print("COLUMNS:", list(df.columns))
            print("SAMPLE ROW:", df.iloc[0].to_dict())
        except Exception as e:
            print(f"ERROR READING FILE: {e}")
    else:
        print("STATUS: ❌ File not found. Check filename.")
    print("-" * 50 + "\n")