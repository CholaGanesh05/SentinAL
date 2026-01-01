import pandas as pd
import os

def check():
    print("--- 🔍 Checking CSV Headers ---")
    
    # 1. Credit Data
    try:
        df = pd.read_csv("data/processed/credit_clean.csv")
        print(f"\n[Credit Data] Columns found: {list(df.columns)}")
        required = ['entity_id', 'liquidity_ratio', 'leverage_ratio', 'operating_margin', 'roa']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"❌ WARNING: Missing columns: {missing}")
        else:
            print("✅ Credit Headers OK")
    except:
        print("❌ Could not read credit_clean.csv")

    # 2. News Data
    try:
        df = pd.read_csv("data/processed/news_clean.csv")
        print(f"\n[News Data] Columns found: {list(df.columns)}")
        if 'entity_id' in df.columns and 'headline' in df.columns:
            print("✅ News Headers OK")
        else:
            print("❌ WARNING: Missing 'entity_id' or 'headline'")
    except:
        print("❌ Could not read news_clean.csv")

if __name__ == "__main__":
    check()