import pandas as pd
import json
import os
import re

# Settings
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_credit_data():
    """
    Cleans the Bankruptcy Dataset.
    Fixes: Strips whitespace from column names.
    """
    print("[1/3] Processing Credit Data...")
    try:
        df = pd.read_csv(f"{RAW_DIR}/financial_metrics.csv")
        
        # FIX: Clean leading/trailing spaces in column names
        df.columns = [c.strip() for c in df.columns]
        
        # Select and Rename key columns
        cols_to_keep = {
            'Bankrupt?': 'target', 
            'ROA(C) before interest and depreciation before interest': 'roa',
            'Debt ratio %': 'debt_ratio',
            'Net Income to Total Assets': 'net_income_assets',
            'Operating Gross Margin': 'operating_margin',
            'Current Liability to Assets': 'liability_asset_ratio'
        }
        
        # Filter and rename
        df = df[list(cols_to_keep.keys())]
        df = df.rename(columns=cols_to_keep)
        
        output_path = f"{PROCESSED_DIR}/credit_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"   -> Saved {len(df)} records to {output_path}")
        
    except Exception as e:
        print(f"   [!] Error processing credit data: {e}")

def process_graph_data():
    """
    Cleans the PaySim Dataset.
    Standardizes Source/Target columns.
    """
    print("[2/3] Processing Transaction Network...")
    try:
        # Load first 200k rows (loading 6M rows might crash a laptop)
        df = pd.read_csv(f"{RAW_DIR}/transaction_network.csv", nrows=200000)
        
        df = df.rename(columns={
            'nameOrig': 'source_id',
            'nameDest': 'target_id',
            'amount': 'amount',
            'isFraud': 'is_fraud',
            'type': 'txn_type'
        })
        
        keep = ['source_id', 'target_id', 'amount', 'txn_type', 'is_fraud']
        df = df[keep]
        
        output_path = f"{PROCESSED_DIR}/network_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"   -> Saved {len(df)} transactions to {output_path}")
        
    except Exception as e:
        print(f"   [!] Error processing graph data: {e}")

def extract_sentiment(decision_str):
    """
    Parses '{"Company": "neutral"}' -> 'neutral'
    """
    try:
        # It looks like JSON, let's try to parse it
        data = json.loads(decision_str)
        # Return the first value found (e.g., "neutral")
        return list(data.values())[0]
    except:
        return "neutral"

def process_sentiment_data():
    """
    Cleans the News Dataset.
    Fixes: Extracts sentiment label from JSON string.
    """
    print("[3/3] Processing Market News...")
    try:
        df = pd.read_csv(f"{RAW_DIR}/market_news.csv", encoding='latin-1')
        
        # Rename columns based on your inspection
        df = df.rename(columns={'Title': 'text', 'Decisions': 'raw_label'})
        
        # Clean the label
        df['sentiment'] = df['raw_label'].apply(extract_sentiment)
        
        # Keep only clean columns
        df = df[['text', 'sentiment']]
        
        output_path = f"{PROCESSED_DIR}/news_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"   -> Saved {len(df)} headlines to {output_path}")
            
    except Exception as e:
        print(f"   [!] Error processing news data: {e}")

if __name__ == "__main__":
    process_credit_data()
    process_graph_data()
    process_sentiment_data()
    print("\n[✓] Data Processing Complete.")