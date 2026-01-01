import sys
import os

# --- FORCE IMMEDIATE OUTPUT ---
# This runs before anything else so you know the script is alive.
print("🚀 Script Launcher: Initializing environment...", flush=True)

import logging
import yaml
import pandas as pd

# Adjust path to root
sys.path.append(os.getcwd())

try:
    from src.ingestion.loaders import SentinelDataLoader
    from src.ingestion.normalizers import EntityLinker, FeatureScaler
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# Setup Logging (Force output to Console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # <--- This ensures you see the logs!
)
logger = logging.getLogger("UnifyData")

def load_config():
    try:
        with open("configs/model_config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config not found, proceeding with defaults.")
        return {}

def run_pipeline():
    logger.info("🚀 Starting Data Ingestion Pipeline...")
    
    config = load_config()
    raw_dir = "data/processed" # Using your existing path structure
    output_path = "data/processed/unified_risk_data.csv"
    
    # 1. Initialize Components
    logger.info("🔧 Initializing Components...")
    loader = SentinelDataLoader(data_dir=raw_dir)
    linker = EntityLinker()
    scaler = FeatureScaler()
    
    # 2. Load Raw Data
    try:
        logger.info("📂 Loading datasets...")
        df_credit = loader.load_credit_data()
        df_network = loader.load_network_data()
        df_news = loader.load_news_data()
    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        return

    # 3. Generate Master IDs (The Single Source of Truth)
    logger.info(f"🆔 Generating IDs for {len(df_credit)} entities...")
    df_credit = linker.generate_ids(df_credit)
    master_ids = df_credit['entity_id'].tolist()
    
    # 4. Link & Harmonize Data Streams
    if not df_network.empty:
        logger.info("🕸️ Linking Network Data (Vectorized)...")
        df_network = linker.map_network_to_ids(df_credit, df_network)
        # Save the harmonized network for the Systemic Engine to use later
        df_network.to_csv("data/processed/network_mapped.csv", index=False)
        logger.info(f"💾 Saved mapped network data (rows: {len(df_network)})")

    if not df_news.empty:
        logger.info("📰 Linking News Data...")
        limit = min(len(df_news), len(master_ids))
        df_news = df_news.iloc[:limit].copy()
        df_news['entity_id'] = master_ids[:limit]
        
        df_news.to_csv("data/processed/news_mapped.csv", index=False)
        logger.info(f"💾 Saved mapped news data (rows: {len(df_news)})")

    # 5. Clean & Save Master Risk Table
    logger.info("✨ Finalizing Unified Risk Table...")
    df_credit = scaler.clean_financials(df_credit)
    
    # Save the Unified Master Table
    df_credit.to_csv(output_path, index=False)
    logger.info(f"✅ Pipeline Complete. Unified Master Data at: {output_path}")

if __name__ == "__main__":
    run_pipeline()