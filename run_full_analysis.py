import json
import logging
import sys
import os
import pandas as pd
from tqdm import tqdm

# --- FORCE CPU (Optional: Uncomment if GPU is unstable or OOM) ---
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 

# Setup Paths
sys.path.append(os.getcwd())

from src.ingestion.loaders import SentinelDataLoader
from src.sentiment_risk.news_loader import NewsLoader
from main import SentinAL

# Configure Logging to File
logging.basicConfig(
    filename='sentinal_full_run.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def run_pipeline():
    print("🚀 STARTING SENTINAL BATCH ANALYSIS...")

    # 1. Initialize The App
    app = SentinAL()
    
    # 2. Load Helper Data Sources (for fast lookups)
    print("📂 Loading Data Sources...")
    # Load Credit Data and index by Entity ID for O(1) access
    loader = SentinelDataLoader(data_dir="data/processed")
    df_credit = loader.load_credit_data()
    if 'entity_id' in df_credit.columns:
        df_credit.set_index('entity_id', inplace=True)
    
    # Load News
    news_loader = NewsLoader(data_path="data/processed/news_mapped.csv")
    
    # 3. Define Workload
    all_entities = list(df_credit.index)
    total = len(all_entities)
    print(f"📊 Found {total} entities to analyze.")

    results = []
    output_file = "outputs/final_risk_analysis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 4. Processing Loop
    # We save every 100 records to prevent data loss on crash
    BATCH_SIZE = 100
    
    print("\n🌊 Diving into Risk Stream...")
    try:
        for i in tqdm(range(0, total, BATCH_SIZE), desc="Batch Processing"):
            batch_ids = all_entities[i : i + BATCH_SIZE]
            
            # Use the app's batch processor
            # We need to adapt the app's method slightly or call it directly here
            # Calling the logic directly here for clarity in the runner:
            
            batch_results = app.analyze_batch(batch_ids, df_credit, news_loader)
            
            # Convert JSON strings to objects if needed
            cleaned_results = []
            for res in batch_results:
                if isinstance(res, str):
                    cleaned_results.append(json.loads(res))
                else:
                    cleaned_results.append(res)
            
            results.extend(cleaned_results)
            
            # Intermediate Save (Checkpointing)
            if i % 500 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

    except KeyboardInterrupt:
        print("\n🛑 Execution Interrupted by User. Saving progress...")
    except Exception as e:
        print(f"\n❌ Critical Failure: {e}")
        logging.error(f"Critical Failure: {e}", exc_info=True)
    finally:
        # Final Save
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ ANALYSIS COMPLETE.")
        print(f"📄 Processed: {len(results)}/{total}")
        print(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    run_pipeline()