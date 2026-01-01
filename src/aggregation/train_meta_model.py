import pandas as pd
import numpy as np
import pickle
import sys
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add root to path
sys.path.append(os.getcwd())

from src.credit_risk.engine import CreditRiskEngine
from src.systemic_risk.engine import SystemicRiskEngine
from src.sentiment_risk.engine import SentimentRiskEngine

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetaTrainer")

def train_meta_learner():
    logger.info("🧠 STARTING META-MODEL TRAINING...")
    
    # 1. Load Unified Data
    data_path = "data/processed/unified_risk_data.csv"
    if not os.path.exists(data_path):
        logger.error(f"❌ Data not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"📂 Loaded {len(df)} records.")

    # 2. Initialize Engines (Fast Mode)
    # We initialize them just to use their logic if needed, 
    # but for training a meta-model, we often just need the SCORES.
    # If your CSV already has 'credit_score', 'graph_score', etc., use those.
    # Assuming we need to GENERATE scores on the fly:
    
    credit_engine = CreditRiskEngine()
    graph_engine = SystemicRiskEngine()
    graph_engine.ingest_data("data/processed/network_mapped.csv") # Important!
    sentiment_engine = SentimentRiskEngine()

    X = []
    y = []

    # 3. Generate Training Vectors
    # We limit to 1000 samples for speed during dev
    sample_df = df.sample(min(len(df), 1000), random_state=42)
    
    logger.info("⚙️ Generating feature vectors (this may take time)...")
    for _, row in sample_df.iterrows():
        eid = str(row['entity_id'])
        target = row.get('target', 0) # 1 = Default/Risk, 0 = Safe
        
        # --- FEATURE 1: CREDIT ---
        # Construct simplified feature dict from row
        feats = {k: float(v) for k,v in row.items() if k in ['roa', 'debt_ratio', 'operating_margin', 'net_income_assets']}
        try:
            c_score = credit_engine.analyze(eid, feats).normalized_score
        except:
            c_score = 50.0 # Impute neutral

        # --- FEATURE 2: SYSTEMIC ---
        try:
            g_score = graph_engine.analyze(eid).normalized_score
        except:
            g_score = 0.0

        # --- FEATURE 3: SENTIMENT ---
        # Assuming we have a headline column, or we fetch it
        # For speed in training, we might mock this or use pre-computed sentiment
        s_score = 0.0 
        # (Real implementation would fetch news, but that's slow for training loop)
        
        # Append [Credit, Systemic, Sentiment]
        X.append([c_score, g_score, s_score])
        y.append(target)

    # 4. Train Random Forest
    logger.info("🌲 Training Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    score = clf.score(X_test, y_test)
    logger.info(f"✅ Model Accuracy: {score:.2f}")
    
    # 6. Save
    os.makedirs("models", exist_ok=True)
    with open("models/meta_fusion_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    logger.info("💾 Meta-Model saved to models/meta_fusion_model.pkl")

if __name__ == "__main__":
    train_meta_learner()