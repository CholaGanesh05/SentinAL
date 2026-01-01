import logging
import sys
import os
import json
from typing import List

# Ensure root path is accessible
sys.path.append(os.getcwd())

# Import The Brain & Limbs
from src.aggregation.fusion_engine import RiskFusionEngine
from src.credit_risk.engine import CreditRiskEngine
from src.sentiment_risk.engine import SentimentRiskEngine
from src.systemic_risk.engine import SystemicRiskEngine
from src.ingestion.loaders import SentinelDataLoader
from src.schemas.risk_objects import RiskSignal 

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SentinAL_Core")

class SentinAL:
    def __init__(self):
        logger.info("🤖 Initializing SentinAL Core Systems...")
        self.loader = SentinelDataLoader(data_dir="data/processed")
        
        # Initialize Engines
        self.credit_engine = CreditRiskEngine(model_path="models/credit_model.json")
        self.sentiment_engine = SentimentRiskEngine()
        self.systemic_engine = SystemicRiskEngine()
        
        # Initialize Brain
        self.brain = RiskFusionEngine(use_ml_model=True, model_path="models/meta_fusion_model.pkl")
        
        # Ingest Graph
        logger.info("🕸️ Ingesting Systemic Context...")
        self.systemic_engine.ingest_data("data/processed/network_mapped.csv")
        logger.info("✅ SentinAL System Ready.")

    def _validate_signal(self, signal, engine_name):
        """
        SAFETY CHECK: Ensures the engine returned a RiskSignal, not a Profile.
        """
        if not isinstance(signal, RiskSignal):
            logger.error(f"❌ TYPE ERROR: {engine_name} returned '{type(signal).__name__}' instead of 'RiskSignal'.")
            return False
        return True

    def analyze_batch(self, entities: List[str], df_credit, news_loader) -> List[dict]:
        results = []
        
        for raw_id in entities:
            try:
                # --- TYPE FIX: Force ID to be a String ---
                entity_id = str(raw_id) 

                # 1. Get Financials
                # Note: We must check against the index using the correct type. 
                # If dataframe index is int, use raw_id. If str, use entity_id.
                if raw_id in df_credit.index:
                    row = df_credit.loc[raw_id]
                    feats = {k: float(v) for k,v in row.items() if k in ['roa', 'debt_ratio', 'operating_margin', 'net_income_assets']}
                elif entity_id in df_credit.index:
                    row = df_credit.loc[entity_id]
                    feats = {k: float(v) for k,v in row.items() if k in ['roa', 'debt_ratio', 'operating_margin', 'net_income_assets']}
                else:
                    logger.warning(f"⚠️ Entity {entity_id} not found in credit data. Skipping.")
                    continue

                # 2. Run Analysis & Check Types
                # --- Credit ---
                sig_credit = self.credit_engine.analyze(entity_id, feats)
                if not self._validate_signal(sig_credit, "CreditEngine"): continue

                # --- Sentiment ---
                sig_sentiment = self.sentiment_engine.analyze(entity_id) 
                if not self._validate_signal(sig_sentiment, "SentimentEngine"): continue
                
                # --- Systemic ---
                sig_systemic = self.systemic_engine.analyze(entity_id)
                if not self._validate_signal(sig_systemic, "SystemicEngine"): continue

                # 3. Fuse
                profile = self.brain.aggregate(entity_id, [sig_credit, sig_sentiment, sig_systemic])
                results.append(profile.to_json())
                
            except Exception as e:
                logger.error(f"❌ Analysis failed for {entity_id}: {e}", exc_info=True)

        return results

if __name__ == "__main__":
    app = SentinAL()
    logger.info("System initialized. Run 'run_full_analysis.py' now.")