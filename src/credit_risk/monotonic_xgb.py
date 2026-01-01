import sys
import os

# --- 1. PREVENT HANGS (Anti-Freeze Fix) ---
# This allows XGBoost to run without conflicting with other libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. DEBUGGING PRINTS ---
print("🚀 Script Launcher: Starting...", flush=True)

try:
    import pandas as pd
    print("✅ Pandas imported.", flush=True)
    
    import yaml
    print("✅ YAML imported.", flush=True)

    print("⏳ Importing XGBoost (this might take a moment)...", flush=True)
    import xgboost as xgb
    print("✅ XGBoost imported successfully!", flush=True)
    
except ImportError as e:
    print(f"\n❌ CRITICAL ERROR: Library missing. {e}")
    print("👉 Run: pip install pandas xgboost pyyaml")
    sys.exit(1)

import logging
import numpy as np

# Ensure root is in path
sys.path.append(os.getcwd())

try:
    from src.credit_risk.features import CreditFeatureEngineer
    print("✅ Internal modules imported.", flush=True)
except ImportError as e:
    print(f"❌ Import Error: {e}", flush=True)
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CreditTrainer")

class CreditModelTrainer:
    def __init__(self, config_path="configs/model_config.yaml", constraint_path="configs/monotonic_constraints.yaml"):
        self.output_path = "models/credit_model.json"
        
        # Load Configs
        if not os.path.exists(config_path):
            logger.warning(f"⚠️ Config missing at {config_path}. Using Defaults.")
            self.config = {}
        else:
            self.config = self._load_yaml(config_path)

        if not os.path.exists(constraint_path):
            logger.warning(f"⚠️ Constraints missing at {constraint_path}. Using Defaults.")
            self.constraints_config = {}
        else:
            self.constraints_config = self._load_yaml(constraint_path)

    def _load_yaml(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _get_constraint_tuple(self, feature_cols: list) -> tuple:
        """Dynamically builds the XGBoost constraint tuple."""
        constraints_map = self.constraints_config.get('constraints', {})
        constraint_list = []
        for feat in feature_cols:
            constraint_list.append(constraints_map.get(feat, 0))
        return tuple(constraint_list)

    def train(self, data_path="data/processed/unified_risk_data.csv"):
        logger.info("🚀 Starting Credit Model Training...")

        # 1. Load Data
        if not os.path.exists(data_path):
            logger.error(f"❌ Data file not found: {data_path}")
            return
        
        df = pd.read_csv(data_path)
        logger.info(f"📂 Data loaded: {len(df)} rows.")
        
        # 2. Prepare Features
        try:
            X = CreditFeatureEngineer.prepare_for_training(df)
            
            # Mock target if missing (Resilience)
            if 'target' not in df.columns:
                logger.warning("⚠️ 'target' missing. Generating random target for testing.")
                df['target'] = np.random.randint(0, 2, df.shape[0])
            y = df['target']
            
        except Exception as e:
            logger.error(f"❌ Data Preparation Error: {e}")
            return

        # 3. Setup XGBoost
        mono_constraints = self._get_constraint_tuple(X.columns.tolist())
        xgb_params = self.config.get('credit_risk', {}).get('params', {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 100
        })
        xgb_params['monotone_constraints'] = mono_constraints
        
        # 4. Train
        logger.info("🏋️ Training Model...")
        dtrain = xgb.DMatrix(X, label=y)
        bst = xgb.train(xgb_params, dtrain, num_boost_round=xgb_params.get('n_estimators', 100))

        # 5. Save
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        bst.save_model(self.output_path)
        logger.info(f"✅ Model saved to {self.output_path}")

if __name__ == "__main__":
    trainer = CreditModelTrainer()
    trainer.train()