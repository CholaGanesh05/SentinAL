import logging
import xgboost as xgb
import os
import sys
from datetime import datetime
from typing import Dict

# Ensure root is in path
sys.path.append(os.getcwd())

from src.schemas.risk_objects import RiskSignal, RiskType
from src.credit_risk.features import CreditFeatureEngineer
from src.credit_risk.calibration import ProbabilityCalibrator
from src.credit_risk.scoring import RiskScorer

class CreditRiskEngine:
    """
    The Main Entry Point for Credit Risk.
    Orchestrates the flow from raw data to a standardized RiskSignal.
    """
    
    def __init__(self, model_path="models/credit_model.json"):
        self.logger = logging.getLogger("CreditEngine")
        self.model_path = model_path
        self.model = self._load_model()
        
        # Initialize helpers
        self.calibrator = ProbabilityCalibrator()
        self.scorer = RiskScorer()

    def _load_model(self):
        """Safely loads the XGBoost model."""
        if not os.path.exists(self.model_path):
            self.logger.warning(f"⚠️ Model not found at {self.model_path}. Engine will fail on analyze().")
            return None
        
        try:
            bst = xgb.Booster()
            bst.load_model(self.model_path)
            self.logger.info("✅ Credit Model loaded successfully.")
            return bst
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            return None

    def analyze(self, entity_id: str, input_features: Dict[str, float]) -> RiskSignal:
        """
        Analyzes a single entity and returns a standardized RiskSignal.
        """
        if not self.model:
            raise RuntimeError("Credit Model is not loaded. Please train the model first.")

        # 1. Validate & Prepare Features (Gatekeeper)
        # This converts the dict to a DMatrix with the exact correct column order
        df_features = CreditFeatureEngineer.prepare_for_inference(input_features)
        dmatrix = xgb.DMatrix(df_features)

        # 2. Raw Prediction (Probability of Default)
        raw_prob = self.model.predict(dmatrix)[0]
        
        # 3. Calibration & Scoring
        # Ensure probability is clean (0.0 - 1.0)
        calibrated_prob = self.calibrator.calibrate(raw_prob)
        # Convert to 0-100 Score
        risk_score = self.calibrator.probability_to_score(calibrated_prob)
        # Determine Level (Low/High/Critical)
        risk_level = self.scorer.get_risk_level(risk_score)

        # 4. Construct Metadata (for Explainability)
        metadata = {
            "risk_level_label": risk_level.value,
            "raw_pd_probability": float(calibrated_prob),
            "input_used": input_features
        }

        # 5. Build Final Signal
        signal = RiskSignal(
            entity_id=entity_id,
            risk_type=RiskType.CREDIT,
            raw_score=float(calibrated_prob),
            normalized_score=risk_score,
            confidence=0.95,  # Statistical confidence is high for XGBoost
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.logger.info(f"🔍 Analyzed {entity_id}: Score={risk_score} ({risk_level.value})")
        return signal