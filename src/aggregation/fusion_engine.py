import logging
import yaml
import numpy as np
import os
import pickle
from typing import List, Dict
from datetime import datetime

# Import Schema Contracts
from src.schemas.risk_objects import RiskSignal, AggregatedRiskProfile, RiskLevel, RiskType

class RiskFusionEngine:
    """
    The Brain of SentinAL.
    Aggregates disparate risk signals into a single unified risk profile.
    Supports both Static Weighted Averaging and ML-based Aggregation.
    """
    
    def __init__(self, use_ml_model: bool = False, model_path: str = "models/meta_fusion_model.pkl"):
        self.logger = logging.getLogger("FusionEngine")
        self.config = self._load_config()
        self.use_ml_model = use_ml_model
        
        # Load Static Weights from Config (Fallback)
        # Default to safe values if config fails
        self.weights = self.config.get('fusion', {}).get('weights', {
            'credit': 0.50,
            'systemic': 0.30,
            'sentiment': 0.20
        })

        # Load ML Model if requested
        self.model = None
        if self.use_ml_model:
            self.model = self._load_meta_model(model_path)

    def _load_config(self):
        try:
            with open("configs/model_config.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception:
            self.logger.warning("⚠️ Config not found. Using internal defaults.")
            return {}

    def _load_meta_model(self, path: str):
        if not os.path.exists(path):
            self.logger.warning(f"⚠️ Meta-Model not found at {path}. Reverting to Static Weights.")
            return None
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            self.logger.info(f"✅ Loaded Meta-Model from {path}")
            return model
        except Exception as e:
            self.logger.error(f"❌ Error loading Meta-Model: {e}")
            return None

    def get_risk_level(self, score: float) -> RiskLevel:
        """Maps a 0-100 score to a RiskLevel Enum based on Config Thresholds."""
        thresholds = self.config.get('global', {}).get('risk_thresholds', {
            'low': 25, 'medium': 50, 'high': 75, 'critical': 90
        })
        
        if score < thresholds['low']: return RiskLevel.LOW
        elif score < thresholds['medium']: return RiskLevel.MEDIUM
        elif score < thresholds['high']: return RiskLevel.HIGH
        else: return RiskLevel.CRITICAL

    def aggregate(self, entity_id: str, signals: List[RiskSignal]) -> AggregatedRiskProfile:
        """
        Fuses a list of RiskSignals into one profile.
        """
        if not signals:
            raise ValueError(f"No signals provided for entity {entity_id}")

        # 1. Validation
        for signal in signals:
            if signal.entity_id != entity_id:
                self.logger.error(f"Signal ID Mismatch: {signal.entity_id} vs {entity_id}")
                continue

        final_score = 0.0
        
        # --- STRATEGY A: ML Model (Random Forest) ---
        if self.model:
            # Extract features in correct order: [Credit, Systemic, Sentiment]
            # We assume the order based on training. Ideally, we map by name.
            feats = [0.0, 0.0, 0.0] 
            for s in signals:
                if s.risk_type == RiskType.CREDIT: feats[0] = s.normalized_score
                elif s.risk_type == RiskType.SYSTEMIC: feats[1] = s.normalized_score
                elif s.risk_type == RiskType.SENTIMENT: feats[2] = s.normalized_score
            
            # Predict Risk Class (0 or 1) or Probability
            try:
                # We use probability of Class 1 (High Risk) * 100
                probs = self.model.predict_proba([feats])[0]
                final_score = probs[1] * 100.0
            except Exception as e:
                self.logger.error(f"ML Prediction failed: {e}. Falling back to weights.")
                self.model = None # Disable for this run to avoid loops

        # --- STRATEGY B: Static Weighted Average (Fallback) ---
        if not self.model:
            total_weight = 0.0
            weighted_sum = 0.0
            
            signal_map = {s.risk_type.value: s for s in signals}

            for r_type, weight in self.weights.items():
                if r_type in signal_map:
                    weighted_sum += signal_map[r_type].normalized_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_score = weighted_sum / total_weight
            else:
                final_score = 0.0

        # 3. Construct Profile
        risk_level = self.get_risk_level(final_score)
        
        return AggregatedRiskProfile(
            entity_id=entity_id,
            composite_risk_score=round(final_score, 2),
            risk_level=risk_level,
            contributing_signals=signals,
            timestamp=datetime.now()
        )