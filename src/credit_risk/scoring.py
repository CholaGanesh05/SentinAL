# Final credit scoring

import yaml
from src.schemas.risk_objects import RiskLevel

class RiskScorer:
    """
    Maps numerical scores (0-100) to Categorical Risk Levels (Low, High, etc.)
    based on the configuration file.
    """
    
    def __init__(self, config_path="configs/model_config.yaml"):
        self.thresholds = self._load_thresholds(config_path)

    def _load_thresholds(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        # Default fallback if config is missing
        return config.get('global', {}).get('risk_thresholds', {
            'low': 25, 
            'medium': 50, 
            'high': 75, 
            'critical': 90
        })

    def get_risk_level(self, score: float) -> RiskLevel:
        """
        Returns the appropriate RiskLevel Enum based on the score.
        """
        if score < self.thresholds['low']:
            return RiskLevel.LOW
        elif score < self.thresholds['medium']:
            return RiskLevel.MEDIUM
        elif score < self.thresholds['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL