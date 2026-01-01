# Probability calibration

import numpy as np

class ProbabilityCalibrator:
    """
    Scales and calibrates raw model outputs into stable probabilities.
    Currently uses a pass-through (identity) strategy for XGBoost logistic outputs,
    but enables future upgrades (e.g., Isotonic Regression) without breaking the API.
    """
    
    def __init__(self):
        pass

    def calibrate(self, raw_score: float) -> float:
        """
        Ensures the score is strictly between 0.0 and 1.0.
        """
        # Clip values to avoid floating point errors (e.g., -0.000001)
        return float(np.clip(raw_score, 0.0, 1.0))

    def probability_to_score(self, prob: float) -> float:
        """
        Converts probability (0.0 - 1.0) to Risk Score (0 - 100).
        """
        return round(prob * 100.0, 2)