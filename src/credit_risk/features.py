# Financial feature engineering

import pandas as pd
import numpy as np
from typing import List, Dict

class CreditFeatureEngineer:
    """
    Central definition of features.
    Ensures training and inference always use the exact same columns in the same order.
    """
    
    # Define the exact list of features expected by the model
    # These must match the columns in your clean CSV and the keys in your constraints.yaml
    FEATURES = [
        'roa', 
        'debt_ratio', 
        'operating_margin', 
        'net_income_assets'
    ]

    @classmethod
    def validate_inputs(cls, data: Dict[str, float]) -> bool:
        """Checks if all required features are present in the input dictionary."""
        missing = [f for f in cls.FEATURES if f not in data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return True

    @classmethod
    def prepare_for_inference(cls, data: Dict[str, float]) -> pd.DataFrame:
        """Converts a dictionary input into a single-row DataFrame with correct column order."""
        cls.validate_inputs(data)
        # Create DataFrame and reorder columns to match training order strictly
        df = pd.DataFrame([data])
        return df[cls.FEATURES]

    @classmethod
    def prepare_for_training(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Selects only relevant columns from the large training dataset."""
        missing = [f for f in cls.FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Training data missing columns: {missing}")
        return df[cls.FEATURES]