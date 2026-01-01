from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class RiskType(Enum):
    CREDIT = "credit"
    MARKET = "market"
    SYSTEMIC = "systemic"
    SENTIMENT = "sentiment"
    LIQUIDITY = "liquidity"

class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class RiskSignal(BaseModel):
    """
    Standardized unit of risk information from any engine.
    """
    entity_id: str
    risk_type: RiskType
    raw_score: float         # The raw output
    normalized_score: float  # 0-100 Scale (This is what Fusion Engine needs!)
    confidence: float
    metadata: Dict[str, Any] = {}
    timestamp: datetime = datetime.now()

class AggregatedRiskProfile(BaseModel):
    """
    The final fused 360-degree view.
    """
    entity_id: str
    composite_risk_score: float
    risk_level: RiskLevel
    contributing_signals: List[RiskSignal]
    timestamp: datetime = datetime.now()
    
    def to_json(self):
        """Helper to serialize to JSON-safe dict"""
        return self.model_dump(mode='json')