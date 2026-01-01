import logging
import sys
import os
from datetime import datetime
from typing import Optional

# Ensure root path is accessible
sys.path.append(os.getcwd())

from src.schemas.risk_objects import RiskSignal, RiskType
from src.sentiment_risk.finbert_inference import FinBERTAnalyzer
from src.sentiment_risk.news_loader import NewsLoader
from src.sentiment_risk.stress_overlay import SentimentStressOverlay

class SentimentRiskEngine:
    """
    The Sentiment Controller.
    1. Fetches news for an entity.
    2. Runs FinBERT to get probability of negative sentiment.
    3. Applies Keyword Stress Test to catch 'Fraud/Bankruptcy' events.
    4. Outputs a standardized RiskSignal.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("SentimentEngine")
        
        # Initialize components
        self.loader = NewsLoader()
        self.analyzer = FinBERTAnalyzer() # Singleton, loads model once
        self.overlay = SentimentStressOverlay()

    def analyze(self, entity_id: str) -> RiskSignal:
        """
        Full Sentiment Pipeline for one entity.
        """
        # 1. Fetch Data
        headlines = self.loader.get_headlines(entity_id)
        
        if not headlines:
            self.logger.info(f"No news found for {entity_id}. Returning Neutral signal.")
            return self._create_neutral_signal(entity_id)

        # 2. AI Inference (FinBERT)
        # Returns {positive: 0.1, negative: 0.8, neutral: 0.1}
        bert_scores = self.analyzer.predict(headlines)
        
        # We use the 'Negative' probability as the base risk score (0-100)
        base_risk = bert_scores["negative"] * 100.0

        # 3. Apply Heuristic Stress (Panic Keywords)
        final_score = self.overlay.apply_shock(base_risk, headlines)
        
        # 4. Construct Metadata
        metadata = {
            "headline_count": len(headlines),
            "finbert_raw_negative": round(base_risk, 2),
            "top_headline": headlines[0][:100] + "..." if headlines else ""
        }

        # 5. Build Signal
        signal = RiskSignal(
            entity_id=entity_id,
            risk_type=RiskType.SENTIMENT,
            raw_score=bert_scores["negative"], # 0.0 - 1.0
            normalized_score=final_score,      # 0 - 100 (possibly boosted by overlay)
            confidence=0.90,                   # BERT is generally confident
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.logger.info(f"📰 Sentiment Risk for {entity_id}: {final_score:.2f}")
        return signal

    def _create_neutral_signal(self, entity_id: str) -> RiskSignal:
        """Fallback for when no news exists."""
        return RiskSignal(
            entity_id=entity_id,
            risk_type=RiskType.SENTIMENT,
            raw_score=0.0,
            normalized_score=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={"note": "No news data available"}
        )