# Sentiment shock logic
import logging
from typing import List

class SentimentStressOverlay:
    """
    Applies heuristic rules to amplify risk when specific 'Panic Keywords' are found.
    This acts as a safety net over the pure ML model.
    """
    def __init__(self):
        self.logger = logging.getLogger("StressOverlay")
        # Words that indicate immediate existential threat to a financial entity
        self.panic_keywords = [
            "fraud", "investigation", "bankruptcy", "insolvency", 
            "default", "sanctions", "embezzlement", "raid", "jail"
        ]

    def apply_shock(self, base_risk_score: float, headlines: List[str]) -> float:
        """
        Adjusts the ML-based risk score based on keyword severity.
        """
        if not headlines:
            return base_risk_score

        penalty = 0.0
        found_keywords = []

        for text in headlines:
            text_lower = text.lower()
            for word in self.panic_keywords:
                if word in text_lower:
                    # Critical keywords add a massive penalty
                    penalty += 20.0
                    found_keywords.append(word)

        # If we found panic words, we log it
        if penalty > 0:
            self.logger.info(f"🚨 Panic Keywords Detected: {list(set(found_keywords))}. Boosting risk by {penalty}.")

        # Combine ML score + Penalty
        final_score = base_risk_score + penalty
        
        # Cap at 100
        return min(final_score, 100.0)