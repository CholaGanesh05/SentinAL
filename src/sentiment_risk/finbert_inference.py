import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict

class FinBERTAnalyzer:
    """
    Singleton wrapper for the ProsusAI/finbert model.
    Handles inference on financial text.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FinBERTAnalyzer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.logger = logging.getLogger("FinBERT")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"⚙️ Using Device: {self.device}")

        try:
            self.logger.info("⏳ Loading FinBERT model (ProsusAI/finbert)...")
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval() # Set to inference mode
            self.logger.info("✅ FinBERT loaded successfully.")
        except Exception as e:
            self.logger.error(f"❌ Failed to load FinBERT: {e}")
            raise e

    def predict(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyzes a list of texts and returns the AVERAGE sentiment probabilities.
        """
        if not texts:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        # Batch processing (e.g. process 16 headlines at a time)
        # This prevents crashing if an entity has 1000 headlines.
        batch_size = 16
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Apply Softmax to get probabilities (Logits -> 0.0-1.0)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        # Concatenate all batches
        if not all_probs:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate Mean Sentiment across all headlines
        # FinBERT Output Order: [Positive, Negative, Neutral]
        mean_probs = all_probs.mean(axis=0)
        
        return {
            "positive": float(mean_probs[0]),
            "negative": float(mean_probs[1]),
            "neutral":  float(mean_probs[2])
        }