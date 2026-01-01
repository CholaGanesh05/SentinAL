# News text ingestion

import pandas as pd
import logging
import os
from typing import List, Dict

class NewsLoader:
    """
    Responsible for loading and indexing news headlines mapped to Entity IDs.
    """
    def __init__(self, data_path="data/processed/news_mapped.csv"):
        self.logger = logging.getLogger("NewsLoader")
        self.data_path = data_path
        self.news_index = {} # dict[entity_id] -> List[headlines]
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            self.logger.warning(f"⚠️ News data not found at {self.data_path}. Sentiment engine will have no data.")
            return

        try:
            df = pd.read_csv(self.data_path)
            
            # Ensure required columns exist
            if 'entity_id' not in df.columns or 'headline' not in df.columns:
                self.logger.error("❌ News CSV missing 'entity_id' or 'headline' columns.")
                return

            # Group by Entity ID for O(1) retrieval
            # This creates a dict: {'ENT-001': ['Bad news...', 'Good news...']}
            self.news_index = df.groupby('entity_id')['headline'].apply(list).to_dict()
            
            self.logger.info(f"✅ Loaded headlines for {len(self.news_index)} entities.")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading news data: {e}")

    def get_headlines(self, entity_id: str) -> List[str]:
        """Returns list of headlines for a specific entity."""
        return self.news_index.get(entity_id, [])