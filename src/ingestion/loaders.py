import pandas as pd
import logging
import os
from typing import Dict, Optional

class SentinelDataLoader:
    """
    Production-grade Data Loader.
    Responsibility: Read raw files, standardize column names, and return clean DataFrames.
    Does NOT handle logic, ID mapping, or merging.
    """
    def __init__(self, data_dir: str):
        self.logger = logging.getLogger("SentinelLoader")
        self.data_dir = data_dir

    def _standardize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to ensure column names are always lowercase and stripped of spaces."""
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        return df

    def load_credit_data(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "credit_clean.csv")
        try:
            df = pd.read_csv(path)
            self.logger.info(f"✅ Loaded Credit Data: {df.shape}")
            return self._standardize_cols(df)
        except FileNotFoundError:
            self.logger.error(f"❌ Critical: Credit data not found at {path}")
            raise

    def load_network_data(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "network_clean.csv")
        if not os.path.exists(path):
            self.logger.warning(f"⚠️ Network data missing at {path}. Returning empty.")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df = self._standardize_cols(df)
        
        # Robust renaming for graph columns
        rename_map = {}
        for col in df.columns:
            if col in ['src', 'sender', 'from', 'origin', 'source_id']: rename_map[col] = 'source'
            elif col in ['dst', 'receiver', 'to', 'dest', 'target_id']: rename_map[col] = 'target'
            elif col in ['amt', 'value', 'weight']: rename_map[col] = 'amount'
        
        df.rename(columns=rename_map, inplace=True)
        self.logger.info(f"✅ Loaded Network Data: {df.shape}")
        return df

    def load_news_data(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "news_clean.csv")
        if not os.path.exists(path):
            self.logger.warning(f"⚠️ News data missing at {path}. Returning empty.")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df = self._standardize_cols(df)
        if 'text' in df.columns:
            df.rename(columns={'text': 'headline'}, inplace=True)
            
        self.logger.info(f"✅ Loaded News Data: {df.shape}")
        return df