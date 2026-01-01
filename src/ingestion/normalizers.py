import pandas as pd
import numpy as np
import logging

class EntityLinker:
    """
    Handles the logic of mapping disparate datasets (Graph/News) to the 
    core Credit Entities.
    """
    def __init__(self):
        self.logger = logging.getLogger("EntityLinker")

    def generate_ids(self, df: pd.DataFrame, prefix="ENT-") -> pd.DataFrame:
        """Assigns stable IDs to the main dataframe."""
        ids = [f"{prefix}{i:05d}" for i in range(len(df))]
        df['entity_id'] = ids
        return df

    def map_network_to_ids(self, df_credit: pd.DataFrame, df_network: pd.DataFrame) -> pd.DataFrame:
        """
        Maps raw graph nodes to generated Entity IDs (ENT-001) using Vectorization.
        This is significantly faster than looping row-by-row.
        """
        if df_network.empty: return df_network

        self.logger.info("🔗 Mapping Graph Nodes to Entity IDs (Vectorized)...")
        
        # Get all unique nodes in graph
        unique_nodes = pd.concat([df_network['source'], df_network['target']]).unique()
        num_nodes = len(unique_nodes)
        
        # Get available Entity IDs
        available_ids = df_credit['entity_id'].values
        num_ids = len(available_ids)
        
        # --- OPTIMIZED VECTORIZED MAPPING ---
        # Instead of a slow loop, we calculate all assignments in one go.
        
        if num_nodes <= num_ids:
            # We have enough IDs for everyone
            assignments = available_ids[:num_nodes]
        else:
            # We have more nodes than IDs.
            # 1. Assign unique IDs to the first batch
            direct_map = available_ids
            
            # 2. Assign RANDOM existing IDs to the excess nodes (in one batch operation)
            extra_needed = num_nodes - num_ids
            random_fill = np.random.choice(available_ids, size=extra_needed)
            
            # Combine them
            assignments = np.concatenate([direct_map, random_fill])

        # Create the Dictionary Map instantly
        node_map = dict(zip(unique_nodes, assignments))

        # Apply Map
        df_network['source'] = df_network['source'].map(node_map)
        df_network['target'] = df_network['target'].map(node_map)
        
        self.logger.info("✅ Mapping Complete.")
        return df_network

class FeatureScaler:
    """
    Handles statistical normalization (MinMax, Scaling, Filling NaNs).
    """
    def clean_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces Infinite values and NaNs with 0 or Mean."""
        # Replace infinity with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaN with 0 (assuming missing data = no assets/debt)
        df.fillna(0, inplace=True)
        return df