import logging
import pandas as pd
import sys
import os
from datetime import datetime
from typing import List, Dict

# Ensure root path is accessible
sys.path.append(os.getcwd())

from src.schemas.risk_objects import RiskSignal, RiskType
from src.systemic_risk.graph_builder import GraphBuilder
from src.systemic_risk.centrality import CentralityCalculator
from src.systemic_risk.contagion import ContagionSimulator

class SystemicRiskEngine:
    """
    The Systemic Risk Controller.
    1. Builds the graph from all transaction data.
    2. Pre-computes centrality metrics (PageRank, etc.) for speed.
    3. Runs contagion simulation on demand.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("SystemicEngine")
        self.builder = GraphBuilder()
        self.calculator = CentralityCalculator()
        self.simulator = ContagionSimulator()
        
        self.graph = None
        self.is_initialized = False

    def ingest_data(self, data_path="data/processed/network_mapped.csv"):
        """
        Loads network data and runs the heavy bulk calculations.
        MUST be called before analyze().
        """
        self.logger.info("🕸️ Initializing Systemic Risk Engine...")
        
        if not os.path.exists(data_path):
            self.logger.warning(f"⚠️ Network data not found at {data_path}. Engine will run in empty mode.")
            self.graph = self.builder.build_graph([])
            self.is_initialized = True
            return

        # Load CSV and convert to list of dicts
        df = pd.read_csv(data_path)
        transactions = df.to_dict(orient='records')
        
        # 1. Build Graph
        self.graph = self.builder.build_graph(transactions)
        
        # 2. Pre-compute Centrality (The "Heavy Lift")
        self.calculator.compute_all_metrics(self.graph)
        
        self.is_initialized = True
        self.logger.info("✅ Systemic Engine Ready.")

    def analyze(self, entity_id: str) -> RiskSignal:
        """
        Returns the Systemic Risk Profile for a specific entity.
        """
        if not self.is_initialized:
            raise RuntimeError("Systemic Engine not initialized. Call ingest_data() first.")

        # 1. Get Static Risk (Centrality)
        centrality_score = self.calculator.get_risk_score(entity_id)
        centrality_metrics = self.calculator.get_metrics(entity_id)
        
        # 2. Get Dynamic Risk (Contagion Simulation)
        contagion_data = self.simulator.simulate_failure(self.graph, entity_id)
        contagion_score = contagion_data.get("contagion_score", 0.0)

        # 3. Blended Score (60% Centrality, 40% Contagion Potential)
        final_score = (centrality_score * 0.6) + (contagion_score * 0.4)
        final_score = min(final_score, 100.0)

        # 4. Construct Metadata
        metadata = {
            "centrality_metrics": centrality_metrics,
            "stress_test_results": contagion_data,
            "network_nodes": self.graph.number_of_nodes()
        }

        # 5. Build Signal
        signal = RiskSignal(
            entity_id=entity_id,
            risk_type=RiskType.SYSTEMIC,
            raw_score=final_score,         # Raw and Normalized are similar here
            normalized_score=final_score,
            confidence=0.85,               # Graph algorithms are deterministic
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.logger.info(f"🕸️ Systemic Score for {entity_id}: {final_score:.2f}")
        return signal