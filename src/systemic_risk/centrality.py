import networkx as nx
import logging
import yaml
from typing import Dict

class CentralityCalculator:
    """
    Calculates network importance metrics (PageRank, Degree, Betweenness).
    Optimized for bulk calculation to avoid re-running expensive graph algos per query.
    """
    
    def __init__(self, config_path="configs/model_config.yaml"):
        self.logger = logging.getLogger("CentralityCalc")
        self.config = self._load_config(config_path)
        
        # Cache for scores
        self.pagerank_scores = {}
        self.degree_scores = {}
        self.betweenness_scores = {}

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f).get('systemic_risk', {})
        except FileNotFoundError:
            return {}

    def compute_all_metrics(self, graph: nx.DiGraph):
        """
        Runs heavy graph algorithms ONCE for the entire network.
        Must be called after building the graph.
        """
        if graph.number_of_nodes() == 0:
            return

        self.logger.info("🧮 Computing Network Centrality Metrics...")
        
        # 1. PageRank (Liquidity Importance)
        damping = self.config.get('damping_factor', 0.85)
        try:
            self.pagerank_scores = nx.pagerank(graph, weight='weight', alpha=damping)
        except Exception as e:
            self.logger.error(f"PageRank failed: {e}")
            self.pagerank_scores = {n: 0.0 for n in graph.nodes()}

        # 2. Degree Centrality (Connectivity)
        self.degree_scores = nx.degree_centrality(graph)
        
        # 3. Betweenness (Bridge Nodes) - Expensive, so we might skip on huge graphs
        # We limit k (samples) for speed if graph is huge
        k_val = min(100, len(graph)) if len(graph) > 500 else None
        self.betweenness_scores = nx.betweenness_centrality(graph, k=k_val)

        self.logger.info("✅ Network Metrics Computed.")

    def get_risk_score(self, node: str) -> float:
        """
        Returns a normalized systemic risk score (0-100) using cached metrics.
        """
        if node not in self.pagerank_scores:
            return 0.0

        # Retrieve raw metrics
        pr = self.pagerank_scores.get(node, 0)
        deg = self.degree_scores.get(node, 0)
        bet = self.betweenness_scores.get(node, 0)

        # Weighted Formula (Heuristic)
        # PageRank is usually very small (e.g. 0.001), so we scale it heavily
        # These weights should be tuned in production
        raw_score = (pr * 50.0) + (deg * 30.0) + (bet * 20.0)
        
        # Scale to 0-100 and clip
        # We use a logarithmic scaler or simple multiplier to make it readable
        final_score = min(raw_score * 100, 100.0)
        
        return round(final_score, 2)

    def get_metrics(self, node: str) -> Dict[str, float]:
        """Returns raw metrics for explainability."""
        return {
            "pagerank": self.pagerank_scores.get(node, 0),
            "degree": self.degree_scores.get(node, 0),
            "betweenness": self.betweenness_scores.get(node, 0)
        }