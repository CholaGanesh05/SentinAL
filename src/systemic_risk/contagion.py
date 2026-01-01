# Network failure simulation

import networkx as nx
import logging

class ContagionSimulator:
    """
    Simulates network failure scenarios (Stress Testing).
    """
    def __init__(self):
        self.logger = logging.getLogger("ContagionSim")

    def simulate_failure(self, graph: nx.DiGraph, start_node: str) -> dict:
        """
        Simulates the collapse of 'start_node' and measures the impact.
        Returns:
            - impacted_neighbors: count of immediate partners affected
            - total_value_at_risk: sum of outgoing transaction value
            - contagion_score: 0-100 severity score
        """
        if start_node not in graph:
            return {"contagion_score": 0.0, "impact_magnitude": 0.0}

        # 1. Direct Impact (First-Order)
        # Who was expecting money from this node?
        successors = list(graph.successors(start_node))
        
        total_value_at_risk = 0.0
        for neighbor in successors:
            # Sum up the weights (transaction amounts) on outgoing edges
            weight = graph[start_node][neighbor].get('weight', 0)
            total_value_at_risk += weight

        # 2. Network Vulnerability (Ego Graph Density)
        # If the immediate network is very dense, failure spreads faster.
        try:
            # Get the subgraph of the node and its neighbors
            ego_graph = nx.ego_graph(graph, start_node, radius=1)
            density = nx.density(ego_graph)
        except Exception:
            density = 0.0

        # 3. Calculate Contagion Score (Heuristic)
        # High Value At Risk + High Density = High Contagion Risk
        # We normalize value_at_risk using a log scale or threshold (e.g., >1M is high)
        # For this implementation, we map density (0-1) heavily.
        
        # Simple heuristic: Density contributes 40%, Connectivity 60%
        # We cap the connection count impact at 20 connections for normalization
        connectivity_factor = min(len(successors) / 20.0, 1.0)
        
        contagion_score = (density * 40) + (connectivity_factor * 60)
        
        return {
            "contagion_score": round(contagion_score, 2),
            "direct_neighbors_at_risk": len(successors),
            "total_value_at_risk": round(total_value_at_risk, 2),
            "local_network_density": round(density, 2)
        }