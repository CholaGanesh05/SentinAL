import networkx as nx
import logging
from typing import List, Dict

class GraphBuilder:
    """
    Constructs a weighted directed financial graph.
    Nodes = Entities
    Edges = Financial Flows (Transactions)
    """
    def __init__(self):
        self.logger = logging.getLogger("GraphBuilder")

    def build_graph(self, transactions: List[Dict]) -> nx.DiGraph:
        """
        Input: List of dicts [{'source': 'ENT-01', 'target': 'ENT-02', 'amount': 100}]
        Output: Directed Graph with weighted edges.
        """
        G = nx.DiGraph()
        
        if not transactions:
            self.logger.warning("⚠️ No transactions provided. Returning empty graph.")
            return G

        count = 0
        for txn in transactions:
            src = txn.get('source')
            dst = txn.get('target')
            amount = float(txn.get('amount', 1.0))

            if not src or not dst:
                continue

            # Add edge with weight aggregation (summing multiple transactions)
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += amount
            else:
                G.add_edge(src, dst, weight=amount)
            count += 1
                
        self.logger.info(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges from {count} txns.")
        return G