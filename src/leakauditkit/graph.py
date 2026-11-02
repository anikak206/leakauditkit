"""Contamination graph: model overlap relationships between train/eval items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass
class GraphSummary:
    n_nodes: int
    n_edges: int
    n_components: int
    largest_component_size: int
    contaminated_eval_count: int
    high_risk_train: list[tuple[str, int]]  # (train_id, number of evals it contaminates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_components": self.n_components,
            "largest_component_size": self.largest_component_size,
            "contaminated_eval_count": self.contaminated_eval_count,
            "high_risk_train": [{"train_id": t, "blast_radius": b} for t, b in self.high_risk_train],
        }


def build_contamination_graph(matches: list[dict[str, Any]]) -> nx.Graph:
    """Build an undirected graph from a list of match dicts.

    Each match must have train_id, eval_id, score.
    """
    G = nx.Graph()
    for m in matches:
        t = m["train_id"]
        e = m["eval_id"]
        score = float(m.get("score", 1.0))
        G.add_node(t, kind="train")
        G.add_node(e, kind="eval")
        if G.has_edge(t, e):
            if score > G[t][e].get("score", 0):
                G[t][e]["score"] = score
        else:
            G.add_edge(t, e, score=score)
    return G


def summarize_graph(G: nx.Graph, top_k: int = 5) -> GraphSummary:
    if G.number_of_nodes() == 0:
        return GraphSummary(0, 0, 0, 0, 0, [])
    components = list(nx.connected_components(G))
    # blast radius per train node
    blast: dict[str, int] = {}
    for node, attrs in G.nodes(data=True):
        if attrs.get("kind") != "train":
            continue
        # count eval nodes reachable within component
        comp = None
        for c in components:
            if node in c:
                comp = c
                break
        if comp is None:
            continue
        evals = [n for n in comp if G.nodes[n].get("kind") == "eval"]
        blast[node] = len(evals)
    high_risk = sorted(blast.items(), key=lambda x: x[1], reverse=True)[:top_k]
    contaminated_evals = {n for n, a in G.nodes(data=True) if a.get("kind") == "eval"}
    return GraphSummary(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_components=len(components),
        largest_component_size=max(len(c) for c in components),
        contaminated_eval_count=len(contaminated_evals),
        high_risk_train=high_risk,
    )


def export_graph_json(G: nx.Graph) -> dict[str, Any]:
    return {
        "nodes": [{"id": n, "kind": G.nodes[n].get("kind", "unknown")} for n in G.nodes],
        "edges": [{"source": u, "target": v, "score": G[u][v].get("score", 0.0)} for u, v in G.edges],
    }