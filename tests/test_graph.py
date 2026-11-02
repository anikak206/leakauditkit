from leakauditkit.graph import build_contamination_graph, export_graph_json, summarize_graph


def test_graph_basic():
    matches = [
        {"train_id": "t1", "eval_id": "e1", "score": 1.0},
        {"train_id": "t1", "eval_id": "e2", "score": 0.8},
        {"train_id": "t2", "eval_id": "e3", "score": 0.9},
    ]
    G = build_contamination_graph(matches)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 3
    s = summarize_graph(G)
    assert s.contaminated_eval_count == 3
    assert s.high_risk_train[0][0] == "t1"
    assert s.high_risk_train[0][1] == 2


def test_graph_empty():
    G = build_contamination_graph([])
    s = summarize_graph(G)
    assert s.n_nodes == 0


def test_export():
    matches = [{"train_id": "t1", "eval_id": "e1", "score": 1.0}]
    G = build_contamination_graph(matches)
    j = export_graph_json(G)
    assert "nodes" in j and "edges" in j