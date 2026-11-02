from leakauditkit.shortcuts import audit_shortcuts


def test_no_shortcut():
    records = [
        {"text": "hello world", "label": "pos", "prediction": "pos", "src": "A"},
        {"text": "foo bar", "label": "neg", "prediction": "neg", "src": "B"},
    ]
    rep = audit_shortcuts(records, text_key="text", feature_keys=["src"])
    assert rep.n == 2


def test_class_imbalance_flag():
    records = [{"label": "pos", "prediction": "pos", "src": "A"} for _ in range(9)] + \
              [{"label": "neg", "prediction": "neg", "src": "B"}]
    rep = audit_shortcuts(records, feature_keys=["src"])
    assert any("class_imbalance" in f for f in rep.flags)


def test_shortcut_feature_detected():
    # feature 'src' perfectly predicts label
    records = []
    for i in range(20):
        lab = "pos" if i < 10 else "neg"
        src = "A" if lab == "pos" else "B"
        records.append({"label": lab, "prediction": lab, "src": src})
    rep = audit_shortcuts(records, feature_keys=["src"])
    assert any("shortcut_feature" in f for f in rep.flags)


def test_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        audit_shortcuts([])