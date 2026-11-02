from unittest.mock import patch

from leakauditkit.hf_audit import audit_hf_dataset


def test_audit_hf_mocked():
    long = "duplicate text alpha beta gamma delta epsilon zeta eta theta iota kappa"
    train = [{"id": "t1", "text": long}]
    eval_ = [{"id": "e1", "text": long}]

    def fake_load(dataset, split, text_col="text", limit=None, id_col=None):
        return train if "train" in dataset else eval_

    with patch("leakauditkit.hf_audit.load_hf", side_effect=fake_load):
        payload = audit_hf_dataset("train_ds", "eval_ds", limit=5)
    assert payload["n_train"] == 1
    assert payload["n_eval"] == 1
    assert payload["result"]["n_matches"] == 1