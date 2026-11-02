from leakauditkit.benchmark import evaluate, make_scenario, threshold_sweep
from leakauditkit.detectors.ngram import NgramDetector


def test_make_scenario_verbatim():
    sc = make_scenario("v", n_train=20, n_eval=20, contamination_rate=0.25, mode="verbatim", seed=1)
    assert len(sc.train) == 20
    assert len(sc.eval_) == 20
    assert len(sc.ground_truth) == 5


def test_make_scenario_clean():
    sc = make_scenario("c", n_train=20, n_eval=20, contamination_rate=0.25, mode="clean", seed=1)
    assert len(sc.ground_truth) == 0


def test_evaluate_perfect():
    sc = make_scenario("v", n_train=20, n_eval=20, contamination_rate=0.25, mode="verbatim", seed=2)
    det = NgramDetector(n=5, threshold=0.3, min_shared=3)
    r = evaluate(det, sc)
    assert r.recall == 1.0
    assert r.precision == 1.0


def test_threshold_sweep():
    sc = make_scenario("v", n_train=20, n_eval=20, contamination_rate=0.25, mode="verbatim", seed=3)
    sweep = threshold_sweep(lambda threshold=0.3: NgramDetector(n=5, threshold=threshold, min_shared=3),
                            sc, [0.2, 0.3, 0.5])
    assert len(sweep) == 3
    assert all("f1" in s for s in sweep)


def test_invalid_rate():
    import pytest
    with pytest.raises(ValueError):
        make_scenario("x", contamination_rate=1.5)