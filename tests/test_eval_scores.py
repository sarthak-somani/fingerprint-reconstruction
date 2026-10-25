"""Tests for evaluation/evaluate.py's headline scores and evaluation/compare_experiments.py's Holm step."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import compare_experiments  # noqa: E402
import evaluate  # noqa: E402


def ridges(h=96, w=96, period=9.0, angle=0.3):
    y, x = np.mgrid[:h, :w]
    return 127 + 100 * np.sin(2 * np.pi * (x * np.cos(angle) + y * np.sin(angle)) / period)


def test_ridge_fidelity_identical_inverted_restyled():
    clean, region = ridges(), np.ones((96, 96), bool)
    assert evaluate.ridge_fidelity(clean, clean, [region])[0] > 0.99
    assert evaluate.ridge_fidelity(clean, 255 - clean, [region])[0] < -0.99
    restyled = 0.3 * clean + 60                       # other brightness and contrast, same ridges
    assert evaluate.ridge_fidelity(clean, restyled, [region])[0] > 0.99
    flat = np.full_like(clean, 127.0)
    assert abs(evaluate.ridge_fidelity(clean, flat, [region])[0]) < 0.05


def test_headline_scores_counts_and_gate():
    n = 40
    probes = pd.DataFrame({
        "key": [f"k{i}" for i in range(n)], "id": [f"i{i}" for i in range(n)], "type": ["oil"] * n,
        "acc_clean": [True] * n,
        "acc_deg": [False] * 10 + [True] * 30,
        "acc_rec": [True] * 8 + [False] * 2 + [False] * 1 + [True] * 29,   # 8 rescued, 1 broken
        "fmr_deg": [0.02] * n, "fmr_rec": [0.02] * n,
        "ridge_region_deg": [0.8] * n, "ridge_region_rec": [0.9] * n})
    all_ = evaluate.headline_scores(probes).set_index("type").loc["all"]
    assert (all_["rescued"], all_["broken"]) == (8, 1)
    assert np.isclose(all_["net_rescue"], 7 / n)
    assert all_["gate"] == "pass" and np.isclose(all_["fmr_delta"], 0)
    assert np.isclose(all_["ridge_delta"], 0.1)
    assert all_["net_rescue_lo"] <= all_["net_rescue"] <= all_["net_rescue_hi"]


def test_gate_fails_on_large_fmr_increase():
    n = 40
    probes = pd.DataFrame({
        "key": [f"k{i}" for i in range(n)], "id": [f"i{i}" for i in range(n)], "type": ["scar"] * n,
        "acc_clean": [True] * n, "acc_deg": [True] * n, "acc_rec": [True] * n,
        "fmr_deg": [0.01] * n, "fmr_rec": [0.03] * n,
        "ridge_region_deg": [0.8] * n, "ridge_region_rec": [0.8] * n})
    assert evaluate.headline_scores(probes).set_index("type").loc["all", "gate"] == "fail"


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(compare_experiments.holm(p), [0.04, 0.09, 0.09, 0.5])


def test_sign_flip_is_mcnemar_for_one_sample_per_finger():
    from scipy import stats
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = rng.choice([-1, 0, 1], size=int(rng.integers(1, 40)))
        gained, lost = int((d > 0).sum()), int((d < 0).sum())
        expected = stats.binomtest(gained, gained + lost).pvalue if gained + lost else 1.0
        assert np.isclose(evaluate.sign_flip_p(d, np.arange(len(d))), expected)


def test_clustered_statistics():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1, 60)
    single = np.arange(60)
    # one sample per finger: the cluster bootstrap is the ordinary one
    plain = x[evaluate.boot_index(60, "s")].mean(axis=1)
    assert np.allclose(evaluate.boot_means(x, single, "s"), plain)
    # the same 20 values three times: clustered CI = CI of the 20 fingers, wider than treating 60 as independent
    x3, ids3 = np.repeat(x[:20], 3), np.repeat(np.arange(20), 3)
    lo, hi = evaluate.boot_ci(x3, ids3, "s")
    naive_lo, naive_hi = evaluate.boot_ci(x3, np.arange(60), "s")
    assert hi - lo > 1.3 * (naive_hi - naive_lo)
    # per-finger tests: triplicating the data must not make it more significant
    d = np.array([1, 1, 1, 1, 1, 0, -1])
    assert np.isclose(evaluate.sign_flip_p(np.repeat(d, 3), np.repeat(np.arange(7), 3)),
                      evaluate.sign_flip_p(d, np.arange(7)))
    assert np.isclose(evaluate.wilcoxon_p(np.repeat(x[:20], 3), np.repeat(np.arange(20), 3)),
                      evaluate.wilcoxon_p(x[:20]))
