"""Tests for fpr/metrics/matching/verification.py on toy score lists with known answers."""
import numpy as np
import pytest

from fpr.metrics.matching import verification


def test_perfect_separation():
    genuine, impostor = np.array([10, 11, 12.0]), np.array([1, 2, 3.0])
    assert verification.eer(genuine, impostor) == 0.0
    assert verification.tar_at_far(genuine, impostor, 0.0) == 1.0


def test_total_overlap_gives_half():
    scores = np.arange(100.0)
    assert verification.eer(scores, scores) == pytest.approx(0.5, abs=0.01)


def test_known_eer():
    # Threshold 5: impostors >= 5 are {5} -> FAR 1/4; genuine < 5 is {4} -> FRR 1/4.
    genuine, impostor = np.array([4, 6, 7, 8.0]), np.array([1, 2, 3, 5.0])
    assert verification.eer(genuine, impostor) == pytest.approx(0.25)


def test_tar_at_far():
    genuine = np.array([5, 6, 7, 8, 9.0])
    impostor = np.arange(100.0) / 10          # 0.0 .. 9.9
    # FAR <= 1% needs threshold > 9.8 (only 9.9 accepted) -> threshold 9.9, genuine >= 9.9 = none.
    assert verification.tar_at_far(genuine, impostor, 0.01) == 0.0
    # FAR <= 10% -> threshold 9.0 (impostors 9.0-9.9 = 10) -> genuine >= 9: one of five.
    assert verification.tar_at_far(genuine, impostor, 0.10) == pytest.approx(0.2)


def test_roc_points_monotone():
    rng = np.random.default_rng(0)
    points = verification.roc_points(rng.normal(2, 1, 200), rng.normal(0, 1, 2000))
    assert np.all(np.diff(points["far"]) <= 0) and np.all(np.diff(points["frr"]) >= 0)
    assert points["far"][-1] == 0.0 and points["frr"][0] == 0.0
