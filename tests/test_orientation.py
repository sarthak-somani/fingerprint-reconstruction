"""Tests for fpr/data/orientation.py on synthetic ridges with known orientation."""
import numpy as np
import pytest

from fpr.data import orientation


def ridges(angle: float, size: int = 128, period: float = 9.0) -> np.ndarray:
    """uint8 sinusoidal ridges running at `angle` (radians from the +x axis)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    # Distance across the ridges, measured along the normal of the ridge direction.
    across = -xx * np.sin(angle) + yy * np.cos(angle)
    return (127.5 + 127.5 * np.cos(2 * np.pi * across / period)).astype(np.uint8)


@pytest.mark.parametrize("angle_deg", [0, 30, 60, 90, 135])
def test_orientation_of_straight_ridges(angle_deg):
    theta, coherence = orientation.orientation_field(ridges(np.radians(angle_deg)))
    centre = np.zeros(theta.shape, bool)
    centre[32:96, 32:96] = True
    error = orientation.orientation_error_deg(theta, np.full(theta.shape, np.radians(angle_deg)), centre)
    assert error < 3.0
    assert coherence[centre].mean() > 0.8


def test_error_respects_180_periodicity():
    a = np.full((4, 4), np.radians(1.0))
    b = np.full((4, 4), np.radians(179.0))
    assert orientation.orientation_error_deg(a, b, np.ones((4, 4), bool)) == pytest.approx(2.0)


def test_fill_recovers_orientation_in_hole():
    angle = np.radians(40)
    theta, _ = orientation.orientation_field(ridges(angle, size=160))
    hole = np.zeros(theta.shape, bool)
    hole[60:100, 60:100] = True
    damaged = theta.copy()
    damaged[hole] = 0.0                                   # wrong angle inside the hole
    filled = orientation.fill_orientation(damaged, ~hole)
    assert orientation.orientation_error_deg(filled, np.full(theta.shape, angle), hole) < 3.0
    assert np.array_equal(filled[~hole], damaged[~hole])
