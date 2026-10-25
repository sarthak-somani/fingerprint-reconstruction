"""Identity 'reconstructor': returns the damaged input unchanged.

This is the floor of every comparison: how well the damaged probe matches
with no repair at all. Registered in reconstructors/__init__.py.
"""
import numpy as np

from fpr.models.base import Reconstructor


class IdentityReconstructor(Reconstructor):
    name = "identity"

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """uint8 (H, W) -> the same uint8 (H, W) image."""
        return degraded.copy()
