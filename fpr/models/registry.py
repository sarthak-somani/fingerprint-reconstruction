"""Registry of reconstruction methods: get_reconstructor(name, **kwargs).

Used by fpr.experiments for experiments without a reconstructor.py (E01_gabor). 'clean' is not a reconstructor: it
is the ceiling condition (the undamaged print), handled by the scripts.
"""
from fpr.models.base import Reconstructor
from fpr.models.gabor import GaborReconstructor
from fpr.models.identity import IdentityReconstructor

REGISTRY = {
    "identity": IdentityReconstructor,
    "gabor": GaborReconstructor,
    # Learned models are not registered here: each lives in its own experiment folder
    # (experiments/<group>/EXX_name/reconstructor.py) and is loaded by fpr.experiments.
}


def get_reconstructor(name: str, **kwargs) -> Reconstructor:
    """Build the method called `name`; kwargs go to its constructor (e.g. checkpoint=...)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
