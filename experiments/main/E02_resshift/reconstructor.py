"""E02_resshift: ResShift's 15-step reverse chain on this experiment's checkpoint (fpr/models/reconstructors.py)."""
from pathlib import Path

from fpr.models.reconstructors import resolve, ResShiftReconstructor

HERE = Path(__file__).resolve().parent


def build_reconstructor(params: dict, random_init: bool = False) -> ResShiftReconstructor:
    checkpoint = None if random_init else resolve(params["inference"]["checkpoint"], HERE)
    return ResShiftReconstructor(params, params["inference"], checkpoint)
