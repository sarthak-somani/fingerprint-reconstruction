"""E11_ctmsr_dtm as a Reconstructor: E10's one-step CTMSR repair with E11's DTM-fine-tuned weights."""
from pathlib import Path

from experiments.one_step.E10_ctmsr import reconstructor as e10

HERE = Path(__file__).resolve().parent


def build_reconstructor(params: dict, random_init: bool = False) -> e10.CtmsrReconstructor:
    return e10.build_reconstructor(params, folder=HERE, random_init=random_init)
