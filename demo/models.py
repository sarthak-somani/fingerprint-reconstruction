"""The repair models the demo offers, built once and kept in memory.

A model is an experiment folder (experiments/<group>/EXX_name/). It is offered only if every file its
reconstructor needs exists (checkpoint, selected.json), so an experiment that is still training
(E12 before its CCDF sweep has written selected.json) is simply left out instead of crashing.
"""
import json
import time

import torch

from fpr import experiments
from fpr.models.reconstructors import resolve

DEFAULT_MODEL = "E09_ccdf"              # E09 and E22 tie for best on v3 test across seeds (E27)
CANDIDATES = {                          # dropdown order; label shown next to the name
    "E09_ccdf": "best (tied): ResShift + CCDF sampling",
    "E22_ccdf_aux_minutiae_cldice": "best (tied): ResShift + minutiae/clDice losses + CCDF",
    "E02_resshift": "ResShift diffusion, 15 steps",
    "E04_msu_stft": "baseline: MSU STFT enhancement (no learning)",
    "E20_fingergan_finetune": "baseline: FingerGAN fine-tuned on our data",
    "E12_norm_ccdf_cldice_cv": "ResShift on normalised prints + clDice/edge losses, CCDF",
}


def missing_files(name: str) -> list[str]:
    """Files the experiment's reconstructor reads that do not exist yet (empty = loadable)."""
    folder, params = experiments.experiment_dir(name), experiments.load_params(name)
    inference = params.get("inference", {})
    needed = [resolve(inference[key], folder) for key in ("checkpoint", "selected") if key in inference]
    if "e02" in params:                                  # E09 samples E02's network
        needed.append(resolve(params["e02"]["checkpoint"], folder))
    if "checkpoint" in params.get("model", {}):          # E22 samples E07's network
        needed.append(resolve(params["model"]["checkpoint"], folder))
    return [str(path) for path in needed if not path.exists()]


def available_models() -> list[str]:
    """CANDIDATES whose folder exists and whose files are all present, in CANDIDATES order."""
    names = set(experiments.list_experiments())
    return [name for name in CANDIDATES if name in names and not missing_files(name)]


def output_normalized(name: str) -> bool:
    """True if the model returns an MSU-normalised print (E12): the evaluator measures it as stored."""
    return bool(experiments.load_params(name).get("output_normalized"))


_LOADED = {}                            # name -> Reconstructor, kept for the life of the server


def load_model(name: str):
    """(Reconstructor, seconds spent loading it now: 0 if it was already in memory)."""
    if name in _LOADED:
        return _LOADED[name], 0.0
    start = time.perf_counter()
    _LOADED[name] = experiments.build_reconstructor(name)
    return _LOADED[name], time.perf_counter() - start


def device_of(model) -> str:
    """'cuda' or 'cpu': where the model runs (non-learned methods report 'cpu')."""
    return getattr(model, "device", "cpu")


def benchmark_line(name: str) -> str:
    """Median seconds per print from experiments/<name>/benchmark.json, e.g. '0.48 s on NVIDIA A40'."""
    path = experiments.experiment_dir(name) / "benchmark.json"
    if not path.exists():
        return "n/a"
    bench = json.loads(path.read_text())
    return f"{bench['seconds_median']:.2f} s ({bench['device_name'].split(',')[0]})"


def run_model(model, degraded, mask):
    """(repaired uint8 (H, W), seconds) for one print; the GPU is synchronised so the time is honest."""
    start = time.perf_counter()
    repaired = model.repair(degraded, mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return repaired, time.perf_counter() - start
