"""Where an experiment's files live and how to build its reconstruction method.

An experiment is a folder experiments/<group>/<name>/ (name like 'E01_gabor', group one of
config.EXPERIMENT_GROUPS) with a params.json and, for learned models, a reconstructor.py.
Used by experiments/run_experiment.py, evaluation/run_eval.py, the trainers and the demo.
"""
import importlib
import json
import re
from pathlib import Path

import config
from fpr.data import eval_sets
from fpr.models.registry import get_reconstructor, Reconstructor

NAME_PATTERN = re.compile(r"^E\d{2}_[a-z0-9_]+$")     # E01_gabor, E12_diffusion_small


def _folders() -> dict[str, Path]:
    """{name: experiments/<group>/<name>/} for every experiment folder."""
    return {p.name: p for group in config.EXPERIMENT_GROUPS for p in (config.EXPERIMENTS_DIR / group).iterdir()
            if p.is_dir() and NAME_PATTERN.match(p.name)}


def experiment_dir(name: str) -> Path:
    """experiments/<group>/<name>/; raises if it does not exist or the name breaks the naming rule."""
    assert NAME_PATTERN.match(name), f"experiment names look like E01_gabor, got {name!r}"
    folders = _folders()
    assert name in folders, f"no experiment folder {name} in {config.EXPERIMENTS_DIR}/<group>/"
    return folders[name]


def list_experiments() -> list[str]:
    """Names of all experiment folders, in number order."""
    return sorted(_folders())


def load_params(name: str) -> dict:
    """The experiment's params.json as a dict."""
    return json.loads((experiment_dir(name) / "params.json").read_text())


def build_reconstructor(name: str, **kwargs) -> Reconstructor:
    """The experiment's method.

    If its folder has a reconstructor.py, its build_reconstructor(params, **kwargs) is called
    (learned models; e.g. random_init=True for a smoke test without a checkpoint). Otherwise
    params['method'] names a built-in method from fpr.models.registry, with params['kwargs'] passed on.
    """
    params = load_params(name)
    folder = experiment_dir(name)
    if (folder / "reconstructor.py").exists():
        # Experiment folders are packages: experiments.<group>.<name>.reconstructor
        module = importlib.import_module(f"experiments.{folder.parent.name}.{name}.reconstructor")
        return module.build_reconstructor(params, **kwargs)
    return get_reconstructor(params["method"], **params.get("kwargs", {}))


def check_data_version(name: str) -> None:
    """Fail early if the experiment's params.json pins a data version other than the active one."""
    wanted = load_params(name).get("data_version")
    assert wanted in (None, config.DATA_VERSION), (
        f"{name} runs on data {wanted}, but config.DATA_VERSION is {config.DATA_VERSION}: "
        f"set FPR_DATA_VERSION={wanted}")


def reconstructions_dir(name: str, split_name: str) -> Path:
    """<experiment folder>/reconstructions/{split}/ for v2, reconstructions/<version>/{split}/ otherwise
    (git-ignored), so one method's repairs of different data versions never mix."""
    base = experiment_dir(name) / "reconstructions"
    return base / split_name if config.DATA_VERSION == "v2" else base / config.DATA_VERSION / split_name


def reconstruction_path(name: str, split_name: str, probe_id: str, contaminant: str) -> Path:
    """<id>_<type>_rec.png inside reconstructions_dir, as evaluation/evaluate.py expects."""
    return reconstructions_dir(name, split_name) / eval_sets.layout_name(probe_id, contaminant,
                                                                         "reconstruction")
