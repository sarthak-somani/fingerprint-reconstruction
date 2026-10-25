"""CCDF sampler selection on v2 val (E09's rule), for E09 and for E21-E23 (E09's sampler on E06-E08's weights).

Usage (E21-E23; E09 runs it from its train.py, with the regression control):
    python -m fpr.validation.ccdf_sweep --exp E22_ccdf_aux_minutiae_cldice
Each configuration (starting estimate, t0, projection) repairs all val probes with the same trained
model; the repaired probes are matched against every val pair (bozorth3, as quick_check).
Selection (params.json 'sweep.selection'): highest mean genuine score among configurations whose
impostor p95 is not above plain E02's and whose EER is at most plain E02's + 1 point.
Writes <exp>/checkpoints/sweep.csv and <exp>/selected.json (read by its reconstructor.py).
"""
import argparse
import json
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd

import config
from fpr import experiments
from fpr.data import eval_sets, image_io, nist
from fpr.metrics.matching import nbis, verification

PLAIN_E02 = {"init": "y0", "t0": 15, "project": False}
REGRESSOR_ONLY = {"init": "regressor", "t0": 0, "project": False}


def configurations(sweep: dict) -> list[dict]:
    """The two references, then every init x t0 x project of params.json's grid."""
    grid = [{"init": i, "t0": t, "project": p} for i in sweep["inits"] for t in sweep["t0"] for p in sweep["project"]]
    return [PLAIN_E02, REGRESSOR_ONLY] + grid


def name(cfg: dict) -> str:
    return f"{cfg['init']}_t{cfg['t0']}" + ("_proj" if cfg["project"] else "")


def _score_probe(args: tuple) -> pd.DataFrame:
    """bozorth3 scores of one repaired probe against its val pairs (worker process)."""
    probe_id, image_path, group, gallery_xyt, tmp = args
    xyt, _ = nbis.run_mindtct(image_io.load_gray(image_path), Path(tmp) / probe_id)
    return group.assign(score=nbis.bozorth_scores(xyt, [gallery_xyt[g] for g in group["gallery_path"]]))


def score(images: dict[str, Path], workers: int) -> dict:
    """Matching summary of repaired val probes {probe_id: png}: EER, genuine mean/median, impostor p95."""
    pairs = pd.read_csv(eval_sets.pairs_path("val"), dtype={"probe_id": str})
    pairs = pairs[pairs["probe_id"].isin(images)]                     # all of val in the sweep
    gallery_xyt = {g: nbis.cached_xyt(nist.raw_path(g), config.XYT_CACHE_DIR / "gallery")
                   for g in pairs["gallery_path"].unique()}
    with tempfile.TemporaryDirectory() as tmp, Pool(workers) as pool:
        jobs = [(pid, images[pid], group, gallery_xyt, tmp) for pid, group in pairs.groupby("probe_id")]
        scores = pd.concat(pool.map(_score_probe, jobs), ignore_index=True)
    genuine = scores.loc[scores["label"] == 1, "score"].to_numpy()
    impostor = scores.loc[scores["label"] == 0, "score"].to_numpy()
    return {"eer": verification.eer(genuine, impostor), "genuine_mean": float(genuine.mean()),
            "genuine_median": float(np.median(genuine)), "impostor_p95": float(np.percentile(impostor, 95)),
            "tar_at_far_0.01": verification.tar_at_far(genuine, impostor, 0.01)}


def repair_val(method, out_dir: Path) -> dict[str, Path]:
    """Repair every val probe with `method` (a Reconstructor); returns {probe_id: png path}."""
    paths = {}
    for row in eval_sets.load_meta("val").itertuples():
        degraded = image_io.load_gray(eval_sets.triplet_path("val", row.probe_id, "degraded"))
        mask = image_io.load_mask(eval_sets.triplet_path("val", row.probe_id, "mask"))
        paths[row.probe_id] = out_dir / f"{row.probe_id}.png"
        image_io.save_png(paths[row.probe_id], method.repair(degraded, mask))
    return paths


def select(table: pd.DataFrame) -> dict:
    """The selected row of a sweep table (params.json 'sweep.selection')."""
    plain = table[table["config"] == name(PLAIN_E02)].iloc[0]
    ok = table[(table["impostor_p95"] <= plain["impostor_p95"]) & (table["eer"] <= plain["eer"] + 0.01)]
    best = ok.sort_values(["genuine_mean", "t0"], ascending=[False, True]).iloc[0]
    return {"init": best["init"], "t0": int(best["t0"]), "project": bool(best["project"]),
            "val": {k: float(best[k]) for k in ("eer", "genuine_mean", "impostor_p95")},
            "plain_e02_val": {k: float(plain[k]) for k in ("eer", "genuine_mean", "impostor_p95")}}


def run_sweep(method, configs: list[dict], out: Path, workers: int) -> pd.DataFrame:
    """Repair val with every configuration (method.config is swapped in place) and score it."""
    rows = []
    for cfg in configs:
        method.config, start = cfg, time.time()
        images = repair_val(method, out / "sweep_val" / name(cfg))
        rows.append({"config": name(cfg), **cfg, **score(images, workers), "seconds": round(time.time() - start)})
        pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="E21_ccdf_aux_minutiae, E22_ccdf_aux_minutiae_cldice or E23_...")
    args = parser.parse_args()
    params, folder = experiments.load_params(args.exp), experiments.experiment_dir(args.exp)
    out = folder / "checkpoints"
    out.mkdir(exist_ok=True)
    method = experiments.build_reconstructor(args.exp, config=PLAIN_E02)
    configs = [cfg for cfg in configurations(params["sweep"]) if cfg["init"] != "regressor"]
    selected = select(run_sweep(method, configs, out, params["sweep"]["workers"]))
    (folder / params["inference"]["selected"]).write_text(json.dumps(selected, indent=2) + "\n")
    print("selected", selected, flush=True)


if __name__ == "__main__":
    main()
