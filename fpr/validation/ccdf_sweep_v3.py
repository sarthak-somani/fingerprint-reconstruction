"""CCDF sampler selection on v3 val (E12's rule), for E12-E16 (normalised prints) and E26 (raw prints).

Usage (after training, FPR_DATA_VERSION=v3):  python -m fpr.validation.ccdf_sweep_v3 --exp E12_norm_ccdf_cldice_cv
Every configuration of params.json 'ccdf' (plus the full 15-step chain, 'plain') repairs every v3 val
sample with the trained model: script 10's normalised copies for a normalised model, the raw damaged prints
for E26 (whose raw output is then normalised, as evaluate.py does). Each repaired sample is matched
(MINDTCT + bozorth3) against every val gallery print its probe is paired with (pairs.csv), the gallery
normalised too. tau = the score at which 1% of the clean normalised val probes' impostor comparisons are
accepted (accept = score > tau, as evaluate.py). Selection: params.json 'ccdf.selection'.
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
from fpr.data.normalization import normalize
from fpr.metrics.matching import nbis, verification

PLAIN = {"init": "y0", "t0": 15, "project": False}     # the full 15-step ResShift chain
FAR = 0.01


def configurations(grid: dict) -> list[dict]:
    """plain, then every init x t0 x project of params.json's 'ccdf' grid."""
    return [PLAIN] + [{"init": i, "t0": t, "project": p}
                      for i in grid["inits"] for t in grid["t0"] for p in grid["project"]]


def name(cfg: dict) -> str:
    return f"{cfg['init']}_t{cfg['t0']}" + ("_proj" if cfg["project"] else "")


def gallery_xyts(pairs: pd.DataFrame, out: Path) -> dict[str, Path]:
    """{gallery_path: .xyt of the normalised gallery print}, cached in checkpoints/gallery_norm/."""
    out = {}
    for g in pairs["gallery_path"].unique():
        xyt = out / "gallery_norm" / f"{Path(g).stem}.xyt"
        if not xyt.exists():
            nbis.run_mindtct(normalize(image_io.load_gray(nist.raw_path(g))), xyt.with_suffix(""))
        out[g] = xyt
    return out


def _score_sample(job: tuple) -> pd.DataFrame:
    """bozorth3 scores of one image against its probe's val pairs (worker process)."""
    sample_id, image_path, group, xyts, tmp = job
    xyt, _ = nbis.run_mindtct(image_io.load_gray(image_path), Path(tmp) / sample_id)
    return group.assign(sample_id=sample_id, score=nbis.bozorth_scores(xyt, [xyts[g] for g in group["gallery_path"]]))


def match(images: dict[str, Path], meta: pd.DataFrame, pairs: pd.DataFrame, xyts: dict, workers: int) -> pd.DataFrame:
    """Scores of {sample_id: png} against their probes' pairs: probe_id, gallery_path, label, sample_id, score."""
    groups = dict(tuple(pairs.groupby("probe_id")))
    probe_of = meta.set_index("sample_id")["probe_id"]
    with tempfile.TemporaryDirectory() as tmp, Pool(workers) as pool:
        jobs = [(s, path, groups[probe_of[s]], xyts, tmp) for s, path in images.items()]
        return pd.concat(pool.map(_score_sample, jobs), ignore_index=True)


def tau_from_clean(scores: pd.DataFrame) -> float:
    """Smallest score t with at most FAR of the clean impostor comparisons above it (evaluate.py's)."""
    impostor = scores.loc[scores["label"] == 0, "score"].to_numpy()
    return next(float(t) for t in np.unique(impostor) if np.mean(impostor > t) <= FAR)


def summary(scores: pd.DataFrame, tau: float) -> dict:
    """Acceptance of genuine comparisons and FMR of impostor comparisons at tau, mean genuine score, EER."""
    genuine = scores.loc[scores["label"] == 1, "score"].to_numpy()
    impostor = scores.loc[scores["label"] == 0, "score"].to_numpy()
    return {"accept": float(np.mean(genuine > tau)), "fmr": float(np.mean(impostor > tau)),
            "genuine_mean": float(genuine.mean()), "eer": verification.eer(genuine, impostor)}


def select(table: pd.DataFrame, margin: float) -> dict:
    """The selected configuration (params.json 'ccdf.selection')."""
    plain = table[table["config"] == name(PLAIN)].iloc[0]
    ok = table[table["fmr"] <= plain["fmr"] + margin]
    best = ok.sort_values(["accept", "genuine_mean", "t0"], ascending=[False, False, True]).iloc[0]
    keys = ("accept", "fmr", "genuine_mean", "eer")
    return {"init": best["init"], "t0": int(best["t0"]), "project": bool(best["project"]),
            "val": {k: float(best[k]) for k in keys}, "plain_val": {k: float(plain[k]) for k in keys},
            "tau_val": float(best["tau"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="e.g. E12_norm_ccdf_cldice_cv")
    args = parser.parse_args()
    params, folder = experiments.load_params(args.exp), experiments.experiment_dir(args.exp)
    out = folder / "checkpoints"
    assert config.DATA_VERSION == params["data_version"], f"set FPR_DATA_VERSION={params['data_version']}"
    normalized = bool(params["output_normalized"])
    grid, workers = params["ccdf"], params["ccdf"]["workers"]
    meta = eval_sets.load_meta("val")
    pairs = pd.read_csv(eval_sets.pairs_path("val"), dtype={"probe_id": str})
    xyts = gallery_xyts(pairs, out)

    first = meta.drop_duplicates("probe_id")               # one clean print per probe
    clean = {s: eval_sets.normalized_triplet_path("val", s, "clean") for s in first["sample_id"]}
    tau = tau_from_clean(match(clean, meta, pairs, xyts, workers))
    damaged = {s: eval_sets.normalized_triplet_path("val", s, "degraded") for s in meta["sample_id"]}
    rows = [{"config": "damaged", **summary(match(damaged, meta, pairs, xyts, workers), tau), "tau": tau}]
    print(f"tau {tau:g}; damaged {rows[0]}", flush=True)

    degraded_path = eval_sets.normalized_triplet_path if normalized else eval_sets.triplet_path
    inputs = {s: (image_io.load_gray(degraded_path("val", s, "degraded")),
                  image_io.load_mask(eval_sets.triplet_path("val", s, "mask"))) for s in meta["sample_id"]}
    method = experiments.build_reconstructor(args.exp, config=PLAIN)
    for cfg in configurations(grid):
        method.config, start = cfg, time.time()
        sweep_dir = out / "sweep_val" / name(cfg)
        images = {}
        for s, (image, mask) in inputs.items():
            images[s] = sweep_dir / f"{s}.png"
            repaired = method.repair_normalized(image, mask) if normalized else normalize(method.repair(image, mask))
            image_io.save_png(images[s], repaired)          # scored as evaluate.py: normalised
        rows.append({"config": name(cfg), **cfg, **summary(match(images, meta, pairs, xyts, workers), tau),
                     "tau": tau, "seconds": round(time.time() - start)})
        pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)
        print(rows[-1], flush=True)
    selected = select(pd.DataFrame(rows[1:]), grid["fmr_margin"])
    (folder / params["inference"]["selected"]).write_text(json.dumps(selected, indent=2) + "\n")
    print("selected", selected, flush=True)


if __name__ == "__main__":
    main()
