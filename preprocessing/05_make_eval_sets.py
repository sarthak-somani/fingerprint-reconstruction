"""Step 05: damage each val/test L print -> {split}/probes/ + meta.csv (one row per damaged sample).

Usage: python preprocessing/05_make_eval_sets.py [--severity 0.4 0.8]
       FPR_DATA_VERSION=v3 python preprocessing/05_make_eval_sets.py    # -> preprocessing/data/v3/
v2 (config.DAMAGE_COPIES = 1): each probe damaged once; contaminants are assigned round-robin over a
seeded shuffle of the probes, so each type gets an equal share. sample_id = probe_id.
v3 (3 copies): each probe damaged 3 times, with 3 different contaminants drawn at random per probe.
sample_id = '{probe_id}_{contaminant}'. Severity is uniform in --severity (default
config.EVAL_SEVERITY_RANGE) for every copy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from fpr.data import data_md, eval_sets, foreground, image_io, nist
from fpr.data.degradations import apply_degradation, CONTAMINANTS


def damage_copies(probe, seed: int, copies: int, contaminant: str) -> list[tuple[str, str, np.random.Generator]]:
    """(sample_id, contaminant, rng) for each damaged copy of one probe.

    copies = 1 (v2): the round-robin contaminant and the probe's rng, exactly as before v3.
    copies > 1 (v3): `copies` different contaminants drawn with the probe's rng, one rng per copy.
    Seeded per probe: re-running gives identical damage, whatever the order.
    """
    rng = np.random.default_rng([seed, int(probe.participant_id), probe.finger])
    if copies == 1:
        return [(probe.probe_id, contaminant, rng)]
    names = rng.choice(CONTAMINANTS, size=copies, replace=False)
    return [(f"{probe.probe_id}_{name}", str(name),
             np.random.default_rng([seed, int(probe.participant_id), probe.finger, j + 1]))
            for j, name in enumerate(names)]


def make_split_set(index: pd.DataFrame, split_name: str, seed: int,
                   severity_range: tuple[float, float], copies: int) -> pd.DataFrame:
    """Write the triplets of one split and return its meta table."""
    probes = eval_sets.probe_table(index, split_name)
    order = np.random.default_rng([seed, len(split_name)]).permutation(len(probes))
    contaminant_of = {probes.loc[k, "probe_id"]: CONTAMINANTS[i % len(CONTAMINANTS)]
                      for i, k in enumerate(order)}

    rows = []
    for probe in tqdm(probes.itertuples(), total=len(probes), desc=split_name):
        clean = image_io.load_gray(nist.raw_path(probe.probe_path))
        fg = foreground.foreground_mask(clean)
        for sample_id, name, rng in damage_copies(probe, seed, copies, contaminant_of[probe.probe_id]):
            severity = float(rng.uniform(*severity_range))
            degraded, mask = apply_degradation(clean, fg, name, severity, rng)
            for kind, image in (("clean", clean), ("degraded", degraded), ("mask", mask)):
                image_io.save_png(eval_sets.triplet_path(split_name, sample_id, kind), image)
            rows.append({"sample_id": sample_id, "probe_id": probe.probe_id, "participant_id": probe.participant_id,
                         "finger": probe.finger, "contaminant": name,
                         "severity": round(severity, 3),
                         "damaged_fraction": round(float(mask.sum() / max(fg.sum(), 1)), 4),
                         "probe_path": probe.probe_path, "gallery_path": probe.gallery_path})
    meta = pd.DataFrame(rows)
    meta.to_csv(eval_sets.meta_path(split_name), index=False)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    parser.add_argument("--severity", type=float, nargs=2, default=config.EVAL_SEVERITY_RANGE,
                        metavar=("LOW", "HIGH"), help="severity drawn uniformly in [LOW, HIGH]")
    args = parser.parse_args()
    index = nist.load_index()

    summaries = []
    for split_name in config.EVAL_SPLITS:
        meta = make_split_set(index, split_name, args.seed, tuple(args.severity), config.DAMAGE_COPIES)
        summary = (meta.groupby("contaminant")
                   .agg(samples=("sample_id", "size"),
                        mean_severity=("severity", lambda s: round(s.mean(), 2)),
                        mean_damaged_fraction=("damaged_fraction", lambda d: round(d.mean(), 3)))
                   .reset_index())
        summary.insert(0, "split", split_name)
        summaries.append(summary)
        print(f"wrote {len(meta)} samples of {meta['probe_id'].nunique()} probes to {eval_sets.split_dir(split_name)}")
    table = pd.concat(summaries)
    times = "once" if config.DAMAGE_COPIES == 1 else f"{config.DAMAGE_COPIES} times (different contaminants)"
    data_md.write_section(data_md.versioned("eval_sets"), f"## Fixed val/test sets, data {config.DATA_VERSION} (script 05)\n\n"
                          + f"Full L plain prints damaged {times} (seed {args.seed}), severity "
                          + f"uniform in {tuple(args.severity)}. damaged_fraction = "
                          + "damaged pixels / foreground pixels.\n\n"
                          + data_md.table_to_markdown(table))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
