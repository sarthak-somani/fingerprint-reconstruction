"""Zero-shot test: can a released segmenter find the damage without fine-tuning?

Usage: python segmenter/evaluate_zero_shot.py --data v1 --split val
Scores, against the true damage masks of the split's probes (pixels of the whole print):
- floors: 'empty' (no damage) and 'foreground' (the whole finger is damaged);
- SegFormer-B0 ADE20K, zero-shot: for each of its 150 classes, the class mask as the damage mask,
  pooled over the split; the best class is an *oracle* upper bound (chosen on the answers);
- CLIPSeg rows, if clipseg_zero_shot.py has written its predictions (it needs transformers).
Writes segmenter/results/<data>_<split>_zero_shot.csv and .md (pooled per contaminant).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fpr.data import data_md, foreground, image_io
from segmenter import data, metrics
from segmenter.predict import full_logits
from segmenter.segformer import load_released

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())


def add(table: dict, key: tuple, counts: dict) -> None:
    """Accumulate confusion counts under `key`."""
    total = table.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for name, value in counts.items():
        total[name] += value


def class_counts(classes: torch.Tensor, truth: torch.Tensor, n: int) -> np.ndarray:
    """(n, 2) per-class pixel counts: [inside the true damage, outside it] for each class index."""
    inside = torch.bincount(classes[truth], minlength=n)
    outside = torch.bincount(classes[~truth], minlength=n)
    return torch.stack([inside, outside], 1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--data", required=True, choices=sorted(data.ROOTS), help="eval-set version")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_released(PARAMS["zero_shot"]["segformer"], num_labels=150).to(device).eval()
    meta = data.load_meta(args.data, args.split).head(args.limit)

    pooled, per_class = {}, {}
    for row in tqdm(meta.itertuples(), total=len(meta), mininterval=10):
        degraded = image_io.load_gray(data.probe_path(args.data, args.split, row.probe_id, "degraded"))
        truth = image_io.load_mask(data.probe_path(args.data, args.split, row.probe_id, "mask"))
        for name, pred in (("empty", np.zeros_like(truth)), ("foreground", foreground.foreground_mask(degraded))):
            for group in (row.contaminant, "all"):
                add(pooled, (name, group), metrics.confusion(pred, truth))
        classes = full_logits(model, degraded).argmax(0)
        counts = class_counts(classes.flatten(), torch.from_numpy(truth).flatten().to(device), 150)
        for group in (row.contaminant, "all"):
            per_class[group] = per_class.get(group, 0) + counts

    damage = per_class["all"][:, 0].sum()                         # true damaged pixels in the split
    iou_all = per_class["all"][:, 0] / (damage + per_class["all"][:, 1])
    best = int(np.argmax(iou_all))                                # oracle: the class that best matches damage
    for group, counts in per_class.items():
        tp, fp = int(counts[best, 0]), int(counts[best, 1])
        total = int(counts[:, 0].sum())
        add(pooled, (f"segformer_ade_oracle_class_{best}", group), {"tp": tp, "fp": fp, "fn": total - tp, "tn": 0})

    clipseg_csv = HERE / "results" / f"{args.data}_{args.split}_clipseg_counts.csv"
    if clipseg_csv.exists():
        for r in pd.read_csv(clipseg_csv).itertuples():
            add(pooled, (r.method, r.group), {"tp": r.tp, "fp": r.fp, "fn": r.fn, "tn": r.tn})

    rows = [{"method": m, "group": g, **{k: round(v, 3) for k, v in metrics.scores(c).items()}}
            for (m, g), c in pooled.items()]
    table = pd.DataFrame(rows).sort_values(["method", "group"])
    out = HERE / "results" / f"{args.data}_{args.split}_zero_shot"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    dice = table.pivot(index="method", columns="group", values="dice").reset_index()
    out.with_suffix(".md").write_text(f"# Zero-shot damage segmentation, {args.data} {args.split} "
                                      f"({len(meta)} probes): pooled Dice\n\n{data_md.table_to_markdown(dice)}\n")
    print(dice.to_string(index=False))


if __name__ == "__main__":
    main()
