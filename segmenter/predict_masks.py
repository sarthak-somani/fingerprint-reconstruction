"""Predict the damage mask of every probe of a split with the fine-tuned segmenter, and score it.

Usage: python segmenter/predict_masks.py --data v2 --split val      (after segmenter/train.py)
Writes segmenter/masks/<data>/<split>/<probe_id>_mask.png (white = damage, as the true masks) and
segmenter/results/<data>_<split>_finetuned.{csv,md}: pooled IoU / Dice / precision / recall per contaminant.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import pandas as pd
import torch
from tqdm import tqdm

from fpr.data import data_md, image_io
from segmenter import data, metrics
from segmenter.predict import DamageSegmenter
from segmenter.segformer import Segformer

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())


def load_segmenter(checkpoint: Path, threshold: float) -> DamageSegmenter:
    """The fine-tuned binary SegFormer-B0 from a train.py checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Segformer(num_labels=PARAMS["model"]["num_labels"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    return DamageSegmenter(model.to(device), threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, choices=sorted(data.ROOTS))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=HERE / PARAMS["model"]["checkpoint"])
    parser.add_argument("--threshold", type=float, default=PARAMS["model"]["threshold"])
    args = parser.parse_args()
    segmenter = load_segmenter(args.checkpoint, args.threshold)
    out_dir = HERE / "masks" / args.data / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {}
    meta = data.load_meta(args.data, args.split)
    for row in tqdm(meta.itertuples(), total=len(meta), mininterval=10):
        degraded = image_io.load_gray(data.probe_path(args.data, args.split, row.probe_id, "degraded"))
        truth = image_io.load_mask(data.probe_path(args.data, args.split, row.probe_id, "mask"))
        pred = segmenter.predict(degraded)
        image_io.save_png(out_dir / f"{row.probe_id}_mask.png", pred)
        for group in (row.contaminant, "all"):
            total = totals.setdefault(group, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            for name, value in metrics.confusion(pred, truth).items():
                total[name] += value

    table = pd.DataFrame([{"group": g, **{k: round(v, 3) for k, v in metrics.scores(c).items()}}
                          for g, c in sorted(totals.items())])
    out = HERE / "results" / f"{args.data}_{args.split}_finetuned"
    table.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(f"# Fine-tuned SegFormer-B0, {args.data} {args.split} ({len(meta)} probes), "
                                      f"threshold {args.threshold}\n\n{data_md.table_to_markdown(table)}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
