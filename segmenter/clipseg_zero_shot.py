"""Zero-shot test of an open-vocabulary segmenter: CLIPSeg (Lüddecke & Ecker, CVPR 2022) with text prompts.

Usage (needs torch >= 2.6 + transformers, in a separate environment):
    python segmenter/clipseg_zero_shot.py --data v1 --split val
Damage = max over the prompts in params.json of CLIPSeg's 352 px probability map, resized to the
print, thresholded at each value in params.json (choosing the best one afterwards is an oracle).
Writes segmenter/results/<data>_<split>_clipseg_counts.csv, read by evaluate_zero_shot.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from fpr.data import image_io
from segmenter import data, metrics

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())


@torch.no_grad()
def damage_probability(model, processor, image: np.ndarray, prompts: list[str]) -> np.ndarray:
    """uint8 (H, W) print -> float (H, W): highest prompt probability per pixel."""
    rgb = Image.fromarray(image).convert("RGB")
    inputs = processor(text=prompts, images=[rgb] * len(prompts), padding=True, return_tensors="pt")
    logits = model(**inputs).logits.view(len(prompts), 1, 352, 352)
    probability = torch.sigmoid(F.interpolate(logits, size=image.shape, mode="bilinear", align_corners=False))
    return probability.max(0).values[0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--data", required=True, choices=sorted(data.ROOTS), help="eval-set version")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    settings = PARAMS["zero_shot"]
    processor = CLIPSegProcessor.from_pretrained(settings["clipseg"])
    model = CLIPSegForImageSegmentation.from_pretrained(settings["clipseg"]).eval()

    totals = {}
    meta = data.load_meta(args.data, args.split).head(args.limit)
    for row in tqdm(meta.itertuples(), total=len(meta), mininterval=10):
        degraded = image_io.load_gray(data.probe_path(args.data, args.split, row.probe_id, "degraded"))
        truth = image_io.load_mask(data.probe_path(args.data, args.split, row.probe_id, "mask"))
        probability = damage_probability(model, processor, degraded, settings["clipseg_prompts"])
        for threshold in settings["clipseg_thresholds"]:
            counts = metrics.confusion(probability > threshold, truth)
            for group in (row.contaminant, "all"):
                total = totals.setdefault((f"clipseg_t{threshold}", group), dict.fromkeys(counts, 0))
                for name, value in counts.items():
                    total[name] += value

    out = HERE / "results" / f"{args.data}_{args.split}_clipseg_counts.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame([{"method": m, "group": g, **c} for (m, g), c in totals.items()]).to_csv(out, index=False)
    print(f"wrote {out} ({len(meta)} probes)")


if __name__ == "__main__":
    main()
