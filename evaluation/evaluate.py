#!/usr/bin/env python
"""Evaluate fingerprint reconstructions against their originals, per degradation type.

Usage:  python evaluate.py ORIGINAL_DIR DEGRADED_DIR RECON_DIR [MASK_DIR] [--gallery GALLERY_DIR]
                           [--benchmark BENCHMARK_JSON]

Writes results/<timestamp>/ (report.md, plots/, CSVs) next to this file. Everything that
depends only on original + degraded images is cached in results/degraded_cache/ and reused.
Every image is first normalised to MSU's cartoon-texture representation (temporary, see
NORMALIZE_SIGMA). Headline: three scores with bootstrap CIs and significance tests:
1 identity recovery (net rescue rate at the threshold tau), 2 false-identity gate (FMR increase
at tau), 3 ridge fidelity in the damaged region; plus cost from experiments/benchmark.py.
Diagnostics: PSNR, MS-SSIM, LPIPS, ridge fidelity (full image); MINDTCT minutiae quality; NFIQ 2;
bozorth3 score against the original; EER, FAR/FRR. Report section 0 defines every metric.
See README.md for the protocol, the inputs and the third-party builds.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import to_rgb
from PIL import Image
from scipy import ndimage, stats
from tqdm import tqdm

# ================================================================ settings
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))   # repo root, for the MSU preprocessing port (NORMALIZE_SIGMA)
THIRD_PARTY = EVAL_DIR / "third_party"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = RESULTS_DIR / "degraded_cache"
TOOL_DIRS = [THIRD_PARTY / "nbis" / "bin", THIRD_PARTY / "NFIQ2" / "bin"]
LPIPS_REPO = THIRD_PARTY / "PerceptualSimilarity"

TYPES = ["pressure", "scar", "oil", "sanitizer", "mehendi", "lotion"]   # lotion: 6th type
TYPE_ALIASES = {t: t for t in TYPES} | {"mehndi": "mehendi", "mehandi": "mehendi"}
IMAGE_EXTS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
STANDARD_PPI = 500

# TEMPORARY: every image (original, degraded, reconstruction, gallery) is replaced by
# the texture part of MSU-LatentAFIS's cartoon-texture decomposition with this sigma before anything is
# measured, so all methods are scored in one ridge representation. It moves into the whole pipeline later;
# None switches it off.
NORMALIZE_SIGMA = 2.5

PSNR_CAP_DB = 100.0                  # PSNR of a pixel-identical region (instead of inf)
MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)   # Wang et al. 2003
MSSSIM_WINDOW, MSSSIM_SIGMA = 11, 1.5
LPIPS_NET = "alex"
RIDGE_BAND_SIGMAS = (1.0, 4.0)       # ridge fidelity: difference-of-Gaussians band-pass (px)
RIDGE_WINDOW_SIGMA = 8.0             # ridge fidelity: Gaussian window of the local correlation (px)

DIFF_THRESH = 8                      # derived mask: |orig - deg| above this many grey levels
MASK_CLOSE_PX, MASK_MIN_AREA_PX, MASK_DILATE_PX = 5, 30, 4
REGION_DILATE_PX = 8                 # "reconstructed region" for minutiae = mask grown by this

N_IMPOSTORS = 50                     # random other references (gallery prints or originals) per probe
SEED = 643
FIXED_THRESHOLD = 40                 # bozorth3 score; accept when score > threshold
FAR_TARGETS = (0.01, 0.001)          # thresholds set on the original images' impostor scores
TAU = "far1"                         # operating point of scores 1 and 2: the FAR = 1% threshold
FMR_GATE = 0.005                     # score 2 passes when the upper 95% CI of the FMR increase is below this
N_BOOT = 10000                       # bootstrap resamples (probes) for every CI
SAMPLE_METRIC = "ridge_region"       # per-image Δ that picks the sample grid (low / median / high gain)
SAMPLE_PCTS = (10, 50, 90)           # Δ percentiles (within a type) of the three samples per type
SAMPLE_DAMAGE_BAND = (25, 75)        # samples are drawn from this damage-percentile band ("median damage")
BOZORTH_BATCH = 500
NFIQ2_BATCH = 20                     # images per nfiq2 call (the model loads once per call)
NFIQ2_MODEL = "nist_plain_tir-ink.txt"
N_WORKERS = os.cpu_count() or 4
EPS = 1e-6

# name -> higher_is_better. Metrics with an _orig column (the clean print's own value) also show it.
PER_IMAGE_METRICS = {
    "ridge_region": True, "ridge_full": True,
    "psnr_full": True, "psnr_region": True,
    "msssim_full": True, "msssim_region": True,
    "lpips_full": False, "lpips_region": False,
    "mq_mean_region": True, "mq_median_region": True, "mq_mean_full": True, "mq_median_full": True,
    "nfiq2": True,
    "orig_match_score": True,     # probe vs its own original (the original vs itself is the most it can reach)
    "match_score": True,          # probe vs gallery print (gallery mode)
}
LABELS = {
    "ridge_region": "Ridge fidelity region", "ridge_full": "Ridge fidelity full",
    "psnr_full": "PSNR full (dB)", "psnr_region": "PSNR region (dB)",
    "msssim_full": "MS-SSIM full", "msssim_region": "MS-SSIM region",
    "lpips_full": "LPIPS full", "lpips_region": "LPIPS region",
    "mq_mean_region": "MINDTCT quality mean", "mq_median_region": "MINDTCT quality median",
    "mq_mean_full": "MINDTCT quality mean (full)", "mq_median_full": "MINDTCT quality median (full)",
    "nfiq2": "NFIQ 2", "orig_match_score": "Original-match score", "match_score": "Gallery match score",
    "eer": "EER", "far_t40": "FAR @40", "frr_t40": "FRR @40",
    "far_far1": "FAR @FAR1%", "frr_far1": "FRR @FAR1%", "far_far01": "FAR @FAR0.1%", "frr_far01": "FRR @FAR0.1%",
}

# Fixed colour per degradation type (validated categorical order, slots 1-6); never by rank.
TYPE_COLORS = {"pressure": "#2a78d6", "scar": "#eb6834", "oil": "#1baf7a",
               "sanitizer": "#eda100", "mehendi": "#e87ba4", "lotion": "#008300"}
ORIGINAL_COLOR = "#8f8e88"
ORIG_MINUTIA_COLOR, REC_MINUTIA_COLOR = "#6250d6", "#e34948"
INK, MUTED_INK, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI = 180


# ================================================================ inputs
def load_gray(path: Path) -> np.ndarray:
    """uint8 (H, W) grayscale image."""
    return np.array(Image.open(path).convert("L"))


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def text_sha1(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def split_type(stem: str) -> tuple[str, str] | None:
    """'00012_mehndi' -> ('00012', 'mehendi'); ids may contain '_', so match the type as a suffix."""
    for alias in sorted(TYPE_ALIASES, key=len, reverse=True):
        if stem.lower().endswith("_" + alias) and len(stem) > len(alias) + 1:
            return stem[:-len(alias) - 1], TYPE_ALIASES[alias]
    return None


def index_dir(folder: Path | None, suffix: str) -> tuple[dict, list]:
    """{(id, type): path} for files named <id>_<type><suffix>.<ext>; plus unparsed file names."""
    found, unparsed = {}, []
    if folder is None or not folder.is_dir():
        return found, unparsed
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = path.stem
        if suffix:
            if not stem.lower().endswith(suffix):
                continue
            stem = stem[:-len(suffix)]
        elif stem.lower().endswith(("_rec", "_mask")):
            continue
        parsed = split_type(stem)
        if parsed is None:
            unparsed.append(path.name)
        else:
            found[parsed] = path
    return found, unparsed


def find_samples(orig_dir: Path, deg_dir: Path, rec_dir: Path, mask_dir: Path | None):
    """Pair every degraded image with its original, reconstruction and (optional) mask.

    Output: (samples, originals, skipped). samples: list of dicts with key, id, type and paths;
    originals: {id: path} of every original (the impostor gallery); skipped: [(file, reason)].
    """
    originals = {p.stem: p for p in sorted(orig_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    degraded, unparsed = index_dir(deg_dir, "")
    recons, _ = index_dir(rec_dir, "_rec")
    masks, _ = index_dir(mask_dir, "_mask") if mask_dir else ({}, [])
    skipped = [(name, "degradation type not recognised") for name in unparsed]
    samples = []
    for (fid, dtype), deg_path in sorted(degraded.items()):
        if fid not in originals:
            skipped.append((deg_path.name, "no original"))
            continue
        if (fid, dtype) not in recons:
            skipped.append((deg_path.name, "no reconstruction"))
            continue
        samples.append({"key": f"{fid}_{dtype}", "id": fid, "type": dtype,
                        "orig_path": originals[fid], "deg_path": deg_path,
                        "rec_path": recons[(fid, dtype)], "mask_path": masks.get((fid, dtype))})
    return samples, originals, skipped


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def derive_mask(orig: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Damage mask from |orig - deg| > DIFF_THRESH, closed, small blobs removed, dilated."""
    diff = np.abs(orig.astype(np.int16) - deg.astype(np.int16)) > DIFF_THRESH
    closed = ndimage.binary_closing(diff, structure=disk(MASK_CLOSE_PX))
    labels, n = ndimage.label(closed)
    if n == 0:
        return closed
    sizes = ndimage.sum(closed, labels, range(1, n + 1))
    keep = np.isin(labels, 1 + np.flatnonzero(sizes >= MASK_MIN_AREA_PX))
    return ndimage.binary_dilation(keep, structure=disk(MASK_DILATE_PX))


def load_sample_images(s: dict) -> tuple:
    """(orig, deg, rec, mask, mask_source) for one sample; raises ValueError on size mismatch."""
    orig, deg, rec = load_gray(s["orig_path"]), load_gray(s["deg_path"]), load_gray(s["rec_path"])
    if not (orig.shape == deg.shape == rec.shape):
        raise ValueError(f"sizes differ: orig {orig.shape}, deg {deg.shape}, rec {rec.shape}")
    if s["mask_path"] is not None:
        mask = load_gray(s["mask_path"]) > 127
        if mask.shape == orig.shape:
            return orig, deg, rec, mask, "given"
    return orig, deg, rec, derive_mask(orig, deg), "derived"


def normalize_image(src: Path, dst: Path) -> None:
    """MSU cartoon-texture texture of `src` (uint8, background grey 128) -> `dst` PNG at 500 ppi."""
    from fpr.data.normalization import normalize          # the same function E12 applies to its inputs
    texture = normalize(load_gray(src), NORMALIZE_SIGMA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(texture).save(dst, dpi=(STANDARD_PPI, STANDARD_PPI))


def normalize_inputs(samples: list[dict], originals: dict, refs: dict, run_dir: Path,
                     rec_normalized: bool = False) -> tuple[dict, dict]:
    """Point every image path at its normalised copy (temporary, see NORMALIZE_SIGMA).

    Originals, gallery prints and degraded images are content-addressed in the cache (by the raw
    file's hash); reconstructions go to the run folder, unless rec_normalized (the method already
    outputs normalised prints, e.g. E12): then they are measured as stored.
    Returns (originals, refs) with the new paths.
    """
    cache = CACHE_DIR / "normalized" / f"sigma{NORMALIZE_SIGMA:g}"
    copies = {p: cache / f"{file_sha1(p)}.png"
              for p in set(originals.values()) | set(refs.values()) | {s["deg_path"] for s in samples}}
    copies |= {s["rec_path"]: s["rec_path"] if rec_normalized else run_dir / "normalized" / f"{s['key']}_rec.png"
               for s in samples}
    jobs = [(src, dst) for src, dst in copies.items() if not dst.exists()]
    if jobs:
        with ProcessPoolExecutor(N_WORKERS) as pool:
            list(tqdm(pool.map(normalize_image, *zip(*jobs), chunksize=4), total=len(jobs),
                      desc="MSU normalisation", leave=False))
    for s in samples:
        s["deg_path"], s["rec_path"] = copies[s["deg_path"]], copies[s["rec_path"]]
        s["orig_path"] = copies[s["orig_path"]]
    return {k: copies[p] for k, p in originals.items()}, {k: copies[p] for k, p in refs.items()}


# ================================================================ image metrics
def psnr(ref: np.ndarray, img: np.ndarray, region: np.ndarray) -> float:
    """PSNR in dB (peak 255) over `region` pixels, capped at PSNR_CAP_DB."""
    err = (img.astype(np.float64) - ref.astype(np.float64))[region]
    if err.size == 0:
        return np.nan
    mse = np.mean(err ** 2)
    return PSNR_CAP_DB if mse == 0 else float(min(PSNR_CAP_DB, 10 * np.log10(255.0 ** 2 / mse)))


def _gaussian_blur(x: np.ndarray) -> np.ndarray:
    half = MSSSIM_WINDOW // 2
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / MSSSIM_SIGMA) ** 2)
    kernel /= kernel.sum()
    x = ndimage.correlate1d(x, kernel, axis=0, mode="reflect")
    return ndimage.correlate1d(x, kernel, axis=1, mode="reflect")


def _ssim_maps(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Luminance and contrast-structure maps (float64, same shape as input)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _gaussian_blur(a), _gaussian_blur(b)
    var_a = _gaussian_blur(a * a) - mu_a ** 2
    var_b = _gaussian_blur(b * b) - mu_b ** 2
    cov = _gaussian_blur(a * b) - mu_a * mu_b
    lum = (2 * mu_a * mu_b + c1) / (mu_a ** 2 + mu_b ** 2 + c1)
    cs = (2 * cov + c2) / (var_a + var_b + c2)
    return lum, cs


def _pool2(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    return x[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def ms_ssim(ref: np.ndarray, img: np.ndarray, regions: list[np.ndarray]) -> list[float]:
    """MS-SSIM (5 scales, Wang et al. 2003) averaged over each region in `regions`.

    At each scale the maps are averaged over the region, downsampled with the image (a cell
    counts if any of its pixels is in the region). Negative cs values are clipped to 0. Scales
    where a region has vanished are dropped and the remaining weights renormalised.
    """
    a, b = ref.astype(np.float64), img.astype(np.float64)
    masks = [r.astype(np.float64) for r in regions]
    values = [[] for _ in regions]            # (value, weight) per region
    for j, weight in enumerate(MSSSIM_WEIGHTS):
        lum, cs = _ssim_maps(a, b)
        last = j == len(MSSSIM_WEIGHTS) - 1
        for k, m in enumerate(masks):
            sel = m > 0
            if sel.any():
                v = (lum * cs)[sel].mean() if last else cs[sel].mean()
                values[k].append((max(float(v), 0.0), weight))
        if not last:
            a, b, masks = _pool2(a), _pool2(b), [_pool2(m) for m in masks]
            if min(a.shape) < 2:
                break
    out = []
    for vals in values:
        if not vals:
            out.append(np.nan)
            continue
        total = sum(w for _, w in vals)
        out.append(float(np.prod([v ** (w / total) for v, w in vals])))
    return out


_LPIPS: dict = {}


def lpips_models() -> dict:
    """Official LPIPS (richzhang/PerceptualSimilarity clone), AlexNet v0.1, mean and spatial."""
    if not _LPIPS:
        sys.path.insert(0, str(LPIPS_REPO))
        import lpips                                         # from the clone, not pip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _LPIPS["device"] = device
            _LPIPS["mean"] = lpips.LPIPS(net=LPIPS_NET, verbose=False).to(device).eval()
            _LPIPS["spatial"] = lpips.LPIPS(net=LPIPS_NET, spatial=True, verbose=False).to(device).eval()
    return _LPIPS


def lpips_full_region(ref: np.ndarray, img: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    """LPIPS of the whole image, and the spatial LPIPS map averaged over `region`."""
    models = lpips_models()

    def to_tensor(x):
        t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)[None, None].repeat(1, 3, 1, 1)
        return t.to(models["device"])

    with torch.no_grad():
        ta, tb = to_tensor(ref), to_tensor(img)
        full = float(models["mean"](ta, tb).item())
        if not region.any():
            return full, np.nan
        spatial = models["spatial"](ta, tb)[0, 0].cpu().numpy()
    return full, float(spatial[region].mean())


def ridge_band(x: np.ndarray) -> np.ndarray:
    """Difference-of-Gaussians band-pass: ridge-scale structure without brightness or fine noise."""
    x = x.astype(np.float64)
    return ndimage.gaussian_filter(x, RIDGE_BAND_SIGMAS[0]) - ndimage.gaussian_filter(x, RIDGE_BAND_SIGMAS[1])


def ridge_fidelity(ref: np.ndarray, img: np.ndarray, regions: list[np.ndarray]) -> list[float]:
    """Mean local correlation of the band-passed ridge patterns of ref and img over each region.

    1 = the same ridges (up to local brightness and contrast), 0 = unrelated, < 0 = ridges and
    valleys swapped. The correlation is taken in a Gaussian window of RIDGE_WINDOW_SIGMA px.
    """
    a, b = ridge_band(ref), ridge_band(img)
    window = lambda z: ndimage.gaussian_filter(z, RIDGE_WINDOW_SIGMA)
    mu_a, mu_b = window(a), window(b)
    var_a, var_b = window(a * a) - mu_a ** 2, window(b * b) - mu_b ** 2
    corr = (window(a * b) - mu_a * mu_b) / np.sqrt(np.maximum(var_a, EPS) * np.maximum(var_b, EPS))
    corr = np.clip(corr, -1, 1)
    return [float(corr[r].mean()) if r.any() else np.nan for r in regions]


def image_metrics(ref: np.ndarray, img: np.ndarray, mask: np.ndarray) -> dict:
    full = np.ones_like(mask, dtype=bool)
    ms_full, ms_region = ms_ssim(ref, img, [full, mask])
    lp_full, lp_region = lpips_full_region(ref, img, mask)
    rf_full, rf_region = ridge_fidelity(ref, img, [full, mask])
    return {"ridge_full": rf_full, "ridge_region": rf_region,
            "psnr_full": psnr(ref, img, full), "psnr_region": psnr(ref, img, mask),
            "msssim_full": ms_full, "msssim_region": ms_region,
            "lpips_full": lp_full, "lpips_region": lp_region}


# ================================================================ NIST tools
def find_tool(name: str) -> str | None:
    for folder in TOOL_DIRS:
        if (folder / name).is_file():
            return str(folder / name)
    return shutil.which(name)


def tool_hash(path: str | None) -> str:
    return file_sha1(Path(path))[:12] if path else "none"


def run_mindtct(mindtct: str, image_path: Path, out_root: Path) -> None:
    """mindtct on one image -> out_root.xyt and out_root.min (skipped if both exist)."""
    if out_root.with_suffix(".xyt").exists() and out_root.with_suffix(".min").exists():
        return
    out_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "in.png"
        Image.open(image_path).convert("L").save(png, dpi=(STANDARD_PPI, STANDARD_PPI))
        subprocess.run([mindtct, str(png), str(Path(tmp) / "out")], check=True, capture_output=True)
        for suffix in (".xyt", ".min"):
            shutil.copy(Path(tmp) / f"out{suffix}", out_root.with_suffix(suffix))


# .min line: "   3 :  210, 345 :  7 :  0.42 : BIF : ..." -> x, y, direction (0-31), reliability, type
MIN_LINE = re.compile(r"^\s*\d+\s*:\s*(\d+),\s*(\d+)\s*:\s*(\d+)\s*:\s*([\d.]+)\s*:\s*(RIG|BIF)")


def read_min(path: Path) -> np.ndarray:
    """float (N, 5): x, y (px, top-left origin), direction (deg), reliability 0-1, type (1 = BIF)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        match = MIN_LINE.match(line)
        if match:
            x, y, direction, quality, kind = match.groups()
            rows.append([float(x), float(y), int(direction) * 11.25, float(quality),
                         1.0 if kind == "BIF" else 0.0])
    return np.array(rows, dtype=float).reshape(-1, 5)


def bozorth_scores(bozorth3: str, probe_xyt: Path, gallery_xyts: list[Path]) -> np.ndarray:
    """bozorth3 score of one probe against each gallery file (higher = more similar)."""
    positions = {}
    for k, p in enumerate(gallery_xyts):
        positions.setdefault(str(p), []).append(k)          # the same file may be listed twice
    scores = np.zeros(len(gallery_xyts))
    for start in range(0, len(gallery_xyts), BOZORTH_BATCH):
        batch = list(dict.fromkeys(str(p) for p in gallery_xyts[start:start + BOZORTH_BATCH]))
        result = subprocess.run([bozorth3, "-A", "outfmt=spg", "-p", str(probe_xyt)] + batch,
                                check=True, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            fields = line.split()                         # score probe gallery
            if len(fields) == 3 and fields[2] in positions:
                scores[positions[fields[2]]] = float(fields[0])
    return scores


def nfiq2_model(nfiq2: str) -> str:
    """The NFIQ 2 model info file shipped with the build (share/ next to bin/)."""
    return str(Path(nfiq2).resolve().parent.parent / "share" / NFIQ2_MODEL)


def nfiq2_scores(nfiq2: str, paths: list[Path]) -> list[tuple[float, str]]:
    """NFIQ 2 unified quality score (0-100) and error text per image, from one nfiq2 call.

    Parses the CSV nfiq2 prints (Filename, FingerCode, QualityScore, OptionalError, ...).
    The score is NaN, with NFIQ 2's error, when it refuses an image.
    """
    paths = [os.path.abspath(p) for p in paths]
    result = subprocess.run([nfiq2, "-F", "-m", nfiq2_model(nfiq2)] + paths,
                            capture_output=True, text=True)
    found = {}
    for row in csv.DictReader(io.StringIO(result.stdout)):
        error = row.get("OptionalError", "NA")
        try:
            score = float(row["QualityScore"])
        except (KeyError, TypeError, ValueError):
            score = np.nan
        found[row.get("Filename")] = (score, "" if error in ("NA", "", None) else error)
    message = (result.stderr or result.stdout).strip().splitlines()
    fallback = (np.nan, message[-1] if message else "no output")
    return [found.get(p, fallback) for p in paths]


def nfiq2_applicability(path: Path) -> str:
    """'' if the file meets NFIQ 2's input requirements we can check, else the reason."""
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
        if dpi is not None and abs(float(dpi[0]) - STANDARD_PPI) > 2:
            return f"resolution {float(dpi[0]):.0f} ppi, not {STANDARD_PPI}"
        if im.mode not in ("L", "P", "1", "I;16"):
            return f"not grayscale (mode {im.mode})"
        if min(im.size) < 196:
            return f"too small ({im.size[0]}x{im.size[1]})"
    return ""


def parallel(fn, items: list, desc: str) -> list:
    """Run fn over items in threads (the work is in subprocesses), with a progress bar."""
    if not items:
        return []
    with ThreadPoolExecutor(N_WORKERS) as pool:
        return list(tqdm(pool.map(fn, items), total=len(items), desc=desc, leave=False))


# ================================================================ minutiae
def inside(mnt: np.ndarray, region: np.ndarray) -> np.ndarray:
    cols = np.clip(mnt[:, 0].astype(int), 0, region.shape[1] - 1)
    rows = np.clip(mnt[:, 1].astype(int), 0, region.shape[0] - 1)
    return region[rows, cols]


def quality_stats(mnt: np.ndarray, region: np.ndarray) -> tuple[float, float, int]:
    """Mean and median MINDTCT reliability of minutiae inside `region`, and their count."""
    q = mnt[inside(mnt, region), 3]
    return (float(q.mean()), float(np.median(q)), len(q)) if len(q) else (np.nan, np.nan, 0)


def minutiae_metrics(mnt: np.ndarray, region: np.ndarray) -> dict:
    """MINDTCT quality (mean and median reliability) and count of minutiae, in the region and full image."""
    out = {}
    for name, reg in (("region", region), ("full", np.ones_like(region, dtype=bool))):
        mean, median, n = quality_stats(mnt, reg)
        out |= {f"mq_mean_{name}": mean, f"mq_median_{name}": median, f"n_minutiae_{name}": n}
    return out


# ================================================================ matching statistics
def far_frr(genuine: np.ndarray, impostor: np.ndarray, t: float) -> tuple[float, float]:
    """Accept when score > t. FAR = impostors accepted, FRR = genuines rejected."""
    far = float(np.mean(impostor > t)) if len(impostor) else np.nan
    frr = float(np.mean(genuine <= t)) if len(genuine) else np.nan
    return far, frr


def roc(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    """FAR and FRR at every distinct score as threshold (ascending), plus one below all scores."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.concatenate([[thresholds[0] - 1], thresholds])
    g, i = np.sort(genuine), np.sort(impostor)
    frr = np.searchsorted(g, thresholds, side="right") / len(g)
    far = 1 - np.searchsorted(i, thresholds, side="right") / len(i)
    return {"threshold": thresholds, "far": far, "frr": frr}


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Equal error rate, linearly interpolated between the operating points where FAR - FRR changes sign."""
    if len(genuine) == 0 or len(impostor) == 0:
        return np.nan
    p = roc(genuine, impostor)
    d = p["far"] - p["frr"]                    # decreasing as the threshold rises
    exact = np.flatnonzero(d == 0)
    if len(exact):
        return float(p["far"][exact[0]])
    k = np.flatnonzero((d[:-1] > 0) & (d[1:] < 0))
    if not len(k):
        return float(np.min(np.maximum(p["far"], p["frr"])))
    k = k[0]
    alpha = d[k] / (d[k] - d[k + 1])
    return float(p["far"][k] + alpha * (p["far"][k + 1] - p["far"][k]))


def threshold_at_far(impostor: np.ndarray, target: float) -> float:
    """Smallest score threshold t whose FAR (impostor > t) is <= target."""
    for t in np.unique(impostor):
        if np.mean(impostor > t) <= target:
            return float(t)
    return float(impostor.max())


# ================================================================ headline scores and statistics
# Statistical unit = the finger (sample id `id`), not the sample: several damaged copies of one finger (v3
# data: 3) share its original and gallery print, so they are not independent. Bootstraps resample fingers
# with all their samples and tests compare per-finger values. With one sample per finger (v2) every
# function below gives exactly the sample-level result (percentile bootstrap, exact McNemar, Wilcoxon).
def boot_index(n: int, seed_text: str) -> np.ndarray:
    """(N_BOOT, n) indices drawn with replacement; fixed per seed_text, so reruns agree."""
    rng = np.random.default_rng([SEED, int(hashlib.sha1(seed_text.encode()).hexdigest()[:8], 16)])
    return rng.integers(0, n, size=(N_BOOT, n))


def cluster_sums(x: np.ndarray, ids) -> tuple[np.ndarray, np.ndarray]:
    """(sum of x, number of samples) per finger, fingers in order of first appearance."""
    codes, uniques = pd.factorize(np.asarray(ids))
    return (np.bincount(codes, weights=np.asarray(x, float), minlength=len(uniques)),
            np.bincount(codes, minlength=len(uniques)).astype(float))


def boot_means(x: np.ndarray, ids, seed_text: str) -> np.ndarray:
    """N_BOOT bootstrap means of per-sample values x, resampling fingers (with all their samples)."""
    sums, counts = cluster_sums(x, ids)
    index = boot_index(len(sums), seed_text)
    return sums[index].sum(axis=1) / counts[index].sum(axis=1)


def boot_ci(x: np.ndarray, ids, seed_text: str) -> tuple[float, float]:
    """95% percentile cluster-bootstrap CI of the mean of per-sample values x."""
    means = boot_means(x, ids, seed_text)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_p(d: np.ndarray, ids) -> float:
    """Exact two-sided sign-flip test of paired 0/1 changes d in {-1, 0, 1} (e.g. rescued +1, broken -1).

    Statistic S = sum of d. Under H0 (repair rescues and breaks equally often) each finger's total
    s_f = sum of its d is equally likely to be +s_f or -s_f, independently across fingers; p = P(|S*| >= |S|),
    computed exactly by convolution. With one sample per finger this is the exact McNemar test.
    """
    sums = np.rint(cluster_sums(d, ids)[0]).astype(int)
    observed, steps = abs(int(sums.sum())), np.abs(sums[sums != 0])
    if not len(steps):
        return 1.0
    dist = np.ones(1)                                   # distribution of S*, index i = value i - total so far
    for s in steps:
        new = np.zeros(len(dist) + 2 * s)
        new[:len(dist)] += 0.5 * dist                   # -s
        new[2 * s:] += 0.5 * dist                       # +s
        dist = new
    values = np.arange(len(dist)) - steps.sum()
    return float(min(1.0, dist[np.abs(values) >= observed].sum()))


def wilcoxon_p(d: np.ndarray, ids=None) -> float:
    """Wilcoxon signed-rank test (two-sided) of paired differences d against 0; with ids, of the
    per-finger mean differences."""
    d = np.asarray(d, float)
    if ids is not None:
        ok = ~np.isnan(d)
        sums, counts = cluster_sums(d[ok], np.asarray(ids)[ok])
        d = sums / counts
    d = d[~np.isnan(d)]
    return float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


def probe_table(scores: pd.DataFrame, df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """One row per probe: genuine scores, acceptance at tau, FMR over its impostors, ridge fidelity.

    acc_* = genuine score > tau for the clean (original), damaged and repaired probe; fmr_* = share
    of the probe's impostor comparisons with score > tau.
    """
    sel = scores[scores["probe"] != "original"]
    gen = sel[sel["label"] == "genuine"].pivot_table(index="key", columns="probe", values="score")
    fmr = (sel[sel["label"] == "impostor"].assign(hit=lambda x: x["score"] > tau)
           .pivot_table(index="key", columns="probe", values="hit", aggfunc="mean"))
    clean = scores[(scores["probe"] == "original") & (scores["label"] == "genuine")].groupby("id")["score"].first()
    out = df[["key", "id", "type", "ridge_region_deg", "ridge_region_rec"]].set_index("key")
    out = out.join(gen.rename(columns={"degraded": "gen_deg", "reconstruction": "gen_rec"}), how="inner")
    out = out.join(fmr.rename(columns={"degraded": "fmr_deg", "reconstruction": "fmr_rec"}), how="inner")
    out["gen_clean"] = out["id"].map(clean)
    for c in ("clean", "deg", "rec"):
        out[f"acc_{c}"] = out[f"gen_{c}"] > tau
    return out.reset_index()


def headline_scores(probes: pd.DataFrame) -> pd.DataFrame:
    """Scores 1-3 per degradation type and for all probes, with 95% bootstrap CIs and p-values.

    Score 1 (identity recovery): rescued, broken, net rescue rate = (rescued - broken) / n, exact
    sign-flip p over fingers (= McNemar with one sample per finger). Score 2 (false-identity gate): FMR repaired - FMR damaged, passes if the CI's upper
    bound is below FMR_GATE. Score 3 (ridge fidelity in the damage mask): repaired - damaged mean,
    Wilcoxon p. CIs resample fingers (each with all its samples and their impostor comparisons)
    N_BOOT times; p-values are finger-level (sign_flip_p, wilcoxon_p with ids).
    """
    rows = []
    groups = [(t, probes[probes["type"] == t]) for t in TYPES if (probes["type"] == t).any()] + [("all", probes)]
    for t, g in groups:
        n, ids = len(g), g["id"].to_numpy()
        rescued = int((~g["acc_deg"] & g["acc_rec"]).sum())
        broken = int((g["acc_deg"] & ~g["acc_rec"]).sum())
        net = g["acc_rec"].to_numpy(float) - g["acc_deg"].to_numpy(float)
        dfmr = (g["fmr_rec"] - g["fmr_deg"]).to_numpy()
        ridge = (g["ridge_region_rec"] - g["ridge_region_deg"]).to_numpy()
        ridge_ok = ~np.isnan(ridge)
        row = {"type": t, "n": n,
               "accept_clean": g["acc_clean"].mean(), "accept_damaged": g["acc_deg"].mean(),
               "accept_repaired": g["acc_rec"].mean(), "rescued": rescued, "broken": broken,
               "n_fingers": g["id"].nunique(),
               "net_rescue": net.mean(), "p_rescue": sign_flip_p(net, ids),
               "fmr_damaged": g["fmr_deg"].mean(), "fmr_repaired": g["fmr_rec"].mean(),
               "fmr_delta": dfmr.mean(),
               "ridge_damaged": g["ridge_region_deg"].mean(), "ridge_repaired": g["ridge_region_rec"].mean(),
               "ridge_delta": np.nanmean(ridge), "p_ridge": wilcoxon_p(ridge, ids)}
        row["net_rescue_lo"], row["net_rescue_hi"] = boot_ci(net, ids, t)
        row["fmr_delta_lo"], row["fmr_delta_hi"] = boot_ci(dfmr, ids, t)
        row["gate"] = "pass" if row["fmr_delta_hi"] < FMR_GATE else "fail"
        row["ridge_delta_lo"], row["ridge_delta_hi"] = boot_ci(ridge[ridge_ok], ids[ridge_ok], t)
        rows.append(row)
    return pd.DataFrame(rows)


def eer_change(scores: pd.DataFrame) -> dict:
    """EER of the damaged and repaired probes (all types) and a 95% bootstrap CI of their difference.

    Resamples fingers with all their samples' genuine and impostor comparisons (paired: the same
    samples for both conditions).
    """
    sel = scores[scores["probe"].isin(["degraded", "reconstruction"]) & scores["label"].isin(["genuine", "impostor"])]
    keys = sorted(sel["key"].unique())
    id_of = sel.groupby("key")["id"].first()
    codes, fingers = pd.factorize(id_of.loc[keys].to_numpy())
    members = [np.flatnonzero(codes == c) for c in range(len(fingers))]     # sample positions per finger
    by_key = {cond: {k: (g[g["label"] == "genuine"]["score"].to_numpy(), g[g["label"] == "impostor"]["score"].to_numpy())
                     for k, g in sel[sel["probe"] == cond].groupby("key")}
              for cond in ("degraded", "reconstruction")}

    def eer_of(cond, chosen):
        parts = [by_key[cond][keys[i]] for i in chosen]
        return eer(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))

    every = np.arange(len(keys))
    out = {"eer_damaged": eer_of("degraded", every), "eer_repaired": eer_of("reconstruction", every)}
    draws = [np.concatenate([members[c] for c in idx]) for idx in boot_index(len(members), "eer")[:N_BOOT // 5]]
    diffs = [eer_of("reconstruction", idx) - eer_of("degraded", idx) for idx in draws]
    out["eer_delta"] = out["eer_repaired"] - out["eer_damaged"]
    out["eer_delta_lo"], out["eer_delta_hi"] = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return out


# ================================================================ cache
def load_cache(settings_hash: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cached degraded rows, cached scores and the NFIQ 2 cache, for these settings only."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("settings_hash") != settings_hash:          # settings changed: start over
        for name in ("degraded.csv", "scores.csv", "nfiq2.json"):
            (CACHE_DIR / name).unlink(missing_ok=True)
        shutil.rmtree(CACHE_DIR / "minutiae", ignore_errors=True)
    rows = pd.read_csv(CACHE_DIR / "degraded.csv", dtype={"id": str}) \
        if (CACHE_DIR / "degraded.csv").exists() else pd.DataFrame(columns=["cache_hash"])
    scores = pd.read_csv(CACHE_DIR / "scores.csv", dtype={"id": str, "gallery_id": str}) \
        if (CACHE_DIR / "scores.csv").exists() else pd.DataFrame(columns=["cache_hash", "gallery_sig"])
    nfiq = json.loads((CACHE_DIR / "nfiq2.json").read_text()) if (CACHE_DIR / "nfiq2.json").exists() else {}
    return rows, scores, nfiq


def save_cache(settings: dict, settings_hash: str, rows: pd.DataFrame, scores: pd.DataFrame, nfiq: dict):
    rows.to_csv(CACHE_DIR / "degraded.csv", index=False)
    scores.to_csv(CACHE_DIR / "scores.csv", index=False)
    (CACHE_DIR / "nfiq2.json").write_text(json.dumps(nfiq))
    (CACHE_DIR / "meta.json").write_text(json.dumps(
        {"settings_hash": settings_hash, "settings": settings, "updated": datetime.now().isoformat()}, indent=1))


# ================================================================ evaluation steps
def run_tools(samples, originals, orig_sha, refs, ref_sha, tools, run_dir, nfiq_cache):
    """mindtct on every image needed, NFIQ 2 on every original/degraded/reconstruction.

    Originals, references (gallery prints or originals) and degraded images are content-addressed
    in the cache; reconstructions go to the run folder. Returns {image path: .xyt/.min root} and
    fills s['nfiq2_*'] per sample.
    """
    min_cache = CACHE_DIR / "minutiae"
    roots = {}
    for fid in sorted({s["id"] for s in samples}):
        roots[originals[fid]] = min_cache / orig_sha[fid]
    needed_refs = {s["id"] for s in samples} | {g for s in samples for g in s["impostors"]}
    for fid in sorted(needed_refs & refs.keys()):
        roots[refs[fid]] = min_cache / ref_sha[fid]
    for s in samples:
        roots[s["deg_path"]] = min_cache / s["deg_sha"]
        roots[s["rec_path"]] = run_dir / "minutiae" / f"{s['key']}_rec"
    jobs = [(p, r) for p, r in roots.items()
            if not (r.with_suffix(".xyt").exists() and r.with_suffix(".min").exists())]
    parallel(lambda job: run_mindtct(tools["mindtct"], *job), jobs, "mindtct")

    if tools["nfiq2"]:
        images = {}
        for s in samples:
            images[s["orig_path"]] = ("cache", orig_sha[s["id"]])
            images[s["deg_path"]] = ("cache", s["deg_sha"])
            images[s["rec_path"]] = ("run", s["rec_sha"])
        todo = [p for p, (where, sha) in images.items() if not (where == "cache" and sha in nfiq_cache)]

        not_applicable = {p: nfiq2_applicability(p) for p in todo}
        results = {p: (np.nan, why) for p, why in not_applicable.items() if why}
        ok = [p for p in todo if not not_applicable[p]]
        batches = [ok[k:k + NFIQ2_BATCH] for k in range(0, len(ok), NFIQ2_BATCH)]
        for batch, scores in zip(batches, parallel(lambda b: nfiq2_scores(tools["nfiq2"], b), batches, "nfiq2")):
            results |= dict(zip(batch, scores))
        for path, (where, sha) in images.items():
            if path in results:
                value, why = results[path]
                if where == "cache":
                    nfiq_cache[sha] = [None if np.isnan(value) else value, why]
            else:
                value, why = nfiq_cache[sha]
                results[path] = (np.nan if value is None else value, why)
        for s in samples:
            for cond, key in (("orig", "orig_path"), ("deg", "deg_path"), ("rec", "rec_path")):
                s[f"nfiq2_{cond}"], s[f"nfiq2_note_{cond}"] = results[s[key]]
    return roots


def evaluate_samples(samples, roots, cached_rows, orig_sha):
    """Per-sample image and minutiae metrics. Degraded-side results come from the cache when possible."""
    cached = {r["cache_hash"]: r for r in cached_rows.to_dict("records")}
    rows, new_cache, hits = [], [], 0
    for s in tqdm(samples, desc="image + minutiae metrics", leave=False):
        try:
            orig, deg, rec, mask, source = load_sample_images(s)
        except ValueError as err:
            s["error"] = str(err)
            continue
        region = ndimage.binary_dilation(mask, structure=disk(REGION_DILATE_PX)) if mask.any() else mask
        m_orig = read_min(roots[s["orig_path"]].with_suffix(".min"))
        m_rec = read_min(roots[s["rec_path"]].with_suffix(".min"))
        row = {"key": s["key"], "id": s["id"], "type": s["type"], "mask_source": source,
               "mask_fraction": float(mask.mean())}
        if s["cache_hash"] in cached:
            deg_metrics = {k[:-4]: v for k, v in cached[s["cache_hash"]].items() if k.endswith("_deg")}
            hits += 1
        else:
            m_deg = read_min(roots[s["deg_path"]].with_suffix(".min"))
            deg_metrics = image_metrics(orig, deg, mask) | minutiae_metrics(m_deg, region)
        new_cache.append({"cache_hash": s["cache_hash"], "key": s["key"], "id": s["id"], "type": s["type"]}
                         | {f"{k}_deg": v for k, v in deg_metrics.items()})
        rec_metrics = image_metrics(orig, rec, mask) | minutiae_metrics(m_rec, region)
        row |= {f"{k}_orig": v for k, v in minutiae_metrics(m_orig, region).items()}
        row |= {f"{k}_deg": v for k, v in deg_metrics.items()}
        row |= {f"{k}_rec": v for k, v in rec_metrics.items()}
        for cond in ("orig", "deg", "rec"):
            row[f"nfiq2_{cond}"] = s.get(f"nfiq2_{cond}", np.nan)
            row[f"nfiq2_note_{cond}"] = s.get(f"nfiq2_note_{cond}", "")
        rows.append(row)
    keep = cached_rows[~cached_rows["cache_hash"].isin([r["cache_hash"] for r in new_cache])]
    cache_rows = pd.concat([keep, pd.DataFrame(new_cache)], ignore_index=True)
    return pd.DataFrame(rows), cache_rows, hits


def run_matching(samples, originals, orig_sha, refs, gallery_mode, roots, tools, cached_scores, ref_sig):
    """bozorth3 scores of every probe against the references.

    References are the gallery prints (gallery mode) or the originals. Three probe conditions:
    'degraded' and 'reconstruction' per sample, and 'original' per id (the undamaged image, which
    sets the ideal). Each probe is compared with its own reference ('genuine'), with the references
    of its impostor ids ('impostor') and, in gallery mode, with its own original image ('original').
    Degraded and original probes are cached; reconstructions are not.
    Returns (all scores, cache scores, number of cached probes reused).
    """
    valid = [s for s in samples if "error" not in s and s["id"] in refs]
    cached = cached_scores[cached_scores["gallery_sig"] == ref_sig]
    have = set(cached["cache_hash"])
    probes = []                                  # (cache_hash or None, condition, ids, probe image)
    for s in valid:
        if s["cache_hash"] not in have:
            probes.append((s["cache_hash"], "degraded", s, s["deg_path"]))
        probes.append((None, "reconstruction", s, s["rec_path"]))
    for fid in sorted({s["id"] for s in valid}):
        h = text_sha1("orig", orig_sha[fid])
        if h not in have:
            ids = {"id": fid, "key": fid, "type": "", "impostors": impostor_ids(fid, sorted(refs))}
            probes.append((h, "original", ids, originals[fid]))

    def score(probe):
        _, cond, s, image = probe
        targets = [(s["id"], "genuine", refs[s["id"]])] + [(g, "impostor", refs[g]) for g in s["impostors"]]
        if gallery_mode:
            targets.append((s["id"], "original", originals[s["id"]]))
        values = bozorth_scores(tools["bozorth3"], roots[image].with_suffix(".xyt"),
                                [roots[p].with_suffix(".xyt") for _, _, p in targets])
        return [{"probe": cond, "key": s["key"], "id": s["id"], "type": s["type"], "gallery_id": g,
                 "label": label, "score": v} for (g, label, _), v in zip(targets, values)]

    results = parallel(score, probes, "bozorth3")
    new = pd.DataFrame([r | {"cache_hash": h, "gallery_sig": ref_sig}
                        for (h, _, _, _), rows in zip(probes, results) for r in rows])
    fresh = new[new["probe"] != "reconstruction"]
    cache_out = pd.concat([cached_scores, fresh], ignore_index=True)
    wanted = {s["cache_hash"] for s in valid} | {text_sha1("orig", orig_sha[s["id"]]) for s in valid}
    reused = cached[cached["cache_hash"].isin(wanted)]
    scores = pd.concat([reused, new], ignore_index=True)
    return scores.drop(columns=["cache_hash", "gallery_sig"]), cache_out, reused["cache_hash"].nunique()


def impostor_ids(fid: str, gallery: list[str]) -> list[str]:
    """N_IMPOSTORS other reference ids for `fid`, the same every run for the same references."""
    others = [g for g in gallery if g != fid]
    rng = np.random.default_rng([SEED, int(hashlib.sha1(fid.encode()).hexdigest()[:8], 16)])
    return sorted(rng.choice(others, size=min(N_IMPOSTORS, len(others)), replace=False).tolist())


# ================================================================ summaries
def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add <m>_delta = rec - deg per metric, signed so that positive = better."""
    df = df.copy()
    for m, higher in PER_IMAGE_METRICS.items():
        if f"{m}_deg" in df and f"{m}_rec" in df:
            df[f"{m}_delta"] = (1 if higher else -1) * (df[f"{m}_rec"] - df[f"{m}_deg"])
    return df


def ci95(x: pd.Series, ids: pd.Series) -> float:
    """Half-width of the normal-approximation 95% CI of the mean, with the finger-clustered standard
    error (= std / sqrt(n) when every finger has one sample)."""
    ok = x.notna()
    sums, counts = cluster_sums(x[ok].to_numpy(), ids[ok].to_numpy())
    k, mean = len(sums), x[ok].mean()
    if k < 2:
        return np.nan
    return float(1.96 * np.sqrt(np.sum((sums - mean * counts) ** 2) * k / (k - 1)) / counts.sum())


def per_image_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (type, metric): clean mean (if any), deg/rec mean ± std, Δ mean ± 95% CI, Wilcoxon p, n."""
    out = []
    groups = [(t, df[df["type"] == t]) for t in TYPES if (df["type"] == t).any()] + [("all", df)]
    for t, g in groups:
        for m in PER_IMAGE_METRICS:
            if f"{m}_deg" not in g:
                continue
            both = g[[f"{m}_deg", f"{m}_rec"]].dropna()
            out.append({"type": t, "metric": m, "n": len(both),
                        "orig_mean": g.loc[both.index, f"{m}_orig"].mean() if f"{m}_orig" in g else np.nan,
                        "deg_mean": both[f"{m}_deg"].mean(), "deg_std": both[f"{m}_deg"].std(),
                        "rec_mean": both[f"{m}_rec"].mean(), "rec_std": both[f"{m}_rec"].std(),
                        "delta_mean": g[f"{m}_delta"].mean(), "delta_ci95": ci95(g[f"{m}_delta"], g["id"]),
                        "p_wilcoxon": wilcoxon_p(g[f"{m}_delta"].to_numpy(), g["id"]) if len(both) else np.nan})
    return pd.DataFrame(out)


def condition_scores(scores: pd.DataFrame, cond: str, t: str, label: str) -> np.ndarray:
    """Scores with `label` of one probe condition for degradation type t ('all' = every type).

    The 'original' condition has one probe per id; for a type it uses the ids of that type's samples.
    """
    sel = scores[(scores["probe"] == cond) & (scores["label"] == label)]
    if t == "all":
        return sel["score"].to_numpy()
    if cond == "original":
        ids = set(scores.loc[(scores["probe"] == "degraded") & (scores["type"] == t), "id"])
        return sel[sel["id"].isin(ids)]["score"].to_numpy()
    return sel[sel["type"] == t]["score"].to_numpy()


def matching_summary(scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Per type and condition (original, degraded, reconstruction): EER, FAR/FRR at each threshold,
    score statistics."""
    def stat(fn, x):
        return float(fn(x)) if len(x) else np.nan

    rows = []
    present = [t for t in TYPES if (scores["type"] == t).any()]
    for t in present + ["all"]:
        for cond in ("original", "degraded", "reconstruction"):
            gen = condition_scores(scores, cond, t, "genuine")
            imp = condition_scores(scores, cond, t, "impostor")
            row = {"type": t, "condition": cond, "n_genuine": len(gen), "n_impostor": len(imp),
                   "gen_mean": stat(np.mean, gen), "gen_median": stat(np.median, gen),
                   "imp_mean": stat(np.mean, imp), "imp_median": stat(np.median, imp),
                   "imp_p95": stat(lambda x: np.percentile(x, 95), imp), "eer": eer(gen, imp)}
            for name, thr in thresholds.items():
                row[f"far_{name}"], row[f"frr_{name}"] = far_frr(gen, imp, thr)
            rows.append(row)
    return pd.DataFrame(rows)


# ================================================================ plots
def set_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED_INK,
        "axes.labelcolor": INK, "xtick.color": MUTED_INK, "ytick.color": MUTED_INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.facecolor": "white", "savefig.bbox": "tight",
        "lines.linewidth": 2, "hatch.linewidth": 0.6})


def tint(color: str, amount: float = 0.55) -> tuple:
    """Mix `color` with white (amount = fraction of white)."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def condition_style(cond: str, t: str) -> dict:
    """Fill, edge and hatch for original / degraded (light, hatched) / reconstruction (full colour)."""
    if cond == "orig":
        return {"facecolor": tint(ORIGINAL_COLOR, 0.3), "edgecolor": ORIGINAL_COLOR, "hatch": ""}
    if cond == "deg":
        return {"facecolor": tint(TYPE_COLORS[t], 0.65), "edgecolor": TYPE_COLORS[t], "hatch": "////"}
    return {"facecolor": TYPE_COLORS[t], "edgecolor": TYPE_COLORS[t], "hatch": ""}


def condition_legend(fig, conds=("deg", "rec")) -> None:
    """Legend for the condition encoding, above the axes (call after tight_layout)."""
    from matplotlib.patches import Patch
    names = {"orig": "original", "deg": "degraded (light, hatched)", "rec": "reconstruction (solid)"}
    handles = [Patch(**(condition_style(c, "pressure") | ({"facecolor": tint(MUTED_INK, 0.65),
                                                           "edgecolor": MUTED_INK} if c == "deg" else
                                                          {"facecolor": MUTED_INK, "edgecolor": MUTED_INK}
                                                          if c == "rec" else {})), label=names[c])
               for c in conds]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0),
               title="colour = degradation type", title_fontsize=8, fontsize=8)


def types_in(df: pd.DataFrame) -> list[str]:
    return [t for t in TYPES if (df["type"] == t).any()]


def boxes(ax, df: pd.DataFrame, metric: str, conds=("deg", "rec")) -> None:
    """Box per (type, condition) for column <metric>_<cond>; types on x."""
    types = types_in(df)
    width = 0.8 / len(conds)
    for i, t in enumerate(types):
        for j, cond in enumerate(conds):
            col = f"{metric}_{cond}"
            if col not in df:
                continue
            data = df.loc[df["type"] == t, col].dropna()
            if data.empty:
                continue
            pos = i + (j - (len(conds) - 1) / 2) * width
            bp = ax.boxplot([data], positions=[pos], widths=width * 0.85, patch_artist=True,
                            medianprops={"color": INK, "linewidth": 1.2},
                            whiskerprops={"color": MUTED_INK, "linewidth": 0.8},
                            capprops={"color": MUTED_INK, "linewidth": 0.8},
                            flierprops={"marker": "o", "markersize": 2, "alpha": 0.4,
                                        "markeredgecolor": MUTED_INK})
            for patch in bp["boxes"]:
                patch.set(**condition_style(cond, t), linewidth=0.8)
    ax.set_xticks(range(len(types)), types)
    ax.set_xlim(-0.6, len(types) - 0.4)


def save(fig, plots_dir: Path, name: str) -> str:
    fig.savefig(plots_dir / name, dpi=DPI)
    plt.close(fig)
    return name


def plot_image_metrics(df, plots_dir):
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5))
    for r, scope in enumerate(("full", "region")):
        for c, base in enumerate(("ridge", "psnr", "msssim", "lpips")):
            m = f"{base}_{scope}"
            boxes(axes[r, c], df, m)
            arrow = "lower is better" if base == "lpips" else "higher is better"
            axes[r, c].set_title(f"{LABELS[m]}  ({arrow})")
    fig.tight_layout()
    condition_legend(fig)
    return save(fig, plots_dir, "image_metrics_box.png")


def plot_scores(headline: pd.DataFrame, plots_dir):
    """The three headline scores per degradation type and for all probes, with 95% CIs."""
    rows = headline["type"].tolist()
    colors = [TYPE_COLORS.get(t, INK) for t in rows]
    panels = [("net_rescue", 100, "Score 1: net rescue rate (points)\nrescued − broken, share of probes"),
              ("fmr_delta", 100, f"Score 2: FMR change (points)\npass if the CI stays below +{FMR_GATE * 100:g} (dashed)"),
              ("ridge_delta", 1, "Score 3: ridge fidelity change\nrepaired − damaged, in the damage mask")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (m, scale, title) in zip(axes, panels):
        v, lo, hi = (headline[c].to_numpy() * scale for c in (m, f"{m}_lo", f"{m}_hi"))
        ax.bar(range(len(rows)), v, 0.6, color=[tint(c, 0.35) for c in colors], edgecolor=colors, linewidth=0.8)
        ax.errorbar(range(len(rows)), v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.axhline(0, color=MUTED_INK, linewidth=0.8)
        if m == "fmr_delta":
            ax.axhline(FMR_GATE * 100, color=MUTED_INK, linewidth=0.8, linestyle="--")
        if m == "net_rescue":
            for k, r in headline.iterrows():
                ax.annotate(f"+{r['rescued']}/−{r['broken']}", (k, max(hi[k], 0)), ha="center", va="bottom",
                            fontsize=7, color=MUTED_INK, xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(range(len(rows)), rows, rotation=30)
        ax.set_title(title)
    fig.tight_layout()
    return save(fig, plots_dir, "scores.png")


def plot_deltas_per_image(df, plots_dir):
    """Per-image Δ (repaired − damaged, positive = better) of the main per-image metrics, by type."""
    metrics = [m for m in ("ridge_region", "psnr_region", "msssim_region", "lpips_region", "mq_mean_region",
                           "nfiq2", "orig_match_score", "match_score")
               if f"{m}_delta" in df and df[f"{m}_delta"].notna().any()]
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), squeeze=False)
    types = types_in(df)
    for ax, m in zip(axes.flat, metrics):
        data = [df.loc[df["type"] == t, f"{m}_delta"].dropna() for t in types]
        bp = ax.boxplot(data, positions=range(len(types)), widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"color": INK}, whiskerprops={"color": MUTED_INK}, capprops={"color": MUTED_INK})
        for patch, t in zip(bp["boxes"], types):
            patch.set(facecolor=tint(TYPE_COLORS[t], 0.25), edgecolor=TYPE_COLORS[t])
        for k, (t, d) in enumerate(zip(types, data)):
            jitter = np.random.default_rng(k).uniform(-0.18, 0.18, len(d))
            ax.scatter(k + jitter, d, s=4, color=TYPE_COLORS[t], alpha=0.5, linewidths=0)
        ax.axhline(0, color=MUTED_INK, linewidth=0.8)
        ax.set_xticks(range(len(types)), types, rotation=30)
        ax.set_title(f"Δ {LABELS[m]}")
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Per-image change, repaired − damaged (signed so that above 0 = better)", y=1.0)
    fig.tight_layout()
    return save(fig, plots_dir, "deltas_per_image.png")


def plot_psnr_scatter(df, plots_dir):
    types = types_in(df)
    fig, axes = plt.subplots(1, len(types), figsize=(3.0 * len(types), 3.2), squeeze=False)
    for ax, t in zip(axes[0], types):
        g = df[df["type"] == t]
        ax.scatter(g["psnr_region_deg"], g["psnr_region_rec"], s=10, color=TYPE_COLORS[t], alpha=0.7, linewidths=0)
        lo = np.nanmin(df[["psnr_region_deg", "psnr_region_rec"]].to_numpy())
        hi = np.nanmax(df[["psnr_region_deg", "psnr_region_rec"]].to_numpy())
        ax.plot([lo, hi], [lo, hi], color=MUTED_INK, linewidth=1, linestyle="--")
        ax.set_title(t)
        ax.set_xlabel("degraded PSNR region (dB)")
    axes[0, 0].set_ylabel("reconstruction PSNR region (dB)")
    fig.suptitle("Region PSNR per image: points above the dashed line improved", y=1.02)
    fig.tight_layout()
    return save(fig, plots_dir, "psnr_paired_scatter.png")


def plot_match_scores(df, plots_dir):
    metrics = [m for m in ("orig_match_score", "match_score") if f"{m}_rec" in df]
    titles = {"orig_match_score": "Original-match score: probe vs its own original\n(original = matched with itself)",
              "match_score": "Gallery match score: probe vs gallery print\n(original = original vs gallery; dotted = 40)"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.0), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        boxes(ax, df, m, conds=("orig", "deg", "rec"))
        if m == "match_score":
            ax.axhline(FIXED_THRESHOLD, color=MUTED_INK, linewidth=0.8, linestyle=":")
        ax.set_ylabel("bozorth3 score")
        ax.set_title(titles[m])
    fig.tight_layout()
    condition_legend(fig, conds=("orig", "deg", "rec"))
    return save(fig, plots_dir, "match_scores.png")


def plot_minutiae_quality(df, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, m in zip(axes, ("mq_mean_region", "mq_median_region")):
        boxes(ax, df, m, conds=("orig", "deg", "rec"))
        ax.set_title(f"{LABELS[m]} in region (MINDTCT reliability 0–1, higher = better)")
    fig.tight_layout()
    condition_legend(fig, conds=("orig", "deg", "rec"))
    return save(fig, plots_dir, "minutiae_quality.png")


def plot_nfiq2(df, plots_dir):
    if df[["nfiq2_orig", "nfiq2_deg", "nfiq2_rec"]].notna().sum().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(1.8 * len(types_in(df)) + 2, 3.8))
    boxes(ax, df, "nfiq2", conds=("orig", "deg", "rec"))
    ax.set_ylim(0, 100)
    ax.set_ylabel("NFIQ 2 unified quality score")
    ax.set_title("NFIQ 2 (0–100, higher = better quality under NFIQ 2.x)")
    fig.tight_layout()
    condition_legend(fig, conds=("orig", "deg", "rec"))
    return save(fig, plots_dir, "nfiq2_box.png")


def plot_score_hist(scores, thresholds, plots_dir):
    types = [t for t in TYPES if (scores["type"] == t).any()]
    fig, axes = plt.subplots(2, len(types), figsize=(3.2 * len(types), 5.6), squeeze=False)
    gen_all = scores[scores["label"] == "genuine"]["score"]
    gen_hi = max(np.percentile(gen_all, 99), FIXED_THRESHOLD + 15) if len(gen_all) else 100
    imp_hi = max(np.percentile(scores[scores["label"] == "impostor"]["score"], 99.9), FIXED_THRESHOLD + 15)
    for k, t in enumerate(types):
        for r, (label, hi) in enumerate((("genuine", gen_hi), ("impostor", imp_hi))):
            ax = axes[r, k]
            edges = np.linspace(0, hi, 40)
            base = condition_scores(scores, "original", t, label)
            ax.hist(np.clip(base, 0, hi), edges, density=True, histtype="stepfilled",
                    color=tint(ORIGINAL_COLOR, 0.6), label="original")
            for cond, style in (("degraded", "--"), ("reconstruction", "-")):
                x = condition_scores(scores, cond, t, label)
                color = tint(TYPE_COLORS[t], 0.35) if cond == "degraded" else TYPE_COLORS[t]
                ax.hist(np.clip(x, 0, hi), edges, density=True, histtype="step", linestyle=style,
                        color=color, linewidth=2, label=cond)
            for thr in thresholds.values():
                if thr <= hi:
                    ax.axvline(thr, color=MUTED_INK, linewidth=0.8, linestyle=":")
            ax.set_title(f"{t}: {label}")
            ax.set_xlabel("bozorth3 score")
            if k == 0:
                ax.set_ylabel("density")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Match score distributions against the references (scores clipped at the right edge; "
                 "dotted lines = thresholds " + ", ".join(f"{n} = {v:g}" for n, v in thresholds.items()) + ")", y=1.01)
    fig.tight_layout()
    return save(fig, plots_dir, "match_score_hist.png")


def plot_det(scores, match, plots_dir):
    types = [t for t in TYPES if (scores["type"] == t).any()]
    fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 3.4), squeeze=False)
    for ax, t in zip(axes[0], types):
        for cond, style in (("original", "-"), ("degraded", "--"), ("reconstruction", "-")):
            gen = condition_scores(scores, cond, t, "genuine")
            imp = condition_scores(scores, cond, t, "impostor")
            if not len(gen) or not len(imp):
                continue
            p = roc(gen, imp)
            floor = 0.5 / len(imp), 0.5 / len(gen)
            color = {"original": ORIGINAL_COLOR, "degraded": tint(TYPE_COLORS[t], 0.35),
                     "reconstruction": TYPE_COLORS[t]}[cond]
            ax.plot(np.maximum(p["far"], floor[0]), np.maximum(p["frr"], floor[1]), style, color=color,
                    drawstyle="steps-post", label=cond, linewidth=1.2 if cond == "original" else 2)
            e = match[(match["type"] == t) & (match["condition"] == cond)]["eer"].iloc[0]
            ax.plot([max(e, floor[0])], [max(e, floor[1])], "o", markersize=7, color=color,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([1e-4, 1], [1e-4, 1], color=GRID, linewidth=1)
        ax.set_title(t)
        ax.set_xlabel("FAR")
    axes[0, 0].set_ylabel("FRR")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("DET curves against the references (dot = EER; lower-left is better; zero rates drawn at the floor)", y=1.02)
    fig.tight_layout()
    return save(fig, plots_dir, "det_curves.png")


def plot_error_bars(match, plots_dir):
    metrics = ["eer", "far_t40", "frr_t40", "frr_far1"]
    types = [t for t in TYPES if (match["type"] == t).any()]
    conds = (("original", "orig"), ("degraded", "deg"), ("reconstruction", "rec"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.4))
    for ax, m in zip(axes, metrics):
        for i, t in enumerate(types):
            for j, (cond, short) in enumerate(conds):
                v = match[(match["type"] == t) & (match["condition"] == cond)][m].iloc[0]
                ax.bar(i + (j - 1) * 0.27, v, 0.25, **condition_style(short, t), linewidth=0.8)
        ax.set_xticks(range(len(types)), types, rotation=30)
        ax.set_title(f"{LABELS[m]} (lower is better)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    condition_legend(fig, conds=("orig", "deg", "rec"))
    return save(fig, plots_dir, "eer_far_frr_bars.png")


def plot_impostor_shift(scores, thresholds, plots_dir):
    types = [t for t in TYPES if (scores["type"] == t).any()]
    fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 3.3), squeeze=False, sharey=True)
    hi = max(np.percentile(scores[scores["label"] == "impostor"]["score"], 99.9), FIXED_THRESHOLD + 10)
    for ax, t in zip(axes[0], types):
        for cond, style in (("original", "-"), ("degraded", "--"), ("reconstruction", "-")):
            x = np.sort(condition_scores(scores, cond, t, "impostor"))
            color = {"original": ORIGINAL_COLOR, "degraded": tint(TYPE_COLORS[t], 0.35),
                     "reconstruction": TYPE_COLORS[t]}[cond]
            ax.plot(x, 1 - np.arange(1, len(x) + 1) / len(x), style, color=color, label=cond)
        ax.axvline(FIXED_THRESHOLD, color=MUTED_INK, linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlim(0, hi)
        ax.set_title(t)
        ax.set_xlabel("impostor bozorth3 score")
    axes[0, 0].set_ylabel("fraction of impostors above score")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Impostor score tails: a reconstruction curve right of the original means spurious identity", y=1.02)
    fig.tight_layout()
    return save(fig, plots_dir, "impostor_shift.png")


def plot_examples(df, samples, roots, plots_dir):
    by_key = {s["key"]: s for s in samples}
    types = types_in(df)
    fig, axes = plt.subplots(len(types), 5, figsize=(13, 2.9 * len(types)), squeeze=False)
    for r, t in enumerate(types):
        g = df[df["type"] == t].dropna(subset=["ridge_region_rec"])
        if g.empty:
            g = df[df["type"] == t]
        pick = g.iloc[(g["ridge_region_rec"] - g["ridge_region_rec"].median()).abs().argsort().iloc[0]]
        s = by_key[pick["key"]]
        orig, deg, rec, mask, _ = load_sample_images(s)
        region = ndimage.binary_dilation(mask, structure=disk(REGION_DILATE_PX)) if mask.any() else mask
        m_orig = read_min(roots[s["orig_path"]].with_suffix(".min"))
        m_rec = read_min(roots[s["rec_path"]].with_suffix(".min"))
        panels = [(orig, "original"), (deg, "degraded + mask"), (rec, "reconstruction"),
                  (np.abs(rec.astype(int) - orig), "|reconstruction − original|"), (rec, "minutiae in region")]
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray_r" if c == 3 else "gray", vmin=0, vmax=255 if c != 3 else max(1, img.max()))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            ax.spines[:].set_visible(False)
            ax.set_title(title if r == 0 else "", fontsize=9)
        axes[r, 1].contour(mask, levels=[0.5], colors=[TYPE_COLORS[t]], linewidths=1.2)
        ax = axes[r, 4]
        ax.contour(region, levels=[0.5], colors=[MUTED_INK], linewidths=0.8)
        o, c = m_orig[inside(m_orig, region)], m_rec[inside(m_rec, region)]
        ax.scatter(o[:, 0], o[:, 1], s=34, marker="s", facecolors="none", edgecolors=ORIG_MINUTIA_COLOR,
                   linewidths=1.4, label="original")
        ax.scatter(c[:, 0], c[:, 1], s=26, marker="x", color=REC_MINUTIA_COLOR, linewidths=1.4, label="reconstruction")
        axes[r, 0].set_ylabel(f"{t}\n{pick['key']}", fontsize=8)
        axes[r, 2].set_xlabel(f"ridge fidelity {pick['ridge_region_deg']:.2f} → {pick['ridge_region_rec']:.2f}", fontsize=8)
        axes[r, 4].set_xlabel(f"original-match {fmt(pick['orig_match_score_deg'], 0)} → "
                              f"{fmt(pick['orig_match_score_rec'], 0)} (ideal {fmt(pick['orig_match_score_orig'], 0)})",
                              fontsize=8)
    axes[0, 4].legend(fontsize=7, loc="lower right", framealpha=0.8, frameon=True)
    fig.suptitle("Typical example per degradation (median ridge fidelity of the reconstruction; MSU-normalised images)", y=1.0)
    fig.tight_layout()
    return save(fig, plots_dir, "examples.png")


def select_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Three samples per type with about median damage, at low / median / high gain.

    Damage = 1 − the degraded image's SAMPLE_METRIC (ridge fidelity in the mask: 1 = undamaged
    ridges); gain = its Δ (repaired − degraded). Within each type, samples whose damage percentile
    lies in SAMPLE_DAMAGE_BAND are candidates (all samples if fewer than 3); for each target in
    SAMPLE_PCTS the candidate whose gain is closest to that gain percentile of the type is picked.
    Percentiles are ranks among all of the type's samples.
    """
    m = SAMPLE_METRIC
    df = df.assign(damage=1 - df[f"{m}_deg"], gain=df[f"{m}_delta"])
    picks = []
    for t in types_in(df):
        g = df[df["type"] == t].dropna(subset=["gain", "damage"])
        if g.empty:
            continue
        g = g.assign(gain_pct=100 * g["gain"].rank(pct=True), damage_pct=100 * g["damage"].rank(pct=True))
        band = g[g["damage_pct"].between(*SAMPLE_DAMAGE_BAND)]
        band = band if len(band) >= len(SAMPLE_PCTS) else g
        for p, level in zip(SAMPLE_PCTS, ("low gain", "median gain", "high gain")):
            left = band[~band["key"].isin([q["key"] for q in picks])]
            if left.empty:
                break
            row = left.loc[(left["gain"] - np.percentile(g["gain"], p)).abs().idxmin()]
            picks.append(row[["key", "type", "gain", "gain_pct", "damage", "damage_pct", f"{m}_deg",
                              f"{m}_rec"]].to_dict() | {"level": level})
    return pd.DataFrame(picks)


def plot_samples(picks: pd.DataFrame, samples, plots_dir):
    """Original / degraded (mask outlined) / reconstruction of the selected samples: one row per type."""
    by_key = {s["key"]: s for s in samples}
    types = [t for t in TYPES if (picks["type"] == t).any()]
    n = len(SAMPLE_PCTS)
    ratios = sum(([1, 1, 1, 0.12] for _ in range(n)), [])[:-1]
    fig = plt.figure(figsize=(1.45 * 3 * n + 0.6, 2.75 * len(types)))
    grid = fig.add_gridspec(len(types), len(ratios), width_ratios=ratios, wspace=0.04, hspace=0.32,
                            top=0.95, bottom=0.02, left=0.04, right=0.99)
    for r, t in enumerate(types):
        for k, (_, pick) in enumerate(picks[picks["type"] == t].iterrows()):
            orig, deg, rec, mask, _ = load_sample_images(by_key[pick["key"]])
            for c, (img, title) in enumerate(((orig, "original"), (deg, "degraded"), (rec, "reconstruction"))):
                ax = fig.add_subplot(grid[r, 4 * k + c])
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
                ax.spines[:].set_visible(False)
                if r == 0:
                    ax.set_title(f"{pick['level']}\n{title}" if c == 1 else f"\n{title}", fontsize=9)
                if c == 0 and k == 0:
                    ax.set_ylabel(t, fontsize=10, color=INK)
                if c == 1:
                    ax.contour(mask, levels=[0.5], colors=[TYPE_COLORS[t]], linewidths=0.9)
                    ax.set_xlabel(f"{pick['key']}\nΔ = {pick['gain']:+.2f} (p{pick['gain_pct']:.0f}), "
                                  f"damage p{pick['damage_pct']:.0f}", fontsize=7.5, color=MUTED_INK)
    fig.suptitle(f"Sample reconstructions at about median damage: low / median / high Δ {LABELS[SAMPLE_METRIC].lower()} "
                 f"(targets p{' / p'.join(map(str, SAMPLE_PCTS))} within each type; mask outlined; "
                 f"MSU-normalised images)", y=0.995)
    return save(fig, plots_dir, "samples.png")


# ================================================================ report
def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return str(v)


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pvalue(p: float) -> str:
    if p is None or np.isnan(p):
        return "–"
    return f"{p:.0e}" if p < 0.001 else f"{p:.3f}"


def metric_table(summary: pd.DataFrame, metric: str) -> str:
    sel = summary[summary["metric"] == metric]
    digits = 2 if metric.startswith("psnr") or metric in ("nfiq2", "orig_match_score", "match_score") else 3
    has_orig = sel["orig_mean"].notna().any()
    header = ["type", "n"] + (["clean"] if has_orig else []) + ["damaged", "repaired", "Δ (± 95% CI)", "p (Wilcoxon)"]
    rows = []
    for _, r in sel.iterrows():
        row = [r["type"], int(r["n"])]
        if has_orig:
            row.append(fmt(r["orig_mean"], digits))
        row += [f"{fmt(r['deg_mean'], digits)} ± {fmt(r['deg_std'], digits)}",
                f"{fmt(r['rec_mean'], digits)} ± {fmt(r['rec_std'], digits)}",
                f"{r['delta_mean']:+.{digits}f} ± {fmt(r['delta_ci95'], digits)}" if not np.isnan(r["delta_mean"]) else "–",
                pvalue(r["p_wilcoxon"])]
        rows.append(row)
    return md_table(header, rows)


def scores_table(headline: pd.DataFrame) -> str:
    """The headline table: scores 1-3 per type and for all probes."""
    pts = lambda v: f"{100 * v:+.1f}"
    header = ["type", "n", "accepted clean / damaged / repaired", "rescued", "broken",
              "**1 net rescue** (95% CI)", "p", "FMR damaged → repaired", "**2 FMR change** (95% CI)", "gate",
              "ridge fid. damaged → repaired", "**3 ridge fid. change** (95% CI)", "p"]
    rows = []
    for _, r in headline.iterrows():
        bold = (lambda s: f"**{s}**") if r["type"] == "all" else (lambda s: s)
        rows.append([bold(r["type"]), int(r["n"]),
                     f"{r['accept_clean']:.1%} / {r['accept_damaged']:.1%} / {r['accept_repaired']:.1%}",
                     int(r["rescued"]), int(r["broken"]),
                     bold(f"{pts(r['net_rescue'])} [{pts(r['net_rescue_lo'])}, {pts(r['net_rescue_hi'])}]"),
                     pvalue(r["p_rescue"]),
                     f"{r['fmr_damaged']:.2%} → {r['fmr_repaired']:.2%}",
                     bold(f"{pts(r['fmr_delta'])} [{pts(r['fmr_delta_lo'])}, {pts(r['fmr_delta_hi'])}]"),
                     bold(r["gate"]),
                     f"{r['ridge_damaged']:.3f} → {r['ridge_repaired']:.3f}",
                     bold(f"{r['ridge_delta']:+.3f} [{r['ridge_delta_lo']:+.3f}, {r['ridge_delta_hi']:+.3f}]"),
                     pvalue(r["p_ridge"])])
    return md_table(header, rows)


def cost_lines(benchmark: dict | None) -> list[str]:
    if benchmark is None:
        return ["Not measured: no `--benchmark` file was given (run `experiments/benchmark.py`).", ""]
    b = benchmark
    gpu = "–" if b.get("peak_gpu_mb") is None else f"{b['peak_gpu_mb']:.0f} MB"
    return [md_table(["device", "seconds per print (median)", "p90", "peak GPU memory", "peak RAM", "parameters"],
                     [[f"{b['device']} ({b['device_name']})", f"{b['seconds_median']:.3f}", f"{b['seconds_p90']:.3f}",
                       gpu, f"{b['peak_ram_mb']:.0f} MB", f"{b['parameters']:,}"]]), "",
            f"From `{b['file']}`: {b['n_prints']} test prints, one at a time, after {b['warmup']} warm-up calls, "
            f"{b['date']}. Peak GPU memory = PyTorch's peak allocated memory during the timed calls; peak RAM = "
            f"the process's peak resident memory (model and data included).", ""]


def definitions(thresholds: dict, info: dict) -> list[str]:
    """Section 0: every base measure and every metric, from first principles."""
    tau = thresholds[TAU]
    ref = "gallery print" if info["gallery"] else "original"
    return [
        "## 0. What is measured, and how", "",
        "### Base measures", "",
        "- **Probe set.** Each test probe *k* = 1…N is one finger with: the clean impression *O* (the *original*), "
        "the damaged copy *D* (a synthetic contaminant painted onto *O*), the damage mask *M* (the pixels the "
        "contaminant changed), the method's repaired image *R*, and the "
        + ("**gallery print** *G*, a separate capture of the same finger that plays the enrolled reference. "
           if info["gallery"] else "reference *G* = *O* itself (no gallery given). ")
        + f"Each probe also gets the same {N_IMPOSTORS} *impostor references*: {ref}s of other fingers drawn at "
          f"random (seed {SEED}), identical for every condition and every experiment. The three *conditions* of a "
          "probe are clean (*O*), damaged (*D*) and repaired (*R*).",
        "- **Normalisation (temporary).** "
        + (f"Before anything is measured, every image (O, D, R and the references) is replaced by the texture "
           f"component of MSU-LatentAFIS's cartoon-texture decomposition (Buades et al.; σ = {NORMALIZE_SIGMA:g}; "
           "`experiments/baselines/E04_msu_stft/preprocess.py`). It removes the slowly varying background, brightness and "
           "stain shading and keeps the ridge pattern (8-bit, background grey 128). All methods, including "
           "whole-image enhancers, are therefore measured in one ridge representation."
           + (" **This method works on normalised prints** (its input is normalised, its output is already a "
              "normalised print), so R is measured as stored and only O, D and the references are normalised."
              if info.get("rec_normalized") else "")
           if NORMALIZE_SIGMA else "Off: images are measured as stored."),
        "- **Minutiae.** NBIS MINDTCT finds the ridge endings and bifurcations of an image: position, direction "
        "and a reliability in [0, 1].",
        "- **Match score s(a, b).** NBIS Bozorth3 compares the minutiae of two images and returns the size of "
        "the largest set of minutia pairs whose relative distances and angles agree in both: an integer, "
        "higher = more likely the same finger, with no upper bound and no probabilistic meaning. The *genuine "
        f"score* of a probe is s(probe, its own {ref}); its *impostor scores* are s(probe, each impostor reference).",
        f"- **Threshold τ = {tau:g}.** A comparison is *accepted* (\"same finger\") when s > τ. τ is the smallest "
        f"score such that at most {FAR_TARGETS[0]:.0%} of the **clean** probes' impostor scores exceed it, i.e. "
        f"the false-match rate (FMR) of undamaged prints is {FAR_TARGETS[0]:.0%}. It is set once from clean data, "
        "so it is the same line for every experiment on this evaluation set; it is a property of the matcher and "
        "the data, not of the method. This is how deployed systems are configured (threshold for a target FMR). "
        "Scores 1 and 2 count events at τ, so τ fixes what *rescued* means: a probe whose genuine score goes "
        f"from below to above τ. At FMR {FAR_TARGETS[1]:.1%} (τ = {thresholds['far01']:g}) damaged prints produce "
        "too few false matches to estimate score 2.",
        f"- **Ridge pattern.** Band-pass B(x) = G_{RIDGE_BAND_SIGMAS[0]:g}∗x − G_{RIDGE_BAND_SIGMAS[1]:g}∗x "
        "(difference of Gaussian blurs, σ in px): keeps structure at ridge scale and removes brightness, slow "
        "shading and pixel noise.",
        "",
        "### Headline scores", "",
        "**Score 1: identity recovery (net rescue rate).** Does the right finger come back?",
        "",
        "- a_c(k) = 1 if probe k's genuine score in condition c exceeds τ, else 0.",
        "- *rescued* = #probes with a_D = 0 and a_R = 1 (rejected when damaged, accepted after repair); "
        "*broken* = #probes with a_D = 1 and a_R = 0.",
        "- **net rescue rate = (rescued − broken) / N** = acceptance rate of R − acceptance rate of D, in "
        "percentage points. For scale: the most a method can gain is the acceptance rate of O − that of D.",
        "- *Why:* it counts what verification needs (a probe accepted or not at the operating point), compares "
        "each probe with itself, and never divides by the clean-vs-damaged gap, which made the old restoration R "
        "unstable when the gap was near zero or negative. Rescued and broken are reported separately because a "
        "method can rescue some probes and break others.",
        "- *Statistics:* exact sign-flip test on the fingers (H0: repair rescues and breaks equally often; each "
        "finger's rescued − broken total is equally likely to have either sign). With one sample per finger this "
        "is the exact McNemar test (binomial test of rescued against rescued + broken at ½). 95% CI: percentile "
        "bootstrap over fingers.",
        "",
        "**Score 2: false-identity gate.** Does repair make prints match the wrong finger?",
        "",
        "- FMR_c = share of all impostor comparisons of condition c with score > τ (N × "
        f"{N_IMPOSTORS} comparisons; FMR_O ≈ {FAR_TARGETS[0]:.0%} by construction of τ).",
        "- **FMR change = FMR_R − FMR_D**, in percentage points. **Pass** when the upper bound of its 95% CI is "
        f"below +{FMR_GATE * 100:g} points.",
        "- *Why:* a repair that invents minutiae can create matches with strangers: the hallucination failure "
        "that genuine scores cannot see. It is a safety requirement, so it is a pass/fail gate, not a score to "
        f"optimise. The margin, +{FMR_GATE * 100:g} points, is half the design FMR. Passing requires evidence "
        "that the increase is small (a non-inferiority test), not merely the absence of a significant increase.",
        "- *Statistics:* 95% CI by bootstrap over fingers, each resampled with all its samples and their impostor "
        "comparisons.",
        "",
        "**Score 3: ridge fidelity.** Does the repaired region show the true ridges?",
        "",
        "- ρ(p) = local correlation of B(O) and B(X) around pixel p: covariance / √(variance × variance), each "
        f"computed with a Gaussian window of σ = {RIDGE_WINDOW_SIGMA:g} px (about two ridge periods).",
        "- RF(X) = mean of ρ over the pixels of the mask M. 1 = the same ridges as the clean print (up to local "
        "brightness and contrast), 0 = unrelated, < 0 = ridges where the valleys should be. RF(O) = 1.",
        "- **ridge fidelity change = RF(R) − RF(D)**, averaged over probes.",
        "- *Why:* matching alone does not show whether the repaired ridges are the real ones. Unlike PSNR / "
        "MS-SSIM / LPIPS, ρ is unaffected by brightness and contrast, so whole-image enhancers are not penalised "
        "for restyling the print, only for wrong ridges. Within a run it is almost uncorrelated with the "
        "genuine-score gain, so it adds information score 1 lacks. It is taken in the mask: "
        "outside it, paste-back methods return the input unchanged.",
        "- *Statistics:* Wilcoxon signed-rank test of the per-finger mean changes; 95% CI by bootstrap over fingers.",
        "",
        "**Cost.** Median seconds per print, peak GPU memory and peak RAM of the repair, from "
        "`experiments/benchmark.py` (same node type for every experiment, one print at a time).",
        "",
        "**How to decide.** 1. Drop methods that fail score 2. 2. Rank the rest by score 1. 3. Break ties "
        "(overlapping CIs) with score 3, then cost.",
        "",
        "### Diagnostics (sections 2-5)", "",
        "Reported for the clean (where defined), damaged and repaired image with **Δ = repaired − damaged**, "
        "signed so that positive = better, a 95% CI (normal approximation, finger-clustered standard error) and a "
        "Wilcoxon signed-rank p-value of the per-finger mean changes. "
        "PSNR (dB, peak 255), MS-SSIM (5 scales) and LPIPS (AlexNet; lower = closer) compare an image with "
        "the original, over the whole image (*full*) or the mask (*region*); *ridge fidelity full* is RF over "
        "the whole image. MINDTCT quality = mean reliability of the detected minutiae in the mask grown by "
        f"{REGION_DILATE_PX} px. NFIQ 2 = NIST's 0-100 quality score (higher = better). Original-match score = "
        "s(probe, its own original). EER = the error rate where FMR equals the false non-match rate (FNMR) "
        "when the threshold is swept; FAR/FRR are FMR/FNMR at fixed thresholds.",
        "",
        f"**Statistics, common to all.** The statistical unit is the **finger**: a finger damaged several times "
        f"(v3 data: 3 samples, each with a different contaminant) shares its clean print and gallery print, so its "
        f"samples are not independent. CIs are 95% percentile bootstrap intervals with {N_BOOT:,} resamples of "
        f"the fingers, each with all its samples ({N_BOOT // 5:,} for the EER), seeded so reruns agree; tests "
        "compare per-finger values. With one sample per finger (v2 data) all of this is the ordinary per-probe "
        "bootstrap, McNemar and Wilcoxon. Per-type numbers (about N/6 samples each) are descriptive: read their "
        "CIs, do not count their p-values as findings. Comparisons *between* experiments are in "
        "`evaluation/compare_experiments.py`.",
        ""]


def matching_table(match: pd.DataFrame, thresholds: dict) -> str:
    header = ["type", "condition", "genuine", "impostor", "EER", "genuine mean", "impostor mean", "impostor p95"] + \
        [f"{a} @{n}" for n in thresholds for a in ("FAR", "FRR")]
    rows = []
    for _, r in match.iterrows():
        pct = lambda v: "–" if v is None or np.isnan(v) else f"{v:.2%}"
        count = lambda v: "–" if v is None or np.isnan(v) else str(int(v))
        rows.append([r["type"], r["condition"], count(r.get("n_genuine", np.nan)), count(r.get("n_impostor", np.nan)),
                     pct(r.get("eer", np.nan)), fmt(r.get("gen_mean", np.nan), 1), fmt(r.get("imp_mean"), 1),
                     fmt(r.get("imp_p95"), 1)] +
                    [pct(r.get(f"{a}_{n}", np.nan)) for n in thresholds for a in ("far", "frr")])
    return md_table(header, rows)


def verdicts(headline: pd.DataFrame) -> list[str]:
    lines = []
    for _, r in headline.iterrows():
        sig = lambda lo, hi: "significant" if lo > 0 or hi < 0 else "not significant"
        lines.append(
            f"- **{r['type']}** (n = {int(r['n'])}): rescued {int(r['rescued'])}, broken {int(r['broken'])}, "
            f"net {100 * r['net_rescue']:+.1f} points ({sig(r['net_rescue_lo'], r['net_rescue_hi'])}; "
            f"acceptance {r['accept_damaged']:.0%} → {r['accept_repaired']:.0%}, clean {r['accept_clean']:.0%}); "
            f"FMR {r['fmr_damaged']:.2%} → {r['fmr_repaired']:.2%} (gate **{r['gate']}**); "
            f"ridge fidelity {r['ridge_damaged']:.3f} → {r['ridge_repaired']:.3f} "
            f"({sig(r['ridge_delta_lo'], r['ridge_delta_hi'])}).")
    return lines


def write_report(run_dir, args, samples, skipped, df, summary, match, thresholds, plots, info) -> Path:
    headline = info["headline"]
    lines = [f"# Fingerprint reconstruction evaluation", "",
             f"Run `{run_dir.name}`, {datetime.now():%Y-%m-%d %H:%M}.", "",
             "| input | path |", "|---|---|",
             f"| originals | `{args.original_dir}` |", f"| degraded | `{args.degraded_dir}` |",
             f"| reconstructions | `{args.recon_dir}` |", f"| masks | `{args.mask_dir or 'not given (derived from |orig − deg|)'}` |",
             f"| gallery | `{args.gallery or 'not given (matching against the originals)'}` |",
             f"| normalisation | {'MSU cartoon-texture, σ = ' + format(NORMALIZE_SIGMA, 'g') + ' (temporary)' if NORMALIZE_SIGMA else 'none'} |",
             "", "## Data", ""]
    counts = []
    for t in types_in(df):
        g = df[df["type"] == t]
        counts.append([t, len(g), int((g["mask_source"] == "given").sum()), int((g["mask_source"] == "derived").sum()),
                       f"{g['mask_fraction'].mean():.1%}"])
    counts.append(["all", len(df), int((df["mask_source"] == "given").sum()),
                   int((df["mask_source"] == "derived").sum()), f"{df['mask_fraction'].mean():.1%}"])
    lines += [md_table(["type", "samples", "mask given", "mask derived", "mean damaged fraction"], counts), "",
              f"{info['n_originals']} originals; matching references: "
              f"{str(info['n_refs']) + ' gallery prints' if info['gallery'] else 'the originals'}. "
              f"Degraded-side cache: {info['cache_hits']} of {len(df)} samples reused, "
              f"{info['score_hits']} cached probes reused for matching "
              f"(`{CACHE_DIR.relative_to(EVAL_DIR)}`).", ""]
    missing = [t for t in TYPES if t not in types_in(df)]
    if missing:
        lines += [f"No samples for: {', '.join(missing)}.", ""]
    if skipped:
        lines += [f"Skipped {len(skipped)} files:", ""] + [f"- `{n}`: {why}" for n, why in skipped[:30]] + \
            (["- …"] if len(skipped) > 30 else []) + [""]

    lines += definitions(thresholds, info)

    e = info["eer"]
    lines += ["## 1. Headline scores", "",
              f"At τ = {thresholds[TAU]:g} (FMR {FAR_TARGETS[0]:.0%} on clean prints). Points = percentage points. "
              "Definitions in section 0.", "",
              scores_table(headline), "",
              f"EER (diagnostic): damaged {e['eer_damaged']:.2%} → repaired {e['eer_repaired']:.2%}, change "
              f"{100 * e['eer_delta']:+.2f} points [{100 * e['eer_delta_lo']:+.2f}, {100 * e['eer_delta_hi']:+.2f}].", "",
              f"![headline scores](plots/{plots['scores']})", "",
              "**Cost**", ""] + cost_lines(info["benchmark"])

    lines += ["## 2. Image fidelity (diagnostics)", "",
              "Each repaired and damaged image against the original. *Full* = whole image; *region* = inside the "
              "damage mask (MS-SSIM averages its per-scale maps over the mask; LPIPS averages its spatial distance "
              "map over the mask).", ""]
    for m in ("ridge_region", "ridge_full", "psnr_full", "psnr_region", "msssim_full", "msssim_region",
              "lpips_full", "lpips_region"):
        lines += [f"**{LABELS[m]}**", "", metric_table(summary, m), ""]
    lines += [f"![image metrics](plots/{plots['image']})", "", f"![PSNR scatter](plots/{plots['scatter']})", "",
              f"![per-image change](plots/{plots['deltas']})", ""]

    lines += ["## 3. Fingerprint features (diagnostics)", "",
              f"MINDTCT (NBIS 5.0, default settings) on original, degraded and reconstruction. The *region* is "
              f"the damage mask grown by {REGION_DILATE_PX} px. MINDTCT quality is the per-minutia reliability "
              f"in [0, 1] from the `.min` file (higher = more reliable); it only covers minutiae that were "
              f"detected.", ""]
    for m in ("mq_mean_region", "mq_median_region", "mq_mean_full"):
        lines += [f"**{LABELS[m]}**", "", metric_table(summary, m), ""]
    lines += [f"![minutiae quality](plots/{plots['quality']})", "",
              "**Original-match score.** bozorth3 score of the degraded image and of the reconstruction "
              "against their own original; *clean* = the original matched with itself (the most that print "
              "can reach). It measures how much matchable minutiae structure survives, not a verification "
              "result: these scores are far above any decision threshold.", "",
              metric_table(summary, "orig_match_score"), "",
              f"![match scores](plots/{plots['match_scores']})", ""]

    lines += ["## 4. NFIQ 2 (diagnostic)", ""]
    if info["nfiq2"] is None:
        lines += ["`nfiq2` was not found, so NFIQ 2 was skipped. See README.md for the build.", ""]
    else:
        notes = pd.concat([df[f"nfiq2_note_{c}"] for c in ("orig", "deg", "rec")]).fillna("")
        bad = notes[notes != ""].value_counts()
        lines += ["**Convention.** NFIQ 2.x (here v2.3, NIST) reports a unified quality score from 0 to 100 where "
                  "**higher = better**. NFIQ 2 is built to predict matcher performance, so it is a diagnostic of "
                  "score 1, not a separate quality axis.", "",
                  f"**Applicability.** NFIQ 2 is defined for {STANDARD_PPI} ppi, 8-bit grayscale, plain or rolled "
                  "contact impressions. Checked per image: PNG DPI tag (missing is taken as 500 ppi), grayscale mode, "
                  "minimum size 196 px; images NFIQ 2 rejected are listed. Normalised and synthetic images are "
                  "outside its training distribution: treat it as a complementary indicator.", ""]
        lines += ([md_table(["reason not scored", "images"], [[k, int(v)] for k, v in bad.items()]), ""]
                  if len(bad) else ["All images met the checked requirements.", ""])
        lines += [metric_table(summary, "nfiq2"), ""]
        if plots.get("nfiq2"):
            lines += [f"![NFIQ 2](plots/{plots['nfiq2']})", ""]

    lines += ["## 5. Biometric matching (diagnostics)", ""]
    if info["gallery"]:
        lines += [f"**References: gallery** `{info['gallery']}` ({info['n_refs']} prints). Each probe is a separate "
                  "capture from its gallery print (enrolment vs verification), so genuine scores vary like in real "
                  "use."
                  + (f" {info['no_gallery']} samples have no gallery print and are left out of matching."
                     if info["no_gallery"] else ""), ""]
    else:
        lines += ["**References: the originals** (no `--gallery` given). Genuine pairs compare a probe with the "
                  "very impression it was made from: outside the damage the images are pixel-identical, so "
                  "genuine scores are far above the threshold. Give `--gallery` with second impressions for a "
                  "real verification test.", ""]
    lines += [f"Matcher: NBIS bozorth3 on MINDTCT `.xyt` minutiae, identical for every condition. Each probe "
              f"(original, degraded, reconstruction) is compared with its own reference (genuine) and with the "
              f"same {N_IMPOSTORS} randomly drawn other references (impostors; seed {SEED}). A probe is accepted "
              f"when score > threshold. Thresholds: fixed {FIXED_THRESHOLD:g}, and the scores giving FAR = 1% and "
              f"0.1% on the original condition's impostors "
              f"({', '.join(f'{n} = {v:g}' for n, v in thresholds.items())}); `{TAU}` is τ. EER is interpolated "
              f"linearly between the two operating points where FAR − FRR changes sign.", "",
              matching_table(match, thresholds), ""]
    if info["gallery"]:
        lines += ["**Gallery match score** (per image; clean = the original vs its gallery print)", "",
                  metric_table(summary, "match_score"), ""]
    lines += [f"![score histograms](plots/{plots['hist']})", "", f"![DET](plots/{plots['det']})", "",
              f"![error rates](plots/{plots['errors']})", "", f"![impostor shift](plots/{plots['impostor']})", ""]

    lines += ["## 6. Verdict per degradation", ""] + verdicts(headline) + [""]
    lines += [f"![examples](plots/{plots['examples']})", ""]

    m = SAMPLE_METRIC
    lines += ["## 7. Sample reconstructions", "",
              f"Three samples per degradation, picked by the per-image change in **{LABELS[m].lower()}** "
              f"(score 3's per-probe value). *Damage* = 1 − the damaged image's ridge fidelity. Candidates are the "
              f"samples between the p{SAMPLE_DAMAGE_BAND[0]} and p{SAMPLE_DAMAGE_BAND[1]} damage percentiles of "
              f"their type, so all three have about median damage; among them, the samples whose change is closest "
              f"to the type's p{', p'.join(map(str, SAMPLE_PCTS))} are shown. *Percentile* = rank among all samples "
              f"of that type. Images are shown after normalisation (what was measured).", "",
              md_table(["type", "sample", "gain", "Δ", "Δ percentile", "damage", "damage percentile",
                        f"{LABELS[m]}: degraded → reconstruction"],
                       [[p["type"], f"`{p['key']}`", p["level"], f"{p['gain']:+.3f}", f"p{p['gain_pct']:.0f}",
                         f"{p['damage']:.2f}", f"p{p['damage_pct']:.0f}",
                         f"{fmt(p[f'{m}_deg'], 3)} → {fmt(p[f'{m}_rec'], 3)}"]
                        for _, p in info["samples"].iterrows()]), "",
              f"![sample reconstructions](plots/{plots['samples']})", ""]

    lines += ["## Files", "",
              "- `headline.csv`: scores 1-3 per type and for all probes, with CIs and p-values; `probes.csv`: the "
              "per-probe values behind them (acceptance at τ, FMR, ridge fidelity)",
              "- `per_image.csv`: every metric per sample (`_orig`, `_deg`, `_rec`, `_delta` = better-signed rec − deg)",
              "- `summary.csv`: per type and metric summary; `matching.csv`: matching summary; "
              "`scores.csv`: every bozorth3 comparison; `config.json`: settings, thresholds, tool hashes, EER change "
              "and the benchmark", ""]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


# ================================================================ main
def add_score_columns(df: pd.DataFrame, scores: pd.DataFrame, label: str, metric: str) -> None:
    """<metric>_deg/_rec/_orig columns from the bozorth3 scores with `label` (one per probe)."""
    sel = scores[scores["label"] == label]
    per_key = sel[sel["probe"] != "original"].pivot_table(index="key", columns="probe", values="score")
    per_id = sel[sel["probe"] == "original"].groupby("id")["score"].first()
    for cond, short in (("degraded", "deg"), ("reconstruction", "rec")):
        df[f"{metric}_{short}"] = df["key"].map(per_key[cond]) if cond in per_key else np.nan
    df[f"{metric}_orig"] = df["id"].map(per_id)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("original_dir", type=Path, help="originals: <id>.png")
    p.add_argument("degraded_dir", type=Path, help="degraded: <id>_<type>.png")
    p.add_argument("recon_dir", type=Path, help="reconstructions: <id>_<type>_rec.png")
    p.add_argument("mask_dir", type=Path, nargs="?", default=None,
                   help="optional masks: <id>_<type>_mask.png (missing ones are derived from |orig - deg|)")
    p.add_argument("--gallery", type=Path, default=None,
                   help="optional second impressions <id>.png, used as matching references instead of the originals")
    p.add_argument("--benchmark", type=Path, default=None,
                   help="optional benchmark.json of the method (experiments/benchmark.py), shown as its cost")
    p.add_argument("--rec-normalized", action="store_true",
                   help="the reconstructions are already MSU-normalised (the method works on normalised prints): "
                        "measure them as stored, normalise everything else")
    return p.parse_args()


def settings_dict(tools: dict) -> dict:
    return {"psnr_cap_db": PSNR_CAP_DB, "msssim": [MSSSIM_WEIGHTS, MSSSIM_WINDOW, MSSSIM_SIGMA],
            "lpips_net": LPIPS_NET, "diff_thresh": DIFF_THRESH,
            "mask": [MASK_CLOSE_PX, MASK_MIN_AREA_PX, MASK_DILATE_PX], "region_dilate_px": REGION_DILATE_PX,
            "n_impostors": N_IMPOSTORS, "seed": SEED, "normalize_sigma": NORMALIZE_SIGMA,
            "ridge": [RIDGE_BAND_SIGMAS, RIDGE_WINDOW_SIGMA],
            "tools": {name: tool_hash(path) for name, path in tools.items()}}


def main() -> None:
    args = parse_args()
    for name in ("original_dir", "degraded_dir", "recon_dir", "gallery"):
        folder = getattr(args, name)
        if folder is not None and not folder.is_dir():
            sys.exit(f"{name}: {folder} is not a folder")
    if args.mask_dir is not None and not args.mask_dir.is_dir():
        print(f"mask folder {args.mask_dir} not found: every mask will be derived from |orig - deg|")
    tools = {name: find_tool(name) for name in ("mindtct", "bozorth3", "nfiq2")}
    if not tools["mindtct"] or not tools["bozorth3"]:
        sys.exit(f"mindtct/bozorth3 not found in {TOOL_DIRS[0]} or on PATH; see {EVAL_DIR / 'README.md'}")
    samples, originals, skipped = find_samples(args.original_dir, args.degraded_dir, args.recon_dir, args.mask_dir)
    if not samples:
        sys.exit("no complete (original, degraded, reconstruction) triplets found")
    settings = settings_dict(tools)
    settings_hash = text_sha1(json.dumps(settings, sort_keys=True))
    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True)
    print(f"{len(samples)} samples, {len(originals)} originals -> {run_dir}")

    gallery_mode = args.gallery is not None
    if gallery_mode:
        refs = {p.stem: p for p in sorted(args.gallery.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
        if not refs.keys() & {s["id"] for s in samples}:
            sys.exit(f"no gallery print in {args.gallery} matches an original id")
    else:
        refs = originals
    if NORMALIZE_SIGMA:
        originals, refs = normalize_inputs(samples, originals, refs, run_dir, args.rec_normalized)
    orig_sha = {fid: file_sha1(p) for fid, p in originals.items()}
    ref_sha = {fid: file_sha1(p) for fid, p in refs.items()} if gallery_mode else orig_sha
    ref_ids = sorted(refs)
    if len(ref_ids) < 2:
        sys.exit("matching needs at least 2 references (originals or gallery prints) for impostor pairs")
    ref_sig = text_sha1("gallery" if gallery_mode else "originals", *[f"{g}:{ref_sha[g]}" for g in ref_ids],
                        str(N_IMPOSTORS), str(SEED))
    for s in samples:
        s["deg_sha"], s["rec_sha"] = file_sha1(s["deg_path"]), file_sha1(s["rec_path"])
        mask_sha = file_sha1(s["mask_path"]) if s["mask_path"] else "derived"
        s["cache_hash"] = text_sha1(settings_hash, orig_sha[s["id"]], s["deg_sha"], mask_sha)
        s["impostors"] = impostor_ids(s["id"], ref_ids)

    cached_rows, cached_scores, nfiq_cache = load_cache(settings_hash)
    roots = run_tools(samples, originals, orig_sha, refs, ref_sha, tools, run_dir, nfiq_cache)
    df, cache_rows, cache_hits = evaluate_samples(samples, roots, cached_rows, orig_sha)
    skipped += [(s["deg_path"].name, s["error"]) for s in samples if "error" in s]
    no_gallery = [s for s in samples if "error" not in s and s["id"] not in refs]
    skipped += [(s["deg_path"].name, "no gallery print (left out of matching only)") for s in no_gallery]
    scores, cache_scores, score_hits = run_matching(samples, originals, orig_sha, refs, gallery_mode, roots,
                                                    tools, cached_scores, ref_sig)
    save_cache(settings, settings_hash, cache_rows, cache_scores, nfiq_cache)

    add_score_columns(df, scores, "genuine", "match_score" if gallery_mode else "orig_match_score")
    if gallery_mode:
        add_score_columns(df, scores, "original", "orig_match_score")
    df = add_deltas(df)

    base_imp = condition_scores(scores, "original", "all", "impostor")
    thresholds = {"t40": float(FIXED_THRESHOLD), "far1": threshold_at_far(base_imp, FAR_TARGETS[0]),
                  "far01": threshold_at_far(base_imp, FAR_TARGETS[1])}
    summary = per_image_summary(df)
    match = matching_summary(scores, thresholds)
    probes = probe_table(scores, df, thresholds[TAU])
    headline = headline_scores(probes)
    eer_info = eer_change(scores)
    benchmark = json.loads(args.benchmark.read_text()) | {"file": str(args.benchmark)} \
        if args.benchmark is not None and args.benchmark.exists() else None
    df.to_csv(run_dir / "per_image.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    match.to_csv(run_dir / "matching.csv", index=False)
    scores.to_csv(run_dir / "scores.csv", index=False)
    probes.to_csv(run_dir / "probes.csv", index=False)
    headline.to_csv(run_dir / "headline.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps(settings | {
        "thresholds": thresholds, "tau": TAU, "fmr_gate": FMR_GATE, "n_boot": N_BOOT, "eer_change": eer_info,
        "benchmark": benchmark, "rec_normalized": args.rec_normalized, "tool_paths": tools}, indent=1))

    picks = select_samples(df)
    set_style()
    plots = {"scores": plot_scores(headline, plots_dir), "deltas": plot_deltas_per_image(df, plots_dir),
             "image": plot_image_metrics(df, plots_dir), "scatter": plot_psnr_scatter(df, plots_dir),
             "match_scores": plot_match_scores(df, plots_dir),
             "quality": plot_minutiae_quality(df, plots_dir), "nfiq2": plot_nfiq2(df, plots_dir),
             "hist": plot_score_hist(scores, thresholds, plots_dir), "det": plot_det(scores, match, plots_dir),
             "errors": plot_error_bars(match, plots_dir),
             "impostor": plot_impostor_shift(scores, thresholds, plots_dir),
             "examples": plot_examples(df, samples, roots, plots_dir),
             "samples": plot_samples(picks, samples, plots_dir)}
    info = {"n_originals": len(originals), "cache_hits": cache_hits, "score_hits": score_hits, "samples": picks,
            "nfiq2": tools["nfiq2"], "gallery": args.gallery, "n_refs": len(refs), "no_gallery": len(no_gallery),
            "headline": headline, "eer": eer_info, "benchmark": benchmark,
            "rec_normalized": args.rec_normalized}
    report = write_report(run_dir, args, samples, skipped, df, summary, match, thresholds, plots, info)
    print(f"report: {report}")


if __name__ == "__main__":
    main()
