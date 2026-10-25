"""Every path, size, seed and threshold used in the repo lives here.

Imported by all `fpr` modules and scripts. No other file hard-codes a path
or a magic number; change things here and nowhere else.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- data version
# Which damage / val-test version every script reads and writes, and records with every leaderboard row.
# v2 = one damaged copy of every val/test print and train crop; v3 = the same damage, 3 damaged copies
# (3 contaminants). (v1, a milder first damage model, is retired.) Pick it with the environment variable, e.g.
#     FPR_DATA_VERSION=v3 python experiments/run_experiment.py --exp E12_norm_ccdf_cldice_cv --split test
DATA_VERSIONS = ("v2", "v3")
DATA_VERSION = os.environ.get("FPR_DATA_VERSION", "v2")
assert DATA_VERSION in DATA_VERSIONS, f"FPR_DATA_VERSION must be one of {DATA_VERSIONS}, got {DATA_VERSION!r}"
DAMAGE_COPIES = {"v2": 1, "v3": 3}[DATA_VERSION]      # damaged copies per val/test print and per train crop
                                                      # and contaminant (scripts 05 and 08)

# ---------------------------------------------------------------- paths
# Pipeline: preprocessing/ -> fpr/ (library) -> experiments/ -> evaluation/ -> demo/.
REPO_ROOT = Path(__file__).resolve().parent

# preprocessing/: raw data (symlinks), the scripts that prepare it, the prepared data.
PREPROCESSING_DIR = REPO_ROOT / "preprocessing"
# Where the downloaded raw data lives (never written to): FPR_RAW_DATA, default ../data next to the repo.
RAW_DATA_SOURCE = Path(os.environ.get("FPR_RAW_DATA", REPO_ROOT.parent / "data"))
RAW_DATA_DIR = PREPROCESSING_DIR / "raw_data"        # symlinks into RAW_DATA_SOURCE
RAW_DATASETS = ("NIST", "SOCOFing")
NIST_DIR = RAW_DATA_DIR / "NIST"                     # read-only
PARTICIPANTS_CSV = NIST_DIR / "participants.csv"
DATA_DIR = Path(os.environ.get("FPR_DATA_DIR", PREPROCESSING_DIR / "data"))   # everything generated (git-ignored)
INDEX_CSV = DATA_DIR / "index.csv"
SPLIT_CSV = DATA_DIR / "split.csv"
VERSION_DIR = DATA_DIR if DATA_VERSION == "v2" else DATA_DIR / DATA_VERSION   # val/, test/, train/ damaged set
TRAIN_DIR = DATA_DIR / "train"                       # shared by every version: clean/, clean_norm/, minutiae_w/
TRAIN_CLEAN_DIR = TRAIN_DIR / "clean"                # pool.csv + K/ L/ M/ P/ U/ V/: clean finger crops
TRAIN_CLEAN_NORM_DIR = TRAIN_DIR / "clean_norm"      # the same crops MSU-normalised (script 10)
TRAIN_SET_DIR = VERSION_DIR / "train"                # the fixed damaged set of this version (= TRAIN_DIR for v2)
TRAIN_DAMAGED_DIR = TRAIN_SET_DIR / "damaged"        # <sample>_degraded.png, <sample>_mask.png (fixed set)
TRAIN_DAMAGED_NORM_DIR = TRAIN_SET_DIR / "damaged_norm"   # <sample>_degraded.png MSU-normalised (script 10)
TRAIN_META_CSV = TRAIN_SET_DIR / "meta.csv"          # one row per fixed damaged sample
TRAIN_MINUTIAE_DIR = TRAIN_DIR / "minutiae_w"        # <clean file>.npy: MINDTCT minutiae maps (script 09)
# val/ and test/ (VERSION_DIR / split) each hold: meta.csv, pairs.csv, probes/, probes_norm/, gallery/, layout/
PROBES_SUBDIR = "probes"                             # <sample>_{clean,degraded,mask}.png
PROBES_NORM_SUBDIR = "probes_norm"                   # <sample>_{clean,degraded}.png MSU-normalised (script 10)
GALLERY_SUBDIR = "gallery"                           # copies of the probes' raw slap prints (NIST names)
LAYOUT_SUBDIR = "layout"                             # the same probes in evaluation/evaluate.py's naming
META_CSV = "meta.csv"
PAIRS_CSV = "pairs.csv"
PREVIEW_FIGURE = DATA_DIR / "figures" / "degradation_preview.png"
DATA_MD = PREPROCESSING_DIR / "DATA.md"              # data counts, written by scripts 02-08

# experiments/: one folder per experiment, grouped (experiments/<group>/EXX_name/).
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENT_GROUPS = ("main", "baselines", "ablations", "one_step")
REFERENCE_PARAMS = EXPERIMENTS_DIR / "main" / "E02_resshift" / "params.json"   # ResShift defaults (fpr/models/schedule.py)

# evaluation/: the main evaluator (evaluation/evaluate.py) and every experiment's results.
EVALUATION_DIR = REPO_ROOT / "evaluation"
EVALUATE_SCRIPT = EVALUATION_DIR / "evaluate.py"
EVAL_RESULTS_DIR = EVALUATION_DIR / "results"        # results/<experiment>/<split>_<timestamp>/
LEADERBOARD_MD = EVAL_RESULTS_DIR / "LEADERBOARD.md"
MATCH_THRESHOLD_JSON = EVAL_RESULTS_DIR / "match_threshold.json"   # used by the demo
COMPARISON_DIR = EVAL_RESULTS_DIR / "comparison"     # cross-experiment figures
CACHE_DIR = EVALUATION_DIR / "cache"                 # fpr-side caches (git-ignored)
XYT_CACHE_DIR = CACHE_DIR / "xyt"
DEMO_CACHE_DIR = CACHE_DIR / "demo"
# NBIS build (evaluation/README.md): FPR_NBIS_BIN, default evaluation/third_party/nbis/bin, else PATH.
NBIS_BIN_DIR = Path(os.environ.get("FPR_NBIS_BIN", EVALUATION_DIR / "third_party" / "nbis" / "bin"))

# ---------------------------------------------------------------- raw data layout
# (folder relative to NIST_DIR, capture label used in index.csv)
# Only 500 ppi folders. Full multi-finger slaps (codes 13/14/15) are excluded
# by using the slap-segmented folders only.
RAW_FOLDERS = [
    ("sd302d/images/auxiliary/flat/K/500/plain/png", "plain"),
    ("sd302d/images/auxiliary/flat/L/500/plain/png", "plain"),
    ("sd302d/images/auxiliary/flat/M/500/plain/png", "plain"),
    ("sd302d/images/auxiliary/flat/P/500/plain/png", "plain"),
    ("sd302b/images/baseline/U/500/roll/png", "rolled"),
    ("sd302b/images/baseline/V/500/roll/png", "rolled"),
    ("sd302b/images/baseline/R/500/slap-segmented/png", "slap"),
    ("sd302b/images/baseline/S/500/slap-segmented/png", "slap"),
]
# Image counts per (device, capture), from the NIST documentation.
EXPECTED_COUNTS = {
    ("K", "plain"): 930, ("L", "plain"): 1953, ("M", "plain"): 1979,
    ("P", "plain"): 279, ("U", "rolled"): 2000, ("V", "rolled"): 2000,
    ("R", "slap"): 920, ("S", "slap"): 1080,
}
PPI = 500
N_FINGERS = 10

# ---------------------------------------------------------------- split
SPLIT_SEED = 643
N_TRAIN, N_VAL, N_TEST = 140, 20, 40
PROBE_DEVICE = "L"            # plain-print scanner used for val/test probes
MIN_EVAL_FINGERS = 9          # val/test people need L + slap for >= this many fingers

# ---------------------------------------------------------------- images
BACKGROUND_VALUE = 255        # white paper / platen background

# ---------------------------------------------------------------- foreground
FG_BLOCK = 16                 # block size (px) for block-variance segmentation
FG_REFERENCE_PERCENTILE = 90  # block std percentile taken as "typical ridge contrast"
FG_RELATIVE_THRESHOLD = 0.4   # foreground = block std > this fraction of the reference
FG_MIN_STD = 0.03             # ...but never below this (image scaled to [0, 1])
FG_OPEN_BLOCKS = 2            # opening radius in blocks: cuts thin bridges (joint strip)
FG_CLOSE_BLOCKS = 2           # closing radius in blocks: fills gaps between ridges
JOINT_SEARCH_FROM = 0.6       # rolled: look for the crease below this fraction of mask height
JOINT_DARKNESS_RATIO = 0.7    # crease row is lighter than this fraction of a typical row
JOINT_SMOOTH_PX = 9           # smoothing of the row-darkness profile (px)

# ---------------------------------------------------------------- training pool / patches
CROP_MARGIN_PX = 8            # margin kept around the foreground box in the pool
PATCH_SIZE = 128
PATCH_MIN_FG = 0.7            # a patch must be at least this fraction foreground
PATCH_MAX_TRIES = 50

# ---------------------------------------------------------------- degradations
# Modelled on the contaminants of Kumari et al. (2021) on an optical scanner.
# All sizes in pixels at 500 ppi (ridge period is about 9-10 px; an L fingertip is
# about 450 x 320 px). Pairs (a, b) are the values at severity 0 and severity 1.
SEVERITY_RANGE_TRAIN = (0.0, 0.8)   # severity 1 is too harsh
BLOB_ROUGHNESS = 0.35         # how far noise bends the blob outline (fraction of radius)
BLOB_NOISE_SIGMA_PX = 12      # smoothness of the blob outline noise
BLOB_EDGE = 0.25              # soft edge width (fraction of radius)
MASK_ALPHA_MIN = 0.15         # a pixel is "damaged" where the blend weight exceeds this

# Shared helpers (shapes.py)
RIDGE_PERIOD_PX = 9           # window for the local valley (background) level
DROPOUT_SIGMA_PX = 2.0        # grain of ridge dropout: gaps are about 2-5 px long
DROPOUT_GREY_MAX = 232        # dropped-out ridge pixels are never brighter than this (no pure white)
WARP_COVER_PX = 1.0           # a pixel displaced by >= this is fully inside the damage weight
MORPH_SMOOTH_SIGMA_PX = 1.0   # light smoothing after a disk min/max filter (removes round dot artefacts)

# pressure: 1-2 large Gaussian lobes of hard or light contact
PRESSURE_LOBES = (1, 2)
PRESSURE_LOBE_RADIUS_PX = (70, 135)    # lobe weight falls to MASK_ALPHA_MIN here; 25-40% coverage at severity 1
PRESSURE_LOBE_ASPECT = (0.6, 1.0)      # minor / major axis of the elliptical lobe
PRESSURE_HARD_PROB = 0.5
PRESSURE_CLOSE_PX = (1, 3)    # hard: ridge closing radius (valleys narrower than ~2r fill)
PRESSURE_DARKEN = (0.9, 0.75) # hard: grey-level multiplier
PRESSURE_WARP_PX = (1.0, 2.0) # hard: displacement amplitude (skin squashed against the platen)
PRESSURE_WARP_SIGMA_PX = 10   # hard: smoothness of the displacement (~45 px wavelength)
PRESSURE_THIN_PX = (1, 2)     # light: ridge thinning radius
PRESSURE_LIGHT_FADE = (0.85, 0.65)     # light: ridge darkness kept
PRESSURE_DROPOUT = (0.2, 0.55)         # light: dropout share at the lobe edge (0 at its centre)

# scar: long, wide, ridge-free strokes with the ridges sheared across them
SCAR_COUNT_MIN = (1, 3)       # fewest strokes
SCAR_COUNT_MAX = (2, 5)       # most strokes
SCAR_LENGTH_PX = (100, 400)
SCAR_WIDTH_PX = (6, 20)       # widest point of a stroke (it tapers to 40% at the ends)
SCAR_BEND = 0.35              # Bezier control-point offset (fraction of length)
SCAR_TONE_PROBS = {"flat": 0.6, "bright": 0.2, "dark": 0.2}
SCAR_MEAN_SIGMA_PX = 12       # flat scar: interior = local mean grey of the print
SCAR_TEXTURE = 5              # grey-level std of the faint interior texture
SCAR_BRIGHT_GREY = (210, 235) # healed scar (never pure white)
SCAR_DARK_GREY = (30, 80)     # fresh cut or dirt-filled crease
SCAR_SHEAR_PX = (2.0, 4.0)    # ridge offset between the two sides of a stroke
SCAR_SHEAR_REACH = 1.5        # shear fades out this many stroke widths beyond the stroke edge

# oil: ridges broaden and merge over most of the fingertip
OIL_BLOBS = (1, 3)
OIL_BLOB_RADIUS_PX = (90, 220)
OIL_VISCOUS_PROB = 0.5        # viscous (thick) oil is strong, thin oil is mild
OIL_VISCOUS_RADIUS_PX = (1, 3)   # grey erosion radius: dark ridges spread into the valleys
OIL_THIN_RADIUS_PX = 1
OIL_DARKEN_VISCOUS = (0.9, 0.7)  # grey-level multiplier under the thickest film
OIL_DARKEN_THIN = 0.95
OIL_FILM_SIGMA_PX = 8         # smoothness of the uneven film thickness
OIL_FILM_MIN = 0.55           # blend weight where the film is thinnest

# lotion: clumps that fill the valleys so neighbouring ridges bridge
LOTION_COVERAGE = (0.10, 0.50)   # share of the foreground inside clumps
LOTION_CLUMP_SIGMA_PX = 10    # smoothness of the random field thresholded into clumps
LOTION_EDGE = 0.3             # field units over which a clump edge ramps up to full strength
LOTION_CLOSE_PX = (2, 4)      # ridge closing radius inside the clumps
LOTION_DARKEN = (0.95, 0.85)

# sanitizer: dry skin, faint broken ridges, up to the whole fingertip
SANITIZER_RADIUS_PX = (80, 450)          # at severity 1 the blob covers the whole fingertip
SANITIZER_THIN_PX = (0, 1)
SANITIZER_BREAK_FRACTION = (0.25, 0.65)  # mean share of pixels turned into gaps
SANITIZER_BREAK_MODULATION = 0.5         # low-frequency field changes the gap share by +-50%
SANITIZER_BREAK_LF_SIGMA_PX = 25         # smoothness of that field
SANITIZER_BREAK_MAX = 0.8                # local gap share never exceeds this (some ridge always left)
SANITIZER_FADE = (0.8, 0.5)              # ridge darkness kept
SANITIZER_GAP_GREY = 200                 # gap cap: broken ridges stay visible as light dashes

# mehendi: a henna design (plus sign or crossing strokes) over the ridges
MEHENDI_PLUS_PROB = 0.5       # plus sign; otherwise 2-3 crossing tapered strokes
MEHENDI_DESIGN_SIZE_PX = (120, 280)   # arm-to-arm length of the design
MEHENDI_DESIGN_WIDTH_PX = (20, 45)
MEHENDI_STROKES = (2, 3)
MEHENDI_CONTRAST_DROP = (0.5, 0.7)    # share of ridge contrast lost inside the design
MEHENDI_DROPOUT = (0.3, 0.6)          # share of pixels turned into gaps
MEHENDI_STAIN_DARKEN = (0.15, 0.3)    # dye darkens the design area
MEHENDI_WARP_PX = (2.0, 4.0)          # displacement amplitude: ridge spacing changes
MEHENDI_WARP_SIGMA_PX = 8             # ~35 px wavelength
MEHENDI_VINE_PROB = 0.5               # also draw dark vines over the finger
MEHENDI_VINES = (1, 4)
MEHENDI_VINE_LENGTH_PX = (60, 260)
MEHENDI_WIDTH_PX = (3, 8)
MEHENDI_INK = 45              # grey level of henna paste lines

# ---------------------------------------------------------------- fixed val/test sets
EVAL_SPLITS = ("val", "test")
EVAL_SEED = 2026              # combined with participant id and finger, one rng per probe
EVAL_SEVERITY_RANGE = (0.4, 0.8)   # default of script 05 --severity (v2 and v3 eval sets)
# DATA_VERSION and DAMAGE_COPIES: top of this file

# ---------------------------------------------------------------- MSU normalisation
NORMALIZE_SIGMA = 2.5         # texture part of MSU's FastCartoonTexture (fpr/data/normalization.py, evaluate.py)
NORMALIZE_BACKGROUND = 128    # grey level of a flat region after normalisation (pad value for normalised images)
NORMALIZE_WORKERS = 32        # processes used by preprocessing/10_normalize.py

# ---------------------------------------------------------------- orientation field (Hong, Wan & Jain 1998)
ORIENT_BLOCK = 16             # block size (px) for the orientation estimate
ORIENT_GRAD_SIGMA = 1.0       # Gaussian derivative scale (px)
ORIENT_SMOOTH_SIGMA = 2.0     # smoothing of doubled-angle field (in blocks)
ORIENT_FILL_SIGMA = 3.0       # fill-in smoothing into damaged region (in blocks)

# ---------------------------------------------------------------- image metrics
LPIPS_NET = "alex"
PSNR_CAP_DB = 100.0           # PSNR reported for a pixel-perfect region (instead of inf)
METRIC_BOX_MARGIN_PX = 8      # margin around the damaged-region box for LPIPS
MIN_BOX_SIDE_PX = 32          # LPIPS box is grown to at least this size

# ---------------------------------------------------------------- minutiae / NBIS
# The build in NBIS_BIN_DIR is used if present, else whatever is on PATH.
MINDTCT = str(NBIS_BIN_DIR / "mindtct") if (NBIS_BIN_DIR / "mindtct").exists() else "mindtct"
BOZORTH3 = str(NBIS_BIN_DIR / "bozorth3") if (NBIS_BIN_DIR / "bozorth3").exists() else "bozorth3"
NFIQ2 = "nfiq2"               # optional; evaluation/evaluate.py runs NFIQ 2 itself
MINUTIA_MATCH_DIST_PX = 10
MINUTIA_MATCH_ANGLE_DEG = 20
MINUTIA_REGION_DILATE_PX = 8  # minutiae are compared inside the damaged region grown by this
BOZORTH_BATCH = 500           # gallery files per bozorth3 call

# ---------------------------------------------------------------- verification
FAR_TARGETS = (0.001, 0.01)
DEMO_FAR = 0.01               # demo MATCH / NO MATCH threshold = val FAR of 1%

# ---------------------------------------------------------------- reconstructors
TILE_PATCH = 128              # full-image tiling for patch-trained models
TILE_OVERLAP = 32
FEATHER_PX = 3                # paste-back seam: blend ramps over this many px inside the mask
GABOR_NORM_MEAN = 100.0       # Hong et al. normalisation target mean and variance
GABOR_NORM_VAR = 100.0
GABOR_SIGMA_PX = 4.0          # Gaussian envelope of the Gabor filter (sigma_x = sigma_y)
GABOR_N_ORIENT = 16           # orientation bins of the filter bank
GABOR_N_FREQ = 4              # frequency bins of the filter bank
GABOR_LOCAL_SIGMA_PX = 16     # window for local contrast normalisation of the response
GABOR_CONTRAST = 1.5          # tanh gain: higher = closer to a binary ridge map
FREQ_WINDOW_PX = (32, 16)     # x-signature window: (across ridges, along ridges)
FREQ_PERIOD_RANGE_PX = (3, 25)   # valid ridge period at 500 ppi

# ---------------------------------------------------------------- skeleton and losses
BINARIZE_WINDOW_PX = 25       # Sauvola local threshold window (about 2.5 ridge periods)
SOFT_SKELETON_ITERS = 10      # soft-skeleton iterations (Shit et al. 2021)

# ---------------------------------------------------------------- datasets
EPOCH_SIZE = 50_000
TRAIN_SEED = 0
FIXED_PATCH_ON_DAMAGE = 0.9   # FixedTrainDataset: share of patches centred on damage

# ---------------------------------------------------------------- demo
DEMO_SPLIT = "test"
DEMO_PORT = 7860

# ---------------------------------------------------------------- report figures
# Validated categorical order: ceiling and floor keep slots 1-2; experiments take the rest in
# experiment-number order, so a colour follows its experiment and is never re-assigned by rank.
CEILING_COLOR, FLOOR_COLOR = "#2a78d6", "#eb6834"      # clean print, damaged probe
EXPERIMENT_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GENUINE_COLOR, IMPOSTOR_COLOR = "#2a78d6", "#eb6834"
INK, MUTED_INK, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIGURE_DPI = 200

# ---------------------------------------------------------------- fixed damaged train set
TRAIN_SET_SEED = 2027         # with the pool row and contaminant index: one rng per sample
TRAIN_SET_SEVERITY_RANGE = (0.0, 0.8)   # same range as on-the-fly training damage
TRAIN_SET_WORKERS = 16        # processes used by preprocessing/08_make_train_set.py

# ---------------------------------------------------------------- minutiae maps of the train crops (script 09)
MINUTIAE_MAP_SIGMA_PX = 8.0       # Gaussian bump around each MINDTCT minutia (E06-E08 spec)
MINUTIAE_MAP_HALF_WINDOW_PX = 17  # filter truncated at this radius: truncate = 17 / 8 sigmas
MINUTIAE_MAP_WORKERS = 12         # processes used by preprocessing/09_make_minutiae_maps.py

