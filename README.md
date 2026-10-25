# Halluci-Nation: repairing contaminated fingerprints so they match again

IE 643 (Deep Learning, IIT Bombay) course project, team **Halluci-Nation**: Sarthak Somani (24B3006),
Raheel Aggarwal (24B2478).

Mehendi, oil, lotion, sanitizer, scars and uneven pressure damage part of a fingerprint, and the print then
fails at Aadhaar-style verification. We train a lightweight residual-shifting diffusion model (ResShift, 8.1M
parameters) that repairs **only the damaged region** (every pixel outside the damage mask is kept bit for bit)
and sample it with CCDF's warm start. We judge a repair by **matching**: does the repaired print match its owner
again (NBIS BOZORTH3 against a clean enrolment print), without starting to match strangers? Ridge fidelity
comes next; image-quality metrics are diagnostics. A SegFormer segmenter predicts the damage mask when no
ground truth exists, and a Gradio demo runs the whole chain live.

## Pipeline

```
NIST SD 302, 500 ppi (200 people)
  │  preprocessing/00-10   index -> split by participant (140 / 20 / 40) -> finger crops -> synthetic contaminants
  ▼                        -> fixed damaged val/test probes + clean gallery + genuine/impostor pairs
fpr/data                   128 px patches: clean, damaged, mask, ridge-orientation field
  ▼
fpr/models                 conditional U-Net (8.1M) + ResShift residual-shift schedule; samplers: ResShift chain, CCDF
  ▼
fpr/training               L_diff (x0-prediction MSE) + ridge losses on x0_hat (minutiae-weighted L1, clDice, ...);
  │                        AdamW 2e-4, warm-up + cosine, EMA, grad clip
  ▼
fpr/validation             val loss / val EER during training; CCDF configuration (start, t0, projection) chosen on val
  ▼
evaluation/                MSU-normalise -> MINDTCT + BOZORTH3 vs clean gallery -> net rescue, false-match gate,
  │                        ridge fidelity (bootstrap CIs, paired tests)
  ▼
demo/                      Gradio: damage a print -> segmenter/ predicts the mask -> repair -> MATCH / NO MATCH
```

## Repository map

| Path | What |
|---|---|
| `config.py` | every path, size, seed and threshold; data version (`FPR_DATA_VERSION`), data paths (`FPR_RAW_DATA`, `FPR_DATA_DIR`) |
| `preprocessing/` | numbered scripts `00`–`10`: link raw data, index, split, crops, damage preview, eval sets, pairs, train set, minutiae maps, normalisation |
| `fpr/` | the library, one sub-package per stage: `data/` → `models/` → `training/` → `validation/`, plus `metrics/` (image metrics, NBIS matching) and `experiments.py` (finds experiments, builds their methods) |
| `experiments/` | one folder per experiment (settings in `params.json`, selected sampler in `selected.json`), grouped as `main/` (the final models and what they are built from), `baselines/`, `ablations/`, `one_step/`; shared entry points `train.py`, `run_experiment.py`, `benchmark.py` |
| `evaluation/` | the evaluator (`evaluate.py`), `run_eval.py`, `compare_experiments.py`, and the result tables in `results/` |
| `segmenter/` | SegFormer-B0 damage-mask predictor (fine-tuned) |
| `demo/` | the Gradio test interface for the live demo |
| `scripts/` | `smoke_test.py` (CPU, no data) and `slurm/` job scripts |
| `tests/` | pytest suite (data-dependent tests skip when the data is absent) |

Why this layout: Python packages cannot start with a digit, so the pipeline order is carried by the library's
sub-packages (`fpr/data` → `models` → `training` → `validation`) and by the numbered preprocessing scripts.
Experiments hold only settings and the little code that is truly theirs; everything they share lives in `fpr/`.

## Quick start

```bash
# 1. install (Python >= 3.10; a CUDA GPU for training, CPU is enough for the smoke test and the demo)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# NBIS (mindtct, bozorth3) and NFIQ 2 are compiled NIST tools: build them as in evaluation/README.md.

# 2. smoke test, no data needed: damage a synthetic print, repair it with E22, score it
python scripts/smoke_test.py

# 3. data: download NIST Special Database 302 (sd302b baseline + sd302d auxiliary, with participants.csv) from
#    https://www.nist.gov/itl/iad/image-group/nist-special-database-302  into  ../data/NIST/
#    (or set FPR_RAW_DATA to the folder that holds NIST/). Then, in order:
for s in preprocessing/0*.py preprocessing/10_*.py; do python "$s"; done   # scripts 00-10 in order, v2 data
for s in 05 06 07 08 10; do FPR_DATA_VERSION=v3 python preprocessing/${s}_*.py; done   # v3: 3 damaged copies

# 4. train (GPU, ~12 h each on an A40) and choose the CCDF sampler on val
python experiments/train.py --exp E02_resshift                       # ResShift, plain diffusion loss
python experiments/train.py --exp E07_aux_minutiae_cldice            # + minutiae L1 + clDice
python experiments/train.py --exp E09_ccdf                           # CCDF sweep on val for E02 -> selected.json
python -m fpr.validation.ccdf_sweep --exp E22_ccdf_aux_minutiae_cldice   # CCDF sweep on val for E07

# 5. repair the test probes and evaluate (v3 test = the headline numbers)
FPR_DATA_VERSION=v3 python experiments/run_experiment.py --exp E22_ccdf_aux_minutiae_cldice --split test
FPR_DATA_VERSION=v3 python evaluation/run_eval.py --exp E22_ccdf_aux_minutiae_cldice --split test

# 6. demo (http://127.0.0.1:7860; on a cluster see demo/README.md)
python demo/gradio_app.py
```

`scripts/slurm/` has one-job versions of these chains (`submit_chain.sh`, `v3_chain.sh`, `resample_chain.sh`).
Checkpoints are not in the repository (git-ignored); every trainer writes them to `<experiment>/checkpoints/`.

## Results

v3 test set: 1,194 damaged samples of 398 fingers of 40 unseen people. **Net rescue** = (probes rejected when
damaged and accepted after repair − the reverse) / N, at the matcher threshold τ for a 1% false-match rate.
**Gate** = the repair must not raise the false-match rate (upper 95% bound of ΔFMR < +0.5 points).
**Ridge fidelity** = local correlation of repaired and clean ridges inside the damage mask (damaged: 0.826).
Final models: mean of 9 runs each (3 training seeds × 3 sampler seeds), hierarchical bootstrap 95% CIs
([E27 results](experiments/main/E27_seed_study/results.md)).

| Method | Net rescue, points (95% CI) | Gate | Ridge fidelity (95% CI) | s / print |
|---|---|---|---|---|
| Damaged print, no repair | 0 | – | 0.826 | – |
| **E04** MSU-LatentAFIS STFT enhancement (main baseline) | +2.18 [+0.34, +4.10] | pass | 0.820 [0.813, 0.827] | 0.22 (CPU) |
| **E09** ResShift + CCDF (final) | **+5.45** [+3.86, +7.12] | 9 / 9 pass | **0.943** [0.939, 0.946] | 0.48 (A40) |
| **E22** ResShift + minutiae/clDice losses + CCDF (final) | **+5.61** [+4.07, +7.25] | 9 / 9 pass | **0.943** [0.940, 0.946] | 0.43 (A40) |

- **Both final models beat the E04 baseline on every seed** (9 / 9 runs, 3 / 3 training seeds, on both scores):
  E09 − E04 = +3.28 points [+1.67, +4.99], E22 − E04 = +3.43 [+1.79, +5.18]; ridge fidelity +0.123 for both.
- **E09 and E22 are tied:** E22 − E09 = +0.16 points [−0.72, +1.00] and +0.0002 ridge fidelity [−0.0005, +0.0009].
  E22 is slightly cheaper.
- A repair can gain at most about 6.5 points here (clean prints are accepted at 78.9%, damaged ones at 72.4% on v2).
- Every experiment, the per-pair tests and the costs: [experiments/README.md](experiments/README.md),
  [evaluation/results/LEADERBOARD.md](evaluation/results/LEADERBOARD.md),
  [comparison (v3)](evaluation/results/comparison/v3/test/comparison.md),
  [experimental log](experiments/experimental_log.md).

## Code walkthrough guide

| Topic (course list) | Where |
|---|---|
| Dataset collection | `preprocessing/00_link_raw_data.py`, `01_index_dataset.py`, `fpr/data/nist.py`, `config.py` (`RAW_FOLDERS`, `EXPECTED_COUNTS`) |
| Preprocessing | `fpr/data/foreground.py`, `preprocessing/03_export_training_pool.py`; synthetic contaminants `fpr/data/degradations/` (one file each) and `04`, `05`, `08`; minutiae maps `09`; MSU normalisation `fpr/data/normalization.py`, `10` |
| Train / val / test split | `preprocessing/02_make_split.py`, `fpr/data/split.py` (`make_split`, `check_split`: by participant, seed 643), `tests/test_split.py` |
| Datasets and inputs | `fpr/data/datasets.py` (`FixedTrainDataset`, `TrainPatchDataset`), `fpr/models/inputs.py` (conditioning: damaged print, mask, orientation) |
| Network structures | `fpr/models/unet.py` (`UNet`, sizes in `experiments/main/E02_resshift/params.json`); baselines `experiments/baselines/E03_fingergan/network.py`, `E05_msu_aec/model.py`; segmenter `segmenter/segformer.py` |
| Diffusion process and samplers | `fpr/models/schedule.py` (η_t, κ), `fpr/models/resshift.py` (`q_sample`, `predict_x0`, `sample`), `fpr/models/ccdf.py` (warm start, projection), `fpr/models/reconstructors.py` + `base.py` (tiling, paste-back) |
| Training process | `experiments/train.py` → `fpr/training/train_resshift.py` (E02), `fpr/training/train_aux.py` (E07 and the other aux-loss models) |
| Loss functions | `fpr/models/resshift.py` (`training_loss`), `fpr/training/aux_losses.py` (`minutiae_term`, `cldice_term`, `cv_edge_term`, `cv_mix_term`, `eta_gate`, `aux_loss`), `fpr/training/skeleton.py` (soft skeleton) |
| Optimiser, learning rate, schedule | `fpr/training/common.py` (`build_model`: AdamW; `lr_factor`: linear warm-up + cosine; `optimiser_step`: clip + EMA), values in each `params.json` `training` block |
| Validation process | `fpr/training/common.py` (`val_batch`) + trainers' `val_loss`; `fpr/validation/quick_check.py` (val EER); `fpr/validation/ccdf_sweep.py`, `ccdf_sweep_v3.py` (sampler selection rules) |
| Evaluation metrics | `evaluation/evaluate.py` (`ridge_fidelity`, `probe_table`, `headline_scores`), `evaluation/run_eval.py`, `evaluation/compare_experiments.py` (paired tests, Holm), `fpr/metrics/` |
| Seed robustness | `experiments/main/E27_seed_study/` (`train.py`, `sample.py`, `analysis.py`) |
| Test interface (demo) | `demo/gradio_app.py` (layout), `demo/handlers.py`, `demo/scoring.py` (scores as the evaluator) |

## Built on

- Z. Yue, J. Wang, C. C. Loy. *ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting*,
  NeurIPS 2023 ([arXiv 2307.12348](https://arxiv.org/abs/2307.12348)); TPAMI 2024 ([arXiv 2403.07319](https://arxiv.org/abs/2403.07319)).
  Code: [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift). `fpr/models/schedule.py` and `resshift.py` are
  adapted from it and follow its **S-Lab License 1.0** (non-commercial use); see the headers of those files.
- H. Chung, B. Sim, J. C. Ye. *Come-Closer-Diffuse-Faster (CCDF)*, CVPR 2022 ([arXiv 2112.05146](https://arxiv.org/abs/2112.05146)).
- Y. Zhu, X. Yin, J. Hu. *FingerGAN*, TPAMI 2023 ([arXiv 2206.12885](https://arxiv.org/abs/2206.12885));
  code and weights: [HubYZ/LatentEnhancement](https://github.com/HubYZ/LatentEnhancement) (E03, E17, E20).
- K. Cao, D.-L. Nguyen, C. Tymoszek, A. K. Jain. *End-to-End Latent Fingerprint Search* (MSU-LatentAFIS), TIFS 2020
  ([arXiv 1812.10213](https://arxiv.org/abs/1812.10213)); code: [prip-lab/MSU-LatentAFIS](https://github.com/prip-lab/MSU-LatentAFIS) (E04, E05, normalisation).
- S. Shit et al. *clDice*, CVPR 2021 ([arXiv 2003.07311](https://arxiv.org/abs/2003.07311)).
- E. Xie et al. *SegFormer*, NeurIPS 2021 ([arXiv 2105.15203](https://arxiv.org/abs/2105.15203)); weights [nvidia/mit-b0](https://huggingface.co/nvidia/mit-b0).
- Y. You et al. *CTMSR* (one-step consistency for super-resolution), ICCV 2025 ([arXiv 2503.20349](https://arxiv.org/abs/2503.20349)); code [CVL-UESTC/CTMSR](https://github.com/CVL-UESTC/CTMSR) (E10, E11).
- L. Hong, Y. Wan, A. K. Jain. *Fingerprint Image Enhancement*, TPAMI 1998 (E01, orientation field).
- NIST Biometric Image Software (NBIS 5.0: MINDTCT, BOZORTH3): [nbis_v5_0_0.zip](https://nigos.nist.gov/nist/nbis/nbis_v5_0_0.zip).
- NIST NFIQ 2: [usnistgov/NFIQ2](https://github.com/usnistgov/NFIQ2). LPIPS: [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).
- G. Fiumara et al. *NIST Special Database 302*, NIST TN 2007, 2019 ([doi:10.6028/NIST.TN.2007](https://doi.org/10.6028/NIST.TN.2007)).
