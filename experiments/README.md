# Experiments

One folder per experiment, grouped by role. A folder holds `params.json` (every setting), the chosen sampler
(`selected.json`) or loss weights (`calibration.json`) where there are any, the cost on one A40
(`benchmark.json`) and only the code that is the experiment's own; the shared model, trainers and samplers
are in `fpr/`. `fpr.experiments.build_reconstructor(name)` builds any method from its folder.

```bash
python experiments/train.py --exp EXX_name                       # picks the trainer (see train.py)
python experiments/run_experiment.py --exp EXX_name --split test # repair the test probes
python experiments/benchmark.py --exp EXX_name                   # seconds per print, memory
python evaluation/run_eval.py --exp EXX_name --split test        # evaluate, add a leaderboard row
```

Scores: v2 test (398 probes) unless marked **v3** (1,194 samples of the same 398 fingers); compare a v3 number
only with v3 numbers. Score 1 = net rescue in points (rescued / broken), 2 = false-match gate, 3 = ridge
fidelity in the damage mask (damaged: 0.827 v2, 0.826 v3); then seconds per print. Full numbers and rationale:
[experimental_log.md](experimental_log.md); every pair tested: `../evaluation/results/comparison/`.
FingerGAN's released weights (E03, E17, E20; 406 MB) are not in git: `gdown 1lS8_wbMjf0_AHYMG4mO_Mrb2TpFVQVTa
-O experiments/baselines/E03_fingergan/weights/FingerGAN_25850831.tar`.

## Main path: the final models E09 and E22 and what they are built from

| Experiment | What | Status | Score 1 / 2 / 3, cost | Code |
|---|---|---|---|---|
| [E02_resshift](main/E02_resshift/) | ResShift residual-shifting diffusion (TPAMI 2024), pixel space, 8.1M-parameter U-Net, 15 steps; **reference model** for E06–E11 | evaluated | +2.8 (19 / 8), pass, 0.930; 0.86 s | `reconstructor.py`, trainer: `fpr/training/train_resshift.py` |
| [E07_aux_minutiae_cldice](main/E07_aux_minutiae_cldice/) | E06 + clDice (Shit et al., CVPR 2021) on soft ridge skeletons inside the dilated mask, same gate; λ minutiae 0.22 (E06's), λ clDice 0.19 | evaluated | +3.8 (25 / 10), pass, 0.930; 0.83 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py` |
| [E09_ccdf](main/E09_ccdf/) | CCDF (Chung et al., CVPR 2022) on E02 v2: warm start at t₀ < T from a starting estimate (regression U-Net / E02 one-call / y₀), optional data-consistency projection; t₀, init and projection chosen on val | evaluated; **co-best** (tied with E22 on v3 and across seeds, E27) | +4.3 (24 / 7), pass, 0.943; 0.48 s; v3: +5.3 (88 / 25), pass, 0.943 | `reconstructor.py`, `train.py` |
| [E22_ccdf_aux_minutiae_cldice](main/E22_ccdf_aux_minutiae_cldice/) | E07's checkpoint (minutiae + clDice, v2) + E09's CCDF sampler, selected on v2 val | evaluated (v2 + v3); **co-best** with E09 (tied across seeds, E27), cheapest | v3: +6.0 (94 / 22), pass, 0.943; 0.43 s | `reconstructor.py`, sweep: `fpr/validation/ccdf_sweep.py` |
| [E27_seed_study](main/E27_seed_study/) | **v3 test.** Seed robustness of E09 and E22: E02 / E07 retrained with training seeds 1, 2 (only `training.seed` changed), all 6 checkpoints sampled with the fixed E09/E22 CCDF config × sampler seeds 0–2 (18 sets), evaluated outside the leaderboard; hierarchical bootstrap (training seed > sampler seed > finger) | done (18 / 18 runs; changelog 0030) | v3, mean of 9 runs: E09 +5.45 [+3.86, +7.12], E22 +5.61 [+4.07, +7.25]; 18 / 18 pass; 0.9426 / 0.9427; E22 − E09 +0.16 [−0.72, +1.00] pts, +0.0002 ridge (tied); no new model, cost as E09 / E22 | `analysis.py`, `chain.sh`, `evaluate_run.py`, `pipeline.py`, `runs.py`, `sample.py`, `train.py` |

## Baselines (published enhancers, raw and adapted to the normalised domain)

| Experiment | What | Status | Score 1 / 2 / 3, cost | Code |
|---|---|---|---|---|
| [E01_gabor](baselines/E01_gabor/) | classical Gabor enhancement (Hong et al. 1998), 0 parameters | v1 only (eval 1, in LEADERBOARD_eval1.md) | — |  |
| [E03_fingergan](baselines/E03_fingergan/) | SOTA baseline: FingerGAN (TPAMI 2023), released weights, zero-shot, 33.9M parameters; whole print, no paste-back | evaluated | −26.6 (10 / 116), pass, 0.551; 1.16 s | `network.py`, `reconstructor.py` |
| [E04_msu_stft](baselines/E04_msu_stft/) | SOTA baseline: MSU-LatentAFIS (TIFS 2020) non-learned enhancement (cartoon-texture + STFT), 0 parameters; whole print, no paste-back; **main baseline** | evaluated | +1.5 (20 / 14), pass, 0.820; 0.22 s CPU | `preprocess.py`, `reconstructor.py` |
| [E05_msu_aec](baselines/E05_msu_aec/) | SOTA baseline: MSU-LatentAFIS enhancement autoencoder, retrained with MSU's recipe (released weights are gone), 0.65M parameters; whole print, no paste-back | evaluated | −52.3 (3 / 211), pass, 0.515; 0.26 s | `model.py`, `reconstructor.py`, `train.py` |
| [E17_fingergan_norm](baselines/E17_fingergan_norm/) | **v3 data.** E03 (FingerGAN, released weights, zero-shot) in the normalised domain: normalised input, ridge map rendered as a normalised print (blur + normalise), measured as stored; input / rendering / paste-back chosen on v3 val | evaluated (v3) | v3: −5.2 (55 / 117), pass, 0.631; 1.69 s | `reconstructor.py`, `sweep.py`, `val_sweep.py` |
| [E18_msu_stft_norm](baselines/E18_msu_stft_norm/) | **v3 data.** E04's STFT on the normalised print, measured as stored (no second normalisation by the evaluator). Normalised *input* alone = E04 (corr 0.999), so tested only if as-stored wins on val | evaluated (v3) | v3: +2.3 (65 / 37), pass, 0.827; 0.37 s CPU | `reconstructor.py`, `sweep.py` |
| [E19_msu_aec_norm](baselines/E19_msu_aec_norm/) | **v3 data.** E05's MSU autoencoder (0.65M) trained on **our** normalised damaged → clean pairs (v3 train set, MSU's optimiser and L2), output normalised | evaluated (v3) | v3: −2.1 (51 / 76), pass, 0.711; 0.23 s | `pairs.py`, `reconstructor.py`, `sweep.py`, `train.py` |
| [E20_fingergan_finetune](baselines/E20_fingergan_finetune/) | **v3 data.** FingerGAN generator fine-tuned from the released weights on normalised v3 pairs (L1 + 0.1 clDice, no GAN term), output normalised; runs only if E17 still fails on val | evaluated (v3) | v3: +2.8 (67 / 33), pass, 0.846; 1.30 s | `reconstructor.py`, `sweep.py`, `train.py` |

## Ablations (losses, sampler, normalisation, data version)

| Experiment | What | Status | Score 1 / 2 / 3, cost | Code |
|---|---|---|---|---|
| [E06_aux_minutiae](ablations/E06_aux_minutiae/) | E02 + minutiae-weighted L1 on the predicted x̂₀ inside the dilated mask (MINDTCT minutiae of the clean print, σ 8 px), low-noise steps only (η_t ≤ 0.5) | evaluated | +2.0 (18 / 10), pass, 0.931; 0.80 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py` |
| [E08_aux_cv_edge_mix](ablations/E08_aux_cv_edge_mix/) | E02 + Laplacian edge term (Charbonnier, λ 0.27) + MS-SSIM/L1 mix (Zhao et al., TCI 2017; α 0.84, λ 0.12) on x̂₀ inside the dilated mask, same gate | evaluated | +3.8 (23 / 8), pass, 0.932; 0.83 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py` |
| [E12_norm_ccdf_cldice_cv](ablations/E12_norm_ccdf_cldice_cv/) | **v3 data.** ResShift trained with clDice + edge + MS-SSIM/L1 (no minutiae term) on MSU-normalised prints, sampled with CCDF (t₀, start, projection chosen on v3 val); input and output normalised | evaluated (v3) | v3: +5.5 (94 / 28), pass, 0.928; 0.58 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py`, sweep: `fpr/validation/ccdf_sweep_v3.py` |
| [E13_norm_no_cldice](ablations/E13_norm_no_cldice/) | **v3 data.** E12 without clDice (edge 0.26 + mix 0.24, E12's λs fixed); everything else E12's; E12 − E13 isolates clDice | evaluated (v3) | v3: +4.7 (85 / 29), pass, 0.905; 0.93 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py`, sweep: `fpr/validation/ccdf_sweep_v3.py` |
| [E14_norm_no_edge](ablations/E14_norm_no_edge/) | **v3 data.** E12 without the Laplacian edge term (clDice 0.38 + mix 0.24); E12 − E14 isolates the edge term | evaluated (v3) | v3: +4.9 (83 / 25), pass, 0.907; 0.94 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py`, sweep: `fpr/validation/ccdf_sweep_v3.py` |
| [E15_norm_no_mix](ablations/E15_norm_no_mix/) | **v3 data.** E12 without the MS-SSIM/L1 mix (clDice 0.38 + edge 0.26); E12 − E15 isolates the mix | evaluated (v3) | v3: +5.6 (94 / 27), pass, 0.921; 0.70 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py`, sweep: `fpr/validation/ccdf_sweep_v3.py` |
| [E16_norm_no_aux](ablations/E16_norm_no_aux/) | **v3 data.** E12 with no extra terms (L_diff only): normalised v3 ResShift + CCDF; E12 − E16 isolates all three terms | evaluated (v3) | v3: +4.9 (81 / 23), pass, 0.929; 0.59 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py`, sweep: `fpr/validation/ccdf_sweep_v3.py` |
| [E21_ccdf_aux_minutiae](ablations/E21_ccdf_aux_minutiae/) | E06's checkpoint (minutiae loss, v2) sampled with E09's CCDF sampler; start / t₀ / projection chosen on v2 val by E09's grid and rule (no regressor); no retraining | evaluated (v2 + v3); val kept the plain chain, so = E06 | v3: +5.7 (94 / 26), pass, 0.932; 0.79 s | `reconstructor.py`, sweep: `fpr/validation/ccdf_sweep.py` |
| [E23_ccdf_aux_cv_edge_mix](ablations/E23_ccdf_aux_cv_edge_mix/) | E08's checkpoint (edge + MS-SSIM/L1, v2) + E09's CCDF sampler, selected on v2 val | evaluated (v2 + v3) | v3: +4.9 (85 / 26), pass, 0.934; 0.70 s | `reconstructor.py`, sweep: `fpr/validation/ccdf_sweep.py` |
| [E24_norm_no_cldice_e12sampler](ablations/E24_norm_no_cldice_e12sampler/) | **v3 data.** E13's checkpoint with E12's selected sampler fixed (one-call, t₀ 2, no projection); E12 − E24 = clDice alone | evaluated (v3) | v3: +5.8 (89 / 20), pass, 0.927; 0.60 s | `reconstructor.py` |
| [E25_norm_no_edge_e12sampler](ablations/E25_norm_no_edge_e12sampler/) | **v3 data.** E14's checkpoint with E12's selected sampler fixed; E12 − E25 = the edge term alone | evaluated (v3) | v3: +5.3 (88 / 25), pass, 0.930; 0.62 s | `reconstructor.py` |
| [E26_raw_ccdf_cldice_cv](ablations/E26_raw_ccdf_cldice_cv/) | **v3 data.** E12 without the MSU normalisation: trained, selected and repairing on raw prints (λs recalibrated on raw, same ~20%-of-L_diff rule), E12's losses, CCDF grid and rule; raw output with E02/E09 paste-back, normalised by the evaluator; E12 − E26 isolates the normalisation (plus the λs: clDice 0.19 / mix 0.12 vs E12's 0.38 / 0.24) | evaluated (v3); ties E09 / E22, costs more | v3: +5.6 (99 / 32), pass, 0.941; 0.53 s | `reconstructor.py`, trainer: `fpr/training/train_aux.py`, sweep: `fpr/validation/ccdf_sweep_v3.py` |

## One-step alternative (consistency distillation)

| Experiment | What | Status | Score 1 / 2 / 3, cost | Code |
|---|---|---|---|---|
| [E10_ctmsr](one_step/E10_ctmsr/) | CTMSR Stage 1 (You et al., ICCV 2025): consistency training on a ResShift-style trajectory, **one step**, no teacher; E02's 8.1M U-Net; 50k steps | evaluated | −1.8 (8 / 15), pass, 0.800; 0.40 s | `consistency.py`, `reconstructor.py`, `train.py` |
| [E11_ctmsr_dtm](one_step/E11_ctmsr_dtm/) | CTMSR Stage 2: E10 + Distribution Trajectory Matching (2k iterations), a realism push; E11 − E10 isolates DTM | evaluated | −1.8 (11 / 18), pass, 0.798; 0.40 s | `reconstructor.py`, `train.py` |
