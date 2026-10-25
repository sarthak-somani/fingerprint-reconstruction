# E27 seed study: results (written by analysis.py; do not edit by hand)

v3 test, 1194 samples / 398 fingers. 18 of 18 runs evaluated: E09: training seeds [0, 1, 2], 9 runs, E22: training seeds [0, 1, 2], 9 runs.
Sampler configuration fixed (selected once on val for training seed 0: one-call x0 start, t0 = 2, projection on); no re-selection per seed. Score 1 in fractions (x100 = points).

## Per run (evaluate.py headline_scores, finger bootstrap within the run)

| run | model | train_seed | sampler_seed | rescued | broken | net_rescue | net_rescue_lo | net_rescue_hi | fmr_delta | fmr_delta_hi | gate | ridge_fidelity | ridge_delta | ridge_delta_lo | ridge_delta_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E09_s0_r0 | E09 | 0 | 0 | 79 | 21 | 0.0486 | 0.0327 | 0.0653 | -0.0018 | -0.0008 | pass | 0.9428 | 0.1164 | 0.1119 | 0.1209 |
| E09_s0_r1 | E09 | 0 | 1 | 88 | 25 | 0.0528 | 0.0369 | 0.0695 | -0.0015 | -0.0004 | pass | 0.9428 | 0.1164 | 0.1120 | 0.1210 |
| E09_s0_r2 | E09 | 0 | 2 | 90 | 28 | 0.0519 | 0.0352 | 0.0695 | -0.0014 | -0.0003 | pass | 0.9428 | 0.1165 | 0.1120 | 0.1210 |
| E09_s1_r0 | E09 | 1 | 0 | 97 | 20 | 0.0645 | 0.0469 | 0.0829 | -0.0024 | -0.0014 | pass | 0.9427 | 0.1163 | 0.1118 | 0.1209 |
| E09_s1_r1 | E09 | 1 | 1 | 89 | 27 | 0.0519 | 0.0343 | 0.0695 | -0.0021 | -0.0011 | pass | 0.9428 | 0.1164 | 0.1119 | 0.1209 |
| E09_s1_r2 | E09 | 1 | 2 | 95 | 25 | 0.0586 | 0.0410 | 0.0762 | -0.0023 | -0.0013 | pass | 0.9428 | 0.1164 | 0.1119 | 0.1209 |
| E09_s2_r0 | E09 | 2 | 0 | 90 | 26 | 0.0536 | 0.0369 | 0.0712 | -0.0022 | -0.0012 | pass | 0.9420 | 0.1157 | 0.1112 | 0.1202 |
| E09_s2_r1 | E09 | 2 | 1 | 87 | 24 | 0.0528 | 0.0352 | 0.0704 | -0.0019 | -0.0008 | pass | 0.9420 | 0.1157 | 0.1112 | 0.1202 |
| E09_s2_r2 | E09 | 2 | 2 | 93 | 26 | 0.0561 | 0.0385 | 0.0745 | -0.0019 | -0.0009 | pass | 0.9421 | 0.1157 | 0.1113 | 0.1202 |
| E22_s0_r0 | E22 | 0 | 0 | 92 | 22 | 0.0586 | 0.0419 | 0.0754 | -0.0018 | -0.0008 | pass | 0.9426 | 0.1162 | 0.1117 | 0.1208 |
| E22_s0_r1 | E22 | 0 | 1 | 83 | 18 | 0.0544 | 0.0385 | 0.0712 | -0.0023 | -0.0013 | pass | 0.9426 | 0.1163 | 0.1118 | 0.1208 |
| E22_s0_r2 | E22 | 0 | 2 | 91 | 20 | 0.0595 | 0.0419 | 0.0779 | -0.0021 | -0.0011 | pass | 0.9428 | 0.1164 | 0.1119 | 0.1210 |
| E22_s1_r0 | E22 | 1 | 0 | 91 | 25 | 0.0553 | 0.0385 | 0.0720 | -0.0022 | -0.0012 | pass | 0.9424 | 0.1160 | 0.1116 | 0.1206 |
| E22_s1_r1 | E22 | 1 | 1 | 93 | 21 | 0.0603 | 0.0419 | 0.0796 | -0.0015 | -0.0004 | pass | 0.9424 | 0.1160 | 0.1115 | 0.1205 |
| E22_s1_r2 | E22 | 1 | 2 | 81 | 22 | 0.0494 | 0.0318 | 0.0678 | -0.0020 | -0.0011 | pass | 0.9425 | 0.1161 | 0.1116 | 0.1206 |
| E22_s2_r0 | E22 | 2 | 0 | 91 | 19 | 0.0603 | 0.0419 | 0.0787 | -0.0017 | -0.0006 | pass | 0.9431 | 0.1167 | 0.1122 | 0.1213 |
| E22_s2_r1 | E22 | 2 | 1 | 88 | 24 | 0.0536 | 0.0377 | 0.0712 | -0.0021 | -0.0011 | pass | 0.9431 | 0.1167 | 0.1121 | 0.1213 |
| E22_s2_r2 | E22 | 2 | 2 | 90 | 26 | 0.0536 | 0.0352 | 0.0720 | -0.0016 | -0.0006 | pass | 0.9431 | 0.1167 | 0.1122 | 0.1213 |

Reference runs (evaluation/results, v3 test):

| run | model | train_seed | sampler_seed | rescued | broken | net_rescue | net_rescue_lo | net_rescue_hi | fmr_delta | fmr_delta_hi | gate | ridge_fidelity | ridge_delta | ridge_delta_lo | ridge_delta_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E04 (evaluation/results) | E04 (evaluation/results) | -1 | -1 | 74 | 48 | 0.0218 | 0.0034 | 0.0410 | -0.0004 | 0.0006 | pass | 0.8196 | -0.0068 | -0.0076 | -0.0059 |
| E09 (evaluation/results) | E09 (evaluation/results) | -1 | -1 | 88 | 25 | 0.0528 | 0.0360 | 0.0704 | -0.0017 | -0.0006 | pass | 0.9428 | 0.1164 | 0.1119 | 0.1209 |
| E22 (evaluation/results) | E22 (evaluation/results) | -1 | -1 | 94 | 22 | 0.0603 | 0.0427 | 0.0787 | -0.0017 | -0.0007 | pass | 0.9427 | 0.1163 | 0.1118 | 0.1209 |

## Spread across seeds

sd_train = SD of the training seeds' means (over sampler seeds); sd_sampler = mean within-training-seed SD over sampler seeds.

| model | score | runs | train_seeds | mean | sd_train | sd_sampler | min | max |
|---|---|---|---|---|---|---|---|---|
| E09 | s1 | 9 | 3 | 0.0545 | 0.0036 | 0.0034 | 0.0486 | 0.0645 |
| E09 | s3 | 9 | 3 | 0.9426 | 0.0004 | 0.0000 | 0.9420 | 0.9428 |
| E22 | s1 | 9 | 3 | 0.0561 | 0.0013 | 0.0040 | 0.0494 | 0.0603 |
| E22 | s3 | 9 | 3 | 0.9427 | 0.0003 | 0.0000 | 0.9424 | 0.9431 |

## Hierarchical bootstrap (training seeds > sampler seeds > fingers, 10,000 resamples)

estimate = mean over all runs; lo / hi = 95% percentile CI; p_le_0 = share of resamples with a difference <= 0. Differences are paired by finger (one finger draw per resample for every model); each model's seeds are drawn independently. With 3 training seeds the seed level is resampled from 3 values only, so read the CI as a sensitivity check, not an exact interval.

| score | what | estimate | lo | hi | p_le_0 |
|---|---|---|---|---|---|
| s1 | E09 | 0.0545 | 0.0386 | 0.0712 |  |
| s1 | E22 | 0.0561 | 0.0407 | 0.0725 |  |
| s1 | E04 | 0.0218 | 0.0034 | 0.0410 |  |
| s1 | damaged | 0.0000 | 0.0000 | 0.0000 |  |
| s1 | E09 - E04 | 0.0328 | 0.0167 | 0.0499 | 0.0000 |
| s1 | E22 - E04 | 0.0343 | 0.0179 | 0.0518 | 0.0001 |
| s1 | E09 - damaged | 0.0545 | 0.0386 | 0.0712 | 0.0000 |
| s1 | E22 - damaged | 0.0561 | 0.0407 | 0.0725 | 0.0000 |
| s1 | E22 - E09 | 0.0016 | -0.0072 | 0.0100 | 0.3555 |
| s3 | E09 | 0.9426 | 0.9394 | 0.9457 |  |
| s3 | E22 | 0.9427 | 0.9396 | 0.9459 |  |
| s3 | E04 | 0.8196 | 0.8128 | 0.8265 |  |
| s3 | damaged | 0.8264 | 0.8195 | 0.8333 |  |
| s3 | E09 - E04 | 0.1229 | 0.1186 | 0.1273 | 0.0000 |
| s3 | E22 - E04 | 0.1231 | 0.1187 | 0.1275 | 0.0000 |
| s3 | E09 - damaged | 0.1162 | 0.1117 | 0.1208 | 0.0000 |
| s3 | E22 - damaged | 0.1163 | 0.1118 | 0.1210 | 0.0000 |
| s3 | E22 - E09 | 0.0002 | -0.0005 | 0.0009 | 0.2946 |

## E22 vs E09 across seeds

train_seed_pairs: all (E22 seed, E09 seed) pairs of training-seed means; matched: same training seed (E02 and E07 share init and data order at a given seed); run_pairs: every (E22 run, E09 run).

| score | train_seed_pairs | matched_seed_pairs | run_pairs |
|---|---|---|---|
| s1 | 6/9 | 2/3 | 54/81 |
| s3 | 5/9 | 1/3 | 46/81 |

## Every seed against E04

| model | score | runs_above_E04 | train_seeds_above_E04 | gate_pass |
|---|---|---|---|---|
| E09 | s1 | 9/9 | 3/3 | 9/9 |
| E09 | s3 | 9/9 | 3/3 | 9/9 |
| E22 | s1 | 9/9 | 3/3 | 9/9 |
| E22 | s3 | 9/9 | 3/3 | 9/9 |

## Sanity: seed-0 / sampler-seed-0 runs vs the original E09 / E22 v3 evaluations

Same checkpoint, configuration and seed; differences can only come from GPU nondeterminism.

| run | reference | same_acceptance | max_ridge_diff |
|---|---|---|---|
| E09_s0_r0 | evaluation/results/E09_ccdf/test_20261023_193652 | 0.9690 | 0.0110 |
| E22_s0_r0 | evaluation/results/E22_ccdf_aux_minutiae_cldice/test_20261024_144531 | 0.9665 | 0.0121 |

![seeds](seed_study.png)
