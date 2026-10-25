### All experiments (test)

| experiment | rescued | broken | 1 net rescue, pts (95% CI) | p | 2 FMR change, pts (95% CI) | gate | 3 ridge fidelity | 3 change (95% CI) | EER % | s / print | GPU MB | RAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E02_resshift | 85 | 23 | +5.2 [+3.6, +6.9] | 1e-09 | -0.16 [-0.27, -0.05] | pass | 0.930 | +0.104 [+0.100, +0.108] | 11.4 | 0.86 cuda | 678 | 1043 |
| E03_fingergan | 43 | 332 | -24.2 [-28.1, -20.4] | 7e-30 | -0.19 [-0.38, -0.01] | pass | 0.546 | -0.281 [-0.288, -0.273] | 20.8 | 1.16 cuda | 3598 | 1025 |
| **E04_msu_stft** | 74 | 48 | +2.2 [+0.3, +4.1] | 0.030 | -0.04 [-0.14, +0.06] | pass | 0.820 | -0.007 [-0.008, -0.006] | 12.8 | 0.22 cpu | – | 572 |
| E05_msu_aec | 8 | 622 | -51.4 [-55.6, -47.2] | 7e-75 | -0.90 [-1.07, -0.73] | pass | 0.509 | -0.317 [-0.328, -0.307] | 25.6 | 0.26 cuda | 37 | 972 |
| E06_aux_minutiae | 94 | 26 | +5.7 [+3.9, +7.6] | 4e-09 | -0.19 [-0.29, -0.09] | pass | 0.932 | +0.105 [+0.101, +0.109] | 10.9 | 0.80 cuda | 678 | 1009 |
| E07_aux_minutiae_cldice | 92 | 33 | +4.9 [+3.2, +6.8] | 2e-07 | -0.21 [-0.31, -0.11] | pass | 0.931 | +0.104 [+0.100, +0.108] | 10.9 | 0.83 cuda | 678 | 970 |
| E08_aux_cv_edge_mix | 89 | 25 | +5.4 [+3.5, +7.3] | 2e-08 | -0.17 [-0.28, -0.07] | pass | 0.933 | +0.106 [+0.102, +0.110] | 11.0 | 0.83 cuda | 678 | 1036 |
| E09_ccdf | 88 | 25 | +5.3 [+3.6, +7.0] | 3e-09 | -0.17 [-0.27, -0.06] | pass | 0.943 | +0.116 [+0.112, +0.121] | 11.1 | 0.48 cuda | 681 | 1011 |
| E10_ctmsr | 47 | 55 | -0.7 [-2.3, +1.0] | 0.498 | -0.03 [-0.11, +0.06] | pass | 0.801 | -0.026 [-0.028, -0.024] | 12.8 | 0.40 cuda | 675 | 1042 |
| E11_ctmsr_dtm | 41 | 53 | -1.0 [-2.8, +0.8] | 0.300 | -0.01 [-0.10, +0.09] | pass | 0.798 | -0.028 [-0.030, -0.027] | 12.9 | 0.40 cuda | 675 | 1019 |
| E12_norm_ccdf_cldice_cv | 94 | 28 | +5.5 [+3.8, +7.4] | 3e-09 | -0.13 [-0.23, -0.02] | pass | 0.928 | +0.102 [+0.098, +0.106] | 10.5 | 0.58 cuda | 682 | 1180 |
| E13_norm_no_cldice | 85 | 29 | +4.7 [+3.0, +6.4] | 2e-07 | -0.19 [-0.31, -0.08] | pass | 0.905 | +0.078 [+0.075, +0.082] | 11.1 | 0.93 cuda | 680 | 1194 |
| E14_norm_no_edge | 83 | 25 | +4.9 [+3.4, +6.4] | 2e-09 | -0.23 [-0.34, -0.12] | pass | 0.907 | +0.081 [+0.077, +0.085] | 10.8 | 0.94 cuda | 680 | 1197 |
| E15_norm_no_mix | 94 | 27 | +5.6 [+3.9, +7.4] | 4e-10 | -0.16 [-0.28, -0.04] | pass | 0.921 | +0.095 [+0.091, +0.099] | 11.3 | 0.70 cuda | 683 | 1197 |
| E16_norm_no_aux | 81 | 23 | +4.9 [+3.2, +6.6] | 3e-08 | -0.18 [-0.28, -0.07] | pass | 0.929 | +0.103 [+0.099, +0.107] | 10.7 | 0.59 cuda | 683 | 1177 |
| E17_fingergan_norm | 55 | 117 | -5.2 [-7.5, -2.9] | 2e-05 | -0.40 [-0.53, -0.28] | pass | 0.631 | -0.195 [-0.204, -0.186] | 13.4 | 1.69 cuda | 3598 | 1025 |
| E18_msu_stft_norm | 65 | 37 | +2.3 [+0.7, +4.0] | 0.008 | -0.16 [-0.28, -0.05] | pass | 0.827 | +0.000 [+0.000, +0.001] | 12.9 | 0.37 cpu | – | 573 |
| E19_msu_aec_norm | 51 | 76 | -2.1 [-3.9, -0.3] | 0.033 | -0.40 [-0.52, -0.28] | pass | 0.711 | -0.115 [-0.123, -0.107] | 12.2 | 0.23 cuda | 37 | 972 |
| E20_fingergan_finetune | 67 | 33 | +2.8 [+1.2, +4.5] | 0.002 | -0.24 [-0.36, -0.12] | pass | 0.846 | +0.019 [+0.018, +0.021] | 11.8 | 1.30 cuda | 3598 | 1013 |
| E21_ccdf_aux_minutiae | 94 | 26 | +5.7 [+3.9, +7.6] | 4e-09 | -0.19 [-0.29, -0.09] | pass | 0.932 | +0.105 [+0.101, +0.109] | 10.9 | 0.79 cuda | 678 | 1026 |
| E22_ccdf_aux_minutiae_cldice | 94 | 22 | +6.0 [+4.3, +7.9] | 8e-11 | -0.17 [-0.28, -0.07] | pass | 0.943 | +0.116 [+0.112, +0.121] | 11.4 | 0.43 cuda | 681 | 1031 |
| E23_ccdf_aux_cv_edge_mix | 85 | 26 | +4.9 [+3.2, +6.8] | 9e-08 | -0.18 [-0.28, -0.07] | pass | 0.934 | +0.108 [+0.104, +0.112] | 11.1 | 0.70 cuda | 682 | 1043 |
| E24_norm_no_cldice_e12sampler | 89 | 20 | +5.8 [+4.2, +7.5] | 2e-11 | -0.18 [-0.29, -0.08] | pass | 0.927 | +0.101 [+0.097, +0.105] | 11.0 | 0.60 cuda | 682 | 1195 |
| E25_norm_no_edge_e12sampler | 88 | 25 | +5.3 [+3.6, +7.0] | 8e-10 | -0.19 [-0.30, -0.08] | pass | 0.930 | +0.104 [+0.100, +0.108] | 11.3 | 0.62 cuda | 682 | 1191 |
| E26_raw_ccdf_cldice_cv | 99 | 32 | +5.6 [+3.8, +7.5] | 4e-09 | -0.20 [-0.31, -0.09] | pass | 0.941 | +0.114 [+0.110, +0.119] | 11.0 | 0.53 cuda | 682 | 1052 |

All samples (n = 1194, 398 fingers). Reference points: accepted at τ: clean 78.9%, damaged 71.4%; FMR damaged 0.99%; ridge fidelity damaged 0.826 (clean = 1); EER damaged 12.9%. p = exact sign-flip test over fingers (rescued vs broken; = exact McNemar with one sample per finger). CIs resample fingers.

### Every pair

Score 1 compares acceptance at τ (exact sign-flip test on per-finger sums; = McNemar with one sample per finger), score 3 the ridge fidelity in the mask (Wilcoxon signed-rank on per-finger means), sample by sample; CIs resample fingers. p-values are Holm-adjusted over all 325 pairs, per score. **equivalent**: the 90% CI of the difference lies within ±3 points (score 1) or ±0.01 (score 3), whatever the p-value. **A better / B better**: otherwise, adjusted p < 0.05. **unresolved**: neither. `damaged` = the unrepaired probes. Score 2 is a per-run gate (table above).

**Score 1 (identity recovery)**: row vs column, ▲ row better, ▼ row worse, ≈ equivalent, · unresolved

|  | dmg | E02 | E03 | E04 | E05 | E06 | E07 | E08 | E09 | E10 | E11 | E12 | E13 | E14 | E15 | E16 | E17 | E18 | E19 | E20 | E21 | E22 | E23 | E24 | E25 | E26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dmg |  | ▼ | ▲ | · | ▲ | ▼ | ▼ | ▼ | ▼ | ≈ | ≈ | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ | · | · | · | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E02 | ▲ |  | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E03 | ▼ | ▼ |  | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E04 | · | · | ▲ |  | ▲ | ▼ | · | · | · | · | · | · | · | · | ▼ | · | ▲ | ≈ | ▲ | ≈ | ▼ | ▼ | · | · | · | · |
| E05 | ▼ | ▼ | ▼ | ▼ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E06 | ▲ | ≈ | ▲ | ▲ | ▲ |  | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E07 | ▲ | ≈ | ▲ | · | ▲ | ≈ |  | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E08 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ |  | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E09 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ |  | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E10 | ≈ | ▼ | ▲ | · | ▲ | ▼ | ▼ | ▼ | ▼ |  | ≈ | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ | · | ≈ | · | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E11 | ≈ | ▼ | ▲ | · | ▲ | ▼ | ▼ | ▼ | ▼ | ≈ |  | ▼ | ▼ | ▼ | ▼ | ▼ | · | · | ≈ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E12 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ |  | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E13 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ |  | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E14 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ |  | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E15 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ |  | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E16 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ |  | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ | ≈ |
| E17 | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | · | ▼ | ▼ | ▼ | ▼ | ▼ |  | ▼ | · | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E18 | · | · | ▲ | ≈ | ▲ | · | · | · | · | · | · | · | · | · | · | · | ▲ |  | ▲ | ≈ | · | · | · | · | · | · |
| E19 | · | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ≈ | ≈ | ▼ | ▼ | ▼ | ▼ | ▼ | · | ▼ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E20 | · | · | ▲ | ≈ | ▲ | · | · | · | · | · | ▲ | · | · | · | · | · | ▲ | ≈ | ▲ |  | · | · | · | · | · | · |
| E21 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · |  | ≈ | ≈ | ≈ | ≈ | ≈ |
| E22 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ |  | ≈ | ≈ | ≈ | ≈ |
| E23 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ |  | ≈ | ≈ | ≈ |
| E24 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ |  | ≈ | ≈ |
| E25 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ |  | ≈ |
| E26 | ▲ | ≈ | ▲ | · | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ≈ | ≈ |  |

**Score 3 (ridge fidelity)**: same reading

|  | dmg | E02 | E03 | E04 | E05 | E06 | E07 | E08 | E09 | E10 | E11 | E12 | E13 | E14 | E15 | E16 | E17 | E18 | E19 | E20 | E21 | E22 | E23 | E24 | E25 | E26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dmg |  | ▼ | ▲ | ≈ | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ | ≈ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E02 | ▲ |  | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ▲ | ≈ | ▼ | ≈ | ≈ | ≈ | ▼ |
| E03 | ▼ | ▼ |  | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E04 | ≈ | ▼ | ▲ |  | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ | ≈ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E05 | ▼ | ▼ | ▼ | ▼ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E06 | ▲ | ≈ | ▲ | ▲ | ▲ |  | ≈ | ≈ | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ▲ | ≈ | ▼ | ≈ | ≈ | ≈ | ≈ |
| E07 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ |  | ≈ | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ▲ | ≈ | ▼ | ≈ | ≈ | ≈ | ▼ |
| E08 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ |  | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ▲ | ≈ | ▼ | ≈ | ≈ | ≈ | ≈ |
| E09 | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ |  | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ≈ | ≈ | ▲ | ▲ | ≈ |
| E10 | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ |  | ≈ | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E11 | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ≈ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E12 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▼ | ▲ | ▲ |  | ▲ | ▲ | ≈ | ≈ | ▲ | ▲ | ▲ | ▲ | ≈ | ▼ | ≈ | ≈ | ≈ | ▼ |
| E13 | ▲ | ▼ | ▲ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▼ |  | ≈ | ▼ | ▼ | ▲ | ▲ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E14 | ▲ | ▼ | ▲ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▼ | ≈ |  | ▼ | ▼ | ▲ | ▲ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E15 | ▲ | ▼ | ▲ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ |  | ≈ | ▲ | ▲ | ▲ | ▲ | ▼ | ▼ | ▼ | ≈ | ≈ | ▼ |
| E16 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ | ≈ |  | ▲ | ▲ | ▲ | ▲ | ≈ | ▼ | ≈ | ≈ | ≈ | ▼ |
| E17 | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E18 | ≈ | ▼ | ▲ | ≈ | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ |  | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E19 | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ | ▼ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E20 | ▲ | ▼ | ▲ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▲ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E21 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ▲ |  | ▼ | ≈ | ≈ | ≈ | ≈ |
| E22 | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ |  | ≈ | ▲ | ▲ | ≈ |
| E23 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ | ▲ | ≈ | ≈ |  | ≈ | ≈ | ≈ |
| E24 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ | ≈ | ≈ | ▲ | ▲ | ▲ | ▲ | ≈ | ▼ | ≈ |  | ≈ | ▼ |
| E25 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▼ | ▲ | ▲ | ≈ | ▲ | ▲ | ≈ | ≈ | ▲ | ▲ | ▲ | ▲ | ≈ | ▼ | ≈ | ≈ |  | ▼ |
| E26 | ▲ | ▲ | ▲ | ▲ | ▲ | ≈ | ▲ | ≈ | ≈ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▲ | ▲ |  |

<details><summary>All pairs with differences, CIs and adjusted p-values</summary>

| A | B | 1: A − B, pts (95% CI) | p (Holm) | verdict | 3: A − B (95% CI) | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|
| damaged | E02_resshift | -5.2 [-7.0, -3.5] | 2e-07 | B better | -0.104 [-0.108, -0.100] | 2e-64 | B better |
| damaged | E03_fingergan | +24.2 [+20.3, +28.1] | 2e-27 | A better | +0.281 [+0.273, +0.288] | 2e-64 | A better |
| damaged | E04_msu_stft | -2.2 [-4.1, -0.3] | 1.000 | unresolved | +0.007 [+0.006, +0.008] | 2e-38 | equivalent |
| damaged | E05_msu_aec | +51.4 [+47.3, +55.7] | 2e-72 | A better | +0.317 [+0.307, +0.328] | 2e-64 | A better |
| damaged | E06_aux_minutiae | -5.7 [-7.6, -3.9] | 7e-07 | B better | -0.105 [-0.109, -0.101] | 2e-64 | B better |
| damaged | E07_aux_minutiae_cldice | -4.9 [-6.9, -3.2] | 3e-05 | B better | -0.104 [-0.108, -0.100] | 2e-64 | B better |
| damaged | E08_aux_cv_edge_mix | -5.4 [-7.3, -3.5] | 4e-06 | B better | -0.106 [-0.110, -0.102] | 2e-64 | B better |
| damaged | E09_ccdf | -5.3 [-7.1, -3.5] | 7e-07 | B better | -0.116 [-0.121, -0.112] | 2e-64 | B better |
| damaged | E10_ctmsr | +0.7 [-1.0, +2.3] | 1.000 | equivalent | +0.026 [+0.024, +0.027] | 3e-63 | A better |
| damaged | E11_ctmsr_dtm | +1.0 [-0.8, +2.8] | 1.000 | equivalent | +0.028 [+0.027, +0.030] | 5e-64 | A better |
| damaged | E12_norm_ccdf_cldice_cv | -5.5 [-7.4, -3.7] | 7e-07 | B better | -0.102 [-0.106, -0.097] | 2e-64 | B better |
| damaged | E13_norm_no_cldice | -4.7 [-6.5, -3.0] | 3e-05 | B better | -0.078 [-0.082, -0.075] | 2e-64 | B better |
| damaged | E14_norm_no_edge | -4.9 [-6.4, -3.4] | 3e-07 | B better | -0.081 [-0.085, -0.077] | 2e-64 | B better |
| damaged | E15_norm_no_mix | -5.6 [-7.4, -3.9] | 9e-08 | B better | -0.095 [-0.099, -0.091] | 2e-64 | B better |
| damaged | E16_norm_no_aux | -4.9 [-6.5, -3.2] | 7e-06 | B better | -0.103 [-0.107, -0.099] | 2e-64 | B better |
| damaged | E17_fingergan_norm | +5.2 [+2.8, +7.5] | 0.004 | A better | +0.195 [+0.186, +0.204] | 2e-64 | A better |
| damaged | E18_msu_stft_norm | -2.3 [-4.0, -0.8] | 1.000 | unresolved | -0.000 [-0.001, -0.000] | 0.028 | equivalent |
| damaged | E19_msu_aec_norm | +2.1 [+0.3, +3.9] | 1.000 | unresolved | +0.115 [+0.107, +0.123] | 3e-63 | A better |
| damaged | E20_fingergan_finetune | -2.8 [-4.6, -1.2] | 0.272 | unresolved | -0.019 [-0.021, -0.018] | 4e-59 | B better |
| damaged | E21_ccdf_aux_minutiae | -5.7 [-7.6, -3.9] | 7e-07 | B better | -0.105 [-0.109, -0.101] | 2e-64 | B better |
| damaged | E22_ccdf_aux_minutiae_cldice | -6.0 [-7.9, -4.3] | 2e-08 | B better | -0.116 [-0.121, -0.112] | 2e-64 | B better |
| damaged | E23_ccdf_aux_cv_edge_mix | -4.9 [-6.8, -3.1] | 2e-05 | B better | -0.108 [-0.112, -0.104] | 2e-64 | B better |
| damaged | E24_norm_no_cldice_e12sampler | -5.8 [-7.5, -4.1] | 4e-09 | B better | -0.101 [-0.105, -0.097] | 2e-64 | B better |
| damaged | E25_norm_no_edge_e12sampler | -5.3 [-7.0, -3.6] | 2e-07 | B better | -0.104 [-0.108, -0.100] | 2e-64 | B better |
| damaged | E26_raw_ccdf_cldice_cv | -5.6 [-7.5, -3.8] | 8e-07 | B better | -0.114 [-0.119, -0.110] | 2e-64 | B better |
| E02_resshift | E03_fingergan | +29.4 [+25.4, +33.5] | 3e-37 | A better | +0.385 [+0.378, +0.391] | 2e-64 | A better |
| E02_resshift | E04_msu_stft | +3.0 [+1.2, +4.9] | 0.295 | unresolved | +0.111 [+0.107, +0.115] | 2e-64 | A better |
| E02_resshift | E05_msu_aec | +56.6 [+52.3, +60.9] | 2e-79 | A better | +0.421 [+0.411, +0.432] | 2e-64 | A better |
| E02_resshift | E06_aux_minutiae | -0.5 [-1.8, +0.8] | 1.000 | equivalent | -0.001 [-0.002, -0.000] | 7e-06 | equivalent |
| E02_resshift | E07_aux_minutiae_cldice | +0.3 [-1.1, +1.6] | 1.000 | equivalent | -0.000 [-0.001, +0.001] | 0.019 | equivalent |
| E02_resshift | E08_aux_cv_edge_mix | -0.2 [-1.3, +1.0] | 1.000 | equivalent | -0.002 [-0.003, -0.001] | 9e-15 | equivalent |
| E02_resshift | E09_ccdf | -0.1 [-1.2, +1.0] | 1.000 | equivalent | -0.012 [-0.013, -0.011] | 4e-64 | B better |
| E02_resshift | E10_ctmsr | +5.9 [+3.9, +7.9] | 5e-07 | A better | +0.130 [+0.125, +0.134] | 2e-64 | A better |
| E02_resshift | E11_ctmsr_dtm | +6.2 [+4.4, +8.0] | 5e-09 | A better | +0.132 [+0.128, +0.137] | 2e-64 | A better |
| E02_resshift | E12_norm_ccdf_cldice_cv | -0.3 [-1.7, +1.0] | 1.000 | equivalent | +0.002 [+0.001, +0.004] | 5e-05 | equivalent |
| E02_resshift | E13_norm_no_cldice | +0.5 [-0.9, +1.9] | 1.000 | equivalent | +0.026 [+0.024, +0.027] | 2e-64 | A better |
| E02_resshift | E14_norm_no_edge | +0.3 [-1.0, +1.7] | 1.000 | equivalent | +0.023 [+0.022, +0.025] | 2e-64 | A better |
| E02_resshift | E15_norm_no_mix | -0.4 [-1.7, +0.9] | 1.000 | equivalent | +0.009 [+0.008, +0.010] | 1e-31 | A better |
| E02_resshift | E16_norm_no_aux | +0.3 [-1.0, +1.8] | 1.000 | equivalent | +0.001 [-0.000, +0.002] | 0.067 | equivalent |
| E02_resshift | E17_fingergan_norm | +10.4 [+7.8, +13.0] | 2e-13 | A better | +0.299 [+0.291, +0.307] | 2e-64 | A better |
| E02_resshift | E18_msu_stft_norm | +2.8 [+0.9, +4.9] | 0.720 | unresolved | +0.104 [+0.100, +0.108] | 2e-64 | A better |
| E02_resshift | E19_msu_aec_norm | +7.3 [+5.4, +9.3] | 6e-11 | A better | +0.219 [+0.213, +0.226] | 2e-64 | A better |
| E02_resshift | E20_fingergan_finetune | +2.3 [+0.5, +4.3] | 1.000 | unresolved | +0.085 [+0.081, +0.088] | 2e-64 | A better |
| E02_resshift | E21_ccdf_aux_minutiae | -0.5 [-1.8, +0.8] | 1.000 | equivalent | -0.001 [-0.002, -0.000] | 7e-06 | equivalent |
| E02_resshift | E22_ccdf_aux_minutiae_cldice | -0.8 [-2.3, +0.5] | 1.000 | equivalent | -0.012 [-0.013, -0.011] | 5e-59 | B better |
| E02_resshift | E23_ccdf_aux_cv_edge_mix | +0.3 [-0.9, +1.5] | 1.000 | equivalent | -0.004 [-0.005, -0.003] | 1e-27 | equivalent |
| E02_resshift | E24_norm_no_cldice_e12sampler | -0.6 [-1.9, +0.8] | 1.000 | equivalent | +0.003 [+0.002, +0.005] | 2e-07 | equivalent |
| E02_resshift | E25_norm_no_edge_e12sampler | -0.1 [-1.3, +1.2] | 1.000 | equivalent | +0.000 [-0.001, +0.002] | 0.348 | equivalent |
| E02_resshift | E26_raw_ccdf_cldice_cv | -0.4 [-1.7, +0.8] | 1.000 | equivalent | -0.010 [-0.011, -0.009] | 2e-55 | B better |
| E03_fingergan | E04_msu_stft | -26.4 [-30.2, -22.6] | 3e-35 | B better | -0.274 [-0.281, -0.266] | 2e-64 | B better |
| E03_fingergan | E05_msu_aec | +27.2 [+23.0, +31.5] | 7e-29 | A better | +0.037 [+0.028, +0.045] | 1e-11 | A better |
| E03_fingergan | E06_aux_minutiae | -29.9 [-33.9, -25.9] | 2e-38 | B better | -0.386 [-0.392, -0.379] | 2e-64 | B better |
| E03_fingergan | E07_aux_minutiae_cldice | -29.1 [-33.2, -25.1] | 1e-36 | B better | -0.385 [-0.391, -0.378] | 2e-64 | B better |
| E03_fingergan | E08_aux_cv_edge_mix | -29.6 [-33.7, -25.5] | 1e-37 | B better | -0.387 [-0.393, -0.380] | 2e-64 | B better |
| E03_fingergan | E09_ccdf | -29.5 [-33.5, -25.4] | 3e-37 | B better | -0.397 [-0.404, -0.390] | 2e-64 | B better |
| E03_fingergan | E10_ctmsr | -23.5 [-27.4, -19.7] | 3e-27 | B better | -0.255 [-0.262, -0.248] | 2e-64 | B better |
| E03_fingergan | E11_ctmsr_dtm | -23.2 [-27.1, -19.3] | 4e-26 | B better | -0.252 [-0.259, -0.245] | 2e-64 | B better |
| E03_fingergan | E12_norm_ccdf_cldice_cv | -29.7 [-33.8, -25.7] | 2e-38 | B better | -0.382 [-0.389, -0.376] | 2e-64 | B better |
| E03_fingergan | E13_norm_no_cldice | -28.9 [-32.8, -25.0] | 6e-38 | B better | -0.359 [-0.365, -0.353] | 2e-64 | B better |
| E03_fingergan | E14_norm_no_edge | -29.1 [-33.2, -25.1] | 7e-37 | B better | -0.361 [-0.368, -0.355] | 2e-64 | B better |
| E03_fingergan | E15_norm_no_mix | -29.8 [-33.8, -25.8] | 2e-38 | B better | -0.376 [-0.382, -0.369] | 2e-64 | B better |
| E03_fingergan | E16_norm_no_aux | -29.1 [-33.1, -25.0] | 6e-36 | B better | -0.384 [-0.391, -0.377] | 2e-64 | B better |
| E03_fingergan | E17_fingergan_norm | -19.0 [-22.4, -15.7] | 3e-23 | B better | -0.086 [-0.090, -0.082] | 2e-64 | B better |
| E03_fingergan | E18_msu_stft_norm | -26.5 [-30.4, -22.8] | 5e-35 | B better | -0.281 [-0.289, -0.273] | 2e-64 | B better |
| E03_fingergan | E19_msu_aec_norm | -22.1 [-25.8, -18.5] | 9e-27 | B better | -0.166 [-0.171, -0.160] | 2e-64 | B better |
| E03_fingergan | E20_fingergan_finetune | -27.1 [-31.0, -23.2] | 6e-35 | B better | -0.300 [-0.307, -0.293] | 2e-64 | B better |
| E03_fingergan | E21_ccdf_aux_minutiae | -29.9 [-33.9, -25.9] | 2e-38 | B better | -0.386 [-0.392, -0.379] | 2e-64 | B better |
| E03_fingergan | E22_ccdf_aux_minutiae_cldice | -30.2 [-34.3, -26.1] | 1e-38 | B better | -0.397 [-0.404, -0.390] | 2e-64 | B better |
| E03_fingergan | E23_ccdf_aux_cv_edge_mix | -29.1 [-33.3, -25.1] | 3e-35 | B better | -0.389 [-0.395, -0.382] | 2e-64 | B better |
| E03_fingergan | E24_norm_no_cldice_e12sampler | -30.0 [-34.0, -26.0] | 7e-39 | B better | -0.381 [-0.388, -0.375] | 2e-64 | B better |
| E03_fingergan | E25_norm_no_edge_e12sampler | -29.5 [-33.5, -25.5] | 2e-39 | B better | -0.384 [-0.391, -0.378] | 2e-64 | B better |
| E03_fingergan | E26_raw_ccdf_cldice_cv | -29.8 [-33.9, -25.7] | 3e-37 | B better | -0.395 [-0.402, -0.388] | 2e-64 | B better |
| E04_msu_stft | E05_msu_aec | +53.6 [+49.5, +57.8] | 4e-77 | A better | +0.311 [+0.300, +0.321] | 2e-64 | A better |
| E04_msu_stft | E06_aux_minutiae | -3.5 [-5.4, -1.8] | 0.027 | B better | -0.112 [-0.116, -0.108] | 2e-64 | B better |
| E04_msu_stft | E07_aux_minutiae_cldice | -2.8 [-4.6, -0.9] | 0.592 | unresolved | -0.111 [-0.115, -0.107] | 2e-64 | B better |
| E04_msu_stft | E08_aux_cv_edge_mix | -3.2 [-5.0, -1.4] | 0.137 | unresolved | -0.113 [-0.117, -0.109] | 2e-64 | B better |
| E04_msu_stft | E09_ccdf | -3.1 [-4.9, -1.4] | 0.106 | unresolved | -0.123 [-0.127, -0.119] | 2e-64 | B better |
| E04_msu_stft | E10_ctmsr | +2.8 [+1.0, +4.7] | 0.691 | unresolved | +0.019 [+0.017, +0.021] | 4e-54 | A better |
| E04_msu_stft | E11_ctmsr_dtm | +3.2 [+1.3, +5.1] | 0.266 | unresolved | +0.022 [+0.020, +0.023] | 8e-57 | A better |
| E04_msu_stft | E12_norm_ccdf_cldice_cv | -3.4 [-5.3, -1.3] | 0.218 | unresolved | -0.108 [-0.112, -0.104] | 2e-64 | B better |
| E04_msu_stft | E13_norm_no_cldice | -2.5 [-4.3, -0.8] | 0.953 | unresolved | -0.085 [-0.089, -0.082] | 2e-64 | B better |
| E04_msu_stft | E14_norm_no_edge | -2.7 [-4.5, -0.8] | 0.838 | unresolved | -0.088 [-0.091, -0.084] | 2e-64 | B better |
| E04_msu_stft | E15_norm_no_mix | -3.4 [-5.2, -1.7] | 0.032 | B better | -0.102 [-0.105, -0.098] | 2e-64 | B better |
| E04_msu_stft | E16_norm_no_aux | -2.7 [-4.6, -0.8] | 1.000 | unresolved | -0.110 [-0.114, -0.106] | 2e-64 | B better |
| E04_msu_stft | E17_fingergan_norm | +7.4 [+5.0, +9.6] | 1e-07 | A better | +0.188 [+0.180, +0.196] | 2e-64 | A better |
| E04_msu_stft | E18_msu_stft_norm | -0.2 [-1.8, +1.5] | 1.000 | equivalent | -0.007 [-0.008, -0.007] | 4e-58 | equivalent |
| E04_msu_stft | E19_msu_aec_norm | +4.3 [+2.3, +6.3] | 0.010 | A better | +0.108 [+0.101, +0.116] | 2e-62 | A better |
| E04_msu_stft | E20_fingergan_finetune | -0.7 [-2.4, +1.1] | 1.000 | equivalent | -0.026 [-0.027, -0.025] | 5e-64 | B better |
| E04_msu_stft | E21_ccdf_aux_minutiae | -3.5 [-5.4, -1.8] | 0.027 | B better | -0.112 [-0.116, -0.108] | 2e-64 | B better |
| E04_msu_stft | E22_ccdf_aux_minutiae_cldice | -3.9 [-5.7, -2.0] | 0.013 | B better | -0.123 [-0.127, -0.119] | 2e-64 | B better |
| E04_msu_stft | E23_ccdf_aux_cv_edge_mix | -2.8 [-4.7, -0.8] | 0.952 | unresolved | -0.115 [-0.119, -0.111] | 2e-64 | B better |
| E04_msu_stft | E24_norm_no_cldice_e12sampler | -3.6 [-5.5, -1.7] | 0.055 | unresolved | -0.108 [-0.111, -0.104] | 2e-64 | B better |
| E04_msu_stft | E25_norm_no_edge_e12sampler | -3.1 [-4.9, -1.3] | 0.212 | unresolved | -0.111 [-0.115, -0.107] | 2e-64 | B better |
| E04_msu_stft | E26_raw_ccdf_cldice_cv | -3.4 [-5.4, -1.5] | 0.145 | unresolved | -0.121 [-0.125, -0.117] | 2e-64 | B better |
| E05_msu_aec | E06_aux_minutiae | -57.1 [-61.4, -52.8] | 2e-79 | B better | -0.422 [-0.433, -0.412] | 2e-64 | B better |
| E05_msu_aec | E07_aux_minutiae_cldice | -56.4 [-60.6, -52.2] | 4e-78 | B better | -0.421 [-0.432, -0.411] | 2e-64 | B better |
| E05_msu_aec | E08_aux_cv_edge_mix | -56.8 [-61.1, -52.5] | 3e-78 | B better | -0.423 [-0.434, -0.413] | 2e-64 | B better |
| E05_msu_aec | E09_ccdf | -56.7 [-61.0, -52.4] | 2e-77 | B better | -0.434 [-0.444, -0.423] | 2e-64 | B better |
| E05_msu_aec | E10_ctmsr | -50.8 [-54.9, -46.6] | 2e-70 | B better | -0.292 [-0.301, -0.282] | 2e-64 | B better |
| E05_msu_aec | E11_ctmsr_dtm | -50.4 [-54.7, -46.2] | 5e-69 | B better | -0.289 [-0.299, -0.279] | 2e-64 | B better |
| E05_msu_aec | E12_norm_ccdf_cldice_cv | -57.0 [-61.2, -52.7] | 5e-78 | B better | -0.419 [-0.429, -0.408] | 2e-64 | B better |
| E05_msu_aec | E13_norm_no_cldice | -56.1 [-60.4, -51.9] | 2e-78 | B better | -0.396 [-0.406, -0.386] | 2e-64 | B better |
| E05_msu_aec | E14_norm_no_edge | -56.3 [-60.5, -52.1] | 1e-79 | B better | -0.398 [-0.408, -0.388] | 2e-64 | B better |
| E05_msu_aec | E15_norm_no_mix | -57.0 [-61.2, -52.8] | 3e-78 | B better | -0.412 [-0.422, -0.402] | 2e-64 | B better |
| E05_msu_aec | E16_norm_no_aux | -56.3 [-60.6, -52.0] | 9e-79 | B better | -0.420 [-0.431, -0.410] | 2e-64 | B better |
| E05_msu_aec | E17_fingergan_norm | -46.2 [-50.3, -42.1] | 3e-70 | B better | -0.122 [-0.131, -0.114] | 5e-60 | B better |
| E05_msu_aec | E18_msu_stft_norm | -53.8 [-58.0, -49.7] | 1e-76 | B better | -0.318 [-0.328, -0.307] | 2e-64 | B better |
| E05_msu_aec | E19_msu_aec_norm | -49.3 [-53.5, -45.3] | 2e-74 | B better | -0.202 [-0.208, -0.197] | 2e-64 | B better |
| E05_msu_aec | E20_fingergan_finetune | -54.3 [-58.5, -50.1] | 5e-76 | B better | -0.337 [-0.346, -0.327] | 2e-64 | B better |
| E05_msu_aec | E21_ccdf_aux_minutiae | -57.1 [-61.4, -52.8] | 2e-79 | B better | -0.422 [-0.433, -0.412] | 2e-64 | B better |
| E05_msu_aec | E22_ccdf_aux_minutiae_cldice | -57.5 [-61.7, -53.2] | 3e-79 | B better | -0.434 [-0.444, -0.423] | 2e-64 | B better |
| E05_msu_aec | E23_ccdf_aux_cv_edge_mix | -56.4 [-60.7, -52.1] | 3e-78 | B better | -0.425 [-0.436, -0.415] | 2e-64 | B better |
| E05_msu_aec | E24_norm_no_cldice_e12sampler | -57.2 [-61.4, -53.0] | 3e-79 | B better | -0.418 [-0.428, -0.408] | 2e-64 | B better |
| E05_msu_aec | E25_norm_no_edge_e12sampler | -56.7 [-60.9, -52.5] | 3e-80 | B better | -0.421 [-0.431, -0.411] | 2e-64 | B better |
| E05_msu_aec | E26_raw_ccdf_cldice_cv | -57.0 [-61.3, -52.8] | 4e-78 | B better | -0.431 [-0.442, -0.421] | 2e-64 | B better |
| E06_aux_minutiae | E07_aux_minutiae_cldice | +0.8 [-0.6, +2.1] | 1.000 | equivalent | +0.001 [+0.000, +0.002] | 0.008 | equivalent |
| E06_aux_minutiae | E08_aux_cv_edge_mix | +0.3 [-1.0, +1.6] | 1.000 | equivalent | -0.001 [-0.002, -0.000] | 7e-04 | equivalent |
| E06_aux_minutiae | E09_ccdf | +0.4 [-0.8, +1.7] | 1.000 | equivalent | -0.011 [-0.012, -0.010] | 9e-59 | B better |
| E06_aux_minutiae | E10_ctmsr | +6.4 [+4.6, +8.2] | 4e-10 | A better | +0.131 [+0.126, +0.135] | 2e-64 | A better |
| E06_aux_minutiae | E11_ctmsr_dtm | +6.7 [+4.9, +8.5] | 3e-11 | A better | +0.133 [+0.129, +0.138] | 2e-64 | A better |
| E06_aux_minutiae | E12_norm_ccdf_cldice_cv | +0.2 [-1.4, +1.8] | 1.000 | equivalent | +0.004 [+0.002, +0.005] | 9e-09 | equivalent |
| E06_aux_minutiae | E13_norm_no_cldice | +1.0 [-0.3, +2.3] | 1.000 | equivalent | +0.027 [+0.025, +0.028] | 2e-64 | A better |
| E06_aux_minutiae | E14_norm_no_edge | +0.8 [-0.5, +2.2] | 1.000 | equivalent | +0.024 [+0.023, +0.026] | 2e-64 | A better |
| E06_aux_minutiae | E15_norm_no_mix | +0.1 [-1.3, +1.4] | 1.000 | equivalent | +0.010 [+0.009, +0.011] | 1e-38 | A better |
| E06_aux_minutiae | E16_norm_no_aux | +0.8 [-0.5, +2.2] | 1.000 | equivalent | +0.002 [+0.001, +0.003] | 9e-04 | equivalent |
| E06_aux_minutiae | E17_fingergan_norm | +10.9 [+8.4, +13.4] | 2e-15 | A better | +0.300 [+0.292, +0.308] | 2e-64 | A better |
| E06_aux_minutiae | E18_msu_stft_norm | +3.4 [+1.5, +5.4] | 0.132 | unresolved | +0.105 [+0.101, +0.109] | 2e-64 | A better |
| E06_aux_minutiae | E19_msu_aec_norm | +7.8 [+5.9, +9.7] | 1e-12 | A better | +0.220 [+0.214, +0.227] | 2e-64 | A better |
| E06_aux_minutiae | E20_fingergan_finetune | +2.8 [+1.0, +4.7] | 0.539 | unresolved | +0.086 [+0.082, +0.089] | 2e-64 | A better |
| E06_aux_minutiae | E21_ccdf_aux_minutiae | +0.0 [+0.0, +0.0] | 1.000 | equivalent | +0.000 [+0.000, +0.000] | 1.000 | equivalent |
| E06_aux_minutiae | E22_ccdf_aux_minutiae_cldice | -0.3 [-1.8, +1.1] | 1.000 | equivalent | -0.011 [-0.012, -0.010] | 7e-58 | B better |
| E06_aux_minutiae | E23_ccdf_aux_cv_edge_mix | +0.8 [-0.5, +2.0] | 1.000 | equivalent | -0.003 [-0.004, -0.002] | 9e-18 | equivalent |
| E06_aux_minutiae | E24_norm_no_cldice_e12sampler | -0.1 [-1.5, +1.3] | 1.000 | equivalent | +0.004 [+0.003, +0.006] | 1e-11 | equivalent |
| E06_aux_minutiae | E25_norm_no_edge_e12sampler | +0.4 [-0.9, +1.8] | 1.000 | equivalent | +0.001 [+0.000, +0.002] | 0.013 | equivalent |
| E06_aux_minutiae | E26_raw_ccdf_cldice_cv | +0.1 [-1.3, +1.5] | 1.000 | equivalent | -0.009 [-0.010, -0.008] | 6e-52 | equivalent |
| E07_aux_minutiae_cldice | E08_aux_cv_edge_mix | -0.4 [-1.8, +0.9] | 1.000 | equivalent | -0.002 [-0.003, -0.001] | 9e-11 | equivalent |
| E07_aux_minutiae_cldice | E09_ccdf | -0.3 [-1.5, +0.8] | 1.000 | equivalent | -0.012 [-0.013, -0.011] | 3e-60 | B better |
| E07_aux_minutiae_cldice | E10_ctmsr | +5.6 [+3.7, +7.6] | 2e-06 | A better | +0.130 [+0.125, +0.134] | 2e-64 | A better |
| E07_aux_minutiae_cldice | E11_ctmsr_dtm | +5.9 [+4.1, +7.9] | 2e-07 | A better | +0.133 [+0.128, +0.137] | 2e-64 | A better |
| E07_aux_minutiae_cldice | E12_norm_ccdf_cldice_cv | -0.6 [-1.9, +0.8] | 1.000 | equivalent | +0.003 [+0.001, +0.004] | 5e-05 | equivalent |
| E07_aux_minutiae_cldice | E13_norm_no_cldice | +0.3 [-1.1, +1.6] | 1.000 | equivalent | +0.026 [+0.024, +0.027] | 4e-64 | A better |
| E07_aux_minutiae_cldice | E14_norm_no_edge | +0.1 [-1.3, +1.4] | 1.000 | equivalent | +0.023 [+0.022, +0.025] | 7e-64 | A better |
| E07_aux_minutiae_cldice | E15_norm_no_mix | -0.7 [-2.1, +0.8] | 1.000 | equivalent | +0.009 [+0.008, +0.011] | 8e-30 | A better |
| E07_aux_minutiae_cldice | E16_norm_no_aux | +0.1 [-1.3, +1.5] | 1.000 | equivalent | +0.001 [-0.000, +0.002] | 0.067 | equivalent |
| E07_aux_minutiae_cldice | E17_fingergan_norm | +10.1 [+7.6, +12.6] | 5e-13 | A better | +0.299 [+0.291, +0.307] | 2e-64 | A better |
| E07_aux_minutiae_cldice | E18_msu_stft_norm | +2.6 [+0.8, +4.5] | 1.000 | unresolved | +0.104 [+0.100, +0.108] | 2e-64 | A better |
| E07_aux_minutiae_cldice | E19_msu_aec_norm | +7.0 [+5.0, +9.1] | 2e-09 | A better | +0.219 [+0.213, +0.226] | 2e-64 | A better |
| E07_aux_minutiae_cldice | E20_fingergan_finetune | +2.1 [+0.3, +4.0] | 1.000 | unresolved | +0.085 [+0.081, +0.089] | 2e-64 | A better |
| E07_aux_minutiae_cldice | E21_ccdf_aux_minutiae | -0.8 [-2.1, +0.6] | 1.000 | equivalent | -0.001 [-0.002, -0.000] | 0.008 | equivalent |
| E07_aux_minutiae_cldice | E22_ccdf_aux_minutiae_cldice | -1.1 [-2.4, +0.3] | 1.000 | equivalent | -0.012 [-0.013, -0.011] | 2e-63 | B better |
| E07_aux_minutiae_cldice | E23_ccdf_aux_cv_edge_mix | +0.0 [-1.3, +1.2] | 1.000 | equivalent | -0.004 [-0.005, -0.003] | 6e-25 | equivalent |
| E07_aux_minutiae_cldice | E24_norm_no_cldice_e12sampler | -0.8 [-2.3, +0.6] | 1.000 | equivalent | +0.003 [+0.002, +0.005] | 3e-07 | equivalent |
| E07_aux_minutiae_cldice | E25_norm_no_edge_e12sampler | -0.3 [-1.7, +1.0] | 1.000 | equivalent | +0.000 [-0.001, +0.002] | 0.317 | equivalent |
| E07_aux_minutiae_cldice | E26_raw_ccdf_cldice_cv | -0.7 [-2.0, +0.7] | 1.000 | equivalent | -0.010 [-0.011, -0.009] | 2e-54 | B better |
| E08_aux_cv_edge_mix | E09_ccdf | +0.1 [-1.1, +1.3] | 1.000 | equivalent | -0.010 [-0.011, -0.009] | 2e-53 | B better |
| E08_aux_cv_edge_mix | E10_ctmsr | +6.0 [+4.1, +8.1] | 6e-07 | A better | +0.132 [+0.127, +0.136] | 2e-64 | A better |
| E08_aux_cv_edge_mix | E11_ctmsr_dtm | +6.4 [+4.6, +8.3] | 4e-10 | A better | +0.134 [+0.130, +0.139] | 2e-64 | A better |
| E08_aux_cv_edge_mix | E12_norm_ccdf_cldice_cv | -0.2 [-1.7, +1.3] | 1.000 | equivalent | +0.005 [+0.003, +0.006] | 1e-12 | equivalent |
| E08_aux_cv_edge_mix | E13_norm_no_cldice | +0.7 [-0.8, +2.1] | 1.000 | equivalent | +0.028 [+0.026, +0.029] | 2e-64 | A better |
| E08_aux_cv_edge_mix | E14_norm_no_edge | +0.5 [-0.9, +1.9] | 1.000 | equivalent | +0.025 [+0.024, +0.027] | 2e-64 | A better |
| E08_aux_cv_edge_mix | E15_norm_no_mix | -0.3 [-1.6, +1.1] | 1.000 | equivalent | +0.011 [+0.010, +0.012] | 4e-41 | A better |
| E08_aux_cv_edge_mix | E16_norm_no_aux | +0.5 [-0.8, +1.8] | 1.000 | equivalent | +0.003 [+0.002, +0.004] | 7e-07 | equivalent |
| E08_aux_cv_edge_mix | E17_fingergan_norm | +10.6 [+8.0, +13.1] | 3e-14 | A better | +0.301 [+0.293, +0.309] | 2e-64 | A better |
| E08_aux_cv_edge_mix | E18_msu_stft_norm | +3.0 [+1.1, +5.0] | 0.556 | unresolved | +0.106 [+0.102, +0.110] | 2e-64 | A better |
| E08_aux_cv_edge_mix | E19_msu_aec_norm | +7.5 [+5.4, +9.5] | 6e-11 | A better | +0.221 [+0.215, +0.228] | 2e-64 | A better |
| E08_aux_cv_edge_mix | E20_fingergan_finetune | +2.5 [+0.6, +4.5] | 1.000 | unresolved | +0.087 [+0.083, +0.090] | 2e-64 | A better |
| E08_aux_cv_edge_mix | E21_ccdf_aux_minutiae | -0.3 [-1.6, +1.0] | 1.000 | equivalent | +0.001 [+0.000, +0.002] | 7e-04 | equivalent |
| E08_aux_cv_edge_mix | E22_ccdf_aux_minutiae_cldice | -0.7 [-2.2, +0.8] | 1.000 | equivalent | -0.010 [-0.011, -0.009] | 4e-54 | B better |
| E08_aux_cv_edge_mix | E23_ccdf_aux_cv_edge_mix | +0.4 [-0.8, +1.8] | 1.000 | equivalent | -0.002 [-0.002, -0.002] | 1e-18 | equivalent |
| E08_aux_cv_edge_mix | E24_norm_no_cldice_e12sampler | -0.4 [-1.8, +1.0] | 1.000 | equivalent | +0.005 [+0.004, +0.007] | 9e-15 | equivalent |
| E08_aux_cv_edge_mix | E25_norm_no_edge_e12sampler | +0.1 [-1.1, +1.3] | 1.000 | equivalent | +0.002 [+0.001, +0.004] | 6e-05 | equivalent |
| E08_aux_cv_edge_mix | E26_raw_ccdf_cldice_cv | -0.3 [-1.6, +1.1] | 1.000 | equivalent | -0.008 [-0.009, -0.007] | 8e-46 | equivalent |
| E09_ccdf | E10_ctmsr | +5.9 [+4.2, +7.8] | 1e-08 | A better | +0.142 [+0.137, +0.147] | 2e-64 | A better |
| E09_ccdf | E11_ctmsr_dtm | +6.3 [+4.6, +8.0] | 4e-11 | A better | +0.145 [+0.140, +0.150] | 2e-64 | A better |
| E09_ccdf | E12_norm_ccdf_cldice_cv | -0.3 [-1.5, +1.0] | 1.000 | equivalent | +0.015 [+0.014, +0.016] | 1e-63 | A better |
| E09_ccdf | E13_norm_no_cldice | +0.6 [-0.8, +1.9] | 1.000 | equivalent | +0.038 [+0.036, +0.040] | 2e-64 | A better |
| E09_ccdf | E14_norm_no_edge | +0.4 [-0.9, +1.7] | 1.000 | equivalent | +0.036 [+0.034, +0.037] | 2e-64 | A better |
| E09_ccdf | E15_norm_no_mix | -0.3 [-1.5, +0.8] | 1.000 | equivalent | +0.021 [+0.020, +0.023] | 2e-64 | A better |
| E09_ccdf | E16_norm_no_aux | +0.4 [-0.8, +1.6] | 1.000 | equivalent | +0.013 [+0.012, +0.014] | 2e-62 | A better |
| E09_ccdf | E17_fingergan_norm | +10.5 [+7.9, +13.1] | 4e-13 | A better | +0.311 [+0.303, +0.319] | 2e-64 | A better |
| E09_ccdf | E18_msu_stft_norm | +2.9 [+1.1, +4.9] | 0.546 | unresolved | +0.116 [+0.112, +0.120] | 2e-64 | A better |
| E09_ccdf | E19_msu_aec_norm | +7.4 [+5.4, +9.4] | 8e-11 | A better | +0.231 [+0.225, +0.238] | 2e-64 | A better |
| E09_ccdf | E20_fingergan_finetune | +2.4 [+0.8, +4.2] | 1.000 | unresolved | +0.097 [+0.093, +0.101] | 2e-64 | A better |
| E09_ccdf | E21_ccdf_aux_minutiae | -0.4 [-1.7, +0.8] | 1.000 | equivalent | +0.011 [+0.010, +0.012] | 9e-59 | A better |
| E09_ccdf | E22_ccdf_aux_minutiae_cldice | -0.8 [-2.0, +0.4] | 1.000 | equivalent | +0.000 [-0.000, +0.001] | 0.348 | equivalent |
| E09_ccdf | E23_ccdf_aux_cv_edge_mix | +0.3 [-1.0, +1.6] | 1.000 | equivalent | +0.008 [+0.007, +0.009] | 2e-47 | equivalent |
| E09_ccdf | E24_norm_no_cldice_e12sampler | -0.5 [-1.8, +0.8] | 1.000 | equivalent | +0.016 [+0.015, +0.017] | 4e-63 | A better |
| E09_ccdf | E25_norm_no_edge_e12sampler | +0.0 [-1.3, +1.3] | 1.000 | equivalent | +0.013 [+0.012, +0.014] | 5e-62 | A better |
| E09_ccdf | E26_raw_ccdf_cldice_cv | -0.3 [-1.6, +0.8] | 1.000 | equivalent | +0.002 [+0.002, +0.003] | 2e-12 | equivalent |
| E10_ctmsr | E11_ctmsr_dtm | +0.3 [-1.0, +1.7] | 1.000 | equivalent | +0.003 [+0.002, +0.003] | 2e-57 | equivalent |
| E10_ctmsr | E12_norm_ccdf_cldice_cv | -6.2 [-8.1, -4.4] | 7e-09 | B better | -0.127 [-0.132, -0.123] | 2e-64 | B better |
| E10_ctmsr | E13_norm_no_cldice | -5.4 [-7.1, -3.7] | 2e-07 | B better | -0.104 [-0.108, -0.100] | 2e-64 | B better |
| E10_ctmsr | E14_norm_no_edge | -5.5 [-7.4, -3.8] | 4e-07 | B better | -0.106 [-0.110, -0.103] | 2e-64 | B better |
| E10_ctmsr | E15_norm_no_mix | -6.3 [-8.3, -4.4] | 3e-08 | B better | -0.121 [-0.125, -0.116] | 2e-64 | B better |
| E10_ctmsr | E16_norm_no_aux | -5.5 [-7.4, -3.7] | 4e-07 | B better | -0.129 [-0.133, -0.124] | 2e-64 | B better |
| E10_ctmsr | E17_fingergan_norm | +4.5 [+2.3, +6.7] | 0.016 | A better | +0.169 [+0.161, +0.177] | 2e-64 | A better |
| E10_ctmsr | E18_msu_stft_norm | -3.0 [-5.0, -1.0] | 0.600 | unresolved | -0.026 [-0.028, -0.024] | 4e-64 | B better |
| E10_ctmsr | E19_msu_aec_norm | +1.4 [-0.3, +3.2] | 1.000 | equivalent | +0.089 [+0.082, +0.096] | 2e-59 | A better |
| E10_ctmsr | E20_fingergan_finetune | -3.5 [-5.4, -1.6] | 0.092 | unresolved | -0.045 [-0.047, -0.043] | 2e-64 | B better |
| E10_ctmsr | E21_ccdf_aux_minutiae | -6.4 [-8.2, -4.6] | 4e-10 | B better | -0.131 [-0.135, -0.126] | 2e-64 | B better |
| E10_ctmsr | E22_ccdf_aux_minutiae_cldice | -6.7 [-8.7, -4.8] | 3e-09 | B better | -0.142 [-0.147, -0.137] | 2e-64 | B better |
| E10_ctmsr | E23_ccdf_aux_cv_edge_mix | -5.6 [-7.5, -3.8] | 2e-06 | B better | -0.134 [-0.138, -0.129] | 2e-64 | B better |
| E10_ctmsr | E24_norm_no_cldice_e12sampler | -6.4 [-8.3, -4.7] | 2e-10 | B better | -0.126 [-0.131, -0.122] | 2e-64 | B better |
| E10_ctmsr | E25_norm_no_edge_e12sampler | -5.9 [-7.8, -4.2] | 3e-08 | B better | -0.129 [-0.134, -0.125] | 2e-64 | B better |
| E10_ctmsr | E26_raw_ccdf_cldice_cv | -6.3 [-8.3, -4.4] | 3e-08 | B better | -0.140 [-0.145, -0.135] | 2e-64 | B better |
| E11_ctmsr_dtm | E12_norm_ccdf_cldice_cv | -6.5 [-8.4, -4.8] | 2e-10 | B better | -0.130 [-0.134, -0.125] | 2e-64 | B better |
| E11_ctmsr_dtm | E13_norm_no_cldice | -5.7 [-7.5, -3.9] | 1e-07 | B better | -0.107 [-0.110, -0.103] | 2e-64 | B better |
| E11_ctmsr_dtm | E14_norm_no_edge | -5.9 [-7.7, -4.2] | 2e-08 | B better | -0.109 [-0.113, -0.105] | 2e-64 | B better |
| E11_ctmsr_dtm | E15_norm_no_mix | -6.6 [-8.5, -4.8] | 8e-10 | B better | -0.123 [-0.127, -0.119] | 2e-64 | B better |
| E11_ctmsr_dtm | E16_norm_no_aux | -5.9 [-7.7, -4.2] | 8e-09 | B better | -0.131 [-0.136, -0.127] | 2e-64 | B better |
| E11_ctmsr_dtm | E17_fingergan_norm | +4.2 [+1.9, +6.4] | 0.058 | unresolved | +0.167 [+0.159, +0.174] | 2e-64 | A better |
| E11_ctmsr_dtm | E18_msu_stft_norm | -3.4 [-5.3, -1.4] | 0.150 | unresolved | -0.029 [-0.030, -0.027] | 2e-64 | B better |
| E11_ctmsr_dtm | E19_msu_aec_norm | +1.1 [-0.7, +2.8] | 1.000 | equivalent | +0.087 [+0.080, +0.093] | 2e-58 | A better |
| E11_ctmsr_dtm | E20_fingergan_finetune | -3.9 [-5.8, -2.0] | 0.018 | B better | -0.048 [-0.049, -0.046] | 2e-64 | B better |
| E11_ctmsr_dtm | E21_ccdf_aux_minutiae | -6.7 [-8.5, -4.9] | 3e-11 | B better | -0.133 [-0.138, -0.129] | 2e-64 | B better |
| E11_ctmsr_dtm | E22_ccdf_aux_minutiae_cldice | -7.0 [-9.0, -5.1] | 1e-10 | B better | -0.145 [-0.149, -0.140] | 2e-64 | B better |
| E11_ctmsr_dtm | E23_ccdf_aux_cv_edge_mix | -5.9 [-7.8, -4.2] | 4e-08 | B better | -0.136 [-0.141, -0.132] | 2e-64 | B better |
| E11_ctmsr_dtm | E24_norm_no_cldice_e12sampler | -6.8 [-8.6, -5.0] | 6e-12 | B better | -0.129 [-0.134, -0.125] | 2e-64 | B better |
| E11_ctmsr_dtm | E25_norm_no_edge_e12sampler | -6.3 [-8.0, -4.7] | 2e-12 | B better | -0.132 [-0.137, -0.127] | 2e-64 | B better |
| E11_ctmsr_dtm | E26_raw_ccdf_cldice_cv | -6.6 [-8.5, -4.8] | 6e-10 | B better | -0.143 [-0.147, -0.138] | 2e-64 | B better |
| E12_norm_ccdf_cldice_cv | E13_norm_no_cldice | +0.8 [-0.8, +2.3] | 1.000 | equivalent | +0.023 [+0.022, +0.025] | 6e-64 | A better |
| E12_norm_ccdf_cldice_cv | E14_norm_no_edge | +0.7 [-0.8, +2.1] | 1.000 | equivalent | +0.021 [+0.019, +0.022] | 3e-63 | A better |
| E12_norm_ccdf_cldice_cv | E15_norm_no_mix | -0.1 [-1.4, +1.3] | 1.000 | equivalent | +0.007 [+0.006, +0.007] | 3e-56 | equivalent |
| E12_norm_ccdf_cldice_cv | E16_norm_no_aux | +0.7 [-0.8, +2.1] | 1.000 | equivalent | -0.001 [-0.002, -0.001] | 7e-08 | equivalent |
| E12_norm_ccdf_cldice_cv | E17_fingergan_norm | +10.7 [+8.2, +13.2] | 1e-14 | A better | +0.297 [+0.289, +0.304] | 2e-64 | A better |
| E12_norm_ccdf_cldice_cv | E18_msu_stft_norm | +3.2 [+1.3, +5.0] | 0.198 | unresolved | +0.101 [+0.097, +0.105] | 2e-64 | A better |
| E12_norm_ccdf_cldice_cv | E19_msu_aec_norm | +7.6 [+5.7, +9.6] | 7e-12 | A better | +0.217 [+0.210, +0.223] | 2e-64 | A better |
| E12_norm_ccdf_cldice_cv | E20_fingergan_finetune | +2.7 [+0.9, +4.4] | 0.596 | unresolved | +0.082 [+0.079, +0.086] | 2e-64 | A better |
| E12_norm_ccdf_cldice_cv | E21_ccdf_aux_minutiae | -0.2 [-1.8, +1.4] | 1.000 | equivalent | -0.004 [-0.005, -0.002] | 9e-09 | equivalent |
| E12_norm_ccdf_cldice_cv | E22_ccdf_aux_minutiae_cldice | -0.5 [-2.0, +0.9] | 1.000 | equivalent | -0.015 [-0.016, -0.014] | 5e-62 | B better |
| E12_norm_ccdf_cldice_cv | E23_ccdf_aux_cv_edge_mix | +0.6 [-0.8, +2.1] | 1.000 | equivalent | -0.006 [-0.008, -0.005] | 3e-25 | equivalent |
| E12_norm_ccdf_cldice_cv | E24_norm_no_cldice_e12sampler | -0.3 [-1.4, +0.8] | 1.000 | equivalent | +0.001 [+0.000, +0.001] | 0.004 | equivalent |
| E12_norm_ccdf_cldice_cv | E25_norm_no_edge_e12sampler | +0.3 [-0.9, +1.4] | 1.000 | equivalent | -0.002 [-0.003, -0.002] | 4e-23 | equivalent |
| E12_norm_ccdf_cldice_cv | E26_raw_ccdf_cldice_cv | -0.1 [-1.4, +1.3] | 1.000 | equivalent | -0.013 [-0.014, -0.012] | 2e-61 | B better |
| E13_norm_no_cldice | E14_norm_no_edge | -0.2 [-1.4, +1.1] | 1.000 | equivalent | -0.002 [-0.003, -0.002] | 7e-06 | equivalent |
| E13_norm_no_cldice | E15_norm_no_mix | -0.9 [-2.4, +0.5] | 1.000 | equivalent | -0.017 [-0.018, -0.015] | 4e-59 | B better |
| E13_norm_no_cldice | E16_norm_no_aux | -0.2 [-1.5, +1.2] | 1.000 | equivalent | -0.025 [-0.026, -0.023] | 2e-64 | B better |
| E13_norm_no_cldice | E17_fingergan_norm | +9.9 [+7.5, +12.2] | 1e-13 | A better | +0.273 [+0.266, +0.281] | 2e-64 | A better |
| E13_norm_no_cldice | E18_msu_stft_norm | +2.3 [+0.7, +4.1] | 1.000 | unresolved | +0.078 [+0.075, +0.081] | 2e-64 | A better |
| E13_norm_no_cldice | E19_msu_aec_norm | +6.8 [+4.9, +8.7] | 4e-10 | A better | +0.193 [+0.187, +0.200] | 2e-64 | A better |
| E13_norm_no_cldice | E20_fingergan_finetune | +1.8 [+0.1, +3.7] | 1.000 | unresolved | +0.059 [+0.056, +0.062] | 2e-64 | A better |
| E13_norm_no_cldice | E21_ccdf_aux_minutiae | -1.0 [-2.3, +0.3] | 1.000 | equivalent | -0.027 [-0.028, -0.025] | 2e-64 | B better |
| E13_norm_no_cldice | E22_ccdf_aux_minutiae_cldice | -1.3 [-2.9, +0.3] | 1.000 | equivalent | -0.038 [-0.040, -0.036] | 2e-64 | B better |
| E13_norm_no_cldice | E23_ccdf_aux_cv_edge_mix | -0.3 [-1.5, +1.1] | 1.000 | equivalent | -0.030 [-0.031, -0.028] | 2e-64 | B better |
| E13_norm_no_cldice | E24_norm_no_cldice_e12sampler | -1.1 [-2.5, +0.3] | 1.000 | equivalent | -0.022 [-0.024, -0.021] | 5e-64 | B better |
| E13_norm_no_cldice | E25_norm_no_edge_e12sampler | -0.6 [-2.0, +0.8] | 1.000 | equivalent | -0.025 [-0.027, -0.024] | 2e-64 | B better |
| E13_norm_no_cldice | E26_raw_ccdf_cldice_cv | -0.9 [-2.5, +0.7] | 1.000 | equivalent | -0.036 [-0.038, -0.034] | 2e-64 | B better |
| E14_norm_no_edge | E15_norm_no_mix | -0.8 [-2.1, +0.5] | 1.000 | equivalent | -0.014 [-0.016, -0.013] | 1e-52 | B better |
| E14_norm_no_edge | E16_norm_no_aux | +0.0 [-1.3, +1.2] | 1.000 | equivalent | -0.022 [-0.024, -0.021] | 2e-64 | B better |
| E14_norm_no_edge | E17_fingergan_norm | +10.1 [+7.7, +12.4] | 7e-15 | A better | +0.276 [+0.268, +0.283] | 2e-64 | A better |
| E14_norm_no_edge | E18_msu_stft_norm | +2.5 [+0.8, +4.4] | 1.000 | unresolved | +0.080 [+0.077, +0.084] | 2e-64 | A better |
| E14_norm_no_edge | E19_msu_aec_norm | +7.0 [+5.1, +9.0] | 3e-10 | A better | +0.196 [+0.190, +0.202] | 2e-64 | A better |
| E14_norm_no_edge | E20_fingergan_finetune | +2.0 [+0.4, +3.6] | 1.000 | unresolved | +0.062 [+0.058, +0.065] | 2e-64 | A better |
| E14_norm_no_edge | E21_ccdf_aux_minutiae | -0.8 [-2.2, +0.5] | 1.000 | equivalent | -0.024 [-0.026, -0.023] | 2e-64 | B better |
| E14_norm_no_edge | E22_ccdf_aux_minutiae_cldice | -1.2 [-2.4, +0.1] | 1.000 | equivalent | -0.036 [-0.037, -0.034] | 2e-64 | B better |
| E14_norm_no_edge | E23_ccdf_aux_cv_edge_mix | -0.1 [-1.4, +1.3] | 1.000 | equivalent | -0.027 [-0.029, -0.026] | 2e-64 | B better |
| E14_norm_no_edge | E24_norm_no_cldice_e12sampler | -0.9 [-2.2, +0.3] | 1.000 | equivalent | -0.020 [-0.021, -0.019] | 3e-61 | B better |
| E14_norm_no_edge | E25_norm_no_edge_e12sampler | -0.4 [-1.7, +0.8] | 1.000 | equivalent | -0.023 [-0.024, -0.022] | 2e-64 | B better |
| E14_norm_no_edge | E26_raw_ccdf_cldice_cv | -0.8 [-2.1, +0.6] | 1.000 | equivalent | -0.033 [-0.035, -0.032] | 2e-64 | B better |
| E15_norm_no_mix | E16_norm_no_aux | +0.8 [-0.6, +2.1] | 1.000 | equivalent | -0.008 [-0.009, -0.007] | 9e-61 | equivalent |
| E15_norm_no_mix | E17_fingergan_norm | +10.8 [+8.2, +13.4] | 4e-14 | A better | +0.290 [+0.282, +0.298] | 2e-64 | A better |
| E15_norm_no_mix | E18_msu_stft_norm | +3.3 [+1.4, +5.2] | 0.124 | unresolved | +0.095 [+0.091, +0.098] | 2e-64 | A better |
| E15_norm_no_mix | E19_msu_aec_norm | +7.7 [+5.7, +9.7] | 2e-11 | A better | +0.210 [+0.204, +0.216] | 2e-64 | A better |
| E15_norm_no_mix | E20_fingergan_finetune | +2.8 [+1.2, +4.4] | 0.225 | unresolved | +0.076 [+0.072, +0.079] | 2e-64 | A better |
| E15_norm_no_mix | E21_ccdf_aux_minutiae | -0.1 [-1.4, +1.3] | 1.000 | equivalent | -0.010 [-0.011, -0.009] | 1e-38 | B better |
| E15_norm_no_mix | E22_ccdf_aux_minutiae_cldice | -0.4 [-1.9, +1.1] | 1.000 | equivalent | -0.021 [-0.023, -0.020] | 2e-64 | B better |
| E15_norm_no_mix | E23_ccdf_aux_cv_edge_mix | +0.7 [-0.7, +2.1] | 1.000 | equivalent | -0.013 [-0.014, -0.012] | 1e-51 | B better |
| E15_norm_no_mix | E24_norm_no_cldice_e12sampler | -0.2 [-1.3, +1.0] | 1.000 | equivalent | -0.006 [-0.006, -0.005] | 5e-51 | equivalent |
| E15_norm_no_mix | E25_norm_no_edge_e12sampler | +0.3 [-0.8, +1.5] | 1.000 | equivalent | -0.009 [-0.010, -0.008] | 4e-62 | equivalent |
| E15_norm_no_mix | E26_raw_ccdf_cldice_cv | +0.0 [-1.3, +1.3] | 1.000 | equivalent | -0.019 [-0.020, -0.018] | 3e-64 | B better |
| E16_norm_no_aux | E17_fingergan_norm | +10.1 [+7.5, +12.6] | 5e-12 | A better | +0.298 [+0.290, +0.306] | 2e-64 | A better |
| E16_norm_no_aux | E18_msu_stft_norm | +2.5 [+0.6, +4.5] | 1.000 | unresolved | +0.103 [+0.099, +0.107] | 2e-64 | A better |
| E16_norm_no_aux | E19_msu_aec_norm | +7.0 [+4.9, +9.0] | 1e-08 | A better | +0.218 [+0.212, +0.224] | 2e-64 | A better |
| E16_norm_no_aux | E20_fingergan_finetune | +2.0 [+0.3, +3.8] | 1.000 | unresolved | +0.084 [+0.080, +0.087] | 2e-64 | A better |
| E16_norm_no_aux | E21_ccdf_aux_minutiae | -0.8 [-2.2, +0.5] | 1.000 | equivalent | -0.002 [-0.003, -0.001] | 9e-04 | equivalent |
| E16_norm_no_aux | E22_ccdf_aux_minutiae_cldice | -1.2 [-2.7, +0.3] | 1.000 | equivalent | -0.013 [-0.014, -0.012] | 2e-60 | B better |
| E16_norm_no_aux | E23_ccdf_aux_cv_edge_mix | -0.1 [-1.3, +1.2] | 1.000 | equivalent | -0.005 [-0.006, -0.004] | 4e-18 | equivalent |
| E16_norm_no_aux | E24_norm_no_cldice_e12sampler | -0.9 [-2.1, +0.2] | 1.000 | equivalent | +0.002 [+0.002, +0.003] | 3e-19 | equivalent |
| E16_norm_no_aux | E25_norm_no_edge_e12sampler | -0.4 [-1.6, +0.8] | 1.000 | equivalent | -0.001 [-0.001, -0.000] | 4e-04 | equivalent |
| E16_norm_no_aux | E26_raw_ccdf_cldice_cv | -0.8 [-2.2, +0.7] | 1.000 | equivalent | -0.011 [-0.012, -0.010] | 1e-58 | B better |
| E17_fingergan_norm | E18_msu_stft_norm | -7.5 [-10.0, -5.1] | 3e-07 | B better | -0.195 [-0.204, -0.187] | 2e-64 | B better |
| E17_fingergan_norm | E19_msu_aec_norm | -3.1 [-5.1, -1.1] | 0.645 | unresolved | -0.080 [-0.085, -0.075] | 2e-64 | B better |
| E17_fingergan_norm | E20_fingergan_finetune | -8.0 [-10.5, -5.5] | 5e-08 | B better | -0.214 [-0.222, -0.206] | 2e-64 | B better |
| E17_fingergan_norm | E21_ccdf_aux_minutiae | -10.9 [-13.4, -8.4] | 2e-15 | B better | -0.300 [-0.308, -0.292] | 2e-64 | B better |
| E17_fingergan_norm | E22_ccdf_aux_minutiae_cldice | -11.2 [-13.7, -8.6] | 3e-15 | B better | -0.311 [-0.319, -0.303] | 2e-64 | B better |
| E17_fingergan_norm | E23_ccdf_aux_cv_edge_mix | -10.1 [-12.6, -7.6] | 1e-12 | B better | -0.303 [-0.311, -0.295] | 2e-64 | B better |
| E17_fingergan_norm | E24_norm_no_cldice_e12sampler | -11.0 [-13.5, -8.5] | 2e-15 | B better | -0.296 [-0.304, -0.288] | 2e-64 | B better |
| E17_fingergan_norm | E25_norm_no_edge_e12sampler | -10.5 [-12.9, -8.0] | 5e-15 | B better | -0.299 [-0.306, -0.291] | 2e-64 | B better |
| E17_fingergan_norm | E26_raw_ccdf_cldice_cv | -10.8 [-13.4, -8.2] | 7e-14 | B better | -0.309 [-0.317, -0.301] | 2e-64 | B better |
| E18_msu_stft_norm | E19_msu_aec_norm | +4.4 [+2.6, +6.4] | 0.002 | A better | +0.115 [+0.108, +0.123] | 1e-63 | A better |
| E18_msu_stft_norm | E20_fingergan_finetune | -0.5 [-2.3, +1.3] | 1.000 | equivalent | -0.019 [-0.020, -0.018] | 1e-60 | B better |
| E18_msu_stft_norm | E21_ccdf_aux_minutiae | -3.4 [-5.4, -1.5] | 0.132 | unresolved | -0.105 [-0.109, -0.101] | 2e-64 | B better |
| E18_msu_stft_norm | E22_ccdf_aux_minutiae_cldice | -3.7 [-5.6, -1.8] | 0.050 | unresolved | -0.116 [-0.120, -0.111] | 2e-64 | B better |
| E18_msu_stft_norm | E23_ccdf_aux_cv_edge_mix | -2.6 [-4.6, -0.7] | 1.000 | unresolved | -0.108 [-0.112, -0.104] | 2e-64 | B better |
| E18_msu_stft_norm | E24_norm_no_cldice_e12sampler | -3.4 [-5.3, -1.6] | 0.060 | unresolved | -0.100 [-0.104, -0.096] | 2e-64 | B better |
| E18_msu_stft_norm | E25_norm_no_edge_e12sampler | -2.9 [-4.9, -1.1] | 0.384 | unresolved | -0.103 [-0.107, -0.099] | 2e-64 | B better |
| E18_msu_stft_norm | E26_raw_ccdf_cldice_cv | -3.3 [-5.4, -1.3] | 0.293 | unresolved | -0.114 [-0.118, -0.109] | 2e-64 | B better |
| E19_msu_aec_norm | E20_fingergan_finetune | -4.9 [-7.0, -2.9] | 6e-04 | B better | -0.134 [-0.141, -0.128] | 2e-64 | B better |
| E19_msu_aec_norm | E21_ccdf_aux_minutiae | -7.8 [-9.7, -5.9] | 1e-12 | B better | -0.220 [-0.227, -0.214] | 2e-64 | B better |
| E19_msu_aec_norm | E22_ccdf_aux_minutiae_cldice | -8.1 [-10.3, -6.0] | 3e-11 | B better | -0.231 [-0.238, -0.225] | 2e-64 | B better |
| E19_msu_aec_norm | E23_ccdf_aux_cv_edge_mix | -7.0 [-9.0, -5.1] | 2e-10 | B better | -0.223 [-0.230, -0.217] | 2e-64 | B better |
| E19_msu_aec_norm | E24_norm_no_cldice_e12sampler | -7.9 [-9.9, -5.9] | 4e-13 | B better | -0.216 [-0.222, -0.209] | 2e-64 | B better |
| E19_msu_aec_norm | E25_norm_no_edge_e12sampler | -7.4 [-9.3, -5.5] | 5e-12 | B better | -0.219 [-0.225, -0.212] | 2e-64 | B better |
| E19_msu_aec_norm | E26_raw_ccdf_cldice_cv | -7.7 [-9.7, -5.8] | 2e-12 | B better | -0.229 [-0.236, -0.223] | 2e-64 | B better |
| E20_fingergan_finetune | E21_ccdf_aux_minutiae | -2.8 [-4.7, -1.0] | 0.539 | unresolved | -0.086 [-0.089, -0.082] | 2e-64 | B better |
| E20_fingergan_finetune | E22_ccdf_aux_minutiae_cldice | -3.2 [-4.9, -1.5] | 0.094 | unresolved | -0.097 [-0.101, -0.093] | 2e-64 | B better |
| E20_fingergan_finetune | E23_ccdf_aux_cv_edge_mix | -2.1 [-4.0, -0.3] | 1.000 | unresolved | -0.089 [-0.092, -0.085] | 2e-64 | B better |
| E20_fingergan_finetune | E24_norm_no_cldice_e12sampler | -2.9 [-4.6, -1.3] | 0.141 | unresolved | -0.082 [-0.085, -0.078] | 2e-64 | B better |
| E20_fingergan_finetune | E25_norm_no_edge_e12sampler | -2.4 [-4.1, -0.8] | 0.906 | unresolved | -0.085 [-0.088, -0.081] | 2e-64 | B better |
| E20_fingergan_finetune | E26_raw_ccdf_cldice_cv | -2.8 [-4.7, -0.8] | 0.886 | unresolved | -0.095 [-0.099, -0.091] | 2e-64 | B better |
| E21_ccdf_aux_minutiae | E22_ccdf_aux_minutiae_cldice | -0.3 [-1.8, +1.1] | 1.000 | equivalent | -0.011 [-0.012, -0.010] | 7e-58 | B better |
| E21_ccdf_aux_minutiae | E23_ccdf_aux_cv_edge_mix | +0.8 [-0.5, +2.0] | 1.000 | equivalent | -0.003 [-0.004, -0.002] | 9e-18 | equivalent |
| E21_ccdf_aux_minutiae | E24_norm_no_cldice_e12sampler | -0.1 [-1.5, +1.3] | 1.000 | equivalent | +0.004 [+0.003, +0.006] | 1e-11 | equivalent |
| E21_ccdf_aux_minutiae | E25_norm_no_edge_e12sampler | +0.4 [-0.9, +1.8] | 1.000 | equivalent | +0.001 [+0.000, +0.002] | 0.013 | equivalent |
| E21_ccdf_aux_minutiae | E26_raw_ccdf_cldice_cv | +0.1 [-1.3, +1.5] | 1.000 | equivalent | -0.009 [-0.010, -0.008] | 6e-52 | equivalent |
| E22_ccdf_aux_minutiae_cldice | E23_ccdf_aux_cv_edge_mix | +1.1 [-0.3, +2.5] | 1.000 | equivalent | +0.008 [+0.007, +0.009] | 5e-49 | equivalent |
| E22_ccdf_aux_minutiae_cldice | E24_norm_no_cldice_e12sampler | +0.3 [-1.3, +1.8] | 1.000 | equivalent | +0.016 [+0.014, +0.017] | 3e-62 | A better |
| E22_ccdf_aux_minutiae_cldice | E25_norm_no_edge_e12sampler | +0.8 [-0.7, +2.2] | 1.000 | equivalent | +0.013 [+0.012, +0.013] | 1e-59 | A better |
| E22_ccdf_aux_minutiae_cldice | E26_raw_ccdf_cldice_cv | +0.4 [-0.9, +1.8] | 1.000 | equivalent | +0.002 [+0.002, +0.003] | 2e-16 | equivalent |
| E23_ccdf_aux_cv_edge_mix | E24_norm_no_cldice_e12sampler | -0.8 [-2.3, +0.6] | 1.000 | equivalent | +0.007 [+0.006, +0.009] | 1e-27 | equivalent |
| E23_ccdf_aux_cv_edge_mix | E25_norm_no_edge_e12sampler | -0.3 [-1.7, +0.9] | 1.000 | equivalent | +0.004 [+0.003, +0.005] | 3e-14 | equivalent |
| E23_ccdf_aux_cv_edge_mix | E26_raw_ccdf_cldice_cv | -0.7 [-2.0, +0.6] | 1.000 | equivalent | -0.006 [-0.007, -0.005] | 3e-38 | equivalent |
| E24_norm_no_cldice_e12sampler | E25_norm_no_edge_e12sampler | +0.5 [-0.6, +1.6] | 1.000 | equivalent | -0.003 [-0.004, -0.003] | 1e-25 | equivalent |
| E24_norm_no_cldice_e12sampler | E26_raw_ccdf_cldice_cv | +0.2 [-1.2, +1.6] | 1.000 | equivalent | -0.013 [-0.014, -0.012] | 8e-62 | B better |
| E25_norm_no_edge_e12sampler | E26_raw_ccdf_cldice_cv | -0.3 [-1.7, +1.1] | 1.000 | equivalent | -0.010 [-0.011, -0.010] | 2e-56 | B better |

</details>

Source: `evaluation/results/E02_resshift/test_20261023_172239`, `evaluation/results/E03_fingergan/test_20261023_175633`, `evaluation/results/E04_msu_stft/test_20261023_180254`, `evaluation/results/E05_msu_aec/test_20261023_181347`, `evaluation/results/E06_aux_minutiae/test_20261023_183632`, `evaluation/results/E07_aux_minutiae_cldice/test_20261023_190230`, `evaluation/results/E08_aux_cv_edge_mix/test_20261023_192942`, `evaluation/results/E09_ccdf/test_20261023_193652`, `evaluation/results/E10_ctmsr/test_20261023_195023`, `evaluation/results/E11_ctmsr_dtm/test_20261023_200339`, `evaluation/results/E12_norm_ccdf_cldice_cv/test_20261023_235228`, `evaluation/results/E13_norm_no_cldice/test_20261024_112659`, `evaluation/results/E14_norm_no_edge/test_20261024_045330`, `evaluation/results/E15_norm_no_mix/test_20261024_135435`, `evaluation/results/E16_norm_no_aux/test_20261024_042714`, `evaluation/results/E17_fingergan_norm/test_20261024_001802`, `evaluation/results/E18_msu_stft_norm/test_20261024_004312`, `evaluation/results/E19_msu_aec_norm/test_20261024_003029`, `evaluation/results/E20_fingergan_finetune/test_20261024_015919`, `evaluation/results/E21_ccdf_aux_minutiae/test_20261024_143815`, `evaluation/results/E22_ccdf_aux_minutiae_cldice/test_20261024_144531`, `evaluation/results/E23_ccdf_aux_cv_edge_mix/test_20261024_145210`, `evaluation/results/E24_norm_no_cldice_e12sampler/test_20261024_132110`, `evaluation/results/E25_norm_no_edge_e12sampler/test_20261024_134601`, `evaluation/results/E26_raw_ccdf_cldice_cv/test_20261024_230555`.
