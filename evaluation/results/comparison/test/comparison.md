### All experiments (test)

| experiment | rescued | broken | 1 net rescue, pts (95% CI) | p | 2 FMR change, pts (95% CI) | gate | 3 ridge fidelity | 3 change (95% CI) | EER % | s / print | GPU MB | RAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E02_resshift | 19 | 8 | +2.8 [+0.3, +5.3] | 0.052 | +0.02 [-0.14, +0.18] | pass | 0.930 | +0.103 [+0.095, +0.111] | 10.9 | 0.86 cuda | 678 | 1043 |
| E03_fingergan | 10 | 116 | -26.6 [-31.7, -21.9] | 5e-24 | -0.15 [-0.39, +0.08] | pass | 0.551 | -0.277 [-0.287, -0.267] | 20.8 | 1.16 cuda | 3598 | 1025 |
| **E04_msu_stft** | 20 | 14 | +1.5 [-1.3, +4.3] | 0.392 | +0.05 [-0.13, +0.23] | pass | 0.820 | -0.008 [-0.009, -0.006] | 11.8 | 0.22 cpu | – | 572 |
| E05_msu_aec | 3 | 211 | -52.3 [-57.3, -47.2] | 1e-58 | -0.84 [-1.05, -0.66] | pass | 0.515 | -0.312 [-0.326, -0.298] | 25.8 | 0.26 cuda | 37 | 972 |
| E06_aux_minutiae | 18 | 10 | +2.0 [-0.5, +4.5] | 0.185 | -0.10 [-0.27, +0.08] | pass | 0.931 | +0.104 [+0.096, +0.112] | 10.3 | 0.80 cuda | 678 | 1009 |
| E07_aux_minutiae_cldice | 25 | 10 | +3.8 [+1.0, +6.5] | 0.017 | -0.05 [-0.20, +0.10] | pass | 0.930 | +0.103 [+0.094, +0.111] | 11.3 | 0.83 cuda | 678 | 970 |
| E08_aux_cv_edge_mix | 23 | 8 | +3.8 [+1.0, +6.5] | 0.011 | -0.06 [-0.22, +0.11] | pass | 0.932 | +0.104 [+0.096, +0.113] | 11.4 | 0.83 cuda | 678 | 1036 |
| E09_ccdf | 24 | 7 | +4.3 [+1.8, +7.0] | 0.003 | +0.03 [-0.13, +0.18] | pass | 0.943 | +0.115 [+0.106, +0.125] | 11.2 | 0.48 cuda | 681 | 1011 |
| E10_ctmsr | 8 | 15 | -1.8 [-4.3, +0.5] | 0.210 | +0.01 [-0.15, +0.17] | pass | 0.800 | -0.027 [-0.030, -0.024] | 12.9 | 0.40 cuda | 675 | 1042 |
| E11_ctmsr_dtm | 11 | 18 | -1.8 [-4.5, +0.8] | 0.265 | -0.02 [-0.17, +0.13] | pass | 0.798 | -0.030 [-0.033, -0.027] | 12.8 | 0.40 cuda | 675 | 1019 |
| E21_ccdf_aux_minutiae | 21 | 10 | +2.8 [+0.0, +5.5] | 0.071 | -0.07 [-0.22, +0.09] | pass | 0.932 | +0.104 [+0.096, +0.113] | 10.9 | 0.79 cuda | 678 | 1026 |
| E22_ccdf_aux_minutiae_cldice | 24 | 10 | +3.5 [+0.8, +6.3] | 0.024 | -0.05 [-0.21, +0.11] | pass | 0.943 | +0.116 [+0.106, +0.125] | 10.8 | 0.43 cuda | 681 | 1031 |
| E23_ccdf_aux_cv_edge_mix | 25 | 5 | +5.0 [+2.3, +7.8] | 3e-04 | -0.04 [-0.20, +0.12] | pass | 0.935 | +0.107 [+0.099, +0.116] | 10.7 | 0.70 cuda | 682 | 1043 |

All samples (n = 398). Reference points: accepted at τ: clean 78.9%, damaged 72.4%; FMR damaged 0.89%; ridge fidelity damaged 0.827 (clean = 1); EER damaged 13.1%. p = exact sign-flip test over fingers (rescued vs broken; = exact McNemar with one sample per finger). CIs resample fingers.

### Every pair

Score 1 compares acceptance at τ (exact sign-flip test on per-finger sums; = McNemar with one sample per finger), score 3 the ridge fidelity in the mask (Wilcoxon signed-rank on per-finger means), sample by sample; CIs resample fingers. p-values are Holm-adjusted over all 91 pairs, per score. **equivalent**: the 90% CI of the difference lies within ±3 points (score 1) or ±0.01 (score 3), whatever the p-value. **A better / B better**: otherwise, adjusted p < 0.05. **unresolved**: neither. `damaged` = the unrepaired probes. Score 2 is a per-run gate (table above).

**Score 1 (identity recovery)**: row vs column, ▲ row better, ▼ row worse, ≈ equivalent, · unresolved

|  | dmg | E02 | E03 | E04 | E05 | E06 | E07 | E08 | E09 | E10 | E11 | E21 | E22 | E23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dmg |  | · | ▲ | · | ▲ | · | · | · | · | · | · | · | · | ▼ |
| E02 | · |  | ▲ | · | ▲ | ≈ | ≈ | ≈ | · | · | · | ≈ | ≈ | · |
| E03 | ▼ | ▼ |  | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E04 | · | · | ▲ |  | ▲ | · | · | · | · | · | · | · | · | · |
| E05 | ▼ | ▼ | ▼ | ▼ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E06 | · | ≈ | ▲ | · | ▲ |  | · | · | · | · | · | ≈ | · | · |
| E07 | · | ≈ | ▲ | · | ▲ | · |  | ≈ | ≈ | ▲ | ▲ | · | ≈ | · |
| E08 | · | ≈ | ▲ | · | ▲ | · | ≈ |  | ≈ | ▲ | ▲ | · | ≈ | · |
| E09 | · | · | ▲ | · | ▲ | · | ≈ | ≈ |  | ▲ | ▲ | · | ≈ | ≈ |
| E10 | · | · | ▲ | · | ▲ | · | ▼ | ▼ | ▼ |  | ≈ | · | ▼ | ▼ |
| E11 | · | · | ▲ | · | ▲ | · | ▼ | ▼ | ▼ | ≈ |  | · | · | ▼ |
| E21 | · | ≈ | ▲ | · | ▲ | ≈ | · | · | · | · | · |  | ≈ | · |
| E22 | · | ≈ | ▲ | · | ▲ | · | ≈ | ≈ | ≈ | ▲ | · | ≈ |  | · |
| E23 | ▲ | · | ▲ | · | ▲ | · | · | · | ≈ | ▲ | ▲ | · | · |  |

**Score 3 (ridge fidelity)**: same reading

|  | dmg | E02 | E03 | E04 | E05 | E06 | E07 | E08 | E09 | E10 | E11 | E21 | E22 | E23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dmg |  | ▼ | ▲ | ≈ | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▼ | ▼ | ▼ |
| E02 | ▲ |  | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▼ | ▲ | ▲ | ≈ | ▼ | ≈ |
| E03 | ▼ | ▼ |  | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E04 | ≈ | ▼ | ▲ |  | ▲ | ▼ | ▼ | ▼ | ▼ | ▲ | ▲ | ▼ | ▼ | ▼ |
| E05 | ▼ | ▼ | ▼ | ▼ |  | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ | ▼ |
| E06 | ▲ | ≈ | ▲ | ▲ | ▲ |  | ≈ | ≈ | ▼ | ▲ | ▲ | ≈ | ▼ | ≈ |
| E07 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ |  | ≈ | ▼ | ▲ | ▲ | ≈ | ▼ | ≈ |
| E08 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ |  | ▼ | ▲ | ▲ | ≈ | ▼ | ≈ |
| E09 | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ |  | ▲ | ▲ | ▲ | ≈ | ≈ |
| E10 | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ |  | ≈ | ▼ | ▼ | ▼ |
| E11 | ▼ | ▼ | ▲ | ▼ | ▲ | ▼ | ▼ | ▼ | ▼ | ≈ |  | ▼ | ▼ | ▼ |
| E21 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ▼ | ▲ | ▲ |  | ▼ | ≈ |
| E22 | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ≈ | ▲ | ▲ | ▲ |  | ≈ |
| E23 | ▲ | ≈ | ▲ | ▲ | ▲ | ≈ | ≈ | ≈ | ≈ | ▲ | ▲ | ≈ | ≈ |  |

<details><summary>All pairs with differences, CIs and adjusted p-values</summary>

| A | B | 1: A − B, pts (95% CI) | p (Holm) | verdict | 3: A − B (95% CI) | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|
| damaged | E02_resshift | -2.8 [-5.3, -0.3] | 1.000 | unresolved | -0.103 [-0.112, -0.095] | 5e-65 | B better |
| damaged | E03_fingergan | +26.6 [+21.9, +31.7] | 3e-22 | A better | +0.277 [+0.267, +0.286] | 5e-65 | A better |
| damaged | E04_msu_stft | -1.5 [-4.3, +1.3] | 1.000 | unresolved | +0.008 [+0.006, +0.009] | 1e-25 | equivalent |
| damaged | E05_msu_aec | +52.3 [+47.2, +57.3] | 1e-56 | A better | +0.312 [+0.298, +0.326] | 5e-65 | A better |
| damaged | E06_aux_minutiae | -2.0 [-4.5, +0.5] | 1.000 | unresolved | -0.104 [-0.113, -0.096] | 5e-65 | B better |
| damaged | E07_aux_minutiae_cldice | -3.8 [-6.8, -0.8] | 0.800 | unresolved | -0.103 [-0.111, -0.094] | 5e-65 | B better |
| damaged | E08_aux_cv_edge_mix | -3.8 [-6.5, -1.0] | 0.534 | unresolved | -0.104 [-0.113, -0.096] | 3e-64 | B better |
| damaged | E09_ccdf | -4.3 [-7.0, -1.8] | 0.176 | unresolved | -0.115 [-0.125, -0.106] | 5e-65 | B better |
| damaged | E10_ctmsr | +1.8 [-0.5, +4.3] | 1.000 | unresolved | +0.027 [+0.024, +0.030] | 1e-41 | A better |
| damaged | E11_ctmsr_dtm | +1.8 [-0.8, +4.5] | 1.000 | unresolved | +0.030 [+0.027, +0.033] | 5e-47 | A better |
| damaged | E21_ccdf_aux_minutiae | -2.8 [-5.5, +0.0] | 1.000 | unresolved | -0.104 [-0.113, -0.096] | 5e-65 | B better |
| damaged | E22_ccdf_aux_minutiae_cldice | -3.5 [-6.3, -0.8] | 1.000 | unresolved | -0.116 [-0.125, -0.106] | 5e-65 | B better |
| damaged | E23_ccdf_aux_cv_edge_mix | -5.0 [-7.8, -2.5] | 0.019 | B better | -0.107 [-0.116, -0.099] | 5e-65 | B better |
| E02_resshift | E03_fingergan | +29.4 [+24.4, +34.4] | 3e-26 | A better | +0.380 [+0.371, +0.388] | 5e-65 | A better |
| E02_resshift | E04_msu_stft | +1.3 [-1.8, +4.5] | 1.000 | unresolved | +0.111 [+0.102, +0.119] | 5e-65 | A better |
| E02_resshift | E05_msu_aec | +55.0 [+50.0, +60.1] | 3e-61 | A better | +0.415 [+0.403, +0.426] | 5e-65 | A better |
| E02_resshift | E06_aux_minutiae | +0.8 [-1.0, +2.5] | 1.000 | equivalent | -0.001 [-0.002, +0.000] | 6e-12 | equivalent |
| E02_resshift | E07_aux_minutiae_cldice | -1.0 [-3.0, +1.0] | 1.000 | equivalent | +0.000 [-0.001, +0.002] | 1e-05 | equivalent |
| E02_resshift | E08_aux_cv_edge_mix | -1.0 [-3.0, +1.0] | 1.000 | equivalent | -0.002 [-0.003, -0.000] | 2e-20 | equivalent |
| E02_resshift | E09_ccdf | -1.5 [-4.0, +1.0] | 1.000 | unresolved | -0.012 [-0.014, -0.011] | 2e-56 | B better |
| E02_resshift | E10_ctmsr | +4.5 [+1.8, +7.3] | 0.079 | unresolved | +0.130 [+0.122, +0.139] | 5e-65 | A better |
| E02_resshift | E11_ctmsr_dtm | +4.5 [+1.8, +7.3] | 0.114 | unresolved | +0.133 [+0.124, +0.142] | 5e-65 | A better |
| E02_resshift | E21_ccdf_aux_minutiae | +0.0 [-2.3, +2.3] | 1.000 | equivalent | -0.001 [-0.002, -0.000] | 5e-15 | equivalent |
| E02_resshift | E22_ccdf_aux_minutiae_cldice | -0.8 [-2.8, +1.3] | 1.000 | equivalent | -0.013 [-0.015, -0.011] | 2e-47 | B better |
| E02_resshift | E23_ccdf_aux_cv_edge_mix | -2.3 [-4.3, -0.3] | 1.000 | unresolved | -0.004 [-0.006, -0.003] | 6e-29 | equivalent |
| E03_fingergan | E04_msu_stft | -28.1 [-32.9, -23.4] | 1e-25 | B better | -0.269 [-0.278, -0.260] | 5e-65 | B better |
| E03_fingergan | E05_msu_aec | +25.6 [+20.4, +30.9] | 7e-18 | A better | +0.035 [+0.025, +0.046] | 1e-08 | A better |
| E03_fingergan | E06_aux_minutiae | -28.6 [-33.4, -23.9] | 4e-26 | B better | -0.381 [-0.389, -0.372] | 5e-65 | B better |
| E03_fingergan | E07_aux_minutiae_cldice | -30.4 [-35.4, -25.4] | 5e-28 | B better | -0.380 [-0.388, -0.371] | 5e-65 | B better |
| E03_fingergan | E08_aux_cv_edge_mix | -30.4 [-35.4, -25.4] | 5e-28 | B better | -0.381 [-0.390, -0.373] | 5e-65 | B better |
| E03_fingergan | E09_ccdf | -30.9 [-35.9, -25.9] | 7e-28 | B better | -0.392 [-0.401, -0.383] | 5e-65 | B better |
| E03_fingergan | E10_ctmsr | -24.9 [-29.9, -19.8] | 6e-19 | B better | -0.249 [-0.258, -0.240] | 5e-65 | B better |
| E03_fingergan | E11_ctmsr_dtm | -24.9 [-29.9, -19.8] | 1e-18 | B better | -0.247 [-0.256, -0.238] | 5e-65 | B better |
| E03_fingergan | E21_ccdf_aux_minutiae | -29.4 [-34.4, -24.6] | 7e-27 | B better | -0.381 [-0.390, -0.372] | 5e-65 | B better |
| E03_fingergan | E22_ccdf_aux_minutiae_cldice | -30.2 [-35.2, -25.1] | 2e-26 | B better | -0.392 [-0.402, -0.383] | 5e-65 | B better |
| E03_fingergan | E23_ccdf_aux_cv_edge_mix | -31.7 [-36.4, -26.6] | 4e-30 | B better | -0.384 [-0.393, -0.375] | 5e-65 | B better |
| E04_msu_stft | E05_msu_aec | +53.8 [+48.7, +58.5] | 1e-59 | A better | +0.305 [+0.290, +0.318] | 5e-65 | A better |
| E04_msu_stft | E06_aux_minutiae | -0.5 [-3.8, +2.8] | 1.000 | unresolved | -0.112 [-0.120, -0.103] | 5e-65 | B better |
| E04_msu_stft | E07_aux_minutiae_cldice | -2.3 [-5.5, +1.0] | 1.000 | unresolved | -0.110 [-0.119, -0.102] | 5e-65 | B better |
| E04_msu_stft | E08_aux_cv_edge_mix | -2.3 [-5.0, +0.5] | 1.000 | unresolved | -0.112 [-0.121, -0.104] | 1e-64 | B better |
| E04_msu_stft | E09_ccdf | -2.8 [-5.8, +0.3] | 1.000 | unresolved | -0.123 [-0.133, -0.114] | 5e-65 | B better |
| E04_msu_stft | E10_ctmsr | +3.3 [+0.3, +6.3] | 1.000 | unresolved | +0.020 [+0.017, +0.023] | 2e-34 | A better |
| E04_msu_stft | E11_ctmsr_dtm | +3.3 [+0.3, +6.3] | 1.000 | unresolved | +0.022 [+0.020, +0.025] | 3e-39 | A better |
| E04_msu_stft | E21_ccdf_aux_minutiae | -1.3 [-4.5, +2.0] | 1.000 | unresolved | -0.112 [-0.120, -0.103] | 5e-65 | B better |
| E04_msu_stft | E22_ccdf_aux_minutiae_cldice | -2.0 [-5.0, +1.0] | 1.000 | unresolved | -0.123 [-0.133, -0.114] | 5e-65 | B better |
| E04_msu_stft | E23_ccdf_aux_cv_edge_mix | -3.5 [-6.5, -0.5] | 1.000 | unresolved | -0.115 [-0.124, -0.106] | 5e-65 | B better |
| E05_msu_aec | E06_aux_minutiae | -54.3 [-59.3, -49.0] | 4e-59 | B better | -0.416 [-0.428, -0.404] | 5e-65 | B better |
| E05_msu_aec | E07_aux_minutiae_cldice | -56.0 [-61.1, -51.0] | 2e-62 | B better | -0.415 [-0.427, -0.403] | 5e-65 | B better |
| E05_msu_aec | E08_aux_cv_edge_mix | -56.0 [-60.8, -51.0] | 8e-64 | B better | -0.417 [-0.428, -0.405] | 5e-65 | B better |
| E05_msu_aec | E09_ccdf | -56.5 [-61.3, -51.5] | 5e-63 | B better | -0.427 [-0.439, -0.416] | 5e-65 | B better |
| E05_msu_aec | E10_ctmsr | -50.5 [-55.5, -45.5] | 1e-54 | B better | -0.285 [-0.298, -0.271] | 5e-65 | B better |
| E05_msu_aec | E11_ctmsr_dtm | -50.5 [-55.5, -45.5] | 1e-54 | B better | -0.282 [-0.295, -0.268] | 6e-65 | B better |
| E05_msu_aec | E21_ccdf_aux_minutiae | -55.0 [-59.8, -50.0] | 3e-61 | B better | -0.416 [-0.428, -0.404] | 5e-65 | B better |
| E05_msu_aec | E22_ccdf_aux_minutiae_cldice | -55.8 [-60.8, -50.8] | 4e-62 | B better | -0.428 [-0.439, -0.416] | 5e-65 | B better |
| E05_msu_aec | E23_ccdf_aux_cv_edge_mix | -57.3 [-62.1, -52.3] | 2e-65 | B better | -0.419 [-0.431, -0.408] | 5e-65 | B better |
| E06_aux_minutiae | E07_aux_minutiae_cldice | -1.8 [-4.0, +0.3] | 1.000 | unresolved | +0.001 [+0.000, +0.002] | 1e-08 | equivalent |
| E06_aux_minutiae | E08_aux_cv_edge_mix | -1.8 [-4.0, +0.5] | 1.000 | unresolved | -0.001 [-0.002, +0.001] | 2e-07 | equivalent |
| E06_aux_minutiae | E09_ccdf | -2.3 [-4.5, +0.0] | 1.000 | unresolved | -0.011 [-0.013, -0.010] | 8e-19 | B better |
| E06_aux_minutiae | E10_ctmsr | +3.8 [+1.0, +6.5] | 0.534 | unresolved | +0.131 [+0.123, +0.140] | 5e-65 | A better |
| E06_aux_minutiae | E11_ctmsr_dtm | +3.8 [+1.0, +6.8] | 0.800 | unresolved | +0.134 [+0.125, +0.143] | 5e-65 | A better |
| E06_aux_minutiae | E21_ccdf_aux_minutiae | -0.8 [-2.8, +1.3] | 1.000 | equivalent | -0.000 [-0.001, +0.000] | 0.327 | equivalent |
| E06_aux_minutiae | E22_ccdf_aux_minutiae_cldice | -1.5 [-4.0, +0.8] | 1.000 | unresolved | -0.012 [-0.014, -0.010] | 3e-26 | B better |
| E06_aux_minutiae | E23_ccdf_aux_cv_edge_mix | -3.0 [-5.3, -0.8] | 0.800 | unresolved | -0.003 [-0.005, -0.002] | 5e-15 | equivalent |
| E07_aux_minutiae_cldice | E08_aux_cv_edge_mix | +0.0 [-2.5, +2.5] | 1.000 | equivalent | -0.002 [-0.003, -0.000] | 2e-15 | equivalent |
| E07_aux_minutiae_cldice | E09_ccdf | -0.5 [-3.0, +2.0] | 1.000 | equivalent | -0.013 [-0.015, -0.010] | 1e-34 | B better |
| E07_aux_minutiae_cldice | E10_ctmsr | +5.5 [+2.8, +8.3] | 0.007 | A better | +0.130 [+0.121, +0.139] | 5e-65 | A better |
| E07_aux_minutiae_cldice | E11_ctmsr_dtm | +5.5 [+2.8, +8.3] | 0.012 | A better | +0.133 [+0.124, +0.142] | 5e-65 | A better |
| E07_aux_minutiae_cldice | E21_ccdf_aux_minutiae | +1.0 [-1.3, +3.3] | 1.000 | unresolved | -0.001 [-0.003, -0.000] | 4e-10 | equivalent |
| E07_aux_minutiae_cldice | E22_ccdf_aux_minutiae_cldice | +0.3 [-1.8, +2.3] | 1.000 | equivalent | -0.013 [-0.015, -0.011] | 3e-55 | B better |
| E07_aux_minutiae_cldice | E23_ccdf_aux_cv_edge_mix | -1.3 [-3.5, +1.0] | 1.000 | unresolved | -0.005 [-0.006, -0.003] | 4e-28 | equivalent |
| E08_aux_cv_edge_mix | E09_ccdf | -0.5 [-2.8, +1.8] | 1.000 | equivalent | -0.011 [-0.013, -0.009] | 1e-16 | B better |
| E08_aux_cv_edge_mix | E10_ctmsr | +5.5 [+2.8, +8.5] | 0.019 | A better | +0.132 [+0.123, +0.141] | 5e-65 | A better |
| E08_aux_cv_edge_mix | E11_ctmsr_dtm | +5.5 [+2.8, +8.5] | 0.019 | A better | +0.134 [+0.125, +0.144] | 5e-65 | A better |
| E08_aux_cv_edge_mix | E21_ccdf_aux_minutiae | +1.0 [-1.3, +3.3] | 1.000 | unresolved | +0.000 [-0.001, +0.002] | 4e-09 | equivalent |
| E08_aux_cv_edge_mix | E22_ccdf_aux_minutiae_cldice | +0.3 [-2.0, +2.5] | 1.000 | equivalent | -0.011 [-0.013, -0.009] | 5e-21 | B better |
| E08_aux_cv_edge_mix | E23_ccdf_aux_cv_edge_mix | -1.3 [-3.3, +0.8] | 1.000 | unresolved | -0.003 [-0.004, -0.002] | 9e-08 | equivalent |
| E09_ccdf | E10_ctmsr | +6.0 [+3.3, +9.0] | 0.004 | A better | +0.143 [+0.133, +0.153] | 5e-65 | A better |
| E09_ccdf | E11_ctmsr_dtm | +6.0 [+3.0, +9.0] | 0.007 | A better | +0.145 [+0.135, +0.156] | 5e-65 | A better |
| E09_ccdf | E21_ccdf_aux_minutiae | +1.5 [-0.5, +3.8] | 1.000 | unresolved | +0.011 [+0.009, +0.013] | 4e-19 | A better |
| E09_ccdf | E22_ccdf_aux_minutiae_cldice | +0.8 [-1.5, +3.0] | 1.000 | equivalent | -0.000 [-0.001, +0.001] | 5e-06 | equivalent |
| E09_ccdf | E23_ccdf_aux_cv_edge_mix | -0.8 [-3.3, +1.5] | 1.000 | equivalent | +0.008 [+0.006, +0.010] | 1e-10 | equivalent |
| E10_ctmsr | E11_ctmsr_dtm | +0.0 [-1.8, +1.8] | 1.000 | equivalent | +0.002 [+0.002, +0.003] | 1e-34 | equivalent |
| E10_ctmsr | E21_ccdf_aux_minutiae | -4.5 [-7.5, -1.5] | 0.205 | unresolved | -0.132 [-0.141, -0.123] | 5e-65 | B better |
| E10_ctmsr | E22_ccdf_aux_minutiae_cldice | -5.3 [-8.3, -2.3] | 0.043 | B better | -0.143 [-0.154, -0.133] | 5e-65 | B better |
| E10_ctmsr | E23_ccdf_aux_cv_edge_mix | -6.8 [-9.8, -4.0] | 2e-04 | B better | -0.135 [-0.144, -0.126] | 5e-65 | B better |
| E11_ctmsr_dtm | E21_ccdf_aux_minutiae | -4.5 [-7.8, -1.5] | 0.404 | unresolved | -0.134 [-0.143, -0.125] | 5e-65 | B better |
| E11_ctmsr_dtm | E22_ccdf_aux_minutiae_cldice | -5.3 [-8.3, -2.3] | 0.060 | unresolved | -0.145 [-0.156, -0.135] | 5e-65 | B better |
| E11_ctmsr_dtm | E23_ccdf_aux_cv_edge_mix | -6.8 [-9.8, -4.0] | 5e-04 | B better | -0.137 [-0.147, -0.128] | 5e-65 | B better |
| E21_ccdf_aux_minutiae | E22_ccdf_aux_minutiae_cldice | -0.8 [-3.0, +1.3] | 1.000 | equivalent | -0.011 [-0.013, -0.010] | 1e-25 | B better |
| E21_ccdf_aux_minutiae | E23_ccdf_aux_cv_edge_mix | -2.3 [-4.8, +0.0] | 1.000 | unresolved | -0.003 [-0.004, -0.002] | 8e-18 | equivalent |
| E22_ccdf_aux_minutiae_cldice | E23_ccdf_aux_cv_edge_mix | -1.5 [-3.8, +0.5] | 1.000 | unresolved | +0.008 [+0.007, +0.010] | 8e-17 | equivalent |

</details>

Source: `evaluation/results/E02_resshift/test_20261023_134052`, `evaluation/results/E03_fingergan/test_20261023_134615`, `evaluation/results/E04_msu_stft/test_20261023_134805`, `evaluation/results/E05_msu_aec/test_20261023_135018`, `evaluation/results/E06_aux_minutiae/test_20261023_135146`, `evaluation/results/E07_aux_minutiae_cldice/test_20261023_135348`, `evaluation/results/E08_aux_cv_edge_mix/test_20261023_135609`, `evaluation/results/E09_ccdf/test_20261023_135852`, `evaluation/results/E10_ctmsr/test_20261023_140106`, `evaluation/results/E11_ctmsr_dtm/test_20261023_140337`, `evaluation/results/E21_ccdf_aux_minutiae/test_20261024_143452`, `evaluation/results/E22_ccdf_aux_minutiae_cldice/test_20261024_144345`, `evaluation/results/E23_ccdf_aux_cv_edge_mix/test_20261024_145025`.
