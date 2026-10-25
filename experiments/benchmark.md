# Repair cost

Written by `experiments/benchmark.py` (see its docstring for the protocol): 50 test prints per method, one at a time, after 3 warm-up calls; peak GPU = PyTorch's peak allocated memory during the timed calls; peak RAM = the process's peak resident memory, model and data included.

| experiment | device | s / print (median) | s / print (p90) | peak GPU MB | peak RAM MB | parameters |
|---|---|---|---|---|---|---|
| E02_resshift | NVIDIA A40 | 0.855 | 1.279 | 678 | 1043 | 8,103,937 |
| E03_fingergan | NVIDIA A40 | 1.162 | 1.570 | 3598 | 1025 | 33,867,137 |
| E04_msu_stft | AMD EPYC 7313 16-Core Processor, 8 threads | 0.220 | 0.403 | – | 572 | 0 |
| E05_msu_aec | NVIDIA A40 | 0.260 | 0.405 | 37 | 972 | 647,441 |
| E06_aux_minutiae | NVIDIA A40 | 0.797 | 1.266 | 678 | 1009 | 8,103,937 |
| E07_aux_minutiae_cldice | NVIDIA A40 | 0.830 | 1.272 | 678 | 970 | 8,103,937 |
| E08_aux_cv_edge_mix | NVIDIA A40 | 0.825 | 1.288 | 678 | 1036 | 8,103,937 |
| E09_ccdf | NVIDIA A40 | 0.478 | 0.735 | 681 | 1011 | 8,103,937 |
| E10_ctmsr | NVIDIA A40 | 0.401 | 0.637 | 675 | 1042 | 8,103,937 |
| E11_ctmsr_dtm | NVIDIA A40 | 0.399 | 0.628 | 675 | 1019 | 8,103,937 |
| E12_norm_ccdf_cldice_cv | NVIDIA A40 | 0.585 | 0.893 | 682 | 1180 | 8,103,937 |
| E13_norm_no_cldice | NVIDIA A40 | 0.926 | 1.440 | 680 | 1194 | 8,103,937 |
| E14_norm_no_edge | NVIDIA A40 | 0.940 | 1.441 | 680 | 1197 | 8,103,937 |
| E15_norm_no_mix | NVIDIA A40 | 0.699 | 0.982 | 683 | 1197 | 8,103,937 |
| E16_norm_no_aux | NVIDIA A40 | 0.585 | 0.890 | 683 | 1177 | 8,103,937 |
| E17_fingergan_norm | NVIDIA A40 | 1.689 | 2.065 | 3598 | 1025 | 33,867,137 |
| E18_msu_stft_norm | AMD EPYC 7313 16-Core Processor, 8 threads | 0.366 | 0.398 | – | 573 | 0 |
| E19_msu_aec_norm | NVIDIA A40 | 0.230 | 0.236 | 37 | 972 | 647,441 |
| E20_fingergan_finetune | NVIDIA A40 | 1.299 | 1.665 | 3598 | 1013 | 33,867,137 |
| E21_ccdf_aux_minutiae | NVIDIA A40 | 0.788 | 1.240 | 678 | 1026 | 8,103,937 |
| E22_ccdf_aux_minutiae_cldice | NVIDIA A40 | 0.433 | 0.688 | 681 | 1031 | 8,103,937 |
| E23_ccdf_aux_cv_edge_mix | NVIDIA A40 | 0.700 | 1.102 | 682 | 1043 | 8,103,937 |
| E24_norm_no_cldice_e12sampler | NVIDIA A40 | 0.596 | 0.909 | 682 | 1195 | 8,103,937 |
| E25_norm_no_edge_e12sampler | NVIDIA A40 | 0.624 | 0.898 | 682 | 1191 | 8,103,937 |
| E26_raw_ccdf_cldice_cv | NVIDIA A40 | 0.532 | 0.791 | 682 | 1052 | 8,103,937 |
