# Preprocessing: raw NIST SD 302 -> everything the experiments and the evaluator read

Run from the repo root, in order. Raw data: NIST SD 302b (rolled U/V, slap-segmented R/S) and 302d (plain
K/L/M/P), 500 ppi only, from https://www.nist.gov/itl/iad/image-group/nist-special-database-302, placed in
`../data/NIST/` (`sd302b/`, `sd302d/`, `participants.csv`) or wherever `FPR_RAW_DATA` points.
Output goes to `preprocessing/data/` (or `FPR_DATA_DIR`), never into git; counts are written to [DATA.md](DATA.md).

```bash
python preprocessing/00_link_raw_data.py        # raw_data/ symlinks to the raw data (never written to)
python preprocessing/01_index_dataset.py        # data/index.csv: every 500 ppi image, NIST counts asserted
python preprocessing/02_make_split.py           # data/split.csv: 140 / 20 / 40 people (by participant, seed 643)
python preprocessing/03_export_training_pool.py # data/train/clean/: train people's prints, cropped to the finger
python preprocessing/04_preview_degradations.py # data/figures/degradation_preview.png: the 6 contaminants
python preprocessing/05_make_eval_sets.py       # data/{val,test}/probes/ + meta.csv: damaged L probes (fixed seed)
python preprocessing/06_make_pairs.py           # data/{val,test}/pairs.csv: genuine / 50 impostor gallery pairs
python preprocessing/07_export_eval_layout.py   # data/{val,test}/layout/: the probes in evaluate.py's naming
python preprocessing/08_make_train_set.py       # data/train/damaged/ + meta.csv: fixed damaged train set
python preprocessing/09_make_minutiae_maps.py   # data/train/minutiae_w/: MINDTCT minutiae maps (E06, E07)
python preprocessing/10_normalize.py            # MSU-normalised copies (models trained on normalised prints)
```

Data versions (`FPR_DATA_VERSION`): **v2** (default) damages each val/test print once and each train crop once
per contaminant (E02-E11, E21-E23 training); **v3** uses the same damage, 3 damaged copies each (3 contaminants
per val/test print: 1,194 test samples), under `data/v3/`. The headline results are on v3 test. For v3, run
05, 06, 07, 08 and 10 again with `FPR_DATA_VERSION=v3` (08 + 10 on a CPU node: `scripts/slurm/make_train_v3.sh`).
