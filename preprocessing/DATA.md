# Data counts

Written by the preprocessing scripts (`fpr/data/data_md.py`); re-run a script to refresh its section.

<!-- begin:index -->
## Image index (script 01)

| device | capture | images | participants | height_range | width_range |
|---|---|---|---|---|---|
| K | plain | 930 | 93 | 319-319 | 248-248 |
| L | plain | 1953 | 199 | 180-549 | 51-375 |
| M | plain | 1979 | 198 | 360-360 | 256-256 |
| P | plain | 279 | 28 | 480-480 | 320-320 |
| R | slap | 920 | 92 | 298-682 | 256-454 |
| S | slap | 1080 | 108 | 276-638 | 256-466 |
| U | rolled | 2000 | 200 | 750-750 | 800-800 |
| V | rolled | 2000 | 200 | 750-750 | 800-800 |

Total: 11141 images at 500 ppi.
<!-- end:index -->

<!-- begin:split -->
## Participant split (script 02)

Seed 643. Val/test people need an L plain print and a slap print for at least 9 fingers.

| split | participants | K/plain | L/plain | M/plain | P/plain | R/slap | S/slap | U/rolled | V/rolled | L+slap fingers |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 140 | 600 | 1358 | 1380 | 170 | 650 | 750 | 1400 | 1400 | 1358 |
| val | 20 | 120 | 197 | 199 | 49 | 80 | 120 | 200 | 200 | 197 |
| test | 40 | 210 | 398 | 400 | 60 | 190 | 210 | 400 | 400 | 398 |
<!-- end:split -->

<!-- begin:pool -->
## Training pool (script 03)

Train-split prints cropped to the foreground box (+8 px), background set to white. 6308 images, 755 MB.

| device | capture | images | median_h | median_w |
|---|---|---|---|---|
| K | plain | 600 | 319.0 | 240.0 |
| L | plain | 1358 | 488.0 | 332.0 |
| M | plain | 1380 | 352.0 | 256.0 |
| P | plain | 170 | 436.0 | 284.0 |
| U | rolled | 1400 | 470.0 | 512.0 |
| V | rolled | 1400 | 470.0 | 528.0 |
<!-- end:pool -->

<!-- begin:eval_sets -->
## Fixed val/test sets (script 05)

Full L plain prints damaged once (seed 2026), severity uniform in (0.4, 0.8). damaged_fraction = damaged pixels / foreground pixels.

| split | contaminant | probes | mean_severity | mean_damaged_fraction |
|---|---|---|---|---|
| val | lotion | 32 | 0.6 | 0.34 |
| val | mehendi | 33 | 0.59 | 0.126 |
| val | oil | 33 | 0.6 | 0.538 |
| val | pressure | 33 | 0.62 | 0.244 |
| val | sanitizer | 33 | 0.61 | 0.71 |
| val | scar | 33 | 0.59 | 0.16 |
| test | lotion | 66 | 0.58 | 0.333 |
| test | mehendi | 66 | 0.61 | 0.129 |
| test | oil | 66 | 0.63 | 0.591 |
| test | pressure | 67 | 0.59 | 0.233 |
| test | sanitizer | 66 | 0.61 | 0.764 |
| test | scar | 67 | 0.62 | 0.187 |
<!-- end:eval_sets -->

<!-- begin:pairs -->
## Verification pairs (script 06)

Probe = damaged L plain print; gallery = clean slap print. Impostors include the same person's other fingers.

| split | probes | gallery prints | genuine | impostor |
|---|---|---|---|---|
| val | 197 | 200 | 197 | 39203 |
| test | 398 | 400 | 398 | 158802 |
<!-- end:pairs -->

<!-- begin:eval_layout -->
## Eval sets in evaluate.py's layout (script 07)

Symlinks under `preprocessing/data/{split}/layout/`: `original/<id>.png` (clean L print), `degraded/<id>_<type>.png`, `mask/<id>_<type>_mask.png`, `gallery/<id>.png` (clean slap print, copied to `{split}/gallery/`). Ids are probe ids. Every link stays inside the split folder.

| split | original | degraded | mask | gallery |
|---|---|---|---|---|
| val | 197 | 197 | 197 | 197 |
| test | 398 | 398 | 398 | 398 |
<!-- end:eval_layout -->

<!-- begin:train_set -->
## Fixed damaged train set (script 08)

Each of the 6308 clean train crops damaged once with each contaminant (seed 2027, severity uniform in (0.0, 0.8)). 37848 samples, 4723 MB in `train/damaged/`.

| contaminant | samples | mean_severity | mean_damaged_fraction |
|---|---|---|---|
| lotion | 6308 | 0.4 | 0.259 |
| mehendi | 6308 | 0.4 | 0.11 |
| oil | 6308 | 0.4 | 0.416 |
| pressure | 6308 | 0.4 | 0.194 |
| sanitizer | 6308 | 0.4 | 0.542 |
| scar | 6308 | 0.4 | 0.121 |
<!-- end:train_set -->

<!-- begin:eval_sets_v3 -->
## Fixed val/test sets, data v3 (script 05)

Full L plain prints damaged 3 times (different contaminants) (seed 2026), severity uniform in (0.4, 0.8). damaged_fraction = damaged pixels / foreground pixels.

| split | contaminant | samples | mean_severity | mean_damaged_fraction |
|---|---|---|---|---|
| val | lotion | 93 | 0.58 | 0.332 |
| val | mehendi | 94 | 0.6 | 0.123 |
| val | oil | 97 | 0.61 | 0.513 |
| val | pressure | 112 | 0.6 | 0.226 |
| val | sanitizer | 97 | 0.59 | 0.72 |
| val | scar | 98 | 0.6 | 0.184 |
| test | lotion | 221 | 0.58 | 0.333 |
| test | mehendi | 193 | 0.61 | 0.13 |
| test | oil | 188 | 0.6 | 0.56 |
| test | pressure | 198 | 0.6 | 0.239 |
| test | sanitizer | 202 | 0.6 | 0.725 |
| test | scar | 192 | 0.6 | 0.17 |
<!-- end:eval_sets_v3 -->

<!-- begin:pairs_v3 -->
## Verification pairs, data v3 (script 06)

Probe = damaged L plain print; gallery = clean slap print. Impostors include the same person's other fingers.

| split | probes | gallery prints | genuine | impostor |
|---|---|---|---|---|
| val | 197 | 200 | 197 | 39203 |
| test | 398 | 400 | 398 | 158802 |
<!-- end:pairs_v3 -->

<!-- begin:eval_layout_v3 -->
## Eval sets in evaluate.py's layout, data v3 (script 07)

Symlinks under `preprocessing/data/{split}/layout/`: `original/<id>.png` (clean L print), `degraded/<id>_<type>.png`, `mask/<id>_<type>_mask.png`, `gallery/<id>.png` (clean slap print, copied to `{split}/gallery/`). Ids are probe ids; a probe damaged several times (v3) has one degraded print and mask per contaminant. Every link stays inside the split folder.

| split | original | degraded | mask | gallery |
|---|---|---|---|---|
| val | 197 | 591 | 591 | 197 |
| test | 398 | 1194 | 1194 | 398 |
<!-- end:eval_layout_v3 -->

<!-- begin:train_set_v3 -->
## Fixed damaged train set, data v3 (script 08)

Each of the 6308 clean train crops damaged 3 times with each contaminant (seed 2027, severity uniform in (0.0, 0.8)). 113544 samples, 14171 MB in `v3/train/damaged/`.

| contaminant | samples | mean_severity | mean_damaged_fraction |
|---|---|---|---|
| lotion | 18924 | 0.4 | 0.258 |
| mehendi | 18924 | 0.4 | 0.11 |
| oil | 18924 | 0.4 | 0.41 |
| pressure | 18924 | 0.4 | 0.195 |
| sanitizer | 18924 | 0.4 | 0.543 |
| scar | 18924 | 0.4 | 0.122 |
<!-- end:train_set_v3 -->

<!-- begin:normalized_v3 -->
## MSU-normalised copies, data v3 (script 10)

Texture of MSU's cartoon-texture decomposition (σ 2.5; `fpr/normalization.py`), for models that work on normalised prints: `train/clean_norm/`, `train/damaged_norm/` (degraded only), `{split}/probes_norm/` (clean and degraded). Masks are the stored ones.

| part | images | written now |
|---|---|---|
| val | 1182 | 1182 |
| test | 2388 | 2388 |
| train | 119852 | 119844 |
<!-- end:normalized_v3 -->
