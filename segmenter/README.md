# Damage-mask segmenter

A scanner has no ground-truth damage mask, so `segformer.py` (SegFormer-B0, a plain-PyTorch port loading the
`nvidia/mit-b0` ImageNet encoder) is fine-tuned on the v2 train set to predict it (`train.py`, Dice + BCE;
`params.json`); `predict.py` / `predict_masks.py` apply it, and the demo offers it as the mask source.
`clipseg_zero_shot.py` + `evaluate_zero_shot.py` are the zero-shot check that showed fine-tuning is needed.
v2 test (398 probes): IoU 0.884, Dice 0.939 (`results/v2_test_finetuned.md`; val in `results/v2_val_finetuned.md`). Train: `python segmenter/train.py`
(or `scripts/slurm/train_segmenter.sh`); the checkpoint goes to `segmenter/checkpoints/` (git-ignored).
