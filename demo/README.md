# Live demo (test interface)

A Gradio page: pick a test print (or upload a 500 ppi print), damage it (the stored test damage, stacked
contaminants, or a smudge you draw), give the model the true mask or the one the segmenter predicts, and compare
up to two models (default E09 vs the E04 baseline; E22, E02, E20 and E12 are offered when their files exist).
It shows the clean, damaged and repaired prints with |repaired − clean| heat maps, and scores each version the
way `evaluation/evaluate.py` does: BOZORTH3 against the clean slap print (MATCH when the score exceeds the
evaluator's τ), ridge fidelity and PSNR in the damage mask, repair time. Needs the test set, the checkpoints
and NBIS (see the main README).

```bash
python demo/gradio_app.py                      # http://127.0.0.1:7860 (CPU is fine for E09 / E22 / E04)
mkdir -p logs && sbatch demo/launch_gpu.sh     # on a SLURM cluster: GPU node, prints the ssh tunnel command
```

Start-up takes 2–4 min (imports, then a warm-up repair with every model; `--no-warmup` skips it). Files:
`gradio_app.py` (layout), `handlers.py` (one click), `damage.py`, `models.py`, `scoring.py`, `views.py`.
