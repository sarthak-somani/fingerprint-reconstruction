"""fpr = fingerprint reconstruction: the Halluci-Nation library, one sub-package per pipeline stage.

data (NIST index, split, synthetic damage, datasets) -> models (ResShift U-Net, schedule, samplers,
reconstructors) -> training (train loops, losses) -> validation (checkpoint and sampler selection on val)
-> metrics (image metrics, NBIS matching). experiments.py finds experiment folders and builds their methods.
"""
