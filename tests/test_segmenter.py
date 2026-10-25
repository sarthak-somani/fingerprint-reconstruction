"""Tests for the segmenter: the SegFormer port (against transformers when available) and the mask metrics."""
import numpy as np
import pytest
import torch

from segmenter import metrics
from segmenter.segformer import load_released, Segformer

ADE = "nvidia/segformer-b0-finetuned-ade-512-512"


def test_segformer_b0_shape_and_size():
    model = Segformer(num_labels=1).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 200, 330))
    assert out.shape == (2, 1, 50, 83)                                   # 1/4 resolution, odd sizes ok
    assert 3.5e6 < sum(p.numel() for p in model.parameters()) < 4e6      # SegFormer-B0: 3.7M


def test_port_matches_transformers():
    """Same logits as Hugging Face's SegformerForSemanticSegmentation (needs torch >= 2.6 + transformers)."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "SegformerForSemanticSegmentation"):
        pytest.skip("transformers has no PyTorch backend in this environment")
    reference = transformers.SegformerForSemanticSegmentation.from_pretrained(ADE).eval()
    ours = load_released(ADE, num_labels=150).eval()
    x = torch.randn(1, 3, 200, 330)
    with torch.no_grad():
        assert torch.allclose(ours(x), reference(pixel_values=x).logits, atol=1e-4)


def test_one_fine_tuning_step_is_finite():
    from segmenter.predict import normalise
    from segmenter.train import loss_fn
    model = Segformer(num_labels=1).train()
    mask = (torch.rand(2, 1, 64, 64) > 0.7).float()
    loss = loss_fn(model(normalise(torch.rand(2, 1, 64, 64))), mask)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_metrics_on_known_masks():
    truth = np.zeros((10, 10), bool)
    truth[:5] = True
    pred = np.zeros((10, 10), bool)
    pred[:, :5] = True
    counts = metrics.confusion(pred, truth)
    assert counts == {"tp": 25, "fp": 25, "fn": 25, "tn": 25}
    scores = metrics.scores(counts)
    assert scores["iou"] == pytest.approx(1 / 3) and scores["dice"] == pytest.approx(0.5)
