"""Tests for fpr/data/datasets.py: shapes, value ranges, determinism, worker safety."""
import pytest
import torch
from torch.utils.data import DataLoader

import config
from fpr.data import datasets, split
from fpr.data.degradations import CONTAMINANTS

needs_pool = pytest.mark.skipif(not (config.TRAIN_CLEAN_DIR / "pool.csv").exists(),
                                reason="run script 03 first")
needs_eval = pytest.mark.skipif(not (config.DATA_DIR / "val" / config.META_CSV).exists(),
                                reason="run script 05 first")


@needs_pool
def test_train_item_shapes_and_ranges():
    item = datasets.TrainPatchDataset(epoch_size=4)[0]
    for key in ("clean", "degraded", "mask"):
        assert item[key].shape == (1, 128, 128) and item[key].dtype == torch.float32
        assert 0.0 <= item[key].min() and item[key].max() <= 1.0
    assert item["orientation"].shape == (2, 128, 128)
    assert item["mask"].sum() > 0
    outside = item["mask"] == 0
    assert torch.equal(item["clean"][outside], item["degraded"][outside])


@needs_pool
def test_train_items_reproducible_and_epoch_dependent():
    a, b = datasets.TrainPatchDataset(epoch_size=4), datasets.TrainPatchDataset(epoch_size=4)
    assert torch.equal(a[1]["degraded"], b[1]["degraded"])
    b.set_epoch(1)
    assert not torch.equal(a[1]["degraded"], b[1]["degraded"])


@needs_pool
def test_pool_only_has_train_people():
    pool = datasets.TrainPatchDataset(epoch_size=1).pool
    assert set(pool["participant_id"]) <= set(split.participants_in("train"))


@needs_pool
def test_dataloader_with_workers():
    loader = DataLoader(datasets.TrainPatchDataset(epoch_size=8), batch_size=4, num_workers=2)
    batch = next(iter(loader))
    assert batch["degraded"].shape == (4, 1, 128, 128)
    assert not torch.equal(batch["degraded"][0], batch["degraded"][1])


@needs_eval
def test_eval_set_item():
    item = datasets.EvalSetDataset("val")[0]
    assert item["clean"].shape == item["degraded"].shape == item["mask"].shape
    assert item["contaminant"] in CONTAMINANTS


needs_train_set = pytest.mark.skipif(not config.TRAIN_META_CSV.exists(), reason="run script 08 first")


@needs_train_set
def test_fixed_train_set_items():
    data = datasets.FixedTrainDataset()
    assert len(data) == len(CONTAMINANTS) * len(datasets.TrainPatchDataset(epoch_size=1).pool)
    with_damage = 0
    for index in range(0, len(data), len(data) // 20):
        item = data[index]
        assert item["clean"].shape == item["degraded"].shape == item["mask"].shape == (1, 128, 128)
        outside = item["mask"] == 0
        assert torch.equal(item["clean"][outside], item["degraded"][outside])
        with_damage += int(item["mask"].sum() > 0)
    assert with_damage >= 15, "most patches should contain damage"
    assert set(data.meta["participant_id"]) <= set(split.participants_in("train"))


@needs_train_set
def test_fixed_train_set_is_reproducible(tmp_path, monkeypatch):
    """Damaging one pool print twice with script 08 gives identical files.

    Writes to a temp folder, never into train/damaged/. The stored set was made with the
    first version of the degradations, so it is not compared here.
    """
    import importlib.util
    import numpy as np
    from fpr.data import image_io
    path = config.PREPROCESSING_DIR / "scripts" / "08_make_train_set.py"
    spec = importlib.util.spec_from_file_location("make_train_set", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    pool = datasets.TrainPatchDataset(epoch_size=1).pool
    outputs = []
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
        monkeypatch.setattr(config, "TRAIN_DAMAGED_DIR", tmp_path / run)
        script.damage_one_print((7, pool.iloc[7].to_dict()))
        outputs.append(sorted((tmp_path / run).iterdir()))
    assert [p.name for p in outputs[0]] == [p.name for p in outputs[1]]
    for a, b in zip(*outputs):
        assert np.array_equal(image_io.load_gray(a), image_io.load_gray(b))


needs_minutiae_maps = pytest.mark.skipif(not (config.TRAIN_MINUTIAE_DIR / "meta.csv").exists(),
                                         reason="run script 09 first")


@needs_minutiae_maps
def test_fixed_train_set_minutiae_maps_keep_the_patches(monkeypatch):
    """Loading the minutiae maps changes nothing else, and each map is cut with the item's window."""
    import numpy as np
    from fpr.data import patches
    corners, original = [], patches.patch_corner

    def recording_corner(*args, **kwargs):
        corners.append(original(*args, **kwargs))
        return corners[-1]

    monkeypatch.setattr(patches, "patch_corner", recording_corner)
    plain = datasets.FixedTrainDataset()
    with_maps = datasets.FixedTrainDataset(minutiae_dir=config.TRAIN_MINUTIAE_DIR)
    for index in (0, 1234, 20_000, len(plain) - 1):
        a, b = plain[index], with_maps[index]
        for key in ("clean", "degraded", "mask", "orientation"):
            assert torch.equal(a[key], b[key]), key
        assert "minutiae_w" not in a and b["minutiae_w"].shape == (1, 128, 128)
        assert b["minutiae_w"].dtype == torch.float32 and 0 <= b["minutiae_w"].min() <= b["minutiae_w"].max() <= 1
        top, left = corners[-1]
        assert corners[-2] == corners[-1]                              # same window both times
        full = with_maps.load_minutiae(with_maps.meta.iloc[index]).astype(np.float32)
        assert np.array_equal(b["minutiae_w"][0].numpy(), full[top:top + 128, left:left + 128])
