"""Tests for fpr/metrics/image_metrics.py: perfect when pred == clean, worse when damaged."""
import numpy as np
import pytest

import config
from fpr.metrics import image_metrics


def make_case(size: int = 96):
    yy, xx = np.mgrid[0:size, 0:size]
    clean = (127.5 + 127.5 * np.cos(2 * np.pi * (xx + 0.5 * yy) / 9)).astype(np.uint8)
    region = np.zeros((size, size), bool)
    region[30:70, 30:70] = True
    damaged = clean.copy()
    damaged[region] = 128                                 # flat grey smudge
    return clean, damaged, region


def test_identical_images_are_perfect():
    clean, _, region = make_case()
    assert image_metrics.masked_psnr(clean, clean, region) == config.PSNR_CAP_DB
    assert image_metrics.masked_ssim(clean, clean, region) == pytest.approx(1.0)
    assert image_metrics.orientation_error(clean, clean, region) == pytest.approx(0.0, abs=1e-4)


def test_damage_makes_metrics_worse():
    clean, damaged, region = make_case()
    assert image_metrics.masked_psnr(damaged, clean, region) < 20
    assert image_metrics.masked_ssim(damaged, clean, region) < 0.5


def test_damage_outside_region_is_ignored():
    clean, damaged, region = make_case()
    outside = ~region
    outside[:, :] = False
    outside[0:10, 0:10] = True                            # untouched corner
    assert image_metrics.masked_psnr(damaged, clean, outside) == config.PSNR_CAP_DB


def test_empty_region_gives_nan():
    clean, damaged, _ = make_case()
    empty = np.zeros(clean.shape, bool)
    assert np.isnan(image_metrics.masked_psnr(damaged, clean, empty))


def test_lpips_orders_correctly():
    pytest.importorskip("lpips")
    clean, damaged, region = make_case()
    try:
        same = image_metrics.lpips_box(clean, clean, region)
    except Exception as error:                            # backbone weights need a download once
        pytest.skip(f"LPIPS weights unavailable: {error}")
    assert same == pytest.approx(0.0, abs=1e-4)
    assert image_metrics.lpips_box(damaged, clean, region) > 0.05


# ---------------------------------------------------------------- minutiae comparison (no NBIS needed)
from fpr.metrics import minutiae  # noqa: E402
from fpr.metrics.matching import nbis  # noqa: E402


def test_compare_minutiae_counts_spurious_and_missed():
    region = np.zeros((100, 100), bool)
    region[20:80, 20:80] = True
    clean = np.array([[30, 30, 90, 0], [50, 50, 180, 1], [70, 70, 0, 0]], float)
    recon = np.array([[32, 31, 95, 0],        # matches clean 0 (2 px, 5 deg)
                      [50, 50, 300, 1],       # same place, direction off by 120 deg -> spurious
                      [60, 40, 0, 0]], float)  # nothing nearby -> spurious
    result = minutiae.compare_minutiae(recon, clean, region)
    assert result["n_recon"] == 3 and result["n_clean"] == 3
    assert result["spurious_rate"] == pytest.approx(2 / 3)
    assert result["missed_rate"] == pytest.approx(2 / 3)


def test_compare_identical_sets_is_perfect():
    region = np.ones((100, 100), bool)
    points = np.array([[10, 10, 45, 0], [40, 60, 200, 1]], float)
    result = minutiae.compare_minutiae(points, points, region)
    assert result["spurious_rate"] == 0 and result["missed_rate"] == 0


def test_read_min_file(tmp_path):
    text = ("Minutiae Detection Results :\n"
            "   0 :  120, 300 :  8 :  0.45 : RIG : APPEARING : 45,10; \n"
            "   1 :   15,  22 : 31 :  0.12 : BIF : DISAPPEARING : 3,4; \n")
    (tmp_path / "x.min").write_text(text)
    parsed = minutiae.read_min_file(tmp_path / "x.min")
    assert parsed.shape == (2, 4)
    assert parsed[0].tolist() == [120, 300, 90.0, 0] and parsed[1, 3] == 1


@pytest.mark.skipif(bool(nbis.missing_tools()), reason="NBIS not installed (tools/install_nbis.md)")
def test_nbis_mindtct_and_bozorth3(tmp_path):
    import config
    from fpr.data import eval_sets, image_io
    meta = eval_sets.load_meta("val")
    clean = image_io.load_gray(eval_sets.triplet_path("val", meta.probe_id[0], "clean"))
    found = minutiae.extract_minutiae(clean)
    assert len(found) > 10
    xyt, _ = nbis.run_mindtct(clean, tmp_path / "a")
    same = nbis.bozorth_scores(xyt, [xyt])
    assert same[0] > 40, "a print should match itself strongly"
