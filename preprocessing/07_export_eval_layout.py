"""Step 07: present the eval sets in the folder layout evaluation/evaluate.py reads.

Usage: python preprocessing/07_export_eval_layout.py
Copies each probe's raw slap print to preprocessing/data/{split}/gallery/, then writes
preprocessing/data/{split}/layout/{original,degraded,mask,gallery}/ as symlinks into the
split folder: original = clean L print, gallery = clean slap print.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import pandas as pd

import config
from fpr.data import data_md, eval_sets, nist


def link(link_path: Path, target: Path) -> None:
    """Create (or refresh) one symlink link_path -> target."""
    assert target.exists(), f"missing {target}"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target.resolve())


def export_split(split_name: str) -> dict:
    """Link every probe of one split; return the file count per part."""
    meta = eval_sets.load_meta(split_name)
    for row in meta.itertuples():
        gallery = eval_sets.gallery_copy_path(split_name, row.gallery_path)
        gallery.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nist.raw_path(row.gallery_path), gallery)   # byte-identical copy of the raw PNG
        targets = {"original": eval_sets.triplet_path(split_name, row.sample_id, "clean"),
                   "degraded": eval_sets.triplet_path(split_name, row.sample_id, "degraded"),
                   "mask": eval_sets.triplet_path(split_name, row.sample_id, "mask"),
                   "gallery": gallery}
        for part, target in targets.items():
            name = eval_sets.layout_name(row.probe_id, row.contaminant, part)
            link(eval_sets.layout_dir(split_name, part) / name, target)
    counts = {part: len(list(eval_sets.layout_dir(split_name, part).glob("*.png")))
              for part in eval_sets.LAYOUT_PARTS}
    # One original and gallery print per probe, one degraded print and mask per sample (v3: 3 per probe).
    expected = {"original": meta["probe_id"].nunique(), "gallery": meta["probe_id"].nunique(),
                "degraded": len(meta), "mask": len(meta)}
    assert counts == expected, f"{split_name}: {counts} vs {expected}"
    return {"split": split_name, **counts}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    table = pd.DataFrame([export_split(s) for s in config.EVAL_SPLITS])
    data_md.write_section(data_md.versioned("eval_layout"),
                          f"## Eval sets in evaluate.py's layout, data {config.DATA_VERSION} (script 07)\n\n"
                          + "Symlinks under `preprocessing/data/{split}/layout/`: `original/<id>.png` "
                          + "(clean L print), `degraded/<id>_<type>.png`, `mask/<id>_<type>_mask.png`, "
                          + "`gallery/<id>.png` (clean slap print, copied to `{split}/gallery/`). "
                          + "Ids are probe ids; a probe damaged several times (v3) has one degraded print "
                          + "and mask per contaminant. Every link stays inside the split folder.\n\n"
                          + data_md.table_to_markdown(table))
    print(table.to_string(index=False))
    print("wrote " + ", ".join(str(eval_sets.split_dir(s) / config.LAYOUT_SUBDIR) for s in config.EVAL_SPLITS))


if __name__ == "__main__":
    main()
