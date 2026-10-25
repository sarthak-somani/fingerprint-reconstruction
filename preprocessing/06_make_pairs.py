"""Step 06: genuine and impostor pairs -> preprocessing/data/{split}/pairs.csv.

Usage: python preprocessing/06_make_pairs.py
Each probe is paired with every slap print of its split: one genuine pair
(same person and finger), the rest impostors. Pairs are per probe (finger), so every damaged
sample of a probe (v3: 3) uses its probe's pairs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import pandas as pd

import config
from fpr.data import data_md, eval_sets, nist
from fpr.metrics.matching import pairs as pair_lists


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    index = nist.load_index()

    rows = []
    for split_name in config.EVAL_SPLITS:
        probes = eval_sets.load_meta(split_name).drop_duplicates("probe_id")   # v3: 3 samples per probe
        gallery = eval_sets.gallery_table(index, split_name)
        pairs = pair_lists.build_pairs(probes, gallery)
        pair_lists.check_pairs(pairs, probes)
        out = eval_sets.pairs_path(split_name)
        pairs.to_csv(out, index=False)
        rows.append({"split": split_name, "probes": len(probes), "gallery prints": len(gallery),
                     "genuine": int(pairs["label"].sum()),
                     "impostor": int((pairs["label"] == 0).sum())})
        print(f"wrote {out} ({len(pairs)} pairs)")

    table = pd.DataFrame(rows)
    data_md.write_section(data_md.versioned("pairs"), f"## Verification pairs, data {config.DATA_VERSION} (script 06)\n\n"
                          + "Probe = damaged L plain print; gallery = clean slap print. "
                          + "Impostors include the same person's other fingers.\n\n"
                          + data_md.table_to_markdown(table))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
