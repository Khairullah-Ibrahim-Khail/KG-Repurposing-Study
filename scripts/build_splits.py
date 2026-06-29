"""Build standard + zero-shot splits for all seeds, then audit for leakage.

    python scripts/build_splits.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from repurpose.graph.split_builder import make_all_splits
from repurpose.graph.leakage_audit import audit_split

SEEDS = [42, 0, 1]


def main() -> int:
    print("Loading kg.csv ...")
    edges = pd.read_csv(ROOT / "data" / "raw" / "kg.csv", low_memory=False)
    print(f"  {len(edges):,} edges")

    make_all_splits(edges, seeds=SEEDS)

    print("\nAuditing zero-shot splits ...")
    clean = True
    for seed in SEEDS:
        verdict = audit_split(seed)
        print(f"  seed={seed}: {verdict['status']} "
              f"(test_diseases={verdict['n_test_diseases']}, leaks={verdict['n_leaking']})")
        clean = clean and verdict["status"] == "PASS"

    if not clean:
        print("\n[FAIL] leakage present — do not use zero-shot results.")
        return 1
    print("\n[PASS] all splits clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
