"""Guard the zero-shot claim.

A zero-shot result is only meaningful if every held-out test/val disease had
*no* treatment edge during training. This module verifies that and records the
verdict at ``results/metrics/leakage_check_seed<s>.json``.

Run standalone:
    python -m repurpose.graph.leakage_audit 42
Exit status 0 = clean, 1 = leakage found.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PKG_ROOT / "data" / "splits"
METRICS_DIR = PKG_ROOT / "results" / "metrics"
TREATMENT_RELATIONS = frozenset({"indication", "contraindication"})


def audit_split(seed: int = 42) -> dict:
    """Check the zero-shot split for the given seed and persist the result."""
    folder = SPLITS_DIR / "zeroshot" / f"seed_{seed}"
    paths = {name: folder / f"{name}.csv" for name in ("train", "val", "test")}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Build splits first (scripts/build_splits.py)."
            )

    train = pd.read_csv(paths["train"])
    val = pd.read_csv(paths["val"])
    test = pd.read_csv(paths["test"])

    trained_diseases = set(
        train[train["relation"].isin(TREATMENT_RELATIONS)]["y_id"].unique()
    )
    test_diseases = set(test["y_id"].unique())
    val_diseases = set(val["y_id"].unique())
    held_out = test_diseases | val_diseases

    leaks = sorted(str(d) for d in (held_out & trained_diseases))

    verdict = {
        "status": "PASS" if not leaks else "FAIL",
        "seed": seed,
        "leaking_diseases": leaks,
        "n_train_diseases": len(trained_diseases),
        "n_test_diseases": len(test_diseases),
        "n_val_diseases": len(val_diseases),
        "n_leaking": len(leaks),
        "checked_at": datetime.utcnow().isoformat(),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"leakage_check_seed{seed}.json", "w") as fh:
        json.dump(verdict, fh, indent=2)
    return verdict


def _cli() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    verdict = audit_split(seed)
    print(json.dumps(verdict, indent=2))
    if verdict["status"] == "FAIL":
        print(f"\n[FAIL] leakage in seed={seed}; fix the split before trusting zero-shot.")
        sys.exit(1)
    print(f"\n[PASS] seed={seed} clean.")


if __name__ == "__main__":
    _cli()
