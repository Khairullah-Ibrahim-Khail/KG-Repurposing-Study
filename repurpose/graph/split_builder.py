"""Construct the two evaluation regimes from PrimeKG's therapeutic edges.

standard
    A plain random hold-out of (drug, relation, disease) edges. A disease may
    have edges in train *and* test — this is the easy regime.

zeroshot
    A set of diseases is removed from training entirely; every therapeutic edge
    touching them is pushed to val/test. No held-out disease keeps any
    treatment edge in train. This is the hard, rare-disease regime.

Output layout (one folder per seed):
    data/splits/standard/seed_<s>/{train,val,test}.csv  + meta.json
    data/splits/zeroshot/seed_<s>/{train,val,test}.csv  + meta.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PKG_ROOT / "data" / "splits"
TREATMENT_RELATIONS = frozenset({"indication", "contraindication"})

TEST_FRACTION = 0.10        # standard: share of edges held out for test
VAL_FRACTION = 0.10         # standard: share held out for validation
HELDOUT_DISEASE_FRACTION = 0.20   # zeroshot: share of diseases removed from train


def make_all_splits(edges: pd.DataFrame, seeds=(42, 0, 1)) -> None:
    """Build both regimes for every seed. seed=42 is the canonical comparison."""
    treat = edges[edges["relation"].isin(TREATMENT_RELATIONS)].copy()
    print(f"[split] therapeutic edges: {len(treat):,}")
    print(f"[split] diseases: {treat['y_id'].nunique():,} | drugs: {treat['x_id'].nunique():,}")

    for seed in seeds:
        print(f"\n[split] seed={seed} — standard")
        _standard_partition(treat, seed)
        print(f"[split] seed={seed} — zeroshot")
        _zeroshot_partition(treat, seed)

    print("\n[split] complete.")


def _write_split(folder: Path, train, val, test, meta) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / "train.csv", index=False)
    val.to_csv(folder / "val.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    with open(folder / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def _standard_partition(treat: pd.DataFrame, seed: int) -> None:
    rng = np.random.default_rng(seed)
    order = np.arange(len(treat))
    rng.shuffle(order)

    n_test = int(len(order) * TEST_FRACTION)
    n_val = int(len(order) * VAL_FRACTION)
    test_pos = order[:n_test]
    val_pos = order[n_test:n_test + n_val]
    train_pos = order[n_test + n_val:]

    meta = {
        "split_type": "standard",
        "seed": seed,
        "n_train": int(len(train_pos)),
        "n_val": int(len(val_pos)),
        "n_test": int(len(test_pos)),
        "note": "Test diseases may also appear in training. Not zero-shot.",
    }
    _write_split(
        SPLITS_DIR / "standard" / f"seed_{seed}",
        treat.iloc[train_pos], treat.iloc[val_pos], treat.iloc[test_pos], meta,
    )
    print(f"  train={len(train_pos):,} val={len(val_pos):,} test={len(test_pos):,}")


def _zeroshot_partition(treat: pd.DataFrame, seed: int) -> None:
    rng = np.random.default_rng(seed)
    diseases = treat["y_id"].unique()
    diseases = diseases.to_numpy() if hasattr(diseases, "to_numpy") else diseases
    rng.shuffle(diseases)

    n_heldout = int(len(diseases) * HELDOUT_DISEASE_FRACTION)
    heldout = set(diseases[:n_heldout])
    seen = set(diseases[n_heldout:])

    heldout_list = list(heldout)
    n_val_disease = max(1, int(n_heldout * 0.20))
    val_diseases = set(heldout_list[:n_val_disease])
    test_diseases = set(heldout_list[n_val_disease:])

    in_train = treat["y_id"].isin(seen)
    in_val = treat["y_id"].isin(val_diseases)
    in_test = treat["y_id"].isin(test_diseases)

    # Invariant: no held-out disease may carry a training edge.
    assert not treat[in_train]["y_id"].isin(heldout).any(), \
        "zeroshot invariant violated: held-out disease found in train"

    meta = {
        "split_type": "zeroshot",
        "seed": seed,
        "n_train_edges": int(in_train.sum()),
        "n_val_edges": int(in_val.sum()),
        "n_test_edges": int(in_test.sum()),
        "n_seen_diseases": len(seen),
        "n_val_diseases": len(val_diseases),
        "n_test_diseases": len(test_diseases),
        "n_total_diseases": len(diseases),
        "held_out_frac": HELDOUT_DISEASE_FRACTION,
        "note": (
            "Held-out diseases have zero treatment edges in train. Their "
            "non-therapeutic edges remain in the full KG for structural context."
        ),
    }
    _write_split(
        SPLITS_DIR / "zeroshot" / f"seed_{seed}",
        treat[in_train], treat[in_val], treat[in_test], meta,
    )
    print(f"  train={in_train.sum():,} val={in_val.sum():,} test={in_test.sum():,}")
    print(f"  seen={len(seen):,} val_dis={len(val_diseases):,} test_dis={len(test_diseases):,}")
