"""Download PrimeKG from Harvard Dataverse and write data/primekg_stats.json
(computed from the files themselves — never copied from a paper).

    python scripts/fetch_primekg.py
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
STATS_PATH = ROOT / "data" / "primekg_stats.json"

DATAVERSE = {
    "kg.csv": "https://dataverse.harvard.edu/api/access/datafile/6180620",
    "nodes.tab": "https://dataverse.harvard.edu/api/access/datafile/6180617",
    "disease_features.tab": "https://dataverse.harvard.edu/api/access/datafile/6180618",
    "drug_features.tab": "https://dataverse.harvard.edu/api/access/datafile/6180619",
}


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[get] {dest.name}")
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done, t0 = 0, time.time()
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total:
                    rate = done / (time.time() - t0 + 1e-6) / 1e6
                    print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB  {rate:.1f} MB/s", end="")
        tmp.rename(dest)
        print(f"\n  saved {dest.stat().st_size / 1e6:.1f} MB")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {url}") from exc


def compute_stats() -> dict:
    kg = pd.read_csv(RAW_DIR / "kg.csv", low_memory=False)
    stats = {
        "source": "computed from downloaded files",
        "kg_csv_rows": len(kg),
        "kg_csv_columns": list(kg.columns),
        "total_edges": int(len(kg)),
    }
    if "relation" in kg:
        rels = kg["relation"].value_counts().to_dict()
        stats["edge_counts_by_relation"] = {k: int(v) for k, v in rels.items()}
        stats["n_relation_types"] = len(rels)
    node_ids = {}
    for tcol, icol in (("x_type", "x_id"), ("y_type", "y_id")):
        for ntype, chunk in kg.groupby(tcol):
            node_ids.setdefault(ntype, set()).update(chunk[icol].unique())
    counts = {k: len(v) for k, v in node_ids.items()}
    stats["node_counts_by_type"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    stats["total_nodes"] = int(sum(counts.values()))
    return stats


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DATAVERSE.items():
        fetch(url, RAW_DIR / name)
    print("\n[stats] computing from downloaded files ...")
    stats = compute_stats()
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"[done] -> {STATS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
