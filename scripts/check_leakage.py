"""Re-run the zero-shot leakage audit for every seed. Exit 1 on any failure.

    python scripts/check_leakage.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from repurpose.graph.leakage_audit import audit_split


def main() -> int:
    failed = False
    for seed in (42, 0, 1):
        verdict = audit_split(seed)
        print(f"seed={seed}: {verdict['status']} | leaks={verdict['n_leaking']}")
        failed = failed or verdict["status"] == "FAIL"
    if failed:
        print("\n[FAIL] fix the split before trusting zero-shot results.")
        return 1
    print("\n[PASS] no leakage across seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
