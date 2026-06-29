"""Append-only JSONL run log at ``results/run_log.jsonl``."""

import json
from datetime import datetime
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PKG_ROOT / "results" / "run_log.jsonl"


def append_run(entry: dict) -> None:
    entry = {**entry, "logged_at": datetime.utcnow().isoformat()}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def load_runs() -> list:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as fh:
        return [json.loads(line) for line in fh if line.strip()]
