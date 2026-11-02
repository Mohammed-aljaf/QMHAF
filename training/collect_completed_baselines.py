#!/usr/bin/env python3
"""Collect successfully completed OOF baselines after a partial model sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path("training/results/ahi_a0h3_local_v1")
    rows = []
    for path in sorted(root.glob("*/*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = dict(payload["overall"])
        row["scenario"] = path.parent.parent.name
        row["model"] = path.parent.name
        row.pop("confusion_matrix", None)
        rows.append(row)
    if not rows:
        raise RuntimeError("no completed metrics found")
    frame = pd.DataFrame(rows).sort_values(["scenario", "accuracy"], ascending=[True, False])
    frame.to_csv(root / "completed_baseline_summary.csv", index=False)
    frame.to_parquet(root / "completed_baseline_summary.parquet", index=False)
    print(frame[["scenario", "model", "accuracy", "macro_f1", "qc_pass_accuracy"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
