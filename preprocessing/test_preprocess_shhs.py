#!/usr/bin/env python3
"""Small integrity checks for an existing SHHS preprocessing cache."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


def main(root: Path) -> int:
    manifest_path = root / "manifest.parquet"
    schema_path = root / "schema.json"
    if not manifest_path.exists() or not schema_path.exists():
        raise SystemExit(f"missing manifest/schema under {root}")
    manifest = pd.read_parquet(manifest_path)
    schema = json.loads(schema_path.read_text())
    assert manifest["subject_id"].is_unique
    assert set(manifest["status"]) <= {"ok", "failed"}
    good = manifest[manifest["status"] == "ok"]
    for sid in good["subject_id"].head(25):
        path = root / "subjects" / f"{sid}.npz"
        with np.load(path, allow_pickle=False) as data:
            seconds = data["spo2_1hz"].size
            assert data["resp_q10"].shape == (3, seconds * 10)
            assert data["resp_valid_1hz"].shape == (3, seconds)
            assert data["hr_1hz"].size == seconds
            assert data["position_1hz"].size == seconds
            assert data["sleep_stage_30s"].size == data["epoch_features_30s"].shape[0]
            assert data["feature_names"].size == data["epoch_features_30s"].shape[1]
            assert str(data["cache_version"].item()) == schema["cache_version"]
    print(f"validated {len(good)} successful caches; deep-checked {min(25, len(good))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "processed_shhs_v1")))
