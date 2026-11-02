#!/usr/bin/env python3
"""Freeze a versioned, checksummed local ahi_a0h3 target and CV folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


TARGET_VERSION = "ahi_a0h3_local_v1"
CLASS_NAMES = np.array(["normal", "mild", "moderate", "severe"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qc_pass(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame["sleep_seconds"] >= 2 * 3600)
        & (frame["record_seconds"] >= 4 * 3600)
        & (frame["spo2_valid_fraction"] >= 0.80)
        & (frame["thor_valid_fraction"] >= 0.70)
        & (frame["abdo_valid_fraction"] >= 0.70)
        & (
            (frame["airflow_valid_fraction"] >= 0.50)
            | ((frame["thor_valid_fraction"] >= 0.90) & (frame["abdo_valid_fraction"] >= 0.90))
        )
    ).to_numpy(bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("processed_shhs_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/targets/ahi_a0h3_local_v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.cache_dir / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path).sort_values("subject_id").reset_index(drop=True)
    if not (manifest["status"] == "ok").all():
        raise ValueError("target cannot be frozen while preprocessing failures exist")

    target = pd.DataFrame(
        {
            "subject_id": manifest["subject_id"],
            "target_ahi": manifest["ahi_xml_desat3_proxy"].astype(float),
            "target_class": manifest["label_severity"].astype(np.int8),
            "target_version": TARGET_VERSION,
            "qc_pass": qc_pass(manifest),
        }
    )
    target.insert(3, "target_class_name", CLASS_NAMES[target["target_class"].to_numpy()])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260919)
    target["fold"] = -1
    for fold, (_, test_indices) in enumerate(splitter.split(target, target["target_class"])):
        target.loc[test_indices, "fold"] = fold
    target["fold"] = target["fold"].astype(np.int8)

    canonical = "".join(
        f"{row.subject_id},{row.target_ahi:.10f},{int(row.target_class)},{int(row.qc_pass)},{int(row.fold)}\n"
        for row in target.itertuples(index=False)
    ).encode("utf-8")
    target_hash = hashlib.sha256(canonical).hexdigest()
    target.to_parquet(args.output_dir / "targets.parquet", index=False)
    target.to_csv(args.output_dir / "targets.csv", index=False, float_format="%.10f")

    definition = {
        "target_version": TARGET_VERSION,
        "status": "frozen_local_reconstruction_not_official_nsrr_csv",
        "target_sha256": target_hash,
        "source_manifest_sha256": sha256_file(manifest_path),
        "subjects": int(len(target)),
        "qc_pass_subjects": int(target["qc_pass"].sum()),
        "formula": "(valid sleep apneas + valid sleep hypopneas paired to >=3% XML oxygen desaturation) / XML sleep hours",
        "denominator": {
            "source": "NSRR XML stage annotations",
            "sleep_stages": ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "REM"],
            "epoch_seconds": 30,
        },
        "apnea_numerator_rule": {
            "types": ["Obstructive apnea", "Central apnea", "Mixed apnea"],
            "minimum_duration_seconds": 10,
            "sleep_rule": "event midpoint must be in a sleep epoch",
            "desaturation_required": False,
        },
        "hypopnea_numerator_rule": {
            "types": ["Hypopnea", "Unsure"],
            "minimum_duration_seconds": 10,
            "sleep_rule": "event midpoint must be in a sleep epoch",
            "desaturation_pairing": "SpO2-desaturation annotation at respiratory-event start + 30.0 seconds, matched by nearest duration",
            "minimum_desaturation_percent": 3.0,
        },
        "severity_thresholds": {"normal": "AHI < 5", "mild": "5 <= AHI < 15", "moderate": "15 <= AHI < 30", "severe": "AHI >= 30"},
        "cv": {"type": "StratifiedKFold", "folds": 5, "shuffle": True, "random_state": 20260919},
        "class_counts": {str(k): int(v) for k, v in target["target_class"].value_counts().sort_index().items()},
        "fold_class_counts": {
            str(fold): {str(k): int(v) for k, v in group["target_class"].value_counts().sort_index().items()}
            for fold, group in target.groupby("fold")
        },
        "upgrade_path": "Create a new immutable target version when official shhs1-dataset-0.21.0.csv is supplied; do not overwrite this version.",
    }
    (args.output_dir / "target_definition.json").write_text(
        json.dumps(definition, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "TARGET_SHA256").write_text(target_hash + "\n", encoding="ascii")
    print(json.dumps(definition, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
