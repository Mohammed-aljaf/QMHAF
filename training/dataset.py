"""Small, strict loader for the frozen ahi_a0h3_local_v1 training assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_VERSION = "ahi_a0h3_local_v1"
CLASS_NAMES = ("normal", "mild", "moderate", "severe")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(root: Path = ROOT) -> dict:
    target_dir = root / "training" / "targets" / TARGET_VERSION
    table_dir = root / "training" / "tables" / TARGET_VERSION
    target_definition = json.loads((target_dir / "target_definition.json").read_text(encoding="utf-8"))
    table_definition = json.loads((table_dir / "table_definition.json").read_text(encoding="utf-8"))
    target = pd.read_parquet(target_dir / "targets.parquet").sort_values("subject_id").reset_index(drop=True)
    canonical = "".join(
        f"{row.subject_id},{row.target_ahi:.10f},{int(row.target_class)},{int(row.qc_pass)},{int(row.fold)}\n"
        for row in target.itertuples(index=False)
    ).encode("utf-8")
    recomputed_target_hash = hashlib.sha256(canonical).hexdigest()
    checks = {
        "target_version": target_definition["target_version"] == TARGET_VERSION,
        "target_canonical_hash": (
            (target_dir / "TARGET_SHA256").read_text(encoding="ascii").strip()
            == target_definition["target_sha256"] == recomputed_target_hash
        ),
        "allnight_file_hash": (
            _sha256(table_dir / "allnight_features.parquet") == table_definition["allnight_sha256"]
        ),
        "sleep_oracle_file_hash": (
            _sha256(table_dir / "sleep_oracle_features.parquet")
            == table_definition["sleep_oracle_sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"training asset verification failed: {checks}")
    return checks


def load_dataset(
    scenario: str = "allnight",
    qc_only: bool = False,
    root: Path = ROOT,
    verify: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return X, severity y, continuous AHI y, and immutable row metadata."""
    if scenario not in {"allnight", "sleep_oracle"}:
        raise ValueError("scenario must be 'allnight' or 'sleep_oracle'")
    if verify:
        verify_assets(root)
    target_path = root / "training" / "targets" / TARGET_VERSION / "targets.parquet"
    feature_path = root / "training" / "tables" / TARGET_VERSION / f"{scenario}_features.parquet"
    targets = pd.read_parquet(target_path)
    features = pd.read_parquet(feature_path)
    merged = targets.merge(features, on="subject_id", how="inner", validate="one_to_one")
    if len(merged) != len(targets):
        raise RuntimeError("target/feature subject mismatch")
    if qc_only:
        merged = merged.loc[merged["qc_pass"]].reset_index(drop=True)
    metadata_columns = ["subject_id", "target_version", "qc_pass", "fold"]
    target_columns = ["target_ahi", "target_class", "target_class_name"]
    x = merged.drop(columns=metadata_columns + target_columns)
    forbidden = ("target", "label", "ahi", "event", "arousal", "sleep_stage")
    offenders = [column for column in x.columns if any(token in column.lower() for token in forbidden)]
    if offenders:
        raise RuntimeError(f"potential target leakage columns: {offenders}")
    return (
        x.astype(np.float32),
        merged["target_class"].astype(np.int8),
        merged["target_ahi"].astype(float),
        merged[metadata_columns].copy(),
    )


def fold_indices(metadata: pd.DataFrame, test_fold: int) -> tuple[np.ndarray, np.ndarray]:
    if test_fold not in range(5):
        raise ValueError("test_fold must be one of 0, 1, 2, 3, 4")
    test = metadata["fold"].to_numpy() == test_fold
    return np.flatnonzero(~test), np.flatnonzero(test)
