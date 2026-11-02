#!/usr/bin/env python3
"""Prepare privately obtained SHHS data and enforce the released frozen splits."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = "ahi_a0h3_local_v1"


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


def check_splits(target_path):
    fixed = pd.read_csv(ROOT / "splits/frozen_subject_folds.csv").sort_values("subject_id").reset_index(drop=True)
    actual = pd.read_parquet(target_path).sort_values("subject_id").reset_index(drop=True)
    for col in ("subject_id", "fold", "qc_pass"):
        if not np.array_equal(fixed[col], actual[col]):
            raise ValueError(f"Prepared {col} differs from released fixed splits; check cohort, target definition and dependency versions")
    canonical = "".join(f"{r.subject_id},{r.target_ahi:.10f},{int(r.target_class)},{int(r.qc_pass)},{int(r.fold)}\n"
        for r in actual.itertuples(index=False)).encode()
    expected_hash = (ROOT / "splits/TARGET_SHA256").read_text().strip()
    if hashlib.sha256(canonical).hexdigest() != expected_hash:
        raise ValueError("Reconstructed target values differ from the released immutable target checksum")
    return len(actual)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edf-dir", type=Path)
    p.add_argument("--xml-dir", type=Path)
    p.add_argument("--existing-cache", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--build-classical-tables", action="store_true")
    args = p.parse_args()
    cache = ROOT / "processed_shhs_v1"
    if args.existing_cache:
        existing = args.existing_cache.resolve()
        if existing != cache.resolve():
            if cache.exists():
                raise ValueError(f"{cache} already exists; use that cache or select a fresh checkout")
            if not (existing / "manifest.parquet").is_file():
                raise ValueError("Existing cache must contain manifest.parquet and subjects/*.npz")
            cache.symlink_to(existing, target_is_directory=True)
    else:
        if not args.edf_dir or not args.xml_dir:
            p.error("Provide --existing-cache, or both --edf-dir and --xml-dir")
        for directory in (args.edf_dir, args.xml_dir):
            if not directory.is_dir(): p.error(f"Directory does not exist: {directory}")
        run("preprocessing/preprocess_shhs.py", "--edf-dir", args.edf_dir.resolve(),
            "--xml-dir", args.xml_dir.resolve(), "--workers", args.workers)
    target_dir = ROOT / f"training/targets/{VERSION}"
    if not (target_dir / "targets.parquet").exists():
        run("training/freeze_target.py")
    n = check_splits(target_dir / "targets.parquet")
    run("deep_learning/prepare_deep_targets.py")
    if args.build_classical_tables:
        run("training/build_training_tables.py", "--workers", args.workers)
    print(f"Prepared {n} private records; all subject IDs, folds and QC flags match the released split.")


if __name__ == "__main__":
    main()
