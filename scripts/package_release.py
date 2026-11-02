#!/usr/bin/env python3
"""Inventory and ZIP this repository, excluding all datasets and new local runs.

Use --inventory-only to refresh SHA256SUMS and MANIFEST without creating an archive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
    "runs", "processed_shhs_v1", "labels", "shhs1", "shsh1", "data_private"}
SKIP_PATHS = ("deep_learning/data/", "deep_learning/pretrained/", "training/targets/", "training/tables/")
FORBIDDEN_SUFFIXES = {".edf", ".xml", ".npz", ".h5", ".hdf5", ".npy", ".pyc", ".zip"}
INVENTORY = {"MANIFEST.json", "SHA256SUMS.txt"}


def files_for_release():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts): continue
        if str(relative).startswith(SKIP_PATHS): continue
        if not path.is_file() or path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in INVENTORY: continue
        if path.is_symlink(): raise ValueError(f"Do not publish symlinks: {relative}")
        if path.stat().st_size >= 100 * 1024 ** 2: raise ValueError(f"Oversized repository file: {relative}")
        yield path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT.parent / "QMHAF_GITHUB_PACKAGE.zip")
    args = p.parse_args(); files = list(files_for_release())
    rows = [dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path)) for path in files]
    (ROOT / "SHA256SUMS.txt").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in rows))
    manifest = dict(package="Q-MHAF dataset-free reproducibility repository", version="1.0.0",
        file_count=len(rows), total_bytes=sum(r["bytes"] for r in rows), dataset_included=False,
        included_checkpoints=39, production_ssl_checkpoints_available=False,
        fixed_split_subjects=5785, evaluated_subjects=5197, generated_tables=12, result_figures=7, original_method_diagrams=5,
        checksum_file_sha256=sha256(ROOT / "SHA256SUMS.txt"), files=rows)
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.inventory_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files + [ROOT / name for name in sorted(INVENTORY)]:
                archive.write(path, str(Path(ROOT.name) / path.relative_to(ROOT)))
        hashpath = args.output.with_name(args.output.name + ".sha256")
        hashpath.write_text(f"{sha256(args.output)}  {args.output.name}\n")
        print(f"Release: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")
    print(f"Inventory: {len(rows)} files, {sum(r['bytes'] for r in rows) / 1024**2:.1f} MiB; no recording or feature caches.")


if __name__ == "__main__":
    main()
