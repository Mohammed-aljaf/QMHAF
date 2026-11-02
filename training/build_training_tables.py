#!/usr/bin/env python3
"""Build leakage-controlled subject-level tables from the immutable SHHS cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter


AGGREGATIONS = ("mean", "std", "p10", "p50", "p90")
SLEEP_CODES = (1, 2, 3, 4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_values(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return values[valid & np.isfinite(values)]


def safe_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p01": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    q = np.percentile(values, [1, 5, 10, 50, 90, 95, 99])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p01": float(q[0]),
        f"{prefix}_p05": float(q[1]),
        f"{prefix}_p10": float(q[2]),
        f"{prefix}_p50": float(q[3]),
        f"{prefix}_p90": float(q[4]),
        f"{prefix}_p95": float(q[5]),
        f"{prefix}_p99": float(q[6]),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def desaturation_features(spo2: np.ndarray, valid: np.ndarray, hours: float) -> dict[str, float]:
    """Signal-only desaturation proxies; deliberately independent of XML events."""
    result: dict[str, float] = {}
    if not np.any(valid) or hours <= 0:
        for threshold in (3, 4):
            result[f"raw_odi{threshold}_recording_hour"] = np.nan
            result[f"raw_drop{threshold}_count"] = np.nan
        return result

    x = spo2.astype(np.float32, copy=True)
    good = valid & np.isfinite(x)
    positions = np.arange(x.size)
    if np.sum(good) < 2:
        x[~good] = float(np.nanmedian(x[good]))
    else:
        x[~good] = np.interp(positions[~good], positions[good], x[good])
    smooth = median_filter(x, size=5, mode="nearest")
    local_baseline = maximum_filter1d(smooth, size=120, mode="nearest")
    drop = local_baseline - smooth
    for threshold in (3, 4):
        active = (drop >= threshold) & good
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        # Merge fragmentary detections within 20 seconds into one physiological episode.
        if starts.size:
            count = 1 + int(np.sum(np.diff(starts) >= 20))
        else:
            count = 0
        result[f"raw_drop{threshold}_count"] = float(count)
        result[f"raw_odi{threshold}_recording_hour"] = float(count / hours)
    return result


def aggregate_epoch_matrix(
    matrix: np.ndarray, names: list[str], indices: np.ndarray, mask: np.ndarray | None, prefix: str
) -> dict[str, float]:
    selected = matrix[:, indices]
    if mask is not None:
        selected = selected[mask]
    output: dict[str, float] = {}
    if selected.shape[0] == 0:
        for name in names:
            for aggregation in AGGREGATIONS:
                output[f"{prefix}{name}__{aggregation}"] = np.nan
        return output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(selected, axis=0)
        stds = np.nanstd(selected, axis=0)
        percentiles = np.nanpercentile(selected, [10, 50, 90], axis=0)
    arrays = (means, stds, percentiles[0], percentiles[1], percentiles[2])
    for column, name in enumerate(names):
        for aggregation, values in zip(AGGREGATIONS, arrays):
            output[f"{prefix}{name}__{aggregation}"] = float(values[column])
    return output


def process_subject(arguments: tuple[str, str, dict[str, str]]) -> tuple[dict, dict]:
    path_string, subject_id, roles = arguments
    path = Path(path_string)
    with np.load(path, allow_pickle=False) as data:
        names_all = [str(value) for value in data["feature_names"]]
        indices = np.array([index for index, name in enumerate(names_all) if roles.get(name) == "model_input"])
        names = [names_all[index] for index in indices]
        matrix = data["epoch_features_30s"].astype(np.float32)
        stages = data["sleep_stage_30s"]
        seconds = int(data["spo2_1hz"].size)
        hours = seconds / 3600.0

        common: dict[str, float | str] = {"subject_id": subject_id, "record_hours": hours}
        spo2 = finite_values(data["spo2_1hz"], data["spo2_valid_1hz"])
        common.update(safe_summary(spo2, "raw_spo2"))
        if spo2.size:
            common["raw_spo2_fraction_below_90"] = float(np.mean(spo2 < 90))
            common["raw_spo2_fraction_below_88"] = float(np.mean(spo2 < 88))
        else:
            common["raw_spo2_fraction_below_90"] = np.nan
            common["raw_spo2_fraction_below_88"] = np.nan
        common.update(desaturation_features(data["spo2_1hz"], data["spo2_valid_1hz"], hours))

        hr = finite_values(data["hr_1hz"], data["hr_valid_1hz"])
        common.update(safe_summary(hr, "raw_hr"))
        common["raw_hr_rmssd"] = float(np.sqrt(np.mean(np.diff(hr) ** 2))) if hr.size > 1 else np.nan

        position = data["position_1hz"]
        for code in range(4):
            common[f"raw_position_{code}_fraction"] = float(np.mean(position == code))

        allnight = dict(common)
        allnight.update(aggregate_epoch_matrix(matrix, names, indices, None, ""))

        sleep_mask = np.isin(stages, SLEEP_CODES)
        oracle = dict(common)
        oracle["oracle_sleep_hours"] = float(np.sum(sleep_mask) * 30 / 3600)
        oracle["oracle_sleep_fraction"] = float(np.mean(sleep_mask))
        oracle.update(aggregate_epoch_matrix(matrix, names, indices, sleep_mask, ""))
    return allnight, oracle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("processed_shhs_v1"))
    parser.add_argument("--target-dir", type=Path, default=Path("training/targets/ahi_a0h3_local_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/tables/ahi_a0h3_local_v1"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_parquet(args.target_dir / "targets.parquet").sort_values("subject_id")
    roles = json.loads((args.cache_dir / "feature_roles.json").read_text(encoding="utf-8"))
    model_names = [name for name, role in roles.items() if role == "model_input"]
    quality_names = [name for name, role in roles.items() if role == "quality_only"]
    jobs = [
        (str(args.cache_dir / "subjects" / f"{sid}.npz"), sid, roles)
        for sid in targets["subject_id"].tolist()
    ]
    allnight_rows: list[dict] = []
    oracle_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for count, (allnight, oracle) in enumerate(executor.map(process_subject, jobs, chunksize=8), 1):
            allnight_rows.append(allnight)
            oracle_rows.append(oracle)
            if count % 500 == 0 or count == len(jobs):
                print(f"aggregated {count}/{len(jobs)} subjects", flush=True)

    allnight_frame = pd.DataFrame(allnight_rows).sort_values("subject_id").reset_index(drop=True)
    oracle_frame = pd.DataFrame(oracle_rows).sort_values("subject_id").reset_index(drop=True)
    if not allnight_frame["subject_id"].equals(targets["subject_id"].reset_index(drop=True)):
        raise RuntimeError("subject order mismatch")
    for frame in (allnight_frame, oracle_frame):
        frame.iloc[:, 1:] = frame.iloc[:, 1:].astype(np.float32)

    allnight_path = args.output_dir / "allnight_features.parquet"
    oracle_path = args.output_dir / "sleep_oracle_features.parquet"
    allnight_frame.to_parquet(allnight_path, index=False, compression="zstd")
    oracle_frame.to_parquet(oracle_path, index=False, compression="zstd")
    metadata = {
        "target_version": "ahi_a0h3_local_v1",
        "subjects": len(targets),
        "model_input_epoch_features": len(model_names),
        "excluded_quality_only_epoch_features": quality_names,
        "aggregations": list(AGGREGATIONS),
        "allnight_columns_including_subject_id": len(allnight_frame.columns),
        "sleep_oracle_columns_including_subject_id": len(oracle_frame.columns),
        "allnight_stage_information_used": False,
        "sleep_oracle_stage_information_used": True,
        "annotation_or_event_arrays_used_as_features": False,
        "allnight_sha256": sha256_file(allnight_path),
        "sleep_oracle_sha256": sha256_file(oracle_path),
    }
    (args.output_dir / "table_definition.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
