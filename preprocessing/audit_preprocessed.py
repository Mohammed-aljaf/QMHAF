#!/usr/bin/env python3
"""Audit the complete SHHS cache before any model design or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


AHI_COLUMNS = (
    "ahi_xml_desat3_proxy",
    "ahi_raw_desat3_proxy",
    "ahi_raw_desat3_or_arousal_proxy",
)
SEVERITY_NAMES = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}


def severity(values: np.ndarray) -> np.ndarray:
    return np.select([values < 5, values < 15, values < 30], [0, 1, 2], default=3).astype(np.int8)


def finite_stats(values: np.ndarray) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "p01": float(np.percentile(x, 1)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def class_counts(values: np.ndarray) -> dict:
    classes = severity(np.asarray(values, float))
    return {SEVERITY_NAMES[int(k)]: int(v) for k, v in sorted(Counter(classes).items())}


def pairwise_label_audit(frame: pd.DataFrame) -> dict:
    report = {}
    for i, first in enumerate(AHI_COLUMNS):
        for second in AHI_COLUMNS[i + 1 :]:
            a = frame[first].to_numpy(float)
            b = frame[second].to_numpy(float)
            good = np.isfinite(a) & np.isfinite(b)
            ca, cb = severity(a[good]), severity(b[good])
            confusion = np.zeros((4, 4), dtype=int)
            np.add.at(confusion, (ca, cb), 1)
            report[f"{first}__vs__{second}"] = {
                "n": int(good.sum()),
                "mae_events_per_hour": float(np.mean(np.abs(a[good] - b[good]))),
                "median_absolute_error": float(np.median(np.abs(a[good] - b[good]))),
                "pearson_r": float(np.corrcoef(a[good], b[good])[0, 1]),
                "severity_agreement": float(np.mean(ca == cb)),
                "confusion_rows_first_cols_second": confusion.tolist(),
            }
    return report


def pairwise_label_audit_by_qc(frame: pd.DataFrame) -> dict:
    high_quality = (
        (frame["sleep_seconds"] >= 2 * 3600)
        & (frame["spo2_valid_fraction"] >= 0.80)
        & (frame["thor_valid_fraction"] >= 0.70)
        & (frame["abdo_valid_fraction"] >= 0.70)
        & (
            (frame["airflow_valid_fraction"] >= 0.50)
            | ((frame["thor_valid_fraction"] >= 0.90) & (frame["abdo_valid_fraction"] >= 0.90))
        )
    )
    return {
        "qc_pass": pairwise_label_audit(frame[high_quality].reset_index(drop=True)),
        "qc_fail": pairwise_label_audit(frame[~high_quality].reset_index(drop=True)),
    }


def quality_audit(frame: pd.DataFrame) -> dict:
    quality_columns = (
        "spo2_valid_fraction",
        "hr_valid_fraction",
        "airflow_valid_fraction",
        "thor_valid_fraction",
        "abdo_valid_fraction",
        "airflow_clip_fraction",
        "thor_clip_fraction",
        "abdo_clip_fraction",
    )
    label = frame["label_ahi"].to_numpy(float)
    classes = frame["label_severity"].to_numpy(int)
    report = {}
    for column in quality_columns:
        values = frame[column].to_numpy(float)
        good = np.isfinite(values) & np.isfinite(label)
        rho, pvalue = stats.spearmanr(values[good], label[good]) if good.sum() > 2 else (np.nan, np.nan)
        medians = {
            SEVERITY_NAMES[c]: float(np.nanmedian(values[classes == c])) for c in range(4)
        }
        report[column] = {
            "distribution": finite_stats(values),
            "spearman_with_selected_ahi": float(rho),
            "spearman_pvalue": float(pvalue),
            "median_by_severity": medians,
        }

    airflow_table = pd.crosstab(frame["airflow_channel"], frame["label_severity"])
    if airflow_table.shape[0] > 1 and airflow_table.shape[1] > 1:
        chi2, pvalue, _, _ = stats.chi2_contingency(airflow_table)
        n = airflow_table.to_numpy().sum()
        cramers_v = math.sqrt(chi2 / max(n * min(airflow_table.shape[0] - 1, airflow_table.shape[1] - 1), 1))
    else:
        chi2, pvalue, cramers_v = np.nan, np.nan, np.nan
    report["airflow_channel_vs_severity"] = {
        "table": {str(idx): {str(c): int(v) for c, v in row.items()} for idx, row in airflow_table.iterrows()},
        "chi2": float(chi2),
        "pvalue": float(pvalue),
        "cramers_v": float(cramers_v),
    }
    return report


def retention_audit(frame: pd.DataFrame) -> dict:
    rules = {
        "sleep_at_least_2h": frame["sleep_seconds"] >= 2 * 3600,
        "record_at_least_4h": frame["record_seconds"] >= 4 * 3600,
        "spo2_valid_at_least_80pct": frame["spo2_valid_fraction"] >= 0.80,
        "both_effort_valid_at_least_70pct": (frame["thor_valid_fraction"] >= 0.70)
        & (frame["abdo_valid_fraction"] >= 0.70),
        "some_flow_or_strong_effort": (frame["airflow_valid_fraction"] >= 0.50)
        | ((frame["thor_valid_fraction"] >= 0.90) & (frame["abdo_valid_fraction"] >= 0.90)),
    }
    keep = np.ones(len(frame), dtype=bool)
    report = {}
    for name, mask in rules.items():
        mask = mask.to_numpy(bool)
        report[name] = {
            "pass": int(mask.sum()),
            "fail": int((~mask).sum()),
            "pass_by_severity": {
                SEVERITY_NAMES[c]: int(np.sum(mask & (frame["label_severity"].to_numpy() == c)))
                for c in range(4)
            },
        }
        keep &= mask
    report["all_initial_qc_rules"] = {
        "pass": int(keep.sum()),
        "fail": int((~keep).sum()),
        "retention_fraction": float(np.mean(keep)),
        "retained_by_severity": {
            SEVERITY_NAMES[c]: int(np.sum(keep & (frame["label_severity"].to_numpy() == c)))
            for c in range(4)
        },
    }
    return report


def deep_cache_audit(root: Path, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    feature_names: list[str] | None = None
    epoch_finite: np.ndarray | None = None
    epoch_total = 0
    subject_feature_rows = []
    subject_feature_ids = []
    event_counts = Counter()
    signal_fingerprints: dict[str, list[str]] = {}
    label_consistency_errors = []
    errors = []
    total_uncompressed_array_bytes = 0
    for index, row in frame.iterrows():
        sid = row["subject_id"]
        path = root / "subjects" / f"{sid}.npz"
        try:
            with np.load(path, allow_pickle=False) as data:
                spo2_cache = data["spo2_1hz"]
                resp_cache = data["resp_q10"]
                seconds = spo2_cache.size
                assert resp_cache.shape == (3, seconds * 10)
                assert data["resp_valid_1hz"].shape == (3, seconds)
                assert data["hr_1hz"].size == seconds
                assert data["position_1hz"].size == seconds
                features = data["epoch_features_30s"].astype(np.float32)
                stages = data["sleep_stage_30s"]
                current_names = [str(x) for x in data["feature_names"]]
                assert features.shape[0] == stages.size
                assert features.shape[1] == len(current_names)
                if feature_names is None:
                    feature_names = current_names
                    epoch_finite = np.zeros(len(feature_names), dtype=np.int64)
                assert current_names == feature_names
                assert epoch_finite is not None
                epoch_finite += np.sum(np.isfinite(features), axis=0)
                epoch_total += features.shape[0]
                sleep = np.isin(stages, [1, 2, 3, 4])
                selected = features[sleep] if np.any(sleep) else features
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    subject_feature_rows.append(np.nanmedian(selected, axis=0))
                subject_feature_ids.append(sid)
                event_counts.update(data["event_type"].tolist())
                fingerprint = hashlib.blake2b(digest_size=16)
                fingerprint.update(np.ascontiguousarray(resp_cache).view(np.uint8))
                fingerprint.update(np.ascontiguousarray(spo2_cache).view(np.uint8))
                digest = fingerprint.hexdigest()
                signal_fingerprints.setdefault(digest, []).append(sid)

                event_type = data["event_type"]
                event_flags = data["event_flags"]
                core = (event_flags & 3) == 3
                xml_valid = core & (np.isin(event_type, [1, 2, 3]) | ((event_type >= 4) & ((event_flags & 4) > 0)))
                reconstructed = float(np.sum(xml_valid) / max(float(row["sleep_seconds"]) / 3600.0, 1e-12))
                if not np.isclose(reconstructed, float(row["ahi_xml_desat3_proxy"]), rtol=0, atol=1e-8):
                    label_consistency_errors.append(
                        {"subject_id": sid, "stored": float(row["ahi_xml_desat3_proxy"]), "reconstructed": reconstructed}
                    )
                total_uncompressed_array_bytes += sum(data[name].nbytes for name in data.files)
        except Exception as exc:
            errors.append({"subject_id": sid, "error": f"{type(exc).__name__}: {exc}"})
        if (index + 1) % 500 == 0:
            print(f"deep-audit {index+1}/{len(frame)}", flush=True)

    feature_names = feature_names or []
    finite_fraction = epoch_finite / max(epoch_total, 1) if epoch_finite is not None else np.array([])
    subject_features = np.asarray(subject_feature_rows, dtype=np.float32)
    ahi_by_id = frame.set_index("subject_id")["label_ahi"]
    selected_ahi = ahi_by_id.loc[subject_feature_ids].to_numpy(float)
    feature_report = {}
    for i, name in enumerate(feature_names):
        x = subject_features[:, i]
        good = np.isfinite(x) & np.isfinite(selected_ahi)
        if good.sum() > 2 and np.nanstd(x[good]) > 0:
            rho = stats.spearmanr(x[good], selected_ahi[good]).statistic
        else:
            rho = np.nan
        feature_report[name] = {
            "epoch_finite_fraction": float(finite_fraction[i]),
            "subject_median_distribution": finite_stats(x),
            "subject_median_spearman_with_ahi": float(rho),
        }
    forbidden = ("label", "ahi", "event_type", "event_flag", "sleep_stage", "arousal")
    leakage_names = [name for name in feature_names if any(token in name.lower() for token in forbidden)]
    quality_only_names = [name for name in feature_names if name.endswith("valid_fraction")]
    duplicate_groups = [ids for ids in signal_fingerprints.values() if len(ids) > 1]
    report = {
        "files_checked": int(len(frame)),
        "errors": errors,
        "epoch_rows_checked": int(epoch_total),
        "feature_count": len(feature_names),
        "feature_leakage_name_violations": leakage_names,
        "quality_only_feature_names": quality_only_names,
        "model_input_feature_count_after_qc_exclusion": len(feature_names) - len(quality_only_names),
        "event_type_counts": {str(k): int(v) for k, v in sorted(event_counts.items())},
        "duplicate_signal_groups": duplicate_groups,
        "label_consistency_errors": label_consistency_errors,
        "compressed_bytes": int(frame["cache_bytes"].sum()),
        "uncompressed_array_bytes": int(total_uncompressed_array_bytes),
        "features": feature_report,
    }
    subject_frame = pd.DataFrame(subject_features, columns=feature_names)
    subject_frame.insert(0, "subject_id", subject_feature_ids)
    return report, subject_frame


def markdown_report(report: dict) -> str:
    overall = report["overall"]
    lines = [
        "# SHHS preprocessing audit",
        "",
        f"- Successful subjects: {overall['subjects_ok']}",
        f"- Failed subjects: {overall['subjects_failed']}",
        f"- Cache size: {overall['cache_gib']:.2f} GiB",
        f"- Label source counts: {overall['label_sources']}",
        f"- Selected severity distribution: {overall['selected_severity_distribution']}",
        "",
        "## Initial QC retention",
        "",
        f"`{report['retention']['all_initial_qc_rules']}`",
        "",
        "## Label agreement",
        "",
    ]
    for name, values in report["label_pairwise"].items():
        lines.append(
            f"- {name}: MAE={values['mae_events_per_hour']:.3f}, "
            f"severity agreement={values['severity_agreement']:.3f}"
        )
    lines += [
        "",
        "## Boundary sensitivity",
        "",
        f"`{report['boundary_counts']}`",
        "",
        "## Deep integrity",
        "",
        f"- Files checked: {report['deep']['files_checked']}",
        f"- Integrity errors: {len(report['deep']['errors'])}",
        f"- Epochs checked: {report['deep']['epoch_rows_checked']}",
        f"- Input feature count: {report['deep']['feature_count']}",
        f"- Forbidden feature-name violations: {report['deep']['feature_leakage_name_violations']}",
        "",
        "This audit describes data readiness only; it does not report model performance.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", nargs="?", type=Path, default=Path("processed_shhs_v1"))
    parser.add_argument("--skip-deep", action="store_true")
    args = parser.parse_args()
    manifest = pd.read_parquet(args.cache_dir / "manifest.parquet")
    if manifest["subject_id"].duplicated().any():
        raise ValueError("duplicate subject ids in manifest")
    ok = manifest[manifest["status"] == "ok"].sort_values("subject_id").reset_index(drop=True)
    selected = ok["label_ahi"].to_numpy(float)
    boundary_counts = {
        f"within_{width:g}_of_{threshold}": int(np.sum(np.abs(selected - threshold) <= width))
        for width in (0.5, 1.0, 2.0)
        for threshold in (5, 15, 30)
    }
    report = {
        "overall": {
            "subjects_total": int(len(manifest)),
            "subjects_ok": int(len(ok)),
            "subjects_failed": int(len(manifest) - len(ok)),
            "cache_gib": float(ok["cache_bytes"].sum() / 2**30),
            "label_sources": ok["label_source"].value_counts().to_dict(),
            "selected_severity_distribution": class_counts(selected),
        },
        "ahi_distributions": {column: finite_stats(ok[column].to_numpy(float)) for column in AHI_COLUMNS},
        "severity_distributions": {column: class_counts(ok[column].to_numpy(float)) for column in AHI_COLUMNS},
        "label_pairwise": pairwise_label_audit(ok),
        "label_pairwise_by_qc": pairwise_label_audit_by_qc(ok),
        "boundary_counts": boundary_counts,
        "quality": quality_audit(ok),
        "retention": retention_audit(ok),
    }
    if args.skip_deep:
        report["deep"] = {
            "files_checked": 0,
            "errors": [],
            "epoch_rows_checked": 0,
            "feature_count": 0,
            "feature_leakage_name_violations": [],
        }
    else:
        report["deep"], subject_features = deep_cache_audit(args.cache_dir, ok)
        subject_features.to_parquet(args.cache_dir / "subject_feature_medians.parquet", index=False)
        roles = {
            name: ("quality_only" if name.endswith("valid_fraction") else "model_input")
            for name in subject_features.columns
            if name != "subject_id"
        }
        (args.cache_dir / "feature_roles.json").write_text(
            json.dumps(roles, indent=2, sort_keys=True), encoding="utf-8"
        )
    (args.cache_dir / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (args.cache_dir / "audit_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))
    return 1 if report["deep"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
