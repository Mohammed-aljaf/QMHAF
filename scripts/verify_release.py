#!/usr/bin/env python3
"""Verify identities, folds, predictions, metrics, calibration and optional weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from analysis.metrics import PCOLS, apply_saved_calibration, summarize

VERSION = "ahi_a0h3_local_v1"
DEEP = ROOT / f"deep_learning/results/{VERSION}"
NAMES = ("xception", "resnet", "cnn_bigru_attention", "multiscale_transformer", "dual_branch_cnn")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def check_frame(frame, expected=None, raw=False):
    if not frame.subject_id.is_unique: raise ValueError("Duplicate subject predictions")
    if not np.isfinite(frame[PCOLS].to_numpy()).all(): raise ValueError("Nonfinite probabilities")
    p = frame[PCOLS].to_numpy()
    np.testing.assert_allclose(p.sum(1), 1, atol=.002 if raw else 1e-5, rtol=0)
    np.testing.assert_array_equal(p.argmax(1), frame.predicted_class)
    if expected is not None:
        np.testing.assert_array_equal(frame.subject_id, expected.subject_id)
        np.testing.assert_array_equal(frame.target_class, expected.target_class)
        np.testing.assert_array_equal(frame.fold, expected.fold)
        np.testing.assert_allclose(frame.target_ahi, expected.target_ahi, rtol=0, atol=1e-5)


def verify(checkpoints=False, checksums=False):
    complete = json.loads((ROOT / "server_run/RUN_COMPLETE.json").read_text())
    if complete["run_mode"] != "production" or not complete["main_complete"]: raise ValueError("Production completion record invalid")
    fixed = pd.read_csv(ROOT / "splits/frozen_subject_folds.csv").sort_values("subject_id").reset_index(drop=True)
    q = pd.read_parquet(DEEP / "main/all/aggregate_oof_predictions.parquet").sort_values("subject_id").reset_index(drop=True)
    expected = fixed.loc[fixed.qc_pass].reset_index(drop=True)
    if len(fixed) != 5785 or len(q) != 5197: raise ValueError("Incorrect cohort size")
    np.testing.assert_array_equal(q.subject_id, expected.subject_id)
    np.testing.assert_array_equal(q.fold, expected.fold)
    roles = pd.read_csv(ROOT / "splits/nested_roles.csv")
    if len(roles) != 5 * len(q) or roles.duplicated(["subject_id", "outer_fold"]).any(): raise ValueError("Invalid nested split coverage")
    for k, rows in roles.groupby("outer_fold"):
        calculated = np.where(rows.assigned_fold == k, "test", np.where(rows.assigned_fold == (k + 1) % 5, "validation", "train"))
        np.testing.assert_array_equal(rows.role, calculated)
    check_frame(q)
    m = summarize(q)
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "quadratic_kappa", "ahi_mae", "ahi_rmse", "confusion_matrix"):
        np.testing.assert_allclose(m[key], complete["metrics"][key], rtol=1e-6, atol=1e-5, err_msg=key)
    checks = dict(subjects=len(q), frozen_subjects=len(fixed), nested_role_rows=len(roles), deep_fold_runs=0,
        comparator_fold_runs=0, calibration_folds_checked=0, checkpoint_files_loaded=0)
    for path in sorted(DEEP.glob("*/*/fold_*/oof_predictions.parquet")):
        frame = pd.read_parquet(path).sort_values("subject_id").reset_index(drop=True)
        report = json.loads(path.with_name("test_metrics.json").read_text())
        check_frame(frame, q.loc[q.fold == report["test_fold"]].reset_index(drop=True), raw=report["calibration_parameters"] is None)
        values = summarize(frame)
        for key, value in report["test_metrics"].items():
            if key in values: np.testing.assert_allclose(values[key], value, rtol=1e-6, atol=1e-5, err_msg=str(path) + key)
        history = pd.read_csv(path.with_name("history.csv"))
        if report["best_epoch"] not in history.epoch.to_numpy(): raise ValueError("Best epoch not in history")
        if report["validation_fold"] != (report["test_fold"] + 1) % 5: raise ValueError("Unexpected validation fold")
        if report["calibration_parameters"] is not None:
            ahi, probability = apply_saved_calibration(frame.raw_predicted_ahi.to_numpy(), frame[["raw_" + c for c in PCOLS]].to_numpy(), report["calibration_parameters"])
            np.testing.assert_allclose(ahi, frame.predicted_ahi, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(probability, frame[PCOLS], rtol=1e-6, atol=1e-6)
            checks["calibration_folds_checked"] += 1
        checks["deep_fold_runs"] += 1
        if checkpoints:
            import torch
            from deep_learning.model import ModelConfig, QMHAFNet
            ck = torch.load(path.with_name("best_model.pt"), map_location="cpu", weights_only=False)
            model = QMHAFNet(ModelConfig(**ck["model_config"]))
            model.load_state_dict(ck["model"], strict=True)
            if ck["test_fold"] != report["test_fold"] or ck["epoch"] != report["best_epoch"]: raise ValueError("Checkpoint metadata mismatch")
            if sum(p.numel() for p in model.parameters()) != report["model_parameters"]: raise ValueError("Parameter count mismatch")
            if not all(torch.isfinite(v).all() for v in ck["model"].values()): raise ValueError("Nonfinite weights")
            checks["checkpoint_files_loaded"] += 1
    for name in NAMES:
        for fold in range(5):
            folder = ROOT / f"comparison_2024plus/results/{name}"
            frame = pd.read_parquet(folder / f"fold_{fold}_oof.parquet").sort_values("subject_id").reset_index(drop=True)
            check_frame(frame, q.loc[q.fold == fold].reset_index(drop=True))
            report = json.loads((folder / f"fold_{fold}_metadata.json").read_text())
            h = pd.read_csv(folder / f"fold_{fold}_history.csv")
            if report["best_epoch"] not in h.epoch.to_numpy(): raise ValueError("Comparator best epoch absent")
            values = summarize(frame)
            for key in ("accuracy", "macro_f1", "balanced_accuracy", "quadratic_kappa"):
                np.testing.assert_allclose(values[key], report["test_" + key], rtol=1e-6, atol=1e-5)
            checks["comparator_fold_runs"] += 1
            if checkpoints:
                import torch
                from comparison_2024plus.train_comparators import build_model
                ck = torch.load(folder / f"fold_{fold}_model.pt", map_location="cpu", weights_only=False)
                model = build_model(name, ck["feature_mean"].shape[-1]); model.load_state_dict(ck["model"], strict=True)
                if sum(p.numel() for p in model.parameters()) != report["parameters"]: raise ValueError("Comparator parameter count mismatch")
                if ck["fold"] != fold or ck["architecture"] != name: raise ValueError("Comparator checkpoint metadata mismatch")
                for key in ("feature_mean", "feature_std"):
                    if not np.isfinite(ck[key]).all(): raise ValueError("Nonfinite comparator normalization")
                if not all(torch.isfinite(v).all() for v in ck["model"].values()): raise ValueError("Nonfinite comparator weights")
                checks["checkpoint_files_loaded"] += 1
    if checks["deep_fold_runs"] != 14 or checks["comparator_fold_runs"] != 25: raise ValueError("Incomplete run artifacts")
    if checksums:
        for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
            digest, relative = line.split("  ", 1)
            if sha256(ROOT / relative) != digest: raise ValueError(f"Checksum mismatch: {relative}")
        checks["checksum_files_checked"] = len((ROOT / "SHA256SUMS.txt").read_text().splitlines())
    checks["status"] = "passed"
    checks["missing_original_artifacts"] = ["Five production SSL initialization checkpoints; see docs/LIMITATIONS.md"]
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", action="store_true")
    p.add_argument("--checksums", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args(); result = verify(args.checkpoints, args.checksums)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
