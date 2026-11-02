#!/usr/bin/env python3
"""Deterministic five-fold OOF baselines for the frozen ahi_a0h3 target."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ("normal", "mild", "moderate", "severe")
SEED = 20260919


def ahi_to_class(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, [5.0, 15.0, 30.0]).astype(np.int8)


def model_spec(name: str):
    common = dict(random_state=SEED, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
    if name == "logistic_balanced":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=0.2, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=SEED),
        )
    if name == "extra_trees_balanced":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=0.7,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=4,
            ),
        )
    if name == "lgbm_unweighted":
        return LGBMClassifier(
            objective="multiclass", num_class=4, n_estimators=500, learning_rate=0.03,
            num_leaves=31, min_child_samples=25, subsample=0.85, colsample_bytree=0.75,
            reg_lambda=2.0, **common,
        )
    if name == "lgbm_balanced":
        return LGBMClassifier(
            objective="multiclass", num_class=4, n_estimators=500, learning_rate=0.03,
            num_leaves=31, min_child_samples=25, subsample=0.85, colsample_bytree=0.75,
            reg_lambda=2.0, class_weight="balanced", **common,
        )
    if name == "catboost_balanced":
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=450,
            depth=7,
            learning_rate=0.04,
            l2_leaf_reg=4.0,
            auto_class_weights="Balanced",
            random_seed=SEED,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        )
    if name == "lgbm_logahi_regression":
        return LGBMRegressor(
            objective="huber", n_estimators=600, learning_rate=0.025, num_leaves=31,
            min_child_samples=25, subsample=0.85, colsample_bytree=0.75, reg_lambda=2.0, **common,
        )
    raise ValueError(name)


def classification_metrics(y_true: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    recalls = recall_score(y_true, predicted, labels=np.arange(4), average=None, zero_division=0)
    output = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, predicted, weights="quadratic")),
        "mean_absolute_class_error": float(np.mean(np.abs(y_true - predicted))),
        "log_loss": float(log_loss(y_true, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=np.arange(4))),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=np.arange(4)).tolist(),
    }
    for index, name in enumerate(CLASS_NAMES):
        output[f"recall_{name}"] = float(recalls[index])
    try:
        output["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        output["roc_auc_ovr_macro"] = np.nan
    return output


def bootstrap_intervals(y: np.ndarray, prediction: np.ndarray, iterations: int = 1000) -> dict:
    generator = np.random.default_rng(SEED)
    accuracy = np.empty(iterations)
    macro_f1 = np.empty(iterations)
    for index in range(iterations):
        sample = generator.integers(0, y.size, size=y.size)
        accuracy[index] = accuracy_score(y[sample], prediction[sample])
        macro_f1[index] = f1_score(y[sample], prediction[sample], average="macro", zero_division=0)
    return {
        "accuracy_95ci": [float(x) for x in np.percentile(accuracy, [2.5, 97.5])],
        "macro_f1_95ci": [float(x) for x in np.percentile(macro_f1, [2.5, 97.5])],
    }


def extract_importance(model, feature_names: list[str]) -> np.ndarray | None:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(np.asarray(estimator.coef_, dtype=float)), axis=0)
    else:
        return None
    # Imputation indicators added by linear/tree pipelines are excluded here.
    values = values[: len(feature_names)]
    total = np.sum(values)
    return values / total if total > 0 else values


def fit_predict_fold(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    ahi_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, object | None]:
    if model_name == "dummy_prior":
        probabilities = np.bincount(y_train, minlength=4).astype(float)
        probabilities /= probabilities.sum()
        probabilities = np.repeat(probabilities[None, :], x_test.shape[0], axis=0)
        predicted = np.argmax(probabilities, axis=1).astype(np.int8)
        return predicted, probabilities, None, None
    model = model_spec(model_name)
    if model_name == "lgbm_logahi_regression":
        model.fit(x_train, np.log1p(ahi_train))
        ahi_prediction = np.maximum(0, np.expm1(model.predict(x_test)))
        predicted = ahi_to_class(ahi_prediction)
        probabilities = np.eye(4, dtype=float)[predicted]
        return predicted, probabilities, ahi_prediction, model
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    predicted = np.argmax(probabilities, axis=1).astype(np.int8)
    return predicted, probabilities, None, model


def evaluate_scenario(
    scenario: str,
    table_path: Path,
    targets: pd.DataFrame,
    models: list[str],
    output_dir: Path,
) -> list[dict]:
    features = pd.read_parquet(table_path)
    data = targets.merge(features, on="subject_id", validate="one_to_one")
    feature_names = [column for column in features.columns if column != "subject_id"]
    x = data[feature_names].to_numpy(np.float32)
    y = data["target_class"].to_numpy(np.int8)
    ahi = data["target_ahi"].to_numpy(float)
    folds = data["fold"].to_numpy(np.int8)
    finite_fraction = np.mean(np.isfinite(x), axis=0)
    keep = finite_fraction > 0.01
    x = x[:, keep]
    feature_names = [name for name, selected in zip(feature_names, keep) if selected]
    print(f"{scenario}: {len(data)} subjects, {len(feature_names)} usable features", flush=True)

    summaries = []
    for model_name in models:
        started = time.time()
        prediction = np.full(len(data), -1, dtype=np.int8)
        probabilities = np.full((len(data), 4), np.nan, dtype=float)
        ahi_prediction = np.full(len(data), np.nan, dtype=float)
        fold_rows = []
        importance_rows = []
        for fold in range(5):
            test = folds == fold
            train = ~test
            predicted, proba, predicted_ahi, model = fit_predict_fold(
                model_name, x[train], y[train], ahi[train], x[test]
            )
            prediction[test] = predicted
            probabilities[test] = proba
            if predicted_ahi is not None:
                ahi_prediction[test] = predicted_ahi
            metrics = classification_metrics(y[test], predicted, proba)
            metrics.update({"fold": fold, "n_test": int(np.sum(test))})
            if predicted_ahi is not None:
                metrics["ahi_mae"] = float(mean_absolute_error(ahi[test], predicted_ahi))
                metrics["ahi_rmse"] = float(np.sqrt(mean_squared_error(ahi[test], predicted_ahi)))
            fold_rows.append(metrics)
            if model is not None:
                importance = extract_importance(model, feature_names)
                if importance is not None:
                    importance_rows.append(importance)
            print(
                f"  {model_name} fold={fold} accuracy={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}", flush=True
            )

        overall = classification_metrics(y, prediction, probabilities)
        overall.update(bootstrap_intervals(y, prediction))
        qc_mask = data["qc_pass"].to_numpy(bool)
        qc_metrics = classification_metrics(y[qc_mask], prediction[qc_mask], probabilities[qc_mask])
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "quadratic_kappa"):
            overall[f"qc_pass_{metric_name}"] = qc_metrics[metric_name]
        overall["qc_pass_subjects"] = int(np.sum(qc_mask))
        overall.update(
            {
                "scenario": scenario,
                "model": model_name,
                "subjects": len(data),
                "features": len(feature_names),
                "runtime_seconds": float(time.time() - started),
                "fold_accuracy_mean": float(np.mean([row["accuracy"] for row in fold_rows])),
                "fold_accuracy_std": float(np.std([row["accuracy"] for row in fold_rows], ddof=1)),
                "fold_macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_rows])),
                "fold_macro_f1_std": float(np.std([row["macro_f1"] for row in fold_rows], ddof=1)),
            }
        )
        if np.any(np.isfinite(ahi_prediction)):
            overall["ahi_mae"] = float(mean_absolute_error(ahi, ahi_prediction))
            overall["ahi_rmse"] = float(np.sqrt(mean_squared_error(ahi, ahi_prediction)))
            overall["ahi_pearson"] = float(np.corrcoef(ahi, ahi_prediction)[0, 1])

        model_dir = output_dir / scenario / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        oof = data[["subject_id", "target_ahi", "target_class", "fold", "qc_pass"]].copy()
        oof["predicted_class"] = prediction
        for class_index, class_name in enumerate(CLASS_NAMES):
            oof[f"probability_{class_name}"] = probabilities[:, class_index]
        if np.any(np.isfinite(ahi_prediction)):
            oof["predicted_ahi"] = ahi_prediction
        oof.to_parquet(model_dir / "oof_predictions.parquet", index=False)
        (model_dir / "metrics.json").write_text(
            json.dumps({"overall": overall, "folds": fold_rows}, indent=2, allow_nan=True), encoding="utf-8"
        )
        if importance_rows:
            importance = np.mean(np.vstack(importance_rows), axis=0)
            importance_frame = pd.DataFrame({"feature": feature_names, "importance": importance})
            importance_frame.sort_values("importance", ascending=False).to_csv(
                model_dir / "feature_importance.csv", index=False
            )
        summaries.append(overall)
        print(
            f"DONE {scenario}/{model_name}: accuracy={overall['accuracy']:.4f} "
            f"macro_f1={overall['macro_f1']:.4f} time={overall['runtime_seconds']:.1f}s", flush=True
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=Path("training/targets/ahi_a0h3_local_v1"))
    parser.add_argument("--table-dir", type=Path, default=Path("training/tables/ahi_a0h3_local_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/results/ahi_a0h3_local_v1"))
    parser.add_argument("--skip-catboost", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(args.target_dir / "targets.parquet")

    allnight_models = [
        "dummy_prior", "logistic_balanced", "extra_trees_balanced",
        "lgbm_unweighted", "lgbm_balanced", "lgbm_logahi_regression",
    ]
    if not args.skip_catboost:
        allnight_models.append("catboost_balanced")
    summaries = evaluate_scenario(
        "allnight_all", args.table_dir / "allnight_features.parquet", targets,
        allnight_models, args.output_dir,
    )
    # The oracle uses XML sleep stages and is reported only as a research upper-bound.
    oracle_models = ["lgbm_unweighted", "lgbm_balanced", "lgbm_logahi_regression"]
    summaries.extend(
        evaluate_scenario(
            "sleep_oracle_all", args.table_dir / "sleep_oracle_features.parquet", targets,
            oracle_models, args.output_dir,
        )
    )
    summary_frame = pd.DataFrame(summaries).sort_values(["scenario", "accuracy"], ascending=[True, False])
    summary_frame.to_csv(args.output_dir / "baseline_summary.csv", index=False)
    summary_frame.to_parquet(args.output_dir / "baseline_summary.parquet", index=False)
    print("\n" + summary_frame[[
        "scenario", "model", "accuracy", "macro_f1", "balanced_accuracy",
        "qc_pass_accuracy", "qc_pass_macro_f1",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
