"""Numerical definitions used by the publication analysis.

Bootstrap draws have the original seed, order, and sample size. Confusion-matrix
arithmetic replaces repeated sklearn calls without changing the estimand.
"""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, recall_score, roc_auc_score)

CLASSES = ("normal", "mild", "moderate", "severe")
PCOLS = ["probability_" + c for c in CLASSES]


def summarize(frame):
    y = frame.target_class.to_numpy(int)
    pred = frame.predicted_class.to_numpy(int)
    cm = confusion_matrix(y, pred, labels=range(4))
    result = dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=range(4), average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        quadratic_kappa=float(cohen_kappa_score(y, pred, weights="quadratic")),
        confusion_matrix=cm.tolist())
    result.update({"recall_" + c: float(v) for c, v in zip(CLASSES,
        recall_score(y, pred, labels=range(4), average=None, zero_division=0))})
    if all(c in frame for c in PCOLS):
        p = frame[PCOLS].to_numpy(float)
        # Final probabilities have unit sums. Normalizing also handles AMP rounding.
        p /= p.sum(axis=1, keepdims=True)
        result.update(macro_auroc_ovr=float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
            macro_auprc_ovr=float(average_precision_score(np.eye(4)[y], p, average="macro")))
    if "predicted_ahi" in frame:
        error = frame.predicted_ahi.to_numpy(float) - frame.target_ahi.to_numpy(float)
        result.update(ahi_mae=float(np.mean(np.abs(error))), ahi_rmse=float(np.sqrt(np.mean(error ** 2))))
    return result


def bootstrap(frame, rng, iterations=2000):
    y = frame.target_class.to_numpy(int)
    pred = frame.predicted_class.to_numpy(int)
    encoded = y * 4 + pred
    samples = np.empty((iterations, 2))
    for b in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        cm = np.bincount(encoded[idx], minlength=16).reshape(4, 4)
        tp = np.diag(cm)
        denom = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2 * tp, denom, out=np.zeros(4), where=denom != 0)
        samples[b] = [tp.sum() / len(y), f1.mean()]
    return {"accuracy_ci": np.quantile(samples[:, 0], [.025, .975]).tolist(),
            "macro_f1_ci": np.quantile(samples[:, 1], [.025, .975]).tolist()}


def reliability(frame, raw=False, bins=15):
    prefix = "raw_" if raw else ""
    p = frame[[prefix + c for c in PCOLS]].to_numpy(float)
    # The original calibration analysis uses saved probability magnitudes verbatim.
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == frame.target_class.to_numpy(int)
    rows = []
    for k in range(bins):
        lower, upper = k / bins, (k + 1) / bins
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            rows.append(dict(bin=k, lower=lower, upper=upper, n=int(mask.sum()),
                confidence=float(confidence[mask].mean()), accuracy=float(correct[mask].mean())))
    return rows


def probability_quality(frame, raw=False):
    prefix = "raw_" if raw else ""
    p = frame[[prefix + c for c in PCOLS]].to_numpy(float)
    y = frame.target_class.to_numpy(int)
    rows = reliability(frame, raw)
    return dict(brier=float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1))),
        nll=float(log_loss(y, p / p.sum(axis=1, keepdims=True), labels=range(4))),
        ece15=float(sum(r["n"] * abs(r["accuracy"] - r["confidence"]) for r in rows) / len(y)),
        accuracy=float(np.mean(p.argmax(axis=1) == y)))


def apply_saved_calibration(raw_ahi, raw_probabilities, parameters):
    """Apply validation-fitted coefficients; never fit on evaluation labels."""
    if parameters is None:
        return np.asarray(raw_ahi), np.asarray(raw_probabilities)
    raw_ahi = np.asarray(raw_ahi)
    raw_probabilities = np.asarray(raw_probabilities)
    log_ahi = np.log1p(np.maximum(raw_ahi, 0))
    ahi = np.maximum(0, np.expm1(log_ahi * parameters["ahi_coefficient"][0] + parameters["ahi_intercept"]))
    features = np.column_stack([np.log(raw_probabilities.clip(1e-6, 1)), np.log1p(ahi)])
    logits = features @ np.asarray(parameters["class_coefficients"]).T + np.asarray(parameters["class_intercepts"])
    return ahi, softmax(logits, axis=1)
