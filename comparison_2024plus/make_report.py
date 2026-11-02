"""Validate aligned OOF predictions and create manuscript-ready comparison assets."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score, roc_auc_score


ROOT = Path("comparison_2024plus")
NAMES = ("xception", "resnet", "cnn_bigru_attention", "multiscale_transformer", "dual_branch_cnn")
LABELS = {"xception": "Xception (2024)", "resnet": "ResNet (2024)",
          "cnn_bigru_attention": "CNN-BiGRU-Attention (2024)",
          "multiscale_transformer": "Multiscale Transformer (2025)",
          "dual_branch_cnn": "Dual-branch CNN (2025)",
          "Q-MHAF": "Q-MHAF (proposed)", "LightGBM": "LightGBM reference"}


def ci(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def metrics(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    y = df.target_class.to_numpy(int)
    pred = df.predicted_class.to_numpy(int)
    n = len(y)
    acc_samples = np.empty(2000)
    f1_samples = np.empty(2000)
    for b in range(2000):
        idx = rng.integers(0, n, n)
        acc_samples[b] = accuracy_score(y[idx], pred[idx])
        f1_samples[b] = f1_score(y[idx], pred[idx], labels=[0, 1, 2, 3], average="macro", zero_division=0)
    ret = {"n": n, "accuracy": accuracy_score(y, pred), "accuracy_ci": ci(acc_samples),
           "macro_f1": f1_score(y, pred, average="macro"), "macro_f1_ci": ci(f1_samples),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "quadratic_kappa": cohen_kappa_score(y, pred, weights="quadratic"),
           "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist()}
    ret.update({f"recall_{name}": float(value) for name, value in
                zip(("normal", "mild", "moderate", "severe"),
                    recall_score(y, pred, labels=[0, 1, 2, 3], average=None))})
    pcols = [f"probability_{group}" for group in ("normal", "mild", "moderate", "severe")]
    if all(col in df for col in pcols):
        probabilities = df[pcols].to_numpy()
        ret["macro_auroc_ovr"] = float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro"))
        ret["macro_auprc_ovr"] = float(average_precision_score(np.eye(4)[y], probabilities, average="macro"))
    if "predicted_ahi" in df:
        ret["ahi_mae"] = float(np.mean(np.abs(df.target_ahi - df.predicted_ahi)))
    return ret


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    # Repository release stores production predictions directly, without the ZIP.
    path = Path("deep_learning/results/ahi_a0h3_local_v1/main/all/aggregate_oof_predictions.parquet")
    qm = pd.read_parquet(path)
    qm = qm.sort_values("subject_id").reset_index(drop=True)
    assert len(qm) == 5197 and qm.subject_id.is_unique and set(qm.fold) == set(range(5))
    all_models = {"Q-MHAF": qm}
    for name in NAMES:
        paths = [ROOT / "results" / name / f"fold_{k}_oof.parquet" for k in range(5)]
        if not all(p.exists() for p in paths):
            raise FileNotFoundError(f"Incomplete model {name}: {[str(p) for p in paths if not p.exists()]}")
        df = pd.concat([pd.read_parquet(p) for p in paths]).sort_values("subject_id").reset_index(drop=True)
        assert len(df) == len(qm) and df.subject_id.is_unique
        assert (df.subject_id == qm.subject_id).all()
        assert (df.target_class == qm.target_class).all()
        assert (df.fold == qm.fold).all()
        assert np.allclose(df.target_ahi, qm.target_ahi, atol=1e-5)
        prob_cols = [f"probability_{group}" for group in ("normal", "mild", "moderate", "severe")]
        probabilities = df[prob_cols].to_numpy()
        assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        assert np.array_equal(df.predicted_class.to_numpy(), probabilities.argmax(axis=1))
        all_models[name] = df
    lightgbm = pd.read_parquet("training/results/ahi_a0h3_local_v1/allnight_all/lgbm_logahi_regression/oof_predictions.parquet")
    lightgbm = lightgbm.loc[lightgbm.qc_pass].sort_values("subject_id").reset_index(drop=True)
    assert (lightgbm.subject_id == qm.subject_id).all()
    assert (lightgbm.target_class == qm.target_class).all()
    all_models["LightGBM"] = lightgbm
    combined = qm[["subject_id", "fold", "target_class", "target_ahi", "predicted_class", "predicted_ahi"]].rename(
        columns={"predicted_class": "qmhaf_predicted_class", "predicted_ahi": "qmhaf_predicted_ahi"})
    for name in (*NAMES, "LightGBM"):
        combined[f"{name}_predicted_class"] = all_models[name].predicted_class.to_numpy()
    combined.to_parquet(ROOT / "aligned_oof_predictions.parquet", index=False)
    rng = np.random.default_rng(20260922)
    results = {name: metrics(df, rng) for name, df in all_models.items()}
    qcorrect = qm.target_class.to_numpy() == qm.predicted_class.to_numpy()
    y = qm.target_class.to_numpy()
    for name in (*NAMES, "LightGBM"):
        df = all_models[name]
        c = y == df.predicted_class.to_numpy()
        better_q = int(np.sum(qcorrect & ~c))
        better_other = int(np.sum(~qcorrect & c))
        results[name]["paired_mcnemar_p_uncorrected"] = binomtest(
            min(better_q, better_other), better_q + better_other, 0.5).pvalue
        results[name]["discordant_q_better"] = better_q
        results[name]["discordant_other_better"] = better_other
        diffs = np.empty(2000)
        for b in range(2000):
            idx = rng.integers(0, len(y), len(y))
            diffs[b] = qcorrect[idx].mean() - c[idx].mean()
        results[name]["delta_accuracy_q_minus_model_ci"] = ci(diffs)
        results[name]["delta_accuracy_q_minus_model"] = float(qcorrect.mean() - c.mean())
    ordered_p = sorted((results[name]["paired_mcnemar_p_uncorrected"], name) for name in NAMES)
    running = 0.0
    for rank, (pvalue, name) in enumerate(ordered_p):
        running = max(running, min(1.0, pvalue * (len(NAMES) - rank)))
        results[name]["paired_mcnemar_p_holm"] = running
    (ROOT / "metrics.json").write_text(json.dumps(results, indent=2))
    rows = []
    for name in (*NAMES, "Q-MHAF", "LightGBM"):
        m = results[name]
        rows.append({"method": LABELS[name], "n": m["n"],
                     "accuracy": m["accuracy"], "accuracy_ci_low": m["accuracy_ci"][0],
                     "accuracy_ci_high": m["accuracy_ci"][1], "macro_f1": m["macro_f1"],
                     "macro_f1_ci_low": m["macro_f1_ci"][0], "macro_f1_ci_high": m["macro_f1_ci"][1],
                     "balanced_accuracy": m["balanced_accuracy"], "quadratic_kappa": m["quadratic_kappa"],
                     "macro_auroc_ovr": m.get("macro_auroc_ovr", np.nan),
                     "macro_auprc_ovr": m.get("macro_auprc_ovr", np.nan),
                     "ahi_mae": m.get("ahi_mae", np.nan),
                     "p_mcnemar_vs_qmhaf": m.get("paired_mcnemar_p_uncorrected", np.nan),
                     "p_mcnemar_holm": m.get("paired_mcnemar_p_holm", np.nan),
                     "delta_accuracy_q_minus_model": m.get("delta_accuracy_q_minus_model", np.nan)})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "comparison_table.csv", index=False)
    recall_rows = []
    for name in (*NAMES, "Q-MHAF", "LightGBM"):
        m = results[name]
        recall_rows.append({"method": LABELS[name], **{f"recall_{group}": m[f"recall_{group}"]
                                                      for group in ("normal", "mild", "moderate", "severe")}})
    pd.DataFrame(recall_rows).to_csv(ROOT / "class_recall_table.csv", index=False)
    latex = [r"\begin{table*}[t]", r"\centering", r"\caption{Subject-level SHHS1 four-class OSA severity comparison on the same 5,197 QC-passing test subjects and frozen five-fold split. Comparator networks are architecture-adapted reimplementations, not original-paper results. The supplemental LightGBM model was trained using all 5,785 subjects and evaluated on the same QC test subset. Brackets are 95\% subject-bootstrap CIs; QWK denotes quadratic weighted kappa. Holm-adjusted p-values come from paired exact McNemar tests against Q-MHAF.}",
             r"\label{tab:recent_comparison}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\hline",
             "Method & Accuracy (95\\% CI) & Macro-F1 (95\\% CI) & Balanced Acc. & QWK & $p_{\\mathrm{Holm}}$ " + "\\\\", r"\hline"]
    for row in rows:
        p_holm = row["p_mcnemar_holm"]
        p_text = "--" if np.isnan(p_holm) else ("$<0.001$" if p_holm < 0.001 else f"{p_holm:.3f}")
        latex.append(f"{row['method']} & {row['accuracy']:.3f} [{row['accuracy_ci_low']:.3f}, {row['accuracy_ci_high']:.3f}] & {row['macro_f1']:.3f} [{row['macro_f1_ci_low']:.3f}, {row['macro_f1_ci_high']:.3f}] & {row['balanced_accuracy']:.3f} & {row['quadratic_kappa']:.3f} & {p_text} " + "\\\\")
    latex += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    (ROOT / "comparison_table.tex").write_text("\n".join(latex) + "\n")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    plot = table.iloc[:6]
    positions = np.arange(len(plot))
    values = plot.accuracy.to_numpy()
    errors = np.vstack((values - plot.accuracy_ci_low.to_numpy(), plot.accuracy_ci_high.to_numpy() - values))
    colors = ["#4682a9"] * 5 + ["#d26639"]
    for i, (value, color) in enumerate(zip(values, colors)):
        ax.errorbar(value, i, xerr=errors[:, i:i+1], fmt="o", markersize=8,
                    linewidth=2, capsize=4, color=color)
        ax.text(value + 0.018, i, f"{value:.3f}", va="center", fontsize=10)
    ax.set_yticks(positions, plot.method)
    ax.invert_yaxis()
    ax.set_xlim(0.68, 0.89)
    ax.set_xlabel("Four-class subject-level accuracy (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ROOT / "comparison_accuracy.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "comparison_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "comparison_accuracy.svg", bbox_inches="tight")
    plt.close(fig)
    best_name = max(NAMES, key=lambda n: results[n]["accuracy"])
    best = results[best_name]
    proposed = results["Q-MHAF"]
    delta = best["delta_accuracy_q_minus_model"]
    dci = best["delta_accuracy_q_minus_model_ci"]
    delta_wording = "in favor of Q-MHAF" if delta >= 0 else "in favor of the comparator"
    manuscript = f"""# Results text for the manuscript

## Experimental comparison

We evaluated five architecture-adapted comparators on the same 5,197 quality-controlled SHHS1 subjects and the same frozen subject-level five-fold partition as Q-MHAF. Each comparator used three folds for training, one for validation and model selection, and one untouched outer fold for testing. Physiological inputs were represented as 256 chronological full-night bins derived from oximetry, heart rate, and respiratory channels. The comparator networks are adapted from the cited architectural families; they are not exact reproductions of their original publications.

## Results

Across pooled out-of-fold predictions, Q-MHAF achieved accuracy {proposed['accuracy']:.3f} (95% subject-bootstrap CI {proposed['accuracy_ci'][0]:.3f}–{proposed['accuracy_ci'][1]:.3f}), macro-F1 {proposed['macro_f1']:.3f}, balanced accuracy {proposed['balanced_accuracy']:.3f}, and quadratic weighted kappa {proposed['quadratic_kappa']:.3f}. The strongest adapted comparator was {LABELS[best_name]} (accuracy {best['accuracy']:.3f}, macro-F1 {best['macro_f1']:.3f}); the paired accuracy difference {delta_wording} was {abs(delta):.3f} (95% subject-bootstrap CI for Q-MHAF minus comparator {dci[0]:.3f}–{dci[1]:.3f}; two-sided exact McNemar test with Holm correction across the five architecture comparisons, p={best['paired_mcnemar_p_holm']:.3g}). Q-MHAF class recalls were {proposed['recall_normal']:.3f}, {proposed['recall_mild']:.3f}, {proposed['recall_moderate']:.3f}, and {proposed['recall_severe']:.3f} for normal, mild, moderate, and severe groups, respectively. Its AHI mean absolute error was {proposed['ahi_mae']:.3f} events/h. Table 1 and Figure 1 present the matched comparison.

## Table 1 caption

Four-class OSA severity classification on the same 5,197 QC-passing SHHS1 participants using pooled subject-level out-of-fold predictions. The five comparator networks are adapted architectural implementations trained on identical 256-bin full-night representations; Q-MHAF uses its original multirate waveform pipeline. Accuracy intervals are 95% subject-bootstrap intervals. QWK denotes quadratic weighted kappa. Published accuracies from the source studies are excluded because tasks, cohorts, signal sets, and AHI definitions differ.

## Figure 1 caption

Four-class subject-level accuracy for five adapted literature architectures and Q-MHAF on the identical SHHS1 test subjects. Error bars denote 95% subject-bootstrap confidence intervals. Orange marks Q-MHAF.

## Interpretation constraint

The common subjects, labels, and outer folds make this a matched evaluation of complete systems. Different signal representations and training procedures mean that the result does not isolate the causal effect of the Q-MHAF architecture. The frozen local AHI target was reconstructed from XML annotations and must not be described as the official NSRR clinical-table AHI. Published performance figures for pediatric, ECG, or event-detection studies are not directly interchangeable with these four-class adult SHHS1 scores.

## Architectural sources

1. Yook et al., *Sleep Medicine* (2024), Xception event-detection architecture. https://doi.org/10.1016/j.sleep.2024.01.015
2. Mortazavi et al., *Scientific Reports* (2024), 1D ResNet and CNN–BiGRU–Attention architectures. https://doi.org/10.1038/s41598-024-67729-9
3. Zhang et al., *Nature and Science of Sleep* (2025), multimodal multiscale Transformer. https://doi.org/10.2147/NSS.S492806
4. Kolhar et al., *Biomedicines* (2025), dual-branch CNN. https://doi.org/10.3390/biomedicines13051090
"""
    (ROOT / "manuscript_results.md").write_text(manuscript)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
