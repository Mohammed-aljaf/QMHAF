#!/usr/bin/env python3
"""Rebuild every numbered manuscript table and result figure without SHHS data.

Usage: python3 -m analysis.reproduce --output generated
Original checkpoints, predictions, logs and reference tables are read-only.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

from analysis.metrics import CLASSES, PCOLS, bootstrap, probability_quality, reliability, summarize

ROOT = Path(__file__).resolve().parents[1]
VERSION = "ahi_a0h3_local_v1"
RESULTS = ROOT / f"deep_learning/results/{VERSION}"
NAMES = ("xception", "resnet", "cnn_bigru_attention", "multiscale_transformer", "dual_branch_cnn")
LABELS = {"Q-MHAF": "Q-MHAF", "xception": "Xception-style CNN", "resnet": "1-D ResNet",
    "cnn_bigru_attention": "CNN–BiGRU–Attention", "multiscale_transformer": "Multiscale Transformer",
    "dual_branch_cnn": "Dual-branch CNN", "LightGBM": "LightGBM reference"}
ARCH = {"A0_supervised_direct": "Supervised direct model", "A1_ssl_direct": "SSL + direct AHI",
    "A2_ssl_factorised": "SSL + factorised AHI", "A3_quality_no_boundary": "SSL + factorisation + quality routing"}
SIGNALS = {"no_hr": "All except HR", "spo2_only": "Oximetry only", "resp_only": "Respiration only",
    "spo2_airflow": "Oximetry + airflow", "spo2_effort": "Oximetry + thoracic/abdominal effort"}
COLORS = ["#2768a5", "#129a98", "#d58b30", "#99539a"]


def load(path):
    return pd.read_parquet(path).sort_values("subject_id").reset_index(drop=True)


def json_save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(output, number, frame, caption):
    stem = output / "tables" / f"table_{number:02d}"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".tex").write_text(frame.to_latex(index=False, escape=True,
        float_format=lambda x: f"{x:.3f}", caption=caption, label=f"tab:reproduced_{number:02d}"))


def save(output, fig, name):
    for ext in ("pdf", "png", "svg"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(output / "figures" / f"{name}.{ext}", dpi=300, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def alignment(reference, other, name):
    if not other.subject_id.is_unique or len(other) != len(reference):
        raise ValueError(f"Invalid identity coverage for {name}")
    for col in ("subject_id", "target_class", "fold"):
        if not np.array_equal(reference[col], other[col]):
            raise ValueError(f"Misaligned {col} in {name}")
    np.testing.assert_allclose(reference.target_ahi, other.target_ahi, atol=1e-5, rtol=0)


def build_metrics(models):
    # The original make_report.py uses a single generator in this exact order.
    rng = np.random.default_rng(20260922)
    results = {name: {**summarize(frame), **bootstrap(frame, rng)} for name, frame in models.items()}
    q = models["Q-MHAF"]
    qcorrect = q.predicted_class.to_numpy() == q.target_class.to_numpy()
    for name in (*NAMES, "LightGBM"):
        frame = models[name]
        correct = frame.predicted_class.to_numpy() == frame.target_class.to_numpy()
        better_q, better_other = int(np.sum(qcorrect & ~correct)), int(np.sum(~qcorrect & correct))
        diffs = np.empty(2000)
        for b in range(2000):
            idx = rng.integers(0, len(q), len(q))
            diffs[b] = qcorrect[idx].mean() - correct[idx].mean()
        results[name].update(discordant_q_better=better_q, discordant_other_better=better_other,
            paired_mcnemar_p_uncorrected=float(binomtest(min(better_q, better_other), better_q + better_other, .5).pvalue) if better_q + better_other else 1.,
            delta_accuracy_q_minus_model=float(qcorrect.mean() - correct.mean()),
            delta_accuracy_q_minus_model_ci=np.quantile(diffs, [.025, .975]).tolist())
    running = 0.
    for rank, (p, name) in enumerate(sorted((results[n]["paired_mcnemar_p_uncorrected"], n) for n in NAMES)):
        running = max(running, min(1., p * (len(NAMES) - rank)))
        results[name]["paired_mcnemar_p_holm"] = running
    return results


def build_tables(output, models, metrics):
    q = models["Q-MHAF"]
    spec = pd.DataFrame([
        ["Respiratory", "Airflow, thoracic effort, abdominal effort", "3 × 300", "Three valid fractions"],
        ["Low-rate", "Absolute SpO2, desaturation depth, ΔSpO2, robust HR, ΔHR", "5 × 30", "SpO2 and HR valid fractions"]],
        columns=["Branch", "Channels", "Tensor", "Quality"])
    table(output, 1, spec, "Inputs per 30-second epoch; tensor shapes exclude the batch and epoch axes.")
    spec = pd.read_csv(ROOT / "paper/reference_tables/table_02.csv").fillna("")
    spec.iloc[0, 1] = "3 × 300; CNN kernels 15/7/5, strides 5/2/2"
    spec.iloc[1, 1] = "5 × 30; CNN kernels 5/5, strides 2/3"
    table(output, 2, spec, "Architecture specification; see deep_learning/model.py.")
    config = json.loads((ROOT / "configs/production.json").read_text())
    table(output, 3, pd.DataFrame([
        ["SSL", "Training", "30 epochs; batch 32; 160-epoch crop; LR 0.0003; AdamW; WD 0.05"],
        ["SSL", "Mask / teacher", "0.45 per modality/epoch; EMA 0.996 to 1"],
        ["Main", "Training", "35-epoch limit; batch 16; accumulation 1; LR 0.0002; WD 0.05; 8% warmup + cosine"],
        ["Main", "Selection", "Validation macro-F1; patience 8; minimum improvement 0.0001"],
        ["Ablations", "Training", "10-epoch limit; outer fold 0 only; batch 16; accumulation 1"],
        ["Calibration", "Fitting", "Validation-only Huber log-AHI, then multinomial logistic probabilities"],
        ["Evaluation", "Splits", "Five frozen subject folds; validation = (test + 1) mod 5"],
        ["Execution", "Environment", f"Python {config['environment']['python']}; torch {config['environment']['torch']}; RTX 3090; workers 0"]],
        columns=["Stage", "Parameter", "Value"]), "Executed configuration, including exploratory ablation limits.")
    definition = json.loads((ROOT / "configs/target_definition.json").read_text())
    counts = q.target_class.value_counts()
    table(output, 4, pd.DataFrame([
        ["Available labeled recordings", definition["subjects"], *[definition["class_counts"][str(i)] for i in range(4)]],
        ["Quality-controlled analysis cohort", len(q), *[int(counts[i]) for i in range(4)]]],
        columns=["Cohort", "Subjects", "Normal", "Mild", "Moderate", "Severe"]), "Frozen cohort composition.")
    table(output, 5, pd.read_csv(ROOT / "paper/reference_tables/table_05.csv"), "Adapted comparator motifs and their manuscript source citations.")
    main = metrics["Q-MHAF"]
    main_ci = bootstrap(q, np.random.default_rng(20260919))["accuracy_ci"]
    table(output, 6, pd.DataFrame([
        ["Accuracy (95% CI)", f"{main['accuracy']:.3f} ({main_ci[0]:.3f}–{main_ci[1]:.3f})"],
        ["Macro-F1", f"{main['macro_f1']:.3f}"], ["Balanced accuracy", f"{main['balanced_accuracy']:.3f}"],
        ["Quadratic weighted kappa", f"{main['quadratic_kappa']:.3f}"],
        ["Macro-AUROC / macro-AUPRC", f"{main['macro_auroc_ovr']:.3f} / {main['macro_auprc_ovr']:.3f}"],
        ["AHI MAE / RMSE (events/h)", f"{main['ahi_mae']:.3f} / {main['ahi_rmse']:.3f}"]],
        columns=["Measure", "Value"]), "Pooled five-fold out-of-fold performance; 2,000 subject bootstrap samples, seed 20260919.")
    folds = []
    for k, part in q.groupby("fold"):
        m = summarize(part)
        report = json.loads((RESULTS / f"main/all/fold_{k}/test_metrics.json").read_text())
        folds.append({"Fold": k + 1, "Test": len(part), "Best epoch": report["best_epoch"],
            "Acc.": m["accuracy"], "Macro-F1": m["macro_f1"], "Bal. acc.": m["balanced_accuracy"], "AHI MAE": m["ahi_mae"]})
    table(output, 7, pd.DataFrame(folds), "Independent outer-fold results. Display folds 1–5 correspond to stored folds 0–4.")
    comparison = []
    for name in (*NAMES, "Q-MHAF"):
        m = metrics[name]
        comparison.append([LABELS[name], f"{m['accuracy']:.3f} ({m['accuracy_ci'][0]:.3f}–{m['accuracy_ci'][1]:.3f})",
            m["macro_f1"], m["balanced_accuracy"], m["quadratic_kappa"]])
    table(output, 8, pd.DataFrame(comparison, columns=["Method", "Accuracy (95% CI)", "Macro-F1", "Bal. acc.", "QWK"]),
        "Matched-subject system comparison. Comparator inputs and optimization differ from Q-MHAF; 2,000 bootstrap samples, seed 20260922.")
    table(output, 9, pd.DataFrame([[c.title(), int(counts[i]), main["recall_" + c]] for i, c in enumerate(CLASSES)],
        columns=["Severity", "Support", "Recall"]), "Class support and recall from pooled outer-test predictions.")
    quality = []
    for raw, label in ((True, "Raw Q-MHAF"), (False, "Validation-calibrated")):
        m = probability_quality(q, raw)
        quality.append([label, m["brier"], m["nll"], m["ece15"], m["accuracy"]])
        pd.DataFrame(reliability(q, raw)).to_csv(output / "metrics" / f"reliability_{'raw' if raw else 'calibrated'}.csv", index=False)
    table(output, 10, pd.DataFrame(quality, columns=["Probability output", "Brier", "NLL", "ECE15", "Accuracy"]),
        "Probability quality; Brier is the mean sum over four classes; ECE uses 15 equal-width (lower, upper] bins.")
    ablation_metrics = []
    for number, variants, header, main_label in ((11, ARCH, "Configuration", "Full Q-MHAF"),
        (12, SIGNALS, "Signals", "All five physiological signals")):
        rows = []
        for name, label in variants.items():
            prefix = f"{name}/all" if number == 11 else f"signal_ablation/{name}"
            frame = load(RESULTS / prefix / "fold_0/oof_predictions.parquet")
            reference = q.loc[q.fold == 0].reset_index(drop=True)
            alignment(reference, frame, name)
            m = summarize(frame)
            rows.append([label, m["accuracy"], m["macro_f1"], m["ahi_mae"]])
            ablation_metrics.append(dict(experiment=name, group="architecture" if number == 11 else "signal",
                maximum_epochs=10, outer_fold=0, **m))
        m = summarize(q.loc[q.fold == 0])
        rows.append([main_label, m["accuracy"], m["macro_f1"], m["ahi_mae"]])
        table(output, number, pd.DataFrame(rows, columns=[header, "Accuracy", "Macro-F1", "AHI MAE"]),
            "Exploratory retraining on outer fold 0 (n=1,033). Reduced configurations: 10-epoch limit; full reference: 35-epoch limit. Effects are confounded by training budget and, for A0 versus A1, calibration.")
    json_save(output / "metrics/ablation_metrics.json", ablation_metrics)


def build_figures(output, q, metrics):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "qmhaf-v1", "pdf.fonttype": 42})
    diagrams = ["qmhaf_overall_architecture_600dpi", "qmhaf_multirate_fusion_600dpi", "qmhaf_hierarchy_head_600dpi",
        "qmhaf_ssl_pretraining_600dpi", "qmhaf_supervised_evaluation_600dpi"]
    for i, name in enumerate(diagrams, 1):
        shutil.copy2(ROOT / f"paper/figures_original/{name}.png", output / f"figures/figure_{i:02d}.png")
    folds = pd.read_csv(output / "tables/table_07.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for col in ("Acc.", "Macro-F1", "Bal. acc."):
        axes[0].plot(folds.Fold, folds[col], "o-", label=col)
    axes[0].set(xlabel="Outer test fold", ylabel="Score", xticks=range(1, 6), ylim=(.78, .88))
    axes[0].legend()
    axes[1].plot(folds.Fold, folds["AHI MAE"], "o-", color=COLORS[2])
    axes[1].set(xlabel="Outer test fold", ylabel="AHI MAE (events/h)", xticks=range(1, 6))
    save(output, fig, "figure_06_fold_stability")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, name in enumerate(NAMES):
        m = metrics[name]; delta = m["delta_accuracy_q_minus_model"]; low, high = m["delta_accuracy_q_minus_model_ci"]
        ax.errorbar(delta, i, xerr=[[delta - low], [high - delta]], fmt="o", capsize=4, color=COLORS[0])
    ax.axvline(0, linestyle="--", color="gray"); ax.invert_yaxis()
    ax.set(yticks=range(5), yticklabels=[LABELS[n] for n in NAMES], xlabel="Q-MHAF minus comparator accuracy (95% paired bootstrap CI)")
    save(output, fig, "figure_07_paired_accuracy_gains")
    cm = np.asarray(metrics["Q-MHAF"]["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, matrix, norm in ((axes[0], cm, False), (axes[1], cm / cm.sum(axis=1, keepdims=True), True)):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if norm else cm.max())
        for i in range(4):
            for j in range(4):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.1%}" if norm else str(int(v)), ha="center", va="center", color="white" if v > (.5 if norm else cm.max() / 2) else "black")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=[c.title() for c in CLASSES],
            yticklabels=[c.title() for c in CLASSES], xlabel="Predicted severity", ylabel="Reference severity", title="Row-normalized" if norm else "Counts")
    save(output, fig, "figure_08_confusion_matrix")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i, (name, color) in enumerate(zip(CLASSES, COLORS)):
        truth = (q.target_class.to_numpy() == i).astype(int); score = q[PCOLS[i]].to_numpy()
        fpr, tpr, _ = roc_curve(truth, score); precision, recall, _ = precision_recall_curve(truth, score)
        axes[0].plot(fpr, tpr, color=color, label=f"{name.title()}: {roc_auc_score(truth, score):.3f}")
        axes[1].plot(recall, precision, color=color, label=f"{name.title()}: {average_precision_score(truth, score):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for ax in axes: ax.legend(fontsize=8); ax.set(xlim=(0, 1), ylim=(0, 1.01))
    save(output, fig, "figure_09_discrimination_curves")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    hi = max(q.target_ahi.max(), q.predicted_ahi.max()) * 1.02
    hb = axes[0].hexbin(q.target_ahi, q.predicted_ahi, gridsize=65, bins="log", mincnt=1, cmap="viridis")
    axes[0].plot([0, hi], [0, hi], "--", color="#b33e3e")
    axes[0].set(xlim=(0, hi), ylim=(0, hi), xlabel="Reference AHI (events/h)", ylabel="Predicted AHI (events/h)")
    fig.colorbar(hb, ax=axes[0], label="Subjects / bin (log scale)")
    error = np.abs(q.predicted_ahi - q.target_ahi)
    axes[1].boxplot([error[q.target_class == i] for i in range(4)], showfliers=False, tick_labels=[c.title() for c in CLASSES])
    axes[1].set(ylabel="Absolute AHI error (events/h)", xlabel="Reference severity")
    save(output, fig, "figure_10_ahi_diagnostics")
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for raw, label, color in ((True, "Raw", COLORS[2]), (False, "Validation-calibrated", COLORS[0])):
        rows = pd.DataFrame(reliability(q, raw))
        ax.plot(rows.confidence, rows.accuracy, "-", color=color, label=label)
        ax.scatter(rows.confidence, rows.accuracy, s=15 + rows.n / 15, color=color, alpha=.7)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="Maximum class probability", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1)); ax.legend()
    save(output, fig, "figure_11_calibration_reliability")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, number, title in ((axes[0], 11, "Component configurations"), (axes[1], 12, "Signal subsets")):
        tab = pd.read_csv(output / f"tables/table_{number:02d}.csv"); x = np.arange(len(tab))
        labels = tab.iloc[:, 0].str.replace(" + ", " +\n", regex=False).str.replace("Oximetry +\nthoracic/abdominal effort", "SpO2 + effort", regex=False)
        ax.bar(x - .18, tab.Accuracy, .36, label="Accuracy", color=COLORS[0])
        ax.bar(x + .18, tab["Macro-F1"], .36, label="Macro-F1", color=COLORS[1])
        ax.set(xticks=x, xticklabels=labels, ylim=(.6, .9), ylabel="Score", title=title)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        other = ax.twinx(); other.plot(x, tab["AHI MAE"], "o--", color=COLORS[2], label="AHI MAE")
        other.set(ylabel="AHI MAE (events/h)", ylim=(1.8, 3.8))
        if number == 11: ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Outer fold 0, n=1,033: reduced configurations ≤10 epochs; full reference ≤35 epochs", fontsize=10)
    save(output, fig, "figure_12_component_signal_analysis")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for fold in range(5):
        h = pd.read_csv(RESULTS / f"main/all/fold_{fold}/history.csv")
        axes[0].plot(h.epoch, h.validation_macro_f1, label=f"Fold {fold + 1}")
        h = pd.read_csv(ROOT / f"histories/ssl_outer_fold_{fold}.csv")
        axes[1].plot(h.epoch, h.ssl_loss, label=f"Fold {fold + 1}")
    axes[0].set(xlabel="Supervised epoch", ylabel="Validation macro-F1")
    axes[1].set(xlabel="SSL epoch", ylabel="SSL loss")
    for ax in axes: ax.legend(fontsize=7)
    save(output, fig, "supplement_training_histories")


def audit_reference_tables(output):
    # Compare displayed result numbers against the latest supplied Word manuscript.
    # Tables 1–5 contain method descriptions and corrected math/specification fields.
    checks = []
    for number in range(6, 13):
        reference = pd.read_csv(ROOT / f"paper/reference_tables/table_{number:02d}.csv", dtype=str).fillna("")
        actual = pd.read_csv(output / f"tables/table_{number:02d}.csv", dtype=str).fillna("")
        if actual.shape != reference.shape:
            checks.append(dict(table=number, match=False, detail="shape differs")); continue
        for row in range(len(actual)):
            for col in range(1, len(actual.columns)):
                ref, value = reference.iloc[row, col], actual.iloc[row, col]
                try:
                    expected = float(ref.replace(",", "")); observed = float(value.replace(",", ""))
                    tolerance = .000500001 if "." in ref else 0
                    ok = abs(expected - observed) <= tolerance
                except ValueError:
                    ok = ref == value
                checks.append(dict(table=number, row=row, column=actual.columns[col], match=bool(ok), reference=ref, regenerated=value))
    json_save(output / "metrics/manuscript_numeric_audit.json", checks)
    failures = [c for c in checks if not c["match"]]
    if failures:
        raise ValueError(f"Manuscript numeric audit failed: {failures}")
    return len(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    parser.add_argument("--deep-results", type=Path, default=RESULTS)
    parser.add_argument("--comparator-results", type=Path, default=ROOT / "comparison_2024plus/results")
    parser.add_argument("--classical-results", type=Path, default=ROOT / f"training/results/{VERSION}")
    parser.add_argument("--skip-reference-audit", action="store_true", help="Analyze new training runs without requiring equality to the released manuscript")
    args = parser.parse_args(); output = args.output.resolve()
    # The same table/figure functions work with a new complete set of model runs.
    globals()["RESULTS"] = args.deep_results.resolve()
    for folder in ("tables", "figures", "metrics"): (output / folder).mkdir(parents=True, exist_ok=True)
    models = {"Q-MHAF": load(RESULTS / "main/all/aggregate_oof_predictions.parquet")}
    for name in NAMES:
        models[name] = pd.concat([load(args.comparator_results / f"{name}/fold_{k}_oof.parquet") for k in range(5)]).sort_values("subject_id").reset_index(drop=True)
        alignment(models["Q-MHAF"], models[name], name)
    lightgbm = load(args.classical_results / "allnight_all/lgbm_logahi_regression/oof_predictions.parquet")
    models["LightGBM"] = lightgbm.loc[lightgbm.qc_pass].reset_index(drop=True)
    alignment(models["Q-MHAF"], models["LightGBM"], "LightGBM")
    metrics = build_metrics(models)
    original = json.loads((ROOT / "comparison_2024plus/metrics.json").read_text())
    comparisons = 0
    for name, fields in ({} if args.skip_reference_audit else original).items():
        for key, value in fields.items():
            if key in metrics[name]:
                np.testing.assert_allclose(metrics[name][key], value, atol=1e-5, rtol=1e-6, err_msg=f"{name}/{key}")
                comparisons += 1
    json_save(output / "metrics/comparison_recomputed.json", metrics)
    build_tables(output, models, metrics); build_figures(output, models["Q-MHAF"], metrics)
    table_checks = 0 if args.skip_reference_audit else audit_reference_tables(output)
    json_save(output / "metrics/reproduction_report.json", {
        "status": "passed", "subjects": len(models["Q-MHAF"]), "original_metric_fields_checked": comparisons,
        "manuscript_numeric_cells_checked": table_checks, "numbered_tables": 12, "numbered_figures": 12,
        "result_figures_regenerated": list(range(6, 13)), "method_diagrams_copied_from_originals": list(range(1, 6)),
        "dataset_used": False, "models_retrained": False})
    print(f"Reproduced 12 tables and 7 result figures; retained 5 original method diagrams. {table_checks} manuscript numeric cells checked. Output: {output}")


if __name__ == "__main__":
    main()
