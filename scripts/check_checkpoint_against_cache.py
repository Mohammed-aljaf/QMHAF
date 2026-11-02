#!/usr/bin/env python3
"""Optional checkpoint-to-OOF check using a private waveform cache, without XML.

This check is separate from dataset-free release verification. CPU float32 vs
the archived CUDA AMP run can differ slightly; tolerances are reported.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from analysis.metrics import PCOLS, apply_saved_calibration
from deep_learning.model import QMHAFNet, ModelConfig
from scripts.predict import waveform_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    p.add_argument("--samples-per-fold", type=int, default=1)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(2)
    base = ROOT / "deep_learning/results/ahi_a0h3_local_v1/main/all"
    results = []
    for fold in args.folds:
        directory = base / f"fold_{fold}"
        ck = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=False)
        model = QMHAFNet(ModelConfig(**ck["model_config"])).eval(); model.load_state_dict(ck["model"], strict=True)
        predictions = pd.read_parquet(directory / "oof_predictions.parquet").sort_values("subject_id").head(args.samples_per_fold)
        parameters = json.loads((directory / "test_metrics.json").read_text())["calibration_parameters"]
        for row in predictions.itertuples(index=False):
            path = args.cache_dir / "subjects" / f"{row.subject_id}.npz"
            with torch.inference_mode(): out = model(waveform_batch(path))
            raw_ahi = out["predicted_ahi"].numpy(); raw_p = out["class_probabilities"].numpy()
            ahi, probability = apply_saved_calibration(raw_ahi, raw_p, parameters)
            raw_reference = np.array([getattr(row, "raw_" + c) for c in PCOLS])
            reference = np.array([getattr(row, c) for c in PCOLS])
            np.testing.assert_allclose(raw_ahi, [row.raw_predicted_ahi], rtol=0, atol=.03)
            np.testing.assert_allclose(ahi, [row.predicted_ahi], rtol=0, atol=.03)
            np.testing.assert_allclose(raw_p[0], raw_reference, rtol=0, atol=.003)
            np.testing.assert_allclose(probability[0], reference, rtol=0, atol=.003)
            results.append(dict(fold=fold, subject_id=row.subject_id,
                raw_ahi_abs_error=float(abs(raw_ahi[0] - row.raw_predicted_ahi)),
                calibrated_ahi_abs_error=float(abs(ahi[0] - row.predicted_ahi)),
                raw_probability_max_abs_error=float(np.abs(raw_p[0] - raw_reference).max()),
                final_probability_max_abs_error=float(np.abs(probability[0] - reference).max()),
                predicted_class_matches=bool(probability.argmax(1)[0] == row.predicted_class)))
    report = dict(status="passed", mode="CPU float32 versus archived CUDA AMP", private_data_used=True,
        annotations_read=False, ahi_absolute_tolerance=.03, probability_absolute_tolerance=.003, comparisons=results)
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
