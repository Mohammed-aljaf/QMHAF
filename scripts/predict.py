#!/usr/bin/env python3
"""Predict from a private waveform cache or a comparator feature sequence.

Q-MHAF inputs require only resp_q10, resp_quant_scale, resp_valid_1hz,
spo2_1hz, spo2_valid_1hz, hr_1hz and hr_valid_1hz. No annotation file is read.
Comparator inputs use the x array produced by build_inputs.py (N,256,16).
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
from analysis.metrics import CLASSES, apply_saved_calibration
from deep_learning.data import _low_rate_channels
from deep_learning.model import ModelConfig, QMHAFNet
from deep_learning.train_supervised import apply_signal_set


def waveform_batch(path):
    with np.load(path, allow_pickle=False) as z:
        epochs = z["resp_q10"].shape[1] // 300
        if epochs < 1: raise ValueError("At least one complete 30-second epoch is required")
        seconds = epochs * 30
        valid = z["resp_valid_1hz"][:, :seconds]
        resp = z["resp_q10"][:, :epochs * 300].astype(np.float32) / float(z["resp_quant_scale"])
        resp[~np.repeat(valid, 10, axis=1)] = 0
        spo2_valid = z["spo2_valid_1hz"][:seconds]; hr_valid = z["hr_valid_1hz"][:seconds]
        low = _low_rate_channels(z["spo2_1hz"][:seconds], spo2_valid, z["hr_1hz"][:seconds], hr_valid)
        batch = dict(resp=resp.reshape(3, epochs, 300).transpose(1, 0, 2),
            low=low.reshape(5, epochs, 30).transpose(1, 0, 2),
            resp_quality=valid.reshape(3, epochs, 30).mean(2).T.astype(np.float32),
            low_quality=np.stack([spo2_valid.reshape(epochs, 30).mean(1), hr_valid.reshape(epochs, 30).mean(1)], axis=1).astype(np.float32),
            epoch_mask=np.ones(epochs, dtype=bool))
    return {key: torch.from_numpy(np.ascontiguousarray(value))[None] for key, value in batch.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration", type=Path, help="Corresponding test_metrics.json containing saved validation-only calibrators")
    p.add_argument("--signal-set", choices=("all", "no_hr", "spo2_only", "resp_only", "spo2_airflow", "spo2_effort"), default="all")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args(); torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with torch.inference_mode():
        if "architecture" in checkpoint:
            from comparison_2024plus.train_comparators import build_model, predict
            with np.load(args.input, allow_pickle=False) as z: x = z["x"]
            if x.ndim != 3 or x.shape[1:] != (256, 16): raise ValueError("Expected x shape (N,256,16)")
            x = np.nan_to_num(np.clip((x - checkpoint["feature_mean"]) / checkpoint["feature_std"], -8, 8)).astype(np.float32)
            model = build_model(checkpoint["architecture"], x.shape[-1]).to(args.device)
            model.load_state_dict(checkpoint["model"], strict=True)
            probability = predict(model, torch.from_numpy(x), args.device)
            frame = pd.DataFrame({"sample_index": np.arange(len(x)), "predicted_class": probability.argmax(1)})
        else:
            model = QMHAFNet(ModelConfig(**checkpoint["model_config"])).to(args.device).eval()
            model.load_state_dict(checkpoint["model"], strict=True)
            batch = {k: v.to(args.device) for k, v in waveform_batch(args.input).items()}
            signal = args.signal_set
            report = json.loads(args.calibration.read_text()) if args.calibration else None
            if report:
                if checkpoint["test_fold"] != report["test_fold"]: raise ValueError("Checkpoint/calibrator fold mismatch")
                if checkpoint["validation_fold"] != report["validation_fold"] or checkpoint["epoch"] != report["best_epoch"] or checkpoint["validation_metrics"] != report["validation_metrics"]:
                    raise ValueError("Report does not identify this selected checkpoint")
                if checkpoint["model_config"]["quality_routing"] != report["quality_routing"] or checkpoint["model_config"]["factorised_ahi"] != report["auxiliary_factorisation"]:
                    raise ValueError("Checkpoint/calibrator architecture mismatch")
                signal = report["signal_set"]
            apply_signal_set(batch, signal)
            with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.device == "cuda"):
                out = model(batch)
            raw_ahi = out["predicted_ahi"].float().cpu().numpy()
            probability = out["class_probabilities"].float().cpu().numpy()
            ahi, calibrated = apply_saved_calibration(raw_ahi, probability, report["calibration_parameters"] if report else None)
            frame = pd.DataFrame({"sample_index": [0], "raw_predicted_ahi": raw_ahi, "predicted_ahi": ahi,
                "predicted_class": calibrated.argmax(1), "calibrated": bool(report and report["calibration_parameters"] is not None)})
            for i, c in enumerate(CLASSES): frame["raw_probability_" + c] = probability[:, i]
            probability = calibrated
        for i, c in enumerate(CLASSES): frame["probability_" + c] = probability[:, i]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()
