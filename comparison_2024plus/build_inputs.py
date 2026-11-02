"""Build a compact, label-free full-night sequence for controlled architecture comparisons.

The 256 bins preserve the order of the entire recording. Every feature is derived
from physiological waveforms or channel-validity masks, never XML annotations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter


FEATURES = [
    "spo2_mean", "spo2_min", "spo2_std", "spo2_fraction_below_90",
    "spo2_drop_mean", "spo2_drop_max", "hr_mean", "hr_std",
    "airflow_rms", "thor_rms", "abdo_rms",
    "spo2_valid", "hr_valid", "airflow_valid", "thor_valid", "abdo_valid",
]


def make_sequence(path: str, bins: int = 256) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        spo2 = data["spo2_1hz"].astype(np.float32)
        sv = data["spo2_valid_1hz"].astype(bool)
        hr = data["hr_1hz"].astype(np.float32)
        hv = data["hr_valid_1hz"].astype(bool)
        rv = data["resp_valid_1hz"].astype(bool)
        resp = data["resp_q10"].astype(np.float32) / float(data["resp_quant_scale"])
    n = min(len(spo2), len(hr), rv.shape[1], resp.shape[1] // 10)
    spo2, sv, hr, hv, rv = spo2[:n], sv[:n], hr[:n], hv[:n], rv[:, :n]
    resp = resp[:, :n * 10].reshape(3, n, 10)
    resp_rms = np.sqrt(np.mean(resp * resp, axis=2))
    resp_rms[~rv] = np.nan
    good = sv & np.isfinite(spo2)
    if good.sum() >= 2:
        spo2[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), spo2[good])
    else:
        spo2[:] = 94
    spo2 = np.clip(spo2, 50, 100)
    spo2 = median_filter(spo2, size=5, mode="nearest")
    drop = np.clip(maximum_filter1d(spo2, size=120, mode="nearest") - spo2, 0, 30)
    hr[~(hv & np.isfinite(hr))] = np.nan
    out = np.zeros((bins, len(FEATURES)), np.float32)
    edges = np.linspace(0, n, bins + 1).astype(int)
    starts = edges[:-1]
    length = np.diff(edges).astype(np.float32)
    def summed(a: np.ndarray) -> np.ndarray:
        return np.add.reduceat(a, starts, axis=-1)
    def averaged(a: np.ndarray) -> np.ndarray:
        return summed(a) / length
    out[:, 0] = averaged(spo2)
    out[:, 1] = np.minimum.reduceat(spo2, starts)
    out[:, 2] = np.sqrt(np.maximum(0, averaged(spo2 * spo2) - out[:, 0] ** 2))
    out[:, 3] = averaged((spo2 < 90).astype(np.float32))
    out[:, 4] = averaged(drop)
    out[:, 5] = np.maximum.reduceat(drop, starts)
    valid_hr = np.isfinite(hr)
    h = np.nan_to_num(hr)
    counts = summed(valid_hr.astype(np.float32))
    out[:, 6] = np.divide(summed(h), counts, out=np.full(bins, 70, np.float32), where=counts > 0)
    h2 = np.divide(summed(h * h), counts, out=np.zeros(bins, np.float32), where=counts > 0)
    out[:, 7] = np.sqrt(np.maximum(0, h2 - np.where(counts > 0, out[:, 6] ** 2, 0)))
    for c in range(3):
        valid = np.isfinite(resp_rms[c])
        count = summed(valid.astype(np.float32))
        out[:, 8 + c] = np.divide(summed(np.nan_to_num(resp_rms[c])), count,
                                        out=np.zeros(bins, np.float32), where=count > 0)
    out[:, 11] = averaged(sv.astype(np.float32))
    out[:, 12] = averaged(hv.astype(np.float32))
    out[:, 13:16] = (summed(rv.astype(np.float32)) / length).T
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("comparison_2024plus/inputs.npz"))
    parser.add_argument("--bins", type=int, default=256)
    args = parser.parse_args()
    frame = pd.read_parquet("deep_learning/data/ahi_a0h3_local_v1/deep_manifest.parquet")
    frame = frame.loc[frame.qc_pass].sort_values("subject_id").reset_index(drop=True)
    x = np.empty((len(frame), args.bins, len(FEATURES)), np.float32)
    for i, path in enumerate(frame.input_npz):
        x[i] = make_sequence(path, args.bins)
        if (i + 1) % 500 == 0:
            print(f"processed {i+1}/{len(frame)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, subject_id=frame.subject_id.to_numpy(dtype="U16"),
                        fold=frame.fold.to_numpy(np.int8), y=frame.target_class.to_numpy(np.int8),
                        ahi=frame.target_ahi.to_numpy(np.float32), features=np.array(FEATURES))
    print(f"saved {args.output}: {x.shape}", flush=True)


if __name__ == "__main__":
    main()
