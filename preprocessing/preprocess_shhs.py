#!/usr/bin/env python3
"""Build a compact, restartable SHHS1 preprocessing cache.

The cache deliberately keeps model inputs separate from labels:

* Full-night respiratory waveforms are filtered, robustly normalised and
  quantised at 10 Hz (int8, scale 1/16).
* SaO2, heart rate, position and signal-validity masks are kept at 1 Hz.
* EEG/EOG/EMG and low-rate cardiorespiratory summaries are converted to
  30-second epoch features to keep the cache small.
* XML annotations are stored as compact event intervals and are never mixed
  into input features.
* Several explicitly named AHI variants are recorded.  The official clinical
  ahi_a0h3 is preferred when a SHHS dataset CSV is supplied; otherwise the
  XML/raw-signal reconstructions remain clearly marked as proxies.

The script reads EDF records directly so that 1 Hz channels are not silently
upsampled to the maximum EDF sampling frequency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import signal


CACHE_VERSION = "1.0.0"
RESP_HZ = 10
LOW_HZ = 1
EPOCH_SECONDS = 30
RESP_QUANT_SCALE = 16.0

AIRFLOW_ALIASES = (
    "NEW AIR",
    "New Air",
    "NEWAIR",
    "New A/F",
    "New AIR",
    "new air",
    "NASAL",
    "AIRFLOW",
)
EEG_PRIMARY_ALIASES = ("EEG",)
EEG_SECONDARY_ALIASES = ("EEG(sec)", "EEG sec", "EEG(SEC)", "EEG 2", "EEG2")

EVENT_CODES = {
    "Obstructive apnea": 1,
    "Central apnea": 2,
    "Mixed apnea": 3,
    "Hypopnea": 4,
    "Unsure": 5,
}

# event_flags bit layout
FLAG_DURATION = 1
FLAG_SLEEP = 2
FLAG_XML_DESAT3 = 4
FLAG_RAW_DESAT3 = 8
FLAG_AROUSAL = 16


def _decode_ascii(raw: bytes) -> str:
    return raw.decode("latin1", "ignore").strip()


@dataclass(frozen=True)
class SignalHeader:
    label: str
    unit: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    samples_per_record: int


class EDFReader:
    """Small EDF reader supporting efficient native-rate channel access."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            fixed = handle.read(256)
            if len(fixed) != 256:
                raise ValueError("short EDF fixed header")
            self.header_bytes = int(_decode_ascii(fixed[184:192]))
            self.n_records = int(_decode_ascii(fixed[236:244]))
            self.record_duration = float(_decode_ascii(fixed[244:252]))
            self.n_signals = int(_decode_ascii(fixed[252:256]))
            variable = handle.read(256 * self.n_signals)
        if len(variable) != 256 * self.n_signals:
            raise ValueError("short EDF signal header")

        offset = 0

        def fields(width: int) -> list[str]:
            nonlocal offset
            values = [
                _decode_ascii(variable[offset + i * width : offset + (i + 1) * width])
                for i in range(self.n_signals)
            ]
            offset += width * self.n_signals
            return values

        labels = fields(16)
        fields(80)  # transducer
        units = fields(8)
        pmin = [float(x) for x in fields(8)]
        pmax = [float(x) for x in fields(8)]
        dmin = [float(x) for x in fields(8)]
        dmax = [float(x) for x in fields(8)]
        fields(80)  # prefiltering
        samples = [int(x) for x in fields(8)]
        fields(32)  # reserved

        self.signals = [
            SignalHeader(labels[i], units[i], pmin[i], pmax[i], dmin[i], dmax[i], samples[i])
            for i in range(self.n_signals)
        ]
        self._starts = np.r_[0, np.cumsum(samples[:-1])].astype(np.int64)
        self._samples_per_record = int(sum(samples))
        bytes_available = path.stat().st_size - self.header_bytes
        inferred_records = bytes_available // (2 * self._samples_per_record)
        if self.n_records < 0:
            self.n_records = int(inferred_records)
        if inferred_records < self.n_records:
            raise ValueError(f"truncated EDF: header={self.n_records}, available={inferred_records}")
        self.duration_seconds = self.n_records * self.record_duration
        self._data = np.memmap(
            path,
            dtype="<i2",
            mode="r",
            offset=self.header_bytes,
            shape=(self.n_records, self._samples_per_record),
        )

    @property
    def labels(self) -> list[str]:
        return [x.label for x in self.signals]

    def indices(self, aliases: Sequence[str]) -> list[int]:
        return [i for i, s in enumerate(self.signals) if s.label in aliases]

    def first_index(self, aliases: Sequence[str]) -> int | None:
        ids = self.indices(aliases)
        return ids[0] if ids else None

    def read(self, index: int) -> tuple[np.ndarray, np.ndarray, float]:
        h = self.signals[index]
        start = int(self._starts[index])
        digital = np.asarray(
            self._data[:, start : start + h.samples_per_record], dtype=np.int16
        ).reshape(-1)
        denominator = h.digital_max - h.digital_min
        if denominator == 0:
            physical = np.zeros(digital.size, dtype=np.float32)
        else:
            physical = (
                (digital.astype(np.float32) - h.digital_min)
                * (h.physical_max - h.physical_min)
                / denominator
                + h.physical_min
            ).astype(np.float32)
        clipped = (digital <= h.digital_min) | (digital >= h.digital_max)
        sampling_rate = h.samples_per_record / self.record_duration
        return physical, clipped, float(sampling_rate)


def resample_exact(x: np.ndarray, source_hz: float, target_hz: int, target_n: int) -> np.ndarray:
    if x.size == 0:
        return np.full(target_n, np.nan, dtype=np.float32)
    if math.isclose(source_hz, target_hz):
        out = np.asarray(x, dtype=np.float32)
    else:
        ratio = Fraction(target_hz / source_hz).limit_denominator(1000)
        out = signal.resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)
    if out.size < target_n:
        out = np.pad(out, (0, target_n - out.size), constant_values=np.nan)
    return out[:target_n]


def resample_mask(mask: np.ndarray, source_hz: float, target_hz: int, target_n: int) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(target_n, dtype=bool)
    source_t = (np.arange(mask.size, dtype=np.float64) + 0.5) / source_hz
    target_t = (np.arange(target_n, dtype=np.float64) + 0.5) / target_hz
    ids = np.clip(np.searchsorted(source_t, target_t), 0, mask.size - 1)
    return np.asarray(mask[ids], dtype=bool)


def interpolate_short_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    out[~valid] = np.nan
    good = np.flatnonzero(np.isfinite(out))
    if good.size == 0:
        return out
    starts = np.flatnonzero(np.diff(np.r_[True, np.isfinite(out), True].astype(np.int8)) == -1)
    ends = np.flatnonzero(np.diff(np.r_[True, np.isfinite(out), True].astype(np.int8)) == 1)
    for start, end in zip(starts, ends):
        if end - start <= max_gap and start > 0 and end < out.size:
            out[start:end] = np.linspace(out[start - 1], out[end], end - start + 2)[1:-1]
    return out


def fill_for_filtering(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    good = np.flatnonzero(np.isfinite(out))
    if good.size == 0:
        return np.zeros_like(out)
    bad = np.flatnonzero(~np.isfinite(out))
    if bad.size:
        out[bad] = np.interp(bad, good, out[good])
    return out


def robust_filter_quantise(
    x: np.ndarray,
    clipped: np.ndarray,
    source_hz: float,
    duration_seconds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    target_n = duration_seconds * RESP_HZ
    xr = resample_exact(x, source_hz, RESP_HZ, target_n)
    cr = resample_mask(clipped, source_hz, RESP_HZ, target_n)
    finite = np.isfinite(xr)
    base_valid = finite & ~cr
    xf = fill_for_filtering(np.where(base_valid, xr, np.nan))
    if xf.size >= 64:
        sos = signal.butter(4, (0.05, 2.0), btype="bandpass", fs=RESP_HZ, output="sos")
        try:
            xf = signal.sosfiltfilt(sos, xf).astype(np.float32)
        except ValueError:
            xf = signal.sosfilt(sos, xf).astype(np.float32)
    centre = float(np.median(xf[base_valid])) if np.any(base_valid) else 0.0
    mad = float(np.median(np.abs(xf[base_valid] - centre))) if np.any(base_valid) else 0.0
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(xf[base_valid])) if np.any(base_valid) else 1.0
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    z = np.clip((xf - centre) / scale, -7.9375, 7.9375)
    quantised = np.rint(z * RESP_QUANT_SCALE).astype(np.int8)

    usable = duration_seconds * RESP_HZ
    sec = xr[:usable].reshape(duration_seconds, RESP_HZ)
    sec_valid = base_valid[:usable].reshape(duration_seconds, RESP_HZ)
    valid_fraction = sec_valid.mean(axis=1)
    spread = np.ptp(np.where(sec_valid, sec, np.nan), axis=1)
    global_span = float(np.nanpercentile(xr, 99) - np.nanpercentile(xr, 1))
    spread_threshold = max(abs(global_span) * 1e-5, 1e-12)
    valid_1hz = (valid_fraction >= 0.8) & np.isfinite(spread) & (spread > spread_threshold)
    stats = {
        "clip_fraction": float(np.mean(cr)),
        "valid_fraction": float(np.mean(valid_1hz)),
        "flat_fraction": float(np.mean(~np.isfinite(spread) | (spread <= spread_threshold))),
        "robust_scale": scale,
    }
    return quantised, valid_1hz, stats


def choose_airflow(reader: EDFReader, duration_seconds: int):
    candidates = []
    for idx in reader.indices(AIRFLOW_ALIASES):
        x, clipped, sf = reader.read(idx)
        q, valid, stats = robust_filter_quantise(x, clipped, sf, duration_seconds)
        label = reader.signals[idx].label
        # Valid coverage is dominant.  Clipping/flatness break ties, and a tiny
        # preference favours nasal-pressure aliases over legacy AIRFLOW.
        score = (
            stats["valid_fraction"]
            - 0.6 * stats["clip_fraction"]
            - 0.4 * stats["flat_fraction"]
            + (0.005 if label != "AIRFLOW" else 0.0)
        )
        candidates.append((score, label, q, valid, stats))
    if not candidates:
        return (
            "MISSING",
            np.zeros(duration_seconds * RESP_HZ, dtype=np.int8),
            np.zeros(duration_seconds, dtype=bool),
            {"clip_fraction": 1.0, "valid_fraction": 0.0, "flat_fraction": 1.0, "robust_scale": 0.0},
        )
    _, label, q, valid, stats = max(candidates, key=lambda item: item[0])
    return label, q, valid, stats


def read_lowrate(reader: EDFReader, aliases: Sequence[str], duration_seconds: int):
    idx = reader.first_index(aliases)
    if idx is None:
        return np.full(duration_seconds, np.nan, np.float32), np.zeros(duration_seconds, bool)
    x, clipped, sf = reader.read(idx)
    target = resample_exact(x, sf, LOW_HZ, duration_seconds)
    clip = resample_mask(clipped, sf, LOW_HZ, duration_seconds)
    return target, ~clip & np.isfinite(target)


def interval_overlap(start: float, duration: float, intervals: Sequence[tuple[float, float]]) -> bool:
    end = start + duration
    return any(a < end and b > start for a, b in intervals)


def parse_annotations(xml_path: Path, record_seconds: int) -> dict:
    root = ET.parse(xml_path).getroot()
    n_epochs = int(math.ceil(record_seconds / EPOCH_SECONDS))
    stages = np.full(n_epochs, 9, dtype=np.uint8)
    respiratory: list[dict] = []
    arousals: list[tuple[float, float]] = []
    artifacts: list[tuple[float, float]] = []
    desaturations: list[dict] = []

    stage_map = {
        "Wake": 0,
        "Stage 1 sleep": 1,
        "Stage 2 sleep": 2,
        "Stage 3 sleep": 3,
        "Stage 4 sleep": 3,
        "REM sleep": 4,
        "Unscored": 9,
        "Movement": 9,
    }
    for event in root.findall(".//ScoredEvent"):
        concept = (event.findtext("EventConcept") or "").split("|")[0]
        try:
            start = float(event.findtext("Start") or 0.0)
            duration = float(event.findtext("Duration") or 0.0)
        except ValueError:
            continue
        if concept in stage_map:
            first = max(0, int(math.floor(start / EPOCH_SECONDS)))
            last = min(n_epochs, int(math.ceil((start + duration) / EPOCH_SECONDS)))
            stages[first:last] = stage_map[concept]
        elif concept in EVENT_CODES:
            respiratory.append({"start": start, "duration": duration, "type": EVENT_CODES[concept]})
        elif "arousal" in concept.lower():
            arousals.append((start, start + duration))
        elif concept == "SpO2 artifact":
            artifacts.append((start, start + duration))
        elif concept == "SpO2 desaturation":
            try:
                baseline = float(event.findtext("SpO2Baseline"))
                nadir = float(event.findtext("SpO2Nadir"))
            except (TypeError, ValueError):
                baseline = np.nan
                nadir = np.nan
            desaturations.append(
                {"start": start, "duration": duration, "baseline": baseline, "nadir": nadir}
            )

    sleep_seconds = float(np.sum(np.isin(stages, [1, 2, 3, 4])) * EPOCH_SECONDS)
    return {
        "stages": stages,
        "respiratory": respiratory,
        "arousals": arousals,
        "artifacts": artifacts,
        "desaturations": desaturations,
        "sleep_seconds": sleep_seconds,
    }


def associate_events(
    annotations: dict,
    spo2: np.ndarray,
    spo2_valid: np.ndarray,
) -> dict:
    respiratory = annotations["respiratory"]
    desaturations = annotations["desaturations"]
    arousals = annotations["arousals"]
    stages = annotations["stages"]
    sleep_seconds = annotations["sleep_seconds"]

    desat_by_start: dict[float, list[dict]] = {}
    for item in desaturations:
        desat_by_start.setdefault(round(item["start"], 1), []).append(item)

    starts_ds: list[int] = []
    durations_ds: list[int] = []
    types: list[int] = []
    flags: list[int] = []
    count_xml3 = 0
    count_raw3 = 0
    count_raw3a = 0

    for event in respiratory:
        start = event["start"]
        duration = event["duration"]
        event_type = event["type"]
        flag = 0
        if duration >= 10.0:
            flag |= FLAG_DURATION
        midpoint_epoch = min(max(int((start + duration / 2) // EPOCH_SECONDS), 0), stages.size - 1)
        if stages.size and stages[midpoint_epoch] in (1, 2, 3, 4):
            flag |= FLAG_SLEEP

        associated = desat_by_start.get(round(start + 30.0, 1), [])
        if associated:
            best = min(associated, key=lambda d: abs(d["duration"] - duration))
            depth = best["baseline"] - best["nadir"]
            if np.isfinite(depth) and depth >= 3.0:
                flag |= FLAG_XML_DESAT3

        end_second = int(round(start + duration))
        before = max(0, end_second - 60)
        after = min(spo2.size, end_second + 30)
        pre_values = spo2[before:end_second]
        pre_valid = spo2_valid[before:end_second]
        post_values = spo2[end_second:after]
        post_valid = spo2_valid[end_second:after]
        if np.count_nonzero(pre_valid) >= 10 and np.count_nonzero(post_valid) >= 5:
            baseline = float(np.nanmax(pre_values[pre_valid]))
            nadir = float(np.nanmin(post_values[post_valid]))
            if baseline - nadir >= 3.0:
                flag |= FLAG_RAW_DESAT3

        if any(a <= start + duration + 5.0 and b >= start + duration for a, b in arousals):
            flag |= FLAG_AROUSAL

        core_valid = (flag & (FLAG_DURATION | FLAG_SLEEP)) == (FLAG_DURATION | FLAG_SLEEP)
        if event_type in (1, 2, 3):
            count_xml3 += int(core_valid)
            count_raw3 += int(core_valid)
            count_raw3a += int(core_valid)
        elif event_type in (4, 5):
            count_xml3 += int(core_valid and bool(flag & FLAG_XML_DESAT3))
            count_raw3 += int(core_valid and bool(flag & FLAG_RAW_DESAT3))
            count_raw3a += int(core_valid and bool(flag & (FLAG_RAW_DESAT3 | FLAG_AROUSAL)))

        starts_ds.append(int(round(start * 10)))
        durations_ds.append(max(0, int(round(duration * 10))))
        types.append(event_type)
        flags.append(flag)

    hours = sleep_seconds / 3600.0

    def rate(count: int) -> float:
        return float(count / hours) if hours > 0 else float("nan")

    return {
        "event_start_ds": np.asarray(starts_ds, dtype=np.int32),
        "event_duration_ds": np.asarray(durations_ds, dtype=np.uint16),
        "event_type": np.asarray(types, dtype=np.uint8),
        "event_flags": np.asarray(flags, dtype=np.uint8),
        "ahi_xml_desat3_proxy": rate(count_xml3),
        "ahi_raw_desat3_proxy": rate(count_raw3),
        "ahi_raw_desat3_or_arousal_proxy": rate(count_raw3a),
        "events_xml_desat3": count_xml3,
        "events_raw_desat3": count_raw3,
        "events_raw_desat3_or_arousal": count_raw3a,
    }


def severity(ahi: float) -> int:
    if not np.isfinite(ahi):
        return -1
    if ahi < 5:
        return 0
    if ahi < 15:
        return 1
    if ahi < 30:
        return 2
    return 3


def epochise(x: np.ndarray, samples_per_epoch: int, n_epochs: int, fill=np.nan) -> np.ndarray:
    target = n_epochs * samples_per_epoch
    if x.size < target:
        x = np.pad(x, (0, target - x.size), constant_values=fill)
    return x[:target].reshape(n_epochs, samples_per_epoch)


def nan_slope(blocks: np.ndarray) -> np.ndarray:
    t = np.arange(blocks.shape[1], dtype=np.float32)
    t = t - t.mean()
    out = np.full(blocks.shape[0], np.nan, np.float32)
    for i, row in enumerate(blocks):
        good = np.isfinite(row)
        if good.sum() >= 3:
            tg = t[good]
            out[i] = np.sum(tg * (row[good] - np.mean(row[good]))) / max(np.sum(tg * tg), 1e-12)
    return out


def lowrate_epoch_features(
    spo2: np.ndarray,
    spo2_valid: np.ndarray,
    hr: np.ndarray,
    hr_valid: np.ndarray,
    position: np.ndarray,
    n_epochs: int,
) -> tuple[np.ndarray, list[str]]:
    s = epochise(np.where(spo2_valid, spo2, np.nan), EPOCH_SECONDS, n_epochs)
    h = epochise(np.where(hr_valid, hr, np.nan), EPOCH_SECONDS, n_epochs)
    p = epochise(position.astype(np.float32), EPOCH_SECONDS, n_epochs, fill=-1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        feats = [
            np.nanmean(s, axis=1),
            np.nanmin(s, axis=1),
            np.nanstd(s, axis=1),
            np.nanmax(s, axis=1) - np.nanmin(s, axis=1),
            np.mean(np.isfinite(s), axis=1),
            np.nanmean(s < 90, axis=1),
            nan_slope(s),
            np.nanmean(h, axis=1),
            np.nanstd(h, axis=1),
            np.nanmax(h, axis=1) - np.nanmin(h, axis=1),
            np.mean(np.isfinite(h), axis=1),
            np.sqrt(np.nanmean(np.diff(h, axis=1) ** 2, axis=1)),
        ]
    names = [
        "spo2_mean",
        "spo2_min",
        "spo2_std",
        "spo2_range",
        "spo2_valid_fraction",
        "spo2_fraction_below_90",
        "spo2_slope",
        "hr_mean",
        "hr_std",
        "hr_range",
        "hr_valid_fraction",
        "hr_rmssd_1s",
    ]
    for code in range(4):
        feats.append(np.mean(p == code, axis=1))
        names.append(f"position_{code}_fraction")
    return np.column_stack(feats).astype(np.float32), names


def respiratory_epoch_features(
    resp_q: np.ndarray,
    resp_valid_1hz: np.ndarray,
    n_epochs: int,
) -> tuple[np.ndarray, list[str]]:
    x = resp_q.astype(np.float32) / RESP_QUANT_SCALE
    features: list[np.ndarray] = []
    names: list[str] = []
    window = np.hanning(EPOCH_SECONDS * RESP_HZ).astype(np.float32)
    freqs = np.fft.rfftfreq(window.size, 1 / RESP_HZ)
    for channel_index, channel_name in enumerate(("airflow", "thor", "abdo")):
        blocks = epochise(x[channel_index], window.size, n_epochs, fill=0.0)
        centred = blocks - np.mean(blocks, axis=1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centred * window, axis=1)) ** 2
        components = [
            np.std(blocks, axis=1),
            np.sqrt(np.mean(blocks * blocks, axis=1)),
            np.ptp(blocks, axis=1),
            np.mean(np.abs(np.diff(blocks, axis=1)), axis=1),
        ]
        component_names = ["std", "rms", "range", "line_length"]
        for lo, hi, band_name in ((0.05, 0.2, "very_low"), (0.2, 0.4, "resp"), (0.4, 0.8, "high_resp"), (0.8, 2.0, "high")):
            mask = (freqs >= lo) & (freqs < hi)
            components.append(np.log1p(np.sum(spectrum[:, mask], axis=1)))
            component_names.append(f"logpower_{band_name}")
        valid_blocks = epochise(resp_valid_1hz[channel_index], EPOCH_SECONDS, n_epochs, fill=False)
        components.append(np.mean(valid_blocks, axis=1))
        component_names.append("valid_fraction")
        features.extend(components)
        names.extend([f"{channel_name}_{name}" for name in component_names])

    blocks = [epochise(x[i], EPOCH_SECONDS * RESP_HZ, n_epochs, fill=0.0) for i in range(3)]
    for a, b, name in ((0, 1, "airflow_thor_corr"), (0, 2, "airflow_abdo_corr"), (1, 2, "thor_abdo_corr")):
        aa = blocks[a] - np.mean(blocks[a], axis=1, keepdims=True)
        bb = blocks[b] - np.mean(blocks[b], axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1))
        features.append(np.divide(np.sum(aa * bb, axis=1), denominator, out=np.zeros(n_epochs), where=denominator > 0))
        names.append(name)
    return np.column_stack(features).astype(np.float32), names


def spectral_epoch_features(
    reader: EDFReader,
    aliases: Sequence[str],
    prefix: str,
    bands: Sequence[tuple[float, float, str]],
    n_epochs: int,
) -> tuple[np.ndarray, list[str]]:
    names = [f"{prefix}_log_total", f"{prefix}_spectral_entropy", f"{prefix}_rms", f"{prefix}_line_length"]
    names += [f"{prefix}_relpower_{name}" for _, _, name in bands]
    names += [f"{prefix}_valid_fraction"]
    idx = reader.first_index(aliases)
    if idx is None:
        return np.full((n_epochs, len(names)), np.nan, np.float32), names
    x, clipped, sf = reader.read(idx)
    samples_per_epoch = int(round(sf * EPOCH_SECONDS))
    blocks = epochise(x, samples_per_epoch, n_epochs)
    clip_blocks = epochise(clipped, samples_per_epoch, n_epochs, fill=True)
    finite = np.isfinite(blocks) & ~clip_blocks
    valid_fraction = np.mean(finite, axis=1)
    blocks = np.where(finite, blocks, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(blocks, axis=1, keepdims=True)
    blocks = np.nan_to_num(blocks - means, nan=0.0)
    scale = np.nanmedian(np.abs(blocks), axis=1, keepdims=True) * 1.4826
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    blocks = np.clip(blocks / scale, -20, 20)
    window = np.hanning(samples_per_epoch).astype(np.float32)
    power = np.abs(np.fft.rfft(blocks * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(samples_per_epoch, 1 / sf)
    allowed = (freqs >= bands[0][0]) & (freqs < bands[-1][1])
    total = np.sum(power[:, allowed], axis=1)
    distribution = power[:, allowed] / np.maximum(total[:, None], 1e-12)
    entropy = -np.sum(distribution * np.log(np.maximum(distribution, 1e-12)), axis=1)
    entropy /= max(math.log(distribution.shape[1]), 1.0)
    columns = [
        np.log1p(total),
        entropy,
        np.sqrt(np.mean(blocks * blocks, axis=1)),
        np.mean(np.abs(np.diff(blocks, axis=1)), axis=1),
    ]
    for lo, hi, _ in bands:
        mask = (freqs >= lo) & (freqs < hi)
        columns.append(np.sum(power[:, mask], axis=1) / np.maximum(total, 1e-12))
    columns.append(valid_fraction)
    return np.column_stack(columns).astype(np.float32), names


def annotation_arrays(annotations: dict) -> dict[str, np.ndarray]:
    arousals = annotations["arousals"]
    artifacts = annotations["artifacts"]
    desats = annotations["desaturations"]
    return {
        "arousal_start_ds": np.asarray([round(a * 10) for a, _ in arousals], dtype=np.int32),
        "arousal_duration_ds": np.asarray([round((b - a) * 10) for a, b in arousals], dtype=np.uint16),
        "spo2_artifact_start_ds": np.asarray([round(a * 10) for a, _ in artifacts], dtype=np.int32),
        "spo2_artifact_duration_ds": np.asarray([round((b - a) * 10) for a, b in artifacts], dtype=np.uint16),
        "desat_start_ds": np.asarray([round(x["start"] * 10) for x in desats], dtype=np.int32),
        "desat_duration_ds": np.asarray([round(x["duration"] * 10) for x in desats], dtype=np.uint16),
        "desat_baseline": np.asarray([x["baseline"] for x in desats], dtype=np.float16),
        "desat_nadir": np.asarray([x["nadir"] for x in desats], dtype=np.float16),
    }


def mark_intervals_invalid(valid: np.ndarray, intervals: Iterable[tuple[float, float]]) -> None:
    for start, end in intervals:
        a = max(0, int(math.floor(start)))
        b = min(valid.size, int(math.ceil(end)))
        valid[a:b] = False


def load_clinical_labels(csv_path: Path | None) -> dict[str, float]:
    if csv_path is None:
        return {}
    frame = pd.read_csv(csv_path, low_memory=False)
    lower = {str(c).lower(): c for c in frame.columns}
    ahi_col = lower.get("ahi_a0h3")
    if ahi_col is None:
        raise ValueError(f"ahi_a0h3 not found in {csv_path}")
    id_col = next((lower[x] for x in ("nsrrid", "id", "pptid") if x in lower), None)
    if id_col is None:
        raise ValueError(f"subject id column not found in {csv_path}")
    labels = {}
    for sid, ahi in zip(frame[id_col], frame[ahi_col]):
        if pd.isna(sid) or pd.isna(ahi):
            continue
        text = str(sid).strip()
        if text.isdigit():
            text = f"shhs1-{int(text):06d}"
        elif not text.startswith("shhs1-"):
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                text = f"shhs1-{int(digits):06d}"
        labels[text] = float(ahi)
    return labels


def process_subject(
    edf_path_str: str,
    xml_path_str: str,
    output_path_str: str,
    clinical_ahi: float | None,
    force: bool,
) -> dict:
    edf_path = Path(edf_path_str)
    xml_path = Path(xml_path_str)
    output_path = Path(output_path_str)
    subject_id = edf_path.stem
    if output_path.exists() and not force:
        try:
            with np.load(output_path, allow_pickle=False) as cached:
                if str(cached["cache_version"].item()) == CACHE_VERSION:
                    return json.loads(str(cached["manifest_json"].item()))
        except Exception:
            pass

    started = time.monotonic()
    reader = EDFReader(edf_path)
    duration_seconds = int(math.floor(reader.duration_seconds))
    annotations = parse_annotations(xml_path, duration_seconds)

    spo2, spo2_base_valid = read_lowrate(reader, ("SaO2",), duration_seconds)
    hr, hr_base_valid = read_lowrate(reader, ("H.R.",), duration_seconds)
    ox, ox_present_valid = read_lowrate(reader, ("OX stat",), duration_seconds)
    position, position_valid = read_lowrate(reader, ("POSITION",), duration_seconds)
    ox_present = np.any(ox_present_valid)
    ox_good = np.ones(duration_seconds, dtype=bool) if not ox_present else (np.rint(ox) == 0)
    spo2_valid = spo2_base_valid & (spo2 >= 50) & (spo2 <= 100) & ox_good
    hr_valid = hr_base_valid & (hr >= 20) & (hr <= 250) & ox_good
    mark_intervals_invalid(spo2_valid, annotations["artifacts"])
    spo2_clean = interpolate_short_gaps(spo2, spo2_valid, max_gap=30)
    hr_clean = interpolate_short_gaps(hr, hr_valid, max_gap=10)
    position_clean = np.where(position_valid, np.rint(position), -1).astype(np.int8)
    ox_status = np.where(ox_present_valid, np.rint(ox), -1).astype(np.int8)

    airflow_name, airflow_q, airflow_valid, airflow_stats = choose_airflow(reader, duration_seconds)
    resp_q = [airflow_q]
    resp_valid = [airflow_valid]
    resp_stats = {"airflow": airflow_stats}
    for key, aliases in (("thor", ("THOR RES",)), ("abdo", ("ABDO RES",))):
        idx = reader.first_index(aliases)
        if idx is None:
            q = np.zeros(duration_seconds * RESP_HZ, dtype=np.int8)
            valid = np.zeros(duration_seconds, dtype=bool)
            stats = {"clip_fraction": 1.0, "valid_fraction": 0.0, "flat_fraction": 1.0, "robust_scale": 0.0}
        else:
            x, clipped, sf = reader.read(idx)
            q, valid, stats = robust_filter_quantise(x, clipped, sf, duration_seconds)
        resp_q.append(q)
        resp_valid.append(valid)
        resp_stats[key] = stats
    resp_q_array = np.stack(resp_q)
    resp_valid_array = np.stack(resp_valid)

    event_data = associate_events(annotations, spo2_clean, spo2_valid)
    n_epochs = annotations["stages"].size
    resp_features, resp_names = respiratory_epoch_features(resp_q_array, resp_valid_array, n_epochs)
    low_features, low_names = lowrate_epoch_features(
        spo2_clean, spo2_valid, hr_clean, hr_valid, position_clean, n_epochs
    )
    feature_parts = [resp_features, low_features]
    feature_names = resp_names + low_names

    spectral_specs = [
        (EEG_PRIMARY_ALIASES, "eeg_primary", ((0.5, 4, "delta"), (4, 8, "theta"), (8, 12, "alpha"), (12, 16, "sigma"), (16, 30, "beta"))),
        (EEG_SECONDARY_ALIASES, "eeg_secondary", ((0.5, 4, "delta"), (4, 8, "theta"), (8, 12, "alpha"), (12, 16, "sigma"), (16, 30, "beta"))),
        (("EOG(L)",), "eog_left", ((0.3, 1, "slow"), (1, 4, "delta"), (4, 8, "theta"), (8, 15, "fast"))),
        (("EOG(R)",), "eog_right", ((0.3, 1, "slow"), (1, 4, "delta"), (4, 8, "theta"), (8, 15, "fast"))),
        (("EMG",), "emg", ((5, 10, "low"), (10, 20, "mid"), (20, 45, "high"))),
        (("ECG",), "ecg", ((0.5, 5, "low"), (5, 15, "qrs"), (15, 40, "high"))),
    ]
    for aliases, prefix, bands in spectral_specs:
        part, names = spectral_epoch_features(reader, aliases, prefix, bands, n_epochs)
        feature_parts.append(part)
        feature_names.extend(names)
    epoch_features = np.column_stack(feature_parts).astype(np.float16)

    official_ahi = float(clinical_ahi) if clinical_ahi is not None else float("nan")
    if np.isfinite(official_ahi):
        selected_ahi = official_ahi
        label_source = "clinical_ahi_a0h3"
    else:
        selected_ahi = event_data["ahi_xml_desat3_proxy"]
        label_source = "xml_desat3_proxy"

    manifest = {
        "subject_id": subject_id,
        "cache_version": CACHE_VERSION,
        "record_seconds": duration_seconds,
        "sleep_seconds": annotations["sleep_seconds"],
        "n_epochs": n_epochs,
        "n_respiratory_events_raw": len(annotations["respiratory"]),
        "n_arousals": len(annotations["arousals"]),
        "n_spo2_artifacts": len(annotations["artifacts"]),
        "n_desaturations": len(annotations["desaturations"]),
        "airflow_channel": airflow_name,
        "airflow_valid_fraction": airflow_stats["valid_fraction"],
        "airflow_clip_fraction": airflow_stats["clip_fraction"],
        "thor_valid_fraction": resp_stats["thor"]["valid_fraction"],
        "thor_clip_fraction": resp_stats["thor"]["clip_fraction"],
        "abdo_valid_fraction": resp_stats["abdo"]["valid_fraction"],
        "abdo_clip_fraction": resp_stats["abdo"]["clip_fraction"],
        "spo2_valid_fraction": float(np.mean(spo2_valid)),
        "hr_valid_fraction": float(np.mean(hr_valid)),
        "ox_status_present": bool(ox_present),
        "ahi_clinical_a0h3": official_ahi,
        "ahi_xml_desat3_proxy": event_data["ahi_xml_desat3_proxy"],
        "ahi_raw_desat3_proxy": event_data["ahi_raw_desat3_proxy"],
        "ahi_raw_desat3_or_arousal_proxy": event_data["ahi_raw_desat3_or_arousal_proxy"],
        "label_ahi": selected_ahi,
        "label_severity": severity(selected_ahi),
        "label_source": label_source,
        "processing_seconds": 0.0,
        "cache_bytes": 0,
        "status": "ok",
    }

    arrays = {
        "cache_version": np.asarray(CACHE_VERSION),
        "subject_id": np.asarray(subject_id),
        "resp_q10": resp_q_array,
        "resp_valid_1hz": resp_valid_array,
        "resp_quant_scale": np.asarray(RESP_QUANT_SCALE, dtype=np.float32),
        "spo2_1hz": spo2_clean.astype(np.float16),
        "spo2_valid_1hz": spo2_valid,
        "hr_1hz": hr_clean.astype(np.float16),
        "hr_valid_1hz": hr_valid,
        "position_1hz": position_clean,
        "ox_status_1hz": ox_status,
        "sleep_stage_30s": annotations["stages"],
        "epoch_features_30s": epoch_features,
        "feature_names": np.asarray(feature_names),
        "airflow_channel": np.asarray(airflow_name),
        **annotation_arrays(annotations),
        **{key: value for key, value in event_data.items() if isinstance(value, np.ndarray)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + f".{os.getpid()}.tmp.npz")
    manifest["processing_seconds"] = time.monotonic() - started
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    np.savez_compressed(temp_path, **arrays)
    os.replace(temp_path, output_path)
    manifest["cache_bytes"] = output_path.stat().st_size
    # Rewrite only the tiny manifest scalar so resumed runs report final bytes.
    # Avoid a second full archive write; the authoritative tables get the size.
    return manifest


def write_schema(output_dir: Path, feature_names: Sequence[str] | None = None) -> None:
    schema = {
        "cache_version": CACHE_VERSION,
        "resp_q10": {
            "shape": "[3, record_seconds*10]",
            "channels": ["selected_airflow", "thor", "abdo"],
            "dtype": "int8",
            "decode": f"float_value = int8_value / {RESP_QUANT_SCALE}",
            "filter_hz": [0.05, 2.0],
            "normalisation": "per-record median/MAD",
        },
        "low_rate_arrays_hz": 1,
        "epoch_seconds": EPOCH_SECONDS,
        "sleep_stage_codes": {"0": "Wake", "1": "N1", "2": "N2", "3": "N3/N4", "4": "REM", "9": "Unknown"},
        "event_type_codes": {"1": "Obstructive apnea", "2": "Central apnea", "3": "Mixed apnea", "4": "Hypopnea", "5": "Unsure"},
        "event_flag_bits": {
            "1": "duration >= 10 seconds",
            "2": "event midpoint is in sleep",
            "4": "paired XML desaturation >= 3%",
            "8": "raw SaO2 desaturation >= 3%",
            "16": "arousal near event termination",
        },
        "ahi_fields": {
            "ahi_clinical_a0h3": "official SHHS dataset value when supplied",
            "ahi_xml_desat3_proxy": "all sleep apneas + hypopneas paired to XML desaturation >=3% per sleep hour",
            "ahi_raw_desat3_proxy": "all sleep apneas + hypopneas with raw SaO2 drop >=3% per sleep hour",
            "ahi_raw_desat3_or_arousal_proxy": "AASM-style 3% desaturation or arousal proxy",
        },
        "feature_names": list(feature_names or []),
        "leakage_rule": "annotation/event arrays are targets only and must never enter model inputs",
    }
    (output_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


def deterministic_proxy_folds(labels: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    rows = []
    for severity_value, group in labels.groupby("label_severity"):
        ordered = sorted(
            group["subject_id"].tolist(),
            key=lambda sid: hashlib.sha256(f"shhs-v1:{sid}".encode()).hexdigest(),
        )
        for index, sid in enumerate(ordered):
            rows.append({"subject_id": sid, "fold": index % folds, "stratify_label": int(severity_value)})
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def build_qc_summary(manifest: pd.DataFrame) -> dict:
    ok = manifest[manifest["status"] == "ok"].copy()

    def quantiles(column: str) -> dict:
        values = pd.to_numeric(ok[column], errors="coerce").dropna()
        return {str(q): float(values.quantile(q)) for q in (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1)}

    summary = {
        "cache_version": CACHE_VERSION,
        "subjects_total": int(len(manifest)),
        "subjects_ok": int(len(ok)),
        "subjects_failed": int((manifest["status"] != "ok").sum()),
        "total_cache_bytes": int(ok["cache_bytes"].sum()),
        "airflow_channels": ok["airflow_channel"].value_counts(dropna=False).to_dict(),
        "label_sources": ok["label_source"].value_counts(dropna=False).to_dict(),
        "severity_distribution": {str(k): int(v) for k, v in ok["label_severity"].value_counts().sort_index().items()},
        "quantiles": {
            key: quantiles(key)
            for key in (
                "record_seconds",
                "sleep_seconds",
                "spo2_valid_fraction",
                "hr_valid_fraction",
                "airflow_valid_fraction",
                "thor_valid_fraction",
                "abdo_valid_fraction",
                "ahi_xml_desat3_proxy",
                "ahi_raw_desat3_proxy",
            )
        },
    }
    return summary


def discover_pairs(edf_dir: Path, xml_dir: Path) -> list[tuple[Path, Path]]:
    xml_by_id = {p.name.removesuffix("-nsrr.xml"): p for p in xml_dir.glob("*-nsrr.xml")}
    pairs = []
    for edf in sorted(edf_dir.glob("*.edf")):
        xml = xml_by_id.get(edf.stem)
        if xml is not None:
            pairs.append((edf, xml))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf-dir", type=Path, default=Path("shsh1"))
    parser.add_argument("--xml-dir", type=Path, default=Path("labels/shhs1"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_shhs_v1"))
    parser.add_argument("--clinical-csv", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(exist_ok=True)
    write_schema(args.output_dir)
    clinical = load_clinical_labels(args.clinical_csv)
    pairs = discover_pairs(args.edf_dir, args.xml_dir)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(
        f"cache_version={CACHE_VERSION} subjects={len(pairs)} workers={args.workers} "
        f"clinical_labels={len(clinical)} output={args.output_dir}",
        flush=True,
    )

    rows: list[dict] = []
    failures: list[dict] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        jobs = {}
        for edf, xml in pairs:
            output = subject_dir / f"{edf.stem}.npz"
            ahi = clinical.get(edf.stem)
            future = executor.submit(
                process_subject, str(edf), str(xml), str(output), ahi, args.force
            )
            jobs[future] = edf.stem
        for completed, future in enumerate(as_completed(jobs), start=1):
            sid = jobs[future]
            try:
                row = future.result()
                if sid in clinical:
                    row["ahi_clinical_a0h3"] = float(clinical[sid])
                    row["label_ahi"] = float(clinical[sid])
                    row["label_severity"] = severity(float(clinical[sid]))
                    row["label_source"] = "clinical_ahi_a0h3"
                row["cache_bytes"] = (subject_dir / f"{sid}.npz").stat().st_size
                rows.append(row)
            except Exception as exc:
                error = {
                    "subject_id": sid,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                }
                failures.append(error)
                print(f"FAILED {sid}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if completed == 1 or completed % 25 == 0 or completed == len(jobs):
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                remaining = (len(jobs) - completed) / rate if rate else float("nan")
                size_gib = sum(x.get("cache_bytes", 0) for x in rows) / 2**30
                print(
                    f"progress={completed}/{len(jobs)} ok={len(rows)} failed={len(failures)} "
                    f"rate={rate:.2f}/s eta_min={remaining/60:.1f} cache_GiB={size_gib:.2f}",
                    flush=True,
                )

    manifest = pd.DataFrame(rows + failures).sort_values("subject_id").reset_index(drop=True)
    manifest.to_parquet(args.output_dir / "manifest.parquet", index=False)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    if rows:
        good = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
        label_columns = [
            "subject_id",
            "ahi_clinical_a0h3",
            "ahi_xml_desat3_proxy",
            "ahi_raw_desat3_proxy",
            "ahi_raw_desat3_or_arousal_proxy",
            "label_ahi",
            "label_severity",
            "label_source",
            "sleep_seconds",
        ]
        labels = good[label_columns]
        labels.to_parquet(args.output_dir / "labels.parquet", index=False)
        labels.to_csv(args.output_dir / "labels.csv", index=False)
        deterministic_proxy_folds(labels).to_parquet(args.output_dir / "splits_5fold.parquet", index=False)
        with np.load(subject_dir / f"{good.iloc[0]['subject_id']}.npz", allow_pickle=False) as sample:
            write_schema(args.output_dir, [str(x) for x in sample["feature_names"]])
    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    summary = build_qc_summary(manifest)
    (args.output_dir / "qc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
