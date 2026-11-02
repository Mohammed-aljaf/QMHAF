#!/usr/bin/env python3
"""Plan or execute the recorded training protocol in a separate runs directory.

Training starts only with --execute; without it, print the complete command plan.
Completed stages can be skipped with --resume (this restarts incomplete fits;
the original training code did not save optimizer state for mid-fit resumption).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def command_plan(config, run_dir, stage, folds):
    py = sys.executable
    ssl = run_dir / "ssl"
    deep = run_dir / "deep"
    commands = []
    if stage in ("all", "ssl"):
        c = config["ssl"]
        for fold in folds:
            commands.append((f"ssl_fold_{fold}", [py, "deep_learning/pretrain_ssl.py", "--test-fold", fold,
                "--epochs", c["epochs"], "--batch-size", c["batch_size"], "--crop-epochs", c["crop_epochs"],
                "--workers", c["workers"], "--learning-rate", c["learning_rate"], "--output-dir", ssl],
                [ssl / f"ssl_outer_fold_{fold}.pt", ssl / f"ssl_outer_fold_{fold}_history.csv"]))
    def supervised(name, signal, fold, epochs, flags=(), pretrained=True):
        c = config["main"]
        cmd = [py, "deep_learning/train_supervised.py", "--test-fold", fold, "--epochs", epochs,
            "--batch-size", c["batch_size"], "--accumulate", c["accumulate"], "--workers", c["workers"],
            "--learning-rate", c["learning_rate"], "--experiment-name", name, "--signal-set", signal, "--output-dir", deep]
        if pretrained: cmd += ["--ssl-checkpoint", ssl / f"ssl_outer_fold_{fold}.pt"]
        cmd += list(flags)
        directory = deep / name / signal / f"fold_{fold}"
        commands.append((f"{name}_{signal}_fold_{fold}", cmd,
            [directory / n for n in ("best_model.pt", "history.csv", "test_metrics.json", "oof_predictions.parquet")]))
    if stage in ("all", "main"):
        for fold in folds: supervised("main", "all", fold, config["main"]["epochs"])
    if stage in ("all", "ablations"):
        c = config["ablations"]
        for fold in c["folds"]:
            if fold not in folds: continue
            for name, flags in c["architecture_flags"].items():
                supervised(name, "all", fold, c["epochs"], flags, pretrained=name != "A0_supervised_direct")
            for signal in c["signal_sets"]: supervised("signal_ablation", signal, fold, c["epochs"])
    if stage in ("all", "aggregate"):
        commands.append(("aggregate", [py, "deep_learning/aggregate_results.py", "--results-dir", deep], []))
    if stage in ("all", "comparators"):
        c = config["comparators"]; inputs = run_dir / "comparator_inputs.npz"; output = run_dir / "comparators"
        commands.append(("comparator_inputs", [py, "comparison_2024plus/build_inputs.py", "--output", inputs], [inputs]))
        cmd = [py, "comparison_2024plus/train_comparators.py", "--inputs", inputs, "--output", output,
            "--epochs", c["epochs"], "--batch-size", c["batch_size"], "--patience", c["patience"],
            "--lr", c["learning_rate"], "--folds", *folds]
        if c["device"] == "cuda": cmd += ["--cuda"]
        # Completeness is checked within the original comparator script and below.
        commands.append(("comparators", cmd, []))
    if stage == "classical":
        commands.append(("classical", [py, "training/evaluate_baselines.py", "--output-dir", run_dir / "classical"], []))
    return [(label, list(map(str, cmd)), markers) for label, cmd, markers in commands]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/production.json")
    p.add_argument("--stage", choices=("all", "ssl", "main", "ablations", "aggregate", "comparators", "classical"), default="all")
    p.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/reproduction")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(); run_dir = args.run_dir.resolve()
    if run_dir == ROOT or ROOT not in run_dir.parents or "runs" != run_dir.relative_to(ROOT).parts[0]:
        p.error("--run-dir must be inside this checkout's runs/ directory to protect released evidence")
    config = json.loads(args.config.read_text()); plan = command_plan(config, run_dir, args.stage, args.folds)
    for label, cmd, _ in plan: print(label + ": " + shlex.join(cmd), flush=True)
    if not args.execute: return
    from scripts.prepare_data import check_splits
    check_splits(ROOT / "training/targets/ahi_a0h3_local_v1/targets.parquet")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=config, started_utc=datetime.now(timezone.utc).isoformat(),
        commands=[dict(label=label, argv=cmd) for label, cmd, _ in plan], completed=[])
    manifest_path = run_dir / "run_manifest.json"
    # Save each invocation independently; a resumed invocation does not erase history.
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("config") != config:
            raise ValueError("Run directory contains a different configuration; select a fresh --run-dir")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        manifest_path = run_dir / f"run_manifest_{stamp}.json"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    for label, cmd, markers in plan:
        if args.resume and markers and all(m.is_file() for m in markers):
            manifest["completed"].append(dict(label=label, status="skipped_complete")); continue
        if not args.resume and any(m.exists() for m in markers):
            raise FileExistsError(f"Stage artifacts already exist for {label}; use --resume or a fresh run directory")
        manifest["current_stage"] = label
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log = run_dir / (label + ".log")
        with log.open("a") as stream:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        manifest["completed"].append(dict(label=label, returncode=result.returncode, log=str(log.relative_to(ROOT))))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if result.returncode: raise SystemExit(f"Stage {label} failed; see {log}")
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
