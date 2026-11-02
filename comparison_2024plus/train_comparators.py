"""Five controlled, architecture-adapted severity comparators (2024+ literature).

These are independent implementations of architectural motifs, not author-code
reproductions. All models share exactly the same cached full-night inputs,
labels, outer folds, validation-fold rule, loss, and training budget.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260919
NAMES = ("xception", "resnet", "cnn_bigru_attention", "multiscale_transformer", "dual_branch_cnn")


class SepBlock(nn.Module):
    def __init__(self, width: int, kernel: int = 5):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width),
                                 nn.Conv1d(width, width, 1), nn.BatchNorm1d(width), nn.GELU())

    def forward(self, x):
        return x + self.net(x)


class ResBlock(nn.Module):
    def __init__(self, width: int, kernel: int = 5):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, kernel, padding=kernel // 2),
                                 nn.BatchNorm1d(width), nn.GELU(),
                                 nn.Conv1d(width, width, kernel, padding=kernel // 2),
                                 nn.BatchNorm1d(width))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class XceptionAdapted(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, 48, 5, stride=2, padding=2), nn.GELU(),
                                 SepBlock(48), SepBlock(48), nn.AvgPool1d(2),
                                 SepBlock(48), SepBlock(48), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(48, 4)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1))


class ResNetAdapted(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, 48, 7, stride=2, padding=3),
                                 nn.BatchNorm1d(48), nn.ReLU(), ResBlock(48),
                                 nn.AvgPool1d(2), ResBlock(48), ResBlock(48),
                                 nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(48, 4)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1))


class CNNBiGRUAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(channels, 40, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(40, 48, 5, stride=2, padding=2), nn.GELU())
        self.gru = nn.GRU(48, 40, batch_first=True, bidirectional=True)
        self.att = nn.Linear(80, 1)
        self.head = nn.Linear(80, 4)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.gru(z)
        weights = torch.softmax(self.att(z), dim=1)
        return self.head((weights * z).sum(dim=1))


class MultiScaleTransformer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.short = nn.Conv1d(channels, 32, 3, stride=2, padding=1)
        self.long = nn.Conv1d(channels, 32, 11, stride=2, padding=5)
        self.pos = nn.Parameter(torch.zeros(1, 64, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        x = x.transpose(1, 2)
        z = torch.cat([self.short(x), self.long(x)], dim=1)
        z = torch.nn.functional.avg_pool1d(z, 2).transpose(1, 2)
        z = self.encoder(z + self.pos[:, :z.shape[1]])
        return self.head(z.mean(dim=1))


class DualBranchCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Parallel branches model local and longer cardiorespiratory patterns.
        self.a = nn.Sequential(nn.Conv1d(channels, 32, 3, padding=1), nn.ReLU(),
                               nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.ReLU())
        self.b = nn.Sequential(nn.Conv1d(channels, 32, 11, padding=5), nn.ReLU(),
                               nn.Conv1d(32, 32, 9, stride=2, padding=4), nn.ReLU())
        self.merge = nn.Sequential(nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
                                   nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(self.merge(torch.cat([self.a(x), self.b(x)], dim=1)).squeeze(-1))


def build_model(name: str, channels: int) -> nn.Module:
    return {"xception": XceptionAdapted,
            "resnet": ResNetAdapted,
            "cnn_bigru_attention": CNNBiGRUAttention,
            "multiscale_transformer": MultiScaleTransformer,
            "dual_branch_cnn": DualBranchCNN}[name](channels)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for chunk in x.split(batch_size):
        out.append(torch.softmax(model(chunk.to(device)), -1).cpu().numpy())
    return np.concatenate(out)


def fit_one(name: str, fold: int, data: dict, args: argparse.Namespace, output: Path) -> None:
    random.seed(SEED + fold)
    np.random.seed(SEED + fold)
    torch.manual_seed(SEED + fold)
    mask_test = data["fold"] == fold
    mask_val = data["fold"] == (fold + 1) % 5
    mask_train = ~(mask_test | mask_val)
    raw = data["x"]
    mean = raw[mask_train].mean(axis=(0, 1), keepdims=True)
    std = raw[mask_train].std(axis=(0, 1), keepdims=True).clip(min=1e-3)
    x = np.clip((raw - mean) / std, -8, 8).astype(np.float32)
    x = np.nan_to_num(x)
    y = data["y"].astype(np.int64)
    x_train = torch.from_numpy(x[mask_train]); y_train = torch.from_numpy(y[mask_train])
    x_val = torch.from_numpy(x[mask_val]); x_test = torch.from_numpy(x[mask_test])
    device = "cuda" if args.cuda else "cpu"
    model = build_model(name, x.shape[-1]).to(device)
    counts = np.bincount(y[mask_train], minlength=4)
    weights = torch.tensor(np.sqrt(counts.sum() / (4 * counts)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    best_score, best_state, best_epoch, patience = -1, None, 0, 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_prob = predict(model, x_val, device)
        val_score = f1_score(y[mask_val], val_prob.argmax(1), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": np.mean(losses), "val_macro_f1": val_score})
        print(f"{name} fold={fold} epoch={epoch} loss={np.mean(losses):.4f} val_f1={val_score:.4f}", flush=True)
        if val_score > best_score + 1e-5:
            best_score, best_epoch, patience = val_score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break
    model.load_state_dict(best_state)
    prob = predict(model, x_test, device)
    pred = prob.argmax(1)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "feature_mean": mean, "feature_std": std,
                "architecture": name, "fold": fold}, output / f"fold_{fold}_model.pt")
    frame = pd.DataFrame({"subject_id": data["subject_id"][mask_test], "fold": fold,
                          "target_class": y[mask_test], "target_ahi": data["ahi"][mask_test],
                          "predicted_class": pred})
    for k, label in enumerate(("normal", "mild", "moderate", "severe")):
        frame[f"probability_{label}"] = prob[:, k]
    frame.to_parquet(output / f"fold_{fold}_oof.parquet", index=False)
    pd.DataFrame(history).to_csv(output / f"fold_{fold}_history.csv", index=False)
    metadata = {"name": name, "fold": fold, "train_n": int(mask_train.sum()),
                "validation_n": int(mask_val.sum()), "test_n": int(mask_test.sum()),
                "best_epoch": best_epoch, "validation_macro_f1": best_score,
                "test_accuracy": accuracy_score(y[mask_test], pred),
                "test_macro_f1": f1_score(y[mask_test], pred, average="macro"),
                "test_balanced_accuracy": balanced_accuracy_score(y[mask_test], pred),
                "test_quadratic_kappa": cohen_kappa_score(y[mask_test], pred, weights="quadratic"),
                "runtime_seconds": time.time() - started, "parameters": sum(p.numel() for p in model.parameters()),
                "device": device, "seed": SEED + fold}
    (output / f"fold_{fold}_metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path("comparison_2024plus/inputs.npz"))
    parser.add_argument("--output", type=Path, default=Path("comparison_2024plus/results"))
    parser.add_argument("--models", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with np.load(args.inputs) as z:
        data = {k: z[k] for k in ("x", "y", "ahi", "fold", "subject_id")}
    for name in args.models:
        for fold in args.folds:
            path = args.output / name
            if all((path / f"fold_{fold}_{suffix}").exists() for suffix in
                   ("oof.parquet", "model.pt", "history.csv", "metadata.json")):
                print(f"skip complete {name} fold={fold}", flush=True)
                continue
            fit_one(name, fold, data, args, path)


if __name__ == "__main__":
    main()
