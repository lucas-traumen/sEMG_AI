"""One seeded training run.

Everything about a run is determined by (config, channel, model, seed, manifest).
The seed is set in four places -- Python, NumPy, torch CPU, torch CUDA -- plus the
DataLoader generator, because a seed that only reaches `random` reproduces nothing
once torch is involved.

Checkpoint selection rule (an epoch-vs-epoch rule inside ONE run):
    higher validation accuracy wins; on an exact tie, lower validation loss wins.
Seeds do not compete with each other anywhere in this file. A seed is one estimate
of the same quantity; picking "the best seed" would be reporting a maximum.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import models as models_mod
from .config import Config


@dataclass
class RunResult:
    """What one training run produced, independent of where it was written."""

    channel: str
    model: str
    seed: int
    best_epoch: int
    best_val_accuracy: float
    best_val_loss: float
    final_train_loss: float
    history: list[dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0
    peak_gpu_memory_mb: float | None = None
    param_count: int = 0
    macs_analytic: int = 0
    flops_measured: int = 0
    environment: dict[str, Any] = field(default_factory=dict)
    scaler_sha256: str = ""
    state_dict: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "model": self.model,
            "seed": int(self.seed),
            "best_epoch": int(self.best_epoch),
            "best_val_accuracy": float(self.best_val_accuracy),
            "best_val_loss": float(self.best_val_loss),
            "final_train_loss": float(self.final_train_loss),
            "train_seconds": float(self.train_seconds),
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "param_count": int(self.param_count),
            "macs_analytic": int(self.macs_analytic),
            "flops_measured": int(self.flops_measured),
            "scaler_sha256": self.scaler_sha256,
            "n_epochs_run": len(self.history),
            "environment": self.environment,
        }


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed every generator that can influence a run."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_memory_mb"] = round(props.total_memory / (1024**2), 1)
    else:
        info["gpu_name"] = None
        info["gpu_total_memory_mb"] = None
    return info


def pick_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    # x is [N, L]; the model wants [N, 1, L]. Unsqueeze once here rather than in
    # the training loop.
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(1)
    y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
    ds = TensorDataset(x_t, y_t)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=gen if shuffle else None,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (loss, accuracy, predictions, targets) over one loader."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * int(yb.numel())
        p = logits.argmax(dim=1)
        correct += int((p == yb).sum().item())
        total += int(yb.numel())
        preds.append(p.detach().cpu().numpy())
        tgts.append(yb.detach().cpu().numpy())
    if total == 0:
        raise ValueError("evaluate() received an empty loader")
    return (
        total_loss / total,
        correct / total,
        np.concatenate(preds),
        np.concatenate(tgts),
    )


def train_one(
    *,
    config: Config,
    channel: str,
    model_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    scaler_sha256: str = "",
    max_epochs: int | None = None,
    device: torch.device | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    """Train one model on one channel for one seed."""
    train_cfg = config.training
    repro = train_cfg["reproducibility"]
    set_seed(seed, deterministic=bool(repro["deterministic_algorithms"]))

    device = device or pick_device()
    epochs_cfg = train_cfg["epochs"]
    n_epochs = int(max_epochs if max_epochs is not None else epochs_cfg["max_epochs"])
    patience = int(epochs_cfg["early_stopping_patience"])
    loader_cfg = train_cfg["loader"]

    train_loader = _make_loader(
        x_train,
        y_train,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=bool(loader_cfg["shuffle_train"]),
        seed=seed,
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]) and device.type == "cuda",
    )
    val_loader = _make_loader(
        x_val,
        y_val,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]) and device.type == "cuda",
    )

    model = models_mod.build(config, model_name).to(device)
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"])
    )
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best: dict[str, Any] = {
        "epoch": -1,
        "val_accuracy": -1.0,
        "val_loss": float("inf"),
        "state": None,
    }
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())
        train_loss = running / max(seen, 1)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }
        )
        if progress is not None:
            progress(history[-1])

        # Selection rule: accuracy first, loss as the tie-break. Comparing floats
        # for exact equality is intentional here -- both numbers come from the same
        # correct/total ratio, so a tie is a genuine tie, not a rounding artefact.
        improved = (
            val_acc > best["val_accuracy"]
            or (val_acc == best["val_accuracy"] and val_loss < best["val_loss"])
        )
        if improved:
            best = {
                "epoch": epoch,
                "val_accuracy": float(val_acc),
                "val_loss": float(val_loss),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    train_seconds = time.perf_counter() - started
    peak_mb = None
    if device.type == "cuda":
        peak_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)

    length = int(config.models["input"]["length"])
    macs = models_mod.macs_analytic(model, length)
    flops = models_mod.flops_measured(model, length, device=device.type)

    return RunResult(
        channel=channel,
        model=model_name,
        seed=int(seed),
        best_epoch=int(best["epoch"]),
        best_val_accuracy=float(best["val_accuracy"]),
        best_val_loss=float(best["val_loss"]),
        final_train_loss=float(history[-1]["train_loss"]),
        history=history,
        train_seconds=round(train_seconds, 3),
        peak_gpu_memory_mb=peak_mb,
        param_count=models_mod.param_count(model),
        macs_analytic=int(macs["macs"]),
        flops_measured=int(flops),
        environment=environment_info(),
        scaler_sha256=scaler_sha256,
        state_dict=best["state"],
    )


def save_checkpoint(
    result: RunResult,
    *,
    path: Path,
    config_sha256: str,
    manifest_sha256: str,
) -> None:
    """Write the checkpoint atomically: a half-written .pt must never be loadable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "channel": result.channel,
        "model": result.model,
        "seed": int(result.seed),
        "state_dict": result.state_dict,
        "best_epoch": int(result.best_epoch),
        "best_val_accuracy": float(result.best_val_accuracy),
        "best_val_loss": float(result.best_val_loss),
        "param_count": int(result.param_count),
        "macs_analytic": int(result.macs_analytic),
        "flops_measured": int(result.flops_measured),
        "scaler_sha256": result.scaler_sha256,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "environment": result.environment,
        "history": result.history,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)

    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(result.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_model(config: Config, payload: dict[str, Any]) -> nn.Module:
    model = models_mod.build(config, str(payload["model"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_logits(
    model: nn.Module, x: np.ndarray, *, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    """Run inference and return raw logits [N, num_classes]."""
    model.eval().to(device)
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(1)
    out: list[np.ndarray] = []
    for i in range(0, x_t.shape[0], batch_size):
        chunk = x_t[i : i + batch_size].to(device)
        out.append(model(chunk).detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, config.num_classes), dtype=np.float32)
