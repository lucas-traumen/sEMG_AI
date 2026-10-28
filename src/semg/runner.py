"""Shared training-grid machinery for Stages B, C, D and E.

All four stages are the same shape: a set of (channel, model, seed) cells, each one
a training run whose Validation accuracy is measured per subject. Writing that loop
once means the idempotency rule, the checkpoint layout, and the row schema cannot
drift between stages.

Idempotency is per cell, not per stage: a stage that is interrupted at cell 12 of
15 resumes at cell 12. Each finished cell appends to the CSV immediately, so the
file on disk is always a truthful record of what has been trained.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import dataset, manifest, metrics, reporting, train
from . import models as models_mod
from .config import Config

SUMMARY_COLUMNS = (    "channel",
    "model",
    "seed",
    "best_epoch",
    "best_val_accuracy",
    "best_val_loss",
    "final_train_loss",
    "val_subject_mean_accuracy",
    "val_subject_sd",
    "n_val_windows",
    "n_train_windows",
    "param_count",
    "macs_analytic",
    "flops_measured",
    "train_seconds",
    "elapsed_seconds",
    "peak_gpu_memory_mb",
    "scaler_sha256",
    "config_sha256",
    "manifest_sha256",
)

SUBJECT_COLUMNS = (
    "channel",
    "model",
    "seed",
    "subject",
    "split",
    "n_windows",
    "n_correct",
    "accuracy",
)


@dataclass
class Cell:
    channel: str
    model: str
    seed: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.channel, self.model, int(self.seed))

    @property
    def label(self) -> str:
        return f"{self.channel}/{self.model}/s{self.seed}"


class RunnerError(RuntimeError):
    """A grid cell has no result and cannot be recovered from an earlier stage."""


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    return df.to_dict("records")


def run_cells(
    *,
    config: Config,
    ctx: Any,
    cells: Sequence[Cell],
    summary_path: Path,
    subject_path: Path,
    max_epochs: int | None,
    subjects: Sequence[int] | None = None,
    save_checkpoints: bool = True,
    progress: Callable[[str], None] | None = None,
    summary_columns: Sequence[str] = SUMMARY_COLUMNS,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train every cell that is not already in `summary_path`."""
    windows = dataset.load_trainval(config)
    manifest_sha = manifest.read_manifest_sha(ctx.manifest_sha_path)["manifest_sha256"]

    entries = _entries(ctx)
    source = _source(entries, ctx=ctx, stage=ctx.report.stage)
    cache = dataset.ensure_emg_cache(
        config=config,
        source=source,
        files=sorted(windows["file"].unique()),
        manifest_sha=manifest_sha,
    )

    done = {tuple([str(r["channel"]), str(r["model"]), int(r["seed"])]) for r in read_rows(summary_path)}
    summary_rows = read_rows(summary_path)
    subject_rows = read_rows(subject_path)

    device = train.pick_device()
    subject_filter = (
        np.fromiter((int(s) for s in subjects), dtype=np.int32) if subjects else None
    )
    per_channel_cache: dict[str, tuple[dataset.SplitArrays, dataset.SplitArrays]] = {}
    n_new = 0

    for i, cell in enumerate(cells, start=1):
        if cell.key in done:
            continue

        if cell.channel not in per_channel_cache:
            scaler = dataset.load_scaler(config, cell.channel)
            tr = dataset.channel_arrays(
                config=config,
                cache_dir=cache,
                windows=windows,
                channel_id=cell.channel,
                scaler=scaler,
                split="train",
            )
            va = dataset.channel_arrays(
                config=config,
                cache_dir=cache,
                windows=windows,
                channel_id=cell.channel,
                scaler=scaler,
                split="validation",
            )
            per_channel_cache[cell.channel] = (tr, va)
        tr, va = per_channel_cache[cell.channel]
        if subject_filter is not None:
            tr_use = dataset.restrict_subjects(tr, subject_filter)
            va_use = dataset.restrict_subjects(va, subject_filter)
        else:
            tr_use, va_use = tr, va

        scaler_sha = dataset.load_scaler(config, cell.channel).sha256
        started = time.perf_counter()
        result = train.train_one(
            config=config,
            channel=cell.channel,
            model_name=cell.model,
            seed=cell.seed,
            x_train=tr_use.x,
            y_train=tr_use.y,
            x_val=va_use.x,
            y_val=va_use.y,
            scaler_sha256=scaler_sha,
            max_epochs=max_epochs,
            device=device,
        )
        elapsed = time.perf_counter() - started

        pred = _predict(config, result, va_use.x, device)
        rows = metrics.accuracy_by_subject(
            seeds=[cell.seed],
            subjects=va_use.subjects,
            predictions={cell.seed: pred},
            targets=va_use.y,
        )
        subject_rows.extend(
            {
                "channel": cell.channel,
                "model": cell.model,
                "seed": int(cell.seed),
                "split": "validation",
                **r,
            }
            for r in rows
        )
        accs = [float(r["accuracy"]) for r in rows]
        row_out = {
            "channel": cell.channel,
            "model": cell.model,
            "seed": int(cell.seed),
            "best_epoch": int(result.best_epoch),
            "best_val_accuracy": float(result.best_val_accuracy),
            "best_val_loss": float(result.best_val_loss),
            "final_train_loss": float(result.final_train_loss),
            "val_subject_mean_accuracy": float(np.mean(accs)),
            "val_subject_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "n_val_windows": int(len(va_use)),
            "n_train_windows": int(len(tr_use)),
            "param_count": int(result.param_count),
            "macs_analytic": int(result.macs_analytic),
            "flops_measured": int(result.flops_measured),
            "train_seconds": float(result.train_seconds),
            "elapsed_seconds": round(elapsed, 2),
            "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
            "scaler_sha256": scaler_sha,
            "config_sha256": config.sha256,
            "manifest_sha256": manifest_sha,
        }
        if extra_summary:
            for key, value in extra_summary.items():
                row_out.setdefault(key, value)
        summary_rows.append(row_out)

        if save_checkpoints:
            train.save_checkpoint(
                result,
                path=checkpoint_path(config, cell),
                config_sha256=config.sha256,
                manifest_sha256=manifest_sha,
            )

        # Write after every cell: the CSV is the resume point and the record.
        reporting.write_csv(summary_rows, summary_path, columns=summary_columns)
        reporting.write_csv(subject_rows, subject_path, columns=SUBJECT_COLUMNS)
        n_new += 1
        if progress is not None:
            progress(
                f"[{i}/{len(cells)}] {cell.label} val_subject_mean="
                f"{summary_rows[-1]['val_subject_mean_accuracy']:.4f} "
                f"({result.train_seconds:.1f}s, {len(result.history)} ep)"
            )

    return {
        "n_cells": len(cells),
        "n_trained_now": n_new,
        "n_reused": len(cells) - n_new,
        "device": str(device),
        "rows": summary_rows,
    }


def checkpoint_path(config: Config, cell: Cell) -> Path:
    """checkpoints/<channel>/<model>/seed_<seed>.pt -- no subject component."""
    naming = str(config.experiment["scope"]["checkpoint_naming"])
    rel = (
        naming.replace("<channel>", cell.channel)
        .replace("<model>", cell.model)
        .replace("<seed>", str(int(cell.seed)))
    )
    return config.path("checkpoints_dir") / rel


def _predict(config: Config, result: train.RunResult, x: np.ndarray, device) -> np.ndarray:
    model = train.restore_model(
        config, {"model": result.model, "state_dict": result.state_dict}
    )
    logits = train.predict_logits(model, x, device=device)
    return logits.argmax(axis=1)


def _entries(ctx: Any) -> list[Any]:
    from . import stages

    return stages.load_entries(ctx)


def _source(entries: list[Any], *, ctx: Any, stage: str):
    from . import stages

    return stages.build_source_for(entries, ctx=ctx, stage=stage)


def model_costs(config: Config, *, device: str = "cpu") -> dict[str, dict[str, Any]]:
    """Parameter/MAC/FLOP numbers for every configured model."""
    out = {}
    for name in config.models["models"]:
        out[name] = models_mod.model_stats(config, name, device=device)
    return out


def write_model_stats(config: Config, path: Path, *, device: str = "cpu") -> dict[str, Any]:
    stats = model_costs(config, device=device)
    payload = {
        "schema_version": 1,
        "note": (
            "macs_analytic comes from an explicit per-layer formula; "
            "flops_measured comes from torch FlopCounterMode. They are separate "
            "numbers and MACs is never derived as flops // 2."
        ),
        "models": stats,
    }
    reporting.write_json(payload, path)
    return payload


def write_run_index(config: Config, ctx: Any, cells: Iterable[Cell], *, stage: str) -> None:
    """Append-only index of every (stage, cell) that has been trained."""
    path = ctx.run_index_path
    existing = read_rows(path) if path.is_file() else []
    seen = {(r["stage"], r["channel"], r["model"], int(r["seed"])) for r in existing}
    for cell in cells:
        key = (stage, cell.channel, cell.model, int(cell.seed))
        if key in seen:
            continue
        existing.append(
            {
                "stage": stage,
                "channel": cell.channel,
                "model": cell.model,
                "seed": int(cell.seed),
                "config_sha256": config.sha256,
            }
        )
    reporting.write_csv(existing, path)
