"""Turning the window manifest into training arrays, one channel at a time.

The manifest holds indices, not samples. This module is where samples appear, and
it appears them exactly once per channel:

    ensure_emg_cache()  reads each .mat once and writes the 16-column EMG as a
                        float32 .npy under data/interim/emg/. 20 files, ~1.8 GB,
                        written once per (config, manifest) and reused by every
                        channel, model, and seed.

    channel_arrays()    memory-maps those files, takes ONE column, and gathers the
                        windows for that column. ~110 MB per role per channel.

Without the cache, 14 channels x 20 files would decompress the same 21 GB of .mat
payloads 280 times. With it, the expensive read happens 20 times in total.

Normalisation is applied here, from the scaler fitted in Stage A1 on Train only.
The scaler is per channel and shared by every model and seed -- five scalers for
five seeds would mean five different inputs and a comparison that is not a
comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import manifest as manifest_mod
from . import windowing
from .config import Config
from .matfile_source import MatFileSource


@dataclass
class SplitArrays:
    """One role's arrays for one channel."""

    x: np.ndarray            # [N, window_samples] float32, already scaled
    y: np.ndarray            # [N] int64 class ids
    subjects: np.ndarray     # [N] int32
    files: np.ndarray        # [N] object
    windows: pd.DataFrame    # the manifest rows this array was built from

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, mask: np.ndarray) -> "SplitArrays":
        return SplitArrays(
            x=self.x[mask],
            y=self.y[mask],
            subjects=self.subjects[mask],
            files=self.files[mask],
            windows=self.windows.loc[mask].reset_index(drop=True),
        )


def interim_dir(config: Config) -> Path:
    return (config.config_dir.parent / "data/interim/emg").resolve()


def cache_signature(config: Config, manifest_sha: str) -> str:
    return f"{config.sha256[:16]}_{manifest_sha[:16]}"


def ensure_emg_cache(
    *,
    config: Config,
    source: MatFileSource,
    files: Sequence[str],
    manifest_sha: str,
    log: Any = None,
) -> Path:
    """Write one .npy per file holding its 16 EMG columns as float32.

    Reuses an existing cache when the config and manifest hashes match: the bytes
    are then provably the same, so re-reading 21 GB would buy nothing.
    """
    out_dir = interim_dir(config) / cache_signature(config, manifest_sha)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / "cache.json"

    if stamp.is_file():
        existing = json.loads(stamp.read_text(encoding="utf-8"))
        if existing.get("config_sha256") == config.sha256 and existing.get(
            "manifest_sha256"
        ) == manifest_sha and all((out_dir / f"{Path(f).stem}.npy").is_file() for f in files):
            return out_dir

    written: dict[str, Any] = {}
    if stamp.is_file():
        # Merge rather than replace: Stage F adds Test files to the same cache, and
        # dropping the Train entries from the stamp would misdescribe the directory.
        try:
            written.update(json.loads(stamp.read_text(encoding="utf-8")).get("files", {}))
        except (json.JSONDecodeError, OSError):
            written = {}

    for name in files:
        path = out_dir / f"{Path(name).stem}.npy"
        if path.is_file() and name in written:
            # Already materialised under this same config+manifest signature.
            continue
        data = source.load(name, reason="materialise EMG for training")
        emg = np.ascontiguousarray(np.asarray(data["emg"], dtype=np.float32))
        np.save(path, emg)
        written[name] = {"shape": list(emg.shape), "dtype": "float32"}
        if log is not None:
            log(name)

    stamp.write_text(
        json.dumps(
            {
                "config_sha256": config.sha256,
                "manifest_sha256": manifest_sha,
                "files": written,
                "note": "float32 EMG, 16 matrix columns preserved; column index == CH id - 1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir


def channel_arrays(
    *,
    config: Config,
    cache_dir: Path,
    windows: pd.DataFrame,
    channel_id: str,
    scaler: windowing.StandardScaler,
    split: str,
) -> SplitArrays:
    """Gather one channel's windows for one split, scaled."""
    matrix_index = int(channel_id[2:]) - 1
    if not 0 <= matrix_index < config.matrix_columns:
        raise ValueError(f"{channel_id}: matrix index {matrix_index} out of range")

    rows = windows[windows["split"] == split]
    if rows.empty:
        raise ValueError(f"no windows for split {split!r}")

    xs: list[np.ndarray] = []
    order: list[np.ndarray] = []
    for name, group in rows.groupby("file", sort=True):
        path = cache_dir / f"{Path(name).stem}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"EMG cache missing {path}; re-run Stage A1")
        arr = np.load(path, mmap_mode="r")
        if arr.shape[1] != config.matrix_columns:
            raise ValueError(
                f"{name}: cached EMG has {arr.shape[1]} columns, expected {config.matrix_columns}"
            )
        col = np.asarray(arr[:, matrix_index], dtype=np.float32)
        bounds = np.stack(
            [group["start"].to_numpy(dtype=np.int64), group["stop"].to_numpy(dtype=np.int64)],
            axis=1,
        )
        xs.append(windowing.gather_channel(col, 0, bounds))
        order.append(group.index.to_numpy())

    x = np.concatenate(xs, axis=0)
    idx = np.concatenate(order)
    rows = windows.loc[idx].reset_index(drop=True)

    return SplitArrays(
        x=scaler.transform(x),
        y=rows["class_id"].to_numpy(dtype=np.int64),
        subjects=rows["subject"].to_numpy(dtype=np.int32),
        files=rows["file"].to_numpy(dtype=object),
        windows=rows,
    )


def load_scaler(config: Config, channel_id: str) -> windowing.StandardScaler:
    path = config.path("scalers_dir") / f"{channel_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"scaler for {channel_id} missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return windowing.StandardScaler(
        mean=float(payload["mean"]),
        std=float(payload["std"]),
        channel=str(payload["channel"]),
        eps=float(payload["eps"]),
    )


def restrict_subjects(arrays: SplitArrays, subjects: Iterable[int]) -> SplitArrays:
    wanted = set(int(s) for s in subjects)
    mask = np.isin(arrays.subjects, np.fromiter(wanted, dtype=np.int32))
    if not mask.any():
        raise ValueError(f"no windows for subjects {sorted(wanted)}")
    return arrays.subset(mask)


def channel_ids_from_map(channel_map: dict[str, Any]) -> list[str]:
    return list(channel_map["candidate_channels"])


def read_channel_map(config: Config) -> dict[str, Any]:
    path = config.path("results_dir") / "channel_map.json"
    if not path.is_file():
        raise FileNotFoundError(f"channel map missing: {path}; run Stage A1 first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_trainval(config: Config) -> pd.DataFrame:
    return manifest_mod.read_windows(config.path("manifest_dir") / "trainval_windows.parquet")
