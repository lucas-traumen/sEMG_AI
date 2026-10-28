"""Segment scan and fully-inside window extraction.

The leakage rule this module implements is the one that makes a per-window
accuracy meaningful:

    A window belongs to exactly one contiguous segment of constant
    (restimulus, rerepetition). It must lie entirely inside that segment.

A window that straddles a rest->grasp transition would carry two labels; one that
straddles a repetition boundary would contain samples from a different trial. Both
inflate accuracy and neither is a real-time situation the device can be in.

Segments come from `restimulus` (the relabelled stimulus: 0 = REST, non-zero =
grasp) combined with `rerepetition` (a per-trial counter). The pair is constant
within a trial of one gesture, so a change in either starts a new segment.

Windowing emits INDICES, not signal. `WindowIndex` rows are the manifest; the
signal is materialised per channel on demand by `gather_channel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """One contiguous run of constant (raw_label, repetition)."""

    segment_id: int
    start: int          # inclusive, sample index in the file
    stop: int           # exclusive
    raw_label: int
    repetition: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def find_segments(restimulus: np.ndarray, rerepetition: np.ndarray) -> list[Segment]:
    """Split a file into maximal constant-(label, repetition) runs."""
    labels = np.asarray(restimulus).reshape(-1)
    reps = np.asarray(rerepetition).reshape(-1)
    if labels.shape != reps.shape:
        raise ValueError(f"restimulus {labels.shape} and rerepetition {reps.shape} disagree")

    n = labels.shape[0]
    if n == 0:
        return []

    # A new segment starts wherever either the label or the repetition changes.
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (labels[1:] != labels[:-1]) | (reps[1:] != reps[:-1])
    starts = np.flatnonzero(change)
    stops = np.append(starts[1:], n)

    return [
        Segment(
            segment_id=i,
            start=int(s),
            stop=int(e),
            raw_label=int(labels[s]),
            repetition=int(reps[s]),
        )
        for i, (s, e) in enumerate(zip(starts, stops))
    ]


def segment_windows(
    segment: Segment,
    *,
    window_samples: int,
    hop_samples: int,
) -> list[tuple[int, int]]:
    """All (start, stop) windows fully inside `segment`, in sample offsets.

    A segment shorter than one window yields nothing -- that is the correct answer
    for the many short REST gaps, and padding one would fabricate data.
    """
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive")
    if segment.length < window_samples:
        return []
    starts = range(segment.start, segment.stop - window_samples + 1, hop_samples)
    return [(s, s + window_samples) for s in starts]


@dataclass(frozen=True)
class WindowIndex:
    """One row of the window manifest. Indices only -- no samples."""

    file: str
    subject: int
    day: int
    session: int
    split: str
    segment_id: int
    repetition: int
    raw_label: int
    class_id: int
    start: int
    stop: int

    @property
    def n_samples(self) -> int:
        return self.stop - self.start

    def as_row(self) -> dict[str, object]:
        return {
            "file": self.file,
            "subject": self.subject,
            "day": self.day,
            "session": self.session,
            "split": self.split,
            "segment_id": self.segment_id,
            "repetition": self.repetition,
            "raw_label": self.raw_label,
            "class_id": self.class_id,
            "start": self.start,
            "stop": self.stop,
        }


WINDOW_COLUMNS = (
    "file",
    "subject",
    "day",
    "session",
    "split",
    "segment_id",
    "repetition",
    "raw_label",
    "class_id",
    "start",
    "stop",
)


def build_window_index(
    *,
    file: str,
    subject: int,
    day: int,
    session: int,
    split: str,
    restimulus: np.ndarray,
    rerepetition: np.ndarray,
    raw_to_class: dict[int, int],
    window_samples: int,
    hop_samples: int,
) -> list[WindowIndex]:
    """Full window index for one file, with labels resolved to class ids."""
    out: list[WindowIndex] = []
    for seg in find_segments(restimulus, rerepetition):
        if seg.raw_label not in raw_to_class:
            # A label the config does not know is a hard error, not a skip: it
            # would silently shrink a class.
            raise KeyError(f"{file}: segment label {seg.raw_label} is not in gestures.yaml")
        class_id = raw_to_class[seg.raw_label]
        for start, stop in segment_windows(
            seg, window_samples=window_samples, hop_samples=hop_samples
        ):
            out.append(
                WindowIndex(
                    file=file,
                    subject=subject,
                    day=day,
                    session=session,
                    split=split,
                    segment_id=seg.segment_id,
                    repetition=seg.repetition,
                    raw_label=seg.raw_label,
                    class_id=class_id,
                    start=start,
                    stop=stop,
                )
            )
    return out


def gather_channel(
    emg: np.ndarray,
    channel_index: int,
    windows: list[WindowIndex] | np.ndarray,
) -> np.ndarray:
    """Materialise one channel's windows as float32 [n_windows, window_samples].

    `emg` may be the full [N, 16] matrix or an already-extracted 1-D column; the
    dataset cache keeps single columns, so both forms are accepted rather than
    forcing every caller to reshape.
    """
    arr = np.asarray(emg)
    col = np.asarray(arr if arr.ndim == 1 else arr[:, channel_index], dtype=np.float32)
    if isinstance(windows, np.ndarray):
        starts = windows[:, 0]
        stops = windows[:, 1]
    else:
        starts = np.fromiter((w.start for w in windows), dtype=np.int64, count=len(windows))
        stops = np.fromiter((w.stop for w in windows), dtype=np.int64, count=len(windows))
    if len(starts) == 0:
        width = 0
        return np.zeros((0, width), dtype=np.float32)
    width = int(stops[0] - starts[0])
    if not np.all(stops - starts == width):
        raise ValueError("windows have inconsistent lengths")
    idx = starts[:, None] + np.arange(width, dtype=np.int64)[None, :]
    return col[idx]


class StandardScaler:
    """Per-channel z-score with statistics fitted on Train D1_T1 only.

    A single (mean, std) pair per channel is a constant transform: it can be baked
    into firmware and applied sample by sample, which is what AGENT.md rule 2
    requires. A per-window z-score would not be, and would also erase the
    amplitude difference between grasp and rest.
    """

    def __init__(self, mean: float, std: float, *, channel: str, eps: float = 1e-8):
        self.mean = float(mean)
        self.std = float(std)
        self.channel = channel
        self.eps = float(eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - np.float32(self.mean)) / np.float32(
            max(self.std, self.eps)
        )).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
            "method": "global_train_zscore",
            "fit_role": "train",
            "n_samples_fit": getattr(self, "n_samples_fit", None),
            "sha256": self.sha256,
        }

    @property
    def sha256(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            {"channel": self.channel, "mean": self.mean, "std": self.std, "eps": self.eps},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def fit_scaler(channel_values: np.ndarray, *, channel: str, eps: float = 1e-8) -> StandardScaler:
    """Fit from the concatenated Train D1_T1 signal of ONE channel."""
    x = np.asarray(channel_values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{channel}: cannot fit a scaler on zero samples")
    scaler = StandardScaler(mean=float(x.mean()), std=float(x.std()), channel=channel, eps=eps)
    scaler.n_samples_fit = int(x.size)  # type: ignore[attr-defined]
    return scaler
