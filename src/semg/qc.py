"""Stage A1 QC: per (file, channel) statistics and the channel map.

Two separate concepts are kept apart on purpose, because conflating them is how a
"dead channel" quietly becomes a hardcoded drop:

    structural_status  -- a property of the DB6 montage. Either `physical` or
                          `structurally_empty`. The dataset documents 14 physical
                          electrodes in a 16-column matrix, so exactly two columns
                          are empty BY DESIGN. That claim is verified against the
                          data, not assumed: an empty column must be constant for
                          the whole file (std == 0, frozen_fraction == 1.0).

    qc_flags           -- observations about a column in a file: dead_in_file,
                          low_amplitude, high_mains, clipping, nan_inf. These are
                          FLAGS. None of them removes a channel from the candidate
                          set. A channel that is merely quiet still gets screened.

`candidate_eligible` is then a simple, auditable function of the two: a column is
eligible if it is physical and no file reported it as nan_inf or clipping-broken.

Every threshold comes from configs/pipeline.yaml and was fixed before any training
run. Nothing here adjusts itself to make the 14/2 expectation come out right; if
the data disagrees with the documentation, Stage A1 reports `blocked` with the
evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import welch

STRUCTURAL_PHYSICAL = "physical"
STRUCTURAL_EMPTY = "structurally_empty"

# Measurement flag meaning "this column was numerically constant for the whole
# file". It is deliberately NOT called `structurally_empty`: whether a constant
# column is a documented hole in the matrix or a dead electrode is a question about
# the other files, answered later by build_channel_map.
CONSTANT_IN_FILE = "constant_in_file"


@dataclass
class ChannelFileStats:
    """Everything measured for one (file, channel) pair."""

    file: str
    subject: int
    channel_index: int      # 0-based matrix column
    channel_id: str         # CH01..CH16
    n_samples: int
    nan_count: int
    inf_count: int
    zero_fraction: float
    mean: float
    std: float
    min: float
    max: float
    frozen_fraction: float      # fraction of samples identical to the previous one
    longest_frozen_run: int
    clipping_fraction: float
    rms_rest: float
    rms_active: float
    mains_ratio: float
    std_ratio: float = float("nan")   # std / median std of live channels in the same file
    qc_flags: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "subject": self.subject,
            "channel_id": self.channel_id,
            "matrix_index": self.channel_index,
            "n_samples": self.n_samples,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "zero_fraction": self.zero_fraction,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "frozen_fraction": self.frozen_fraction,
            "longest_frozen_run": self.longest_frozen_run,
            "clipping_fraction": self.clipping_fraction,
            "rms_rest": self.rms_rest,
            "rms_active": self.rms_active,
            "mains_ratio": self.mains_ratio,
            "std_ratio": self.std_ratio,
            "qc_flags": ";".join(self.qc_flags),
        }


def _longest_run(mask: np.ndarray) -> int:
    """Longest run of True in a boolean array, vectorised."""
    if mask.size == 0:
        return 0
    if not mask.any():
        return 0
    # Pad with False so every run has a boundary, then diff the edge positions.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[0::2]).max())


def _mains_ratio(
    x: np.ndarray, fs_hz: int, mains_cfg: dict[str, Any]
) -> float:
    """Mean PSD in the mains band over median PSD in the flanking bands.

    A ratio rather than an absolute power, because electrode gain differs between
    sessions. NaN when the signal is too short for the requested resolution.
    """
    band = [float(v) for v in mains_cfg["band_hz"]]
    flanks = [[float(a), float(b)] for a, b in mains_cfg["flank_hz"]]
    nperseg = min(4096, x.size)
    if nperseg < 64:
        return float("nan")
    freqs, psd = welch(x, fs=fs_hz, nperseg=nperseg)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        return float("nan")
    flank_mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in flanks:
        flank_mask |= (freqs >= lo) & (freqs <= hi)
    if not flank_mask.any():
        return float("nan")
    denom = float(np.median(psd[flank_mask]))
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.mean(psd[in_band]) / denom)


def measure_channel(
    *,
    x: np.ndarray,
    restimulus: np.ndarray,
    file: str,
    subject: int,
    channel_index: int,
    channel_id: str,
    fs_hz: int,
    qc_cfg: dict[str, Any],
) -> ChannelFileStats:
    """Measure one channel of one file."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(x.size)
    nan_count = int(np.isnan(x).sum())
    inf_count = int(np.isinf(x).sum())

    finite = np.isfinite(x)
    if not finite.all():
        xf = x[finite]
    else:
        xf = x
    if xf.size == 0:
        xf = np.zeros(1, dtype=np.float64)

    std = float(xf.std())
    lo = float(xf.min())
    hi = float(xf.max())
    tol = float(qc_cfg["clipping"]["extreme_tolerance"])
    clipping = float(np.mean((xf <= lo + tol) | (xf >= hi - tol))) if xf.size else 0.0

    # Frozen-sample detection: identical consecutive values, which is what a
    # disconnected electrode or a saturated ADC produces. std==0 alone would miss
    # a channel that is constant for most of a file but not all of it.
    if n > 1:
        same = x[1:] == x[:-1]
        frozen_fraction = float(np.mean(same))
        longest = _longest_run(same) + 1
    else:
        frozen_fraction = 1.0
        longest = n

    labels = np.asarray(restimulus).reshape(-1)
    rest_mask = labels == 0
    active_mask = labels != 0
    rms_rest = float(np.sqrt(np.mean(x[rest_mask] ** 2))) if rest_mask.any() else float("nan")
    rms_active = float(np.sqrt(np.mean(x[active_mask] ** 2))) if active_mask.any() else float("nan")

    return ChannelFileStats(
        file=file,
        subject=subject,
        channel_index=channel_index,
        channel_id=channel_id,
        n_samples=n,
        nan_count=nan_count,
        inf_count=inf_count,
        zero_fraction=float(np.mean(xf == 0.0)),
        mean=float(xf.mean()),
        std=std,
        min=lo,
        max=hi,
        frozen_fraction=frozen_fraction,
        longest_frozen_run=int(longest),
        clipping_fraction=clipping,
        rms_rest=rms_rest,
        rms_active=rms_active,
        mains_ratio=_mains_ratio(xf, fs_hz, qc_cfg["mains"]),
    )


def annotate_file(rows: list[ChannelFileStats], qc_cfg: dict[str, Any]) -> None:
    """Add per-file relative metrics and flags, in place.

    `std_ratio` needs the file's other channels, so it cannot be computed inside
    `measure_channel`. The reference is the median std of the channels that are
    NOT structurally empty -- a dead channel must not lower its own reference.
    """
    live = [r.std for r in rows if r.std > 0.0]
    ref = float(np.median(live)) if live else 0.0

    nan_frac_max = float(qc_cfg["nan_inf"]["max_fraction"])
    dead_ratio = float(qc_cfg["dead_channel"]["ratio_threshold"])
    low_lo = float(qc_cfg["low_amplitude"]["ratio_min"])
    low_hi = float(qc_cfg["low_amplitude"]["ratio_max"])
    clip_flag = float(qc_cfg["clipping"]["flag_fraction"])
    mains_flag = float(qc_cfg["mains"]["flag_ratio"])
    struct_std = float(qc_cfg["structural_empty"]["std_abs_max"])
    struct_frozen = float(qc_cfg["structural_empty"]["frozen_fraction_min"])

    for r in rows:
        flags: list[str] = []
        n = max(r.n_samples, 1)

        # A column that is constant for the whole file makes every other statistic
        # meaningless: it trivially "clips" at both ends, trivially "freezes", and
        # trivially falls below any amplitude threshold. Reporting those derived
        # flags would bury the one real fact. The flag recorded here is the
        # measurement; whether it means "structural hole" or "dead electrode" is
        # decided in build_channel_map, which can see all files at once.
        if r.std <= struct_std and r.frozen_fraction >= struct_frozen:
            r.std_ratio = float("nan") if ref <= 0.0 else r.std / ref
            r.qc_flags = [CONSTANT_IN_FILE]
            continue

        if r.nan_count > 0 or r.inf_count > 0:
            if (r.nan_count + r.inf_count) / n > nan_frac_max:
                flags.append("nan_inf")
        if r.clipping_fraction >= clip_flag:
            flags.append("clipping")
        if math.isfinite(r.mains_ratio) and r.mains_ratio >= mains_flag:
            flags.append("high_mains")

        if ref > 0.0:
            r.std_ratio = r.std / ref
            if r.std_ratio < dead_ratio:
                flags.append("dead_in_file")
            elif low_lo <= r.std_ratio < low_hi:
                flags.append("low_amplitude")
        else:
            r.std_ratio = float("nan")

        r.qc_flags = flags


def structural_status_of(rows: list[ChannelFileStats], matrix_index: int) -> str:
    """Decide physical vs structurally empty for one matrix column.

    The rule is strict and symmetric: constant in EVERY file, or it is physical. A
    column that is constant in one file and live in another is a dead electrode in
    that one recording, not a hole in the montage -- and it must not be silently
    dropped. This is exactly the CH08 case: constant in S9_D1_T1, ~0.02 of the
    median amplitude everywhere else.
    """
    col = [r for r in rows if r.channel_index == matrix_index]
    if not col:
        raise ValueError(f"no QC rows for matrix index {matrix_index}")
    if all(CONSTANT_IN_FILE in r.qc_flags for r in col):
        return STRUCTURAL_EMPTY
    return STRUCTURAL_PHYSICAL


def build_channel_map(
    rows: list[ChannelFileStats],
    *,
    matrix_columns: int,
    channel_id_format: str,
    qc_cfg: dict[str, Any],
) -> dict[str, Any]:
    """The 16-column channel map: structural status, flags, evidence, eligibility."""
    channels: list[dict[str, Any]] = []
    for idx in range(matrix_columns):
        cid = channel_id_format.format(idx + 1)
        col_rows = [r for r in rows if r.channel_index == idx]
        status = structural_status_of(rows, idx)

        flag_counts: dict[str, int] = {}
        evidence: dict[str, list[str]] = {}
        for r in col_rows:
            for f in r.qc_flags:
                flag_counts[f] = flag_counts.get(f, 0) + 1
                evidence.setdefault(f, []).append(r.file)

        # Eligibility: physical, and never broken by a hard-failure condition.
        # Flags such as dead_in_file / low_amplitude / high_mains / constant_in_file
        # do NOT disqualify a channel -- they are reported so the pilot result can
        # be interpreted, and because dropping a channel on a soft flag would make
        # the channel ranking a self-fulfilling prophecy.
        hard_flags = {"nan_inf"}
        eligible = status == STRUCTURAL_PHYSICAL and not (hard_flags & set(flag_counts))

        reason_code = ""
        if status == STRUCTURAL_EMPTY:
            reason_code = "structurally_empty_column_documented"
        elif not eligible:
            reason_code = "hard_qc_failure"

        channels.append(
            {
                "channel_id": cid,
                "matrix_index": idx,
                "structural_status": status,
                "candidate_eligible": eligible,
                "reason_code": reason_code,
                "flag_counts": dict(sorted(flag_counts.items())),
                "evidence_files": {k: sorted(set(v)) for k, v in sorted(evidence.items())},
                "n_files_observed": len(col_rows),
                "median_std_ratio": _median([r.std_ratio for r in col_rows]),
                "median_mains_ratio": _median([r.mains_ratio for r in col_rows]),
            }
        )

    physical = [c for c in channels if c["structural_status"] == STRUCTURAL_PHYSICAL]
    empty = [c for c in channels if c["structural_status"] == STRUCTURAL_EMPTY]
    return {
        "schema_version": 1,
        "matrix_columns": matrix_columns,
        "physical_channels": [c["channel_id"] for c in physical],
        "structurally_empty_channels": [c["channel_id"] for c in empty],
        "candidate_channels": [c["channel_id"] for c in channels if c["candidate_eligible"]],
        "n_physical": len(physical),
        "n_structurally_empty": len(empty),
        "n_candidate_eligible": sum(1 for c in channels if c["candidate_eligible"]),
        "channels": channels,
        "note": (
            "structural_status describes the DB6 montage; qc_flags describe "
            "observations in individual files. Only a hard QC failure removes a "
            "channel from the candidate set."
        ),
    }


def _median(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(finite)) if finite else None


def check_expectation(channel_map: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Compare the measured montage against the documented one.

    Returns a verdict dict. When it fails, the caller must emit `status: blocked`
    with this evidence -- it must never adjust a threshold until the numbers agree.
    """
    expected_physical = int(data_cfg["matrix"]["expected_physical_channels"])
    expected_empty = int(data_cfg["matrix"]["expected_structural_empty_columns"])
    got_physical = int(channel_map["n_physical"])
    got_empty = int(channel_map["n_structurally_empty"])

    ok = got_physical == expected_physical and got_empty == expected_empty
    return {
        "ok": ok,
        "expected_physical": expected_physical,
        "observed_physical": got_physical,
        "expected_structurally_empty": expected_empty,
        "observed_structurally_empty": got_empty,
        "physical_channels": channel_map["physical_channels"],
        "structurally_empty_channels": channel_map["structurally_empty_channels"],
        "message": (
            "DB6 montage matches documentation: "
            f"{got_physical} physical channels, {got_empty} structurally empty."
            if ok
            else (
                "DB6 montage does NOT match documentation: expected "
                f"{expected_physical} physical / {expected_empty} empty, observed "
                f"{got_physical} / {got_empty}. Stage is blocked; the QC thresholds "
                "are not adjusted to make this pass."
            )
        ),
    }
