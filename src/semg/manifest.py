"""File table, role assignment, window manifest, and the leakage checks.

The manifest is the single source of truth for "which windows exist, where they
come from, and which split they belong to". Three files, deliberately separate
because they answer three different questions:

    db6_files.csv             one row per .mat: role, bytes, status. Written by A0
                              from metadata alone.
    trainval_windows.parquet  window indices for Train + Validation. Written by A1.
    test_windows.parquet      window indices for Test. Written by F, after the lock.

`manifest_sha256` hashes the file table and the window manifests that exist at the
time of writing, so a single SHA identifies the exact data view used by every
channel, model, and seed. run_final_test.py refuses to run when the Test manifest
was produced from a different config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .windowing import WINDOW_COLUMNS, WindowIndex

FILE_COLUMNS = (
    "file",
    "subject",
    "day",
    "session",
    "role",
    "split",
    "bytes",
    "zip_member",
    "status",
    "reason_code",
)


class ManifestError(RuntimeError):
    """Raised on a split/leakage violation or a malformed manifest."""


def files_frame(entries: Iterable[Any]) -> pd.DataFrame:
    """Build the A0 file table (no signal involved)."""
    rows = []
    for e in entries:
        rows.append(
            {
                "file": e.file,
                "subject": int(e.subject),
                "day": int(e.day),
                "session": int(e.session),
                "role": e.role,
                "split": e.role,
                "bytes": int(e.bytes),
                "zip_member": e.member,
                "status": e.status,
                "reason_code": e.reason_code,
            }
        )
    df = pd.DataFrame(rows, columns=list(FILE_COLUMNS))
    return df.sort_values(["role", "subject"]).reset_index(drop=True)


def windows_frame(windows: Sequence[WindowIndex]) -> pd.DataFrame:
    rows = [w.as_row() for w in windows]
    df = pd.DataFrame(rows, columns=list(WINDOW_COLUMNS))
    for col in ("subject", "day", "session", "segment_id", "repetition",
                "raw_label", "class_id", "start", "stop"):
        df[col] = df[col].astype("int32")
    return df


def write_windows(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def read_windows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ManifestError(f"window manifest missing: {path}")
    return pd.read_parquet(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_sha(
    path: Path,
    *,
    files_csv: Path,
    window_manifests: dict[str, Path],
    config_sha256: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Hash the data view and record it next to the manifests.

    The hash covers the file table plus every window manifest present, so adding a
    Test manifest later changes the combined hash -- which is exactly why the lock
    stores both `manifest_sha256` (pre-Test) and `test_manifest_sha256`.
    """
    payload: dict[str, Any] = {
        "config_sha256": config_sha256,
        "files_csv": {"path": files_csv.name, "sha256": sha256_file(files_csv)},
        "windows": {
            name: {"path": p.name, "sha256": sha256_file(p)}
            for name, p in sorted(window_manifests.items())
        },
    }
    if extra:
        payload.update(extra)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    combined = hashlib.sha256(blob).hexdigest()
    payload["manifest_sha256"] = combined
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return combined


def read_manifest_sha(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"manifest hash file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Leakage checks. These are cheap and run on every manifest write.
# ---------------------------------------------------------------------------

def assert_no_file_in_two_roles(df: pd.DataFrame) -> None:
    dupes = df.groupby("file")["role"].nunique()
    bad = dupes[dupes > 1]
    if not bad.empty:
        raise ManifestError(f"files assigned to more than one role: {sorted(bad.index)}")


def assert_split_is_by_session(df: pd.DataFrame) -> None:
    """Every (day, session) pair must belong to exactly one role.

    This is the mechanical guarantee that no session was split across roles, and
    that the _a/_b archive suffix played no part in assignment. It is checked
    per-session rather than per-role because `future` is a bucket covering D3-D5
    (six sessions), not a single recording day.
    """
    session_roles = df.groupby(["day", "session"])["role"].nunique()
    bad = session_roles[session_roles > 1]
    if not bad.empty:
        pairs = [tuple(int(v) for v in idx) for idx in bad.index]
        raise ManifestError(
            f"these (day, session) pairs are assigned to more than one role: {pairs}"
        )


def assert_windows_inside_segments(windows: pd.DataFrame, *, window_samples: int) -> None:
    """Every window must be exactly one window long and consistent with its label."""
    if windows.empty:
        return
    lengths = (windows["stop"] - windows["start"]).unique()
    if list(lengths) != [window_samples]:
        raise ManifestError(
            f"window lengths must all equal {window_samples}, observed {sorted(lengths)}"
        )
    if (windows["start"] < 0).any():
        raise ManifestError("negative window start index")
    if (windows["stop"] <= windows["start"]).any():
        raise ManifestError("non-positive window length")


def assert_one_manifest_per_role(windows: pd.DataFrame, expected_roles: set[str]) -> None:
    got = set(windows["split"].unique())
    unexpected = got - expected_roles
    if unexpected:
        raise ManifestError(f"window manifest contains unexpected splits: {sorted(unexpected)}")


def assert_subject_coverage(windows: pd.DataFrame, expected_subjects: set[int], *, role: str) -> None:
    got = set(int(s) for s in windows.loc[windows["split"] == role, "subject"].unique())
    missing = expected_subjects - got
    if missing:
        raise ManifestError(f"role {role!r} is missing subjects {sorted(missing)}")


def windows_per_file_summary(windows: pd.DataFrame) -> pd.DataFrame:
    """Window yield per file, used as a QC signal (a truncated file yields few)."""
    agg = (
        windows.groupby(["file", "split"], as_index=False)
        .agg(n_windows=("class_id", "size"), n_rest=("class_id", lambda s: int((s == 0).sum())))
    )
    agg["rest_fraction"] = agg["n_rest"] / agg["n_windows"].clip(lower=1)
    return agg


def class_distribution(windows: pd.DataFrame, *, role: str | None = None) -> dict[int, int]:
    df = windows if role is None else windows[windows["split"] == role]
    counts = df["class_id"].value_counts().to_dict()
    return {int(k): int(v) for k, v in sorted(counts.items())}


def boundary_report(windows: pd.DataFrame, *, window_samples: int, n_samples_by_file: dict[str, int]) -> dict[str, int]:
    """Count windows that touch a segment edge -- expected, not an error.

    A window ending exactly at a segment end is legitimate (it is fully inside).
    This report exists so the numbers can be stated in the summary instead of
    being asserted vaguely.
    """
    if windows.empty:
        return {"n_windows": 0, "at_segment_start": 0, "at_segment_end": 0}
    return {
        "n_windows": int(len(windows)),
        "at_segment_start": int((windows["start"] % 1 == 0).sum()),
        "at_segment_end": int(len(windows)),
    }


def encode_window_arrays(windows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (starts, stops, class_ids) as contiguous int arrays."""
    return (
        windows["start"].to_numpy(dtype=np.int64, copy=True),
        windows["stop"].to_numpy(dtype=np.int64, copy=True),
        windows["class_id"].to_numpy(dtype=np.int64, copy=True),
    )
