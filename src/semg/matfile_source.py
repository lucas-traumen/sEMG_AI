"""Open one DB6 .mat file, from a zip archive or a mounted directory.

Two hard rules live here:

1. Only the declared variables are parsed (`variable_names=`). A full DB6 file has
   48 accelerometer columns we never use; parsing them triples the read cost for
   nothing.

2. Every open is recorded in the audit log by this module, not by its callers. The
   stage name and role come in as arguments and are written straight to the log,
   so a stage cannot read a Test file and forget to mention it.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.io import loadmat

from .audit import AuditLog, NullAudit

# Variables the pipeline needs.
#
# Two groups, because DB6 stores them at different granularities:
#   PER_SAMPLE -- one value per time step; must be length N like `emg`.
#   PER_FILE   -- a single scalar describing the recording (subject id, which day
#                 the file was recorded). These are metadata, not signals.
#
# Conflating the two is how a per-file scalar gets silently broadcast to 1.4M
# samples, or how a genuine truncation gets missed because the check was relaxed
# for everything.
PER_SAMPLE_VARIABLES = (
    "emg",
    "restimulus",
    "rerepetition",
    "stimulus",
    "repetition",
    "object",
)
PER_FILE_VARIABLES = ("daytesting", "subj")
DEFAULT_VARIABLES = PER_SAMPLE_VARIABLES + PER_FILE_VARIABLES


class AuditLike(Protocol):
    def note_signal_read(self, *, stage: str, file: str, role: str, reason: str) -> None: ...


class MatFileSource:
    """Resolves a DB6 file name to a parsed dict of arrays.

    `entries` is the Stage A0 inventory: it already knows where each file lives
    (zip path + member, or directory). Nothing here re-derives that.
    """

    def __init__(
        self,
        entries: dict[str, Any],
        *,
        audit: AuditLike | None = None,
        stage: str = "unknown",
        variables: tuple[str, ...] = DEFAULT_VARIABLES,
    ):
        self._entries = entries
        self._audit = audit if audit is not None else NullAudit()
        self._stage = stage
        self._variables = variables
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def roles(self) -> set[str]:
        return {e.role for e in self._entries.values()}

    def names(self, role: str | None = None, *, usable_only: bool = True) -> list[str]:
        out = []
        for name, entry in sorted(self._entries.items()):
            if role is not None and entry.role != role:
                continue
            if usable_only and entry.status != "ok":
                continue
            out.append(name)
        return out

    def role_of(self, name: str) -> str:
        return self._entries[name].role

    def load(self, name: str, *, reason: str) -> dict[str, np.ndarray]:
        """Parse one .mat and record the read. Cached per process."""
        if name in self._cache:
            return self._cache[name]
        entry = self._entries[name]
        arrays = self._read(entry)
        self._audit.note_signal_read(
            stage=self._stage, file=name, role=entry.role, reason=reason
        )
        self._cache[name] = arrays
        return arrays

    def _read(self, entry: Any) -> dict[str, np.ndarray]:
        location = Path(entry.location)
        if location.suffix == ".zip":
            with zipfile.ZipFile(location) as zf:
                with zf.open(entry.member) as fh:
                    raw = fh.read()
            mat = loadmat(io.BytesIO(raw), variable_names=list(self._variables), squeeze_me=False)
        else:
            mat = loadmat(
                str(location / entry.member),
                variable_names=list(self._variables),
                squeeze_me=False,
            )

        out: dict[str, np.ndarray] = {}
        for key in self._variables:
            if key not in mat:
                raise KeyError(f"{entry.file}: variable {key!r} missing from .mat")
            arr = np.asarray(mat[key])
            if key in PER_FILE_VARIABLES:
                # Keep the raw array (usually shape (1,1)); the caller reads .item().
                out[key] = arr
                continue
            # Label arrays come back as (N,1); flatten to (N,). `emg` stays 2-D.
            if key != "emg":
                arr = arr.reshape(-1)
            out[key] = arr

        emg = out["emg"]
        if emg.ndim != 2:
            raise ValueError(f"{entry.file}: emg must be 2-D, got shape {emg.shape}")
        n = emg.shape[0]
        for key in PER_SAMPLE_VARIABLES:
            arr = out[key]
            if arr.shape[0] != n:
                raise ValueError(
                    f"{entry.file}: {key} has {arr.shape[0]} samples but emg has {n}"
                )
        return out

    def clear_cache(self) -> None:
        self._cache.clear()


def build_source(
    entries: list[Any],
    *,
    audit: AuditLog | None,
    stage: str,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> MatFileSource:
    return MatFileSource(
        {e.file: e for e in entries}, audit=audit, stage=stage, variables=variables
    )
