"""Append-only record of which .mat files were opened, in which stage, and why.

The rule this enforces is the one that makes the Test number trustworthy: no Test,
Reserve, or Future signal may be read before selection_lock.json exists. A rule
nobody can check is a preference; this log is what turns it into an auditable
fact.

Two distinct events are recorded:

    inventory    -- the file's name and byte size were read (zip central directory
                    or a directory listing). NOT a signal read. Stage A0 does only
                    this, for all 100 files.
    signal_read  -- the .mat was decompressed and parsed. Recorded with the role
                    and the stage, so tests/test_no_peek.py can assert that
                    signal_read appears only for train/validation before the lock
                    and only for test after it.

Every read goes through `AuditLog.note_signal_read`, which is called by
matfile_source -- never by a caller. Centralising it means a future module cannot
quietly bypass the log by opening a file itself.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INVENTORY = "inventory"
SIGNAL_READ = "signal_read"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditLog:
    """Thread-safe, append-only JSONL audit log."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def note_inventory(self, *, stage: str, files: list[str], source: str) -> None:
        """One line recording that N file names and sizes were enumerated."""
        self._append(
            {
                "ts": _now(),
                "kind": INVENTORY,
                "stage": stage,
                "n_files": len(files),
                "source": source,
            }
        )

    def note_signal_read(self, *, stage: str, file: str, role: str, reason: str) -> None:
        """One line per .mat actually decompressed."""
        self._append(
            {
                "ts": _now(),
                "kind": SIGNAL_READ,
                "stage": stage,
                "file": file,
                "role": role,
                "reason": reason,
            }
        )

    # -- readers used by the tests and by run_final_test.py -------------------
    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def signal_reads(self, *, stage: str | None = None) -> list[dict[str, Any]]:
        return [
            r
            for r in self.records()
            if r.get("kind") == SIGNAL_READ and (stage is None or r.get("stage") == stage)
        ]

    def inventory_records(self) -> list[dict[str, Any]]:
        return [r for r in self.records() if r.get("kind") == INVENTORY]

    def roles_read(self, *, stage: str | None = None) -> set[str]:
        return {r["role"] for r in self.signal_reads(stage=stage)}


class NullAudit:
    """Drop-in no-op for unit tests that do not care about the log."""

    def note_inventory(self, **_kwargs: Any) -> None:
        return None

    def note_signal_read(self, **_kwargs: Any) -> None:
        return None

    def records(self) -> list[dict[str, Any]]:
        return []

    def signal_reads(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def inventory_records(self) -> list[dict[str, Any]]:
        return []

    def roles_read(self, **_kwargs: Any) -> set[str]:
        return set()
