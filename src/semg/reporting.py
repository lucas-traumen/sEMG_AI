"""Writers: CSVs, JSON reports, the RESULT_JSON line, and the markdown summary.

The RESULT_JSON contract is the interface a Kaggle orchestrator (and a human
reading the kernel log) relies on:

    exactly ONE line starting with "RESULT_JSON " per stage, containing
    {schema_version, stage, status, git_commit, config_sha256, manifest_sha256,
     metrics, artifacts, warnings, errors}

"Exactly one" is enforced by `emit_result`, which refuses to be called twice in a
process. A stage that crashes before emitting produces no line at all -- the
orchestrator sees a missing result and fails the stage, which is the correct
outcome. A stage that prints a line and then dies would be much worse: it would
look successful.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

RESULT_PREFIX = "RESULT_JSON "
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"

_emitted = False


class ReportError(RuntimeError):
    """Raised on a malformed or duplicate report."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(cwd: Path | None = None) -> str:
    """Current commit, or 'unknown' outside a repo.

    A report that cannot name the code it ran is not reproducible, so this is
    recorded even when it is unhelpful -- 'unknown' is visible, silence is not.

    A Kaggle kernel has no `.git`, and it runs a checked-out bundle rather than
    a working tree, so the bundle records the commit it was packed from in
    `SEMG_GIT_COMMIT`. That is consulted first: the bundle hash names the code
    far more precisely than a bare `git rev-parse` could.
    """
    override = os.environ.get("SEMG_GIT_COMMIT", "").strip()
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def host_label() -> str:
    """Where this run happened: 'kaggle' inside a Kaggle kernel, else 'local'.

    The two hosts mount the raw data differently -- a read-only dataset
    directory versus the local zip cache -- so the label is part of the hashed
    manifest payload. A hash minted on one host is then not silently reused as
    a claim about the other, and a reader can tell a laptop draft from a GPU run.

    `SEMG_HOST` overrides the detection for a host that looks like neither.
    """
    override = os.environ.get("SEMG_HOST", "").strip()
    if override:
        return override
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").is_dir():
        return "kaggle"
    return "local"


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path, *, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if columns is not None:
        df = df.reindex(columns=list(columns))
    df.to_csv(path, index=False)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise ReportError(f"expected JSON file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


@dataclass
class StageReport:
    """Accumulates one stage's outcome and emits it exactly once."""

    stage: str
    config_sha256: str
    manifest_sha256: str = ""
    git_commit: str = ""
    status: str = STATUS_OK
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.status = STATUS_FAILED

    def block(self, message: str) -> None:
        self.errors.append(message)
        self.status = STATUS_BLOCKED

    def artifact(self, name: str, path: Path) -> None:
        self.artifacts[name] = str(path)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": self.stage,
            "status": self.status,
            "git_commit": self.git_commit or git_commit(),
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "errors": self.errors,
            "emitted_at": _now(),
        }


def emit_result(report: StageReport, *, reports_dir: Path | None = None) -> dict[str, Any]:
    """Write reports/<stage>.json and print the single RESULT_JSON line."""
    global _emitted
    if _emitted:
        raise ReportError(
            "RESULT_JSON has already been emitted in this process; a stage must "
            "produce exactly one result line"
        )
    payload = report.payload()
    if reports_dir is not None:
        write_json(payload, reports_dir / f"{report.stage}.json")
    line = RESULT_PREFIX + json.dumps(payload, sort_keys=True, default=_default)
    if "\n" in line:
        raise ReportError("RESULT_JSON payload must be a single line")
    _emitted = True
    print(line, flush=True)
    return payload


def reset_emitter() -> None:
    """Test hook: allow a second emit in a fresh logical stage."""
    global _emitted
    _emitted = False


# ---------------------------------------------------------------------------
# Markdown summary (§7.5)
# ---------------------------------------------------------------------------

def markdown_table(
    headers: Sequence[str], rows: Iterable[Sequence[Any]], *, aligns: Sequence[str] | None = None
) -> str:
    rows = [list(r) for r in rows]
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("| " + " | ".join(aligns) + " |")
    for r in rows:
        out.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return "\n".join(out)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def fmt_pm(mean: float, sd: float, *, digits: int = 4) -> str:
    """`mean ± sd`, with n/a for a missing spread instead of a fake zero."""
    if mean != mean:
        return "n/a"
    if sd != sd:
        return f"{mean:.{digits}f} ± n/a"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
