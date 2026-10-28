"""Stage A0: enumerate the dataset from metadata only.

"Read the file" and "read the file's name" are different operations, and this
module exists to keep them apart. Inventory touches the zip central directory (or
a directory listing), which yields the member name and its uncompressed byte size.
It never calls loadmat, never decompresses a payload, and never imports scipy.

That distinction is what lets Stage A0 be honest about all 100 files while Stages
B-E remain unable to see a single Test sample.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Iterable

# DB6 filenames look like S1_D1_T1.mat: subject 1, day 1, session 1.
_NAME_RE = re.compile(r"^S(?P<subject>\d+)_D(?P<day>\d+)_T(?P<session>\d+)$")

ROLE_TRAIN = "train"
ROLE_VALIDATION = "validation"
ROLE_RESERVE = "reserve"
ROLE_TEST = "test"
ROLE_FUTURE = "future"


@dataclass(frozen=True)
class FileEntry:
    """One DB6 .mat file, described without reading its contents."""

    file: str                 # e.g. S1_D1_T1.mat
    subject: int
    day: int
    session: int
    role: str
    bytes: int
    location: str             # zip path or directory path holding the member
    member: str               # name inside the container
    status: str = "ok"        # ok | excluded
    reason_code: str = ""

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["split"] = self.role
        return row


class InventoryError(RuntimeError):
    """Raised when the dataset is missing or incomplete."""


def parse_name(name: str) -> tuple[int, int, int]:
    """`S3_D2_T2.mat` -> (3, 2, 2). Raises on anything else."""
    stem = Path(name).stem
    m = _NAME_RE.match(stem)
    if not m:
        raise InventoryError(f"unexpected DB6 filename: {name!r} (expected S<n>_D<d>_T<s>.mat)")
    return int(m["subject"]), int(m["day"]), int(m["session"])


def role_for(day: int, session: int, roles_cfg: dict) -> str:
    """Map (day, session) to a role.

    Only the day/session letters decide the split. The _a / _b archive suffix is a
    download artefact and is never consulted (AGENT.md rule 1).
    """
    train = roles_cfg[ROLE_TRAIN]
    if day == int(train["day"]) and session == int(train["session"]):
        return ROLE_TRAIN
    val = roles_cfg[ROLE_VALIDATION]
    if day == int(val["day"]) and session == int(val["session"]):
        return ROLE_VALIDATION
    res = roles_cfg[ROLE_RESERVE]
    if day == int(res["day"]) and session == int(res["session"]):
        return ROLE_RESERVE
    test = roles_cfg[ROLE_TEST]
    if day == int(test["day"]) and session == int(test["session"]):
        return ROLE_TEST
    fut = roles_cfg[ROLE_FUTURE]
    if day in [int(d) for d in fut["day"]] and session in [int(s) for s in fut["session"]]:
        return ROLE_FUTURE
    raise InventoryError(f"no role configured for D{day}_T{session}")


def inventory_from_zips(zip_paths: Iterable[Path], roles_cfg: dict) -> list[FileEntry]:
    """Scan zip central directories. Name + size only; no decompression."""
    entries: list[FileEntry] = []
    for zpath in sorted(zip_paths):
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                base = os.path.basename(info.filename)
                # Drop macOS resource-fork junk (__MACOSX/._*.mat) if a badly
                # packaged archive ever contains it: it is a .mat by extension but
                # not a recording.
                if "/__MACOSX/" in info.filename or base.startswith("._"):
                    continue
                if base.endswith("/"):
                    continue
                name = Path(base).name
                if Path(name).suffix != ".mat":
                    continue
                subject, day, session = parse_name(name)
                entries.append(
                    FileEntry(
                        file=name,
                        subject=subject,
                        day=day,
                        session=session,
                        role=role_for(day, session, roles_cfg),
                        bytes=int(info.file_size),
                        location=str(zpath),
                        member=info.filename,
                    )
                )
    return entries


def inventory_from_dir(root: Path, roles_cfg: dict) -> list[FileEntry]:
    """Scan a mounted directory tree (the Kaggle layout). No .mat opened."""
    entries: list[FileEntry] = []
    for path in sorted(root.rglob("*.mat")):
        name = path.name
        if name.startswith("._"):
            continue
        subject, day, session = parse_name(name)
        entries.append(
            FileEntry(
                file=name,
                subject=subject,
                day=day,
                session=session,
                role=role_for(day, session, roles_cfg),
                bytes=int(path.stat().st_size),
                location=str(path.parent),
                member=name,
            )
        )
    return entries


def apply_exclusion_rule(entries: list[FileEntry], rule: dict) -> list[FileEntry]:
    """Mark files that are implausibly small for their role, using bytes only.

    The threshold is relative to the role's median, so it survives a different
    download but still catches a truncated recording. Decided before any Test
    signal exists, which is the whole point: the exclusion cannot be a reaction to
    Test accuracy.
    """
    if not rule.get("enabled", False):
        return list(entries)

    role = str(rule.get("role", ROLE_TEST))
    ratio = float(rule["ratio_threshold"])
    sizes = [e.bytes for e in entries if e.role == role and e.status == "ok"]
    if not sizes:
        raise InventoryError(f"exclusion rule targets role {role!r} but no such files were found")
    sizes.sort()
    mid = len(sizes) // 2
    median = float(sizes[mid]) if len(sizes) % 2 else (sizes[mid - 1] + sizes[mid]) / 2.0
    cutoff = ratio * median

    out: list[FileEntry] = []
    for e in entries:
        if e.role == role and e.bytes < cutoff:
            out.append(
                replace(e, status="excluded", reason_code=f"{Path(e.file).stem}_truncated")
            )
        else:
            out.append(e)
    return out


def validate_counts(entries: list[FileEntry], roles_cfg: dict, *, expected_total: int = 100) -> dict:
    """Check the inventory against the declared expectation.

    Returns a summary dict; raises InventoryError when the dataset is not the one
    the config describes. Never returns a "patched" view.
    """
    if len(entries) != expected_total:
        raise InventoryError(
            f"expected {expected_total} DB6 .mat files, found {len(entries)}. "
            "Re-attach dataset trilucas/semg-db6 (version pinned) or check "
            "data/raw/db6/*.zip."
        )
    names = [e.file for e in entries]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise InventoryError(f"duplicate DB6 files in inventory: {sorted(dupes)}")

    per_role: dict[str, dict[str, int]] = {}
    for role, spec in roles_cfg.items():
        got = [e for e in entries if e.role == role]
        excluded = [e for e in got if e.status != "ok"]
        expected = int(spec["files_expected"])
        if len(got) != expected:
            raise InventoryError(
                f"role {role!r}: expected {expected} files, found {len(got)}"
            )
        per_role[role] = {
            "present": len(got),
            "excluded": len(excluded),
            "usable": len(got) - len(excluded),
        }
    return {"total": len(entries), "per_role": per_role}
