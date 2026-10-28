"""Stage A0 + A1 orchestration.

A0 (inventory) and A1 (QC + windowing) live in one script but are separate stages
with separate RESULT_JSON lines, because they have different trust properties:

    A0 -- reads names and byte sizes for all 100 files. Touches no signal.
    A1 -- reads the 20 Train+Validation files. Touches nothing else.

Keeping A0 runnable on its own is what lets `run_preflight.py --stage inventory`
prove, before any signal exists, which Test file is excluded and why.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import config as config_mod
from . import gestures as gestures_mod
from . import inventory as inventory_mod
from . import manifest as manifest_mod
from . import qc as qc_mod
from . import reporting
from . import windowing
from .audit import AuditLog
from .config import Config
from .matfile_source import build_source


@dataclass
class StageContext:
    """Shared plumbing for every stage script."""

    config: Config
    root: Path
    report: reporting.StageReport
    audit: AuditLog

    @property
    def manifest_dir(self) -> Path:
        return self.config.path("manifest_dir")

    @property
    def results_dir(self) -> Path:
        return self.config.path("results_dir")

    @property
    def scalers_dir(self) -> Path:
        return self.config.path("scalers_dir")

    @property
    def checkpoints_dir(self) -> Path:
        return self.config.path("checkpoints_dir")

    @property
    def reports_dir(self) -> Path:
        return self.config.path("reports_dir")

    @property
    def figures_dir(self) -> Path:
        return self.config.path("figures_dir")

    @property
    def files_csv(self) -> Path:
        return self.manifest_dir / "db6_files.csv"

    @property
    def trainval_path(self) -> Path:
        return self.manifest_dir / "trainval_windows.parquet"

    @property
    def test_windows_path(self) -> Path:
        return self.manifest_dir / "test_windows.parquet"

    @property
    def manifest_sha_path(self) -> Path:
        return self.manifest_dir / "manifest_sha256.txt"

    @property
    def lock_path(self) -> Path:
        return self.results_dir / "selection_lock.json"

    @property
    def run_index_path(self) -> Path:
        return self.results_dir / "run_index.json"

    @property
    def audit_path(self) -> Path:
        return self.results_dir / "access_audit.jsonl"


def make_context(config_dir: str | Path, root: Path, stage: str) -> StageContext:
    cfg = config_mod.load(root / Path(config_dir))
    reporting.ensure_dirs(
        cfg.path("manifest_dir"),
        cfg.path("results_dir"),
        cfg.path("scalers_dir"),
        cfg.path("checkpoints_dir"),
        cfg.path("reports_dir"),
        cfg.path("figures_dir"),
    )
    audit = AuditLog(cfg.path("results_dir") / "access_audit.jsonl")
    report = reporting.StageReport(
        stage=stage,
        config_sha256=cfg.sha256,
        git_commit=reporting.git_commit(root),
    )
    return StageContext(config=cfg, root=root, report=report, audit=audit)


# ---------------------------------------------------------------------------
# A0
# ---------------------------------------------------------------------------

def run_inventory(ctx: StageContext) -> list[inventory_mod.FileEntry]:
    cfg = ctx.config
    roles = cfg.data["roles"]
    zip_dir = ctx.root / Path(cfg.data["dataset"]["local_zip_dir"])
    raw_dir = cfg.path("raw_zip_dir")

    zips = sorted(raw_dir.glob("*.zip")) if raw_dir.is_dir() else []
    mounted = ctx.root / "data/raw/db6_mount"
    if mounted.is_dir() and any(mounted.rglob("*.mat")):
        entries = inventory_mod.inventory_from_dir(mounted, roles)
        ctx.audit.note_inventory(stage="inventory", files=[e.file for e in entries], source=str(mounted))
    elif zips:
        entries = inventory_mod.inventory_from_zips(zips, roles)
        ctx.audit.note_inventory(stage="inventory", files=[e.file for e in entries], source=f"{len(zips)} zip archives")
    else:
        raise inventory_mod.InventoryError(
            f"no DB6 data found: expected *.zip in {raw_dir} (symlinks to {zip_dir}) "
            f"or a mounted tree in {mounted}"
        )

    entries = inventory_mod.apply_exclusion_rule(entries, cfg.data["exclusion_rule"])
    summary = inventory_mod.validate_counts(entries, roles)

    df = manifest_mod.files_frame(entries)
    manifest_mod.assert_no_file_in_two_roles(df)
    manifest_mod.assert_split_is_by_session(df)
    ctx.files_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ctx.files_csv, index=False)

    sha = manifest_mod.write_manifest_sha(
        ctx.manifest_sha_path,
        files_csv=ctx.files_csv,
        window_manifests=_existing_windows(ctx),
        config_sha256=cfg.sha256,
        extra={"stage_recorded": "inventory"},
    )
    ctx.report.manifest_sha256 = sha

    excluded = [e for e in entries if e.status != "ok"]
    ctx.report.metrics.update(
        {
            "files_total": summary["total"],
            "per_role": summary["per_role"],
            "excluded": [{"file": e.file, "bytes": e.bytes, "reason_code": e.reason_code} for e in excluded],
            "usable_test_files": summary["per_role"]["test"]["usable"],
            "signal_reads": 0,
        }
    )
    ctx.report.artifact("db6_files", ctx.files_csv)
    ctx.report.artifact("manifest_sha256", ctx.manifest_sha_path)
    if len(excluded) != len(cfg.data["exclusion_rule"].get("expected_exclusions", [])):
        ctx.report.warn(
            f"exclusion rule marked {len(excluded)} file(s); "
            f"config expects {len(cfg.data['exclusion_rule'].get('expected_exclusions', []))}"
        )
    return entries


def _existing_windows(ctx: StageContext) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if ctx.trainval_path.is_file():
        out["trainval"] = ctx.trainval_path
    if ctx.test_windows_path.is_file():
        out["test"] = ctx.test_windows_path
    return out


def build_source_for(entries: list[inventory_mod.FileEntry], *, ctx: StageContext, stage: str):
    """Construct a MatFileSource that writes every read to the audit log."""
    variables = tuple(ctx.config.data["signals"]["variables"])
    return build_source(entries, audit=ctx.audit, stage=stage, variables=variables)


def load_entries(ctx: StageContext) -> list[inventory_mod.FileEntry]:
    """Re-read the A0 file table so later stages never re-invent the inventory."""
    import pandas as pd

    if not ctx.files_csv.is_file():
        raise inventory_mod.InventoryError(
            f"{ctx.files_csv} missing; run scripts/run_preflight.py --stage inventory first"
        )
    df = pd.read_csv(ctx.files_csv)
    return [
        inventory_mod.FileEntry(
            file=row.file,
            subject=int(row.subject),
            day=int(row.day),
            session=int(row.session),
            role=str(row.role),
            bytes=int(row.bytes),
            location=str(_location_for(ctx, row.file)),
            member=str(row.zip_member),
            status=str(row.status),
            reason_code=str(row.reason_code) if isinstance(row.reason_code, str) else "",
        )
        for row in df.itertuples()
    ]


_LOCATION_CACHE: dict[str, str] = {}


def _location_for(ctx: StageContext, file: str) -> str:
    """Resolve where a .mat lives, from the zip central directories."""
    if not _LOCATION_CACHE:
        raw_dir = ctx.config.path("raw_zip_dir")
        mounted = ctx.root / "data/raw/db6_mount"
        if mounted.is_dir() and any(mounted.rglob("*.mat")):
            for path in mounted.rglob("*.mat"):
                _LOCATION_CACHE[path.name] = str(path.parent)
        else:
            import zipfile

            for zpath in sorted(raw_dir.glob("*.zip")):
                with zipfile.ZipFile(zpath) as zf:
                    for info in zf.infolist():
                        base = Path(info.filename).name
                        if base.endswith(".mat"):
                            _LOCATION_CACHE[base] = str(zpath)
    try:
        return _LOCATION_CACHE[file]
    except KeyError as exc:
        raise inventory_mod.InventoryError(f"{file}: cannot locate the archive holding it") from exc


def member_for(ctx: StageContext, file: str) -> str:
    raw_dir = ctx.config.path("raw_zip_dir")
    import zipfile

    for zpath in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if Path(info.filename).name == file:
                    return info.filename
    return file


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------

@dataclass
class TrainValData:
    """Everything A1 produced, for reuse by later stages."""

    windows: Any                     # DataFrame
    channel_map: dict[str, Any]
    qc_rows: list[dict[str, Any]]
    scalers: dict[str, windowing.StandardScaler]
    expectation: dict[str, Any]


def _trainval_files(entries: list[inventory_mod.FileEntry]) -> list[inventory_mod.FileEntry]:
    return [e for e in entries if e.role in (inventory_mod.ROLE_TRAIN, inventory_mod.ROLE_VALIDATION) and e.status == "ok"]


def run_qc(ctx: StageContext, entries: list[inventory_mod.FileEntry]) -> TrainValData:
    cfg = ctx.config
    label_map = gestures_mod.build(cfg.gestures)
    qc_cfg = cfg.pipeline["qc"]
    files = _trainval_files(entries)
    if not files:
        raise inventory_mod.InventoryError("no Train/Validation files found in the file table")

    variables = tuple(cfg.data["signals"]["variables"])
    source = build_source(files, audit=ctx.audit, stage="qc", variables=variables)

    matrix_columns = cfg.matrix_columns
    raw_to_class = label_map.raw_to_class
    all_windows: list[windowing.WindowIndex] = []
    qc_rows: list[qc_mod.ChannelFileStats] = []
    channel_sum: dict[int, float] = {i: 0.0 for i in range(matrix_columns)}
    channel_sumsq: dict[int, float] = {i: 0.0 for i in range(matrix_columns)}
    channel_count: dict[int, int] = {i: 0 for i in range(matrix_columns)}

    for entry in files:
        data = source.load(entry.file, reason="Stage A1 QC + windowing (Train/Validation only)")
        emg = np.asarray(data["emg"], dtype=np.float64)
        if emg.shape[1] != matrix_columns:
            raise ValueError(f"{entry.file}: emg has {emg.shape[1]} columns, config expects {matrix_columns}")
        labels = np.asarray(data["restimulus"]).reshape(-1)
        reps = np.asarray(data["rerepetition"]).reshape(-1)
        gestures_mod.validate_against_data(label_map, np.unique(labels), file=entry.file)

        rows = [
            qc_mod.measure_channel(
                x=emg[:, idx],
                restimulus=labels,
                file=entry.file,
                subject=entry.subject,
                channel_index=idx,
                channel_id=cfg.channel_id(idx),
                fs_hz=cfg.fs_hz,
                qc_cfg=qc_cfg,
            )
            for idx in range(matrix_columns)
        ]
        qc_mod.annotate_file(rows, qc_cfg)
        qc_rows.extend(rows)

        # Scaler statistics accumulate over Train only. Validation contributes
        # windows, never normalisation statistics.
        if entry.role == inventory_mod.ROLE_TRAIN:
            for idx in range(matrix_columns):
                col = emg[:, idx]
                col = col[np.isfinite(col)]
                channel_sum[idx] += float(col.sum())
                channel_sumsq[idx] += float(np.square(col).sum())
                channel_count[idx] += int(col.size)

        all_windows.extend(
            windowing.build_window_index(
                file=entry.file,
                subject=entry.subject,
                day=entry.day,
                session=entry.session,
                split=entry.role,
                restimulus=labels,
                rerepetition=reps,
                raw_to_class=raw_to_class,
                window_samples=cfg.window_samples,
                hop_samples=cfg.hop_samples,
            )
        )

    channel_map = qc_mod.build_channel_map(
        qc_rows,
        matrix_columns=matrix_columns,
        channel_id_format=cfg.data["matrix"]["channel_id_format"],
        qc_cfg=qc_cfg,
    )
    expectation = qc_mod.check_expectation(channel_map, cfg.data)

    windows_df = manifest_mod.windows_frame(all_windows)
    manifest_mod.assert_windows_inside_segments(windows_df, window_samples=cfg.window_samples)
    manifest_mod.assert_one_manifest_per_role(windows_df, {"train", "validation"})

    scalers: dict[str, windowing.StandardScaler] = {}
    eps = float(cfg.pipeline["normalization"]["eps"])
    for idx in range(matrix_columns):
        cid = cfg.channel_id(idx)
        n = channel_count[idx]
        if n == 0:
            continue
        mean = channel_sum[idx] / n
        var = max(channel_sumsq[idx] / n - mean * mean, 0.0)
        scaler = windowing.StandardScaler(mean=mean, std=float(np.sqrt(var)), channel=cid, eps=eps)
        scaler.n_samples_fit = n  # type: ignore[attr-defined]
        scalers[cid] = scaler
        reporting.write_json(scaler.to_dict(), ctx.scalers_dir / f"{cid}.json")

    manifest_mod.write_windows(windows_df, ctx.trainval_path)
    qc_df_rows = [r.as_row() for r in qc_rows]
    reporting.write_csv(qc_df_rows, ctx.results_dir / "channel_qc.csv")
    reporting.write_json(channel_map, ctx.results_dir / "channel_map.json")
    per_file = manifest_mod.windows_per_file_summary(windows_df)
    per_file.to_csv(ctx.results_dir / "qc_windows_per_file.csv", index=False)

    sha = manifest_mod.write_manifest_sha(
        ctx.manifest_sha_path,
        files_csv=ctx.files_csv,
        window_manifests=_existing_windows(ctx),
        config_sha256=cfg.sha256,
        extra={
            "stage_recorded": "qc",
            "channel_map_sha256": manifest_mod.sha256_file(ctx.results_dir / "channel_map.json"),
            "host": reporting.host_label(),
        },
    )
    ctx.report.manifest_sha256 = sha

    return TrainValData(
        windows=windows_df,
        channel_map=channel_map,
        qc_rows=qc_df_rows,
        scalers=scalers,
        expectation=expectation,
    )


def summarise_qc(ctx: StageContext, data: TrainValData) -> None:
    cfg = ctx.config
    qc_cfg = cfg.pipeline["qc"]
    windows = data.windows
    per_file = windows.groupby("file").size()
    too_few = per_file[per_file < int(qc_cfg["min_windows_per_file"])]
    classes = manifest_mod.class_distribution(windows)
    n_windows = int(len(windows))

    ctx.report.metrics.update(
        {
            "files_qc": int(windows["file"].nunique()),
            "windows_trainval": n_windows,
            "windows_by_split": {k: int(v) for k, v in windows["split"].value_counts().items()},
            "class_distribution": classes,
            "rest_fraction": float(classes.get(0, 0) / max(n_windows, 1)),
            "n_physical": int(data.channel_map["n_physical"]),
            "n_structurally_empty": int(data.channel_map["n_structurally_empty"]),
            "physical_channels": data.channel_map["physical_channels"],
            "structurally_empty_channels": data.channel_map["structurally_empty_channels"],
            "candidate_channels": data.channel_map["candidate_channels"],
            "montage_expectation": data.expectation,
            "windows_per_file_min": int(per_file.min()),
            "windows_per_file_median": float(per_file.median()),
            "files_below_min_windows": sorted(too_few.index.tolist()),
            "flag_counts": {
                ch["channel_id"]: ch["flag_counts"] for ch in data.channel_map["channels"]
            },
        }
    )
    ctx.report.artifact("trainval_windows", ctx.trainval_path)
    ctx.report.artifact("channel_qc", ctx.results_dir / "channel_qc.csv")
    ctx.report.artifact("channel_map", ctx.results_dir / "channel_map.json")
    ctx.report.artifact("manifest_sha256", ctx.manifest_sha_path)

    if not data.expectation["ok"]:
        ctx.report.block(str(data.expectation["message"]))
    if len(too_few):
        ctx.report.warn(
            f"{len(too_few)} file(s) yielded fewer than {qc_cfg['min_windows_per_file']} windows: "
            f"{sorted(too_few.index.tolist())}"
        )


def clean_stale(ctx: StageContext) -> None:
    """Remove derived artefacts from a previous run so a fresh A1 is truly fresh."""
    for path in (
        ctx.trainval_path,
        ctx.test_windows_path,
        ctx.manifest_sha_path,
        ctx.results_dir / "channel_map.json",
        ctx.results_dir / "channel_qc.csv",
        ctx.results_dir / "qc_windows_per_file.csv",
        ctx.lock_path,
    ):
        if path.exists():
            path.unlink()
    if ctx.scalers_dir.is_dir():
        shutil.rmtree(ctx.scalers_dir, ignore_errors=True)
        ctx.scalers_dir.mkdir(parents=True, exist_ok=True)
