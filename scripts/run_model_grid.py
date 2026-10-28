#!/usr/bin/env python
"""Stage D: joint channel x model grid.

Confirmed channels x every grid model x the confirm seeds.

The grid is joint on purpose. Choosing the channel with one architecture and the
architecture with that channel, then combining the two winners, is the classic way
to select a pair that was never actually the best. Here every cell is trained, and
the pair is chosen from measured cells.

Idempotency reuses any cell already trained in Stage B/C: `cnn_base` on a confirmed
channel with a confirm seed may already exist, and re-training it would produce a
different number for the same configuration.

Emits `results/selection_lock.json` — after this, the channel, model, seeds,
scaler hashes and manifest hash are frozen, and Test is unblocked.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semg import dataset, manifest, ranking, reporting, runner, stages  # noqa: E402

GRID_COLUMNS = runner.SUMMARY_COLUMNS + ("grid_source",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)

    ctx = stages.make_context(args.config_dir, ROOT, "channel_model_grid")
    config = ctx.config
    try:
        stages.load_entries(ctx)
        shortlist_path = ctx.results_dir / "channel_shortlist.json"
        if not shortlist_path.is_file():
            raise FileNotFoundError(f"{shortlist_path} missing; run Stage C first")
        shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
        channels = list(shortlist["confirmed_channels"])
        if not channels:
            raise ValueError("confirmed channel list is empty")

        models = [str(m) for m in config.models["grid_models"]]
        seeds = [int(s) for s in config.seeds_confirm]
        tolerance = config.selection_tolerance

        cells = [
            runner.Cell(channel=c, model=m, seed=s)
            for c in channels
            for m in models
            for s in seeds
        ]
        grid_csv = ctx.results_dir / "channel_model_grid.csv"
        subject_csv = ctx.results_dir / "channel_model_grid_subject.csv"
        _seed_from_stage_c(ctx, channels, models, seeds, grid_csv, subject_csv)

        outcome = runner.run_cells(
            config=config,
            ctx=ctx,
            cells=cells,
            summary_path=grid_csv,
            subject_path=subject_csv,
            max_epochs=args.max_epochs,
            save_checkpoints=False,
            progress=lambda m: print("  " + m, flush=True),
            summary_columns=GRID_COLUMNS,
            extra_summary={"grid_source": "stage_d"},
        )

        rows = pd.read_csv(grid_csv).to_dict("records")
        if len(rows) != len(cells):
            raise runner.RunnerError(f"grid has {len(rows)} rows, expected {len(cells)}")

        ranked = ranking.rank_pairs(rows, tolerance=tolerance)
        best = ranked[0]
        lock = build_lock(
            config=config,
            ctx=ctx,
            channels=channels,
            models=models,
            seeds=seeds,
            ranked=ranked,
            best=best,
            shortlist=shortlist,
        )
        reporting.write_json(lock, ctx.lock_path)

        ctx.report.metrics.update(
            {
                "channels": channels,
                "models": models,
                "seeds": seeds,
                "n_cells": len(cells),
                "n_trained_now": outcome["n_trained_now"],
                "n_reused": outcome["n_reused"],
                "pair_ranking": ranked,
                "selected": {
                    "channel": best["channel"],
                    "model": best["model"],
                    "mean_val_accuracy": best["mean_val_accuracy"],
                    "mean_val_loss": best["mean_val_loss"],
                    "param_count": best["param_count"],
                    "tied_with": best["tied_with"],
                },
                "selection_tolerance": tolerance,
                "device": outcome["device"],
            }
        )
        ctx.report.artifact("channel_model_grid", ctx.results_dir / "channel_model_grid.csv")
        ctx.report.artifact(
            "channel_model_grid_subject", ctx.results_dir / "channel_model_grid_subject.csv"
        )
        ctx.report.artifact("selection_lock", ctx.lock_path)
        ctx.report.manifest_sha256 = lock["manifest_sha256"]

        expected = len(channels) * len(models) * len(seeds)
        if len(rows) != expected:
            ctx.report.warn(f"grid has {len(rows)} rows, expected {expected}")
        if best["tied_with"]:
            ctx.report.warn(
                f"selected pair is within tolerance of {best['tied_with']}; "
                "the tie-break (loss, then parameters) decided it and the report must say so"
            )
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        ctx.report.fail(f"{type(exc).__name__}: {exc}")
        ctx.report.warn(traceback.format_exc(limit=5))
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 1


def _seed_from_stage_c(
    ctx: stages.StageContext,
    channels: list[str],
    models: list[str],
    seeds: list[int],
    grid_csv: Path,
    subject_csv: Path,
) -> None:
    """Pre-fill grid cells that Stage C already trained, before the grid runs.

    Stage C trained `screening_model` on the confirmed channels with the confirm
    seeds -- cells that are also in the grid. Re-training them would produce a
    different number for an identical configuration, so they are copied in first
    and `run_cells` skips them. `grid_source` records where each number came from.
    """
    donor_path = ctx.results_dir / "channel_confirm.csv"
    donor_subject = ctx.results_dir / "channel_confirm_subject.csv"
    if not donor_path.is_file():
        return
    existing = pd.read_csv(grid_csv).to_dict("records") if grid_csv.is_file() else []
    have = {(str(r["channel"]), str(r["model"]), int(r["seed"])) for r in existing}
    wanted = {(c, m, s) for c in channels for m in models for s in seeds}
    donors = []
    for row in pd.read_csv(donor_path).to_dict("records"):
        key = (str(row["channel"]), str(row["model"]), int(row["seed"]))
        if key in wanted and key not in have:
            row = dict(row)
            row["grid_source"] = "stage_c"
            donors.append(row)
            have.add(key)

    if not donors:
        return

    reporting.write_csv(existing + donors, grid_csv, columns=GRID_COLUMNS)

    if donor_subject.is_file():
        keep = {(d["channel"], d["model"], int(d["seed"])) for d in donors}
        subject_rows = pd.read_csv(subject_csv).to_dict("records") if subject_csv.is_file() else []
        for row in pd.read_csv(donor_subject).to_dict("records"):
            key = (str(row["channel"]), str(row["model"]), int(row["seed"]))
            if key in keep:
                subject_rows.append(row)
        reporting.write_csv(subject_rows, subject_csv, columns=runner.SUBJECT_COLUMNS)

    print(
        f"  reused {len(donors)} grid cell(s) from Stage C: "
        f"{sorted(d['channel'] for d in donors)}",
        flush=True,
    )


def build_lock(
    *,
    config,
    ctx: stages.StageContext,
    channels: list[str],
    models: list[str],
    seeds: list[int],
    ranked: list[dict],
    best: dict,
    shortlist: dict,
) -> dict:
    from semg import manifest as manifest_mod

    manifest_sha = manifest_mod.read_manifest_sha(ctx.manifest_sha_path)
    scaler_hashes = {}
    for channel in {best["channel"], *channels}:
        scaler = dataset.load_scaler(config, channel)
        scaler_hashes[channel] = scaler.sha256

    files_df = pd.read_csv(ctx.files_csv)
    test_rows = files_df[files_df["role"] == "test"]

    # The lock records the six items section 6 requires (channel, model, ranking,
    # config hash, manifest hash, seeds) plus what Stage F re-checks against.
    # `git_commit` and `manifest_files` are additive: without them a reader cannot
    # tell which code, and which file table, produced the winning number.
    return {
        "schema_version": 1,
        "locked_at": reporting._now(),
        "git_commit": reporting.git_commit(ctx.root),
        "config_sha256": config.sha256,
        "manifest_sha256": manifest_sha["manifest_sha256"],
        "manifest_files": {
            **dict(manifest_sha.get("files_csv") or {}),
            **dict(manifest_sha.get("windows") or {}),
        },
        "scope": {
            "training_scope": config.experiment["scope"]["training_scope"],
            "selected_channel_scope": config.experiment["scope"]["selected_channel_scope"],
            "subject_role": config.experiment["scope"]["subject_role"],
        },
        "selection": {
            "selected_channel": best["channel"],
            "selected_model": best["model"],
            "mean_val_accuracy": best["mean_val_accuracy"],
            "mean_val_loss": best["mean_val_loss"],
            "param_count": best["param_count"],
            "tied_with": best["tied_with"],
            "selection_tolerance": config.selection_tolerance,
            "tie_break_chain": list(config.experiment["selection"]["tie_break_chain"]),
            "ranked_pairs": ranked,
        },
        "candidates": {
            "confirmed_channels": channels,
            "models": models,
            "seeds_confirm": seeds,
            "seeds_extra": [int(s) for s in config.seeds_extra],
            "pilot_shortlist": shortlist.get("pilot_shortlist", []),
            "pilot_ranking": shortlist.get("pilot_ranking", []),
            "confirm_ranking": shortlist.get("confirm_ranking", []),
        },
        "scalers": {
            "rule": "one scaler per channel, fitted on Train D1_T1, shared by all models and seeds",
            "sha256": scaler_hashes,
            "selected_channel_sha256": scaler_hashes[best["channel"]],
        },
        "test_files": {
            "n_usable": int(len(test_rows[test_rows["status"] == "ok"])),
            "n_excluded": int(len(test_rows[test_rows["status"] != "ok"])),
            "excluded": test_rows[test_rows["status"] != "ok"][
                ["file", "bytes", "reason_code"]
            ].to_dict("records"),
            "note": (
                "Test files were enumerated by name and size only; no Test signal "
                "was read before this lock"
            ),
        },
        "windowing": {
            "fs_hz": config.fs_hz,
            "window_samples": config.window_samples,
            "hop_samples": config.hop_samples,
            "normalization": config.pipeline["normalization"]["method"],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
