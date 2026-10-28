#!/usr/bin/env python
"""Stage E: complete the five-seed set for the locked configuration.

Stage D locked one (channel, model) pair and trained it with the three confirm
seeds. This stage adds the two extra seeds so the final number is a mean over five
seeds rather than three.

Two properties matter here:

    It refuses to run without a lock. The configuration is not "whatever is in the
    config file today"; it is the configuration the lock recorded.

    It verifies the lock still describes the current world. A changed config or
    manifest means the locked numbers no longer refer to these files, and continuing
    would silently attach five-seed results to a selection made under other
    conditions.
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

from semg import manifest, reporting, runner, stages  # noqa: E402


def verify_lock(ctx: stages.StageContext) -> dict:
    if not ctx.lock_path.is_file():
        raise FileNotFoundError(
            f"{ctx.lock_path} missing; Stage D must complete before the five-seed set"
        )
    lock = json.loads(ctx.lock_path.read_text(encoding="utf-8"))
    config = ctx.config

    if lock["config_sha256"] != config.sha256:
        raise ValueError(
            "config changed since the selection lock "
            f"({lock['config_sha256'][:12]} -> {config.sha256[:12]}); "
            "the lock no longer describes this run"
        )
    current = manifest.read_manifest_sha(ctx.manifest_sha_path)["manifest_sha256"]
    if lock["manifest_sha256"] != current:
        raise ValueError(
            "window manifest changed since the selection lock "
            f"({lock['manifest_sha256'][:12]} -> {current[:12]})"
        )

    channel = lock["selection"]["selected_channel"]
    model = lock["selection"]["selected_model"]
    expected = lock["scalers"]["selected_channel_sha256"]
    actual = runner.dataset.load_scaler(config, channel).sha256
    if expected != actual:
        raise ValueError(
            f"scaler for {channel} changed since the lock ({expected[:12]} -> {actual[:12]})"
        )
    return lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)

    ctx = stages.make_context(args.config_dir, ROOT, "complete_five_seeds")
    config = ctx.config
    try:
        lock = verify_lock(ctx)
        channel = lock["selection"]["selected_channel"]
        model = lock["selection"]["selected_model"]
        seeds = [int(s) for s in config.seeds_all]

        cells = [runner.Cell(channel=channel, model=model, seed=s) for s in seeds]
        outcome = runner.run_cells(
            config=config,
            ctx=ctx,
            cells=cells,
            summary_path=ctx.results_dir / "five_seed_runs.csv",
            subject_path=ctx.results_dir / "five_seed_runs_subject.csv",
            max_epochs=args.max_epochs,
            save_checkpoints=True,
            progress=lambda m: print("  " + m, flush=True),
        )

        rows = pd.read_csv(ctx.results_dir / "five_seed_runs.csv").to_dict("records")
        missing = [s for s in seeds if s not in {int(r["seed"]) for r in rows}]
        if missing:
            raise runner.RunnerError(f"seeds still missing after Stage E: {missing}")

        checkpoints = [runner.checkpoint_path(config, c) for c in cells]
        absent = [str(p) for p in checkpoints if not p.is_file()]
        if absent:
            raise runner.RunnerError(f"checkpoints missing: {absent}")

        from semg import metrics as metrics_mod

        subject_rows = pd.read_csv(ctx.results_dir / "five_seed_runs_subject.csv").to_dict(
            "records"
        )
        summary = metrics_mod.summarise(subject_rows)
        payload = {
            "schema_version": 1,
            "channel": channel,
            "model": model,
            "seeds": seeds,
            "split": "validation",
            "metric": config.experiment["evaluation"]["primary_metric"],
            "summary": summary.as_dict(),
            "per_seed": [
                {
                    "seed": int(r["seed"]),
                    "val_subject_mean_accuracy": float(r["val_subject_mean_accuracy"]),
                    "val_subject_sd": float(r["val_subject_sd"]),
                    "best_epoch": int(r["best_epoch"]),
                    "best_val_loss": float(r["best_val_loss"]),
                }
                for r in rows
            ],
        }
        reporting.write_json(payload, ctx.results_dir / "five_seed_summary.json")

        ctx.report.metrics.update(
            {
                "channel": channel,
                "model": model,
                "seeds": seeds,
                "n_trained_now": outcome["n_trained_now"],
                "n_reused": outcome["n_reused"],
                "validation_summary": summary.as_dict(),
                "checkpoints": [str(p.relative_to(ROOT)) for p in checkpoints],
                "device": outcome["device"],
            }
        )
        ctx.report.artifact("five_seed_runs", ctx.results_dir / "five_seed_runs.csv")
        ctx.report.artifact("five_seed_summary", ctx.results_dir / "five_seed_summary.json")
        ctx.report.manifest_sha256 = lock["manifest_sha256"]
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        ctx.report.fail(f"{type(exc).__name__}: {exc}")
        ctx.report.warn(traceback.format_exc(limit=5))
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
