#!/usr/bin/env python
"""Stage B: channel pilot.

Every QC-eligible channel x `screening_model` x the screening seed, on the three
pilot subjects only.

The pilot is cheap on purpose: its job is to discard channels that cannot work,
not to rank the survivors. With 3 subjects and 1 seed the standard error of the
mean is roughly 0.02 in accuracy, which is far too coarse to separate the top few
channels -- so this stage's output is a shortlist, and Stage C produces the ranking.

Output: results/channel_pilot.csv (one row per channel) and
results/channel_pilot_subject.csv (three rows per channel).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semg import dataset, reporting, runner, stages  # noqa: E402

PILOT_COLUMNS = runner.SUMMARY_COLUMNS + ("pilot_subjects",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)

    ctx = stages.make_context(args.config_dir, ROOT, "channel_pilot")
    config = ctx.config
    try:
        entries = stages.load_entries(ctx)
        channel_map = dataset.read_channel_map(config)
        channels = dataset.channel_ids_from_map(channel_map)
        pilot_subjects = [int(s) for s in config.experiment["selection"]["pilot_subjects"]]
        model = str(config.models["screening_model"])
        seed = int(config.training["seeds"]["screening"][0])

        cells = [runner.Cell(channel=c, model=model, seed=seed) for c in channels]
        outcome = runner.run_cells(
            config=config,
            ctx=ctx,
            cells=cells,
            summary_path=ctx.results_dir / "channel_pilot.csv",
            subject_path=ctx.results_dir / "channel_pilot_subject.csv",
            max_epochs=args.max_epochs,
            subjects=pilot_subjects,
            save_checkpoints=False,
            progress=lambda m: print("  " + m, flush=True),
            summary_columns=PILOT_COLUMNS,
            extra_summary={"pilot_subjects": ",".join(str(s) for s in pilot_subjects)},
        )

        ctx.report.metrics.update(
            {
                "n_channels": len(channels),
                "channels": channels,
                "model": model,
                "seed": seed,
                "pilot_subjects": pilot_subjects,
                "n_trained_now": outcome["n_trained_now"],
                "n_reused": outcome["n_reused"],
                "device": outcome["device"],
            }
        )
        ctx.report.artifact("channel_pilot", ctx.results_dir / "channel_pilot.csv")
        ctx.report.artifact("channel_pilot_subject", ctx.results_dir / "channel_pilot_subject.csv")
        ctx.report.manifest_sha256 = _manifest_sha(ctx)
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        ctx.report.fail(f"{type(exc).__name__}: {exc}")
        ctx.report.warn(traceback.format_exc(limit=5))
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 1


def _manifest_sha(ctx: stages.StageContext) -> str:
    from semg import manifest

    try:
        return manifest.read_manifest_sha(ctx.manifest_sha_path)["manifest_sha256"]
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
