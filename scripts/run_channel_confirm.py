#!/usr/bin/env python
"""Stage C: channel confirm.

Top `pilot_keep` channels from Stage B, x the three confirm seeds, on ALL ten
subjects.

This is where channel ranking actually happens. Stage B used three subjects and one
seed to make a shortlist; here every subject contributes, three seeds are averaged,
and the shortlist is cut to `confirm_keep` using the pre-registered tolerance.

Reading Stage B's CSV rather than re-deriving the shortlist means the shortlist is
a recorded artefact, not a recomputed guess.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semg import dataset, manifest, ranking, reporting, runner, stages  # noqa: E402

CONFIRM_COLUMNS = runner.SUMMARY_COLUMNS + ("shortlist_rank",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)

    ctx = stages.make_context(args.config_dir, ROOT, "channel_confirm")
    config = ctx.config
    try:
        entries = stages.load_entries(ctx)
        pilot_path = ctx.results_dir / "channel_pilot.csv"
        if not pilot_path.is_file():
            raise FileNotFoundError(f"{pilot_path} missing; run Stage B first")
        pilot_rows = pd.read_csv(pilot_path).to_dict("records")
        if len(pilot_rows) != len(dataset.channel_ids_from_map(dataset.read_channel_map(config))):
            raise ValueError(
                f"Stage B produced {len(pilot_rows)} rows, expected one per eligible channel"
            )

        tolerance = config.selection_tolerance
        pilot_keep = int(config.experiment["selection"]["pilot_keep"])
        confirm_keep = int(config.experiment["selection"]["confirm_keep"])

        ranked = ranking.rank_channels(pilot_rows, tolerance=tolerance)
        shortlist = [r["channel"] for r in ranked[:pilot_keep]]
        seeds = [int(s) for s in config.seeds_confirm]
        model = str(config.models["screening_model"])

        cells = [
            runner.Cell(channel=c, model=model, seed=s) for c in shortlist for s in seeds
        ]
        outcome = runner.run_cells(
            config=config,
            ctx=ctx,
            cells=cells,
            summary_path=ctx.results_dir / "channel_confirm.csv",
            subject_path=ctx.results_dir / "channel_confirm_subject.csv",
            max_epochs=args.max_epochs,
            save_checkpoints=False,
            progress=lambda m: print("  " + m, flush=True),
            summary_columns=CONFIRM_COLUMNS,
            extra_summary={"shortlist_rank": 0},
        )

        rows = pd.read_csv(ctx.results_dir / "channel_confirm.csv").to_dict("records")
        for row in rows:
            row["shortlist_rank"] = shortlist.index(str(row["channel"])) + 1
        reporting.write_csv(rows, ctx.results_dir / "channel_confirm.csv", columns=CONFIRM_COLUMNS)

        confirmed = ranking.rank_channels(rows, tolerance=tolerance)
        top = ranking.select_within_tolerance(
            confirmed, tolerance=tolerance, keep=confirm_keep
        )
        shortlist_payload = {
            "schema_version": 1,
            "stage": "channel_confirm",
            "model": model,
            "seeds": seeds,
            "pilot_subjects": [int(s) for s in config.experiment["selection"]["pilot_subjects"]],
            "selection_tolerance": tolerance,
            "pilot_shortlist": shortlist,
            "pilot_ranking": ranked,
            "confirm_ranking": confirmed,
            "confirmed_channels": [r["channel"] for r in top],
            "tie_note": _tie_note(confirmed, tolerance),
        }
        reporting.write_json(shortlist_payload, ctx.results_dir / "channel_shortlist.json")

        ctx.report.metrics.update(
            {
                "pilot_shortlist": shortlist,
                "confirmed_channels": shortlist_payload["confirmed_channels"],
                "n_confirmed": len(shortlist_payload["confirmed_channels"]),
                "confirm_ranking": confirmed,
                "selection_tolerance": tolerance,
                "seeds": seeds,
                "model": model,
                "n_trained_now": outcome["n_trained_now"],
                "n_reused": outcome["n_reused"],
                "device": outcome["device"],
            }
        )
        ctx.report.artifact("channel_confirm", ctx.results_dir / "channel_confirm.csv")
        ctx.report.artifact("channel_confirm_subject", ctx.results_dir / "channel_confirm_subject.csv")
        ctx.report.artifact("channel_shortlist", ctx.results_dir / "channel_shortlist.json")
        ctx.report.manifest_sha256 = manifest.read_manifest_sha(ctx.manifest_sha_path)["manifest_sha256"]

        if len(rows) != len(shortlist) * len(seeds):
            ctx.report.warn(
                f"expected {len(shortlist) * len(seeds)} confirm rows, found {len(rows)}"
            )
        if len(shortlist_payload["confirmed_channels"]) > confirm_keep:
            ctx.report.warn(
                f"tie at the cut extended the confirmed set to "
                f"{len(shortlist_payload['confirmed_channels'])} channels "
                f"(confirm_keep={confirm_keep}); see tie_note"
            )
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        ctx.report.fail(f"{type(exc).__name__}: {exc}")
        ctx.report.warn(traceback.format_exc(limit=5))
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 1


def _tie_note(ranked: list[dict], tolerance: float) -> str:
    if not ranked:
        return "no ranking produced"
    groups: dict[int, list[str]] = {}
    for entry in ranked:
        bucket = int(entry["mean_val_accuracy"] / tolerance)
        groups.setdefault(bucket, []).append(entry["channel"])
    tied = [chans for chans in groups.values() if len(chans) > 1]
    if not tied:
        return "no channels within tolerance of each other"
    return "within-tolerance groups: " + "; ".join(",".join(sorted(g)) for g in tied)


if __name__ == "__main__":
    raise SystemExit(main())
