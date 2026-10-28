#!/usr/bin/env python
"""Stage A: preflight.

    --stage inventory   A0: names and byte sizes for all 100 files. No signal read.
    --stage qc          A1: QC all 16 columns and window the 20 Train/Validation
                            files. Still no Test, Reserve, or Future signal.

Exit codes: 0 ok, 1 failed, 2 blocked.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semg import reporting, stages  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["inventory", "qc", "all"], default="all")
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--clean", action="store_true", help="delete derived manifests before running qc")
    args = parser.parse_args(argv)

    if args.stage in ("inventory", "all"):
        ctx = stages.make_context(args.config_dir, ROOT, "inventory")
        try:
            stages.run_inventory(ctx)
        except Exception as exc:  # noqa: BLE001 - the report is the error channel
            ctx.report.fail(f"{type(exc).__name__}: {exc}")
            ctx.report.warn(traceback.format_exc(limit=3))
            reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
            return 1
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)

    if args.stage in ("qc", "all"):
        ctx = stages.make_context(args.config_dir, ROOT, "qc")
        try:
            if args.clean:
                stages.clean_stale(ctx)
            entries = stages.load_entries(ctx)
            data = stages.run_qc(ctx, entries)
            stages.summarise_qc(ctx, data)
        except Exception as exc:  # noqa: BLE001
            ctx.report.fail(f"{type(exc).__name__}: {exc}")
            ctx.report.warn(traceback.format_exc(limit=3))
            reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
            return 1
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 2 if ctx.report.status == reporting.STATUS_BLOCKED else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
