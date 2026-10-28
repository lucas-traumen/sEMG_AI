#!/usr/bin/env python
"""Stage F: final Test evaluation. Runs only after the selection lock exists.

This is the single place where Test signal is read. It runs exactly the five locked
checkpoints on exactly one Test manifest and writes the numbers that go in the
report.

Three guards, because a mistake here cannot be undone by re-running:

    The lock must exist and must still match the current config, manifest and
    scaler hashes. Otherwise the "Test result" would belong to a selection that no
    longer exists.

    The Test manifest must not exist yet, or must have been produced by this same
    lock. Building it twice from different configs is how a leaked Test set happens.

    The five checkpoints must all load and must all carry the lock's channel, model,
    scaler hash, config hash and manifest hash. A checkpoint from a different run
    would silently change the headline number.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semg import (  # noqa: E402
    dataset,
    gestures,
    manifest,
    metrics,
    reporting,
    runner,
    stages,
    train,
    windowing,
)

FINAL_COLUMNS = (
    "channel",
    "model",
    "seed",
    "subject",
    "split",
    "n_windows",
    "n_correct",
    "accuracy",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument(
        "--reason",
        default="first_open",
        choices=["first_open", "rerun"],
        help="rerun is allowed only with identical config/manifest/checkpoints",
    )
    args = parser.parse_args(argv)

    ctx = stages.make_context(args.config_dir, ROOT, "final_test_report")
    config = ctx.config
    try:
        lock = _verify_lock(ctx)
        channel = lock["selection"]["selected_channel"]
        model = lock["selection"]["selected_model"]
        seeds = [int(s) for s in config.seeds_all]

        # --- build or reuse the Test window manifest -------------------------
        test_windows = _test_manifest(ctx, lock)
        n_test_subjects = int(test_windows["subject"].nunique())

        # --- evaluate the five locked checkpoints ----------------------------
        device = train.pick_device()
        scaler = dataset.load_scaler(config, channel)
        subject_rows: list[dict] = []
        per_seed: list[dict] = []
        predictions: dict[int, np.ndarray] = {}
        targets = None

        entries = stages.load_entries(ctx)
        source = stages.build_source_for(entries, ctx=ctx, stage="final_test")
        manifest_sha = manifest.read_manifest_sha(ctx.manifest_sha_path)["manifest_sha256"]
        cache = dataset.ensure_emg_cache(
            config=config,
            source=source,
            files=sorted(test_windows["file"].unique()),
            manifest_sha=manifest_sha,
        )

        test_arrays = dataset.channel_arrays(
            config=config,
            cache_dir=cache,
            windows=test_windows,
            channel_id=channel,
            scaler=scaler,
            split="test",
        )
        targets = test_arrays.y

        # Confusion counts are summed over seeds rather than taken from one seed:
        # picking a seed's matrix would report a matrix for a run that is not the
        # headline. Every seed contributes equally.
        counts = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)

        for seed in seeds:
            cell = runner.Cell(channel=channel, model=model, seed=seed)
            path = runner.checkpoint_path(config, cell)
            payload = _load_locked_checkpoint(path, lock, cell)
            net = train.restore_model(config, payload)
            logits = train.predict_logits(net, test_arrays.x, device=device)
            pred = logits.argmax(axis=1)
            predictions[seed] = pred
            np.add.at(counts, (targets.astype(np.int64), pred.astype(np.int64)), 1)

            rows = metrics.accuracy_by_subject(
                seeds=[seed],
                subjects=test_arrays.subjects,
                predictions={seed: pred},
                targets=targets,
            )
            subject_rows.extend({**r, "channel": channel, "model": model, "split": "test"} for r in rows)
            accs = [float(r["accuracy"]) for r in rows]
            per_seed.append(
                {
                    "seed": seed,
                    "subject_mean_accuracy": float(np.mean(accs)),
                    "subject_sd": float(np.std(accs, ddof=1)),
                    "n_windows": int(len(test_arrays)),
                    "checkpoint": str(path.relative_to(ROOT)),
                }
            )
            print(f"  test seed {seed}: {np.mean(accs):.4f}", flush=True)

        summary = metrics.summarise(subject_rows)
        cm = _normalise(counts)
        recall = _recall_from_counts(counts)
        baseline = metrics.majority_baseline(targets, config.num_classes)
        label_map = gestures.build(config.gestures)

        _write_outputs(
            ctx=ctx,
            channel=channel,
            model=model,
            seeds=seeds,
            subject_rows=subject_rows,
            per_seed=per_seed,
            summary=summary,
            cm=cm,
            counts=counts,
            recall=recall,
            baseline=baseline,
            label_map=label_map,
            n_test_windows=int(len(test_arrays)),
            n_test_subjects=n_test_subjects,
        )

        ctx.report.metrics.update(
            {
                "channel": channel,
                "model": model,
                "seeds": seeds,
                "test_summary": summary.as_dict(),
                "per_seed": per_seed,
                "n_test_windows": int(len(test_arrays)),
                "n_test_subjects": n_test_subjects,
                "majority_baseline": baseline,
                "per_class_recall": recall,
                "reason": args.reason,
            }
        )
        ctx.report.artifact("final_accuracy_by_seed", ctx.results_dir / "final/final_accuracy_by_seed.csv")
        ctx.report.artifact(
            "final_accuracy_by_subject", ctx.results_dir / "final/final_accuracy_by_subject.csv"
        )
        ctx.report.artifact("per_class_recall", ctx.results_dir / "final/per_class_recall.csv")
        ctx.report.artifact("test_summary", ctx.results_dir / "final/test_summary.json")
        ctx.report.manifest_sha256 = manifest_sha

        if n_test_subjects != lock["test_files"]["n_usable"]:
            ctx.report.warn(
                f"Test covers {n_test_subjects} subjects but the lock recorded "
                f"{lock['test_files']['n_usable']} usable Test files"
            )
        if baseline["majority_class_fraction"] >= summary.overall_mean:
            ctx.report.warn(
                "majority-class baseline is at or above model accuracy; "
                "the result must be reported as no better than chance"
            )
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        ctx.report.fail(f"{type(exc).__name__}: {exc}")
        ctx.report.warn(traceback.format_exc(limit=6))
        reporting.emit_result(ctx.report, reports_dir=ctx.reports_dir)
        return 1


def _normalise(counts: np.ndarray) -> np.ndarray:
    """Row-normalised confusion matrix: of the windows that truly were class i,
    what fraction was called j."""
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)


def _recall_from_counts(counts: np.ndarray) -> list[dict]:
    total = int(counts.sum())
    rows = []
    for c in range(counts.shape[0]):
        support = int(counts[c, :].sum())
        correct = int(counts[c, c])
        rows.append(
            {
                "class_id": c,
                "n_windows": support,
                "n_correct": correct,
                "recall": (correct / support) if support else float("nan"),
                "fraction_of_windows": float(support / total) if total else 0.0,
            }
        )
    return rows


def _verify_lock(ctx: stages.StageContext) -> dict:
    if not ctx.lock_path.is_file():
        raise FileNotFoundError(
            f"{ctx.lock_path} missing; Stage D must lock the selection before Test is opened"
        )
    lock = json.loads(ctx.lock_path.read_text(encoding="utf-8"))
    if lock["config_sha256"] != ctx.config.sha256:
        raise ValueError(
            "config changed since the lock; refusing to evaluate Test under a "
            "different configuration"
        )
    current = manifest.read_manifest_sha(ctx.manifest_sha_path)["manifest_sha256"]
    if lock["manifest_sha256"] != current:
        raise ValueError("Train/Validation manifest changed since the lock")
    return lock


def _test_manifest(ctx: stages.StageContext, lock: dict) -> pd.DataFrame:
    """Build the Test window manifest, or reuse the one this lock produced."""
    config = ctx.config
    if ctx.test_windows_path.is_file():
        stamp = manifest.read_manifest_sha(ctx.manifest_sha_path)
        recorded = stamp.get("test_manifest_sha256")
        if recorded is None:
            raise ValueError(
                f"{ctx.test_windows_path} exists but is not recorded in the manifest "
                "hash file; delete it and re-run to build it under this lock"
            )
        windows = manifest.read_windows(ctx.test_windows_path)
        if manifest.sha256_file(ctx.test_windows_path) != recorded:
            raise ValueError("Test window manifest does not match its recorded hash")
        return windows

    entries = stages.load_entries(ctx)
    test_entries = [
        e for e in entries if e.role == "test" and e.status == "ok"
    ]
    if len(test_entries) != lock["test_files"]["n_usable"]:
        raise ValueError(
            f"{len(test_entries)} usable Test files present, lock recorded "
            f"{lock['test_files']['n_usable']}"
        )

    label_map = gestures.build(config.gestures)
    source = stages.build_source_for(entries, ctx=ctx, stage="final_test")
    all_windows = []
    for entry in test_entries:
        data = source.load(entry.file, reason="Stage F: open Test once, after the lock")
        labels = np.asarray(data["restimulus"]).reshape(-1)
        reps = np.asarray(data["rerepetition"]).reshape(-1)
        all_windows.extend(
            windowing.build_window_index(
                file=entry.file,
                subject=entry.subject,
                day=entry.day,
                session=entry.session,
                split="test",
                restimulus=labels,
                rerepetition=reps,
                raw_to_class=label_map.raw_to_class,
                window_samples=config.window_samples,
                hop_samples=config.hop_samples,
            )
        )

    windows = manifest.windows_frame(all_windows)
    manifest.assert_windows_inside_segments(windows, window_samples=config.window_samples)
    manifest.assert_one_manifest_per_role(windows, {"test"})
    manifest.write_windows(windows, ctx.test_windows_path)

    stamp = manifest.read_manifest_sha(ctx.manifest_sha_path)
    stamp["test_manifest_sha256"] = manifest.sha256_file(ctx.test_windows_path)
    stamp["test_manifest_rows"] = int(len(windows))
    stamp["test_files_used"] = sorted(windows["file"].unique().tolist())
    reporting.write_json(stamp, ctx.manifest_sha_path)
    return windows


def _load_locked_checkpoint(path: Path, lock: dict, cell: runner.Cell) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"locked checkpoint missing: {path}")
    payload = train.load_checkpoint(path)
    checks = {
        "channel": (payload["channel"], cell.channel),
        "model": (payload["model"], cell.model),
        "seed": (int(payload["seed"]), int(cell.seed)),
        "config_sha256": (payload["config_sha256"], lock["config_sha256"]),
        "manifest_sha256": (payload["manifest_sha256"], lock["manifest_sha256"]),
        "scaler_sha256": (
            payload["scaler_sha256"],
            lock["scalers"]["selected_channel_sha256"],
        ),
    }
    mismatched = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatched:
        raise ValueError(f"{path.name} does not match the lock: {mismatched}")
    return payload


def _write_outputs(
    *,
    ctx: stages.StageContext,
    channel: str,
    model: str,
    seeds: list[int],
    subject_rows: list[dict],
    per_seed: list[dict],
    summary,
    cm: np.ndarray,
    counts: np.ndarray,
    recall: list[dict],
    baseline: dict,
    label_map,
    n_test_windows: int,
    n_test_subjects: int,
) -> None:
    final_dir = ctx.results_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    reporting.write_csv(per_seed, final_dir / "final_accuracy_by_seed.csv")
    reporting.write_csv(subject_rows, final_dir / "final_accuracy_by_subject.csv", columns=FINAL_COLUMNS)

    names = label_map.ordered_names()
    recall_rows = []
    for row in recall:
        class_id = int(row["class_id"])
        recall_rows.append(
            {
                "class_id": class_id,
                "gesture": names[class_id],
                "raw_label": int(label_map.class_to_raw[class_id]),
                **{k: v for k, v in row.items() if k != "class_id"},
            }
        )
    reporting.write_csv(recall_rows, final_dir / "per_class_recall.csv")

    np.save(final_dir / "confusion_matrix_window.npy", cm)
    np.save(final_dir / "confusion_counts_window.npy", counts)
    reporting.write_csv(
        [
            {"true_class": i, "gesture": names[i], **{names[j]: int(cm[i, j]) for j in range(cm.shape[1])}}
            for i in range(cm.shape[0])
        ],
        final_dir / "confusion_matrix_window.csv",
    )

    payload = {
        "schema_version": 1,
        "channel": channel,
        "model": model,
        "seeds": seeds,
        "split": "test",
        "unit": "per_subject_then_per_seed",
        "primary_metric": ctx.config.experiment["evaluation"]["primary_metric"],
        "headline_rule": "mean of five per-seed subject-means",
        "summary": summary.as_dict(),
        "per_seed": per_seed,
        "majority_baseline": baseline,
        "n_test_windows": n_test_windows,
        "n_test_subjects": n_test_subjects,
        "test_files_excluded": "S2_D2_T2.mat (truncated, excluded by size rule before any Test read)",
        "confusion_matrix_row_normalised": cm.tolist(),
        "confusion_matrix_counts": counts.tolist(),
        "confusion_counts_note": (
            "counts are summed over the five seeds; per_class_recall n_windows is "
            "therefore 5x the Test window count, and recall is the mean recall of "
            "the five runs weighted by support (identical support per seed)"
        ),
        "class_order": names,
    }
    reporting.write_json(payload, final_dir / "test_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
