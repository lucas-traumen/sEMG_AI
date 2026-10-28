"""Accuracy aggregation: per subject, then per seed, then across seeds.

The headline number is the mean over five seeds of the per-seed mean over subjects.
Windows are never pooled into one correct/total ratio, and a seed is never picked
as "the best one". Both of those shortcuts make a number look better while telling
you less:

  - Pooling windows lets a subject with more windows dominate the average, and
    hides exactly the per-subject spread the report has to show.

  - Reporting the best of five seeds reports a maximum, not an estimate. The mean
    across seeds is the estimate; the SD across seeds is how much of the result is
    training noise.

`accuracy_by_subject` therefore returns one row per (seed, subject) and every
higher-level number is derived from those rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AccuracySummary:
    """Aggregated accuracy at the levels the report needs."""

    overall_mean: float
    overall_sd_across_seeds: float
    seed_means: dict[int, float]
    subject_means: dict[int, float]        # averaged across seeds, per subject
    subject_sd_across_seeds: dict[int, float]
    subject_se: float                      # SE of the mean over subjects
    min_subject_mean: float
    max_subject_mean: float
    n_seeds: int
    n_subjects: int
    n_windows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_mean": self.overall_mean,
            "overall_sd_across_seeds": self.overall_sd_across_seeds,
            "seed_means": {int(k): float(v) for k, v in self.seed_means.items()},
            "subject_means": {int(k): float(v) for k, v in self.subject_means.items()},
            "subject_sd_across_seeds": {
                int(k): float(v) for k, v in self.subject_sd_across_seeds.items()
            },
            "subject_se": self.subject_se,
            "min_subject_mean": self.min_subject_mean,
            "max_subject_mean": self.max_subject_mean,
            "n_seeds": self.n_seeds,
            "n_subjects": self.n_subjects,
            "n_windows": self.n_windows,
        }


def accuracy_by_subject(
    *,
    seeds: Sequence[int],
    subjects: np.ndarray,
    predictions: dict[int, np.ndarray],
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    """One row per (seed, subject). The atom every other number is built from."""
    subjects = np.asarray(subjects).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        pred = np.asarray(predictions[int(seed)]).reshape(-1)
        if pred.shape != targets.shape:
            raise ValueError(f"seed {seed}: {pred.shape[0]} predictions vs {targets.shape[0]} targets")
        for subject in np.unique(subjects):
            mask = subjects == subject
            n = int(mask.sum())
            correct = int((pred[mask] == targets[mask]).sum())
            rows.append(
                {
                    "seed": int(seed),
                    "subject": int(subject),
                    "n_windows": n,
                    "n_correct": correct,
                    "accuracy": (correct / n) if n else float("nan"),
                }
            )
    return rows


def summarise(rows: Iterable[dict[str, Any]]) -> AccuracySummary:
    """Fold (seed, subject) accuracy rows into the reported numbers."""
    rows = list(rows)
    if not rows:
        raise ValueError("no accuracy rows to summarise")

    by_seed: dict[int, dict[int, float]] = {}
    counts: dict[int, int] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {})[int(r["subject"])] = float(r["accuracy"])
        counts[int(r["seed"])] = counts.get(int(r["seed"]), 0) + int(r["n_windows"])

    seed_means = {s: float(np.mean(list(v.values()))) for s, v in sorted(by_seed.items())}
    all_subjects = sorted({subj for v in by_seed.values() for subj in v})

    subject_means: dict[int, float] = {}
    subject_sds: dict[int, float] = {}
    for subj in all_subjects:
        per_seed = [by_seed[s][subj] for s in sorted(by_seed) if subj in by_seed[s]]
        subject_means[subj] = float(np.mean(per_seed))
        subject_sds[subj] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0

    values = np.array(list(seed_means.values()), dtype=float)
    per_subject = np.array(list(subject_means.values()), dtype=float)
    n_subjects = len(per_subject)
    # SE of the mean over subjects, using the across-subject spread of the seeds'
    # own means. This is the number that says whether a channel difference of a
    # given size is even resolvable by 10 subjects.
    subject_se = float(np.std(per_subject, ddof=1) / np.sqrt(n_subjects)) if n_subjects > 1 else float("nan")

    return AccuracySummary(
        overall_mean=float(values.mean()),
        overall_sd_across_seeds=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        seed_means=seed_means,
        subject_means=subject_means,
        subject_sd_across_seeds=subject_sds,
        subject_se=subject_se,
        min_subject_mean=float(per_subject.min()),
        max_subject_mean=float(per_subject.max()),
        n_seeds=len(seed_means),
        n_subjects=n_subjects,
        n_windows=int(sum(counts.values())),
    )


def confusion_matrix(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    normalize: bool = True,
) -> np.ndarray:
    """Confusion matrix, rows = true class (the convention §7.3 asks for).

    Row-normalised by default, so each cell answers "of the windows that really
    were class i, what fraction did we call j". Un-normalised counts are returned
    when normalize=False, and per-class recall is the diagonal either way.
    """
    targets = np.asarray(targets).reshape(-1).astype(np.int64)
    predictions = np.asarray(predictions).reshape(-1).astype(np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(cm, (targets, predictions), 1.0)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm


def per_class_recall(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> list[dict[str, Any]]:
    """Recall per class plus its support. Recall is the matrix diagonal."""
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    rows = []
    for c in range(num_classes):
        mask = targets == c
        support = int(mask.sum())
        correct = int((predictions[mask] == c).sum()) if support else 0
        rows.append(
            {
                "class_id": c,
                "n_windows": support,
                "n_correct": correct,
                "recall": (correct / support) if support else float("nan"),
                "fraction_of_windows": float(support / max(len(targets), 1)),
            }
        )
    return rows


def majority_baseline(targets: np.ndarray, num_classes: int) -> dict[str, Any]:
    """Accuracy of always predicting the most frequent class.

    Required next to every accuracy table: with REST at ~24.5 % of windows, an
    eight-class problem where 0.245 is beatable by a constant predictor is a very
    different claim than one where chance is 0.125.
    """
    targets = np.asarray(targets).reshape(-1).astype(np.int64)
    counts = np.bincount(targets, minlength=num_classes)
    total = int(counts.sum())
    majority = int(counts.argmax()) if total else -1
    return {
        "majority_class_id": majority,
        "majority_class_fraction": float(counts[majority] / total) if total else float("nan"),
        "chance_uniform": 1.0 / num_classes,
        "class_counts": [int(c) for c in counts],
    }


def paired_difference(
    a: dict[int, float], b: dict[int, float], *, subjects: Sequence[int] | None = None
) -> dict[str, float]:
    """Paired per-subject difference between two configurations.

    Pairing on subject removes the between-subject variance, which is much larger
    than the between-channel effect we are trying to see. Reported with a Wilcoxon
    signed-rank p-value over subjects -- a rank test, so it does not assume the
    per-subject differences are normal.
    """
    keys = sorted(set(a) & set(b)) if subjects is None else [s for s in subjects if s in a and s in b]
    diffs = np.array([a[k] - b[k] for k in keys], dtype=float)
    out: dict[str, float] = {
        "n_subjects": int(len(diffs)),
        "mean_difference": float(diffs.mean()) if len(diffs) else float("nan"),
        "sd_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
    }
    out["wilcoxon_p"] = _wilcoxon(diffs)
    return out


def _wilcoxon(diffs: np.ndarray) -> float:
    if diffs.size < 6 or np.allclose(diffs, 0.0):
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(diffs).pvalue)
    except Exception:  # pragma: no cover - scipy present in both envs
        return float("nan")
