"""Ranking helpers shared by Stages B, C and D.

The ranking rule has exactly two levels, and seeds are not one of them:

    1. mean validation accuracy, measured per subject then averaged over seeds
    2. within `selection_tolerance`, lower mean validation loss, then fewer parameters

The tolerance step is not decoration. With three seeds and ten subjects, differences
of a few thousandths of accuracy are noise, and ordering by the third decimal would
be reporting noise as a finding. Entries within tolerance are therefore treated as a
tied cluster, and the cluster is ordered by the pre-registered tie-break rather than
by whichever entry happened to score 0.0004 higher.

Seeds are averaging units. Sorting by "best seed" would report a maximum of five
noisy estimates and would systematically favour configurations that drew a lucky
initialisation.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def mean_accuracy(rows: Iterable[dict[str, Any]]) -> float:
    values = [float(r["val_subject_mean_accuracy"]) for r in rows]
    if not values:
        raise ValueError("no rows to average")
    return sum(values) / len(values)


def mean_validation_loss(rows: Iterable[dict[str, Any]]) -> float:
    values = [float(r["best_val_loss"]) for r in rows]
    return sum(values) / len(values) if values else float("nan")


def group_by_channel(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r["channel"]), []).append(r)
    return out


def rank_channels(
    rows: Sequence[dict[str, Any]],
    *,
    tolerance: float,
    param_count_by_model: dict[str, int] | None = None,
    keep: int | None = None,
) -> list[dict[str, Any]]:
    """Rank channels by mean validation accuracy, with the two-level tie-break.

    Returns one dict per channel, best first. `tied_with` lists the channels in the
    same tolerance cluster, so a reader can see that the top three are not
    distinguishable rather than assuming the ordering is meaningful.
    """
    per_channel = group_by_channel(rows)
    entries: list[dict[str, Any]] = []
    for channel, group in per_channel.items():
        accs = [float(r["val_subject_mean_accuracy"]) for r in group]
        entries.append(
            {
                "channel": channel,
                "mean_val_accuracy": sum(accs) / len(accs),
                "sd_across_seeds": _sd(accs),
                "n_seeds": len(group),
                "mean_val_loss": mean_validation_loss(group),
                # Taken from the rows when present: for a single-model stage this
                # is the only parameter count that exists, and it must not be
                # reported as 0 just because no model->params mapping was passed.
                "param_count": _param_count(group, param_count_by_model),
            }
        )

    return _order(entries, tolerance=tolerance, keep=keep)


def rank_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    tolerance: float,
    keep: int | None = None,
) -> list[dict[str, Any]]:
    """Rank (channel, model) pairs, same rule as rank_channels."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((str(r["channel"]), str(r["model"])), []).append(r)

    entries: list[dict[str, Any]] = []
    for (channel, model), group in grouped.items():
        accs = [float(r["val_subject_mean_accuracy"]) for r in group]
        entries.append(
            {
                "channel": channel,
                "model": model,
                "mean_val_accuracy": sum(accs) / len(accs),
                "sd_across_seeds": _sd(accs),
                "n_seeds": len(group),
                "mean_val_loss": mean_validation_loss(group),
                "param_count": int(group[0].get("param_count", 0)),
            }
        )

    return _order(entries, tolerance=tolerance, keep=keep)


def select_within_tolerance(
    ranked: Sequence[dict[str, Any]], *, tolerance: float, keep: int
) -> list[dict[str, Any]]:
    """Take the top `keep`, but do not split a tie cluster at the boundary.

    If the 5th and 6th entries are within tolerance of each other, taking only five
    would be an arbitrary cut dressed up as a decision. The function extends the
    selection to the end of the tie cluster and says so.
    """
    if keep >= len(ranked):
        return list(ranked)
    cutoff = ranked[keep - 1]["mean_val_accuracy"]
    out = list(ranked[:keep])
    for entry in ranked[keep:]:
        if abs(entry["mean_val_accuracy"] - cutoff) < tolerance:
            out.append(entry)
        else:
            break
    return out


def _order(
    entries: list[dict[str, Any]], *, tolerance: float, keep: int | None
) -> list[dict[str, Any]]:
    """Sort into tolerance clusters, then apply the tie-break inside each cluster.

    Clusters are anchored on their highest-accuracy member. Inside a cluster the
    pre-registered order is: lower mean validation loss, then fewer parameters.
    That is what makes the outcome reproducible rather than an artefact of the
    third decimal place.
    """
    if not entries:
        return []

    by_accuracy = sorted(entries, key=lambda e: -e["mean_val_accuracy"])
    clusters: list[list[dict[str, Any]]] = []
    for entry in by_accuracy:
        if clusters and abs(
            entry["mean_val_accuracy"] - clusters[-1][0]["mean_val_accuracy"]
        ) < tolerance:
            clusters[-1].append(entry)
        else:
            clusters.append([entry])

    ordered: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster.sort(key=lambda e: (e["mean_val_loss"], e["param_count"]))
        for entry in cluster:
            entry["tied_with"] = sorted(
                _label(other) for other in cluster if _label(other) != _label(entry)
            )
            entry["tie_break_applied"] = len(cluster) > 1
            ordered.append(entry)

    for i, entry in enumerate(ordered, start=1):
        entry["rank"] = i
    return ordered[:keep] if keep is not None else ordered


def _label(entry: dict[str, Any]) -> str:
    model = entry.get("model")
    return f"{entry['channel']}/{model}" if model else str(entry["channel"])


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return float((sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5)


def _param_count(
    group: Sequence[dict[str, Any]], by_model: dict[str, int] | None
) -> int:
    """Parameter count for a group of runs, from the rows or a model mapping."""
    for row in group:
        value = row.get("param_count")
        if value is not None and int(value) > 0:
            return int(value)
    if by_model:
        return int(by_model.get(str(group[0].get("model", "")), 0))
    return 0
