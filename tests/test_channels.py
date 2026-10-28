"""Plan section 10 bullets T04, T05, T06 and T07: the channel map and QC coverage.

T04  "CH01..CH16 map to matrix columns 0..15 in both directions."
T05  "channel_map.json covers 16 columns, each with a status and, where it is
      excluded, a reason."
T06  "Preflight confirms 14 physical / 2 empty; otherwise the stage is blocked with
      evidence and the thresholds are not quietly adjusted."
T07  "Every QC-passing channel appears in channel_pilot.csv."

The point of T05 is that a channel can be *measured* and still excluded: the map
must say which of `structural_status`, `candidate_eligible` and `reason_code`
produced the decision, and it must do so for all 16 columns even though only 14
are candidates. T06 is the inverse check: the moment the data disagrees with the
documentation, the pipeline has to stop rather than reinterpret.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from semg import config as config_mod
from semg import qc as qc_mod

from conftest import require


@pytest.fixture(scope="module")
def cfg(root):
    return config_mod.load(root / "configs")


@pytest.fixture(scope="module")
def channel_map(root):
    path = require(root / "results" / "channel_map.json", "A1 channel map")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _row(file: str, index: int, flags: list[str], std: float = 1.0) -> qc_mod.ChannelFileStats:
    return qc_mod.ChannelFileStats(
        file=file,
        subject=1,
        channel_index=index,
        channel_id=f"CH{index + 1:02d}",
        n_samples=1000,
        nan_count=0,
        inf_count=0,
        zero_fraction=0.0,
        mean=0.0,
        std=std,
        min=-1.0,
        max=1.0,
        frozen_fraction=0.0,
        longest_frozen_run=1,
        clipping_fraction=0.0,
        rms_rest=0.0,
        rms_active=0.0,
        mains_ratio=1.0,
        std_ratio=1.0,
        qc_flags=list(flags),
    )


# ---------------------------------------------------------------------------
# T04 -- the two-way column <-> channel-id mapping
# ---------------------------------------------------------------------------

def test_t04_channel_id_is_one_based_over_sixteen_columns(cfg) -> None:
    ids = [cfg.channel_id(i) for i in range(16)]
    assert ids == [f"CH{n:02d}" for n in range(1, 17)]
    assert len(set(ids)) == 16
    # The reverse direction is the one the loaders use: CH07 -> column 6.
    for index, cid in enumerate(ids):
        assert int(cid[2:]) - 1 == index


def test_t04_channel_map_indices_are_zero_based_and_complete(channel_map) -> None:
    channels = channel_map["channels"]
    assert len(channels) == 16
    assert [c["matrix_index"] for c in channels] == list(range(16))
    assert [c["channel_id"] for c in channels] == [f"CH{n:02d}" for n in range(1, 17)]
    for c in channels:
        assert int(c["channel_id"][2:]) - 1 == c["matrix_index"]


def test_t04_channel_qc_rows_agree_with_the_map(root, channel_map) -> None:
    path = require(root / "results" / "channel_qc.csv", "A1 per-file QC table")
    frame = pd.read_csv(path)
    assert set(frame["channel_id"]) == {f"CH{n:02d}" for n in range(1, 17)}
    # channel_qc.csv carries both the id and the 0-based index; they must agree.
    pairs = frame[["channel_id", "matrix_index"]].drop_duplicates()
    assert len(pairs) == 16
    for row in pairs.itertuples():
        assert int(row.channel_id[2:]) - 1 == int(row.matrix_index)


def test_t04_matrix_columns_match_the_config(cfg) -> None:
    matrix = cfg.data["matrix"]
    assert matrix["matrix_columns"] == 16
    assert matrix["expected_physical_channels"] + matrix["expected_structural_empty_columns"] == 16
    assert cfg.matrix_columns == 16


# ---------------------------------------------------------------------------
# T05 -- the map covers every column and explains every exclusion
# ---------------------------------------------------------------------------

def test_t05_every_column_has_a_status(channel_map) -> None:
    assert channel_map["matrix_columns"] == 16
    assert len(channel_map["channels"]) == channel_map["matrix_columns"]
    for c in channel_map["channels"]:
        assert c["structural_status"] in {qc_mod.STRUCTURAL_PHYSICAL, qc_mod.STRUCTURAL_EMPTY}
        assert isinstance(c["candidate_eligible"], bool)
        assert isinstance(c["reason_code"], str)


def test_t05_every_exclusion_has_a_reason(channel_map) -> None:
    excluded = [c for c in channel_map["channels"] if not c["candidate_eligible"]]
    assert excluded, "the map claims all 16 columns are candidates; DB6 has only 14"
    for c in excluded:
        assert c["reason_code"], f"{c['channel_id']} is excluded with no reason"
    # A candidate carries no exclusion reason, and vice versa: the two fields are
    # a partition, so a reader never has to guess which one to trust.
    for c in channel_map["channels"]:
        if c["candidate_eligible"]:
            assert c["reason_code"] == ""
            assert c["structural_status"] == qc_mod.STRUCTURAL_PHYSICAL


def test_t05_empty_columns_are_documented_not_guessed(channel_map) -> None:
    empty = [c for c in channel_map["channels"] if c["structural_status"] == qc_mod.STRUCTURAL_EMPTY]
    assert len(empty) == 2
    for c in empty:
        assert c["reason_code"] == "structurally_empty_column_documented"
        # The evidence is per file: "constant everywhere we looked" is a property of
        # the montage, whereas "constant in one file" would be a dead electrode.
        assert c["flag_counts"].get(qc_mod.CONSTANT_IN_FILE) == c["n_files_observed"]
        assert len(c["evidence_files"][qc_mod.CONSTANT_IN_FILE]) == c["n_files_observed"]


def test_t05_map_bookkeeping_is_internally_consistent(channel_map) -> None:
    physical = [c["channel_id"] for c in channel_map["channels"] if c["structural_status"] == "physical"]
    empty = [
        c["channel_id"] for c in channel_map["channels"] if c["structural_status"] == "structurally_empty"
    ]
    assert channel_map["physical_channels"] == physical
    assert channel_map["structurally_empty_channels"] == empty
    assert channel_map["n_physical"] == len(physical)
    assert channel_map["n_structurally_empty"] == len(empty)
    assert channel_map["n_candidate_eligible"] == len(channel_map["candidate_channels"])
    assert set(channel_map["physical_channels"]) | set(channel_map["structurally_empty_channels"]) == {
        f"CH{n:02d}" for n in range(1, 17)
    }
    assert not set(channel_map["physical_channels"]) & set(channel_map["structurally_empty_channels"])


def test_t05_soft_flags_never_disqualify(channel_map) -> None:
    """A quiet or mains-contaminated channel is reported, not dropped.

    Dropping on a soft flag would make the channel ranking a self-fulfilling
    prophecy: the pilot exists precisely to let a quiet channel prove itself.
    """
    soft = {"dead_in_file", "low_amplitude", "high_mains", "clipping", qc_mod.CONSTANT_IN_FILE}
    for c in channel_map["channels"]:
        if c["structural_status"] != qc_mod.STRUCTURAL_PHYSICAL:
            continue
        assert c["candidate_eligible"] is True, (
            f"{c['channel_id']} is physical but not a candidate on soft flags {c['flag_counts']}"
        )
        assert not (set(c["flag_counts"]) & {"nan_inf"}), c["channel_id"]
        assert set(c["flag_counts"]) <= soft | {"nan_inf"}


def test_t05_no_physical_channel_is_dropped_by_position(channel_map) -> None:
    """CH01-CH07 and CH16 must be present as candidates, whatever their amplitude."""
    must_survive = {f"CH{n:02d}" for n in (*range(1, 8), 16)}
    assert must_survive <= set(channel_map["candidate_channels"])
    assert len(channel_map["candidate_channels"]) == 14


# ---------------------------------------------------------------------------
# T06 -- the 14/2 expectation is checked, and a mismatch blocks
# ---------------------------------------------------------------------------

def test_t06_montage_matches_documentation(channel_map, cfg) -> None:
    verdict = qc_mod.check_expectation(channel_map, cfg.data)
    assert verdict["ok"] is True
    assert verdict["observed_physical"] == 14
    assert verdict["observed_structurally_empty"] == 2
    assert verdict["physical_channels"] == channel_map["physical_channels"]


def test_t06_mismatch_blocks_with_evidence(channel_map, cfg) -> None:
    broken = {**channel_map, "n_physical": 13, "n_structurally_empty": 3}
    verdict = qc_mod.check_expectation(broken, cfg.data)
    assert verdict["ok"] is False
    assert "blocked" in verdict["message"].lower()
    # The message must name both the expectation and the observation, and must say
    # out loud that the thresholds were not moved.
    assert "expected 14 physical / 2 empty" in verdict["message"]
    assert "observed 13 / 3" in verdict["message"]
    assert "not adjusted" in verdict["message"]


def test_t06_blocked_stage_is_reported_as_blocked(root) -> None:
    """The QC report's status is `blocked` exactly when the montage disagrees."""
    import json

    path = require(root / "reports" / "qc.json", "A1 report")
    report = json.loads(path.read_text(encoding="utf-8"))
    expectation = report["metrics"]["montage_expectation"]
    if expectation["ok"]:
        assert report["status"] == "ok"
    else:
        assert report["status"] == "blocked"


def test_t06_structural_status_needs_every_file_constant() -> None:
    """One live file makes a column physical, even if 19 files are flat."""
    flat = [_row(f"S{s}_D1_T1.mat", 7, [qc_mod.CONSTANT_IN_FILE]) for s in range(1, 20)]
    assert qc_mod.structural_status_of(flat, 7) == qc_mod.STRUCTURAL_EMPTY
    with_one_live = flat + [_row("S20_D1_T1.mat", 7, [])]
    assert qc_mod.structural_status_of(with_one_live, 7) == qc_mod.STRUCTURAL_PHYSICAL


def test_t06_structural_status_requires_rows() -> None:
    with pytest.raises(ValueError, match="no QC rows"):
        qc_mod.structural_status_of([_row("S1_D1_T1.mat", 0, [])], 5)


# ---------------------------------------------------------------------------
# T07 -- every eligible channel was screened
# ---------------------------------------------------------------------------

def test_t07_pilot_covers_every_candidate_channel(root, channel_map) -> None:
    path = require(root / "results" / "channel_pilot.csv", "Stage B pilot table")
    pilot = pd.read_csv(path)
    assert set(pilot["channel"]) == set(channel_map["candidate_channels"])
    assert len(pilot) == channel_map["n_candidate_eligible"] == 14
    assert pilot["channel"].is_unique
    # And the two structurally empty columns were NOT trained on.
    assert not set(pilot["channel"]) & set(channel_map["structurally_empty_channels"])


def test_t07_pilot_uses_one_seed_and_one_model(root, cfg) -> None:
    path = require(root / "results" / "channel_pilot.csv", "Stage B pilot table")
    pilot = pd.read_csv(path)
    assert set(pilot["model"]) == {cfg.models["screening_model"]}
    screening_seeds = [int(s) for s in cfg.training["seeds"]["screening"]]
    assert set(int(s) for s in pilot["seed"]) == set(screening_seeds)
    assert len(screening_seeds) == 1


def test_t07_pilot_scores_exactly_the_pilot_subjects(root, channel_map, cfg) -> None:
    """Stage B screens on the three pilot subjects; all ten arrive at Stage C."""
    path = require(root / "results" / "channel_pilot_subject.csv", "Stage B subject table")
    subject_rows = pd.read_csv(path)
    pilot_subjects = {int(s) for s in cfg.experiment["selection"]["pilot_subjects"]}
    assert pilot_subjects == {1, 3, 7}
    expected = len(channel_map["candidate_channels"]) * len(pilot_subjects)
    assert len(subject_rows) == expected, f"expected {expected} subject rows, got {len(subject_rows)}"
    for channel, part in subject_rows.groupby("channel"):
        assert set(part["subject"]) == pilot_subjects, channel
        assert part["n_windows"].sum() > 0


def test_t07_pilot_numbers_are_finite_and_above_chance(root, cfg) -> None:
    path = require(root / "results" / "channel_pilot.csv", "Stage B pilot table")
    pilot = pd.read_csv(path)
    for column in ("val_subject_mean_accuracy", "best_val_accuracy", "best_val_loss"):
        assert pilot[column].map(math.isfinite).all(), column
    num_classes = cfg.num_classes
    assert (pilot["val_subject_mean_accuracy"] > 1.0 / num_classes).all()
    assert (pilot["val_subject_mean_accuracy"] <= 1.0).all()
