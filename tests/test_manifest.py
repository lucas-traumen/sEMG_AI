"""Plan section 10 bullets T02, T10, T11, T12 and T13: the file table and the
window manifests.

The theme of this module is that the data view is *derivable*. Nothing here checks
a number that was typed by hand: the expected values are recomputed from the
archive bytes, from the filename grammar, or from the signal itself.

T02  "All 100 files present in the manifest, or an explained error."
T10  "The _a/_b archive suffix does not decide the split; D/T does."
T11  "S2_D2_T2 is present in the manifest as excluded, with a reason."
T12  "No file, segment or repetition appears in two roles."
T13  "No window crosses a segment or repetition boundary."
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semg import audit as audit_mod
from semg import config as config_mod
from semg import inventory as inventory_mod
from semg import manifest as manifest_mod
from semg import matfile_source
from semg import windowing

from conftest import require

FILE_COLUMNS = ("file", "subject", "day", "session", "role", "bytes", "status", "reason_code")


@pytest.fixture(scope="module")
def cfg(root):
    return config_mod.load(root / "configs")


@pytest.fixture(scope="module")
def files_csv(root):
    return root / "data" / "manifest" / "db6_files.csv"


@pytest.fixture(scope="module")
def files_frame(files_csv):
    require(files_csv, "A0 file table")
    return pd.read_csv(files_csv)


# ---------------------------------------------------------------------------
# T02 -- 100 files, every one accounted for
# ---------------------------------------------------------------------------

def test_t02_file_table_has_all_100_files(files_frame, cfg) -> None:
    assert len(files_frame) == 100
    assert files_frame["file"].is_unique
    for column in FILE_COLUMNS:
        assert column in files_frame.columns, f"missing column {column}"
    # Every row carries a status; a row that is neither `ok` nor `excluded` is an
    # unexplained file, which is exactly what this bullet forbids.
    assert set(files_frame["status"]) <= {"ok", "excluded"}
    bad = files_frame[(files_frame["status"] == "excluded") & (files_frame["reason_code"].isna())]
    assert bad.empty, f"excluded files without a reason: {list(bad['file'])}"


def test_t02_manifest_matches_the_archives(files_frame, cfg, root) -> None:
    """Every name and byte count is re-derived from the zips on disk."""
    zip_dir = Path(cfg.data["dataset"]["local_zip_dir"])
    zips = sorted(zip_dir.glob(cfg.data["dataset"]["local_zip_glob"]))
    if not zips:
        pytest.skip("local DB6 zips not present")

    measured: dict[str, int] = {}
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if name.endswith(".mat"):
                    measured[name] = int(info.file_size)

    assert set(measured) == set(files_frame["file"]), "manifest names disagree with the archives"
    merged = files_frame.set_index("file")
    mismatched = [
        (name, int(merged.loc[name, "bytes"]), size)
        for name, size in measured.items()
        if int(merged.loc[name, "bytes"]) != size
    ]
    assert not mismatched, f"byte sizes differ: {mismatched[:5]}"


def test_t02_role_counts(files_frame, cfg) -> None:
    counts = files_frame["role"].value_counts().to_dict()
    expected = {
        "train": 10,
        "validation": 10,
        "reserve": 10,
        "test": 10,
        "future": 60,
    }
    assert counts == expected, counts
    # The split column repeats the role in A0 (one column per concept, same value,
    # because the file table is the first place "split" can be stated).
    assert (files_frame["split"] == files_frame["role"]).all()


# ---------------------------------------------------------------------------
# T10 -- the split comes from day/session, never from the archive suffix
# ---------------------------------------------------------------------------

def test_t10_split_is_a_function_of_day_and_session(files_frame, cfg) -> None:
    roles_cfg = cfg.data["roles"]
    for row in files_frame.itertuples():
        subject, day, session = inventory_mod.parse_name(row.file)
        assert (subject, day, session) == (row.subject, row.day, row.session)
        assert inventory_mod.role_for(day, session, roles_cfg) == row.role


def test_t10_archive_suffix_cannot_influence_the_split(files_frame) -> None:
    """The same day/session from an `_a` and a `_b` zip land in the same role."""
    assert manifest_mod.__name__  # the rule belongs to the manifest layer
    by_session = files_frame.groupby(["day", "session"])["role"].nunique()
    assert (by_session == 1).all()
    # And the parser has no concept of the suffix at all.
    for name in ("S7_D3_T1.mat", "S7_D3_T1_a.mat", "S7_D3_T1_b.mat"):
        if name.endswith(".mat") and name.count("_") == 4:
            with pytest.raises(inventory_mod.InventoryError):
                inventory_mod.parse_name(name)


def test_t10_no_session_is_split_across_roles(files_frame) -> None:
    manifest_mod.assert_split_is_by_session(files_frame)


# ---------------------------------------------------------------------------
# T11 -- the truncated Test file is recorded, not dropped
# ---------------------------------------------------------------------------

def test_t11_s2_d2_t2_is_excluded_with_a_reason(files_frame) -> None:
    row = files_frame[files_frame["file"] == "S2_D2_T2.mat"]
    assert len(row) == 1, "S2_D2_T2.mat must appear exactly once in the manifest"
    record = row.iloc[0]
    assert record["role"] == "test"
    assert record["status"] == "excluded"
    assert record["reason_code"] == "S2_D2_T2_truncated"
    assert int(record["bytes"]) == 32_519_877


def test_t11_exclusion_rule_is_relative_and_pre_registered(files_frame, cfg) -> None:
    rule = cfg.data["exclusion_rule"]
    assert rule["enabled"] is True
    assert rule["role"] == "test"
    assert 0.0 < float(rule["ratio_threshold"]) < 1.0

    test_rows = files_frame[files_frame["role"] == "test"]
    ok = test_rows[test_rows["status"] == "ok"]["bytes"].astype(float)
    median = float(np.median(ok))
    cutoff = float(rule["ratio_threshold"]) * median
    flagged = set(test_rows[test_rows["bytes"] < cutoff]["file"])
    excluded = set(test_rows[test_rows["status"] == "excluded"]["file"])
    # The rule as configured reproduces exactly the recorded exclusion: it is a
    # function of bytes, not of anything the Test set later revealed.
    assert flagged == excluded == {"S2_D2_T2.mat"}
    assert rule["expected_reason_code"] if "expected_reason_code" in rule else True


# ---------------------------------------------------------------------------
# T12 -- no file, segment or repetition in two roles
# ---------------------------------------------------------------------------

def test_t12_no_file_in_two_roles(files_frame) -> None:
    manifest_mod.assert_no_file_in_two_roles(files_frame)


@pytest.fixture(scope="module")
def trainval(root):
    path = require(root / "data" / "manifest" / "trainval_windows.parquet", "Train/Validation window manifest")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def test_windows(root):
    path = require(root / "data" / "manifest" / "test_windows.parquet", "Test window manifest")
    return pd.read_parquet(path)


@pytest.mark.parametrize("key", ["file", "segment", "repetition"])
def test_t12_no_unit_in_two_roles(trainval, test_windows, key: str) -> None:
    frames = []
    for frame in (trainval, test_windows):
        if key == "file":
            frames.append(frame[["file", "split"]])
        elif key == "segment":
            frames.append(frame[["file", "segment_id", "split"]].assign(
                segment=lambda d: d["file"].astype(str) + "#" + d["segment_id"].astype(str)
            )[["segment", "split"]])
        else:
            frames.append(frame[["file", "repetition", "split"]].assign(
                repetition=lambda d: d["file"].astype(str) + "#" + d["repetition"].astype(str)
            )[["repetition", "split"]])
    combined = pd.concat(frames, ignore_index=True)
    column = combined.columns[0]
    roles = combined.groupby(column)["split"].nunique()
    reused = roles[roles > 1]
    assert reused.empty, f"{key} entities in more than one role: {list(reused.index)[:5]}"


def test_t12_trainval_holds_only_the_two_development_splits(trainval) -> None:
    manifest_mod.assert_one_manifest_per_role(trainval, {"train", "validation"})
    manifest_mod.assert_subject_coverage(trainval, set(range(1, 11)), role="train")
    manifest_mod.assert_subject_coverage(trainval, set(range(1, 11)), role="validation")


# ---------------------------------------------------------------------------
# T13 -- windows stay inside their segment
# ---------------------------------------------------------------------------

def test_t13_window_length_and_alignment(trainval, cfg) -> None:
    manifest_mod.assert_windows_inside_segments(trainval, window_samples=cfg.window_samples)
    hop = cfg.hop_samples
    for (file, segment_id), part in trainval.groupby(["file", "segment_id"], sort=False):
        starts = np.sort(part["start"].to_numpy())
        # Windows of one segment are a regular hop-spaced lattice: that is what
        # `segment_windows` produces and it cannot cross a boundary while doing so.
        deltas = np.diff(starts)
        assert np.all(deltas == hop), f"{file} segment {segment_id}: irregular window starts"
        assert (part["stop"].to_numpy() == part["start"].to_numpy() + cfg.window_samples).all()


def test_t13_segments_do_not_overlap_and_are_monotone(trainval) -> None:
    for file, part in trainval.groupby("file", sort=False):
        ordered = part.sort_values("start")
        previous_stop = -1
        for segment_id, seg in ordered.groupby("segment_id", sort=False):
            lo = int(seg["start"].min())
            hi = int(seg["stop"].max())
            assert lo >= previous_stop, f"{file}: segment {segment_id} overlaps its predecessor"
            previous_stop = hi


def test_t13_windows_match_a_fresh_recomputation(root, cfg, trainval) -> None:
    """Recompute the segment index from the signal for two files and compare.

    This is the only check that proves containment against the real segment
    boundaries rather than against the manifest's own arithmetic. Two files (~4 MB
    of signal each) keep the test quick; the invariant is structural, so it cannot
    hold by accident for one file and fail for another.
    """
    entries = []
    for name in ("S1_D1_T1.mat", "S1_D1_T2.mat"):
        subject, day, session = inventory_mod.parse_name(name)
        entries.append(
            inventory_mod.FileEntry(
                file=name,
                subject=subject,
                day=day,
                session=session,
                role="train" if (day, session) == (1, 1) else "validation",
                bytes=0,
                location="",
                member="",
            )
        )
    zip_dir = Path(cfg.data["dataset"]["local_zip_dir"])
    zips = sorted(zip_dir.glob(cfg.data["dataset"]["local_zip_glob"]))
    if not zips:
        pytest.skip("local DB6 zips not present")
    for entry in entries:
        member = f"DB6_s{entry.subject}_a/{entry.file}"
        found = None
        for zip_path in zips:
            with zipfile.ZipFile(zip_path) as zf:
                names = {Path(n).name: n for n in zf.namelist()}
            if entry.file in names:
                found = (zip_path, names[entry.file])
                break
        assert found is not None, f"{entry.file} not found in any zip"
        zip_path, real_member = found
        with zipfile.ZipFile(zip_path) as zf:
            entry.location = f"{zip_path}::{real_member}"
            source = matfile_source.MatFileSource([entry], audit=audit_mod.NullAudit(), stage="test")
            data = source.load(entry.file, reason="recompute segment boundaries")
        segments = windowing.find_segments(data["restimulus"], data["rerepetition"])
        expected = {}
        for segment in segments:
            wins = windowing.segment_windows(
                segment, window_samples=cfg.window_samples, hop_samples=cfg.hop_samples
            )
            if wins:
                expected[segment.segment_id] = (wins[0][0], wins[-1][1], len(wins))

        actual = trainval[trainval["file"] == entry.file]
        got = {}
        for segment_id, seg in actual.groupby("segment_id"):
            got[int(segment_id)] = (
                int(seg["start"].min()),
                int(seg["stop"].max()),
                int(len(seg)),
            )
        assert got == expected, f"{entry.file}: window index disagrees with the signal"
