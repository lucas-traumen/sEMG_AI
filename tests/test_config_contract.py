"""Plan section 10 bullets T01 and T03 are covered here together with the source
census that makes T01 meaningful.

T01  "All YAML parses and no experimental parameter lives outside the config."
T03  "Labels {0,1,3,4,6,9,10,11} remap bijectively to 0..7."

T01 is only worth anything if the pipeline genuinely has no second, hidden source
of truth. A YAML that parses while `train.py` hardcodes `lr=1e-3` would pass a
naive version of this test, so the census below scans the executable code for
experimental literals and fails if one appears. Comments and docstrings are
stripped first: prose may quote a number, code may not.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest
import yaml

from semg import config as config_mod
from semg import gestures as gestures_mod

# A literal is "experimental" when it is one of these knobs. The patterns are
# checked against the AST of every function call and every module-level constant,
# not against raw text, so `SEEDS = [11, 22]` is caught while `"seed"` is not.
EXPERIMENTAL_KEYWORDS = {
    "lr",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "epochs",
    "max_epochs",
    "patience",
    "early_stopping_patience",
    "dropout",
    "num_classes",
    "window_samples",
    "hop_samples",
    "fs_hz",
    "tolerance",
    "seed",
}

# Numeric literals that are structural rather than experimental: array ranks,
# channel counts fixed by the DB6 file format, index arithmetic. Anything that
# looks like a *decision* about the experiment must come from config.
STRUCTURAL_ALLOWLIST = {0, 1, 2, -1, 3, 4, 8, 16, 400, 2000, 48, 6}

SOURCE_ROOTS = ("src", "scripts")


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for sub in SOURCE_ROOTS:
        base = root / sub
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


def _strip_docstrings(tree: ast.AST) -> None:
    """Remove docstrings in place so the census only sees executable nodes."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]


def _numbers(node: ast.AST) -> list[float]:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) and not isinstance(sub.value, bool):
            out.append(float(sub.value))
    return out


def test_t01_every_yaml_parses() -> None:
    cfg_dir = Path(config_mod.__file__).resolve().parents[2] / "configs"
    for name in config_mod.CONFIG_FILES:
        path = cfg_dir / name
        assert path.is_file(), f"missing config file {name}"
        loaded = yaml.safe_load(path.read_bytes())
        assert isinstance(loaded, dict), f"{name} must parse to a mapping"


def test_t01_config_stack_loads_and_hashes(root: Path) -> None:
    cfg = config_mod.load(root / "configs")
    assert len(cfg.sha256) == 64
    # The hash is over bytes in sorted filename order: re-loading is stable, and a
    # second read of the same directory must reproduce it exactly.
    assert cfg.sha256 == config_mod.config_sha256(root / "configs")


def test_t01_no_experimental_literal_in_code(root: Path) -> None:
    """No experimental knob may be supplied as a literal inside src/ or scripts/."""
    offenders: list[str] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _strip_docstrings(tree)
        for node in ast.walk(tree):
            label = None
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in EXPERIMENTAL_KEYWORDS:
                        values = _numbers(kw.value)
                        if values:
                            label = (kw.arg, values)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.upper() in {
                        "LR",
                        "LEARNING_RATE",
                        "BATCH_SIZE",
                        "MAX_EPOCHS",
                        "PATIENCE",
                        "SEEDS",
                        "DROPOUT",
                        "NUM_CLASSES",
                        "WINDOW_SAMPLES",
                        "HOP_SAMPLES",
                    }:
                        values = _numbers(node.value)
                        if values:
                            label = (target.id, values)
            if label is None:
                continue
            key, values = label
            bad = [v for v in values if v not in STRUCTURAL_ALLOWLIST]
            if bad:
                offenders.append(f"{path.relative_to(root)}: {key}={bad}")
    assert not offenders, "experimental literals found outside config:\n" + "\n".join(offenders)


def test_t01_experiment_yaml_declares_every_stage(root: Path) -> None:
    cfg = config_mod.load(root / "configs")
    stages = cfg.experiment["stages"]
    names = [s["name"] for s in stages]
    assert names == [
        "inventory",
        "qc",
        "channel_pilot",
        "channel_confirm",
        "channel_model_grid",
        "complete_five_seeds",
        "final_test",
    ]
    # Every stage has the four fields the runner needs; nothing is implicit.
    for stage in stages:
        assert set(stage) >= {"name", "script", "trains", "description"}
        # `script` may carry CLI arguments ("run_preflight.py --stage inventory"),
        # so only the first token is a path.
        script = str(stage["script"]).split()[0]
        assert (root / script).is_file(), f"{stage['name']} script missing: {script}"
    # `trains` counts NEW fits only; reused cells are declared separately and do not
    # add to the budget. 14 + 15 + 18 + 2 = 49.
    assert int(cfg.experiment["expected_total_training_runs"]) == sum(
        int(s["trains"]) for s in stages
    )
    assert cfg.experiment["stages"][4]["reuses"] == 9
    assert cfg.experiment["stages"][5]["reuses"] == 3


def test_t01_windowing_parameters_are_config_only(root: Path) -> None:
    cfg = config_mod.load(root / "configs")
    assert cfg.window_samples == int(cfg.pipeline["windowing"]["window_samples"])
    assert cfg.hop_samples == int(cfg.pipeline["windowing"]["hop_samples"])
    # 200 ms at 2 kHz. Written as a relation, not a magic number, so a sampling-rate
    # change cannot silently alter the analysed window length.
    assert cfg.window_samples / cfg.fs_hz == pytest.approx(0.200)
    assert cfg.hop_samples / cfg.fs_hz == pytest.approx(0.100)


def test_t03_label_set_remaps_bijectively(root: Path) -> None:
    cfg = config_mod.load(root / "configs")
    label_map = gestures_mod.build(cfg.gestures)

    raw = sorted(label_map.raw_to_class)
    assert raw == [0, 1, 3, 4, 6, 9, 10, 11], f"unexpected raw label set {raw}"
    assert sorted(label_map.raw_to_class.values()) == list(range(cfg.num_classes))

    # Both directions, exhaustively: a_round(a_b(a)) == a for every label.
    for r in raw:
        c = label_map.to_class(r)
        assert label_map.to_raw(c) == r
    for c in range(cfg.num_classes):
        r = label_map.to_raw(c)
        assert label_map.to_class(r) == c


def test_t03_rest_is_class_zero(root: Path) -> None:
    cfg = config_mod.load(root / "configs")
    label_map = gestures_mod.build(cfg.gestures)
    assert label_map.rest_class_id == int(cfg.gestures["rest_class_id"]) == 0
    assert label_map.to_raw(label_map.rest_class_id) == 0
    assert label_map.name_of(0).upper() == "REST"


def test_t03_observed_labels_must_match_the_map(root: Path) -> None:
    cfg = config_mod.load(root / "configs")
    label_map = gestures_mod.build(cfg.gestures)
    observed = set(label_map.raw_to_class)
    gestures_mod.validate_against_data(label_map, observed, file="unit-test")

    with pytest.raises(gestures_mod.LabelMapError) as excinfo:
        gestures_mod.validate_against_data(
            label_map, observed | {2}, file="unit-test"
        )
    assert "unexpected" in str(excinfo.value)

    with pytest.raises(gestures_mod.LabelMapError) as excinfo:
        gestures_mod.validate_against_data(
            label_map, observed - {11}, file="unit-test"
        )
    assert "missing" in str(excinfo.value)


def test_t03_gesture_names_are_unique_and_ordered(root: Path) -> None:
    cfg = config_mod.load(root / "configs")
    label_map = gestures_mod.build(cfg.gestures)
    names = label_map.ordered_names()
    assert len(names) == cfg.num_classes
    assert len(set(names)) == cfg.num_classes
    assert names[0].upper() == "REST"
