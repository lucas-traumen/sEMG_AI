"""Plan section 10 bullet T26 and the scope invariants behind it.

T26  "No adaptation, deployment or Edge AI artefact is mixed into the current
      result."

The check is deliberately narrow: it scans *executable* code, not prose. The words
"adaptation" and "Edge AI" appear legitimately in PLAN.md, AGENT.md and several
docstrings (this phase is defined partly by what it refuses to do), so a naive
text grep would either fail on the documentation or force the documentation to
lie. What must not exist is a callable code path that trains an adaptation head,
exports a deployment model, or evaluates D3-D5.

The module also asserts the three scope keys that make the run count 49 instead
of 490, and that no subject ever becomes a checkpoint or a model boundary.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from semg import config as config_mod
from semg import metrics as metrics_mod
from semg import runner

SOURCE_ROOTS = ("src", "scripts")
NOTEBOOK_ROOT = "notebooks"

# Names that would only exist if an out-of-scope capability were implemented.
# Matched as whole identifiers against the AST, so `adapt` inside a comment or a
# docstring is invisible and `AdaptationHead` is not.
FORBIDDEN_IDENTIFIERS = {
    # adaptation / few-shot
    "adapt",
    "adaptation",
    "finetune",
    "fine_tune",
    "proto_net",
    "prototypical",
    "prototype_loss",
    "maml",
    "episodic_sampler",
    # calibration on the reserve day
    "calibrate",
    "calibration_session",
    # deployment / edge
    "export_onnx",
    "onnxruntime",
    "tflite",
    "tf_lite",
    "quantize",
    "quantization",
    "prune",
    "pruning",
    "int8",
    "latency_benchmark",
    "flash_bytes",
    "ram_bytes",
}

# Modules that would only be imported by such a capability. `torch.ao.quantization`
# is covered by the `quantize` naming; these are the top-level offenders.
FORBIDDEN_IMPORTS = {
    "onnx",
    "onnxruntime",
    "tensorflow",
    "tflite_runtime",
    "tensorrt",
    "openvino",
    "torch.ao",
    "torch.quantization",
}

# D3-D5 are the `future` bucket. Reading them is out of scope in this phase and is
# enforced at the data layer by `signal_read_allowed: false`; here we assert the
# config still says so.
OUT_OF_SCOPE_ROLES = {"future": False, "reserve": False}


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for sub in (*SOURCE_ROOTS, NOTEBOOK_ROOT):
        base = root / sub
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


def _identifiers(tree: ast.AST) -> set[str]:
    """Every identifier the code actually binds or reads, lowercased."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name.lower())
        elif isinstance(node, ast.arg):
            names.add(node.arg.lower())
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).lower())
    return names


def _imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.fixture(scope="module")
def cfg(root):
    return config_mod.load(root / "configs")


def test_t26_no_out_of_scope_identifier_in_executable_code(root) -> None:
    offenders: list[str] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _identifiers(tree) & FORBIDDEN_IDENTIFIERS
        if hits:
            offenders.append(f"{path.relative_to(root)}: {sorted(hits)}")
    assert not offenders, "out-of-scope capability in executable code:\n" + "\n".join(offenders)


def test_t26_no_deployment_or_adaptation_dependency(root) -> None:
    offenders: list[str] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imports(tree):
            top = module.split(".")[0]
            if module in FORBIDDEN_IMPORTS or any(
                module == bad or module.startswith(bad + ".") for bad in FORBIDDEN_IMPORTS
            ) or top in {"onnx", "onnxruntime", "tensorflow", "tflite_runtime", "tensorrt", "openvino"}:
                offenders.append(f"{path.relative_to(root)}: imports {module}")
    assert not offenders, "out-of-scope dependency:\n" + "\n".join(offenders)


def test_t26_requirements_do_not_pull_in_a_deployment_stack(root) -> None:
    banned = re.compile(
        r"^\s*(onnx|onnxruntime|tensorflow|tflite|tensorrt|openvino|torch-tensorrt|"
        r"ai-edge|edgeimpulse|tinyml)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    for name in ("requirements-common.txt", "requirements-local.txt", "requirements-kaggle.txt"):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert not banned.search(text), f"{name} pulls in an out-of-scope stack"


def test_t26_future_and_reserve_are_not_readable(cfg) -> None:
    roles = cfg.data["roles"]
    for role, allowed in OUT_OF_SCOPE_ROLES.items():
        assert role in roles, f"role {role} vanished from data_db6.yaml"
        assert roles[role]["signal_read_allowed"] is allowed, (
            f"role {role} must stay unreadable in this phase"
        )
    # D3-D5 exist as a bucket so the manifest accounts for 100 files; they are
    # never a modelling target.
    assert roles["future"]["day"] == [3, 4, 5]
    assert roles["future"]["files_expected"] == 60


def test_t26_training_scope_is_pooled(cfg) -> None:
    """49 runs only holds for one global channel and one pooled model."""
    scope = cfg.experiment["scope"]
    assert scope["training_scope"] == "pooled_subjects"
    assert scope["selected_channel_scope"] == "global"
    assert scope["subject_role"] == "reporting_unit_only"
    assert "subject" not in str(scope["checkpoint_naming"])

    stages = cfg.experiment["stages"]
    trains = sum(int(s["trains"]) for s in stages)
    assert trains == 49, f"stage ladder trains {trains} runs, not 49"
    assert cfg.experiment["expected_total_training_runs"] == 49


def test_t26_scope_drift_is_rejected(cfg) -> None:
    """A per-subject experiment must fail loudly, not silently cost 10x."""
    changed = {
        **cfg.experiment,
        "scope": {**cfg.experiment["scope"], "training_scope": "per_subject"},
    }
    with pytest.raises(config_mod.ConfigError, match="training_scope"):
        config_mod.assert_scope(changed)

    named = {
        **cfg.experiment,
        "scope": {
            **cfg.experiment["scope"],
            "checkpoint_naming": "<subject>/<channel>/<model>/seed_<seed>.pt",
        },
    }
    with pytest.raises(config_mod.ConfigError, match="checkpoint_naming"):
        config_mod.assert_scope(named)


def test_t26_no_subject_in_checkpoint_paths(root, cfg) -> None:
    """Checkpoint filenames are <channel>/<model>/seed_<seed>.pt, nothing else."""
    checkpoint_dir = cfg.path("checkpoints_dir")
    if not checkpoint_dir.is_dir():
        pytest.skip("no checkpoints on disk yet")
    pattern = re.compile(r"^CH\d{2}$")
    for path in checkpoint_dir.rglob("*.pt"):
        parts = path.relative_to(checkpoint_dir).parts
        assert len(parts) == 3, f"unexpected checkpoint depth: {path}"
        assert pattern.match(parts[0]), f"channel directory malformed: {parts[0]}"
        assert parts[1] in {"cnn_small", "cnn_base", "cnn_wide"}
        assert re.match(r"^seed_\d+\.pt$", parts[2]), parts[2]


def test_t26_headline_is_never_the_best_seed(cfg) -> None:
    ev = cfg.experiment["evaluation"]
    assert ev["headline_seed_rule"] == "five_seed_mean"
    assert ev["forbid_best_seed_headline"] is True
    assert ev["unit"] == "per_subject_then_per_seed"
    # And the selection block treats the seed as an averaging unit, not a contestant.
    sel = cfg.experiment["selection"]
    assert sel["seed_role"] == "averaging_unit"
    assert sel["forbid_seed_ranking"] is True


def test_t26_accuracy_is_not_a_pooled_ratio() -> None:
    """Two subjects with unequal window counts must weight equally."""
    import numpy as np

    subjects = np.array([1] * 10 + [2] * 1000)
    targets = np.zeros(subjects.shape, dtype=int)
    predictions = {11: np.concatenate([np.zeros(10, dtype=int), np.zeros(1000, dtype=int)])}
    predictions[11][:5] = 1  # subject 1 is 50% wrong, subject 2 is perfect
    rows = metrics_mod.accuracy_by_subject(
        seeds=[11], subjects=subjects, predictions=predictions, targets=targets
    )
    summary = metrics_mod.summarise(rows)
    # Subject-mean = (0.5 + 1.0) / 2 = 0.75. The pooled ratio would be 1005/1010,
    # i.e. 0.995 -- a completely different (and flattering) number.
    assert summary.overall_mean == pytest.approx(0.75)
    assert summary.n_subjects == 2


def test_t26_summary_columns_carry_no_edge_metric() -> None:
    """The run summary records cost as description, never latency/RAM/Flash."""
    banned = {"latency_ms", "ram_bytes", "flash_bytes", "macs_per_second", "fps"}
    assert not (set(runner.SUMMARY_COLUMNS) & banned)
    # Cost fields that ARE recorded, and are description-only.
    assert {"param_count", "macs_analytic", "flops_measured"} <= set(runner.SUMMARY_COLUMNS)
