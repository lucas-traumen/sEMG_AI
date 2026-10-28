"""Config loading, hashing, and the scope assertions that keep the experiment honest.

Every experimental parameter lives in configs/*.yaml (AGENT.md rule 3). This
module is the single place that reads them, so "where does this number come from"
always has one answer.

`config_sha256` is the hash of the whole config directory. It is written into
every RESULT_JSON and into selection_lock.json; run_final_test.py refuses to run
when the current hash no longer matches the locked one. That is what makes
"nothing is tunable on test data" a mechanism instead of a promise.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The config stack, in dependency order. Order matters for readability only; the
# hash is order-independent (sorted by filename).
CONFIG_FILES = (
    "gestures.yaml",
    "data_db6.yaml",
    "pipeline.yaml",
    "models.yaml",
    "training.yaml",
    "experiment.yaml",
)

REQUIRED_SCOPE = {
    "training_scope": "pooled_subjects",
    "selected_channel_scope": "global",
    "subject_role": "reporting_unit_only",
}


class ConfigError(RuntimeError):
    """Raised when the config stack is missing, malformed, or violates scope."""


@dataclass(frozen=True)
class Config:
    """An immutable view of the whole config stack."""

    gestures: dict[str, Any]
    data: dict[str, Any]
    pipeline: dict[str, Any]
    models: dict[str, Any]
    training: dict[str, Any]
    experiment: dict[str, Any]
    config_dir: Path
    sha256: str

    # -- convenience accessors -------------------------------------------------
    @property
    def num_classes(self) -> int:
        return int(self.gestures["num_classes"])

    @property
    def fs_hz(self) -> int:
        return int(self.pipeline["windowing"]["fs_hz"])

    @property
    def window_samples(self) -> int:
        return int(self.pipeline["windowing"]["window_samples"])

    @property
    def hop_samples(self) -> int:
        return int(self.pipeline["windowing"]["hop_samples"])

    @property
    def matrix_columns(self) -> int:
        return int(self.data["matrix"]["matrix_columns"])

    @property
    def seeds_all(self) -> list[int]:
        return [int(s) for s in self.training["seeds"]["all"]]

    @property
    def seeds_confirm(self) -> list[int]:
        return [int(s) for s in self.training["seeds"]["confirm"]]

    @property
    def seeds_extra(self) -> list[int]:
        return [int(s) for s in self.training["seeds"]["extra"]]

    @property
    def selection_tolerance(self) -> float:
        return float(self.experiment["selection"]["tolerance"])

    @property
    def repro_tolerance(self) -> dict[str, float]:
        return self.training["reproducibility"]["tolerance"]

    def path(self, key: str) -> Path:
        """Resolve a path from experiment.yaml:paths, relative to the repo root."""
        rel = self.experiment["paths"][key]
        return (self.config_dir.parent / rel).resolve()

    def channel_id(self, matrix_index: int) -> str:
        """0-based matrix column -> CH01..CH16."""
        return self.data["matrix"]["channel_id_format"].format(matrix_index + 1)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open("rb") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path.name} must parse to a mapping, got {type(loaded).__name__}")
    return loaded


def config_sha256(config_dir: Path) -> str:
    """SHA-256 over every config file's bytes, in filename order.

    Hashes bytes rather than parsed values on purpose: a whitespace-only edit is
    still an edit, and we want the lock to notice it.
    """
    digest = hashlib.sha256()
    for name in sorted(CONFIG_FILES):
        path = config_dir / name
        if not path.is_file():
            raise ConfigError(f"missing config file: {path}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def assert_scope(experiment: dict[str, Any]) -> None:
    """Fail loud if the experiment scope drifted from pooled-subject training.

    A per-subject model would multiply the run count by the number of subjects
    (49 -> 490) and require subject-tagged checkpoints. Both are different
    experiments; neither may happen by accident.
    """
    scope = experiment.get("scope")
    if not isinstance(scope, dict):
        raise ConfigError("experiment.yaml is missing the `scope` block")
    for key, expected in REQUIRED_SCOPE.items():
        actual = scope.get(key)
        if actual != expected:
            raise ConfigError(
                f"scope.{key} must be {expected!r}, found {actual!r}. "
                "Changing this changes the experiment's run count and checkpoint "
                "layout; it is not a tuning knob."
            )
    naming = str(scope.get("checkpoint_naming", ""))
    if "subject" in naming:
        raise ConfigError(
            "scope.checkpoint_naming must not contain a subject component: "
            "this experiment trains one pooled model, not one model per subject."
        )


def load(config_dir: str | os.PathLike[str] = "configs") -> Config:
    """Load and validate the whole config stack."""
    cdir = Path(config_dir).resolve()
    if not cdir.is_dir():
        raise ConfigError(f"config directory not found: {cdir}")

    raw = {name: _read_yaml(cdir / name) for name in CONFIG_FILES}

    exp = raw["experiment.yaml"]
    assert_scope(exp)

    # Cross-file consistency: these pairs are easy to desynchronise and expensive
    # to debug later, so they are checked at load time rather than at run time.
    models = raw["models.yaml"]
    n_classes = int(raw["gestures.yaml"]["num_classes"])
    for name, spec in models["models"].items():
        out_features = spec["head"][-1]["out_features"]
        if int(out_features) != n_classes:
            raise ConfigError(
                f"models.{name} outputs {out_features} classes but gestures.yaml "
                f"declares {n_classes}"
            )
    if int(models["common"]["num_classes"]) != n_classes:
        raise ConfigError("models.common.num_classes disagrees with gestures.yaml")

    pipe = raw["pipeline.yaml"]
    length = int(pipe["windowing"]["window_samples"])
    if int(models["input"]["length"]) != length:
        raise ConfigError(
            f"models.input.length ({models['input']['length']}) must equal "
            f"pipeline.windowing.window_samples ({length})"
        )
    if int(pipe["windowing"]["fs_hz"]) != int(raw["data_db6.yaml"]["sampling"]["fs_hz"]):
        raise ConfigError("pipeline and data_db6 disagree on fs_hz")

    data = raw["data_db6.yaml"]
    mat = data["matrix"]
    if int(mat["matrix_columns"]) != (
        int(mat["expected_physical_channels"]) + int(mat["expected_structural_empty_columns"])
    ):
        raise ConfigError(
            "matrix columns must equal physical + structurally empty channels: "
            f"{mat['matrix_columns']} != {mat['expected_physical_channels']} + "
            f"{mat['expected_structural_empty_columns']}"
        )

    # Scope/tolerance sanity: a tolerance wider than the metric's range would make
    # every comparison a tie.
    tol = float(exp["selection"]["tolerance"])
    if not 0.0 < tol < 0.1:
        raise ConfigError(f"selection.tolerance must be a small positive accuracy delta, got {tol}")

    return Config(
        gestures=raw["gestures.yaml"],
        data=data,
        pipeline=pipe,
        models=models,
        training=raw["training.yaml"],
        experiment=exp,
        config_dir=cdir,
        sha256=config_sha256(cdir),
    )


def dumps(obj: Any) -> str:
    """Deterministic JSON for hashing and for the RESULT_JSON line."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    # numpy scalars leak into dicts constantly (np.int64 keys/values). Casting here
    # keeps every writer honest instead of failing at json.dumps time.
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
