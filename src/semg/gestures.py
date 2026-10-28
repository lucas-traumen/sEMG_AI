"""The raw-label <-> class-id bijection, read from configs/gestures.yaml.

DB6's `restimulus` uses sparse, non-contiguous ids (0, 1, 3, 4, 6, 9, 10, 11).
The model's output layer uses compact 0..7. Getting this mapping wrong is silent:
accuracy would still be computed, just against the wrong class. So the mapping is
built once, asserted to be a bijection, and every conversion goes through it.

No gesture name appears in this file or anywhere else in src/ -- names live in the
config (AGENT.md rule 3), and code only ever handles ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


class LabelMapError(RuntimeError):
    """Raised when the config's label mapping is not a bijection."""


@dataclass(frozen=True)
class LabelMap:
    raw_to_class: dict[int, int]
    class_to_raw: dict[int, int]
    names: dict[int, str]
    rest_class_id: int

    @property
    def num_classes(self) -> int:
        return len(self.raw_to_class)

    @property
    def raw_labels(self) -> list[int]:
        return sorted(self.raw_to_class)

    def to_class(self, raw: np.ndarray) -> np.ndarray:
        """Vectorised raw -> class id. Unknown raw labels raise, never map to 0."""
        raw = np.asarray(raw).reshape(-1)
        out = np.full(raw.shape, -1, dtype=np.int64)
        for r, c in self.raw_to_class.items():
            out[raw == r] = c
        if (out < 0).any():
            bad = np.unique(raw[out < 0])[:10].tolist()
            raise LabelMapError(f"raw labels not in the configured mapping: {bad}")
        return out

    def to_raw(self, class_id: np.ndarray) -> np.ndarray:
        class_id = np.asarray(class_id).reshape(-1)
        out = np.full(class_id.shape, -1, dtype=np.int64)
        for c, r in self.class_to_raw.items():
            out[class_id == c] = r
        if (out < 0).any():
            bad = np.unique(class_id[out < 0])[:10].tolist()
            raise LabelMapError(f"class ids not in the configured mapping: {bad}")
        return out

    def name_of(self, class_id: int) -> str:
        return self.names[int(class_id)]

    def ordered_names(self) -> list[str]:
        return [self.names[c] for c in sorted(self.names)]


def build(gestures_cfg: dict[str, Any]) -> LabelMap:
    """Validate and index the configured mapping."""
    items = gestures_cfg.get("gestures")
    if not isinstance(items, list) or not items:
        raise LabelMapError("gestures.yaml must define a non-empty `gestures` list")

    raw_to_class: dict[int, int] = {}
    class_to_raw: dict[int, int] = {}
    names: dict[int, str] = {}

    for item in items:
        raw = int(item["raw_label"])
        cls = int(item["class_id"])
        if raw in raw_to_class:
            raise LabelMapError(f"duplicate raw_label {raw}")
        if cls in class_to_raw:
            raise LabelMapError(f"duplicate class_id {cls}")
        raw_to_class[raw] = cls
        class_to_raw[cls] = raw
        names[cls] = str(item["name"])

    declared = int(gestures_cfg["num_classes"])
    if len(raw_to_class) != declared:
        raise LabelMapError(
            f"gestures.yaml declares num_classes={declared} but lists {len(raw_to_class)} entries"
        )
    expected_classes = set(range(declared))
    if set(class_to_raw) != expected_classes:
        raise LabelMapError(
            f"class_id values must be exactly 0..{declared - 1}, got {sorted(class_to_raw)}"
        )

    rest = int(gestures_cfg["rest_class_id"])
    if rest not in class_to_raw:
        raise LabelMapError(f"rest_class_id {rest} is not one of the declared classes")
    if class_to_raw[rest] != 0:
        raise LabelMapError(
            f"rest_class_id {rest} maps to raw label {class_to_raw[rest]}, expected 0"
        )

    return LabelMap(
        raw_to_class=raw_to_class,
        class_to_raw=class_to_raw,
        names=names,
        rest_class_id=rest,
    )


def validate_against_data(label_map: LabelMap, observed: Iterable[int], *, file: str) -> None:
    """Assert a file's observed raw labels are exactly the configured set."""
    observed_set = {int(v) for v in observed}
    expected = set(label_map.raw_to_class)
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        extra = sorted(observed_set - expected)
        raise LabelMapError(
            f"{file}: raw label set differs from config. missing={missing} unexpected={extra}"
        )
