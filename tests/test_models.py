"""Plan section 10 bullets T17 and T18, plus the shape contract advertised in
`configs/models.yaml`.

T17  "Every model takes [B,1,400] and returns logits [B,8]."
T18  "Parameter count and MACs are stored for all three models."

"Returns logits" is asserted, not assumed: the raw output must NOT be a probability
distribution. A model that smuggled a softmax into `forward()` would still produce
argmax-identical predictions, so accuracy would look fine while `cross_entropy`
silently trained on already-squashed values. The check is therefore on the output
distribution itself, plus on the absence of a softmax module in the graph.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from semg import config as config_mod
from semg import models as models_mod

from conftest import require

MODEL_NAMES = ("cnn_small", "cnn_base", "cnn_wide")


@pytest.fixture(scope="module")
def cfg(root):
    return config_mod.load(root / "configs")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_t17_model_shape_contract(cfg, name: str) -> None:
    model = models_mod.build(cfg, name)
    model.eval()
    batch = 4
    x = torch.randn(batch, int(cfg.models["input"]["channels"]), int(cfg.models["input"]["length"]))
    with torch.no_grad():
        out = model(x)

    assert tuple(x.shape) == (batch, 1, 400)
    assert tuple(out.shape) == (batch, int(cfg.num_classes)) == (batch, 8)
    assert out.dtype == torch.float32

    # Logits, not probabilities: rows must not sum to 1 and values must not all be
    # inside [0, 1]. A softmax head fails both.
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    assert bool((out < 0).any() or (out > 1).any()), "output looks like probabilities"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_t17_no_softmax_module_in_forward(cfg, name: str) -> None:
    model = models_mod.build(cfg, name)
    assert not any(isinstance(m, nn.Softmax) for m in model.modules())
    assert not any(isinstance(m, nn.LogSoftmax) for m in model.modules())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_t17_model_accepts_a_single_window(cfg, name: str) -> None:
    """Batch size 1 is the inference path; BatchNorm in eval mode must allow it."""
    model = models_mod.build(cfg, name).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, cfg.window_samples))
    assert tuple(out.shape) == (1, cfg.num_classes)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_t18_parameter_count_matches_config(cfg, name: str) -> None:
    model = models_mod.build(cfg, name)
    expected = int(cfg.models["models"][name]["expected_params"])
    actual = models_mod.param_count(model)
    assert actual == expected, f"{name}: {actual} params vs {expected} declared"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_t18_macs_are_layer_by_layer(cfg, name: str) -> None:
    model = models_mod.build(cfg, name)
    result = models_mod.macs_analytic(model, cfg.window_samples)

    assert result["macs"] > 0
    assert result["length_before_gap"] >= 1
    # One entry per linear layer and one per conv block: no layer may be skipped.
    n_conv = sum(1 for b in model.conv if isinstance(b, models_mod.ConvBlock))
    n_fc = len([p for p in result["per_layer"] if p["type"] == "Linear"])
    assert len(result["per_layer"]) == n_conv + n_fc
    assert sum(p["macs"] for p in result["per_layer"]) == result["macs"]

    # MACs must be derived from shapes, never from measured FLOPs / 2.
    for layer in result["per_layer"]:
        if layer["type"] == "Conv1d":
            assert layer["macs"] == (
                layer["out_length"]
                * layer["out_channels"]
                * layer["in_channels"]
                * layer["kernel_size"]
            )
        else:
            assert layer["macs"] == layer["in_features"] * layer["out_features"]


def _stats_by_name(stats: dict) -> dict[str, dict]:
    """`model_stats.json` stores a list under `models` (one entry per model)."""
    entries = stats["models"] if isinstance(stats.get("models"), list) else stats
    return {str(e["model"]): e for e in entries}


def test_t18_model_stats_artifact_covers_all_three(cfg, results_dir) -> None:
    path = require(results_dir / "model_stats.json", "model stats")
    stats = _stats_by_name(json.loads(path.read_text(encoding="utf-8")))

    missing = [n for n in MODEL_NAMES if n not in stats]
    assert not missing, f"model_stats.json missing {missing}"

    for name in MODEL_NAMES:
        entry = stats[name]
        for key in ("params", "macs_analytic", "flops_measured", "params_match_config"):
            assert key in entry, f"{name} missing {key}"
        assert entry["params"] == int(cfg.models["models"][name]["expected_params"])
        assert entry["params_match_config"] is True
        assert entry["output_is_logits"] is True
        assert tuple(entry["input_shape"]) == (1, 1, 400)
        assert tuple(entry["output_shape"]) == (1, 8)
        # The two cost numbers are independent measurements. On CPU `FlopCounterMode`
        # happens to count exactly 2 FLOPs per MAC, so the ratio is 2.0 -- recorded as
        # information, never used to derive one number from the other.
        assert entry["flops_measured"] > entry["macs_analytic"]
        assert entry["flops_over_macs_ratio"] == pytest.approx(
            entry["flops_measured"] / entry["macs_analytic"]
        )


def test_t18_model_stats_ordering_is_by_capacity(cfg, results_dir) -> None:
    """The three models must differ in capacity, monotonically, in a fixed order."""
    path = require(results_dir / "model_stats.json", "model stats")
    stats = _stats_by_name(json.loads(path.read_text(encoding="utf-8")))
    params = [stats[n]["params"] for n in MODEL_NAMES]
    macs = [stats[n]["macs_analytic"] for n in MODEL_NAMES]
    assert params == sorted(params) and len(set(params)) == 3
    assert macs == sorted(macs) and len(set(macs)) == 3
