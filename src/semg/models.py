"""The three 1D-CNNs, plus the two parameter-cost numbers the report must show.

Design constraint: the three architectures differ ONLY in capacity. Same kernel
ladder (7 -> 5 -> 3), same BatchNorm placement, same global average pooling, same
head shape, same dropout. That makes Stage D a capacity comparison rather than an
architecture lottery.

Two cost numbers, deliberately not derived from each other:

    macs_analytic   computed layer by layer from the actual shapes:
                      Conv1d:  out_len * out_ch * in_ch * k
                      Linear:  in_features * out_features
                    One MAC = one multiply-accumulate = one "connection".

    flops_measured  torch's FlopCounterMode, which counts the FLOPs of the ops that
                    actually ran. This is NOT simply 2 x MACs: it counts bias adds,
                    BatchNorm, pooling, and the activation depending on version.

Reporting `flops // 2` as MACs is wrong often enough that this project does not do
it. Both numbers are stored, and the ratio is recorded as information only.

Shape contract: [B, 1, 400] -> [B, 8] raw logits. No softmax in forward(): the
training loss is cross_entropy (which expects logits) and the evaluation path takes
argmax over the same logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .config import Config


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> optional MaxPool.

    bias is False because BatchNorm's beta absorbs it; keeping both would add
    parameters that provably cannot change the function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int,
        pool: bool,
        conv_bias: bool = False,
        batchnorm: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=conv_bias
        )
        self.bn = nn.BatchNorm1d(out_channels) if batchnorm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class HeadBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.fc(x)))


class ConvNet1D(nn.Module):
    """The single architecture family; `name` selects the capacity."""

    def __init__(self, spec: dict[str, Any], common: dict[str, Any], name: str):
        super().__init__()
        self.name = name
        self.spec = spec
        blocks = []
        for b in spec["conv_blocks"]:
            blocks.append(
                ConvBlock(
                    int(b["in_channels"]),
                    int(b["out_channels"]),
                    int(b["kernel_size"]),
                    padding=int(b.get("padding", 0)),
                    pool=bool(b.get("pool", False)),
                    conv_bias=bool(common["conv_bias"]),
                    batchnorm=bool(common["batchnorm"]),
                )
            )
        self.conv = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)

        head = []
        for i, h in enumerate(spec["head"]):
            head.append(
                HeadBlock(
                    int(h["in_features"]),
                    int(h["out_features"]),
                    activation=str(h.get("activation", "none")),
                    dropout=float(h.get("dropout", 0.0)),
                    bias=bool(h.get("bias", True)),
                )
            )
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, L]
        z = self.conv(x)
        z = self.gap(z).flatten(1)
        return self.head(z)


def build(config: Config, name: str) -> ConvNet1D:
    models = config.models["models"]
    if name not in models:
        raise KeyError(f"unknown model {name!r}; configured: {sorted(models)}")
    return ConvNet1D(models[name], config.models["common"], name)


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Analytic MACs
# ---------------------------------------------------------------------------

def macs_analytic(model: ConvNet1D, input_length: int) -> dict[str, Any]:
    """Per-layer multiply-accumulate counts, from the layer shapes.

    Walks the modules in order, tracking the sequence length through the same
    arithmetic the forward pass uses (conv keeps length with `padding=k//2`, pool
    halves it). Fails loud if the computed final length does not reach the global
    average pool with a sane width, because a silent mismatch here would produce a
    wrong cost number that looks plausible.
    """
    per_layer: list[dict[str, Any]] = []
    length = int(input_length)
    total = 0

    for block in model.conv:
        if not isinstance(block, ConvBlock):
            continue
        conv = block.conv
        k = int(conv.kernel_size[0])
        out_len = (length + 2 * int(conv.padding[0]) - k) // int(conv.stride[0]) + 1
        macs = int(out_len * conv.out_channels * conv.in_channels * k)
        total += macs
        per_layer.append(
            {
                "layer": f"conv{len(per_layer) + 1}",
                "type": "Conv1d",
                "in_channels": int(conv.in_channels),
                "out_channels": int(conv.out_channels),
                "kernel_size": k,
                "out_length": int(out_len),
                "macs": macs,
            }
        )
        length = out_len
        if isinstance(block.pool, nn.MaxPool1d):
            length = length // 2

    if length < 1:
        raise ValueError(f"{model.name}: sequence collapsed to {length} before global pooling")

    for head in model.head:
        if not isinstance(head, HeadBlock):
            continue
        fc = head.fc
        macs = int(fc.in_features * fc.out_features)
        total += macs
        per_layer.append(
            {
                "layer": f"fc{len([p for p in per_layer if p['type'] == 'Linear']) + 1}",
                "type": "Linear",
                "in_features": int(fc.in_features),
                "out_features": int(fc.out_features),
                "macs": macs,
            }
        )

    return {"macs": int(total), "per_layer": per_layer, "length_before_gap": int(length)}


def flops_measured(model: ConvNet1D, input_length: int, *, device: str = "cpu") -> int:
    """FLOPs of one forward pass, measured with torch's FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode

    model = model.to(device).eval()
    x = torch.zeros(1, 1, int(input_length), device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    return int(counter.get_total_flops())


def model_stats(config: Config, name: str, *, device: str = "cpu") -> dict[str, Any]:
    """Everything the report needs about one architecture."""
    model = build(config, name)
    length = int(config.models["input"]["length"])
    macs = macs_analytic(model, length)
    params = param_count(model)
    flops = flops_measured(model, length, device=device)

    # A shape check belongs here, not in a comment: the contract is the interface.
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, length))
    expected = (2, int(config.num_classes))
    if tuple(out.shape) != expected:
        raise ValueError(f"{name}: forward produced {tuple(out.shape)}, expected {expected}")

    return {
        "model": name,
        "params": params,
        "expected_params": int(config.models["models"][name].get("expected_params", params)),
        "params_match_config": params
        == int(config.models["models"][name].get("expected_params", params)),
        "macs_analytic": macs["macs"],
        "macs_per_layer": macs["per_layer"],
        "flops_measured": flops,
        # Recorded as information only. Not used to derive MACs.
        "flops_over_macs_ratio": (flops / macs["macs"]) if macs["macs"] else None,
        "input_shape": [1, 1, length],
        "output_shape": [1, int(config.num_classes)],
        "output_is_logits": True,
    }
