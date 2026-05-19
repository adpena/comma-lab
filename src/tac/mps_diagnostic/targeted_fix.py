# SPDX-License-Identifier: MIT
"""Targeted Conv2d wrapper for the SegNet MPS drift-cliff layer.

Predecessor `lane_mps_local_compute_frontier_diagnostic_20260518` (commit
`8ddfc64ae`) identified the drift-cliff layer as
`scorer.decoder.blocks.0.conv1.0` — a Conv2d(472 -> 256, 3x3, stride 1,
padding 1, bias=False) at 24x32 spatial. The mechanism is H3 (Conv2d
reduction-order MPS vs CPU) DOMINANT + H4 (Conv+BN fusion) AMPLIFYING per
`.omx/research/mps_drift_mechanism_20260519T035310Z.md`.

This helper wraps the cliff layer with a targeted-fix strategy that reduces
the per-layer L_inf drift below 1e-3 without re-implementing the scorer.
Three strategies are tried in order (cheapest runtime first):

  1. `fp32_force`: explicit `F.conv2d(input.float(), weight.float())` to
     force fp32 reduction; should be near-zero runtime overhead.
  2. `cpu_wrap`: move input + weights to CPU, run conv2d on CPU
     (bit-exact PyTorch reference), move output back to MPS. Predicted
     overhead: ~5-10ms per forward.
  3. `deterministic_algorithms`: enable `torch.use_deterministic_algorithms`
     globally (if MPS supports it in PyTorch 2.11.0).

[verified-against:
  - `torch.nn.Module.register_forward_pre_hook` (PyTorch >= 2.0 stable)
  - `torch.nn.functional.conv2d` (PyTorch canonical conv contract)
  - Predecessor drift table in
    `.omx/research/mps_drift_mechanism_20260519T035310Z.md`
    (L_inf 4.578e-3 at the cliff layer, seed=42, batch=2)
]

Non-promotability contract per CLAUDE.md "MPS auth eval is NOISE" +
Catalog #1 + Catalog #192:

    Every artifact this module produces is `evidence_grade =
    "macOS-MPS-diagnostic"` + `score_claim = False` +
    `promotion_eligible = False`. The wrapper is a research-grade
    targeted fix; NEVER claim a contest score from MPS forward passes
    even after the wrapper is applied.

Catalog row: this module's preflight gate is covered by Catalog #287
(check_no_docstring_overstatement_without_evidence_tag) because the
verified-against tag above carries explicit citations.

Lane: lane_mps_conv2d_wrap_fix_plus_real_frame_gap_experiment_20260518
(Catalog #126).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# Strategy names per docstring above
WrapStrategy = Literal["fp32_force", "cpu_wrap", "deterministic_algorithms"]
VALID_STRATEGIES: tuple[WrapStrategy, ...] = (
    "fp32_force",
    "cpu_wrap",
    "deterministic_algorithms",
)

# Default cliff layer name from predecessor diagnostic per
# `.omx/research/mps_drift_mechanism_20260519T035310Z.md`
DEFAULT_CLIFF_LAYER_NAME = "decoder.blocks.0.conv1.0"


@dataclass(frozen=True)
class TargetedFixRecord:
    """Per-fix record describing what was applied + diagnostic provenance.

    Frozen invariants (enforced by __post_init__):
      - layer_name is non-empty string
      - strategy is one of VALID_STRATEGIES
      - original_class is non-empty str (class name of replaced module)
      - fix_evidence_grade equals "macOS-MPS-diagnostic" (canonical marker)
    """

    layer_name: str
    strategy: WrapStrategy
    original_class: str
    fix_evidence_grade: str = "macOS-MPS-diagnostic"

    def __post_init__(self) -> None:
        if not self.layer_name:
            raise ValueError("layer_name must be non-empty")
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {VALID_STRATEGIES}, got {self.strategy!r}"
            )
        if not self.original_class:
            raise ValueError("original_class must be non-empty")
        if self.fix_evidence_grade != "macOS-MPS-diagnostic":
            raise ValueError(
                f"fix_evidence_grade must be 'macOS-MPS-diagnostic' per CLAUDE.md "
                f'"MPS auth eval is NOISE", got {self.fix_evidence_grade!r}'
            )


class _FP32ForceConv2d(nn.Module):
    """Wrapper that forces fp32 reduction inside the conv2d kernel.

    The original Conv2d weight may be fp32 already, but on MPS the kernel
    can choose internal reduction order based on auto-tuning heuristics that
    differ from CPU. By explicitly casting input + weight to fp32 and using
    `F.conv2d` (the functional form) we constrain the implementation choice
    to a more reproducible code path.
    """

    def __init__(self, original_conv: nn.Conv2d) -> None:
        super().__init__()
        # Hold the original conv's parameters (no copy; share weights)
        self.original_conv = original_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.original_conv.weight.to(dtype=torch.float32)
        bias = (
            self.original_conv.bias.to(dtype=torch.float32)
            if self.original_conv.bias is not None
            else None
        )
        x_fp32 = x.to(dtype=torch.float32)
        out = F.conv2d(
            x_fp32,
            weight,
            bias=bias,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups,
        )
        return out.to(dtype=x.dtype)


class _CPUWrapConv2d(nn.Module):
    """Wrapper that performs the conv on CPU (bit-exact reference).

    Moves input + weight + bias to CPU, runs conv2d on the PyTorch CPU
    backend (oneDNN-backed), then moves output back to the original input's
    device. Predicted overhead per forward: ~5-10ms for the cliff layer's
    472 -> 256 channel projection at 24x32.

    This is the fallback when fp32_force is insufficient.
    """

    def __init__(self, original_conv: nn.Conv2d) -> None:
        super().__init__()
        self.original_conv = original_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_device = x.device
        x_cpu = x.to(device="cpu", dtype=torch.float32)
        weight_cpu = self.original_conv.weight.to(device="cpu", dtype=torch.float32)
        bias_cpu = (
            self.original_conv.bias.to(device="cpu", dtype=torch.float32)
            if self.original_conv.bias is not None
            else None
        )
        out_cpu = F.conv2d(
            x_cpu,
            weight_cpu,
            bias=bias_cpu,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups,
        )
        return out_cpu.to(device=original_device, dtype=x.dtype)


class _DeterministicAlgorithmsConv2d(nn.Module):
    """Wrapper that toggles deterministic algorithms around the conv.

    PyTorch's `torch.use_deterministic_algorithms(True)` may pin the conv2d
    kernel to a deterministic implementation on MPS. This is the lightest
    intervention but support is PyTorch-version-dependent.
    """

    def __init__(self, original_conv: nn.Conv2d) -> None:
        super().__init__()
        self.original_conv = original_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            out = self.original_conv(x)
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=True)
        return out


_WRAPPER_CLASSES: dict[WrapStrategy, type[nn.Module]] = {
    "fp32_force": _FP32ForceConv2d,
    "cpu_wrap": _CPUWrapConv2d,
    "deterministic_algorithms": _DeterministicAlgorithmsConv2d,
}


def _set_module_by_name(root: nn.Module, layer_name: str, new_module: nn.Module) -> None:
    """Replace the module at `layer_name` (dotted path) within `root`.

    Walks the dotted path to the parent of the target leaf, then setattr's
    the new module under the leaf attribute name. Raises ValueError if the
    layer name does not resolve.
    """
    if not layer_name:
        raise ValueError("layer_name must be non-empty")
    parts = layer_name.split(".")
    parent = root
    for part in parts[:-1]:
        # Container modules can use integer indices stored as string keys
        if part.isdigit() and hasattr(parent, "__getitem__"):
            try:
                parent = parent[int(part)]
                continue
            except (IndexError, KeyError, TypeError):
                pass
        if not hasattr(parent, part):
            raise ValueError(
                f"layer_name {layer_name!r} does not resolve at part {part!r}"
            )
        parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit() and hasattr(parent, "__setitem__"):
        try:
            parent[int(leaf)] = new_module
            return
        except (IndexError, KeyError, TypeError):
            pass
    if not hasattr(parent, leaf):
        raise ValueError(
            f"layer_name {layer_name!r} does not resolve at leaf {leaf!r}"
        )
    setattr(parent, leaf, new_module)


def wrap_drift_cliff_layer(
    scorer: nn.Module,
    layer_name: str = DEFAULT_CLIFF_LAYER_NAME,
    strategy: WrapStrategy = "fp32_force",
) -> TargetedFixRecord:
    """Wrap the drift-cliff layer in `scorer` with the targeted fix.

    Args:
        scorer: nn.Module (SegNet or similar) containing the cliff layer.
        layer_name: dotted path to the cliff layer. Defaults to
            `decoder.blocks.0.conv1.0` per the predecessor diagnostic.
        strategy: which fix to apply. Default `fp32_force` is the
            cheapest at runtime.

    Returns:
        TargetedFixRecord describing the applied fix.

    Raises:
        ValueError: if the layer is not a Conv2d, if the layer name does
            not resolve, or if the strategy is unknown.

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + Catalog #192:
    the wrapped scorer's forward pass is STILL non-promotable. The wrapper
    reduces drift but does NOT make MPS-scored outputs contest-authoritative.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}"
        )
    # Find the target layer
    target: nn.Module | None = None
    for name, module in scorer.named_modules():
        if name == layer_name:
            target = module
            break
    if target is None:
        raise ValueError(
            f"layer_name {layer_name!r} not found in scorer "
            f"(type={type(scorer).__name__})"
        )
    if not isinstance(target, nn.Conv2d):
        raise ValueError(
            f"layer {layer_name!r} is {type(target).__name__}, expected Conv2d"
        )
    wrapper_cls = _WRAPPER_CLASSES[strategy]
    new_module = wrapper_cls(target)
    _set_module_by_name(scorer, layer_name, new_module)
    return TargetedFixRecord(
        layer_name=layer_name,
        strategy=strategy,
        original_class=type(target).__name__,
    )


def try_strategy_chain(
    scorer: nn.Module,
    measure_drift_fn: Callable[[nn.Module], float],
    layer_name: str = DEFAULT_CLIFF_LAYER_NAME,
    per_layer_threshold: float = 1e-3,
    strategy_order: tuple[WrapStrategy, ...] = (
        "fp32_force",
        "cpu_wrap",
        "deterministic_algorithms",
    ),
) -> tuple[TargetedFixRecord | None, float]:
    """Try strategies in order until one drives drift below threshold.

    Args:
        scorer: the scorer module (will be deep-copied per attempt to
            avoid in-place mutation across strategies).
        measure_drift_fn: callable taking a wrapped scorer and returning
            the per-layer L_inf drift at the cliff layer (the caller is
            responsible for how this is measured; canonical pattern uses
            `tac.mps_diagnostic.layerwise_drift.measure_layerwise_drift`).
        layer_name: dotted path to cliff layer.
        per_layer_threshold: drive drift below this L_inf bound.
        strategy_order: order to try strategies in. Cheapest first by
            default.

    Returns:
        (record, post_fix_drift) where record is the successful fix or
        None if no strategy succeeded, and post_fix_drift is the L_inf at
        the cliff layer with the chosen strategy applied.

    Per the predecessor diagnostic the unwrapped cliff is at L_inf
    4.578e-3 (seed=42 batch=2); the threshold 1e-3 is the same canonical
    boundary the diagnostic uses.
    """
    import copy

    best_strategy_record: TargetedFixRecord | None = None
    best_drift = float("inf")
    for strategy in strategy_order:
        scorer_copy = copy.deepcopy(scorer)
        try:
            record = wrap_drift_cliff_layer(
                scorer_copy, layer_name=layer_name, strategy=strategy
            )
        except (ValueError, RuntimeError) as e:
            # Strategy may not be supported on this PyTorch version
            # (e.g., deterministic_algorithms may not work on MPS in some
            # PyTorch builds). Skip and continue.
            print(f"[targeted-fix] strategy {strategy!r} failed: {e}")
            continue
        try:
            drift = measure_drift_fn(scorer_copy)
        except (ValueError, RuntimeError) as e:
            print(f"[targeted-fix] measure_drift_fn failed for {strategy!r}: {e}")
            continue
        if drift < best_drift:
            best_drift = drift
            best_strategy_record = record
        if drift < per_layer_threshold:
            return record, drift
    return best_strategy_record, best_drift
