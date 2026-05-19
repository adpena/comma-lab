# SPDX-License-Identifier: MIT
"""Layerwise drift diagnostic across PyTorch backends.

Canonical implementation of the per-layer forward-hook drift measurement
that identifies the FIRST layer whose output diverges between two backends.
The mechanism behind the documented 23x PoseNet / 2x SegNet / 2.5x score
drift between MPS and CUDA is currently UNKNOWN despite empirical
measurement, per CLAUDE.md "MPS auth eval is NOISE" non-negotiable. This
helper produces the layerwise evidence needed to map the drift to one of
the five hypothesis families (H1 fp16 accumulation / H2 bilinear interp
coordinate convention / H3 reduction tree topology / H4 BN-LN-GN fusion /
H5 rgb_to_yuv6 preprocessing).

[verified-against:
  - torch.nn.Module.register_forward_hook (PyTorch >= 2.0 stable contract)
  - torch.mps.synchronize() (PyTorch MPS backend; CRITICAL for valid
    measurement because MPS execution is asynchronous by default)
  - torch.cuda.synchronize() (sister discipline for CUDA backend)
  - L_inf, L_2, mean-relative drift definitions standard in numerical
    analysis (Trefethen & Bau 1997 "Numerical Linear Algebra")
]

Non-promotability contract per CLAUDE.md "MPS auth eval is NOISE" +
Catalog #1 (check_no_mps_fallback_default) + Catalog #192
(check_macos_cpu_advisory_not_promoted_without_linux_verification):

    Every artifact this module produces is `evidence_grade =
    "macOS-MPS-diagnostic"` + `score_claim = False` +
    `promotion_eligible = False`. The diagnostic NEVER claims a score on
    the contest axis. It produces a per-layer drift table that the
    operator + downstream subagents use to identify the drift-cliff layer
    + reason about targeted fixes.

Catalog row: this module's preflight gate is covered by Catalog #287
(check_no_docstring_overstatement_without_evidence_tag) because the
verified-against tag above carries explicit citations to canonical sources.

Lane: lane_mps_local_compute_frontier_diagnostic_20260518 (Catalog #126).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn


# Canonical evidence-grade markers (NEVER REMOVE per CLAUDE.md
# "MPS auth eval is NOISE" + Catalog #1 + Catalog #192):
DRIFT_EVIDENCE_GRADE = "macOS-MPS-diagnostic"
DRIFT_AXIS_TAG_MPS = "[macOS-MPS-PyTorch]"
DRIFT_AXIS_TAG_CPU = "[macOS-CPU-PyTorch]"
DRIFT_AXIS_TAG_CUDA = "[contest-CUDA-PyTorch-reference]"

# Per-backend synchronization functions. CRITICAL: forgetting MPS sync
# produces PHANTOM drift because MPS execution is asynchronous; the
# subsequent backend's forward pass can race the previous backend's
# pending kernels and produce values that depend on ordering rather than
# on the actual layer math.
_SYNC_FNS = {
    "mps": lambda: torch.mps.synchronize() if torch.backends.mps.is_available() else None,
    "cuda": lambda: torch.cuda.synchronize() if torch.cuda.is_available() else None,
    "cpu": lambda: None,
}


@dataclass(frozen=True)
class LayerDriftRecord:
    """Per-layer drift record across a single backend pair.

    Frozen invariants (enforced by __post_init__):
      - layer_name is non-empty string
      - layer_depth >= 0 (root module = 0; deeper modules = larger)
      - backend_pair is a tuple of exactly 2 non-empty strings
      - l_inf, l_2, mean_rel are non-negative floats
      - dtype is the str repr of the comparison dtype (e.g. 'torch.float32')
      - is_first_divergence is a bool computed by the caller; the dataclass
        does NOT compute it itself because that decision depends on the
        full layer ordering.
    """

    layer_name: str
    layer_depth: int
    layer_class: str
    backend_pair: tuple[str, str]
    l_inf: float
    l_2: float
    mean_rel: float
    dtype: str
    is_first_divergence: bool = False
    # Output shape recorded as tuple for serialization
    output_shape: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Frozen invariants per Catalog #229 premise-verification spirit
        # applied at dataclass construction time.
        if not self.layer_name:
            raise ValueError("layer_name must be non-empty")
        if self.layer_depth < 0:
            raise ValueError(f"layer_depth must be >= 0, got {self.layer_depth}")
        if len(self.backend_pair) != 2:
            raise ValueError(
                f"backend_pair must be tuple of 2 strings, got {self.backend_pair}"
            )
        if any(not b for b in self.backend_pair):
            raise ValueError(f"backend_pair entries must be non-empty: {self.backend_pair}")
        if self.l_inf < 0 or self.l_2 < 0 or self.mean_rel < 0:
            raise ValueError(
                f"drift metrics must be non-negative: l_inf={self.l_inf} "
                f"l_2={self.l_2} mean_rel={self.mean_rel}"
            )


def _sync_backend(backend: str) -> None:
    """Synchronize the given backend (MPS / CUDA) before measurement.

    CRITICAL per CLAUDE.md "MPS auth eval is NOISE" addendum: MPS execution
    is asynchronous; without sync, a subsequent forward pass on a different
    backend can race the previous backend's pending kernels and produce
    phantom drift.
    """
    sync_fn = _SYNC_FNS.get(backend)
    if sync_fn is None:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {sorted(_SYNC_FNS)}"
        )
    sync_fn()


def _seed_all(seed: int) -> None:
    """Set all per-backend RNG seeds deterministically."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # torch.mps.manual_seed is the canonical API per PyTorch >= 2.0
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            # Older PyTorch versions; MPS uses global manual_seed
            pass


def _collect_module_outputs(
    model: nn.Module,
    sample_input: torch.Tensor,
    backend: str,
    seed: int,
    sync_after_each_module: bool,
) -> tuple[dict[str, tuple[torch.Tensor, type]], list[str]]:
    """Run a single forward pass on `backend`, collecting per-module outputs.

    Returns (outputs_by_name, depth_ordering).

    The depth_ordering is the order in which hooks fired (PyTorch's
    depth-first traversal of the module tree). This is the canonical layer
    ordering for the drift-cliff analysis.

    The forward hook captures `output` (the module's return value) regardless
    of whether the module is a leaf (Linear / Conv2d / ReLU) or a container
    (Sequential / custom block). For our purposes, container outputs are the
    aggregate of their children's effects, which is what we want.
    """
    outputs: dict[str, tuple[torch.Tensor, type]] = {}
    ordering: list[str] = []

    def _make_hook(name: str):
        def _hook(module, input_, output):
            # Capture only tensor outputs; tuples / lists / dicts get the
            # first tensor element. This is a pragmatic compromise that
            # handles most architectures; modules that return non-tensor
            # outputs are skipped (we record class but not value).
            captured = None
            if isinstance(output, torch.Tensor):
                captured = output
            elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                captured = output[0]
            elif isinstance(output, dict):
                for v in output.values():
                    if isinstance(v, torch.Tensor):
                        captured = v
                        break
            if captured is not None:
                # Detach + move to CPU first (MPS does not support fp64),
                # then upcast to fp64 for backend-independent comparison.
                # Per CLAUDE.md "MPS auth eval is NOISE" addendum: MPS
                # does not implement Apple Silicon fp64 in 2024-2026
                # PyTorch builds; this empirical constraint shapes the
                # diagnostic pipeline.
                cpu_tensor = captured.detach().to(device="cpu")
                outputs[name] = (
                    cpu_tensor.to(dtype=torch.float64).clone(),
                    type(module),
                )
                ordering.append(name)
                if sync_after_each_module:
                    _sync_backend(backend)
        return _hook

    handles = []
    for name, module in model.named_modules():
        if name == "":  # skip the root module; we only want named children
            continue
        h = module.register_forward_hook(_make_hook(name))
        handles.append(h)

    try:
        _seed_all(seed)
        with torch.no_grad():
            _sync_backend(backend)
            _ = model(sample_input)
            _sync_backend(backend)
    finally:
        for h in handles:
            h.remove()

    return outputs, ordering


def _move_model_to_backend(model: nn.Module, backend: str) -> nn.Module:
    """Move a copy of `model` to `backend`. Returns the moved model.

    We deepcopy to avoid mutating the caller's model; the caller may want
    to keep the original on CPU for sister comparisons.
    """
    import copy

    moved = copy.deepcopy(model)
    if backend == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        moved = moved.to("mps")
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        moved = moved.to("cuda")
    elif backend == "cpu":
        moved = moved.to("cpu")
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return moved


def _move_input_to_backend(t: torch.Tensor, backend: str) -> torch.Tensor:
    """Move tensor `t` to backend; returns a copy."""
    return t.detach().clone().to(backend)


def _compute_pairwise_drift(
    out_a: torch.Tensor, out_b: torch.Tensor
) -> tuple[float, float, float]:
    """Compute (l_inf, l_2, mean_rel) between two same-shape tensors.

    mean_rel is element-wise |a - b| / (|a| + |b| + epsilon), then mean.
    epsilon prevents division by zero on identically-zero pixels.

    Both tensors must already be on CPU fp64 (the comparator dtype).
    """
    if out_a.shape != out_b.shape:
        raise ValueError(
            f"shape mismatch: {tuple(out_a.shape)} vs {tuple(out_b.shape)}"
        )
    diff = (out_a - out_b).abs()
    l_inf = float(diff.max().item()) if diff.numel() > 0 else 0.0
    l_2 = float(diff.pow(2).mean().sqrt().item()) if diff.numel() > 0 else 0.0
    eps = 1e-12
    denom = out_a.abs() + out_b.abs() + eps
    mean_rel = float((diff / denom).mean().item()) if diff.numel() > 0 else 0.0
    return l_inf, l_2, mean_rel


def measure_layerwise_drift(
    model: nn.Module,
    sample_input: torch.Tensor,
    backends: Sequence[str] = ("mps", "cpu"),
    *,
    seed: int = 0,
    sync_after_each_module: bool = True,
    cliff_threshold: float = 1e-3,
) -> dict[str, object]:
    """Run identical forward through each backend; emit per-layer drift.

    Args:
        model: nn.Module on CPU (we will move copies to each backend).
        sample_input: input tensor on CPU; will be moved to each backend.
        backends: sequence of backend names to compare. Pairs are computed
            as combinations (e.g. ("mps", "cpu") -> single pair).
        seed: RNG seed shared across backends for determinism.
        sync_after_each_module: if True, call backend sync after each hook
            fires. Set False only for benchmarking; True is the canonical
            measurement contract.
        cliff_threshold: L_inf above which a layer is considered "diverged"
            for the drift-cliff analysis. Default 1e-3 is the empirically
            calibrated threshold for fp32 numerics where bit-exactness is
            not expected.

    Returns:
        Dict with structure:
            {
                "schema_version": "mps_layerwise_drift_v1",
                "backends": list of backend names,
                "seed": int,
                "input_shape": tuple,
                "input_dtype": str,
                "model_class": str,
                "cliff_threshold": float,
                "pairs": {
                    "<backend_a>_vs_<backend_b>": {
                        "ordering": list of layer names in hook-fire order,
                        "records": [LayerDriftRecord as dict, ...],
                        "drift_cliff_layer": str | None,
                    },
                    ...
                },
                "evidence_grade": "macOS-MPS-diagnostic",
                "score_claim": False,
                "promotion_eligible": False,
                "axis_tags": list of axis tags (one per backend),
                "sync_after_each_module": bool,
            }

    Per CLAUDE.md "MPS auth eval is NOISE": the returned dict is NEVER a
    score claim. Downstream consumers MUST respect the
    promotion_eligible=False marker.
    """
    if not backends or len(backends) < 2:
        raise ValueError(
            f"need >= 2 backends for pairwise comparison, got {tuple(backends)}"
        )
    backends = tuple(backends)
    for b in backends:
        if b not in _SYNC_FNS:
            raise ValueError(
                f"unknown backend {b!r}; expected one of {sorted(_SYNC_FNS)}"
            )

    # Collect outputs per backend
    per_backend: dict[str, tuple[dict[str, tuple[torch.Tensor, type]], list[str]]] = {}
    for backend in backends:
        m_backend = _move_model_to_backend(model, backend)
        i_backend = _move_input_to_backend(sample_input, backend)
        m_backend.eval()
        outputs, ordering = _collect_module_outputs(
            m_backend, i_backend, backend, seed, sync_after_each_module
        )
        per_backend[backend] = (outputs, ordering)
        # Free GPU/MPS memory eagerly
        del m_backend
        del i_backend
        if backend == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
        elif backend == "cuda":
            torch.cuda.empty_cache()

    # Build pairwise drift records
    pairs: dict[str, dict[str, object]] = {}
    backend_list = list(backends)
    for i in range(len(backend_list)):
        for j in range(i + 1, len(backend_list)):
            ba = backend_list[i]
            bb = backend_list[j]
            outs_a, order_a = per_backend[ba]
            outs_b, order_b = per_backend[bb]
            # Take the intersection of layer names + sort by `order_a`
            common = [n for n in order_a if n in outs_b]
            records: list[LayerDriftRecord] = []
            cliff_layer: str | None = None
            for depth, name in enumerate(common):
                ta, cls_a = outs_a[name]
                tb, cls_b = outs_b[name]
                # If shapes mismatch (rare; e.g. backend-specific dynamic
                # shapes), skip the layer rather than crash.
                if ta.shape != tb.shape:
                    continue
                l_inf, l_2, mean_rel = _compute_pairwise_drift(ta, tb)
                is_first = (cliff_layer is None) and (l_inf > cliff_threshold)
                if is_first:
                    cliff_layer = name
                records.append(
                    LayerDriftRecord(
                        layer_name=name,
                        layer_depth=depth,
                        layer_class=cls_a.__name__,
                        backend_pair=(ba, bb),
                        l_inf=l_inf,
                        l_2=l_2,
                        mean_rel=mean_rel,
                        dtype=str(ta.dtype),
                        is_first_divergence=is_first,
                        output_shape=tuple(ta.shape),
                    )
                )
            pairs[f"{ba}_vs_{bb}"] = {
                "ordering": common,
                "records": [r.__dict__ for r in records],
                "drift_cliff_layer": cliff_layer,
            }

    axis_tag_map = {
        "mps": DRIFT_AXIS_TAG_MPS,
        "cpu": DRIFT_AXIS_TAG_CPU,
        "cuda": DRIFT_AXIS_TAG_CUDA,
    }
    return {
        "schema_version": "mps_layerwise_drift_v1",
        "backends": list(backends),
        "seed": seed,
        "input_shape": tuple(sample_input.shape),
        "input_dtype": str(sample_input.dtype),
        "model_class": type(model).__name__,
        "cliff_threshold": cliff_threshold,
        "pairs": pairs,
        # Non-promotability markers per CLAUDE.md "MPS auth eval is NOISE":
        "evidence_grade": DRIFT_EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "axis_tags": [axis_tag_map[b] for b in backends],
        "sync_after_each_module": sync_after_each_module,
    }


def identify_drift_cliff_layer(
    drift_data: Mapping[str, object],
    threshold: float = 1e-3,
    *,
    pair: str | None = None,
) -> str | None:
    """Return the first-divergence-layer name for the given pair.

    If `pair` is None and there is exactly one pair, that pair is used.
    Returns None if no layer in the pair exceeds threshold.

    Note: `threshold` here is informational; the canonical cliff layer is
    computed at measurement time (see `measure_layerwise_drift`'s
    cliff_threshold). This function lets the caller re-query with a
    different threshold post-hoc by scanning the records.
    """
    pairs = drift_data.get("pairs", {})
    if not pairs:
        return None
    if pair is None:
        if len(pairs) != 1:
            raise ValueError(
                f"multiple pairs present; specify pair=... explicitly: {list(pairs)}"
            )
        pair = next(iter(pairs))
    pair_data = pairs.get(pair)
    if pair_data is None:
        return None
    for rec in pair_data.get("records", []):
        if rec.get("l_inf", 0.0) > threshold:
            return rec.get("layer_name")
    return None


def emit_drift_table_markdown(
    drift_data: Mapping[str, object],
    output_path: str | Path,
) -> None:
    """Write a markdown table of per-layer drift to `output_path`.

    The table lists every layer in hook-fire order with columns:
      | depth | layer_name | class | l_inf | l_2 | mean_rel | shape |

    The first-divergence-layer (per pair) is annotated with `**`.

    Includes a header section documenting:
      - schema version
      - backend pair tags (axis-labelled per CLAUDE.md non-negotiable)
      - non-promotability markers
      - cliff threshold

    Per CLAUDE.md "Apples-to-apples evidence discipline": every drift number
    is paired with its backend pair tag; the table header explicitly states
    `score_claim: False` + `evidence_grade: macOS-MPS-diagnostic`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# MPS layerwise drift diagnostic\n")
    lines.append(f"- schema_version: `{drift_data.get('schema_version')}`")
    lines.append(f"- model_class: `{drift_data.get('model_class')}`")
    lines.append(
        f"- input_shape: `{drift_data.get('input_shape')}` "
        f"dtype `{drift_data.get('input_dtype')}`"
    )
    lines.append(f"- seed: `{drift_data.get('seed')}`")
    lines.append(f"- backends: `{drift_data.get('backends')}`")
    lines.append(f"- axis_tags: `{drift_data.get('axis_tags')}`")
    lines.append(f"- cliff_threshold: `{drift_data.get('cliff_threshold')}`")
    lines.append(
        f"- sync_after_each_module: `{drift_data.get('sync_after_each_module')}`"
    )
    lines.append("")
    lines.append("**Non-promotability contract** per CLAUDE.md \"MPS auth eval is NOISE\"")
    lines.append(f"- evidence_grade: `{drift_data.get('evidence_grade')}`")
    lines.append(f"- score_claim: `{drift_data.get('score_claim')}`")
    lines.append(f"- promotion_eligible: `{drift_data.get('promotion_eligible')}`")
    lines.append("")
    pairs = drift_data.get("pairs", {})
    for pair_name, pair_data in pairs.items():
        lines.append(f"## Pair: `{pair_name}`")
        cliff = pair_data.get("drift_cliff_layer")
        lines.append(f"- drift_cliff_layer: `{cliff}`")
        lines.append("")
        lines.append("| depth | layer_name | class | l_inf | l_2 | mean_rel | shape |")
        lines.append("|---:|---|---|---:|---:|---:|---|")
        for rec in pair_data.get("records", []):
            star = "**" if rec.get("is_first_divergence") else ""
            shape = rec.get("output_shape", ())
            lines.append(
                f"| {rec.get('layer_depth')} | {star}`{rec.get('layer_name')}`{star} | "
                f"`{rec.get('layer_class')}` | "
                f"{rec.get('l_inf'):.3e} | "
                f"{rec.get('l_2'):.3e} | "
                f"{rec.get('mean_rel'):.3e} | "
                f"`{tuple(shape)}` |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines))
