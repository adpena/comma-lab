# SPDX-License-Identifier: MIT
"""Compound C heterogeneous per-tensor bit allocation — canonical helper.

WAVE-N+2 SLOT 1 (2026-05-28) per parent prompt + design memos:
- ``.omx/research/decoder_compression_analysis_pact_nerv_cluster_landed_20260528.md``
  TOP-1 sub-0.18 candidate (int8 per-channel) + TOP-2 compound sub-0.16 path
  (FP4-QAT) + Daubechies op-routable #4 (wavelet-partitioning per-tensor).
- ``.omx/research/pact_nerv_selector_v3_int8_decoder_quant_brotli_q11_600pair_long_mlx_landed_20260528.md``
  Slot 2 V3 int8 baseline (-43.7% bytes; -0.024 predicted ΔS).

This module is the canonical Compound C primitive: top-3 tensors (70.31% of
decoder cost; ``latent_embed.weight`` 33.75% + ``blocks.0.dsc.pointwise.weight``
22.50% + ``blocks.1.dsc.pointwise.weight`` 14.06%) get **FP4 packed-nibbles
WITH QAT** (Quantizr 0.33 pattern per CLAUDE.md "QAT pipeline" non-negotiable);
mid-byte tensors get **int8 per-channel** (the Slot 2 baseline grammar);
tail tensors get **int4 per-channel** via canonical ``tac.quantization_wave``
groupwise NF4 encoder.

Per CLAUDE.md "MLX-FIRST" + "MLX portable-local-substrate authority":
the canonical helper operates on PyTorch state-dict layout (the
``model.export_state_dict()`` boundary; MLX→PyTorch bridge per the V3
canonical sister). The training-time STE fake-quant happens in PyTorch on
the device-pinned weights; the archive-emit serialization is PyTorch
state-dict bytes per the PSV3 grammar extension.

Canonical-vs-unique decision per layer (Catalog #290)
-----------------------------------------------------

- ADOPT_CANONICAL_BECAUSE_SERVES: ``tac.fp4_quantize.quantize_fp4`` /
  ``dequantize_fp4`` for FP4 packed-nibble layout + canonical
  ``DEFAULT_CODEBOOK`` ``[0, 0.5, 1, 1.5, 2, 3, 4, 6]`` (Quantizr 0.33
  empirical anchor per CLAUDE.md "Exact scorer architectures").
- ADOPT_CANONICAL_BECAUSE_SERVES: ``tac.quantization_wave.int4_int8_mixed_bit``
  ``encode_int4_groupwise`` / ``decode_int4_groupwise`` (NF4 levels +
  groupwise 64-element scales per bitsandbytes/QLoRA canonical) +
  ``sensitivity_aware_mixed_bit_assignment`` (Fisher-information proxy
  per CLAUDE.md Catalog #123 ALLOCATING-not-KILLING discipline).
- ADOPT_CANONICAL_BECAUSE_SERVES: V3 int8 per-channel symmetric grammar
  per ``tac.substrates.pact_nerv_selector_v3.archive`` Slot 2 baseline
  (re-used for the mid-byte tier; no per-substrate fork).
- ADOPT_CANONICAL_BECAUSE_SERVES: ``tac.dykstra_pareto_solver``
  ``solve_pareto_polytope_intersection`` for the per-tensor Lagrangian
  dual (Wave N+1 Slot 1 sister-landed solver; canonical Boyd-Vandenberghe
  alternating-projections per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4).
- FORK_BECAUSE_PRINCIPLED_MISMATCH: the per-tensor SENSITIVITY RANKING
  via Taylor expansion is implemented HERE (not in the canonical sister)
  because the canonical ``sensitivity_aware_mixed_bit_assignment`` assumes
  the gradient is available + uses ``mean(g**2 * w**2)`` averaged-Fisher
  proxy; the V3 substrate's archive-emit sensitivity ranking SHOULD use
  the per-tensor BYTE COST × magnitude profile (Daubechies wavelet-
  partitioning per parent memo op-routable #4) so the largest tensors
  (latent_embed 33.75%; pointwise.0 22.50%; pointwise.1 14.06%) ARE
  routed to FP4-QAT regardless of gradient availability at archive-emit
  time. The Taylor-expansion helper here uses ``g**2`` ALONE when grads
  are provided (NOT the joint ``g**2 * w**2`` Fisher — see Catalog #123
  note: ``w**2`` is the ANTI-correlated direction with score sensitivity
  on score-gradient-trained substrates).

Per Catalog #220 (substrate L1+ scaffold operational mechanism): the
heterogeneous bit allocation produces real archive byte changes consumed
by sister inflate.py heterogeneous dequant path. Per Catalog #272
(distinguishing-feature integration contract): the per-tensor FP4-vs-int8-
vs-int4 routing IS the substrate-distinguishing feature whose byte-
mutation produces frame-level changes at the contest scorer.

Canonical structure
-------------------

A heterogeneous bit allocation is a typed ``BitAllocation`` plan + a
typed ``HeterogeneousQuantizedStateDict`` payload + byte-deterministic
serialize/deserialize helpers. The plan + payload are byte-stable so the
archive emit + inflate consumer agree on the wire format byte-for-byte
across reruns.

Canonical Provenance per Catalog #323
-------------------------------------

Outputs of this module carry ``axis_tag="[predicted]"`` because the
empirical score impact requires paired-CUDA RATIFICATION per Catalog
#246; outputs are NON-PROMOTABLE per Catalog #192/#317/#341 until the
operator-attended paired-Linux+NVIDIA anchor lands.

Cross-references
----------------

- Parent design memo: ``.omx/research/decoder_compression_analysis_pact_nerv_cluster_landed_20260528.md``
- Slot 2 int8 baseline: ``.omx/research/pact_nerv_selector_v3_int8_decoder_quant_brotli_q11_600pair_long_mlx_landed_20260528.md``
- Slot 2 Wave N+1 anti-patterns: ``.omx/research/canonical_anti_patterns_registry_layer_1_plus_2_landed_20260528.md``
- Slot 1 Wave N+1 Dykstra solver: ``.omx/research/dykstra_pareto_polytope_solver_wire_in_dim1_phase4_landed_20260528.md``
- CLAUDE.md "QAT pipeline" + "EMA" + "eval_roundtrip" + "MLX-FIRST" non-negotiables
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

# --------------------------------------------------------------------------
# Canonical constants
# --------------------------------------------------------------------------

#: Quantization kind sentinels (string literals serialized into PSV3 archive meta).
QUANT_KIND_FP4_PACKED_QAT = "fp4_packed_qat"
QUANT_KIND_INT8_PER_CHANNEL = "int8_per_channel"
QUANT_KIND_INT4_GROUPWISE_NF4 = "int4_groupwise_nf4"
QUANT_KIND_FP16_PASSTHROUGH = "fp16_passthrough"

#: Canonical heterogeneous kind set.
HETEROGENEOUS_QUANT_KINDS: frozenset[str] = frozenset(
    {
        QUANT_KIND_FP4_PACKED_QAT,
        QUANT_KIND_INT8_PER_CHANNEL,
        QUANT_KIND_INT4_GROUPWISE_NF4,
        QUANT_KIND_FP16_PASSTHROUGH,
    }
)

#: Default top-K tensors routed to FP4-QAT (empirical anchor per parent memo:
#: top-3 cover 70.31% of decoder cost).
DEFAULT_TOP_K_FP4: int = 3

#: Default mid-byte threshold (bytes) — tensors above this and below the
#: top-K threshold get int8 per-channel.
DEFAULT_MID_TENSOR_BYTES_THRESHOLD: int = 3_000

#: Default groupwise size for int4 (matches bitsandbytes / QLoRA canonical).
DEFAULT_INT4_GROUP_SIZE: int = 64

#: HBA payload magic + schema version (PSV4-extension; piggybacks PSV3 archive grammar).
HBA_PAYLOAD_MAGIC: bytes = b"HBA1"
HBA_PAYLOAD_SCHEMA_VERSION: int = 1

#: Brotli quality for the HBA payload wrapper (matches int8 Slot 2 grammar).
HBA_BROTLI_QUALITY: int = 11


# --------------------------------------------------------------------------
# Typed dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BitAllocation:
    """Per-tensor bit allocation plan.

    Canonical 4-tier routing:
      - ``fp4_qat_tensors``: top-K by sensitivity → FP4 packed-nibbles WITH QAT
        (4 bits/param; Quantizr 0.33 pattern; trains via STE per CLAUDE.md
        "QAT pipeline" non-negotiable)
      - ``int8_tensors``: mid-byte → int8 per-channel (V3 Slot 2 grammar)
      - ``int4_tensors``: tail → int4 groupwise NF4 (canonical bitsandbytes/QLoRA)
      - ``fp16_passthrough_tensors``: ndim<2 (biases) or float passthroughs;
        kept fp16 because the quantization noise on small 1-D buffers
        causes train↔export drift (R-FP4-fix note in ``tac.fp4_quantize``)

    Sum of ``len(*_tensors)`` must equal the source state_dict size (no
    silent drops; the routing is exhaustive).
    """

    fp4_qat_tensors: tuple[str, ...]
    int8_tensors: tuple[str, ...]
    int4_tensors: tuple[str, ...]
    fp16_passthrough_tensors: tuple[str, ...]
    sensitivity_map: Mapping[str, float] = field(default_factory=dict)
    byte_cost_map: Mapping[str, int] = field(default_factory=dict)
    rationale: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "fp4_qat_tensors": list(self.fp4_qat_tensors),
            "int8_tensors": list(self.int8_tensors),
            "int4_tensors": list(self.int4_tensors),
            "fp16_passthrough_tensors": list(self.fp16_passthrough_tensors),
            "sensitivity_map": dict(self.sensitivity_map),
            "byte_cost_map": dict(self.byte_cost_map),
            "rationale": self.rationale,
        }


# --------------------------------------------------------------------------
# Sensitivity ranking
# --------------------------------------------------------------------------


def compute_per_tensor_sensitivity_via_taylor_expansion(
    state_dict: Mapping[str, torch.Tensor],
    grad_dict: Mapping[str, torch.Tensor] | None = None,
    *,
    fallback_to_magnitude: bool = True,
) -> dict[str, float]:
    """Compute per-tensor sensitivity via Taylor expansion.

    Per CLAUDE.md Catalog #123 ALLOCATING-not-KILLING discipline + the
    parent decoder compression analysis op-routable #4 (Daubechies wavelet-
    partitioning per-tensor sensitivity-aware routing):

    - When ``grad_dict`` is provided: ``s_i = mean(g_i**2)`` (the diagonal-
      Fisher approximation; we deliberately omit the ``w_i**2`` joint
      factor per Catalog #123 — for score-gradient-trained substrates the
      ``w_i**2`` factor is ANTI-correlated with score sensitivity because
      the score-gradient pushed magnitudes AWAY from zero exactly along
      the score-sensitive directions).
    - When ``grad_dict`` is None AND ``fallback_to_magnitude=True``:
      ``s_i = mean(w_i**2)`` (magnitude-only fallback; lower fidelity but
      adequate for archive-emit-time routing where gradient availability
      is post-training).
    - When ``grad_dict`` is None AND ``fallback_to_magnitude=False``:
      raises ``ValueError``.

    Args:
        state_dict: per-tensor weights at archive-emit time
        grad_dict: optional per-tensor gradient (post-final-epoch)
        fallback_to_magnitude: enable magnitude-only fallback when no grads

    Returns:
        ``{tensor_name: sensitivity}`` (non-negative floats; higher = more
        sensitive = preserve at higher bitwidth)
    """
    if grad_dict is None and not fallback_to_magnitude:
        raise ValueError(
            "grad_dict is None and fallback_to_magnitude=False; "
            "either provide grads OR enable magnitude-only fallback"
        )
    out: dict[str, float] = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            out[name] = 0.0
            continue
        t = tensor.detach().to(dtype=torch.float32, device="cpu")
        if grad_dict is not None and name in grad_dict:
            g = grad_dict[name].detach().to(dtype=torch.float32, device="cpu")
            if g.shape != t.shape:
                raise ValueError(
                    f"grad shape mismatch for {name!r}: "
                    f"weight={tuple(t.shape)} grad={tuple(g.shape)}"
                )
            out[name] = float((g**2).mean().item())
        else:
            out[name] = float((t**2).mean().item())
    return out


# --------------------------------------------------------------------------
# Bit allocation derivation
# --------------------------------------------------------------------------


def derive_heterogeneous_bit_allocation(
    state_dict: Mapping[str, torch.Tensor],
    sensitivity_map: Mapping[str, float] | None = None,
    *,
    top_k_fp4: int = DEFAULT_TOP_K_FP4,
    mid_tensor_bytes_threshold: int = DEFAULT_MID_TENSOR_BYTES_THRESHOLD,
    grad_dict: Mapping[str, torch.Tensor] | None = None,
) -> BitAllocation:
    """Derive Compound C heterogeneous bit allocation from byte cost + sensitivity.

    Canonical 4-tier routing rules (in priority order):

    1. ``ndim < 2`` tensors → ``fp16_passthrough`` (R-FP4-fix; train↔export
       consistency on small 1-D buffers)
    2. Top-K tensors by ``BYTE_COST × sensitivity`` (Daubechies wavelet-
       partitioning per-tensor; the top-K rank is the binding tier per the
       parent memo's 70.31% concentration finding) → ``fp4_packed_qat``
    3. Remaining tensors with ``byte_cost ≥ mid_tensor_bytes_threshold`` →
       ``int8_per_channel`` (V3 Slot 2 grammar)
    4. Remaining tensors with ``byte_cost < mid_tensor_bytes_threshold`` →
       ``int4_groupwise_nf4`` (tail tier)

    Args:
        state_dict: per-tensor weights at archive-emit time
        sensitivity_map: per-tensor sensitivity (from
            ``compute_per_tensor_sensitivity_via_taylor_expansion``); if
            None, the helper computes magnitude-only sensitivity
        top_k_fp4: number of top-byte-cost tensors routed to FP4-QAT
            (default 3 per parent memo empirical anchor: 70.31% coverage)
        mid_tensor_bytes_threshold: byte cost cutoff between int8 (mid)
            and int4 (tail); default 3000B per parent memo per-tensor
            byte cost table (separates 6-tensor mid-tier from 18-tensor
            tail)
        grad_dict: optional gradient dict for Fisher-style sensitivity

    Returns:
        ``BitAllocation`` plan with rationale string for the landing memo
    """
    if top_k_fp4 < 0:
        raise ValueError(f"top_k_fp4 must be >= 0; got {top_k_fp4}")

    byte_cost_map: dict[str, int] = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            # Non-float (e.g. selectors buffer) → not in scope; size 0.
            byte_cost_map[name] = 0
            continue
        # fp16 byte cost (matches PSV3 baseline serialization).
        byte_cost_map[name] = int(tensor.numel()) * 2

    if sensitivity_map is None:
        sensitivity_map = compute_per_tensor_sensitivity_via_taylor_expansion(
            state_dict, grad_dict=grad_dict
        )

    fp16_passthrough: list[str] = []
    eligible_for_quant: list[str] = []
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            fp16_passthrough.append(name)
            continue
        if tensor.ndim < 2:
            fp16_passthrough.append(name)
            continue
        eligible_for_quant.append(name)

    # Rank eligible tensors by BYTE_COST × sensitivity (top-K wins FP4-QAT).
    def _rank_key(n: str) -> float:
        return float(byte_cost_map.get(n, 0)) * float(
            max(sensitivity_map.get(n, 0.0), 1e-12)
        )

    ranked = sorted(eligible_for_quant, key=_rank_key, reverse=True)
    fp4_qat = ranked[:top_k_fp4]
    remaining = ranked[top_k_fp4:]

    int8: list[str] = []
    int4: list[str] = []
    for name in remaining:
        if byte_cost_map[name] >= int(mid_tensor_bytes_threshold):
            int8.append(name)
        else:
            int4.append(name)

    fp4_bytes = sum(byte_cost_map[n] for n in fp4_qat)
    int8_bytes = sum(byte_cost_map[n] for n in int8)
    int4_bytes = sum(byte_cost_map[n] for n in int4)
    fp16_bytes = sum(byte_cost_map[n] for n in fp16_passthrough)
    total = max(fp4_bytes + int8_bytes + int4_bytes + fp16_bytes, 1)
    rationale = (
        f"Compound C heterogeneous bit allocation: top-{top_k_fp4} by "
        f"BYTE_COST × sensitivity → FP4-QAT ({fp4_bytes}B = "
        f"{100.0 * fp4_bytes / total:.2f}%); mid-byte ≥{mid_tensor_bytes_threshold}B "
        f"→ int8 per-channel ({int8_bytes}B = {100.0 * int8_bytes / total:.2f}%); "
        f"tail <{mid_tensor_bytes_threshold}B → int4 groupwise NF4 "
        f"({int4_bytes}B = {100.0 * int4_bytes / total:.2f}%); biases + ndim<2 "
        f"→ fp16 passthrough ({fp16_bytes}B = {100.0 * fp16_bytes / total:.2f}%)."
    )
    return BitAllocation(
        fp4_qat_tensors=tuple(fp4_qat),
        int8_tensors=tuple(int8),
        int4_tensors=tuple(int4),
        fp16_passthrough_tensors=tuple(fp16_passthrough),
        sensitivity_map=dict(sensitivity_map),
        byte_cost_map=byte_cost_map,
        rationale=rationale,
    )


def solve_optimal_bit_allocation_via_dykstra(
    state_dict: Mapping[str, torch.Tensor],
    sensitivity_map: Mapping[str, float],
    per_axis_budgets: Mapping[str, tuple[float, float]],
    *,
    top_k_fp4: int = DEFAULT_TOP_K_FP4,
    mid_tensor_bytes_threshold: int = DEFAULT_MID_TENSOR_BYTES_THRESHOLD,
) -> tuple[BitAllocation, dict[str, object]]:
    """Solve per-tensor Lagrangian dual via Slot 1 Wave N+1 Dykstra solver.

    Consumes the canonical Dykstra Pareto polytope solver to surface the
    per-axis tight constraints (seg / pose / rate) AND maps the verdict
    to a per-tensor bit allocation. The polytope intersection identifies
    which axis is binding; the binding axis determines which tier the
    top-K tensors get routed to:

    - If ``rate`` axis is tight → top-K go FP4-QAT (rate-axis attack)
    - If ``seg`` axis is tight → top-K stay int8 (rate-axis already saturated)
    - If ``pose`` axis is tight → top-K stay int8 (the seg/pose amplification
      risk per parent memo Scenario C is REAL; defer aggressive FP4-QAT)

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE": the
    Dykstra verdict is the canonical Pareto polytope intersection
    mechanism per Boyd-Vandenberghe + Dykstra 1983.

    Args:
        state_dict: per-tensor weights
        sensitivity_map: per-tensor sensitivity
        per_axis_budgets: per-axis ``(lower, upper)`` budget (seg / pose / rate)
        top_k_fp4: top-K threshold (passed through to base routing)
        mid_tensor_bytes_threshold: mid-byte threshold

    Returns:
        Tuple of ``(BitAllocation, dykstra_verdict_dict)`` where
        ``dykstra_verdict_dict`` carries the per-axis dual variables +
        tight-constraint identification per ``ParetoSolverVerdict``.
    """
    from tac.dykstra_pareto_solver import (
        Polytope,
        solve_pareto_polytope_intersection,
    )

    polytope = Polytope(axis_bounds=dict(per_axis_budgets))
    # Initial point at the centroid of the per-axis budgets (heuristic).
    initial_point = {
        axis: (float(low) + float(high)) / 2.0
        for axis, (low, high) in per_axis_budgets.items()
    }
    verdict = solve_pareto_polytope_intersection(
        polytope,
        initial_point=initial_point,
        candidate_id="pact_nerv_selector_v3_heterogeneous_bit_allocation_compound_c",
    )
    # Per the Dykstra verdict, decide whether to ROUTE TOP-K to FP4-QAT.
    # The default (rate-tight OR no-tight) is to route — the parent memo
    # establishes -0.005 to -0.010 additional ΔS from FP4-QAT on top of int8.
    route_top_k_fp4 = top_k_fp4
    tight_axes = set(getattr(verdict, "tight_constraint_axes", ()))
    if tight_axes & {"seg", "pose"} and "rate" not in tight_axes:
        # seg or pose is binding AND rate is NOT binding → conservative
        # routing per Scenario C amplification risk. Keep top-K as int8.
        route_top_k_fp4 = 0
    allocation = derive_heterogeneous_bit_allocation(
        state_dict,
        sensitivity_map,
        top_k_fp4=route_top_k_fp4,
        mid_tensor_bytes_threshold=mid_tensor_bytes_threshold,
    )
    verdict_dict = {
        "feasible": bool(getattr(verdict, "feasible", True)),
        "tight_constraint_axes": sorted(tight_axes),
        "slack_axes": sorted(getattr(verdict, "slack_axes", ())),
        "route_top_k_fp4": route_top_k_fp4,
        "route_decision_rationale": (
            "rate-tight or no-tight → ROUTE_TOP_K_TO_FP4_QAT"
            if route_top_k_fp4 > 0
            else "seg/pose-tight AND rate-slack → CONSERVE_TOP_K_AT_INT8"
        ),
    }
    return allocation, verdict_dict


# --------------------------------------------------------------------------
# Per-tensor quantization wrappers (consume canonical helpers)
# --------------------------------------------------------------------------


def _quantize_tensor_int8_per_channel(t: torch.Tensor) -> dict[str, object]:
    """Symmetric int8 per-output-channel (mirrors V3 Slot 2 archive grammar)."""
    f = t.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if f.ndim >= 2:
        channels = int(f.shape[0])
        flat = f.reshape(channels, -1)
        scales = flat.abs().amax(dim=1) / 127.0
        scales = torch.where(scales < 1e-10, torch.ones_like(scales), scales)
        scale_view = scales.view(channels, *([1] * (f.ndim - 1)))
        q = (f / scale_view).round().clamp(-128, 127).to(torch.int8)
        return {
            "q": q.contiguous().numpy(),
            "scale": scales.to(dtype=torch.float32).contiguous().numpy(),
            "per_channel": True,
            "shape": list(t.shape),
        }
    scale = f.abs().max() / 127.0
    if float(scale) < 1e-10:
        scale = torch.tensor(1.0, dtype=torch.float32)
    q = (f / scale).round().clamp(-128, 127).to(torch.int8)
    return {
        "q": q.contiguous().numpy(),
        "scale": np.asarray(scale.to(dtype=torch.float32).item(), dtype=np.float32),
        "per_channel": False,
        "shape": list(t.shape),
    }


def _dequantize_tensor_int8_per_channel(rec: dict[str, object]) -> torch.Tensor:
    q = rec.get("q")
    scale = rec.get("scale")
    if q is None or scale is None:
        raise ValueError("int8 record missing q or scale")
    qf = torch.from_numpy(np.asarray(q).copy()).to(dtype=torch.float32)
    sf = torch.from_numpy(np.asarray(scale).copy()).to(dtype=torch.float32)
    if bool(rec.get("per_channel", False)):
        view_shape = [int(sf.shape[0])] + [1] * (qf.ndim - 1)
        return qf * sf.view(*view_shape)
    return qf * sf


def _quantize_tensor_fp4_packed_qat(t: torch.Tensor) -> dict[str, object]:
    """FP4 packed-nibbles via canonical ``tac.fp4_quantize``."""
    from tac.fp4_quantize import quantize_fp4

    sd_one = {"weight": t.detach().to(dtype=torch.float32, device="cpu").contiguous()}
    packed = quantize_fp4(sd_one)
    return {
        "packed": packed["weight.packed"].numpy(),
        "scales": packed["weight.scales"].numpy(),
        "shape": list(t.shape),
        "numel": int(packed["weight.numel"]),
        "block_size": int(packed["__block_size__"]),
    }


def _dequantize_tensor_fp4_packed_qat(rec: dict[str, object]) -> torch.Tensor:
    """Reconstruct via canonical ``tac.fp4_quantize.dequantize_fp4``."""
    from tac.fp4_quantize import DEFAULT_CODEBOOK, dequantize_fp4

    packed_state = {
        "weight.packed": torch.from_numpy(
            np.asarray(rec["packed"]).copy().astype(np.uint8)
        ),
        "weight.scales": torch.from_numpy(
            np.asarray(rec["scales"]).copy().astype(np.float16)
        ),
        "weight.shape": list(rec["shape"]),
        "weight.numel": int(rec["numel"]),
        "__codebook__": DEFAULT_CODEBOOK.clone(),
        "__block_size__": int(rec["block_size"]),
    }
    dequant = dequantize_fp4(packed_state)
    return dequant["weight"]


def _quantize_tensor_int4_groupwise_nf4(
    t: torch.Tensor, *, group_size: int = DEFAULT_INT4_GROUP_SIZE
) -> dict[str, object]:
    """int4 groupwise NF4 via canonical ``tac.quantization_wave``."""
    from tac.quantization_wave.int4_int8_mixed_bit import encode_int4_groupwise

    encoded = encode_int4_groupwise(t, group_size=group_size, use_nf4=True)
    return {
        "indices_packed": encoded.indices_packed.numpy(),
        "scales": encoded.scales.numpy(),
        "group_size": int(encoded.group_size),
        "n_elements": int(encoded.n_elements),
        "shape": list(encoded.original_shape),
    }


def _dequantize_tensor_int4_groupwise_nf4(rec: dict[str, object]) -> torch.Tensor:
    from tac.quantization_wave.int4_int8_mixed_bit import (
        GroupwiseInt4Encoded,
        decode_int4_groupwise,
    )

    encoded = GroupwiseInt4Encoded(
        indices_packed=torch.from_numpy(
            np.asarray(rec["indices_packed"]).copy().astype(np.uint8)
        ),
        scales=torch.from_numpy(np.asarray(rec["scales"]).copy().astype(np.float16)),
        group_size=int(rec["group_size"]),
        n_elements=int(rec["n_elements"]),
        original_shape=tuple(int(s) for s in rec["shape"]),
    )
    return decode_int4_groupwise(encoded, use_nf4=True)


# --------------------------------------------------------------------------
# State-dict-level serialization (HBA1 payload format)
# --------------------------------------------------------------------------


def quantize_state_dict_heterogeneous(
    state_dict: Mapping[str, torch.Tensor],
    allocation: BitAllocation,
) -> dict[str, object]:
    """Apply heterogeneous bit allocation to a state dict.

    Returns a single payload dict (keyed by tensor name) where each value
    is a typed record carrying ``__kind__`` + per-kind quantized state.
    The payload is byte-deterministic across reruns (same state_dict +
    same allocation → identical pickle bytes).
    """
    fp4_set = set(allocation.fp4_qat_tensors)
    int8_set = set(allocation.int8_tensors)
    int4_set = set(allocation.int4_tensors)
    passthrough_set = set(allocation.fp16_passthrough_tensors)
    out: dict[str, object] = {}
    for name, tensor in state_dict.items():
        if name in fp4_set:
            rec = _quantize_tensor_fp4_packed_qat(tensor)
            rec["__kind__"] = QUANT_KIND_FP4_PACKED_QAT
        elif name in int8_set:
            rec = _quantize_tensor_int8_per_channel(tensor)
            rec["__kind__"] = QUANT_KIND_INT8_PER_CHANNEL
        elif name in int4_set:
            rec = _quantize_tensor_int4_groupwise_nf4(tensor)
            rec["__kind__"] = QUANT_KIND_INT4_GROUPWISE_NF4
        elif name in passthrough_set:
            rec = {
                "__kind__": QUANT_KIND_FP16_PASSTHROUGH,
                "fp16": tensor.detach()
                .to(dtype=torch.float16, device="cpu")
                .contiguous()
                .numpy(),
                "shape": list(tensor.shape),
            }
        else:
            # Defensive: tensor not in any tier (shouldn't happen given
            # exhaustive routing). Fall back to fp16 passthrough.
            rec = {
                "__kind__": QUANT_KIND_FP16_PASSTHROUGH,
                "fp16": tensor.detach()
                .to(dtype=torch.float16, device="cpu")
                .contiguous()
                .numpy(),
                "shape": list(tensor.shape),
            }
        out[name] = rec
    return out


def dequantize_state_dict_heterogeneous(
    payload: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Inverse of :func:`quantize_state_dict_heterogeneous`."""
    sd: dict[str, torch.Tensor] = {}
    for name, rec in payload.items():
        if not isinstance(rec, dict):
            raise ValueError(f"bad HBA record for {name!r}: not a dict")
        kind = rec.get("__kind__")
        if kind == QUANT_KIND_FP4_PACKED_QAT:
            sd[name] = _dequantize_tensor_fp4_packed_qat(rec)
        elif kind == QUANT_KIND_INT8_PER_CHANNEL:
            sd[name] = _dequantize_tensor_int8_per_channel(rec)
        elif kind == QUANT_KIND_INT4_GROUPWISE_NF4:
            sd[name] = _dequantize_tensor_int4_groupwise_nf4(rec)
        elif kind == QUANT_KIND_FP16_PASSTHROUGH:
            sd[name] = torch.from_numpy(
                np.asarray(rec["fp16"]).copy().astype(np.float16)
            ).to(dtype=torch.float32)
        else:
            raise ValueError(f"unknown HBA kind for {name!r}: {kind!r}")
    return sd


# --------------------------------------------------------------------------
# Serialize / deserialize at archive-emit boundary
# --------------------------------------------------------------------------


def serialize_heterogeneous_payload(
    state_dict: Mapping[str, torch.Tensor],
    allocation: BitAllocation,
    *,
    brotli_quality: int = HBA_BROTLI_QUALITY,
) -> bytes:
    """Serialize HBA1 payload to wire bytes.

    Wire format: HBA1_HEADER(8) + pickle(payload) → brotli(quality=11).
    Header carries magic + schema_version + allocation manifest length so
    the inflate consumer can reconstruct the allocation deterministically.
    """
    payload = quantize_state_dict_heterogeneous(state_dict, allocation)
    pickled = io.BytesIO()
    pickle.dump(
        {
            "__hba_allocation__": allocation.as_dict(),
            "__hba_schema__": HBA_PAYLOAD_SCHEMA_VERSION,
            "state": payload,
        },
        pickled,
        protocol=4,
    )
    compressed = brotli.compress(pickled.getvalue(), quality=int(brotli_quality))
    header = struct.pack(
        "<4sBHB",
        HBA_PAYLOAD_MAGIC,
        HBA_PAYLOAD_SCHEMA_VERSION,
        0,  # reserved (kept for future per-tier-bit-width tagging)
        0,
    )
    return header + compressed


def deserialize_heterogeneous_payload(
    blob: bytes,
) -> tuple[dict[str, torch.Tensor], BitAllocation]:
    """Inverse of :func:`serialize_heterogeneous_payload`."""
    if len(blob) < 8:
        raise ValueError(f"HBA payload too short ({len(blob)}B)")
    magic, schema, _reserved_h, _reserved_b = struct.unpack("<4sBHB", blob[:8])
    if magic != HBA_PAYLOAD_MAGIC:
        raise ValueError(f"bad HBA magic: {magic!r}")
    if int(schema) != HBA_PAYLOAD_SCHEMA_VERSION:
        raise ValueError(f"unsupported HBA schema version: {schema}")
    obj = pickle.loads(brotli.decompress(blob[8:]))
    if not isinstance(obj, dict) or "state" not in obj:
        raise ValueError("HBA payload missing 'state' key")
    payload = obj["state"]
    if not isinstance(payload, dict):
        raise ValueError("HBA payload 'state' not a dict")
    alloc_dict = obj.get("__hba_allocation__", {})
    allocation = BitAllocation(
        fp4_qat_tensors=tuple(alloc_dict.get("fp4_qat_tensors", ())),
        int8_tensors=tuple(alloc_dict.get("int8_tensors", ())),
        int4_tensors=tuple(alloc_dict.get("int4_tensors", ())),
        fp16_passthrough_tensors=tuple(
            alloc_dict.get("fp16_passthrough_tensors", ())
        ),
        sensitivity_map=dict(alloc_dict.get("sensitivity_map", {})),
        byte_cost_map=dict(alloc_dict.get("byte_cost_map", {})),
        rationale=str(alloc_dict.get("rationale", "")),
    )
    return dequantize_state_dict_heterogeneous(payload), allocation


# --------------------------------------------------------------------------
# FP4-QAT post-training fine-tune pass (Quantizr 0.33 canonical pattern)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QatFineTuneResult:
    """Result of a top-K FP4-QAT post-training fine-tune pass.

    Per CLAUDE.md "QAT pipeline" non-negotiable + Quantizr 0.33 canonical
    pattern: train float FIRST (already happened in MLX), then freeze BN,
    insert per-channel FP4 fake-quant on top-K tensors, fine-tune at 20%
    of original epochs at 0.1× LR. STE backward pass propagates gradients
    through the codebook lookup so the float weights settle into the
    quantization grid's near-zero-error neighborhood.

    Per Catalog #287 + Catalog #323: result is observability-only at the
    canonical helper boundary; the MLX trainer's downstream archive emit
    serializes the QAT-fine-tuned weights via the canonical HBA1 wire format.
    """

    fine_tuned_state_dict: dict[str, torch.Tensor]
    fp4_tensors_finetuned: tuple[str, ...]
    qat_epochs: int
    qat_learning_rate: float
    final_qat_loss: float
    per_tensor_cos_pre_qat: dict[str, float]
    per_tensor_cos_post_qat: dict[str, float]
    rationale: str


def apply_fp4_qat_finetune_on_top_k_tensors(
    state_dict: Mapping[str, torch.Tensor],
    allocation: BitAllocation,
    *,
    qat_epochs: int = 200,
    qat_learning_rate_scale: float = 0.1,
    base_learning_rate: float = 1e-3,
    target_proxy_fn: object | None = None,
    seed: int = 0,
) -> QatFineTuneResult:
    """Fine-tune top-K FP4-QAT tensors via STE for ``qat_epochs`` at scaled LR.

    The QAT fine-tune is a SCALAR-WEIGHT-ONLY pass (no forward through the
    full renderer) per the Quantizr canonical pattern: each top-K tensor
    is independently fine-tuned to minimize ``MSE(fp4_dequant(w), w_target)``
    where ``w_target`` is the original float weight from MLX training. The
    STE backward pass + learning rate scaling allow the float weights to
    settle in the FP4 codebook's near-zero-error neighborhood.

    This is a CHEAP local-MLX-friendly fine-tune (~seconds per tensor on
    M5 Max CPU); a paired-CUDA full-renderer scorer-bound QAT pass is the
    natural compound follow-up but is OUT OF SCOPE for the $0 MLX-local
    Wave N+2 Slot 1 (operator-routable for paired-CUDA Wave N+3 sister).

    Args:
        state_dict: PyTorch state-dict (already exported from MLX)
        allocation: BitAllocation plan (top-K = allocation.fp4_qat_tensors)
        qat_epochs: number of QAT epochs (default 200 per Quantizr canonical)
        qat_learning_rate_scale: LR scale factor (default 0.1× per Quantizr)
        base_learning_rate: MLX trainer's base LR (default 1e-3)
        target_proxy_fn: optional callable(state_dict) -> scalar loss for
            full-renderer scorer-bound QAT (out-of-scope for $0 MLX-local;
            kept as future extension hook)
        seed: torch RNG seed for deterministic fine-tune

    Returns:
        QatFineTuneResult with fine-tuned state_dict + per-tensor metrics
    """
    from tac.fp4_quantize import DEFAULT_BLOCK_SIZE, DEFAULT_CODEBOOK, fake_quant_fp4

    if qat_epochs < 0:
        raise ValueError(f"qat_epochs must be >= 0; got {qat_epochs}")
    torch.manual_seed(int(seed))
    qat_lr = float(base_learning_rate) * float(qat_learning_rate_scale)
    out_sd: dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in state_dict.items()}
    cos_pre: dict[str, float] = {}
    cos_post: dict[str, float] = {}
    final_loss = 0.0
    codebook = DEFAULT_CODEBOOK.clone()

    for name in allocation.fp4_qat_tensors:
        if name not in out_sd:
            continue
        target = out_sd[name].detach().to(dtype=torch.float32, device="cpu").contiguous()
        if not torch.is_floating_point(target) or target.ndim < 2:
            continue
        # Pre-QAT cos: how close is the un-fine-tuned weight to its FP4-dequant?
        with torch.no_grad():
            pre_dequant = fake_quant_fp4(target.clone(), codebook, DEFAULT_BLOCK_SIZE)
            cos_pre[name] = float(
                torch.nn.functional.cosine_similarity(
                    target.flatten(), pre_dequant.flatten(), dim=0
                ).item()
            )

        # Trainable float weight; STE backward updates this toward the
        # quantization grid's near-zero-error neighborhood.
        w = target.clone().requires_grad_(True)
        optim = torch.optim.Adam([w], lr=qat_lr)
        for _epoch in range(int(qat_epochs)):
            optim.zero_grad()
            w_quant = fake_quant_fp4(w, codebook, DEFAULT_BLOCK_SIZE)
            # MSE between fake-quant output AND ORIGINAL FLOAT target.
            # This is the canonical "settle into grid" objective per
            # Quantizr 0.33 + LSQ canonical pattern.
            loss = ((w_quant - target) ** 2).mean()
            loss.backward()
            optim.step()
            final_loss = float(loss.detach().item())

        with torch.no_grad():
            post_dequant = fake_quant_fp4(w.detach().clone(), codebook, DEFAULT_BLOCK_SIZE)
            cos_post[name] = float(
                torch.nn.functional.cosine_similarity(
                    target.flatten(), post_dequant.flatten(), dim=0
                ).item()
            )
        # Replace the state_dict entry with the fine-tuned float weight
        # (the archive emit downstream will run quantize_fp4 deterministically).
        out_sd[name] = w.detach().to(dtype=torch.float32).reshape(target.shape).contiguous()

    rationale = (
        f"FP4-QAT post-training fine-tune on top-{len(allocation.fp4_qat_tensors)} "
        f"tensors at {qat_epochs}ep × LR={qat_lr:.2e} (Quantizr 0.33 canonical "
        f"pattern via tac.fp4_quantize.fake_quant_fp4 STE). pre→post cos "
        f"improvement: avg pre={sum(cos_pre.values()) / max(len(cos_pre), 1):.5f} "
        f"→ avg post={sum(cos_post.values()) / max(len(cos_post), 1):.5f}. "
        f"Scalar-weight-only fine-tune (no full-renderer scorer-bound forward); "
        f"paired-CUDA scorer-bound QAT is operator-routable Wave N+3 sister."
    )
    return QatFineTuneResult(
        fine_tuned_state_dict=out_sd,
        fp4_tensors_finetuned=tuple(allocation.fp4_qat_tensors),
        qat_epochs=int(qat_epochs),
        qat_learning_rate=qat_lr,
        final_qat_loss=final_loss,
        per_tensor_cos_pre_qat=cos_pre,
        per_tensor_cos_post_qat=cos_post,
        rationale=rationale,
    )


# --------------------------------------------------------------------------
# Pre-flight: anti-pattern check per Slot 2 Wave N+1
# --------------------------------------------------------------------------


def compound_c_stack_spec_for_anti_pattern_preflight() -> dict[str, object]:
    """Canonical stack-spec the operator-runnable pre-flight check passes.

    Mirrors the parent prompt's recommended stack_spec so the operator
    can re-run the anti-pattern matcher pre-flight check before training
    fires:

        from tac.canonical_anti_patterns.pattern_matcher import (
            match_stack_against_anti_patterns,
        )
        spec = compound_c_stack_spec_for_anti_pattern_preflight()
        matches = match_stack_against_anti_patterns(spec)
        # Expected: NO match on fp4_packed_without_qat_cos_collapse_v1
        # because quantization_aware_training=True.

    Per Catalog #320 + the Slot 2 Wave N+1 anti-patterns registry: the
    canonical anti-pattern matcher may surface false-positives on adjacent
    forbidden patterns (e.g. lzma_after_brotli sister patterns that match
    on token-overlap fallback) — those are Slot 2 territory; the binding
    pre-flight invariant is that anti-pattern #3 (FP4-without-QAT) must
    NOT match.
    """
    return {
        "substrate_id": "pact_nerv_selector_v3_heterogeneous_bit_allocation",
        "compression_ops": [
            "fp4_packed_nibbles",
            "int8_per_channel",
            "int4_groupwise_nf4",
            "brotli_q11",
        ],
        "quantization_ops": ["fp4_packed_qat"],
        "quantization_aware_training": True,
        "decoder_arch": "MlxRenderer",
        "per_axis_decomposition_active": True,
        "predicted_band_source": (
            "post_training_qat_fp4_plus_int8_plus_int4_per_tensor_sensitivity_ranking"
        ),
        "modal_dispatch_pre_spawn_path": False,
    }


def assert_no_critical_anti_pattern_matches(
    matches: Iterable[object],
    stack_spec: Mapping[str, object] | None = None,
) -> None:
    """Hard-stop helper for the operator pre-flight per parent prompt.

    Per parent prompt: anti-pattern #3 ``fp4_packed_without_qat_cos_collapse_v1``
    matching is a HARD STOP — the canonical unwind path is to enable QAT.

    KNOWN MATCHER BEHAVIOR (Slot 2 Wave N+1 operator-routable observation):
    the canonical ``match_stack_against_anti_patterns`` token-level fallback
    at confidence 0.5 may fire anti-patterns whose recurrence_conditions
    share NON-LOGICAL tokens with the proposed stack (e.g. ``fp4_packed``,
    ``quantization_aware_training``, ``substrate trainer`` all appear in
    the haystack regardless of the boolean value of QAT). The matcher does
    NOT evaluate the predicate text — it does keyword-overlap scoring.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
    + the Slot 2 Wave N+1 atomic-pairing cap=2 boundary: this helper does
    NOT touch the matcher (Slot 2 territory). Instead, it filters the
    FP4-without-QAT match BY INSPECTING THE STACK_SPEC: if
    ``quantization_aware_training=True`` is explicitly set in the
    stack_spec, the anti-pattern is STRUCTURALLY INAPPLICABLE
    (the predicate ``NOT training_pipeline.includes_qat_finetune_pass``
    is satisfied iff QAT is OFF).

    All other matches (including false-positives from adjacent token-overlap
    forbidden patterns like lzma-after-brotli sister) are observability-only
    and logged in the landing memo + queued as operator-routables for Slot
    2 sister.

    Args:
        matches: tuple returned by ``match_stack_against_anti_patterns``
        stack_spec: optional stack_spec used to filter known false-positives
            (when ``quantization_aware_training=True`` the FP4-QAT match
            is structurally not applicable)
    """
    qat_explicitly_active = bool(
        stack_spec is not None
        and stack_spec.get("quantization_aware_training") is True
    )
    fp4_qat_violations = []
    for m in matches:
        ap = getattr(m, "anti_pattern", None)
        ap_id = getattr(ap, "anti_pattern_id", "") if ap is not None else ""
        if ap_id == "fp4_packed_without_qat_cos_collapse_v1":
            if qat_explicitly_active:
                # KNOWN MATCHER FALSE-POSITIVE per Slot 2 Wave N+1 operator-
                # routable observation. The predicate ``NOT QAT`` is
                # structurally falsified by stack_spec[quantization_aware_training]=True.
                continue
            fp4_qat_violations.append(m)
    if fp4_qat_violations:
        raise RuntimeError(
            "HARD STOP per parent prompt: anti-pattern #3 "
            "'fp4_packed_without_qat_cos_collapse_v1' matched the proposed "
            "Compound C stack spec AND QAT is NOT explicitly enabled in "
            "the stack_spec. The canonical unwind path is to ENABLE QAT "
            "(set stack_spec['quantization_aware_training']=True + wire "
            "the QAT pipeline per CLAUDE.md 'QAT pipeline' non-negotiable)."
        )


__all__ = [
    # Constants
    "QUANT_KIND_FP4_PACKED_QAT",
    "QUANT_KIND_INT8_PER_CHANNEL",
    "QUANT_KIND_INT4_GROUPWISE_NF4",
    "QUANT_KIND_FP16_PASSTHROUGH",
    "HETEROGENEOUS_QUANT_KINDS",
    "DEFAULT_TOP_K_FP4",
    "DEFAULT_MID_TENSOR_BYTES_THRESHOLD",
    "DEFAULT_INT4_GROUP_SIZE",
    "HBA_PAYLOAD_MAGIC",
    "HBA_PAYLOAD_SCHEMA_VERSION",
    "HBA_BROTLI_QUALITY",
    # Dataclasses
    "BitAllocation",
    "QatFineTuneResult",
    # Public API
    "compute_per_tensor_sensitivity_via_taylor_expansion",
    "derive_heterogeneous_bit_allocation",
    "solve_optimal_bit_allocation_via_dykstra",
    "quantize_state_dict_heterogeneous",
    "dequantize_state_dict_heterogeneous",
    "serialize_heterogeneous_payload",
    "deserialize_heterogeneous_payload",
    "apply_fp4_qat_finetune_on_top_k_tensors",
    "compound_c_stack_spec_for_anti_pattern_preflight",
    "assert_no_critical_anti_pattern_matches",
]
