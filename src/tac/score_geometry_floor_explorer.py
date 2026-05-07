"""What-if Shannon-floor explorer for weight-reducing techniques.

The empirical Shannon floor (``tac.score_geometry_shannon_floor``) is the
hard lower bound on archive bytes given:
  - the schema (tensor shapes)
  - the quantization scheme (n_quant)
  - the per-tensor empirical entropy

To go BELOW that floor, you need to change one of those inputs. This
module quantifies the theoretical savings each technique enables,
giving the operator a planning signal for which technique class to
attack first.

## Techniques modeled

1. **Lossy quantization** (lower n_quant): reduces uniform bits per
   element; usually reduces empirical bits proportionally if the
   shape of the distribution stays similar. Hard cap: distortion
   floor (must be empirically validated via parity check).

2. **Sparsity / ablation** (mask alpha fraction of elements to zero):
   reduces effective non-zero count. Encoder cost includes a
   sparsity-pattern overhead (1 bit per masked position naively;
   far less with run-length or hash-based encoding).

3. **Per-tensor entropy reduction** (training-side): user supplies
   target entropy per tensor (e.g., from QAT with sparsity loss
   prediction). Floor recomputed at lower entropy.

4. **Architecture shrink** (parameter-count reduction): user supplies
   element-count multiplier (e.g., 0.4 for going from 228K elements
   to 90K like Quantizr).

5. **Mixed precision** (per-tensor n_quant): user supplies a dict
   of tensor to n_quant. Floor accumulates per-tensor contributions.

6. **Water-filling** (uneven bit allocation by Hessian sensitivity):
   user supplies per-tensor sensitivity weights; the optimizer
   distributes a TOTAL bit budget proportional to weight times log2 of
   element count.

The `rank_technique_results()` helper takes independently computed
what-if results and returns a deterministic byte-savings ranking.

Pure CPU + math; no torch dependencies.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from tac.score_geometry_shannon_floor import (
    ShannonFloorReport,
    compute_shannon_floor,
)


@dataclass(frozen=True)
class TechniqueResult:
    """One what-if scenario's predicted floor + savings vs baseline."""

    name: str
    description: str
    bytes_floor: int
    bytes_savings_vs_baseline: int
    score_at_floor_zero_distortion: float
    distortion_risk: str
    """Heuristic risk tag: 'lossless' | 'lossy_low' | 'lossy_high' |
    'training_side' | 'architectural'."""
    notes: list[str] = field(default_factory=list)


def explore_lossy_quantization(
    *,
    schema: Iterable[tuple[str, tuple[int, ...]]],
    baseline_n_quant: int,
    baseline_per_tensor_bits: dict[str, float],
    target_n_quant: int,
    archive_overhead_bytes: int,
) -> TechniqueResult:
    """Reduce n_quant globally. Empirical bits scale with log2(n_quant)
    ratio (rough but conservative; actual entropy may compress further
    if the lower-bit distribution gets more concentrated)."""
    if target_n_quant >= baseline_n_quant:
        raise ValueError(
            f"target_n_quant {target_n_quant} >= baseline {baseline_n_quant}; "
            "use baseline path for no-op"
        )
    ratio = math.log2(target_n_quant) / math.log2(baseline_n_quant)
    new_per_tensor_bits = {
        name: bits * ratio for name, bits in baseline_per_tensor_bits.items()
    }
    report = compute_shannon_floor(
        schema=schema,
        n_quant=target_n_quant,
        per_tensor_empirical_bits=new_per_tensor_bits,
        archive_overhead_bytes=archive_overhead_bytes,
        schema_label=f"lossy_n_quant_{target_n_quant}",
    )
    bytes_floor = report.total_bytes_empirical_floor or report.total_bytes_uniform_floor
    risk = (
        "lossy_low" if target_n_quant >= 64
        else "lossy_high"
    )
    return TechniqueResult(
        name=f"lossy_int_n_quant={target_n_quant}",
        description=f"Replace n_quant=127 with n_quant={target_n_quant}",
        bytes_floor=bytes_floor,
        bytes_savings_vs_baseline=0,  # caller fills
        score_at_floor_zero_distortion=(
            report.score_at_empirical_floor_zero_distortion
            or report.score_at_uniform_floor_zero_distortion
        ),
        distortion_risk=risk,
        notes=[
            f"Empirical bits scale by log2({target_n_quant})/log2({baseline_n_quant}) = {ratio:.4f}",
            "ASSUMES distribution shape preserved under finer quantization. "
            "Actual savings may be larger if quantization concentrates symbols.",
            f"REQUIRES distortion-parity validation (basin-parity gate, see "
            f"apogee_int{target_n_quant if target_n_quant < 8 else 'N'} forensic).",
        ],
    )


def explore_sparsity(
    *,
    schema: Iterable[tuple[str, tuple[int, ...]]],
    baseline_n_quant: int,
    baseline_per_tensor_bits: dict[str, float],
    sparsity_fraction: float,
    archive_overhead_bytes: int,
    sparsity_overhead_bits_per_element: float = 0.04,
) -> TechniqueResult:
    """Mask `sparsity_fraction` of elements to zero. Effective entropy:
    (1-alpha) * original_entropy + alpha * 0 + sparsity_overhead.

    Default sparsity_overhead_bits_per_element=0.04 corresponds to
    ~2-3% mask overhead from run-length-encoded sparsity patterns
    (typical for 50% sparse tensors).
    """
    if not 0 < sparsity_fraction < 1:
        raise ValueError("sparsity_fraction must be in (0, 1)")
    new_per_tensor_bits: dict[str, float] = {}
    for name, bits in baseline_per_tensor_bits.items():
        # Non-zero elements still encoded at original entropy
        new_bits = (
            (1 - sparsity_fraction) * bits
            + sparsity_overhead_bits_per_element
        )
        new_per_tensor_bits[name] = new_bits
    report = compute_shannon_floor(
        schema=schema,
        n_quant=baseline_n_quant,
        per_tensor_empirical_bits=new_per_tensor_bits,
        archive_overhead_bytes=archive_overhead_bytes,
        schema_label=f"sparse_alpha={sparsity_fraction:.2f}",
    )
    bytes_floor = report.total_bytes_empirical_floor or report.total_bytes_uniform_floor
    return TechniqueResult(
        name=f"sparsity_alpha={sparsity_fraction:.2f}",
        description=f"{sparsity_fraction * 100:.0f}% of weights ablated to zero",
        bytes_floor=bytes_floor,
        bytes_savings_vs_baseline=0,
        score_at_floor_zero_distortion=(
            report.score_at_empirical_floor_zero_distortion
            or report.score_at_uniform_floor_zero_distortion
        ),
        distortion_risk="training_side",
        notes=[
            f"Sparsity overhead {sparsity_overhead_bits_per_element} bits/elem applied uniformly.",
            "Requires QAT/IMP retraining; cannot apply to a finished checkpoint.",
            "Highest leverage on tensors with high entropy_ratio (less skewed).",
        ],
    )


def explore_architecture_shrink(
    *,
    schema: Iterable[tuple[str, tuple[int, ...]]],
    baseline_n_quant: int,
    baseline_per_tensor_bits: dict[str, float],
    element_multiplier: float,
    archive_overhead_bytes: int,
) -> TechniqueResult:
    """Scale every tensor's element count by `element_multiplier` (e.g.,
    0.4 for 88K-param target from 228K-element baseline). Per-element
    entropy preserved (assumption: smaller arch trains to similar
    distribution shape)."""
    if element_multiplier <= 0 or element_multiplier >= 1:
        raise ValueError("element_multiplier must be in (0, 1)")
    new_schema: list[tuple[str, tuple[int, ...]]] = []
    new_per_tensor_bits: dict[str, float] = {}
    for name, shape in schema:
        n = 1
        for dim in shape:
            n *= int(dim)
        target_n = max(1, int(n * element_multiplier))
        # Preserve shape's "leading dim" structure by scaling the FIRST dim
        # (a heuristic; real architectures might shrink mid-dims). For
        # floor estimation only the element COUNT matters.
        new_shape = (target_n,)
        new_schema.append((name, new_shape))
        new_per_tensor_bits[name] = baseline_per_tensor_bits.get(name, 0.0)
    report = compute_shannon_floor(
        schema=new_schema,
        n_quant=baseline_n_quant,
        per_tensor_empirical_bits=new_per_tensor_bits,
        archive_overhead_bytes=archive_overhead_bytes,
        schema_label=f"arch_shrink_x{element_multiplier:.2f}",
    )
    bytes_floor = report.total_bytes_empirical_floor or report.total_bytes_uniform_floor
    return TechniqueResult(
        name=f"arch_shrink_x{element_multiplier:.2f}",
        description=f"Smaller renderer x{element_multiplier:.2f} elements (e.g., Quantizr 88K)",
        bytes_floor=bytes_floor,
        bytes_savings_vs_baseline=0,
        score_at_floor_zero_distortion=(
            report.score_at_empirical_floor_zero_distortion
            or report.score_at_uniform_floor_zero_distortion
        ),
        distortion_risk="architectural",
        notes=[
            f"Element count x {element_multiplier:.4f}; per-element entropy preserved.",
            "Requires full retrain; distortion likely higher than baseline.",
            "Reference: Quantizr 88K-param 0.33 archive (88K elements vs PR101's ~229K).",
        ],
    )


def explore_water_filling(
    *,
    schema: Iterable[tuple[str, tuple[int, ...]]],
    baseline_n_quant: int,
    baseline_per_tensor_bits: dict[str, float],
    sensitivities: dict[str, float],
    total_bit_budget: int,
    archive_overhead_bytes: int,
) -> TechniqueResult:
    """Allocate `total_bit_budget` across tensors proportional to
    sensitivity times element_count, capped at baseline empirical entropy.
    This is a deterministic planning proxy for byte allocation; exact
    promotion still requires score-coupled distortion validation."""
    if total_bit_budget <= 0:
        raise ValueError("total_bit_budget must be positive")
    schema_list = list(schema)
    if any(value < 0 for value in sensitivities.values()):
        raise ValueError("sensitivities must be nonnegative")
    # Compute weights (sensitivity times element_count)
    weights: dict[str, float] = {}
    elements: dict[str, int] = {}
    for name, shape in schema_list:
        n = 1
        for dim in shape:
            n *= int(dim)
        elements[name] = n
        sens = sensitivities.get(name, 1.0)
        weights[name] = sens * n
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        raise ValueError("all sensitivities zero")
    # Initial allocation: budget times weight / sum_weight, in bits per tensor.
    new_per_tensor_bits: dict[str, float] = {}
    overflow = 0.0
    for name, n in elements.items():
        share_bits = total_bit_budget * (weights[name] / weight_sum)
        per_elem_bits = share_bits / n if n > 0 else 0
        baseline_bits = baseline_per_tensor_bits.get(name, math.log2(baseline_n_quant))
        # Cap at baseline empirical entropy (can't allocate more bits/elem
        # than the actual entropy needs)
        if per_elem_bits > baseline_bits:
            overflow += (per_elem_bits - baseline_bits) * n
            per_elem_bits = baseline_bits
        new_per_tensor_bits[name] = per_elem_bits

    # Reallocate overflow to under-allocated tensors (one pass)
    if overflow > 0:
        under = {
            name: baseline_per_tensor_bits.get(name, math.log2(baseline_n_quant))
            - new_per_tensor_bits[name]
            for name in elements
        }
        under_total_bits = sum(u * elements[name] for name, u in under.items() if u > 0)
        if under_total_bits > 0:
            extra_ratio = min(1.0, overflow / under_total_bits)
            for name in elements:
                if under[name] > 0:
                    new_per_tensor_bits[name] += under[name] * extra_ratio

    report = compute_shannon_floor(
        schema=schema_list,
        n_quant=baseline_n_quant,
        per_tensor_empirical_bits=new_per_tensor_bits,
        archive_overhead_bytes=archive_overhead_bytes,
        schema_label="water_filling",
    )
    bytes_floor = report.total_bytes_empirical_floor or report.total_bytes_uniform_floor
    return TechniqueResult(
        name=f"water_filling_budget={total_bit_budget}",
        description=f"Water-fill {total_bit_budget} bits across tensors by sensitivity times n",
        bytes_floor=bytes_floor,
        bytes_savings_vs_baseline=0,
        score_at_floor_zero_distortion=(
            report.score_at_empirical_floor_zero_distortion
            or report.score_at_uniform_floor_zero_distortion
        ),
        distortion_risk="lossy_low",
        notes=[
            "Deterministic water-fill proxy: bits per tensor proportional to sensitivity times element_count.",
            "Capped at baseline empirical entropy (no over-allocation).",
            "Requires distortion-parity validation per Lane Omega-W playbook.",
        ],
    )


def baseline_floor_summary(
    *,
    schema: Iterable[tuple[str, tuple[int, ...]]],
    baseline_n_quant: int,
    baseline_per_tensor_bits: dict[str, float],
    archive_overhead_bytes: int,
) -> ShannonFloorReport:
    """Compute the baseline Shannon floor (the bar to beat)."""
    return compute_shannon_floor(
        schema=schema,
        n_quant=baseline_n_quant,
        per_tensor_empirical_bits=baseline_per_tensor_bits,
        archive_overhead_bytes=archive_overhead_bytes,
        schema_label="baseline_empirical",
    )


def rank_technique_results(
    *,
    baseline_floor_bytes: int,
    results: Iterable[TechniqueResult],
) -> list[TechniqueResult]:
    """Return deterministic byte-savings ranking for closed-form floor scenarios."""
    if baseline_floor_bytes <= 0:
        raise ValueError("baseline_floor_bytes must be positive")
    ranked = [
        replace(
            result,
            bytes_savings_vs_baseline=baseline_floor_bytes - result.bytes_floor,
        )
        for result in results
    ]
    return sorted(
        ranked,
        key=lambda result: (
            -result.bytes_savings_vs_baseline,
            result.bytes_floor,
            result.name,
        ),
    )


__all__ = [
    "TechniqueResult",
    "baseline_floor_summary",
    "explore_architecture_shrink",
    "explore_lossy_quantization",
    "explore_sparsity",
    "explore_water_filling",
    "rank_technique_results",
]
