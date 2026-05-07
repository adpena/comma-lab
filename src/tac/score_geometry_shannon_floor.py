"""Closed-form Shannon-floor lower bound for the contest score.

Computes the analytical information-theoretic lower bound on archive
size given:

  - The renderer schema (PR101's 28-tensor FIXED_STATE_SCHEMA, or
    a caller-supplied (name, shape) iterable)
  - The chosen quantization scheme (n_quant levels per element)
  - The empirical per-tensor symbol entropy (or a worst-case uniform
    upper bound)

This is a HARD LOWER BOUND on archive bytes — no encoder, no clever
brotli/lzma/ANS variant, can produce a smaller archive without
compromising decode-correctness OR retraining the renderer to produce
lower-entropy weights.

Conversely, the score-geometry information_floor() (in
tac.score_geometry) gives the floor in **score points** at fixed
archive bytes; this module gives the floor in **archive bytes** given
a quantization budget, then tells you what score that converts to.

The two together give the operator: "you cannot beat score X without
either (a) reducing distortion to zero — impossible — or (b) training
a smaller-entropy renderer." That's the Shannon floor of THIS
architecture.

## Math

For a tensor of N elements quantized to n_quant levels with empirical
per-element entropy H bits, the minimum encoded bytes is:

    bytes_min = ceil(N * H / 8)

Sum over all tensors gives the total schema floor. With UNIFORM
distribution (worst case) H = log2(n_quant). With observed entropy
(best case) H = empirical entropy of the symbol stream.

## Cross-references

- ``tac.score_geometry.information_floor`` — score floor at fixed bytes
- ``tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA`` — concrete renderer
- ``feedback_pr101_lgwin13_q10_8byte_savings_20260507.md`` — empirical
  baseline (162,164 B) we are computing the floor for

CLAUDE.md compliance: pure CPU + math + numpy; no scorer load; no torch;
no contest score claims (only architectural information bounds).
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from tac.score_geometry import (
    CONTEST_REFERENCE_BYTES,
    RATE_COEFFICIENT,
    contest_score,
)


@dataclass(frozen=True)
class ShannonFloorComponent:
    """One tensor's contribution to the schema's Shannon floor."""

    name: str
    n_elements: int
    n_quant: int
    bits_per_element_uniform: float
    bits_per_element_empirical: float | None
    bytes_uniform_floor: int
    bytes_empirical_floor: int | None


@dataclass(frozen=True)
class ShannonFloorReport:
    """Total Shannon floor for a renderer schema + quantization scheme."""

    schema_label: str
    total_elements: int
    n_quant: int
    components: list[ShannonFloorComponent]
    total_bytes_uniform_floor: int
    total_bytes_empirical_floor: int | None
    score_at_uniform_floor_zero_distortion: float
    score_at_empirical_floor_zero_distortion: float | None
    notes: list[str]


def _uniform_bits_per_element(n_quant: int) -> float:
    """Uniform-distribution entropy in bits per element."""
    if n_quant < 1:
        raise ValueError(f"n_quant must be >= 1, got {n_quant}")
    if n_quant == 1:
        return 0.0
    return math.log2(n_quant)


def _shannon_entropy(symbol_counts: dict[int, int]) -> float:
    """Empirical entropy in bits per element from symbol counts."""
    total = sum(symbol_counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for n in symbol_counts.values():
        if n == 0:
            continue
        p = n / total
        h -= p * math.log2(p)
    return h


def compute_shannon_floor(
    *,
    schema: Iterable[tuple[str, tuple[int, ...]]],
    n_quant: int,
    schema_label: str = "<unnamed>",
    per_tensor_empirical_bits: dict[str, float] | None = None,
    archive_overhead_bytes: int = 0,
) -> ShannonFloorReport:
    """Compute the closed-form Shannon floor for a renderer schema.

    Parameters
    ----------
    schema : iterable of (name, shape) tuples
        E.g. ``tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA``.
        Each shape's product is the element count.
    n_quant : int
        Number of distinct quantization levels (e.g., 127 for PR101's
        N_QUANT, 16 for int4, 256 for uint8). Used for the uniform
        upper bound.
    per_tensor_empirical_bits : dict, optional
        Per-tensor empirical entropy in bits/element. If provided, the
        report includes the empirical-bound floor (always <= uniform
        bound). Tensors not present default to the uniform bound.
    archive_overhead_bytes : int, optional
        Fixed overhead (zip header, magic bytes, latents/sidecar tail
        size, etc.) added to the floor. For PR101 this is 16,094 bytes
        (latent_blob 15,387 + sidecar 607 + zip overhead 100).

    Returns
    -------
    ShannonFloorReport
        Per-tensor breakdown + total floors + score conversion.
    """
    if n_quant < 1:
        raise ValueError(f"n_quant must be >= 1, got {n_quant}")
    if archive_overhead_bytes < 0:
        raise ValueError("archive_overhead_bytes must be non-negative")

    bits_uniform_per_elem = _uniform_bits_per_element(n_quant)
    components: list[ShannonFloorComponent] = []
    total_elements = 0
    total_bits_uniform = 0.0
    total_bits_empirical: float | None = (
        0.0 if per_tensor_empirical_bits is not None else None
    )

    for name, shape in schema:
        n = 1
        for dim in shape:
            n *= int(dim)
        bits_emp_per_elem = (
            per_tensor_empirical_bits.get(name, bits_uniform_per_elem)
            if per_tensor_empirical_bits is not None
            else None
        )
        bytes_uniform = math.ceil(n * bits_uniform_per_elem / 8.0)
        bytes_emp = (
            math.ceil(n * bits_emp_per_elem / 8.0)
            if bits_emp_per_elem is not None
            else None
        )
        components.append(ShannonFloorComponent(
            name=name,
            n_elements=n,
            n_quant=n_quant,
            bits_per_element_uniform=bits_uniform_per_elem,
            bits_per_element_empirical=bits_emp_per_elem,
            bytes_uniform_floor=bytes_uniform,
            bytes_empirical_floor=bytes_emp,
        ))
        total_elements += n
        total_bits_uniform += n * bits_uniform_per_elem
        if total_bits_empirical is not None and bits_emp_per_elem is not None:
            total_bits_empirical += n * bits_emp_per_elem

    total_bytes_uniform = math.ceil(total_bits_uniform / 8.0) + archive_overhead_bytes
    total_bytes_empirical = (
        math.ceil(total_bits_empirical / 8.0) + archive_overhead_bytes
        if total_bits_empirical is not None
        else None
    )

    # Score conversion at zero distortion (information floor)
    score_uniform = contest_score(0.0, 0.0, total_bytes_uniform)
    score_empirical = (
        contest_score(0.0, 0.0, total_bytes_empirical)
        if total_bytes_empirical is not None
        else None
    )

    notes: list[str] = [
        f"Schema '{schema_label}' has {total_elements:,} elements "
        f"quantized at n_quant={n_quant} ({bits_uniform_per_elem:.3f} bits/elem uniform).",
        f"Uniform Shannon floor: {total_bytes_uniform:,} bytes "
        f"({total_bits_uniform:,.0f} bits content + {archive_overhead_bytes} overhead).",
    ]
    if total_bytes_empirical is not None:
        savings = total_bytes_uniform - total_bytes_empirical
        notes.append(
            f"Empirical Shannon floor: {total_bytes_empirical:,} bytes "
            f"(savings {savings:+,} vs uniform; assumes per-tensor entropy "
            f"matches empirical anchor)."
        )
    notes.append(
        f"At zero distortion this floor scores {score_uniform:.5f} "
        f"(rate term ~ {RATE_COEFFICIENT * total_bytes_uniform / CONTEST_REFERENCE_BYTES:.5f})."
    )
    if score_empirical is not None:
        notes.append(
            f"Empirical-floor score: {score_empirical:.5f}. "
            f"Below this you cannot go without either (a) lower-entropy "
            f"weights via retraining, or (b) lossy quantization that "
            f"increases distortion."
        )

    return ShannonFloorReport(
        schema_label=schema_label,
        total_elements=total_elements,
        n_quant=n_quant,
        components=components,
        total_bytes_uniform_floor=total_bytes_uniform,
        total_bytes_empirical_floor=total_bytes_empirical,
        score_at_uniform_floor_zero_distortion=score_uniform,
        score_at_empirical_floor_zero_distortion=score_empirical,
        notes=notes,
    )


def shannon_floor_pr101_default() -> ShannonFloorReport:
    """Convenience: Shannon floor for PR101's exact schema.

    PR101 uses N_QUANT=127 (signed int8 codes -63..63 + zero); the
    archive overhead is the latent_blob (15,387 B) + sidecar (607 B)
    + ZIP header (~100 B) ≈ 16,094 B. This is the floor on the
    decoder-blob-with-archive bytes for a perfectly-compressed PR101
    that uses uniform symbol distribution.
    """
    from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA, N_QUANT
    return compute_shannon_floor(
        schema=FIXED_STATE_SCHEMA,
        n_quant=N_QUANT,
        schema_label="PR101_FIXED_STATE_SCHEMA",
        archive_overhead_bytes=15_387 + 607 + 100,
    )


__all__ = [
    "ShannonFloorComponent",
    "ShannonFloorReport",
    "compute_shannon_floor",
    "shannon_floor_pr101_default",
]
