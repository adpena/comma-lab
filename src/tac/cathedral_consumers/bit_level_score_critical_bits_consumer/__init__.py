# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #8 - bit-level score-critical bits.

Per RESPAWN-MG-7-BUNDLE 2026-05-20. Extends sister exploit #3 (top-K bytes)
from byte-level to BIT-level. Bit-level master gradient is sparser than
byte-level; this consumer identifies "score-critical bits" that must NEVER
flip via entropy coding errors. Per Catalog #318 the bit-level gradient is
DERIVED from byte-level via 8x expansion + per-bit sensitivity weighting -
never raw bit-flip finite differences. Auto-discovered by cathedral autopilot
ranker per Catalog #335 canonical contract.

## Canonical-vs-unique decision per layer

- Bit-level sensitivity extraction: FORK from byte-level via 8x expansion
  + per-bit positional weighting (factor 2^bit_pos for the MSB-aligned
  interpretation; MSB matters more than LSB for unsigned int8 byte
  semantics). The producer surface
  ``tac.master_gradient_comparison.multi_granularity.extract_M_archive_via_chain_rule``
  returns BYTE-level gradient; bit-level derivation is THIS consumer's
  responsibility. NO raw bit-flip FD per Catalog #318.
- Provenance contract: ADOPT canonical Provenance per Catalog #323.
- Routing markers: ADOPT Catalog #341 canonical non-promotable markers
  (bit-criticality is ENTROPY-CODER guidance, not a score-promotion).

## Observability surface

Per Catalog #305:

1. Inspectable per layer: ``extract_M_archive_bit_level`` returns
   (8 * N_bytes, 3) ndarray; per-bit sensitivity is queryable.
2. Decomposable per signal: per-bit decomposes into (byte_idx, bit_pos)
   pairs.
3. Diff-able across runs: bit-level rankings tied to byte-level
   M_archive sha256.
4. Queryable post-hoc: operator-facing CLI
   ``tools/list_score_critical_bits.py``.
5. Cite-able: Catalog #318 cited in provenance.
6. Counterfactual-able: per-bit ranking lets operator ask "what if we
   protected only the top-N bits?" without re-running scorer.

## 9-dimension success checklist evidence

1. UNIQUENESS: bit-level granularity is canonically distinct from
   byte-level (sister #3) because entropy coders operate at the bit
   level; sub-byte protection is impossible without bit-level signal.
2. BEAUTY+ELEGANCE: ~180 LOC consumer; 8x expansion + weighted sort.
3. DISTINCTNESS: distinct from sister #3 (byte-level); finer granularity.
4. RIGOR: Catalog #318 chain-rule respected (derived from byte-level,
   never raw FD); MSB-aligned weighting is principled (per-bit
   contribution to integer value is 2^bit_pos).
5. OPTIMIZATION-PER-TECHNIQUE: per-bit MSB weighting reflects entropy-
   coder bit semantics.
6. STACK-OF-STACKS-COMPOSABILITY: bit-level consumer composes with sister
   #3 byte-level; together they inform multi-granularity protection.
7. DETERMINISTIC-REPRODUCIBILITY: pure numpy.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_bytes * 8 * log(N_bytes * 8)).
9. OPTIMAL-MINIMAL-CONTEST-SCORE: bit-level protection IS the canonical
   entropy-coder safeguard; future substrates with arithmetic / range
   coders consume this signal directly.

## Cargo-cult audit per assumption

- ASSUMPTION: MSB-aligned weighting (2^bit_pos) is the correct bit-level
  sensitivity model. CLASSIFICATION: HARD-EARNED for unsigned int8
  byte-as-integer interpretation; CARGO-CULTED for arbitrary codec
  semantics (e.g. some codecs use sign-extended bits or fractional
  arithmetic). Consumer documents the assumption and exposes the
  weighting as an optional override.
- ASSUMPTION: bit-level sensitivity from byte-level chain-rule is exact.
  CLASSIFICATION: CARGO-CULTED. The chain rule via 8x expansion is an
  APPROXIMATION; true bit-level chain rule requires the inflate Jacobian
  at bit granularity (currently producer surface operates at byte
  granularity). Consumer documents the approximation and reports it as
  per-bit ADVISORY signal.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "bit_level_score_critical_bits_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Future contest-CUDA anchors with empirical evidence on per-bit
    protection thresholds SHOULD inform substrate-specific defaults via
    this hook.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution."""
    score_critical_bits = candidate.get("score_critical_bit_indices")
    m_archive_sha = candidate.get("m_archive_array_sha256")
    k_top = candidate.get("k_top_bits")

    rationale_parts = [
        "bit-level score-critical bits consumer (exploit #8)",
        "Catalog #318 chain-rule discipline (derived from byte-level; raw bit-flip FD FORBIDDEN)",
        "MSB-aligned weighting (factor 2^bit_pos for unsigned int8 byte semantics)",
    ]
    if k_top is not None:
        rationale_parts.append(f"k_top_bits={k_top}")
    if score_critical_bits is not None:
        rationale_parts.append(
            f"upstream score_critical_bits n={len(score_critical_bits)}"
        )
    if m_archive_sha is not None:
        rationale_parts.append(f"M_archive sha256[:12]={str(m_archive_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "consumer_signal_kind": "bit_level_entropy_coder_protection_recommendation",
        "score_critical_bit_indices": score_critical_bits,
        "k_top_bits": k_top,
        "m_archive_array_sha256": m_archive_sha,
    }


def extract_M_archive_bit_level(
    M_archive_byte_level,
    *,
    msb_aligned: bool = True,
):
    """Derive bit-level master gradient from byte-level via 8x expansion.

    Per Catalog #318: ``M_archive_byte_level`` MUST be chain-rule-derived
    (the producer surface enforces this). The bit-level expansion is:

        M_archive_bit[bit_idx, axis] = M_archive_byte[byte_idx, axis] *
                                       2^(bit_pos)

    where ``byte_idx = bit_idx // 8`` and ``bit_pos = bit_idx % 8`` (or
    ``7 - bit_pos`` for MSB-first interpretation per ``msb_aligned``).

    This is an APPROXIMATION; true bit-level chain rule requires the
    inflate Jacobian at bit granularity which the producer surface does
    not currently expose.

    Args:
        M_archive_byte_level: np.ndarray of shape (N_bytes, 3) -
            chain-rule-derived byte-level gradient.
        msb_aligned: if True (default), bit 0 is the MSB; else bit 0 is
            the LSB.

    Returns:
        np.ndarray of shape (N_bytes * 8, 3) - per-bit sensitivity.

    Raises:
        ValueError: on shape mismatch.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for bit-level extraction") from exc

    arr = np.asarray(M_archive_byte_level, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"M_archive_byte_level must have shape (N_bytes, 3); got {arr.shape}"
        )
    n_bytes = arr.shape[0]
    # Per-bit MSB-aligned weighting: bit 0 contributes 2^7 = 128, bit 7
    # contributes 2^0 = 1.
    if msb_aligned:
        bit_weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.float64)
    else:
        bit_weights = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.float64)
    # Outer product: per-byte sensitivity * per-bit weight.
    # Shape: (N_bytes, 8, 3) -> reshape (N_bytes * 8, 3).
    expanded = arr[:, None, :] * bit_weights[None, :, None]
    return expanded.reshape(n_bytes * 8, 3)


def rank_score_critical_bits(
    M_archive_bit_level,
    k_top: int,
    *,
    axis: int | None = None,
) -> list[int]:
    """Rank bit-level sensitivity descending; return top-K bit indices.

    Args:
        M_archive_bit_level: np.ndarray of shape (N_bits, 3) - bit-level
            per-axis sensitivity from ``extract_M_archive_bit_level``.
        k_top: number of top bits to return.
        axis: optional axis index in {0, 1, 2}; None ranks by aggregate L2.

    Returns:
        List of bit indices, sorted DESCENDING by sensitivity.

    Raises:
        ValueError: on shape mismatch / invalid k_top / invalid axis.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for bit ranking") from exc

    arr = np.asarray(M_archive_bit_level, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"M_archive_bit_level must have shape (N_bits, 3); got {arr.shape}"
        )
    n_bits = arr.shape[0]
    if k_top < 0:
        raise ValueError(f"k_top must be >= 0; got {k_top}")
    k_top_clamped = min(k_top, n_bits)
    if k_top_clamped == 0:
        return []

    if axis is None:
        magnitude = np.linalg.norm(arr, axis=1)
    else:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be in {{0, 1, 2}}; got {axis}")
        magnitude = np.abs(arr[:, axis])

    order = np.argsort(-magnitude, kind="stable")
    return [int(i) for i in order[:k_top_clamped]]


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "consume_candidate",
    "extract_M_archive_bit_level",
    "rank_score_critical_bits",
    "update_from_anchor",
]
