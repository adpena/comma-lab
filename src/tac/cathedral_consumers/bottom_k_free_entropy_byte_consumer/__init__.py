# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #4 - bottom-K free-entropy byte ranking.

Sister of exploit #3 (top-K byte sensitivity) per RESPAWN-MG-7-BUNDLE
2026-05-20. Consumes ``M_archive`` per-byte gradients (derived via the CHAIN
RULE per Catalog #318) and identifies near-zero ``|partial S / partial byte|``
bytes - those eligible for further coarsening / constant prior replacement /
removal. Auto-discovered by cathedral autopilot ranker per Catalog #335
canonical contract.

## Canonical-vs-unique decision per layer

- Per-byte sensitivity extraction: ADOPT canonical
  ``tac.master_gradient_comparison.multi_granularity.extract_M_archive_via_chain_rule``
  producer surface (Catalog #318 chain-rule discipline).
- Provenance contract: ADOPT canonical Provenance per Catalog #323.
- Routing markers: ADOPT Catalog #341 canonical non-promotable markers
  (bottom-K is CODEC-DESIGN guidance, not a score promotion).
- Catalog #220 substrate L1+ scaffold operational mechanism: CITE. The
  bottom-K identification is the structural protection that prevents the
  "byte addition without operational mechanism" anti-pattern - if the
  caller proposes to add bytes that fall in the bottom-K free-entropy
  range, those bytes are by construction not operational.

## Observability surface

Per Catalog #305:

1. Inspectable per layer: ``rank_archive_bytes_by_low_sensitivity``
   returns a sorted list of byte indices.
2. Decomposable per signal: bottom-K decomposes by per-axis sensitivity.
3. Diff-able across runs: rankings tied to ``M_archive`` sha256.
4. Queryable post-hoc: operator-facing CLI
   ``tools/list_bottom_k_free_entropy_bytes.py``.
5. Cite-able: Catalog #318 + Catalog #220 cited in provenance.
6. Counterfactual-able: ranked list lets operator ask "what if we replaced
   the bottom-100 bytes with a constant prior?" without re-running scorer.

## 9-dimension success checklist evidence

1. UNIQUENESS: bottom-K free-entropy bytes are canonically distinct from
   top-K (sister exploit #3); together they partition the archive.
2. BEAUTY+ELEGANCE: ~180 LOC consumer; sorting + threshold filter.
3. DISTINCTNESS: distinct from sister exploit #3 (top-K) which targets
   the OPPOSITE end of the sensitivity distribution.
4. RIGOR: Catalog #318 chain-rule enforced at producer; Catalog #220
   operational-mechanism cited.
5. OPTIMIZATION-PER-TECHNIQUE: threshold + K-cap allows substrate to
   define "free-entropy" empirically.
6. STACK-OF-STACKS-COMPOSABILITY: composes with sister #3 (top-K
   protection); together they fully partition the archive.
7. DETERMINISTIC-REPRODUCIBILITY: ``np.argsort`` stable.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_bytes log N_bytes).
9. OPTIMAL-MINIMAL-CONTEST-SCORE: bottom-K bytes are the candidates for
   coarsening / removal; substrate can replace with constant prior +
   recover the bytes for entropy gain.

## Cargo-cult audit per assumption

- ASSUMPTION: bytes with low |dS/db| are SAFE to coarsen. CLASSIFICATION:
  CARGO-CULTED-PENDING-EMPIRICAL. Catalog #318 chain-rule sensitivity is
  LOCAL at the operating point; bytes that look low-sensitivity locally
  may matter globally (e.g. corrupting them may corrupt the inflate path
  entirely if they are control bytes). The consumer surfaces candidates
  without claiming safety; substrate must verify via byte-mutation smoke
  per Catalog #139 / #272 distinguishing-feature contract.
- ASSUMPTION: ``sensitivity_threshold=1e-6`` is the right cutoff.
  CLASSIFICATION: CARGO-CULTED. The consumer exposes threshold as a
  parameter; per-substrate empirical calibration required.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "bottom_k_free_entropy_byte_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Future empirical anchors that prove a specific sensitivity threshold
    optimizes contest-CUDA score SHOULD inform a substrate-specific
    default threshold via this hook.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Routing-style consumer per Catalog #341: returns canonical non-promotable
    markers because bottom-K is CODEC-DESIGN guidance.
    """
    bottom_k_indices = candidate.get("bottom_k_free_entropy_byte_indices")
    m_archive_sha = candidate.get("m_archive_array_sha256")
    threshold = candidate.get("sensitivity_threshold")

    rationale_parts = [
        "bottom-K free-entropy byte consumer (exploit #4)",
        "Catalog #318 chain-rule discipline (raw bit-flip FD FORBIDDEN)",
        "Catalog #220 operational-mechanism cited (byte must be verified safe to coarsen)",
    ]
    if threshold is not None:
        rationale_parts.append(f"sensitivity_threshold={threshold}")
    if bottom_k_indices is not None:
        rationale_parts.append(f"upstream bottom_k_indices n={len(bottom_k_indices)}")
    if m_archive_sha is not None:
        rationale_parts.append(f"M_archive sha256[:12]={str(m_archive_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "consumer_signal_kind": "bottom_k_free_entropy_coarsening_candidates",
        "bottom_k_free_entropy_byte_indices": bottom_k_indices,
        "sensitivity_threshold": threshold,
        "m_archive_array_sha256": m_archive_sha,
    }


def rank_archive_bytes_by_low_sensitivity(
    M_archive,
    k_bottom: int,
    sensitivity_threshold: float = 1e-6,
    *,
    axis: int | None = None,
) -> list[int]:
    """Rank archive bytes by |partial S / partial byte| ASCENDING; return bottom-K with sensitivity below threshold.

    Per Catalog #318: ``M_archive`` MUST be chain-rule-derived; raw bit-flip
    FD is FORBIDDEN.

    Args:
        M_archive: np.ndarray of shape (N_bytes, 3) - chain-rule-derived
            per-byte gradient.
        k_bottom: maximum number of bottom-sensitivity bytes to return;
            must be >= 0.
        sensitivity_threshold: only bytes with sensitivity STRICTLY BELOW
            this value are included in the result; default 1e-6.
        axis: optional axis index in {0, 1, 2}; None ranks by aggregate L2.

    Returns:
        List of byte indices, sorted by sensitivity ASCENDING. Length is
        AT MOST min(k_bottom, N_bytes) (may be less if fewer than k_bottom
        bytes fall below threshold).

    Raises:
        ValueError: on shape mismatch / invalid k_bottom / invalid axis /
            negative threshold.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for byte ranking") from exc

    arr = np.asarray(M_archive, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"M_archive must have shape (N_bytes, 3); got {arr.shape}"
        )
    n_bytes = arr.shape[0]
    if k_bottom < 0:
        raise ValueError(f"k_bottom must be >= 0; got {k_bottom}")
    if sensitivity_threshold < 0:
        raise ValueError(
            f"sensitivity_threshold must be >= 0; got {sensitivity_threshold}"
        )
    if k_bottom == 0:
        return []

    if axis is None:
        magnitude = np.linalg.norm(arr, axis=1)
    else:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be in {{0, 1, 2}}; got {axis}")
        magnitude = np.abs(arr[:, axis])

    # Ascending sort; take first k_bottom; filter to those strictly below threshold.
    order = np.argsort(magnitude, kind="stable")
    result: list[int] = []
    for idx in order[:k_bottom]:
        if magnitude[int(idx)] < sensitivity_threshold:
            result.append(int(idx))
        else:
            # Sorted ascending, so once we exceed threshold, all subsequent
            # are >= threshold; can early-exit.
            break
    return result


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "consume_candidate",
    "rank_archive_bytes_by_low_sensitivity",
    "update_from_anchor",
]
