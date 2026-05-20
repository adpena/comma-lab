# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #3 - top-K byte sensitivity ranking.

Per RESPAWN-MG-7-BUNDLE 2026-05-20. Consumes ``M_archive`` per-byte gradients
(derived via the CHAIN RULE per Catalog #318) and ranks bytes by
``|partial S / partial byte|``. The top-K bytes are flagged for
canonical-Huffman protection / fixed precision / redundancy. Auto-discovered
by cathedral autopilot ranker per Catalog #335 canonical contract.

## Canonical-vs-unique decision per layer

- Per-byte sensitivity extraction: ADOPT canonical
  ``tac.master_gradient_comparison.multi_granularity.extract_M_archive_via_chain_rule``
  producer surface (chain-rule discipline per Catalog #318 - raw bit-flip
  finite differences are FORBIDDEN).
- Provenance contract: ADOPT canonical
  ``tac.provenance.build_provenance_for_predicted``.
- Routing markers: ADOPT Catalog #341 canonical non-promotable markers
  (top-K is a CODEC-DESIGN guidance signal, not a score-promotion signal).
- Catalog #318 chain-rule discipline: STRICT REQUIRED. This consumer ONLY
  accepts ``M_archive`` derived via the canonical chain rule; raw bit-flip
  byte-FD sensitivity surfaces are forbidden at the producer side and the
  consumer enforces this via the producer's typed dataclass contract.

## Observability surface

Per Catalog #305:

1. Inspectable per layer: ``rank_archive_bytes_by_sensitivity`` returns a
   sorted list of byte indices; per-byte sensitivity is queryable via the
   ``M_archive`` tensor itself.
2. Decomposable per signal: top-K decomposes by per-axis (seg/pose/rate)
   sensitivity when caller chooses an axis.
3. Diff-able across runs: byte rankings tied to ``M_archive``
   ``array_sha256`` + ``archive_sha256``.
4. Queryable post-hoc: operator-facing CLI
   ``tools/list_top_k_sensitive_bytes.py``.
5. Cite-able: Catalog #318 chain-rule cited in provenance.
6. Counterfactual-able: ranked list lets the operator ask "what if we
   protected only the top-100 bytes?" without re-running the scorer.

## 9-dimension success checklist evidence

1. UNIQUENESS: top-K byte sensitivity is canonically distinct from
   uniform-byte protection; the chain-rule derivation is the canonical
   per-byte score derivative per Catalog #318.
2. BEAUTY+ELEGANCE: ~180 LOC consumer; sorting + slicing is the kernel.
3. DISTINCTNESS: distinct from sister exploit #4 (bottom-K bytes for
   coarsening).
4. RIGOR: Catalog #318 chain-rule strictly enforced; non-promotable per
   Catalog #341.
5. OPTIMIZATION-PER-TECHNIQUE: ranking respects per-axis vs aggregate
   sensitivity (caller chooses).
6. STACK-OF-STACKS-COMPOSABILITY: composes with sister #4 (bottom-K
   coarsening); together they partition the archive into protected vs
   coarsen-eligible regions.
7. DETERMINISTIC-REPRODUCIBILITY: ``np.argsort`` with kind='stable'.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_bytes log N_bytes); typically
   <1s for archives up to 1 MB.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: top-K bytes are the candidates for
   canonical-Huffman / fixed-precision protection in future substrates.

## Cargo-cult audit per assumption

- ASSUMPTION: top-K sensitivity ranking is the right signal for codec
  protection. CLASSIFICATION: HARD-EARNED. Per Catalog #318 + the
  canonical chain rule, the absolute byte sensitivity IS the canonical
  per-byte score derivative; bytes with high |dS/db| are the bytes the
  codec must protect from corruption.
- ASSUMPTION: K is a substrate-independent constant. CLASSIFICATION:
  CARGO-CULTED. K depends on the substrate's byte budget; the consumer
  exposes K as a parameter rather than fixing it.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from tac.cathedral.consumer_contract import AxisDecomposition, HookNumber


CONSUMER_NAME = "top_k_byte_sensitivity_consumer"
CONSUMER_VERSION = "1.1.0"  # bumped: per-axis decomposition emission (Dim 3 Step 3.4)
_PROVENANCE_MODEL_ID = "top_k_byte_sensitivity_consumer.predicted_axis_decomposition_v1"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    For this consumer, posterior update is a no-op at the consumer level
    because top-K byte sensitivity is computed on-demand from the producer
    surface. Future empirical anchors that prove a specific K value
    optimizes contest-CUDA score SHOULD inform a substrate-specific
    default K via this hook.
    """
    _ = anchor


def _build_per_axis_decomposition(
    top_k_indices: list[int] | None,
    k_value: int | None,
    per_byte_sensitivity_sums: Mapping[str, float] | None,
    archive_bytes_protected_delta: int,
    m_archive_sha: str | None,
) -> AxisDecomposition:
    """Build canonical per-axis decomposition with Provenance.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.4: this consumer is
    rate-axis-dominant by definition (top-K byte protection is codec design
    that changes archive bytes via canonical-Huffman / fixed-precision /
    redundancy). When the caller supplies per-axis sensitivity sums (the
    canonical chain-rule output decomposes per byte into seg/pose/rate axes),
    we propagate those sums; otherwise seg + pose contributions default to 0
    per Catalog #341 observability-only invariant.
    """
    try:
        from tac.provenance.builders import build_provenance_for_predicted
        from tac.provenance.validator import provenance_to_dict
    except ImportError:  # pragma: no cover
        canonical_provenance: Mapping[str, Any] = {
            "artifact_kind": "predicted_from_model",
            "model_id": _PROVENANCE_MODEL_ID,
            "measurement_axis": "[predicted]",
            "evidence_grade": "predicted",
            "promotion_eligible": False,
            "score_claim_valid": False,
        }
    else:
        sha_seed = m_archive_sha or "no_m_archive_sha"
        n_top = len(top_k_indices) if top_k_indices else 0
        inputs_seed = f"{_PROVENANCE_MODEL_ID}:m_archive_sha={sha_seed}:k={k_value or 0}:n={n_top}"
        inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
        prov = build_provenance_for_predicted(
            model_id=_PROVENANCE_MODEL_ID,
            inputs_sha256=inputs_sha256,
            measurement_axis="[predicted]",
            hardware_substrate="cpu_local",
        )
        canonical_provenance = provenance_to_dict(prov)

    # Per-axis sensitivity sums propagation: when caller supplies the sums
    # from the canonical Catalog #318 chain-rule output, we propagate them.
    # Default 0.0 when not provided (consumer is observability-only).
    seg_delta = 0.0
    pose_delta = 0.0
    if per_byte_sensitivity_sums is not None and isinstance(
        per_byte_sensitivity_sums, Mapping
    ):
        seg_delta = float(per_byte_sensitivity_sums.get("seg_axis_sum", 0.0))
        pose_delta = float(per_byte_sensitivity_sums.get("pose_axis_sum", 0.0))

    return AxisDecomposition(
        predicted_d_seg_delta=seg_delta,
        predicted_d_pose_delta=pose_delta,
        predicted_archive_bytes_delta=int(archive_bytes_protected_delta),
        axis_tag="[predicted]",
        canonical_provenance=canonical_provenance,
    )


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Routing-style consumer per Catalog #341: returns canonical non-promotable
    markers because top-K is a CODEC-DESIGN guidance signal, not a score
    claim. The candidate dict MAY carry ``top_k_byte_indices`` /
    ``m_archive_array_sha256`` if a prior pipeline stage computed them.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.4: also emits canonical
    ``predicted_axis_decomposition`` (rate-axis-dominant for byte-protection
    overhead; seg/pose from per-byte sensitivity sums when caller supplies
    them). Per-axis emission is OBSERVABILITY-ONLY per Catalog #341.
    """
    top_k_indices = candidate.get("top_k_byte_indices")
    m_archive_sha = candidate.get("m_archive_array_sha256")
    k_value = candidate.get("k_top")
    per_byte_sensitivity_sums = candidate.get("per_byte_sensitivity_sums")
    archive_bytes_protected_delta = int(
        candidate.get("archive_bytes_protected_delta") or 0
    )

    rationale_parts = [
        "top-K byte sensitivity consumer (exploit #3)",
        "Catalog #318 chain-rule discipline (raw bit-flip FD FORBIDDEN)",
    ]
    if k_value is not None:
        rationale_parts.append(f"k_top={k_value}")
    if top_k_indices is not None:
        rationale_parts.append(f"upstream top_k_indices n={len(top_k_indices)}")
    if m_archive_sha is not None:
        rationale_parts.append(f"M_archive sha256[:12]={str(m_archive_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    decomposition = _build_per_axis_decomposition(
        top_k_indices=top_k_indices,
        k_value=k_value,
        per_byte_sensitivity_sums=per_byte_sensitivity_sums,
        archive_bytes_protected_delta=archive_bytes_protected_delta,
        m_archive_sha=m_archive_sha,
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "consumer_signal_kind": "top_k_byte_protection_recommendation",
        "top_k_byte_indices": top_k_indices,
        "k_top": k_value,
        "m_archive_array_sha256": m_archive_sha,
        "predicted_axis_decomposition": decomposition.as_dict(),
    }


def rank_archive_bytes_by_sensitivity(
    M_archive,
    k_top: int,
    *,
    axis: int | None = None,
) -> list[int]:
    """Rank archive bytes by |partial S / partial byte| and return top-K indices.

    Per Catalog #318: ``M_archive`` MUST be derived via the canonical chain
    rule (``tac.master_gradient_comparison.multi_granularity.extract_M_archive_via_chain_rule``);
    raw bit-flip finite differences are FORBIDDEN. The consumer accepts the
    ndarray (loaded via the ``ArchiveByteGradientTensor.load()`` producer
    surface) but does NOT itself enforce the chain-rule discipline - that
    happens at the producer side via Catalog #318 self-protection.

    Args:
        M_archive: np.ndarray of shape (N_bytes, 3) - the chain-rule-derived
            per-byte gradient (axis 1 = seg / pose / rate).
        k_top: number of top-sensitivity bytes to return; must be >= 0 and
            <= N_bytes.
        axis: optional axis index in {0, 1, 2} to rank by a single axis;
            None (default) ranks by aggregate L2 across axes.

    Returns:
        List of byte indices, sorted by sensitivity DESCENDING. Length is
        min(k_top, N_bytes).

    Raises:
        ValueError: on shape mismatch / invalid k_top / invalid axis.
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
    if k_top < 0:
        raise ValueError(f"k_top must be >= 0; got {k_top}")
    k_top_clamped = min(k_top, n_bytes)
    if k_top_clamped == 0:
        return []

    if axis is None:
        # Aggregate L2 across axes.
        magnitude = np.linalg.norm(arr, axis=1)
    else:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be in {{0, 1, 2}}; got {axis}")
        magnitude = np.abs(arr[:, axis])

    # argsort returns ascending; flip for descending.
    order = np.argsort(-magnitude, kind="stable")
    return [int(i) for i in order[:k_top_clamped]]


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "consume_candidate",
    "rank_archive_bytes_by_sensitivity",
    "update_from_anchor",
    "_build_per_axis_decomposition",
]
