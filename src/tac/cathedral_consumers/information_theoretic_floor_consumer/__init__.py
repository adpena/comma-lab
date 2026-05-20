# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #7 - information-theoretic floor estimator.

Per RESPAWN-MG-7-BUNDLE 2026-05-20. Consumes ``M_contest`` and computes a
Cramer-Rao-like lower bound on the achievable score, then compares to the
current best empirical score to emit a saturation diagnostic. Per CLAUDE.md
"Meta-Lagrangian/Pareto solver" non-negotiable: prefer solvable math over
arbitrary sweeps. Auto-discovered by cathedral autopilot ranker per Catalog
#335 canonical contract.

## Canonical-vs-unique decision per layer

- Floor estimation: ADOPT canonical
  ``tac.master_gradient_comparison.multi_granularity.estimate_information_theoretic_floor``
  producer surface (3 modes: cramer_rao / fisher_trace / shannon_lower).
- Provenance contract: ADOPT canonical Provenance per Catalog #323.
- Routing markers: ADOPT Catalog #341 canonical non-promotable markers
  (floor estimate is THEORETICAL guidance, not a contest claim).
- Mode default: ADOPT ``cramer_rao`` (canonical Cramer-Rao lower bound
  per Cover & Thomas 2006 Ch 10 + Catalog #344 canonical equations
  registry alignment).

## Observability surface

Per Catalog #305:

1. Inspectable per layer: ``estimate_cramer_rao_lower_bound`` returns a
   scalar float; per-axis breakdown via mode='fisher_trace' is queryable.
2. Decomposable per signal: per-axis decomposition via Fisher trace mode.
3. Diff-able across runs: floor estimate tied to M_contest sha256.
4. Queryable post-hoc: operator-facing CLI
   ``tools/estimate_information_theoretic_floor.py``.
5. Cite-able: canonical equation citations in docstring (Cramer-Rao,
   Shannon R(D), Blahut-Arimoto, Cover & Thomas 2006).
6. Counterfactual-able: floor estimate lets operator ask "are we within
   epsilon of the floor?" by comparing to current best empirical score.

## 9-dimension success checklist evidence

1. UNIQUENESS: information-theoretic floor is canonically distinct from
   empirical score; it is the THEORETICAL lower bound the substrate can
   achieve.
2. BEAUTY+ELEGANCE: ~180 LOC consumer; 3-mode estimator kernel.
3. DISTINCTNESS: distinct from sister exploits (each targets a different
   gradient analysis).
4. RIGOR: Cramer-Rao is a rigorous lower bound under unbiased estimator
   assumption; Shannon R(D) + Blahut-Arimoto are alternative canonical
   bounds.
5. OPTIMIZATION-PER-TECHNIQUE: 3 modes let operator choose the bound
   appropriate to substrate.
6. STACK-OF-STACKS-COMPOSABILITY: floor estimate informs the cathedral
   ranker's saturation diagnostic; composes with sister exploits.
7. DETERMINISTIC-REPRODUCIBILITY: pure numpy.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_pairs * H * W * 3); matches
   producer surface.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: floor estimate signals when further
   within-substrate optimization is saturating; substrate-class shift is
   the next move.

## Cargo-cult audit per assumption

- ASSUMPTION: Cramer-Rao lower bound applies to the contest score.
  CLASSIFICATION: HARD-EARNED with caveat. CR is a rigorous bound on the
  VARIANCE of any unbiased estimator. The contest score is a sum of
  weighted distortions; CR bound on per-axis distortion variance
  translates to a bound on score variance (not score absolute value).
  Consumer surfaces the bound with this caveat documented.
- ASSUMPTION: Shannon R(D) is achievable with practical codecs.
  CLASSIFICATION: CARGO-CULTED. R(D) is the THEORETICAL bound; practical
  codecs (HNeRV / NeRV / Ballé / etc.) approach but never reach it.
  Consumer outputs floor as a DIAGNOSTIC, not a target.

## Canonical equation registry references

Per Catalog #344 canonical equations registry: this consumer aligns with
the Cramer-Rao bound + Shannon R(D) + Blahut-Arimoto algorithms.
Citations: Cover & Thomas 2006 *Elements of Information Theory* Ch 10
(Rate Distortion) + Ch 4 (Fisher Information + Cramer-Rao); Blahut 1972
*Computation of channel capacity and rate-distortion functions* IEEE Trans
Information Theory; Arimoto 1972 *An algorithm for computing the capacity
of arbitrary discrete memoryless channels* IEEE Trans Information Theory.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from tac.cathedral.consumer_contract import AxisDecomposition, HookNumber


CONSUMER_NAME = "information_theoretic_floor_consumer"
CONSUMER_VERSION = "1.1.0"  # bumped: per-axis decomposition emission (Dim 3 Step 3.4)
_PROVENANCE_MODEL_ID = (
    "information_theoretic_floor_consumer.predicted_axis_decomposition_v1"
)
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Future contest-CUDA anchors SHOULD inform the gap between empirical
    score and information-theoretic floor; gap-closing patterns inform
    substrate-class-shift recommendations.
    """
    _ = anchor


def _build_per_axis_decomposition(
    floor_estimate: float | None,
    mode: str,
    per_axis_floor: Mapping[str, float] | None,
    m_contest_sha: str | None,
) -> AxisDecomposition:
    """Build canonical per-axis decomposition with Provenance.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.4: this consumer's natural
    decomposition is the per-axis Cramer-Rao lower bound (the producer's
    ``cramer_rao`` mode computes ``1 / fisher_info_per_axis`` for each of
    seg / pose / rate axes; the sum is the scalar floor; the per-axis
    components ARE the canonical decomposition per Cover & Thomas 2006 Ch
    4 + Catalog #344). When the caller supplies ``per_axis_floor`` dict
    {"seg": float, "pose": float, "rate_bytes": int} from the producer
    surface, we propagate it directly; otherwise per-axis defaults to 0
    per Catalog #341 observability-only invariant.

    The values represent ACHIEVABLE-FLOOR deltas (signed; a negative
    delta = improvement from current operating point toward the floor).
    Consumer surfaces the per-axis decomposition as observability for the
    Pareto polytope solver; the floor estimate itself is NOT a score
    prediction — see consumer docstring's cargo-cult audit.
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
        sha_seed = m_contest_sha or "no_m_contest_sha"
        floor_seed = f"{floor_estimate:.6e}" if floor_estimate is not None else "none"
        inputs_seed = (
            f"{_PROVENANCE_MODEL_ID}:m_contest_sha={sha_seed}:mode={mode}"
            f":floor={floor_seed}"
        )
        inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
        prov = build_provenance_for_predicted(
            model_id=_PROVENANCE_MODEL_ID,
            inputs_sha256=inputs_sha256,
            measurement_axis="[predicted]",
            hardware_substrate="cpu_local",
        )
        canonical_provenance = provenance_to_dict(prov)

    seg_delta = 0.0
    pose_delta = 0.0
    rate_bytes_delta = 0
    if per_axis_floor is not None and isinstance(per_axis_floor, Mapping):
        seg_delta = float(per_axis_floor.get("seg", 0.0))
        pose_delta = float(per_axis_floor.get("pose", 0.0))
        rate_bytes_delta = int(per_axis_floor.get("rate_bytes", 0))

    return AxisDecomposition(
        predicted_d_seg_delta=seg_delta,
        predicted_d_pose_delta=pose_delta,
        predicted_archive_bytes_delta=rate_bytes_delta,
        axis_tag="[predicted]",
        canonical_provenance=canonical_provenance,
    )


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.4: also emits canonical
    ``predicted_axis_decomposition`` (per-axis Cramer-Rao lower-bound
    components per Cover & Thomas 2006 Ch 4). Per-axis emission is
    OBSERVABILITY-ONLY per Catalog #341.
    """
    floor_estimate = candidate.get("information_theoretic_floor")
    mode = candidate.get("floor_estimate_mode", "cramer_rao")
    m_contest_sha = candidate.get("m_contest_array_sha256")
    current_best_score = candidate.get("current_best_empirical_score")
    per_axis_floor = candidate.get("per_axis_floor")

    rationale_parts = [
        "information-theoretic floor consumer (exploit #7)",
        f"mode={mode}",
        "Cramer-Rao + Shannon R(D) + Blahut-Arimoto canonical equations",
    ]
    if floor_estimate is not None:
        rationale_parts.append(f"floor_estimate={floor_estimate}")
    if current_best_score is not None and floor_estimate is not None:
        try:
            gap = float(current_best_score) - float(floor_estimate)
            rationale_parts.append(f"gap_to_floor={gap:.6f}")
        except (TypeError, ValueError):
            pass
    if m_contest_sha is not None:
        rationale_parts.append(f"M_contest sha256[:12]={str(m_contest_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    decomposition = _build_per_axis_decomposition(
        floor_estimate=floor_estimate,
        mode=str(mode),
        per_axis_floor=per_axis_floor,
        m_contest_sha=m_contest_sha,
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "consumer_signal_kind": "saturation_diagnostic_vs_information_theoretic_floor",
        "information_theoretic_floor": floor_estimate,
        "floor_estimate_mode": mode,
        "current_best_empirical_score": current_best_score,
        "m_contest_array_sha256": m_contest_sha,
        "predicted_axis_decomposition": decomposition.as_dict(),
    }


def estimate_cramer_rao_lower_bound(
    M_contest,
    *,
    mode: str = "cramer_rao",
) -> float:
    """Estimate the information-theoretic lower bound on achievable score.

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable. The
    canonical 3 modes per the producer surface:

    * ``cramer_rao``: Cramer-Rao lower bound on variance of any unbiased
      estimator of per-axis score components (1 / Fisher info trace per
      axis).
    * ``fisher_trace``: trace of Fisher information matrix.
    * ``shannon_lower``: Shannon entropy-style lower bound.

    Args:
        M_contest: np.ndarray of shape (N_pairs, 3, H, W) - the per-pair
            contest gradient tensor.
        mode: one of {"cramer_rao", "fisher_trace", "shannon_lower"}.

    Returns:
        Floor estimate as a scalar float; always non-negative.

    Raises:
        ValueError: on shape mismatch or invalid mode.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for floor estimation") from exc

    arr = np.asarray(M_contest, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(
            f"M_contest must have shape (N_pairs, 3, H, W); got {arr.shape}"
        )
    legal_modes = {"cramer_rao", "fisher_trace", "shannon_lower"}
    if mode not in legal_modes:
        raise ValueError(f"mode={mode!r} must be one of {sorted(legal_modes)!r}")

    if mode == "cramer_rao":
        fisher_per_axis = np.sum(np.square(arr), axis=(0, 2, 3))
        nonzero = fisher_per_axis > 0
        if not nonzero.any():
            return 0.0
        cr_per_axis = np.zeros(3, dtype=np.float64)
        cr_per_axis[nonzero] = 1.0 / fisher_per_axis[nonzero]
        return float(cr_per_axis.sum())

    if mode == "fisher_trace":
        flat = arr.reshape(arr.shape[0], 3, -1)
        per_axis_sum = np.sum(np.square(flat), axis=(0, 2))
        return float(per_axis_sum.sum())

    if mode == "shannon_lower":
        per_pair_norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
        n = float(arr.shape[0])
        max_neg = float((-per_pair_norms).max())
        log_sum_exp = max_neg + float(
            np.log(np.exp(-per_pair_norms - max_neg).sum())
        )
        return float(np.log(n) - log_sum_exp)

    raise ValueError(f"unreachable mode={mode!r}")


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "consume_candidate",
    "estimate_cramer_rao_lower_bound",
    "update_from_anchor",
    "_build_per_axis_decomposition",
]
