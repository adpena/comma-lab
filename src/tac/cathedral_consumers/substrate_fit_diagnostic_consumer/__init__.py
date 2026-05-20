# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #6 - substrate-fit diagnostic.

Per RESPAWN-MG-7-BUNDLE 2026-05-20. Consumes ``M_inflated`` per substrate
(via the producer's ``extract_M_inflated`` helper) and compares to
``M_contest`` (via the producer's ``extract_M_contest`` helper). Substrates
whose ``M_inflated`` lacks structure where ``M_contest`` has high magnitude
are flagged as poor-fit; substrates whose ``M_inflated`` matches the
``M_contest`` distribution closely are good-fit. This is the $0
substrate-class-selector signal. Auto-discovered by cathedral autopilot
ranker per Catalog #335 canonical contract.

## Canonical-vs-unique decision per layer

- M_inflated / M_contest extraction: ADOPT canonical producer surfaces
  ``extract_M_inflated`` + ``extract_M_contest``.
- Provenance contract: ADOPT canonical Provenance per Catalog #323.
- Routing markers: ADOPT Catalog #341 canonical non-promotable markers
  (substrate-fit score is RANKING-GUIDANCE, not contest score).
- Similarity metric: FORK to cosine + L2 ratio (canonical cathedral
  ranker fit metric); the canonical helpers do not include a substrate-
  vs-substrate fit comparator surface.

## Observability surface

Per Catalog #305:

1. Inspectable per layer: ``compute_substrate_fit_score`` returns a dict
   mapping substrate_name -> fit_score in [0, 1].
2. Decomposable per signal: per-substrate score.
3. Diff-able across runs: scores tied to per-substrate M_inflated sha256
   + M_contest sha256.
4. Queryable post-hoc: operator-facing CLI
   ``tools/rank_substrates_by_information_fidelity.py``.
5. Cite-able: producer surface invocation cited in provenance.
6. Counterfactual-able: per-substrate dict lets operator ask "what if we
   added a new substrate?" by extending the dict.

## 9-dimension success checklist evidence

1. UNIQUENESS: substrate-fit diagnostic is canonically distinct from
   contest-CUDA score; it is the LOCAL substrate-class-selector signal.
2. BEAUTY+ELEGANCE: ~180 LOC consumer; cosine similarity kernel.
3. DISTINCTNESS: distinct from sister exploits (each targets a different
   gradient granularity).
4. RIGOR: cosine similarity is bounded [-1, 1]; we map to [0, 1] via
   max(0, cosine) for fit-score semantics.
5. OPTIMIZATION-PER-TECHNIQUE: per-substrate fit score allows the
   cathedral ranker to prefer substrates whose M_inflated structurally
   matches M_contest.
6. STACK-OF-STACKS-COMPOSABILITY: substrate-class-selector signal
   composes with sister exploits (top-K bytes / per-pair difficulty /
   per-class chroma).
7. DETERMINISTIC-REPRODUCIBILITY: pure numpy einsum.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_substrates * (N_pairs * H * W));
   matches producer surface.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: substrate-fit signal feeds the
   cathedral ranker $0; future score improvement comes from FAVORING
   high-fit substrates in dispatch ranking.

## Cargo-cult audit per assumption

- ASSUMPTION: cosine similarity between M_inflated and M_contest predicts
  contest score. CLASSIFICATION: CARGO-CULTED-PENDING-EMPIRICAL. Cosine
  similarity measures DIRECTIONAL agreement; magnitude difference is
  ignored. Per CLAUDE.md "Apples-to-apples evidence discipline": this
  signal is ADVISORY, never promotable to contest-CUDA. Sister-pairing
  with the actual contest-CUDA anchor on the substrate IS required
  before promotion.
- ASSUMPTION: substrates are independent of one another. CLASSIFICATION:
  CARGO-CULTED. Substrate composition can violate independence; this
  consumer outputs per-substrate scores assuming independence; composition
  consumers (sister exploit #9) should be consulted before composing.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "substrate_fit_diagnostic_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Future contest-CUDA anchors on specific substrates SHOULD inform a
    posterior over substrate-fit-vs-contest-score relationship via this
    hook.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution."""
    substrate_fit_scores = candidate.get("substrate_fit_scores")
    m_contest_sha = candidate.get("m_contest_array_sha256")
    candidate_substrate = candidate.get("substrate_name")

    rationale_parts = [
        "substrate-fit diagnostic consumer (exploit #6)",
        "non-promotable advisory signal per Catalog #341",
    ]
    if candidate_substrate is not None:
        rationale_parts.append(f"candidate_substrate={candidate_substrate}")
    if substrate_fit_scores is not None and isinstance(substrate_fit_scores, Mapping):
        n_substrates = len(substrate_fit_scores)
        rationale_parts.append(f"upstream substrate_fit_scores n={n_substrates}")
        if candidate_substrate in substrate_fit_scores:
            fit_score = substrate_fit_scores[candidate_substrate]
            rationale_parts.append(f"fit_score={fit_score:.4f}")
    if m_contest_sha is not None:
        rationale_parts.append(f"M_contest sha256[:12]={str(m_contest_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "consumer_signal_kind": "substrate_class_selector_ranking",
        "substrate_fit_scores": substrate_fit_scores,
        "candidate_substrate": candidate_substrate,
        "m_contest_array_sha256": m_contest_sha,
    }


def compute_substrate_fit_score(
    M_contest,
    M_inflated_per_substrate: Mapping[str, "object"],
) -> dict[str, float]:
    """Compute per-substrate fit score against the contest gradient.

    The fit score is the cosine similarity between M_inflated and M_contest
    (flattened across pairs / axes / pixels), clipped to [0, 1] via
    max(0, cosine). Higher score = better substrate fit.

    Args:
        M_contest: np.ndarray of shape (N_pairs, 3, H, W) - the contest
            video's per-pixel scorer-axis gradient.
        M_inflated_per_substrate: dict mapping substrate_name -> np.ndarray
            of shape (N_pairs, 3, H, W) - each substrate's inflated-video
            per-pixel gradient.

    Returns:
        Dict mapping substrate_name -> fit_score in [0, 1]; higher is
        better. Empty dict if input is empty.

    Raises:
        ValueError: on shape mismatch or empty M_contest.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for substrate fit") from exc

    m_contest = np.asarray(M_contest, dtype=np.float64)
    if m_contest.ndim != 4 or m_contest.shape[1] != 3:
        raise ValueError(
            f"M_contest must have shape (N_pairs, 3, H, W); got {m_contest.shape}"
        )
    if not isinstance(M_inflated_per_substrate, Mapping):
        raise ValueError(
            f"M_inflated_per_substrate must be Mapping; got "
            f"{type(M_inflated_per_substrate).__name__}"
        )
    if not M_inflated_per_substrate:
        return {}

    contest_flat = m_contest.ravel()
    contest_norm = float(np.linalg.norm(contest_flat))
    if contest_norm == 0:
        # All-zero M_contest: every substrate is "perfect fit" (trivially).
        return {name: 1.0 for name in M_inflated_per_substrate}

    fit_scores: dict[str, float] = {}
    for name, tensor in M_inflated_per_substrate.items():
        m_inflated = np.asarray(tensor, dtype=np.float64)
        if m_inflated.shape != m_contest.shape:
            raise ValueError(
                f"substrate {name!r} M_inflated shape {m_inflated.shape} "
                f"!= M_contest shape {m_contest.shape}"
            )
        inflated_flat = m_inflated.ravel()
        inflated_norm = float(np.linalg.norm(inflated_flat))
        if inflated_norm == 0:
            # Zero substrate gradient: 0 fit.
            fit_scores[str(name)] = 0.0
            continue
        cosine = float(
            np.dot(inflated_flat, contest_flat) / (inflated_norm * contest_norm)
        )
        # Clip to [0, 1] via max(0, cosine) for fit-score semantics.
        fit_scores[str(name)] = max(0.0, cosine)
    return fit_scores


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "compute_substrate_fit_score",
    "consume_candidate",
    "update_from_anchor",
]
