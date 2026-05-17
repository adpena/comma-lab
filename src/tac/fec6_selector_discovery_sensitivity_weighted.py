# SPDX-License-Identifier: MIT
"""Sensitivity-weighted reweighting helper for fec6 selector discovery (Ext 3).

This module is the canonical helper for Ext 3 of the fec6 stacking wave
(lane ``lane_fec6_stacking_wave_5_grammar_extensions_20260517``). It is a
small, focused per-pair scoring helper that consumes the canonical
:mod:`tac.sensitivity_map.axis_weights` API and applies a per-pair
reweighting to the fec6 selector-discovery loop's candidate scores.

The fec6 selector-discovery loop is OFFLINE (no GPU; macOS-CPU is
sufficient). It picks one palette mode per pair (out of K=16 modes) to
minimize a per-pair score. The unweighted score is typically a sum of
per-pair distortion contributions; this helper rebalances those
contributions so that high-sensitivity pairs receive more optimization
budget, per the operating-point-aware axis weights from
``axis_weights_for_named_operating_point("pr106_r2_frontier")``.

Design memo: ``.omx/research/fec6_plus_sensitivity_weighted_discovery_design_20260517.md``

Per CLAUDE.md "Apples-to-apples evidence discipline": every score row that
flows through this helper carries an ``[axis_weights v1;
operating_point=...; basis=...]`` evidence tag emitted by
``AxisWeights.evidence_tag()`` so downstream consumers can trace the
operating point.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": this
module's predicted ΔS band is ``[0.0, -0.0005] [predicted, theoretical]``
on contest-CPU axis; the empirical effect is paired-axis dispatch
pending.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from tac.sensitivity_map.axis_weights import AxisWeights

__all__ = [
    "compute_weighted_per_pair_scores",
    "pick_weighted_per_pair_modes",
    "reweight_per_pair_candidate_table",
]


def compute_weighted_per_pair_scores(
    *,
    pair_id: int,
    candidate_modes: Sequence[Mapping[str, float]],
    axis_weights: AxisWeights,
) -> list[float]:
    """Compute the per-pair weighted score for each candidate palette mode.

    Each ``candidate_modes[k]`` is a mapping with keys ``"delta_d_pose"``
    and ``"delta_d_seg"`` (delta-distortion contributions if this mode is
    chosen for this pair). The weighted score is::

        score_k = axis_weights.pose * delta_d_pose_k + axis_weights.seg * delta_d_seg_k

    Returns a list of float scores, one per candidate mode. The optimal
    mode for the pair under this weighting is ``argmin(scores)``.

    The unweighted baseline (``axis_weights.pose = axis_weights.seg = 1.0``)
    recovers the canonical per-pair score sum. The PR106 r2 frontier
    weighting (``pose=2.71, seg=1.0``) rebalances toward pose-sensitive
    pairs.

    Per CLAUDE.md "SegNet vs PoseNet importance — operating-point
    dependent": the canonical operating point for fec6 / PR106 r2 frontier
    is documented in
    ``tac.sensitivity_map.axis_weights.PR106_R2_FRONTIER_AXIS_WEIGHTS``.

    Parameters
    ----------
    pair_id : int
        The per-pair identifier (used only for diagnostic error messages).
    candidate_modes : sequence of mapping
        Per-pair candidate palette modes with ``delta_d_pose`` and
        ``delta_d_seg`` keys.
    axis_weights : AxisWeights
        The operating-point-aware axis weights instance.

    Returns
    -------
    list[float]
        Per-mode weighted scores; argmin is the optimal mode.

    Raises
    ------
    KeyError
        If any candidate mode is missing either of the required keys.
    """
    if not candidate_modes:
        raise ValueError(
            f"pair_id={pair_id}: candidate_modes is empty; need at least one mode"
        )
    scores: list[float] = []
    for k, mode in enumerate(candidate_modes):
        try:
            delta_d_pose = float(mode["delta_d_pose"])
            delta_d_seg = float(mode["delta_d_seg"])
        except KeyError as exc:
            raise KeyError(
                f"pair_id={pair_id} candidate_modes[{k}] missing required key {exc!r}; "
                f"need both 'delta_d_pose' and 'delta_d_seg'"
            ) from exc
        if not math.isfinite(delta_d_pose):
            raise ValueError(
                f"pair_id={pair_id} candidate_modes[{k}] has non-finite "
                f"delta_d_pose={delta_d_pose!r}"
            )
        if not math.isfinite(delta_d_seg):
            raise ValueError(
                f"pair_id={pair_id} candidate_modes[{k}] has non-finite "
                f"delta_d_seg={delta_d_seg!r}"
            )
        score = axis_weights.pose * delta_d_pose + axis_weights.seg * delta_d_seg
        if not math.isfinite(score):
            raise ValueError(
                f"pair_id={pair_id} candidate_modes[{k}] produced non-finite "
                f"weighted_score={score!r}"
            )
        scores.append(score)
    return scores


def pick_weighted_per_pair_modes(
    *,
    per_pair_candidate_modes: Sequence[Sequence[Mapping[str, float]]],
    axis_weights: AxisWeights,
) -> list[int]:
    """Pick the optimal per-pair palette mode under the axis-weighted scoring.

    For each pair, computes the weighted score per candidate mode and
    returns ``argmin`` as the chosen mode index. The output is suitable
    for replacement of the per-pair palette indices in
    ``selector_policy_sample.json``.

    Parameters
    ----------
    per_pair_candidate_modes : sequence of sequence of mapping
        Outer length == number of pairs (typically 600). Inner length ==
        number of candidate palette modes per pair (typically K=16 for
        fec6 fixed-Huffman-k16).
    axis_weights : AxisWeights
        The operating-point-aware axis weights instance.

    Returns
    -------
    list[int]
        Per-pair chosen mode index. Length == number of pairs.
    """
    if not per_pair_candidate_modes:
        raise ValueError("per_pair_candidate_modes is empty; need at least one pair")
    chosen: list[int] = []
    for pair_id, candidate_modes in enumerate(per_pair_candidate_modes):
        scores = compute_weighted_per_pair_scores(
            pair_id=pair_id,
            candidate_modes=candidate_modes,
            axis_weights=axis_weights,
        )
        # argmin
        best_k = 0
        best_score = scores[0]
        for k, score in enumerate(scores[1:], start=1):
            if score < best_score:
                best_k = k
                best_score = score
        chosen.append(best_k)
    return chosen


def reweight_per_pair_candidate_table(
    *,
    per_pair_candidate_modes: Sequence[Sequence[Mapping[str, float]]],
    axis_weights: AxisWeights,
) -> dict[str, object]:
    """Reweight a per-pair candidate table and emit a result manifest.

    This is the canonical entry-point consumed by
    ``tools/reweight_fec6_selector_discovery.py`` to convert a
    sensitivity-unaware ``selector_policy_sample.json`` into a
    sensitivity-weighted variant.

    Returns a dict with:

    - ``chosen_per_pair_indices`` (list[int]): the weighted per-pair mode picks
    - ``per_pair_weighted_scores`` (list[list[float]]): per-pair score
      per candidate mode (for observability per Catalog #305)
    - ``axis_weights_evidence_tag`` (str): the canonical evidence tag
      from ``AxisWeights.evidence_tag()``
    - ``per_pair_unweighted_indices`` (list[int]): the unweighted argmin
      per pair (for diff observability)
    - ``per_pair_divergence_count`` (int): number of pairs where the
      weighted and unweighted picks differ

    Per CLAUDE.md "Apples-to-apples evidence discipline" the
    ``axis_weights_evidence_tag`` MUST be propagated into any downstream
    artifact that consumes this result.
    """
    per_pair_weighted_scores: list[list[float]] = []
    per_pair_unweighted_indices: list[int] = []
    chosen_per_pair_indices: list[int] = []
    divergence_count = 0

    # Unweighted baseline = uniform per-axis weight; we reuse the same
    # helper with a baseline AxisWeights to get the canonical unweighted
    # argmin (rather than re-implementing).
    baseline_axis_weights = AxisWeights(
        pose=1.0,
        seg=1.0,
        rate=1.0,
        mixed=1.0,
        operating_point_tag="baseline_unweighted",
        basis="uniform 1.0 per axis; diagnostic-only",
    )

    for pair_id, candidate_modes in enumerate(per_pair_candidate_modes):
        weighted_scores = compute_weighted_per_pair_scores(
            pair_id=pair_id,
            candidate_modes=candidate_modes,
            axis_weights=axis_weights,
        )
        unweighted_scores = compute_weighted_per_pair_scores(
            pair_id=pair_id,
            candidate_modes=candidate_modes,
            axis_weights=baseline_axis_weights,
        )

        # argmin for each
        weighted_best = 0
        weighted_min = weighted_scores[0]
        for k, score in enumerate(weighted_scores[1:], start=1):
            if score < weighted_min:
                weighted_best = k
                weighted_min = score

        unweighted_best = 0
        unweighted_min = unweighted_scores[0]
        for k, score in enumerate(unweighted_scores[1:], start=1):
            if score < unweighted_min:
                unweighted_best = k
                unweighted_min = score

        per_pair_weighted_scores.append(weighted_scores)
        per_pair_unweighted_indices.append(unweighted_best)
        chosen_per_pair_indices.append(weighted_best)
        if weighted_best != unweighted_best:
            divergence_count += 1

    return {
        "chosen_per_pair_indices": chosen_per_pair_indices,
        "per_pair_weighted_scores": per_pair_weighted_scores,
        "axis_weights_evidence_tag": axis_weights.evidence_tag(),
        "per_pair_unweighted_indices": per_pair_unweighted_indices,
        "per_pair_divergence_count": divergence_count,
        "n_pairs": len(per_pair_candidate_modes),
        "score_claim": False,  # diagnostic-only; per Catalog #287 evidence-tag discipline
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_provider_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "evidence_tag": "[predicted, theoretical] sensitivity-weighted fec6 selector discovery",
    }
