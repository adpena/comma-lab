"""Evaluator-action waterfilling law — every interpreter atom must pay rent.

THE BUG THIS EXTINCTS (2026-06-08 sidecar incident): a target-region action that
PARSES + APPLIES is NOT a valid archive word.  It is valid only if it improves the
EXACT contest score relative to the current base archive:

    S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes / 37_545_489
    admit action sigma against base P  iff  S(P + sigma) < S(P)

The shipped sidecar was admitted on the WRONG predicate ("it rescues a collapsed
backend") against a PHANTOM base (the backend was believed dead at ~2 wins).  The
true backend survived (~11306 wins), so the sidecar rewrote pixels the backend
already won, collapsing the result to ~3 wins AND adding ~8 KB — a negative action
on BOTH the distortion and rate terms.  A rate-distortion admission law would have
rejected it automatically.

This module is the canonical, reusable law every atom family plugs into — target
region sidecars, semantic/pose primitives, backend precision deltas, SNeRV
source-state actions, codec choices.  One currency: exact contest ΔS.

ANTI-DRIFT INVARIANT: an action evaluation is bound to the base it was measured
against (``base_archive_sha256`` + optional ``base_scorer_state_hash``).  When the
base changes, the evaluation is STALE (``is_stale_for_base``) and must be
re-measured — a candidate computed against a phantom/old base must never be
admitted (the exact failure mode of the original sidecar).

NONLINEAR + NONCOMMUTATIVE: SegNet/PoseNet are nonlocal + discontinuous, so atom
effects do not add.  ``action_commutator`` measures interaction; the waterfill
selector recomputes after each accepted atom (caller loop) rather than assuming
separability — raw per-pixel waterfilling is wrong because one pixel overwrite has
a receptive-field footprint (2286 px -> ~11319 argmax flips in the incident).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tac.optimization.bayesian_experimental_design import contest_score

CONTEST_ARCHIVE_RATE_DENOM = 37_545_489
_SEG = 100.0
_POSE_INNER = 10.0
_RATE = 25.0


def _nonrate(d_seg: float, d_pose: float) -> float:
    return _SEG * float(d_seg) + math.sqrt(_POSE_INNER * float(d_pose))


@dataclass(frozen=True)
class CandidateActionEvaluation:
    """Exact rate-distortion evaluation of one interpreter atom against a base.

    All score terms are EXACT contest units (``contest_score``).  ``pays_rent`` is
    the admission law: the action lowers the exact score AND its scorer effect
    survives parse-back.  Planning-control evidence; promotion still requires the
    paired upstream eval providing the d_seg/d_pose terms on contest hardware.
    """

    action_id: str
    action_kind: str
    base_archive_sha256: str
    with_action_archive_sha256: str
    d_seg_base: float
    d_pose_base: float
    bytes_base: int
    d_seg_with_action: float
    d_pose_with_action: float
    bytes_with_action: int
    scorer_effect_survived: bool = True
    base_scorer_state_hash: str | None = None
    backend_wrong_to_target: int | None = None
    with_action_wrong_to_target: int | None = None
    evidence_grade: str = "advisory"

    def __post_init__(self) -> None:
        for name in ("d_seg_base", "d_pose_base", "d_seg_with_action", "d_pose_with_action"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative; got {getattr(self, name)!r}")
        if int(self.bytes_base) < 0 or int(self.bytes_with_action) < 0:
            raise ValueError("archive bytes must be non-negative")

    # --- exact score terms ---
    @property
    def score_base(self) -> float:
        return contest_score(self.d_seg_base, self.d_pose_base, self.bytes_base)

    @property
    def score_with_action(self) -> float:
        return contest_score(self.d_seg_with_action, self.d_pose_with_action, self.bytes_with_action)

    @property
    def delta_score_total(self) -> float:
        return self.score_with_action - self.score_base

    @property
    def delta_score_nonrate(self) -> float:
        return _nonrate(self.d_seg_with_action, self.d_pose_with_action) - _nonrate(
            self.d_seg_base, self.d_pose_base
        )

    @property
    def delta_bytes(self) -> int:
        return int(self.bytes_with_action) - int(self.bytes_base)

    @property
    def delta_rate_score(self) -> float:
        return _RATE * float(self.delta_bytes) / float(CONTEST_ARCHIVE_RATE_DENOM)

    @property
    def value_per_byte(self) -> float | None:
        """Exact score reduction per added byte (the waterfill ranking key).

        ``-delta_score_total / delta_bytes`` when the action adds bytes; for a
        byte-NEGATIVE action (frees bytes) any score reduction is unconditionally
        admissible, so value-per-byte is +inf (sorted first)."""
        if self.delta_bytes <= 0:
            return math.inf if self.delta_score_total < 0 else None
        return -self.delta_score_total / float(self.delta_bytes)

    @property
    def pays_rent(self) -> bool:
        """Admission law: lowers exact score AND scorer effect survives parse-back."""
        return self.delta_score_total < 0.0 and bool(self.scorer_effect_survived)

    def is_stale_for_base(
        self, current_base_archive_sha256: str, current_base_scorer_state_hash: str | None = None
    ) -> bool:
        """ANTI-DRIFT: True when this evaluation was computed against a DIFFERENT
        base than the current one — it must be re-measured, never admitted (the
        phantom-base failure mode of the original sidecar)."""
        if str(self.base_archive_sha256) != str(current_base_archive_sha256):
            return True
        return (
            current_base_scorer_state_hash is not None
            and self.base_scorer_state_hash is not None
            and str(self.base_scorer_state_hash) != str(current_base_scorer_state_hash)
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "schema": "hi_nerv_candidate_action_evaluation.v1",
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "base_archive_sha256": self.base_archive_sha256,
            "with_action_archive_sha256": self.with_action_archive_sha256,
            "base_scorer_state_hash": self.base_scorer_state_hash,
            "d_seg_base": self.d_seg_base,
            "d_pose_base": self.d_pose_base,
            "bytes_base": int(self.bytes_base),
            "d_seg_with_action": self.d_seg_with_action,
            "d_pose_with_action": self.d_pose_with_action,
            "bytes_with_action": int(self.bytes_with_action),
            "score_base": self.score_base,
            "score_with_action": self.score_with_action,
            "delta_score_total": self.delta_score_total,
            "delta_score_nonrate": self.delta_score_nonrate,
            "delta_bytes": self.delta_bytes,
            "delta_rate_score": self.delta_rate_score,
            "value_per_byte": self.value_per_byte,
            "scorer_effect_survived": bool(self.scorer_effect_survived),
            "backend_wrong_to_target": self.backend_wrong_to_target,
            "with_action_wrong_to_target": self.with_action_wrong_to_target,
            "pays_rent": self.pays_rent,
            "verdict": "admit" if self.pays_rent else "reject",
            "evidence_grade": self.evidence_grade,
            "authority": "planning_control_false_authority",
            "score_claim": False,
            "promotion_eligible": False,
            "promotable": False,
        }


def action_commutator(
    delta_score_a: float, delta_score_b: float, delta_score_ab: float
) -> float:
    """Interaction term ``ΔS(a∘b) - ΔS(a) - ΔS(b)``.

    < 0 => synergistic (the pair lowers score MORE than the sum of singles);
    > 0 => interfering (the atoms partially cancel / harm each other);
    ~ 0 => approximately additive.  Because SegNet/PoseNet are nonlinear, this is
    measured, never assumed."""
    return float(delta_score_ab) - float(delta_score_a) - float(delta_score_b)


def waterfill_select_actions(
    evaluations: list[CandidateActionEvaluation],
    *,
    current_base_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Static waterfill pass: drop stale + non-rent-paying atoms, rank survivors by
    exact value-per-byte (most score-reduction-per-byte first).

    This is ONE pass; because atom effects are noncommutative the caller MUST
    re-measure-and-re-select after accepting each atom (accepted atom changes the
    base -> remaining evaluations become stale).  Returns the ranked admissible
    set + the rejected/stale sets so the decision is auditable.
    """

    admissible: list[CandidateActionEvaluation] = []
    rejected: list[CandidateActionEvaluation] = []
    stale: list[CandidateActionEvaluation] = []
    for ev in evaluations:
        if current_base_archive_sha256 is not None and ev.is_stale_for_base(
            current_base_archive_sha256
        ):
            stale.append(ev)
            continue
        (admissible if ev.pays_rent else rejected).append(ev)

    def _key(ev: CandidateActionEvaluation) -> float:
        vpb = ev.value_per_byte
        return -(vpb if vpb is not None and vpb != math.inf else 1e18)

    admissible.sort(key=_key)
    best = admissible[0] if admissible else None
    return {
        "schema": "hi_nerv_evaluator_action_waterfill_pass.v1",
        "admissible_ranked": [ev.to_row() for ev in admissible],
        "rejected": [ev.to_row() for ev in rejected],
        "stale_for_base": [ev.to_row() for ev in stale],
        "best_action_id": best.action_id if best is not None else None,
        "best_delta_score_total": best.delta_score_total if best is not None else None,
        "n_admissible": len(admissible),
        "n_rejected": len(rejected),
        "n_stale": len(stale),
        "requires_recompute_after_accept": True,
        "authority": "planning_control_false_authority",
        "promotable": False,
    }
