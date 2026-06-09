"""Vehicle-AGNOSTIC harvest → typed evidence → machine-readable decision.

THE SELF-DRIVING LOOP CORE, for ALL vehicles (operator 2026-06-09: "is this going to
work with all vehicles?" — yes, by design).  The evaluator-action waterfiller is one
currency for every backend (V1 HiNeRV, V2 SNeRV, V3 PACT-NeRV / atoms): a candidate is
just "an archive.zip -> exact d_seg/d_pose/bytes -> a CandidateActionEvaluation judged by
exact ΔS."  So the harvest loop is ONE primitive; vehicles differ only in (a) their
checkpoint->archive EXPORT adapter and (b) their VEHICLE-SPECIFIC hard-fail predicates.

  loop (any vehicle):
    checkpoint/source-state
      -> [vehicle export adapter] -> archive.zip
      -> inflate.sh -> evaluate.py -> exact d_seg/d_pose/bytes      (B2 bridge, shared)
      -> build_harvest_receipt(...)            # mechanical proof   (shared)
      -> build_candidate_action_evaluation_row # waterfiller action (shared)
      -> apply_campaign_decision(hard_fail_checks=<vehicle's>)      # encoded policy (shared core)
      -> CONTINUE / STOP / BRANCH / PREP_CUDA / HARD_FAIL / INSPECT

Per-vehicle plug-in (only the hard_fail_checks differ):
  - V1 HiNeRV B1:  ("muon_active_before_stage_8", bool), ("sidecar_exported", bool)
  - V2 SNeRV:      ("hf_restorer_diverged", bool), ("source_forward_unproven", bool)
  - V3 atom:       ("scorer_effect_not_survived", bool), ("stale_for_base", bool)

All rows are advisory (``[macOS-CPU advisory]`` / ``promotion_eligible=false``) until the
authoritative Linux-x86_64-CPU + T4-CUDA B2 lands.  ep-N is a binding-constraint PROBE,
never an auto-kill (CLAUDE.md "Forbidden premature KILL").
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from tac.optimization.evaluator_action_waterfill import (
    CONTEST_ARCHIVE_RATE_DENOM,
    contest_score,
)

# Vehicle labels (the witness backends + the atom family; all judged by one ΔS currency).
VEHICLE_HI_NERV = "hi_nerv"
VEHICLE_SNERV = "snerv"
VEHICLE_PACT_NERV = "pact_nerv"
VEHICLE_ATOM = "atom"
_VEHICLES = (VEHICLE_HI_NERV, VEHICLE_SNERV, VEHICLE_PACT_NERV, VEHICLE_ATOM)

_SEG_COEF = 100.0
_POSE_INNER = 10.0
_RATE_COEF = 25.0

# Decision sentinels (machine-readable; the encoded campaign policy — same for all vehicles).
HARD_FAIL = "HARD_FAIL"
PATCH_BRIDGE = "PATCH_BRIDGE"
CONTINUE_PREP_CUDA = "CONTINUE_PREP_CUDA"
CONTINUE_TRAINING = "CONTINUE_TRAINING"
INSPECT_BINDING_CONSTRAINT = "INSPECT_BINDING_CONSTRAINT"

_PROMISING_MARGIN = 0.0001  # score <= frontier - this => beats frontier, prep CUDA.
_DIRECTIONAL_MARGIN = 0.001  # score <= baseline - this => directionally promising.


def _check_vehicle(vehicle: str) -> str:
    if vehicle not in _VEHICLES:
        raise ValueError(f"vehicle must be one of {_VEHICLES}; got {vehicle!r}")
    return vehicle


def build_harvest_receipt(
    *,
    vehicle: str,
    run_dir: str,
    checkpoint_label: str,
    checkpoint_sha256: str,
    export_status: str,
    archive_path: str | None,
    archive_sha256: str | None,
    archive_bytes: int | None,
    sidecar_exported: bool,
    pay_rent_gate_active: bool,
    inflate_status: str,
    evaluate_status: str,
    result_json_path: str | None,
) -> dict[str, Any]:
    """``checkpoint_harvest_receipt.v1`` — the mechanical-pipeline proof (any vehicle).

    ``checkpoint_label`` is vehicle-flexible (e.g. ``"ep250"`` for a NeRV checkpoint or a
    SNeRV source-state tag).  ``idempotency_key`` makes a re-harvest a provable no-op.
    """

    _check_vehicle(vehicle)
    idempotency_key = "+".join(
        str(x)
        for x in (vehicle, run_dir, checkpoint_sha256, archive_sha256 or "no_archive")
    )
    return {
        "schema": "checkpoint_harvest_receipt.v1",
        "vehicle": vehicle,
        "run_dir": run_dir,
        "checkpoint_label": checkpoint_label,
        "checkpoint_sha256": checkpoint_sha256,
        "export_status": export_status,
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "sidecar_exported": bool(sidecar_exported),
        "pay_rent_gate_active": bool(pay_rent_gate_active),
        "inflate_status": inflate_status,
        "evaluate_status": evaluate_status,
        "result_json_path": result_json_path,
        "idempotency_key": idempotency_key,
        "authority": "macOS-CPU advisory",
        "promotion_eligible": False,
        "score_claim": False,
    }


def build_candidate_action_evaluation_row(
    *,
    vehicle: str,
    candidate_kind: str,
    candidate_archive_sha256: str,
    candidate_d_seg: float,
    candidate_d_pose: float,
    candidate_bytes: int,
    base_archive_sha256: str,
    base_score: float,
    checkpoint_label: str | None = None,
    axis_tag: str = "[macOS-CPU advisory]",
) -> dict[str, Any]:
    """``candidate_action_evaluation.v1`` — a vehicle's candidate archive as a waterfiller
    action.  Score-based delta vs the frontier (the frontier carries only its score);
    ``pays_rent`` iff the candidate BEATS the frontier on the EXACT contest score.  Works
    identically for a HiNeRV checkpoint, an SNeRV source-state archive, or an atom."""

    _check_vehicle(vehicle)
    candidate_score = contest_score(
        float(candidate_d_seg), float(candidate_d_pose), int(candidate_bytes)
    )
    delta_score_total = candidate_score - float(base_score)
    return {
        "schema": "candidate_action_evaluation.v1",
        "vehicle": vehicle,
        "candidate_kind": candidate_kind,
        "checkpoint_label": checkpoint_label,
        "base_archive_sha256": base_archive_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "candidate_d_seg": float(candidate_d_seg),
        "candidate_d_pose": float(candidate_d_pose),
        "candidate_bytes": int(candidate_bytes),
        "candidate_score": candidate_score,
        "base_score": float(base_score),
        "delta_score_total": delta_score_total,
        "pays_rent": delta_score_total < 0.0,
        "verdict": "beats_frontier" if delta_score_total < 0.0 else "above_frontier",
        "score_formula": "100*d_seg + sqrt(10*d_pose) + 25*bytes/37545489",
        "authority": axis_tag,
        "promotion_eligible": False,
        "score_claim": False,
        "promotable": False,
    }


def binding_constraint(
    d_seg: float, d_pose: float, archive_bytes: int
) -> dict[str, Any]:
    """Which exact-score term dominates: 'rate vs pose vs seg?' for ANY vehicle's archive."""

    seg_term = _SEG_COEF * float(d_seg)
    pose_term = math.sqrt(_POSE_INNER * float(d_pose))
    rate_term = _RATE_COEF * float(archive_bytes) / float(CONTEST_ARCHIVE_RATE_DENOM)
    total = seg_term + pose_term + rate_term
    terms = {"seg": seg_term, "pose": pose_term, "rate": rate_term}
    binding = max(terms, key=lambda k: terms[k])
    shares = {k: (v / total if total > 0 else 0.0) for k, v in terms.items()}
    return {
        "binding_constraint": binding,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "total": total,
        "shares": shares,
    }


def apply_campaign_decision(
    *,
    receipt: dict[str, Any],
    candidate_eval: dict[str, Any],
    frontier_score: float,
    hard_fail_checks: Sequence[tuple[str, bool]] = (),
    baseline_score: float | None = None,
) -> dict[str, Any]:
    """The ENCODED campaign policy (machine-readable) — SHARED across vehicles.

    ``hard_fail_checks`` are the VEHICLE-SPECIFIC invariants (name, is_violated): e.g.
    HiNeRV passes ``("muon_active_before_stage_8", bool)``; SNeRV passes
    ``("hf_restorer_diverged", bool)``; an atom passes ``("scorer_effect_not_survived",
    bool)``.  The falling-rule core (vehicle hard-fails -> bridge failure -> score ladder)
    is identical for all.  A high score is never an auto-kill — it routes to INSPECT with
    the binding constraint named (Forbidden premature KILL)."""

    # --- vehicle-specific hard-fail invariants ---
    for name, violated in hard_fail_checks:
        if bool(violated):
            return _decision(HARD_FAIL, str(name))
    # --- universal structural hard-fails ---
    if bool(receipt.get("sidecar_exported")):
        return _decision(HARD_FAIL, "sidecar_exported_without_rent")
    if not bool(receipt.get("pay_rent_gate_active", True)):
        return _decision(HARD_FAIL, "pay_rent_gate_inactive")
    # --- bridge failure -> patch the export/inflate/eval bridge, keep training if healthy ---
    if receipt.get("export_status") != "passed":
        return _decision(PATCH_BRIDGE, "export_from_checkpoint_failed")
    if receipt.get("inflate_status") != "passed":
        return _decision(PATCH_BRIDGE, "inflate_sh_failed")
    if receipt.get("evaluate_status") != "passed":
        return _decision(PATCH_BRIDGE, "exact_eval_bridge_failed")
    # --- score ladder ---
    candidate_score = float(candidate_eval.get("candidate_score", math.inf))
    bc = binding_constraint(
        float(candidate_eval.get("candidate_d_seg", 0.0)),
        float(candidate_eval.get("candidate_d_pose", 0.0)),
        int(candidate_eval.get("candidate_bytes", 0)),
    )
    if candidate_score <= float(frontier_score) - _PROMISING_MARGIN:
        return _decision(
            CONTINUE_PREP_CUDA, "cpu_advisory_beats_frontier_prep_authoritative", bc
        )
    ref = float(baseline_score) if baseline_score is not None else float(frontier_score)
    if candidate_score <= ref - _DIRECTIONAL_MARGIN:
        return _decision(CONTINUE_TRAINING, "directionally_promising", bc)
    return _decision(
        INSPECT_BINDING_CONSTRAINT,
        f"early_high_score_binding_is_{bc['binding_constraint']}",
        bc,
    )


def _decision(
    decision: str, reason: str, binding: dict[str, Any] | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": "campaign_decision.v1",
        "decision": decision,
        "reason": reason,
        "authority": "macOS-CPU advisory",
        "promotion_eligible": False,
        "auto_kill": False,  # never auto-kill (Forbidden premature KILL).
    }
    if binding is not None:
        row["binding_constraint"] = binding
    return row
