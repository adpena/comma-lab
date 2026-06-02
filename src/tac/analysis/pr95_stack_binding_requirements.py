# SPDX-License-Identifier: MIT
"""PR95-grade stack binding requirements for NeRV-family candidates.

PR95's useful lesson is not "use HNeRV"; it is the binding depth: scorer-in-loop
training, eval-roundtrip simulation, QAT/coder pressure, archive-in-loop byte
feedback, and receiver/eval gates all active on the same candidate. This helper
turns that lesson into reusable planner state for HiNeRV/SNeRV without granting
score authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SCHEMA = "pr95_stack_binding_requirements.v1"

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


@dataclass(frozen=True)
class StackRequirement:
    """One PR95-grade binding requirement.

    ``evidence_key`` is intentionally a simple boolean-ish input. Callers own
    the substrate-specific proof; this module owns the cross-carrier contract.
    """

    requirement_id: str
    evidence_key: str
    title: str
    pr95_source: str
    rationale: str


REQUIREMENTS: tuple[StackRequirement, ...] = (
    StackRequirement(
        "carrier_source_or_documented_adaptation",
        "carrier_source_or_documented_adaptation",
        "carrier is source-faithful or adaptations are documented",
        "PR95 used an HNeRV-style custom contest decoder, not blind OSS vendoring",
        "OSS controls are useful only when the contest adaptation is explicit.",
    ),
    StackRequirement(
        "modelsize_archive_budget",
        "modelsize_archive_budget",
        "modelsize/capacity candidate is bound to an archive byte ceiling",
        "PR95 fixed a compact 229K-param decoder plus 28-d pair latents",
        "Capacity must be chosen by charged bytes, not arbitrary channel counts.",
    ),
    StackRequirement(
        "pr95_staged_curriculum",
        "pr95_staged_curriculum",
        "candidate uses a staged PR95-style curriculum",
        "PR95 ran 8 stages totaling 29,650 default epochs",
        "Carrier fit, QAT, coder pressure, and optimizer polish are coupled.",
    ),
    StackRequirement(
        "real_segnet_teacher",
        "real_segnet_teacher",
        "real SegNet teacher is active",
        "PR95 trained against frozen challenge SegNet targets",
        "SegNet boundary behavior is the dominant scorer-facing shape signal.",
    ),
    StackRequirement(
        "real_posenet_teacher",
        "real_posenet_teacher",
        "real PoseNet teacher is active",
        "PR95 trained against frozen challenge PoseNet targets",
        "PoseNet is pair-coupled and can dominate near low pose distortion.",
    ),
    StackRequirement(
        "differentiable_pose_preprocess",
        "differentiable_pose_preprocess",
        "PoseNet preprocessing is differentiable through YUV6",
        "PR95 patched upstream rgb_to_yuv6 so pose gradients reached the decoder",
        "A scorer loop with severed pose gradients is a false fit signal.",
    ),
    StackRequirement(
        "eval_roundtrip_ste",
        "eval_roundtrip_ste",
        "uint8/eval roundtrip is simulated with STE",
        "PR95 trained through 384->874->384 plus straight-through rounding",
        "Proxy loss must see the same quantized surface the scorer consumes.",
    ),
    StackRequirement(
        "ema_archive_selection",
        "ema_archive_selection",
        "EMA weights are evaluated and selected by archive score",
        "PR95 saved best EMA archive every eval interval",
        "Training loss alone is not archive promotion evidence.",
    ),
    StackRequirement(
        "qat_forward",
        "qat_forward",
        "QAT/fake-quant forward path is active",
        "PR95 applied per-tensor INT8 fake quant from stage 4 onward",
        "The fitted weights must learn the quantized receiver surface.",
    ),
    StackRequirement(
        "coder_aware_regularizer",
        "coder_aware_regularizer",
        "coder-aware entropy/MDL regularizer is active",
        "PR95 C1a shaped post-INT8 weight histograms for brotli",
        "Rate pressure belongs inside training, not only after export.",
    ),
    StackRequirement(
        "muon_adamw_partition",
        "muon_adamw_partition",
        "Muon/AdamW partition is available for final-stage polish",
        "PR95 used Muon on hidden conv weights and AdamW on stem/heads/latents",
        "The optimizer split is part of the proven binding stack.",
    ),
    StackRequirement(
        "archive_in_loop_byte_oracle",
        "archive_in_loop_byte_oracle",
        "packed archive byte oracle feeds the planner",
        "PR95 built and parsed an archive every eval interval",
        "Nominal payload estimates must be corrected by measured packet bytes.",
    ),
    StackRequirement(
        "byte_closed_archive_export",
        "byte_closed_archive_export",
        "candidate exports a byte-closed archive packet",
        "PR95 emitted the final archive through its codec stage",
        "No local tensor or advisory packet can promote without archive bytes.",
    ),
    StackRequirement(
        "receiver_proof",
        "receiver_proof",
        "receiver consumes the exported archive under inflate contract",
        "PR95 shipped inflate.py/inflate.sh with the archive grammar",
        "Decoded-frame authority requires runtime consumption proof.",
    ),
    StackRequirement(
        "full_video_local_prefilter",
        "full_video_local_prefilter",
        "full-video local scorer prefilter is attached",
        "PR95 evaluated all pairs repeatedly during training",
        "Partial-pair smokes are useful but not promotion evidence.",
    ),
    StackRequirement(
        "local_cpu_replay_gate",
        "local_cpu_replay_gate",
        "local CPU replay gate is present before exact auth",
        "PR95 archive score was recomputed through parsed runtime artifacts",
        "MLX advisory candidates need CPU replay before exact dispatch.",
    ),
    StackRequirement(
        "exact_auth_gate_plan",
        "exact_auth_gate_plan",
        "exact CPU/CUDA auth gate is planned but not bypassed",
        "Contest axes are separate; PR95 public anchors were axis-tagged",
        "Promotion must remain fail-closed until true local winners clear gates.",
    ),
)


def build_pr95_stack_binding_requirements(
    *,
    family: str,
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a fail-closed PR95-grade binding audit for one candidate family."""

    family_token = str(family or "unknown").strip() or "unknown"
    ev = dict(evidence or {})
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    satisfied = 0
    for requirement in REQUIREMENTS:
        present = bool(ev.get(requirement.evidence_key))
        if present:
            satisfied += 1
        blocker = (
            None
            if present
            else f"{family_token}_{requirement.requirement_id}_missing"
        )
        if blocker is not None:
            blockers.append(blocker)
        rows.append(
            {
                "schema": "pr95_stack_binding_requirement_row.v1",
                "requirement_id": requirement.requirement_id,
                "title": requirement.title,
                "evidence_key": requirement.evidence_key,
                "satisfied": present,
                "status": "satisfied" if present else "missing",
                "blocker": blocker,
                "evidence_value": ev.get(requirement.evidence_key),
                "pr95_source": requirement.pr95_source,
                "rationale": requirement.rationale,
            }
        )
    required = len(REQUIREMENTS)
    complete = satisfied == required
    return {
        "schema": SCHEMA,
        "family": family_token,
        "required_count": required,
        "satisfied_count": satisfied,
        "missing_count": required - satisfied,
        "complete": complete,
        "rows": rows,
        "blockers": blockers,
        "policy": {
            "pr95_is_control_arm_not_design_ceiling": True,
            "oss_controls_must_be_bound_to_contest_stack": True,
            "exact_auth_forbidden_until_complete_and_local_winner": True,
        },
        **FALSE_AUTHORITY,
    }


def build_pr95_stack_binding_evidence(
    *,
    carrier_source_or_documented_adaptation: bool = True,
    modelsize_archive_budget: bool = False,
    pr95_staged_curriculum: bool = False,
    real_segnet_teacher: bool = False,
    real_posenet_teacher: bool = False,
    differentiable_pose_preprocess: bool = False,
    eval_roundtrip_ste: bool = False,
    ema_archive_selection: bool = False,
    qat_forward: bool = False,
    coder_aware_regularizer: bool = False,
    muon_adamw_partition: bool = False,
    archive_in_loop_byte_oracle: bool = False,
    byte_closed_archive_export: bool = False,
    receiver_proof: bool = False,
    full_video_local_prefilter: bool = False,
    local_cpu_replay_gate: bool = False,
    exact_auth_gate_plan: bool = True,
) -> dict[str, bool]:
    """Build a complete evidence map with explicit defaults for every key."""

    return {
        "carrier_source_or_documented_adaptation": bool(
            carrier_source_or_documented_adaptation
        ),
        "modelsize_archive_budget": bool(modelsize_archive_budget),
        "pr95_staged_curriculum": bool(pr95_staged_curriculum),
        "real_segnet_teacher": bool(real_segnet_teacher),
        "real_posenet_teacher": bool(real_posenet_teacher),
        "differentiable_pose_preprocess": bool(differentiable_pose_preprocess),
        "eval_roundtrip_ste": bool(eval_roundtrip_ste),
        "ema_archive_selection": bool(ema_archive_selection),
        "qat_forward": bool(qat_forward),
        "coder_aware_regularizer": bool(coder_aware_regularizer),
        "muon_adamw_partition": bool(muon_adamw_partition),
        "archive_in_loop_byte_oracle": bool(archive_in_loop_byte_oracle),
        "byte_closed_archive_export": bool(byte_closed_archive_export),
        "receiver_proof": bool(receiver_proof),
        "full_video_local_prefilter": bool(full_video_local_prefilter),
        "local_cpu_replay_gate": bool(local_cpu_replay_gate),
        "exact_auth_gate_plan": bool(exact_auth_gate_plan),
    }


__all__ = [
    "FALSE_AUTHORITY",
    "REQUIREMENTS",
    "SCHEMA",
    "StackRequirement",
    "build_pr95_stack_binding_evidence",
    "build_pr95_stack_binding_requirements",
]
