#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan C067/apogee multi-resolution stack candidates.

This planner is bounded and planning-only.  It inspects existing C067 fixed-
slice, CMG/PMG/downsample, high-resolution repair, and packer/profile artifacts,
then emits deterministic JSON for multi-pass stack policies:

pass 0 anchor, pass 1 coarse/global representation, pass 2 high-resolution
hard-pair/foveal/boundary repair, pass 3 entropy/packer/self-compression, and
optional pass 4 pose/runtime co-adaptation.

No scorer is loaded, no archive is written, no GPU/remote job is dispatched, and
no score claim is made.  Additive deltas remain predictions until the full
stacked archive bytes pass exact CUDA auth eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_c067_multiresolution_stack_candidates.py"
SCHEMA = "c067_multiresolution_stack_planner_v1"
EVIDENCE_GRADE = "planning_only"
ORIGINAL_VIDEO_BYTES = 37_545_489
LAMBDA_RATE = 25.0 / ORIGINAL_VIDEO_BYTES
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
ADDITIVE_DELTA_WARNING = (
    "Component and pass deltas are first-order predictions only. They must not "
    "promote, rank, or support paper claims until the complete stacked archive "
    "itself has exact CUDA auth eval on the exact archive bytes."
)

DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/results/c067_multiresolution_stack_planner_20260502/"
    "c067_multiresolution_stack_plan.json"
)

DEFAULT_ARTIFACTS: tuple[tuple[str, str], ...] = (
    (
        "pass0_c067_fixedslice_anchor_exact",
        "experiments/results/lightning_batch/"
        "exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/"
        "contest_auth_eval.adjudicated.json",
    ),
    (
        "pass1_cmg2_downsample2x2",
        "experiments/results/c067_cmg2_downsample2x2_candidate_20260502T1010Z/"
        "build_manifest.json",
    ),
    (
        "pass1_cmg3_nonzero_top2",
        "experiments/results/c067_cmg3_nonzero_runs_top2_candidate_20260502T105724Z/"
        "build_manifest.json",
    ),
    (
        "pass1_pmg_hotspot_lzma",
        "experiments/results/pmg_hotspot_candidate_c067_20260502_lzma/build_manifest.json",
    ),
    (
        "pass1_pmg_hotspot_exact_negative",
        "experiments/results/lightning_batch/exact_eval_pmg_hotspot_c067_t4_20260502T1402Z/"
        "contest_auth_eval.json",
    ),
    (
        "pass1_cmg3_nonzero_top2_exact_negative",
        "experiments/results/lightning_batch/exact_eval_c067_cmg3_nonzero_top2_t4_20260502T1100Z/"
        "contest_auth_eval.json",
    ),
    (
        "pass2_cmg2_foveated_top256_exact",
        "experiments/results/lightning_batch/"
        "exact_eval_c067_cmg2_foveated_top256_t4_20260502T1007Z/"
        "contest_auth_eval.json",
    ),
    (
        "pass2_hotspot_geometry_plan",
        "experiments/results/c067_hotspot_mask_geometry_compiler_20260502/"
        "c067_hotspot_mask_geometry_plan.json",
    ),
    (
        "pass2_multimask_reconciliation_plan",
        "experiments/results/c067_multimask_reconciliation_20260502/"
        "multimask_reconciliation_plan.json",
    ),
    (
        "pass3_archive_bit_budget_profile",
        "experiments/results/archive_bit_budget_profile_c067_pr67_mix_20260502T1548Z/"
        "archive_bit_budget_profile.json",
    ),
    (
        "pass4_ego_motion_field_plan",
        "experiments/results/c067_ego_motion_field_atoms_20260502/ego_motion_field_plan.json",
    ),
)


class PlannerError(ValueError):
    """Raised when planner inputs violate the planning-only contract."""


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    label: str
    path: Path
    sha256: str
    schema: str | None
    pass_index: int
    pass_name: str
    resolution_layer: str
    component_type: str
    logical_members: tuple[str, ...]
    evidence_grade: str
    score_claim: bool
    promotion_eligible: bool
    archive_bytes: int | None
    archive_sha256: str | None
    payload_bytes: int | None
    delta_bytes_vs_anchor: int | None
    observed_score: float | None
    observed_pose: float | None
    observed_seg: float | None
    disagreement_fraction: float | None
    exact_negative: bool
    source_builder: str | None
    builder_consumable: bool
    notes: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PlannerError(f"{path} must contain a JSON object")
    return payload


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _archive_bytes(payload: dict[str, Any]) -> tuple[int | None, str | None]:
    for key in ("output_archive", "archive", "frontier_archive"):
        record = payload.get(key)
        if isinstance(record, dict):
            bytes_value = _int_or_none(record.get("bytes"))
            sha = record.get("sha256")
            if bytes_value is not None or isinstance(sha, str):
                return bytes_value, sha if isinstance(sha, str) else None
    bytes_value = _int_or_none(payload.get("archive_size_bytes"))
    sha = payload.get("archive_sha256")
    return bytes_value, sha if isinstance(sha, str) else None


def _score_triplet(payload: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    score = _finite_or_none(
        payload.get("score_recomputed_from_components", payload.get("final_score"))
    )
    pose = _finite_or_none(payload.get("avg_posenet_dist", payload.get("pose_dist")))
    seg = _finite_or_none(payload.get("avg_segnet_dist", payload.get("seg_dist")))
    return score, pose, seg


def _default_pass_name(pass_index: int) -> str:
    return {
        0: "anchor",
        1: "coarse_global_representation",
        2: "high_resolution_repair",
        3: "entropy_packer_self_compression",
        4: "pose_runtime_coadaptation",
    }[pass_index]


def _classify(label: str, payload: dict[str, Any]) -> tuple[int, str, str, tuple[str, ...]]:
    schema = str(payload.get("schema", ""))
    lower = f"{label} {schema}".lower()
    if "ego_motion" in lower or "pose_runtime" in lower:
        return 4, "pose_runtime_coadaptation", "pose_runtime_field", ("optimized_poses.bin", "runtime")
    if "archive_bit_budget" in lower or "self_compression" in lower or "packer" in lower:
        return 3, "entropy_packer_self_compression", "archive_entropy_layer", ("p",)
    if "multimask" in lower or "hotspot_geometry" in lower or "foveated" in lower or "repair" in lower:
        return 2, "high_resolution_repair", "high_res_hard_pair_foveal_boundary_repair", ("masks.mkv",)
    if "cmg2" in lower:
        return 1, "coarse_global_representation", "coarse_downsample_mask_grid", ("masks.mkv",)
    if "cmg3" in lower:
        return 1, "coarse_global_representation", "coarse_row_run_mask_grammar", ("masks.mkv",)
    if "pmg" in lower:
        return 1, "coarse_global_representation", "predictive_mask_grammar_global", ("masks.mkv",)
    if "exact_eval" in lower or "fixedslice" in lower or "anchor" in lower:
        return 0, "anchor", "fixedslice_full_resolution_anchor", (
            "masks.mkv",
            "renderer.bin",
            "optimized_poses.bin",
        )
    return 0, "anchor", "unclassified_anchor_context", ()


def _payload_bytes(payload: dict[str, Any]) -> int | None:
    for key in ("cmg2", "cmg3", "pmg_hotspot_cmg3"):
        record = payload.get(key)
        if isinstance(record, dict):
            for byte_key in ("payload_bytes", "body_bytes", "residual_record_bytes"):
                value = _int_or_none(record.get(byte_key))
                if value is not None:
                    return value
    return None


def _disagreement(payload: dict[str, Any]) -> float | None:
    for key in ("cmg2", "cmg3", "pmg_hotspot_cmg3"):
        record = payload.get(key)
        if isinstance(record, dict):
            for value_key in (
                "pixel_disagreement_vs_source",
                "pixel_disagreement_vs_full",
                "final_pixel_disagreement_vs_source_fraction",
                "base_pixel_disagreement_vs_source_fraction",
            ):
                value = _finite_or_none(record.get(value_key))
                if value is not None:
                    return value
    return None


def _source_builder(schema: str | None, pass_index: int) -> tuple[str | None, bool]:
    if schema == "cmg2_downsample_candidate_v1":
        return "experiments/build_cmg2_downsample_candidate.py", True
    if schema == "cmg3_nonzero_row_runs_candidate_v1":
        return "experiments/build_cmg3_nonzero_runs_candidate.py", True
    if schema == "pmg_hotspot_cmg3_candidate_v1":
        return "experiments/build_pmg_hotspot_candidate.py", True
    if schema == "c067_hotspot_mask_geometry_compiler_v1":
        return "experiments/build_cmg3_adaptive_runs_candidate.py --field-policy-json", True
    if pass_index == 3:
        return None, False
    return None, False


def load_artifact(label: str, path: Path) -> Artifact:
    resolved = path.resolve()
    payload = _read_json(resolved)
    if payload.get("score_claim") is True:
        raise PlannerError(f"{resolved} has score_claim=true; planner accepts non-claim inputs only")
    schema = payload.get("compiler_schema", payload.get("schema"))
    schema_text = str(schema) if isinstance(schema, str) else None
    pass_index, pass_name, resolution_layer, logical_members = _classify(label, payload)
    archive_bytes, archive_sha = _archive_bytes(payload)
    score, pose, seg = _score_triplet(payload)
    payload_byte_count = _payload_bytes(payload)
    delta = _int_or_none(_nested(payload, "output_archive", "delta_bytes_vs_frontier"))
    evidence_grade = str(payload.get("evidence_grade", "exact_cuda_observation" if score is not None else EVIDENCE_GRADE))
    promotion_eligible = bool(payload.get("promotion_eligible", False))
    exact_negative = False
    if score is not None and pass_index != 0:
        exact_negative = bool(score > 1.0 or (pose is not None and pose > 0.02) or (seg is not None and seg > 0.01))
    builder, consumable = _source_builder(schema_text, pass_index)
    notes: list[str] = []
    if score is not None:
        notes.append("exact_cuda_observation_input_not_a_new_score_claim")
    if exact_negative:
        notes.append("known_exact_negative_or_component_collapse_for_standalone_candidate")
    if pass_index in {1, 2}:
        notes.append("changes_mask_stream_geometry_or_reconstruction")
    return Artifact(
        artifact_id=_safe_id(label),
        label=label,
        path=resolved,
        sha256=_sha256_file(resolved),
        schema=schema_text,
        pass_index=pass_index,
        pass_name=pass_name,
        resolution_layer=resolution_layer,
        component_type=resolution_layer,
        logical_members=logical_members,
        evidence_grade=evidence_grade,
        score_claim=False,
        promotion_eligible=promotion_eligible,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha,
        payload_bytes=payload_byte_count,
        delta_bytes_vs_anchor=delta,
        observed_score=score,
        observed_pose=pose,
        observed_seg=seg,
        disagreement_fraction=_disagreement(payload),
        exact_negative=exact_negative,
        source_builder=builder,
        builder_consumable=consumable,
        notes=tuple(notes),
    )


def _safe_id(raw: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:96] or "artifact"


def _artifact_record(artifact: Artifact) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "label": artifact.label,
        "path": str(artifact.path),
        "sha256": artifact.sha256,
        "schema": artifact.schema,
        "pass_index": artifact.pass_index,
        "pass_name": artifact.pass_name,
        "resolution_layer": artifact.resolution_layer,
        "component_type": artifact.component_type,
        "logical_members": list(artifact.logical_members),
        "evidence_grade": artifact.evidence_grade,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_bytes": artifact.archive_bytes,
        "archive_sha256": artifact.archive_sha256,
        "payload_bytes": artifact.payload_bytes,
        "delta_bytes_vs_anchor": artifact.delta_bytes_vs_anchor,
        "observed_exact_cuda": {
            "score_recomputed_from_components": artifact.observed_score,
            "avg_posenet_dist": artifact.observed_pose,
            "avg_segnet_dist": artifact.observed_seg,
            "score_claim": False,
        },
        "disagreement_fraction": artifact.disagreement_fraction,
        "exact_negative": artifact.exact_negative,
        "source_builder": artifact.source_builder,
        "builder_consumable": artifact.builder_consumable,
        "notes": list(artifact.notes),
    }


def _best_anchor(artifacts: list[Artifact]) -> Artifact | None:
    anchors = [item for item in artifacts if item.pass_index == 0 and item.archive_bytes is not None]
    if not anchors:
        return None
    return sorted(
        anchors,
        key=lambda item: (
            item.observed_score is None,
            item.observed_score if item.observed_score is not None else float("inf"),
            item.archive_bytes or 0,
            item.artifact_id,
        ),
    )[0]


def _by_pass(artifacts: list[Artifact], pass_index: int) -> list[Artifact]:
    return sorted(
        [item for item in artifacts if item.pass_index == pass_index],
        key=lambda item: (
            item.exact_negative,
            item.archive_bytes if item.archive_bytes is not None else 10**18,
            item.payload_bytes if item.payload_bytes is not None else 10**18,
            item.artifact_id,
        ),
    )


def _predicted_delta(component_ids: list[str], artifacts_by_id: dict[str, Artifact]) -> dict[str, Any]:
    byte_values: list[int] = []
    score_values: list[float] = []
    for component_id in component_ids:
        artifact = artifacts_by_id[component_id]
        if artifact.delta_bytes_vs_anchor is not None:
            byte_values.append(artifact.delta_bytes_vs_anchor)
        if artifact.observed_score is not None and artifact.pass_index != 0:
            score_values.append(artifact.observed_score)
    delta_bytes = sum(byte_values) if byte_values else None
    return {
        "kind": "prediction",
        "score_claim": False,
        "additive_delta_warning": ADDITIVE_DELTA_WARNING,
        "first_order_delta_bytes_vs_anchor": delta_bytes,
        "first_order_rate_score_delta": round(LAMBDA_RATE * delta_bytes, 12) if delta_bytes is not None else None,
        "standalone_exact_scores_seen": score_values,
        "standalone_scores_are_not_stack_scores": True,
    }


def _interaction_edges(component_ids: list[str], artifacts_by_id: dict[str, Artifact]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    components = [artifacts_by_id[item] for item in component_ids]
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            shared = sorted(set(left.logical_members) & set(right.logical_members))
            if not shared:
                continue
            if left.pass_index == 0 or right.pass_index == 0:
                relation = "synergy"
                reason = "anchor supplies custody and baseline bytes for the later pass"
            elif left.pass_index == 3 or right.pass_index == 3:
                relation = "synergy"
                reason = "entropy/packer pass may compress a chosen representation but must be rebuilt on stacked bytes"
            elif left.pass_index == 4 or right.pass_index == 4:
                relation = "antagonism"
                reason = "pose/runtime adaptation can change scorer basin after mask repair; exact stacked eval required"
            else:
                relation = "antagonism"
                reason = "overlapping score-affecting mask atoms target the same logical member and are not blindly additive"
            edges.append(
                {
                    "from_component": left.artifact_id,
                    "to_component": right.artifact_id,
                    "relation": relation,
                    "shared_logical_members": shared,
                    "score_claim": False,
                    "reason": reason,
                }
            )
    for item in components:
        if item.exact_negative:
            edges.append(
                {
                    "from_component": item.artifact_id,
                    "to_component": "policy",
                    "relation": "antagonism",
                    "shared_logical_members": list(item.logical_members),
                    "score_claim": False,
                    "reason": "standalone exact CUDA observation shows component collapse; do not assume repair or pass stacking recovers it",
                }
            )
    return edges


def _contextual_negative_edges(
    component_ids: list[str],
    artifacts_by_id: dict[str, Artifact],
) -> list[dict[str, Any]]:
    components = [artifacts_by_id[item] for item in component_ids]
    negatives = [item for item in artifacts_by_id.values() if item.exact_negative]
    edges: list[dict[str, Any]] = []
    for component in components:
        if component.pass_index == 0:
            continue
        for negative in negatives:
            if negative.artifact_id in component_ids:
                continue
            shared = sorted(set(component.logical_members) & set(negative.logical_members))
            if component.pass_index != negative.pass_index or not shared:
                continue
            edges.append(
                {
                    "from_component": component.artifact_id,
                    "to_component": negative.artifact_id,
                    "relation": "antagonism",
                    "shared_logical_members": shared,
                    "score_claim": False,
                    "reason": (
                        "same-pass/same-member standalone exact-negative context exists; "
                        "treat additive recovery as prediction until stacked exact CUDA eval"
                    ),
                }
            )
    return edges


def _pass_records(component_ids: list[str], artifacts_by_id: dict[str, Artifact]) -> list[dict[str, Any]]:
    used = {artifacts_by_id[item].pass_index for item in component_ids}
    records: list[dict[str, Any]] = []
    for pass_index in range(5):
        component_list = [
            artifacts_by_id[item]
            for item in component_ids
            if artifacts_by_id[item].pass_index == pass_index
        ]
        records.append(
            {
                "pass_index": pass_index,
                "pass_name": _default_pass_name(pass_index),
                "active": pass_index in used,
                "optional": pass_index == 4,
                "components": [
                    {
                        "component_id": item.artifact_id,
                        "resolution_layer": item.resolution_layer,
                        "component_type": item.component_type,
                        "logical_members": list(item.logical_members),
                        "score_claim": False,
                    }
                    for item in component_list
                ],
            }
        )
    return records


def _policy(
    policy_id: str,
    component_ids: list[str],
    artifacts_by_id: dict[str, Artifact],
    *,
    intent: str,
    dispatchable: bool,
) -> dict[str, Any]:
    edges = _interaction_edges(component_ids, artifacts_by_id)
    edges.extend(_contextual_negative_edges(component_ids, artifacts_by_id))
    return {
        "policy_id": policy_id,
        "intent": intent,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "dispatchable_from_this_plan": False,
        "existing_builder_can_consume_full_stack": dispatchable,
        "builder_commands": [],
        "builder_command_status": (
            "no existing builder consumes this full multi-pass stack"
            if not dispatchable
            else "builder command intentionally omitted unless a byte-closed stack builder is selected"
        ),
        "component_ids": component_ids,
        "passes": _pass_records(component_ids, artifacts_by_id),
        "predicted_delta": _predicted_delta(component_ids, artifacts_by_id),
        "pass_synergies": [edge for edge in edges if edge["relation"] == "synergy"],
        "pass_antagonisms": [edge for edge in edges if edge["relation"] == "antagonism"],
        "exact_eval_branch_rule": {
            "score_claim": False,
            "additive_delta_warning": ADDITIVE_DELTA_WARNING,
            "branch": (
                "materialize one byte-closed stacked archive -> verify payload closure, "
                "runtime tree hash, manifest, and no sidecars -> claim dispatch lane -> "
                "run exact CUDA auth eval on the exact archive bytes"
            ),
            "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
            "dispatch_claim_required_before_remote_job": True,
        },
    }


def build_candidate_policies(artifacts: list[Artifact]) -> list[dict[str, Any]]:
    artifacts_by_id = {item.artifact_id: item for item in artifacts}
    anchor = _best_anchor(artifacts)
    pass1 = _by_pass(artifacts, 1)
    pass2 = _by_pass(artifacts, 2)
    pass3 = _by_pass(artifacts, 3)
    pass4 = _by_pass(artifacts, 4)
    policies: list[dict[str, Any]] = []

    if anchor is not None:
        components = [anchor.artifact_id]
        if pass3:
            components.append(pass3[0].artifact_id)
        policies.append(
            _policy(
                "c067_multires_p00_anchor_entropy_guard",
                components,
                artifacts_by_id,
                intent="Preserve the apogee fixed-slice scorer basin and only test packer/self-compression around identical logical streams.",
                dispatchable=False,
            )
        )

    if anchor is not None and pass1:
        components = [anchor.artifact_id, pass1[0].artifact_id]
        if pass2:
            components.append(pass2[0].artifact_id)
        if pass3:
            components.append(pass3[0].artifact_id)
        policies.append(
            _policy(
                "c067_multires_p01_coarse_global_with_highres_repair",
                components,
                artifacts_by_id,
                intent="Replace broad mask bytes with a coarse/global representation, then spend high-resolution repair only on hard-pair/foveal/boundary atoms.",
                dispatchable=False,
            )
        )

    if anchor is not None and len(pass1) >= 2:
        components = [anchor.artifact_id, pass1[1].artifact_id]
        if len(pass2) >= 2:
            components.append(pass2[1].artifact_id)
        elif pass2:
            components.append(pass2[0].artifact_id)
        policies.append(
            _policy(
                "c067_multires_p02_conservative_mask_grammar_recovery",
                components,
                artifacts_by_id,
                intent="Use the next-best coarse mask grammar with an explicit repair/reconciliation pass and keep overlap conflicts visible.",
                dispatchable=False,
            )
        )

    if anchor is not None and pass1 and pass2 and pass4:
        components = [anchor.artifact_id, pass1[0].artifact_id, pass2[0].artifact_id, pass4[0].artifact_id]
        policies.append(
            _policy(
                "c067_multires_p03_optional_pose_runtime_coadaptation",
                components,
                artifacts_by_id,
                intent="Treat pose/runtime co-adaptation as an optional final-pass basin shift after mask-stack repair, not as an additive byte-only win.",
                dispatchable=False,
            )
        )
    return policies


def _pass_catalog(artifacts: list[Artifact]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pass_index in range(5):
        items = _by_pass(artifacts, pass_index)
        out.append(
            {
                "pass_index": pass_index,
                "pass_name": _default_pass_name(pass_index),
                "artifact_ids": [item.artifact_id for item in items],
                "resolution_layers": sorted({item.resolution_layer for item in items}),
                "score_claim": False,
            }
        )
    return out


def build_plan(
    *,
    output_json: Path,
    artifact_specs: list[tuple[str, Path]] | None = None,
    include_default_artifacts: bool = True,
) -> dict[str, Any]:
    specs: list[tuple[str, Path]] = []
    if include_default_artifacts:
        specs.extend((label, REPO_ROOT / rel_path) for label, rel_path in DEFAULT_ARTIFACTS)
    specs.extend(artifact_specs or [])

    seen_labels: set[str] = set()
    artifacts: list[Artifact] = []
    missing: list[dict[str, Any]] = []
    for label, path in specs:
        if label in seen_labels:
            raise PlannerError(f"duplicate artifact label: {label}")
        seen_labels.add(label)
        resolved = path if path.is_absolute() else (REPO_ROOT / path)
        if not resolved.exists():
            missing.append({"label": label, "path": str(resolved), "score_claim": False})
            continue
        artifacts.append(load_artifact(label, resolved))

    artifacts.sort(key=lambda item: (item.pass_index, item.artifact_id))
    policies = build_candidate_policies(artifacts)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "no_score_claim": True,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "planning_only": True,
        "cuda_jobs_launched": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "additive_delta_contract": ADDITIVE_DELTA_WARNING,
        "contest_formula": {
            "score": "100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489",
            "lambda_rate": LAMBDA_RATE,
            "score_claim": False,
        },
        "pass_contract": [
            {
                "pass_index": 0,
                "pass_name": "anchor",
                "required_role": "C067/apogee fixed-slice archive custody and exact-eval baseline",
                "resolution_layer": "fixedslice_full_resolution_anchor",
                "score_claim": False,
            },
            {
                "pass_index": 1,
                "pass_name": "coarse_global_representation",
                "required_role": "coarse mask/video representation that replaces broad bytes",
                "resolution_layer": "coarse_global_mask_or_video",
                "score_claim": False,
            },
            {
                "pass_index": 2,
                "pass_name": "high_resolution_repair",
                "required_role": "hard-pair, foveal, boundary, and residual atoms charged inside archive",
                "resolution_layer": "high_res_repair_atoms",
                "score_claim": False,
            },
            {
                "pass_index": 3,
                "pass_name": "entropy_packer_self_compression",
                "required_role": "deterministic packer or entropy pass over the selected stack bytes",
                "resolution_layer": "archive_entropy_layer",
                "score_claim": False,
            },
            {
                "pass_index": 4,
                "pass_name": "pose_runtime_coadaptation",
                "required_role": "optional pose/runtime co-adaptation after mask-stack geometry is fixed",
                "resolution_layer": "pose_runtime_field",
                "optional": True,
                "score_claim": False,
            },
        ],
        "loaded_artifacts": [_artifact_record(item) for item in artifacts],
        "missing_artifacts": missing,
        "pass_catalog": _pass_catalog(artifacts),
        "candidate_policies": policies,
        "exact_eval_branch_rule": {
            "score_claim": False,
            "required_for_any_candidate_policy": True,
            "additive_delta_warning": ADDITIVE_DELTA_WARNING,
            "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
            "steps": [
                "build a byte-closed stacked archive with every pass payload charged",
                "record deterministic manifest, archive bytes, archive SHA-256, and runtime tree hash",
                "claim lane dispatch before any remote exact eval",
                "run experiments/contest_auth_eval.py --device cuda on the exact archive bytes",
                "promote only if component gates, payload closure, and recomputed formula pass",
            ],
        },
    }
    _write_json(output_json, payload)
    return payload


def _parse_artifact_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("artifact spec must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("artifact label must be non-empty")
    return label, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--artifact-json",
        action="append",
        default=[],
        type=_parse_artifact_spec,
        help="Additional artifact input as LABEL=PATH.",
    )
    parser.add_argument(
        "--no-default-artifacts",
        action="store_true",
        help="Only inspect artifacts supplied with --artifact-json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(
        output_json=args.output_json,
        artifact_specs=list(args.artifact_json),
        include_default_artifacts=not args.no_default_artifacts,
    )
    print(json.dumps({"output_json": str(args.output_json), "candidate_policy_count": len(plan["candidate_policies"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
