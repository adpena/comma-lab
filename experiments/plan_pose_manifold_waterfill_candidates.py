#!/usr/bin/env python3
"""Plan exact-eval-ready pose-manifold water-fill candidates.

This planner consumes exact eval JSONs, component traces, optional per-pair
atom policies, and optional active-subspace metadata. It emits deterministic
candidate specs only. It does not build an archive, dispatch remote work, or
make a score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "experiments/plan_pose_manifold_waterfill_candidates.py"
SCHEMA_VERSION = 1
EXPECTED_SAMPLES = 600
SCORE_DENOMINATOR_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / SCORE_DENOMINATOR_BYTES
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
NON_PROMOTABLE_WARNING = (
    "This artifact is a planning/specification artifact only. It cannot "
    "promote, rank, kill, retire, or support a score claim until a closed "
    "archive is evaluated through exact CUDA auth eval on identical bytes."
)

DEFAULT_FRONTIER_DIR = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z"
)
DEFAULT_DIAGNOSTIC_DIR = (
    REPO_ROOT
    / "experiments/results/vast_harvest/"
    "archive_eval_ls_c059_weighted_pairs_top32_h100_20260502"
)
DEFAULT_POSE_ATOM_PLAN = (
    REPO_ROOT / "experiments/results/pose_atom_plan_c059_20260502/pose_atom_policies.json"
)
DEFAULT_ACTIVE_METADATA = (
    REPO_ROOT
    / "experiments/results/vast_harvest/"
    "line_search_qzs3_qp1_pr67_active_subspace_c057_fix2_20260502T0240Z_latest/"
    "archive.accepted_latest.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pose_manifold_waterfill_c059_20260502_codex"
)
DEFAULT_LEDGER = (
    REPO_ROOT / ".omx/research/pose_manifold_waterfill_candidates_20260502_codex.md"
)


class PoseManifoldPlanError(ValueError):
    """Raised when the candidate plan inputs fail custody or schema checks."""


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PoseManifoldPlanError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PoseManifoldPlanError(f"{label} must be a JSON object: {path}")
    return payload


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseManifoldPlanError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise PoseManifoldPlanError(f"{field} must be finite")
    return out


def _int_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PoseManifoldPlanError(f"{field} must be an integer")
    return int(value)


def _nested_get(payload: Mapping[str, Any], keys: Iterable[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_present(payload: Mapping[str, Any], candidates: Iterable[Any]) -> Any:
    for candidate in candidates:
        if isinstance(candidate, tuple):
            value = _nested_get(payload, candidate)
        else:
            value = payload.get(candidate)
        if value is not None:
            return value
    return None


def _formula_terms(*, archive_size_bytes: int, avg_pose: float, avg_seg: float) -> dict[str, float]:
    pose_term = math.sqrt(10.0 * avg_pose)
    seg_term = 100.0 * avg_seg
    rate_term = RATE_SCORE_PER_BYTE * archive_size_bytes
    return {
        "pose_score_term": pose_term,
        "seg_score_term": seg_term,
        "rate_score_term": rate_term,
        "formula_score": pose_term + seg_term + rate_term,
    }


def _load_eval(path: Path, *, label: str, require_cuda: bool = True) -> dict[str, Any]:
    payload = _read_json_object(path, label=label)
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    n_samples = int(
        _finite_float(
            _first_present(payload, ["n_samples", "sample_count"]),
            field=f"{label}.n_samples",
        )
    )
    if n_samples != EXPECTED_SAMPLES:
        raise PoseManifoldPlanError(
            f"{label}.n_samples must be {EXPECTED_SAMPLES}, got {n_samples}"
        )
    device = _first_present(payload, ("device", ("provenance", "device")))
    if require_cuda and device != "cuda":
        raise PoseManifoldPlanError(f"{label}.device must be cuda")
    archive_sha = _first_present(
        payload,
        ("archive_sha256", ("provenance", "archive_sha256"), ("provenance", "archive_sha")),
    )
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise PoseManifoldPlanError(f"{label}.archive_sha256 must be a SHA-256 hex string")
    archive_size = int(
        _finite_float(
            _first_present(payload, ["archive_size_bytes", "archive_bytes"]),
            field=f"{label}.archive_size_bytes",
        )
    )
    avg_pose = _finite_float(
        _first_present(payload, ["avg_posenet_dist", "pose_dist", "posenet_dist"]),
        field=f"{label}.avg_posenet_dist",
    )
    avg_seg = _finite_float(
        _first_present(payload, ["avg_segnet_dist", "seg_dist", "segnet_dist"]),
        field=f"{label}.avg_segnet_dist",
    )
    score = _finite_float(
        _first_present(payload, ["score_recomputed_from_components", "final_score", "score"]),
        field=f"{label}.score_recomputed_from_components",
    )
    terms = _formula_terms(archive_size_bytes=archive_size, avg_pose=avg_pose, avg_seg=avg_seg)
    return {
        "label": label,
        "path": path,
        "payload": payload,
        "file": _file_meta(path),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "avg_posenet_dist": avg_pose,
        "avg_segnet_dist": avg_seg,
        "score_recomputed_from_components": score,
        "formula_terms": terms,
        "formula_score_delta_vs_recorded": terms["formula_score"] - score,
        "n_samples": n_samples,
        "device": device,
        "gpu_model": _first_present(payload, ("gpu_model", ("provenance", "gpu_model"))),
        "gpu_t4_match": bool(
            _first_present(payload, ("gpu_t4_match", ("provenance", "gpu_t4_match")))
        ),
        "sys_argv": _first_present(payload, ("sys_argv", ("provenance", "sys_argv"))),
    }


def _load_component_trace(path: Path, *, label: str) -> dict[str, Any]:
    payload = _read_json_object(path, label=label)
    n_samples = _int_value(payload.get("n_samples"), field=f"{label}.n_samples")
    if n_samples != EXPECTED_SAMPLES:
        raise PoseManifoldPlanError(
            f"{label}.n_samples must be {EXPECTED_SAMPLES}, got {n_samples}"
        )
    if payload.get("score_claim") is not False:
        raise PoseManifoldPlanError(f"{label}.score_claim must be false")
    cross = payload.get("contest_auth_eval_cross_check")
    if not isinstance(cross, Mapping) or cross.get("all_match") is not True:
        raise PoseManifoldPlanError(f"{label} must cross-check contest_auth_eval")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLES:
        raise PoseManifoldPlanError(f"{label}.samples must have {EXPECTED_SAMPLES} rows")
    avg_pose = _finite_float(payload.get("avg_posenet_dist"), field=f"{label}.avg_posenet_dist")
    pair_weight = 0.0 if avg_pose <= 0.0 else 5.0 / math.sqrt(10.0 * avg_pose)
    pairs: list[dict[str, Any]] = []
    for row, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise PoseManifoldPlanError(f"{label}.samples[{row}] must be an object")
        pair_index = _int_value(sample.get("pair_index"), field=f"{label}.samples[{row}].pair_index")
        pose = _finite_float(
            sample.get("posenet_dist"),
            field=f"{label}.samples[{row}].posenet_dist",
        )
        seg = _finite_float(
            sample.get("segnet_dist"),
            field=f"{label}.samples[{row}].segnet_dist",
        )
        frames = sample.get("frame_indices")
        if not isinstance(frames, list) or not all(isinstance(item, int) for item in frames):
            frames = [2 * pair_index, 2 * pair_index + 1]
        pairs.append(
            {
                "pair_index": pair_index,
                "frame_indices": [int(item) for item in frames],
                "video_name": sample.get("video_name"),
                "posenet_dist": pose,
                "segnet_dist": seg,
                "pose_score_contribution": pair_weight * pose / EXPECTED_SAMPLES,
                "seg_score_contribution": 100.0 * seg / EXPECTED_SAMPLES,
            }
        )
    pairs.sort(
        key=lambda item: (
            item["pose_score_contribution"] + item["seg_score_contribution"],
            item["pose_score_contribution"],
            -item["pair_index"],
        ),
        reverse=True,
    )
    return {
        "path": path,
        "file": _file_meta(path),
        "avg_posenet_dist": avg_pose,
        "avg_segnet_dist": _finite_float(
            payload.get("avg_segnet_dist"), field=f"{label}.avg_segnet_dist"
        ),
        "score_recomputed_from_components": _finite_float(
            payload.get("score_recomputed_from_components"),
            field=f"{label}.score_recomputed_from_components",
        ),
        "archive_size_bytes": _int_value(
            payload.get("archive_size_bytes"), field=f"{label}.archive_size_bytes"
        ),
        "top_hard_pairs": pairs[:128],
    }


def _load_pose_atom_plan(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _read_json_object(path, label="pose_atom_plan")
    if payload.get("score_claim") is not False:
        raise PoseManifoldPlanError("pose_atom_plan.score_claim must be false")
    if payload.get("promotion_eligible") is not False:
        raise PoseManifoldPlanError("pose_atom_plan.promotion_eligible must be false")
    policies = payload.get("recommended_policies")
    top_atoms = payload.get("top_atoms")
    if not isinstance(policies, list) or not isinstance(top_atoms, list):
        raise PoseManifoldPlanError("pose_atom_plan must contain policies and top_atoms")
    return {
        "path": path,
        "file": _file_meta(path),
        "payload": payload,
        "policies": [item for item in policies if isinstance(item, Mapping)],
        "top_atoms": [item for item in top_atoms if isinstance(item, Mapping)],
    }


def _load_active_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _read_json_object(path, label="active_metadata")
    refinement = payload.get("refinement")
    if not isinstance(refinement, Mapping):
        refinement = payload
    raw_pairs = refinement.get("basis_pair_indices")
    pair_indices: list[int] = []
    if isinstance(raw_pairs, list):
        pair_indices = sorted(
            {
                int(item)
                for item in raw_pairs
                if isinstance(item, int) and 0 <= item < EXPECTED_SAMPLES
            }
        )
    return {
        "path": str(path),
        "file": _file_meta(path),
        "score_claim": payload.get("score_claim"),
        "evidence_grade": payload.get("evidence_grade"),
        "basis_kind": refinement.get("basis_kind"),
        "basis_index": refinement.get("basis_index"),
        "basis_signed_magnitude": refinement.get("basis_signed_magnitude"),
        "basis_pair_indices": pair_indices,
        "archive_sha256": payload.get("archive_sha256"),
        "archive_size_bytes": payload.get("archive_bytes"),
    }


def _validate_archive_against_eval(archive: Path, eval_record: Mapping[str, Any]) -> dict[str, Any]:
    meta = _file_meta(archive)
    expected_sha = eval_record["archive_sha256"]
    expected_bytes = eval_record["archive_size_bytes"]
    if meta["sha256"] != expected_sha:
        raise PoseManifoldPlanError(
            f"archive SHA mismatch for {archive}: {meta['sha256']} != {expected_sha}"
        )
    if meta["size_bytes"] != expected_bytes:
        raise PoseManifoldPlanError(
            f"archive byte mismatch for {archive}: {meta['size_bytes']} != {expected_bytes}"
        )
    return meta


def _eval_command(*, archive: Path, work_dir: Path) -> list[str]:
    return [
        ".venv/bin/python",
        "-u",
        "experiments/contest_auth_eval.py",
        "--archive",
        str(archive),
        "--inflate-sh",
        "submissions/robust_current/inflate.sh",
        "--upstream-dir",
        "upstream",
        "--device",
        "cuda",
        "--keep-work-dir",
        "--work-dir",
        str(work_dir),
    ]


def _delta_record(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    cand_terms = candidate["formula_terms"]
    base_terms = baseline["formula_terms"]
    return {
        "score_delta_vs_frontier": candidate["score_recomputed_from_components"]
        - baseline["score_recomputed_from_components"],
        "formula_score_delta_vs_frontier": cand_terms["formula_score"]
        - base_terms["formula_score"],
        "archive_bytes_delta_vs_frontier": candidate["archive_size_bytes"]
        - baseline["archive_size_bytes"],
        "avg_posenet_delta_vs_frontier": candidate["avg_posenet_dist"]
        - baseline["avg_posenet_dist"],
        "avg_segnet_delta_vs_frontier": candidate["avg_segnet_dist"]
        - baseline["avg_segnet_dist"],
        "pose_score_term_delta_vs_frontier": cand_terms["pose_score_term"]
        - base_terms["pose_score_term"],
        "seg_score_term_delta_vs_frontier": cand_terms["seg_score_term"]
        - base_terms["seg_score_term"],
        "rate_score_term_delta_vs_frontier": cand_terms["rate_score_term"]
        - base_terms["rate_score_term"],
    }


def _macro_specs_from_atom_plan(
    pose_atom_plan: Mapping[str, Any] | None,
    *,
    frontier: Mapping[str, Any],
    active_metadata: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if pose_atom_plan is None:
        return []
    active_pairs = set()
    if active_metadata is not None:
        active_pairs = set(active_metadata["basis_pair_indices"])

    specs: list[dict[str, Any]] = []
    for policy in pose_atom_plan["policies"]:
        selected_pairs = [
            int(item)
            for item in policy.get("selected_pair_indices", [])
            if isinstance(item, int)
        ]
        if not selected_pairs:
            continue
        active_overlap = sorted(active_pairs.intersection(selected_pairs))
        charged_bytes = float(policy.get("charged_bytes_estimate", 0.0))
        expected_saved = float(policy.get("expected_score_saved_sum", 0.0))
        rate_cost = charged_bytes * RATE_SCORE_PER_BYTE
        specs.append(
            {
                "candidate_id": f"{policy.get('policy_name')}_macro_build_spec",
                "candidate_kind": "pose_residual_macro_bundle_spec",
                "status": "build_required_before_eval",
                "archive_path": None,
                "archive_sha256": None,
                "archive_size_bytes": None,
                "source_frontier_archive_sha256": frontier["archive_sha256"],
                "source_frontier_archive_size_bytes": frontier["archive_size_bytes"],
                "selected_pair_indices": selected_pairs,
                "selected_pair_count": len(selected_pairs),
                "active_subspace_pair_overlap": active_overlap,
                "active_subspace_pair_overlap_count": len(active_overlap),
                "estimated_charged_payload_bytes": charged_bytes,
                "estimated_rate_score_cost": rate_cost,
                "expected_score_saved_formula_only": expected_saved,
                "expected_net_score_utility_formula_only": expected_saved - rate_cost,
                "measured_delta_atom_count": policy.get("measured_delta_atom_count"),
                "prior_atom_count": policy.get("prior_atom_count"),
                "payload_contract": (
                    "charged PVR1/QP pose residual pair atoms inside archive.zip; "
                    "no scorer-side sidecar or runtime fetch"
                ),
                "next_build_step": (
                    "materialize selected pair residuals into a closed archive, "
                    "then run exact CUDA auth eval on that archive"
                ),
                "exact_eval_command_after_build": None,
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
    specs.sort(
        key=lambda item: (
            item["expected_net_score_utility_formula_only"],
            item["active_subspace_pair_overlap_count"],
            -item["selected_pair_count"],
        ),
        reverse=True,
    )
    return specs


def _diagnostic_candidate_spec(
    *,
    label: str,
    diagnostic: Mapping[str, Any],
    frontier: Mapping[str, Any],
    archive: Path,
    output_dir: Path,
) -> dict[str, Any]:
    archive_meta = _validate_archive_against_eval(archive, diagnostic)
    delta = _delta_record(diagnostic, frontier)
    diagnostic_only = not diagnostic["gpu_t4_match"]
    return {
        "candidate_id": label,
        "candidate_kind": "already_built_pose_manifold_archive",
        "status": "ready_for_t4_confirmation" if diagnostic_only else "already_t4_evaluated",
        "archive": archive_meta,
        "diagnostic_eval": diagnostic["file"],
        "diagnostic_hardware": {
            "device": diagnostic["device"],
            "gpu_model": diagnostic["gpu_model"],
            "gpu_t4_match": diagnostic["gpu_t4_match"],
        },
        "diagnostic_delta_vs_frontier": delta,
        "requires_t4_confirmation": diagnostic_only,
        "dispatch_priority": "highest" if diagnostic_only and delta["score_delta_vs_frontier"] < 0 else "normal",
        "exact_eval_command": _eval_command(
            archive=archive,
            work_dir=output_dir / f"{label}_t4_eval_work",
        ),
        "conflict_guard": (
            "Before dispatch, claim a lane with tools/claim_lane_dispatch.py "
            "and confirm no active Lightning T4 promotion already covers this "
            "archive SHA."
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _ledger_text(payload: Mapping[str, Any]) -> str:
    frontier = payload["frontier"]
    recs = payload["dispatch_recommendations"]
    top = recs[0] if recs else None
    lines = [
        "# Pose Manifold Water-Fill Candidates - 2026-05-02 Codex",
        "",
        "## Evidence Boundary",
        "",
        f"- Evidence grade: `{payload['evidence_grade']}`.",
        f"- Score claim: `{payload['score_claim']}`.",
        f"- Required score truth: `{payload['required_promotion_eval']}`.",
        "- H100/L40S/A100 diagnostics remain diagnostic until identical bytes pass T4/equivalent CUDA auth eval.",
        "",
        "## Frontier Anchor",
        "",
        f"- Label: `{frontier['label']}`.",
        f"- Archive bytes: `{frontier['archive_size_bytes']}`.",
        f"- Archive SHA-256: `{frontier['archive_sha256']}`.",
        f"- Score recomputed from components: `{frontier['score_recomputed_from_components']}`.",
        f"- Avg PoseNet: `{frontier['avg_posenet_dist']}`.",
        f"- Avg SegNet: `{frontier['avg_segnet_dist']}`.",
        "",
        "## Output Artifacts",
        "",
        f"- Plan JSON: `{payload['artifacts']['plan_json']['path']}`.",
        f"- Dispatch recommendations: `{payload['artifacts']['dispatch_recommendations_json']['path']}`.",
        "",
        "## Top Recommendation",
        "",
    ]
    if top is None:
        lines.append("- No dispatch recommendation was produced.")
    else:
        lines.extend(
            [
                f"- Candidate: `{top['candidate_id']}`.",
                f"- Status: `{top['status']}`.",
                f"- Archive SHA-256: `{top.get('archive', {}).get('sha256')}`.",
                f"- Archive bytes: `{top.get('archive', {}).get('size_bytes')}`.",
                f"- Requires T4 confirmation: `{top.get('requires_t4_confirmation')}`.",
                "- Exact eval command:",
                "",
                "```bash",
                " ".join(top.get("exact_eval_command", []) or []),
                "```",
                "",
                "- Dispatch guard: claim the lane first and do not duplicate an active Lightning T4 promotion claim.",
            ]
        )
    lines.extend(
        [
            "",
            "## Build-Only Macro Specs",
            "",
            f"- Macro specs emitted: `{len(payload['macro_candidate_specs'])}`.",
            "- These specs require a closed archive builder before exact eval; their expected utility is formula-only planning signal.",
            "",
        ]
    )
    return "\n".join(lines)


def build_pose_manifold_plan(
    *,
    frontier_contest_eval: Path,
    frontier_component_trace: Path,
    output_dir: Path,
    ledger_md: Path | None = None,
    diagnostic_contest_eval: Path | None = None,
    diagnostic_archive: Path | None = None,
    diagnostic_label: str = "ls_c059_weighted_pairs_top32_h100",
    pose_atom_plan_path: Path | None = None,
    active_metadata_path: Path | None = None,
    frontier_label: str = "C-059",
) -> dict[str, Any]:
    frontier = _load_eval(frontier_contest_eval, label=frontier_label)
    component_trace = _load_component_trace(frontier_component_trace, label=f"{frontier_label}_trace")
    if component_trace["archive_size_bytes"] != frontier["archive_size_bytes"]:
        raise PoseManifoldPlanError("frontier trace and eval archive bytes disagree")
    if abs(component_trace["score_recomputed_from_components"] - frontier["score_recomputed_from_components"]) > 1e-5:
        raise PoseManifoldPlanError("frontier trace and eval scores disagree")

    pose_atom_plan = _load_pose_atom_plan(pose_atom_plan_path)
    active_metadata = _load_active_metadata(active_metadata_path)
    macro_specs = _macro_specs_from_atom_plan(
        pose_atom_plan,
        frontier=frontier,
        active_metadata=active_metadata,
    )

    diagnostics: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    if diagnostic_contest_eval is not None and diagnostic_archive is not None:
        diagnostic = _load_eval(diagnostic_contest_eval, label=diagnostic_label)
        diagnostics.append(diagnostic)
        recommendations.append(
            _diagnostic_candidate_spec(
                label=diagnostic_label,
                diagnostic=diagnostic,
                frontier=frontier,
                archive=diagnostic_archive,
                output_dir=output_dir,
            )
        )
    elif diagnostic_contest_eval is not None or diagnostic_archive is not None:
        raise PoseManifoldPlanError("diagnostic eval and diagnostic archive must be supplied together")

    recommendations.sort(
        key=lambda item: (
            item["dispatch_priority"] == "highest",
            -float(item["diagnostic_delta_vs_frontier"]["score_delta_vs_frontier"])
            if "diagnostic_delta_vs_frontier" in item
            else 0.0,
        ),
        reverse=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_json = output_dir / "pose_manifold_waterfill_plan.json"
    rec_json = output_dir / "exact_eval_recommendations.json"
    artifact_manifest_json = output_dir / "artifact_manifest.json"
    ledger_path = ledger_md
    if ledger_path is not None:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "diagnostic_planning_non_promotable",
        "non_promotable_warning": NON_PROMOTABLE_WARNING,
        "required_promotion_eval": CUDA_AUTH_EVAL_PATH,
        "contest_formula": {
            "score": "100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/37545489",
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        },
        "frontier": {
            "label": frontier_label,
            "archive_sha256": frontier["archive_sha256"],
            "archive_size_bytes": frontier["archive_size_bytes"],
            "score_recomputed_from_components": frontier["score_recomputed_from_components"],
            "avg_posenet_dist": frontier["avg_posenet_dist"],
            "avg_segnet_dist": frontier["avg_segnet_dist"],
            "n_samples": frontier["n_samples"],
            "device": frontier["device"],
            "gpu_model": frontier["gpu_model"],
            "gpu_t4_match": frontier["gpu_t4_match"],
            "formula_score_delta_vs_recorded": frontier["formula_score_delta_vs_recorded"],
        },
        "inputs": {
            "frontier_contest_eval": frontier["file"],
            "frontier_component_trace": component_trace["file"],
            "pose_atom_plan": None if pose_atom_plan is None else pose_atom_plan["file"],
            "active_metadata": None if active_metadata is None else active_metadata["file"],
            "diagnostic_evals": [item["file"] for item in diagnostics],
        },
        "frontier_hard_pair_head": component_trace["top_hard_pairs"][:16],
        "active_metadata_summary": active_metadata,
        "macro_candidate_specs": macro_specs,
        "dispatch_recommendations": recommendations,
        "artifacts": {
            "plan_json": {
                "path": str(plan_json),
                "sha256": None,
                "size_bytes": None,
                "sha256_omitted_reason": (
                    "self-referential output; final hash is recorded in artifact_manifest"
                ),
            },
            "dispatch_recommendations_json": {
                "path": str(rec_json),
            },
            "artifact_manifest_json": {
                "path": str(artifact_manifest_json),
            },
            "ledger_md": None if ledger_path is None else {"path": str(ledger_path)},
        },
    }

    _write_json(plan_json, payload)
    rec_payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "required_promotion_eval": CUDA_AUTH_EVAL_PATH,
        "dispatch_recommendations": recommendations,
    }
    _write_json(rec_json, rec_payload)
    payload["artifacts"]["dispatch_recommendations_json"].update(_file_meta(rec_json))
    _write_json(plan_json, payload)
    if ledger_path is not None:
        ledger_path.write_text(_ledger_text(payload) + "\n")
        payload["artifacts"]["ledger_md"] = _file_meta(ledger_path)
        _write_json(plan_json, payload)
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "artifacts": {
            "plan_json": _file_meta(plan_json),
            "dispatch_recommendations_json": _file_meta(rec_json),
            "ledger_md": None if ledger_path is None else _file_meta(ledger_path),
        },
        "manifest_self_hash_omitted_reason": "self-referential output",
    }
    _write_json(artifact_manifest_json, manifest_payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontier-contest-eval",
        type=Path,
        default=DEFAULT_FRONTIER_DIR / "contest_auth_eval.adjudicated.json",
    )
    parser.add_argument(
        "--frontier-component-trace",
        type=Path,
        default=DEFAULT_FRONTIER_DIR / "component_trace.json",
    )
    parser.add_argument("--frontier-label", default="C-059")
    parser.add_argument(
        "--diagnostic-contest-eval",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_DIR / "contest_auth_eval.json",
    )
    parser.add_argument(
        "--diagnostic-archive",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_DIR / "archive.zip",
    )
    parser.add_argument("--diagnostic-label", default="ls_c059_weighted_pairs_top32_h100")
    parser.add_argument("--pose-atom-plan", type=Path, default=DEFAULT_POSE_ATOM_PLAN)
    parser.add_argument("--active-metadata", type=Path, default=DEFAULT_ACTIVE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)

    payload = build_pose_manifold_plan(
        frontier_contest_eval=args.frontier_contest_eval,
        frontier_component_trace=args.frontier_component_trace,
        output_dir=args.output_dir,
        ledger_md=args.ledger_md,
        diagnostic_contest_eval=args.diagnostic_contest_eval,
        diagnostic_archive=args.diagnostic_archive,
        diagnostic_label=args.diagnostic_label,
        pose_atom_plan_path=args.pose_atom_plan,
        active_metadata_path=args.active_metadata,
        frontier_label=args.frontier_label,
    )
    top = payload["dispatch_recommendations"][0] if payload["dispatch_recommendations"] else None
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "plan_json": payload["artifacts"]["plan_json"]["path"],
                "dispatch_recommendations_json": payload["artifacts"][
                    "dispatch_recommendations_json"
                ]["path"],
                "artifact_manifest_json": payload["artifacts"]["artifact_manifest_json"]["path"],
                "macro_candidate_specs": len(payload["macro_candidate_specs"]),
                "top_dispatch_candidate": None if top is None else top["candidate_id"],
                "score_claim": payload["score_claim"],
                "promotion_eligible": payload["promotion_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
