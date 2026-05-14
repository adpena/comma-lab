#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Triage C067 non-training big-move structural candidates.

This planner only reads existing local artifacts. It does not load scorers,
build archives, inspect dispatch state, or launch GPU work. The output ranks
non-retraining structural-mask, multiresolution, multimask, topology, repair,
and SJ-KL context by exact-eval readiness, byte-gate eligibility, and
fail-closed blockers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_c067_bigmove_nontrain_candidates.py"
SCHEMA = "c067_bigmove_nontrain_candidate_triage_v1"
OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/results/c067_bigmove_nontrain_candidate_triage_20260502/"
    "c067_bigmove_nontrain_candidate_triage.json"
)

ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
C067_FRONTIER_SCORE = 0.31561703078448233
C067_FRONTIER_ARCHIVE_BYTES = 276_214
C067_FRONTIER_ARCHIVE_SHA256 = (
    "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
)
C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE = 252_760
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
NO_REMOTE_DISPATCH_NOTE = (
    "This planner does not dispatch. Any later remote exact eval requires a "
    "fresh lane claim and an active-eval conflict check outside this run."
)

MULTIMASK_THRESHOLD_SUMMARY = (
    "experiments/results/"
    "c067_multimask_reconciliation_20260502_cmg3a_reconciler_threshold_fix1/"
    "byte_screen_summary.json"
)
MULTIMASK_FIX1_SUMMARY = (
    "experiments/results/c067_multimask_reconciliation_20260502_cmg3a_reconciler_fix1/"
    "byte_screen_summary.json"
)
MULTIRESOLUTION_PLAN = (
    "experiments/results/c067_multiresolution_stack_planner_20260502/"
    "c067_multiresolution_stack_plan.json"
)
HOTSPOT_POSESAFE_PLAN = (
    "experiments/results/c067_hotspot_mask_geometry_compiler_20260502/"
    "next_pose_safe_plan_after_extra065_072_negatives.json"
)
HOTSPOT_GEOMETRY_PLAN = (
    "experiments/results/c067_hotspot_mask_geometry_compiler_20260502/"
    "c067_hotspot_mask_geometry_plan.json"
)
HOTSPOT_GEOMETRY_MANIFEST = (
    "experiments/results/c067_hotspot_mask_geometry_compiler_20260502/"
    "candidate_top0128_c067_anchor/build_manifest.json"
)
PMG_BYTE_SCREEN_JSONL = (
    "experiments/results/pmg_hotspot_candidate_c067_20260502_lzma/"
    "pmg_atomtop_byte_screen.jsonl"
)
PMG_STRIDE1DYN_MANIFEST = (
    "experiments/results/pmg_hotspot_candidate_c067_stride1dyn_pairs64_atoms512_20260502/"
    "build_manifest.json"
)
MICRO_MASK_PLAN = (
    "experiments/results/c067_micro_mask_reencode_plan_20260502/micro_mask_reencode_plan.json"
)
MICRO_MASK_SAVE12_PACKED_PROVENANCE = (
    "experiments/results/c067_micro_mask_reencode_plan_20260502/builds/"
    "c067_micro_av1_mask_reencode_save12k/packed_pr64_maskfirst_qp1/"
    "packed_renderer_payload_provenance.json"
)
POSTDECODE_PAIR_PLAN = (
    "experiments/results/c067_postdecode_mask_repair_candidate_20260502/"
    "c067_postdecode_mask_repair_waterfill_pair_class_plan.json"
)
POSTDECODE_BUDGET4000_MANIFEST = (
    "experiments/results/c067_postdecode_mask_repair_candidate_20260502/"
    "save12k_waterfill_pair_sweep/budget4000/c067_postdecode_mask_repair_manifest.json"
)
POSTDECODE_BUDGET8000_MANIFEST = (
    "experiments/results/c067_postdecode_mask_repair_candidate_20260502/"
    "save12k_budget_sweep/budget8000/c067_postdecode_mask_repair_manifest.json"
)
POSTDECODE_TOP10_MANIFEST = (
    "experiments/results/c067_postdecode_mask_repair_candidate_20260502/"
    "save12k_trace_frame_sweep/top10/c067_postdecode_mask_repair_manifest.json"
)
REVERSED_BASE_CDO1_ECONOMICS = (
    "experiments/results/c067_reversed_base_cdo1_overlay_economics_20260502/"
    "c067_reversed_base_cdo1_overlay_economics.json"
)
TRAINED_RENDERER_EXPORT_UNLOCK_PLAN = (
    "experiments/results/trained_renderer_export_unlock_20260502_codex/"
    "trained_renderer_export_unlock_plan.json"
)

EXACT_EVALS: dict[str, str] = {
    "c067_multimask_reconciler_extra065k_fix1": (
        "experiments/results/lightning_batch/"
        "exact_eval_c067_multimask_reconciler_extra065k_fix1_l40sdiag_20260502T1903Z/"
        "contest_auth_eval.json"
    ),
    "c067_multimask_reconciler_extra072k_fix1": (
        "experiments/results/lightning_batch/"
        "exact_eval_c067_multimask_reconciler_extra072k_fix1_l40sdiag_20260502T1910Z/"
        "contest_auth_eval.json"
    ),
    "c067_micro_av1_mask_reencode_save12k_packed": (
        "experiments/results/lightning_batch/"
        "exact_eval_c067_micro_mask_save12k_l40sdiag_20260502T2034Z/"
        "contest_auth_eval.json"
    ),
    "c067_postdecode_repair_save12k_top10": (
        "experiments/results/lightning_batch/"
        "exact_eval_c067_postdecode_repair_save12k_top10_l40sdiag_20260502T2054Z/"
        "contest_auth_eval.json"
    ),
    "c067_postdecode_repair_save12k_budget8000": (
        "experiments/results/lightning_batch/"
        "exact_eval_c067_postdecode_repair_save12k_budget8000_l40sdiag_20260502T2101Z/"
        "contest_auth_eval.json"
    ),
    "c067_postdecode_repair_save12k_pairwaterfill4k": (
        "experiments/results/lightning_batch/"
        "exact_eval_c067_postdecode_repair_save12k_pairwaterfill4k_l40sdiag_20260502T2114Z/"
        "contest_auth_eval.json"
    ),
    "pmg_hotspot_c067": (
        "experiments/results/lightning_batch/"
        "exact_eval_pmg_hotspot_c067_t4_20260502T1402Z/contest_auth_eval.json"
    ),
    "pmg_hotspot_atomtop4068": (
        "experiments/results/lightning_batch/"
        "exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/"
        "contest_auth_eval.json"
    ),
    "c067_hotspot_geometry_top0128": (
        "experiments/results/lightning_batch/"
        "exact_eval_c067_hotspot_geometry_top0128_l40sdiag_20260502T1733Z/"
        "contest_auth_eval.json"
    ),
}

SJ_KL_DIAGNOSTICS: tuple[tuple[str, str], ...] = (
    (
        "sjkl_c067_l40sdiag_existing_active_diagnostic",
        "experiments/results/lightning_batch/sjkl_c067_l40sdiag_20260502T151434Z/"
        "contest_auth_eval.json",
    ),
    (
        "sjkl_c067_v2_k4_a5_cap32k_existing_active_diagnostic",
        "experiments/results/sjkl_c067_v2_k4_a5_cap32k_l40s_20260502T181718Z/"
        "contest_auth_eval.json",
    ),
)


class PlannerError(ValueError):
    """Raised for malformed local triage artifacts."""


@dataclass
class ArtifactLog:
    loaded: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def load_json(self, repo_root: Path, rel_path: str) -> dict[str, Any] | None:
        path = repo_root / rel_path
        if not path.exists():
            self._missing(rel_path)
            return None
        payload = _read_json(path)
        self._loaded(repo_root, path, kind="json")
        return payload

    def load_jsonl(self, repo_root: Path, rel_path: str) -> list[dict[str, Any]] | None:
        path = repo_root / rel_path
        if not path.exists():
            self._missing(rel_path)
            return None
        rows = _read_jsonl(path)
        self._loaded(repo_root, path, kind="jsonl")
        return rows

    def _loaded(self, repo_root: Path, path: Path, *, kind: str) -> None:
        rel = _display_path(path, repo_root)
        self.loaded[rel] = {
            "kind": kind,
            "path": rel,
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }

    def _missing(self, rel_path: str) -> None:
        if rel_path not in self.missing:
            self.missing.append(rel_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise PlannerError(f"{path}:{lineno} is not valid JSONL") from exc
        if not isinstance(row, dict):
            raise PlannerError(f"{path}:{lineno} must contain a JSON object")
        rows.append(row)
    return rows


def _display_path(raw: str | Path | None, repo_root: Path) -> str | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _round(value: float | None, digits: int = 12) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _archive_from_record(record: dict[str, Any], repo_root: Path) -> dict[str, Any] | None:
    archive = record.get("archive")
    if isinstance(archive, dict):
        bytes_value = _int(archive.get("bytes"))
        if bytes_value is None:
            bytes_value = _int(archive.get("size_bytes"))
        delta = _int(archive.get("delta_bytes_vs_frontier"))
        if delta is None and bytes_value is not None:
            delta = bytes_value - C067_FRONTIER_ARCHIVE_BYTES
        rate_delta = _float(archive.get("formula_only_rate_delta_vs_frontier"))
        if rate_delta is None and bytes_value is not None:
            rate_delta = (bytes_value - C067_FRONTIER_ARCHIVE_BYTES) * RATE_SCORE_PER_BYTE
        return {
            "bytes": bytes_value,
            "delta_bytes_vs_frontier": delta,
            "formula_only_rate_delta_vs_frontier": _round(rate_delta),
            "path": _display_path(archive.get("path"), repo_root),
            "sha256": archive.get("sha256") if isinstance(archive.get("sha256"), str) else None,
        }

    output_archive = record.get("output_archive")
    if isinstance(output_archive, dict):
        bytes_value = _int(output_archive.get("bytes"))
        if bytes_value is None:
            bytes_value = _int(output_archive.get("size_bytes"))
        delta = _int(output_archive.get("delta_bytes_vs_frontier"))
        if delta is None and bytes_value is not None:
            delta = bytes_value - C067_FRONTIER_ARCHIVE_BYTES
        rate_delta = _float(output_archive.get("formula_only_rate_delta_vs_frontier"))
        if rate_delta is None and bytes_value is not None:
            rate_delta = (bytes_value - C067_FRONTIER_ARCHIVE_BYTES) * RATE_SCORE_PER_BYTE
        return {
            "bytes": bytes_value,
            "delta_bytes_vs_frontier": delta,
            "formula_only_rate_delta_vs_frontier": _round(rate_delta),
            "path": _display_path(output_archive.get("path"), repo_root),
            "sha256": output_archive.get("sha256")
            if isinstance(output_archive.get("sha256"), str)
            else None,
        }

    output_bytes = _int(record.get("output_archive_bytes"))
    if output_bytes is not None:
        return {
            "bytes": output_bytes,
            "delta_bytes_vs_frontier": output_bytes - C067_FRONTIER_ARCHIVE_BYTES,
            "formula_only_rate_delta_vs_frontier": _round(
                (output_bytes - C067_FRONTIER_ARCHIVE_BYTES) * RATE_SCORE_PER_BYTE
            ),
            "path": _display_path(record.get("output_archive"), repo_root),
            "sha256": record.get("output_archive_sha256")
            if isinstance(record.get("output_archive_sha256"), str)
            else None,
        }

    archive_size = _int(record.get("archive_size_bytes"))
    if archive_size is not None:
        return {
            "bytes": archive_size,
            "delta_bytes_vs_frontier": archive_size - C067_FRONTIER_ARCHIVE_BYTES,
            "formula_only_rate_delta_vs_frontier": _round(
                (archive_size - C067_FRONTIER_ARCHIVE_BYTES) * RATE_SCORE_PER_BYTE
            ),
            "path": None,
            "sha256": None,
        }
    return None


def _exact_eval_record(
    *,
    repo_root: Path,
    rel_path: str,
    log: ArtifactLog,
) -> dict[str, Any] | None:
    payload = log.load_json(repo_root, rel_path)
    if payload is None:
        return None
    score = _float(payload.get("score_recomputed_from_components"))
    if score is None:
        score = _float(payload.get("final_score"))
    archive_bytes = _int(payload.get("archive_size_bytes"))
    measured_delta = score - C067_FRONTIER_SCORE if score is not None else None
    status = "exact_eval_present"
    if measured_delta is not None and measured_delta > 1e-12:
        status = "exact_negative"
    elif measured_delta is not None:
        status = "exact_nonworse_than_c067_frontier"
    archive_path = Path(rel_path).parent / "archive.zip"
    archive_full_path = repo_root / archive_path
    return {
        "archive_path": str(archive_path) if archive_full_path.exists() else None,
        "archive_sha256": _sha256_file(archive_full_path) if archive_full_path.exists() else None,
        "archive_bytes": archive_bytes,
        "avg_posenet_dist": _round(_float(payload.get("avg_posenet_dist"))),
        "avg_segnet_dist": _round(_float(payload.get("avg_segnet_dist"))),
        "evidence_grade": "A_diagnostic_exact_cuda"
        if payload.get("n_samples") == 600
        else "diagnostic_exact_cuda",
        "measured_score_delta_vs_c067": _round(measured_delta),
        "n_samples": _int(payload.get("n_samples")),
        "path": rel_path,
        "score_recomputed_from_components": _round(score),
        "status": status,
    }


def _archive_from_exact(exact: dict[str, Any] | None) -> dict[str, Any] | None:
    if exact is None:
        return None
    archive_bytes = _int(exact.get("archive_bytes"))
    if archive_bytes is None:
        return None
    return {
        "bytes": archive_bytes,
        "delta_bytes_vs_frontier": archive_bytes - C067_FRONTIER_ARCHIVE_BYTES,
        "formula_only_rate_delta_vs_frontier": _round(
            (archive_bytes - C067_FRONTIER_ARCHIVE_BYTES) * RATE_SCORE_PER_BYTE
        ),
        "path": exact.get("archive_path") if isinstance(exact.get("archive_path"), str) else None,
        "sha256": exact.get("archive_sha256") if isinstance(exact.get("archive_sha256"), str) else None,
    }


def _rate_delta_for_bytes(archive_bytes: int | None) -> float | None:
    if archive_bytes is None:
        return None
    return (archive_bytes - C067_FRONTIER_ARCHIVE_BYTES) * RATE_SCORE_PER_BYTE


def _candidate(
    *,
    candidate_id: str,
    lane: str,
    family: str,
    artifact_path: str,
    archive: dict[str, Any] | None,
    evidence_grade: str,
    policy_id: str | None = None,
    expected_component_benefit_score: float | None = None,
    component_benefit_evidence: str | None = None,
    unchanged_distortion_score: float | None = None,
    exact_eval: dict[str, Any] | None = None,
    fail_closed_blockers: Iterable[str] = (),
    notes: Iterable[str] = (),
    builder_command: Iterable[str] | None = None,
) -> dict[str, Any]:
    archive_bytes = _int(archive.get("bytes")) if isinstance(archive, dict) else None
    rate_delta = _rate_delta_for_bytes(archive_bytes)
    if exact_eval and exact_eval.get("measured_score_delta_vs_c067") is not None:
        expected_net = _float(exact_eval.get("measured_score_delta_vs_c067"))
        expectation_kind = "measured_exact_eval_delta"
    elif expected_component_benefit_score is not None:
        expected_net = (rate_delta or 0.0) - expected_component_benefit_score
        expectation_kind = "first_order_component_benefit_minus_rate"
    else:
        expected_net = rate_delta
        expectation_kind = "unchanged_distortion_byte_screen"

    dispatch_gate_passed = bool(
        archive_bytes is not None
        and archive_bytes <= C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE
    )
    component_benefit_gate_passed = bool(
        expected_component_benefit_score is not None
        and expected_component_benefit_score > 0.0
        and component_benefit_evidence
    )

    blockers = list(dict.fromkeys(fail_closed_blockers))
    if archive_bytes is None:
        blockers.append("no byte-closed archive artifact is present yet")
    if archive_bytes is not None and not dispatch_gate_passed and not component_benefit_gate_passed:
        blockers.append(
            "archive bytes exceed 252760 and no clear expected component benefit evidence is attached"
        )
    if exact_eval and exact_eval.get("status") == "exact_negative":
        blockers.append("exact CUDA eval already measured a regression versus C067")
    if exact_eval and exact_eval.get("status") == "exact_eval_present":
        blockers.append("exact CUDA eval exists; do not duplicate without new evidence")

    dispatchable = bool(
        archive_bytes is not None
        and (dispatch_gate_passed or component_benefit_gate_passed)
        and not blockers
    )
    if dispatchable:
        dispatch_status = "dispatchable_after_lane_claim_and_active_eval_check"
    else:
        dispatch_status = "no_dispatch"

    return {
        "archive": archive,
        "artifact_path": artifact_path,
        "builder_command_if_materialization_needed": list(builder_command)
        if builder_command is not None
        else [],
        "candidate_id": candidate_id,
        "dispatch": {
            "byte_gate_passed": dispatch_gate_passed,
            "component_benefit_gate_passed": component_benefit_gate_passed,
            "dispatchable": dispatchable,
            "no_remote_dispatch_from_this_run": True,
            "status": dispatch_status,
        },
        "evidence_grade": evidence_grade,
        "exact_eval": exact_eval,
        "expected_value": {
            "component_benefit_evidence": component_benefit_evidence,
            "expected_component_benefit_score": _round(expected_component_benefit_score),
            "expected_net_score_delta_vs_c067": _round(expected_net),
            "expectation_kind": expectation_kind,
            "rate_score_delta_vs_c067": _round(rate_delta),
            "unchanged_distortion_score_if_applicable": _round(unchanged_distortion_score),
        },
        "fail_closed_blockers": blockers,
        "family": family,
        "lane": lane,
        "notes": list(notes),
        "policy_id": policy_id,
        "promotion_eligible": False,
        "score_claim": False,
    }


def _rank_key(candidate: dict[str, Any]) -> tuple[int, float, str]:
    dispatchable = bool(candidate["dispatch"]["dispatchable"])
    has_archive = candidate.get("archive") is not None
    exact_status = _nested(candidate, "exact_eval", "status")
    expected = _float(_nested(candidate, "expected_value", "expected_net_score_delta_vs_c067"))
    if dispatchable:
        group = 0
    elif has_archive and exact_status != "exact_negative":
        group = 1
    elif not has_archive and candidate["dispatch"]["component_benefit_gate_passed"]:
        group = 2
    elif exact_status == "exact_negative":
        group = 3
    else:
        group = 4
    return (group, expected if expected is not None else 9999.0, str(candidate["candidate_id"]))


def _collect_multimask(repo_root: Path, log: ArtifactLog) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    lineage_blocker = (
        "same-lineage multimask extra065k/extra072k exact CUDA diagnostics collapsed PoseNet; "
        "operator review required before any fresh exact eval"
    )
    summary = log.load_json(repo_root, MULTIMASK_THRESHOLD_SUMMARY)
    if summary:
        for record in summary.get("candidate_records", []):
            if not isinstance(record, dict):
                continue
            archive = _archive_from_record(record, repo_root)
            extra_runs = _int(record.get("target_extra_runs"))
            candidate_id = f"c067_multimask_threshold_fix1_extra{extra_runs or 'unknown'}"
            out.append(
                _candidate(
                    candidate_id=candidate_id,
                    lane="multiresolution_multimask_reconciliation",
                    family="cmg3a_multimask_reconciler_threshold_fix1",
                    artifact_path=MULTIMASK_THRESHOLD_SUMMARY,
                    archive=archive,
                    evidence_grade=str(
                        record.get(
                            "evidence_grade",
                            "empirical_byte_screen_archive_candidate_until_exact_cuda",
                        )
                    ),
                    policy_id=f"extra{extra_runs}" if extra_runs is not None else None,
                    fail_closed_blockers=[lineage_blocker],
                    notes=[
                        "byte-closed archive exists and is non-training",
                        "threshold_fix1 reduces bytes but has no exact component benefit yet",
                    ],
                )
            )

    fix1 = log.load_json(repo_root, MULTIMASK_FIX1_SUMMARY)
    exact65 = _exact_eval_record(
        repo_root=repo_root, rel_path=EXACT_EVALS["c067_multimask_reconciler_extra065k_fix1"], log=log
    )
    exact72 = _exact_eval_record(
        repo_root=repo_root, rel_path=EXACT_EVALS["c067_multimask_reconciler_extra072k_fix1"], log=log
    )
    if fix1:
        for record in fix1.get("candidate_records", []):
            if not isinstance(record, dict):
                continue
            extra_runs = _int(record.get("target_extra_runs"))
            if extra_runs not in {65_000, 72_000}:
                continue
            candidate_id = (
                "c067_multimask_reconciler_extra065k_fix1"
                if extra_runs == 65_000
                else "c067_multimask_reconciler_extra072k_fix1"
            )
            out.append(
                _candidate(
                    candidate_id=candidate_id,
                    lane="multiresolution_multimask_reconciliation",
                    family="cmg3a_multimask_reconciler_fix1_exact_negative",
                    artifact_path=MULTIMASK_FIX1_SUMMARY,
                    archive=_archive_from_record(record, repo_root),
                    evidence_grade=str(
                        record.get(
                            "evidence_grade",
                            "empirical_byte_screen_archive_candidate_until_exact_cuda",
                        )
                    ),
                    policy_id=f"extra{extra_runs}",
                    exact_eval=exact65 if extra_runs == 65_000 else exact72,
                    notes=["preserved as a structural exact negative, not a dispatch target"],
                )
            )
    return out


def _collect_pmg(repo_root: Path, log: ArtifactLog) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    _exact_eval_record(
        repo_root=repo_root, rel_path=EXACT_EVALS["pmg_hotspot_c067"], log=log
    )
    exact_4068 = _exact_eval_record(
        repo_root=repo_root, rel_path=EXACT_EVALS["pmg_hotspot_atomtop4068"], log=log
    )
    family_blocker = (
        "PMG/topology byte screens have same-family exact negatives; unchanged-distortion "
        "score is not component evidence"
    )
    rows = log.load_jsonl(repo_root, PMG_BYTE_SCREEN_JSONL)
    if rows:
        wanted = {64, 128, 512, 2048, 4068}
        for row in rows:
            n = _int(row.get("n"))
            if n not in wanted:
                continue
            exact = exact_4068 if n == 4068 else None
            out.append(
                _candidate(
                    candidate_id=f"pmg_hotspot_atomtop{n}",
                    lane="scorer_weighted_mask_topology_repair_atoms",
                    family="pmg_hotspot_atomtop_byte_screen",
                    artifact_path=PMG_BYTE_SCREEN_JSONL,
                    archive={
                        "bytes": _int(row.get("bytes")),
                        "delta_bytes_vs_frontier": _int(row.get("delta")),
                        "formula_only_rate_delta_vs_frontier": _round(_float(row.get("rate_delta"))),
                        "path": (
                            "experiments/results/"
                            f"pmg_hotspot_candidate_c067_atomtop{n}_20260502/archive.zip"
                        ),
                        "sha256": row.get("sha") if isinstance(row.get("sha"), str) else None,
                    },
                    evidence_grade="empirical_byte_screen_with_exact_negative_family_context",
                    policy_id=f"atomtop{n}",
                    unchanged_distortion_score=_float(row.get("unchanged_distortion_score")),
                    exact_eval=exact,
                    fail_closed_blockers=[family_blocker],
                    notes=[
                        f"residual atom records: {row.get('atom_records')}",
                        f"pixel disagreement proxy: {row.get('disagreement')}",
                    ],
                )
            )

    stride = log.load_json(repo_root, PMG_STRIDE1DYN_MANIFEST)
    if stride:
        cmg3 = stride.get("cmg3") if isinstance(stride.get("cmg3"), dict) else {}
        ledger = cmg3.get("atom_ledger_selection") if isinstance(cmg3, dict) else {}
        out.append(
            _candidate(
                candidate_id="pmg_hotspot_stride1dyn_pairs64_atoms512",
                lane="scorer_weighted_mask_topology_repair_atoms",
                family="pmg_dynamic_ego_foveal_pair_protected_topology",
                artifact_path=PMG_STRIDE1DYN_MANIFEST,
                archive=_archive_from_record(stride, repo_root),
                evidence_grade=str(stride.get("evidence_grade", "empirical_archive_candidate")),
                policy_id="stride1dyn_pairs64_atoms512",
                fail_closed_blockers=[family_blocker],
                notes=[
                    f"selected_row_run_atom_count={_nested(ledger, 'selected_row_run_atom_count')}",
                    "uses scorer-weighted topology and pair protection, but needs a new exact-safe shape filter",
                ],
            )
        )
    return out


def _collect_hotspot_poseguard(repo_root: Path, log: ArtifactLog) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    exact_geometry = _exact_eval_record(
        repo_root=repo_root, rel_path=EXACT_EVALS["c067_hotspot_geometry_top0128"], log=log
    )
    geometry = log.load_json(repo_root, HOTSPOT_GEOMETRY_MANIFEST)
    if geometry:
        out.append(
            _candidate(
                candidate_id="c067_hotspot_geometry_top0128",
                lane="scorer_weighted_mask_topology_repair_atoms",
                family="broad_hotspot_geometry_exact_negative",
                artifact_path=HOTSPOT_GEOMETRY_MANIFEST,
                archive=_archive_from_record(geometry, repo_root),
                evidence_grade=str(geometry.get("evidence_grade", "empirical_archive_candidate")),
                policy_id="c067_hotspot_geometry_top0128",
                exact_eval=exact_geometry,
                notes=["broad topology atom set is preserved as an exact negative"],
            )
        )
    log.load_json(repo_root, HOTSPOT_GEOMETRY_PLAN)

    pose_safe = log.load_json(repo_root, HOTSPOT_POSESAFE_PLAN)
    if not pose_safe:
        return out
    for policy in pose_safe.get("candidate_policies", [])[:4]:
        if not isinstance(policy, dict):
            continue
        proxy = policy.get("estimated_proxy") if isinstance(policy.get("estimated_proxy"), dict) else {}
        benefit = _float(proxy.get("first_order_score_saved_proxy"))
        policy_id = str(policy.get("policy_id", "unknown_policy"))
        builder = str(policy.get("builder", ""))
        out.append(
            _candidate(
                candidate_id=policy_id,
                lane="scorer_weighted_mask_topology_repair_atoms",
                family="poseguard_topology_repair_atoms_after_exact_negatives",
                artifact_path=HOTSPOT_POSESAFE_PLAN,
                archive=None,
                evidence_grade=str(policy.get("evidence_grade", "planning_only")),
                policy_id=policy_id,
                expected_component_benefit_score=benefit,
                component_benefit_evidence="component-trace weighted topology atoms after exact-negative filters",
                fail_closed_blockers=[
                    "policy is not a byte-closed archive yet; build and byte-screen before exact eval",
                    "top0128 broad hotspot geometry exact negative requires tiny-policy review",
                ],
                notes=[
                    f"selected_atom_count={policy.get('selected_atom_count')}",
                    f"selected_residual_pixels={proxy.get('selected_residual_pixels')}",
                ],
                builder_command=builder.split() if builder else None,
            )
        )
    return out


def _collect_micro_and_postdecode(repo_root: Path, log: ArtifactLog) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    micro_plan = log.load_json(repo_root, MICRO_MASK_PLAN)
    micro_prov = log.load_json(repo_root, MICRO_MASK_SAVE12_PACKED_PROVENANCE)
    micro_exact = _exact_eval_record(
        repo_root=repo_root,
        rel_path=EXACT_EVALS["c067_micro_av1_mask_reencode_save12k_packed"],
        log=log,
    )
    if micro_prov:
        out.append(
            _candidate(
                candidate_id="c067_micro_av1_mask_reencode_save12k_packed",
                lane="scorer_weighted_mask_topology_repair_atoms",
                family="micro_mask_reencode_trust_region",
                artifact_path=MICRO_MASK_SAVE12_PACKED_PROVENANCE,
                archive=_archive_from_record(micro_prov, repo_root),
                evidence_grade="A_negative_exact_cuda" if micro_exact else "empirical_archive_candidate",
                policy_id="c067_micro_av1_mask_reencode_save12k",
                exact_eval=micro_exact,
                notes=["micro trust-region byte win still regressed PoseNet under exact CUDA"],
            )
        )
    if micro_plan:
        for idx, screen in enumerate(micro_plan.get("measured_candidate_byte_screen", [])[:4]):
            if not isinstance(screen, dict):
                continue
            screen_path = Path(str(screen.get("path", f"candidate_{idx}")))
            parent = screen_path.parent
            screen_label = f"{parent.parent.name}_{parent.name}" if parent.parent.name else parent.name
            out.append(
                _candidate(
                    candidate_id=f"micro_mask_screen_{idx}_{screen_label}",
                    lane="scorer_weighted_mask_topology_repair_atoms",
                    family="micro_mask_measured_byte_screen_refused",
                    artifact_path=MICRO_MASK_PLAN,
                    archive={
                        "bytes": _int(screen.get("archive_size_bytes")),
                        "delta_bytes_vs_frontier": (
                            _int(screen.get("archive_size_bytes")) - C067_FRONTIER_ARCHIVE_BYTES
                            if _int(screen.get("archive_size_bytes")) is not None
                            else None
                        ),
                        "formula_only_rate_delta_vs_frontier": _round(
                            _rate_delta_for_bytes(_int(screen.get("archive_size_bytes")))
                        ),
                        "path": _display_path(screen.get("path"), repo_root),
                        "sha256": screen.get("sha256")
                        if isinstance(screen.get("sha256"), str)
                        else None,
                    },
                    evidence_grade=str(micro_plan.get("evidence_grade", "planning_only")),
                    fail_closed_blockers=[
                        str(reason) for reason in screen.get("reject_reasons", [])
                    ],
                    notes=["refused by existing micro planner byte/trust-region screen"],
                )
            )

    post_plan = log.load_json(repo_root, POSTDECODE_PAIR_PLAN)
    exact_top10 = _exact_eval_record(
        repo_root=repo_root,
        rel_path=EXACT_EVALS["c067_postdecode_repair_save12k_top10"],
        log=log,
    )
    exact_budget8000 = _exact_eval_record(
        repo_root=repo_root,
        rel_path=EXACT_EVALS["c067_postdecode_repair_save12k_budget8000"],
        log=log,
    )
    exact_pairwaterfill4k = _exact_eval_record(
        repo_root=repo_root,
        rel_path=EXACT_EVALS["c067_postdecode_repair_save12k_pairwaterfill4k"],
        log=log,
    )
    budget4000_manifest = log.load_json(repo_root, POSTDECODE_BUDGET4000_MANIFEST)
    budget8000_manifest = log.load_json(repo_root, POSTDECODE_BUDGET8000_MANIFEST)
    top10_manifest = log.load_json(repo_root, POSTDECODE_TOP10_MANIFEST)
    if post_plan:
        exact_by_policy = {
            "save12k_exact_trace_pair_waterfill_budget4000": exact_pairwaterfill4k,
            "save12k_exact_trace_pair_waterfill_budget8000": exact_budget8000,
            "save12k_exact_trace_pair_waterfill_top10": exact_top10,
        }
        manifest_by_policy = {
            "save12k_exact_trace_pair_waterfill_budget4000": budget4000_manifest,
            "save12k_exact_trace_pair_waterfill_budget8000": budget8000_manifest,
            "save12k_exact_trace_pair_waterfill_top10": top10_manifest,
        }
        for policy in post_plan.get("budget_policies", [])[:4]:
            if not isinstance(policy, dict):
                continue
            terms = (
                policy.get("expected_marginal_score_terms")
                if isinstance(policy.get("expected_marginal_score_terms"), dict)
                else {}
            )
            policy_id = str(policy.get("policy_id", "unknown_policy"))
            exact = exact_by_policy.get(policy_id)
            manifest = manifest_by_policy.get(policy_id)
            out.append(
                _candidate(
                    candidate_id=f"c067_postdecode_repair_{policy_id}",
                    lane="scorer_weighted_mask_topology_repair_atoms",
                    family="postdecode_mask_repair_waterfill_pair_class",
                    artifact_path=POSTDECODE_PAIR_PLAN,
                    archive=_archive_from_record(manifest, repo_root)
                    if manifest
                    else _archive_from_exact(exact),
                    evidence_grade="planning_only_with_exact_negative_context",
                    policy_id=policy_id,
                    expected_component_benefit_score=_float(
                        terms.get("component_score_improvement_first_order")
                    ),
                    component_benefit_evidence="pair/class component-trace waterfill prior",
                    exact_eval=exact,
                    fail_closed_blockers=[],
                    notes=[
                        f"estimated_payload_bytes={policy.get('estimated_payload_bytes')}",
                        f"selected_atom_count={policy.get('selected_atom_count')}",
                    ],
                    builder_command=(
                        _nested(policy, "builder_contract", "cli_args_fragment") or []
                    ),
                )
            )
        if top10_manifest:
            out.append(
                _candidate(
                    candidate_id="c067_postdecode_repair_save12k_top10",
                    lane="scorer_weighted_mask_topology_repair_atoms",
                    family="postdecode_mask_repair_trace_frame_top10",
                    artifact_path=POSTDECODE_TOP10_MANIFEST,
                    archive=_archive_from_record(top10_manifest, repo_root),
                    evidence_grade="A_negative_exact_cuda" if exact_top10 else "empirical_archive_candidate",
                    policy_id="save12k_trace_frame_top10",
                    exact_eval=exact_top10,
                    notes=["trace-frame top10 repair is preserved as an exact negative"],
                )
            )
    return out


def _collect_reversed_base_cdo1(repo_root: Path, log: ArtifactLog) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    plan = log.load_json(repo_root, REVERSED_BASE_CDO1_ECONOMICS)
    if not plan:
        return out
    rows = plan.get("best_candidates")
    if not isinstance(rows, list):
        return out
    for row in rows[:6]:
        if not isinstance(row, dict):
            continue
        gates = row.get("gates") if isinstance(row.get("gates"), dict) else {}
        estimated = row.get("estimated_archive") if isinstance(row.get("estimated_archive"), dict) else {}
        residual = (
            row.get("mask_disagreement")
            if isinstance(row.get("mask_disagreement"), dict)
            else {}
        )
        blockers = [
            "reversed-base CDO1 rows are lower-bound economics only; no byte-closed archive was built"
        ]
        if not bool(gates.get("residual_geometry_gate")):
            blockers.append("residual decoded-mask geometry gate failed")
        if not bool(gates.get("byte_gate_sub0300_if_distortion_unchanged")):
            blockers.append("estimated bytes do not pass unchanged-distortion sub-0.30 byte gate")
        if not bool(gates.get("joint_sub0300_geometry_gate")):
            blockers.append("no joint byte+geometry gate pass for exact-eval dispatch")
        out.append(
            _candidate(
                candidate_id=f"reversed_base_cdo1_{row.get('candidate_id', 'unknown')}",
                lane="cdo1_reversed_base_mask_topology",
                family="cdo1_overlay_over_smaller_base_lower_bound",
                artifact_path=REVERSED_BASE_CDO1_ECONOMICS,
                archive={
                    "bytes": _int(estimated.get("estimated_archive_bytes")),
                    "delta_bytes_vs_frontier": _int(estimated.get("estimated_delta_vs_c067")),
                    "formula_only_rate_delta_vs_frontier": _round(
                        _float(estimated.get("estimated_rate_delta_vs_c067"))
                    ),
                    "path": None,
                    "sha256": None,
                },
                evidence_grade="derivation_lower_bound_no_score",
                policy_id=str(_nested(row, "policy", "policy_id")),
                fail_closed_blockers=blockers,
                notes=[
                    f"base={_nested(row, 'base', 'label')}",
                    "residual_after_overlay="
                    f"{residual.get('residual_vs_target_fraction_after_overlay')}",
                    "overlay_lzma_bytes="
                    f"{_nested(row, 'cdo1_payload', 'compressed_payloads', 'lzma_xz', 'bytes')}",
                ],
            )
        )
    return out


def _collect_trained_renderer_export(repo_root: Path, log: ArtifactLog) -> list[dict[str, Any]]:
    plan = log.load_json(repo_root, TRAINED_RENDERER_EXPORT_UNLOCK_PLAN)
    if not plan:
        return []
    rows = plan.get("candidates")
    if not isinstance(rows, list):
        rows = plan.get("best_candidates") if isinstance(plan.get("best_candidates"), list) else []
    best_rows = [row for row in rows[:4] if isinstance(row, dict)]
    if not best_rows:
        best_rows = [
            {
                "kind": "trained_renderer_export_unlock",
                "bytes": None,
                "path": TRAINED_RENDERER_EXPORT_UNLOCK_PLAN,
                "sha256": None,
                "blockers": ["no non-surrogate trained renderer export candidate found"],
            }
        ]
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(best_rows):
        blockers = [
            str(blocker)
            for blocker in (row.get("blockers") if isinstance(row.get("blockers"), list) else [])
        ]
        if int(plan.get("h100_ready_preflight_count") or 0) <= 0:
            blockers.append("no H100-ready non-surrogate trained renderer preflight exists")
        if int(plan.get("non_surrogate_candidate_count") or 0) <= 0:
            blockers.append("no non-surrogate trained JointFrameGenerator export exists")
        row_bytes = _int(row.get("bytes"))
        out.append(
            _candidate(
                candidate_id=f"trained_renderer_export_unlock_{idx}_{row.get('kind', 'candidate')}",
                lane="renderer_self_compression",
                family="trained_joint_frame_generator_export",
                artifact_path=TRAINED_RENDERER_EXPORT_UNLOCK_PLAN,
                archive={
                    "bytes": row_bytes,
                    "delta_bytes_vs_frontier": (
                        row_bytes - C067_FRONTIER_ARCHIVE_BYTES if row_bytes is not None else None
                    ),
                    "formula_only_rate_delta_vs_frontier": _round(
                        _rate_delta_for_bytes(row_bytes)
                    ),
                    "path": _display_path(row.get("path"), repo_root),
                    "sha256": row.get("sha256") if isinstance(row.get("sha256"), str) else None,
                },
                evidence_grade="empirical_export_readiness_scan",
                policy_id=str(row.get("kind", "trained_renderer_export")),
                fail_closed_blockers=blockers,
                notes=[
                    f"h100_ready_preflight_count={plan.get('h100_ready_preflight_count')}",
                    f"non_surrogate_candidate_count={plan.get('non_surrogate_candidate_count')}",
                ],
            )
        )
    return out


def _collect_sjkl_diagnostics(repo_root: Path, log: ArtifactLog) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for diagnostic_id, rel_path in SJ_KL_DIAGNOSTICS:
        exact = _exact_eval_record(repo_root=repo_root, rel_path=rel_path, log=log)
        if exact is None:
            continue
        diagnostics.append(
            {
                "diagnostic_id": diagnostic_id,
                "exact_eval": exact,
                "lane": "sjkl_existing_active_diagnostic",
                "no_duplicate_dispatch": True,
                "notes": [
                    "SJ-KL is preserved as existing active diagnostic context, not duplicated in ranked candidates",
                    NO_REMOTE_DISPATCH_NOTE,
                ],
                "score_claim": False,
            }
        )
    return diagnostics


def _assert_no_score_claim_true(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("score_claim") is True:
            raise PlannerError("planner output attempted to emit score_claim=true")
        for child in value.values():
            _assert_no_score_claim_true(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_score_claim_true(child)


def build_plan(
    *,
    repo_root: Path = REPO_ROOT,
    output_json: Path | None = None,
    max_ranked_candidates: int = 24,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    log = ArtifactLog()
    candidates: list[dict[str, Any]] = []

    log.load_json(repo_root, MULTIRESOLUTION_PLAN)
    candidates.extend(_collect_multimask(repo_root, log))
    candidates.extend(_collect_pmg(repo_root, log))
    candidates.extend(_collect_hotspot_poseguard(repo_root, log))
    candidates.extend(_collect_micro_and_postdecode(repo_root, log))
    candidates.extend(_collect_reversed_base_cdo1(repo_root, log))
    candidates.extend(_collect_trained_renderer_export(repo_root, log))
    diagnostics = _collect_sjkl_diagnostics(repo_root, log)

    ranked = sorted(candidates, key=_rank_key)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    ranked = ranked[:max_ranked_candidates]

    lane_ids = sorted({candidate["lane"] for candidate in ranked})
    plan = {
        "active_diagnostics": diagnostics,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "candidate_count": len(candidates),
        "dispatch_policy": {
            "dispatch_byte_gate": C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE,
            "dispatch_claim_required_before_any_remote_job": True,
            "dispatchable_rule": (
                "candidate must have archive bytes <=252760 or clear expected component "
                "benefit with evidence, and must have no fail-closed blockers"
            ),
            "no_remote_dispatch_from_this_run": True,
            "note": NO_REMOTE_DISPATCH_NOTE,
            "score_claim": False,
        },
        "frontier": {
            "archive_bytes": C067_FRONTIER_ARCHIVE_BYTES,
            "archive_sha256": C067_FRONTIER_ARCHIVE_SHA256,
            "score": C067_FRONTIER_SCORE,
            "unchanged_distortion_sub0300_byte_gate": C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE,
        },
        "grand_council_structural_lane_coverage": {
            "multiresolution_multimask_reconciliation": (
                "multiresolution_multimask_reconciliation" in lane_ids
            ),
            "cdo1_reversed_base_mask_topology": (
                "cdo1_reversed_base_mask_topology" in lane_ids
            ),
            "renderer_self_compression": "renderer_self_compression" in lane_ids,
            "scorer_weighted_mask_topology_repair_atoms": (
                "scorer_weighted_mask_topology_repair_atoms" in lane_ids
            ),
            "sjkl_existing_active_diagnostic_not_duplicated": bool(diagnostics),
        },
        "loaded_artifacts": sorted(log.loaded.values(), key=lambda item: item["path"]),
        "missing_artifacts": sorted(log.missing),
        "producer": TOOL,
        "promotion_eligible": False,
        "ranked_candidates": ranked,
        "record_date": "2026-05-02",
        "remote_jobs_dispatched": False,
        "schema": SCHEMA,
        "score_claim": False,
    }
    _assert_no_score_claim_true(plan)

    if output_json is not None:
        _write_json(output_json, plan)
    return plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--max-ranked-candidates", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_ranked_candidates <= 0:
        raise PlannerError("--max-ranked-candidates must be positive")
    plan = build_plan(
        repo_root=args.repo_root,
        output_json=args.output_json,
        max_ranked_candidates=args.max_ranked_candidates,
    )
    print(json.dumps({"output_json": str(args.output_json), "candidate_count": plan["candidate_count"]}))


if __name__ == "__main__":
    main()
