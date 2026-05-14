#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan local-only SJ-KL trace benefit-per-byte successors.

This planner consumes exact CUDA auth JSON, component traces, and local SJ-KL
pack/repack manifests.  It does not build archives, load scorers, or dispatch
remote jobs.  The output is a deterministic planning artifact that separates
component-positive diagnostics from true score-frontier movement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sjkl_trace_benefit_allocator_plan_v1"
TOOL = "experiments/plan_sjkl_trace_benefit_allocator.py"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_PER_ARCHIVE_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
EXPECTED_CONTEST_SAMPLES = 600
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)

DEFAULT_BASELINE_AUTH_JSON = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_BASELINE_TRACE_JSON = DEFAULT_BASELINE_AUTH_JSON.with_name("component_trace.json")
DEFAULT_SPARSE_BYTE_SCREEN_JSON = REPO_ROOT / (
    "experiments/results/sjkl_c067_sparse_repack_screen_20260502T_local/"
    "byte_screen_summary.json"
)
DEFAULT_LOCAL_REPACK_ROOTS = (
    REPO_ROOT / "experiments/results/sjkl_c067_sparse_repack_screen_20260502T_local",
    REPO_ROOT / "experiments/results/sjkl_c067_trace_selected_repack_20260502T_local",
)
DEFAULT_OUTPUT_JSON = REPO_ROOT / (
    "experiments/results/sjkl_trace_benefit_allocator_20260502/"
    "sjkl_trace_benefit_plan.json"
)


class PlannerError(ValueError):
    """Raised when custody, trace, or metric inputs are missing or inconsistent."""


@dataclass(frozen=True)
class CandidatePaths:
    """Input files for one already-built/evaluated SJ-KL candidate."""

    label: str
    auth_json: Path
    trace_json: Path
    pack_manifest: Path
    repack_manifest: Path
    adjudication_json: Path | None = None
    eval_provenance_json: Path | None = None


@dataclass(frozen=True)
class AuthMetrics:
    """Score terms extracted from contest auth eval JSON."""

    path: Path
    archive_bytes: int
    score: float
    pose_dist: float
    seg_dist: float
    n_samples: int
    rate_score: float
    pose_score: float
    seg_score: float

    @property
    def component_score(self) -> float:
        return self.pose_score + self.seg_score


def default_candidates() -> tuple[CandidatePaths, ...]:
    """Return the current local SJ-KL diagnostics requested by the runbook."""

    sparse_root = REPO_ROOT / (
        "experiments/results/sjkl_c067_sparse_repack_screen_20260502T_local/"
        "k1_g08x06_p32_a3_gain003125"
    )
    trace_root = REPO_ROOT / (
        "experiments/results/sjkl_c067_trace_selected_repack_20260502T_local/"
        "old_selected_positive9"
    )
    sparse_eval = REPO_ROOT / (
        "experiments/results/lightning_batch/"
        "exact_eval_sjkl_c067_sparse_k1g08_p32_a3_rtxprodiag_20260502T2123Z"
    )
    trace_eval = REPO_ROOT / (
        "experiments/results/lightning_batch/"
        "exact_eval_sjkl_c067_tracepos9_rtxprodiag_20260502T2146Z"
    )
    return (
        CandidatePaths(
            label="sparse_k1_g08x06_p32_a3_gain003125",
            auth_json=sparse_eval / "contest_auth_eval.adjudicated.json",
            trace_json=sparse_eval / "component_trace.json",
            pack_manifest=sparse_root / "pack/sjkl_c067_archive_manifest.json",
            repack_manifest=sparse_root / "repack/sjkl_repack_manifest.json",
        ),
        CandidatePaths(
            label="trace_selected_positive9",
            auth_json=trace_eval / "contest_auth_eval.adjudicated.json",
            trace_json=trace_eval / "component_trace.json",
            pack_manifest=trace_root / "pack/sjkl_c067_archive_manifest.json",
            repack_manifest=trace_root / "repack/sjkl_repack_manifest.json",
        ),
    )


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise PlannerError(f"required JSON input is missing: {_rel(path)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"required JSON input is invalid: {_rel(path)}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    if not path.exists():
        raise PlannerError(f"required custody file is missing: {_rel(path)}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _finite_float(value: Any, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerError(f"{_rel(path)} missing numeric field {field}")
    out = float(value)
    if not math.isfinite(out):
        raise PlannerError(f"{_rel(path)} has non-finite field {field}")
    return out


def _int_field(payload: dict[str, Any], field: str, *, path: Path) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerError(f"{_rel(path)} missing integer field {field}")
    return int(value)


def _score_field(payload: dict[str, Any], *, path: Path) -> float:
    if "score_recomputed_from_components" in payload:
        return _finite_float(payload["score_recomputed_from_components"], field="score_recomputed_from_components", path=path)
    if "final_score" in payload:
        return _finite_float(payload["final_score"], field="final_score", path=path)
    raise PlannerError(f"{_rel(path)} missing recomputed score")


def _load_auth_metrics(path: Path) -> AuthMetrics:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{_rel(path)} must contain a JSON object")
    archive_bytes = _int_field(payload, "archive_size_bytes", path=path)
    n_samples = _int_field(payload, "n_samples", path=path)
    metrics = AuthMetrics(
        path=path,
        archive_bytes=archive_bytes,
        score=_score_field(payload, path=path),
        pose_dist=_finite_float(payload.get("avg_posenet_dist"), field="avg_posenet_dist", path=path),
        seg_dist=_finite_float(payload.get("avg_segnet_dist"), field="avg_segnet_dist", path=path),
        n_samples=n_samples,
        rate_score=_finite_float(payload.get("score_rate_contribution"), field="score_rate_contribution", path=path),
        pose_score=_finite_float(payload.get("score_pose_contribution"), field="score_pose_contribution", path=path),
        seg_score=_finite_float(payload.get("score_seg_contribution"), field="score_seg_contribution", path=path),
    )
    if metrics.n_samples != EXPECTED_CONTEST_SAMPLES:
        raise PlannerError(f"{_rel(path)} has n_samples={metrics.n_samples}, expected {EXPECTED_CONTEST_SAMPLES}")
    return metrics


def _archive_from_pack_manifest(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    archive = payload.get("output_archive") or payload.get("archive")
    if not isinstance(archive, dict):
        raise PlannerError(f"{_rel(path)} missing output_archive custody")
    return archive


def _archive_sha_from_eval_provenance(path: Path) -> str:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{_rel(path)} must contain a JSON object")
    sha = payload.get("archive_sha256")
    if not isinstance(sha, str) or not sha:
        raise PlannerError(f"{_rel(path)} missing archive_sha256")
    return sha


def _archive_sha_from_trace(payload: dict[str, Any], *, path: Path) -> str:
    trace_inputs = payload.get("trace_inputs")
    if not isinstance(trace_inputs, dict):
        raise PlannerError(f"{_rel(path)} missing trace_inputs custody")
    sha = trace_inputs.get("archive_sha256")
    if not isinstance(sha, str) or not sha:
        raise PlannerError(f"{_rel(path)} missing trace_inputs.archive_sha256")
    return sha


def _validate_trace_against_auth(trace_path: Path, auth: AuthMetrics, *, expected_archive_sha: str) -> dict[str, Any]:
    trace = _read_json(trace_path)
    if not isinstance(trace, dict):
        raise PlannerError(f"{_rel(trace_path)} must contain a JSON object")
    if _int_field(trace, "n_samples", path=trace_path) != auth.n_samples:
        raise PlannerError(f"{_rel(trace_path)} n_samples does not match auth JSON")
    if _int_field(trace, "archive_size_bytes", path=trace_path) != auth.archive_bytes:
        raise PlannerError(f"{_rel(trace_path)} archive_size_bytes does not match auth JSON")
    if _archive_sha_from_trace(trace, path=trace_path) != expected_archive_sha:
        raise PlannerError(f"{_rel(trace_path)} archive SHA does not match custody provenance")
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) != auth.n_samples:
        raise PlannerError(f"{_rel(trace_path)} must contain one per-sample trace row for every contest sample")
    trace_score = _score_field(trace, path=trace_path)
    if abs(trace_score - auth.score) > 1e-5:
        raise PlannerError(f"{_rel(trace_path)} score does not match auth JSON within tolerance")
    return trace


def _sample_map(trace: dict[str, Any], *, path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for sample in trace["samples"]:
        if not isinstance(sample, dict):
            raise PlannerError(f"{_rel(path)} contains a non-object sample")
        pair_index = sample.get("pair_index")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int):
            raise PlannerError(f"{_rel(path)} sample missing integer pair_index")
        if pair_index in out:
            raise PlannerError(f"{_rel(path)} duplicate pair_index {pair_index}")
        out[pair_index] = sample
    return out


def _sample_component(sample: dict[str, Any], key: str) -> float:
    value = sample.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerError(f"component trace sample missing numeric {key}")
    out = float(value)
    if not math.isfinite(out):
        raise PlannerError(f"component trace sample has non-finite {key}")
    return out


def _sample_frames(sample: dict[str, Any], pair_index: int) -> list[int]:
    frames = sample.get("frame_indices")
    if isinstance(frames, list) and frames and all(isinstance(item, int) and item >= 0 for item in frames):
        return [int(item) for item in frames]
    frame_start = sample.get("frame_start")
    if isinstance(frame_start, int) and frame_start >= 0:
        return [frame_start, frame_start + 1]
    return [pair_index * 2, pair_index * 2 + 1]


def _selected_pair_indices(repack_manifest: dict[str, Any], *, path: Path) -> list[int]:
    selected = repack_manifest.get("selected_pair_indices") or repack_manifest.get("requested_pair_indices")
    if not isinstance(selected, list) or not selected:
        raise PlannerError(f"{_rel(path)} missing selected_pair_indices")
    out: list[int] = []
    for value in selected:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PlannerError(f"{_rel(path)} has invalid selected pair index")
        out.append(int(value))
    if len(set(out)) != len(out):
        raise PlannerError(f"{_rel(path)} selected_pair_indices contains duplicates")
    return out


def _classification(score_delta: float, component_benefit: float, promotion_eligible: bool) -> str:
    if score_delta < 0.0 and promotion_eligible:
        return "true_frontier_movement"
    if score_delta < 0.0:
        return "diagnostic_frontier_movement_requires_promotion_hardware"
    if component_benefit > 0.0:
        return "score_negative_component_positive"
    if abs(score_delta) <= 1e-12:
        return "score_neutral"
    return "score_negative_component_negative"


def _load_adjudication(path: Path | None, auth_json: Path) -> dict[str, Any]:
    if path is None:
        path = auth_json.with_name("adjudication_provenance.json")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{_rel(path)} must contain a JSON object")
    return payload


def _load_eval_provenance(path: Path | None, auth_json: Path) -> dict[str, Any]:
    if path is None:
        path = auth_json.with_name("eval_provenance.json")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{_rel(path)} must contain a JSON object")
    return payload


def _summarize_candidate(
    *,
    candidate: CandidatePaths,
    baseline_auth: AuthMetrics,
    baseline_trace: dict[str, Any],
    baseline_archive_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    auth = _load_auth_metrics(candidate.auth_json)
    pack = _read_json(candidate.pack_manifest)
    repack = _read_json(candidate.repack_manifest)
    if not isinstance(pack, dict) or not isinstance(repack, dict):
        raise PlannerError(f"{candidate.label} manifests must be JSON objects")

    output_archive = _archive_from_pack_manifest(pack, path=candidate.pack_manifest)
    output_sha = output_archive.get("sha256")
    output_bytes = output_archive.get("bytes")
    if output_sha != _archive_sha_from_eval_provenance(
        candidate.eval_provenance_json or candidate.auth_json.with_name("eval_provenance.json")
    ):
        raise PlannerError(f"{candidate.label} output archive SHA does not match eval provenance")
    if output_sha != _archive_sha_from_trace(
        _read_json(candidate.trace_json), path=candidate.trace_json
    ):
        raise PlannerError(f"{candidate.label} output archive SHA does not match trace")
    if output_sha != (auth_payload_sha := _load_adjudication(
        candidate.adjudication_json, candidate.auth_json
    ).get("contest_cuda_archive_sha256")):
        raise PlannerError(
            f"{candidate.label} output archive SHA does not match adjudication "
            f"({output_sha!r} != {auth_payload_sha!r})"
        )
    if output_bytes != auth.archive_bytes:
        raise PlannerError(f"{candidate.label} output archive bytes do not match auth JSON")

    source_archive = pack.get("source_archive")
    if not isinstance(source_archive, dict):
        raise PlannerError(f"{_rel(candidate.pack_manifest)} missing source_archive")
    if source_archive.get("sha256") != baseline_archive_sha:
        raise PlannerError(f"{candidate.label} source archive SHA does not match baseline")
    if source_archive.get("bytes") != baseline_auth.archive_bytes:
        raise PlannerError(f"{candidate.label} source archive bytes do not match baseline")

    sjkl_payload = pack.get("sjkl_payload")
    out_payload = repack.get("out")
    if not isinstance(sjkl_payload, dict) or not isinstance(out_payload, dict):
        raise PlannerError(f"{candidate.label} missing SJ-KL payload custody")
    if sjkl_payload.get("sha256") != out_payload.get("sha256"):
        raise PlannerError(f"{candidate.label} pack and repack payload SHA differ")
    if sjkl_payload.get("bytes") != out_payload.get("bytes"):
        raise PlannerError(f"{candidate.label} pack and repack payload bytes differ")
    if pack.get("score_claim") is not False or repack.get("score_claim") is not False:
        raise PlannerError(f"{candidate.label} local pack/repack manifests must remain score_claim=false")

    trace = _validate_trace_against_auth(candidate.trace_json, auth, expected_archive_sha=str(output_sha))
    baseline_samples = _sample_map(baseline_trace, path=DEFAULT_BASELINE_TRACE_JSON)
    candidate_samples = _sample_map(trace, path=candidate.trace_json)
    if set(baseline_samples) != set(candidate_samples):
        raise PlannerError(f"{candidate.label} trace sample set does not match baseline")

    selected = set(_selected_pair_indices(repack, path=candidate.repack_manifest))
    pair_deltas: list[dict[str, Any]] = []
    for pair_index in sorted(baseline_samples):
        base_sample = baseline_samples[pair_index]
        cand_sample = candidate_samples[pair_index]
        pose_benefit = _sample_component(base_sample, "score_pose_contribution_first_order") - _sample_component(
            cand_sample, "score_pose_contribution_first_order"
        )
        seg_benefit = _sample_component(base_sample, "score_seg_contribution_exact") - _sample_component(
            cand_sample, "score_seg_contribution_exact"
        )
        component_benefit = pose_benefit + seg_benefit
        pair_deltas.append(
            {
                "pair_index": pair_index,
                "frame_indices": _sample_frames(base_sample, pair_index),
                "selected_by_candidate": pair_index in selected,
                "component_benefit": round(component_benefit, 15),
                "pose_benefit": round(pose_benefit, 15),
                "seg_benefit": round(seg_benefit, 15),
                "baseline_posenet_dist": _sample_component(base_sample, "posenet_dist"),
                "candidate_posenet_dist": _sample_component(cand_sample, "posenet_dist"),
                "baseline_segnet_dist": _sample_component(base_sample, "segnet_dist"),
                "candidate_segnet_dist": _sample_component(cand_sample, "segnet_dist"),
            }
        )

    adjudication = _load_adjudication(candidate.adjudication_json, candidate.auth_json)
    archive_delta_bytes = auth.archive_bytes - baseline_auth.archive_bytes
    rate_cost = auth.rate_score - baseline_auth.rate_score
    component_benefit = baseline_auth.component_score - auth.component_score
    score_delta = auth.score - baseline_auth.score
    break_even_delta_bytes = math.floor(max(component_benefit, 0.0) / RATE_PER_ARCHIVE_BYTE)
    shrink_bytes_required = max(0, archive_delta_bytes - break_even_delta_bytes)
    promotion_eligible = bool(adjudication.get("promotion_eligible"))
    positive_pairs = sum(1 for row in pair_deltas if row["component_benefit"] > 0)
    negative_pairs = sum(1 for row in pair_deltas if row["component_benefit"] < 0)
    selected_benefit = sum(row["component_benefit"] for row in pair_deltas if row["selected_by_candidate"])

    summary = {
        "label": candidate.label,
        "archive": {
            "bytes": auth.archive_bytes,
            "sha256": output_sha,
            "delta_bytes_vs_baseline": archive_delta_bytes,
        },
        "sjkl_payload": {
            "bytes": sjkl_payload.get("bytes"),
            "sha256": sjkl_payload.get("sha256"),
            "basis_bytes": repack.get("basis_bytes"),
            "coefficient_block_bytes": repack.get("coefficient_block_bytes"),
            "selected_pair_count": repack.get("selected_pair_count"),
            "selected_pair_indices": sorted(selected),
            "k": repack.get("k"),
            "alpha_bits": repack.get("alpha_bits"),
            "residual_gain": repack.get("residual_gain"),
            "basis_grid": [repack.get("basis_grid_w"), repack.get("basis_grid_h")],
        },
        "score_terms": {
            "score": auth.score,
            "score_delta_vs_baseline": score_delta,
            "rate_cost_vs_baseline": rate_cost,
            "component_benefit_vs_baseline": component_benefit,
            "pose_score_benefit": baseline_auth.pose_score - auth.pose_score,
            "seg_score_benefit": baseline_auth.seg_score - auth.seg_score,
            "break_even_component_benefit_required": rate_cost,
            "break_even_archive_delta_bytes": break_even_delta_bytes,
            "shrink_bytes_required_to_break_even": shrink_bytes_required,
            "benefit_per_added_archive_byte": component_benefit / archive_delta_bytes
            if archive_delta_bytes > 0
            else None,
            "rate_per_archive_byte": RATE_PER_ARCHIVE_BYTE,
        },
        "classification": {
            "frontier_class": _classification(score_delta, component_benefit, promotion_eligible),
            "component_positive": component_benefit > 0.0,
            "score_improves": score_delta < 0.0,
            "promotion_eligible": promotion_eligible,
            "scientific_score_eligible": bool(adjudication.get("scientific_score_eligible")),
            "evidence_grade": adjudication.get("evidence_grade"),
            "gpu_model": adjudication.get("contest_cuda_gpu_model"),
            "gpu_t4_match": bool(adjudication.get("contest_cuda_gpu_t4_match")),
        },
        "pair_response": {
            "positive_pair_count": positive_pairs,
            "negative_pair_count": negative_pairs,
            "selected_pair_component_benefit": round(selected_benefit, 15),
            "selected_positive_pair_count": sum(
                1 for row in pair_deltas if row["selected_by_candidate"] and row["component_benefit"] > 0
            ),
            "worst_selected_pairs": sorted(
                [row for row in pair_deltas if row["selected_by_candidate"]],
                key=lambda item: (item["component_benefit"], item["pair_index"]),
            )[:8],
            "top_positive_pairs": sorted(
                pair_deltas,
                key=lambda item: (-item["component_benefit"], item["pair_index"]),
            )[:16],
            "top_negative_pairs": sorted(pair_deltas, key=lambda item: (item["component_benefit"], item["pair_index"]))[:12],
        },
        "custody": {
            "auth_json": _rel(candidate.auth_json),
            "trace_json": _rel(candidate.trace_json),
            "pack_manifest": _rel(candidate.pack_manifest),
            "pack_manifest_sha256": _sha256_file(candidate.pack_manifest),
            "repack_manifest": _rel(candidate.repack_manifest),
            "repack_manifest_sha256": _sha256_file(candidate.repack_manifest),
            "eval_provenance": _rel(candidate.eval_provenance_json or candidate.auth_json.with_name("eval_provenance.json")),
            "adjudication_provenance": _rel(
                candidate.adjudication_json or candidate.auth_json.with_name("adjudication_provenance.json")
            ),
        },
    }
    return summary, pair_deltas


def _collect_local_byte_screens(byte_screen_jsons: Iterable[Path], local_repack_roots: Iterable[Path]) -> list[dict[str, Any]]:
    screens: list[dict[str, Any]] = []
    for path in byte_screen_jsons:
        if not path.exists():
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise PlannerError(f"{_rel(path)} must contain a JSON object")
        for candidate in payload.get("candidates", []):
            if isinstance(candidate, dict):
                screens.append(
                    {
                        "name": candidate.get("name"),
                        "archive_bytes": candidate.get("archive_bytes"),
                        "archive_sha256": candidate.get("archive_sha256"),
                        "delta_bytes": candidate.get("delta_bytes"),
                        "sjkl_bytes": candidate.get("sjkl_bytes"),
                        "selected_pair_count": candidate.get("selected_pair_count"),
                        "basis_grid": candidate.get("basis_grid"),
                        "k": candidate.get("k"),
                        "alpha_bits": candidate.get("alpha_bits"),
                        "residual_gain": candidate.get("residual_gain"),
                        "score_claim": candidate.get("score_claim"),
                        "source": _rel(path),
                    }
                )
    for root in local_repack_roots:
        if not root.exists():
            continue
        for pack_manifest_path in sorted(root.glob("*/pack/sjkl_c067_archive_manifest.json")):
            repack_manifest_path = pack_manifest_path.parents[1] / "repack/sjkl_repack_manifest.json"
            if not repack_manifest_path.exists():
                continue
            pack = _read_json(pack_manifest_path)
            repack = _read_json(repack_manifest_path)
            if not isinstance(pack, dict) or not isinstance(repack, dict):
                raise PlannerError(f"{_rel(pack_manifest_path)} and repack manifest must be objects")
            archive = _archive_from_pack_manifest(pack, path=pack_manifest_path)
            selected = repack.get("selected_pair_indices") or repack.get("requested_pair_indices") or []
            screens.append(
                {
                    "name": pack_manifest_path.parents[1].name,
                    "archive_bytes": archive.get("bytes"),
                    "archive_sha256": archive.get("sha256"),
                    "delta_bytes": archive.get("delta_bytes_vs_source_archive"),
                    "sjkl_bytes": (pack.get("sjkl_payload") or {}).get("bytes"),
                    "selected_pair_count": len(selected) if isinstance(selected, list) else None,
                    "selected_pair_indices": selected if isinstance(selected, list) else None,
                    "basis_grid": [repack.get("basis_grid_w"), repack.get("basis_grid_h")],
                    "k": repack.get("k"),
                    "alpha_bits": repack.get("alpha_bits"),
                    "residual_gain": repack.get("residual_gain"),
                    "score_claim": pack.get("score_claim"),
                    "source": _rel(pack_manifest_path),
                }
            )
    deduped: dict[str, dict[str, Any]] = {}
    for item in screens:
        key = str(item.get("archive_sha256") or item.get("name") or item.get("source"))
        deduped[key] = item
    return sorted(deduped.values(), key=lambda item: (item.get("delta_bytes") is None, item.get("delta_bytes") or 0, str(item.get("name"))))


def _recommendations(candidate_summaries: list[dict[str, Any]], local_byte_screens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    exact_positive = [
        item
        for item in candidate_summaries
        if item["classification"]["frontier_class"] == "score_negative_component_positive"
    ]
    if exact_positive:
        best = min(exact_positive, key=lambda item: item["score_terms"]["shrink_bytes_required_to_break_even"])
        top_pairs = [row["pair_index"] for row in best["pair_response"]["top_positive_pairs"][:16]]
        recommendations.append(
            {
                "rank": 1,
                "kind": "explicit_pair_policy",
                "name": "trace_delta_top16_positive_pairs",
                "recommended_pair_indices": top_pairs,
                "basis": "per-pair component benefit vs C067 from exact diagnostic traces",
                "why": (
                    "Current exact SJ-KL diagnostics are component-positive but rate-negative; "
                    "the next explicit policy should target the observed positive response pairs "
                    "instead of the old selected set."
                ),
                "non_dispatchable_until": "local archive build plus exact CUDA eval; no remote dispatch from this planner",
                "avoid_pair_indices": [
                    row["pair_index"] for row in best["pair_response"]["top_negative_pairs"][:8]
                ],
            }
        )
        recommendations.append(
            {
                "rank": 2,
                "kind": "payload_shrink_direction",
                "name": "cap_archive_delta_to_measured_break_even",
                "target_max_archive_delta_bytes": min(
                    item["score_terms"]["break_even_archive_delta_bytes"] for item in exact_positive
                ),
                "minimum_shrink_bytes_from_best_exact": best["score_terms"]["shrink_bytes_required_to_break_even"],
                "why": (
                    "Measured component benefit only buys a few hundred archive bytes. "
                    "Do not promote another SJ-KL payload unless the archive delta is under "
                    "the measured break-even budget or the component benefit materially grows."
                ),
                "directions": [
                    "remove or amortize the charged basis block before adding more pairs",
                    "prefer coefficient-only or smaller-grid payloads when parity permits",
                    "delta-code selected pair ids and fail closed on duplicate/no-op pair selections",
                    "reject policies containing measured negative-response pairs unless a new trace reverses them",
                ],
            }
        )
    frontier = [
        item
        for item in candidate_summaries
        if item["classification"]["frontier_class"] in {
            "true_frontier_movement",
            "diagnostic_frontier_movement_requires_promotion_hardware",
        }
    ]
    if frontier:
        recommendations.append(
            {
                "rank": len(recommendations) + 1,
                "kind": "promotion_review",
                "name": "frontier_candidate_requires_hardware_gate",
                "candidate_labels": [item["label"] for item in frontier],
                "why": "A candidate improved recomputed score, but promotion still depends on T4/equivalent custody.",
            }
        )
    trace_top16 = [item for item in local_byte_screens if item.get("name") == "trace_top16"]
    if trace_top16:
        item = trace_top16[0]
        recommendations.append(
            {
                "rank": len(recommendations) + 1,
                "kind": "local_byte_screen_followup",
                "name": "trace_top16_existing_local_candidate",
                "archive_delta_bytes": item.get("delta_bytes"),
                "sjkl_bytes": item.get("sjkl_bytes"),
                "selected_pair_indices": item.get("selected_pair_indices"),
                "why": (
                    "This local candidate already encodes the top positive trace-delta pair set; "
                    "it is empirical byte-screen evidence only until exact CUDA lands."
                ),
                "score_claim": False,
            }
        )
    return recommendations


def build_plan(
    *,
    output_json: Path,
    baseline_auth_json: Path = DEFAULT_BASELINE_AUTH_JSON,
    baseline_trace_json: Path = DEFAULT_BASELINE_TRACE_JSON,
    candidates: Iterable[CandidatePaths] | None = None,
    byte_screen_jsons: Iterable[Path] = (DEFAULT_SPARSE_BYTE_SCREEN_JSON,),
    local_repack_roots: Iterable[Path] = DEFAULT_LOCAL_REPACK_ROOTS,
) -> dict[str, Any]:
    """Build and write the deterministic SJ-KL allocator plan."""

    baseline_auth = _load_auth_metrics(baseline_auth_json)
    baseline_eval_provenance = _load_eval_provenance(None, baseline_auth_json)
    baseline_archive_sha = _archive_sha_from_eval_provenance(baseline_auth_json.with_name("eval_provenance.json"))
    baseline_trace = _validate_trace_against_auth(
        baseline_trace_json,
        baseline_auth,
        expected_archive_sha=baseline_archive_sha,
    )
    candidate_summaries: list[dict[str, Any]] = []
    all_pair_deltas: dict[int, list[dict[str, Any]]] = {}
    for candidate in tuple(candidates or default_candidates()):
        summary, pair_deltas = _summarize_candidate(
            candidate=candidate,
            baseline_auth=baseline_auth,
            baseline_trace=baseline_trace,
            baseline_archive_sha=baseline_archive_sha,
        )
        candidate_summaries.append(summary)
        for row in pair_deltas:
            all_pair_deltas.setdefault(row["pair_index"], []).append(
                {
                    "candidate": candidate.label,
                    "component_benefit": row["component_benefit"],
                    "pose_benefit": row["pose_benefit"],
                    "seg_benefit": row["seg_benefit"],
                    "frame_indices": row["frame_indices"],
                }
            )

    consensus_pairs: list[dict[str, Any]] = []
    for pair_index, rows in sorted(all_pair_deltas.items()):
        if len(rows) != len(candidate_summaries):
            continue
        min_benefit = min(row["component_benefit"] for row in rows)
        mean_benefit = sum(row["component_benefit"] for row in rows) / len(rows)
        if min_benefit > 0.0:
            consensus_pairs.append(
                {
                    "pair_index": pair_index,
                    "frame_indices": rows[0]["frame_indices"],
                    "min_component_benefit": round(min_benefit, 15),
                    "mean_component_benefit": round(mean_benefit, 15),
                    "candidate_benefits": rows,
                }
            )
    consensus_pairs.sort(key=lambda item: (-item["min_component_benefit"], item["pair_index"]))

    local_byte_screens = _collect_local_byte_screens(byte_screen_jsons, local_repack_roots)
    plan = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "frontier": {
            "archive_bytes": baseline_auth.archive_bytes,
            "archive_sha256": baseline_archive_sha,
            "score": baseline_auth.score,
            "pose_dist": baseline_auth.pose_dist,
            "seg_dist": baseline_auth.seg_dist,
            "rate_score": baseline_auth.rate_score,
            "pose_score": baseline_auth.pose_score,
            "seg_score": baseline_auth.seg_score,
            "component_trace_json": _rel(baseline_trace_json),
            "auth_json": _rel(baseline_auth_json),
            "eval_provenance_sha256": _sha256_file(baseline_auth_json.with_name("eval_provenance.json")),
            "gpu_model": baseline_eval_provenance.get("gpu_model"),
            "gpu_t4_match": baseline_eval_provenance.get("gpu_t4_match"),
        },
        "candidate_summaries": candidate_summaries,
        "consensus_positive_pairs": consensus_pairs[:32],
        "local_byte_screens": local_byte_screens,
        "recommendations": _recommendations(candidate_summaries, local_byte_screens),
        "fail_closed_contract": {
            "requires_auth_json": True,
            "requires_component_trace_samples": EXPECTED_CONTEST_SAMPLES,
            "requires_eval_provenance_archive_sha_match": True,
            "requires_pack_manifest_source_archive_match": True,
            "requires_repack_payload_sha_match": True,
            "no_remote_dispatch": True,
        },
    }
    _write_json(output_json, plan)
    return plan


def _parse_candidate(value: str) -> CandidatePaths:
    parts = value.split("|")
    if len(parts) not in {5, 7}:
        raise argparse.ArgumentTypeError(
            "--candidate must be label|auth_json|trace_json|pack_manifest|repack_manifest"
            "[|adjudication_json|eval_provenance_json]"
        )
    label, auth_json, trace_json, pack_manifest, repack_manifest, *rest = parts
    return CandidatePaths(
        label=label,
        auth_json=Path(auth_json),
        trace_json=Path(trace_json),
        pack_manifest=Path(pack_manifest),
        repack_manifest=Path(repack_manifest),
        adjudication_json=Path(rest[0]) if rest else None,
        eval_provenance_json=Path(rest[1]) if rest else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSON, help="output JSON artifact path")
    parser.add_argument("--baseline-auth-json", type=Path, default=DEFAULT_BASELINE_AUTH_JSON)
    parser.add_argument("--baseline-trace-json", type=Path, default=DEFAULT_BASELINE_TRACE_JSON)
    parser.add_argument(
        "--candidate",
        type=_parse_candidate,
        action="append",
        help=(
            "repeatable candidate spec: "
            "label|auth_json|trace_json|pack_manifest|repack_manifest"
            "[|adjudication_json|eval_provenance_json]"
        ),
    )
    parser.add_argument("--byte-screen-json", type=Path, action="append", default=[DEFAULT_SPARSE_BYTE_SCREEN_JSON])
    parser.add_argument("--local-repack-root", type=Path, action="append", default=list(DEFAULT_LOCAL_REPACK_ROOTS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(
        output_json=args.out,
        baseline_auth_json=args.baseline_auth_json,
        baseline_trace_json=args.baseline_trace_json,
        candidates=args.candidate,
        byte_screen_jsons=args.byte_screen_json,
        local_repack_roots=args.local_repack_root,
    )
    print(json.dumps({"out": _rel(args.out), "schema": plan["schema"], "score_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
