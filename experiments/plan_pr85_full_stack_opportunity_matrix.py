#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planning-only PR85 full-stack opportunity matrix.

The planner consumes already-produced local artifacts, ranks stackable
candidate families, and preserves exact negatives as guardrails. It does not
build archives, touch dispatch state, run scorers, or launch remote/GPU work.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

from tac.repo_io import json_text, sha256_bytes, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_pr85_full_stack_opportunity_matrix.py"
SCHEMA = "pr85_full_stack_opportunity_matrix_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_full_stack_opportunity_matrix_20260504_worker"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_POINTS_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

DEFAULT_GLOBS: dict[str, tuple[str, ...]] = {
    "pr85_bit_budget": ("experiments/results/**/profile_pr85_archive_bit_budget.json",),
    "qma9_context_entropy": ("experiments/results/**/pr85_qma9_context_entropy_profile.json",),
    "qma9_residual_sufficient_program": (
        "experiments/results/**/pr85_residual_sufficient_program_profile.json",
    ),
    "qma9_escape_screen_manifest": (
        "experiments/results/pr85_qma9_escape_screens_*/**/manifest.json",
    ),
    "qma9_native_grammar_summary": (
        "experiments/results/**/pr85_qma9_native_grammar_candidates_*/candidate_summary.json",
    ),
    "qma9_alt_grammar_summary": (
        "experiments/results/**/pr85_qma9_alt_grammar_candidates_*/candidate_summary.json",
    ),
    "qma9_run_grammar_summary": (
        "experiments/results/**/pr85_qma9_run_grammar_candidates_*/candidate_summary.json",
    ),
    "qh0_model_self_compression": (
        "experiments/results/**/profile_pr85_model_payload_self_compression.json",
        "experiments/results/**/pr85_qh0_serializer_candidates_*/candidate_summary.json",
    ),
    "pr86_hpac_contract_plan": (
        "experiments/results/**/pr86_hpac_pr85_contract_port_plan.json",
        "experiments/results/**/pr86_hpac_pr85_contract_port_*/plan.json",
    ),
    "pr86_hpac_probability_variants": (
        "experiments/results/**/pr86_hpac_probability_contract_*/pr86_hpac_probability_contract_variants.json",
    ),
    "pr85_sidechannel_ablation_summary": (
        "experiments/results/**/public_pr85_sidechannel_ablations_*/candidate_summary.json",
    ),
    "pr85_randmulti_policy_summary": (
        "experiments/results/**/pr85_randmulti_group_policy_candidates_*/candidate_summary.json",
    ),
    "pr85_post_motion_policy_summary": (
        "experiments/results/**/pr85_post_motion_group_policy_candidates_*/candidate_summary.json",
    ),
    "pr85_correction_recode_summary": (
        "experiments/results/**/pr85_correction_recodes_*/candidate_summary.json",
    ),
    "pr85_final_bias_stack_summary": (
        "experiments/results/**/pr85_final_bias_stack_candidates_*/candidate_summary.json",
    ),
    "pr85_bridge_sparse_action_summary": (
        "experiments/results/**/pr85_bridge_sparse_action_candidates_*/candidate_summary.json",
    ),
    "pr85_pair_atom_readiness": (
        "experiments/results/**/pr85_pair_atom_candidates_*/planning.json",
        "experiments/results/**/pr85_pair_atom_candidates_*/candidate_summary.json",
    ),
    "pr85_pair_action_specs": (
        "experiments/results/**/pr85_pair_action_candidates_*/candidate_specs.json",
        "/tmp/pr85_pair_action_candidates_*/candidate_specs.json",
    ),
    "pr85_exact_eval": ("experiments/results/lightning_batch/**/contest_auth_eval.json",),
}


class MatrixError(ValueError):
    """Raised when an explicitly supplied planning input is malformed."""


def _rel(path: Path | str, *, root: Path = REPO_ROOT) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MatrixError(f"invalid JSON input: {_rel(path)}") from exc
    if not isinstance(payload, dict):
        raise MatrixError(f"JSON input must be an object: {_rel(path)}")
    return payload


def _stable_digest(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"stable_matrix_digest_sha256"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _rate_bound(bytes_at_stake: int | None) -> float | None:
    if bytes_at_stake is None:
        return None
    return round(bytes_at_stake * RATE_POINTS_PER_BYTE, 12)


def _coerce_paths(root: Path, paths: list[str] | tuple[str, ...] | None) -> list[Path]:
    if not paths:
        return []
    out = []
    for raw in paths:
        if any(ch in raw for ch in "*?["):
            if Path(raw).is_absolute():
                out.extend(Path(path) for path in glob.glob(raw, recursive=True))
            else:
                out.extend(root / Path(path) for path in glob.glob(raw, recursive=True))
            continue
        path = Path(raw)
        out.append(path if path.is_absolute() else root / path)
    return out


def discover_inputs(
    root: Path = REPO_ROOT,
    *,
    overrides: dict[str, list[str]] | None = None,
) -> dict[str, list[Path]]:
    """Discover optional input artifacts by sorted globs.

    Overrides replace discovery for a category and are intended for tests or
    explicit operator input. Missing override paths are kept in the inventory
    but ignored by readers.
    """

    found: dict[str, list[Path]] = {}
    overrides = overrides or {}
    for category, patterns in DEFAULT_GLOBS.items():
        if category in overrides:
            found[category] = sorted(_coerce_paths(root, overrides[category]), key=lambda p: str(p))
            continue
        paths: list[Path] = []
        for pattern in patterns:
            if Path(pattern).is_absolute():
                paths.extend(Path(path) for path in glob.glob(pattern, recursive=True))
            else:
                paths.extend(root.glob(pattern))
        found[category] = sorted({path for path in paths if path.is_file()}, key=lambda p: str(p))
    return found


def _inventory(root: Path, inputs: dict[str, list[Path]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for category in sorted(inputs):
        entries = []
        for path in inputs[category]:
            exists = path.exists()
            entry: dict[str, Any] = {
                "path": _rel(path, root=root),
                "exists": exists,
            }
            if exists and path.is_file():
                entry["bytes"] = path.stat().st_size
                entry["sha256"] = sha256_file(path)
            entries.append(entry)
        rows[category] = {
            "count": sum(1 for entry in entries if entry["exists"]),
            "entries": entries,
        }
    return rows


def _load_existing(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, _read_json(path)) for path in paths if path.exists()]


def _latest_payload(inputs: dict[str, list[Path]], category: str) -> tuple[Path, dict[str, Any]] | None:
    loaded = _load_existing(inputs.get(category, []))
    if not loaded:
        return None
    return loaded[-1]


def _segment_lengths(bit_budget: dict[str, Any]) -> dict[str, int]:
    bundle = bit_budget.get("bundle", {})
    lengths = bundle.get("segment_lengths", {}) if isinstance(bundle, dict) else {}
    if isinstance(lengths, dict):
        return {
            str(name): int(value)
            for name, value in lengths.items()
            if _as_int(value) is not None
        }
    segments = bit_budget.get("segments", [])
    if isinstance(segments, list):
        return {
            str(row.get("name")): int(row.get("bytes"))
            for row in segments
            if isinstance(row, dict) and row.get("name") and _as_int(row.get("bytes")) is not None
        }
    return {}


def _best_pr85_baseline(exact_evals: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any] | None:
    candidates = []
    for path, payload in exact_evals:
        path_text = str(path)
        score = _as_float(payload.get("score_recomputed_from_components"))
        samples = _as_int(payload.get("n_samples"))
        if "public_pr85" in path_text and samples == 600 and score is not None:
            candidates.append((score, path, payload))
    if not candidates:
        return None
    score, path, payload = sorted(candidates, key=lambda row: (row[0], str(row[1])))[0]
    return {
        "path": _rel(path),
        "archive_bytes": _as_int(payload.get("archive_size_bytes")),
        "n_samples": _as_int(payload.get("n_samples")),
        "score": score,
        "seg_dist": _as_float(payload.get("avg_segnet_dist")),
        "pose_dist": _as_float(payload.get("avg_posenet_dist")),
        "evidence_status": "exact_cuda_auth_eval_full_600_samples",
    }


def _exact_negative_rows(
    exact_evals: list[tuple[Path, dict[str, Any]]],
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    baseline_score = _as_float((baseline or {}).get("score"))
    rows = []
    for path, payload in exact_evals:
        if "minus_" not in str(path):
            continue
        score = _as_float(payload.get("score_recomputed_from_components"))
        samples = _as_int(payload.get("n_samples"))
        row = {
            "path": _rel(path),
            "route": path.parent.name.replace("exact_eval_pr85_", ""),
            "archive_bytes": _as_int(payload.get("archive_size_bytes")),
            "n_samples": samples,
            "score": score,
            "score_delta_vs_baseline": (
                round(score - baseline_score, 12)
                if score is not None and baseline_score is not None
                else None
            ),
            "evidence_status": (
                "exact_cuda_negative_full_600_samples"
                if samples == 600 and score is not None
                else "non_promotable_or_incomplete"
            ),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["route"]))


def _scoped_negative_rows(
    exact_evals: list[tuple[Path, dict[str, Any]]],
    baseline: dict[str, Any] | None,
    *,
    path_markers: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return exact full-sample regressions for a scoped candidate family."""
    baseline_score = _as_float((baseline or {}).get("score"))
    if baseline_score is None:
        return []
    rows: list[dict[str, Any]] = []
    for path, payload in exact_evals:
        path_text = str(path)
        if not any(marker in path_text for marker in path_markers):
            continue
        score = _as_float(payload.get("score_recomputed_from_components"))
        samples = _as_int(payload.get("n_samples"))
        if score is None or samples != 600 or score <= baseline_score:
            continue
        rows.append(
            {
                "path": _rel(path),
                "archive_bytes": _as_int(payload.get("archive_size_bytes")),
                "n_samples": samples,
                "score": score,
                "score_delta_vs_baseline": round(score - baseline_score, 12),
                "pose_dist": _as_float(payload.get("avg_posenet_dist")),
                "seg_dist": _as_float(payload.get("avg_segnet_dist")),
                "evidence_status": "exact_cuda_negative_full_600_samples",
            }
        )
    return sorted(rows, key=lambda row: (row["score_delta_vs_baseline"], row["path"]))


def _best_by_delta(candidates: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    eligible = [
        row
        for row in candidates
        if isinstance(row, dict) and _as_int(row.get(field)) is not None
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (int(row[field]), str(row.get("policy_id") or row.get("candidate_id"))))[0]


def _summary_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates", [])
    return [row for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []


def _opportunity(
    *,
    family_id: str,
    surface: str,
    bytes_at_stake: int | None,
    exact_evidence_status: str,
    stackability: str,
    risks: list[str],
    gates: list[str],
    recommendation: str,
    source_artifacts: list[str],
    already_refuted: bool = False,
    blocked: bool = False,
    priority: int,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "family_id": family_id,
        "surface": surface,
        "bytes_at_stake": bytes_at_stake,
        "estimated_rate_only_score_bound": _rate_bound(bytes_at_stake),
        "exact_evidence_status": exact_evidence_status,
        "stackability": stackability,
        "synergy_antagonism_risks": risks,
        "required_gates": gates,
        "recommendation": recommendation,
        "already_refuted": already_refuted,
        "blocked": blocked,
        "priority": priority,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "source_artifacts": source_artifacts,
        "notes": notes or [],
    }


def _build_opportunities(
    inputs: dict[str, list[Path]],
    loaded: dict[str, list[tuple[Path, dict[str, Any]]]],
    baseline: dict[str, Any] | None,
    exact_evals: list[tuple[Path, dict[str, Any]]],
    exact_negatives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    bit_budget_row = _latest_payload(inputs, "pr85_bit_budget")
    bit_budget = bit_budget_row[1] if bit_budget_row else {}
    bit_budget_path = [_rel(bit_budget_row[0])] if bit_budget_row else []
    segments = _segment_lengths(bit_budget)
    mask_bytes = segments.get("mask")
    model_bytes = segments.get("model")
    randmulti_bytes = segments.get("randmulti")
    post_motion_bytes = sum(segments.get(name, 0) for name in ("post", "shift", "frac", "frac2", "frac3"))

    qh0_row = _latest_payload(inputs, "qh0_model_self_compression")
    if qh0_row:
        qh0 = qh0_row[1]
        route = qh0.get("top_implementable_route", {})
        no_real_byte_win = qh0.get("blocker_class") == "no_real_byte_win"
        best_screened = qh0.get("best_screened_candidate", {})
        route_bytes = _as_int(route.get("bytes_at_stake")) or model_bytes
        opportunities.append(
            _opportunity(
                family_id="qh0_record_level_model_repack",
                surface="PR85 QH0 joint-frame model payload",
                bytes_at_stake=route_bytes,
                exact_evidence_status=(
                    "empirical_serializer_screen_no_real_byte_win"
                    if no_real_byte_win
                    else "planning_profile_only_no_runtime_parity"
                ),
                stackability="high: changes model segment only and should compose with mask/post/randmulti work after runtime parity",
                risks=[
                    "QH0 serializer errors can change renderer tensors and PoseNet/SegNet terms",
                    (
                        "deterministic runtime-compatible QH0/QM0 serializer screen was byte-neutral or byte-negative"
                        if no_real_byte_win
                        else "generic recompression of the charged encoded segment was already non-improving"
                    ),
                ],
                gates=[
                    (
                        "new representation-level QH0 transform with byte-positive local screen"
                        if no_real_byte_win
                        else "reviewed deterministic QH0 serializer"
                    ),
                    "decoded tensor/record parity",
                    "PR85 runtime output parity",
                    "deterministic archive byte closure",
                    "exact CUDA auth eval before any score claim",
                ],
                recommendation=str(
                    (
                        "Measured deterministic QH0/QM0 serializer recodes are neutral or byte-negative; pursue only representation-changing model compression, not wrapper recode."
                        if no_real_byte_win
                        else route.get("recommendation")
                        or "Build a record-level QH0 serializer/repacker; do not rely on generic wrapper compression."
                    )
                ),
                source_artifacts=[_rel(qh0_row[0])] + bit_budget_path,
                already_refuted=no_real_byte_win,
                blocked=no_real_byte_win,
                priority=10,
                notes=[
                    f"blocker_class={qh0.get('blocker_class')}",
                    f"best_model_delta={best_screened.get('candidate_model_delta_bytes_vs_source') if isinstance(best_screened, dict) else None}",
                    f"built_candidate_count={qh0.get('built_candidate_count')}",
                ],
            )
        )

    qma9_row = _latest_payload(inputs, "qma9_context_entropy")
    if qma9_row:
        qma9 = qma9_row[1]
        context_rank = qma9.get("opportunity_ranking", [])
        best_context = context_rank[0] if isinstance(context_rank, list) and context_rank else {}
        run_lengths = qma9.get("run_length_opportunities", {})
        opportunities.append(
            _opportunity(
                family_id="qma9_native_run_grammar_or_table_reduction",
                surface="PR85 QMA9 mask payload",
                bytes_at_stake=mask_bytes,
                exact_evidence_status="planning_profile_only_mask_bytes_no_candidate_archive",
                stackability="medium: mask-only if decoded-token and runtime parity are preserved",
                risks=[
                    "simple context entropy replacement is byte-negative against current QMA9",
                    "mask-token parity is necessary but not sufficient for inflate/runtime parity",
                ],
                gates=[
                    "full-stream deterministic QMA9-compatible encoder",
                    "decoded token SHA parity against PR85 baseline",
                    "archive byte closure and runtime output parity",
                    "exact CUDA auth eval before score promotion",
                ],
                recommendation=(
                    "Target native run grammar/table-overhead reductions, not generic symbol/context entropy replacement."
                ),
                source_artifacts=[_rel(qma9_row[0])] + bit_budget_path,
                priority=20,
                notes=[
                    f"best_simple_context={best_context.get('model')}",
                    f"row_run_avg={((run_lengths.get('row_sequences_along_width_axis') or {}) if isinstance(run_lengths, dict) else {}).get('average_run_length')}",
                ],
            )
        )
        opportunities.append(
            _opportunity(
                family_id="qma9_simple_context_entropy_replacement",
                surface="PR85 QMA9 mask payload",
                bytes_at_stake=mask_bytes,
                exact_evidence_status="empirical_static_profile_refutes_byte_economics",
                stackability="low: refuted as a replacement family unless it becomes a parity control",
                risks=[
                    "estimated ideal simple-context payload exceeds charged PR85 QMA9 bytes before overhead",
                ],
                gates=["none for dispatch; keep only as a local parity control"],
                recommendation="Already refuted for dispatch: do not build a simple entropy-context replacement as a score candidate.",
                source_artifacts=[_rel(qma9_row[0])],
                already_refuted=True,
                blocked=True,
                priority=900,
                notes=[f"best_context_byte_gap={best_context.get('break_even_overhead_bytes')}"],
            )
        )

    residual_row = _latest_payload(inputs, "qma9_residual_sufficient_program")
    if residual_row:
        residual = residual_row[1]
        programs = residual.get("residual_programs", [])
        best_program = programs[0] if isinstance(programs, list) and programs else {}
        saved = _as_float(best_program.get("estimated_bytes_saved_vs_charged_mask")) if isinstance(best_program, dict) else None
        lower_bound = _as_float(best_program.get("best_lower_bound_bytes")) if isinstance(best_program, dict) else None
        predictor = best_program.get("predictor") if isinstance(best_program, dict) else None
        refuted_as_direct_coder = saved is not None and saved <= 0.0
        opportunities.append(
            _opportunity(
                family_id="qma9_residual_sufficient_program_density",
                surface="PR85 QMA9 decoded mask token sufficient statistics",
                bytes_at_stake=mask_bytes,
                exact_evidence_status=(
                    "empirical_residual_sufficient_program_direct_coder_negative"
                    if refuted_as_direct_coder
                    else "empirical_residual_sufficient_program_byte_positive_profile"
                ),
                stackability=(
                    "high as a learned-coder curriculum/profile field; low as a direct residual bitmap"
                    if refuted_as_direct_coder
                    else "medium if lowered into a byte-closed runtime-supported coder"
                ),
                risks=[
                    "naive residual-map location cost can dominate despite high zero fraction",
                    "profile must guide learned/native coder design, not dispatch by itself",
                ],
                gates=[
                    "lower profile into a concrete byte-closed codec or training objective",
                    "decoded token SHA parity against PR85 baseline",
                    "runtime output parity and deterministic archive closure",
                    "exact CUDA auth eval after lane claim",
                ],
                recommendation=(
                    "Use this as the non-arbitrary density/curriculum field for HPAC/native/learned mask coding; do not dispatch a naive residual-map coder."
                    if refuted_as_direct_coder
                    else "Prototype the best residual sufficient program as a byte-closed local coder before any exact eval."
                ),
                source_artifacts=[_rel(residual_row[0])],
                already_refuted=False,
                blocked=True,
                priority=22,
                notes=[
                    f"best_predictor={predictor}",
                    f"best_lower_bound_bytes={lower_bound}",
                    f"estimated_bytes_saved_vs_charged_mask={saved}",
                ],
            )
        )

    escape_rows = loaded.get("qma9_escape_screen_manifest", [])
    if escape_rows:
        negative_count = 0
        best_delta: int | None = None
        for _, payload in escape_rows:
            decision = payload.get("decision", {})
            if isinstance(decision, dict) and decision.get("local_screen_negative") is True:
                negative_count += 1
            subset = payload.get("subset", {})
            delta = _as_int(subset.get("delta_bytes_vs_subset_qma9")) if isinstance(subset, dict) else None
            if delta is not None:
                best_delta = delta if best_delta is None else min(best_delta, delta)
        all_negative = negative_count == len(escape_rows)
        opportunities.append(
            _opportunity(
                family_id="qma9_block_copy_escape_screens",
                surface="PR85 QMA9 block/copy escape variants",
                bytes_at_stake=mask_bytes,
                exact_evidence_status="empirical_prefix_screen_negative" if all_negative else "empirical_prefix_screen_mixed",
                stackability="low until a full-stream screen beats QMA9 and preserves decode parity",
                risks=[
                    "prefix projection can overstate full-stream economics",
                    "block-copy flags add overhead against an already compact adaptive QMA9 stream",
                ],
                gates=[
                    "full-stream deterministic byte screen beats current QMA9",
                    "decode parity on all 600 frames",
                    "runtime parity before exact eval",
                ],
                recommendation=(
                    "Already refuted for the screened variants; revisit only with a structurally different native run grammar."
                    if all_negative
                    else "Keep as non-promotable screen data until a full-stream positive appears."
                ),
                source_artifacts=[_rel(path) for path, _ in escape_rows],
                already_refuted=all_negative,
                blocked=all_negative,
                priority=910,
                notes=[f"screen_count={len(escape_rows)}", f"best_subset_delta_bytes={best_delta}"],
            )
        )

    qma9_native_row = _latest_payload(inputs, "qma9_native_grammar_summary")
    if qma9_native_row:
        qma9_native = qma9_native_row[1]
        candidate_count = _as_int(qma9_native.get("candidate_count")) or 0
        best_delta = _as_int(qma9_native.get("best_byte_delta"))
        blockers = qma9_native.get("blockers", [])
        if not isinstance(blockers, list):
            blockers = []
        no_candidate = candidate_count == 0
        opportunities.append(
            _opportunity(
                family_id="qma9_native_runtime_supported_grammar_screen",
                surface="PR85 QMA9 runtime-supported grammar reductions",
                bytes_at_stake=abs(best_delta) if best_delta is not None else mask_bytes,
                exact_evidence_status=(
                    "empirical_runtime_supported_screen_no_byte_win"
                    if no_candidate
                    else "candidate_archive_ready_unscored"
                ),
                stackability="medium only after decoded-token parity and exact stacked eval",
                risks=[
                    "current runtime-supported QMA9 suffix/trim screens emitted no byte-positive archive",
                    "alternate QMA9 magics require runtime edits and replay parity before dispatch",
                ],
                gates=[
                    (
                        "new QMA9 runtime grammar implementation with token parity"
                        if no_candidate
                        else "review candidate archive custody"
                    ),
                    "deterministic archive byte closure",
                    "exact CUDA auth eval after lane claim",
                ],
                recommendation=(
                    "Measured runtime-supported QMA9 grammar trims are closed; next QMA9 work must implement an alternate grammar/table reduction, not a suffix trim."
                    if no_candidate
                    else "Candidate archive exists; review custody and lane claim before exact eval."
                ),
                source_artifacts=[_rel(qma9_native_row[0])],
                already_refuted=no_candidate,
                blocked=no_candidate,
                priority=905 if no_candidate else 20,
                notes=[
                    f"candidate_count={candidate_count}",
                    f"best_byte_delta={best_delta}",
                    f"blockers={','.join(str(item) for item in blockers[:6])}",
                ],
            )
        )

    qma9_alt_row = _latest_payload(inputs, "qma9_alt_grammar_summary")
    if qma9_alt_row:
        qma9_alt = qma9_alt_row[1]
        byte_positive_count = _as_int(qma9_alt.get("byte_positive_candidate_count")) or 0
        runtime_supported_positive_count = (
            _as_int(qma9_alt.get("runtime_supported_byte_positive_candidate_count")) or 0
        )
        best_alt = qma9_alt.get("best_alt_candidate", {})
        fail_closed = qma9_alt.get("fail_closed", {})
        source_qma9 = qma9_alt.get("source_qma9", {})
        best_alt_delta = None
        best_alt_mode = None
        best_alt_payload = None
        if isinstance(best_alt, dict):
            best_alt_delta = _as_int(best_alt.get("delta_bytes_vs_source_qma9"))
            best_alt_mode = best_alt.get("mode") or best_alt.get("candidate_id")
            best_alt_payload = _as_int(best_alt.get("payload_bytes"))
        if isinstance(fail_closed, dict):
            best_alt_delta = best_alt_delta if best_alt_delta is not None else _as_int(
                fail_closed.get("best_alt_delta_bytes_vs_source_qma9")
            )
            best_alt_mode = best_alt_mode or fail_closed.get("best_alt_mode")
            best_alt_payload = best_alt_payload if best_alt_payload is not None else _as_int(
                fail_closed.get("best_alt_payload_bytes")
            )
        source_payload = None
        if isinstance(source_qma9, dict):
            source_payload = _as_int(source_qma9.get("segment_bytes"))
        no_byte_positive = byte_positive_count == 0
        opportunities.append(
            _opportunity(
                family_id="qma9_alternate_neighbor_table_grammar_screen",
                surface="PR85 QMA9 alternate neighbor/table grammars",
                bytes_at_stake=source_payload or mask_bytes,
                exact_evidence_status=(
                    "empirical_alt_grammar_full_stream_no_byte_win"
                    if no_byte_positive
                    else "empirical_alt_grammar_byte_positive_runtime_locked"
                ),
                stackability="medium only if a future byte-positive grammar has token parity and an explicit charged runtime mode",
                risks=[
                    "screened alternate neighbor/table grammars were full-stream local byte-negative"
                    if no_byte_positive
                    else "byte-positive alternate grammars require runtime edits before any exact eval",
                    "QMA9 magic cannot silently change semantics; runtime must use a distinct magic or charged mode header",
                ],
                gates=[
                    "new structurally different grammar, not another screened neighbor-table variant"
                    if no_byte_positive
                    else "runtime parity for selected alternate grammar",
                    "decoded token SHA parity against PR85 baseline",
                    "deterministic archive byte closure",
                    "exact CUDA auth eval after lane claim",
                ],
                recommendation=(
                    "Do not redispatch screened alternate neighbor/table modes; push only a structurally different run grammar or HPAC-style model."
                    if no_byte_positive
                    else "Port only after runtime parity and archive byte win are both proven locally."
                ),
                source_artifacts=[_rel(qma9_alt_row[0])],
                already_refuted=no_byte_positive,
                blocked=runtime_supported_positive_count == 0,
                priority=906 if no_byte_positive else 25,
                notes=[
                    f"byte_positive_candidate_count={byte_positive_count}",
                    f"runtime_supported_byte_positive_candidate_count={runtime_supported_positive_count}",
                    f"best_alt_mode={best_alt_mode}",
                    f"best_alt_delta_bytes_vs_source_qma9={best_alt_delta}",
                    f"best_alt_payload_bytes={best_alt_payload}",
                ],
            )
        )

    qma9_run_row = _latest_payload(inputs, "qma9_run_grammar_summary")
    if qma9_run_row:
        qma9_run = qma9_run_row[1]
        byte_positive_count = _as_int(qma9_run.get("byte_positive_candidate_count")) or 0
        runtime_supported_positive_count = (
            _as_int(qma9_run.get("runtime_supported_byte_positive_candidate_count")) or 0
        )
        best = qma9_run.get("best_bytes_vs_pr85_qma9_159011B", {})
        best_payload = qma9_run.get("best_payload_candidate", {})
        source_qma9 = qma9_run.get("source_qma9", {})
        best_mode = None
        best_payload_bytes = None
        best_delta = None
        if isinstance(best, dict):
            best_mode = best.get("best_mode")
            best_payload_bytes = _as_int(best.get("best_payload_bytes"))
            best_delta = _as_int(best.get("best_delta_bytes"))
        if isinstance(best_payload, dict):
            best_mode = best_mode or best_payload.get("mode") or best_payload.get("candidate_id")
            best_payload_bytes = best_payload_bytes if best_payload_bytes is not None else _as_int(
                best_payload.get("payload_bytes")
            )
            best_delta = best_delta if best_delta is not None else _as_int(
                best_payload.get("delta_bytes_vs_pr85_qma9_159011B")
            )
        source_payload = _as_int(source_qma9.get("segment_bytes")) if isinstance(source_qma9, dict) else None
        no_byte_positive = byte_positive_count == 0
        opportunities.append(
            _opportunity(
                family_id="qma9_qrg1_row_run_grammar_screen",
                surface="PR85 QMA9 row-run grammar replacement",
                bytes_at_stake=source_payload or mask_bytes,
                exact_evidence_status=(
                    "empirical_qrg1_full_stream_no_byte_win"
                    if no_byte_positive
                    else "empirical_qrg1_byte_positive_runtime_locked"
                ),
                stackability="low after current screen; only future model-based or radically different grammars should revisit this surface",
                risks=[
                    "QRG1 row-run payloads were much larger than PR85 adaptive QMA9"
                    if no_byte_positive
                    else "QRG1 byte-positive payload would still require runtime output parity",
                    "runtime does not decode QRG1 and must not silently reinterpret QMA9 bytes",
                ],
                gates=[
                    "new non-row-RLE mask entropy model or learned proposal beats QMA9 locally"
                    if no_byte_positive
                    else "QRG1 runtime output parity",
                    "decoded token SHA parity",
                    "deterministic archive byte closure",
                    "exact CUDA auth eval after lane claim",
                ],
                recommendation=(
                    "Do not pursue raw row-run grammar as a score lane; PR85 QMA9 is already far smaller. Shift QMA9 work to learned/HPAC-style entropy models."
                    if no_byte_positive
                    else "Port only after local byte win and runtime parity."
                ),
                source_artifacts=[_rel(qma9_run_row[0])],
                already_refuted=no_byte_positive,
                blocked=runtime_supported_positive_count == 0,
                priority=907 if no_byte_positive else 25,
                notes=[
                    f"byte_positive_candidate_count={byte_positive_count}",
                    f"runtime_supported_byte_positive_candidate_count={runtime_supported_positive_count}",
                    f"best_mode={best_mode}",
                    f"best_payload_bytes={best_payload_bytes}",
                    f"best_delta_bytes_vs_pr85_qma9={best_delta}",
                ],
            )
        )

    hpac_row = _latest_payload(inputs, "pr86_hpac_contract_plan")
    if hpac_row:
        hpac = hpac_row[1]
        gross = hpac.get("gross_byte_math", {})
        gross_bytes = _as_int(gross.get("gross_mask_byte_opportunity")) if isinstance(gross, dict) else None
        fail_closed = hpac.get("fail_closed_reasons", [])
        opportunities.append(
            _opportunity(
                family_id="pr86_hpac_pr85_mask_contract_port",
                surface="PR85 mask payload via PR86 HPAC contract",
                bytes_at_stake=gross_bytes,
                exact_evidence_status="fail_closed_planning_only_no_pr86_exact_score_or_pr85_hpac_parity",
                stackability="medium after parity: mask-contract replacement should compose with model and protected sidechannels",
                risks=[
                    "PR86 exact-score evidence is missing or invalid in current plan",
                    "PR86 full decode/reencode and PR85 HPAC token parity are not passed",
                ],
                gates=[
                    "PR86 full decode/reencode byte parity",
                    "PR85 HPAC decoded-token parity",
                    "PR85 runtime output parity",
                    "candidate archive byte closure",
                    "exact CUDA auth eval before score claim",
                ],
                recommendation="Do not dispatch; unblock local HPAC parity gates first.",
                source_artifacts=[_rel(hpac_row[0])],
                blocked=True,
                priority=30,
                notes=[f"fail_closed_gate_count={len(fail_closed) if isinstance(fail_closed, list) else None}"],
            )
        )

    hpac_prob_row = _latest_payload(inputs, "pr86_hpac_probability_variants")
    if hpac_prob_row:
        hpac_prob = hpac_prob_row[1]
        variants = hpac_prob.get("variant_results", [])
        if not isinstance(variants, list):
            variants = []
        parity_variants = hpac_prob.get("byte_parity_variants", [])
        source_parity_variants = hpac_prob.get("source_contract_byte_parity_variants", [])
        if not isinstance(parity_variants, list):
            parity_variants = []
        if not isinstance(source_parity_variants, list):
            source_parity_variants = []
        best_prefix: dict[str, Any] | None = None
        for row in variants:
            if not isinstance(row, dict):
                continue
            decode = row.get("hpac_decode", {})
            count = _as_int(decode.get("decoded_symbol_count_before_failure")) if isinstance(decode, dict) else None
            if count is None:
                continue
            if best_prefix is None or count > int(best_prefix.get("decoded_symbol_count_before_failure", -1)):
                variant = row.get("probability_variant", {})
                best_prefix = {
                    "variant": variant.get("name") if isinstance(variant, dict) else None,
                    "decoded_symbol_count_before_failure": count,
                }
        no_full_decode = not parity_variants and not source_parity_variants
        opportunities.append(
            _opportunity(
                family_id="pr86_hpac_probability_contract_variants",
                surface="PR86 HPAC submitted-token probability model",
                bytes_at_stake=None,
                exact_evidence_status=(
                    "fail_closed_probability_variants_no_full_decode"
                    if no_full_decode
                    else "byte_parity_variant_available_for_review"
                ),
                stackability="medium after full PR86 decode/reencode parity; currently blocks HPAC transfer",
                risks=[
                    "float32/perfect variants advance the prefix but still fail before full submitted-token decode",
                    "off-contract probability variants cannot justify PR85 transfer without byte-exact tokens.bin parity",
                ],
                gates=[
                    "full submitted tokens.bin decode",
                    "byte-exact reencode of submitted tokens.bin",
                    "dependency/version contract proof",
                    "PR85 token transfer parity before any exact eval",
                ],
                recommendation=(
                    "Do not dispatch or transfer HPAC from probability variants; next HPAC work must recover the missing full-stream contract, not tune scalar dtype/perfect flags."
                    if no_full_decode
                    else "Review byte-parity variant and lower into PR85 transfer only after source-contract proof."
                ),
                source_artifacts=[_rel(hpac_prob_row[0])],
                already_refuted=no_full_decode,
                blocked=no_full_decode,
                priority=908 if no_full_decode else 25,
                notes=[
                    f"variant_count={len(variants)}",
                    f"byte_parity_variant_count={len(parity_variants)}",
                    f"source_contract_byte_parity_variant_count={len(source_parity_variants)}",
                    f"best_prefix_variant={(best_prefix or {}).get('variant')}",
                    f"best_prefix_decoded_symbols={(best_prefix or {}).get('decoded_symbol_count_before_failure')}",
                ],
            )
        )

    rand_row = _latest_payload(inputs, "pr85_randmulti_policy_summary")
    if rand_row:
        best = _best_by_delta(_summary_candidates(rand_row[1]), "byte_delta_vs_source_archive")
        saved = -int(best["byte_delta_vs_source_archive"]) if best else randmulti_bytes
        randmulti_exact_negatives = _scoped_negative_rows(
            exact_evals,
            baseline,
            path_markers=("pr85_randmulti",),
        )
        randmulti_refuted = bool(randmulti_exact_negatives)
        opportunities.append(
            _opportunity(
                family_id="protected_randmulti_group_waterfill",
                surface="PR85 randmulti sidechannel groups",
                bytes_at_stake=saved,
                exact_evidence_status=(
                    "exact_cuda_negative_full_600_samples"
                    if randmulti_refuted
                    else "candidate_archive_empirical_plus_whole_stream_exact_negative_guardrail"
                ),
                stackability="high with model/mask work; medium with post/motion because scorer interactions are not proven additive",
                risks=[
                    "whole randmulti deletion is exact CUDA negative",
                    (
                        "group-level randmulti waterfill candidates have exact CUDA regressions"
                        if randmulti_refuted
                        else "group-level rescue estimates are planning only until exact eval"
                    ),
                ],
                gates=[
                    (
                        "no dispatch for measured group-deletion/waterfill configs"
                        if randmulti_refuted
                        else "preserve exact candidate archive custody"
                    ),
                    "runtime output sanity/parity where applicable",
                    "exact CUDA auth eval after lane claim",
                    "component-gate adjudication against PR85 baseline",
                ],
                recommendation=(
                    "Measured randmulti group-deletion/waterfill configs are exact negatives; pursue only decoded-output-preserving recode or new component-response microatoms."
                    if randmulti_refuted
                    else f"Best current policy is {best.get('policy_id')} with rate-only byte saving; keep as protected group-level candidate, not whole deletion."
                    if best
                    else "Build or refresh randmulti group candidate summary before ranking."
                ),
                source_artifacts=[_rel(rand_row[0])],
                already_refuted=randmulti_refuted,
                blocked=randmulti_refuted,
                priority=40,
                notes=[
                    f"best_policy={best.get('policy_id') if best else None}",
                    f"exact_negative_count={len(randmulti_exact_negatives)}",
                    *[
                        f"negative={row['path']} score_delta={row['score_delta_vs_baseline']}"
                        for row in randmulti_exact_negatives[:4]
                    ],
                ],
            )
        )

    post_row = _latest_payload(inputs, "pr85_post_motion_policy_summary")
    if post_row:
        best = _best_by_delta(_summary_candidates(post_row[1]), "byte_delta_vs_source_archive")
        saved = -int(best["byte_delta_vs_source_archive"]) if best else post_motion_bytes
        post_motion_exact_negatives = _scoped_negative_rows(
            exact_evals,
            baseline,
            path_markers=("minus_post", "minus_motion", "preserve_post"),
        )
        best_policy = str(best.get("policy_id")) if best else None
        post_motion_refuted = bool(post_motion_exact_negatives) and best_policy in {
            "preserve_motion_only",
            "preserve_post_all_shift_frac2_frac3",
        }
        opportunities.append(
            _opportunity(
                family_id="protected_post_motion_group_policy",
                surface="PR85 post/motion micro sidechannels",
                bytes_at_stake=saved,
                exact_evidence_status=(
                    "exact_cuda_negative_full_600_samples"
                    if post_motion_refuted
                    else "candidate_archive_empirical_plus_whole_stream_exact_negative_guardrail"
                ),
                stackability="medium: small bytes, high scorer sensitivity, likely stackable only after exact component trace",
                risks=[
                    "whole post and motion deletions are exact CUDA negative",
                    (
                        "best current protected policy falls into an exact-negative deletion basin"
                        if post_motion_refuted
                        else "small rate wins can be dominated by PoseNet/SegNet movement"
                    ),
                ],
                gates=[
                    (
                        "new post/motion policy outside measured exact-negative deletion basin"
                        if post_motion_refuted
                        else "semantic group validation remains passed"
                    ),
                    "exact CUDA auth eval after lane claim",
                    "component trace comparison against baseline",
                ],
                recommendation=(
                    "Current best post/motion policy is covered by exact-negative deletion evidence; do not dispatch this candidate family until a new policy preserves the measured sensitive groups."
                    if post_motion_refuted
                    else f"Best current protected policy is {best.get('policy_id')}; treat as a narrow exact-eval candidate only."
                    if best
                    else "Refresh post/motion candidate summary before ranking."
                ),
                source_artifacts=[_rel(post_row[0])],
                already_refuted=post_motion_refuted,
                blocked=post_motion_refuted,
                priority=50,
                notes=[
                    f"best_policy={best_policy}",
                    f"exact_negative_count={len(post_motion_exact_negatives)}",
                    *[
                        f"negative={row['path']} score_delta={row['score_delta_vs_baseline']}"
                        for row in post_motion_exact_negatives[:4]
                    ],
                ],
            )
        )

    correction_recode_row = _latest_payload(inputs, "pr85_correction_recode_summary")
    if correction_recode_row:
        correction = correction_recode_row[1]
        best_delta = _as_int(correction.get("best_byte_delta_vs_source_archive"))
        unlocked = correction.get("exact_eval_unlocked") is True
        no_win = not unlocked and best_delta is not None and best_delta >= 0
        opportunities.append(
            _opportunity(
                family_id="decoded_parity_correction_stream_recode",
                surface="PR85 post/motion/randmulti/bias/region decoded-parity recodes",
                bytes_at_stake=abs(best_delta) if best_delta is not None else post_motion_bytes,
                exact_evidence_status=(
                    "empirical_decoded_parity_recode_no_byte_win"
                    if no_win
                    else "candidate_archive_ready_unscored"
                    if unlocked
                    else "planning_only_recode_screen_incomplete"
                ),
                stackability="high if byte-positive because decoded semantics are preserved",
                risks=[
                    "decoded-parity recode cannot improve score unless archive bytes drop",
                    "current runtime-supported correction grammars all selected source bytes",
                ],
                gates=[
                    "byte-positive decoded-parity archive candidate",
                    "semantic SHA parity for every changed stream",
                    "exact CUDA auth eval after lane claim",
                ],
                recommendation=(
                    "Measured runtime-supported decoded-parity recodes are byte-neutral; pursue grammar-changing sidechannel compression or component-benefit atoms."
                    if no_win
                    else "Candidate archive exists; review manifest before exact eval."
                    if unlocked
                    else "Refresh recode screen with complete runtime support."
                ),
                source_artifacts=[_rel(correction_recode_row[0])],
                already_refuted=no_win,
                blocked=not unlocked,
                priority=906 if no_win else 45,
                notes=[
                    f"best_byte_delta_vs_source_archive={best_delta}",
                    f"archive_candidate_count={correction.get('archive_candidate_count')}",
                    f"result_class={correction.get('result_class')}",
                ],
            )
        )

    pair_rows = loaded.get("pr85_pair_atom_readiness", [])
    if pair_rows:
        latest_pair_path, latest_pair = pair_rows[-1]
        action_rows = loaded.get("pr85_pair_action_specs", [])
        latest_action_path, latest_action = action_rows[-1] if action_rows else (None, {})
        dispatch_unlocked = latest_pair.get("dispatch_unlocked") is True
        action_ready_count = _as_int(latest_action.get("ready_for_exact_eval_after_lane_claim_count"))
        action_candidate_count = _as_int(latest_action.get("candidate_count"))
        action_blocker = latest_action.get("blocker_class")
        action_unlocked = latest_action.get("dispatch_unlocked") is True or (
            action_ready_count is not None and action_ready_count > 0
        )
        blockers = latest_pair.get("blockers")
        if not isinstance(blockers, list):
            blocker = latest_pair.get("blocker_class")
            blockers = [blocker] if blocker else []
        if action_blocker and action_blocker != "none":
            blockers = [*blockers, str(action_blocker)]
        top_pairs = latest_pair.get("top_pair_opportunities")
        if not isinstance(top_pairs, list):
            top_pairs = latest_pair.get("top_pairs")
        if not isinstance(top_pairs, list):
            top_pairs = []
        if not top_pairs and isinstance(latest_action.get("candidates"), list):
            for candidate in latest_action["candidates"]:
                if not isinstance(candidate, dict):
                    continue
                selected = candidate.get("selected_pairs")
                if isinstance(selected, list) and selected and isinstance(selected[0], dict):
                    top_pairs = selected
                    break
        best_pair = top_pairs[0] if top_pairs and isinstance(top_pairs[0], dict) else {}
        break_even = _as_float(best_pair.get("break_even_bytes")) if best_pair else None
        if break_even is None and isinstance(best_pair.get("component_signal"), dict):
            break_even = _as_float(best_pair["component_signal"].get("combined_break_even_bytes"))
        opportunities.append(
            _opportunity(
                family_id="scorer_gradient_pair_atom_policy",
                surface="PR85 pair-index scorer-gradient correction atoms",
                bytes_at_stake=int(math.ceil(break_even)) if break_even is not None else None,
                exact_evidence_status=(
                    "candidate_archive_ready_unscored"
                    if dispatch_unlocked or action_unlocked
                    else (
                        "fail_closed_pair_action_lowered_missing_grounded_action_evidence"
                        if action_rows
                        else "fail_closed_planning_only_missing_pair_action_contract"
                    )
                ),
                stackability="medium after contract: pair-local corrections should compose only after exact stacked eval",
                risks=[
                    "scorer-gradient rankings do not define legal stream/value edits",
                    "pair-specific runtime atoms need charged payload bytes and no scorer/runtime sidecars",
                ],
                gates=[
                    "explicit pair-action spec",
                    "reviewed pair-atom runtime contract",
                    "non-noop payload or decoded-output proof",
                    "exact CUDA auth eval after lane claim",
                ],
                recommendation=(
                    "Candidate archives exist; run custody review before exact eval dispatch."
                    if dispatch_unlocked or action_unlocked
                    else (
                        "Pair-action lowering exists but remains blocked; generate grounded stream/value action evidence with a non-noop archive-changing path before dispatch."
                        if action_rows
                        else "Do not dispatch scorer-gradient pair rankings until they are lowered into explicit PR85 stream/value actions or a decoded-output-preserving recode."
                    )
                ),
                source_artifacts=[
                    *[_rel(path) for path, _ in pair_rows],
                    *([_rel(latest_action_path)] if latest_action_path is not None else []),
                ],
                blocked=not (dispatch_unlocked or action_unlocked),
                priority=45,
                notes=[
                    f"latest={_rel(latest_pair_path)}",
                    f"dispatch_unlocked={dispatch_unlocked}",
                    f"pair_action_latest={_rel(latest_action_path) if latest_action_path is not None else None}",
                    f"pair_action_candidate_count={action_candidate_count}",
                    f"pair_action_ready_for_exact_eval_after_lane_claim_count={action_ready_count}",
                    f"blockers={','.join(str(item) for item in blockers)}",
                    f"best_pair={best_pair.get('pair_id') or best_pair.get('pair_index') or best_pair.get('atom_id') if best_pair else None}",
                    f"best_pair_break_even_bytes={break_even}",
                ],
            )
        )

    fb_row = _latest_payload(inputs, "pr85_final_bias_stack_summary")
    if fb_row:
        best = _best_by_delta(_summary_candidates(fb_row[1]), "byte_delta_vs_source_x_archive")
        cost = int(best["byte_delta_vs_source_x_archive"]) if best else None
        opportunities.append(
            _opportunity(
                family_id="pr89_final_bias_stack_on_pr85",
                surface="PR85 plus final-bias sidecar member",
                bytes_at_stake=abs(cost) if cost is not None else None,
                exact_evidence_status="empirical_stack_candidate_no_exact_component_benefit",
                stackability="medium: explicit sidecar-style stack, but requires PR89 inflate family and component benefit proof",
                risks=[
                    "rate-only bound is negative because the stack adds charged bytes",
                    "component benefit has not been proven for this PR85 stack",
                ],
                gates=[
                    "public inflate runtime compatibility",
                    "byte-closed archive with required members",
                    "exact CUDA auth eval proves component benefit exceeds charged bytes",
                ],
                recommendation=(
                    "Do not rank by rate; keep only if exact eval can prove final-bias component gain."
                ),
                source_artifacts=[_rel(fb_row[0])],
                blocked=True,
                priority=70,
                notes=[f"best_candidate={best.get('candidate_id') if best else None}", f"charged_byte_delta={cost}"],
            )
        )

    bridge_row = _latest_payload(inputs, "pr85_bridge_sparse_action_summary")
    if bridge_row:
        candidates = _summary_candidates(bridge_row[1])
        blocked_count = 0
        best_saving: int | None = None
        for row in candidates:
            preflight = row.get("dispatch_preflight", {})
            if isinstance(preflight, dict) and preflight.get("status") == "blocked":
                blocked_count += 1
            deltas = row.get("charged_byte_deltas", {})
            delta = _as_int(deltas.get("archive_delta_bytes_vs_pr85_bridge")) if isinstance(deltas, dict) else None
            if delta is not None and delta < 0:
                saving = -delta
                best_saving = saving if best_saving is None else max(best_saving, saving)
        opportunities.append(
            _opportunity(
                family_id="fixed_runtime_bridge_sparse_action_deletions",
                surface="PR85 fixed-runtime bridge qpost/randmulti sparse actions",
                bytes_at_stake=best_saving,
                exact_evidence_status="preflight_blocked_planning_only",
                stackability="low until protected qpost/randmulti deletion blockers have exact evidence override",
                risks=[
                    "protected qpost groups were deleted without covering exact evidence",
                    "bridge runtime changes confound archive-vs-runtime custody",
                ],
                gates=[
                    "preflight blockers cleared with exact evidence override",
                    "runtime custody manifest",
                    "exact CUDA auth eval on fixed runtime",
                ],
                recommendation="Already blocked by preflight; do not dispatch sparse-action bridge deletions without exact blocker coverage.",
                source_artifacts=[_rel(bridge_row[0])],
                already_refuted=blocked_count == len(candidates) and len(candidates) > 0,
                blocked=True,
                priority=920,
                notes=[f"blocked_candidate_count={blocked_count}", f"candidate_count={len(candidates)}"],
            )
        )

    deletion_routes = [row for row in exact_negatives if row.get("score_delta_vs_baseline", 0) is not None and row.get("score_delta_vs_baseline", 0) > 0]
    if deletion_routes:
        bytes_saved = max(
            (
                ((baseline or {}).get("archive_bytes") or 0) - (row.get("archive_bytes") or 0)
                for row in deletion_routes
            ),
            default=None,
        )
        opportunities.append(
            _opportunity(
                family_id="whole_sidechannel_deletion_routes",
                surface="PR85 whole post/motion/randmulti sidechannel deletion",
                bytes_at_stake=bytes_saved,
                exact_evidence_status="exact_cuda_negative_full_600_samples",
                stackability="none: deletion routes are guardrails, not candidate families",
                risks=["exact eval shows score regression despite byte savings"],
                gates=["no dispatch; convert scoped negatives into guards"],
                recommendation="Already refuted: do not pursue whole sidechannel deletion as a stack plan.",
                source_artifacts=[str(row["path"]) for row in deletion_routes],
                already_refuted=True,
                blocked=True,
                priority=930,
            )
        )

    return sorted(
        opportunities,
        key=lambda row: (
            row["already_refuted"],
            row["priority"],
            -(row["bytes_at_stake"] or 0),
            row["family_id"],
        ),
    )


def _top_stack_plans(opportunities: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    stackable = [
        row
        for row in opportunities
        if not row["already_refuted"]
    ]
    out = []
    for row in stackable[:limit]:
        out.append(
            {
                "rank": len(out) + 1,
                "family_id": row["family_id"],
                "surface": row["surface"],
                "bytes_at_stake": row["bytes_at_stake"],
                "estimated_rate_only_score_bound": row["estimated_rate_only_score_bound"],
                "blocked": row["blocked"],
                "blocked_on": row["required_gates"][0] if row["required_gates"] else None,
                "recommendation": row["recommendation"],
            }
        )
    return out


def _markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# PR85 Full-Stack Opportunity Matrix",
        "",
        "- planning_only: true",
        "- score_claim: false",
        "- dispatch_performed: false",
        "",
        "## Baseline",
    ]
    baseline = matrix.get("baseline_pr85_exact_eval")
    if baseline:
        lines.append(
            f"- PR85 baseline: score {baseline.get('score')} at {baseline.get('archive_bytes')} bytes "
            f"from `{baseline.get('path')}`."
        )
    else:
        lines.append("- PR85 baseline exact eval was not found in the discovered artifacts.")
    lines.extend(["", "## Top Stack Plans", ""])
    lines.append("| Rank | Family | Bytes at stake | Rate-only bound | Blocked | First gate |")
    lines.append("|---:|---|---:|---:|---|---|")
    for row in matrix["top_stack_plans"]:
        lines.append(
            "| {rank} | `{family_id}` | {bytes_at_stake} | {bound} | {blocked} | {gate} |".format(
                rank=row["rank"],
                family_id=row["family_id"],
                bytes_at_stake=row["bytes_at_stake"],
                bound=row["estimated_rate_only_score_bound"],
                blocked=str(row["blocked"]).lower(),
                gate=row["blocked_on"],
            )
        )
    lines.extend(["", "## Opportunity Records", ""])
    lines.append("| Family | Surface | Evidence | Stackability | Recommendation |")
    lines.append("|---|---|---|---|---|")
    for row in matrix["opportunities"]:
        recommendation = str(row["recommendation"]).replace("|", "/")
        lines.append(
            f"| `{row['family_id']}` | {row['surface']} | {row['exact_evidence_status']} | "
            f"{row['stackability']} | {recommendation} |"
        )
    lines.extend(["", "## Refuted Or Blocked Routes", ""])
    for row in matrix["opportunities"]:
        if row["already_refuted"] or row["blocked"]:
            lines.append(f"- `{row['family_id']}`: {row['recommendation']}")
    return "\n".join(lines) + "\n"


def build_matrix(
    *,
    repo_root: Path = REPO_ROOT,
    overrides: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    inputs = discover_inputs(repo_root, overrides=overrides)
    loaded = {category: _load_existing(paths) for category, paths in inputs.items()}
    exact_evals = loaded.get("pr85_exact_eval", [])
    baseline = _best_pr85_baseline(exact_evals)
    exact_negatives = _exact_negative_rows(exact_evals, baseline)
    opportunities = _build_opportunities(inputs, loaded, baseline, exact_evals, exact_negatives)
    matrix: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "deterministic": True,
        "score_rate_formula": {
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "points_per_byte": RATE_POINTS_PER_BYTE,
            "meaning": "rate-only lower-is-better score contribution for bytes saved when components are unchanged",
        },
        "input_inventory": _inventory(repo_root, inputs),
        "baseline_pr85_exact_eval": baseline,
        "exact_negative_sidechannel_evals": exact_negatives,
        "opportunities": opportunities,
        "top_stack_plans": _top_stack_plans(opportunities),
        "refuted_family_ids": [
            row["family_id"] for row in opportunities if row["already_refuted"]
        ],
        "blocked_family_ids": [
            row["family_id"] for row in opportunities if row["blocked"]
        ],
    }
    matrix["stable_matrix_digest_sha256"] = _stable_digest(matrix)
    return matrix


def write_outputs(matrix: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pr85_full_stack_opportunity_matrix.json"
    md_path = out_dir / "pr85_full_stack_opportunity_matrix.md"
    json_path.write_text(json_text(matrix), encoding="utf-8")
    md_path.write_text(_markdown(matrix), encoding="utf-8")
    return {"json": _rel(json_path), "markdown": _rel(md_path)}


def _parse_overrides(values: list[str]) -> dict[str, list[str]]:
    overrides: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise MatrixError("--input entries must be category=path")
        category, path = value.split("=", 1)
        if category not in DEFAULT_GLOBS:
            raise MatrixError(f"unknown input category: {category}")
        overrides.setdefault(category, []).append(path)
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Override a discovered input category with category=path. May be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = _parse_overrides(args.input)
    matrix = build_matrix(repo_root=args.repo_root, overrides=overrides or None)
    outputs = write_outputs(matrix, args.out_dir)
    print(json_text({"outputs": outputs, "stable_matrix_digest_sha256": matrix["stable_matrix_digest_sha256"]}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
