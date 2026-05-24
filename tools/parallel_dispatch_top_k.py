#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Parallel-dispatch actuator for exact-eval-ready ranked candidates.

This tool is intentionally strict. It refuses prediction-only, forensic,
local-proxy, or missing-readiness candidates. The input must contain candidates
already marked `ready_for_exact_eval_dispatch=true` by a separate exact-SHA
readiness gate; this actuator only fans out those already-authorized jobs and
harvests their result JSONL.

The full closed loop:
    1. local: meta_lagrangian_search_cli.py ranks N candidates → top-K
    2. PARALLEL: this script fires K dispatches concurrently
    3. harvest: each dispatch writes contest_auth_eval.json to its result dir
    4. reseed: harvested empirical anchors update .omx/calibration/anchors_*.json
    5. repeat from step 1 with the updated calibration

Prediction sweeps belong upstream as planning-only feedback. They do not become
remote jobs here.

Per CLAUDE.md cost discipline: every dispatch must include `--max-dph` and
`--estimated-cost` so a runaway sweep cannot exceed the operator's budget.

Usage (typical apogee_intN sweep):
    .venv/bin/python tools/meta_lagrangian_search_cli.py \\
        --lane-class apogee_intN --auto-sweep-bits 4,5,6,7,8 \\
        --top-k 16 --output reports/sweep_ranked.json

    .venv/bin/python tools/parallel_dispatch_top_k.py \\
        --ranked-input reports/sweep_ranked.json \\
        --max-concurrency 16 \\
        --provider lightning \\
        --estimated-cost-per-dispatch 0.11 \\
        --max-total-cost 5.00 \\
        --harvest-output reports/sweep_harvested.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.auth_eval_result import parse_auth_eval_score_claim  # noqa: E402
from tac.exact_eval_custody import (  # noqa: E402
    extract_observed_runtime_content_tree_sha256,
    extract_runtime_tree_sha256,
    is_sha256_hex,
)
from tac.hdm8_selector_cuda_gate import validate_hdm8_selector_cuda_gate_context  # noqa: E402
from tac.hnerv_frontier_defaults import (  # noqa: E402
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL,
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE,
    ACTIVE_RATE_ONLY_FLOOR_SCORE,
    ACTIVE_SCORE_FRONTIER_LABEL,
    ACTIVE_SCORE_FRONTIER_SCORE,
)
from tac.optimizer.exact_dispatch_authority import (  # noqa: E402
    ClaimPolicy,
    exact_dispatch_authority,
)
from tac.optimizer.exact_readiness import QUEUE_SCHEMA as EXACT_READY_QUEUE_SCHEMA  # noqa: E402
from tac.optimizer.exact_ready_audit import audit_exact_ready_queue  # noqa: E402
from tac.zipwire_archive import inspect_zip_headers  # noqa: E402


@dataclass
class DispatchResult:
    candidate_id: str
    label: str
    archive_sha256: str | None
    archive_size_bytes: int | None
    started_utc: str
    elapsed_seconds: float
    returncode: int
    stdout_tail: str
    stderr_tail: str
    score_json_path: str | None
    contest_cuda_score: float | None
    score_axis: str | None = None
    score_claim_source_key: str | None = None


_LIGHTNING_DISPATCH = REPO / "tools" / "lightning_dispatch_pr106_stack.py"
_VASTAI_DISPATCH = REPO / "scripts" / "launch_lane_on_vastai.py"
DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES = ACTIVE_FLOOR_ARCHIVE_BYTES
DEFAULT_ACTIVE_RATE_ONLY_FLOOR_SCORE = ACTIVE_RATE_ONLY_FLOOR_SCORE
DEFAULT_ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE = (
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE
)
DEFAULT_ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL = (
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL
)
DEFAULT_ACTIVE_SCORE_FRONTIER_SCORE = ACTIVE_SCORE_FRONTIER_SCORE
DEFAULT_ACTIVE_SCORE_FRONTIER_LABEL = ACTIVE_SCORE_FRONTIER_LABEL
# Backward-compatible flag/default name. Score comparisons use the active
# promotable score frontier; archive-byte comparisons use the separate
# rate-only byte floor. The lower exact-CUDA reference is preserved as
# non-promotional evidence, not as an unqualified dispatch floor.
DEFAULT_ACTIVE_FLOOR_SCORE = DEFAULT_ACTIVE_SCORE_FRONTIER_SCORE
BLOCKED_EVIDENCE_SEMANTICS = {
    "prediction_only_forensic",
    "local_proxy_prediction_forensic",
    "byte_only_forensic",
    "cpu_substrate_predicted_band",
}
BLOCKED_EVIDENCE_MARKERS = (
    "predict",
    "proxy",
    "forensic",
    "mps",
    "cpu-build",
    "cpu_build",
    "cpu build",
    "cpu-only",
    "cpu_only",
    "cpu only",
    "cpu-prep",
    "cpu_prep",
    "cpu prep",
    "cpu research",
    "local-only",
    "local_only",
    "local only",
    "research-signal",
    "research_signal",
    "research signal",
)
EVIDENCE_GUARD_TEXT_KEYS = (
    "evidence_semantics",
    "evidence_grade",
    "evidence_marker",
    "evidence_markers",
    "score_evidence_grade",
    "source",
    "source_text",
    "source_note",
    "source_notes",
    "source_provenance",
    "provenance",
    "notes",
    "tags",
)
CONTEST_DISPATCH_VERDICT_KEYS = (
    "contest_dispatch_verdict",
    "dispatch_verdict",
    "cuda_eval_verdict",
)
BLOCKED_CONTEST_DISPATCH_VERDICT_MARKERS = (
    "deferred",
    "refuse",
    "blocked",
    "do not dispatch",
    "do_not_dispatch",
    "not dispatch",
    "not_dispatch",
    "pending research",
    "pending-research",
    "pending_research",
    "research only",
    "research-only",
    "research_signal",
    "research-signal",
    "proxy",
    "forensic",
    "mps",
    "cpu-build",
    "cpu_build",
    "cpu build",
    "cpu-only",
    "cpu_only",
    "cpu only",
    "cpu-prep",
    "cpu_prep",
    "cpu prep",
    "prediction",
    "predicted",
)
PREDICTED_SCORE_FIELDS = (
    "predicted_score",
    "predicted_score_point_estimate",
    "predicted_score_band",
)
CONTEST_TARGET_MARKERS = {
    "contest_exact_eval",
}
PRODUCTION_TARGET_MARKERS = {
    "production",
    "production_only",
    "comma_ai",
    "comma_ai_production",
    "openpilot",
    "openpilot_edge",
    "production_generalized",
    "production_edge_adaptive",
    "edge",
    "edge_device",
    "outside_contest",
    "outside_contest_mode",
}
SELF_NEURAL_EDGE_MARKERS = (
    "self_compress",
    "self-compress",
    "self_compression",
    "self-compression",
    "neural_compress",
    "neural-compress",
    "neural_compression",
    "neural weight compression",
    "on_device_learning",
    "on-device-learning",
    "edge_learning",
    "edge-learning",
    "online_learning",
)
TRUE_BIT_CHANGE_FIELDS = (
    "score_affecting_payload_changed",
    "charged_bits_changed",
    "bits_changed",
    "payload_changed",
    "byte_different",
    "archive_changed",
    "archive_bytes_changed",
)
NO_OP_FIELDS = (
    "no_op",
    "is_noop",
    "no_op_payload",
    "noop",
)
SHA_DIFF_FIELD_PAIRS = (
    ("source_archive_sha256", "candidate_archive_sha256"),
    ("source_archive_sha256", "archive_sha256"),
    ("input_archive_sha256", "output_archive_sha256"),
    ("old_archive_sha256", "new_archive_sha256"),
    ("source_payload_sha256", "candidate_payload_sha256"),
    ("input_payload_sha256", "output_payload_sha256"),
    ("old_payload_sha256", "new_payload_sha256"),
)
BYTE_DIFF_FIELD_PAIRS = (
    ("source_archive_bytes", "candidate_archive_bytes"),
    ("source_archive_bytes", "archive_size_bytes"),
    ("source_archive_bytes", "expected_archive_size_bytes"),
    ("source_charged_bytes", "candidate_charged_bytes"),
    ("old_archive_bytes", "new_archive_bytes"),
    ("input_archive_bytes", "output_archive_bytes"),
)


class DispatchInputError(ValueError):
    """Raised when ranked input is not exact-eval dispatch-ready."""


def _is_sha256(value: object) -> bool:
    return is_sha256_hex(value)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _candidate_archive_path_value(candidate: dict) -> object:
    for key in ("archive_path", "candidate_archive_path"):
        value = candidate.get(key)
        if value:
            return value
    return None


def _resolve_archive_path(path_value: object, *, ranked_input_dir: Path | None) -> Path | None:
    if isinstance(path_value, Path):
        path = path_value
    elif isinstance(path_value, str) and path_value.strip():
        path = Path(path_value)
    else:
        return None
    if path.is_absolute():
        return path
    candidates = [REPO / path]
    if ranked_input_dir is not None:
        candidates.append(ranked_input_dir / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _candidate_archive_byte_values(candidate: dict) -> dict[str, int]:
    values: dict[str, int] = {}
    for key in (
        "candidate_archive_bytes",
        "archive_size_bytes",
        "expected_archive_size_bytes",
        "archive_bytes",
    ):
        parsed = _coerce_positive_int(candidate.get(key))
        if parsed is not None:
            values[key] = parsed
    return values


def _candidate_archive_bytes(candidate: dict) -> int | None:
    values = _candidate_archive_byte_values(candidate)
    return next(iter(values.values()), None)


def _candidate_archive_sha256(candidate: dict) -> str:
    for key in ("candidate_archive_sha256", "archive_sha256", "expected_archive_sha256"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _candidate_runtime_tree_sha256(candidate: dict) -> str:
    return extract_runtime_tree_sha256(candidate)


def _candidate_runtime_content_tree_sha256(candidate: dict) -> str:
    return extract_observed_runtime_content_tree_sha256(candidate)


def _candidate_exact_score(candidate: dict) -> float | None:
    for key in ("contest_cuda_score", "final_score", "contest_score", "score"):
        parsed = _coerce_float(candidate.get(key))
        if parsed is not None:
            return parsed
    return None


def _flatten_text(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bool | int | float):
        return [str(value)]
    if isinstance(value, dict):
        out: list[str] = []
        for key, inner in value.items():
            out.extend(_flatten_text(key))
            out.extend(_flatten_text(inner))
        return out
    if isinstance(value, list | tuple | set):
        out: list[str] = []
        for inner in value:
            out.extend(_flatten_text(inner))
        return out
    return [str(value)]


def _searchable_text(raw_text: str) -> str:
    text = raw_text.strip().lower()
    variants = {
        text,
        text.replace("_", "-"),
        text.replace("_", " "),
        text.replace("-", "_"),
        text.replace("-", " "),
        text.replace("_", " ").replace("-", " "),
    }
    return " ".join(sorted(variant for variant in variants if variant))


def _candidate_guard_text(candidate: dict, keys: tuple[str, ...]) -> tuple[str, str]:
    parts: list[str] = []
    for key in keys:
        if key in candidate and candidate.get(key) is not None:
            parts.extend(_flatten_text(candidate.get(key)))
    raw_text = " ".join(part.strip().lower() for part in parts if part.strip())
    return raw_text, _searchable_text(raw_text)


def _candidate_text(candidate: dict) -> str:
    keys = (
        "candidate_id",
        "lane_id",
        "lane_class",
        "family",
        "family_group",
        "pareto_scope",
        "op_class",
        "op_module",
        "cathedral_op",
        "optimization_target",
        "target_mode",
        "target_modes",
        "deployment_target",
        "deployment_targets",
        "paradigm",
        "paradigms",
        "tags",
        "notes",
    )
    parts: list[str] = []
    for key in keys:
        parts.extend(_flatten_text(candidate.get(key)))
    return " ".join(parts).strip().lower()


def _normalize_target_token(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _candidate_target_modes(candidate: dict) -> set[str]:
    modes: set[str] = set()
    for key in (
        "optimization_target",
        "target_mode",
        "target_modes",
        "dispatch_target",
        "deployment_target",
        "deployment_targets",
    ):
        for part in _flatten_text(candidate.get(key)):
            token = _normalize_target_token(part)
            if token:
                modes.add(token)
    if candidate.get("contest_mode") is True:
        modes.add("contest")
    if candidate.get("contest_mode") is False:
        modes.add("outside_contest_mode")
    if candidate.get("production_target") is True:
        modes.add("production")
    if candidate.get("openpilot_target") is True:
        modes.add("openpilot")
    if candidate.get("comma_ai_target") is True:
        modes.add("comma_ai_production")
    if candidate.get("outside_contest_mode") is True:
        modes.add("outside_contest_mode")
    return modes


def _candidate_targets_contest_dispatch(candidate: dict) -> bool:
    modes = _candidate_target_modes(candidate)
    if not modes:
        return False
    return bool(modes & CONTEST_TARGET_MARKERS)


def _target_mode_blockers(candidate: dict) -> list[str]:
    modes = _candidate_target_modes(candidate)
    if not modes:
        return [
            "target_modes_missing; parallel_dispatch_top_k requires explicit "
            "contest_exact_eval target metadata for paid dispatch"
        ]
    blockers: list[str] = []
    if not _candidate_targets_contest_dispatch(candidate):
        mode_text = ",".join(sorted(modes))
        blockers.append(
            f"contest_exact_eval_target_mode_missing:{mode_text}; "
            "parallel_dispatch_top_k requires explicit contest_exact_eval "
            "target metadata for paid dispatch"
        )
    if modes & PRODUCTION_TARGET_MARKERS and not _candidate_targets_contest_dispatch(candidate):
        mode_text = ",".join(sorted(modes))
        blockers.append(
            f"non_contest_target_mode:{mode_text}; "
            "parallel_dispatch_top_k only dispatches contest exact-eval archives"
        )
    return blockers


def _candidate_is_self_neural_or_edge_learning(candidate: dict) -> bool:
    text = _candidate_text(candidate)
    return any(marker in text for marker in SELF_NEURAL_EDGE_MARKERS)


def _field_true(mapping: dict, key: str) -> bool | None:
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "1", "passed"}:
            return True
        if token in {"false", "no", "0", "failed"}:
            return False
    return None


def _explicit_no_op(mapping: dict) -> bool:
    for key in NO_OP_FIELDS:
        value = _field_true(mapping, key)
        if value is True:
            return True
    status = str(mapping.get("no_op_status") or "").strip().lower()
    return "no_op" in status or "noop" in status


def _recursive_dicts(value: object) -> list[dict]:
    out: list[dict] = []
    if isinstance(value, dict):
        out.append(value)
        for inner in value.values():
            out.extend(_recursive_dicts(inner))
    elif isinstance(value, list | tuple):
        for inner in value:
            out.extend(_recursive_dicts(inner))
    return out


def _candidate_bits_changed(candidate: dict) -> bool:
    for mapping in _recursive_dicts(candidate):
        if _explicit_no_op(mapping):
            return False
    for mapping in _recursive_dicts(candidate):
        for key in TRUE_BIT_CHANGE_FIELDS:
            value = _field_true(mapping, key)
            if value is True:
                return True
            if value is False and key in {"score_affecting_payload_changed", "charged_bits_changed"}:
                return False
        for old_key, new_key in SHA_DIFF_FIELD_PAIRS:
            old = mapping.get(old_key)
            new = mapping.get(new_key)
            if _is_sha256(old) and _is_sha256(new):
                return str(old).lower() != str(new).lower()
        for old_key, new_key in BYTE_DIFF_FIELD_PAIRS:
            old = _coerce_positive_int(mapping.get(old_key))
            new = _coerce_positive_int(mapping.get(new_key))
            if old is not None and new is not None:
                return old != new
    return False


def _self_neural_edge_blockers(candidate: dict) -> list[str]:
    if not _candidate_is_self_neural_or_edge_learning(candidate):
        return []
    if _candidate_bits_changed(candidate):
        return []
    return [
        "self_neural_edge_candidate_missing_charged_bits_changed_proof:"
        " provide score_affecting_payload_changed/charged_bits_changed=true "
        "or old/new SHA-256 or charged-byte delta proof"
    ]


def _archive_custody_blockers(
    candidate: dict,
    *,
    ranked_input_dir: Path | None,
) -> list[str]:
    blockers: list[str] = []
    path = _resolve_archive_path(
        _candidate_archive_path_value(candidate),
        ranked_input_dir=ranked_input_dir,
    )
    expected_sha = _candidate_archive_sha256(candidate)
    byte_values = _candidate_archive_byte_values(candidate)
    expected_bytes = next(iter(byte_values.values()), None)
    if path is None:
        blockers.append("archive_path_missing")
    elif not path.is_file():
        blockers.append("archive_file_missing")
    if not _is_sha256(expected_sha):
        blockers.append("archive_sha256_missing_or_invalid")
    if expected_bytes is None:
        blockers.append("archive_bytes_missing_or_invalid")
    elif len(set(byte_values.values())) > 1:
        fields = ",".join(f"{key}={value}" for key, value in sorted(byte_values.items()))
        blockers.append(f"archive_bytes_field_mismatch:{fields}")
    if path is not None and path.is_file():
        actual_bytes = path.stat().st_size
        actual_sha = _sha256_file(path)
        if expected_bytes is not None and actual_bytes != expected_bytes:
            blockers.append(f"archive_bytes_mismatch:{actual_bytes}!={expected_bytes}")
        if _is_sha256(expected_sha) and actual_sha != expected_sha:
            blockers.append("archive_sha256_mismatch")
        try:
            zipwire = inspect_zip_headers(path)
        except (OSError, ValueError) as exc:
            blockers.append(f"archive_zip_unreadable:{exc}")
        else:
            if int(zipwire.get("member_count") or 0) == 0:
                blockers.append("archive_zip_empty")
            for name in zipwire.get("duplicate_member_names") or []:
                blockers.append(f"archive_zip_duplicate_member:{name}")
            if zipwire.get("duplicate_member_names"):
                blockers.append("archive_zip_duplicate_members")
            for blocker in zipwire.get("blockers") or []:
                blockers.append(f"archive_zipwire:{blocker}")
                if "unsafe_member_name:" in blocker:
                    blockers.append(f"archive_zip_unsafe_member:{blocker}")
                if "local_central_name_mismatch" in blocker:
                    blockers.append(f"archive_zip_local_header_mismatch:{blocker}")
            if zipwire.get("zip_strict") is not True and not (zipwire.get("blockers") or []):
                blockers.append("archive_zipwire:not_strict_without_blockers")
    return blockers


def _runtime_custody_blockers(candidate: dict) -> list[str]:
    if (
        candidate.get("ready_for_exact_eval_dispatch") is not True
        and not _candidate_targets_contest_dispatch(candidate)
    ):
        return []
    runtime_tree_sha256 = _candidate_runtime_tree_sha256(candidate)
    if not _is_sha256(runtime_tree_sha256):
        return ["runtime_tree_sha256_missing_or_invalid"]
    runtime_content_tree_sha256 = _candidate_runtime_content_tree_sha256(candidate)
    if not _is_sha256(runtime_content_tree_sha256):
        return ["runtime_content_tree_sha256_missing_or_invalid"]
    return []


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _authority_repo_root_for_candidate(
    candidate: dict,
    ranked_input_dir: Path | None,
) -> Path:
    """Use a detached queue root when the ranked input carries its own runtime.

    Production queues normally live inside this checkout and should be checked
    against ``REPO``. For forensic or fixture bundles, the queue directory can
    be the custody root containing ``upstream/evaluate.py`` and relative
    submission paths. In that case recomputing the runtime hash against this
    checkout would create a false mismatch.
    """

    if ranked_input_dir is None:
        return REPO
    queue_root = ranked_input_dir.resolve()
    if not (queue_root / "upstream" / "evaluate.py").is_file():
        return REPO
    for key in (
        "archive_path",
        "candidate_archive_path",
        "submission_dir",
        "submission_path",
        "archive_manifest_path",
        "manifest_path",
        "runtime_packet_manifest_path",
    ):
        value = candidate.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        queue_path = path if path.is_absolute() else queue_root / path
        if queue_path.exists() and _path_is_relative_to(queue_path, queue_root):
            return queue_root
    return REPO


def _candidate_blockers(
    candidate: dict,
    *,
    ranked_input_dir: Path | None = None,
    active_floor_archive_bytes: int | None = DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = DEFAULT_ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
    dispatch_claims_path: Path | None = REPO / ".omx" / "state" / "active_lane_dispatch_claims.md",
    claim_policy: ClaimPolicy = "preclaim_conflict_check",
    required_claim_platform: str | None = None,
    required_claim_instance_job_ids: tuple[str, ...] = (),
) -> list[str]:
    blockers: list[str] = []
    # Round 5 R5-1 fix (2026-05-06, 95% CRITICAL): the historical
    # `if candidate.get("dispatch_blockers"): blockers.append(...)` rejected
    # any candidate with a non-empty list. Round 4 R4-B documented that
    # `build_wavelet_apply_gate` and `build_wavelet_apply_transform_candidate`
    # both emit fail-closed-by-design dispatch_blockers (4+ unconditional
    # entries) — operators must clear those entries by hand after providing
    # the corresponding evidence. So a non-empty list is NOT itself a
    # blocker; the canonical clearance signal is `ready_for_exact_eval_dispatch
    # == True`. That signal is already checked above. The dispatch_blockers
    # list is informational ("next required evidence") and must not be the
    # gating predicate for the actuator.
    if candidate.get("ready_for_exact_eval_dispatch") is not True:
        blockers.append("candidate_not_ready_for_exact_eval_dispatch")
    authority = exact_dispatch_authority(
        candidate,
        repo_root=_authority_repo_root_for_candidate(candidate, ranked_input_dir),
        queue_dir=ranked_input_dir,
        source="parallel_dispatch_top_k",
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_score=active_floor_score,
        allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
        operator_override_reason=operator_override_reason,
        dispatch_claims_path=dispatch_claims_path,
        claim_policy=claim_policy,
        required_claim_platform=required_claim_platform,
        required_claim_instance_job_ids=required_claim_instance_job_ids,
        required_score_axis="contest_cuda",
    )
    blockers.extend(
        f"exact_dispatch_authority:{blocker}"
        for blocker in authority.blockers
    )
    gate_blockers, _gate_facts = validate_hdm8_selector_cuda_gate_context(
        candidate,
        None,
        expected_archive_sha256=_candidate_archive_sha256(candidate) or None,
    )
    blockers.extend(
        f"hdm8_selector_cuda_component_gate:{blocker}"
        for blocker in gate_blockers
    )
    if not str(candidate.get("evidence_semantics") or "").strip():
        blockers.append("evidence_semantics_missing")
    evidence_text, searchable_evidence_text = _candidate_guard_text(
        candidate,
        EVIDENCE_GUARD_TEXT_KEYS,
    )
    if (
        evidence_text in BLOCKED_EVIDENCE_SEMANTICS
        or searchable_evidence_text in BLOCKED_EVIDENCE_SEMANTICS
        or any(marker in searchable_evidence_text for marker in BLOCKED_EVIDENCE_MARKERS)
    ):
        blockers.append(f"blocked_evidence_semantics:{evidence_text or 'missing'}")
    verdict_text, searchable_verdict_text = _candidate_guard_text(
        candidate,
        CONTEST_DISPATCH_VERDICT_KEYS,
    )
    if any(
        marker in searchable_verdict_text
        for marker in BLOCKED_CONTEST_DISPATCH_VERDICT_MARKERS
    ):
        blockers.append(f"blocked_contest_dispatch_verdict:{verdict_text or 'missing'}")
    predicted_score_fields = [key for key in PREDICTED_SCORE_FIELDS if key in candidate]
    if predicted_score_fields:
        blockers.append(
            "predicted_score_field_present:"
            + ",".join(sorted(predicted_score_fields))
        )
    if candidate.get("score_claim") is True and candidate.get("score_claim_verified") is not True:
        blockers.append("unverified_score_claim")
    blockers.extend(_target_mode_blockers(candidate))
    blockers.extend(_self_neural_edge_blockers(candidate))
    blockers.extend(
        f"archive_custody:{blocker}"
        for blocker in _archive_custody_blockers(
            candidate,
            ranked_input_dir=ranked_input_dir,
        )
    )
    blockers.extend(
        f"runtime_custody:{blocker}"
        for blocker in _runtime_custody_blockers(candidate)
    )
    archive_bytes = _candidate_archive_bytes(candidate)
    if (
        active_floor_archive_bytes is not None
        and archive_bytes is not None
        and archive_bytes > active_floor_archive_bytes
    ):
        if not allow_above_active_floor_dispatch:
            score_note = (
                f", active_score_frontier={active_floor_score:.12f}"
                if active_floor_score is not None else ""
            )
            blockers.append(
                "above_active_floor_archive_bytes:"
                f"{archive_bytes}>{active_floor_archive_bytes}{score_note}; "
                "above rate-only byte floor; treat as research/calibration unless "
                "explicitly overridden"
            )
        elif not operator_override_reason:
            blockers.append("above_active_floor_override_missing_reason")
    exact_score = _candidate_exact_score(candidate)
    if (
        active_floor_score is not None
        and exact_score is not None
        and exact_score > active_floor_score
    ):
        if not allow_above_active_floor_dispatch:
            blockers.append(
                "above_active_floor_score:"
                f"{exact_score:.12f}>{active_floor_score:.12f}; "
                "above active score frontier; treat as research/calibration unless "
                "explicitly overridden"
            )
        elif not operator_override_reason:
            blockers.append("above_active_floor_override_missing_reason")
    return blockers


def _build_dispatch_cmd(
    candidate: dict,
    *,
    provider: str,
    lane_script: str,
    label_prefix: str,
    estimated_cost: float,
    max_dph: float,
    dispatch_claims_path: Path | None = REPO / ".omx" / "state" / "active_lane_dispatch_claims.md",
    claim_policy: ClaimPolicy = "preclaim_conflict_check",
    required_claim_instance_job_ids: tuple[str, ...] = (),
) -> list[str]:
    candidate_id = candidate["candidate_id"]
    required_claim_job_id = _dispatch_job_id_for_candidate(
        candidate,
        required_claim_instance_job_ids=required_claim_instance_job_ids,
    )
    label = required_claim_job_id or f"{label_prefix}_{candidate_id}"
    band = candidate.get("predicted_band", [candidate.get("band_low", 0.0), candidate.get("band_high", 1.0)])

    if provider == "lightning":
        if not _LIGHTNING_DISPATCH.is_file():
            raise FileNotFoundError(f"missing lightning dispatcher: {_LIGHTNING_DISPATCH}")
        archive_path = _candidate_archive_path_value(candidate)
        if not archive_path:
            raise DispatchInputError(
                f"candidate {candidate_id!r} missing archive_path/candidate_archive_path for Lightning dispatch"
            )
        cmd = [
            sys.executable, str(_LIGHTNING_DISPATCH),
            "--lane", str(candidate.get("lane_id") or candidate_id),
            "--archive", str(archive_path),
            "--predicted-low", str(band[0]),
            "--predicted-high", str(band[1]),
            "--job-name", label,
            "--dispatch-lane-id", str(candidate.get("lane_id") or candidate_id),
        ]
        if dispatch_claims_path is not None:
            cmd += ["--dispatch-claims-path", str(dispatch_claims_path)]
        if claim_policy == "require_active_claim":
            cmd.append("--use-existing-dispatch-claim")
        gate_json = candidate.get("apogee_distortion_gate_json")
        if gate_json:
            cmd += ["--apogee-distortion-gate-json", str(gate_json)]
        _ = lane_script
        return cmd

    if provider == "vastai":
        if not _VASTAI_DISPATCH.is_file():
            raise FileNotFoundError(f"missing vastai dispatcher: {_VASTAI_DISPATCH}")
        return [
            sys.executable, str(_VASTAI_DISPATCH), "full",
            "--lane-script", lane_script,
            "--label", label,
            "--predicted-band", str(band[0]), str(band[1]),
            "--estimated-cost", str(estimated_cost),
            "--council-priority", "1",
            "--max-dph", str(max_dph),
        ]

    raise ValueError(f"unknown provider: {provider} (expected: lightning | vastai)")


def _dispatch_job_id_for_candidate(
    candidate: dict,
    *,
    required_claim_instance_job_ids: tuple[str, ...],
) -> str | None:
    allowed = [job_id.strip() for job_id in required_claim_instance_job_ids if job_id.strip()]
    for key in ("dispatch_job_id", "required_claim_instance_job_id", "job_name"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            job_id = value.strip()
            if not allowed or job_id in allowed:
                return job_id
    if len(allowed) == 1:
        return allowed[0]
    if allowed:
        raise DispatchInputError(
            "require_active_claim with multiple required job ids needs "
            "candidate.dispatch_job_id or candidate.required_claim_instance_job_id"
        )
    return None


def _current_run_auth_eval_candidates(label: str, *, started_unix: float) -> list[Path]:
    """Return label-bound auth-eval JSON files written by this dispatch run."""

    candidates = sorted(REPO.glob(f"experiments/results/*{label}*/contest_auth_eval.json"))
    out: list[Path] = []
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        # A small slack avoids filesystem timestamp edge cases while still
        # rejecting stale artifacts from earlier runs with the same label.
        if mtime >= started_unix - 2.0:
            out.append(path)
    return out


def _parse_current_contest_cuda_score(path: Path) -> tuple[float, str, str] | None:
    """Parse a score only when the auth-eval claim contract authorizes CUDA."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    claim = parse_auth_eval_score_claim(
        payload,
        required_score_axis="contest_cuda",
        require_component_recompute=True,
    )
    if claim is None:
        return None
    return claim.score, claim.score_axis, claim.source_key


def _harvest_current_contest_cuda_score(
    label: str,
    *,
    started_unix: float,
) -> tuple[str | None, float | None, str | None, str | None]:
    """Find a current-run auth-eval artifact and return validated CUDA score."""

    score_json_path: str | None = None
    for path in _current_run_auth_eval_candidates(label, started_unix=started_unix):
        score_json_path = str(path)
        parsed = _parse_current_contest_cuda_score(path)
        if parsed is None:
            continue
        score, score_axis, source_key = parsed
        return score_json_path, score, score_axis, source_key
    return score_json_path, None, None, None


def _harvest_tag(result: DispatchResult) -> str:
    if result.returncode != 0:
        return "[dispatch-failed]"
    if result.contest_cuda_score is not None and result.score_axis == "contest_cuda":
        return "[contest-CUDA]"
    return "[dispatch-completed-no-contest-cuda-score]"


def _fire_one(
    candidate: dict,
    *,
    provider: str,
    lane_script: str,
    label_prefix: str,
    estimated_cost: float,
    max_dph: float,
    timeout_seconds: float,
    dispatch_claims_path: Path | None = REPO / ".omx" / "state" / "active_lane_dispatch_claims.md",
    claim_policy: ClaimPolicy = "preclaim_conflict_check",
    required_claim_instance_job_ids: tuple[str, ...] = (),
) -> DispatchResult:
    candidate_id = candidate["candidate_id"]
    label = (
        _dispatch_job_id_for_candidate(
            candidate,
            required_claim_instance_job_ids=required_claim_instance_job_ids,
        )
        or f"{label_prefix}_{candidate_id}"
    )
    cmd = _build_dispatch_cmd(
        candidate,
        provider=provider, lane_script=lane_script,
        label_prefix=label_prefix,
        estimated_cost=estimated_cost, max_dph=max_dph,
        dispatch_claims_path=dispatch_claims_path,
        claim_policy=claim_policy,
        required_claim_instance_job_ids=required_claim_instance_job_ids,
    )
    started_unix = time.time()
    started = time.gmtime(started_unix)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", started)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # subprocess-no-check-OK: we capture rc explicitly + log to harvested JSONL
            cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False,
            cwd=str(REPO),
        )
        rc = proc.returncode
        stdout_tail = (proc.stdout or "")[-2000:]
        stderr_tail = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired as exc:
        rc = -1
        stdout_tail = (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = f"TIMEOUT after {timeout_seconds}s"
    elapsed = time.monotonic() - t0

    # Harvest contest_auth_eval.json from the lane's expected output directory.
    # This is a score-authority boundary: raw finite scores are not enough.
    # Only the canonical auth-eval claim parser may mint a [contest-CUDA] score.
    (
        score_json_path,
        contest_cuda_score,
        score_axis,
        score_claim_source_key,
    ) = _harvest_current_contest_cuda_score(label, started_unix=started_unix)

    return DispatchResult(
        candidate_id=candidate_id,
        label=label,
        archive_sha256=_candidate_archive_sha256(candidate) or None,
        archive_size_bytes=_candidate_archive_bytes(candidate),
        started_utc=started_utc,
        elapsed_seconds=elapsed,
        returncode=rc,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        score_json_path=score_json_path,
        contest_cuda_score=contest_cuda_score,
        score_axis=score_axis,
        score_claim_source_key=score_claim_source_key,
    )


def _load_top_k(
    ranked_input: Path,
    k: int | None,
    *,
    dispatch_claims_path: Path | None = REPO / ".omx" / "state" / "active_lane_dispatch_claims.md",
    active_floor_archive_bytes: int | None = DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = DEFAULT_ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
    claim_policy: ClaimPolicy = "preclaim_conflict_check",
    required_claim_platform: str | None = None,
    required_claim_instance_job_ids: tuple[str, ...] = (),
) -> list[dict]:
    """Load candidates from a meta-Lagrangian ranked-output JSON file."""
    ranked_input = ranked_input.resolve()
    payload = json.loads(ranked_input.read_text())
    if isinstance(payload, dict) and payload.get("ready_for_exact_eval_dispatch") is False:
        raise DispatchInputError(
            f"{ranked_input} is marked ready_for_exact_eval_dispatch=false; refusing parallel dispatch"
        )
    if isinstance(payload, dict) and payload.get("schema") == EXACT_READY_QUEUE_SCHEMA:
        candidates = payload.get("dispatch_ready")
        if not isinstance(candidates, list):
            raise DispatchInputError(
                f"{ranked_input} exact-ready queue missing dispatch_ready list"
            )
        declared_ready_count = payload.get("dispatch_ready_count")
        if declared_ready_count != len(candidates):
            raise DispatchInputError(
                f"{ranked_input} exact-ready dispatch_ready_count mismatch: "
                f"{declared_ready_count!r}!={len(candidates)}"
            )
        if not candidates:
            raise DispatchInputError(
                f"{ranked_input} exact-ready queue has no dispatch_ready rows; "
                "refusing top_k fallback"
            )
    else:
        candidates = (
            payload.get("dispatch_ready") or payload.get("top_k")
            if isinstance(payload, dict)
            else payload
        )
    if not isinstance(candidates, list):
        raise ValueError(f"ranked-input must contain a top_k or dispatch_ready list, got {type(candidates)}")
    if k is not None:
        candidates = candidates[:k]
    selected_candidate_ids = [
        str(candidate.get("candidate_id"))
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("candidate_id") is not None
    ]
    blocked: list[str] = []
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            blocked.append(f"candidate[{idx}]: not an object")
            continue
        candidate_id = candidate.get("candidate_id", f"candidate[{idx}]")
        for blocker in _candidate_blockers(
            candidate,
            ranked_input_dir=ranked_input.parent,
            active_floor_archive_bytes=active_floor_archive_bytes,
            active_floor_score=active_floor_score,
            allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
            operator_override_reason=operator_override_reason,
            dispatch_claims_path=dispatch_claims_path,
            claim_policy=claim_policy,
            required_claim_platform=required_claim_platform,
            required_claim_instance_job_ids=required_claim_instance_job_ids,
        ):
            blocked.append(f"{candidate_id}: {blocker}")
    if blocked:
        details = "\n  - ".join(blocked[:20])
        raise DispatchInputError(
            "ranked-input contains non-dispatch-ready candidates; refusing paid dispatch:\n  - "
            + details
        )
    if (
        selected_candidate_ids
        and dispatch_claims_path is not None
        and isinstance(payload, dict)
    ):
        try:
            ranked_input.relative_to(REPO.resolve())
            audit_repo_root = REPO
        except ValueError:
            audit_repo_root = ranked_input.parent
        audit = audit_exact_ready_queue(
            ranked_input,
            repo_root=audit_repo_root,
            dispatch_claims_path=dispatch_claims_path,
            active_floor_score=active_floor_score,
            candidate_ids=selected_candidate_ids,
            allowed_active_claim_platform=required_claim_platform
            if claim_policy == "require_active_claim"
            else None,
            allowed_active_claim_instance_job_ids=required_claim_instance_job_ids
            if claim_policy == "require_active_claim"
            else (),
        )
        stale_rows = audit.get("stale_ready_rows")
        if isinstance(stale_rows, list) and stale_rows:
            details: list[str] = []
            for row in stale_rows[:10]:
                if not isinstance(row, dict):
                    continue
                candidate_id = row.get("candidate_id")
                lane_id = row.get("lane_id")
                blockers = row.get("blockers")
                blocker_text = (
                    ",".join(str(blocker) for blocker in blockers[:6])
                    if isinstance(blockers, list)
                    else str(blockers)
                )
                details.append(f"{candidate_id}:{lane_id}:{blocker_text}")
            raise DispatchInputError(
                "ranked-input exact-ready audit failed; refusing paid dispatch:\n  - "
                + "\n  - ".join(details)
            )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-input", type=Path, required=True,
                        help="JSON output from tools/meta_lagrangian_search_cli.py")
    parser.add_argument("--top-k", type=int, default=None,
                        help="cap dispatch count; defaults to full ranked list length")
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="max simultaneous dispatches (default: 8). Pin to your provider's quota.")
    parser.add_argument("--provider", choices=["lightning", "vastai"], default="lightning")
    parser.add_argument("--lane-script", default="scripts/remote_lane_apogee_intN.sh",
                        help="path to the remote lane script (relative to repo root)")
    parser.add_argument("--label-prefix", default="parallel_sweep",
                        help="label prefix for each dispatch (used to namespace result dirs)")
    parser.add_argument("--estimated-cost-per-dispatch", type=float, default=0.30,
                        help="$ estimate per dispatch for budget gating")
    parser.add_argument("--max-total-cost", type=float, default=5.00,
                        help="hard cap on total $ across all dispatches; refuse if exceeded")
    parser.add_argument("--active-floor-archive-bytes", type=int, default=DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES,
                        help="refuse paid dispatch for rate-only candidates larger than the current "
                        "rate-only archive-byte floor unless --allow-above-active-floor-dispatch is set "
                        f"(default: {DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES})")
    parser.add_argument("--active-floor-score", type=float, default=DEFAULT_ACTIVE_FLOOR_SCORE,
                        help="active score frontier for terminal/exact-score routing; "
                        "kept as --active-floor-score for CLI compatibility "
                        f"(default: {DEFAULT_ACTIVE_FLOOR_SCORE:.12f})")
    parser.add_argument("--allow-above-active-floor-dispatch", action="store_true",
                        help="operator override for calibration/non-rate experiments whose archives "
                        "are larger than --active-floor-archive-bytes")
    parser.add_argument("--operator-override-reason", default=None,
                        help="required with --allow-above-active-floor-dispatch")
    parser.add_argument("--dispatch-claims-path", type=Path,
                        default=REPO / ".omx/state/active_lane_dispatch_claims.md",
                        help="lane-claim ledger used by the exact-ready audit before paid dispatch")
    parser.add_argument("--claim-policy",
                        choices=["preclaim_conflict_check", "require_active_claim"],
                        default="preclaim_conflict_check",
                        help="preclaim mode refuses active same-lane claims; require-active mode is for claim-then-dispatch queues")
    parser.add_argument("--required-claim-platform", default=None,
                        help="Optional platform required when --claim-policy=require_active_claim")
    parser.add_argument("--required-claim-instance-job-id", action="append", default=[],
                        help="Optional active claim job id allowed when --claim-policy=require_active_claim; may be repeated")
    parser.add_argument("--max-dph", type=float, default=0.30,
                        help="passed to vastai dispatcher to gate per-hour cost")
    parser.add_argument("--per-dispatch-timeout-seconds", type=float, default=1800.0,
                        help="kill any individual dispatch after this many seconds (default 30min)")
    parser.add_argument("--harvest-output", type=Path, default=None,
                        help="write harvested-results JSONL to this path (one DispatchResult per line)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the dispatch commands that WOULD fire, without firing them")
    args = parser.parse_args(argv)
    if args.allow_above_active_floor_dispatch and not args.operator_override_reason:
        print(
            "FATAL: --allow-above-active-floor-dispatch requires "
            "--operator-override-reason",
            file=sys.stderr,
        )
        return 2
    if args.provider == "vastai" and not args.dry_run:
        print(
            "FATAL: provider=vastai is disabled for exact-ready paid fan-out "
            "until scripts/launch_lane_on_vastai.py owns a mandatory "
            "claim_lane_dispatch.py pre-instance claim and terminal claim update. "
            "Use dry-run only or a provider launcher with enforced lane claims.",
            file=sys.stderr,
        )
        return 2

    try:
        candidates = _load_top_k(
            args.ranked_input,
            args.top_k,
            dispatch_claims_path=args.dispatch_claims_path,
            active_floor_archive_bytes=args.active_floor_archive_bytes,
            active_floor_score=args.active_floor_score,
            allow_above_active_floor_dispatch=args.allow_above_active_floor_dispatch,
            operator_override_reason=args.operator_override_reason,
            claim_policy=args.claim_policy,
            required_claim_platform=args.required_claim_platform,
            required_claim_instance_job_ids=tuple(args.required_claim_instance_job_id),
        )
    except DispatchInputError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    if not candidates:
        print(f"FATAL: no candidates in {args.ranked_input}", file=sys.stderr)
        return 2

    n = len(candidates)
    estimated_total = args.estimated_cost_per_dispatch * n
    if estimated_total > args.max_total_cost:
        print(
            f"FATAL: estimated total ${estimated_total:.2f} (={n}x${args.estimated_cost_per_dispatch}) "
            f"exceeds --max-total-cost ${args.max_total_cost:.2f}. "
            f"Lower --top-k or raise --max-total-cost.",
            file=sys.stderr,
        )
        return 3

    print(f"[parallel-dispatch] {n} candidates, max_concurrency={args.max_concurrency}, provider={args.provider}")
    print(f"[parallel-dispatch] estimated total cost: ${estimated_total:.2f} (cap ${args.max_total_cost:.2f})")
    print(f"[parallel-dispatch] timeout per dispatch: {args.per_dispatch_timeout_seconds}s")

    if args.dry_run:
        print("[parallel-dispatch] DRY-RUN — printing commands only:")
        for c in candidates:
            cmd = _build_dispatch_cmd(
                c, provider=args.provider, lane_script=args.lane_script,
                label_prefix=args.label_prefix,
                estimated_cost=args.estimated_cost_per_dispatch,
                max_dph=args.max_dph,
                dispatch_claims_path=args.dispatch_claims_path,
                claim_policy=args.claim_policy,
                required_claim_instance_job_ids=tuple(
                    args.required_claim_instance_job_id
                ),
            )
            print("  " + " ".join(cmd))
        return 0

    results: list[DispatchResult] = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        futures = {
            ex.submit(
                _fire_one, c,
                provider=args.provider, lane_script=args.lane_script,
                label_prefix=args.label_prefix,
                estimated_cost=args.estimated_cost_per_dispatch,
                max_dph=args.max_dph,
                timeout_seconds=args.per_dispatch_timeout_seconds,
                dispatch_claims_path=args.dispatch_claims_path,
                claim_policy=args.claim_policy,
                required_claim_instance_job_ids=tuple(
                    args.required_claim_instance_job_id
                ),
            ): c["candidate_id"]
            for c in candidates
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover — defensive
                print(f"[parallel-dispatch] EXCEPTION dispatching {cid}: {exc}", file=sys.stderr)
                continue
            results.append(result)
            score_str = (
                f"score={result.contest_cuda_score:.4f}"
                if result.contest_cuda_score is not None else "score=PENDING"
            )
            symbol = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
            print(
                f"[parallel-dispatch] [{symbol}] {result.candidate_id} "
                f"elapsed={result.elapsed_seconds:.1f}s {score_str}"
            )

    elapsed_total = time.monotonic() - t0
    n_ok = sum(1 for r in results if r.returncode == 0)
    n_with_score = sum(1 for r in results if r.contest_cuda_score is not None)
    print(
        f"[parallel-dispatch] DONE — {n_ok}/{len(results)} dispatches succeeded "
        f"({n_with_score} with parsed contest-CUDA score) in {elapsed_total:.1f}s wall-clock"
    )

    if args.harvest_output:
        args.harvest_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.harvest_output, "w") as f:
            for r in results:
                f.write(json.dumps({
                    "candidate_id": r.candidate_id,
                    "label": r.label,
                    "archive_sha256": r.archive_sha256,
                    "archive_size_bytes": r.archive_size_bytes,
                    "started_utc": r.started_utc,
                    "elapsed_seconds": r.elapsed_seconds,
                    "returncode": r.returncode,
                    "stdout_tail": r.stdout_tail,
                    "stderr_tail": r.stderr_tail,
                    "score_json_path": r.score_json_path,
                    "contest_cuda_score": r.contest_cuda_score,
                    "score_axis": r.score_axis,
                    "score_claim_source_key": r.score_claim_source_key,
                    "tag": _harvest_tag(r),
                }) + "\n")
        print(f"[parallel-dispatch] harvested → {args.harvest_output}")

    # Best score in the batch (lower is better for comma's contest)
    successful_scores = [(r.candidate_id, r.contest_cuda_score) for r in results if r.contest_cuda_score is not None]
    if successful_scores:
        successful_scores.sort(key=lambda t: t[1])
        best_id, best_score = successful_scores[0]
        print(f"[parallel-dispatch] best in batch: {best_id} = {best_score:.4f} [contest-CUDA]")

    return 0 if n_ok > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
