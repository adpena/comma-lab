# SPDX-License-Identifier: MIT
"""Exact-eval readiness gate for optimizer candidate queues.

Optimizer, Optuna, CMA-ES, Kaggle, M5, and local proxy rows are useful for
ranking. They are not dispatch authority until a separate byte-closed archive
and runtime custody gate proves that the candidate can enter the canonical
``archive.zip -> inflate.sh -> upstream/evaluate.py`` path.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.hdm8_selector_cuda_gate import validate_hdm8_selector_cuda_gate_context
from tac.hnerv_frontier_defaults import (
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL,
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE,
    ACTIVE_RATE_ONLY_FLOOR_SCORE,
    ACTIVE_SCORE_FRONTIER_LABEL,
    ACTIVE_SCORE_FRONTIER_SCORE,
)
from tac.optimization.family_agnostic_materializers import (
    verify_renderer_payload_dfl1_full_frame_inflate_parity_proof,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.optimization.serialized_archive_economics import (
    CANDIDATE_ARCHIVE_LARGER_BLOCKER,
    MISSING_ARCHIVE_BYTES_BLOCKER,
    MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER,
    SERIALIZED_ARCHIVE_DELTA_SCHEMA,
    SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER,
    serialized_archive_delta_blockers,
)
from tac.zipwire_archive import inspect_zip_headers

QUEUE_SCHEMA = "optimizer_candidate_exact_eval_ready_queue_v1"
TOOL_NAME = "tools/promote_optimizer_candidate_for_exact_eval.py"
PR101_RUNTIME_CONSUMPTION_PROOF_SCHEMA = "pr101_kaggle_proxy_runtime_consumption_proof_v1"
FAMILY_AGNOSTIC_RUNTIME_CONSUMPTION_PROOF_SCHEMAS = frozenset(
    {
        "family_agnostic_runtime_consumption_proof_v1",
        "family_agnostic_runtime_consumption_proof_verification.v1",
    }
)
# Backward-compatible name used by dispatch gates for score comparisons.
# Score comparisons use the active promotable CUDA frontier. The lower
# exact-CUDA reference is preserved above as non-promotional evidence and must
# not silently become the dispatch floor. Archive-byte comparisons still use
# ACTIVE_FLOOR_ARCHIVE_BYTES, which is the separate rate-only PR103-on-PR106
# byte floor.
ACTIVE_FLOOR_SCORE = ACTIVE_SCORE_FRONTIER_SCORE
SHA256_HEX = frozenset("0123456789abcdef")
PREDICTED_SCORE_FIELDS = frozenset(
    {
        "predicted_score",  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
        "predicted_score_point_estimate",
        "predicted_score_band",
        "predicted_contest_cpu_gha",
        "proxy_score",
        "macos_cpu_score",
        "rank_score",
        "fitness",
        "proxy_objective",
        "cpu_score",
        "contest_cpu_score",
        "contest_cuda_score",
        "predicted_contest_cuda",
        "predicted_contest_cuda_score",
    }
)
CLEARABLE_SOURCE_BLOCKERS = frozenset(
    {
        "optimizer_candidate_queue_is_planning_only",
        "requires_exact_eval_readiness_gate",
        "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
        "requires_non_proxy_score_evidence_before_promotion",
        "exact_cuda_auth_eval_missing",
        "runtime_tree_sha_required_before_exact_dispatch",
        "a1_runtime_variant_requires_cpu_or_cuda_eval",
        "archive_bytes_unchanged_score_depends_on_inflate_runtime",
        "gha_eval_required_before_exact_cuda_promotion",
        "operator_decision_required_before_gha_or_exact_cuda",
        "macos_cpu_is_not_contest_cuda_evidence",
    }
)
BLOCKED_SOURCE_BLOCKER_PREFIXES = (
    "candidate_archive_missing",
    "codec_op_payload_not_archive_zip",
    "archive_substitution_surgery_required",
    "kaggle_proxy",
    "no_archive_zip_emitted",
    "proxy_score_is_not_score_evidence",
    "predicted_score_is_not_score_evidence",
)
TERMINAL_CLAIM_PREFIXES = (
    "completed_",
    "failed_",
    "timed_out",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
    "falsified_",
    "retired_",
    "config_retired_",
    "measured_implementation_retired_",
    "stop_attempt_timeout_duplicate_after_primary_negative",
)
TERMINAL_NEGATIVE_STATUS_MARKERS = (
    "negative",
    "falsified",
    "retired",
    "component_collapse",
)
FLOAT_TEXT_RE = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
TERMINAL_SCORE_RE = re.compile(
    rf"(?:score_recomputed|score|canonical_score)=({FLOAT_TEXT_RE})"
)
TERMINAL_RUNTIME_TREE_SHA_RE = re.compile(
    r"(?:runtime_tree_sha256|runtime_tree_sha)=([0-9a-fA-F]{64})"
)
TERMINAL_RUNTIME_CONTENT_TREE_SHA_RE = re.compile(
    r"(?:runtime_content_tree_sha256|runtime_content_tree_sha)=([0-9a-fA-F]{64})"
)
TRUE_CHANGE_FIELDS = (
    "score_affecting_payload_changed",
    "charged_bits_changed",
    "score_affecting_runtime_changed",
    "byte_different",
    "archive_changed",
)
SCORE_AFFECTING_PROOF_OBJECT_KEYS = (
    "score_affecting_change_proof",
    "payload_diff_proof",
    "archive_diff_proof",
    "runtime_diff_proof",
    "byte_diff_proof",
)
NO_OP_FIELDS = ("no_op", "is_noop", "no_op_payload", "noop")
SHA_DIFF_FIELD_PAIRS = (
    ("source_archive_sha256", "candidate_archive_sha256"),
    ("source_archive_sha256", "archive_sha256"),
    ("input_archive_sha256", "output_archive_sha256"),
    ("old_archive_sha256", "new_archive_sha256"),
    ("source_payload_sha256", "candidate_payload_sha256"),
    ("input_payload_sha256", "output_payload_sha256"),
    ("old_payload_sha256", "new_payload_sha256"),
    ("source_runtime_tree_sha256", "runtime_tree_sha256"),
)
BYTE_DIFF_FIELD_PAIRS = (
    ("source_archive_bytes", "candidate_archive_bytes"),
    ("source_archive_bytes", "archive_size_bytes"),
    ("source_archive_bytes", "archive_bytes"),
    ("source_charged_bytes", "candidate_charged_bytes"),
    ("old_archive_bytes", "new_archive_bytes"),
    ("input_archive_bytes", "output_archive_bytes"),
)
INVERSE_SCORER_CHAIN_SCHEMA = "inverse_scorer_cell_candidate_chain_v1"
INVERSE_SCORER_CHAIN_KIND = "inverse_scorer_cell_candidate_chain"
RENDERER_PAYLOAD_DFL1_SCHEMA = "renderer_payload_dfl1_candidate.v1"
RENDERER_PAYLOAD_DFL1_TARGET_KIND = "renderer_payload_dfl1_v1"
RENDERER_PAYLOAD_DFL1_FULL_FRAME_BLOCKER = (
    "renderer_payload_dfl1_full_frame_inflate_parity_missing"
)
INVERSE_SCORER_EXACT_AUTH_BOUNDARY_TOKENS = frozenset(
    {
        "contest_auth_eval",
        "exact_auth_eval_required_before_score_claim",
    }
)
INVERSE_SCORER_ALLOWED_READINESS_BLOCKERS = frozenset(
    {"exact_auth_eval_required_before_score_claim"}
)
INVERSE_SCORER_CLEARABLE_SOURCE_BLOCKERS = frozenset(
    {
        "inverse_scorer_cell_candidate_chain_is_not_dispatch_authorization",
        "exact_auth_eval_required_before_score_claim",
    }
)
INVERSE_SCORER_REQUIRED_FALSE_AUTHORITY_FIELDS = tuple(
    field
    for field in PROXY_FALSE_AUTHORITY_FIELDS
    if field not in {"score_affecting_payload_changed", "charged_bits_changed"}
)
FAMILY_AGNOSTIC_RUNTIME_PROOF_REQUIRED_FALSE_AUTHORITY_FIELDS = tuple(
    field
    for field in PROXY_FALSE_AUTHORITY_FIELDS
    if field not in {"score_affecting_payload_changed", "charged_bits_changed"}
)
RENDERER_PAYLOAD_DFL1_CLEARABLE_SOURCE_BLOCKERS = frozenset(
    {
        RENDERER_PAYLOAD_DFL1_FULL_FRAME_BLOCKER,
        "renderer_payload_dfl1_receiver_contract_not_satisfied",
        "runtime_consumption_proof_not_passed",
        "family_agnostic_receiver_contract_not_satisfied",
        "renderer_payload_dfl1_requires_same_runtime_full_frame_parity",
        "renderer_payload_dfl1_requires_source_runtime_unpack_proof",
    }
)


class ExactReadinessError(ValueError):
    """Raised when a candidate cannot be promoted for exact-eval dispatch."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return len(text) == 64 and all(ch in SHA256_HEX for ch in text)


def as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def as_integral(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("-").isdigit():
            return int(text)
    return None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "1", "passed"}:
            return True
        if token in {"false", "no", "0", "failed"}:
            return False
    return None


def resolve_path(
    path_value: Any,
    *,
    repo_root: Path,
    queue_dir: Path | None = None,
) -> Path | None:
    if isinstance(path_value, Path):
        path = path_value
    elif isinstance(path_value, str) and path_value.strip():
        path = Path(path_value)
    else:
        return None
    if path.is_absolute():
        return path
    candidates = [repo_root / path]
    if queue_dir is not None:
        candidates.append(queue_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def candidate_archive_path(row: Mapping[str, Any]) -> Any:
    for key in ("candidate_archive_path", "archive_path"):
        value = row.get(key)
        if value:
            return value
    return None


def candidate_archive_sha(row: Mapping[str, Any]) -> str | None:
    for key in ("candidate_archive_sha256", "archive_sha256", "expected_archive_sha256"):
        value = row.get(key)
        if is_sha256(value):
            return str(value).lower()
    return None


def candidate_archive_byte_values(row: Mapping[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for key in (
        "candidate_archive_bytes",
        "archive_size_bytes",
        "archive_bytes",
        "expected_archive_size_bytes",
    ):
        parsed = as_positive_int(row.get(key))
        if parsed is not None:
            values[key] = parsed
    return values


def source_archive_byte_value(row: Mapping[str, Any]) -> int | None:
    mappings: list[Mapping[str, Any]] = [row]
    for key in SCORE_AFFECTING_PROOF_OBJECT_KEYS:
        value = row.get(key)
        if isinstance(value, Mapping):
            mappings.append(value)
    delta = row.get("serialized_archive_delta")
    if isinstance(delta, Mapping):
        mappings.append(delta)
    for mapping in mappings:
        for key in (
            "source_archive_bytes",
            "input_archive_bytes",
            "old_archive_bytes",
        ):
            parsed = as_positive_int(mapping.get(key))
            if parsed is not None:
                return parsed
    return None


def is_rate_only_control(row: Mapping[str, Any]) -> bool:
    if as_bool(row.get("rate_only_control")) is True:
        return True
    delta = row.get("serialized_archive_delta")
    if isinstance(delta, Mapping):
        if as_bool(delta.get("rate_only_control")) is True:
            return True
        if str(delta.get("status") or "").strip().lower() == "rate_only_control":
            return True
    for key in (
        "control_kind",
        "exact_eval_control_kind",
        "dispatch_control_kind",
        "experiment_kind",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip().lower() == "rate_only_control":
            return True
    tags = row.get("tags")
    if isinstance(tags, list | tuple | set):
        return any(
            isinstance(tag, str) and tag.strip().lower() == "rate_only_control"
            for tag in tags
        )
    return False


def find_candidate(queue: Mapping[str, Any], candidate_id: str) -> tuple[dict[str, Any] | None, str | None]:
    for list_name in ("dispatch_ready", "top_k"):
        rows = queue.get(list_name)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and str(row.get("candidate_id") or "") == candidate_id:
                return dict(row), list_name
    return None, None


def iter_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for inner in value.values():
            yield from iter_mappings(inner)
    elif isinstance(value, list | tuple):
        for inner in value:
            yield from iter_mappings(inner)


def score_affecting_change_proof(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    no_op_markers: list[str] = []
    for mapping in iter_mappings(row):
        for key in NO_OP_FIELDS:
            if as_bool(mapping.get(key)) is True:
                no_op_markers.append(f"{key}=true")
        status = str(mapping.get("no_op_status") or "").strip().lower()
        if "no_op" in status or "noop" in status:
            no_op_markers.append(f"no_op_status={status}")
    if no_op_markers:
        return False, [f"explicit_no_op_marker:{','.join(no_op_markers)}"]

    proof_mappings: list[Mapping[str, Any]] = [row]
    for key in SCORE_AFFECTING_PROOF_OBJECT_KEYS:
        value = row.get(key)
        if isinstance(value, Mapping):
            proof_mappings.append(value)
    delta = row.get("serialized_archive_delta")
    if isinstance(delta, Mapping):
        proof_mappings.append(delta)

    proofs: list[str] = []
    explicit_false: list[str] = []
    for mapping in proof_mappings:
        for key in TRUE_CHANGE_FIELDS:
            parsed = as_bool(mapping.get(key))
            if parsed is True:
                proofs.append(f"{key}=true")
            elif parsed is False and key in {
                "score_affecting_payload_changed",
                "charged_bits_changed",
            }:
                explicit_false.append(f"{key}=false")
        for old_key, new_key in SHA_DIFF_FIELD_PAIRS:
            old = mapping.get(old_key)
            new = mapping.get(new_key)
            if is_sha256(old) and is_sha256(new) and str(old).lower() != str(new).lower():
                proofs.append(f"{old_key}!={new_key}")
        for old_key, new_key in BYTE_DIFF_FIELD_PAIRS:
            old = as_positive_int(mapping.get(old_key))
            new = as_positive_int(mapping.get(new_key))
            if old is not None and new is not None and old != new:
                proofs.append(f"{old_key}!={new_key}")
    if proofs:
        return True, sorted(set(proofs))
    if explicit_false:
        return False, sorted(set(explicit_false))
    return False, []


def _expected_serialized_delta_status(
    *,
    source_bytes: int | None,
    candidate_bytes: int | None,
    rate_only_control: bool,
) -> str:
    if source_bytes is None or candidate_bytes is None:
        return "missing_archive_bytes"
    realized_saved = source_bytes - candidate_bytes
    if rate_only_control:
        return "rate_only_control"
    if realized_saved > 0:
        return "realized_saving"
    if realized_saved == 0:
        return "zero_delta"
    return "realized_cost"


def validate_serialized_archive_delta_contract(
    row: Mapping[str, Any],
    *,
    actual_candidate_archive_bytes: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate canonical archive-byte economics if a row carries the contract."""

    raw = row.get("serialized_archive_delta")
    if raw is None:
        return [], {}
    if not isinstance(raw, Mapping):
        return ["serialized_archive_delta_not_object"], {"present": True}

    blockers = serialized_archive_delta_blockers(raw)
    schema = raw.get("schema")
    source_bytes = as_positive_int(raw.get("source_archive_bytes"))
    candidate_bytes = as_positive_int(raw.get("candidate_archive_bytes"))
    declared_delta = as_integral(raw.get("archive_delta_bytes"))
    declared_saved = as_integral(raw.get("realized_saved_bytes"))
    require_realized_saving = as_bool(raw.get("require_realized_saving")) is True
    rate_only_control = as_bool(raw.get("rate_only_control")) is True
    status = str(raw.get("status") or "").strip()
    expected_status = _expected_serialized_delta_status(
        source_bytes=source_bytes,
        candidate_bytes=candidate_bytes,
        rate_only_control=rate_only_control,
    )
    facts: dict[str, Any] = {
        "present": True,
        "schema": schema,
        "status": status or None,
        "expected_status": expected_status,
        "materializer_rate_outcome": expected_status,
        "rate_positive": expected_status == "realized_saving",
        "signal_semantics": (
            "realized_archive_saving"
            if expected_status == "realized_saving"
            else "successful_quality_spend_not_byte_saving_progress"
        ),
        "quality_spend_allowed": as_bool(row.get("quality_spend_allowed")) is True,
        "source_archive_bytes": source_bytes,
        "candidate_archive_bytes": candidate_bytes,
        "archive_delta_bytes": declared_delta,
        "realized_saved_bytes": declared_saved,
        "require_realized_saving": require_realized_saving,
        "rate_only_control": rate_only_control,
        "contract_blockers": list(blockers),
    }

    if schema != SERIALIZED_ARCHIVE_DELTA_SCHEMA:
        blockers.append(f"serialized_archive_delta_schema_unsupported:{schema!r}")
    if not status:
        blockers.append("serialized_archive_delta_status_missing")
    elif status != expected_status:
        blockers.append(
            f"serialized_archive_delta_status_mismatch:{status}!={expected_status}"
        )
    if source_bytes is None or candidate_bytes is None:
        blockers.append(MISSING_ARCHIVE_BYTES_BLOCKER)
    else:
        computed_delta = candidate_bytes - source_bytes
        computed_saved = source_bytes - candidate_bytes
        facts["computed_archive_delta_bytes"] = computed_delta
        facts["computed_realized_saved_bytes"] = computed_saved
        if declared_delta is None:
            blockers.append("serialized_archive_delta_archive_delta_bytes_missing")
        elif declared_delta != computed_delta:
            blockers.append(
                "serialized_archive_delta_archive_delta_bytes_mismatch:"
                f"{declared_delta}!={computed_delta}"
            )
        if declared_saved is None:
            blockers.append("serialized_archive_delta_realized_saved_bytes_missing")
        elif declared_saved != computed_saved:
            blockers.append(
                "serialized_archive_delta_realized_saved_bytes_mismatch:"
                f"{declared_saved}!={computed_saved}"
            )
        savings_realized = as_bool(raw.get("savings_realized"))
        expected_savings_realized = computed_saved > 0
        facts["savings_realized"] = savings_realized
        facts["expected_savings_realized"] = expected_savings_realized
        if savings_realized is None:
            blockers.append("serialized_archive_delta_savings_realized_missing")
        elif savings_realized != expected_savings_realized:
            blockers.append(
                "serialized_archive_delta_savings_realized_mismatch:"
                f"{savings_realized}!={expected_savings_realized}"
            )
        if actual_candidate_archive_bytes is not None and (
            candidate_bytes != actual_candidate_archive_bytes
        ):
            blockers.append(
                "serialized_archive_delta_candidate_bytes_mismatch:"
                f"{candidate_bytes}!={actual_candidate_archive_bytes}"
            )

    if require_realized_saving and expected_status != "realized_saving":
        if expected_status == "realized_cost":
            blockers.append(CANDIDATE_ARCHIVE_LARGER_BLOCKER)
        else:
            blockers.append(SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER)

    modeled_saved = as_integral(raw.get("modeled_saved_bytes"))
    if (
        modeled_saved is not None
        and modeled_saved > 0
        and not rate_only_control
        and expected_status != "realized_saving"
    ):
        blockers.append(MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER)

    return sorted(set(blockers)), facts


def manifest_sha(payload: Mapping[str, Any]) -> str | None:
    for mapping in _manifest_archive_authority_mappings(payload):
        for key in ("candidate_archive_sha256", "archive_sha256", "sha256"):
            value = mapping.get(key)
            if is_sha256(value):
                return str(value).lower()
    return None


def manifest_size(payload: Mapping[str, Any]) -> int | None:
    for mapping in _manifest_archive_authority_mappings(payload):
        for key in (
            "candidate_archive_bytes",
            "candidate_archive_size_bytes",
            "archive_size_bytes",
            "archive_bytes",
            "bytes",
        ):
            parsed = as_positive_int(mapping.get(key))
            if parsed is not None:
                return parsed
    return None


def _manifest_archive_authority_mappings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only mappings whose schema/position can describe the archive."""

    mappings: list[Mapping[str, Any]] = [payload]
    for key in (
        "archive",
        "archive_zip",
        "candidate_archive",
        "packet_archive",
        "submission_archive",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            mappings.append(value)
    return mappings


def manifest_member_names(payload: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for mapping in iter_mappings(payload):
        value = mapping.get("member_name")
        if isinstance(value, str) and value:
            names.append(value)
        members = mapping.get("members")
        if isinstance(members, list):
            for member in members:
                if isinstance(member, Mapping) and isinstance(member.get("name"), str):
                    names.append(str(member["name"]))
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def source_blockers(row: Mapping[str, Any]) -> list[str]:
    blockers = row.get("dispatch_blockers")
    if isinstance(blockers, str):
        return [blockers]
    if isinstance(blockers, list | tuple):
        return [str(item) for item in blockers if str(item)]
    return []


def _as_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _is_inverse_scorer_cell_candidate_chain(row: Mapping[str, Any]) -> bool:
    return (
        row.get("schema") == INVERSE_SCORER_CHAIN_SCHEMA
        or row.get("kind") == INVERSE_SCORER_CHAIN_KIND
    )


def _is_renderer_payload_dfl1_candidate(row: Mapping[str, Any]) -> bool:
    return (
        row.get("schema") == RENDERER_PAYLOAD_DFL1_SCHEMA
        or row.get("target_kind") == RENDERER_PAYLOAD_DFL1_TARGET_KIND
        or row.get("candidate_family") == "renderer_payload_dfl1"
    )


def _has_strict_inverse_scorer_full_frame_parity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
) -> bool:
    if row.get("inflate_parity_satisfied") is not True:
        return False
    steps = row.get("chain_steps")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if step.get("step_id") != "build_inflate_parity_probe":
            continue
        if step.get("full_frame_inflate_output_parity_claim") is not True:
            continue
        step_blockers = _as_text_values(step.get("blockers"))
        if not step_blockers and _has_verified_inverse_scorer_parity_artifact(
            step,
            repo_root=repo_root,
            queue_dir=queue_dir,
        ):
            return True
    return False


def _has_strict_renderer_payload_dfl1_full_frame_parity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
) -> bool:
    if not _is_renderer_payload_dfl1_candidate(row):
        return False
    if (
        row.get("renderer_payload_dfl1_full_frame_inflate_parity_satisfied") is not True
        and row.get("renderer_payload_dfl1_inflate_parity_satisfied") is not True
        and row.get("full_frame_inflate_parity_proven") is not True
    ):
        return False
    proof_ref = row.get("renderer_payload_dfl1_full_frame_inflate_parity_proof_path")
    if proof_ref is None:
        proof_ref = row.get("renderer_payload_dfl1_inflate_parity_proof_path")
    verification = row.get("full_frame_inflate_parity_verification")
    if proof_ref is None and isinstance(verification, Mapping):
        proof_ref = verification.get("proof_path")
    proof_path = resolve_path(proof_ref, repo_root=repo_root, queue_dir=queue_dir)
    if proof_path is None or not proof_path.is_file() or proof_path.is_symlink():
        return False
    if not _path_is_repo_confined(proof_path, repo_root):
        return False
    expected_proof_sha = row.get(
        "renderer_payload_dfl1_full_frame_inflate_parity_proof_sha256"
    )
    if expected_proof_sha is None:
        expected_proof_sha = row.get("renderer_payload_dfl1_inflate_parity_proof_sha256")
    if expected_proof_sha is None and isinstance(verification, Mapping):
        expected_proof_sha = verification.get("proof_sha256")
    if is_sha256(expected_proof_sha) and sha256_file(proof_path) != str(
        expected_proof_sha
    ).lower():
        return False
    candidate_sha = (
        str(row.get("archive_sha256") or "")
        if is_sha256(row.get("archive_sha256"))
        else str(row.get("candidate_archive_sha256") or "")
    )
    source_sha = (
        str(row.get("source_archive_sha256") or "")
        if is_sha256(row.get("source_archive_sha256"))
        else ""
    )
    if not is_sha256(candidate_sha) or not is_sha256(source_sha):
        return False
    verified = verify_renderer_payload_dfl1_full_frame_inflate_parity_proof(
        full_frame_inflate_parity_proof=proof_path,
        required_source_archive_sha256=source_sha,
        required_candidate_archive_sha256=candidate_sha,
        repo_root=repo_root,
    )
    return verified.get("full_frame_inflate_parity_satisfied") is True


def _has_strict_full_frame_parity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
) -> bool:
    return _has_strict_inverse_scorer_full_frame_parity(
        row,
        repo_root=repo_root,
        queue_dir=queue_dir,
    ) or _has_strict_renderer_payload_dfl1_full_frame_parity(
        row,
        repo_root=repo_root,
        queue_dir=queue_dir,
    )


def _path_is_repo_confined(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return True


def _inverse_scorer_parity_payload_verified(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("schema") != "inverse_scorer_cell_inflate_parity_probe_v1":
        return False
    for key in (
        "full_frame_inflate_output_parity_claim",
        "output_bytes_identical",
        "output_contract_nonempty",
        "output_contract_paths_match",
    ):
        if payload.get(key) is not True:
            return False
    if payload.get("proof_scope") != "full_frame_inflate_output_tree":
        return False
    if payload.get("differing_path_count") not in (0, None):
        return False
    for key in ("blockers", "missing_from_candidate", "extra_in_candidate"):
        if payload.get(key):
            return False
    for key in PROXY_FALSE_AUTHORITY_FIELDS:
        if payload.get(key) is not False:
            return False
    source_tree = payload.get("source_output_tree")
    candidate_tree = payload.get("candidate_output_tree")
    if not isinstance(source_tree, Mapping) or not isinstance(candidate_tree, Mapping):
        return False
    if source_tree.get("tree_sha256") != candidate_tree.get("tree_sha256"):
        return False
    for tree in (source_tree, candidate_tree):
        if tree.get("exists") is not True or tree.get("blockers"):
            return False
        if not isinstance(tree.get("file_count"), int) or tree["file_count"] <= 0:
            return False
        if not isinstance(tree.get("total_bytes"), int) or tree["total_bytes"] <= 0:
            return False
    return True


def _has_verified_inverse_scorer_parity_artifact(
    step: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
) -> bool:
    if step.get("schema") != "inverse_scorer_cell_inflate_parity_probe_v1":
        return False
    artifact = step.get("artifact")
    if not isinstance(artifact, Mapping):
        return False
    expected_sha = artifact.get("sha256")
    if not is_sha256(expected_sha):
        return False
    artifact_path = resolve_path(artifact.get("path"), repo_root=repo_root, queue_dir=queue_dir)
    if artifact_path is None or not artifact_path.is_file() or artifact_path.is_symlink():
        return False
    if not _path_is_repo_confined(artifact_path, repo_root):
        return False
    if sha256_file(artifact_path) != str(expected_sha).lower():
        return False
    try:
        payload = read_json(artifact_path)
    except (OSError, json.JSONDecodeError):
        return False
    return _inverse_scorer_parity_payload_verified(payload)


def _has_inverse_scorer_exact_auth_boundary(row: Mapping[str, Any]) -> bool:
    if (
        row.get("exact_auth_eval_required_before_score_claim") is True
        or row.get("requires_exact_auth_eval_before_score_claim") is True
    ):
        return True
    for key in (
        "readiness_blockers",
        "dispatch_blockers",
        "score_claim_blockers",
        "promotion_blockers",
        "next_required_gates",
    ):
        for item in _as_text_values(row.get(key)):
            if item in INVERSE_SCORER_EXACT_AUTH_BOUNDARY_TOKENS:
                return True
    return False


def inverse_scorer_chain_authority_blockers(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
) -> list[str]:
    """Return fail-closed exact-readiness blockers for IAS1 chain rows.

    IAS1 chain artifacts can prove descriptor consumption and, later, full-frame
    inflate parity. They are never score or promotion evidence by themselves.
    """

    if not _is_inverse_scorer_cell_candidate_chain(row):
        return []
    blockers: list[str] = []
    for key in INVERSE_SCORER_REQUIRED_FALSE_AUTHORITY_FIELDS:
        if row.get(key) is not False:
            blockers.append(f"inverse_scorer_cell_candidate_chain_{key}_not_false")
    if not _has_strict_inverse_scorer_full_frame_parity(
        row,
        repo_root=repo_root,
        queue_dir=queue_dir,
    ):
        blockers.append(
            "inverse_scorer_cell_candidate_chain_strict_full_frame_inflate_parity_missing"
        )
    if not _has_inverse_scorer_exact_auth_boundary(row):
        blockers.append(
            "inverse_scorer_cell_candidate_chain_exact_auth_eval_boundary_missing"
        )
    readiness = {str(item) for item in row.get("readiness_blockers") or [] if str(item)}
    for blocker in sorted(readiness - INVERSE_SCORER_ALLOWED_READINESS_BLOCKERS):
        blockers.append(
            "inverse_scorer_cell_candidate_chain_unresolved_readiness_blocker:"
            f"{blocker}"
        )
    return blockers


def renderer_payload_dfl1_authority_blockers(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
) -> list[str]:
    """Return DFL1 exact-readiness blockers cleared only by verified parity."""

    if not _is_renderer_payload_dfl1_candidate(row):
        return []
    if _has_strict_renderer_payload_dfl1_full_frame_parity(
        row,
        repo_root=repo_root,
        queue_dir=queue_dir,
    ):
        return []
    return ["renderer_payload_dfl1_strict_full_frame_inflate_parity_missing"]


def source_blocker_violations(
    row: Mapping[str, Any],
    *,
    extra_clearable_source_blockers: Iterable[str] = (),
) -> list[str]:
    clearable = set(CLEARABLE_SOURCE_BLOCKERS)
    clearable.update(
        str(item)
        for item in extra_clearable_source_blockers
        if str(item)
        and not any(
            str(item).startswith(prefix) for prefix in BLOCKED_SOURCE_BLOCKER_PREFIXES
        )
    )
    violations: list[str] = []
    for blocker in source_blockers(row):
        if any(blocker.startswith(prefix) for prefix in BLOCKED_SOURCE_BLOCKER_PREFIXES):
            violations.append(f"blocked_source_dispatch_blocker:{blocker}")
        elif blocker in clearable:
            continue
        else:
            violations.append(f"unknown_uncleared_source_dispatch_blocker:{blocker}")
    return violations


def parse_utc(value: str) -> dt.datetime | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def claim_status_terminal(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_PREFIXES)


def parse_claim_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "lane_id" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 8 or set(cells) <= {"---"}:
            continue
        rows.append(
            {
                "timestamp_utc": cells[0],
                "agent": cells[1],
                "lane_id": cells[2],
                "platform": cells[3],
                "instance_job_id": cells[4],
                "predicted_eta_utc": cells[5],
                "status": cells[6],
                "notes": cells[7],
            }
        )
    return rows


def active_claim_conflicts(
    lane_id: str,
    *,
    dispatch_claims_path: Path | None,
    now_utc: dt.datetime | None = None,
    ttl_hours: float = 24.0,
    allowed_active_claim_platform: str | None = None,
    allowed_active_claim_instance_job_ids: Iterable[str] = (),
) -> list[str]:
    if dispatch_claims_path is None or not dispatch_claims_path.is_file():
        return []
    now = now_utc or dt.datetime.now(tz=dt.UTC).replace(microsecond=0)
    allowed_platform = str(allowed_active_claim_platform or "").strip().lower()
    allowed_job_ids = {
        str(item).strip()
        for item in allowed_active_claim_instance_job_ids
        if str(item).strip()
    }
    blockers: list[str] = []
    for row in _latest_claim_rows_by_job(dispatch_claims_path).values():
        if row["lane_id"] != lane_id or claim_status_terminal(row["status"]):
            continue
        ts = parse_utc(row["timestamp_utc"])
        age_hours = None if ts is None else max((now - ts).total_seconds() / 3600.0, 0.0)
        allowed_active_claim = (
            bool(allowed_job_ids)
            and row["instance_job_id"].strip() in allowed_job_ids
            and (
                not allowed_platform
                or row["platform"].strip().lower() == allowed_platform
            )
        )
        if ts is None or age_hours is None or age_hours > ttl_hours:
            blockers.append(
                "same_lane_stale_nonterminal_dispatch_claim:"
                f"{row['lane_id']}:{row['instance_job_id']}:{row['status']}"
            )
        elif allowed_active_claim:
            continue
        else:
            blockers.append(
                "same_lane_active_dispatch_claim:"
                f"{row['lane_id']}:{row['instance_job_id']}:{row['status']}"
            )
    return blockers


def _claim_rows_by_job(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    rows_by_job: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in parse_claim_rows(path):
        key = (row["lane_id"], row["instance_job_id"])
        rows_by_job.setdefault(key, []).append(row)
    return rows_by_job


def _latest_claim_rows_by_job(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return _latest_claim_rows_from_grouped(_claim_rows_by_job(path))


def _latest_claim_rows_from_grouped(
    rows_by_job: Mapping[tuple[str, str], Iterable[dict[str, str]]],
) -> dict[tuple[str, str], dict[str, str]]:
    latest_by_job: dict[tuple[str, str], dict[str, str]] = {}
    for key, rows in rows_by_job.items():
        for row in rows:
            prev = latest_by_job.get(key)
            row_ts = parse_utc(row["timestamp_utc"])
            prev_ts = parse_utc(prev["timestamp_utc"]) if prev is not None else None
            if prev is None or prev_ts is None or (row_ts is not None and row_ts > prev_ts):
                latest_by_job[key] = row
    return latest_by_job


def _claim_job_notes(rows: Iterable[Mapping[str, str]]) -> str:
    return " ".join(row.get("notes", "") for row in rows)


def _terminal_claim_runtime_tree_shas(notes: str) -> set[str]:
    return {match.lower() for match in TERMINAL_RUNTIME_TREE_SHA_RE.findall(notes)}


def _terminal_claim_runtime_content_tree_shas(notes: str) -> set[str]:
    return {
        match.lower()
        for match in TERMINAL_RUNTIME_CONTENT_TREE_SHA_RE.findall(notes)
    }


def _terminal_claim_score(notes: str) -> float | None:
    match = TERMINAL_SCORE_RE.search(notes)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def terminal_claim_result_conflicts(
    lane_id: str,
    archive_sha256: str | None,
    *,
    dispatch_claims_path: Path | None,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
    runtime_tree_sha256: str | None = None,
    runtime_content_tree_sha256: str | None = None,
    score_affecting_runtime_changed: bool | None = None,
    block_runtime_mismatch_for_same_archive: bool = False,
) -> list[str]:
    """Block stale exact-ready rows after terminal evidence on same archive.

    This is intentionally narrow: infrastructure failures such as missing
    provider dependencies do not block a corrected rerun, while terminal
    measured negatives and completed exact-CUDA scores that fail to beat the
    active score frontier do block silent re-promotion of the same
    lane/archive row.
    """

    if (
        dispatch_claims_path is None
        or not dispatch_claims_path.is_file()
        or not isinstance(archive_sha256, str)
        or not archive_sha256
    ):
        return []
    blockers: list[str] = []
    claim_rows_by_job = _claim_rows_by_job(dispatch_claims_path)
    latest_rows = _latest_claim_rows_from_grouped(claim_rows_by_job)
    candidate_runtime_sha = (
        runtime_tree_sha256.lower() if is_sha256(runtime_tree_sha256) else None
    )
    candidate_runtime_content_sha = (
        runtime_content_tree_sha256.lower()
        if is_sha256(runtime_content_tree_sha256)
        else None
    )
    for key, row in latest_rows.items():
        if row["lane_id"] != lane_id or not claim_status_terminal(row["status"]):
            continue
        notes = _claim_job_notes(claim_rows_by_job.get(key, [row]))
        if archive_sha256 not in notes:
            continue
        claim_id = f"{row['lane_id']}:{row['instance_job_id']}:{row['status']}"
        terminal_runtime_shas = _terminal_claim_runtime_tree_shas(row.get("notes", ""))
        claim_runtime_shas = _terminal_claim_runtime_tree_shas(notes)
        runtime_shas_for_match = terminal_runtime_shas or claim_runtime_shas
        terminal_runtime_content_shas = _terminal_claim_runtime_content_tree_shas(
            row.get("notes", "")
        )
        claim_runtime_content_shas = _terminal_claim_runtime_content_tree_shas(notes)
        runtime_content_shas_for_match = (
            terminal_runtime_content_shas or claim_runtime_content_shas
        )
        runtime_identity_evidence_seen = False
        runtime_identity_matched = False
        if candidate_runtime_sha is not None and runtime_shas_for_match:
            runtime_identity_evidence_seen = True
            if candidate_runtime_sha not in runtime_shas_for_match:
                if block_runtime_mismatch_for_same_archive:
                    blockers.append(
                        "same_lane_terminal_runtime_mismatch_for_same_archive:"
                        f"{candidate_runtime_sha}:terminal_runtime="
                        f"{','.join(sorted(runtime_shas_for_match))}:{claim_id}"
                    )
                elif terminal_runtime_shas:
                    continue
                elif score_affecting_runtime_changed is not True:
                    blockers.append(
                        "same_lane_terminal_runtime_mismatch_for_same_archive:"
                        f"{candidate_runtime_sha}:terminal_runtime="
                        f"{','.join(sorted(runtime_shas_for_match))}:{claim_id}"
                    )
                continue
            runtime_identity_matched = True
        if candidate_runtime_content_sha is not None and runtime_content_shas_for_match:
            runtime_identity_evidence_seen = True
            if candidate_runtime_content_sha not in runtime_content_shas_for_match:
                if block_runtime_mismatch_for_same_archive:
                    blockers.append(
                        "same_lane_terminal_runtime_content_mismatch_for_same_archive:"
                        f"{candidate_runtime_content_sha}:terminal_runtime_content="
                        f"{','.join(sorted(runtime_content_shas_for_match))}:{claim_id}"
                    )
                elif terminal_runtime_content_shas:
                    continue
                elif score_affecting_runtime_changed is not True:
                    blockers.append(
                        "same_lane_terminal_runtime_content_mismatch_for_same_archive:"
                        f"{candidate_runtime_content_sha}:terminal_runtime_content="
                        f"{','.join(sorted(runtime_content_shas_for_match))}:{claim_id}"
                    )
                continue
            runtime_identity_matched = True
        if (
            score_affecting_runtime_changed is True
            and (candidate_runtime_sha is not None or candidate_runtime_content_sha is not None)
            and not runtime_identity_evidence_seen
            and not runtime_identity_matched
        ):
            continue
        status = row["status"].lower()
        if status.startswith("refused_dispatch"):
            blockers.append(f"same_lane_terminal_refused_dispatch_for_same_archive:{claim_id}")
            continue
        if any(marker in status for marker in TERMINAL_NEGATIVE_STATUS_MARKERS):
            blockers.append(f"same_lane_terminal_negative_for_same_archive:{claim_id}")
            continue
        score = _terminal_claim_score(notes)
        if status.startswith("completed_contest_cuda") and score is None:
            blockers.append(
                "same_lane_terminal_cuda_score_missing_for_same_archive:"
                f"{claim_id}"
            )
            continue
        if status.startswith("completed_contest_cuda") and score is not None:
            if active_floor_score is not None and score >= active_floor_score:
                blockers.append(
                    "same_lane_terminal_score_not_below_active_floor_for_same_archive:"
                    f"{score:.12g}>={active_floor_score:.12g}:{claim_id}"
                )
            elif active_floor_score is not None and score < active_floor_score:
                blockers.append(
                    "same_lane_terminal_score_already_below_active_floor_for_same_archive:"
                    f"{score:.12g}<{active_floor_score:.12g}:{claim_id}"
                )
    return blockers


def runtime_dependency_manifest(submission_dir: Path, repo_root: Path) -> dict[str, Any]:
    module_path = repo_root / "experiments" / "contest_auth_eval.py"
    if not module_path.is_file():
        module_path = Path(__file__).resolve().parents[3] / "experiments" / "contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location(
        "pact_exact_readiness_contest_auth_eval",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load contest_auth_eval from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime_manifest_fn = module._runtime_dependency_manifest

    return runtime_manifest_fn(
        submission_dir / "inflate.sh",
        repo_root / "upstream",
        repo_root=repo_root,
    )


def default_manifest_path(submission_dir: Path) -> Path:
    for name in ("archive_manifest.json", "manifest.json", "runtime_packet_manifest.json"):
        candidate = submission_dir / name
        if candidate.is_file():
            return candidate
    return submission_dir / "archive_manifest.json"


def _same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _runtime_consumption_pr101_binding_blockers(
    proof_raw: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
    submission_dir: Path | None,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    facts: dict[str, Any] = {}
    if submission_dir is None:
        blockers.append("runtime_consumption_proof_submission_dir_missing")
        return blockers, facts

    packet_dir = resolve_path(
        proof_raw.get("packet_dir"),
        repo_root=repo_root,
        queue_dir=queue_dir,
    )
    facts["runtime_consumption_proof_packet_dir"] = packet_dir
    if packet_dir is None:
        blockers.append("runtime_consumption_proof_packet_dir_missing")
    elif not _same_resolved_path(packet_dir, submission_dir):
        blockers.append("runtime_consumption_proof_packet_dir_mismatch")

    manifest_path = resolve_path(
        proof_raw.get("manifest_path"),
        repo_root=repo_root,
        queue_dir=queue_dir,
    )
    facts["runtime_consumption_proof_manifest_path"] = manifest_path
    if manifest_path is None:
        blockers.append("runtime_consumption_proof_manifest_path_missing")
    elif not manifest_path.is_file():
        blockers.append("runtime_consumption_proof_manifest_file_missing")
    else:
        manifest_sha = proof_raw.get("manifest_sha256")
        if is_sha256(manifest_sha):
            actual_manifest_sha = sha256_file(manifest_path)
            facts["runtime_consumption_proof_manifest_sha256"] = actual_manifest_sha
            if actual_manifest_sha != str(manifest_sha).lower():
                blockers.append("runtime_consumption_proof_manifest_sha_mismatch")
        else:
            blockers.append("runtime_consumption_proof_manifest_sha_missing")

    inflate_sh = submission_dir / "inflate.sh"
    route_proof = _mapping_at(proof_raw, "inflate_wrapper_route_proof")
    expected_inflate_sh_sha = route_proof.get("inflate_sh_sha256")
    if not inflate_sh.is_file():
        blockers.append("runtime_consumption_proof_inflate_sh_file_missing")
    else:
        actual_inflate_sh_sha = sha256_file(inflate_sh)
        facts["runtime_consumption_proof_actual_inflate_sh_sha256"] = (
            actual_inflate_sh_sha
        )
        if not is_sha256(expected_inflate_sh_sha):
            blockers.append("runtime_consumption_proof_inflate_sh_sha_missing")
        elif actual_inflate_sh_sha != str(expected_inflate_sh_sha).lower():
            blockers.append("runtime_consumption_proof_inflate_sh_sha_mismatch")

    if route_proof.get("wrapper_invoked_packet_inflate_py") is not True:
        blockers.append("runtime_consumption_proof_wrapper_route_not_proven")

    static_proof = _mapping_at(proof_raw, "inflate_static_bias_patch_proof")
    runtime_proof = _mapping_at(proof_raw, "inflate_runtime_bias_logic_proof")
    expected_inflate_py_shas = {
        str(value).lower()
        for value in (
            static_proof.get("inflate_sha256"),
            route_proof.get("packet_inflate_py_sha256"),
            runtime_proof.get("inflate_py_sha256"),
        )
        if is_sha256(value)
    }
    if not expected_inflate_py_shas:
        blockers.append("runtime_consumption_proof_inflate_py_sha_missing")
    else:
        inflate_py = submission_dir / "inflate.py"
        if not inflate_py.is_file():
            blockers.append("runtime_consumption_proof_inflate_py_file_missing")
        else:
            actual_inflate_py_sha = sha256_file(inflate_py)
            facts["runtime_consumption_proof_actual_inflate_py_sha256"] = (
                actual_inflate_py_sha
            )
            mismatches = sorted(
                sha for sha in expected_inflate_py_shas if sha != actual_inflate_py_sha
            )
            if mismatches:
                blockers.append("runtime_consumption_proof_inflate_py_sha_mismatch")

    if runtime_proof.get("packet_inflate_function_executed") is not True:
        blockers.append("runtime_consumption_proof_runtime_logic_not_proven")

    return blockers, facts


def _family_agnostic_runtime_proof_passed(proof: Mapping[str, Any]) -> bool:
    if proof.get("receiver_contract_satisfied") is True:
        return True
    if proof.get("runtime_consumption_proof_passed") is True:
        return True
    if proof.get("passed") is True:
        return True
    probe = proof.get("runtime_consumption_probe")
    return isinstance(probe, Mapping) and probe.get("passed") is True


def _family_agnostic_runtime_proof_archive_sha(proof: Mapping[str, Any]) -> str | None:
    candidate_archive = proof.get("candidate_archive")
    if isinstance(candidate_archive, Mapping) and is_sha256(candidate_archive.get("sha256")):
        return str(candidate_archive["sha256"]).lower()
    value = proof.get("candidate_archive_sha256") or proof.get("archive_sha256")
    return str(value).lower() if is_sha256(value) else None


def _family_agnostic_runtime_proof_member_sha(proof: Mapping[str, Any]) -> str | None:
    candidate_member = proof.get("candidate_member")
    if isinstance(candidate_member, Mapping) and is_sha256(candidate_member.get("sha256")):
        return str(candidate_member["sha256"]).lower()
    value = proof.get("candidate_member_sha256") or proof.get("member_sha256")
    return str(value).lower() if is_sha256(value) else None


def validate_runtime_consumption_proof(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
    submission_dir: Path | None,
    archive_sha256: str | None,
) -> tuple[list[str], dict[str, Any]]:
    proof_backed_by_full_frame_parity = _has_strict_full_frame_parity(
        row,
        repo_root=repo_root,
        queue_dir=queue_dir,
    )
    required = row.get("runtime_consumption_proof_required") is True or (
        any(as_bool(row.get(field)) for field in TRUE_CHANGE_FIELDS)
        and not proof_backed_by_full_frame_parity
    )
    status = row.get("runtime_consumption_proof_status")
    proof_ref = row.get("runtime_consumption_proof_path")
    if not required and status is None and proof_ref is None:
        return [], {}

    blockers: list[str] = []
    facts: dict[str, Any] = {
        "runtime_consumption_proof_required": required,
        "runtime_consumption_proof_status": status,
        "runtime_consumption_proof_backed_by_full_frame_parity": (
            proof_backed_by_full_frame_parity
        ),
    }
    if status != "present":
        blockers.append("runtime_consumption_proof_missing")

    proof_path = resolve_path(proof_ref, repo_root=repo_root, queue_dir=queue_dir)
    if proof_path is None and submission_dir is not None:
        proof_path = submission_dir / "runtime_consumption_proof.json"
    facts["runtime_consumption_proof_path"] = proof_path
    if proof_path is None or not proof_path.is_file():
        blockers.append("runtime_consumption_proof_file_missing")
        return blockers, facts

    try:
        proof_raw = read_json(proof_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"runtime_consumption_proof_json_invalid:{exc}")
        return blockers, facts
    if not isinstance(proof_raw, dict):
        blockers.append("runtime_consumption_proof_not_object")
        return blockers, facts

    facts["runtime_consumption_proof_schema"] = proof_raw.get("schema")
    facts["runtime_consumption_proof_sha256"] = sha256_file(proof_path)
    proof_schema = proof_raw.get("schema")
    if proof_schema == PR101_RUNTIME_CONSUMPTION_PROOF_SCHEMA:
        if proof_raw.get("runtime_consumption_proven_for_supported_bias_params") is not True:
            blockers.append("runtime_consumption_proof_not_proven")
        if proof_raw.get("inflate_sh_routes_to_packet_inflate_py") is not True:
            blockers.append("runtime_consumption_proof_wrapper_route_not_proven")
        archive_proof = proof_raw.get("archive_unchanged_proof")
        proof_archive_sha = (
            archive_proof.get("archive_sha256")
            if isinstance(archive_proof, Mapping)
            else None
        )
        if archive_sha256 is not None and proof_archive_sha != archive_sha256:
            blockers.append("runtime_consumption_proof_archive_sha_mismatch")
        for false_authority_field in (
            field for field in PROXY_FALSE_AUTHORITY_FIELDS if field in proof_raw
        ):
            if proof_raw.get(false_authority_field) is not False:
                blockers.append(
                    f"runtime_consumption_proof_false_authority_violation:{false_authority_field}"
                )
        binding_blockers, binding_facts = _runtime_consumption_pr101_binding_blockers(
            proof_raw,
            repo_root=repo_root,
            queue_dir=queue_dir,
            submission_dir=submission_dir,
        )
        blockers.extend(binding_blockers)
        facts.update(binding_facts)
    elif proof_schema in FAMILY_AGNOSTIC_RUNTIME_CONSUMPTION_PROOF_SCHEMAS:
        if not _family_agnostic_runtime_proof_passed(proof_raw) and not (
            _is_renderer_payload_dfl1_candidate(row)
            and proof_backed_by_full_frame_parity
        ):
            blockers.append("runtime_consumption_proof_not_proven")
        for field in ("target_kind", "materializer_id", "receiver_contract_kind"):
            expected = row.get(field)
            if not isinstance(expected, str) or not expected.strip():
                blockers.append(
                    f"candidate_row_{field}_missing_for_family_agnostic_runtime_proof"
                )
                continue
            observed = proof_raw.get(field)
            facts[f"runtime_consumption_proof_{field}"] = observed
            if not isinstance(observed, str) or not observed.strip():
                blockers.append(f"runtime_consumption_proof_{field}_missing")
            elif observed.strip() != expected.strip():
                blockers.append(f"runtime_consumption_proof_{field}_mismatch")
        proof_archive_sha = _family_agnostic_runtime_proof_archive_sha(proof_raw)
        facts["runtime_consumption_proof_archive_sha256"] = proof_archive_sha
        if archive_sha256 is not None:
            if proof_archive_sha is None:
                blockers.append("runtime_consumption_proof_archive_sha_missing")
            elif proof_archive_sha != archive_sha256:
                blockers.append("runtime_consumption_proof_archive_sha_mismatch")
        expected_member_sha = row.get("candidate_member_sha256")
        if is_sha256(expected_member_sha):
            proof_member_sha = _family_agnostic_runtime_proof_member_sha(proof_raw)
            facts["runtime_consumption_proof_candidate_member_sha256"] = (
                proof_member_sha
            )
            if proof_member_sha is None:
                blockers.append("runtime_consumption_proof_candidate_member_sha_missing")
            elif proof_member_sha != str(expected_member_sha).lower():
                blockers.append("runtime_consumption_proof_candidate_member_sha_mismatch")
        for false_authority_field in (
            field
            for field in FAMILY_AGNOSTIC_RUNTIME_PROOF_REQUIRED_FALSE_AUTHORITY_FIELDS
            if field in proof_raw
        ):
            if proof_raw.get(false_authority_field) is not False:
                blockers.append(
                    f"runtime_consumption_proof_false_authority_violation:{false_authority_field}"
                )
    else:
        blockers.append("runtime_consumption_proof_schema_unsupported")

    return blockers, facts


def readiness_blockers(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path | None,
    submission_dir: Path | None = None,
    archive_manifest_path: Path | None = None,
    lane_id: str | None = None,
    active_floor_archive_bytes: int | None = ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
    extra_clearable_source_blockers: Iterable[str] = (),
    dispatch_claims_path: Path | None = None,
    claim_ttl_hours: float = 24.0,
    ignore_active_claim_conflicts: bool = False,
    allowed_active_claim_platform: str | None = None,
    allowed_active_claim_instance_job_ids: Iterable[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    facts: dict[str, Any] = {}

    if row.get("score_claim") is True:
        blockers.append("source_row_already_score_claiming")
    if row.get("proxy_only") is True:
        blockers.append("source_row_proxy_only")
    for key in PREDICTED_SCORE_FIELDS:
        if key in row:
            facts.setdefault("stripped_source_score_fields", []).append(key)
    inverse_scorer_blockers = inverse_scorer_chain_authority_blockers(
        row,
        repo_root=repo_root,
        queue_dir=queue_dir,
    )
    blockers.extend(inverse_scorer_blockers)
    renderer_dfl1_blockers = renderer_payload_dfl1_authority_blockers(
        row,
        repo_root=repo_root,
        queue_dir=queue_dir,
    )
    blockers.extend(renderer_dfl1_blockers)
    effective_clearable_source_blockers = list(extra_clearable_source_blockers)
    if _is_inverse_scorer_cell_candidate_chain(row) and not inverse_scorer_blockers:
        effective_clearable_source_blockers.extend(
            sorted(INVERSE_SCORER_CLEARABLE_SOURCE_BLOCKERS)
        )
    if _is_renderer_payload_dfl1_candidate(row) and not renderer_dfl1_blockers:
        effective_clearable_source_blockers.extend(
            sorted(RENDERER_PAYLOAD_DFL1_CLEARABLE_SOURCE_BLOCKERS)
        )
    blockers.extend(
        source_blocker_violations(
            row,
            extra_clearable_source_blockers=effective_clearable_source_blockers,
        )
    )

    effective_lane_id = lane_id or row.get("lane_id")
    if not isinstance(effective_lane_id, str) or not effective_lane_id.strip():
        blockers.append("lane_id_missing")
    facts["lane_id"] = effective_lane_id
    if (
        isinstance(effective_lane_id, str)
        and effective_lane_id.strip()
        and not ignore_active_claim_conflicts
    ):
        blockers.extend(
            active_claim_conflicts(
                effective_lane_id,
                dispatch_claims_path=dispatch_claims_path,
                ttl_hours=claim_ttl_hours,
                allowed_active_claim_platform=allowed_active_claim_platform,
                allowed_active_claim_instance_job_ids=(
                    allowed_active_claim_instance_job_ids
                ),
            )
        )

    change_proven, change_proofs = score_affecting_change_proof(row)
    facts["score_affecting_change_proofs"] = change_proofs
    facts["score_affecting_change_proven"] = change_proven
    if not change_proven:
        blockers.append("score_affecting_change_proof_missing")

    archive_path = resolve_path(candidate_archive_path(row), repo_root=repo_root, queue_dir=queue_dir)
    if archive_path is None:
        blockers.append("archive_path_missing")
    elif not archive_path.is_file():
        blockers.append("archive_file_missing")
    facts["archive_path"] = archive_path

    expected_sha = candidate_archive_sha(row)
    byte_values = candidate_archive_byte_values(row)
    expected_bytes = next(iter(byte_values.values()), None)
    if expected_sha is None:
        blockers.append("archive_sha256_missing_or_invalid")
    if expected_bytes is None:
        blockers.append("archive_bytes_missing_or_invalid")
    elif len(set(byte_values.values())) > 1:
        details = ",".join(f"{key}={value}" for key, value in sorted(byte_values.items()))
        blockers.append(f"archive_bytes_field_mismatch:{details}")

    zipwire: dict[str, Any] | None = None
    if archive_path is not None and archive_path.is_file():
        actual_sha = sha256_file(archive_path)
        actual_bytes = archive_path.stat().st_size
        facts["archive_sha256"] = actual_sha
        facts["archive_bytes"] = actual_bytes
        if expected_sha is not None and actual_sha != expected_sha:
            blockers.append("archive_sha256_mismatch")
        if expected_bytes is not None and actual_bytes != expected_bytes:
            blockers.append(f"archive_bytes_mismatch:{actual_bytes}!={expected_bytes}")
        try:
            zipwire = inspect_zip_headers(archive_path)
        except (OSError, ValueError) as exc:
            blockers.append(f"archive_zip_unreadable:{exc}")
        else:
            facts["zipwire"] = zipwire
            if zipwire.get("zip_strict") is not True:
                blockers.append("archive_zip_not_strict")
            if int(zipwire.get("member_count") or 0) < 1:
                blockers.append("archive_zip_empty")

    serialized_delta_blockers, serialized_delta_facts = (
        validate_serialized_archive_delta_contract(
            row,
            actual_candidate_archive_bytes=facts.get("archive_bytes")
            if isinstance(facts.get("archive_bytes"), int)
            else None,
        )
    )
    blockers.extend(serialized_delta_blockers)
    if serialized_delta_facts:
        facts["serialized_archive_delta"] = serialized_delta_facts

    source_bytes = source_archive_byte_value(row)
    if source_bytes is not None:
        facts["source_archive_bytes"] = source_bytes
    if (
        _is_inverse_scorer_cell_candidate_chain(row)
        and source_bytes is not None
        and isinstance(facts.get("archive_bytes"), int)
        and _has_strict_inverse_scorer_full_frame_parity(
            row,
            repo_root=repo_root,
            queue_dir=queue_dir,
        )
    ):
        archive_byte_delta = int(facts["archive_bytes"]) - source_bytes
        facts["realized_archive_byte_delta"] = archive_byte_delta
        facts["inverse_scorer_full_frame_output_parity"] = True
        if archive_byte_delta > 0 and not is_rate_only_control(row):
            blockers.append(
                "inverse_scorer_full_frame_parity_byte_increase_without_rate_only_control:"
                f"{facts['archive_bytes']}>{source_bytes}"
            )

    if submission_dir is None and archive_path is not None:
        submission_dir = archive_path.parent
    if submission_dir is None or not submission_dir.is_dir():
        blockers.append("submission_dir_missing")
    facts["submission_dir"] = submission_dir

    inflate_sh = submission_dir / "inflate.sh" if submission_dir is not None else None
    if inflate_sh is None or not inflate_sh.is_file():
        blockers.append("inflate_sh_missing")
    elif not os.access(inflate_sh, os.X_OK):
        blockers.append("inflate_sh_not_executable")
    facts["inflate_sh"] = inflate_sh

    report_path = submission_dir / "report.txt" if submission_dir is not None else None
    if report_path is None or not report_path.is_file():
        blockers.append("report_txt_missing")
    facts["report_path"] = report_path

    manifest_path = archive_manifest_path or (
        default_manifest_path(submission_dir) if submission_dir is not None else None
    )
    manifest_payload: dict[str, Any] | None = None
    if manifest_path is None or not manifest_path.is_file():
        blockers.append("archive_manifest_missing")
    else:
        try:
            raw_manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"archive_manifest_json_invalid:{exc}")
        else:
            if not isinstance(raw_manifest, dict):
                blockers.append("archive_manifest_not_object")
            else:
                manifest_payload = raw_manifest
                manifest_archive_sha = manifest_sha(raw_manifest)
                manifest_archive_size = manifest_size(raw_manifest)
                if archive_path is not None and archive_path.is_file():
                    actual_sha = facts.get("archive_sha256")
                    actual_bytes = facts.get("archive_bytes")
                    if manifest_archive_sha != actual_sha:
                        blockers.append(
                            f"archive_manifest_sha_mismatch:{manifest_archive_sha}!={actual_sha}"
                        )
                    if manifest_archive_size != actual_bytes:
                        blockers.append(
                            f"archive_manifest_size_mismatch:{manifest_archive_size}!={actual_bytes}"
                        )
                    if zipwire is not None:
                        actual_names = {
                            str(member.get("name"))
                            for member in zipwire.get("members", [])
                            if isinstance(member, Mapping) and member.get("name")
                        }
                        for name in manifest_member_names(raw_manifest):
                            if name not in actual_names:
                                blockers.append(f"archive_manifest_member_mismatch:{name}")
    facts["archive_manifest_path"] = manifest_path
    facts["archive_manifest"] = manifest_payload
    gate_blockers, gate_facts = validate_hdm8_selector_cuda_gate_context(
        row,
        manifest_payload,
        expected_archive_sha256=facts.get("archive_sha256")
        if isinstance(facts.get("archive_sha256"), str)
        else expected_sha,
    )
    blockers.extend(gate_blockers)
    facts.update(gate_facts)

    runtime_manifest: dict[str, Any] | None = None
    if inflate_sh is not None and inflate_sh.is_file():
        try:
            runtime_manifest = runtime_dependency_manifest(submission_dir, repo_root)
        except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
            blockers.append(f"runtime_manifest_error:{exc}")
    if not isinstance(runtime_manifest, dict) or not is_sha256(
        runtime_manifest.get("runtime_tree_sha256")
    ):
        blockers.append("runtime_tree_sha256_missing")
    if not isinstance(runtime_manifest, dict) or not is_sha256(
        runtime_manifest.get("runtime_content_tree_sha256")
    ):
        blockers.append("runtime_content_tree_sha256_missing")
    facts["runtime_manifest"] = runtime_manifest

    if isinstance(effective_lane_id, str) and effective_lane_id.strip():
        candidate_runtime_sha = (
            runtime_manifest.get("runtime_tree_sha256")
            if isinstance(runtime_manifest, Mapping)
            else None
        )
        candidate_runtime_content_sha = (
            runtime_manifest.get("runtime_content_tree_sha256")
            if isinstance(runtime_manifest, Mapping)
            else None
        )
        blockers.extend(
            terminal_claim_result_conflicts(
                effective_lane_id,
                facts.get("archive_sha256")
                if isinstance(facts.get("archive_sha256"), str)
                else expected_sha,
                dispatch_claims_path=dispatch_claims_path,
                active_floor_score=active_floor_score,
                runtime_tree_sha256=candidate_runtime_sha
                if isinstance(candidate_runtime_sha, str)
                else None,
                runtime_content_tree_sha256=candidate_runtime_content_sha
                if isinstance(candidate_runtime_content_sha, str)
                else None,
                score_affecting_runtime_changed=as_bool(
                    row.get("score_affecting_runtime_changed")
                ),
            )
        )

    proof_blockers, proof_facts = validate_runtime_consumption_proof(
        row,
        repo_root=repo_root,
        queue_dir=queue_dir,
        submission_dir=submission_dir,
        archive_sha256=facts.get("archive_sha256")
        if isinstance(facts.get("archive_sha256"), str)
        else None,
    )
    blockers.extend(proof_blockers)
    facts.update(proof_facts)

    if (
        active_floor_archive_bytes is not None
        and isinstance(facts.get("archive_bytes"), int)
        and int(facts["archive_bytes"]) > active_floor_archive_bytes
    ):
        if not allow_above_active_floor_dispatch:
            score_text = (
                f", active_score_frontier={active_floor_score:.12f}"
                if active_floor_score is not None
                else ""
            )
            blockers.append(
                "above_active_floor_archive_bytes_without_operator_override:"
                f"{facts['archive_bytes']}>{active_floor_archive_bytes}{score_text}; "
                "above rate-only byte floor"
            )
        elif not operator_override_reason:
            blockers.append("above_active_floor_override_missing_reason")

    return blockers, facts


def promoted_row(
    source_row: Mapping[str, Any],
    *,
    source_queue_path: Path,
    source_queue_list: str,
    facts: Mapping[str, Any],
    repo_root: Path,
    operator_override_reason: str | None = None,
) -> dict[str, Any]:
    archive_path = facts["archive_path"]
    submission_dir = facts["submission_dir"]
    inflate_sh = facts["inflate_sh"]
    manifest_path = facts["archive_manifest_path"]
    runtime_manifest = facts["runtime_manifest"]
    archive_bytes = int(facts["archive_bytes"])
    archive_sha = str(facts["archive_sha256"])
    candidate_id = str(source_row["candidate_id"])
    source_fields = {
        key: source_row[key]
        for key in (
            "candidate_family",
            "lane_class",
            "optimizer_tool",
            "op_params",
            "coords",
            "source_paths",
            "source_manifest_path",
        )
        if key in source_row
    }
    return {
        "candidate_id": candidate_id,
        "source_candidate_id": candidate_id,
        "source_queue_path": repo_rel(source_queue_path, repo_root),
        "source_queue_list": source_queue_list,
        "source_candidate_fields": source_fields,
        "lane_id": facts["lane_id"],
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime",
        "score_axis": "contest_cuda",
        "target_score_axis": "contest_cuda",
        "ready_for_exact_eval_dispatch": True,
        "dispatch_packet_ready": True,
        "dispatch_attempted": False,
        "score_claim": False,
        "score_claim_verified": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "evidence_semantics": "byte_closed_archive_runtime_ready_for_exact_eval",
        "evidence_grade": "[exact-eval-ready-no-score]",
        "cpu_or_proxy_score_not_cuda_evidence": True,
        "cuda_gap_review_required_before_promotion": True,
        "contest_dispatch_verdict": "ready_for_contest_exact_eval_dispatch_after_lane_claim",
        "archive_path": repo_rel(archive_path, repo_root),
        "candidate_archive_path": repo_rel(archive_path, repo_root),
        "archive_sha256": archive_sha,
        "candidate_archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "archive_bytes": archive_bytes,
        "candidate_archive_bytes": archive_bytes,
        "submission_dir": repo_rel(submission_dir, repo_root),
        "inflate_sh_path": repo_rel(inflate_sh, repo_root),
        "archive_manifest_path": repo_rel(manifest_path, repo_root),
        "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
        "runtime_content_tree_sha256": runtime_manifest[
            "runtime_content_tree_sha256"
        ],
        "runtime_manifest": runtime_manifest,
        "runtime_consumption_proof_required": facts.get(
            "runtime_consumption_proof_required"
        ),
        "runtime_consumption_proof_status": facts.get(
            "runtime_consumption_proof_status"
        ),
        "runtime_consumption_proof_backed_by_full_frame_parity": facts.get(
            "runtime_consumption_proof_backed_by_full_frame_parity"
        ),
        "runtime_consumption_proof_path": repo_rel(
            facts["runtime_consumption_proof_path"],
            repo_root,
        )
        if isinstance(facts.get("runtime_consumption_proof_path"), Path)
        else None,
        "runtime_consumption_proof_sha256": facts.get(
            "runtime_consumption_proof_sha256"
        ),
        "runtime_consumption_proof_schema": facts.get(
            "runtime_consumption_proof_schema"
        ),
        "runtime_consumption_proof_archive_sha256": facts.get(
            "runtime_consumption_proof_archive_sha256"
        ),
        "runtime_consumption_proof_candidate_member_sha256": facts.get(
            "runtime_consumption_proof_candidate_member_sha256"
        ),
        "renderer_payload_dfl1_inflate_parity_satisfied": source_row.get(
            "renderer_payload_dfl1_inflate_parity_satisfied"
        ),
        "renderer_payload_dfl1_inflate_parity_proof_path": source_row.get(
            "renderer_payload_dfl1_inflate_parity_proof_path"
        )
        or source_row.get("renderer_payload_dfl1_full_frame_inflate_parity_proof_path"),
        "renderer_payload_dfl1_inflate_parity_proof_sha256": source_row.get(
            "renderer_payload_dfl1_inflate_parity_proof_sha256"
        )
        or source_row.get("renderer_payload_dfl1_full_frame_inflate_parity_proof_sha256"),
        "full_frame_inflate_parity_proven": source_row.get(
            "full_frame_inflate_parity_proven"
        ),
        "full_frame_inflate_parity_verification": source_row.get(
            "full_frame_inflate_parity_verification"
        ),
        "cuda_component_risk_gate_required": bool(
            facts.get("hdm8_selector_cuda_component_gate_required")
        ),
        "cuda_component_risk_gate_status": facts.get(
            "hdm8_selector_cuda_component_gate_status"
        ),
        "cuda_component_risk_gate": facts.get("hdm8_selector_cuda_component_gate"),
        "score_affecting_payload_changed": bool(
            as_bool(source_row.get("score_affecting_payload_changed"))
        ),
        "charged_bits_changed": bool(as_bool(source_row.get("charged_bits_changed"))),
        "score_affecting_runtime_changed": bool(
            as_bool(source_row.get("score_affecting_runtime_changed"))
        ),
        "score_affecting_change_proofs": list(
            facts.get("score_affecting_change_proofs") or []
        ),
        "serialized_archive_delta": source_row.get("serialized_archive_delta"),
        "readiness_gate_tool": TOOL_NAME,
        "readiness_gate_generated_at_utc": utc_now(),
        "operator_override_reason": operator_override_reason,
        "dispatch_claim_required_before_gpu_or_remote_eval": True,
        "dispatch_blockers": [],
    }


def build_promoted_queue(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": QUEUE_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "n_candidates": 1,
        "top_k_count": 1,
        "dispatch_ready_count": 1,
        "dispatch_ready": [dict(row)],
        "top_k": [dict(row)],
        "top_k_forensic": [dict(row)],
        "evidence_boundary": {
            "score_claim": False,
            "exact_cuda_required_before_score_or_rank_claim": True,
            "lane_dispatch_claim_required_before_gpu_or_remote_eval": True,
            "readiness_scope": "local_byte_closed_archive_runtime_custody_only",
            "cpu_or_proxy_score_not_cuda_evidence": True,
            "cuda_gap_review_required_before_promotion": True,
        },
    }


def promote_candidate_for_exact_eval(
    queue_path: Path,
    candidate_id: str,
    *,
    repo_root: Path,
    submission_dir: Path | None = None,
    archive_manifest_path: Path | None = None,
    lane_id: str | None = None,
    active_floor_archive_bytes: int | None = ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
    extra_clearable_source_blockers: Iterable[str] = (),
    dispatch_claims_path: Path | None = None,
    claim_ttl_hours: float = 24.0,
) -> dict[str, Any]:
    queue_payload = read_json(queue_path)
    if not isinstance(queue_payload, dict):
        raise ExactReadinessError("source_queue_not_object")
    if queue_payload.get("schema") != "optimizer_candidate_queue_v1":
        raise ExactReadinessError(
            f"source_queue_schema_unsupported:{queue_payload.get('schema')!r}"
        )
    source_row, source_list = find_candidate(queue_payload, candidate_id)
    if source_row is None or source_list is None:
        raise ExactReadinessError("source_candidate_missing")
    blockers, facts = readiness_blockers(
        source_row,
        repo_root=repo_root,
        queue_dir=queue_path.parent,
        submission_dir=submission_dir,
        archive_manifest_path=archive_manifest_path,
        lane_id=lane_id,
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_score=active_floor_score,
        allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
        operator_override_reason=operator_override_reason,
        extra_clearable_source_blockers=extra_clearable_source_blockers,
        dispatch_claims_path=dispatch_claims_path,
        claim_ttl_hours=claim_ttl_hours,
    )
    report = {
        "schema": "optimizer_candidate_exact_eval_readiness_report_v1",
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "source_queue_path": repo_rel(queue_path, repo_root),
        "candidate_id": candidate_id,
        "ready_for_exact_eval_dispatch": not blockers,
        "blockers": blockers,
        "facts": {
            "archive_path": repo_rel(facts["archive_path"], repo_root)
            if isinstance(facts.get("archive_path"), Path)
            else None,
            "archive_sha256": facts.get("archive_sha256"),
            "archive_bytes": facts.get("archive_bytes"),
            "source_archive_bytes": facts.get("source_archive_bytes"),
            "realized_archive_byte_delta": facts.get("realized_archive_byte_delta"),
            "serialized_archive_delta": facts.get("serialized_archive_delta"),
            "inverse_scorer_full_frame_output_parity": facts.get(
                "inverse_scorer_full_frame_output_parity"
            ),
            "submission_dir": repo_rel(facts["submission_dir"], repo_root)
            if isinstance(facts.get("submission_dir"), Path)
            else None,
            "inflate_sh": repo_rel(facts["inflate_sh"], repo_root)
            if isinstance(facts.get("inflate_sh"), Path)
            else None,
            "archive_manifest_path": repo_rel(facts["archive_manifest_path"], repo_root)
            if isinstance(facts.get("archive_manifest_path"), Path)
            else None,
            "runtime_tree_sha256": (
                facts.get("runtime_manifest") or {}
            ).get("runtime_tree_sha256")
            if isinstance(facts.get("runtime_manifest"), dict)
            else None,
            "runtime_consumption_proof_required": facts.get(
                "runtime_consumption_proof_required"
            ),
            "runtime_consumption_proof_status": facts.get(
                "runtime_consumption_proof_status"
            ),
            "runtime_consumption_proof_path": repo_rel(
                facts["runtime_consumption_proof_path"],
                repo_root,
            )
            if isinstance(facts.get("runtime_consumption_proof_path"), Path)
            else None,
            "runtime_consumption_proof_schema": facts.get(
                "runtime_consumption_proof_schema"
            ),
            "runtime_consumption_proof_sha256": facts.get(
                "runtime_consumption_proof_sha256"
            ),
            "runtime_consumption_proof_archive_sha256": facts.get(
                "runtime_consumption_proof_archive_sha256"
            ),
            "runtime_consumption_proof_candidate_member_sha256": facts.get(
                "runtime_consumption_proof_candidate_member_sha256"
            ),
            "cuda_component_risk_gate_required": facts.get(
                "hdm8_selector_cuda_component_gate_required"
            ),
            "cuda_component_risk_gate_status": facts.get(
                "hdm8_selector_cuda_component_gate_status"
            ),
            "cuda_component_risk_gate_evidence_axis": facts.get(
                "hdm8_selector_cuda_component_gate_evidence_axis"
            ),
            "source_score_fields_stripped": sorted(
                facts.get("stripped_source_score_fields") or []
            ),
            "score_affecting_change_proofs": sorted(
                facts.get("score_affecting_change_proofs") or []
            ),
        },
    }
    if blockers:
        return {"report": report, "promoted_queue": None}
    row = promoted_row(
        source_row,
        source_queue_path=queue_path,
        source_queue_list=source_list,
        facts=facts,
        repo_root=repo_root,
        operator_override_reason=operator_override_reason,
    )
    queue = build_promoted_queue(row)
    return {"report": report, "promoted_queue": queue}


def json_dumps(payload: Any) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n"


__all__ = [
    "ACTIVE_FLOOR_ARCHIVE_BYTES",
    "ACTIVE_FLOOR_SCORE",
    "ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL",
    "ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE",
    "ACTIVE_RATE_ONLY_FLOOR_SCORE",
    "ACTIVE_SCORE_FRONTIER_LABEL",
    "ACTIVE_SCORE_FRONTIER_SCORE",
    "QUEUE_SCHEMA",
    "ExactReadinessError",
    "as_integral",
    "promote_candidate_for_exact_eval",
    "validate_serialized_archive_delta_contract",
]
