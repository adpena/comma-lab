# SPDX-License-Identifier: MIT
"""Compile frontier materializer feedback into queue-owned follow-up surfaces."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.storage_tiers import DEFAULT_RESERVE_FREE_GB
from tac.optimization.dqs1_materializer_feedback_bridge import (
    DQS1_OBSERVATION_SOURCE_SCHEMA,
    DQS1_OBSERVATION_SWEEP_CONFIG_ID,
    FALSE_AUTHORITY,
    build_dqs1_materializer_feedback_bridge,
)
from tac.optimization.local_cpu_contest_drift import (
    EUREKA_SIGNAL_SCHEMA,
    LocalCPUContestDriftError,
    require_eureka_false_authority,
)
from tac.optimization.materializer_feedback import (
    FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
    FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
    materializer_observation_feedback_rows,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    MLXDynamicSweepObservationError,
    load_observation_rows,
    observation_duplicate_key,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .dqs1_local_first_queue import (
    DEFAULT_QUEUE_ID,
    DEFAULT_RESULTS_ROOT,
    PAIR_FRAME_GEOMETRY_QUEUE_REQUEST_SCHEMA,
    build_queue_from_action_summary,
)
from .experiment_queue import ExperimentQueueError, normalize_queue_definition

FEEDBACK_REFRESH_SCHEMA = "frontier_rate_attack_feedback_refresh.v1"
FRONTIER_RATE_ATTACK_FEEDBACK_REFRESH_SCHEMA = FEEDBACK_REFRESH_SCHEMA
PAIR_FRAME_GEOMETRY_DISCOVERY_SCHEMA = (
    "frontier_rate_attack_pair_frame_geometry_discovery.v1"
)
DISCOVERED_PAIR_FRAME_GEOMETRY_SCHEMA = (
    "frontier_rate_attack_discovered_pair_frame_geometry_lattice.v1"
)
PAIR_FRAME_GEOMETRY_LATTICE_SCHEMA = "pair_frame_scorer_geometry_lattice.v1"
MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA = (
    "frontier_rate_attack_materializer_feedback_discovery.v1"
)
DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA = (
    "frontier_rate_attack_discovered_materializer_feedback.v1"
)
LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA = "frontier_rate_attack_local_cpu_eureka_discovery.v1"
LOCAL_CPU_EUREKA_PLANNER_HINT_SCHEMA = "frontier_rate_attack_local_cpu_eureka_planner_hint.v1"
LOCAL_CPU_EUREKA_PAIRSET_PROFILE_SCHEMA = (
    "frontier_rate_attack_local_cpu_eureka_pairset_acquisition_profile.v1"
)
DQS1_OBSERVATION_DISCOVERY_SCHEMA = "frontier_rate_attack_dqs1_observation_discovery.v1"


class FrontierRateAttackFeedbackError(ExperimentQueueError):
    """Raised when frontier feedback discovery or compilation is unsafe."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrontierRateAttackFeedbackError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackError(f"{path}: expected JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FrontierRateAttackFeedbackError(f"{path}: cannot read JSONL") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}:{index}: invalid JSONL row: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise FrontierRateAttackFeedbackError(
                f"{path}:{index}: expected JSON object row"
            )
        rows.append(row)
    return rows


def _finite_float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _is_materializer_feedback_payload(payload: Mapping[str, Any]) -> bool:
    schema = str(payload.get("schema") or "")
    observation_kind = str(payload.get("observation_kind") or "")
    if schema in {
        FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
        FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
    }:
        return True
    if observation_kind == "family_agnostic_materializer_empirical_observation":
        return True
    observations = payload.get("observations")
    if observations is None:
        observations = payload.get("rows")
    if isinstance(observations, list):
        return any(
            isinstance(row, Mapping) and _is_materializer_feedback_payload(row)
            for row in observations
        )
    return False


def _materializer_feedback_paths(root: Path, *, max_files: int) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        raise FrontierRateAttackFeedbackError(f"feedback root does not exist: {root}")
    candidates: list[Path] = []
    scanned_files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        scanned_files += 1
        if scanned_files > max_files:
            raise FrontierRateAttackFeedbackError(
                f"{root}: materializer feedback discovery exceeded max_files={max_files}"
            )
        if path.name in {"sweep.json", "observations.jsonl"} or (path.suffix in {".json", ".jsonl"} and "materializer" in path.as_posix()):
            candidates.append(path)
    return candidates


def _payload_from_materializer_feedback_path(path: Path) -> dict[str, Any] | None:
    if path.suffix == ".jsonl":
        rows = _load_jsonl(path)
        materializer_rows = [
            row for row in rows if _is_materializer_feedback_payload(row)
        ]
        if not materializer_rows:
            return None
        for index, row in enumerate(materializer_rows):
            require_no_truthy_authority_fields(
                row,
                context=f"frontier_rate_attack_feedback.jsonl[{index}]",
            )
        return {
            "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
            "source_format": "jsonl_observation_rows",
            "observations": materializer_rows,
            **FALSE_AUTHORITY,
        }
    payload = _load_json(path)
    if not _is_materializer_feedback_payload(payload):
        return None
    require_no_truthy_authority_fields(
        payload,
        context="frontier_rate_attack_feedback.materializer_payload",
    )
    return payload


def _pair_frame_geometry_paths(root: Path, *, max_files: int) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        raise FrontierRateAttackFeedbackError(
            f"pair-frame geometry root does not exist: {root}"
        )
    candidates: list[Path] = []
    scanned_files = 0
    patterns = (
        "*pair_frame*geometry*lattice*.json",
        "*pair_frame_lattice*.json",
    )
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            scanned_files += 1
            if scanned_files > max_files:
                raise FrontierRateAttackFeedbackError(
                    f"{root}: pair-frame geometry discovery exceeded max_files={max_files}"
                )
            key = path.resolve(strict=False).as_posix()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)
    return candidates


def _pair_frame_geometry_requests(payload: Mapping[str, Any], *, path: Path) -> list[dict[str, Any]]:
    if payload.get("schema") != PAIR_FRAME_GEOMETRY_LATTICE_SCHEMA:
        return []
    require_no_truthy_authority_fields(
        payload,
        context=f"{path} pair-frame geometry lattice",
    )
    requests = payload.get("queue_executable_pairset_drop_requests")
    if not isinstance(requests, list):
        return []
    out: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            continue
        if request.get("schema") != PAIR_FRAME_GEOMETRY_QUEUE_REQUEST_SCHEMA:
            continue
        require_no_truthy_authority_fields(
            request,
            context=f"{path} pair-frame geometry request[{index}]",
        )
        if request.get("queue_executable") is not True:
            continue
        out.append(dict(request))
    return out


def discover_pair_frame_geometry_queue_requests(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    pair_frame_geometry_paths: Sequence[str | Path] = (),
    max_files_per_root: int = 256,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[str, Any]]:
    """Discover queue-executable pair-frame geometry requests.

    These requests are local DQS1 starts, not scorer authority.  Discovery is
    deliberately conservative: only typed lattice JSONs with false authority and
    typed queue-executable request rows are forwarded to the queue builder.
    """

    repo = Path(repo_root)
    default_roots = not frontier_artifact_roots
    roots: Sequence[str | Path] = (
        frontier_artifact_roots if not default_roots else (repo / ".omx" / "research",)
    )
    paths: list[Path] = []
    seen_paths: set[str] = set()
    for value in pair_frame_geometry_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            paths.append(path)
    for value in roots:
        root = _resolve_path(value, repo_root=repo)
        if default_roots and not root.exists():
            continue
        for path in _pair_frame_geometry_paths(root, max_files=max_files_per_root):
            if path.as_posix() in seen_paths:
                continue
            seen_paths.add(path.as_posix())
            paths.append(path)

    requests: list[dict[str, Any]] = []
    source_paths: list[str] = []
    discovered: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()
    duplicate_request_count = 0
    for path in paths:
        rel_path = _repo_rel(path, repo)
        payload = _load_json(path)
        try:
            path_requests = _pair_frame_geometry_requests(payload, path=path)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(f"{path}: {exc}") from exc
        if not path_requests:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": "no_queue_executable_pair_frame_geometry_requests",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        unique_requests: list[dict[str, Any]] = []
        for request in path_requests:
            request_id = str(request.get("candidate_id") or "")
            if request_id in seen_request_ids:
                duplicate_request_count += 1
                continue
            seen_request_ids.add(request_id)
            unique_requests.append(request)
        if not unique_requests:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": "duplicate_pair_frame_geometry_requests",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        requests.extend(unique_requests)
        source_paths.extend([rel_path] * len(unique_requests))
        discovered.append(
            {
                "schema": DISCOVERED_PAIR_FRAME_GEOMETRY_SCHEMA,
                "path": rel_path,
                "request_count": len(unique_requests),
                "candidate_ids": [
                    str(request.get("candidate_id")) for request in unique_requests
                ],
                "drop_counts": [
                    len(request.get("dropped_pair_indices") or [])
                    for request in unique_requests
                ],
                **FALSE_AUTHORITY,
            }
        )

    discovery = {
        "schema": PAIR_FRAME_GEOMETRY_DISCOVERY_SCHEMA,
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in roots
        ],
        "explicit_pair_frame_geometry_paths": [
            _repo_rel(_resolve_path(path, repo_root=repo), repo)
            for path in pair_frame_geometry_paths
        ],
        "scanned_candidate_path_count": len(paths),
        "discovered_lattice_count": len(discovered),
        "queue_executable_request_count": len(requests),
        "duplicate_request_count": duplicate_request_count,
        "discovered_lattices": discovered,
        "ignored_lattices": ignored,
        **FALSE_AUTHORITY,
    }
    return tuple(requests), tuple(source_paths), discovery


def _materializer_observation_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("observation_id") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("target_kind") or ""),
        str(row.get("materializer_id") or ""),
        str(row.get("source_archive_sha256") or ""),
        str(row.get("candidate_archive_sha256") or ""),
        str(row.get("saved_bytes") or ""),
        str(row.get("selected_member_name") or ""),
        ",".join(str(item) for item in row.get("selected_member_names") or []),
    )


def discover_materializer_feedback_payloads(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    materializer_feedback_paths: Sequence[str | Path] = (),
    max_files_per_root: int = 256,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[str, Any]]:
    """Discover family-agnostic materializer feedback under frontier roots."""

    repo = Path(repo_root)
    paths: list[Path] = []
    seen_paths: set[str] = set()
    for value in materializer_feedback_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            paths.append(path)
    for value in frontier_artifact_roots:
        root = _resolve_path(value, repo_root=repo)
        for path in _materializer_feedback_paths(root, max_files=max_files_per_root):
            if path.as_posix() in seen_paths:
                continue
            seen_paths.add(path.as_posix())
            paths.append(path)

    payloads: list[dict[str, Any]] = []
    source_paths: list[str] = []
    discovered: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    seen_observation_keys: set[tuple[str, ...]] = set()
    duplicate_observation_count = 0
    for path in paths:
        try:
            payload = _payload_from_materializer_feedback_path(path)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(f"{path}: {exc}") from exc
        rel_path = _repo_rel(path, repo)
        if payload is None:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": "not_family_agnostic_materializer_feedback",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        try:
            rows = materializer_observation_feedback_rows(payload, source_path=rel_path)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(f"{path}: {exc}") from exc
        unique_rows: list[dict[str, Any]] = []
        duplicate_rows = 0
        for row in rows:
            key = _materializer_observation_key(row)
            if key in seen_observation_keys:
                duplicate_rows += 1
                duplicate_observation_count += 1
                continue
            seen_observation_keys.add(key)
            unique_rows.append(row)
        if not unique_rows:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": (
                        "duplicate_materializer_observations"
                        if duplicate_rows
                        else "materializer_feedback_has_no_observation_rows"
                    ),
                    "duplicate_observation_count": duplicate_rows,
                    **FALSE_AUTHORITY,
                }
            )
            continue
        target_kinds = sorted(
            {
                str(row.get("target_kind"))
                for row in unique_rows
                if str(row.get("target_kind") or "").strip()
            }
        )
        payloads.append(
            {
                "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
                "source_payload_schema": payload.get("schema"),
                "source_format": payload.get("source_format") or path.suffix.lstrip("."),
                "observations": unique_rows,
                **FALSE_AUTHORITY,
            }
        )
        source_paths.append(rel_path)
        discovered.append(
            {
                "schema": DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA,
                "path": rel_path,
                "payload_schema": payload.get("schema"),
                "observation_count": len(unique_rows),
                "duplicate_observation_count": duplicate_rows,
                "target_kinds": target_kinds,
                "rate_positive_count": sum(
                    1 for row in unique_rows if row.get("rate_positive") is True
                ),
                "receiver_positive_rate_saving_count": sum(
                    1
                    for row in unique_rows
                    if row.get("rate_positive") is True
                    and (
                        row.get("receiver_contract_satisfied") is True
                        or row.get("inflate_parity_satisfied") is True
                    )
                ),
                **FALSE_AUTHORITY,
            }
        )

    discovery = {
        "schema": MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA,
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in frontier_artifact_roots
        ],
        "explicit_materializer_feedback_paths": [
            _repo_rel(_resolve_path(path, repo_root=repo), repo)
            for path in materializer_feedback_paths
        ],
        "scanned_candidate_path_count": len(paths),
        "discovered_feedback_count": len(payloads),
        "duplicate_observation_count": duplicate_observation_count,
        "discovered_feedback": discovered,
        "ignored_feedback_candidates": ignored,
        **FALSE_AUTHORITY,
    }
    return tuple(payloads), tuple(source_paths), discovery


def _eureka_signal_paths(root: Path, *, max_files: int) -> list[Path]:
    if root.is_file():
        return [root] if root.name.startswith("local_cpu_contest_drift_eureka_") else []
    if not root.exists():
        raise FrontierRateAttackFeedbackError(f"eureka root does not exist: {root}")
    candidates: list[Path] = []
    scanned_files = 0
    for path in sorted(root.rglob("local_cpu_contest_drift_eureka_*.json")):
        if not path.is_file():
            continue
        scanned_files += 1
        if scanned_files > max_files:
            raise FrontierRateAttackFeedbackError(
                f"{root}: eureka discovery exceeded max_files={max_files}"
            )
        candidates.append(path)
    return candidates


def _eureka_candidate_family(candidate_id: str) -> str:
    if candidate_id.startswith("pairset_drop_one_"):
        return "decoder_q_pairset_drop_one"
    if candidate_id.startswith("pairset_drop_two_"):
        return "decoder_q_pairset_drop_two"
    if candidate_id.startswith("pairset_component_combo_"):
        return "decoder_q_learned_multi_drop"
    if candidate_id.startswith("pairset_"):
        return "decoder_q_pairset"
    return "unknown"


def _eureka_gap_row(payload: Mapping[str, Any], *, path: Path, repo_root: Path) -> dict[str, Any]:
    try:
        require_eureka_false_authority(
            payload,
            context=f"{path} local CPU eureka signal",
        )
    except LocalCPUContestDriftError as exc:
        raise FrontierRateAttackFeedbackError(str(exc)) from exc
    require_no_truthy_authority_fields(
        payload,
        context=f"{path} local CPU eureka signal",
    )
    candidate_id = str(payload.get("candidate_id") or "")
    auth_frontier = _finite_float_or_none(payload.get("auth_frontier_score"))
    projected = _finite_float_or_none(payload.get("projected_contest_score"))
    conservative = _finite_float_or_none(
        payload.get("conservative_projected_contest_score")
    )
    eureka_margin = _finite_float_or_none(payload.get("eureka_margin"))
    projected_gap = None
    conservative_gap = None
    if auth_frontier is not None and projected is not None:
        projected_gap = projected - auth_frontier
    if auth_frontier is not None and conservative is not None:
        conservative_gap = conservative - auth_frontier
    return {
        "schema": EUREKA_SIGNAL_SCHEMA,
        "path": _repo_rel(path, repo_root),
        "candidate_id": candidate_id,
        "candidate_family": _eureka_candidate_family(candidate_id),
        "candidate_archive_sha256": str(payload.get("candidate_archive_sha256") or ""),
        "local_score": _finite_float_or_none(payload.get("local_score")),
        "projected_contest_score": projected,
        "conservative_projected_contest_score": conservative,
        "auth_frontier_score": auth_frontier,
        "projected_gap_vs_auth_frontier": projected_gap,
        "conservative_gap_vs_auth_frontier": conservative_gap,
        "eureka_margin": eureka_margin,
        "eureka_trigger": payload.get("eureka_trigger") is True,
        "recommended_action": str(payload.get("recommended_action") or ""),
        "trust_region": str(payload.get("trust_region") or ""),
        "candidate_trust_region_matches_calibration": (
            payload.get("candidate_trust_region_matches_calibration") is True
        ),
        "source_artifact": str(payload.get("source_artifact") or ""),
        **FALSE_AUTHORITY,
    }


def _eureka_sort_key(row: Mapping[str, Any]) -> tuple[float, float, str]:
    conservative_gap = row.get("conservative_gap_vs_auth_frontier")
    projected_gap = row.get("projected_gap_vs_auth_frontier")
    return (
        float(conservative_gap) if conservative_gap is not None else float("inf"),
        float(projected_gap) if projected_gap is not None else float("inf"),
        str(row.get("candidate_id") or ""),
    )


def _eureka_planner_hints(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    near_rows = [
        row
        for row in rows
        if row.get("recommended_action") == "observe_only"
        and row.get("eureka_trigger") is False
        and (
            (
                row.get("conservative_gap_vs_auth_frontier") is not None
                and float(row["conservative_gap_vs_auth_frontier"]) <= 1.0e-5
            )
            or (
                row.get("projected_gap_vs_auth_frontier") is not None
                and float(row["projected_gap_vs_auth_frontier"]) <= 5.0e-6
            )
        )
    ]
    drop_two_near = [
        row
        for row in near_rows
        if row.get("candidate_family") == "decoder_q_pairset_drop_two"
    ]
    hints: list[dict[str, Any]] = []
    if drop_two_near:
        source_ids = [
            str(row.get("candidate_id"))
            for row in sorted(drop_two_near, key=_eureka_sort_key)[:8]
        ]
        hints.append(
            {
                "schema": LOCAL_CPU_EUREKA_PLANNER_HINT_SCHEMA,
                "hint_id": "dqs1_expand_beyond_drop_two_near_boundary",
                "trigger": "near_frontier_observe_only_drop_two_cluster",
                "source_candidate_ids": source_ids,
                "recommended_candidate_families": [
                    "learned_multi_drop",
                    "drop_many_beam_pairwise_interaction_waterfill",
                    "within_selected_set_mask_feather_probe",
                    "master_gradient_constrained_low_sensitivity_drop",
                    "inverse_scorer_null_direction_masked_variant",
                ],
                "pairset_acquisition_profile": {
                    "schema": LOCAL_CPU_EUREKA_PAIRSET_PROFILE_SCHEMA,
                    "active": True,
                    "max_drop_two": 512,
                    "max_drop_many": 96,
                    "drop_many_counts": [3, 4, 6, 8],
                    "candidate_family": "dqs1_pairset_drop_many_local_first",
                    "rate_distortion_levels_considered": [
                        "bit",
                        "byte",
                        "packet_member",
                        "tensor_channel",
                        "pixel",
                        "region",
                        "boundary",
                        "frame",
                        "pair",
                        "batch",
                        "full_video",
                        "scorer_axis",
                        "receiver_runtime",
                    ],
                    "starting_point_policy": (
                        "expand from near-frontier drop-two rows into bounded "
                        "drop-many local probes before exact-axis authority"
                    ),
                    "blocked_family_requests": [
                        {
                            "family": "global_low_impact_full_pair_drop_probe",
                            "blocker": (
                                "requires pair-frame scorer-geometry lattice "
                                "binding before full-board pair/frame drops are "
                                "queue-executable"
                            ),
                            **FALSE_AUTHORITY,
                        },
                        {
                            "family": "within_selected_set_mask_feather_probe",
                            "blocker": (
                                "requires receiver/materializer support for "
                                "non-pair-drop mask semantics"
                            ),
                            **FALSE_AUTHORITY,
                        },
                        {
                            "family": "inverse_scorer_null_direction_masked_variant",
                            "blocker": (
                                "requires inverse-scorer action cell to runtime "
                                "materializer binding"
                            ),
                            **FALSE_AUTHORITY,
                        },
                    ],
                    **FALSE_AUTHORITY,
                },
                "rationale": (
                    "drop-two local CPU drift rows are close enough to the frontier "
                    "to guide acquisition, but too conservative to treat as the "
                    "optimization endpoint"
                ),
                "planner_consumers": [
                    "pairset_component_marginal_model",
                    "master_gradient",
                    "inverse_scorer_action_surface",
                    "frontier_rate_attack_feedback_cycle",
                ],
                "forbidden_use": "score_claim_or_exact_eval_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
    return hints


def discover_local_cpu_eureka_planning_signals(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    max_files_per_root: int = 256,
) -> dict[str, Any]:
    """Discover local advisory eureka rows and compile acquisition hints.

    These rows are not observations for score/rank.  They are acquisition
    priors for the next local queue cycle, especially when near-boundary
    drop-two rows imply the search should expand beyond drop-two.
    """

    repo = Path(repo_root)
    paths: list[Path] = []
    seen_paths: set[str] = set()
    for value in frontier_artifact_roots:
        root = _resolve_path(value, repo_root=repo)
        for path in _eureka_signal_paths(root, max_files=max_files_per_root):
            key = path.resolve(strict=False).as_posix()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            paths.append(path)

    rows: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    seen_signal_keys: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    for path in paths:
        payload = _load_json(path)
        if payload.get("schema") != EUREKA_SIGNAL_SCHEMA:
            ignored.append(
                {
                    "path": _repo_rel(path, repo),
                    "reason": "not_local_cpu_contest_drift_eureka_signal",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        row = _eureka_gap_row(payload, path=path, repo_root=repo)
        key = (
            str(row.get("candidate_id") or ""),
            str(row.get("candidate_archive_sha256") or ""),
            str(row.get("source_artifact") or ""),
        )
        if key in seen_signal_keys:
            duplicate_count += 1
            continue
        seen_signal_keys.add(key)
        rows.append(row)

    rows = sorted(rows, key=_eureka_sort_key)
    family_counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("candidate_family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
    hints = _eureka_planner_hints(rows)
    best = rows[0] if rows else None
    return {
        "schema": LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA,
        "active": bool(rows),
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in frontier_artifact_roots
        ],
        "signal_count": len(rows),
        "duplicate_signal_count": duplicate_count,
        "ignored_signal_candidates": ignored,
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "near_frontier_observe_only_count": len(
            [
                row
                for row in rows
                if row.get("recommended_action") == "observe_only"
                and row.get("eureka_trigger") is False
                and row.get("projected_gap_vs_auth_frontier") is not None
                and float(row["projected_gap_vs_auth_frontier"]) <= 5.0e-6
            ]
        ),
        "best_projected_gap_vs_auth_frontier": (
            None if best is None else best.get("projected_gap_vs_auth_frontier")
        ),
        "best_conservative_gap_vs_auth_frontier": (
            None if best is None else best.get("conservative_gap_vs_auth_frontier")
        ),
        "planner_hint_count": len(hints),
        "planner_hints": hints,
        "signal_rows": rows[:32],
        "allowed_use": "local_advisory_acquisition_prior_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def load_dqs1_observations(
    *,
    repo_root: str | Path,
    observation_paths: Sequence[str | Path],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Load and dedupe canonical DQS1 local-first observation JSONL rows."""

    repo = Path(repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []
    seen_rows: set[tuple[tuple[str, str | None], ...]] = set()
    seen_paths: set[str] = set()
    for value in observation_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.suffix != ".jsonl":
            raise FrontierRateAttackFeedbackError(
                f"{path}: DQS1 observations must be JSONL rows"
            )
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            source_paths.append(_repo_rel(path, repo))
        try:
            loaded = load_observation_rows(path)
        except OSError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}: cannot read DQS1 observation JSONL"
            ) from exc
        except MLXDynamicSweepObservationError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}: invalid DQS1 observation JSONL: {exc}"
            ) from exc
        for row in loaded:
            if (
                row.get("source_schema") != DQS1_OBSERVATION_SOURCE_SCHEMA
                or row.get("sweep_config_id") != DQS1_OBSERVATION_SWEEP_CONFIG_ID
            ):
                raise FrontierRateAttackFeedbackError(
                    f"{path}: non-local-first DQS1 observation row refused "
                    f"for candidate {row.get('candidate_id')!r}"
                )
            key = observation_duplicate_key(row)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            rows.append(row)
    return tuple(rows), tuple(source_paths)


def discover_dqs1_observation_jsonl_paths(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Find append-only DQS1 local-first observation JSONLs for queue feedback."""

    repo = Path(repo_root)
    roots: Sequence[str | Path] = (
        frontier_artifact_roots
        if frontier_artifact_roots
        else (repo / ".omx" / "research",)
    )
    paths: list[Path] = []
    seen: set[str] = set()
    for value in roots:
        root = _resolve_path(value, repo_root=repo)
        if root.is_file():
            candidates = [root] if root.name.startswith("dqs1_local_first_harvest_observations_") and root.suffix == ".jsonl" else []
        elif root.exists():
            candidates = list(root.rglob("dqs1_local_first_harvest_observations_*.jsonl"))
        else:
            candidates = []
        for path in sorted(candidates):
            key = path.resolve(strict=False).as_posix()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return {
        "schema": DQS1_OBSERVATION_DISCOVERY_SCHEMA,
        "active": bool(paths),
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in roots
        ],
        "discovered_observation_count": len(paths),
        "discovered_observation_jsonl_paths": [_repo_rel(path, repo) for path in paths],
        "allowed_use": "local_advisory_observation_replanning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _queue_summary(queue: Mapping[str, Any]) -> dict[str, Any]:
    experiments = queue.get("experiments")
    experiment_rows = experiments if isinstance(experiments, list) else []
    return {
        "queue_id": queue.get("queue_id"),
        "experiment_count": len(experiment_rows),
        "step_count": sum(
            len(exp.get("steps", []))
            for exp in experiment_rows
            if isinstance(exp, Mapping)
        ),
        "selected_candidate_ids": [
            str(exp.get("id"))
            for exp in experiment_rows
            if isinstance(exp, Mapping) and exp.get("id")
        ],
        **FALSE_AUTHORITY,
    }


def build_frontier_rate_attack_feedback_refresh(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    materializer_feedback_paths: Sequence[str | Path] = (),
    pair_frame_geometry_paths: Sequence[str | Path] = (),
    dqs1_observation_paths: Sequence[str | Path] = (),
    action_summary_path: str | Path | None = None,
    results_root: str = DEFAULT_RESULTS_ROOT,
    queue_id: str = DEFAULT_QUEUE_ID,
    candidate_limit: int = 4,
    skip_observed_dqs1_candidates: bool = True,
    local_cpu_concurrency: int = 1,
    local_io_concurrency: int = 1,
    include_raw_retention_plan: bool = True,
    raw_retention_execute: bool = False,
    raw_retention_action: str = "move",
    raw_retention_cold_store_roots: Sequence[str] = (),
    raw_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    include_mlx_retention_plan: bool = True,
    mlx_retention_execute: bool = False,
    mlx_retention_action: str = "move",
    mlx_retention_cold_store_roots: Sequence[str] = (),
    mlx_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
) -> dict[str, Any]:
    """Build a forest-level feedback refresh and optional DQS1 follow-up queue."""

    repo = Path(repo_root)
    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    payloads, source_paths, discovery = discover_materializer_feedback_payloads(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
        materializer_feedback_paths=materializer_feedback_paths,
    )
    pair_frame_requests, pair_frame_source_paths, pair_frame_discovery = (
        discover_pair_frame_geometry_queue_requests(
            repo_root=repo,
            frontier_artifact_roots=frontier_artifact_roots,
            pair_frame_geometry_paths=pair_frame_geometry_paths,
        )
    )
    eureka_planning = discover_local_cpu_eureka_planning_signals(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
    )
    dqs1_observation_discovery = discover_dqs1_observation_jsonl_paths(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
    )
    discovered_dqs1_observation_paths = tuple(
        str(path)
        for path in dqs1_observation_discovery.get(
            "discovered_observation_jsonl_paths",
            (),
        )
    )
    dqs1_observations, dqs1_source_paths = load_dqs1_observations(
        repo_root=repo,
        observation_paths=(
            *dqs1_observation_paths,
            *discovered_dqs1_observation_paths,
        ),
    )
    queue_payload: dict[str, Any] | None = None
    bridge: dict[str, Any] | None = None
    selected_pairset_acquisition: dict[str, Any] | None = None
    selected_candidate_ids: list[str] = []
    if action_summary_path is not None:
        result = build_queue_from_action_summary(
            _resolve_path(action_summary_path, repo_root=repo),
            repo_root=repo,
            results_root=results_root,
            queue_id=queue_id,
            candidate_limit=candidate_limit,
            materializer_feedback_payloads=payloads,
            materializer_feedback_source_paths=source_paths,
            dqs1_observations=dqs1_observations,
            dqs1_observation_source_paths=dqs1_source_paths,
            skip_observed_dqs1_candidates=skip_observed_dqs1_candidates,
            additional_queue_requests=pair_frame_requests,
            additional_queue_request_source_paths=pair_frame_source_paths,
            local_cpu_concurrency=local_cpu_concurrency,
            local_io_concurrency=local_io_concurrency,
            include_raw_retention_plan=include_raw_retention_plan,
            raw_retention_execute=raw_retention_execute,
            raw_retention_action=raw_retention_action,
            raw_retention_cold_store_roots=tuple(raw_retention_cold_store_roots),
            raw_retention_cold_store_reserve_gb=raw_retention_cold_store_reserve_gb,
            include_mlx_retention_plan=include_mlx_retention_plan,
            mlx_retention_execute=mlx_retention_execute,
            mlx_retention_action=mlx_retention_action,
            mlx_retention_cold_store_roots=tuple(mlx_retention_cold_store_roots),
            mlx_retention_cold_store_reserve_gb=mlx_retention_cold_store_reserve_gb,
        )
        queue_payload = normalize_queue_definition(result.queue)
        if eureka_planning.get("active") is True:
            for experiment in queue_payload.get("experiments", []):
                if not isinstance(experiment, dict):
                    continue
                metadata = experiment.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["frontier_feedback_eureka_planning"] = eureka_planning
        bridge = result.materializer_feedback_bridge
        selected_pairset_acquisition = result.selected_pairset_acquisition
        selected_candidate_ids = [selection.candidate_id for selection in result.selections]
    else:
        try:
            bridge = build_dqs1_materializer_feedback_bridge(
                materializer_feedback_payloads=payloads,
                materializer_feedback_source_paths=source_paths,
                candidate_limit=candidate_limit,
                dqs1_observations=dqs1_observations,
                dqs1_observation_source_paths=dqs1_source_paths,
            )
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(str(exc)) from exc

    return {
        "schema": FEEDBACK_REFRESH_SCHEMA,
        "generated_at_utc": _utc_now(),
        "discovery": discovery,
        "pair_frame_geometry_discovery": pair_frame_discovery,
        "pair_frame_geometry_request_source_paths": list(pair_frame_source_paths),
        "pair_frame_geometry_queue_request_count": len(pair_frame_requests),
        "local_cpu_eureka_planning": eureka_planning,
        "dqs1_observation_discovery": dqs1_observation_discovery,
        "materializer_feedback_source_paths": list(source_paths),
        "materializer_feedback_payload_count": len(payloads),
        "dqs1_observation_source_paths": list(dqs1_source_paths),
        "dqs1_observation_count": len(dqs1_observations),
        "action_summary_path": (
            None
            if action_summary_path is None
            else _repo_rel(_resolve_path(action_summary_path, repo_root=repo), repo)
        ),
        "queue_id": queue_id if queue_payload is not None else None,
        "results_root": results_root,
        "selected_candidate_ids": selected_candidate_ids,
        "selected_pairset_acquisition": selected_pairset_acquisition,
        "materializer_feedback_bridge": bridge,
        "queue_summary": None if queue_payload is None else _queue_summary(queue_payload),
        "queue": queue_payload,
        "retention_policy": {
            "schema": "frontier_rate_attack_feedback_retention_policy.v1",
            "raw_retention_plan_included": include_raw_retention_plan,
            "raw_retention_execute": raw_retention_execute,
            "raw_retention_action": raw_retention_action,
            "raw_retention_cold_store_roots": list(raw_retention_cold_store_roots),
            "raw_retention_cold_store_reserve_gb": raw_retention_cold_store_reserve_gb,
            "mlx_retention_plan_included": include_mlx_retention_plan,
            "mlx_retention_execute": mlx_retention_execute,
            "mlx_retention_action": mlx_retention_action,
            "mlx_retention_cold_store_roots": list(mlx_retention_cold_store_roots),
            "mlx_retention_cold_store_reserve_gb": mlx_retention_cold_store_reserve_gb,
            **FALSE_AUTHORITY,
        },
        "allowed_use": "queue_owned_frontier_feedback_replanning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


__all__ = [
    "DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA",
    "DQS1_OBSERVATION_DISCOVERY_SCHEMA",
    "FEEDBACK_REFRESH_SCHEMA",
    "FRONTIER_RATE_ATTACK_FEEDBACK_REFRESH_SCHEMA",
    "LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA",
    "LOCAL_CPU_EUREKA_PAIRSET_PROFILE_SCHEMA",
    "LOCAL_CPU_EUREKA_PLANNER_HINT_SCHEMA",
    "MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA",
    "PAIR_FRAME_GEOMETRY_DISCOVERY_SCHEMA",
    "FrontierRateAttackFeedbackError",
    "build_frontier_rate_attack_feedback_refresh",
    "discover_dqs1_observation_jsonl_paths",
    "discover_local_cpu_eureka_planning_signals",
    "discover_materializer_feedback_payloads",
    "discover_pair_frame_geometry_queue_requests",
    "load_dqs1_observations",
]
