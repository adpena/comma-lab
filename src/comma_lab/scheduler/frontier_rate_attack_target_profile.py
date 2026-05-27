# SPDX-License-Identifier: MIT
"""Target video/corpus binding for queue-owned frontier-rate attacks."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import ExperimentQueueError

TARGET_OPTIMIZATION_PROFILE_SCHEMA = (
    "frontier_rate_attack_target_optimization_profile.v1"
)
TARGET_OPTIMIZATION_PROFILE_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_target_optimization_profile_queue_metadata.v1"
)
DEFAULT_CONTEST_TARGET_VIDEO = Path("upstream/videos/0.mkv")
TARGET_OPTIMIZATION_MODES = frozenset(
    {
        "contest_video_overfit",
        "corpus_generalization",
        "hybrid_contest_plus_corpus",
    }
)


class FrontierRateAttackTargetProfileError(ExperimentQueueError):
    """Raised when target video/corpus binding is ambiguous or unsafe."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray, str)):
        return []
    return _unique_strings(values)


def _target_video_record(
    path: str | Path,
    *,
    repo_root: Path,
    role: str,
) -> dict[str, Any]:
    resolved = _resolve_path(path, repo_root=repo_root)
    exists = resolved.is_file()
    return {
        "schema": "frontier_rate_attack_target_video_record.v1",
        "path": _repo_rel(resolved, repo_root),
        "role": role,
        "exists": exists,
        "sha256": _sha256_file(resolved) if exists else None,
        "bytes": resolved.stat().st_size if exists else None,
        "allowed_use": "declared_optimization_target_video_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }


def build_frontier_target_optimization_profile(
    *,
    repo_root: str | Path,
    target_profile_id: str = "contest_video_0",
    target_mode: str = "contest_video_overfit",
    target_video_paths: Sequence[str | Path] = (),
    target_corpus_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Declare the video/corpus target for queue-owned optimization.

    Contest specialization is valid for this challenge, but it is declared as
    data here so the same materializer and receiver machinery can run on a
    different video or corpus manifest without hidden path assumptions.
    """

    repo = Path(repo_root)
    mode = str(target_mode or "").strip()
    if mode not in TARGET_OPTIMIZATION_MODES:
        raise FrontierRateAttackTargetProfileError(
            "target_mode must be one of "
            f"{sorted(TARGET_OPTIMIZATION_MODES)}; got {target_mode!r}"
        )
    profile_id = str(target_profile_id or "").strip()
    if not profile_id:
        raise FrontierRateAttackTargetProfileError("target_profile_id must be non-empty")
    raw_video_paths = tuple(target_video_paths) or (DEFAULT_CONTEST_TARGET_VIDEO,)
    videos = [
        _target_video_record(
            path,
            repo_root=repo,
            role="primary_contest_target" if index == 0 else "auxiliary_target",
        )
        for index, path in enumerate(raw_video_paths)
    ]
    corpus_manifest = None
    if target_corpus_manifest_path is not None:
        manifest_path = _resolve_path(target_corpus_manifest_path, repo_root=repo)
        manifest_exists = manifest_path.is_file()
        corpus_manifest = {
            "schema": "frontier_rate_attack_target_corpus_manifest_ref.v1",
            "path": _repo_rel(manifest_path, repo),
            "exists": manifest_exists,
            "sha256": _sha256_file(manifest_path) if manifest_exists else None,
            "bytes": manifest_path.stat().st_size if manifest_exists else None,
            "allowed_use": "declared_generalization_corpus_pointer_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
            **FALSE_AUTHORITY,
        }

    blockers: list[str] = []
    if not videos:
        blockers.append("target_profile_requires_at_least_one_video")
    for video in videos:
        if video.get("exists") is not True:
            blockers.append(f"target_video_missing:{video.get('path')}")
    if mode in {"corpus_generalization", "hybrid_contest_plus_corpus"}:
        if corpus_manifest is None:
            blockers.append("target_corpus_manifest_required_for_mode")
        elif corpus_manifest.get("exists") is not True:
            blockers.append(
                f"target_corpus_manifest_missing:{corpus_manifest.get('path')}"
            )

    profile = {
        "schema": TARGET_OPTIMIZATION_PROFILE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "target_profile_id": profile_id,
        "target_mode": mode,
        "declared_overfit_allowed": mode
        in {"contest_video_overfit", "hybrid_contest_plus_corpus"},
        "target_video_count": len(videos),
        "target_videos": videos,
        "target_corpus_manifest": corpus_manifest,
        "portability_contract": {
            "schema": "frontier_rate_attack_target_portability_contract.v1",
            "contest_specialization_is_declared_data_not_hardcoded_tool_behavior": True,
            "materializers_must_consume_target_profile_or_runtime_contracts": True,
            "corpus_runs_must_bind_manifest_before_execution": True,
            "default_contest_video_path": DEFAULT_CONTEST_TARGET_VIDEO.as_posix(),
            "allowed_use": "target_binding_contract_for_queue_owned_optimization",
            "forbidden_use": "score_claim_or_dispatch_authority",
            **FALSE_AUTHORITY,
        },
        "optimization_policy": {
            "schema": "frontier_rate_attack_target_optimization_policy.v1",
            "contest_policy": (
                "overfit_to_declared_target_video_for_contest_score_when_mode_allows"
            ),
            "generalization_policy": (
                "reuse_same_materializer_search_and_receiver_contracts_on_declared_"
                "video_or_corpus_inputs"
            ),
            "blocked_behavior": (
                "implicit_path_defaults_that_bind_non_target_archives_or_pr_specific_"
                "contexts_without_profile_contract"
            ),
            "allowed_use": "planner_target_policy_only",
            "forbidden_use": "score_claim_or_rank_kill_authority",
            **FALSE_AUTHORITY,
        },
        "blockers": _unique_strings(blockers),
        "profile_ready": not blockers,
        "allowed_use": "queue_owned_target_optimization_binding_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        profile,
        context=f"frontier_target_optimization_profile:{profile_id}",
    )
    return profile


def target_optimization_profile_queue_metadata(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the compact queue metadata form of a target profile."""

    require_no_truthy_authority_fields(
        profile,
        context="frontier_target_optimization_profile_queue_metadata.source_profile",
    )
    return {
        "schema": TARGET_OPTIMIZATION_PROFILE_QUEUE_METADATA_SCHEMA,
        "target_profile_schema": profile.get("schema"),
        "target_profile_id": profile.get("target_profile_id"),
        "target_mode": profile.get("target_mode"),
        "declared_overfit_allowed": profile.get("declared_overfit_allowed") is True,
        "target_video_paths": [
            str(row.get("path") or "")
            for row in profile.get("target_videos") or []
            if isinstance(row, Mapping)
        ],
        "target_video_sha256s": [
            str(row.get("sha256") or "")
            for row in profile.get("target_videos") or []
            if isinstance(row, Mapping) and row.get("sha256")
        ],
        "target_corpus_manifest_path": (
            profile.get("target_corpus_manifest", {}).get("path")
            if isinstance(profile.get("target_corpus_manifest"), Mapping)
            else None
        ),
        "profile_ready": profile.get("profile_ready") is True,
        "blockers": _string_list(profile.get("blockers")),
        "allowed_use": "queue_metadata_pointer_to_target_optimization_profile",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


__all__ = [
    "DEFAULT_CONTEST_TARGET_VIDEO",
    "TARGET_OPTIMIZATION_MODES",
    "TARGET_OPTIMIZATION_PROFILE_QUEUE_METADATA_SCHEMA",
    "TARGET_OPTIMIZATION_PROFILE_SCHEMA",
    "FrontierRateAttackTargetProfileError",
    "build_frontier_target_optimization_profile",
    "target_optimization_profile_queue_metadata",
]
