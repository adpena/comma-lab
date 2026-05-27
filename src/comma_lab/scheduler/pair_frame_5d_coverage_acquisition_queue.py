# SPDX-License-Identifier: MIT
"""Queue builder for 5D canvas coverage-acquisition work orders."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.deploy.modal.paired_dispatch import (
    PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT,
    PAIRED_AUTH_EVAL_DISPATCH_TOOL,
    paired_auth_eval_dispatch_command_template,
)
from tac.deploy.modal.paired_dispatch_contract import (
    paired_auth_eval_dispatch_command_blockers,
)
from tac.local_acceleration.mlx_cache_audit import cache_audit_stamp_blockers
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_coverage import (
    COVERAGE_AUDIT_SCHEMA,
    WORK_ORDER_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.submission_packet.builder import (
    SUBMISSION_BUNDLE_SCHEMA_VERSION,
    submission_bundle_result_from_dict,
)

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA = (
    "pair_frame_5d_canvas_coverage_acquisition_plan.v1"
)
PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA = (
    "pair_frame_5d_canvas_coverage_acquisition_queue.v1"
)
PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA = (
    "pair_frame_5d_canvas_exact_axis_anchor_request.v1"
)
PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA = (
    "pair_frame_5d_canvas_mlx_negative_delta_request.v1"
)
PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA = (
    "pair_frame_5d_canvas_coverage_followup_readiness_report.v1"
)
PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA = (
    "pair_frame_5d_canvas_coverage_followup_execution_queue.v1"
)
PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA = (
    "pair_frame_5d_canvas_coverage_followup_input_binding_report.v1"
)
POPULATED_CANVAS_SCHEMA = "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1"
EXTENDED_OPERATOR_QUEUE_SCHEMA = "experiment_queue.v1"
WORKER_RESULT_SCHEMA = "experiment_queue_worker_result.v1"
MLX_SCORER_RESPONSE_SCHEMA = "mlx_scorer_response.v1"
REQUIRED_MLX_CACHE_ARRAY_KEYS = frozenset(
    {"pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"}
)
REQUIRED_MLX_CACHE_ARRAY_FILES = {
    "pair_indices": "pair_indices.npy",
    "posenet_yuv6_pair": "posenet_yuv6_pair.npy",
    "segnet_last_rgb": "segnet_last_rgb.npy",
}

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_packet_ready": False,
    "reproduction_claim": False,
    "reproduction_equivalence": False,
}
MLX_CACHE_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "promotable",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_") or "item"


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentQueueError(f"{path}: could not load JSON object") from exc
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{path}: expected JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_coverage_audit(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    if payload.get("schema") != COVERAGE_AUDIT_SCHEMA:
        raise ExperimentQueueError(
            f"{path}: expected {COVERAGE_AUDIT_SCHEMA}, got {payload.get('schema')!r}"
        )
    try:
        require_no_truthy_authority_fields(
            payload,
            context=f"pair_frame_5d_coverage_audit:{path}",
        )
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    return payload


def _work_orders(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(audit.get("work_orders") or [], start=1):
        if not isinstance(row, Mapping):
            continue
        if row.get("schema") != WORK_ORDER_SCHEMA:
            raise ExperimentQueueError(
                f"coverage audit work_orders[{index}] schema mismatch: "
                f"{row.get('schema')!r}"
            )
        try:
            require_no_truthy_authority_fields(
                row,
                context=f"pair_frame_5d_coverage_work_order:{row.get('id')}",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        rows.append(dict(row))
    rows.sort(key=lambda item: (int(item.get("priority") or 100), str(item.get("id"))))
    return rows


def _find_work_order(audit: Mapping[str, Any], work_order_id: str) -> dict[str, Any]:
    for row in _work_orders(audit):
        if row.get("id") == work_order_id:
            return row
    raise ExperimentQueueError(f"coverage audit does not contain work order {work_order_id!r}")


def _plan_class(work_order_id: str) -> tuple[str, bool, str, list[str]]:
    if work_order_id == "populate_missing_paired_cpu_cuda_axis_anchors":
        return (
            "contest_axis_anchor_acquisition_required",
            False,
            "cuda_auth",
            [
                "requires_byte_closed_submission_bundle_for_paired_auth_eval",
                "requires_dispatch_claim_before_paid_or_remote_exact_axis_work",
            ],
        )
    if work_order_id == "acquire_negative_delta_cells_before_operator_fanout":
        return (
            "local_mlx_negative_delta_probe_required",
            False,
            "local_mlx",
            [
                "requires_reference_and_candidate_mlx_cache_pair",
                "requires_mlx_scorer_response_or_inverse_surface_seed",
            ],
        )
    if work_order_id in {
        "densify_pair_coverage_for_grouped_search",
        "densify_frame_coverage_for_masked_and_feathered_search",
        "populate_receiver_runtime_mode_diversity",
        "populate_5d_canvas_from_master_gradient_anchors",
    }:
        return (
            "local_master_gradient_canvas_refresh",
            True,
            "local_cpu",
            [],
        )
    return ("manual_consumer_binding_required", False, "local_cpu", ["unknown_work_order"])


def _paired_axis_anchor_request(
    *,
    archive_sha256: str,
    output_base: Path,
    repo_root: Path,
) -> dict[str, Any]:
    lane_id_base = "lane_pair_frame_5d_coverage_axis_fill"
    dispatch_command = paired_auth_eval_dispatch_command_template(
        archive_path="<submission_dir>/archive.zip",
        submission_dir="<submission_dir>",
        lane_id_base=lane_id_base,
        archive_sha256=archive_sha256 or "<archive_sha256>",
        execute=False,
        label="pair_frame_5d_coverage_axis_fill",
        run_id="pair_frame_5d_coverage_axis_fill",
        output_root=_repo_rel(output_base / "paired_auth_eval", repo_root),
        claim_agent=PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT,
        claim_notes=(
            "5D coverage acquisition requires paired contest CPU/CUDA axis anchors"
        ),
    )
    dispatch_command_text = shlex.join(str(part) for part in dispatch_command)
    blockers = paired_auth_eval_dispatch_command_blockers(
        paired_dispatch_tool=PAIRED_AUTH_EVAL_DISPATCH_TOOL,
        command_template=dispatch_command_text,
        require_command=True,
    )
    request = {
        "schema": PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA,
        "work_order_id": "populate_missing_paired_cpu_cuda_axis_anchors",
        "archive_sha256": archive_sha256,
        "required_input_schema_version": SUBMISSION_BUNDLE_SCHEMA_VERSION,
        "required_inputs": [
            {
                "id": "submission_bundle_result",
                "schema_version": SUBMISSION_BUNDLE_SCHEMA_VERSION,
                "path_placeholder": "<submission_bundle_result.json>",
                "purpose": (
                    "bind archive.zip, submission_dir, inflate.sh, archive bytes, "
                    "and runtime closure before paired exact-axis dispatch"
                ),
            }
        ],
        "canonical_plan_entrypoint": [
            ".venv/bin/python",
            "tools/paired_auth_eval_cli.py",
            "--from-submission-bundle",
            "<submission_bundle_result.json>",
            "--cost-band",
            "smoke",
            "--cuda-gpu",
            "T4",
            "--cuda-platform",
            "modal",
            "--cpu-target",
            "linux_x86_64_modal",
            "--budget-usd",
            "1.00",
            "--dry-run",
            "--json",
        ],
        "canonical_paired_dispatch_tool": PAIRED_AUTH_EVAL_DISPATCH_TOOL,
        "paired_dispatch_command_template": dispatch_command,
        "paired_dispatch_command_template_text": dispatch_command_text,
        "paired_dispatch_command_contract_blockers": blockers,
        "exact_axes_required": ["contest_cpu", "contest_cuda"],
        "dispatch_claim_required": True,
        "execute_default": False,
        "allowed_use": "paired_exact_axis_anchor_acquisition_planning_only",
        "forbidden_use": "single_axis_dispatch_or_score_claim_without_auth_eval",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        request,
        context="pair_frame_5d_exact_axis_anchor_request",
    )
    return request


def _mlx_negative_delta_request(
    *,
    archive_sha256: str,
    output_base: Path,
    repo_root: Path,
) -> dict[str, Any]:
    response_path = output_base / "mlx_negative_delta" / "mlx_scorer_response.json"
    components_dir = output_base / "mlx_negative_delta" / "components"
    request = {
        "schema": PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA,
        "work_order_id": "acquire_negative_delta_cells_before_operator_fanout",
        "archive_sha256": archive_sha256,
        "required_inputs": [
            {
                "id": "reference_mlx_cache_dir",
                "path_placeholder": "<reference_mlx_cache_dir>",
                "requires_manifest": True,
            },
            {
                "id": "candidate_mlx_cache_dir",
                "path_placeholder": "<candidate_mlx_cache_dir>",
                "requires_manifest": True,
            },
            {
                "id": "archive_size_bytes",
                "value_placeholder": "<archive_size_bytes>",
            },
        ],
        "canonical_execution_tool": "tools/run_mlx_scorer_response_cache.py",
        "preferred_execution_queue_schema": "mlx_scorer_response_execution_queue_plan.v1",
        "command_template": [
            ".venv/bin/python",
            "tools/run_mlx_scorer_response_cache.py",
            "--reference-cache-dir",
            "<reference_mlx_cache_dir>",
            "--candidate-cache-dir",
            "<candidate_mlx_cache_dir>",
            "--archive-size-bytes",
            "<archive_size_bytes>",
            "--output",
            _repo_rel(response_path, repo_root),
            "--repo-root",
            ".",
            "--batch-pairs",
            "1",
            "--start-pair",
            "0",
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--allow-local-cpu-advisory-cache-identity",
            "--components-dir",
            _repo_rel(components_dir, repo_root),
            "--response-family",
            "pair_frame_5d_coverage_negative_delta",
        ],
        "response_output": _repo_rel(response_path, repo_root),
        "components_dir": _repo_rel(components_dir, repo_root),
        "batch_pairs": 1,
        "device": "gpu",
        "cache_identity_mode": "local_research_signal_only",
        "requires_exact_eval_before_spend_or_promotion": True,
        "allowed_use": "local_mlx_negative_delta_acquisition_planning_only",
        "forbidden_use": "score_claim_or_auth_axis_transfer_without_calibration",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        request,
        context="pair_frame_5d_mlx_negative_delta_request",
    )
    return request


def _followup_lane_requests(
    *,
    work_order_id: str,
    archive_sha256: str,
    output_base: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if work_order_id == "populate_missing_paired_cpu_cuda_axis_anchors":
        return [
            _paired_axis_anchor_request(
                archive_sha256=archive_sha256,
                output_base=output_base,
                repo_root=repo_root,
            )
        ]
    if work_order_id == "acquire_negative_delta_cells_before_operator_fanout":
        return [
            _mlx_negative_delta_request(
                archive_sha256=archive_sha256,
                output_base=output_base,
                repo_root=repo_root,
            )
        ]
    return []


def _optional_existing_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    try:
        return _load_json_object(path)
    except ExperimentQueueError:
        return None


def _replace_command_placeholders(
    command: Sequence[object],
    replacements: Mapping[str, object],
) -> list[str]:
    rendered: list[str] = []
    for item in command:
        text = str(item)
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, str(value))
        rendered.append(text)
    return rendered


def _candidate_search_roots(
    search_roots: Sequence[str | Path] | None,
    *,
    repo_root: Path,
) -> list[Path]:
    roots: list[Path] = []
    for raw in search_roots or []:
        root = _resolve_path(raw, repo_root=repo_root)
        if root.exists():
            roots.append(root)
    return roots


def _iter_named_json_files(roots: Sequence[Path], name: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        candidates = [root] if root.name == name and root.is_file() else []
        if root.is_dir():
            candidates.extend(root.rglob(name))
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            paths.append(path)
    return sorted(paths, key=lambda item: item.as_posix())


def _load_submission_bundle_contract(
    path: Path,
) -> tuple[Any | None, list[str]]:
    try:
        payload = _load_json_object(path)
    except ExperimentQueueError as exc:
        return None, [f"submission_bundle_result_not_json_object:{type(exc).__name__}"]
    try:
        return submission_bundle_result_from_dict(payload), []
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"submission_bundle_result_contract_invalid:{type(exc).__name__}"]


def _select_submission_bundle_for_archive(
    *,
    archive_sha256: str,
    repo_root: Path,
    explicit_path: Path | None,
    search_roots: Sequence[Path],
) -> tuple[Path | None, int | None, list[str], int]:
    paths = [explicit_path] if explicit_path is not None else _iter_named_json_files(
        search_roots,
        "submission_bundle_result.json",
    )
    paths = [path for path in paths if path is not None]
    if not paths:
        return None, None, ["submission_bundle_result_not_found_for_archive_sha256"], 0

    matches: list[tuple[Path, int]] = []
    blockers: list[str] = []
    for path in paths:
        resolved = _resolve_path(path, repo_root=repo_root)
        if not resolved.is_file():
            blockers.append("submission_bundle_result_path_not_found")
            continue
        bundle, bundle_blockers = _load_submission_bundle_contract(resolved)
        if bundle_blockers:
            blockers.extend(bundle_blockers)
            continue
        bundle_sha = str(bundle.archive_sha256)
        if archive_sha256 and bundle_sha != archive_sha256:
            continue
        matches.append((resolved, int(bundle.archive_bytes)))

    if len(matches) == 1:
        selected, archive_bytes = matches[0]
        return selected, archive_bytes, [], len(paths)
    if len(matches) > 1:
        return (
            None,
            None,
            ["submission_bundle_result_ambiguous_for_archive_sha256"],
            len(paths),
        )
    return (
        None,
        None,
        blockers or ["submission_bundle_result_not_found_for_archive_sha256"],
        len(paths),
    )


def _manifest_artifact_file_blockers(
    *,
    cache_dir: Path,
    manifest: Mapping[str, Any],
    key: str,
) -> list[str]:
    blockers: list[str] = []
    filename = REQUIRED_MLX_CACHE_ARRAY_FILES[key]
    path = cache_dir / filename
    if not path.is_file():
        return [f"mlx_cache_{key}_array_file_missing"]
    artifacts = manifest.get("artifacts")
    artifact = artifacts.get(key) if isinstance(artifacts, Mapping) else None
    if not isinstance(artifact, Mapping):
        return [f"mlx_cache_artifact_{key}_missing"]
    expected_bytes = artifact.get("bytes")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        blockers.append(f"mlx_cache_artifact_{key}_bytes_missing")
    elif path.stat().st_size != expected_bytes:
        blockers.append(f"mlx_cache_artifact_{key}_bytes_mismatch")
    expected_sha = artifact.get("sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        blockers.append(f"mlx_cache_artifact_{key}_sha256_missing")
    elif _sha256_file(path) != expected_sha:
        blockers.append(f"mlx_cache_artifact_{key}_sha256_mismatch")
    return blockers


def _mlx_cache_manifest_blockers(
    cache_dir: Path,
    *,
    candidate: bool,
    allow_local_cpu_advisory_identity: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, ["mlx_cache_manifest_missing"]
    try:
        manifest = _load_json_object(manifest_path)
    except ExperimentQueueError as exc:
        return None, [f"mlx_cache_manifest_invalid:{type(exc).__name__}"]

    blockers: list[str] = []
    if manifest.get("hash_only") is True:
        blockers.append("mlx_cache_hash_only_no_tensors")
    arrays = manifest.get("array_sha256")
    if not isinstance(arrays, Mapping):
        blockers.append("mlx_cache_array_sha256_missing")
        arrays = {}
    else:
        for key in sorted(REQUIRED_MLX_CACHE_ARRAY_KEYS):
            if not isinstance(arrays.get(key), str) or not arrays.get(key):
                blockers.append(f"mlx_cache_array_sha256_{key}_missing")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        blockers.append("mlx_cache_artifacts_missing")
    for key in sorted(REQUIRED_MLX_CACHE_ARRAY_KEYS):
        blockers.extend(
            _manifest_artifact_file_blockers(
                cache_dir=cache_dir,
                manifest=manifest,
                key=key,
            )
        )
    for key in (
        "archive_sha256",
        "raw_sha256",
        "inflated_outputs_aggregate_sha256",
        "hash_domain",
    ):
        if not isinstance(manifest.get(key), str) or not manifest.get(key):
            blockers.append(f"mlx_cache_{key}_missing")
    pair_count = manifest.get("pair_count")
    if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count < 1:
        blockers.append("mlx_cache_pair_count_missing_or_invalid")
    for field in MLX_CACHE_FALSE_AUTHORITY_FIELDS:
        if manifest.get(field) is not False:
            blockers.append(f"mlx_cache_manifest_{field}_not_false")

    if candidate:
        auth_blockers = cache_audit_stamp_blockers(
            manifest,
            cache_root=cache_dir,
            stamp_key="auth_eval_identity_audit",
            expected_verdict="PASS_CACHE_AUTH_EVAL_IDENTITY",
            require_identity_residual_zero=True,
            require_cache_shapes=True,
        )
        auth_audit = manifest.get("auth_eval_identity_audit")
        auth_ok = (
            manifest.get("eligible_for_local_mlx_transfer_calibration") is True
            and isinstance(auth_audit, Mapping)
            and auth_audit.get("verdict") == "PASS_CACHE_AUTH_EVAL_IDENTITY"
            and auth_audit.get("passed") is True
            and auth_audit.get("identity_residual") == 0
            and not auth_blockers
        )
        local_blockers = cache_audit_stamp_blockers(
            manifest,
            cache_root=cache_dir,
            stamp_key="local_cpu_advisory_cache_identity_audit",
            expected_verdict="PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
            require_identity_residual_zero=False,
            require_cache_shapes=False,
        )
        local_audit = manifest.get("local_cpu_advisory_cache_identity_audit")
        local_ok = (
            allow_local_cpu_advisory_identity
            and manifest.get("eligible_for_local_mlx_local_advisory_debug") is True
            and isinstance(local_audit, Mapping)
            and local_audit.get("verdict")
            == "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY"
            and local_audit.get("passed") is True
            and not local_blockers
        )
        if not auth_ok and not local_ok:
            blockers.append("candidate_mlx_cache_identity_audit_missing_or_invalid")
            if auth_blockers:
                blockers.extend(f"auth_identity:{blocker}" for blocker in auth_blockers)
            if local_blockers:
                blockers.extend(f"local_advisory_identity:{blocker}" for blocker in local_blockers)

    return manifest, blockers


def _mlx_cache_pair_blockers(
    reference_manifest: Mapping[str, Any] | None,
    candidate_manifest: Mapping[str, Any] | None,
) -> list[str]:
    if reference_manifest is None or candidate_manifest is None:
        return []
    blockers: list[str] = []
    for key in (
        "pair_count",
        "pair_indices_shape",
    ):
        if reference_manifest.get(key) != candidate_manifest.get(key):
            blockers.append(f"mlx_cache_pair_{key}_mismatch")
    reference_arrays = reference_manifest.get("array_sha256")
    candidate_arrays = candidate_manifest.get("array_sha256")
    if (
        isinstance(reference_arrays, Mapping)
        and isinstance(candidate_arrays, Mapping)
        and reference_arrays.get("pair_indices") != candidate_arrays.get("pair_indices")
    ):
        blockers.append("mlx_cache_pair_indices_sha256_mismatch")
    return blockers


def _iter_mlx_cache_dirs(roots: Sequence[Path]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            continue
        manifests: list[Path] = []
        if (root / "manifest.json").is_file():
            manifests.append(root / "manifest.json")
        if root.is_dir():
            manifests.extend(root.rglob("manifest.json"))
        for manifest in manifests:
            cache_dir = manifest.parent
            if not any((cache_dir / filename).is_file() for filename in REQUIRED_MLX_CACHE_ARRAY_FILES.values()):
                continue
            try:
                resolved = cache_dir.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            dirs.append(cache_dir)
    return sorted(dirs, key=lambda item: item.as_posix())


def _select_mlx_cache_for_request(
    *,
    request: Mapping[str, Any],
    repo_root: Path,
    explicit_path: Path | None,
    search_roots: Sequence[Path],
    candidate: bool,
) -> tuple[Path | None, dict[str, Any] | None, list[str], int]:
    archive_sha256 = str(request.get("archive_sha256") or "")
    paths = [explicit_path] if explicit_path is not None else _iter_mlx_cache_dirs(
        search_roots,
    )
    paths = [path for path in paths if path is not None]
    if explicit_path is None and paths:
        needle = "candidate" if candidate else "reference"
        preferred = [path for path in paths if needle in path.as_posix()]
        if preferred:
            paths = preferred
    if not paths:
        blocker = (
            "candidate_mlx_cache_dir_not_found_for_archive_sha256"
            if candidate
            else "reference_mlx_cache_dir_not_found"
        )
        return None, None, [blocker], 0

    matches: list[tuple[Path, dict[str, Any]]] = []
    blockers: list[str] = []
    for path in paths:
        resolved = _resolve_path(path, repo_root=repo_root)
        manifest, cache_blockers = _mlx_cache_manifest_blockers(
            resolved,
            candidate=candidate,
        )
        if cache_blockers:
            blockers.extend(cache_blockers)
            continue
        if candidate and archive_sha256 and manifest.get("archive_sha256") != archive_sha256:
            continue
        matches.append((resolved, manifest))

    if len(matches) == 1:
        return matches[0][0], matches[0][1], [], len(paths)
    if len(matches) > 1:
        blocker = (
            "candidate_mlx_cache_dir_ambiguous_for_archive_sha256"
            if candidate
            else "reference_mlx_cache_dir_ambiguous"
        )
        return None, None, [blocker], len(paths)
    blocker = (
        "candidate_mlx_cache_dir_not_found_for_archive_sha256"
        if candidate
        else "reference_mlx_cache_dir_not_found"
    )
    return None, None, blockers or [blocker], len(paths)


def _exact_axis_request_readiness(
    request: Mapping[str, Any],
    *,
    repo_root: Path,
    submission_bundle_path: Path | None,
) -> tuple[list[str], list[str] | None]:
    blockers = list(request.get("paired_dispatch_command_contract_blockers") or [])
    if submission_bundle_path is None:
        blockers.append("submission_bundle_result_path_missing")
        return blockers, None
    if not submission_bundle_path.is_file():
        blockers.append("submission_bundle_result_path_not_found")
        return blockers, None
    bundle = _optional_existing_json(submission_bundle_path)
    if bundle is None:
        blockers.append("submission_bundle_result_not_json_object")
        return blockers, None
    expected_schema = str(request.get("required_input_schema_version") or "")
    if bundle.get("schema_version") != expected_schema:
        blockers.append(
            "submission_bundle_result_schema_mismatch:"
            f"{bundle.get('schema_version')!r}"
        )
    try:
        bundle_result = submission_bundle_result_from_dict(bundle)
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(f"submission_bundle_result_contract_invalid:{type(exc).__name__}")
        return blockers, None
    request_sha = str(request.get("archive_sha256") or "")
    bundle_sha = str(bundle_result.archive_sha256)
    if request_sha and bundle_sha and request_sha != bundle_sha:
        blockers.append("submission_bundle_archive_sha256_mismatch")
    command = [
        ".venv/bin/python",
        "tools/paired_auth_eval_cli.py",
        "--from-submission-bundle",
        _repo_rel(submission_bundle_path, repo_root),
        "--cost-band",
        "smoke",
        "--cuda-gpu",
        "T4",
        "--cuda-platform",
        "modal",
        "--cpu-target",
        "linux_x86_64_modal",
        "--budget-usd",
        "1.00",
        "--dry-run",
        "--json",
    ]
    return blockers, None if blockers else command


def _mlx_request_readiness(
    request: Mapping[str, Any],
    *,
    repo_root: Path,
    reference_mlx_cache_dir: Path | None,
    candidate_mlx_cache_dir: Path | None,
    archive_size_bytes: int | None,
) -> tuple[list[str], list[str] | None]:
    blockers: list[str] = []
    reference_manifest: dict[str, Any] | None = None
    candidate_manifest: dict[str, Any] | None = None
    if reference_mlx_cache_dir is None:
        blockers.append("reference_mlx_cache_dir_missing")
    else:
        reference_manifest, reference_blockers = _mlx_cache_manifest_blockers(
            reference_mlx_cache_dir,
            candidate=False,
        )
        blockers.extend(f"reference:{blocker}" for blocker in reference_blockers)
    if candidate_mlx_cache_dir is None:
        blockers.append("candidate_mlx_cache_dir_missing")
    else:
        candidate_manifest, candidate_blockers = _mlx_cache_manifest_blockers(
            candidate_mlx_cache_dir,
            candidate=True,
        )
        blockers.extend(f"candidate:{blocker}" for blocker in candidate_blockers)
        request_sha = str(request.get("archive_sha256") or "")
        candidate_sha = (
            str(candidate_manifest.get("archive_sha256") or "")
            if candidate_manifest is not None
            else ""
        )
        if request_sha and candidate_sha and request_sha != candidate_sha:
            blockers.append("candidate:mlx_cache_archive_sha256_mismatch")
    blockers.extend(_mlx_cache_pair_blockers(reference_manifest, candidate_manifest))
    if (
        archive_size_bytes is None
        or isinstance(archive_size_bytes, bool)
        or archive_size_bytes <= 0
    ):
        blockers.append("archive_size_bytes_missing_or_invalid")
    replacements: dict[str, object] = {
        "<reference_mlx_cache_dir>": (
            ""
            if reference_mlx_cache_dir is None
            else _repo_rel(reference_mlx_cache_dir, repo_root)
        ),
        "<candidate_mlx_cache_dir>": (
            ""
            if candidate_mlx_cache_dir is None
            else _repo_rel(candidate_mlx_cache_dir, repo_root)
        ),
        "<archive_size_bytes>": archive_size_bytes or "",
    }
    command = _replace_command_placeholders(
        request.get("command_template") or [],
        replacements,
    )
    return blockers, None if blockers else command


def build_coverage_followup_readiness_report(
    *,
    repo_root: str | Path,
    plan_paths: Sequence[str | Path],
    submission_bundle_path: str | Path | None = None,
    reference_mlx_cache_dir: str | Path | None = None,
    candidate_mlx_cache_dir: str | Path | None = None,
    archive_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Classify typed follow-up requests as ready or fail-closed blocked."""

    repo = Path(repo_root)
    bundle = (
        None
        if submission_bundle_path is None
        else _resolve_path(submission_bundle_path, repo_root=repo)
    )
    reference_cache = (
        None
        if reference_mlx_cache_dir is None
        else _resolve_path(reference_mlx_cache_dir, repo_root=repo)
    )
    candidate_cache = (
        None
        if candidate_mlx_cache_dir is None
        else _resolve_path(candidate_mlx_cache_dir, repo_root=repo)
    )
    rows: list[dict[str, Any]] = []
    for raw_path in plan_paths:
        plan_path = _resolve_path(raw_path, repo_root=repo)
        plan = _load_json_object(plan_path)
        if plan.get("schema") != PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA:
            raise ExperimentQueueError(
                f"{plan_path}: expected {PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA}"
            )
        try:
            require_no_truthy_authority_fields(
                plan,
                context=f"pair_frame_5d_followup_readiness_plan:{plan_path}",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        for request in plan.get("followup_lane_requests") or []:
            if not isinstance(request, Mapping):
                continue
            try:
                require_no_truthy_authority_fields(
                    request,
                    context=(
                        "pair_frame_5d_followup_readiness_request:"
                        f"{plan_path}:{request.get('schema')}"
                    ),
                )
            except ValueError as exc:
                raise ExperimentQueueError(str(exc)) from exc
            schema = str(request.get("schema") or "")
            if schema == PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA:
                blockers, command = _exact_axis_request_readiness(
                    request,
                    repo_root=repo,
                    submission_bundle_path=bundle,
                )
            elif schema == PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA:
                blockers, command = _mlx_request_readiness(
                    request,
                    repo_root=repo,
                    reference_mlx_cache_dir=reference_cache,
                    candidate_mlx_cache_dir=candidate_cache,
                    archive_size_bytes=archive_size_bytes,
                )
            else:
                blockers = [f"unknown_followup_request_schema:{schema}"]
                command = None
            rows.append(
                {
                    "schema": "pair_frame_5d_canvas_followup_readiness_row.v1",
                    "plan_path": _repo_rel(plan_path, repo),
                    "work_order_id": request.get("work_order_id"),
                    "request_schema": schema,
                    "ready": not blockers,
                    "blockers": blockers,
                    "materialized_command": command,
                    "allowed_use": "queue_followup_readiness_routing_only",
                    **FALSE_AUTHORITY,
                }
            )
    ready_count = sum(1 for row in rows if row["ready"])
    report = {
        "schema": PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA,
        "plan_count": len(plan_paths),
        "request_count": len(rows),
        "ready_request_count": ready_count,
        "blocked_request_count": len(rows) - ready_count,
        "requests": rows,
        "allowed_use": "queue_followup_readiness_routing_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="pair_frame_5d_followup_readiness_report",
    )
    return report


def build_coverage_followup_input_binding_report(
    *,
    repo_root: str | Path,
    plan_paths: Sequence[str | Path],
    search_roots: Sequence[str | Path] | None = None,
    submission_bundle_path: str | Path | None = None,
    reference_mlx_cache_dir: str | Path | None = None,
    candidate_mlx_cache_dir: str | Path | None = None,
    archive_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Bind follow-up requests to concrete, custody-checked local inputs.

    The report is deliberately advisory: it discovers the strongest available
    local paths for the existing readiness gate, but does not itself claim
    score, promotion, dispatch readiness, or runtime parity.
    """

    repo = Path(repo_root)
    roots = _candidate_search_roots(search_roots, repo_root=repo)
    explicit_bundle = (
        None
        if submission_bundle_path is None
        else _resolve_path(submission_bundle_path, repo_root=repo)
    )
    explicit_reference_cache = (
        None
        if reference_mlx_cache_dir is None
        else _resolve_path(reference_mlx_cache_dir, repo_root=repo)
    )
    explicit_candidate_cache = (
        None
        if candidate_mlx_cache_dir is None
        else _resolve_path(candidate_mlx_cache_dir, repo_root=repo)
    )

    rows: list[dict[str, Any]] = []
    selected_bundle_paths: set[str] = set()
    selected_reference_cache_paths: set[str] = set()
    selected_candidate_cache_paths: set[str] = set()
    selected_archive_size_bytes: set[int] = set()
    for raw_path in plan_paths:
        plan_path = _resolve_path(raw_path, repo_root=repo)
        plan = _load_json_object(plan_path)
        if plan.get("schema") != PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA:
            raise ExperimentQueueError(
                f"{plan_path}: expected {PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA}"
            )
        try:
            require_no_truthy_authority_fields(
                plan,
                context=f"pair_frame_5d_followup_input_binding_plan:{plan_path}",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        for request in plan.get("followup_lane_requests") or []:
            if not isinstance(request, Mapping):
                continue
            try:
                require_no_truthy_authority_fields(
                    request,
                    context=(
                        "pair_frame_5d_followup_input_binding_request:"
                        f"{plan_path}:{request.get('schema')}"
                    ),
                )
            except ValueError as exc:
                raise ExperimentQueueError(str(exc)) from exc
            schema = str(request.get("schema") or "")
            blockers: list[str] = []
            row: dict[str, Any] = {
                "schema": "pair_frame_5d_canvas_followup_input_binding_row.v1",
                "plan_path": _repo_rel(plan_path, repo),
                "work_order_id": request.get("work_order_id"),
                "request_schema": schema,
                "archive_sha256": request.get("archive_sha256"),
                "allowed_use": "queue_followup_input_binding_only",
                **FALSE_AUTHORITY,
            }
            if schema == PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA:
                bundle, bundle_bytes, bundle_blockers, considered = (
                    _select_submission_bundle_for_archive(
                        archive_sha256=str(request.get("archive_sha256") or ""),
                        repo_root=repo,
                        explicit_path=explicit_bundle,
                        search_roots=roots,
                    )
                )
                blockers.extend(bundle_blockers)
                row["candidate_submission_bundle_count"] = considered
                if bundle is not None:
                    bundle_ref = _repo_rel(bundle, repo)
                    row["selected_submission_bundle_path"] = bundle_ref
                    selected_bundle_paths.add(bundle_ref)
                if bundle_bytes is not None:
                    row["selected_archive_size_bytes"] = bundle_bytes
                    selected_archive_size_bytes.add(bundle_bytes)
            elif schema == PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA:
                reference_cache, reference_manifest, reference_blockers, ref_count = (
                    _select_mlx_cache_for_request(
                        request=request,
                        repo_root=repo,
                        explicit_path=explicit_reference_cache,
                        search_roots=roots,
                        candidate=False,
                    )
                )
                candidate_cache, candidate_manifest, candidate_blockers, cand_count = (
                    _select_mlx_cache_for_request(
                        request=request,
                        repo_root=repo,
                        explicit_path=explicit_candidate_cache,
                        search_roots=roots,
                        candidate=True,
                    )
                )
                blockers.extend(f"reference:{blocker}" for blocker in reference_blockers)
                blockers.extend(f"candidate:{blocker}" for blocker in candidate_blockers)
                blockers.extend(_mlx_cache_pair_blockers(reference_manifest, candidate_manifest))
                row["candidate_reference_mlx_cache_count"] = ref_count
                row["candidate_candidate_mlx_cache_count"] = cand_count
                if reference_cache is not None:
                    reference_ref = _repo_rel(reference_cache, repo)
                    row["selected_reference_mlx_cache_dir"] = reference_ref
                    selected_reference_cache_paths.add(reference_ref)
                if candidate_cache is not None:
                    candidate_ref = _repo_rel(candidate_cache, repo)
                    row["selected_candidate_mlx_cache_dir"] = candidate_ref
                    selected_candidate_cache_paths.add(candidate_ref)
                if (
                    archive_size_bytes is not None
                    and not isinstance(archive_size_bytes, bool)
                    and archive_size_bytes > 0
                ):
                    row["selected_archive_size_bytes"] = int(archive_size_bytes)
                    selected_archive_size_bytes.add(int(archive_size_bytes))
                else:
                    bundle, bundle_bytes, bundle_blockers, considered = (
                        _select_submission_bundle_for_archive(
                            archive_sha256=str(request.get("archive_sha256") or ""),
                            repo_root=repo,
                            explicit_path=explicit_bundle,
                            search_roots=roots,
                        )
                    )
                    row["candidate_submission_bundle_count"] = considered
                    if bundle is not None:
                        bundle_ref = _repo_rel(bundle, repo)
                        row["selected_submission_bundle_path"] = bundle_ref
                        selected_bundle_paths.add(bundle_ref)
                    if bundle_bytes is not None:
                        row["selected_archive_size_bytes"] = bundle_bytes
                        selected_archive_size_bytes.add(bundle_bytes)
                    else:
                        blockers.extend(
                            f"archive_size:{blocker}" for blocker in bundle_blockers
                        )
            else:
                blockers.append(f"unknown_followup_request_schema:{schema}")
            row["ready_for_readiness_refresh"] = not blockers
            row["blockers"] = blockers
            rows.append(row)

    selected_inputs: dict[str, Any] = {}
    global_blockers: list[str] = []
    if len(selected_bundle_paths) == 1:
        selected_inputs["submission_bundle_path"] = next(iter(selected_bundle_paths))
    elif len(selected_bundle_paths) > 1:
        global_blockers.append("multiple_submission_bundles_selected")
    if len(selected_reference_cache_paths) == 1:
        selected_inputs["reference_mlx_cache_dir"] = next(
            iter(selected_reference_cache_paths)
        )
    elif len(selected_reference_cache_paths) > 1:
        global_blockers.append("multiple_reference_mlx_cache_dirs_selected")
    if len(selected_candidate_cache_paths) == 1:
        selected_inputs["candidate_mlx_cache_dir"] = next(
            iter(selected_candidate_cache_paths)
        )
    elif len(selected_candidate_cache_paths) > 1:
        global_blockers.append("multiple_candidate_mlx_cache_dirs_selected")
    if len(selected_archive_size_bytes) == 1:
        selected_inputs["archive_size_bytes"] = next(iter(selected_archive_size_bytes))
    elif len(selected_archive_size_bytes) > 1:
        global_blockers.append("multiple_archive_size_byte_values_selected")

    ready_count = sum(1 for row in rows if row["ready_for_readiness_refresh"])
    report = {
        "schema": PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA,
        "plan_count": len(plan_paths),
        "request_count": len(rows),
        "bound_request_count": ready_count,
        "blocked_request_count": len(rows) - ready_count,
        "search_roots": [_repo_rel(root, repo) for root in roots],
        "selected_inputs": selected_inputs,
        "global_blockers": global_blockers,
        "requests": rows,
        "allowed_use": "queue_followup_input_binding_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="pair_frame_5d_followup_input_binding_report",
    )
    return report


def _load_followup_readiness_report(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    if payload.get("schema") != PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA:
        raise ExperimentQueueError(
            f"{path}: expected {PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA}, "
            f"got {payload.get('schema')!r}"
        )
    try:
        require_no_truthy_authority_fields(
            payload,
            context=f"pair_frame_5d_followup_readiness_report:{path}",
        )
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    return payload


def _command_flag_value(command: Sequence[str], flag: str) -> str | None:
    values: list[str] = []
    for index, item in enumerate(command):
        if item != flag:
            continue
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ExperimentQueueError(f"follow-up command flag {flag} requires a value")
        values.append(command[index + 1])
    if len(values) > 1:
        raise ExperimentQueueError(f"follow-up command flag {flag} appears multiple times")
    return values[0] if values else None


def _ready_followup_command(row: Mapping[str, Any]) -> list[str]:
    command = row.get("materialized_command")
    if not isinstance(command, list) or not command:
        raise ExperimentQueueError(
            f"ready follow-up row {row.get('work_order_id')!r} lacks materialized_command"
        )
    rendered = [str(item) for item in command]
    request_schema = str(row.get("request_schema") or "")
    if request_schema == PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA:
        if "tools/paired_auth_eval_cli.py" not in rendered:
            raise ExperimentQueueError("exact-axis follow-up must use paired_auth_eval_cli")
        if "--dry-run" not in rendered:
            raise ExperimentQueueError(
                "exact-axis follow-up execution queue only accepts dry-run commands"
            )
    elif request_schema == PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA:
        if "tools/run_mlx_scorer_response_cache.py" not in rendered:
            raise ExperimentQueueError(
                "MLX negative-delta follow-up must use run_mlx_scorer_response_cache"
            )
        if "--allow-gpu-research-signal" not in rendered:
            raise ExperimentQueueError(
                "MLX negative-delta follow-up must remain explicit research signal"
            )
    else:
        raise ExperimentQueueError(f"unsupported ready follow-up schema: {request_schema!r}")
    return rendered


def _followup_resource_kind(row: Mapping[str, Any]) -> str:
    schema = str(row.get("request_schema") or "")
    if schema == PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA:
        return "local_mlx"
    return "local_cpu"


def _followup_experiment_status(row: Mapping[str, Any], default_status: str) -> str:
    if row.get("request_schema") == PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA:
        return "frozen"
    return default_status


def _followup_step_postconditions(command: Sequence[str], repo_root: Path) -> list[dict[str, Any]]:
    output = _command_flag_value(command, "--output")
    if output is None:
        return []
    output_path = _resolve_path(output, repo_root=repo_root)
    postconditions = [
        {"type": "path_exists", "path": _repo_rel(output_path, repo_root)},
        {
            "type": "json_false_authority",
            "path": _repo_rel(output_path, repo_root),
        },
    ]
    if "tools/run_mlx_scorer_response_cache.py" in command:
        postconditions.insert(
            1,
            {
                "type": "json_equals",
                "path": _repo_rel(output_path, repo_root),
                "key": "schema_version",
                "equals": MLX_SCORER_RESPONSE_SCHEMA,
            },
        )
    return postconditions


def build_coverage_followup_execution_queue(
    *,
    repo_root: str | Path,
    readiness_report_path: str | Path,
    queue_id: str = "pair_frame_5d_followup_execution_queue",
    lane_id: str = "lane_pair_frame_5d_coverage_followup_execution_20260527",
    local_cpu_concurrency: int = 2,
    local_mlx_concurrency: int = 1,
    timeout_seconds: int = 0,
    limit: int | None = None,
    status: str = "queued",
) -> dict[str, Any]:
    """Compile ready 5D follow-up requests into executable local queue work."""

    if status not in {"queued", "frozen", "disabled"}:
        raise ExperimentQueueError("status must be queued, frozen, or disabled")
    for label, value in (
        ("local_cpu_concurrency", local_cpu_concurrency),
        ("local_mlx_concurrency", local_mlx_concurrency),
    ):
        if isinstance(value, bool) or value < 1:
            raise ExperimentQueueError(f"{label} must be >= 1")
    if isinstance(timeout_seconds, bool) or timeout_seconds < 0:
        raise ExperimentQueueError("timeout_seconds must be non-negative")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ExperimentQueueError("limit must be >= 1 when provided")

    repo = Path(repo_root)
    report_path = _resolve_path(readiness_report_path, repo_root=repo)
    report = _load_followup_readiness_report(report_path)
    rows = [
        dict(row)
        for row in report.get("requests") or []
        if isinstance(row, Mapping)
    ]
    blocked_rows: list[dict[str, Any]] = []
    ready_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            require_no_truthy_authority_fields(
                row,
                context=(
                    "pair_frame_5d_followup_execution_row:"
                    f"{row.get('work_order_id')}"
                ),
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        if row.get("ready") is True:
            ready_rows.append(row)
        else:
            blocked_rows.append(
                {
                    "work_order_id": row.get("work_order_id"),
                    "request_schema": row.get("request_schema"),
                    "blockers": list(row.get("blockers") or []),
                    "plan_path": row.get("plan_path"),
                }
            )
    selected_rows = ready_rows[:limit] if limit is not None else ready_rows

    experiments: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        command = _ready_followup_command(row)
        resource_kind = _followup_resource_kind(row)
        output = _command_flag_value(command, "--output")
        artifact_paths = []
        if output is not None:
            artifact_paths.append(_repo_rel(_resolve_path(output, repo_root=repo), repo))
        request_schema = str(row.get("request_schema") or "")
        work_order_id = str(row.get("work_order_id") or f"request_{index:04d}")
        experiment_id = f"followup_{index:04d}_{_slug(work_order_id)}"
        experiments.append(
            {
                "id": experiment_id,
                "lane_id": lane_id,
                "priority": index,
                "status": _followup_experiment_status(row, status),
                "tags": [
                    "pair-frame-5d-canvas",
                    "followup-execution",
                    resource_kind,
                    "no-score-authority",
                ],
                "metadata": {
                    "schema": PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA,
                    "source_readiness_report_path": _repo_rel(report_path, repo),
                    "work_order_id": work_order_id,
                    "request_schema": request_schema,
                    "plan_path": row.get("plan_path"),
                    "resource_kind": resource_kind,
                    "execution_scope": (
                        "local_mlx_research_signal"
                        if resource_kind == "local_mlx"
                        else "operator_gated_exact_axis_dry_run_plan"
                    ),
                    "operator_gated": (
                        request_schema == PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA
                    ),
                    "allowed_use": "queue_owned_followup_execution_only",
                    "forbidden_use": "score_claim_or_promotion_or_paid_dispatch",
                    **FALSE_AUTHORITY,
                },
                "steps": [
                    {
                        "id": "execute_ready_followup_request",
                        "kind": "command",
                        "command": command,
                        "resources": {"kind": resource_kind},
                        "timeout_seconds": timeout_seconds,
                        "postconditions": _followup_step_postconditions(command, repo),
                        "telemetry": {
                            "artifact_paths": artifact_paths,
                            "input_artifact_paths": [
                                _repo_rel(report_path, repo),
                                str(row.get("plan_path") or ""),
                            ],
                            "recursive": bool(artifact_paths),
                            "include_postcondition_paths": True,
                        },
                    }
                ],
            }
        )
    if not experiments:
        experiments.append(
            {
                "id": "no_ready_followup_requests",
                "lane_id": lane_id,
                "priority": 999,
                "status": "disabled",
                "tags": [
                    "pair-frame-5d-canvas",
                    "followup-execution",
                    "blocked-input-refusal",
                    "no-score-authority",
                ],
                "metadata": {
                    "schema": PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA,
                    "source_readiness_report_path": _repo_rel(report_path, repo),
                    "blocked_request_count": len(blocked_rows),
                    "allowed_use": "queue_owned_followup_execution_only",
                    "forbidden_use": "score_claim_or_promotion_or_paid_dispatch",
                    **FALSE_AUTHORITY,
                },
                "steps": [
                    {
                        "id": "no_ready_followups",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "-c",
                            "print('no ready 5D coverage follow-up requests')",
                        ],
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 30,
                        "postconditions": [],
                        "telemetry": {
                            "artifact_paths": [],
                            "input_artifact_paths": [_repo_rel(report_path, repo)],
                            "recursive": False,
                        },
                    }
                ],
            }
        )

    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": local_cpu_concurrency,
                    "local_mlx": local_mlx_concurrency,
                },
            },
            "metadata": {
                "schema": PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA,
                "source_readiness_report_path": _repo_rel(report_path, repo),
                "request_count": len(rows),
                "ready_request_count": len(ready_rows),
                "blocked_request_count": len(blocked_rows),
                "selected_request_count": len(selected_rows),
                "blocked_requests": blocked_rows,
                "allowed_use": "queue_owned_followup_execution_only",
                "forbidden_use": "score_claim_or_promotion_or_paid_dispatch",
                **FALSE_AUTHORITY,
            },
            "experiments": experiments,
        }
    )
    require_no_truthy_authority_fields(
        queue.get("metadata") if isinstance(queue.get("metadata"), Mapping) else {},
        context="pair_frame_5d_followup_execution_queue:metadata",
    )
    return queue


def build_coverage_acquisition_plan(
    *,
    coverage_audit: Mapping[str, Any],
    work_order_id: str,
    coverage_audit_path: str | Path,
    repo_root: str | Path,
    canvas_path: str | Path | None = None,
    output_root: str | Path | None = None,
    mode: str = "mlx-local",
) -> dict[str, Any]:
    """Build a false-authority acquisition plan for one coverage work order."""

    if mode not in {"mlx-local", "local-cpu"}:
        raise ExperimentQueueError("mode must be mlx-local or local-cpu")
    repo = Path(repo_root)
    audit_path = _resolve_path(coverage_audit_path, repo_root=repo)
    output_base = _resolve_path(
        output_root or audit_path.parent / "pair_frame_5d_coverage_acquisition",
        repo_root=repo,
    )
    work_order = _find_work_order(coverage_audit, work_order_id)
    archive_sha256 = str(coverage_audit.get("archive_sha256") or "")
    plan_class, executable_now, resource_kind, blockers = _plan_class(work_order_id)
    refreshed_canvas = output_base / "refreshed_5d_canvas.json"
    generated_commands: list[list[str]] = []
    if executable_now:
        generated_commands.append(
            [
                ".venv/bin/python",
                "tools/populate_5d_canvas_cli.py",
                "--archive-sha256",
                archive_sha256,
                "--output-path",
                _repo_rel(refreshed_canvas, repo),
                "--json",
            ]
        )
    followup_requests = _followup_lane_requests(
        work_order_id=work_order_id,
        archive_sha256=archive_sha256,
        output_base=output_base,
        repo_root=repo,
    )
    command_templates = [
        list(request.get("command_template") or request.get("canonical_plan_entrypoint") or [])
        for request in followup_requests
    ]
    target = work_order.get("target") if isinstance(work_order.get("target"), Mapping) else {}
    payload: dict[str, Any] = {
        "schema": PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA,
        "mode": mode,
        "work_order_id": work_order_id,
        "work_order": work_order,
        "archive_sha256": archive_sha256,
        "coverage_audit_path": _repo_rel(audit_path, repo),
        "canvas_path": (
            _repo_rel(_resolve_path(canvas_path, repo_root=repo), repo)
            if canvas_path is not None
            else None
        ),
        "output_root": _repo_rel(output_base, repo),
        "plan_class": plan_class,
        "executable_now": executable_now,
        "preferred_resource_kind": resource_kind,
        "blocking_conditions": blockers,
        "generated_commands": generated_commands,
        "command_templates_requiring_inputs": command_templates,
        "followup_lane_requests": followup_requests,
        "suggested_next_tools": list(work_order.get("suggested_next_tools") or []),
        "target": dict(target),
        "queue_followup": {
            "schema": "pair_frame_5d_canvas_coverage_acquisition_followup.v1",
            "local_refresh_canvas_path": _repo_rel(refreshed_canvas, repo),
            "refire_extended_operators_after_refresh": True,
            "requires_exact_auth_before_score_claim": True,
            **FALSE_AUTHORITY,
        },
        "allowed_use": "local_encoder_side_coverage_acquisition_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context=f"pair_frame_5d_coverage_acquisition_plan:{work_order_id}",
    )
    return payload


def _resource_for_work_order(work_order_id: str) -> str:
    return "local_mlx" if work_order_id == "acquire_negative_delta_cells_before_operator_fanout" else "local_cpu"


def _false_authority_postcondition(path: str) -> dict[str, Any]:
    return {"type": "json_false_authority", "path": path}


def build_pair_frame_5d_coverage_acquisition_queue(
    *,
    repo_root: str | Path,
    coverage_audit_path: str | Path,
    canvas_path: str | Path,
    output_root: str | Path,
    queue_id: str = "pair_frame_5d_coverage_acquisition_queue",
    top_n: int = 4,
    local_cpu_concurrency: int = 2,
    local_mlx_concurrency: int = 1,
    local_io_concurrency: int = 1,
    status: str = "queued",
) -> dict[str, Any]:
    """Build queue-owned acquisition work from a 5D coverage audit."""

    if top_n < 1:
        raise ExperimentQueueError("top_n must be >= 1")
    for label, value in {
        "local_cpu_concurrency": local_cpu_concurrency,
        "local_mlx_concurrency": local_mlx_concurrency,
        "local_io_concurrency": local_io_concurrency,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ExperimentQueueError(f"{label} must be >= 1")
    if status not in {"queued", "frozen", "disabled"}:
        raise ExperimentQueueError("status must be queued, frozen, or disabled")

    repo = Path(repo_root)
    audit_path = _resolve_path(coverage_audit_path, repo_root=repo)
    canvas = _resolve_path(canvas_path, repo_root=repo)
    out_root = _resolve_path(output_root, repo_root=repo)
    audit = _load_coverage_audit(audit_path)
    work_orders = _work_orders(audit)
    if not work_orders:
        raise ExperimentQueueError("coverage audit has no acquisition work orders")
    audit_ref = _repo_rel(audit_path, repo)
    canvas_ref = _repo_rel(canvas, repo)
    out_ref = _repo_rel(out_root, repo)
    experiments: list[dict[str, Any]] = []
    plan_step_refs: list[str] = []
    executable_work_order_ids: list[str] = []
    blocked_work_order_ids: list[str] = []
    plan_classes_by_work_order: dict[str, str] = {}
    for order in work_orders:
        work_order_id = str(order["id"])
        slug = _slug(work_order_id)
        plan_class, executable_now, preferred_resource_kind, blockers = _plan_class(
            work_order_id
        )
        plan_classes_by_work_order[work_order_id] = plan_class
        if executable_now:
            executable_work_order_ids.append(work_order_id)
        else:
            blocked_work_order_ids.append(work_order_id)
        plan_path = out_root / "plans" / f"{slug}_acquisition_plan.json"
        plan_ref = _repo_rel(plan_path, repo)
        experiment_id = f"coverage_acquisition_{slug}"
        step_id = "emit_acquisition_plan"
        plan_step_refs.append(f"{experiment_id}.{step_id}")
        experiments.append(
            {
                "id": experiment_id,
                "lane_id": "lane_pair_frame_5d_coverage_acquisition_queue_20260527",
                "priority": int(order.get("priority") or 100),
                "status": status,
                "tags": [
                    "pair-frame-5d-canvas",
                    "coverage-acquisition",
                    "encoder-side-planning",
                    "no-score-authority",
                    work_order_id,
                ],
                "metadata": {
                    "schema": PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA,
                    "coverage_audit_path": audit_ref,
                    "canvas_path": canvas_ref,
                    "work_order_id": work_order_id,
                    "work_order": order,
                    "plan_class": plan_class,
                    "executable_now": executable_now,
                    "preferred_resource_kind": preferred_resource_kind,
                    "blocking_conditions": blockers,
                    "plan_path": plan_ref,
                    "allowed_use": "local_encoder_side_coverage_acquisition_planning_only",
                    **FALSE_AUTHORITY,
                },
                "steps": [
                    {
                        "id": step_id,
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/emit_5d_canvas_coverage_acquisition_plan.py",
                            "--coverage-audit",
                            audit_ref,
                            "--work-order-id",
                            work_order_id,
                            "--canvas-path",
                            canvas_ref,
                            "--output-root",
                            out_ref,
                            "--output",
                            plan_ref,
                            "--mode",
                            "mlx-local",
                        ],
                        "resources": {"kind": _resource_for_work_order(work_order_id)},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": plan_ref,
                                "key": "schema",
                                "equals": PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA,
                            },
                            {
                                "type": "json_equals",
                                "path": plan_ref,
                                "key": "work_order_id",
                                "equals": work_order_id,
                            },
                            _false_authority_postcondition(plan_ref),
                        ],
                        "telemetry": {
                            "artifact_paths": [plan_ref],
                            "input_artifact_paths": [audit_ref, canvas_ref],
                            "include_postcondition_paths": True,
                        },
                    }
                ],
            }
        )

    followup_readiness_report = out_root / "followup_readiness_report.json"
    followup_input_binding_report = out_root / "followup_input_binding_report.json"
    followup_execution_queue = out_root / "followup_execution_queue.json"
    followup_execution_worker_result = (
        out_root / "followup_execution_worker_result.json"
    )
    followup_readiness_report_ref = _repo_rel(followup_readiness_report, repo)
    followup_input_binding_report_ref = _repo_rel(followup_input_binding_report, repo)
    followup_execution_queue_ref = _repo_rel(followup_execution_queue, repo)
    followup_execution_worker_result_ref = _repo_rel(
        followup_execution_worker_result,
        repo,
    )
    experiments.append(
        {
            "id": "audit_blocked_followup_requests",
            "lane_id": "lane_pair_frame_5d_coverage_acquisition_queue_20260527",
            "priority": 900,
            "status": status,
            "tags": [
                "pair-frame-5d-canvas",
                "coverage-followup-readiness",
                "coverage-followup-execution",
                "blocked-input-refusal",
                "no-score-authority",
            ],
            "metadata": {
                "schema": PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA,
                "coverage_audit_path": audit_ref,
                "canvas_path": canvas_ref,
                "followup_input_binding_report_path": followup_input_binding_report_ref,
                "followup_readiness_report_path": followup_readiness_report_ref,
                "followup_execution_queue_path": followup_execution_queue_ref,
                "followup_execution_worker_result_path": (
                    followup_execution_worker_result_ref
                ),
                "blocked_work_order_ids": blocked_work_order_ids,
                "allowed_use": (
                    "local_encoder_side_coverage_followup_readiness_and_"
                    "bounded_mlx_execution_only"
                ),
                **FALSE_AUTHORITY,
            },
            "steps": [
                {
                    "id": "bind_followup_inputs_and_emit_readiness_report",
                    "kind": "command",
                    "requires": plan_step_refs,
                    "command": [
                        ".venv/bin/python",
                        "tools/bind_5d_coverage_followup_inputs.py",
                        "--plans-dir",
                        f"{out_ref}/plans",
                        "--search-root",
                        out_ref,
                        "--output",
                        followup_input_binding_report_ref,
                        "--refreshed-readiness-output",
                        followup_readiness_report_ref,
                        "--overwrite",
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": followup_input_binding_report_ref,
                            "key": "schema",
                            "equals": (
                                PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA
                            ),
                        },
                        {
                            "type": "json_equals",
                            "path": followup_readiness_report_ref,
                            "key": "schema",
                            "equals": PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA,
                        },
                        _false_authority_postcondition(
                            followup_input_binding_report_ref
                        ),
                        _false_authority_postcondition(followup_readiness_report_ref),
                    ],
                    "telemetry": {
                        "artifact_paths": [
                            followup_input_binding_report_ref,
                            followup_readiness_report_ref,
                        ],
                        "input_artifact_paths": [f"{out_ref}/plans"],
                        "recursive": False,
                        "include_postcondition_paths": True,
                    },
                },
                {
                    "id": "emit_followup_execution_queue",
                    "kind": "command",
                    "requires": ["bind_followup_inputs_and_emit_readiness_report"],
                    "command": [
                        ".venv/bin/python",
                        "tools/build_5d_coverage_followup_execution_queue.py",
                        "--readiness-report",
                        followup_readiness_report_ref,
                        "--queue-out",
                        followup_execution_queue_ref,
                        "--queue-id",
                        f"{queue_id}_followup_execution",
                        "--overwrite",
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": followup_execution_queue_ref,
                            "key": "schema",
                            "equals": QUEUE_SCHEMA,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [followup_execution_queue_ref],
                        "input_artifact_paths": [followup_readiness_report_ref],
                        "recursive": False,
                        "include_postcondition_paths": True,
                    },
                },
                {
                    "id": "validate_followup_execution_queue",
                    "kind": "command",
                    "requires": ["emit_followup_execution_queue"],
                    "command": [
                        ".venv/bin/python",
                        "tools/experiment_queue.py",
                        "--queue",
                        followup_execution_queue_ref,
                        "validate",
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 120,
                    "postconditions": [],
                    "telemetry": {
                        "input_artifact_paths": [followup_execution_queue_ref],
                        "recursive": False,
                    },
                },
                {
                    "id": "run_followup_execution_queue_bounded_local",
                    "kind": "command",
                    "requires": ["validate_followup_execution_queue"],
                    "command": [
                        ".venv/bin/python",
                        "tools/experiment_queue.py",
                        "--queue",
                        followup_execution_queue_ref,
                        "run-worker",
                        "--execute",
                        "--max-steps",
                        "4",
                        "--max-experiments",
                        "2",
                        "--max-parallel",
                        "1",
                        "--output",
                        followup_execution_worker_result_ref,
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 900,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": followup_execution_worker_result_ref,
                            "key": "schema",
                            "equals": WORKER_RESULT_SCHEMA,
                        },
                        {
                            "type": "json_equals",
                            "path": followup_execution_worker_result_ref,
                            "key": "failure_count",
                            "equals": 0,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [followup_execution_worker_result_ref],
                        "input_artifact_paths": [followup_execution_queue_ref],
                        "recursive": False,
                        "include_postcondition_paths": True,
                    },
                },
            ],
        }
    )

    archive_sha256 = str(audit.get("archive_sha256") or "")
    refreshed_canvas = out_root / "refreshed_5d_canvas.json"
    refreshed_audit = out_root / "refreshed_5d_canvas_coverage_audit.json"
    refired_output_root = out_root / "refired_5d_extended_operator_outputs"
    refired_queue = out_root / "refired_5d_extended_operator_queue.json"
    refired_worker_result = out_root / "refired_5d_extended_operator_worker_result.json"
    refreshed_canvas_ref = _repo_rel(refreshed_canvas, repo)
    refreshed_audit_ref = _repo_rel(refreshed_audit, repo)
    refired_queue_ref = _repo_rel(refired_queue, repo)
    refired_output_root_ref = _repo_rel(refired_output_root, repo)
    refired_worker_result_ref = _repo_rel(refired_worker_result, repo)
    refresh_experiment_id = "refresh_reaudit_and_refire_extended_operators"
    experiments.append(
        {
            "id": refresh_experiment_id,
            "lane_id": "lane_pair_frame_5d_coverage_acquisition_queue_20260527",
            "priority": 1000,
            "status": status,
            "tags": [
                "pair-frame-5d-canvas",
                "coverage-refresh",
                "extended-operator-refire",
                "no-score-authority",
            ],
            "metadata": {
                "schema": PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA,
                "coverage_audit_path": audit_ref,
                "canvas_path": canvas_ref,
                "refreshed_canvas_path": refreshed_canvas_ref,
                "refreshed_coverage_audit_path": refreshed_audit_ref,
                "refired_extended_operator_queue_path": refired_queue_ref,
                "refired_worker_result_path": refired_worker_result_ref,
                "work_order_count": len(work_orders),
                "local_refresh_consumes_work_order_ids": executable_work_order_ids,
                "external_blocking_work_order_ids": blocked_work_order_ids,
                "refresh_semantics": (
                    "refreshes_from_currently_available_local_anchors_after_plan_"
                    "emission;_non_executable_work_orders_remain_external_blockers"
                ),
                "allowed_use": "local_encoder_side_coverage_acquisition_planning_only",
                **FALSE_AUTHORITY,
            },
            "steps": [
                {
                    "id": "populate_refreshed_canvas_from_master_gradient",
                    "kind": "command",
                    "requires": plan_step_refs,
                    "command": [
                        ".venv/bin/python",
                        "tools/populate_5d_canvas_cli.py",
                        "--archive-sha256",
                        archive_sha256,
                        "--output-path",
                        refreshed_canvas_ref,
                        "--json",
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 300,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": refreshed_canvas_ref,
                            "key": "schema",
                            "equals": POPULATED_CANVAS_SCHEMA,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [refreshed_canvas_ref],
                        "input_artifact_paths": [audit_ref, canvas_ref],
                        "include_postcondition_paths": True,
                    },
                },
                {
                    "id": "reaudit_refreshed_canvas",
                    "kind": "command",
                    "requires": ["populate_refreshed_canvas_from_master_gradient"],
                    "command": [
                        ".venv/bin/python",
                        "tools/audit_5d_canvas_coverage.py",
                        "--canvas-path",
                        refreshed_canvas_ref,
                        "--output",
                        refreshed_audit_ref,
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": refreshed_audit_ref,
                            "key": "schema",
                            "equals": COVERAGE_AUDIT_SCHEMA,
                        },
                        _false_authority_postcondition(refreshed_audit_ref),
                    ],
                    "telemetry": {
                        "artifact_paths": [refreshed_audit_ref],
                        "input_artifact_paths": [refreshed_canvas_ref],
                        "include_postcondition_paths": True,
                    },
                },
                {
                    "id": "build_refired_extended_operator_queue",
                    "kind": "command",
                    "requires": ["reaudit_refreshed_canvas"],
                    "command": [
                        ".venv/bin/python",
                        "tools/build_5d_extended_operator_queue.py",
                        "--canvas-path",
                        refreshed_canvas_ref,
                        "--output-root",
                        refired_output_root_ref,
                        "--queue-out",
                        refired_queue_ref,
                        "--queue-id",
                        f"{queue_id}_refired_extended_operators",
                        "--top-n",
                        str(top_n),
                        "--overwrite",
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": refired_queue_ref,
                            "key": "schema",
                            "equals": EXTENDED_OPERATOR_QUEUE_SCHEMA,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [refired_queue_ref],
                        "input_artifact_paths": [refreshed_canvas_ref, refreshed_audit_ref],
                        "include_postcondition_paths": True,
                    },
                },
                {
                    "id": "run_refired_extended_operator_queue",
                    "kind": "command",
                    "requires": ["build_refired_extended_operator_queue"],
                    "command": [
                        ".venv/bin/python",
                        "tools/experiment_queue.py",
                        "--queue",
                        refired_queue_ref,
                        "run-worker",
                        "--execute",
                        "--max-steps",
                        "8",
                        "--max-experiments",
                        "8",
                        "--max-parallel",
                        "1",
                        "--output",
                        refired_worker_result_ref,
                    ],
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 900,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": refired_worker_result_ref,
                            "key": "schema",
                            "equals": WORKER_RESULT_SCHEMA,
                        },
                        {
                            "type": "json_equals",
                            "path": refired_worker_result_ref,
                            "key": "failure_count",
                            "equals": 0,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [refired_worker_result_ref, refired_output_root_ref],
                        "input_artifact_paths": [refired_queue_ref],
                        "recursive": True,
                        "include_postcondition_paths": True,
                    },
                },
            ],
        }
    )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": local_cpu_concurrency,
                    "local_mlx": local_mlx_concurrency,
                    "local_io_heavy": local_io_concurrency,
                },
            },
            "metadata": {
                "schema": PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA,
                "coverage_audit_path": audit_ref,
                "canvas_path": canvas_ref,
                "output_root": out_ref,
                "work_order_count": len(work_orders),
                "executable_work_order_ids": executable_work_order_ids,
                "blocked_work_order_ids": blocked_work_order_ids,
                "plan_classes_by_work_order": plan_classes_by_work_order,
                "followup_readiness_report_path": followup_readiness_report_ref,
                "followup_execution_queue_path": followup_execution_queue_ref,
                "allowed_use": "local_encoder_side_coverage_acquisition_planning_only",
                **FALSE_AUTHORITY,
            },
            "experiments": experiments,
        }
    )
    require_no_truthy_authority_fields(
        queue.get("metadata") if isinstance(queue.get("metadata"), Mapping) else {},
        context="pair_frame_5d_coverage_acquisition_queue:metadata",
    )
    return queue


__all__ = [
    "FALSE_AUTHORITY",
    "PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA",
    "PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA",
    "PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA",
    "PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA",
    "PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA",
    "PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA",
    "PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA",
    "build_coverage_acquisition_plan",
    "build_coverage_followup_execution_queue",
    "build_coverage_followup_input_binding_report",
    "build_coverage_followup_readiness_report",
    "build_pair_frame_5d_coverage_acquisition_queue",
]
