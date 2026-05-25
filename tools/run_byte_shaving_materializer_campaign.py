#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and run a byte-shaving materializer campaign queue.

This is the one-command control surface for the local materializer loop:
compile the campaign, emit executable materializer rows, append per-row
harvest/exact-readiness/paused-dispatch-plan follow-ups, initialize the
canonical experiment queue state, run a bounded worker, and write live
observation/performance artifacts. It never performs paid dispatch.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.byte_shaving_materializer_registry import (  # noqa: E402
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    INVERSE_SCORER_CELL_TARGET_KIND,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
)
from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
    DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
    QUEUE_SCHEMA,
    load_queue_definition,
    normalize_queue_definition,
)
from comma_lab.scheduler.queue_feedback_replan_policy import (  # noqa: E402
    QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
    QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_FLAGS,
    build_queue_feedback_candidate_actuation_planning_queue,
    build_queue_feedback_candidate_widening_queue,
    build_queue_feedback_replan_continuation_queue,
    build_queue_feedback_replan_policy,
    build_queue_observation_recovery_plan,
    build_queue_observation_recovery_queue,
    validate_feedback_followup_queue,
    validate_queue_observation_recovery_queue,
)
from comma_lab.scheduler.staircase_dag import (  # noqa: E402
    STAIRCASE_DEPENDENT_QUEUE_REF_SCHEMA,
    build_staircase_dag_from_experiment_queue,
    experiment_queue_status_map,
    parse_resource_pool_spec,
    plan_staircase_dispatch,
)
from comma_lab.scheduler.storage_preflight import (  # noqa: E402
    validate_scheduler_storage_preflight_config,
)
from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY  # noqa: E402
from tac.optimization.materializer_feedback import (  # noqa: E402
    materializer_archive_delta,
    selected_materializer_delta,
)
from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    truthy_authority_field_violations,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402
from tac.score_composition import (  # noqa: E402
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
)

RUN_SCHEMA = "byte_shaving_materializer_campaign_run.v1"
RUN_CONFIG_SCHEMA = "byte_shaving_materializer_campaign_run_config.v1"
RESPONSE_UPDATE_PLACEHOLDER_SCHEMA = "byte_shaving_campaign_response_update_placeholder.v1"
QUEUE_PERFORMANCE_SUMMARY_SCHEMA = "experiment_queue_performance_summary.v1"
UNAVAILABLE_QUEUE_PERFORMANCE_SUMMARY_SCHEMA = "experiment_queue_performance_summary_unavailable.v1"
QUEUE_FEEDBACK_REPLAN_REQUEST_SCHEMA = "byte_shaving_materializer_campaign_feedback_replan_request.v1"
RESULT_REVIEW_PACKET_SCHEMA = "tac_result_review_packet_v1"
QUEUE_FEEDBACK_REPLAN_EXPERIMENT_METADATA_SCHEMA = (
    "byte_shaving_materializer_campaign_feedback_replan_experiment_metadata.v1"
)
QUEUE_FEEDBACK_REPLAN_FOLLOWUP_EXECUTION_SCHEMA = (
    "byte_shaving_materializer_campaign_feedback_replan_followup_execution.v1"
)
QUEUE_OBSERVATION_RECOVERY_EXECUTION_SCHEMA = (
    "byte_shaving_materializer_campaign_queue_observation_recovery_execution.v1"
)
POST_RECOVERY_FEEDBACK_REPLAN_SCHEMA = (
    "byte_shaving_materializer_campaign_post_recovery_feedback_replan.v1"
)
FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA = (
    "family_agnostic_materializer_empirical_observation.v1"
)
FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA = (
    "family_agnostic_materializer_empirical_sweep.v1"
)
MAX_AUTO_DISCOVERED_FEEDBACK_OBSERVATIONS = 128
MAX_AUTO_DISCOVERED_FEEDBACK_OBSERVATION_BYTES = 8 * 1024 * 1024
MAX_AUTO_DISCOVERED_MATERIALIZER_CHAIN_MANIFESTS = 128
MAX_AUTO_DISCOVERED_MATERIALIZER_CHAIN_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_AUTO_DISCOVERED_RECEIVER_FEEDBACK_MANIFESTS = 128
MAX_AUTO_DISCOVERED_RECEIVER_FEEDBACK_MANIFEST_BYTES = 8 * 1024 * 1024
DEFAULT_LOCAL_QUEUE_POLL_INTERVAL_SECONDS = 0.005
IN_PROCESS_TOOL_MODULE_BY_RELATIVE_SCRIPT = {
    "tools/build_byte_shaving_campaign_queue.py": "tools.build_byte_shaving_campaign_queue",
    "tools/build_inverse_steganalysis_action_functional.py": "tools.build_inverse_steganalysis_action_functional",
    "tools/build_mlx_acquisition_batch.py": "tools.build_mlx_acquisition_batch",
    "tools/experiment_queue.py": "tools.experiment_queue",
    "tools/plan_byte_shaving_campaign.py": "tools.plan_byte_shaving_campaign",
}
RECEIVER_NEGATIVE_MATERIALIZER_AXIS = "[local-materializer-receiver advisory]"
CONTEST_RATE_SCORE_PER_BYTE = CANONICAL_RATE_MULTIPLIER / float(
    CANONICAL_RATE_DENOM_BYTES
)
QUEUE_FEEDBACK_REPLAN_REQUIRED_FALSE_FIELDS = tuple(
    dict.fromkeys(
        (
            *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
            "ready_for_exact_eval_dispatch",
        )
    )
)
_RUN_CONFIG_METADATA_KEYS = frozenset({"schema", "notes", "description", "owner"})
QUEUE_RUNTIME_IDENTITY_FILE_PATHS = (
    "tools/run_byte_shaving_materializer_campaign.py",
    "tools/experiment_queue.py",
    "tools/run_family_agnostic_materializer.py",
    "tools/run_inverse_scorer_cell_candidate_chain.py",
    "src/comma_lab/scheduler/experiment_queue.py",
    "src/comma_lab/scheduler/byte_shaving_campaign_queue.py",
    "src/comma_lab/scheduler/final_byte_operation_contexts.py",
    "src/tac/optimization/family_agnostic_materializers.py",
    "src/tac/optimization/inverse_scorer_cell_chain.py",
)


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _path_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _default_materializer_context_output_root(
    args: argparse.Namespace,
    *,
    run_dir: Path,
) -> Path:
    if args.materializer_context_default_output_root is not None:
        return _resolve(args.materializer_context_default_output_root)
    if args.include_storage_preflight and args.storage_expected_workload_root:
        expected_workload_root = _resolve(args.storage_expected_workload_root)
        return (
            run_dir / "materializer_outputs"
            if _path_under_root(run_dir, expected_workload_root)
            else expected_workload_root / "materializer_outputs"
        )
    return run_dir / "materializer_outputs"


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _optional_sha256_file(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() and path.is_file() else None


def _load_run_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: run config must be JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: run config must be a JSON object")
    schema = payload.get("schema")
    if schema is not None and schema != RUN_CONFIG_SCHEMA:
        raise SystemExit(f"{path}: schema must be {RUN_CONFIG_SCHEMA}")
    args = payload.get("args", payload)
    if not isinstance(args, dict):
        raise SystemExit(f"{path}: run config args must be a JSON object")
    return dict(args)


def _normalize_config_defaults(
    parser: argparse.ArgumentParser,
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    actions = {action.dest: action for action in parser._actions if action.dest and action.dest != "help"}
    defaults: dict[str, Any] = {}
    for raw_key, raw_value in config.items():
        key = str(raw_key).strip().replace("-", "_")
        if key in _RUN_CONFIG_METADATA_KEYS:
            continue
        action = actions.get(key)
        if action is None:
            raise SystemExit(f"{config_path}: unknown run config key {raw_key!r}")
        if isinstance(action, argparse._AppendAction):
            if raw_value is None:
                value: list[Any] = []
            elif isinstance(raw_value, list):
                value = raw_value
            else:
                value = [raw_value]
            defaults[key] = [str(item) for item in value]
        elif isinstance(action, argparse._StoreTrueAction):
            if not isinstance(raw_value, bool):
                raise SystemExit(f"{config_path}: {raw_key} must be boolean")
            defaults[key] = raw_value
        elif isinstance(action, argparse._StoreAction):
            defaults[key] = raw_value
        else:
            raise SystemExit(f"{config_path}: unsupported config action for {raw_key}")
    return defaults


def _run(command: list[str], *, check: bool = True) -> CommandResult:
    started = time.monotonic()
    result = _run_in_process_tool_command(
        command,
        started_monotonic=started,
    )
    if result is None:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        result = CommandResult(
            command=command,
            returncode=int(proc.returncode),
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_seconds=time.monotonic() - started,
        )
    if check and result.returncode != 0:
        raise SystemExit(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result


def _run_in_process_tool_command(
    command: Sequence[str],
    *,
    started_monotonic: float | None = None,
) -> CommandResult | None:
    command_items = [str(item) for item in command]
    if len(command_items) < 2:
        return None
    interpreter = Path(command_items[0]).resolve(strict=False)
    if interpreter != Path(sys.executable).resolve(strict=False):
        return None
    script = command_items[1]
    script_path = Path(script)
    resolved_script = (
        script_path
        if script_path.is_absolute()
        else REPO_ROOT / script_path
    ).resolve(strict=False)
    module_name = None
    for relative_script, candidate_module in IN_PROCESS_TOOL_MODULE_BY_RELATIVE_SCRIPT.items():
        if resolved_script == (REPO_ROOT / relative_script).resolve(strict=False):
            module_name = candidate_module
            break
    if module_name is None:
        return None

    stdout = io.StringIO()
    stderr = io.StringIO()
    started = time.monotonic() if started_monotonic is None else started_monotonic
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module = importlib.import_module(module_name)
            main = getattr(module, "main", None)
            if not callable(main):
                return None
            returncode = int(main(command_items[2:]))
    except SystemExit as exc:
        code = exc.code
        returncode = int(code) if isinstance(code, int) else 2
        if not isinstance(code, int) and code is not None:
            stderr.write(str(code))
            stderr.write("\n")
    except Exception:
        traceback.print_exc(file=stderr)
        returncode = 1
    return CommandResult(
        command=command_items,
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        elapsed_seconds=time.monotonic() - started,
    )


def _run_in_process_experiment_queue_command(
    command: Sequence[str],
    *,
    started_monotonic: float | None = None,
) -> CommandResult | None:
    """Compatibility wrapper for older tests and forensic notebooks."""

    return _run_in_process_tool_command(
        command,
        started_monotonic=started_monotonic,
    )


def _json_from_stdout(result: CommandResult) -> dict[str, Any] | None:
    text = result.stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _require_json_stdout(
    result: CommandResult,
    *,
    label: str,
    allow_nonzero: bool = False,
) -> dict[str, Any]:
    if result.returncode != 0 and not allow_nonzero:
        raise SystemExit(f"{label} failed ({result.returncode}): {result.stderr or result.stdout}")
    payload = _json_from_stdout(result)
    if payload is None:
        raise SystemExit(f"{label} did not emit a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        write_json_artifact(path, payload)
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc


def _false_authority_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(row),
        **FALSE_AUTHORITY,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _as_sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _display_path_list(paths: Sequence[str | Path]) -> list[str]:
    return [_display_path(_resolve(path)) for path in paths]


def _feedback_observation_paths(args: argparse.Namespace, *, run_dir: Path) -> list[str]:
    explicit = _display_path_list(args.observation)
    discovered = [
        _display_path(path)
        for path in _discover_feedback_observation_paths(run_dir)
        if _display_path(path) not in explicit
    ]
    return list(dict.fromkeys([*explicit, *discovered]))


def _materializer_chain_manifest_paths(
    args: argparse.Namespace,
    *,
    run_dir: Path,
) -> list[str]:
    explicit = _display_path_list(args.materializer_chain_manifest)
    discovered = [
        _display_path(path)
        for path in _discover_materializer_chain_manifest_paths(run_dir)
        if _display_path(path) not in explicit
    ]
    return list(dict.fromkeys([*explicit, *discovered]))


def _discover_feedback_observation_paths(run_dir: Path) -> list[Path]:
    root = run_dir.resolve(strict=False)
    paths: list[Path] = []
    if not root.exists():
        return paths
    for path in sorted(root.rglob("*")):
        if len(paths) >= MAX_AUTO_DISCOVERED_FEEDBACK_OBSERVATIONS:
            break
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if not _path_under_root(path, root):
            continue
        try:
            if path.stat().st_size > MAX_AUTO_DISCOVERED_FEEDBACK_OBSERVATION_BYTES:
                continue
        except OSError:
            continue
        if _path_has_family_agnostic_materializer_observation(path):
            paths.append(path)
    return paths


def _discover_materializer_chain_manifest_paths(run_dir: Path) -> list[Path]:
    root = run_dir.resolve(strict=False)
    paths: list[Path] = []
    if not root.exists():
        return paths
    for path in sorted(root.rglob("*.json")):
        if len(paths) >= MAX_AUTO_DISCOVERED_MATERIALIZER_CHAIN_MANIFESTS:
            break
        if not path.is_file() or not _path_under_root(path, root):
            continue
        try:
            if path.stat().st_size > MAX_AUTO_DISCOVERED_MATERIALIZER_CHAIN_MANIFEST_BYTES:
                continue
        except OSError:
            continue
        if _path_has_materializer_chain_archive_delta(path):
            paths.append(path)
    return paths


def _path_has_materializer_chain_archive_delta(path: Path) -> bool:
    payload = _load_json_object_if_present(path)
    if payload is None:
        return False
    if not isinstance(payload.get("serialized_archive_delta"), Mapping):
        selected_key, _selected = selected_materializer_delta(payload)
        if not selected_key:
            return False
    delta = _materializer_manifest_archive_delta(payload)
    if delta is None:
        return False
    return bool(
        str(delta.get("status") or "").strip()
        and delta.get("realized_saved_bytes") is not None
    )


def _path_has_family_agnostic_materializer_observation(path: Path) -> bool:
    if path.suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8") as handle:
                for index, raw_line in enumerate(handle, start=1):
                    if index > 256:
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        return False
                    if (
                        isinstance(payload, Mapping)
                        and payload.get("schema")
                        == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA
                    ):
                        return True
        except OSError:
            return False
        return False
    payload = _load_json_object_if_present(path)
    if payload is None:
        return False
    if payload.get("schema") == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA:
        return True
    if payload.get("schema") == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA:
        return True
    observations = payload.get("observations")
    if isinstance(observations, Sequence) and not isinstance(observations, (bytes, bytearray, str)):
        return any(
            isinstance(row, Mapping)
            and row.get("schema") == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA
            for row in observations
        )
    return False


def _load_json_object_if_present(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _slug(value: Any, *, default: str = "item") -> str:
    text = str(value or "").strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or default


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _materializer_manifest_archive_delta(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    explicit_delta = payload.get("serialized_archive_delta")
    if isinstance(explicit_delta, Mapping):
        saved_bytes = _optional_int(explicit_delta.get("realized_saved_bytes"))
        if saved_bytes is not None:
            return {
                "archive_delta_source_kind": "serialized_archive_delta",
                "status": explicit_delta.get("status"),
                "realized_saved_bytes": saved_bytes,
                "source_archive_bytes": _optional_int(
                    explicit_delta.get("source_archive_bytes")
                ),
                "candidate_archive_bytes": _optional_int(
                    explicit_delta.get("candidate_archive_bytes")
                ),
                "savings_realized": explicit_delta.get("savings_realized") is True,
            }

    archive_delta = materializer_archive_delta(payload)
    if archive_delta is not None:
        return {
            "archive_delta_source_kind": archive_delta.get(
                "selected_materialization_key"
            ),
            "status": archive_delta.get("status"),
            "realized_saved_bytes": archive_delta.get("realized_saved_bytes"),
            "source_archive_bytes": archive_delta.get("source_archive_bytes"),
            "candidate_archive_bytes": archive_delta.get("candidate_archive_bytes"),
            "savings_realized": archive_delta.get("savings_realized") is True,
        }
    return None


def _blocker_counts(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value).strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _step_lookup_by_artifact_path(
    observation: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for section in (
        "failed_steps",
        "blocked_steps",
        "succeeded_artifact_failure_steps",
        "running_steps",
        "queued_steps",
    ):
        for step in _as_sequence(observation.get(section)):
            if not isinstance(step, Mapping):
                continue
            paths: list[str] = []
            for artifact in _as_sequence(step.get("expected_artifacts")):
                if isinstance(artifact, Mapping):
                    path = str(artifact.get("path") or "").strip()
                    if path:
                        paths.append(path)
            paths.extend(
                str(path).strip()
                for path in _as_sequence(step.get("expected_artifact_paths"))
                if str(path).strip()
            )
            for path in paths:
                resolved = _resolve(path).resolve(strict=False)
                for key in {path, _display_path(resolved), resolved.as_posix()}:
                    lookup[key] = step
    return lookup


def _first_nonempty_sequence_value(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        for item in _as_sequence(value):
            text = str(item).strip()
            if text:
                return text
    return None


def _materializer_receiver_feedback_row(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    step: Mapping[str, Any] | None,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    if truthy_authority_field_violations(manifest):
        return None
    if manifest.get("receiver_contract_satisfied") is not False:
        return None
    candidate_archive = (
        manifest.get("candidate_archive")
        if isinstance(manifest.get("candidate_archive"), Mapping)
        else {}
    )
    if not candidate_archive:
        return None
    byte_closed = (
        manifest.get("byte_closed_candidate_emitted") is True
        or bool(str(candidate_archive.get("path") or "").strip())
    )
    if not byte_closed:
        return None
    source_archive = (
        manifest.get("source_archive")
        if isinstance(manifest.get("source_archive"), Mapping)
        else {}
    )
    selected_key, selected_delta = selected_materializer_delta(manifest)
    archive_delta = _materializer_manifest_archive_delta(manifest) or {}
    source_bytes = _optional_int(
        archive_delta.get("source_archive_bytes") or source_archive.get("bytes")
    )
    candidate_bytes = _optional_int(
        archive_delta.get("candidate_archive_bytes")
        or candidate_archive.get("bytes")
    )
    saved_bytes = _optional_int(archive_delta.get("realized_saved_bytes"))
    if saved_bytes is None and source_bytes is not None and candidate_bytes is not None:
        saved_bytes = source_bytes - candidate_bytes
    saved_bytes = int(saved_bytes or 0)
    readiness_blockers = [
        str(item)
        for item in _as_sequence(manifest.get("readiness_blockers"))
        if str(item)
    ]
    receiver_verification = (
        manifest.get("receiver_verification")
        if isinstance(manifest.get("receiver_verification"), Mapping)
        else {}
    )
    receiver_blockers = [
        str(item)
        for item in _as_sequence(receiver_verification.get("blockers"))
        if str(item)
    ]
    candidate_sha = str(candidate_archive.get("sha256") or "").strip()
    source_sha = str(source_archive.get("sha256") or "").strip()
    candidate_id = _first_nonempty_sequence_value(
        manifest.get("candidate_id"),
        None if step is None else step.get("candidate_ids"),
        None if step is None else step.get("source_unit_ids"),
        f"receiver_negative_{candidate_sha[:12]}" if candidate_sha else None,
    )
    if candidate_id is None:
        candidate_id = f"receiver_negative_{_sha256_json(dict(manifest))[:12]}"
    expected_artifact_paths = (
        []
        if step is None
        else [
            str(path).strip()
            for path in _as_sequence(step.get("expected_artifact_paths"))
            if str(path).strip()
        ]
    )
    cache = dict(cache_identity)
    cache.setdefault(
        "cache_sha256",
        candidate_sha or source_sha or _sha256_json(dict(manifest)),
    )
    rate_positive = saved_bytes > 0
    observed_rate_gain = (
        CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes) if rate_positive else 0.0
    )
    return _false_authority_payload(
        {
            "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
            "observation_kind": "family_agnostic_materializer_empirical_observation",
            "observation_id": (
                "materializer_receiver_negative_"
                f"{_slug(candidate_id)}_"
                f"{candidate_sha[:12] if candidate_sha else _sha256_json(dict(manifest))[:12]}"
            ),
            "candidate_id": candidate_id,
            "axis": RECEIVER_NEGATIVE_MATERIALIZER_AXIS,
            "resource_kind": "local_cpu",
            "runtime_identity": dict(runtime_identity),
            "cache_identity": cache,
            "source_path": _display_path(manifest_path),
            "queue_id": None if step is None else step.get("queue_id"),
            "experiment_id": None if step is None else step.get("experiment_id"),
            "step_id": None if step is None else step.get("step_id"),
            "target_kind": manifest.get("target_kind"),
            "materializer_id": manifest.get("materializer_id"),
            "portability_contract": manifest.get("portability_contract"),
            "receiver_contract_kind": manifest.get("receiver_contract_kind"),
            "source_archive_path": source_archive.get("path"),
            "source_archive_sha256": source_sha or None,
            "source_archive_bytes": source_bytes,
            "candidate_archive_path": candidate_archive.get("path"),
            "candidate_archive_sha256": candidate_sha or None,
            "candidate_archive_bytes": candidate_bytes,
            "artifact_bytes": candidate_bytes or source_bytes or 0,
            "saved_bytes": saved_bytes,
            "archive_delta_status": archive_delta.get("status"),
            "archive_delta_source_kind": archive_delta.get("archive_delta_source_kind"),
            "observed_rate_gain": observed_rate_gain,
            "observed_score_gain": 0.0,
            "rate_positive": rate_positive,
            "receiver_contract_satisfied": False,
            "receiver_verification_blockers": receiver_blockers,
            "readiness_blockers": list(
                dict.fromkeys([*readiness_blockers, *receiver_blockers])
            ),
            "runtime_consumption_proof_path": manifest.get(
                "runtime_consumption_proof_path"
            ),
            "manifest_path": _display_path(manifest_path),
            "selected_member_name": manifest.get("selected_member_name"),
            "selected_member_names": manifest.get("selected_member_names") or [],
            "selected_materialization_key": selected_key or None,
            "selected_materialization": dict(selected_delta),
            "selected_compression": (
                dict(selected_delta) if selected_key == "selected_compression" else {}
            ),
            "factorization": (
                dict(selected_delta) if selected_key == "factorization" else {}
            ),
            "selected_elision": (
                dict(selected_delta) if selected_key == "selected_elision" else {}
            ),
            "source_unit_ids": (
                [] if step is None else _as_sequence(step.get("source_unit_ids"))
            ),
            "source_selection_ids": (
                [] if step is None else _as_sequence(step.get("source_selection_ids"))
            ),
            "work_ids": [] if step is None else _as_sequence(step.get("work_ids")),
            "backlog_keys": [] if step is None else _as_sequence(step.get("backlog_keys")),
            "expected_artifact_paths": expected_artifact_paths,
            "signal_semantics": "byte_closed_receiver_negative_materializer_feedback",
            "recommended_planner_action": (
                "demote_or_repair_matching_receiver_contract_before_refilling_bucket"
            ),
            "observation_feedback_is_not_score_authority": True,
        }
    )


_RECEIVER_FEEDBACK_MERGE_SEQUENCE_KEYS = (
    "readiness_blockers",
    "receiver_verification_blockers",
    "selected_member_names",
    "source_unit_ids",
    "source_selection_ids",
    "work_ids",
    "backlog_keys",
    "expected_artifact_paths",
)


def _empty_for_receiver_feedback_merge(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _merged_sequence_strings(*values: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _as_sequence(value):
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def _materializer_receiver_feedback_merge_key(row: Mapping[str, Any]) -> str:
    candidate_sha = str(row.get("candidate_archive_sha256") or "").strip()
    target_kind = str(row.get("target_kind") or "").strip()
    materializer_id = str(row.get("materializer_id") or "").strip()
    receiver_kind = str(row.get("receiver_contract_kind") or "").strip()
    if candidate_sha and (target_kind or materializer_id or receiver_kind):
        return "|".join(
            (
                "candidate_archive",
                target_kind,
                materializer_id,
                receiver_kind,
                candidate_sha,
            )
        )
    observation_id = str(row.get("observation_id") or "").strip()
    if observation_id:
        return f"observation_id|{observation_id}"
    return f"row_sha256|{_sha256_json(dict(row))}"


def _merge_materializer_receiver_feedback_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_dict = dict(row)
        key = _materializer_receiver_feedback_merge_key(row_dict)
        existing = merged.get(key)
        if existing is None:
            merged[key] = row_dict
            continue
        for field in _RECEIVER_FEEDBACK_MERGE_SEQUENCE_KEYS:
            values = _merged_sequence_strings(existing.get(field), row_dict.get(field))
            if values:
                existing[field] = values
        for field, value in row_dict.items():
            if field in _RECEIVER_FEEDBACK_MERGE_SEQUENCE_KEYS:
                continue
            if _empty_for_receiver_feedback_merge(existing.get(field)) and not (
                _empty_for_receiver_feedback_merge(value)
            ):
                existing[field] = value
    return list(merged.values())


def _write_materializer_receiver_feedback_sweep(
    *,
    run_dir: Path,
    queue_observation: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
) -> Path | None:
    root = (run_dir / "materializer_outputs").resolve(strict=False)
    if not root.exists():
        return None
    step_lookup = _step_lookup_by_artifact_path(queue_observation)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("*.json")):
        if len(rows) >= MAX_AUTO_DISCOVERED_RECEIVER_FEEDBACK_MANIFESTS:
            break
        if not _path_under_root(manifest_path, root):
            continue
        try:
            if manifest_path.stat().st_size > MAX_AUTO_DISCOVERED_RECEIVER_FEEDBACK_MANIFEST_BYTES:
                continue
        except OSError:
            continue
        manifest = _load_json_object_if_present(manifest_path)
        if manifest is None:
            continue
        step = None
        for key in {
            _display_path(manifest_path),
            manifest_path.resolve(strict=False).as_posix(),
        }:
            step = step_lookup.get(key)
            if step is not None:
                break
        row = _materializer_receiver_feedback_row(
            manifest_path=manifest_path,
            manifest=manifest,
            step=step,
            runtime_identity=runtime_identity,
            cache_identity=cache_identity,
        )
        if row is not None:
            rows.append(row)
    rows = _merge_materializer_receiver_feedback_rows(rows)
    if not rows:
        return None
    blockers = []
    if len(rows) >= MAX_AUTO_DISCOVERED_RECEIVER_FEEDBACK_MANIFESTS:
        blockers.append("materializer_receiver_feedback_manifest_limit_reached")
    rate_positive_count = sum(1 for row in rows if row.get("rate_positive") is True)
    output = run_dir / "materializer_receiver_feedback_observations.json"
    _write_json(
        output,
        _false_authority_payload(
            {
                "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
                "observation_kind": "family_agnostic_materializer_receiver_feedback_sweep",
                "generated_at_utc": _utc_stamp(),
                "run_dir": _display_path(run_dir),
                "archive_count": len(rows),
                "observation_count": len(rows),
                "rate_positive_count": rate_positive_count,
                "rate_nonpositive_count": len(rows) - rate_positive_count,
                "receiver_negative_count": len(rows),
                "planner_feedback": {
                    "schema": "family_agnostic_materializer_planner_feedback.v1",
                    "observation_kind": "materializer_receiver_feedback",
                    "receiver_negative_count": len(rows),
                    "blocker_counts": _blocker_counts(
                        [
                            blocker
                            for row in rows
                            for blocker in _as_sequence(row.get("readiness_blockers"))
                        ]
                    ),
                    "recommended_acquisition_rule": (
                        "demote_or_repair_matching_receiver_contract_before_refilling_bucket"
                    ),
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": blockers,
                "observations": rows,
                "allowed_use": "local_materializer_receiver_feedback_for_replanning_only",
                "forbidden_use": (
                    "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                ),
            }
        ),
    )
    return output


def _review_packet_axis(packet: Mapping[str, Any]) -> str | None:
    axis = str(packet.get("score_axis") or "").strip().lower()
    if axis == "contest_cpu" and packet.get("exact_cpu_evidence") is True:
        return axis
    if axis == "contest_cuda" and packet.get("exact_cuda_evidence") is True:
        return axis
    return None


def _nested_mapping(value: Any, key: str) -> Mapping[str, Any]:
    item = value.get(key) if isinstance(value, Mapping) else None
    return item if isinstance(item, Mapping) else {}


def _packet_pair_identity(packet: Mapping[str, Any]) -> tuple[str, int, int, str] | None:
    custody = _nested_mapping(packet, "custody")
    runtime = _nested_mapping(packet, "runtime_custody")
    archive_sha = str(custody.get("archive_sha256") or "").strip()
    runtime_content = str(runtime.get("runtime_content_tree_sha256") or "").strip()
    try:
        archive_bytes = int(custody.get("archive_bytes"))
        n_samples = int(custody.get("n_samples"))
    except (TypeError, ValueError):
        return None
    if not (archive_sha and runtime_content):
        return None
    return archive_sha, archive_bytes, n_samples, runtime_content


def _candidate_id_from_packet_pair(cpu: Mapping[str, Any], cuda: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "technique"):
        cpu_value = str(cpu.get(key) or "").strip()
        cuda_value = str(cuda.get(key) or "").strip()
        if cpu_value and cpu_value == cuda_value:
            return cpu_value
    archive_sha = _nested_mapping(cpu, "custody").get("archive_sha256")
    return f"exact_auth_pair_{str(archive_sha or '')[:12] or 'unknown'}"


def _validate_discovered_exact_auth_packet_pair(
    *,
    candidate_id: str,
    cpu_packet: Mapping[str, Any],
    cuda_packet: Mapping[str, Any],
    cpu_path: Path,
    cuda_path: Path,
) -> str | None:
    from tac.optimization.inverse_steganalysis_acquisition import (
        InverseSteganalysisAcquisitionError,
        paired_exact_auth_calibration_observations_from_review_packets,
    )

    try:
        paired_exact_auth_calibration_observations_from_review_packets(
            [cpu_packet, cuda_packet],
            candidate_id=candidate_id,
            packet_paths=[_display_path(cpu_path), _display_path(cuda_path)],
        )
    except InverseSteganalysisAcquisitionError as exc:
        return str(exc)
    return None


def _iter_exact_auth_calibration_packet_candidates(
    roots: Sequence[str | Path],
    *,
    glob_pattern: str,
) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    blockers: list[str] = []
    for raw_root in roots:
        root = _resolve(raw_root)
        if root.is_file():
            paths.append(root)
            continue
        if root.is_dir():
            paths.extend(path for path in root.glob(glob_pattern) if path.is_file())
            continue
        blockers.append(f"exact_auth_calibration_packet_root_missing:{_display_path(root)}")
    return sorted(dict.fromkeys(paths), key=lambda path: path.as_posix()), blockers


def _derived_exact_auth_calibration_packet_roots(*, run_dir: Path) -> list[Path]:
    roots = [run_dir]
    roots.extend(path for path in run_dir.rglob("exact_eval_handoff") if path.is_dir())
    roots.extend(path.parent for path in run_dir.rglob("*result_review*.json") if path.is_file())
    return sorted(dict.fromkeys(roots), key=lambda path: path.as_posix())


def _feedback_exact_auth_calibration_inputs(
    args: argparse.Namespace,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    explicit_packet_paths = [_resolve(path) for path in args.exact_auth_calibration_packet]
    explicit_paths = [_display_path(path) for path in explicit_packet_paths]
    explicit_candidate_id = args.exact_auth_calibration_candidate_id or args.candidate_id
    if explicit_packet_paths:
        blockers: list[str] = []
        discovery_pair: dict[str, Any] | None = None
        packet_paths = explicit_paths
        if not explicit_candidate_id:
            blockers.append("exact_auth_calibration_candidate_id_missing_for_explicit_packets")
        if len(explicit_packet_paths) != 2:
            blockers.append("exact_auth_calibration_explicit_packets_must_be_paired_cpu_cuda")
        else:
            packets = [
                _load_json_object_if_present(path)
                for path in explicit_packet_paths
            ]
            if any(packet is None for packet in packets):
                blockers.append("exact_auth_calibration_explicit_packet_json_invalid")
            elif explicit_candidate_id:
                by_axis: dict[str, tuple[Path, dict[str, Any]]] = {}
                for path, packet in zip(explicit_packet_paths, packets, strict=True):
                    if packet is None:
                        continue
                    axis = _review_packet_axis(packet)
                    if axis is None:
                        blockers.append(
                            f"exact_auth_calibration_explicit_packet_axis_invalid:{_display_path(path)}"
                        )
                        continue
                    by_axis[axis] = (path, packet)
                if {"contest_cpu", "contest_cuda"} <= set(by_axis):
                    cpu_path, cpu_packet = by_axis["contest_cpu"]
                    cuda_path, cuda_packet = by_axis["contest_cuda"]
                    invalid_reason = _validate_discovered_exact_auth_packet_pair(
                        candidate_id=str(explicit_candidate_id),
                        cpu_packet=cpu_packet,
                        cuda_packet=cuda_packet,
                        cpu_path=cpu_path,
                        cuda_path=cuda_path,
                    )
                    if invalid_reason is not None:
                        blockers.append(
                            "exact_auth_calibration_explicit_packet_pair_invalid:"
                            f"{invalid_reason}"
                        )
                    else:
                        identity = _packet_pair_identity(cpu_packet)
                        if identity is not None:
                            packet_paths = [_display_path(cpu_path), _display_path(cuda_path)]
                            discovery_pair = {
                                "archive_sha256": identity[0],
                                "archive_bytes": identity[1],
                                "n_samples": identity[2],
                                "runtime_content_tree_sha256": identity[3],
                                "contest_cpu_packet_path": _display_path(cpu_path),
                                "contest_cuda_packet_path": _display_path(cuda_path),
                            }
                else:
                    blockers.append(
                        "exact_auth_calibration_explicit_packets_require_one_cpu_one_cuda"
                    )
        return {
            "packet_paths": packet_paths,
            "candidate_id": explicit_candidate_id,
            "source": "explicit_cli",
            "discovery_roots": [],
            "discovery_pair": discovery_pair,
            "blockers": blockers,
        }
    explicit_roots = list(args.exact_auth_calibration_packet_root)
    roots: list[str | Path] = explicit_roots
    required_roots = bool(explicit_roots)
    source = "auto_discovery"
    if not roots and run_dir is not None:
        roots = _derived_exact_auth_calibration_packet_roots(run_dir=run_dir)
        source = "run_derived_discovery"
    if not roots:
        return {
            "packet_paths": [],
            "candidate_id": args.exact_auth_calibration_candidate_id or args.candidate_id,
            "source": "none",
            "discovery_roots": [],
            "discovery_pair": None,
            "blockers": [],
        }

    candidate_paths, discovery_blockers = _iter_exact_auth_calibration_packet_candidates(
        roots,
        glob_pattern=args.exact_auth_calibration_packet_glob,
    )
    grouped: dict[tuple[str, int, int, str], dict[str, tuple[Path, dict[str, Any]]]] = {}
    for path in candidate_paths:
        packet = _load_json_object_if_present(path)
        if packet is None or packet.get("schema") != RESULT_REVIEW_PACKET_SCHEMA:
            continue
        axis = _review_packet_axis(packet)
        identity = _packet_pair_identity(packet)
        if axis is None or identity is None:
            continue
        current = grouped.setdefault(identity, {})
        previous = current.get(axis)
        if previous is None or path.stat().st_mtime >= previous[0].stat().st_mtime:
            current[axis] = (path, packet)

    pairs: list[tuple[tuple[str, int, int, str], dict[str, tuple[Path, dict[str, Any]]]]] = [
        (identity, rows)
        for identity, rows in grouped.items()
        if {"contest_cpu", "contest_cuda"} <= set(rows)
    ]
    if not pairs:
        if not required_roots and not candidate_paths and not discovery_blockers:
            return {
                "packet_paths": [],
                "candidate_id": args.exact_auth_calibration_candidate_id or args.candidate_id,
                "source": source,
                "discovery_roots": _display_path_list(roots),
                "discovery_pair": None,
                "blockers": [],
            }
        return {
            "packet_paths": [],
            "candidate_id": args.exact_auth_calibration_candidate_id or args.candidate_id,
            "source": source,
            "discovery_roots": _display_path_list(roots),
            "discovery_pair": None,
            "blockers": [
                *discovery_blockers,
                "exact_auth_calibration_packet_pair_not_found",
            ],
        }
    pairs.sort(
        key=lambda item: (
            max(path.stat().st_mtime for path, _packet in item[1].values()),
            item[0][0],
            item[0][1],
            item[0][2],
            item[0][3],
        ),
        reverse=True,
    )
    invalid_pair_reasons: list[str] = []
    selected_identity: tuple[str, int, int, str] | None = None
    for pair_identity, rows in pairs:
        cpu_path, cpu_packet = rows["contest_cpu"]
        cuda_path, cuda_packet = rows["contest_cuda"]
        candidate_id = (
            args.exact_auth_calibration_candidate_id
            or args.candidate_id
            or _candidate_id_from_packet_pair(cpu_packet, cuda_packet)
        )
        invalid_reason = _validate_discovered_exact_auth_packet_pair(
            candidate_id=candidate_id,
            cpu_packet=cpu_packet,
            cuda_packet=cuda_packet,
            cpu_path=cpu_path,
            cuda_path=cuda_path,
        )
        if invalid_reason is None:
            selected_identity = pair_identity
            break
        invalid_pair_reasons.append(
            f"exact_auth_calibration_packet_pair_invalid:{invalid_reason}"
        )
    else:
        return {
            "packet_paths": [],
            "candidate_id": args.exact_auth_calibration_candidate_id or args.candidate_id,
            "source": source,
            "discovery_roots": _display_path_list(roots),
            "discovery_pair": None,
            "blockers": [
                *discovery_blockers,
                *list(dict.fromkeys(invalid_pair_reasons)),
                "exact_auth_calibration_packet_pair_not_found",
            ],
        }
    if selected_identity is None:  # pragma: no cover - guarded by the loop else above.
        raise RuntimeError("exact-auth calibration packet selection invariant failed")
    return {
        "packet_paths": [_display_path(cpu_path), _display_path(cuda_path)],
        "candidate_id": candidate_id,
        "source": source,
        "discovery_roots": _display_path_list(roots),
        "discovery_pair": {
            "archive_sha256": selected_identity[0],
            "archive_bytes": selected_identity[1],
            "n_samples": selected_identity[2],
            "runtime_content_tree_sha256": selected_identity[3],
            "contest_cpu_packet_path": _display_path(cpu_path),
            "contest_cuda_packet_path": _display_path(cuda_path),
        },
        "blockers": [],
    }


def _queue_feedback_replan_continuation_lane_id(
    queue: Mapping[str, Any],
    *,
    plan_path: Path | None = None,
) -> str:
    if plan_path is not None:
        try:
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            plan_payload = None
        if isinstance(plan_payload, Mapping):
            lane_id = str(plan_payload.get("lane_id") or "").strip()
            if lane_id:
                return lane_id
    for experiment in _as_sequence(queue.get("experiments")):
        if not isinstance(experiment, Mapping):
            continue
        lane_id = str(experiment.get("lane_id") or "").strip()
        if lane_id:
            return lane_id
    queue_id = str(queue.get("queue_id") or "materializer_campaign").strip()
    return f"{queue_id or 'materializer_campaign'}_feedback_replan"


def _text_excerpt(text: str, *, limit: int = 4096) -> str:
    return text[-limit:] if len(text) > limit else text


def _queue_performance_summary_payload(
    performance_result: CommandResult,
    *,
    queue: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _json_from_stdout(performance_result)
    if payload is None or payload.get("schema") != QUEUE_PERFORMANCE_SUMMARY_SCHEMA:
        blocker = (
            "queue_performance_command_failed"
            if performance_result.returncode != 0
            else "queue_performance_stdout_not_json_object"
        )
        payload = {
            "schema": UNAVAILABLE_QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
            "queue_id": str(queue.get("queue_id") or "unknown_queue"),
            "telemetry_only": True,
            "event_count": 0,
            "performance_command_failed": performance_result.returncode != 0,
            "performance_command_stdout_excerpt": _text_excerpt(performance_result.stdout),
            "performance_command_stderr_excerpt": _text_excerpt(performance_result.stderr),
            "blockers": [blocker],
        }
    else:
        authority_violations = truthy_authority_field_violations(payload)
        if authority_violations:
            payload = {
                "schema": UNAVAILABLE_QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
                "queue_id": str(queue.get("queue_id") or payload.get("queue_id") or "unknown_queue"),
                "telemetry_only": True,
                "event_count": 0,
                "performance_command_failed": False,
                "performance_command_stdout_excerpt": _text_excerpt(performance_result.stdout),
                "performance_command_stderr_excerpt": _text_excerpt(performance_result.stderr),
                "authority_violations": authority_violations,
                "blockers": ["queue_performance_summary_truthy_authority_fields"],
            }
        else:
            payload = dict(payload)
            payload.setdefault("blockers", [])
    payload["performance_command_returncode"] = performance_result.returncode
    payload["performance_command_elapsed_seconds"] = performance_result.elapsed_seconds
    payload["source_kind"] = "byte_shaving_materializer_campaign_runner"
    payload["allowed_use"] = "inverse_action_acquisition_runtime_denominator_update"
    payload["forbidden_use"] = "score_claim_or_promotion_or_rank_kill_authority"
    return _false_authority_payload(payload)


def _queue_performance_replan_blockers(
    args: argparse.Namespace,
    *,
    queue_performance_summary: Mapping[str, Any],
    runtime_identity_path: Path | None,
    cache_identity_path: Path | None,
    exact_auth_calibration_inputs: Mapping[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    if queue_performance_summary.get("schema") != QUEUE_PERFORMANCE_SUMMARY_SCHEMA:
        blockers.append("queue_performance_summary_not_consumable")
    if int(queue_performance_summary.get("event_count") or 0) < 1:
        blockers.append("queue_performance_summary_has_no_completed_step_events")
    if runtime_identity_path is None or not runtime_identity_path.exists():
        blockers.append("queue_performance_runtime_identity_missing")
    if cache_identity_path is None or not cache_identity_path.exists():
        blockers.append("queue_performance_cache_identity_missing")
    if exact_auth_calibration_inputs is not None:
        blockers.extend(
            str(item)
            for item in _as_sequence(exact_auth_calibration_inputs.get("blockers"))
        )
    return blockers


def _feedback_action_functional_command_hint(
    args: argparse.Namespace,
    *,
    plan_path: Path,
    run_dir: Path,
    queue_performance_summary_path: Path,
    runtime_identity_path: Path | None,
    cache_identity_path: Path | None,
    feedback_observation_paths: Sequence[str | Path] = (),
    materializer_chain_manifest_paths: Sequence[str | Path] = (),
    queue_observation_paths: Sequence[str | Path] = (),
    exact_auth_calibration_inputs: Mapping[str, Any] | None = None,
    output_stem: str = "inverse_steganalysis_action_functional.feedback",
) -> list[str]:
    command = [
        sys.executable,
        QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
        "--output",
        _display_path(run_dir / f"{output_stem}.json"),
        "--md-out",
        _display_path(run_dir / f"{output_stem}.md"),
        "--repo-root",
        REPO_ROOT.as_posix(),
        "--resource-kind",
        str(args.resource_kind),
        "--byte-shaving-campaign-plan",
        _display_path(plan_path),
        "--queue-performance-summary",
        _display_path(queue_performance_summary_path),
        "--queue-performance-axis",
        str(args.queue_performance_axis),
    ]
    if runtime_identity_path is not None:
        command.extend(
            [
                "--queue-performance-runtime-identity",
                _display_path(runtime_identity_path),
            ]
        )
    if cache_identity_path is not None:
        command.extend(
            [
                "--queue-performance-cache-identity",
                _display_path(cache_identity_path),
            ]
        )
    if args.queue_performance_candidate_map is not None:
        command.extend(
            [
                "--queue-performance-candidate-map",
                _display_path(_resolve(args.queue_performance_candidate_map)),
            ]
        )
    _append_path_args(command, "--observation", list(feedback_observation_paths))
    _append_path_args(
        command,
        "--materializer-chain-manifest",
        list(materializer_chain_manifest_paths),
    )
    _append_path_args(command, "--queue-observation", list(queue_observation_paths))
    calibration_inputs = (
        exact_auth_calibration_inputs
        if exact_auth_calibration_inputs is not None
        else _feedback_exact_auth_calibration_inputs(args, run_dir=run_dir)
    )
    _append_path_args(
        command,
        "--exact-auth-calibration-packet",
        [str(path) for path in _as_sequence(calibration_inputs.get("packet_paths"))],
    )
    calibration_candidate = calibration_inputs.get("candidate_id")
    if calibration_candidate:
        command.extend(
            [
                "--exact-auth-calibration-candidate-id",
                str(calibration_candidate),
            ]
        )
    if args.total_byte_budget is not None:
        command.extend(["--total-byte-budget", str(args.total_byte_budget)])
    if args.lambda_rate is not None:
        command.extend(["--lambda-rate", str(args.lambda_rate)])
    return command


def _queue_feedback_replan_request_payload(
    args: argparse.Namespace,
    *,
    summary_path: Path,
    plan_path: Path,
    queue_performance_summary_path: Path,
    queue_performance_summary: Mapping[str, Any],
    runtime_identity_path: Path | None,
    cache_identity_path: Path | None,
    generated_runtime_identity: bool,
    generated_cache_identity: bool,
    run_dir: Path,
    execution_queue: Path,
    state_path: Path,
    queue_observation_path: Path | None = None,
    queue_observation_recovery_plan_path: Path | None = None,
    queue_observation_recovery_plan: Mapping[str, Any] | None = None,
    action_functional_output_stem: str = (
        "inverse_steganalysis_action_functional.feedback"
    ),
) -> dict[str, Any]:
    exact_auth_calibration_inputs = _feedback_exact_auth_calibration_inputs(
        args,
        run_dir=run_dir,
    )
    feedback_observation_paths = _feedback_observation_paths(args, run_dir=run_dir)
    materializer_chain_manifest_paths = _materializer_chain_manifest_paths(
        args,
        run_dir=run_dir,
    )
    queue_observation_paths = (
        [queue_observation_path] if queue_observation_path is not None else []
    )
    blockers = _queue_performance_replan_blockers(
        args,
        queue_performance_summary=queue_performance_summary,
        runtime_identity_path=runtime_identity_path,
        cache_identity_path=cache_identity_path,
        exact_auth_calibration_inputs=exact_auth_calibration_inputs,
    )
    command_hint = _feedback_action_functional_command_hint(
        args,
        plan_path=plan_path,
        run_dir=run_dir,
        queue_performance_summary_path=queue_performance_summary_path,
        runtime_identity_path=runtime_identity_path,
        cache_identity_path=cache_identity_path,
        feedback_observation_paths=feedback_observation_paths,
        materializer_chain_manifest_paths=materializer_chain_manifest_paths,
        queue_observation_paths=queue_observation_paths,
        exact_auth_calibration_inputs=exact_auth_calibration_inputs,
        output_stem=action_functional_output_stem,
    )
    return _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_REPLAN_REQUEST_SCHEMA,
            "source_run_path": _display_path(summary_path),
            "run_dir": _display_path(run_dir),
            "plan_path": _display_path(plan_path),
            "queue_path": _display_path(execution_queue),
            "queue_state_path": _display_path(state_path),
            "queue_performance_summary_path": _display_path(queue_performance_summary_path),
            "queue_performance_runtime_identity_path": (
                None if runtime_identity_path is None else _display_path(runtime_identity_path)
            ),
            "queue_performance_cache_identity_path": (
                None if cache_identity_path is None else _display_path(cache_identity_path)
            ),
            "queue_performance_runtime_identity_generated": generated_runtime_identity,
            "queue_performance_cache_identity_generated": generated_cache_identity,
            "feedback_observation_paths": feedback_observation_paths,
            "feedback_observation_auto_discovery_root": _display_path(run_dir),
            "feedback_observation_auto_discovery_enabled": True,
            "materializer_chain_manifest_paths": materializer_chain_manifest_paths,
            "materializer_chain_manifest_auto_discovery_root": _display_path(run_dir),
            "materializer_chain_manifest_auto_discovery_enabled": True,
            "queue_observation_path": (
                None if queue_observation_path is None else _display_path(queue_observation_path)
            ),
            "queue_observation_recovery_plan_path": (
                None
                if queue_observation_recovery_plan_path is None
                else _display_path(queue_observation_recovery_plan_path)
            ),
            "queue_observation_recovery_plan": (
                None
                if queue_observation_recovery_plan is None
                else dict(queue_observation_recovery_plan)
            ),
            "queue_observation_recovery_required": (
                isinstance(queue_observation_recovery_plan, Mapping)
                and queue_observation_recovery_plan.get("recovery_required") is True
            ),
            "queue_observation_maintenance_recommended": (
                isinstance(queue_observation_recovery_plan, Mapping)
                and queue_observation_recovery_plan.get("maintenance_recommended")
                is True
            ),
            "exact_auth_calibration_packet_paths": exact_auth_calibration_inputs[
                "packet_paths"
            ],
            "exact_auth_calibration_candidate_id": exact_auth_calibration_inputs[
                "candidate_id"
            ],
            "exact_auth_calibration_packet_source": exact_auth_calibration_inputs[
                "source"
            ],
            "exact_auth_calibration_discovery_roots": exact_auth_calibration_inputs[
                "discovery_roots"
            ],
            "exact_auth_calibration_discovery_pair": exact_auth_calibration_inputs[
                "discovery_pair"
            ],
            "performance_schema": queue_performance_summary.get("schema"),
            "performance_event_count": queue_performance_summary.get("event_count"),
            "ready_for_action_functional_feedback": not blockers,
            "blockers": blockers,
            "consumer_contract": {
                "tool": QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                "flag": "--queue-performance-summary",
                "role": "telemetry_denominator_calibration_only",
                "requires_non_performance_action_source": True,
                "non_performance_action_source": "--byte-shaving-campaign-plan",
            },
            "suggested_action_functional_command": command_hint if not blockers else None,
            "command_template": command_hint,
            "allowed_use": "next_inverse_action_replan_input_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
        }
    )


def _command_path_arg(command: Sequence[str], flag: str) -> Path | None:
    command_items = [str(item) for item in command]
    for index, item in enumerate(command_items):
        if item == flag:
            value_index = index + 1
            if value_index >= len(command_items):
                return None
            return _resolve(command_items[value_index])
        prefix = f"{flag}="
        if item.startswith(prefix):
            value = item[len(prefix) :].strip()
            return _resolve(value) if value else None
    return None


def _queue_feedback_replan_forbidden_flag_uses(command: Sequence[str]) -> list[str]:
    uses: list[str] = []
    for item in command:
        for flag in QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_FLAGS:
            if item == flag or item.startswith(f"{flag}="):
                uses.append(item)
    return list(dict.fromkeys(uses))


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _queue_feedback_replan_followup_queue_payload(
    args: argparse.Namespace,
    *,
    queue_feedback_replan_request: Mapping[str, Any],
    queue_feedback_replan_request_path: Path,
    run_dir: Path,
    execution_queue: Path,
    state_path: Path,
    source_queue: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    blockers = [str(item) for item in _as_sequence(queue_feedback_replan_request.get("blockers"))]
    if queue_feedback_replan_request.get("queue_observation_recovery_required") is True:
        blockers.append("queue_observation_recovery_required")
    if queue_feedback_replan_request.get("ready_for_action_functional_feedback") is not True:
        blockers.append("queue_feedback_replan_request_not_ready")
    command = queue_feedback_replan_request.get("command_template")
    if not isinstance(command, list) or not command:
        blockers.append("queue_feedback_replan_command_missing")
        command_items: list[str] = []
    else:
        command_items = [str(item) for item in command]
        if len(command_items) < 2 or command_items[1] != (
            QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL
        ):
            blockers.append("queue_feedback_replan_command_not_action_functional_tool")
        forbidden_flags = sorted(_queue_feedback_replan_forbidden_flag_uses(command_items))
        blockers.extend(
            f"queue_feedback_replan_command_forbidden_flag:{flag}"
            for flag in forbidden_flags
        )
    output_path = _command_path_arg(command_items, "--output") if command_items else None
    md_path = _command_path_arg(command_items, "--md-out") if command_items else None
    if output_path is None:
        blockers.append("queue_feedback_replan_output_path_missing")
    elif not _path_is_under(output_path, run_dir):
        blockers.append("queue_feedback_replan_output_path_outside_run_dir")
    if md_path is not None and not _path_is_under(md_path, run_dir):
        blockers.append("queue_feedback_replan_md_outside_run_dir")
    if blockers:
        return None, list(dict.fromkeys(blockers))

    input_artifacts = [
        queue_feedback_replan_request_path,
        queue_feedback_replan_request.get("plan_path"),
        queue_feedback_replan_request.get("queue_performance_summary_path"),
        queue_feedback_replan_request.get("queue_performance_runtime_identity_path"),
        queue_feedback_replan_request.get("queue_performance_cache_identity_path"),
        queue_feedback_replan_request.get("queue_observation_path"),
        queue_feedback_replan_request.get("queue_observation_recovery_plan_path"),
        execution_queue,
        state_path,
    ]
    input_artifacts.extend(
        str(path)
        for path in _as_sequence(queue_feedback_replan_request.get("feedback_observation_paths"))
    )
    input_artifacts.extend(
        str(path)
        for path in _as_sequence(
            queue_feedback_replan_request.get("materializer_chain_manifest_paths")
        )
    )
    input_artifacts.extend(
        str(path)
        for path in _as_sequence(queue_feedback_replan_request.get("exact_auth_calibration_packet_paths"))
    )
    artifact_paths = [output_path]
    if md_path is not None:
        artifact_paths.append(md_path)
    experiment_id = "queue_feedback_replan_action_functional"
    step_id = "build_feedback_action_functional"
    queue_id = f"{source_queue.get('queue_id', 'materializer_campaign')}_feedback_replan"
    metadata = _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_REPLAN_EXPERIMENT_METADATA_SCHEMA,
            "source_run_dir": _display_path(run_dir),
            "source_queue_id": source_queue.get("queue_id"),
            "source_queue_path": _display_path(execution_queue),
            "source_queue_state_path": _display_path(state_path),
            "queue_feedback_replan_request_path": _display_path(queue_feedback_replan_request_path),
            "queue_performance_summary_path": queue_feedback_replan_request.get(
                "queue_performance_summary_path"
            ),
            "queue_performance_runtime_identity_path": queue_feedback_replan_request.get(
                "queue_performance_runtime_identity_path"
            ),
            "queue_performance_cache_identity_path": queue_feedback_replan_request.get(
                "queue_performance_cache_identity_path"
            ),
            "queue_performance_runtime_identity_generated": queue_feedback_replan_request.get(
                "queue_performance_runtime_identity_generated"
            ),
            "queue_performance_cache_identity_generated": queue_feedback_replan_request.get(
                "queue_performance_cache_identity_generated"
            ),
            "ready_for_action_functional_feedback": True,
            "allowed_use": "paused_local_feedback_replan_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
            "dispatch_blockers": [
                "paused_followup_queue_requires_explicit_operator_or_autopilot_resume",
                "exact_auth_eval_required_before_score_claim",
            ],
        }
    )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "paused",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": experiment_id,
                    "lane_id": args.lane_id,
                    "priority": 90,
                    "status": "queued",
                    "tags": [
                        "byte-shaving",
                        "inverse-steganalysis",
                        "queue-feedback",
                        "paused-followup",
                        "no-score-authority",
                    ],
                    "metadata": metadata,
                    "steps": [
                        {
                            "id": step_id,
                            "kind": "command",
                            "command": command_items,
                            "requires": [],
                            "resources": {"kind": "local_cpu"},
                            "timeout_seconds": 0,
                            "telemetry": {
                                "artifact_paths": [_display_path(path) for path in artifact_paths],
                                "input_artifact_paths": [
                                    _display_path(_resolve(str(path)))
                                    for path in input_artifacts
                                    if path is not None
                                ],
                                "recursive": False,
                            },
                            "postconditions": [
                                {
                                    "type": "path_exists",
                                    "path": _display_path(output_path),
                                },
                                {
                                    "type": "json_completion_contract",
                                    "path": _display_path(output_path),
                                    "required_equals": {
                                        "schema": "inverse_steganalysis_discrete_action_functional.v1"
                                    },
                                    "required_false": [
                                        *QUEUE_FEEDBACK_REPLAN_REQUIRED_FALSE_FIELDS,
                                    ],
                                    "false_or_missing": list(
                                        DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS
                                    ),
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    )
    return queue, []


def _queue_feedback_replan_followup_local_autopolicy_blockers(
    child_queue: Mapping[str, Any],
    *,
    run_dir: Path | None = None,
) -> list[str]:
    validation = validate_feedback_followup_queue(
        child_queue,
        run_dir=None if run_dir is None else _display_path(run_dir),
    )
    return [str(item) for item in validation["blockers"]]


def _queue_feedback_replan_followup_activation_policy(args: argparse.Namespace) -> str:
    if args.execute_queue_feedback_replan_followup:
        return "explicit_cli"
    if args.queue_feedback_replan_followup_policy_local_autopilot:
        return "local_autopilot_policy"
    return "paused_only"


def _execution_stdout_json(
    result: CommandResult,
    *,
    label: str,
    blockers: list[str],
) -> dict[str, Any] | None:
    payload = _json_from_stdout(result)
    if payload is None:
        blockers.append(f"{label}_stdout_not_json_object")
    return payload


def _queue_feedback_replan_followup_execution_payload(
    args: argparse.Namespace,
    *,
    child_queue: Mapping[str, Any],
    child_queue_path: Path,
    run_dir: Path,
    activation_policy: str,
) -> tuple[dict[str, Any], list[CommandResult], int]:
    state_path = (
        _resolve(args.queue_feedback_replan_followup_state)
        if args.queue_feedback_replan_followup_state is not None
        else run_dir / "queue_feedback_replan_followup.sqlite"
    )
    command_results: list[CommandResult] = []
    blockers: list[str] = []
    parsed: dict[str, Any] = {}

    def run_step(label: str, subcommand: Sequence[str]) -> CommandResult:
        result = _run(
            _experiment_queue_command(
                execution_queue=child_queue_path,
                state_path=state_path,
                subcommand=subcommand,
            ),
            check=False,
        )
        command_results.append(result)
        if result.returncode != 0:
            blockers.append(f"{label}_failed")
        parsed[label] = _execution_stdout_json(
            result,
            label=label,
            blockers=blockers,
        )
        return result

    validate_result = run_step("validate", ["validate"])
    init_result = run_step("init", ["init"]) if validate_result.returncode == 0 else None
    if init_result is not None and init_result.returncode == 0:
        default_control_reason = (
            "queue-owned local feedback replan policy"
            if activation_policy == "local_autopilot_policy"
            else "materializer campaign explicit local-only feedback replan autorun"
        )
        control_reason = (
            args.queue_feedback_replan_followup_activation_reason
            or default_control_reason
        )
        control_result = run_step(
            "control",
            ["control", "running", "--reason", control_reason],
        )
    else:
        control_result = None

    worker_result: CommandResult | None = None
    if control_result is not None and control_result.returncode == 0:
        worker_command = [
            "run-worker",
            "--execute",
            "--max-steps",
            str(args.queue_feedback_replan_followup_max_steps),
            "--max-parallel",
            str(args.queue_feedback_replan_followup_max_parallel),
            "--idle-sleep-seconds",
            str(args.queue_feedback_replan_followup_idle_sleep_seconds),
            "--poll-interval-seconds",
            str(args.queue_feedback_replan_followup_poll_interval_seconds),
            "--max-idle-cycles",
            str(args.queue_feedback_replan_followup_max_idle_cycles),
            "--noncanonical-state-rationale",
            (
                args.queue_feedback_replan_followup_state_rationale
                or "run-scoped materializer feedback replan child queue state"
            ),
        ]
        worker_result = run_step("worker", worker_command)

    if init_result is not None and init_result.returncode == 0:
        run_step(
            "observation",
            [
                "observe",
                "--tail-lines",
                str(args.queue_feedback_replan_followup_tail_lines),
                "--format",
                "json",
            ],
        )
        run_step("performance", ["performance"])

    output_path = None
    md_path = None
    try:
        step = child_queue["experiments"][0]["steps"][0]  # type: ignore[index]
        command = [str(item) for item in step["command"]]
        output_path = _command_path_arg(command, "--output")
        md_path = _command_path_arg(command, "--md-out")
    except (KeyError, IndexError, TypeError):
        blockers.append("child_queue_command_unreadable")

    worker_payload = parsed.get("worker")
    if isinstance(worker_payload, dict):
        if int(worker_payload.get("failure_count") or 0):
            blockers.append("worker_reported_failures")
        if int(worker_payload.get("success_count") or 0) < 1:
            blockers.append("worker_did_not_complete_feedback_action_functional")
    elif worker_result is not None and worker_result.returncode == 0:
        blockers.append("worker_result_missing")

    success = not blockers
    payload = _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_REPLAN_FOLLOWUP_EXECUTION_SCHEMA,
            "enabled": True,
            "success": success,
            "blockers": list(dict.fromkeys(blockers)),
            "queue_path": _display_path(child_queue_path),
            "queue_id": child_queue.get("queue_id"),
            "state_path": _display_path(state_path),
            "action_functional_output_path": (
                None if output_path is None else _display_path(output_path)
            ),
            "action_functional_md_path": None if md_path is None else _display_path(md_path),
            "max_steps": args.queue_feedback_replan_followup_max_steps,
            "max_parallel": args.queue_feedback_replan_followup_max_parallel,
            "activation_policy": activation_policy,
            "allowed_use": "local_queue_feedback_replan_execution_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
            "validate": parsed.get("validate"),
            "init": parsed.get("init"),
            "control": parsed.get("control"),
            "worker": parsed.get("worker"),
            "observation": parsed.get("observation"),
            "performance": parsed.get("performance"),
            "commands": [result.to_dict() for result in command_results],
        }
    )
    return payload, command_results, (0 if success else 2)


def _queue_feedback_replan_followup_execution_refusal_payload(
    *,
    blockers: Sequence[str],
    child_queue_path: Path,
    activation_policy: str,
) -> dict[str, Any]:
    return _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_REPLAN_FOLLOWUP_EXECUTION_SCHEMA,
            "enabled": True,
            "success": False,
            "blockers": list(dict.fromkeys(str(item) for item in blockers)),
            "queue_path": _display_path(child_queue_path),
            "queue_id": None,
            "state_path": None,
            "action_functional_output_path": None,
            "action_functional_md_path": None,
            "activation_policy": activation_policy,
            "allowed_use": "local_queue_feedback_replan_execution_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
            "commands": [],
        }
    )


def _queue_observation_recovery_activation_policy(args: argparse.Namespace) -> str:
    if args.execute_queue_observation_recovery:
        return "explicit_cli"
    if args.queue_observation_recovery_policy_local_autopilot:
        return "local_autopilot_policy"
    return "paused_only"


def _queue_observation_recovery_local_autopolicy_blockers(
    recovery_queue: Mapping[str, Any],
) -> list[str]:
    validation = validate_queue_observation_recovery_queue(recovery_queue)
    return [str(item) for item in validation["blockers"]]


def _queue_observation_recovery_source_paths(
    recovery_queue: Mapping[str, Any],
) -> tuple[Path | None, Path | None]:
    try:
        metadata = recovery_queue["experiments"][0]["metadata"]  # type: ignore[index]
        source_queue_path = metadata.get("source_queue_path")
        source_state_path = metadata.get("source_queue_state_path")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None, None
    if not source_queue_path or not source_state_path:
        return None, None
    return _resolve(str(source_queue_path)), _resolve(str(source_state_path))


def _queue_observation_recovery_execution_payload(
    args: argparse.Namespace,
    *,
    recovery_queue: Mapping[str, Any],
    recovery_queue_path: Path,
    recovery_queue_state_path: Path,
    activation_policy: str,
) -> tuple[dict[str, Any], list[CommandResult], int]:
    state_path = (
        _resolve(args.queue_observation_recovery_state)
        if args.queue_observation_recovery_state is not None
        else recovery_queue_state_path
    )
    command_results: list[CommandResult] = []
    blockers: list[str] = []
    parsed: dict[str, Any] = {}

    def run_step(label: str, subcommand: Sequence[str]) -> CommandResult:
        result = _run(
            _experiment_queue_command(
                execution_queue=recovery_queue_path,
                state_path=state_path,
                subcommand=subcommand,
            ),
            check=False,
        )
        command_results.append(result)
        if result.returncode != 0:
            blockers.append(f"{label}_failed")
        parsed[label] = _execution_stdout_json(
            result,
            label=label,
            blockers=blockers,
        )
        return result

    validation = validate_queue_observation_recovery_queue(recovery_queue)
    parsed["validation"] = validation
    blockers.extend(
        f"queue_observation_recovery_validation:{item}"
        for item in validation["blockers"]
    )

    validate_result = run_step("validate", ["validate"]) if validation["valid"] else None
    init_result = (
        run_step("init", ["init"])
        if validate_result is not None and validate_result.returncode == 0
        else None
    )
    if init_result is not None and init_result.returncode == 0:
        default_control_reason = (
            "queue-owned local observation recovery policy"
            if activation_policy == "local_autopilot_policy"
            else "materializer campaign explicit local-only queue observation recovery"
        )
        control_reason = (
            args.queue_observation_recovery_activation_reason
            or default_control_reason
        )
        control_result = run_step(
            "control",
            ["control", "running", "--reason", control_reason],
        )
    else:
        control_result = None

    worker_result: CommandResult | None = None
    if control_result is not None and control_result.returncode == 0:
        worker_command = [
            "run-worker",
            "--execute",
            "--max-steps",
            str(args.queue_observation_recovery_max_steps),
            "--max-parallel",
            str(args.queue_observation_recovery_max_parallel),
            "--idle-sleep-seconds",
            str(args.queue_observation_recovery_idle_sleep_seconds),
            "--poll-interval-seconds",
            str(args.queue_observation_recovery_poll_interval_seconds),
            "--max-idle-cycles",
            str(args.queue_observation_recovery_max_idle_cycles),
            "--noncanonical-state-rationale",
            (
                args.queue_observation_recovery_state_rationale
                or "run-scoped materializer queue observation recovery child queue state"
            ),
        ]
        worker_result = run_step("worker", worker_command)

    if init_result is not None and init_result.returncode == 0:
        run_step(
            "observation",
            [
                "observe",
                "--tail-lines",
                str(args.queue_observation_recovery_tail_lines),
                "--format",
                "json",
            ],
        )
        run_step("performance", ["performance"])

    source_queue_path, source_state_path = _queue_observation_recovery_source_paths(
        recovery_queue
    )
    if source_queue_path is None or source_state_path is None:
        blockers.append("source_queue_paths_missing")
    elif worker_result is not None and worker_result.returncode == 0:
        source_result = _run(
            _experiment_queue_command(
                execution_queue=source_queue_path,
                state_path=source_state_path,
                subcommand=[
                    "observe",
                    "--include-orphans",
                    "--tail-lines",
                    str(args.queue_observation_recovery_tail_lines),
                    "--format",
                    "json",
                ],
            ),
            check=False,
        )
        command_results.append(source_result)
        if source_result.returncode != 0:
            blockers.append("source_observation_after_failed")
        parsed["source_observation_after"] = _execution_stdout_json(
            source_result,
            label="source_observation_after",
            blockers=blockers,
        )

    worker_payload = parsed.get("worker")
    if isinstance(worker_payload, dict):
        if int(worker_payload.get("failure_count") or 0):
            blockers.append("worker_reported_failures")
        expected_successes = int(validation.get("command_count") or 0)
        if int(worker_payload.get("success_count") or 0) < expected_successes:
            blockers.append("worker_did_not_complete_all_recovery_actions")
    elif worker_result is not None and worker_result.returncode == 0:
        blockers.append("worker_result_missing")

    source_after = parsed.get("source_observation_after")
    if isinstance(source_after, dict) and source_after.get("healthy") is not True:
        blockers.append("source_observation_after_unhealthy")

    success = not blockers
    payload = _false_authority_payload(
        {
            "schema": QUEUE_OBSERVATION_RECOVERY_EXECUTION_SCHEMA,
            "enabled": True,
            "success": success,
            "blockers": list(dict.fromkeys(blockers)),
            "queue_path": _display_path(recovery_queue_path),
            "queue_id": recovery_queue.get("queue_id"),
            "state_path": _display_path(state_path),
            "source_queue_path": (
                None if source_queue_path is None else _display_path(source_queue_path)
            ),
            "source_queue_state_path": (
                None if source_state_path is None else _display_path(source_state_path)
            ),
            "max_steps": args.queue_observation_recovery_max_steps,
            "max_parallel": args.queue_observation_recovery_max_parallel,
            "activation_policy": activation_policy,
            "allowed_use": "local_queue_observation_recovery_execution_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
            "validation": parsed.get("validation"),
            "validate": parsed.get("validate"),
            "init": parsed.get("init"),
            "control": parsed.get("control"),
            "worker": parsed.get("worker"),
            "observation": parsed.get("observation"),
            "performance": parsed.get("performance"),
            "source_observation_after": parsed.get("source_observation_after"),
            "commands": [result.to_dict() for result in command_results],
        }
    )
    return payload, command_results, (0 if success else 2)


def _queue_observation_recovery_execution_refusal_payload(
    *,
    blockers: Sequence[str],
    recovery_queue_path: Path,
    activation_policy: str,
) -> dict[str, Any]:
    return _false_authority_payload(
        {
            "schema": QUEUE_OBSERVATION_RECOVERY_EXECUTION_SCHEMA,
            "enabled": True,
            "success": False,
            "blockers": list(dict.fromkeys(str(item) for item in blockers)),
            "queue_path": _display_path(recovery_queue_path),
            "queue_id": None,
            "state_path": None,
            "source_queue_path": None,
            "source_queue_state_path": None,
            "activation_policy": activation_policy,
            "allowed_use": "local_queue_observation_recovery_execution_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
            "commands": [],
        }
    )


def _post_recovery_feedback_replan_run_summary(
    *,
    run_dir: Path,
    summary_path: Path,
    plan_path: Path,
    execution_queue: Path,
    state_path: Path,
    queue: Mapping[str, Any],
    queue_performance_summary_path: Path,
    queue_performance_summary: Mapping[str, Any],
    queue_feedback_replan_request: Mapping[str, Any],
    queue_feedback_replan_followup_queue_path: Path,
    queue_feedback_replan_followup_queue: Mapping[str, Any] | None,
    queue_feedback_replan_followup_blockers: Sequence[str],
    queue_feedback_replan_followup_policy: str,
    queue_feedback_replan_followup_policy_enabled: bool,
    queue_feedback_replan_followup_policy_blockers: Sequence[str],
    queue_feedback_replan_followup_execution_requested: bool,
    queue_feedback_replan_followup_executed: bool,
    queue_feedback_replan_followup_execution: Mapping[str, Any] | None,
    queue_observation_path: Path,
    queue_observation_recovery_plan_path: Path,
    queue_observation_recovery_plan: Mapping[str, Any],
    exact_readiness_handoffs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _false_authority_payload(
        {
            "schema": RUN_SCHEMA,
            "run_dir": _display_path(run_dir),
            "source_run_path": _display_path(summary_path),
            "plan": _display_path(plan_path),
            "queue_id": queue.get("queue_id"),
            "queue_path": _display_path(execution_queue),
            "state_path": _display_path(state_path),
            "queue_performance_summary_path": _display_path(
                queue_performance_summary_path
            ),
            "performance": dict(queue_performance_summary),
            "queue_feedback_replan_request": dict(queue_feedback_replan_request),
            "queue_feedback_replan_ready": queue_feedback_replan_request.get(
                "ready_for_action_functional_feedback"
            )
            is True,
            "queue_feedback_replan_blockers": _as_sequence(
                queue_feedback_replan_request.get("blockers")
            ),
            "queue_feedback_replan_followup_queue_path": (
                _display_path(queue_feedback_replan_followup_queue_path)
                if queue_feedback_replan_followup_queue is not None
                else None
            ),
            "queue_feedback_replan_followup_queue_emitted": (
                queue_feedback_replan_followup_queue is not None
            ),
            "queue_feedback_replan_followup_queue_blockers": list(
                dict.fromkeys(str(item) for item in queue_feedback_replan_followup_blockers)
            ),
            "queue_feedback_replan_followup_policy": queue_feedback_replan_followup_policy,
            "queue_feedback_replan_followup_execution_policy": (
                queue_feedback_replan_followup_policy
            ),
            "queue_feedback_replan_followup_policy_enabled": (
                queue_feedback_replan_followup_policy_enabled
            ),
            "queue_feedback_replan_followup_policy_blockers": list(
                dict.fromkeys(
                    str(item) for item in queue_feedback_replan_followup_policy_blockers
                )
            ),
            "queue_feedback_replan_followup_execution_requested": (
                queue_feedback_replan_followup_execution_requested
            ),
            "queue_feedback_replan_followup_executed": (
                queue_feedback_replan_followup_executed
            ),
            "queue_feedback_replan_followup_execution_success": (
                None
                if queue_feedback_replan_followup_execution is None
                else bool(queue_feedback_replan_followup_execution.get("success"))
            ),
            "queue_feedback_replan_followup_execution": (
                None
                if queue_feedback_replan_followup_execution is None
                else dict(queue_feedback_replan_followup_execution)
            ),
            "queue_feedback_replan_followup_state_path": (
                None
                if queue_feedback_replan_followup_execution is None
                else queue_feedback_replan_followup_execution.get("state_path")
            ),
            "queue_feedback_replan_followup_action_functional_path": (
                None
                if queue_feedback_replan_followup_execution is None
                else queue_feedback_replan_followup_execution.get(
                    "action_functional_output_path"
                )
            ),
            "queue_observation_path": _display_path(queue_observation_path),
            "queue_observation_recovery_plan_path": _display_path(
                queue_observation_recovery_plan_path
            ),
            "queue_observation_recovery_plan": dict(queue_observation_recovery_plan),
            "queue_observation_recovery_required": (
                queue_observation_recovery_plan.get("recovery_required") is True
            ),
            "queue_observation_maintenance_recommended": (
                queue_observation_recovery_plan.get("maintenance_recommended") is True
            ),
            "exact_readiness_handoff_count": len(exact_readiness_handoffs),
            "exact_readiness_handoff_paths": [dict(item) for item in exact_readiness_handoffs],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )


def _post_recovery_feedback_replan_payload(
    args: argparse.Namespace,
    *,
    recovery_execution: Mapping[str, Any] | None,
    run_dir: Path,
    commands: list[CommandResult],
    queue: Mapping[str, Any],
    summary_path: Path,
    plan_path: Path,
    execution_queue: Path,
    state_path: Path,
    queue_performance_summary_path: Path,
    queue_performance_summary: Mapping[str, Any],
    runtime_identity_path: Path | None,
    cache_identity_path: Path | None,
    generated_runtime_identity: bool,
    generated_cache_identity: bool,
    feedback_lane_id: str,
    feedback_policy_activation: str,
    exact_readiness_handoffs: Sequence[Mapping[str, Any]],
    staircase_artifacts: dict[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    blockers: list[str] = []
    source_observation = (
        recovery_execution.get("source_observation_after")
        if isinstance(recovery_execution, Mapping)
        else None
    )
    if not isinstance(recovery_execution, Mapping):
        blockers.append("queue_observation_recovery_execution_missing")
    elif recovery_execution.get("success") is not True:
        blockers.append("queue_observation_recovery_execution_not_successful")
    if not isinstance(source_observation, Mapping):
        blockers.append("source_observation_after_missing")

    post_observation_path = run_dir / "queue_observation_after_recovery.json"
    post_recovery_plan_path = (
        run_dir / "queue_observation_recovery_plan_after_recovery.json"
    )
    post_request_path = run_dir / "queue_feedback_replan_request_after_recovery.json"
    post_followup_queue_path = (
        run_dir / "queue_feedback_replan_followup_queue_after_recovery.json"
    )
    post_policy_path = run_dir / "queue_feedback_replan_policy_after_recovery.json"
    post_continuation_queue_path = (
        run_dir / "queue_feedback_replan_continuation_queue_after_recovery.json"
    )

    post_recovery_plan: dict[str, Any] | None = None
    post_request: dict[str, Any] | None = None
    post_followup_queue: dict[str, Any] | None = None
    post_followup_blockers: list[str] = []
    post_followup_policy_blockers: list[str] = []
    post_followup_execution: dict[str, Any] | None = None
    post_followup_execution_returncode = 0
    post_followup_execution_attempted = False
    post_policy: dict[str, Any] | None = None
    post_continuation_queue: dict[str, Any] | None = None
    post_continuation_blockers: list[str] = []
    post_continuation_staircase_artifacts: dict[str, Any] | None = None
    post_followup_state_path = (
        _resolve(args.queue_feedback_replan_followup_state)
        if args.queue_feedback_replan_followup_state is not None
        else run_dir / "queue_feedback_replan_followup.sqlite"
    )
    post_continuation_state_path = (
        run_dir / "queue_feedback_replan_continuation_queue_after_recovery.sqlite"
    )

    if not blockers and isinstance(source_observation, Mapping):
        _write_json(post_observation_path, source_observation)
        post_recovery_plan = build_queue_observation_recovery_plan(
            source_observation,
            queue_path=_display_path(execution_queue),
            state_path=_display_path(state_path),
            reason="post-recovery materializer queue observation replan",
        )
        _write_json(post_recovery_plan_path, post_recovery_plan)
        post_request = _queue_feedback_replan_request_payload(
            args,
            summary_path=summary_path,
            plan_path=plan_path,
            queue_performance_summary_path=queue_performance_summary_path,
            queue_performance_summary=queue_performance_summary,
            runtime_identity_path=runtime_identity_path,
            cache_identity_path=cache_identity_path,
            generated_runtime_identity=generated_runtime_identity,
            generated_cache_identity=generated_cache_identity,
            run_dir=run_dir,
            execution_queue=execution_queue,
            state_path=state_path,
            queue_observation_path=post_observation_path,
            queue_observation_recovery_plan_path=post_recovery_plan_path,
            queue_observation_recovery_plan=post_recovery_plan,
            action_functional_output_stem=(
                "inverse_steganalysis_action_functional.after_recovery"
            ),
        )
        post_followup_queue, post_followup_blockers = (
            _queue_feedback_replan_followup_queue_payload(
                args,
                queue_feedback_replan_request=post_request,
                queue_feedback_replan_request_path=post_request_path,
                run_dir=run_dir,
                execution_queue=execution_queue,
                state_path=state_path,
                source_queue=queue,
            )
        )
        post_request.update(
            {
                "queue_owned_followup_queue_path": (
                    _display_path(post_followup_queue_path)
                    if post_followup_queue is not None
                    else None
                ),
                "queue_owned_followup_queue_emitted": post_followup_queue is not None,
                "queue_owned_followup_queue_blockers": post_followup_blockers,
                "source": "post_queue_observation_recovery",
                "allowed_use": "post_recovery_next_inverse_action_replan_input_only",
            }
        )
        _write_json(post_request_path, post_request)

        if post_followup_queue is not None:
            _write_json(post_followup_queue_path, post_followup_queue)
            if staircase_artifacts is not None:
                post_staircase = _build_queue_feedback_staircase_artifacts(
                    args,
                    run_dir=run_dir,
                    parent_queue=queue,
                    parent_queue_path=execution_queue,
                    parent_state_path=state_path,
                    child_queue=post_followup_queue,
                    child_queue_path=post_followup_queue_path,
                    artifact_stem="post_recovery_queue_feedback_replan_followup",
                )
                staircase_artifacts["post_recovery_feedback_child"] = post_staircase
            if args.queue_feedback_replan_followup_policy_local_autopilot:
                post_followup_policy_blockers = (
                    _queue_feedback_replan_followup_local_autopolicy_blockers(
                        post_followup_queue,
                        run_dir=run_dir,
                    )
                )
            should_execute_followup = (
                args.execute_queue_feedback_replan_followup
                or (
                    args.queue_feedback_replan_followup_policy_local_autopilot
                    and not post_followup_policy_blockers
                )
            )
            if should_execute_followup:
                post_followup_execution_attempted = True
                (
                    post_followup_execution,
                    post_followup_commands,
                    post_followup_execution_returncode,
                ) = _queue_feedback_replan_followup_execution_payload(
                    args,
                    child_queue=post_followup_queue,
                    child_queue_path=post_followup_queue_path,
                    run_dir=run_dir,
                    activation_policy=feedback_policy_activation,
                )
                commands.extend(post_followup_commands)
            elif args.queue_feedback_replan_followup_policy_local_autopilot:
                post_followup_execution = (
                    _queue_feedback_replan_followup_execution_refusal_payload(
                        blockers=(
                            post_followup_policy_blockers
                            or ["queue_feedback_replan_followup_policy_not_enabled"]
                        ),
                        child_queue_path=post_followup_queue_path,
                        activation_policy=feedback_policy_activation,
                    )
                )
        elif args.execute_queue_feedback_replan_followup:
            post_followup_execution_returncode = 2
            post_followup_execution = (
                _queue_feedback_replan_followup_execution_refusal_payload(
                    blockers=(
                        post_followup_blockers
                        or ["queue_feedback_replan_followup_queue_not_emitted"]
                    ),
                    child_queue_path=post_followup_queue_path,
                    activation_policy=feedback_policy_activation,
                )
            )
        elif args.queue_feedback_replan_followup_policy_local_autopilot:
            post_followup_execution = (
                _queue_feedback_replan_followup_execution_refusal_payload(
                    blockers=(
                        post_followup_blockers
                        or ["queue_feedback_replan_followup_queue_not_emitted"]
                    ),
                    child_queue_path=post_followup_queue_path,
                    activation_policy=feedback_policy_activation,
                )
            )

        post_run_summary = _post_recovery_feedback_replan_run_summary(
            run_dir=run_dir,
            summary_path=summary_path,
            plan_path=plan_path,
            execution_queue=execution_queue,
            state_path=state_path,
            queue=queue,
            queue_performance_summary_path=queue_performance_summary_path,
            queue_performance_summary=queue_performance_summary,
            queue_feedback_replan_request=post_request,
            queue_feedback_replan_followup_queue_path=post_followup_queue_path,
            queue_feedback_replan_followup_queue=post_followup_queue,
            queue_feedback_replan_followup_blockers=post_followup_blockers,
            queue_feedback_replan_followup_policy=feedback_policy_activation,
            queue_feedback_replan_followup_policy_enabled=bool(
                args.queue_feedback_replan_followup_policy_local_autopilot
            ),
            queue_feedback_replan_followup_policy_blockers=post_followup_policy_blockers,
            queue_feedback_replan_followup_execution_requested=bool(
                args.execute_queue_feedback_replan_followup
                or args.queue_feedback_replan_followup_policy_local_autopilot
            ),
            queue_feedback_replan_followup_executed=post_followup_execution_attempted,
            queue_feedback_replan_followup_execution=post_followup_execution,
            queue_observation_path=post_observation_path,
            queue_observation_recovery_plan_path=post_recovery_plan_path,
            queue_observation_recovery_plan=post_recovery_plan,
            exact_readiness_handoffs=exact_readiness_handoffs,
        )
        post_policy = build_queue_feedback_replan_policy(
            post_run_summary,
            feedback_followup_queue=post_followup_queue,
            source_run_path=_display_path(summary_path),
            iteration_index=args.queue_feedback_replan_policy_iteration,
            max_iterations=args.queue_feedback_replan_policy_max_iterations,
        )
        _write_json(post_policy_path, post_policy)
        post_continuation_queue, post_continuation_blockers = (
            build_queue_feedback_replan_continuation_queue(
                post_policy,
                lane_id=feedback_lane_id,
                source_policy_path=_display_path(post_policy_path),
            )
        )
        if post_continuation_queue is not None:
            _write_json(post_continuation_queue_path, post_continuation_queue)
            post_continuation_staircase_artifacts = (
                _build_queue_feedback_staircase_artifacts(
                    args,
                    run_dir=run_dir,
                    parent_queue=queue,
                    parent_queue_path=execution_queue,
                    parent_state_path=state_path,
                    child_queue=post_continuation_queue,
                    child_queue_path=post_continuation_queue_path,
                    artifact_stem="post_recovery_queue_feedback_replan_continuation",
                )
            )
            if staircase_artifacts is not None:
                staircase_artifacts["post_recovery_feedback_continuation_child"] = (
                    post_continuation_staircase_artifacts
                )

    post_followup_execution_success = (
        post_followup_execution is not None
        and post_followup_execution.get("success") is True
    )
    post_continuation_emitted = post_continuation_queue is not None
    payload = _false_authority_payload(
        {
            "schema": POST_RECOVERY_FEEDBACK_REPLAN_SCHEMA,
            "attempted": bool(recovery_execution),
            "triggered": not blockers,
            "artifacts_emitted": post_policy is not None,
            "success": (
                post_policy is not None
                and post_policy.get("decision")
                == "run_next_materializer_campaign_iteration"
                and post_followup_execution_success
                and post_continuation_emitted
                and post_followup_execution_returncode == 0
            ),
            "blockers": list(dict.fromkeys(blockers)),
            "source_recovery_execution_success": (
                isinstance(recovery_execution, Mapping)
                and recovery_execution.get("success") is True
            ),
            "queue_observation_path": (
                _display_path(post_observation_path) if post_observation_path.exists() else None
            ),
            "queue_observation_recovery_plan_path": (
                _display_path(post_recovery_plan_path)
                if post_recovery_plan_path.exists()
                else None
            ),
            "queue_observation_recovery_plan": post_recovery_plan,
            "queue_feedback_replan_request_path": (
                _display_path(post_request_path) if post_request_path.exists() else None
            ),
            "queue_feedback_replan_request": post_request,
            "queue_feedback_replan_followup_queue_path": (
                _display_path(post_followup_queue_path)
                if post_followup_queue is not None
                else None
            ),
            "queue_feedback_replan_followup_state_path": (
                _display_path(post_followup_state_path)
                if post_followup_queue is not None
                else None
            ),
            "queue_feedback_replan_followup_queue_emitted": (
                post_followup_queue is not None
            ),
            "queue_feedback_replan_followup_queue_blockers": post_followup_blockers,
            "queue_feedback_replan_followup_policy": feedback_policy_activation,
            "queue_feedback_replan_followup_policy_enabled": bool(
                args.queue_feedback_replan_followup_policy_local_autopilot
            ),
            "queue_feedback_replan_followup_policy_blockers": (
                post_followup_policy_blockers
            ),
            "queue_feedback_replan_followup_execution_requested": bool(
                args.execute_queue_feedback_replan_followup
                or args.queue_feedback_replan_followup_policy_local_autopilot
            ),
            "queue_feedback_replan_followup_executed": post_followup_execution_attempted,
            "queue_feedback_replan_followup_execution_success": (
                None
                if post_followup_execution is None
                else post_followup_execution_success
            ),
            "queue_feedback_replan_followup_execution": post_followup_execution,
            "queue_feedback_replan_followup_action_functional_path": (
                None
                if post_followup_execution is None
                else post_followup_execution.get("action_functional_output_path")
            ),
            "queue_feedback_replan_policy_path": (
                _display_path(post_policy_path) if post_policy_path.exists() else None
            ),
            "queue_feedback_replan_policy": post_policy,
            "queue_feedback_replan_policy_decision": (
                None if post_policy is None else post_policy.get("decision")
            ),
            "queue_feedback_replan_policy_should_continue": (
                post_policy is not None
                and post_policy.get("should_continue_feedback_loop") is True
            ),
            "queue_feedback_replan_continuation_queue_path": (
                _display_path(post_continuation_queue_path)
                if post_continuation_queue is not None
                else None
            ),
            "queue_feedback_replan_continuation_queue_state_path": (
                _display_path(post_continuation_state_path)
                if post_continuation_queue is not None
                else None
            ),
            "queue_feedback_replan_continuation_queue_emitted": (
                post_continuation_emitted
            ),
            "queue_feedback_replan_continuation_queue_blockers": (
                post_continuation_blockers
            ),
            "queue_feedback_replan_continuation_staircase_artifacts": (
                post_continuation_staircase_artifacts
            ),
            "allowed_use": "post_recovery_local_feedback_replan_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
        }
    )
    return payload, post_followup_execution_returncode


def _git_head_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _queue_runtime_identity_payload(
    *,
    run_dir: Path,
    execution_queue: Path,
    state_path: Path,
    queue: Mapping[str, Any],
    runtime_policy_output_path: Path | None,
    runtime_policy_applied_queue_path: Path | None,
) -> dict[str, Any]:
    runtime_file_sha256 = {
        relative_path: _sha256_file(REPO_ROOT / relative_path)
        for relative_path in QUEUE_RUNTIME_IDENTITY_FILE_PATHS
        if (REPO_ROOT / relative_path).is_file()
    }
    runtime_tree_sha256 = _sha256_json(runtime_file_sha256)
    return _false_authority_payload(
        {
            "schema": "byte_shaving_materializer_campaign_queue_runtime_identity.v1",
            "identity_kind": "local_runner_queue_runtime",
            "runner_tool": "tools/run_byte_shaving_materializer_campaign.py",
            "repo_root": REPO_ROOT.as_posix(),
            "git_head_sha": _git_head_sha(),
            "runtime_tree_sha256": runtime_tree_sha256,
            "runtime_content_tree_sha256": runtime_tree_sha256,
            "runtime_file_sha256": runtime_file_sha256,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": sys.platform,
            "run_dir": _display_path(run_dir),
            "queue_id": queue.get("queue_id"),
            "queue_path": _display_path(execution_queue),
            "queue_state_path": _display_path(state_path),
            "runtime_policy_path": (
                None if runtime_policy_output_path is None else _display_path(runtime_policy_output_path)
            ),
            "runtime_policy_applied_queue_path": (
                None if runtime_policy_applied_queue_path is None else _display_path(runtime_policy_applied_queue_path)
            ),
            "allowed_use": "queue_performance_observation_runtime_identity_only",
            "forbidden_use": "score_claim_or_promotion_or_dispatch_authority",
        }
    )


def _queue_cache_identity_payload(
    *,
    run_dir: Path,
    execution_queue: Path,
    state_path: Path,
    queue_performance_summary_path: Path,
    queue: Mapping[str, Any],
) -> dict[str, Any]:
    return _false_authority_payload(
        {
            "schema": "byte_shaving_materializer_campaign_queue_cache_identity.v1",
            "identity_kind": "local_runner_queue_cache",
            "runner_tool": "tools/run_byte_shaving_materializer_campaign.py",
            "run_dir": _display_path(run_dir),
            "queue_id": queue.get("queue_id"),
            "queue_path": _display_path(execution_queue),
            "queue_definition_sha256": _sha256_file(execution_queue),
            "queue_state_path": _display_path(state_path),
            "queue_state_exists": state_path.exists(),
            "queue_state_sha256": _optional_sha256_file(state_path),
            "queue_performance_summary_path": _display_path(queue_performance_summary_path),
            "queue_performance_summary_sha256": _sha256_file(queue_performance_summary_path),
            "cache_sha256": _sha256_file(queue_performance_summary_path),
            "allowed_use": "queue_performance_observation_cache_identity_only",
            "forbidden_use": "score_claim_or_promotion_or_dispatch_authority",
        }
    )


def _handoff_dir_from_path(path: Path) -> Path | None:
    parts = path.parts
    if "exact_eval_handoff" not in parts:
        return None
    index = parts.index("exact_eval_handoff")
    return Path(*parts[: index + 1])


def _queue_exact_readiness_handoff_dirs(queue: Mapping[str, Any]) -> list[Path]:
    out: list[Path] = []
    for experiment in queue.get("experiments", []):
        if not isinstance(experiment, Mapping):
            continue
        for step in experiment.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            telemetry = step.get("telemetry")
            if not isinstance(telemetry, Mapping):
                continue
            for key in ("artifact_paths", "pullback_artifact_paths"):
                raw_paths = telemetry.get(key)
                if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, str | bytes):
                    continue
                for raw_path in raw_paths:
                    handoff_dir = _handoff_dir_from_path(_resolve(str(raw_path)))
                    if handoff_dir is not None:
                        out.append(handoff_dir)
    return out


def _exact_readiness_handoff_paths(
    *,
    run_dir: Path,
    queue: Mapping[str, Any],
) -> list[dict[str, Any]]:
    handoff_dirs = [
        *[path for path in run_dir.rglob("exact_eval_handoff") if path.is_dir()],
        *_queue_exact_readiness_handoff_dirs(queue),
    ]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for handoff_dir in sorted(handoff_dirs, key=lambda item: item.as_posix()):
        key = handoff_dir.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        source_queue = handoff_dir / "source_queue.json"
        harvest_report = handoff_dir / "harvest_report.json"
        readiness_dir = handoff_dir / "exact_readiness"
        bridge_report = handoff_dir / "exact_readiness_bridge_report.json"
        dispatch_plan = handoff_dir / "dispatch_plan.json"
        dispatch_queue = handoff_dir / "dispatch_queue.json"
        records.append(
            {
                "handoff_dir": _display_path(handoff_dir),
                "source_queue_path": _display_path(source_queue),
                "harvest_report_path": _display_path(harvest_report),
                "readiness_dir_path": _display_path(readiness_dir),
                "exact_readiness_bridge_report_path": _display_path(bridge_report),
                "dispatch_plan_path": _display_path(dispatch_plan),
                "dispatch_queue_path": _display_path(dispatch_queue),
                "source_queue_exists": source_queue.exists(),
                "harvest_report_exists": harvest_report.exists(),
                "exact_readiness_bridge_report_exists": bridge_report.exists(),
                "dispatch_plan_exists": dispatch_plan.exists(),
                "dispatch_queue_exists": dispatch_queue.exists(),
                "readiness_dir_exists": readiness_dir.exists(),
            }
        )
    return records


def _response_update_placeholder_payload(
    *,
    summary_path: Path,
    queue_performance_summary_path: Path,
    queue_feedback_replan_request_path: Path,
    queue_feedback_replan_followup_queue_path: Path | None,
    queue_feedback_replan_followup_blockers: Sequence[str],
    queue_feedback_replan_blockers: Sequence[str],
    next_action_functional_command_hint: Sequence[str],
    run_dir: Path,
    execution_queue: Path,
    state_path: Path,
    runtime_policy_output_path: Path | None,
    runtime_policy_applied_queue_path: Path | None,
    exact_readiness_handoffs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    performance_arg = _display_path(queue_performance_summary_path)
    return _false_authority_payload(
        {
            "schema": RESPONSE_UPDATE_PLACEHOLDER_SCHEMA,
            "source_run_path": _display_path(summary_path),
            "run_dir": _display_path(run_dir),
            "queue_path": _display_path(execution_queue),
            "queue_state_path": _display_path(state_path),
            "runtime_policy_path": (
                None
                if runtime_policy_output_path is None
                else _display_path(runtime_policy_output_path)
            ),
            "runtime_policy_applied_queue_path": (
                None
                if runtime_policy_applied_queue_path is None
                else _display_path(runtime_policy_applied_queue_path)
            ),
            "queue_performance_summary_path": performance_arg,
            "queue_feedback_replan_request_path": _display_path(queue_feedback_replan_request_path),
            "queue_feedback_replan_followup_queue_path": (
                None
                if queue_feedback_replan_followup_queue_path is None
                else _display_path(queue_feedback_replan_followup_queue_path)
            ),
            "queue_feedback_replan_followup_queue_emitted": (
                queue_feedback_replan_followup_queue_path is not None
            ),
            "queue_feedback_replan_followup_queue_blockers": [
                str(item) for item in queue_feedback_replan_followup_blockers
            ],
            "exact_readiness_handoff_count": len(exact_readiness_handoffs),
            "exact_readiness_handoff_paths": [dict(item) for item in exact_readiness_handoffs],
            "response_update_applied": False,
            "replan_required": True,
            "next_run_hint": [
                "--queue-performance-summary",
                performance_arg,
            ],
            "next_action_functional_command_hint": list(next_action_functional_command_hint),
            "not_scorer_response_dataset": True,
            "consumer_contract": {
                "next_real_artifact_schema": "scorer_response_dataset.v1",
                "next_real_artifact_flag": "--scorer-response",
                "placeholder_must_not_be_used_as_scorer_response": True,
            },
            "allowed_use": "next_inverse_action_replan_input_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
            "blockers": [
                "response_update_not_applied",
                "placeholder_not_scorer_response_dataset",
                "requires_next_action_functional_replan",
                "requires_exact_auth_eval_before_score_claim",
                *queue_feedback_replan_blockers,
                *queue_feedback_replan_followup_blockers,
            ],
        }
    )


def _parse_resource_concurrency(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit("--materializer-resource-concurrency entries must be KIND=LIMIT")
        key, value = raw.split("=", 1)
        if not key.strip():
            raise SystemExit("--materializer-resource-concurrency KIND must be non-empty")
        try:
            limit = int(value)
        except ValueError as exc:
            raise SystemExit(f"--materializer-resource-concurrency has non-integer limit: {raw!r}") from exc
        if limit < 1:
            raise SystemExit(f"--materializer-resource-concurrency limit must be >= 1: {raw!r}")
        out.extend(["--materializer-resource-concurrency", f"{key.strip()}={limit}"])
    return out


def _parse_remote_repo_roots(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        machine, sep, root = raw.partition("=")
        if not sep or not machine.strip() or not root.strip():
            raise SystemExit("--staircase-ssh-remote-repo-root entries must be MACHINE_ID=PATH")
        out.extend(["--remote-repo-root", f"{machine.strip()}={root.strip()}"])
    return out


def _parse_artifact_path_maps(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        local, sep, remote = raw.partition("=")
        if not sep or not local.strip() or not remote.strip():
            raise SystemExit("--staircase-ssh-artifact-path-map entries must be LOCAL_PREFIX=REMOTE_PREFIX")
        out.extend(["--artifact-path-map", f"{local.strip()}={remote.strip()}"])
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help=(
            "JSON run config with schema "
            f"{RUN_CONFIG_SCHEMA}; CLI scalar flags override config fields and "
            "repeatable source flags append to configured lists"
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help=(
            "existing byte_shaving_campaign_plan.v1. If omitted, the runner "
            "builds one from high-level scorer-response/action sources first."
        ),
    )
    parser.add_argument("--campaign-id", default="byte_shaving_acquisition_campaign")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--scorer-response", action="append", default=[])
    parser.add_argument("--inverse-scorer-surface", action="append", default=[])
    parser.add_argument("--byte-shaving-signal-surface", action="append", default=[])
    parser.add_argument("--byte-shaving-campaign-plan", action="append", default=[])
    parser.add_argument("--mlx-acquisition-batch", action="append", default=[])
    parser.add_argument("--mlx-effective-spend-triage-selection", action="append", default=[])
    parser.add_argument(
        "--mlx-effective-spend-triage-selection-mode",
        choices=("batch", "direct"),
        default="batch",
        help=(
            "batch builds mlx_acquisition_batch.v1 artifacts inside this run "
            "before action-functional construction; direct preserves the older "
            "row-level bridge"
        ),
    )
    parser.add_argument("--mlx-acquisition-set-size", type=int, default=1)
    parser.add_argument("--mlx-acquisition-limit", type=int, default=None)
    parser.add_argument("--atom", action="append", default=[])
    parser.add_argument("--observation", action="append", default=[])
    parser.add_argument("--materializer-chain-manifest", action="append", default=[])
    parser.add_argument("--exact-auth-calibration-packet", action="append", default=[])
    parser.add_argument(
        "--exact-auth-calibration-packet-root",
        action="append",
        default=[],
        help=(
            "file or directory root to scan for paired tac_result_review_packet_v1 "
            "contest_cpu/contest_cuda calibration packets for feedback replans"
        ),
    )
    parser.add_argument(
        "--exact-auth-calibration-packet-glob",
        default="**/*result_review*.json",
        help="glob used under each calibration packet root",
    )
    parser.add_argument("--exact-auth-calibration-candidate-id", default=None)
    parser.add_argument("--queue-performance-summary", action="append", default=[])
    parser.add_argument("--queue-performance-runtime-identity", type=Path, default=None)
    parser.add_argument("--queue-performance-cache-identity", type=Path, default=None)
    parser.add_argument("--queue-performance-candidate-map", type=Path, default=None)
    parser.add_argument(
        "--queue-performance-axis",
        default="[local-queue-performance advisory]",
    )
    parser.add_argument("--resource-kind", default="local_mlx")
    parser.add_argument("--elapsed-seconds", type=float, default=None)
    parser.add_argument("--artifact-bytes", type=int, default=None)
    parser.add_argument("--total-byte-budget", type=int, default=None)
    parser.add_argument("--lambda-rate", type=float, default=None)
    parser.add_argument("--inverse-scorer-max-units", type=int, default=32)
    parser.add_argument("--inverse-scorer-null-delta-epsilon", type=float, default=1e-6)
    parser.add_argument("--inverse-scorer-fragile-delta-threshold", type=float, default=0.0)
    parser.add_argument(
        "--inverse-scorer-allow-native-mlx-window-objective",
        action="store_true",
    )
    parser.add_argument("--campaign-plan-max-k", type=int, default=None)
    parser.add_argument("--materializer-contexts", type=Path, default=None)
    parser.add_argument(
        "--materializer-artifact-map",
        type=Path,
        default=None,
        help=("artifact/custody hints used to auto-generate materializer contexts inside the campaign run directory"),
    )
    parser.add_argument(
        "--materializer-context-default-output-root",
        type=Path,
        default=None,
        help=("default output root for generated materializer contexts; defaults to RUN_DIR/materializer_outputs"),
    )
    parser.add_argument("--materializer-contexts-fail-if-blocked", action="store_true")
    parser.add_argument(
        "--inverse-scorer-action-functional",
        type=Path,
        default=None,
        help=(
            "inverse_steganalysis_discrete_action_functional.v1 used when "
            "auto-generating an inverse-scorer materializer artifact map from "
            "an existing --plan; defaults to the action functional generated "
            "from high-level sources"
        ),
    )
    parser.add_argument(
        "--inverse-scorer-candidate-archive-template",
        type=Path,
        default=None,
        help=("candidate archive template used to auto-generate an inverse-scorer materializer artifact map"),
    )
    parser.add_argument(
        "--inverse-scorer-raw-contest-video-digest",
        default=None,
        help=("raw contest video digest recorded in generated inverse-scorer materializer contexts"),
    )
    parser.add_argument("--inverse-scorer-atom-id", action="append", default=[])
    parser.add_argument("--inverse-scorer-selected-limit", type=int, default=None)
    parser.add_argument("--inverse-scorer-chain-output-dir", type=Path, default=None)
    parser.add_argument("--inverse-scorer-source-inflate-output-dir", type=Path, default=None)
    parser.add_argument("--inverse-scorer-candidate-inflate-output-dir", type=Path, default=None)
    parser.add_argument("--inverse-scorer-inflate-runtime-dir", type=Path, default=None)
    parser.add_argument("--inverse-scorer-source-archive-for-parity", type=Path, default=None)
    parser.add_argument("--inverse-scorer-inflate-work-dir", type=Path, default=None)
    parser.add_argument("--inverse-scorer-runtime-consumption-proof", type=Path, default=None)
    parser.add_argument("--inverse-scorer-min-free-bytes", type=int, default=None)
    parser.add_argument("--inverse-scorer-inflate-timeout-seconds", type=int, default=None)
    parser.add_argument("--inverse-scorer-descriptor-probe-only", action="store_true")
    parser.add_argument("--inverse-scorer-fail-if-receiver-blocked", action="store_true")
    parser.add_argument("--inverse-scorer-fail-if-inflate-parity-blocked", action="store_true")
    parser.add_argument("--inverse-scorer-keep-inflate-work-dir", action="store_true")
    parser.add_argument("--archive-section-archive-path", type=Path, default=None)
    parser.add_argument("--archive-section-manifest", type=Path, default=None)
    parser.add_argument("--archive-section-name", action="append", default=[])
    parser.add_argument("--archive-section-output-archive", type=Path, default=None)
    parser.add_argument("--archive-section-output-manifest", type=Path, default=None)
    parser.add_argument("--archive-section-brotli-quality", action="append", default=[])
    parser.add_argument("--archive-section-runtime-consumption-proof", type=Path, default=None)
    parser.add_argument("--archive-section-min-free-bytes", type=int, default=None)
    parser.add_argument("--archive-section-allow-size-regression", action="store_true")
    parser.add_argument("--archive-section-allow-overwrite", action="store_true")
    parser.add_argument("--packet-member-archive-path", type=Path, default=None)
    parser.add_argument(
        "--packet-member-target-kind",
        choices=(
            PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
            PACKET_MEMBER_MERGE_TARGET_KIND,
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            RENDERER_PAYLOAD_DFL1_TARGET_KIND,
        ),
        default=PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    )
    parser.add_argument("--packet-member-manifest", type=Path, default=None)
    parser.add_argument("--packet-member-name", default=None)
    parser.add_argument("--packet-member-names", action="append", default=[])
    parser.add_argument("--packet-member-payload-member-name", default=None)
    parser.add_argument("--packet-member-output-archive", type=Path, default=None)
    parser.add_argument("--packet-member-output-manifest", type=Path, default=None)
    parser.add_argument("--packet-member-zip-compression-method", action="append", default=[])
    parser.add_argument("--packet-member-zip-compresslevel", action="append", default=[])
    parser.add_argument("--packet-member-runtime-consumption-proof", type=Path, default=None)
    parser.add_argument("--packet-member-min-free-bytes", type=int, default=None)
    parser.add_argument("--packet-member-allow-size-regression", action="store_true")
    parser.add_argument("--packet-member-allow-overwrite", action="store_true")
    parser.add_argument(
        "--renderer-payload-dfl1-source-runtime-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--renderer-payload-dfl1-candidate-runtime-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--renderer-payload-dfl1-full-frame-file-list",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--renderer-payload-dfl1-full-frame-file-list-entry",
        action="append",
        default=[],
    )
    parser.add_argument("--renderer-payload-dfl1-expected-full-frame-file-list-sha256")
    parser.add_argument(
        "--renderer-payload-dfl1-expected-full-frame-entry-count",
        type=int,
    )
    parser.add_argument("--renderer-payload-dfl1-full-frame-file-list-source")
    parser.add_argument(
        "--renderer-payload-dfl1-inflate-parity-output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--tensor-factorize-archive-path", type=Path, default=None)
    parser.add_argument("--tensor-factorize-manifest", type=Path, default=None)
    parser.add_argument("--tensor-factorize-contract", type=Path, default=None)
    parser.add_argument("--tensor-factorize-rank", type=int, default=None)
    parser.add_argument("--tensor-factorize-output-archive", type=Path, default=None)
    parser.add_argument("--tensor-factorize-output-manifest", type=Path, default=None)
    parser.add_argument("--tensor-factorize-runtime-consumption-proof", type=Path, default=None)
    parser.add_argument("--tensor-factorize-min-free-bytes", type=int, default=None)
    parser.add_argument("--tensor-factorize-allow-size-regression", action="store_true")
    parser.add_argument("--tensor-factorize-allow-overwrite", action="store_true")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="artifact directory; defaults to .omx/research/byte_shaving_materializer_campaign_<UTC>",
    )
    parser.add_argument(
        "--queue-state",
        type=Path,
        default=None,
        help=(
            "SQLite state path for the generated materializer execution queue; "
            "defaults to the run-dir artifact state path"
        ),
    )
    parser.add_argument(
        "--queue-state-rationale",
        default=None,
        help=(
            "Specific rationale passed to experiment_queue run-worker when a "
            "custom --queue-state points at a noncanonical execution-state path"
        ),
    )
    parser.add_argument(
        "--derive-runtime-policy",
        action="store_true",
        help="derive scheduler_runtime_policy.v1 from the queue state before local execution",
    )
    parser.add_argument(
        "--apply-runtime-policy",
        action="store_true",
        help=("derive scheduler_runtime_policy.v1 and run the local worker against a policy-applied queue definition"),
    )
    parser.add_argument("--runtime-policy-output", type=Path, default=None)
    parser.add_argument("--runtime-policy-applied-queue-output", type=Path, default=None)
    parser.add_argument("--runtime-policy-cpu-count", type=int, default=None)
    parser.add_argument("--runtime-policy-timeout-multiplier", type=float, default=3.0)
    parser.add_argument("--runtime-policy-min-timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--runtime-policy-max-timeout-seconds",
        type=int,
        default=24 * 60 * 60,
    )
    parser.add_argument(
        "--runtime-policy-no-apply-concurrency",
        action="store_true",
        help="when applying a runtime policy, leave queue max_concurrency unchanged",
    )
    parser.add_argument(
        "--runtime-policy-apply-timeouts",
        action="store_true",
        help=(
            "also apply timeout recommendations; off by default to avoid definition-hash churn in existing queue state"
        ),
    )
    parser.add_argument("--queue-id", default="byte_shaving_materializer_local_proof_chain")
    parser.add_argument("--lane-id", default=None)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--materializer-execution-limit", type=int, default=None)
    parser.add_argument("--materializer-execution-timeout-seconds", type=int, default=0)
    parser.add_argument(
        "--materializer-resource-concurrency",
        action="append",
        default=[],
        metavar="KIND=LIMIT",
    )
    parser.add_argument("--local-cpu-concurrency", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--idle-sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_LOCAL_QUEUE_POLL_INTERVAL_SECONDS,
        help=(
            "active subprocess poll interval for the local materializer worker; "
            "lower than experiment_queue's generic default because campaign "
            "materializer steps are often short local micro-commands"
        ),
    )
    parser.add_argument("--max-idle-cycles", type=int, default=1)
    parser.add_argument("--tail-lines", type=int, default=20)
    parser.add_argument(
        "--execute-queue-feedback-replan-followup",
        action="store_true",
        help=(
            "after emitting the paused feedback child queue, explicitly resume "
            "and run bounded local-only steps to build the next action functional"
        ),
    )
    parser.add_argument(
        "--queue-feedback-replan-followup-policy-local-autopilot",
        action="store_true",
        help=(
            "allow the queue-owned local autopolicy to resume the paused feedback child "
            "queue when strict local-only and false-authority checks pass"
        ),
    )
    parser.add_argument("--queue-feedback-replan-followup-state", type=Path, default=None)
    parser.add_argument(
        "--queue-feedback-replan-followup-state-rationale",
        default=None,
        help="rationale used when the feedback child queue runs against a noncanonical state path",
    )
    parser.add_argument("--queue-feedback-replan-followup-activation-reason", default=None)
    parser.add_argument("--queue-feedback-replan-followup-max-steps", type=int, default=1)
    parser.add_argument("--queue-feedback-replan-followup-max-parallel", type=int, default=1)
    parser.add_argument("--queue-feedback-replan-followup-idle-sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--queue-feedback-replan-followup-poll-interval-seconds",
        type=float,
        default=DEFAULT_LOCAL_QUEUE_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument("--queue-feedback-replan-followup-max-idle-cycles", type=int, default=1)
    parser.add_argument("--queue-feedback-replan-followup-tail-lines", type=int, default=20)
    parser.add_argument(
        "--execute-queue-observation-recovery",
        action="store_true",
        help=(
            "after emitting the paused queue-observation recovery queue, explicitly "
            "resume and run bounded local-only recovery actions"
        ),
    )
    parser.add_argument(
        "--queue-observation-recovery-policy-local-autopilot",
        action="store_true",
        help=(
            "allow the queue-owned local autopolicy to resume the paused recovery "
            "queue when strict local-only and false-authority checks pass"
        ),
    )
    parser.add_argument("--queue-observation-recovery-state", type=Path, default=None)
    parser.add_argument(
        "--queue-observation-recovery-state-rationale",
        default=None,
        help="rationale used when the recovery child queue runs against a noncanonical state path",
    )
    parser.add_argument("--queue-observation-recovery-activation-reason", default=None)
    parser.add_argument("--queue-observation-recovery-max-steps", type=int, default=8)
    parser.add_argument("--queue-observation-recovery-max-parallel", type=int, default=1)
    parser.add_argument("--queue-observation-recovery-idle-sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--queue-observation-recovery-poll-interval-seconds",
        type=float,
        default=DEFAULT_LOCAL_QUEUE_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument("--queue-observation-recovery-max-idle-cycles", type=int, default=1)
    parser.add_argument("--queue-observation-recovery-tail-lines", type=int, default=20)
    parser.add_argument("--queue-feedback-replan-policy-iteration", type=int, default=0)
    parser.add_argument("--queue-feedback-replan-policy-max-iterations", type=int, default=3)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--include-storage-preflight", action="store_true")
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--storage-tier", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--storage-workload-subdir", default=None)
    parser.add_argument("--storage-expected-workload-root", default=None)
    parser.add_argument("--storage-reserve-free-gb", type=float, default=40.0)
    parser.add_argument("--storage-expected-bytes", type=int, default=0)
    parser.add_argument("--proactive-cleanup-root", action="append", default=[])
    parser.add_argument("--proactive-cleanup-action", choices=("move", "delete"), default="move")
    parser.add_argument("--proactive-cleanup-min-bytes", default="1")
    parser.add_argument("--proactive-cleanup-cold-store-root", action="append", default=[])
    parser.add_argument("--proactive-cleanup-cold-store-reserve-gb", type=float, default=40.0)
    parser.add_argument("--exact-readiness-require-ready", action="store_true")
    parser.add_argument(
        "--require-renderer-payload-dfl1-parity-followup",
        action="store_true",
        help=(
            "forward a fail-closed DFL1 shell-inflate parity requirement to the "
            "materializer execution queue builder"
        ),
    )
    parser.add_argument("--exact-eval-dispatch-require-authorized", action="store_true")
    parser.add_argument("--exact-eval-dispatch-provider", choices=("lightning", "vastai"), default="lightning")
    parser.add_argument("--exact-eval-dispatch-label-prefix", default="materializer_exact_eval")
    parser.add_argument("--exact-eval-dispatch-estimated-cost-per-dispatch", type=float, default=0.30)
    parser.add_argument("--exact-eval-dispatch-max-total-cost", type=float, default=5.00)
    parser.add_argument(
        "--emit-staircase-plan",
        action="store_true",
        help="emit staircase_dag.v1 and staircase_dispatch_plan.v1 from the generated queue",
    )
    parser.add_argument(
        "--staircase-resource-pool",
        action="append",
        default=[],
        metavar="SPEC",
        help="resource pool spec accepted by tools/plan_staircase_dag.py",
    )
    parser.add_argument("--staircase-max-nodes", type=int, default=None)
    parser.add_argument("--staircase-allow-cloud", action="store_true")
    parser.add_argument("--staircase-diversity-bucket-limit", type=int, default=None)
    parser.add_argument(
        "--staircase-ssh-dry-run",
        action="store_true",
        help="run tools/run_staircase_ssh_executor.py without --execute against the emitted plan",
    )
    parser.add_argument(
        "--staircase-ssh-execute",
        action="store_true",
        help="run tools/run_staircase_ssh_executor.py --execute against the emitted plan",
    )
    parser.add_argument("--staircase-ssh-max-steps", type=int, default=1)
    parser.add_argument("--staircase-ssh-machine-id", default=None)
    parser.add_argument(
        "--staircase-ssh-remote-repo-root",
        action="append",
        default=[],
        metavar="MACHINE=PATH",
    )
    parser.add_argument(
        "--staircase-ssh-require-artifact-mobility",
        action="store_true",
        help="require SSH input push plus artifact pullback/shared-storage visibility",
    )
    parser.add_argument(
        "--staircase-ssh-artifact-path-map",
        action="append",
        default=[],
        metavar="LOCAL_PREFIX=REMOTE_PREFIX",
    )
    parser.add_argument("--staircase-ssh-artifact-shared-path-rationale", default=None)
    parser.add_argument("--staircase-ssh-allow-dirty-remote-git", action="store_true")
    parser.add_argument("--staircase-ssh-dirty-remote-git-rationale", default=None)
    parser.add_argument("--staircase-ssh-rsync-binary", default="rsync")
    parser.add_argument("--staircase-ssh-artifact-pull-timeout-seconds", type=int, default=300)
    parser.add_argument("--staircase-ssh-allow-future-executor", action="store_true")
    pre_args, _unknown = parser.parse_known_args(argv)
    if pre_args.run_config is not None:
        run_config_path = _resolve(pre_args.run_config)
        defaults = _normalize_config_defaults(
            parser,
            _load_run_config(run_config_path),
            config_path=run_config_path,
        )
        defaults["run_config"] = pre_args.run_config
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def _action_source_count(args: argparse.Namespace) -> int:
    return sum(
        len(getattr(args, name))
        for name in (
            "scorer_response",
            "inverse_scorer_surface",
            "byte_shaving_signal_surface",
            "byte_shaving_campaign_plan",
            "mlx_acquisition_batch",
            "mlx_effective_spend_triage_selection",
            "atom",
            "observation",
            "exact_auth_calibration_packet",
            "queue_performance_summary",
        )
    )


def _require_plan_path(args: argparse.Namespace, plan_path: Path | None = None) -> Path:
    if plan_path is not None:
        return plan_path
    if args.plan is None:
        raise SystemExit(
            "provide --plan or high-level action sources such as "
            "--scorer-response, --inverse-scorer-surface, "
            "--byte-shaving-signal-surface, --byte-shaving-campaign-plan, "
            "--mlx-acquisition-batch, --mlx-effective-spend-triage-selection, "
            "or --atom"
        )
    return _resolve(args.plan)


def _append_path_args(command: list[str], flag: str, values: list[str | Path]) -> None:
    for raw in values:
        command.extend([flag, _display_path(_resolve(raw))])


def _mlx_acquisition_batch_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"mlx_acquisition_batch_{index:04d}.json"


def _build_mlx_acquisition_batch_commands(
    args: argparse.Namespace,
    *,
    run_dir: Path,
) -> list[tuple[Path, list[str]]]:
    commands: list[tuple[Path, list[str]]] = []
    if args.mlx_effective_spend_triage_selection_mode != "batch":
        return commands
    for index, raw_path in enumerate(args.mlx_effective_spend_triage_selection):
        output = _mlx_acquisition_batch_path(run_dir, index)
        command = [
            sys.executable,
            "tools/build_mlx_acquisition_batch.py",
            "--mlx-effective-spend-triage-selection",
            _display_path(_resolve(raw_path)),
            "--output",
            _display_path(output),
            "--repo-root",
            REPO_ROOT.as_posix(),
            "--set-size",
            str(args.mlx_acquisition_set_size),
        ]
        if args.mlx_acquisition_limit is not None:
            command.extend(["--limit", str(args.mlx_acquisition_limit)])
        if args.overwrite_output:
            command.append("--allow-overwrite")
        commands.append((output, command))
    return commands


def _build_action_functional_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    generated_mlx_acquisition_batches: list[Path] | None = None,
) -> list[str]:
    action_path = run_dir / "inverse_steganalysis_action_functional.json"
    md_path = run_dir / "inverse_steganalysis_action_functional.md"
    command = [
        sys.executable,
        QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
        "--output",
        _display_path(action_path),
        "--md-out",
        _display_path(md_path),
        "--repo-root",
        REPO_ROOT.as_posix(),
        "--resource-kind",
        str(args.resource_kind),
    ]
    _append_path_args(command, "--scorer-response", args.scorer_response)
    _append_path_args(command, "--inverse-scorer-surface", args.inverse_scorer_surface)
    _append_path_args(
        command,
        "--byte-shaving-signal-surface",
        args.byte_shaving_signal_surface,
    )
    _append_path_args(
        command,
        "--byte-shaving-campaign-plan",
        args.byte_shaving_campaign_plan,
    )
    _append_path_args(command, "--mlx-acquisition-batch", args.mlx_acquisition_batch)
    _append_path_args(
        command,
        "--mlx-acquisition-batch",
        list(generated_mlx_acquisition_batches or []),
    )
    if args.mlx_effective_spend_triage_selection_mode == "direct":
        _append_path_args(
            command,
            "--mlx-effective-spend-triage-selection",
            args.mlx_effective_spend_triage_selection,
        )
    _append_path_args(command, "--atom", args.atom)
    _append_path_args(command, "--observation", args.observation)
    _append_path_args(
        command,
        "--materializer-chain-manifest",
        args.materializer_chain_manifest,
    )
    _append_path_args(
        command,
        "--exact-auth-calibration-packet",
        args.exact_auth_calibration_packet,
    )
    _append_path_args(
        command,
        "--queue-performance-summary",
        args.queue_performance_summary,
    )
    if args.candidate_id:
        command.extend(["--candidate-id", str(args.candidate_id)])
    if args.exact_auth_calibration_candidate_id:
        command.extend(
            [
                "--exact-auth-calibration-candidate-id",
                str(args.exact_auth_calibration_candidate_id),
            ]
        )
    if args.queue_performance_runtime_identity is not None:
        command.extend(
            [
                "--queue-performance-runtime-identity",
                _display_path(_resolve(args.queue_performance_runtime_identity)),
            ]
        )
    if args.queue_performance_cache_identity is not None:
        command.extend(
            [
                "--queue-performance-cache-identity",
                _display_path(_resolve(args.queue_performance_cache_identity)),
            ]
        )
    if args.queue_performance_candidate_map is not None:
        command.extend(
            [
                "--queue-performance-candidate-map",
                _display_path(_resolve(args.queue_performance_candidate_map)),
            ]
        )
    if args.queue_performance_axis:
        command.extend(["--queue-performance-axis", str(args.queue_performance_axis)])
    if args.elapsed_seconds is not None:
        command.extend(["--elapsed-seconds", str(args.elapsed_seconds)])
    if args.artifact_bytes is not None:
        command.extend(["--artifact-bytes", str(args.artifact_bytes)])
    if args.total_byte_budget is not None:
        command.extend(["--total-byte-budget", str(args.total_byte_budget)])
    if args.lambda_rate is not None:
        command.extend(["--lambda-rate", str(args.lambda_rate)])
    command.extend(
        [
            "--inverse-scorer-max-units",
            str(args.inverse_scorer_max_units),
            "--inverse-scorer-null-delta-epsilon",
            str(args.inverse_scorer_null_delta_epsilon),
            "--inverse-scorer-fragile-delta-threshold",
            str(args.inverse_scorer_fragile_delta_threshold),
        ]
    )
    if args.inverse_scorer_allow_native_mlx_window_objective:
        command.append("--inverse-scorer-allow-native-mlx-window-objective")
    return command


def _build_campaign_plan_command(
    args: argparse.Namespace,
    *,
    action_functional_path: Path,
    run_dir: Path,
) -> list[str]:
    plan_path = run_dir / "byte_shaving_campaign_plan.json"
    md_path = run_dir / "byte_shaving_campaign_plan.md"
    command = [
        sys.executable,
        "tools/plan_byte_shaving_campaign.py",
        "--source",
        _display_path(action_functional_path),
        "--from-inverse-action-functional",
        "--campaign-id",
        str(args.campaign_id),
        "--output",
        _display_path(plan_path),
        "--md-out",
        _display_path(md_path),
        "--repo-root",
        REPO_ROOT.as_posix(),
    ]
    if args.campaign_plan_max_k is not None:
        command.extend(["--max-k", str(args.campaign_plan_max_k)])
    return command


def _build_queue_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    plan_path: Path | None = None,
    generated_materializer_artifact_map: Path | None = None,
) -> list[str]:
    if args.include_storage_preflight:
        try:
            validate_scheduler_storage_preflight_config(
                proactive_cleanup_execute=True,
                proactive_cleanup_action=args.proactive_cleanup_action,
                proactive_cleanup_cold_store_roots=tuple(args.proactive_cleanup_cold_store_root),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    materialization = run_dir / "materialization.json"
    portfolio = run_dir / "portfolio.json"
    action_summary = run_dir / "action_summary.json"
    backlog = run_dir / "materializer_backlog.json"
    work_queue = run_dir / "materializer_work_queue.json"
    execution_queue = run_dir / "materializer_execution_queue.json"
    command = [
        sys.executable,
        "tools/build_byte_shaving_campaign_queue.py",
        "--repo-root",
        REPO_ROOT.as_posix(),
        "--plan",
        _display_path(_require_plan_path(args, plan_path)),
        "--materialization-out",
        _display_path(materialization),
        "--portfolio-out",
        _display_path(portfolio),
        "--action-summary-out",
        _display_path(action_summary),
        "--materializer-backlog-out",
        _display_path(backlog),
        "--materializer-work-queue-out",
        _display_path(work_queue),
        "--materializer-execution-queue-out",
        _display_path(execution_queue),
        "--materializer-execution-queue-id",
        args.queue_id,
        "--materializer-execution-state",
        _display_path(
            _resolve(args.queue_state)
            if args.queue_state is not None
            else run_dir / "materializer_execution_queue.sqlite"
        ),
        "--candidate-limit",
        str(args.candidate_limit),
        "--local-cpu-concurrency",
        str(args.local_cpu_concurrency),
        "--results-root",
        str(args.results_root),
        "--include-materializer-exact-readiness-followup",
        "--materializer-exact-eval-dispatch-provider",
        args.exact_eval_dispatch_provider,
        "--materializer-exact-eval-dispatch-label-prefix",
        args.exact_eval_dispatch_label_prefix,
        "--materializer-exact-eval-dispatch-estimated-cost-per-dispatch",
        str(args.exact_eval_dispatch_estimated_cost_per_dispatch),
        "--materializer-exact-eval-dispatch-max-total-cost",
        str(args.exact_eval_dispatch_max_total_cost),
    ]
    artifact_map = (
        _resolve(args.materializer_artifact_map)
        if args.materializer_artifact_map is not None
        else generated_materializer_artifact_map
    )
    if args.materializer_contexts is not None:
        command.extend(["--materializer-contexts", _display_path(_resolve(args.materializer_contexts))])
    else:
        default_context_output_root = _default_materializer_context_output_root(
            args,
            run_dir=run_dir,
        )
        command.extend(
            [
                "--materializer-contexts-out",
                _display_path(run_dir / "materializer_contexts.json"),
                "--materializer-context-default-output-root",
                _display_path(default_context_output_root),
            ]
        )
        if artifact_map is not None:
            command.extend(
                [
                    "--materializer-artifact-map",
                    _display_path(artifact_map),
                ]
            )
        if args.materializer_contexts_fail_if_blocked:
            command.append("--materializer-contexts-fail-if-blocked")
    if args.lane_id:
        command.extend(["--materializer-execution-lane-id", str(args.lane_id)])
    if args.materializer_execution_limit is not None:
        command.extend(["--materializer-execution-limit", str(args.materializer_execution_limit)])
    if args.materializer_execution_timeout_seconds:
        command.extend(
            [
                "--materializer-execution-timeout-seconds",
                str(args.materializer_execution_timeout_seconds),
            ]
        )
    command.extend(_parse_resource_concurrency(args.materializer_resource_concurrency))
    if args.overwrite_output:
        command.append("--overwrite-output")
    if args.exact_readiness_require_ready:
        command.append("--materializer-exact-readiness-followup-require-ready")
    if (
        args.require_renderer_payload_dfl1_parity_followup
        or (
            generated_materializer_artifact_map is not None
            and args.packet_member_target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND
        )
    ):
        command.append("--require-renderer-payload-dfl1-parity-followup")
    if args.exact_eval_dispatch_require_authorized:
        command.append("--materializer-exact-eval-dispatch-require-authorized")
    if args.include_storage_preflight:
        command.append("--include-materializer-scheduler-preflight")
        command.extend(["--materializer-scheduler-storage-reserve-free-gb", str(args.storage_reserve_free_gb)])
        command.extend(["--materializer-scheduler-storage-expected-bytes", str(args.storage_expected_bytes)])
        command.extend(["--materializer-scheduler-proactive-cleanup-action", args.proactive_cleanup_action])
        command.extend(["--materializer-scheduler-proactive-cleanup-min-bytes", str(args.proactive_cleanup_min_bytes)])
        command.extend(
            [
                "--materializer-scheduler-proactive-cleanup-cold-store-reserve-gb",
                str(args.proactive_cleanup_cold_store_reserve_gb),
            ]
        )
        command.append("--materializer-scheduler-proactive-cleanup-execute")
        if args.storage_workload_subdir:
            command.extend(["--materializer-scheduler-storage-workload-subdir", args.storage_workload_subdir])
        if args.storage_expected_workload_root:
            command.extend(
                [
                    "--materializer-scheduler-storage-expected-workload-root",
                    args.storage_expected_workload_root,
                ]
            )
        for tier in args.storage_tier:
            command.extend(["--materializer-scheduler-storage-tier", tier])
        for root in args.proactive_cleanup_root:
            command.extend(["--materializer-scheduler-proactive-cleanup-root", root])
        for root in args.proactive_cleanup_cold_store_root:
            command.extend(["--materializer-scheduler-proactive-cleanup-cold-store-root", root])
    return command


def _experiment_queue_command(
    *,
    execution_queue: Path,
    state_path: Path,
    subcommand: Sequence[str],
) -> list[str]:
    return [
        sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        _display_path(execution_queue),
        "--state",
        _display_path(state_path),
        *subcommand,
    ]


def _queue_state_rationale_for_worker(
    args: argparse.Namespace,
    *,
    run_dir: Path,
) -> str | None:
    if args.queue_state_rationale:
        return str(args.queue_state_rationale)
    if args.queue_state is not None:
        return None
    return (
        "runner-owned isolated materializer campaign state under run-dir; "
        f"queue_id={args.queue_id}; "
        f"run_dir={_display_path(run_dir)}; "
        "execution is bound to the generated queue and state paths emitted by this run"
    )


def _runtime_policy_command(
    args: argparse.Namespace,
    *,
    execution_queue: Path,
    state_path: Path,
    run_dir: Path,
) -> tuple[Path, Path | None, list[str]]:
    policy_output = (
        _resolve(args.runtime_policy_output)
        if args.runtime_policy_output
        else (run_dir / "scheduler_runtime_policy.json")
    )
    applied_queue_output = (
        _resolve(args.runtime_policy_applied_queue_output)
        if args.runtime_policy_applied_queue_output
        else (run_dir / "materializer_execution_queue.runtime_policy.json")
        if args.apply_runtime_policy
        else None
    )
    command = _experiment_queue_command(
        execution_queue=execution_queue,
        state_path=state_path,
        subcommand=[
            "runtime-policy",
            "--timeout-multiplier",
            str(args.runtime_policy_timeout_multiplier),
            "--min-timeout-seconds",
            str(args.runtime_policy_min_timeout_seconds),
            "--max-timeout-seconds",
            str(args.runtime_policy_max_timeout_seconds),
            "--policy-output",
            _display_path(policy_output),
        ],
    )
    if args.runtime_policy_cpu_count is not None:
        command.extend(["--cpu-count", str(args.runtime_policy_cpu_count)])
    if applied_queue_output is not None:
        command.extend(["--applied-queue-output", _display_path(applied_queue_output)])
        if args.runtime_policy_no_apply_concurrency:
            command.append("--no-apply-concurrency")
        if not args.runtime_policy_apply_timeouts:
            command.append("--no-apply-timeouts")
    return policy_output, applied_queue_output, command


def _inverse_scorer_auto_artifact_map_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            args.inverse_scorer_action_functional is not None,
            args.inverse_scorer_candidate_archive_template is not None,
            bool(str(args.inverse_scorer_raw_contest_video_digest or "").strip()),
            bool(args.inverse_scorer_atom_id),
            args.inverse_scorer_selected_limit is not None,
            args.inverse_scorer_chain_output_dir is not None,
            args.inverse_scorer_source_inflate_output_dir is not None,
            args.inverse_scorer_candidate_inflate_output_dir is not None,
            args.inverse_scorer_inflate_runtime_dir is not None,
            args.inverse_scorer_source_archive_for_parity is not None,
            args.inverse_scorer_inflate_work_dir is not None,
            args.inverse_scorer_runtime_consumption_proof is not None,
            args.inverse_scorer_min_free_bytes is not None,
            args.inverse_scorer_inflate_timeout_seconds is not None,
            args.inverse_scorer_descriptor_probe_only,
            args.inverse_scorer_fail_if_receiver_blocked,
            args.inverse_scorer_fail_if_inflate_parity_blocked,
            args.inverse_scorer_keep_inflate_work_dir,
        )
    )


def _archive_section_auto_artifact_map_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            args.archive_section_archive_path is not None,
            args.archive_section_manifest is not None,
            bool(args.archive_section_name),
            args.archive_section_output_archive is not None,
            args.archive_section_output_manifest is not None,
            bool(args.archive_section_brotli_quality),
            args.archive_section_runtime_consumption_proof is not None,
            args.archive_section_min_free_bytes is not None,
            args.archive_section_allow_size_regression,
            args.archive_section_allow_overwrite,
        )
    )


def _packet_member_auto_artifact_map_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            args.packet_member_archive_path is not None,
            args.packet_member_target_kind != PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
            args.packet_member_manifest is not None,
            bool(str(args.packet_member_name or "").strip()),
            bool(args.packet_member_names),
            bool(str(args.packet_member_payload_member_name or "").strip()),
            args.packet_member_output_archive is not None,
            args.packet_member_output_manifest is not None,
            bool(args.packet_member_zip_compression_method),
            bool(args.packet_member_zip_compresslevel),
            args.packet_member_runtime_consumption_proof is not None,
            args.packet_member_min_free_bytes is not None,
            args.packet_member_allow_size_regression,
            args.packet_member_allow_overwrite,
            _renderer_payload_dfl1_parity_artifact_args_used(args),
        )
    )


def _tensor_factorize_auto_artifact_map_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            args.tensor_factorize_archive_path is not None,
            args.tensor_factorize_manifest is not None,
            args.tensor_factorize_contract is not None,
            args.tensor_factorize_rank is not None,
            args.tensor_factorize_output_archive is not None,
            args.tensor_factorize_output_manifest is not None,
            args.tensor_factorize_runtime_consumption_proof is not None,
            args.tensor_factorize_min_free_bytes is not None,
            args.tensor_factorize_allow_size_regression,
            args.tensor_factorize_allow_overwrite,
        )
    )


def _auto_artifact_map_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            _inverse_scorer_auto_artifact_map_requested(args),
            _archive_section_auto_artifact_map_requested(args),
            _packet_member_auto_artifact_map_requested(args),
            _tensor_factorize_auto_artifact_map_requested(args),
        )
    )


def _positive_optional_int(value: int | None, *, flag: str) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise SystemExit(f"{flag} must be >= 1")
    return int(value)


def _path_value(path: Path | None) -> str | None:
    if path is None:
        return None
    return _display_path(_resolve(path))


def _canonical_sha256_arg(value: str | None, *, flag: str) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise SystemExit(f"{flag} must be a 64-character hexadecimal SHA-256")
    return text


def _add_path_context_value(context: dict[str, Any], key: str, path: Path | None) -> None:
    value = _path_value(path)
    if value is not None:
        context[key] = value


def _add_positive_int_context_value(
    context: dict[str, Any],
    key: str,
    value: int | None,
    *,
    flag: str,
) -> None:
    parsed = _positive_optional_int(value, flag=flag)
    if parsed is not None:
        context[key] = parsed


def _add_string_list_context_value(
    context: dict[str, Any],
    key: str,
    values: Sequence[Any],
) -> None:
    parsed = [str(value).strip() for value in values if str(value).strip()]
    if parsed:
        context[key] = parsed


def _ordered_dfl1_file_list_entries(values: Sequence[Any]) -> tuple[str, ...]:
    entries = tuple(str(value).strip() for value in values if str(value).strip())
    if not entries:
        raise SystemExit(
            "--renderer-payload-dfl1-full-frame-file-list-entry must include "
            "at least one non-empty entry"
        )
    return entries


def _file_list_identity_from_entries(entries: Sequence[Any]) -> tuple[str, int]:
    ordered = _ordered_dfl1_file_list_entries(entries)
    payload = ("\n".join(ordered) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(ordered)


def _renderer_payload_dfl1_full_frame_identity(
    args: argparse.Namespace,
) -> tuple[str | None, int | None, str | None]:
    expected_sha = _canonical_sha256_arg(
        args.renderer_payload_dfl1_expected_full_frame_file_list_sha256,
        flag="--renderer-payload-dfl1-expected-full-frame-file-list-sha256",
    )
    expected_count = _positive_optional_int(
        args.renderer_payload_dfl1_expected_full_frame_entry_count,
        flag="--renderer-payload-dfl1-expected-full-frame-entry-count",
    )
    source = str(args.renderer_payload_dfl1_full_frame_file_list_source or "").strip()
    source = source or None
    measured_sha: str | None = None
    measured_count: int | None = None
    measured_source: str | None = None
    if args.renderer_payload_dfl1_full_frame_file_list is not None:
        file_list_path = _resolve(args.renderer_payload_dfl1_full_frame_file_list)
        if not file_list_path.is_file():
            raise SystemExit(
                "--renderer-payload-dfl1-full-frame-file-list does not exist: "
                f"{file_list_path}"
            )
        measured_sha = _sha256_file(file_list_path)
        measured_count = len(
            _ordered_dfl1_file_list_entries(
                file_list_path.read_text(encoding="utf-8").splitlines()
            )
        )
        measured_source = _display_path(file_list_path)
    elif args.renderer_payload_dfl1_full_frame_file_list_entry:
        measured_sha, measured_count = _file_list_identity_from_entries(
            args.renderer_payload_dfl1_full_frame_file_list_entry
        )
        measured_source = "inline_renderer_payload_dfl1_full_frame_file_list_entries"

    if expected_sha is not None and measured_sha is not None and expected_sha != measured_sha:
        raise SystemExit(
            "--renderer-payload-dfl1-expected-full-frame-file-list-sha256 "
            "does not match the provided full-frame file list"
        )
    if (
        expected_count is not None
        and measured_count is not None
        and expected_count != measured_count
    ):
        raise SystemExit(
            "--renderer-payload-dfl1-expected-full-frame-entry-count does not "
            "match the provided full-frame file list"
        )
    return (
        expected_sha or measured_sha,
        expected_count or measured_count,
        source or measured_source,
    )


def _renderer_payload_dfl1_parity_artifact_args_used(
    args: argparse.Namespace,
) -> bool:
    return any(
        (
            args.renderer_payload_dfl1_source_runtime_dir is not None,
            args.renderer_payload_dfl1_candidate_runtime_dir is not None,
            args.renderer_payload_dfl1_full_frame_file_list is not None,
            bool(args.renderer_payload_dfl1_full_frame_file_list_entry),
            bool(
                str(
                    args.renderer_payload_dfl1_expected_full_frame_file_list_sha256
                    or ""
                ).strip()
            ),
            args.renderer_payload_dfl1_expected_full_frame_entry_count is not None,
            bool(
                str(
                    args.renderer_payload_dfl1_full_frame_file_list_source or ""
                ).strip()
            ),
            args.renderer_payload_dfl1_inflate_parity_output_dir is not None,
        )
    )


def _build_inverse_scorer_artifact_context(
    args: argparse.Namespace,
    *,
    action_functional_path: Path | None,
) -> dict[str, Any]:
    template = _path_value(args.inverse_scorer_candidate_archive_template)
    raw_digest = str(args.inverse_scorer_raw_contest_video_digest or "").strip()
    if template is None:
        raise SystemExit(
            "--inverse-scorer-candidate-archive-template is required to "
            "auto-generate inverse-scorer materializer contexts"
        )
    if not raw_digest:
        raise SystemExit(
            "--inverse-scorer-raw-contest-video-digest is required to "
            "auto-generate inverse-scorer materializer contexts"
        )
    if action_functional_path is None:
        raise SystemExit(
            "--inverse-scorer-action-functional is required with --plan when "
            "auto-generating inverse-scorer materializer contexts"
        )

    context: dict[str, Any] = {
        "candidate_archive_template": template,
        "inverse_action_functional": _display_path(action_functional_path),
        "raw_contest_video_digest": raw_digest,
        **FALSE_AUTHORITY,
    }
    for key, value in (
        ("output_dir", _path_value(args.inverse_scorer_chain_output_dir)),
        ("chain_output_dir", _path_value(args.inverse_scorer_chain_output_dir)),
        (
            "source_inflate_output_dir",
            _path_value(args.inverse_scorer_source_inflate_output_dir),
        ),
        (
            "candidate_inflate_output_dir",
            _path_value(args.inverse_scorer_candidate_inflate_output_dir),
        ),
        ("inflate_runtime_dir", _path_value(args.inverse_scorer_inflate_runtime_dir)),
        (
            "source_archive_for_parity",
            _path_value(args.inverse_scorer_source_archive_for_parity),
        ),
        ("inflate_work_dir", _path_value(args.inverse_scorer_inflate_work_dir)),
        (
            "runtime_consumption_proof",
            _path_value(args.inverse_scorer_runtime_consumption_proof),
        ),
    ):
        if value is not None:
            context[key] = value
    if args.inverse_scorer_atom_id:
        context["atom_ids"] = [str(atom_id).strip() for atom_id in args.inverse_scorer_atom_id if str(atom_id).strip()]
    for key, value, flag in (
        ("selected_limit", args.inverse_scorer_selected_limit, "--inverse-scorer-selected-limit"),
        ("min_free_bytes", args.inverse_scorer_min_free_bytes, "--inverse-scorer-min-free-bytes"),
        (
            "inflate_timeout_seconds",
            args.inverse_scorer_inflate_timeout_seconds,
            "--inverse-scorer-inflate-timeout-seconds",
        ),
    ):
        parsed = _positive_optional_int(value, flag=flag)
        if parsed is not None:
            context[key] = parsed
    for key, enabled in (
        ("descriptor_probe_only", args.inverse_scorer_descriptor_probe_only),
        ("fail_if_receiver_blocked", args.inverse_scorer_fail_if_receiver_blocked),
        (
            "fail_if_inflate_parity_blocked",
            args.inverse_scorer_fail_if_inflate_parity_blocked,
        ),
        ("keep_inflate_work_dir", args.inverse_scorer_keep_inflate_work_dir),
    ):
        if enabled:
            context[key] = True
    return context


def _build_archive_section_artifact_context(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = _path_value(args.archive_section_archive_path)
    section_manifest = _path_value(args.archive_section_manifest)
    if archive_path is None:
        raise SystemExit(
            "--archive-section-archive-path is required to auto-generate archive-section materializer contexts"
        )
    if section_manifest is None:
        raise SystemExit(
            "--archive-section-manifest is required to auto-generate archive-section materializer contexts"
        )
    context: dict[str, Any] = {
        "archive_path": archive_path,
        "source_archive": archive_path,
        "section_manifest": section_manifest,
        **FALSE_AUTHORITY,
    }
    _add_path_context_value(context, "output_archive", args.archive_section_output_archive)
    _add_path_context_value(context, "json_out", args.archive_section_output_manifest)
    _add_path_context_value(
        context,
        "runtime_consumption_proof",
        args.archive_section_runtime_consumption_proof,
    )
    _add_string_list_context_value(
        context,
        "target_sections",
        args.archive_section_name,
    )
    _add_string_list_context_value(
        context,
        "brotli_quality",
        args.archive_section_brotli_quality,
    )
    _add_positive_int_context_value(
        context,
        "min_free_bytes",
        args.archive_section_min_free_bytes,
        flag="--archive-section-min-free-bytes",
    )
    if args.archive_section_allow_size_regression:
        context["allow_size_regression"] = True
    if args.archive_section_allow_overwrite:
        context["allow_overwrite"] = True
    return context


def _build_packet_member_artifact_context(args: argparse.Namespace) -> dict[str, Any]:
    dfl1_parity_args_used = _renderer_payload_dfl1_parity_artifact_args_used(args)
    if (
        dfl1_parity_args_used
        and args.packet_member_target_kind != RENDERER_PAYLOAD_DFL1_TARGET_KIND
    ):
        raise SystemExit(
            "renderer-payload DFL1 parity flags require "
            f"--packet-member-target-kind {RENDERER_PAYLOAD_DFL1_TARGET_KIND}"
        )
    archive_path = _path_value(args.packet_member_archive_path)
    if archive_path is None:
        raise SystemExit(
            "--packet-member-archive-path is required to auto-generate packet-member materializer contexts"
        )
    context: dict[str, Any] = {
        "archive_path": archive_path,
        "source_archive": archive_path,
        **FALSE_AUTHORITY,
    }
    _add_path_context_value(context, "packet_member_manifest", args.packet_member_manifest)
    member_name = str(args.packet_member_name or "").strip()
    if member_name:
        context["member_name"] = member_name
    _add_string_list_context_value(
        context,
        "member_names",
        args.packet_member_names,
    )
    payload_member_name = str(args.packet_member_payload_member_name or "").strip()
    if payload_member_name:
        context["payload_member_name"] = payload_member_name
    _add_path_context_value(context, "output_archive", args.packet_member_output_archive)
    _add_path_context_value(context, "json_out", args.packet_member_output_manifest)
    _add_path_context_value(
        context,
        "runtime_consumption_proof",
        args.packet_member_runtime_consumption_proof,
    )
    _add_string_list_context_value(
        context,
        "zip_compression_method",
        args.packet_member_zip_compression_method,
    )
    _add_string_list_context_value(
        context,
        "zip_compresslevel",
        args.packet_member_zip_compresslevel,
    )
    _add_positive_int_context_value(
        context,
        "min_free_bytes",
        args.packet_member_min_free_bytes,
        flag="--packet-member-min-free-bytes",
    )
    if args.packet_member_allow_size_regression:
        context["allow_size_regression"] = True
    if args.packet_member_allow_overwrite:
        context["allow_overwrite"] = True
    if args.packet_member_target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        expected_sha, expected_count, file_list_source = (
            _renderer_payload_dfl1_full_frame_identity(args)
        )
        _add_path_context_value(
            context,
            "renderer_payload_dfl1_source_runtime_dir",
            args.renderer_payload_dfl1_source_runtime_dir,
        )
        _add_path_context_value(
            context,
            "renderer_payload_dfl1_candidate_runtime_dir",
            args.renderer_payload_dfl1_candidate_runtime_dir,
        )
        _add_path_context_value(
            context,
            "renderer_payload_dfl1_full_frame_file_list",
            args.renderer_payload_dfl1_full_frame_file_list,
        )
        _add_string_list_context_value(
            context,
            "renderer_payload_dfl1_full_frame_file_list_entries",
            args.renderer_payload_dfl1_full_frame_file_list_entry,
        )
        if expected_sha:
            context["renderer_payload_dfl1_expected_full_frame_file_list_sha256"] = (
                expected_sha
            )
        if expected_count is not None:
            context["renderer_payload_dfl1_expected_full_frame_entry_count"] = (
                expected_count
            )
        if file_list_source is not None:
            context["renderer_payload_dfl1_full_frame_file_list_source"] = (
                file_list_source
            )
        _add_path_context_value(
            context,
            "renderer_payload_dfl1_inflate_parity_output_dir",
            args.renderer_payload_dfl1_inflate_parity_output_dir,
        )
    return context


def _build_tensor_factorize_artifact_context(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = _path_value(args.tensor_factorize_archive_path)
    tensor_manifest = _path_value(args.tensor_factorize_manifest)
    if archive_path is None:
        raise SystemExit(
            "--tensor-factorize-archive-path is required to auto-generate tensor-factorize materializer contexts"
        )
    if tensor_manifest is None:
        raise SystemExit(
            "--tensor-factorize-manifest is required to auto-generate tensor-factorize materializer contexts"
        )
    if args.tensor_factorize_contract is None and args.tensor_factorize_rank is None:
        raise SystemExit(
            "--tensor-factorize-contract or --tensor-factorize-rank is required "
            "to auto-generate tensor-factorize materializer contexts"
        )
    context: dict[str, Any] = {
        "archive_path": archive_path,
        "source_archive": archive_path,
        "tensor_manifest": tensor_manifest,
        **FALSE_AUTHORITY,
    }
    _add_path_context_value(
        context,
        "factorization_contract",
        args.tensor_factorize_contract,
    )
    _add_positive_int_context_value(
        context,
        "rank",
        args.tensor_factorize_rank,
        flag="--tensor-factorize-rank",
    )
    _add_path_context_value(context, "output_archive", args.tensor_factorize_output_archive)
    _add_path_context_value(context, "json_out", args.tensor_factorize_output_manifest)
    _add_path_context_value(
        context,
        "runtime_consumption_proof",
        args.tensor_factorize_runtime_consumption_proof,
    )
    _add_positive_int_context_value(
        context,
        "min_free_bytes",
        args.tensor_factorize_min_free_bytes,
        flag="--tensor-factorize-min-free-bytes",
    )
    if args.tensor_factorize_allow_size_regression:
        context["allow_size_regression"] = True
    if args.tensor_factorize_allow_overwrite:
        context["allow_overwrite"] = True
    return context


def _build_generated_materializer_artifact_map_payload(
    args: argparse.Namespace,
    *,
    action_functional_path: Path | None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if _inverse_scorer_auto_artifact_map_requested(args):
        artifacts[INVERSE_SCORER_CELL_TARGET_KIND] = _build_inverse_scorer_artifact_context(
            args,
            action_functional_path=action_functional_path,
        )
    if _archive_section_auto_artifact_map_requested(args):
        artifacts[ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND] = _build_archive_section_artifact_context(args)
    if _packet_member_auto_artifact_map_requested(args):
        artifacts[args.packet_member_target_kind] = _build_packet_member_artifact_context(args)
    if _tensor_factorize_auto_artifact_map_requested(args):
        artifacts[TENSOR_FACTORIZE_TARGET_KIND] = _build_tensor_factorize_artifact_context(args)
    return {
        "schema": "final_byte_artifact_map.generated.v1",
        "generated_by": "tools/run_byte_shaving_materializer_campaign.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": artifacts,
        **FALSE_AUTHORITY,
    }


def _write_generated_materializer_artifact_map(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    generated_action_functional_path: Path | None,
) -> Path | None:
    if args.materializer_artifact_map is not None or args.materializer_contexts is not None:
        return None
    if not _auto_artifact_map_requested(args):
        return None
    action_functional_path = (
        _resolve(args.inverse_scorer_action_functional)
        if args.inverse_scorer_action_functional is not None
        else generated_action_functional_path
    )
    output = run_dir / "materializer_artifact_map.json"
    _write_json(
        output,
        _build_generated_materializer_artifact_map_payload(
            args,
            action_functional_path=action_functional_path,
        ),
    )
    return output


def _build_staircase_artifacts(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    execution_queue: Path,
    state_path: Path,
    queue: dict[str, Any],
    dependent_queue_refs: Sequence[Mapping[str, Any]] | None = None,
    suffix: str = "",
) -> dict[str, Any]:
    resource_pools = [parse_resource_pool_spec(spec) for spec in args.staircase_resource_pool] or None
    status_map = experiment_queue_status_map(
        queue_path=execution_queue,
        repo_root=REPO_ROOT,
        state_path=state_path,
    )
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id=f"{queue['queue_id']}_staircase",
        source_path=_display_path(execution_queue),
        resource_pools=resource_pools,
        dependent_queue_refs=dependent_queue_refs,
    )
    plan = plan_staircase_dispatch(
        dag,
        status_map=status_map,
        max_nodes=args.staircase_max_nodes,
        allow_cloud=args.staircase_allow_cloud,
        diversity_bucket_limit=args.staircase_diversity_bucket_limit,
    )
    dag_path = run_dir / f"staircase_dag{suffix}.json"
    plan_path = run_dir / f"staircase_dispatch_plan{suffix}.json"
    _write_json(dag_path, dag)
    _write_json(plan_path, plan)
    return {
        "dag_path": _display_path(dag_path),
        "dispatch_plan_path": _display_path(plan_path),
        "dag_hash": dag.get("dag_hash"),
        "plan_hash": plan.get("plan_hash"),
        "selected_count": plan.get("selected_count"),
        "blocked_count": plan.get("blocked_count"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _queue_feedback_dependent_queue_ref_payload(
    *,
    parent_queue: Mapping[str, Any],
    parent_queue_path: Path,
    parent_state_path: Path,
    child_queue: Mapping[str, Any],
    child_queue_path: Path,
    child_dag_path: Path,
    child_plan_path: Path,
    child_dag: Mapping[str, Any],
    child_plan: Mapping[str, Any],
) -> dict[str, Any]:
    controls = dict(child_queue.get("controls") or {})
    return _false_authority_payload(
        {
            "schema": STAIRCASE_DEPENDENT_QUEUE_REF_SCHEMA,
            "kind": "experiment_queue",
            "relationship": "feedback_replan_child",
            "parent_queue_id": parent_queue.get("queue_id"),
            "parent_queue_path": _display_path(parent_queue_path),
            "parent_state_path": _display_path(parent_state_path),
            "child_queue_id": child_queue.get("queue_id"),
            "child_queue_path": _display_path(child_queue_path),
            "child_queue_sha256": _sha256_file(child_queue_path),
            "child_controls": {
                "mode": controls.get("mode"),
                "max_concurrency": dict(controls.get("max_concurrency") or {}),
            },
            "control_mode": controls.get("mode"),
            "required_parent_artifacts": [
                "queue_feedback_replan_request.json",
                "queue_performance_summary.json",
                "queue_performance_runtime_identity.json",
                "queue_performance_cache_identity.json",
            ],
            "dag_id": child_dag.get("dag_id"),
            "dag_path": _display_path(child_dag_path),
            "dag_hash": child_dag.get("dag_hash"),
            "dispatch_plan_path": _display_path(child_plan_path),
            "dispatch_plan_hash": child_plan.get("plan_hash"),
            "selected_count": child_plan.get("selected_count"),
            "blocked_count": child_plan.get("blocked_count"),
            "activation_policy": "manual_or_autopilot_resume_required",
            "allowed_use": "staircase_dependent_feedback_replan_planning_only",
            "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
        }
    )


def _build_queue_feedback_staircase_artifacts(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    parent_queue: Mapping[str, Any],
    parent_queue_path: Path,
    parent_state_path: Path,
    child_queue: Mapping[str, Any],
    child_queue_path: Path,
    artifact_stem: str = "queue_feedback_replan",
) -> dict[str, Any]:
    resource_pools = [parse_resource_pool_spec(spec) for spec in args.staircase_resource_pool] or None
    child_dag = build_staircase_dag_from_experiment_queue(
        child_queue,
        dag_id=f"{child_queue['queue_id']}_staircase",
        source_path=_display_path(child_queue_path),
        resource_pools=resource_pools,
    )
    child_plan = plan_staircase_dispatch(
        child_dag,
        max_nodes=args.staircase_max_nodes,
        allow_cloud=args.staircase_allow_cloud,
        diversity_bucket_limit=args.staircase_diversity_bucket_limit,
    )
    child_dag_path = run_dir / f"{artifact_stem}_staircase_dag.json"
    child_plan_path = run_dir / f"{artifact_stem}_staircase_dispatch_plan.json"
    _write_json(child_dag_path, child_dag)
    _write_json(child_plan_path, child_plan)
    dependent_ref = _queue_feedback_dependent_queue_ref_payload(
        parent_queue=parent_queue,
        parent_queue_path=parent_queue_path,
        parent_state_path=parent_state_path,
        child_queue=child_queue,
        child_queue_path=child_queue_path,
        child_dag_path=child_dag_path,
        child_plan_path=child_plan_path,
        child_dag=child_dag,
        child_plan=child_plan,
    )
    dependent_refs_path = run_dir / f"{artifact_stem}_dependent_queue_refs.json"
    _write_json(
        dependent_refs_path,
        _false_authority_payload(
            {
                "schema": "staircase_dependent_queue_refs.v1",
                "refs": [dependent_ref],
                "allowed_use": "staircase_dag_child_queue_composition_only",
                "forbidden_use": "score_claim_or_promotion_or_paid_dispatch_authority",
            }
        ),
    )
    return {
        "dependent_queue_refs_path": _display_path(dependent_refs_path),
        "dependent_queue_ref_count": 1,
        "dependent_queue_refs": [dependent_ref],
        "child_dag_path": _display_path(child_dag_path),
        "child_dispatch_plan_path": _display_path(child_plan_path),
        "child_dag_hash": child_dag.get("dag_hash"),
        "child_dispatch_plan_hash": child_plan.get("plan_hash"),
        "child_selected_count": child_plan.get("selected_count"),
        "child_blocked_count": child_plan.get("blocked_count"),
        "child_control_mode": dict(child_queue.get("controls") or {}).get("mode"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _ssh_executor_command(
    args: argparse.Namespace,
    *,
    execution_queue: Path,
    state_path: Path,
    staircase_plan_path: Path,
    run_dir: Path,
    execute: bool,
) -> list[str]:
    command = [
        sys.executable,
        "tools/run_staircase_ssh_executor.py",
        "--plan",
        _display_path(staircase_plan_path),
        "--queue",
        _display_path(execution_queue),
        "--state",
        _display_path(state_path),
        "--output",
        _display_path(
            run_dir / ("staircase_ssh_executor_execute.json" if execute else "staircase_ssh_executor_dry_run.json")
        ),
    ]
    if execute:
        command.append("--execute")
    command.extend(["--max-steps", str(args.staircase_ssh_max_steps)])
    if args.staircase_ssh_machine_id:
        command.extend(["--machine-id", str(args.staircase_ssh_machine_id)])
    if args.staircase_ssh_allow_future_executor:
        command.append("--allow-future-executor")
    if args.staircase_ssh_allow_dirty_remote_git:
        command.append("--allow-dirty-remote-git")
    if args.staircase_ssh_dirty_remote_git_rationale:
        command.extend(
            [
                "--dirty-remote-git-rationale",
                str(args.staircase_ssh_dirty_remote_git_rationale),
            ]
        )
    command.extend(_parse_remote_repo_roots(args.staircase_ssh_remote_repo_root))
    if execute or args.staircase_ssh_require_artifact_mobility:
        command.append("--require-artifact-mobility")
    command.extend(_parse_artifact_path_maps(args.staircase_ssh_artifact_path_map))
    if args.staircase_ssh_artifact_shared_path_rationale:
        command.extend(
            [
                "--artifact-shared-path-rationale",
                str(args.staircase_ssh_artifact_shared_path_rationale),
            ]
        )
    command.extend(["--rsync-binary", str(args.staircase_ssh_rsync_binary)])
    command.extend(
        [
            "--artifact-pull-timeout-seconds",
            str(args.staircase_ssh_artifact_pull_timeout_seconds),
        ]
    )
    return command


def _ssh_executor_dry_run_command(
    args: argparse.Namespace,
    *,
    execution_queue: Path,
    state_path: Path,
    staircase_plan_path: Path,
    run_dir: Path,
) -> list[str]:
    return _ssh_executor_command(
        args,
        execution_queue=execution_queue,
        state_path=state_path,
        staircase_plan_path=staircase_plan_path,
        run_dir=run_dir,
        execute=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    action_source_count = _action_source_count(args)
    if args.plan is None and action_source_count == 0:
        raise SystemExit(
            "provide --plan or high-level action sources such as "
            "--scorer-response, --inverse-scorer-surface, "
            "--byte-shaving-signal-surface, --byte-shaving-campaign-plan, "
            "--mlx-acquisition-batch, --mlx-effective-spend-triage-selection, "
            "or --atom"
        )
    if args.plan is not None and action_source_count:
        raise SystemExit(
            "--plan is mutually exclusive with high-level action sources; start from one authority surface per run"
        )
    if args.materializer_contexts is not None and args.materializer_artifact_map is not None:
        raise SystemExit("--materializer-contexts and --materializer-artifact-map are mutually exclusive")
    if _auto_artifact_map_requested(args) and (
        args.materializer_contexts is not None or args.materializer_artifact_map is not None
    ):
        raise SystemExit(
            "auto artifact-map flags cannot be combined with --materializer-contexts or --materializer-artifact-map"
        )
    if args.candidate_limit < 1:
        raise SystemExit("--candidate-limit must be >= 1")
    if args.mlx_acquisition_set_size < 1:
        raise SystemExit("--mlx-acquisition-set-size must be >= 1")
    if args.mlx_acquisition_limit is not None and args.mlx_acquisition_limit < 1:
        raise SystemExit("--mlx-acquisition-limit must be >= 1")
    if args.campaign_plan_max_k is not None and args.campaign_plan_max_k < 1:
        raise SystemExit("--campaign-plan-max-k must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.poll_interval_seconds < 0:
        raise SystemExit("--poll-interval-seconds must be non-negative")
    if args.queue_feedback_replan_followup_max_steps < 1:
        raise SystemExit("--queue-feedback-replan-followup-max-steps must be >= 1")
    if args.queue_feedback_replan_followup_max_parallel < 1:
        raise SystemExit("--queue-feedback-replan-followup-max-parallel must be >= 1")
    if args.queue_feedback_replan_followup_max_idle_cycles < 1:
        raise SystemExit("--queue-feedback-replan-followup-max-idle-cycles must be >= 1")
    if args.queue_feedback_replan_followup_idle_sleep_seconds < 0:
        raise SystemExit("--queue-feedback-replan-followup-idle-sleep-seconds must be non-negative")
    if args.queue_feedback_replan_followup_poll_interval_seconds < 0:
        raise SystemExit("--queue-feedback-replan-followup-poll-interval-seconds must be non-negative")
    if args.queue_feedback_replan_followup_tail_lines < 0:
        raise SystemExit("--queue-feedback-replan-followup-tail-lines must be non-negative")
    if args.queue_observation_recovery_max_steps < 1:
        raise SystemExit("--queue-observation-recovery-max-steps must be >= 1")
    if args.queue_observation_recovery_max_parallel < 1:
        raise SystemExit("--queue-observation-recovery-max-parallel must be >= 1")
    if args.queue_observation_recovery_max_idle_cycles < 1:
        raise SystemExit("--queue-observation-recovery-max-idle-cycles must be >= 1")
    if args.queue_observation_recovery_idle_sleep_seconds < 0:
        raise SystemExit("--queue-observation-recovery-idle-sleep-seconds must be non-negative")
    if args.queue_observation_recovery_poll_interval_seconds < 0:
        raise SystemExit("--queue-observation-recovery-poll-interval-seconds must be non-negative")
    if args.queue_observation_recovery_tail_lines < 0:
        raise SystemExit("--queue-observation-recovery-tail-lines must be non-negative")
    if args.queue_feedback_replan_policy_iteration < 0:
        raise SystemExit("--queue-feedback-replan-policy-iteration must be non-negative")
    if args.queue_feedback_replan_policy_max_iterations < 1:
        raise SystemExit("--queue-feedback-replan-policy-max-iterations must be >= 1")
    if args.staircase_ssh_max_steps < 1:
        raise SystemExit("--staircase-ssh-max-steps must be >= 1")
    if args.staircase_ssh_execute and args.execute:
        raise SystemExit("--staircase-ssh-execute and top-level --execute cannot target the same queue run")
    if args.staircase_ssh_artifact_pull_timeout_seconds < 1:
        raise SystemExit("--staircase-ssh-artifact-pull-timeout-seconds must be >= 1")
    if args.runtime_policy_timeout_multiplier <= 0:
        raise SystemExit("--runtime-policy-timeout-multiplier must be positive")
    if args.runtime_policy_min_timeout_seconds < 0:
        raise SystemExit("--runtime-policy-min-timeout-seconds must be non-negative")
    if args.runtime_policy_max_timeout_seconds < 1:
        raise SystemExit("--runtime-policy-max-timeout-seconds must be positive")
    if args.runtime_policy_min_timeout_seconds > args.runtime_policy_max_timeout_seconds:
        raise SystemExit("--runtime-policy-min-timeout-seconds must be <= --runtime-policy-max-timeout-seconds")
    if args.runtime_policy_cpu_count is not None and args.runtime_policy_cpu_count < 1:
        raise SystemExit("--runtime-policy-cpu-count must be >= 1")
    if (
        args.apply_runtime_policy
        and args.runtime_policy_no_apply_concurrency
        and not (args.runtime_policy_apply_timeouts)
    ):
        raise SystemExit("--apply-runtime-policy would apply neither concurrency nor timeouts")
    if args.staircase_ssh_dirty_remote_git_rationale and not args.staircase_ssh_allow_dirty_remote_git:
        raise SystemExit("--staircase-ssh-dirty-remote-git-rationale requires --staircase-ssh-allow-dirty-remote-git")
    if args.staircase_ssh_execute and not (
        args.staircase_ssh_artifact_path_map or args.staircase_ssh_artifact_shared_path_rationale
    ):
        raise SystemExit(
            "--staircase-ssh-execute requires --staircase-ssh-artifact-path-map "
            "or --staircase-ssh-artifact-shared-path-rationale"
        )
    if args.staircase_ssh_artifact_path_map and args.staircase_ssh_artifact_shared_path_rationale:
        raise SystemExit(
            "--staircase-ssh-artifact-path-map and "
            "--staircase-ssh-artifact-shared-path-rationale are mutually exclusive"
        )
    if args.execute and args.queue_state is not None and not args.queue_state_rationale:
        raise SystemExit("--queue-state requires --queue-state-rationale when executing the generated queue")
    if (
        (
            args.execute_queue_feedback_replan_followup
            or args.queue_feedback_replan_followup_policy_local_autopilot
        )
        and args.queue_feedback_replan_followup_state is not None
        and not args.queue_feedback_replan_followup_state_rationale
    ):
        raise SystemExit(
            "--queue-feedback-replan-followup-state requires "
            "--queue-feedback-replan-followup-state-rationale when executing the feedback queue"
        )
    if (
        (
            args.execute_queue_observation_recovery
            or args.queue_observation_recovery_policy_local_autopilot
        )
        and args.queue_observation_recovery_state is not None
        and not args.queue_observation_recovery_state_rationale
    ):
        raise SystemExit(
            "--queue-observation-recovery-state requires "
            "--queue-observation-recovery-state-rationale when executing the recovery queue"
        )
    run_dir = (
        _resolve(args.run_dir)
        if args.run_dir is not None
        else (REPO_ROOT / ".omx" / "research" / f"byte_shaving_materializer_campaign_{_utc_stamp()}")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    execution_queue = run_dir / "materializer_execution_queue.json"
    run_config_record = None
    if args.run_config is not None:
        run_config_path = _resolve(args.run_config)
        run_config_record = {
            "schema": RUN_CONFIG_SCHEMA,
            "path": _display_path(run_config_path),
            "sha256": _sha256_file(run_config_path),
            "bytes": run_config_path.stat().st_size,
            "cli_overrides_allowed": True,
            "authority": "configuration_only_not_queue_or_score_authority",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    commands: list[CommandResult] = []
    generated_action_functional_path: Path | None = None
    generated_campaign_plan_path: Path | None = None
    generated_materializer_artifact_map_path: Path | None = None
    generated_mlx_acquisition_batch_paths: list[Path] = []
    plan_path = _resolve(args.plan) if args.plan is not None else None
    if plan_path is None:
        generated_action_functional_path = run_dir / "inverse_steganalysis_action_functional.json"
        generated_campaign_plan_path = run_dir / "byte_shaving_campaign_plan.json"
        for output_path, command in _build_mlx_acquisition_batch_commands(
            args,
            run_dir=run_dir,
        ):
            commands.append(_run(command))
            generated_mlx_acquisition_batch_paths.append(output_path)
        action_result = _run(
            _build_action_functional_command(
                args,
                run_dir=run_dir,
                generated_mlx_acquisition_batches=generated_mlx_acquisition_batch_paths,
            )
        )
        commands.append(action_result)
        plan_result = _run(
            _build_campaign_plan_command(
                args,
                action_functional_path=generated_action_functional_path,
                run_dir=run_dir,
            )
        )
        commands.append(plan_result)
        plan_path = generated_campaign_plan_path

    generated_materializer_artifact_map_path = _write_generated_materializer_artifact_map(
        args,
        run_dir=run_dir,
        generated_action_functional_path=generated_action_functional_path,
    )

    build_result = _run(
        _build_queue_command(
            args,
            run_dir=run_dir,
            plan_path=plan_path,
            generated_materializer_artifact_map=generated_materializer_artifact_map_path,
        )
    )
    commands.append(build_result)
    queue = load_queue_definition(execution_queue)
    state_path = (
        _resolve(args.queue_state)
        if args.queue_state is not None
        else run_dir / "materializer_execution_queue.sqlite"
    )

    for command in (
        _experiment_queue_command(
            execution_queue=execution_queue,
            state_path=state_path,
            subcommand=["validate"],
        ),
        _experiment_queue_command(
            execution_queue=execution_queue,
            state_path=state_path,
            subcommand=["init"],
        ),
    ):
        commands.append(_run(command))

    runtime_policy_result: CommandResult | None = None
    runtime_policy_payload: dict[str, Any] | None = None
    runtime_policy_output_path: Path | None = None
    runtime_policy_applied_queue_path: Path | None = None
    if args.derive_runtime_policy or args.apply_runtime_policy:
        (
            runtime_policy_output_path,
            runtime_policy_applied_queue_path,
            runtime_policy_command,
        ) = _runtime_policy_command(
            args,
            execution_queue=execution_queue,
            state_path=state_path,
            run_dir=run_dir,
        )
        runtime_policy_result = _run(runtime_policy_command)
        commands.append(runtime_policy_result)
        runtime_policy_payload = _json_from_stdout(runtime_policy_result)
        if runtime_policy_applied_queue_path is not None:
            execution_queue = runtime_policy_applied_queue_path
            queue = load_queue_definition(execution_queue)
            for command in (
                _experiment_queue_command(
                    execution_queue=execution_queue,
                    state_path=state_path,
                    subcommand=["validate"],
                ),
                _experiment_queue_command(
                    execution_queue=execution_queue,
                    state_path=state_path,
                    subcommand=["init"],
                ),
            ):
                commands.append(_run(command))

    staircase_artifacts: dict[str, Any] | None = None
    ssh_executor_dry_run: dict[str, Any] | None = None
    ssh_executor_execute: dict[str, Any] | None = None
    ssh_execute_result: CommandResult | None = None
    if args.emit_staircase_plan or args.staircase_ssh_dry_run or args.staircase_ssh_execute:
        staircase_artifacts = _build_staircase_artifacts(
            args,
            run_dir=run_dir,
            execution_queue=execution_queue,
            state_path=state_path,
            queue=queue,
        )
        if args.staircase_ssh_dry_run:
            ssh_result = _run(
                _ssh_executor_dry_run_command(
                    args,
                    execution_queue=execution_queue,
                    state_path=state_path,
                    staircase_plan_path=run_dir / "staircase_dispatch_plan.json",
                    run_dir=run_dir,
                ),
                check=False,
            )
            commands.append(ssh_result)
            ssh_executor_dry_run = _require_json_stdout(
                ssh_result,
                label="staircase SSH executor dry-run",
            )
        if args.staircase_ssh_execute:
            ssh_execute_result = _run(
                _ssh_executor_command(
                    args,
                    execution_queue=execution_queue,
                    state_path=state_path,
                    staircase_plan_path=run_dir / "staircase_dispatch_plan.json",
                    run_dir=run_dir,
                    execute=True,
                ),
                check=False,
            )
            commands.append(ssh_execute_result)
            ssh_executor_execute = _require_json_stdout(
                ssh_execute_result,
                label="staircase SSH executor execute",
                allow_nonzero=True,
            )

    worker_command = _experiment_queue_command(
        execution_queue=execution_queue,
        state_path=state_path,
        subcommand=[
            "run-worker",
            "--max-steps",
            str(args.max_steps),
            "--max-parallel",
            str(args.max_parallel),
            "--idle-sleep-seconds",
            str(args.idle_sleep_seconds),
            "--poll-interval-seconds",
            str(args.poll_interval_seconds),
            "--max-idle-cycles",
            str(args.max_idle_cycles),
        ],
    )
    if args.max_experiments is not None:
        worker_command.extend(["--max-experiments", str(args.max_experiments)])
    queue_state_rationale = _queue_state_rationale_for_worker(
        args,
        run_dir=run_dir,
    )
    if queue_state_rationale:
        worker_command.extend(
            [
                "--noncanonical-state-rationale",
                queue_state_rationale,
            ]
        )
    if args.execute:
        worker_command.append("--execute")
    worker_result = _run(worker_command, check=False)
    commands.append(worker_result)

    observe_result = _run(
        _experiment_queue_command(
            execution_queue=execution_queue,
            state_path=state_path,
            subcommand=[
                "observe",
                "--tail-lines",
                str(args.tail_lines),
                "--include-orphans",
                "--format",
                "json",
            ],
        ),
        check=False,
    )
    commands.append(observe_result)
    performance_result = _run(
        _experiment_queue_command(
            execution_queue=execution_queue,
            state_path=state_path,
            subcommand=["performance"],
        ),
        check=False,
    )
    commands.append(performance_result)

    summary_path = run_dir / "materializer_campaign_run.json"
    queue_observation_path = run_dir / "queue_observation.json"
    queue_observation_recovery_plan_path = run_dir / "queue_observation_recovery_plan.json"
    queue_observation_recovery_queue_path = run_dir / "queue_observation_recovery_queue.json"
    queue_observation_recovery_queue_state_path = (
        run_dir / "queue_observation_recovery_queue.sqlite"
    )
    queue_performance_summary_path = run_dir / "queue_performance_summary.json"
    queue_feedback_replan_request_path = run_dir / "queue_feedback_replan_request.json"
    queue_feedback_replan_followup_queue_path = run_dir / "queue_feedback_replan_followup_queue.json"
    queue_feedback_replan_policy_path = run_dir / "queue_feedback_replan_policy.json"
    queue_feedback_replan_continuation_queue_path = (
        run_dir / "queue_feedback_replan_continuation_queue.json"
    )
    queue_feedback_candidate_widening_queue_path = (
        run_dir / "queue_feedback_candidate_widening_queue.json"
    )
    queue_feedback_candidate_actuation_planning_queue_path = (
        run_dir / "queue_feedback_candidate_actuation_planning_queue.json"
    )
    generated_runtime_identity_path = run_dir / "queue_performance_runtime_identity.json"
    generated_cache_identity_path = run_dir / "queue_performance_cache_identity.json"
    response_update_placeholder_path = run_dir / "canonical_response_update_placeholder.json"
    queue_feedback_replan_staircase_artifacts: dict[str, Any] | None = None
    queue_feedback_replan_continuation_staircase_artifacts: dict[str, Any] | None = None
    queue_feedback_candidate_widening_staircase_artifacts: dict[str, Any] | None = None
    queue_feedback_candidate_actuation_planning_staircase_artifacts: (
        dict[str, Any] | None
    ) = None
    queue_feedback_replan_continuation_queue: dict[str, Any] | None = None
    queue_feedback_replan_continuation_queue_blockers: list[str] = []
    queue_feedback_candidate_widening_queue: dict[str, Any] | None = None
    queue_feedback_candidate_widening_queue_blockers: list[str] = []
    queue_feedback_candidate_actuation_planning_queue: dict[str, Any] | None = None
    queue_feedback_candidate_actuation_planning_queue_blockers: list[str] = []
    queue_observation_recovery_queue: dict[str, Any] | None = None
    queue_observation_recovery_queue_blockers: list[str] = []
    queue_observation_recovery_staircase_artifacts: dict[str, Any] | None = None
    queue_observation_recovery_execution: dict[str, Any] | None = None
    queue_observation_recovery_execution_returncode = 0
    queue_observation_recovery_policy = _queue_observation_recovery_activation_policy(args)
    queue_observation_recovery_policy_blockers: list[str] = []
    queue_observation_recovery_execution_attempted = False
    queue_feedback_replan_followup_execution: dict[str, Any] | None = None
    queue_feedback_replan_followup_execution_returncode = 0
    post_recovery_feedback_replan: dict[str, Any] | None = None
    post_recovery_feedback_replan_returncode = 0
    queue_feedback_replan_followup_policy = _queue_feedback_replan_followup_activation_policy(args)
    queue_feedback_replan_followup_policy_blockers: list[str] = []
    queue_feedback_replan_followup_execution_attempted = False
    queue_observation = _json_from_stdout(observe_result)
    if queue_observation is None:
        queue_observation = _false_authority_payload(
            {
                "schema": "experiment_queue_observation_unavailable.v1",
                "generated_at_utc": _utc_stamp(),
                "queue_id": queue.get("queue_id"),
                "state": _display_path(state_path),
                "observe_returncode": observe_result.returncode,
                "healthy": False,
                "blockers": ["queue_observation_command_failed_or_invalid_json"],
                "blocker_count": 1,
                "observe_read_only": True,
                "allowed_use": "local_queue_observation_only",
                "forbidden_use": "score_claim_or_promotion_or_dispatch_authority",
            }
        )
    _write_json(queue_observation_path, queue_observation)
    queue_observation_recovery_plan = build_queue_observation_recovery_plan(
        queue_observation,
        queue_path=_display_path(execution_queue),
        state_path=_display_path(state_path),
        reason="materializer campaign queue observation recovery",
    )
    _write_json(queue_observation_recovery_plan_path, queue_observation_recovery_plan)
    queue_performance_summary = _queue_performance_summary_payload(
        performance_result,
        queue=queue,
    )
    queue_performance_summary.update(
        {
            "generated_at_utc": _utc_stamp(),
            "producer": "tools/run_byte_shaving_materializer_campaign.py",
            "source_run_path": _display_path(summary_path),
            "queue_path": _display_path(execution_queue),
            "queue_state_path": _display_path(state_path),
            "queue_definition_sha256": _sha256_file(execution_queue),
            "worker_returncode": worker_result.returncode,
        }
    )
    _write_json(queue_performance_summary_path, queue_performance_summary)
    runtime_identity_path = (
        _resolve(args.queue_performance_runtime_identity)
        if args.queue_performance_runtime_identity is not None
        else generated_runtime_identity_path
    )
    cache_identity_path = (
        _resolve(args.queue_performance_cache_identity)
        if args.queue_performance_cache_identity is not None
        else generated_cache_identity_path
    )
    if args.queue_performance_runtime_identity is None:
        _write_json(
            runtime_identity_path,
            _queue_runtime_identity_payload(
                run_dir=run_dir,
                execution_queue=execution_queue,
                state_path=state_path,
                queue=queue,
                runtime_policy_output_path=runtime_policy_output_path,
                runtime_policy_applied_queue_path=runtime_policy_applied_queue_path,
            ),
        )
    if args.queue_performance_cache_identity is None:
        _write_json(
            cache_identity_path,
            _queue_cache_identity_payload(
                run_dir=run_dir,
                execution_queue=execution_queue,
                state_path=state_path,
                queue_performance_summary_path=queue_performance_summary_path,
                queue=queue,
            ),
        )
    runtime_identity_payload = _load_json_object_if_present(runtime_identity_path) or {}
    cache_identity_payload = _load_json_object_if_present(cache_identity_path) or {}
    materializer_receiver_feedback_observation_path = (
        _write_materializer_receiver_feedback_sweep(
            run_dir=run_dir,
            queue_observation=queue_observation,
            runtime_identity=runtime_identity_payload,
            cache_identity=cache_identity_payload,
        )
    )
    queue_feedback_replan_request = _queue_feedback_replan_request_payload(
        args,
        summary_path=summary_path,
        plan_path=plan_path,
        queue_performance_summary_path=queue_performance_summary_path,
        queue_performance_summary=queue_performance_summary,
        runtime_identity_path=runtime_identity_path,
        cache_identity_path=cache_identity_path,
        generated_runtime_identity=args.queue_performance_runtime_identity is None,
        generated_cache_identity=args.queue_performance_cache_identity is None,
        run_dir=run_dir,
        execution_queue=execution_queue,
        state_path=state_path,
        queue_observation_path=queue_observation_path,
        queue_observation_recovery_plan_path=queue_observation_recovery_plan_path,
        queue_observation_recovery_plan=queue_observation_recovery_plan,
    )
    (
        queue_feedback_replan_followup_queue,
        queue_feedback_replan_followup_blockers,
    ) = _queue_feedback_replan_followup_queue_payload(
        args,
        queue_feedback_replan_request=queue_feedback_replan_request,
        queue_feedback_replan_request_path=queue_feedback_replan_request_path,
        run_dir=run_dir,
        execution_queue=execution_queue,
        state_path=state_path,
        source_queue=queue,
    )
    queue_feedback_replan_request.update(
        {
            "queue_owned_followup_queue_path": (
                _display_path(queue_feedback_replan_followup_queue_path)
                if queue_feedback_replan_followup_queue is not None
                else None
            ),
            "queue_owned_followup_queue_emitted": queue_feedback_replan_followup_queue is not None,
            "queue_owned_followup_queue_blockers": queue_feedback_replan_followup_blockers,
        }
    )
    _write_json(queue_feedback_replan_request_path, queue_feedback_replan_request)
    if queue_feedback_replan_followup_queue is not None:
        _write_json(
            queue_feedback_replan_followup_queue_path,
            queue_feedback_replan_followup_queue,
        )
        queue_feedback_replan_staircase_artifacts = _build_queue_feedback_staircase_artifacts(
            args,
            run_dir=run_dir,
            parent_queue=queue,
            parent_queue_path=execution_queue,
            parent_state_path=state_path,
            child_queue=queue_feedback_replan_followup_queue,
            child_queue_path=queue_feedback_replan_followup_queue_path,
        )
        if staircase_artifacts is not None:
            staircase_artifacts["feedback_child_queue"] = (
                queue_feedback_replan_staircase_artifacts
            )
            staircase_artifacts["with_feedback_child"] = _build_staircase_artifacts(
                args,
                run_dir=run_dir,
                execution_queue=execution_queue,
                state_path=state_path,
                queue=queue,
                dependent_queue_refs=queue_feedback_replan_staircase_artifacts[
                    "dependent_queue_refs"
                ],
                suffix=".with_feedback_child",
            )
        if args.queue_feedback_replan_followup_policy_local_autopilot:
            queue_feedback_replan_followup_policy_blockers = (
                _queue_feedback_replan_followup_local_autopolicy_blockers(
                    queue_feedback_replan_followup_queue,
                    run_dir=run_dir,
                )
            )
        should_execute_feedback_followup = (
            args.execute_queue_feedback_replan_followup
            or (
                args.queue_feedback_replan_followup_policy_local_autopilot
                and not queue_feedback_replan_followup_policy_blockers
            )
        )
        if should_execute_feedback_followup:
            queue_feedback_replan_followup_execution_attempted = True
            (
                queue_feedback_replan_followup_execution,
                followup_execution_commands,
                queue_feedback_replan_followup_execution_returncode,
            ) = _queue_feedback_replan_followup_execution_payload(
                args,
                child_queue=queue_feedback_replan_followup_queue,
                child_queue_path=queue_feedback_replan_followup_queue_path,
                run_dir=run_dir,
                activation_policy=queue_feedback_replan_followup_policy,
            )
            commands.extend(followup_execution_commands)
        elif args.queue_feedback_replan_followup_policy_local_autopilot:
            queue_feedback_replan_followup_execution = (
                _queue_feedback_replan_followup_execution_refusal_payload(
                    blockers=(
                        queue_feedback_replan_followup_policy_blockers
                        or ["queue_feedback_replan_followup_policy_not_enabled"]
                    ),
                    child_queue_path=queue_feedback_replan_followup_queue_path,
                    activation_policy=queue_feedback_replan_followup_policy,
                )
            )
    elif args.execute_queue_feedback_replan_followup:
        queue_feedback_replan_followup_execution_returncode = 2
        queue_feedback_replan_followup_execution = (
            _queue_feedback_replan_followup_execution_refusal_payload(
                blockers=(
                    queue_feedback_replan_followup_blockers
                    or ["queue_feedback_replan_followup_queue_not_emitted"]
                ),
                child_queue_path=queue_feedback_replan_followup_queue_path,
                activation_policy=queue_feedback_replan_followup_policy,
            )
        )
    elif args.queue_feedback_replan_followup_policy_local_autopilot:
        queue_feedback_replan_followup_execution = (
            _queue_feedback_replan_followup_execution_refusal_payload(
                blockers=(
                    queue_feedback_replan_followup_blockers
                    or ["queue_feedback_replan_followup_queue_not_emitted"]
                ),
                child_queue_path=queue_feedback_replan_followup_queue_path,
                activation_policy=queue_feedback_replan_followup_policy,
            )
        )
    exact_readiness_handoffs = _exact_readiness_handoff_paths(
        run_dir=run_dir,
        queue=queue,
    )
    response_update_placeholder = _response_update_placeholder_payload(
        summary_path=summary_path,
        queue_performance_summary_path=queue_performance_summary_path,
        queue_feedback_replan_request_path=queue_feedback_replan_request_path,
        queue_feedback_replan_followup_queue_path=(
            queue_feedback_replan_followup_queue_path
            if queue_feedback_replan_followup_queue is not None
            else None
        ),
        queue_feedback_replan_followup_blockers=queue_feedback_replan_followup_blockers,
        queue_feedback_replan_blockers=queue_feedback_replan_request["blockers"],
        next_action_functional_command_hint=queue_feedback_replan_request["command_template"],
        run_dir=run_dir,
        execution_queue=execution_queue,
        state_path=state_path,
        runtime_policy_output_path=runtime_policy_output_path,
        runtime_policy_applied_queue_path=runtime_policy_applied_queue_path,
        exact_readiness_handoffs=exact_readiness_handoffs,
    )
    _write_json(response_update_placeholder_path, response_update_placeholder)

    payload = {
        "schema": RUN_SCHEMA,
        "run_dir": _display_path(run_dir),
        "run_config": run_config_record,
        "plan": _display_path(plan_path),
        "generated_action_functional_path": (
            None if generated_action_functional_path is None else _display_path(generated_action_functional_path)
        ),
        "generated_campaign_plan_path": (
            None if generated_campaign_plan_path is None else _display_path(generated_campaign_plan_path)
        ),
        "generated_mlx_acquisition_batch_paths": [
            _display_path(path) for path in generated_mlx_acquisition_batch_paths
        ],
        "generated_materializer_artifact_map_path": (
            None
            if generated_materializer_artifact_map_path is None
            else _display_path(generated_materializer_artifact_map_path)
        ),
        "high_level_action_source_count": action_source_count,
        "queue_path": _display_path(execution_queue),
        "state_path": _display_path(state_path),
        "runtime_policy_path": (
            None if runtime_policy_output_path is None else _display_path(runtime_policy_output_path)
        ),
        "runtime_policy_applied_queue_path": (
            None if runtime_policy_applied_queue_path is None else _display_path(runtime_policy_applied_queue_path)
        ),
        "queue_performance_summary_path": _display_path(queue_performance_summary_path),
        "queue_feedback_replan_request_path": _display_path(queue_feedback_replan_request_path),
        "queue_feedback_replan_followup_queue_path": (
            _display_path(queue_feedback_replan_followup_queue_path)
            if queue_feedback_replan_followup_queue is not None
            else None
        ),
        "queue_feedback_replan_followup_queue_emitted": queue_feedback_replan_followup_queue is not None,
        "queue_feedback_replan_followup_queue_blockers": queue_feedback_replan_followup_blockers,
        "queue_feedback_replan_followup_policy": queue_feedback_replan_followup_policy,
        "queue_feedback_replan_followup_execution_policy": queue_feedback_replan_followup_policy,
        "queue_feedback_replan_followup_policy_enabled": bool(
            args.queue_feedback_replan_followup_policy_local_autopilot
        ),
        "queue_feedback_replan_followup_policy_blockers": (
            queue_feedback_replan_followup_policy_blockers
        ),
        "queue_feedback_replan_followup_execution_requested": bool(
            args.execute_queue_feedback_replan_followup
            or args.queue_feedback_replan_followup_policy_local_autopilot
        ),
        "queue_feedback_replan_followup_executed": queue_feedback_replan_followup_execution_attempted,
        "queue_feedback_replan_followup_execution_success": (
            None
            if queue_feedback_replan_followup_execution is None
            else bool(queue_feedback_replan_followup_execution.get("success"))
        ),
        "queue_feedback_replan_followup_execution": queue_feedback_replan_followup_execution,
        "queue_feedback_replan_followup_state_path": (
            None
            if queue_feedback_replan_followup_execution is None
            else queue_feedback_replan_followup_execution.get("state_path")
        ),
        "queue_feedback_replan_followup_action_functional_path": (
            None
            if queue_feedback_replan_followup_execution is None
            else queue_feedback_replan_followup_execution.get(
                "action_functional_output_path"
            )
        ),
        "queue_feedback_replan_staircase_artifacts": queue_feedback_replan_staircase_artifacts,
        "queue_performance_runtime_identity_path": _display_path(runtime_identity_path),
        "queue_performance_cache_identity_path": _display_path(cache_identity_path),
        "materializer_receiver_feedback_observation_path": (
            None
            if materializer_receiver_feedback_observation_path is None
            else _display_path(materializer_receiver_feedback_observation_path)
        ),
        "response_update_placeholder_path": _display_path(response_update_placeholder_path),
        "response_update_applied": False,
        "replan_required": True,
        "queue_feedback_replan_ready": queue_feedback_replan_request[
            "ready_for_action_functional_feedback"
        ],
        "queue_feedback_replan_blockers": queue_feedback_replan_request["blockers"],
        "queue_observation_path": _display_path(queue_observation_path),
        "queue_observation_recovery_plan_path": _display_path(
            queue_observation_recovery_plan_path
        ),
        "queue_observation_recovery_plan": queue_observation_recovery_plan,
        "queue_observation_recovery_required": queue_observation_recovery_plan[
            "recovery_required"
        ],
        "queue_observation_maintenance_recommended": (
            queue_observation_recovery_plan["maintenance_recommended"]
        ),
        "next_run_hint": response_update_placeholder["next_run_hint"],
        "exact_readiness_handoff_count": len(exact_readiness_handoffs),
        "exact_readiness_handoff_paths": exact_readiness_handoffs,
        "execute": bool(args.execute),
        "staircase_ssh_execute": bool(args.staircase_ssh_execute),
        "queue_id": queue["queue_id"],
        "experiment_count": len(queue["experiments"]),
        "build": _json_from_stdout(build_result),
        "runtime_policy": runtime_policy_payload,
        "worker": _json_from_stdout(worker_result),
        "staircase": staircase_artifacts,
        "ssh_executor_dry_run": ssh_executor_dry_run,
        "ssh_executor_execute": ssh_executor_execute,
        "observation": queue_observation,
        "performance": queue_performance_summary,
        "queue_feedback_replan_request": queue_feedback_replan_request,
        "response_update_placeholder": response_update_placeholder,
        "commands": [result.to_dict() for result in commands],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    queue_feedback_replan_policy = build_queue_feedback_replan_policy(
        payload,
        feedback_followup_queue=queue_feedback_replan_followup_queue,
        source_run_path=_display_path(summary_path),
        iteration_index=args.queue_feedback_replan_policy_iteration,
        max_iterations=args.queue_feedback_replan_policy_max_iterations,
    )
    _write_json(queue_feedback_replan_policy_path, queue_feedback_replan_policy)
    feedback_lane_id = args.lane_id or _queue_feedback_replan_continuation_lane_id(
        queue,
        plan_path=plan_path,
    )
    (
        queue_observation_recovery_queue,
        queue_observation_recovery_queue_blockers,
    ) = build_queue_observation_recovery_queue(
        queue_feedback_replan_policy,
        lane_id=feedback_lane_id,
        source_policy_path=_display_path(queue_feedback_replan_policy_path),
    )
    if queue_observation_recovery_queue is not None:
        _write_json(
            queue_observation_recovery_queue_path,
            queue_observation_recovery_queue,
        )
        queue_observation_recovery_staircase_artifacts = (
            _build_queue_feedback_staircase_artifacts(
                args,
                run_dir=run_dir,
                parent_queue=queue,
                parent_queue_path=execution_queue,
                parent_state_path=state_path,
                child_queue=queue_observation_recovery_queue,
                child_queue_path=queue_observation_recovery_queue_path,
                artifact_stem="queue_observation_recovery",
            )
        )
        if staircase_artifacts is not None:
            staircase_artifacts["queue_observation_recovery_child"] = (
                queue_observation_recovery_staircase_artifacts
            )
        if args.queue_observation_recovery_policy_local_autopilot:
            queue_observation_recovery_policy_blockers = (
                _queue_observation_recovery_local_autopolicy_blockers(
                    queue_observation_recovery_queue
                )
            )
        should_execute_recovery = (
            args.execute_queue_observation_recovery
            or (
                args.queue_observation_recovery_policy_local_autopilot
                and not queue_observation_recovery_policy_blockers
            )
        )
        if should_execute_recovery:
            queue_observation_recovery_execution_attempted = True
            (
                queue_observation_recovery_execution,
                recovery_execution_commands,
                queue_observation_recovery_execution_returncode,
            ) = _queue_observation_recovery_execution_payload(
                args,
                recovery_queue=queue_observation_recovery_queue,
                recovery_queue_path=queue_observation_recovery_queue_path,
                recovery_queue_state_path=queue_observation_recovery_queue_state_path,
                activation_policy=queue_observation_recovery_policy,
            )
            commands.extend(recovery_execution_commands)
        elif args.queue_observation_recovery_policy_local_autopilot:
            queue_observation_recovery_execution = (
                _queue_observation_recovery_execution_refusal_payload(
                    blockers=(
                        queue_observation_recovery_policy_blockers
                        or ["queue_observation_recovery_policy_not_enabled"]
                    ),
                    recovery_queue_path=queue_observation_recovery_queue_path,
                    activation_policy=queue_observation_recovery_policy,
                )
            )
    elif args.execute_queue_observation_recovery:
        queue_observation_recovery_execution_returncode = 2
        queue_observation_recovery_execution = (
            _queue_observation_recovery_execution_refusal_payload(
                blockers=(
                    queue_observation_recovery_queue_blockers
                    or ["queue_observation_recovery_queue_not_emitted"]
                ),
                recovery_queue_path=queue_observation_recovery_queue_path,
                activation_policy=queue_observation_recovery_policy,
            )
        )
    elif args.queue_observation_recovery_policy_local_autopilot:
        queue_observation_recovery_execution = (
            _queue_observation_recovery_execution_refusal_payload(
                blockers=(
                    queue_observation_recovery_queue_blockers
                    or ["queue_observation_recovery_queue_not_emitted"]
                ),
                recovery_queue_path=queue_observation_recovery_queue_path,
                activation_policy=queue_observation_recovery_policy,
            )
        )
    if (
        queue_observation_recovery_execution is not None
        and queue_observation_recovery_execution.get("success") is True
    ):
        (
            post_recovery_feedback_replan,
            post_recovery_feedback_replan_returncode,
        ) = _post_recovery_feedback_replan_payload(
            args,
            recovery_execution=queue_observation_recovery_execution,
            run_dir=run_dir,
            commands=commands,
            queue=queue,
            summary_path=summary_path,
            plan_path=plan_path,
            execution_queue=execution_queue,
            state_path=state_path,
            queue_performance_summary_path=queue_performance_summary_path,
            queue_performance_summary=queue_performance_summary,
            runtime_identity_path=runtime_identity_path,
            cache_identity_path=cache_identity_path,
            generated_runtime_identity=args.queue_performance_runtime_identity is None,
            generated_cache_identity=args.queue_performance_cache_identity is None,
            feedback_lane_id=feedback_lane_id,
            feedback_policy_activation=queue_feedback_replan_followup_policy,
            exact_readiness_handoffs=exact_readiness_handoffs,
            staircase_artifacts=staircase_artifacts,
        )
    (
        queue_feedback_replan_continuation_queue,
        queue_feedback_replan_continuation_queue_blockers,
    ) = build_queue_feedback_replan_continuation_queue(
        queue_feedback_replan_policy,
        lane_id=feedback_lane_id,
        source_policy_path=_display_path(queue_feedback_replan_policy_path),
    )
    if queue_feedback_replan_continuation_queue is not None:
        _write_json(
            queue_feedback_replan_continuation_queue_path,
            queue_feedback_replan_continuation_queue,
        )
        queue_feedback_replan_continuation_staircase_artifacts = (
            _build_queue_feedback_staircase_artifacts(
                args,
                run_dir=run_dir,
                parent_queue=queue,
                parent_queue_path=execution_queue,
                parent_state_path=state_path,
                child_queue=queue_feedback_replan_continuation_queue,
                child_queue_path=queue_feedback_replan_continuation_queue_path,
                artifact_stem="queue_feedback_replan_continuation",
            )
        )
        if staircase_artifacts is not None:
            staircase_artifacts["feedback_continuation_child"] = (
                queue_feedback_replan_continuation_staircase_artifacts
            )
    (
        queue_feedback_candidate_widening_queue,
        queue_feedback_candidate_widening_queue_blockers,
    ) = build_queue_feedback_candidate_widening_queue(
        queue_feedback_replan_policy,
        lane_id=feedback_lane_id,
        source_policy_path=_display_path(queue_feedback_replan_policy_path),
    )
    if queue_feedback_candidate_widening_queue is not None:
        _write_json(
            queue_feedback_candidate_widening_queue_path,
            queue_feedback_candidate_widening_queue,
        )
        queue_feedback_candidate_widening_staircase_artifacts = (
            _build_queue_feedback_staircase_artifacts(
                args,
                run_dir=run_dir,
                parent_queue=queue,
                parent_queue_path=execution_queue,
                parent_state_path=state_path,
                child_queue=queue_feedback_candidate_widening_queue,
                child_queue_path=queue_feedback_candidate_widening_queue_path,
                artifact_stem="queue_feedback_candidate_widening",
            )
        )
        if staircase_artifacts is not None:
            staircase_artifacts["feedback_candidate_widening_child"] = (
                queue_feedback_candidate_widening_staircase_artifacts
            )
    (
        queue_feedback_candidate_actuation_planning_queue,
        queue_feedback_candidate_actuation_planning_queue_blockers,
    ) = build_queue_feedback_candidate_actuation_planning_queue(
        queue_feedback_replan_policy,
        lane_id=feedback_lane_id,
        source_policy_path=_display_path(queue_feedback_replan_policy_path),
    )
    if queue_feedback_candidate_actuation_planning_queue is not None:
        _write_json(
            queue_feedback_candidate_actuation_planning_queue_path,
            queue_feedback_candidate_actuation_planning_queue,
        )
        queue_feedback_candidate_actuation_planning_staircase_artifacts = (
            _build_queue_feedback_staircase_artifacts(
                args,
                run_dir=run_dir,
                parent_queue=queue,
                parent_queue_path=execution_queue,
                parent_state_path=state_path,
                child_queue=queue_feedback_candidate_actuation_planning_queue,
                child_queue_path=queue_feedback_candidate_actuation_planning_queue_path,
                artifact_stem="queue_feedback_candidate_actuation_planning",
            )
        )
        if staircase_artifacts is not None:
            staircase_artifacts["feedback_candidate_actuation_planning_child"] = (
                queue_feedback_candidate_actuation_planning_staircase_artifacts
            )
    payload["queue_feedback_replan_policy_path"] = _display_path(
        queue_feedback_replan_policy_path
    )
    payload["queue_feedback_replan_policy"] = queue_feedback_replan_policy
    payload["queue_feedback_replan_policy_decision"] = queue_feedback_replan_policy[
        "decision"
    ]
    payload["queue_feedback_replan_policy_should_continue"] = (
        queue_feedback_replan_policy["should_continue_feedback_loop"]
    )
    payload["queue_feedback_candidate_widening_queue_path"] = (
        _display_path(queue_feedback_candidate_widening_queue_path)
        if queue_feedback_candidate_widening_queue is not None
        else None
    )
    payload["queue_feedback_candidate_widening_queue_emitted"] = (
        queue_feedback_candidate_widening_queue is not None
    )
    payload["queue_feedback_candidate_widening_queue_blockers"] = (
        queue_feedback_candidate_widening_queue_blockers
    )
    payload["queue_feedback_candidate_widening_staircase_artifacts"] = (
        queue_feedback_candidate_widening_staircase_artifacts
    )
    payload["queue_feedback_candidate_actuation_planning_queue_path"] = (
        _display_path(queue_feedback_candidate_actuation_planning_queue_path)
        if queue_feedback_candidate_actuation_planning_queue is not None
        else None
    )
    payload["queue_feedback_candidate_actuation_planning_queue_emitted"] = (
        queue_feedback_candidate_actuation_planning_queue is not None
    )
    payload["queue_feedback_candidate_actuation_planning_queue_blockers"] = (
        queue_feedback_candidate_actuation_planning_queue_blockers
    )
    payload["queue_feedback_candidate_actuation_planning_staircase_artifacts"] = (
        queue_feedback_candidate_actuation_planning_staircase_artifacts
    )
    payload["queue_observation_recovery_queue_path"] = (
        _display_path(queue_observation_recovery_queue_path)
        if queue_observation_recovery_queue is not None
        else None
    )
    payload["queue_observation_recovery_queue_state_path"] = (
        _display_path(queue_observation_recovery_queue_state_path)
    )
    payload["queue_observation_recovery_queue_emitted"] = (
        queue_observation_recovery_queue is not None
    )
    payload["queue_observation_recovery_queue_blockers"] = (
        queue_observation_recovery_queue_blockers
    )
    payload["queue_observation_recovery_policy"] = queue_observation_recovery_policy
    payload["queue_observation_recovery_execution_policy"] = (
        queue_observation_recovery_policy
    )
    payload["queue_observation_recovery_policy_enabled"] = bool(
        args.queue_observation_recovery_policy_local_autopilot
    )
    payload["queue_observation_recovery_policy_blockers"] = (
        queue_observation_recovery_policy_blockers
    )
    payload["queue_observation_recovery_execution_requested"] = bool(
        args.execute_queue_observation_recovery
        or args.queue_observation_recovery_policy_local_autopilot
    )
    payload["queue_observation_recovery_executed"] = (
        queue_observation_recovery_execution_attempted
    )
    payload["queue_observation_recovery_execution_success"] = (
        None
        if queue_observation_recovery_execution is None
        else bool(queue_observation_recovery_execution.get("success"))
    )
    payload["queue_observation_recovery_execution"] = (
        queue_observation_recovery_execution
    )
    payload["queue_observation_recovery_execution_state_path"] = (
        None
        if queue_observation_recovery_execution is None
        else queue_observation_recovery_execution.get("state_path")
    )
    payload["queue_observation_recovery_source_observation_after"] = (
        None
        if queue_observation_recovery_execution is None
        else queue_observation_recovery_execution.get("source_observation_after")
    )
    payload["queue_observation_recovery_staircase_artifacts"] = (
        queue_observation_recovery_staircase_artifacts
    )
    payload["post_recovery_feedback_replan"] = post_recovery_feedback_replan
    payload["post_recovery_feedback_replan_attempted"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get("attempted") is True
    )
    payload["post_recovery_feedback_replan_triggered"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get("triggered") is True
    )
    payload["post_recovery_feedback_replan_artifacts_emitted"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get("artifacts_emitted") is True
    )
    payload["post_recovery_feedback_replan_success"] = (
        None
        if post_recovery_feedback_replan is None
        else bool(post_recovery_feedback_replan.get("success"))
    )
    payload["post_recovery_feedback_replan_blockers"] = (
        []
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get("blockers", [])
    )
    payload["post_recovery_source_recovery_execution_success"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get("source_recovery_execution_success")
        is True
    )
    payload["post_recovery_queue_observation_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get("queue_observation_path")
    )
    payload["post_recovery_queue_observation_recovery_plan_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get("queue_observation_recovery_plan_path")
    )
    payload["post_recovery_queue_feedback_replan_request_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get("queue_feedback_replan_request_path")
    )
    payload["post_recovery_queue_feedback_replan_policy_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get("queue_feedback_replan_policy_path")
    )
    payload["post_recovery_queue_feedback_replan_policy_decision"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get("queue_feedback_replan_policy_decision")
    )
    payload["post_recovery_queue_feedback_replan_policy_should_continue"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get(
            "queue_feedback_replan_policy_should_continue"
        )
        is True
    )
    payload["post_recovery_queue_feedback_replan_followup_queue_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_queue_path"
        )
    )
    payload["post_recovery_queue_feedback_replan_followup_state_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_state_path"
        )
    )
    payload["post_recovery_queue_feedback_replan_followup_queue_emitted"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_queue_emitted"
        )
        is True
    )
    payload["post_recovery_queue_feedback_replan_followup_queue_blockers"] = (
        []
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_queue_blockers",
            [],
        )
    )
    payload["post_recovery_queue_feedback_replan_followup_policy_enabled"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_policy_enabled"
        )
        is True
    )
    payload["post_recovery_queue_feedback_replan_followup_policy_blockers"] = (
        []
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_policy_blockers",
            [],
        )
    )
    payload["post_recovery_queue_feedback_replan_followup_execution_requested"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_execution_requested"
        )
        is True
    )
    payload["post_recovery_queue_feedback_replan_followup_executed"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_executed"
        )
        is True
    )
    payload["post_recovery_queue_feedback_replan_followup_execution_success"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_execution_success"
        )
    )
    payload["post_recovery_queue_feedback_replan_followup_action_functional_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_followup_action_functional_path"
        )
    )
    payload["post_recovery_queue_feedback_replan_continuation_queue_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_continuation_queue_path"
        )
    )
    payload["post_recovery_queue_feedback_replan_continuation_queue_state_path"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_continuation_queue_state_path"
        )
    )
    payload["post_recovery_queue_feedback_replan_continuation_queue_emitted"] = (
        post_recovery_feedback_replan is not None
        and post_recovery_feedback_replan.get(
            "queue_feedback_replan_continuation_queue_emitted"
        )
        is True
    )
    payload["post_recovery_queue_feedback_replan_continuation_queue_blockers"] = (
        []
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_continuation_queue_blockers",
            [],
        )
    )
    payload["post_recovery_queue_feedback_replan_continuation_staircase_artifacts"] = (
        None
        if post_recovery_feedback_replan is None
        else post_recovery_feedback_replan.get(
            "queue_feedback_replan_continuation_staircase_artifacts"
        )
    )
    payload["queue_feedback_replan_continuation_queue_path"] = (
        _display_path(queue_feedback_replan_continuation_queue_path)
        if queue_feedback_replan_continuation_queue is not None
        else None
    )
    payload["queue_feedback_replan_continuation_queue_emitted"] = (
        queue_feedback_replan_continuation_queue is not None
    )
    payload["queue_feedback_replan_continuation_queue_blockers"] = (
        queue_feedback_replan_continuation_queue_blockers
    )
    payload["queue_feedback_replan_continuation_staircase_artifacts"] = (
        queue_feedback_replan_continuation_staircase_artifacts
    )
    _write_json(summary_path, payload)
    payload["summary_path"] = _display_path(summary_path)
    _json_print(payload)
    ssh_execute_returncode = ssh_execute_result.returncode if ssh_execute_result is not None else 0
    return (
        2
        if worker_result.returncode != 0
        or observe_result.returncode != 0
        or ssh_execute_returncode != 0
        or queue_feedback_replan_followup_execution_returncode != 0
        or queue_observation_recovery_execution_returncode != 0
        or post_recovery_feedback_replan_returncode != 0
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
