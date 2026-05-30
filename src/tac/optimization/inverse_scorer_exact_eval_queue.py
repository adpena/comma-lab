# SPDX-License-Identifier: MIT
"""Build exact-readiness source queues from verified inverse-scorer chains."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_contract import (
    ArchiveBoundCandidateContractError,
    archive_bound_candidate_contract_fields_for_row,
    has_archive_bound_candidate_contract_payload,
    selected_archive_bound_candidate_contract_from_payload,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.repo_io import tree_sha256

CHAIN_SCHEMA = "inverse_scorer_cell_candidate_chain_v1"
CHAIN_KIND = "inverse_scorer_cell_candidate_chain"
QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
ARCHIVE_MANIFEST_SCHEMA = "inverse_scorer_exact_eval_archive_manifest_v1"
TOOL_NAME = "tools/build_inverse_scorer_exact_eval_queue.py"
DEFAULT_LANE_ID = "lane_inverse_scorer_exact_eval_queue_bridge_20260523"
PARITY_PROBE_SCHEMA = "inverse_scorer_cell_inflate_parity_probe_v1"
PARITY_PROOF_SCOPE = "full_frame_inflate_output_tree"
PARITY_CLEARED_BLOCKER = "candidate_inflate_output_parity_missing"

SOURCE_FALSE_AUTHORITY = {
    key: value
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items()
    if key not in {"score_affecting_payload_changed", "charged_bits_changed"}
}
PARITY_REQUIRED_FALSE_AUTHORITY = tuple(PROXY_FALSE_AUTHORITY_FIELDS)
CHAIN_ALLOWED_READINESS_BLOCKERS = frozenset({"exact_auth_eval_required_before_score_claim"})
CHAIN_ALLOWED_DISPATCH_BLOCKERS = frozenset(
    {
        "inverse_scorer_cell_candidate_chain_is_not_dispatch_authorization",
        "exact_auth_eval_required_before_score_claim",
    }
)


class InverseScorerExactEvalQueueError(ValueError):
    """Raised when an inverse-scorer chain is not safe to bridge."""


@dataclass(frozen=True)
class ExactEvalQueueBuildResult:
    queue: dict[str, Any]
    archive_manifest: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dumps_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        raise InverseScorerExactEvalQueueError(f"path_outside_repo:{path}") from None


def resolve_path(value: Any, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        raise InverseScorerExactEvalQueueError(f"path_must_be_repo_relative:{value}")
    resolved = (repo_root / path).resolve(strict=False)
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        raise InverseScorerExactEvalQueueError(f"path_outside_repo:{value}") from None
    return resolved


def resolve_runtime_dir_arg(value: Path, repo_root: Path) -> Path:
    path = value if value.is_absolute() else repo_root / value
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        raise InverseScorerExactEvalQueueError(
            f"runtime_submission_dir_outside_repo:{value}"
        ) from None
    return resolved


def resolve_repo_path_arg(value: Path, repo_root: Path, *, label: str) -> Path:
    path = value if value.is_absolute() else repo_root / value
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        raise InverseScorerExactEvalQueueError(f"{label}_outside_repo:{value}") from None
    return resolved


def file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InverseScorerExactEvalQueueError(f"file missing: {path}")
    if path.is_symlink():
        raise InverseScorerExactEvalQueueError(f"file must not be symlink: {path}")
    return {
        "path": repo_relative(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _required_file_from_record(record: Mapping[str, Any], repo_root: Path, label: str) -> Path:
    path = resolve_path(record.get("path"), repo_root)
    if path is None:
        raise InverseScorerExactEvalQueueError(f"{label}_path_missing")
    if not path.is_file():
        raise InverseScorerExactEvalQueueError(f"{label}_file_missing:{path}")
    if path.is_symlink():
        raise InverseScorerExactEvalQueueError(f"{label}_file_is_symlink:{path}")
    expected_sha = record.get("sha256")
    if isinstance(expected_sha, str) and len(expected_sha) == 64 and sha256_file(path) != expected_sha:
        raise InverseScorerExactEvalQueueError(f"{label}_sha256_mismatch")
    expected_bytes = record.get("bytes")
    if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
        raise InverseScorerExactEvalQueueError(f"{label}_bytes_mismatch")
    return path


def _record_matches(
    proof_record: Mapping[str, Any],
    expected_record: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key in ("path", "sha256", "bytes"):
        if proof_record.get(key) != expected_record.get(key):
            raise InverseScorerExactEvalQueueError(f"{label}_{key}_mismatch")


def _false_authority_fields_ok(payload: Mapping[str, Any], *, label: str) -> None:
    for key, value in SOURCE_FALSE_AUTHORITY.items():
        if payload.get(key) is not value:
            raise InverseScorerExactEvalQueueError(f"{label}_{key}_not_{str(value).lower()}")


def _validate_output_tree(
    tree: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(tree, Mapping):
        raise InverseScorerExactEvalQueueError(f"{label}_missing")
    if tree.get("exists") is not True:
        raise InverseScorerExactEvalQueueError(f"{label}_not_existing")
    if tree.get("blockers"):
        raise InverseScorerExactEvalQueueError(f"{label}_has_blockers")
    if not isinstance(tree.get("files"), list) or not tree["files"]:
        raise InverseScorerExactEvalQueueError(f"{label}_files_missing")
    if not isinstance(tree.get("tree_sha256"), str) or len(tree["tree_sha256"]) != 64:
        raise InverseScorerExactEvalQueueError(f"{label}_tree_sha256_missing")
    if not isinstance(tree.get("file_count"), int) or tree["file_count"] <= 0:
        raise InverseScorerExactEvalQueueError(f"{label}_file_count_missing")
    if not isinstance(tree.get("total_bytes"), int) or tree["total_bytes"] <= 0:
        raise InverseScorerExactEvalQueueError(f"{label}_total_bytes_missing")
    return tree


def _validate_inflate_run(run: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(run, Mapping):
        raise InverseScorerExactEvalQueueError(f"{label}_missing")
    if run.get("returncode") != 0:
        raise InverseScorerExactEvalQueueError(f"{label}_returncode_nonzero")
    if run.get("full_frame_file_list_claim") is not True:
        raise InverseScorerExactEvalQueueError(f"{label}_not_full_frame")
    if not isinstance(run.get("file_list_entries"), list) or not run["file_list_entries"]:
        raise InverseScorerExactEvalQueueError(f"{label}_file_list_entries_missing")
    return run


def _candidate_archive_record(chain: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    record = chain.get("candidate_archive")
    if not isinstance(record, Mapping):
        raise InverseScorerExactEvalQueueError("candidate_archive_missing")
    path = _required_file_from_record(record, repo_root, "candidate_archive")
    member = _validated_archive_member(path, record, label="candidate_archive")
    out = dict(record)
    out["path"] = repo_relative(path, repo_root)
    out["bytes"] = path.stat().st_size
    out["sha256"] = sha256_file(path)
    out.update(member)
    return out


def _validated_archive_member(
    path: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    member_name = record.get("member_name")
    if not isinstance(member_name, str) or not member_name:
        raise InverseScorerExactEvalQueueError(f"{label}_member_name_missing")
    try:
        with zipfile.ZipFile(path) as archive:
            try:
                info = archive.getinfo(member_name)
                with archive.open(info) as member:
                    payload = member.read()
            except KeyError as exc:
                raise InverseScorerExactEvalQueueError(
                    f"{label}_member_missing:{member_name}"
                ) from exc
    except zipfile.BadZipFile as exc:
        raise InverseScorerExactEvalQueueError(f"{label}_bad_zip") from exc

    expected_bytes = record.get("member_bytes")
    if isinstance(expected_bytes, int) and len(payload) != expected_bytes:
        raise InverseScorerExactEvalQueueError(f"{label}_member_bytes_mismatch")
    expected_sha = record.get("member_sha256")
    observed_sha = hashlib.sha256(payload).hexdigest()
    if (
        isinstance(expected_sha, str)
        and len(expected_sha) == 64
        and observed_sha != expected_sha
    ):
        raise InverseScorerExactEvalQueueError(f"{label}_member_sha256_mismatch")
    return {
        "member_name": member_name,
        "member_bytes": len(payload),
        "member_sha256": observed_sha,
        "member_compressed_bytes": info.compress_size,
    }


def _argv_value(argv: Sequence[Any], flag: str) -> str | None:
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            value = argv[index + 1]
            return str(value) if value else None
    return None


def _source_archive_record(chain: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    record = chain.get("source_archive") or chain.get("template_archive")
    if isinstance(record, Mapping) and record.get("path"):
        path = _required_file_from_record(record, repo_root, "source_archive")
        member = _validated_archive_member(path, record, label="source_archive")
        return {
            "path": repo_relative(path, repo_root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            **member,
        }
    tool_run = chain.get("tool_run_manifest")
    argv = tool_run.get("argv") if isinstance(tool_run, Mapping) else None
    path_text = _argv_value(argv, "--candidate-archive-template") if isinstance(argv, Sequence) else None
    path = resolve_path(path_text, repo_root)
    if path is None or not path.is_file():
        raise InverseScorerExactEvalQueueError("source_archive_path_missing")
    if path.is_symlink():
        raise InverseScorerExactEvalQueueError(f"source_archive_file_is_symlink:{path}")
    out = file_record(path, repo_root)
    try:
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if len(members) == 1:
                member = members[0]
                payload = archive.read(member)
                out.update(
                    {
                        "member_name": member.filename,
                        "member_bytes": len(payload),
                        "member_sha256": hashlib.sha256(payload).hexdigest(),
                        "member_compressed_bytes": member.compress_size,
                    }
                )
    except zipfile.BadZipFile as exc:
        raise InverseScorerExactEvalQueueError("source_archive_bad_zip") from exc
    return out


def _parity_step(chain: Mapping[str, Any]) -> Mapping[str, Any]:
    steps = chain.get("chain_steps")
    if not isinstance(steps, list):
        raise InverseScorerExactEvalQueueError("chain_steps_missing")
    for step in steps:
        if isinstance(step, Mapping) and step.get("step_id") == "build_inflate_parity_probe":
            return step
    raise InverseScorerExactEvalQueueError("inflate_parity_step_missing")


def _step_by_id(chain: Mapping[str, Any], step_id: str) -> Mapping[str, Any] | None:
    steps = chain.get("chain_steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, Mapping) and step.get("step_id") == step_id:
            return step
    return None


def _require_no_unresolved_chain_blockers(chain: Mapping[str, Any]) -> None:
    readiness = {str(item) for item in chain.get("readiness_blockers") or [] if str(item)}
    extra_readiness = sorted(readiness - CHAIN_ALLOWED_READINESS_BLOCKERS)
    if extra_readiness:
        raise InverseScorerExactEvalQueueError(
            "chain_unresolved_readiness_blockers:" + ",".join(extra_readiness)
        )
    if "exact_auth_eval_required_before_score_claim" not in readiness:
        raise InverseScorerExactEvalQueueError("exact_auth_boundary_missing")
    dispatch = {str(item) for item in chain.get("dispatch_blockers") or [] if str(item)}
    extra_dispatch = sorted(dispatch - CHAIN_ALLOWED_DISPATCH_BLOCKERS)
    if extra_dispatch:
        raise InverseScorerExactEvalQueueError(
            "chain_unresolved_dispatch_blockers:" + ",".join(extra_dispatch)
        )


def _validate_parity_payload_basics(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != PARITY_PROBE_SCHEMA:
        raise InverseScorerExactEvalQueueError("inflate_parity_payload_schema_mismatch")
    for key in (
        "full_frame_inflate_output_parity_claim",
        "output_bytes_identical",
        "output_contract_nonempty",
        "output_contract_paths_match",
    ):
        if payload.get(key) is not True:
            raise InverseScorerExactEvalQueueError(f"inflate_parity_payload_{key}_not_true")
    if payload.get("proof_scope") != "full_frame_inflate_output_tree":
        raise InverseScorerExactEvalQueueError("inflate_parity_payload_scope_not_full_frame")
    if payload.get("differing_path_count") not in (0, None):
        raise InverseScorerExactEvalQueueError("inflate_parity_payload_has_differing_paths")
    for key in ("blockers", "missing_from_candidate", "extra_in_candidate"):
        if payload.get(key):
            raise InverseScorerExactEvalQueueError(f"inflate_parity_payload_{key}_not_empty")
    for key in PARITY_REQUIRED_FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise InverseScorerExactEvalQueueError(
                f"inflate_parity_payload_{key}_not_false"
            )
    source_tree = payload.get("source_output_tree")
    candidate_tree = payload.get("candidate_output_tree")
    if not isinstance(source_tree, Mapping) or not isinstance(candidate_tree, Mapping):
        raise InverseScorerExactEvalQueueError("inflate_parity_output_tree_missing")
    for label, tree in (("source", source_tree), ("candidate", candidate_tree)):
        if tree.get("exists") is not True:
            raise InverseScorerExactEvalQueueError(
                f"inflate_parity_{label}_output_tree_missing"
            )
        if tree.get("blockers"):
            raise InverseScorerExactEvalQueueError(
                f"inflate_parity_{label}_output_tree_has_blockers"
            )
        if not isinstance(tree.get("file_count"), int) or tree["file_count"] <= 0:
            raise InverseScorerExactEvalQueueError(
                f"inflate_parity_{label}_output_tree_empty"
            )
        if not isinstance(tree.get("total_bytes"), int) or tree["total_bytes"] <= 0:
            raise InverseScorerExactEvalQueueError(
                f"inflate_parity_{label}_output_tree_empty"
            )
    if source_tree.get("tree_sha256") != candidate_tree.get("tree_sha256"):
        raise InverseScorerExactEvalQueueError("inflate_parity_output_tree_sha_mismatch")


def _validate_parity_payload_linkage(
    payload: Mapping[str, Any],
    *,
    candidate_archive: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    runtime_submission: Mapping[str, Any],
) -> None:
    for label, expected in (
        ("candidate_archive", candidate_archive),
        ("source_archive", source_archive),
    ):
        observed = payload.get(label)
        if not isinstance(observed, Mapping):
            raise InverseScorerExactEvalQueueError(f"inflate_parity_{label}_missing")
        if observed.get("sha256") != expected.get("sha256"):
            raise InverseScorerExactEvalQueueError(f"inflate_parity_{label}_sha_mismatch")
        if observed.get("bytes") != expected.get("bytes"):
            raise InverseScorerExactEvalQueueError(f"inflate_parity_{label}_bytes_mismatch")
    runtime = payload.get("inflate_runtime")
    if not isinstance(runtime, Mapping):
        raise InverseScorerExactEvalQueueError("inflate_parity_runtime_missing")
    if runtime.get("path") != runtime_submission.get("path"):
        raise InverseScorerExactEvalQueueError("inflate_parity_runtime_path_mismatch")
    if runtime.get("inflate_sh_sha256") != runtime_submission["inflate_sh"]["sha256"]:
        raise InverseScorerExactEvalQueueError("inflate_parity_runtime_inflate_sh_mismatch")


def _load_parity_payload(step: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    artifact = step.get("artifact")
    if not isinstance(artifact, Mapping):
        raise InverseScorerExactEvalQueueError("inflate_parity_step_artifact_missing")
    artifact_path = _required_file_from_record(
        artifact, repo_root, "inflate_parity_step_artifact"
    )
    raw = read_json(artifact_path)
    if not isinstance(raw, dict):
        raise InverseScorerExactEvalQueueError("inflate_parity_payload_not_object")
    _validate_parity_payload_basics(raw)
    return raw


def _archive_bound_contract_for_chain(
    chain: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not has_archive_bound_candidate_contract_payload(chain):
        return {}
    try:
        contract = selected_archive_bound_candidate_contract_from_payload(chain, label=label)
    except ArchiveBoundCandidateContractError as exc:
        raise InverseScorerExactEvalQueueError(
            f"archive_bound_candidate_contract_invalid:{exc}"
        ) from exc
    return dict(contract or {})


def _validate_chain(chain: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    if chain.get("schema") != CHAIN_SCHEMA:
        raise InverseScorerExactEvalQueueError("chain_schema_mismatch")
    archive_bound_contract = _archive_bound_contract_for_chain(
        chain,
        label="inverse_scorer_chain_manifest",
    )
    if archive_bound_contract:
        if archive_bound_contract.get("byte_closed_candidate_materialized") is not True:
            raise InverseScorerExactEvalQueueError(
                "archive_bound_candidate_contract_not_materialized"
            )
        if archive_bound_contract.get("receiver_contract_satisfied") is not True:
            raise InverseScorerExactEvalQueueError(
                "archive_bound_candidate_contract_receiver_not_satisfied"
            )
        if archive_bound_contract.get("runtime_consumption_proof_ready") is not True:
            raise InverseScorerExactEvalQueueError(
                "archive_bound_candidate_contract_runtime_proof_missing"
            )
    if chain.get("byte_closed_candidate_emitted") is not True:
        raise InverseScorerExactEvalQueueError("byte_closed_candidate_not_emitted")
    if chain.get("receiver_contract_satisfied") is not True:
        raise InverseScorerExactEvalQueueError("receiver_contract_not_satisfied")
    if chain.get("inflate_parity_satisfied") is not True:
        raise InverseScorerExactEvalQueueError("inflate_parity_not_satisfied")
    _require_no_unresolved_chain_blockers(chain)
    for key in SOURCE_FALSE_AUTHORITY:
        if chain.get(key) is not False:
            raise InverseScorerExactEvalQueueError(f"chain_{key}_not_false")
    runtime_step = _step_by_id(chain, "build_runtime_adapter")
    if not isinstance(runtime_step, Mapping):
        raise InverseScorerExactEvalQueueError("runtime_adapter_step_missing")
    if runtime_step.get("status") != "succeeded":
        raise InverseScorerExactEvalQueueError("runtime_adapter_step_not_succeeded")
    runtime_tree = runtime_step.get("runtime_tree_sha256")
    if not isinstance(runtime_tree, str) or len(runtime_tree) != 64:
        raise InverseScorerExactEvalQueueError("runtime_adapter_tree_sha_missing")
    if runtime_step.get("readiness_blockers"):
        raise InverseScorerExactEvalQueueError("runtime_adapter_step_has_blockers")
    step = _parity_step(chain)
    if step.get("schema") != PARITY_PROBE_SCHEMA:
        raise InverseScorerExactEvalQueueError("inflate_parity_step_schema_mismatch")
    if step.get("status") != "succeeded":
        raise InverseScorerExactEvalQueueError("inflate_parity_step_not_succeeded")
    if step.get("full_frame_inflate_output_parity_claim") is not True:
        raise InverseScorerExactEvalQueueError("inflate_parity_step_not_full_frame")
    if step.get("blockers"):
        raise InverseScorerExactEvalQueueError("inflate_parity_step_has_blockers")
    return _load_parity_payload(step, repo_root)


def _validate_source_candidate_diff(
    candidate_archive: Mapping[str, Any],
    source_archive: Mapping[str, Any],
) -> None:
    if candidate_archive["sha256"] == source_archive["sha256"]:
        raise InverseScorerExactEvalQueueError("source_candidate_archive_sha256_unchanged")


def _validate_inflate_parity_artifact(
    chain: Mapping[str, Any],
    *,
    candidate_archive: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    repo_root: Path,
) -> None:
    step = _parity_step(chain)
    artifact = step.get("artifact")
    if not isinstance(artifact, Mapping):
        raise InverseScorerExactEvalQueueError("inflate_parity_step_artifact_missing")
    artifact_path = _required_file_from_record(
        artifact,
        repo_root,
        "inflate_parity_step_artifact",
    )
    raw = read_json(artifact_path)
    if not isinstance(raw, Mapping):
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_not_object")
    if raw.get("schema") != PARITY_PROBE_SCHEMA:
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_schema_mismatch")
    if raw.get("proof_scope") != PARITY_PROOF_SCOPE:
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_scope_mismatch")
    if raw.get("full_frame_inflate_output_parity_claim") is not True:
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_not_full_frame")
    if raw.get("expect_output_byte_identical") is not True:
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_expected_identity_missing")
    if raw.get("output_bytes_identical") is not True:
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_output_bytes_not_identical")
    if raw.get("output_contract_nonempty") is not True:
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_output_contract_empty")
    if raw.get("output_contract_paths_match") is not True:
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_output_paths_mismatch")
    if raw.get("differing_path_count") != 0:
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_differing_paths")
    for key in ("blockers", "missing_from_candidate", "extra_in_candidate"):
        value = raw.get(key)
        if not isinstance(value, list):
            raise InverseScorerExactEvalQueueError(f"inflate_parity_artifact_{key}_not_list")
        if value:
            raise InverseScorerExactEvalQueueError(f"inflate_parity_artifact_{key}_not_empty")
    if PARITY_CLEARED_BLOCKER not in set(raw.get("cleared_blockers") or []):
        raise InverseScorerExactEvalQueueError("inflate_parity_artifact_cleared_blocker_missing")
    dispatch_blockers = {str(item) for item in raw.get("dispatch_blockers") or []}
    for required in (
        "inverse_scorer_cell_inflate_parity_is_not_score_authority",
        "exact_auth_eval_required_before_score_claim",
    ):
        if required not in dispatch_blockers:
            raise InverseScorerExactEvalQueueError(
                f"inflate_parity_artifact_dispatch_blocker_missing:{required}"
            )
    _false_authority_fields_ok(raw, label="inflate_parity_artifact")
    _record_matches(
        raw.get("candidate_archive") or {},
        candidate_archive,
        label="inflate_parity_artifact_candidate_archive",
    )
    _record_matches(
        raw.get("source_archive") or {},
        source_archive,
        label="inflate_parity_artifact_source_archive",
    )
    _record_matches(
        raw.get("candidate_archive_inflated") or {},
        candidate_archive,
        label="inflate_parity_artifact_candidate_archive_inflated",
    )
    _record_matches(
        raw.get("source_archive_inflated") or {},
        source_archive,
        label="inflate_parity_artifact_source_archive_inflated",
    )
    source_run = _validate_inflate_run(raw.get("source_inflate_run"), label="source_inflate_run")
    candidate_run = _validate_inflate_run(
        raw.get("candidate_inflate_run"),
        label="candidate_inflate_run",
    )
    if source_run["file_list_entries"] != candidate_run["file_list_entries"]:
        raise InverseScorerExactEvalQueueError("inflate_parity_file_list_entries_mismatch")
    source_tree = _validate_output_tree(raw.get("source_output_tree"), label="source_output_tree")
    candidate_tree = _validate_output_tree(
        raw.get("candidate_output_tree"),
        label="candidate_output_tree",
    )
    for key in ("tree_sha256", "file_count", "total_bytes", "files"):
        if source_tree.get(key) != candidate_tree.get(key):
            raise InverseScorerExactEvalQueueError(f"inflate_parity_output_tree_{key}_mismatch")


def _runtime_submission_record(runtime_submission_dir: Path, repo_root: Path) -> dict[str, Any]:
    runtime = resolve_runtime_dir_arg(runtime_submission_dir, repo_root)
    if not runtime.is_dir():
        raise InverseScorerExactEvalQueueError(f"runtime_submission_dir_missing:{runtime}")
    if runtime.is_symlink():
        raise InverseScorerExactEvalQueueError(f"runtime_submission_dir_is_symlink:{runtime}")
    inflate_sh = runtime / "inflate.sh"
    report = runtime / "report.txt"
    if not inflate_sh.is_file():
        raise InverseScorerExactEvalQueueError("runtime_inflate_sh_missing")
    if not report.is_file():
        raise InverseScorerExactEvalQueueError("runtime_report_txt_missing")
    return {
        "path": repo_relative(runtime, repo_root),
        "inflate_sh": file_record(inflate_sh, repo_root),
        "report_txt": file_record(report, repo_root),
        "runtime_tree_sha256": tree_sha256(runtime),
    }


def _sanitized_chain_steps(chain: Mapping[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    steps = chain.get("chain_steps")
    if not isinstance(steps, list):
        return out
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        item: dict[str, Any] = {}
        for key in ("step_id", "status", "schema"):
            if key in step:
                item[key] = step[key]
        artifact = step.get("artifact")
        if isinstance(artifact, Mapping):
            path = _required_file_from_record(
                artifact, repo_root, str(step.get("step_id") or "chain_step_artifact")
            )
            item["artifact"] = file_record(path, repo_root)
        if step.get("step_id") == "build_inflate_parity_probe":
            item["full_frame_inflate_output_parity_claim"] = True
            item["blockers"] = []
        if step.get("step_id") == "build_runtime_adapter":
            item["runtime_tree_sha256"] = step.get("runtime_tree_sha256")
        out.append(item)
    return out


def build_archive_manifest(
    *,
    chain_manifest_path: Path,
    chain: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    runtime_submission: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    member = {
        "name": candidate_archive["member_name"],
        "sha256": candidate_archive["member_sha256"],
        "bytes": candidate_archive["member_bytes"],
        "compressed_bytes": candidate_archive["member_compressed_bytes"],
    }
    archive_changed = candidate_archive["sha256"] != source_archive["sha256"]
    byte_changed = candidate_archive["bytes"] != source_archive["bytes"]
    return {
        **SOURCE_FALSE_AUTHORITY,
        "schema": ARCHIVE_MANIFEST_SCHEMA,
        "generated_at_utc": utc_now(),
        "tool": TOOL_NAME,
        "source_chain_manifest": file_record(chain_manifest_path, repo_root),
        "runtime_submission": dict(runtime_submission),
        "source_archive": dict(source_archive),
        "candidate_archive": dict(candidate_archive),
        "archive_sha256": candidate_archive["sha256"],
        "archive_bytes": candidate_archive["bytes"],
        "candidate_archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "members": [member],
        "source_member": {
            "name": source_archive.get("member_name"),
            "sha256": source_archive.get("member_sha256"),
            "bytes": source_archive.get("member_bytes"),
            "compressed_bytes": source_archive.get("member_compressed_bytes"),
        },
        "receiver_contract_satisfied": chain.get("receiver_contract_satisfied") is True,
        "inflate_parity_satisfied": chain.get("inflate_parity_satisfied") is True,
        "exact_auth_eval_required_before_score_claim": True,
        "score_affecting_payload_changed": archive_changed,
        "charged_bits_changed": byte_changed,
        "score_affecting_change_proof": {
            "source_archive_sha256": source_archive["sha256"],
            "candidate_archive_sha256": candidate_archive["sha256"],
            "source_archive_bytes": source_archive["bytes"],
            "candidate_archive_bytes": candidate_archive["bytes"],
            "archive_changed": archive_changed,
            "byte_different": byte_changed,
        },
    }


def build_source_queue(
    *,
    chain_manifest_path: Path,
    chain: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    runtime_submission: Mapping[str, Any],
    parity_payload: Mapping[str, Any],
    archive_manifest_path: Path,
    repo_root: Path,
    candidate_id: str | None,
    lane_id: str,
) -> dict[str, Any]:
    candidate_id = candidate_id or f"ias1_chain_{str(candidate_archive['sha256'])[:12]}"
    archive_changed = candidate_archive["sha256"] != source_archive["sha256"]
    byte_changed = candidate_archive["bytes"] != source_archive["bytes"]
    chain_steps = _sanitized_chain_steps(chain, repo_root)
    runtime_consumption_proof_path = None
    for step in chain_steps:
        if step.get("step_id") == "build_inflate_parity_probe":
            artifact = step.get("artifact")
            if isinstance(artifact, Mapping):
                runtime_consumption_proof_path = artifact.get("path")
            break
    row = {
        **SOURCE_FALSE_AUTHORITY,
        "candidate_id": candidate_id,
        "lane_id": lane_id,
        "schema": CHAIN_SCHEMA,
        "kind": CHAIN_KIND,
        "candidate_family": "inverse_scorer_cell",
        "optimizer_tool": TOOL_NAME,
        "candidate_archive_path": candidate_archive["path"],
        "archive_path": candidate_archive["path"],
        "candidate_archive_sha256": candidate_archive["sha256"],
        "archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "archive_bytes": candidate_archive["bytes"],
        "source_archive_path": source_archive["path"],
        "source_archive_sha256": source_archive["sha256"],
        "source_archive_bytes": source_archive["bytes"],
        "archive_manifest_path": repo_relative(archive_manifest_path, repo_root),
        "runtime_submission_dir": runtime_submission["path"],
        "runtime_tree_sha256": runtime_submission["runtime_tree_sha256"],
        "score_affecting_payload_changed": archive_changed,
        "charged_bits_changed": byte_changed,
        "score_affecting_change_proof": {
            "source_archive_sha256": source_archive["sha256"],
            "candidate_archive_sha256": candidate_archive["sha256"],
            "source_archive_bytes": source_archive["bytes"],
            "candidate_archive_bytes": candidate_archive["bytes"],
            "archive_changed": archive_changed,
            "byte_different": byte_changed,
        },
        "byte_closed_candidate_emitted": chain.get("byte_closed_candidate_emitted") is True,
        "runtime_consumption_proof_ready": True,
        "runtime_consumption_proof_path": runtime_consumption_proof_path,
        "receiver_contract_satisfied": chain.get("receiver_contract_satisfied") is True,
        "inflate_parity_satisfied": chain.get("inflate_parity_satisfied") is True,
        "runtime_adapter_ready": True,
        "contest_runtime_decoder_adapter_ready": True,
        "readiness_blockers": list(chain.get("readiness_blockers") or []),
        "dispatch_blockers": list(chain.get("dispatch_blockers") or []),
        "next_required_gates": list(chain.get("next_required_gates") or []),
        "chain_steps": chain_steps,
        "inflate_parity_probe": {
            "schema": parity_payload.get("schema"),
            "proof_scope": parity_payload.get("proof_scope"),
            "output_bytes_identical": parity_payload.get("output_bytes_identical"),
            "output_contract_nonempty": parity_payload.get("output_contract_nonempty"),
            "output_tree_sha256": parity_payload.get("candidate_output_tree", {}).get(
                "tree_sha256"
            )
            if isinstance(parity_payload.get("candidate_output_tree"), Mapping)
            else None,
        },
        "source_paths": {
            "chain_manifest": repo_relative(chain_manifest_path, repo_root),
            "archive_manifest": repo_relative(archive_manifest_path, repo_root),
            "runtime_submission_dir": runtime_submission["path"],
        },
    }
    row.update(
        archive_bound_candidate_contract_fields_for_row(
            row,
            repo_root=repo_root,
            selected_transform_kind="inverse_scorer_exact_eval_archive_bridge_v1",
            family_id="inverse_scorer_cell",
            candidate_chain_id=candidate_id,
            entropy_position_label="after_entropy_coder",
        )
    )
    return {
        "schema": QUEUE_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "source_chain_manifest": repo_relative(chain_manifest_path, repo_root),
        "archive_manifest_path": repo_relative(archive_manifest_path, repo_root),
        "runtime_submission_dir": runtime_submission["path"],
        "top_k_count": 1,
        "dispatch_ready_count": 0,
        "top_k": [row],
        "dispatch_ready": [],
        "evidence_boundary": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "exact_auth_eval_required_before_score_claim": True,
            "lane_dispatch_claim_required_before_gpu_or_remote_eval": True,
        },
    }


def build_inverse_scorer_exact_eval_source_queue(
    *,
    chain_manifest_path: Path,
    runtime_submission_dir: Path,
    archive_manifest_path: Path,
    repo_root: Path,
    candidate_id: str | None = None,
    lane_id: str = DEFAULT_LANE_ID,
) -> ExactEvalQueueBuildResult:
    chain_path = resolve_repo_path_arg(
        chain_manifest_path,
        repo_root,
        label="chain_manifest",
    )
    raw = read_json(chain_path)
    if not isinstance(raw, dict):
        raise InverseScorerExactEvalQueueError("chain_manifest_not_object")
    parity_payload = _validate_chain(raw, repo_root)
    candidate_archive = _candidate_archive_record(raw, repo_root)
    source_archive = _source_archive_record(raw, repo_root)
    _validate_source_candidate_diff(candidate_archive, source_archive)
    _validate_inflate_parity_artifact(
        raw,
        candidate_archive=candidate_archive,
        source_archive=source_archive,
        repo_root=repo_root,
    )
    runtime_submission = _runtime_submission_record(runtime_submission_dir, repo_root)
    _validate_parity_payload_linkage(
        parity_payload,
        candidate_archive=candidate_archive,
        source_archive=source_archive,
        runtime_submission=runtime_submission,
    )
    manifest = build_archive_manifest(
        chain_manifest_path=chain_path,
        chain=raw,
        candidate_archive=candidate_archive,
        source_archive=source_archive,
        runtime_submission=runtime_submission,
        repo_root=repo_root,
    )
    queue = build_source_queue(
        chain_manifest_path=chain_path,
        chain=raw,
        candidate_archive=candidate_archive,
        source_archive=source_archive,
        runtime_submission=runtime_submission,
        parity_payload=parity_payload,
        archive_manifest_path=archive_manifest_path,
        repo_root=repo_root,
        candidate_id=candidate_id,
        lane_id=lane_id,
    )
    return ExactEvalQueueBuildResult(queue=queue, archive_manifest=manifest)


__all__ = [
    "ARCHIVE_MANIFEST_SCHEMA",
    "CHAIN_SCHEMA",
    "DEFAULT_LANE_ID",
    "QUEUE_SCHEMA",
    "ExactEvalQueueBuildResult",
    "InverseScorerExactEvalQueueError",
    "build_inverse_scorer_exact_eval_source_queue",
    "dumps_json",
]
