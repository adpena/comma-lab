# SPDX-License-Identifier: MIT
"""Executable byte-range entropy-recode proof chain.

This module binds the PR103 arithmetic materializer, PR103 runtime adapter, and
byte-range receiver proof into one local, fail-closed artifact. It is a queue
actuator surface, not score authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.hnerv_pr103_lc_ac_schema import (
    AC_STREAM_SPECS,
    HI_SYMBOL_COUNT,
    PUBLIC_PR103_LAYOUT,
    Pr103LcAcLayout,
)
from tac.optimization.byte_range_entropy_recode_materializer import (
    FALSE_AUTHORITY,
    build_byte_range_entropy_recode_receiver_proof,
    materialize_byte_range_entropy_recode_candidate,
    verify_byte_range_entropy_recode_candidate_manifest,
)
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)
from tac.pr103_arithmetic_transform_plan import Pr103ArithmeticTransformPlanError
from tac.pr103_lc_ac_runtime_adapter import (
    Pr103RuntimeAdapterError,
    build_pr103_lc_ac_runtime_adapter,
)
from tac.repo_io import (
    ArtifactWriteError,
    repo_relative,
    sha256_file,
    write_json_artifact,
)

CHAIN_SCHEMA = "byte_range_entropy_recode_chain_v1"
CHAIN_MANIFEST_NAME = "byte_range_entropy_recode_chain_manifest.json"
BYTE_RANGE_CANDIDATE_MANIFEST_NAME = "byte_range_candidate_manifest.json"
PR103_CANDIDATE_MANIFEST_NAME = "pr103_candidate_manifest.json"
RUNTIME_ADAPTER_MANIFEST_NAME = "pr103_runtime_adapter_manifest.json"
RECEIVER_PROOF_NAME = "byte_range_receiver_proof.json"
VERIFIED_CANDIDATE_NAME = "byte_range_verified_candidate.json"
CANDIDATE_ARCHIVE_NAME = "candidate_archive.zip"
RUNTIME_ADAPTER_DIR_NAME = "runtime_adapter"


class ByteRangeEntropyRecodeChainError(ValueError):
    """Raised when the executable byte-range recode chain cannot run safely."""


def build_byte_range_entropy_recode_chain(
    *,
    schema_manifest: str | Path | Mapping[str, Any],
    beam_probe_reports: Sequence[str | Path | Mapping[str, Any]],
    source_runtime_dir: str | Path,
    output_dir: str | Path,
    source_archive: str | Path | None = None,
    global_combo_report: str | Path | Mapping[str, Any] | None = None,
    member_name: str | None = None,
    repo_root: str | Path | None = None,
    layout: Pr103LcAcLayout = PUBLIC_PR103_LAYOUT,
    stream_specs: Sequence[tuple[str, int, int | None]] = AC_STREAM_SPECS,
    hi_symbol_count: int = HI_SYMBOL_COUNT,
    retune_brotli_sections: Sequence[str] = (),
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Run materialization -> adapter -> receiver proof -> verified manifest."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    out = _repo_path(Path(output_dir), repo)
    runtime_source = _repo_path(Path(source_runtime_dir), repo)
    _prepare_new_output_dir(out)

    candidate_archive_path = out / CANDIDATE_ARCHIVE_NAME
    byte_range_candidate_manifest_path = out / BYTE_RANGE_CANDIDATE_MANIFEST_NAME
    pr103_candidate_manifest_path = out / PR103_CANDIDATE_MANIFEST_NAME
    runtime_adapter_dir = out / RUNTIME_ADAPTER_DIR_NAME
    runtime_adapter_manifest_path = out / RUNTIME_ADAPTER_MANIFEST_NAME
    receiver_proof_path = out / RECEIVER_PROOF_NAME
    verified_candidate_path = out / VERIFIED_CANDIDATE_NAME
    chain_manifest_path = out / CHAIN_MANIFEST_NAME

    try:
        candidate = materialize_byte_range_entropy_recode_candidate(
            schema_manifest=schema_manifest,
            beam_probe_reports=beam_probe_reports,
            output_archive=candidate_archive_path,
            source_archive=source_archive,
            global_combo_report=global_combo_report,
            member_name=member_name,
            repo_root=repo,
            layout=layout,
            stream_specs=stream_specs,
            hi_symbol_count=hi_symbol_count,
            retune_brotli_sections=retune_brotli_sections,
        )
        _write_json(
            byte_range_candidate_manifest_path,
            candidate,
            min_free_bytes=min_free_bytes,
        )
        pr103_candidate = _mapping(candidate.get("pr103_candidate"))
        if not pr103_candidate:
            raise ByteRangeEntropyRecodeChainError(
                "byte-range candidate missing embedded PR103 candidate manifest"
            )
        _write_json(
            pr103_candidate_manifest_path,
            pr103_candidate,
            min_free_bytes=min_free_bytes,
        )

        adapter = build_pr103_lc_ac_runtime_adapter(
            candidate_manifest=pr103_candidate_manifest_path,
            source_runtime_dir=runtime_source,
            output_runtime_dir=runtime_adapter_dir,
            repo_root=repo,
        )
        _write_json(runtime_adapter_manifest_path, adapter, min_free_bytes=min_free_bytes)

        receiver_proof = build_byte_range_entropy_recode_receiver_proof(
            runtime_adapter_manifest=runtime_adapter_manifest_path,
            candidate_manifest=pr103_candidate_manifest_path,
            repo_root=repo,
        )
        _write_json(receiver_proof_path, receiver_proof, min_free_bytes=min_free_bytes)

        verified = verify_byte_range_entropy_recode_candidate_manifest(
            candidate_manifest=byte_range_candidate_manifest_path,
            runtime_consumption_proof=receiver_proof_path,
            repo_root=repo,
        )
        _write_json(verified_candidate_path, verified, min_free_bytes=min_free_bytes)
    except (
        ArtifactWriteError,
        ByteRangeEntropyRecodeChainError,
        OSError,
        Pr103ArithmeticTransformPlanError,
        Pr103RuntimeAdapterError,
        ValueError,
    ) as exc:
        raise ByteRangeEntropyRecodeChainError(str(exc)) from exc

    chain = _chain_manifest(
        repo=repo,
        output_dir=out,
        byte_range_candidate_manifest_path=byte_range_candidate_manifest_path,
        pr103_candidate_manifest_path=pr103_candidate_manifest_path,
        runtime_adapter_manifest_path=runtime_adapter_manifest_path,
        receiver_proof_path=receiver_proof_path,
        verified_candidate_path=verified_candidate_path,
        candidate=candidate,
        adapter=adapter,
        receiver_proof=receiver_proof,
        verified=verified,
    )
    _write_json(chain_manifest_path, chain, min_free_bytes=min_free_bytes)
    return chain


def _chain_manifest(
    *,
    repo: Path,
    output_dir: Path,
    byte_range_candidate_manifest_path: Path,
    pr103_candidate_manifest_path: Path,
    runtime_adapter_manifest_path: Path,
    receiver_proof_path: Path,
    verified_candidate_path: Path,
    candidate: Mapping[str, Any],
    adapter: Mapping[str, Any],
    receiver_proof: Mapping[str, Any],
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_archive = _mapping(candidate.get("candidate_archive"))
    source_archive = _mapping(candidate.get("source_archive"))
    serialized_archive_delta = build_serialized_archive_delta_contract(
        source_archive=source_archive,
        candidate_archive=candidate_archive,
        require_realized_saving=True,
    )
    adapter_blockers = _string_list(adapter.get("readiness_blockers"))
    verified_blockers = _string_list(verified.get("readiness_blockers"))
    readiness_blockers = _ordered_unique(
        [
            *verified_blockers,
            *adapter_blockers,
            *serialized_archive_delta["blockers"],
        ]
    )
    candidate_runtime_blocker_cleared = (
        "candidate_runtime_adapter_missing" not in verified_blockers
        and verified.get("receiver_contract_satisfied") is True
    )
    return {
        "schema": CHAIN_SCHEMA,
        "output_dir": repo_relative(output_dir, repo),
        "source_archive": source_archive,
        "source_archive_sha256": source_archive.get("sha256")
        or source_archive.get("archive_sha256")
        or "",
        "source_archive_bytes": source_archive.get("bytes")
        or source_archive.get("archive_bytes"),
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": candidate_archive.get("sha256") or "",
        "candidate_archive_bytes": candidate_archive.get("bytes"),
        "candidate_member_sha256": candidate_archive.get("member_sha256") or "",
        "serialized_archive_delta": serialized_archive_delta,
        "byte_closed_candidate_emitted": candidate.get("byte_closed_candidate_emitted")
        is True,
        "runtime_adapter_ready": not adapter_blockers,
        "receiver_proof_ready": receiver_proof.get("ready_for_exact_eval_runtime")
        is True,
        "receiver_contract_satisfied": verified.get("receiver_contract_satisfied")
        is True,
        "candidate_runtime_adapter_blocker_cleared": candidate_runtime_blocker_cleared,
        "full_frame_or_shell_parity_required": any(
            blocker
            in {
                "candidate_inflate_output_parity_missing",
                "full_frame_render_output_parity_missing",
                "shell_inflate_output_parity_missing",
            }
            for blocker in readiness_blockers
        ),
        "readiness_blockers": readiness_blockers,
        "dispatch_blockers": [
            "byte_range_entropy_recode_chain_is_not_dispatch_authorization",
            *readiness_blockers,
        ],
        "artifacts": {
            "byte_range_candidate_manifest": _file_record(
                byte_range_candidate_manifest_path,
                repo=repo,
            ),
            "pr103_candidate_manifest": _file_record(
                pr103_candidate_manifest_path,
                repo=repo,
            ),
            "runtime_adapter_manifest": _file_record(
                runtime_adapter_manifest_path,
                repo=repo,
            ),
            "receiver_proof": _file_record(receiver_proof_path, repo=repo),
            "verified_candidate": _file_record(verified_candidate_path, repo=repo),
        },
        "chain_steps": [
            {
                "step_id": "materialize_candidate",
                "status": "succeeded",
                "schema": candidate.get("schema"),
                "artifact": _file_record(
                    byte_range_candidate_manifest_path,
                    repo=repo,
                ),
                "pr103_candidate_artifact": _file_record(
                    pr103_candidate_manifest_path,
                    repo=repo,
                ),
                "archive": candidate_archive,
            },
            {
                "step_id": "build_runtime_adapter",
                "status": "succeeded",
                "schema": adapter.get("schema"),
                "artifact": _file_record(runtime_adapter_manifest_path, repo=repo),
                "runtime_tree_sha256": adapter.get("runtime_tree_sha256") or "",
                "readiness_blockers": adapter_blockers,
            },
            {
                "step_id": "build_receiver_proof",
                "status": "succeeded",
                "schema": receiver_proof.get("schema"),
                "artifact": _file_record(receiver_proof_path, repo=repo),
                "ready_for_exact_eval_runtime": receiver_proof.get(
                    "ready_for_exact_eval_runtime"
                )
                is True,
            },
            {
                "step_id": "verify_candidate_with_receiver_proof",
                "status": "succeeded",
                "schema": verified.get("schema"),
                "artifact": _file_record(verified_candidate_path, repo=repo),
                "receiver_contract_satisfied": verified.get(
                    "receiver_contract_satisfied"
                )
                is True,
                "readiness_blockers": verified_blockers,
            },
        ],
        "next_required_gates": _next_required_gates(readiness_blockers),
        **FALSE_AUTHORITY,
    }


def _next_required_gates(blockers: Sequence[str]) -> list[str]:
    gates: list[str] = []
    blocker_set = set(blockers)
    if {
        "candidate_inflate_output_parity_missing",
        "full_frame_render_output_parity_missing",
        "shell_inflate_output_parity_missing",
    } & blocker_set:
        gates.append("inflate_or_full_frame_parity")
    if "strict_pre_submission_compliance_json_missing" in blocker_set:
        gates.append("strict_pre_submission_compliance")
    if "lane_dispatch_claim_missing" in blocker_set:
        gates.append("lane_dispatch_claim")
    if "exact_cuda_auth_eval_missing" in blocker_set:
        gates.append("contest_auth_eval")
    return gates


def _prepare_new_output_dir(path: Path) -> None:
    if path.exists():
        raise ByteRangeEntropyRecodeChainError(
            f"output directory already exists; choose a new chain directory: {path}"
        )
    path.mkdir(parents=True)


def _write_json(path: Path, payload: Mapping[str, Any], *, min_free_bytes: int) -> None:
    write_json_artifact(path, payload, min_free_bytes=min_free_bytes)


def _file_record(path: Path, *, repo: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _repo_path(path: Path, repo: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value or [] if str(item)]


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "BYTE_RANGE_CANDIDATE_MANIFEST_NAME",
    "CANDIDATE_ARCHIVE_NAME",
    "CHAIN_MANIFEST_NAME",
    "CHAIN_SCHEMA",
    "PR103_CANDIDATE_MANIFEST_NAME",
    "RECEIVER_PROOF_NAME",
    "RUNTIME_ADAPTER_DIR_NAME",
    "RUNTIME_ADAPTER_MANIFEST_NAME",
    "VERIFIED_CANDIDATE_NAME",
    "ByteRangeEntropyRecodeChainError",
    "build_byte_range_entropy_recode_chain",
]
