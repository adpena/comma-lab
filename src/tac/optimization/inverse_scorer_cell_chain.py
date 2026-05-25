# SPDX-License-Identifier: MIT
"""Executable inverse-scorer IAS1 candidate proof chain.

This local chain materializes an IAS1 candidate, consumes the descriptor packet
through the canonical receiver adapter, builds a receiver proof, and verifies
the candidate manifest. It is queue actuator evidence, not score authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.inverse_scorer_cell_inflate_parity import (
    build_inverse_scorer_cell_inflate_parity_probe,
    build_inverse_scorer_cell_inflate_parity_probe_from_archives,
)
from tac.optimization.inverse_scorer_cell_materializer import (
    FALSE_AUTHORITY,
    build_inverse_scorer_cell_receiver_proof,
    build_inverse_scorer_cell_runtime_adapter_manifest,
    materialize_inverse_scorer_cell_candidate,
    verify_inverse_scorer_cell_candidate_manifest,
)
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)
from tac.repo_io import (
    ArtifactWriteError,
    repo_relative,
    sha256_file,
    write_json_artifact,
)

CHAIN_SCHEMA = "inverse_scorer_cell_candidate_chain_v1"
CHAIN_MANIFEST_NAME = "inverse_scorer_cell_candidate_chain_manifest.json"
CANDIDATE_ARCHIVE_NAME = "candidate_archive.zip"
CANDIDATE_MANIFEST_NAME = "candidate_manifest.json"
RUNTIME_ADAPTER_MANIFEST_NAME = "runtime_adapter_manifest.json"
RECEIVER_PROOF_NAME = "receiver_proof.json"
INFLATE_PARITY_PROBE_NAME = "inflate_parity_probe.json"
VERIFIED_CANDIDATE_NAME = "verified_candidate.json"


class InverseScorerCellChainError(ValueError):
    """Raised when the inverse-scorer cell proof chain cannot run safely."""


def build_inverse_scorer_cell_candidate_chain(
    *,
    raw_contest_video_digest: str | Mapping[str, Any],
    candidate_archive_template: str | Path,
    inverse_action_functional: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    atom_ids: Sequence[str] = (),
    selected_limit: int | None = None,
    repo_root: str | Path | None = None,
    min_free_bytes: int = 0,
    source_inflate_output_dir: str | Path | None = None,
    candidate_inflate_output_dir: str | Path | None = None,
    inflate_runtime_dir: str | Path | None = None,
    source_archive_for_parity: str | Path | None = None,
    inflate_timeout_seconds: int = 3600,
    inflate_work_dir: str | Path | None = None,
    keep_inflate_work_dir: bool = False,
) -> dict[str, Any]:
    """Run materialization -> receiver adapter -> proof -> verified manifest."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    if inflate_runtime_dir is not None and (
        source_inflate_output_dir is not None or candidate_inflate_output_dir is not None
    ):
        raise InverseScorerCellChainError(
            "choose either inflate_runtime_dir or precomputed inflate output dirs, not both"
        )
    out = _repo_path(Path(output_dir), repo)
    _prepare_new_output_dir(out)

    candidate_archive_path = out / CANDIDATE_ARCHIVE_NAME
    candidate_manifest_path = out / CANDIDATE_MANIFEST_NAME
    runtime_adapter_manifest_path = out / RUNTIME_ADAPTER_MANIFEST_NAME
    receiver_proof_path = out / RECEIVER_PROOF_NAME
    inflate_parity_probe_path = out / INFLATE_PARITY_PROBE_NAME
    verified_candidate_path = out / VERIFIED_CANDIDATE_NAME
    chain_manifest_path = out / CHAIN_MANIFEST_NAME
    inflate_parity_probe: dict[str, Any] | None = None

    try:
        candidate = materialize_inverse_scorer_cell_candidate(
            raw_contest_video_digest=raw_contest_video_digest,
            candidate_archive_template=candidate_archive_template,
            inverse_action_functional=inverse_action_functional,
            output_archive=candidate_archive_path,
            atom_ids=atom_ids,
            selected_limit=selected_limit,
            repo_root=repo,
        )
        _write_json(candidate_manifest_path, candidate, min_free_bytes=min_free_bytes)

        adapter = build_inverse_scorer_cell_runtime_adapter_manifest(
            candidate_manifest=candidate_manifest_path,
            repo_root=repo,
        )
        _write_json(
            runtime_adapter_manifest_path,
            adapter,
            min_free_bytes=min_free_bytes,
        )

        receiver_proof = build_inverse_scorer_cell_receiver_proof(
            runtime_adapter_manifest=runtime_adapter_manifest_path,
            candidate_manifest=candidate_manifest_path,
            repo_root=repo,
        )
        _write_json(receiver_proof_path, receiver_proof, min_free_bytes=min_free_bytes)

        if inflate_runtime_dir is not None:
            inflate_parity_probe = build_inverse_scorer_cell_inflate_parity_probe_from_archives(
                candidate_manifest=candidate_manifest_path,
                inflate_runtime_dir=inflate_runtime_dir,
                source_archive=source_archive_for_parity or candidate_archive_template,
                candidate_archive=candidate_archive_path,
                repo_root=repo,
                timeout_seconds=inflate_timeout_seconds,
                work_dir=inflate_work_dir,
                keep_work_dir=keep_inflate_work_dir,
            )
            _write_json(
                inflate_parity_probe_path,
                inflate_parity_probe,
                min_free_bytes=min_free_bytes,
            )
        elif source_inflate_output_dir is not None or candidate_inflate_output_dir is not None:
            inflate_parity_probe = build_inverse_scorer_cell_inflate_parity_probe(
                candidate_manifest=candidate_manifest_path,
                source_output_dir=source_inflate_output_dir,
                candidate_output_dir=candidate_inflate_output_dir,
                repo_root=repo,
            )
            _write_json(
                inflate_parity_probe_path,
                inflate_parity_probe,
                min_free_bytes=min_free_bytes,
            )

        verified = verify_inverse_scorer_cell_candidate_manifest(
            candidate_manifest=candidate_manifest_path,
            runtime_consumption_proof=receiver_proof_path,
            inflate_parity_probe=inflate_parity_probe_path if inflate_parity_probe is not None else None,
            repo_root=repo,
        )
        _write_json(verified_candidate_path, verified, min_free_bytes=min_free_bytes)
    except (ArtifactWriteError, OSError, ValueError) as exc:
        _write_failure_manifest(
            chain_manifest_path,
            repo=repo,
            output_dir=out,
            error=exc,
        )
        raise InverseScorerCellChainError(str(exc)) from exc

    chain = _chain_manifest(
        repo=repo,
        output_dir=out,
        candidate_manifest_path=candidate_manifest_path,
        runtime_adapter_manifest_path=runtime_adapter_manifest_path,
        receiver_proof_path=receiver_proof_path,
        inflate_parity_probe_path=inflate_parity_probe_path if inflate_parity_probe is not None else None,
        verified_candidate_path=verified_candidate_path,
        candidate=candidate,
        adapter=adapter,
        receiver_proof=receiver_proof,
        inflate_parity_probe=inflate_parity_probe,
        verified=verified,
    )
    _write_json(chain_manifest_path, chain, min_free_bytes=min_free_bytes)
    return chain


def _chain_manifest(
    *,
    repo: Path,
    output_dir: Path,
    candidate_manifest_path: Path,
    runtime_adapter_manifest_path: Path,
    receiver_proof_path: Path,
    inflate_parity_probe_path: Path | None,
    verified_candidate_path: Path,
    candidate: Mapping[str, Any],
    adapter: Mapping[str, Any],
    receiver_proof: Mapping[str, Any],
    inflate_parity_probe: Mapping[str, Any] | None,
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_archive = _mapping(candidate.get("candidate_archive"))
    source_archive = _mapping(candidate.get("template_archive"))
    serialized_archive_delta = build_serialized_archive_delta_contract(
        source_archive=source_archive,
        candidate_archive=candidate_archive,
        modeled_cost_bytes=_selected_cell_cost_bytes(candidate.get("selected_cells")),
    )
    adapter_blockers = _string_list(adapter.get("readiness_blockers"))
    proof_blockers = _string_list(receiver_proof.get("blockers"))
    parity_blockers = _string_list((inflate_parity_probe or {}).get("blockers"))
    verified_blockers = _string_list(verified.get("readiness_blockers"))
    readiness_blockers = _ordered_unique([*adapter_blockers, *proof_blockers, *parity_blockers, *verified_blockers])
    artifacts = {
        "candidate_manifest": _file_record(candidate_manifest_path, repo=repo),
        "runtime_adapter_manifest": _file_record(
            runtime_adapter_manifest_path,
            repo=repo,
        ),
        "receiver_proof": _file_record(receiver_proof_path, repo=repo),
        "verified_candidate": _file_record(verified_candidate_path, repo=repo),
    }
    if inflate_parity_probe_path is not None:
        artifacts["inflate_parity_probe"] = _file_record(
            inflate_parity_probe_path,
            repo=repo,
        )
    chain_steps = [
        {
            "step_id": "materialize_candidate",
            "status": "succeeded",
            "schema": candidate.get("schema"),
            "artifact": _file_record(candidate_manifest_path, repo=repo),
            "archive": candidate_archive,
        },
        {
            "step_id": "build_runtime_adapter",
            "status": "succeeded",
            "schema": adapter.get("schema"),
            "artifact": _file_record(
                runtime_adapter_manifest_path,
                repo=repo,
            ),
            "runtime_tree_sha256": adapter.get("runtime_tree_sha256") or "",
            "readiness_blockers": adapter_blockers,
        },
        {
            "step_id": "build_receiver_proof",
            "status": "succeeded",
            "schema": receiver_proof.get("schema"),
            "artifact": _file_record(receiver_proof_path, repo=repo),
            "ready_for_receiver_verification": receiver_proof.get("ready_for_receiver_verification") is True,
            "ready_for_exact_eval_runtime": receiver_proof.get("ready_for_exact_eval_runtime") is True,
            "blockers": proof_blockers,
        },
    ]
    if inflate_parity_probe is not None and inflate_parity_probe_path is not None:
        chain_steps.append(
            {
                "step_id": "build_inflate_parity_probe",
                "status": "succeeded",
                "schema": inflate_parity_probe.get("schema"),
                "artifact": _file_record(inflate_parity_probe_path, repo=repo),
                "full_frame_inflate_output_parity_claim": (
                    inflate_parity_probe.get("full_frame_inflate_output_parity_claim") is True
                ),
                "blockers": parity_blockers,
            }
        )
    chain_steps.append(
        {
            "step_id": "verify_candidate_with_receiver_proof",
            "status": "succeeded",
            "schema": verified.get("schema"),
            "artifact": _file_record(verified_candidate_path, repo=repo),
            "receiver_contract_satisfied": verified.get("receiver_contract_satisfied") is True,
            "inflate_parity_satisfied": verified.get("inflate_parity_satisfied") is True,
            "readiness_blockers": verified_blockers,
        }
    )
    realized_saved_bytes = serialized_archive_delta.get("realized_saved_bytes")
    rate_positive = (
        serialized_archive_delta.get("status") == "realized_saving"
        and serialized_archive_delta.get("savings_realized") is True
        and isinstance(realized_saved_bytes, int)
        and realized_saved_bytes > 0
    )
    rate_semantics = (
        "realized_archive_saving"
        if rate_positive
        else "successful_quality_spend_not_byte_saving_progress"
    )
    return {
        "schema": CHAIN_SCHEMA,
        "output_dir": repo_relative(output_dir, repo),
        "source_archive": source_archive,
        "source_archive_sha256": source_archive.get("sha256") or "",
        "source_archive_bytes": source_archive.get("bytes"),
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": candidate_archive.get("sha256") or "",
        "candidate_archive_bytes": candidate_archive.get("bytes"),
        "candidate_member_sha256": candidate_archive.get("member_sha256") or "",
        "serialized_archive_delta": serialized_archive_delta,
        "materializer_rate_outcome": serialized_archive_delta.get("status"),
        "rate_positive": rate_positive,
        "realized_saved_bytes": realized_saved_bytes,
        "signal_semantics": rate_semantics,
        "evidence_semantics": rate_semantics,
        "quality_spend_allowed": False,
        "byte_closed_candidate_emitted": candidate.get("byte_closed_candidate_emitted") is True,
        "runtime_adapter_ready": not adapter_blockers,
        "receiver_proof_ready": receiver_proof.get("ready_for_receiver_verification") is True,
        "receiver_contract_satisfied": verified.get("receiver_contract_satisfied") is True,
        "inflate_parity_satisfied": verified.get("inflate_parity_satisfied") is True,
        "candidate_runtime_adapter_blocker_cleared": (
            "runtime_consumption_proof_missing" not in verified_blockers
            and "inverse_scorer_cell_receiver_contract_not_satisfied" not in verified_blockers
        ),
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
            "inverse_scorer_cell_candidate_chain_is_not_dispatch_authorization",
            *readiness_blockers,
        ],
        "artifacts": artifacts,
        "chain_steps": chain_steps,
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
    if "exact_auth_eval_required_before_score_claim" in blocker_set:
        gates.append("contest_auth_eval")
    return gates


def _prepare_new_output_dir(path: Path) -> None:
    if path.exists():
        raise InverseScorerCellChainError(f"output directory already exists; choose a new chain directory: {path}")
    path.mkdir(parents=True)


def _write_json(path: Path, payload: Mapping[str, Any], *, min_free_bytes: int) -> None:
    write_json_artifact(path, payload, min_free_bytes=min_free_bytes)


def _write_failure_manifest(
    path: Path,
    *,
    repo: Path,
    output_dir: Path,
    error: BaseException,
) -> None:
    if path.exists():
        return
    payload = {
        "schema": CHAIN_SCHEMA,
        "output_dir": repo_relative(output_dir, repo),
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "readiness_blockers": ["inverse_scorer_cell_candidate_chain_failed"],
        "dispatch_blockers": ["inverse_scorer_cell_candidate_chain_failed"],
        **FALSE_AUTHORITY,
    }
    try:
        write_json_artifact(path, payload)
    except (ArtifactWriteError, OSError):
        return


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


def _selected_cell_cost_bytes(value: Any) -> int | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    total = 0
    seen = False
    for item in value:
        if not isinstance(item, Mapping):
            continue
        raw = item.get("water_fill_cost_bytes")
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            total += parsed
            seen = True
    return total if seen else None


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "CANDIDATE_ARCHIVE_NAME",
    "CANDIDATE_MANIFEST_NAME",
    "CHAIN_MANIFEST_NAME",
    "CHAIN_SCHEMA",
    "RECEIVER_PROOF_NAME",
    "RUNTIME_ADAPTER_MANIFEST_NAME",
    "VERIFIED_CANDIDATE_NAME",
    "InverseScorerCellChainError",
    "build_inverse_scorer_cell_candidate_chain",
]
