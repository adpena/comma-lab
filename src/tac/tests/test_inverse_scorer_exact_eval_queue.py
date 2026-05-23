# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest

from tac.optimization.inverse_scorer_exact_eval_queue import (
    DEFAULT_LANE_ID,
    InverseScorerExactEvalQueueError,
    build_inverse_scorer_exact_eval_source_queue,
)
from tac.optimizer.exact_readiness import promote_candidate_for_exact_eval


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", payload)


def _write_runtime(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    inflate = path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (path / "report.txt").write_text("IAS1 exact readiness fixture\n", encoding="utf-8")


def _write_chain_fixture(
    repo: Path,
    *,
    runtime: Path,
    parity_backed: bool = True,
    parity_artifact_minimal: bool = False,
    byte_closed: bool = True,
    unchanged_archive: bool = False,
    member_sha256: str | None = None,
    readiness_blockers: list[str] | None = None,
    candidate_path_override: str | None = None,
) -> tuple[Path, Path]:
    chain_dir = repo / "chain"
    chain_dir.mkdir(parents=True, exist_ok=True)
    source_archive = chain_dir / "source.zip"
    candidate_archive = chain_dir / "candidate.zip"
    parity_probe = chain_dir / "inflate_parity_probe.json"
    _write_zip(source_archive, b"source")
    _write_zip(candidate_archive, b"candidate-ias1")
    candidate_payload = b"candidate-ias1"
    if unchanged_archive:
        candidate_archive.write_bytes(source_archive.read_bytes())
        candidate_payload = b"source"
    candidate_record = {
        "path": candidate_path_override or candidate_archive.relative_to(repo).as_posix(),
        "bytes": candidate_archive.stat().st_size,
        "sha256": _sha(candidate_archive),
        "member_name": "x",
        "member_bytes": len(candidate_payload),
        "member_sha256": member_sha256 or hashlib.sha256(candidate_payload).hexdigest(),
    }
    source_record = {
        "path": source_archive.relative_to(repo).as_posix(),
        "bytes": source_archive.stat().st_size,
        "sha256": _sha(source_archive),
        "member_name": "x",
        "member_bytes": len(b"source"),
        "member_sha256": hashlib.sha256(b"source").hexdigest(),
    }
    if parity_backed:
        source_tree = {
            "exists": True,
            "file_count": 1,
            "total_bytes": 16,
            "tree_sha256": "1" * 64,
            "blockers": [],
            "files": [{"path": "0.raw", "bytes": 16, "sha256": "2" * 64}],
        }
        parity_payload = (
            {"schema": "inverse_scorer_cell_inflate_parity_probe_v1"}
            if parity_artifact_minimal
            else {
                "schema": "inverse_scorer_cell_inflate_parity_probe_v1",
                "proof_scope": "full_frame_inflate_output_tree",
                "full_frame_inflate_output_parity_claim": True,
                "expect_output_byte_identical": True,
                "output_bytes_identical": True,
                "output_contract_nonempty": True,
                "output_contract_paths_match": True,
                "differing_path_count": 0,
                "differing_paths_sample": [],
                "missing_from_candidate": [],
                "extra_in_candidate": [],
                "blockers": [],
                "cleared_blockers": ["candidate_inflate_output_parity_missing"],
                "dispatch_blockers": [
                    "inverse_scorer_cell_inflate_parity_is_not_score_authority",
                    "exact_auth_eval_required_before_score_claim",
                ],
                "source_archive": source_record,
                "candidate_archive": candidate_record,
                "source_archive_inflated": source_record,
                "candidate_archive_inflated": candidate_record,
                "source_inflate_run": {
                    "returncode": 0,
                    "full_frame_file_list_claim": True,
                    "file_list_entries": ["0.mkv"],
                },
                "candidate_inflate_run": {
                    "returncode": 0,
                    "full_frame_file_list_claim": True,
                    "file_list_entries": ["0.mkv"],
                },
                "source_output_tree": source_tree,
                "candidate_output_tree": source_tree,
                "inflate_runtime": {
                    "path": runtime.relative_to(repo).as_posix(),
                    "inflate_sh": (runtime / "inflate.sh").relative_to(repo).as_posix(),
                    "inflate_sh_sha256": _sha(runtime / "inflate.sh"),
                    "full_frame_file_list_claim": True,
                    "file_list_entries": ["0.mkv"],
                },
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "score_claim": False,
                "score_claim_valid": False,
                "score_claim_eligible": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "field_selection_ready_for_exact_eval_dispatch": False,
                "exact_cuda_auth_eval": False,
                "contest_cuda_auth_eval": False,
                "score_affecting_payload_changed": False,
                "charged_bits_changed": False,
            }
        )
        parity_probe.write_text(
            json.dumps(parity_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    chain = {
        "schema": "inverse_scorer_cell_candidate_chain_v1",
        "candidate_archive": candidate_record,
        "source_archive": source_record,
        "receiver_contract_satisfied": True,
        "inflate_parity_satisfied": True,
        "byte_closed_candidate_emitted": byte_closed,
        "readiness_blockers": readiness_blockers
        or ["exact_auth_eval_required_before_score_claim"],
        "dispatch_blockers": [
            "inverse_scorer_cell_candidate_chain_is_not_dispatch_authorization",
            "exact_auth_eval_required_before_score_claim",
        ],
        "next_required_gates": ["contest_auth_eval"],
        "chain_steps": [
            {
                "step_id": "build_runtime_adapter",
                "status": "succeeded",
                "schema": "inverse_scorer_cell_runtime_adapter_v1",
                "artifact": {
                    "path": parity_probe.relative_to(repo).as_posix(),
                    "bytes": parity_probe.stat().st_size if parity_probe.exists() else 1,
                    "sha256": _sha(parity_probe) if parity_probe.exists() else "0" * 64,
                },
                "readiness_blockers": [],
                "runtime_tree_sha256": "a" * 64,
            },
            {
                "step_id": "build_inflate_parity_probe",
                "status": "succeeded",
                "schema": "inverse_scorer_cell_inflate_parity_probe_v1",
                "artifact": {
                    "path": parity_probe.relative_to(repo).as_posix(),
                    "bytes": parity_probe.stat().st_size if parity_probe.exists() else 1,
                    "sha256": _sha(parity_probe) if parity_probe.exists() else "0" * 64,
                },
                "full_frame_inflate_output_parity_claim": True,
                "blockers": [],
            }
        ],
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_claim": False,
        "score_claim_valid": False,
        "score_claim_eligible": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }
    chain_path = chain_dir / "inverse_scorer_cell_candidate_chain_manifest.json"
    chain_path.write_text(json.dumps(chain, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return chain_path, candidate_archive


def test_build_inverse_scorer_exact_eval_queue_promotes_after_readiness(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, candidate_archive = _write_chain_fixture(tmp_path, runtime=runtime)
    archive_manifest = tmp_path / "out" / "archive_manifest.json"
    queue_path = tmp_path / "out" / "source_queue.json"

    result = build_inverse_scorer_exact_eval_source_queue(
        chain_manifest_path=chain_path,
        runtime_submission_dir=runtime,
        archive_manifest_path=archive_manifest,
        repo_root=tmp_path,
        candidate_id="ias1_fixture",
        lane_id=DEFAULT_LANE_ID,
    )
    archive_manifest.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest.write_text(json.dumps(result.archive_manifest), encoding="utf-8")
    queue_path.write_text(json.dumps(result.queue), encoding="utf-8")

    row = result.queue["top_k"][0]
    assert row["schema"] == "inverse_scorer_cell_candidate_chain_v1"
    assert row["candidate_archive_sha256"] == _sha(candidate_archive)
    assert row["score_affecting_payload_changed"] is True
    assert row["charged_bits_changed"] is True
    assert row["score_claim"] is False
    assert result.archive_manifest["ready_for_exact_eval_dispatch"] is False
    assert result.archive_manifest["score_affecting_payload_changed"] is True
    assert result.archive_manifest["charged_bits_changed"] is True

    promoted = promote_candidate_for_exact_eval(
        queue_path,
        "ias1_fixture",
        repo_root=tmp_path,
        submission_dir=runtime,
        archive_manifest_path=archive_manifest,
        active_floor_archive_bytes=None,
    )

    assert promoted["report"]["ready_for_exact_eval_dispatch"] is True
    ready = promoted["promoted_queue"]["dispatch_ready"][0]
    assert ready["ready_for_exact_eval_dispatch"] is True
    assert ready["score_claim"] is False
    assert ready["promotion_eligible"] is False
    assert ready["archive_sha256"] == _sha(candidate_archive)


def test_build_inverse_scorer_exact_eval_queue_refuses_unbacked_parity(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(
        tmp_path,
        runtime=runtime,
        parity_backed=False,
    )

    with pytest.raises(InverseScorerExactEvalQueueError, match="artifact_file_missing"):
        build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=chain_path,
            runtime_submission_dir=runtime,
            archive_manifest_path=tmp_path / "archive_manifest.json",
            repo_root=tmp_path,
        )


def test_build_inverse_scorer_exact_eval_queue_refuses_minimal_parity_artifact(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(
        tmp_path,
        runtime=runtime,
        parity_artifact_minimal=True,
    )

    with pytest.raises(
        InverseScorerExactEvalQueueError,
        match="inflate_parity_payload_full_frame_inflate_output_parity_claim_not_true",
    ):
        build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=chain_path,
            runtime_submission_dir=runtime,
            archive_manifest_path=tmp_path / "archive_manifest.json",
            repo_root=tmp_path,
        )


def test_build_inverse_scorer_exact_eval_queue_refuses_non_byte_closed_chain(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(
        tmp_path,
        runtime=runtime,
        byte_closed=False,
    )

    with pytest.raises(
        InverseScorerExactEvalQueueError,
        match="byte_closed_candidate_not_emitted",
    ):
        build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=chain_path,
            runtime_submission_dir=runtime,
            archive_manifest_path=tmp_path / "archive_manifest.json",
            repo_root=tmp_path,
        )


def test_build_inverse_scorer_exact_eval_queue_refuses_unchanged_archive(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(
        tmp_path,
        runtime=runtime,
        unchanged_archive=True,
    )

    with pytest.raises(
        InverseScorerExactEvalQueueError,
        match="source_candidate_archive_sha256_unchanged",
    ):
        build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=chain_path,
            runtime_submission_dir=runtime,
            archive_manifest_path=tmp_path / "archive_manifest.json",
            repo_root=tmp_path,
        )


def test_build_inverse_scorer_exact_eval_queue_refuses_member_sha_mismatch(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(
        tmp_path,
        runtime=runtime,
        member_sha256="0" * 64,
    )

    with pytest.raises(
        InverseScorerExactEvalQueueError,
        match="candidate_archive_member_sha256_mismatch",
    ):
        build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=chain_path,
            runtime_submission_dir=runtime,
            archive_manifest_path=tmp_path / "archive_manifest.json",
            repo_root=tmp_path,
        )


def test_build_inverse_scorer_exact_eval_queue_refuses_runtime_mismatch(
    tmp_path: Path,
) -> None:
    parity_runtime = tmp_path / "runtime"
    submitted_runtime = tmp_path / "other_runtime"
    _write_runtime(parity_runtime)
    _write_runtime(submitted_runtime)
    chain_path, _candidate_archive = _write_chain_fixture(
        tmp_path,
        runtime=parity_runtime,
    )

    with pytest.raises(
        InverseScorerExactEvalQueueError,
        match="inflate_parity_runtime_path_mismatch",
    ):
        build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=chain_path,
            runtime_submission_dir=submitted_runtime,
            archive_manifest_path=tmp_path / "archive_manifest.json",
            repo_root=tmp_path,
        )


def test_build_inverse_scorer_exact_eval_queue_refuses_extra_readiness_blocker(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(
        tmp_path,
        runtime=runtime,
        readiness_blockers=[
            "exact_auth_eval_required_before_score_claim",
            "full_frame_inflate_output_parity_missing",
        ],
    )

    with pytest.raises(
        InverseScorerExactEvalQueueError,
        match="chain_unresolved_readiness_blockers:full_frame_inflate_output_parity_missing",
    ):
        build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=chain_path,
            runtime_submission_dir=runtime,
            archive_manifest_path=tmp_path / "archive_manifest.json",
            repo_root=tmp_path,
        )


def test_build_inverse_scorer_exact_eval_queue_refuses_parent_traversal_path(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(
        tmp_path,
        runtime=runtime,
        candidate_path_override="../candidate.zip",
    )

    with pytest.raises(InverseScorerExactEvalQueueError, match="path_outside_repo"):
        build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=chain_path,
            runtime_submission_dir=runtime,
            archive_manifest_path=tmp_path / "archive_manifest.json",
            repo_root=tmp_path,
        )


def test_exact_readiness_rechecks_parity_payload_content(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(tmp_path, runtime=runtime)
    archive_manifest = tmp_path / "out" / "archive_manifest.json"
    queue_path = tmp_path / "out" / "source_queue.json"
    result = build_inverse_scorer_exact_eval_source_queue(
        chain_manifest_path=chain_path,
        runtime_submission_dir=runtime,
        archive_manifest_path=archive_manifest,
        repo_root=tmp_path,
        candidate_id="ias1_fixture",
        lane_id=DEFAULT_LANE_ID,
    )
    archive_manifest.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest.write_text(json.dumps(result.archive_manifest), encoding="utf-8")

    parity_probe = tmp_path / result.queue["top_k"][0]["chain_steps"][1]["artifact"]["path"]
    parity_probe.write_text(
        json.dumps({"schema": "inverse_scorer_cell_inflate_parity_probe_v1"}),
        encoding="utf-8",
    )
    row = result.queue["top_k"][0]
    row["chain_steps"][1]["artifact"]["bytes"] = parity_probe.stat().st_size
    row["chain_steps"][1]["artifact"]["sha256"] = _sha(parity_probe)
    queue_path.write_text(json.dumps(result.queue), encoding="utf-8")

    promoted = promote_candidate_for_exact_eval(
        queue_path,
        "ias1_fixture",
        repo_root=tmp_path,
        submission_dir=runtime,
        archive_manifest_path=archive_manifest,
        active_floor_archive_bytes=None,
    )

    assert promoted["report"]["ready_for_exact_eval_dispatch"] is False
    assert (
        "inverse_scorer_cell_candidate_chain_strict_full_frame_inflate_parity_missing"
        in promoted["report"]["blockers"]
    )


def test_exact_readiness_refuses_unresolved_chain_readiness_blocker(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _write_runtime(runtime)
    chain_path, _candidate_archive = _write_chain_fixture(tmp_path, runtime=runtime)
    archive_manifest = tmp_path / "out" / "archive_manifest.json"
    queue_path = tmp_path / "out" / "source_queue.json"
    result = build_inverse_scorer_exact_eval_source_queue(
        chain_manifest_path=chain_path,
        runtime_submission_dir=runtime,
        archive_manifest_path=archive_manifest,
        repo_root=tmp_path,
        candidate_id="ias1_fixture",
        lane_id=DEFAULT_LANE_ID,
    )
    archive_manifest.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest.write_text(json.dumps(result.archive_manifest), encoding="utf-8")
    result.queue["top_k"][0]["readiness_blockers"].append(
        "manual_runtime_review_still_open"
    )
    queue_path.write_text(json.dumps(result.queue), encoding="utf-8")

    promoted = promote_candidate_for_exact_eval(
        queue_path,
        "ias1_fixture",
        repo_root=tmp_path,
        submission_dir=runtime,
        archive_manifest_path=archive_manifest,
        active_floor_archive_bytes=None,
    )

    assert promoted["report"]["ready_for_exact_eval_dispatch"] is False
    assert (
        "inverse_scorer_cell_candidate_chain_unresolved_readiness_blocker:"
        "manual_runtime_review_still_open"
    ) in promoted["report"]["blockers"]
