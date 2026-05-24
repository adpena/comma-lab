# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

import comma_lab.artifact_retention as retention
from comma_lab.artifact_retention import (
    ArtifactRetentionError,
    build_retention_plan,
    execute_retention_plan,
    load_json_object,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[3]


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _load_compact_tool():
    spec = importlib.util.spec_from_file_location(
        "compact_experiment_artifacts_under_test",
        REPO / "tools" / "compact_experiment_artifacts.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_locality_candidate(
    root: Path,
    *,
    passed: bool = True,
    locality_root_name: str = "locality_work",
    manifest_name: str = "locality_controls.json",
) -> Path:
    candidate = root / "candidate_a"
    inflated = candidate / locality_root_name / "selective" / "inflated"
    raw_path = inflated / "0.raw"
    _write(raw_path, b"r" * 32)
    archive_zip = candidate / "submission_dir" / "archive.zip"
    entrypoint = candidate / "submission_dir" / "inflate.sh"
    _write(archive_zip, b"zip")
    _write(entrypoint, b"#!/bin/sh\n")
    manifest = {
        "schema": "decoder_q_selective_runtime_locality_controls.v1",
        "locality_controls_passed": passed,
        "mismatch_counts": {
            "missing_raw_file_count": 0,
            "raw_size_mismatch_count": 0,
            "selected_frame_mismatch_count": 0,
            "unselected_frame_mismatch_count": 0,
        },
        "targets": {
            "selective": {
                "archive_zip": str(archive_zip),
                "entrypoint_path": str(entrypoint),
                "output_dir": str(inflated),
                "returncode": 0,
                "archive_sha256": "a" * 64,
                "entrypoint_sha256": "b" * 64,
            }
        },
        "hashes": {
            "0.raw": {
                "raw_files": {
                    "selective": sha256_file(raw_path),
                }
            }
        },
    }
    (candidate / manifest_name).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return inflated


def _write_mlx_delta_cache(root: Path, *, stamp: bool = True) -> Path:
    cache = root / "candidate_mlx" / "mlx_delta_cache"
    _write(cache / "pair_indices.npy", b"pairs")
    _write(cache / "posenet_yuv6_pair.npy", b"pose")
    _write(cache / "segnet_last_rgb.npy", b"seg")
    local_advisory = root / "candidate_mlx" / "local_cpu_advisory.json"
    local_advisory.write_text(
        json.dumps({"score_axis": "cpu_advisory", **_false_authority()}),
        encoding="utf-8",
    )
    manifest = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "raw_sha256": "c" * 64,
        "pair_count": 1,
        "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
        "array_sha256": {
            "pair_indices": "1" * 64,
            "posenet_yuv6_pair": "2" * 64,
            "segnet_last_rgb": "3" * 64,
        },
        "artifacts": {
            "pair_indices": {"sha256": sha256_file(cache / "pair_indices.npy")},
            "posenet_yuv6_pair": {"sha256": sha256_file(cache / "posenet_yuv6_pair.npy")},
            "segnet_last_rgb": {"sha256": sha256_file(cache / "segnet_last_rgb.npy")},
        },
    }
    if stamp:
        audit = {
            "schema_version": "mlx_scorer_input_cache_local_cpu_advisory_audit.v1",
            "verdict": "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
            "passed": True,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "cache": {
                "archive_sha256": manifest["archive_sha256"],
                "inflated_outputs_aggregate_sha256": manifest["inflated_outputs_aggregate_sha256"],
                "raw_sha256": manifest["raw_sha256"],
                "pair_count": manifest["pair_count"],
                "hash_domain": manifest["hash_domain"],
                "array_sha256": manifest["array_sha256"],
            },
        }
        audit_path = root / ".omx" / "research" / "mlx_cache_identity.json"
        audit_path.parent.mkdir(parents=True)
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        manifest["eligible_for_local_mlx_local_advisory_debug"] = True
        manifest["eligible_for_local_mlx_transfer_calibration"] = False
        manifest["local_cpu_advisory_cache_identity_audit"] = {
            "schema_version": audit["schema_version"],
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "verdict": audit["verdict"],
            "passed": True,
            "local_cpu_advisory_path": str(local_advisory),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache


def _proxy_false_authority() -> dict[str, bool]:
    return {
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


def _relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _inverse_tree_record(path: Path, repo_root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        files.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    tree_sha = hashlib.sha256(
        json.dumps(
            {"files": files},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "path": _relative(path, repo_root),
        "exists": True,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "tree_sha256": tree_sha,
        "files": files,
        "blockers": [],
    }


def _write_inverse_scorer_parity_fixture(
    repo_root: Path,
    *,
    proof_mutation: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    chain = repo_root / "chain"
    work = chain / "inflate_work"
    source_out = work / "source" / "out"
    candidate_out = work / "candidate" / "out"
    _write(source_out / "0.raw", b"frame-bytes")
    _write(candidate_out / "0.raw", b"frame-bytes")
    source_archive = chain / "source.zip"
    candidate_archive = chain / "candidate.zip"
    candidate_manifest = chain / "candidate_manifest.json"
    runtime = chain / "runtime"
    inflate_sh = runtime / "inflate.sh"
    _write(source_archive, b"source-zip")
    _write(candidate_archive, b"candidate-zip")
    _write(inflate_sh, b"#!/bin/sh\n")
    candidate_manifest_payload = {
        "schema": "inverse_scorer_cell_candidate_v1",
        **_proxy_false_authority(),
    }
    candidate_manifest.write_text(
        json.dumps(candidate_manifest_payload),
        encoding="utf-8",
    )
    proof = {
        "schema": "inverse_scorer_cell_inflate_parity_probe_v1",
        "proof_scope": "full_frame_inflate_output_tree",
        "candidate_manifest": {
            "provided_inline": False,
            "path": _relative(candidate_manifest, repo_root),
            "bytes": candidate_manifest.stat().st_size,
            "sha256": sha256_file(candidate_manifest),
        },
        "source_archive": {
            "path": _relative(source_archive, repo_root),
            "bytes": source_archive.stat().st_size,
            "sha256": sha256_file(source_archive),
        },
        "candidate_archive": {
            "path": _relative(candidate_archive, repo_root),
            "bytes": candidate_archive.stat().st_size,
            "sha256": sha256_file(candidate_archive),
            "member_sha256": "d" * 64,
        },
        "inverse_scorer_cell_descriptor": {
            "schema": "inverse_scorer_cell_descriptor_v1",
            "packet_offset": 12,
            "packet_bytes": 34,
            "packet_sha256": "a" * 64,
            "json_sha256": "b" * 64,
        },
        "source_output_tree": _inverse_tree_record(source_out, repo_root),
        "candidate_output_tree": _inverse_tree_record(candidate_out, repo_root),
        "output_contract_paths_match": True,
        "output_contract_nonempty": True,
        "expect_output_byte_identical": True,
        "output_bytes_identical": True,
        "full_frame_inflate_output_parity_claim": True,
        "cleared_blockers": ["candidate_inflate_output_parity_missing"],
        "differing_paths_sample": [],
        "differing_path_count": 0,
        "missing_from_candidate": [],
        "extra_in_candidate": [],
        "blockers": [],
        "dispatch_blockers": [
            "inverse_scorer_cell_inflate_parity_is_not_score_authority",
            "exact_auth_eval_required_before_score_claim",
        ],
        "inflate_runtime": {
            "path": _relative(runtime, repo_root),
            "inflate_sh": _relative(inflate_sh, repo_root),
            "inflate_sh_sha256": sha256_file(inflate_sh),
            "timeout_seconds": 30,
            "file_list_entries": ["0.mkv"],
            "full_frame_file_list_claim": True,
        },
        "source_inflate_run": {
            "returncode": 0,
            "timeout_seconds": 30,
            "file_list_entries": ["0.mkv"],
            "full_frame_file_list_claim": True,
            "output_dir": _relative(source_out, repo_root),
        },
        "candidate_inflate_run": {
            "returncode": 0,
            "timeout_seconds": 30,
            "file_list_entries": ["0.mkv"],
            "full_frame_file_list_claim": True,
            "output_dir": _relative(candidate_out, repo_root),
        },
        "source_archive_inflated": {
            "path": _relative(source_archive, repo_root),
            "bytes": source_archive.stat().st_size,
            "sha256": sha256_file(source_archive),
        },
        "candidate_archive_inflated": {
            "path": _relative(candidate_archive, repo_root),
            "bytes": candidate_archive.stat().st_size,
            "sha256": sha256_file(candidate_archive),
        },
        "work_dir": _relative(work, repo_root),
        "work_dir_retained": True,
        **_proxy_false_authority(),
    }
    if proof_mutation:
        proof.update(proof_mutation)
    proof_path = chain / "inflate_parity_probe.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    return source_out, candidate_out, proof_path


def test_retention_deletes_only_certified_locality_raw(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path)
    plan = build_retention_plan(
        [tmp_path],
        repo_root=tmp_path,
        min_bytes=1,
    )

    assert plan.total_reclaimable_bytes == 32
    assert [row.kind for row in plan.candidates] == ["locality_inflated_raw"]
    assert plan.candidates[0].certificate["raw_sha256"] == sha256_file(inflated / "0.raw")

    execution = execute_retention_plan(plan, action="delete")

    assert execution["executed_count"] == 1
    assert not inflated.exists()
    assert (tmp_path / "candidate_a" / "locality_controls.json").is_file()
    assert (tmp_path / "candidate_a" / "submission_dir" / "archive.zip").is_file()


def test_compact_cli_execute_stdout_writes_default_journal(
    tmp_path: Path,
    capsys,
) -> None:
    inflated = _write_locality_candidate(tmp_path)
    tool = _load_compact_tool()

    rc = tool.main(
        [
            str(tmp_path),
            "--repo-root",
            str(tmp_path),
            "--min-bytes",
            "1",
            "--execute",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    journal_path = Path(payload["execution"]["journal_path"])
    assert rc == 0
    assert not inflated.exists()
    assert journal_path.is_file()
    assert journal_path.relative_to(tmp_path).parts[:3] == (
        ".omx",
        "state",
        "artifact_retention_journals",
    )
    events = [
        json.loads(line)["event"]
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["start", "candidate_start", "candidate_end"]


def test_retention_blocks_failed_locality_control(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path, passed=False)
    plan = build_retention_plan(
        [tmp_path],
        repo_root=tmp_path,
        min_bytes=1,
    )

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 1
    assert "locality_controls_not_passed" in plan.blocked_candidates[0].blockers
    assert inflated.exists()


def test_retention_matches_named_locality_controls_work_manifest(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(
        tmp_path,
        locality_root_name="locality_controls_work",
        manifest_name="locality_controls_pair501.json",
    )

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert len(plan.candidates) == 1
    assert plan.candidates[0].path.endswith("locality_controls_work/selective/inflated")
    assert plan.candidates[0].certificate["manifest_path"].endswith(
        "locality_controls_pair501.json"
    )
    assert inflated.exists()


def test_retention_moves_local_cpu_advisory_scratch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    work = repo / "candidate_b" / "dqs1_pair501_cpu_advisory_work_venv"
    inflated = work / "inflated"
    _write(inflated / "0.raw", b"r" * 16)
    _write(work / "archive.zip", b"zip")
    (work / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                **_false_authority(),
                "score_claim_valid": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "n_samples": 600,
            }
        ),
        encoding="utf-8",
    )
    (work / "inflated_outputs_manifest.json").write_text(
        json.dumps(
            {
                "payload": {
                    "files": [
                        {
                            "path": "0.raw",
                            "sha256": sha256_file(inflated / "0.raw"),
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (work / "provenance.json").write_text(
        json.dumps({"command": ["inflate"]}),
        encoding="utf-8",
    )
    cold_store = tmp_path / "cold"
    cold_store.mkdir()

    plan = build_retention_plan(
        [repo / "candidate_b"],
        repo_root=repo,
        min_bytes=1,
    )
    execution = execute_retention_plan(plan, action="move", cold_store_root=cold_store)

    assert execution["executed_count"] == 1
    assert execution["cold_store_contract"]["write_probe_passed"] is True
    assert not inflated.exists()
    moved = cold_store / plan.candidates[0].path
    assert (moved / "0.raw").read_bytes() == b"r" * 16
    assert execution["rows"][0]["cold_store_verification"]["method"] == "same_device_rename"
    assert execution["rows"][0]["local_bytes_reclaimed"] == 0
    assert execution["rows"][0]["cold_store_verification"]["source_digest"]["sha256"]
    assert load_json_object(work / "contest_auth_eval.json")["n_samples"] == 600


def test_retention_same_device_move_does_not_copytree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    inflated = _write_locality_candidate(repo)
    cold_store = tmp_path / "cold"
    cold_store.mkdir()
    plan = build_retention_plan([repo], repo_root=repo, min_bytes=1)

    def fail_copytree(*args: object, **kwargs: object) -> None:
        raise AssertionError("same-device retention should use rename")

    monkeypatch.setattr(shutil, "copytree", fail_copytree)

    execution = execute_retention_plan(plan, action="move", cold_store_root=cold_store)

    row = execution["rows"][0]
    assert row["cold_store_verification"]["method"] == "same_device_rename"
    assert row["local_bytes_reclaimed"] == 0
    assert not inflated.exists()
    assert (cold_store / plan.candidates[0].path / "0.raw").is_file()


def test_retention_move_rejects_cold_store_inside_repo(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path)
    cold_store = tmp_path / "cold"
    cold_store.mkdir()
    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    with pytest.raises(ArtifactRetentionError, match="outside repo_root"):
        execute_retention_plan(plan, action="move", cold_store_root=cold_store)

    assert inflated.exists()


def test_retention_move_copy_failure_journals_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    inflated = _write_locality_candidate(repo)
    cold_store = tmp_path / "cold"
    cold_store.mkdir()
    journal = tmp_path / "retention.journal.jsonl"
    plan = build_retention_plan([repo], repo_root=repo, min_bytes=1)
    original_digest = retention.directory_digest

    def mismatching_destination_digest(path: Path) -> dict[str, object]:
        digest = original_digest(path)
        if ".partial-" in path.name:
            digest = dict(digest)
            digest["sha256"] = "0" * 64
        return digest

    monkeypatch.setattr(retention, "directory_digest", mismatching_destination_digest)

    with pytest.raises(ArtifactRetentionError, match="copy verification failed"):
        execute_retention_plan(
            plan,
            action="move",
            cold_store_root=cold_store,
            journal_path=journal,
        )

    assert inflated.exists()
    assert not any(cold_store.rglob("*.partial-*"))
    events = [
        json.loads(line)["event"]
        for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["start", "candidate_start", "candidate_error", "candidate_end"]


def test_retention_tiered_move_uses_first_cold_store_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    inflated = _write_locality_candidate(repo)
    cold_fast = tmp_path / "cold_fast"
    cold_slow = tmp_path / "cold_slow"
    cold_fast.mkdir()
    cold_slow.mkdir()
    plan = build_retention_plan([repo], repo_root=repo, min_bytes=1)

    execution = execute_retention_plan(
        plan,
        action="move",
        cold_store_roots=[cold_fast, cold_slow],
    )

    assert execution["tiered_cold_store"] is True
    assert execution["executed_count"] == 1
    assert execution["rows"][0]["cold_store_tier_index"] == 0
    assert execution["rows"][0]["cold_store_root"] == str(cold_fast)
    assert execution["local_bytes_reclaimed"] == 0
    assert not inflated.exists()
    assert (cold_fast / plan.candidates[0].path / "0.raw").is_file()
    assert not any(cold_slow.rglob("0.raw"))


def test_retention_tiered_move_prefers_different_source_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    inflated = _write_locality_candidate(repo)
    cold_same = tmp_path / "cold_same"
    cold_other = tmp_path / "cold_other"
    cold_same.mkdir()
    cold_other.mkdir()
    plan = build_retention_plan([repo], repo_root=repo, min_bytes=1)
    original_device_id = retention._path_device_id

    def fake_device_id(path: Path) -> int:
        resolved = path.resolve(strict=False)
        if resolved == cold_other or cold_other in resolved.parents:
            return 2
        if resolved == cold_same or cold_same in resolved.parents:
            return 1
        if resolved == inflated or inflated in resolved.parents:
            return 1
        return original_device_id(path)

    monkeypatch.setattr(retention, "_path_device_id", fake_device_id)

    execution = execute_retention_plan(
        plan,
        action="move",
        cold_store_roots=[cold_same, cold_other],
    )

    assert execution["rows"][0]["cold_store_tier_index"] == 1
    assert execution["rows"][0]["cold_store_root"] == str(cold_other)
    assert not inflated.exists()
    assert not any(cold_same.rglob("0.raw"))
    assert (cold_other / plan.candidates[0].path / "0.raw").is_file()


def test_retention_tiered_move_respects_cold_store_reserve(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    inflated = _write_locality_candidate(repo)
    cold = tmp_path / "cold"
    cold.mkdir()
    plan = build_retention_plan([repo], repo_root=repo, min_bytes=1)

    with pytest.raises(ArtifactRetentionError, match="cold-store free space insufficient"):
        execute_retention_plan(
            plan,
            action="move",
            cold_store_roots=[cold],
            cold_store_reserve_bytes=10**18,
        )
    assert inflated.exists()


def test_retention_can_delete_certified_external_source_under_plan_root(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    external_root = tmp_path / "external_results"
    inflated = _write_locality_candidate(external_root)
    plan = build_retention_plan([external_root], repo_root=repo, min_bytes=1)

    assert Path(plan.candidates[0].path).is_absolute()

    execution = execute_retention_plan(plan, action="delete")

    assert execution["executed_count"] == 1
    assert not inflated.exists()


def test_retention_blocks_mutated_locality_raw_after_manifest(tmp_path: Path) -> None:
    inflated = _write_locality_candidate(tmp_path)
    (inflated / "0.raw").write_bytes(b"mutated")

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 1
    assert "locality_raw_sha_mismatch:0.raw" in plan.blocked_candidates[0].blockers


def test_retention_certifies_mlx_cache_with_external_identity_stamp(tmp_path: Path) -> None:
    cache = _write_mlx_delta_cache(tmp_path)

    plan = build_retention_plan(
        [tmp_path],
        repo_root=tmp_path,
        include_kinds={"mlx_scorer_input_cache"},
        min_bytes=1,
    )

    assert plan.blocked_candidates == []
    assert len(plan.candidates) == 1
    candidate = plan.candidates[0]
    assert candidate.path == cache.relative_to(tmp_path).as_posix()
    assert candidate.kind == "mlx_scorer_input_cache"
    identity = candidate.certificate["identity_audit"]
    assert identity["stamp_key"] == "local_cpu_advisory_cache_identity_audit"
    assert identity["source"]["key"] == "local_cpu_advisory_path"
    assert candidate.certificate["archive_sha256"] == "a" * 64


def test_retention_blocks_mlx_cache_without_identity_stamp(tmp_path: Path) -> None:
    cache = _write_mlx_delta_cache(tmp_path, stamp=False)

    plan = build_retention_plan(
        [tmp_path],
        repo_root=tmp_path,
        include_kinds={"mlx_scorer_input_cache"},
        min_bytes=1,
    )

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 1
    assert plan.blocked_candidates[0].path == cache.relative_to(tmp_path).as_posix()
    assert "mlx_cache_identity_audit_stamp_missing" in plan.blocked_candidates[0].blockers


def test_retention_certifies_inverse_scorer_strict_inflate_parity_raw_outputs(
    tmp_path: Path,
) -> None:
    source_out, candidate_out, proof_path = _write_inverse_scorer_parity_fixture(tmp_path)

    plan = build_retention_plan([tmp_path / "chain"], repo_root=tmp_path, min_bytes=1)

    assert plan.blocked_candidates == []
    assert [row.kind for row in plan.candidates] == [
        "inverse_scorer_inflate_parity_raw_output",
        "inverse_scorer_inflate_parity_raw_output",
    ]
    assert {row.certificate["role"] for row in plan.candidates} == {"source", "candidate"}
    assert all(row.certificate["proof_sha256"] == sha256_file(proof_path) for row in plan.candidates)
    assert all(row.certificate["descriptor_packet_sha256"] == "a" * 64 for row in plan.candidates)

    execution = execute_retention_plan(plan, action="delete")

    assert execution["executed_count"] == 2
    assert not source_out.exists()
    assert not candidate_out.exists()
    assert proof_path.is_file()
    assert (tmp_path / "chain" / "candidate_manifest.json").is_file()
    assert (tmp_path / "chain" / "runtime" / "inflate.sh").is_file()


def test_retention_blocks_inverse_scorer_parity_raw_with_truthy_authority(
    tmp_path: Path,
) -> None:
    source_out, candidate_out, _proof_path = _write_inverse_scorer_parity_fixture(
        tmp_path,
        proof_mutation={"score_claim": True},
    )

    plan = build_retention_plan([tmp_path / "chain"], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert {row.path for row in plan.blocked_candidates} == {
        source_out.relative_to(tmp_path).as_posix(),
        candidate_out.relative_to(tmp_path).as_posix(),
    }
    assert all(
        "inverse_scorer_parity_probe_score_claim_not_false" in row.blockers
        for row in plan.blocked_candidates
    )


def test_retention_blocks_inverse_scorer_parity_raw_when_proof_not_strict(
    tmp_path: Path,
) -> None:
    _source_out, _candidate_out, _proof_path = _write_inverse_scorer_parity_fixture(
        tmp_path,
        proof_mutation={
            "full_frame_inflate_output_parity_claim": False,
            "cleared_blockers": [],
            "blockers": ["inflate_output_bytes_not_identical"],
        },
    )

    plan = build_retention_plan([tmp_path / "chain"], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 2
    assert all(
        "inverse_scorer_full_frame_parity_claim_not_true" in row.blockers
        and "inverse_scorer_parity_probe_has_blockers" in row.blockers
        for row in plan.blocked_candidates
    )


def test_retention_blocks_inverse_scorer_parity_raw_without_runtime_rebuild_contract(
    tmp_path: Path,
) -> None:
    _source_out, _candidate_out, _proof_path = _write_inverse_scorer_parity_fixture(
        tmp_path,
        proof_mutation={
            "inflate_runtime": None,
            "source_archive_inflated": None,
            "candidate_archive_inflated": None,
        },
    )

    plan = build_retention_plan([tmp_path / "chain"], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 2
    assert all("inverse_scorer_inflate_runtime_missing" in row.blockers for row in plan.blocked_candidates)
    assert all("inverse_scorer_source_archive_inflated_missing" in row.blockers for row in plan.blocked_candidates)
    assert all("inverse_scorer_candidate_archive_inflated_missing" in row.blockers for row in plan.blocked_candidates)


def test_retention_blocks_inverse_scorer_parity_raw_with_symlink_reference(
    tmp_path: Path,
) -> None:
    _source_out, _candidate_out, proof_path = _write_inverse_scorer_parity_fixture(tmp_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    link = tmp_path / "chain" / "candidate-link.zip"
    try:
        link.symlink_to(tmp_path / "chain" / "candidate.zip")
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    proof["candidate_archive_inflated"]["path"] = _relative(link, tmp_path)
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    plan = build_retention_plan([tmp_path / "chain"], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 2
    assert all(
        "inverse_scorer_candidate_archive_inflated_path_is_symlink" in row.blockers
        for row in plan.blocked_candidates
    )


def test_retention_blocks_inverse_scorer_parity_raw_with_tree_blockers(
    tmp_path: Path,
) -> None:
    _source_out, _candidate_out, _proof_path = _write_inverse_scorer_parity_fixture(
        tmp_path,
        proof_mutation={
            "source_output_tree": {
                **_inverse_tree_record(
                    tmp_path / "chain" / "inflate_work" / "source" / "out",
                    tmp_path,
                ),
                "blockers": ["source_inflate_output_tree_contains_symlink"],
            }
        },
    )

    plan = build_retention_plan([tmp_path / "chain"], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert len(plan.blocked_candidates) == 2
    assert all(
        "inverse_scorer_source_output_tree_has_blockers" in row.blockers
        for row in plan.blocked_candidates
    )


def test_retention_reports_unknown_raw_surface(tmp_path: Path) -> None:
    raw_dir = tmp_path / "candidate_c" / "local_macos_cpu_eval_work" / "inflated"
    _write(raw_dir / "0.raw", b"r" * 8)

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert plan.candidates == []
    assert any(
        row.kind == "blocked_unknown_raw_surface"
        and "unknown_raw_surface_no_certifier" in row.blockers
        for row in plan.blocked_candidates
    )


def test_retention_reports_known_nested_raw_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "candidate_d" / "contest_auth_eval_cpu_workdir"
    raw_dir = workdir / "nested" / "inflated"
    _write(raw_dir / "0.raw", b"r" * 8)

    plan = build_retention_plan([tmp_path], repo_root=tmp_path, min_bytes=1)

    assert any(
        row.path == workdir.relative_to(tmp_path).as_posix()
        and row.kind == "blocked_unknown_raw_surface"
        and "unknown_raw_workdir_no_certifier" in row.blockers
        and row.certificate["nested_raw_file_count"] == 1
        for row in plan.blocked_candidates
    )
