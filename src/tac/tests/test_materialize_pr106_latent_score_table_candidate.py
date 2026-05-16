# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.repo_io import read_json, write_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]
FORMAT07_FIXTURE = REPO_ROOT / "src/tac/tests/fixtures/pr106_hdm8_format07.archive.zip"


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/materialize_pr106_latent_score_table_candidate.py",
        "materialize_pr106_latent_score_table_candidate_test",
    )


def _stored_zip(path: Path, payload: bytes = b"payload", *, member_name: str = "0.bin") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        zf.writestr(info, payload)


def _complete_score_table_manifest(source_archive: Path, npy: Path) -> dict[str, object]:
    with zipfile.ZipFile(source_archive) as zf:
        member_name = zf.namelist()[0]
        payload = zf.read(member_name)
    return {
        "manifest_schema": "pr106_latent_score_table_manifest_v1",
        "producer": "experiments/build_pr106_latent_score_table.py",
        "score_claim": False,
        "ready_for_builder": True,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "remote_jobs_dispatched": False,
        "source_archive_path": str(source_archive),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        "source_archive_member_name": member_name,
        "source_archive_member_sha256": hashlib.sha256(payload).hexdigest(),
        "source_payload_kind": "raw_pr106_packed_archive",
        "runtime_dir": "submissions/pr106_latent_sidecar_r2_pr101_grammar",
        "source_zero_bin_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_grid_path": "candidate_grid.npy",
        "candidate_grid_sha256": "a" * 64,
        "candidate_grid_npy_sha256": "b" * 64,
        "score_table_npy_path": str(npy),
        "score_table_npy_bytes": npy.stat().st_size,
        "score_table_npy_sha256": hashlib.sha256(npy.read_bytes()).hexdigest(),
        "delta_radius": 2,
        "latent_dim": 28,
        "candidate_count": 113,
        "n_pairs": 600,
        "score_table_shape": [600, 113],
        "objective": "100*seg_dist + sqrt(10*pose_dist), without rate constant",
        "pair_marginal_semantics": "one latent perturbation scored against the official two-frame pair",
        "noop_candidate_index": 0,
        "strict_improvement_pair_count": 0,
        "best_improvement_min": 0.0,
        "best_improvement_mean": 0.0,
        "best_improvement_max": 0.0,
        "device": "cuda",
        "torch_version": "test",
        "cuda_version": "test",
        "elapsed_seconds": 1.0,
        "lane_claim_verified": True,
        "lane_claim": {"lane_id": "lane_pr106_latent_sidecar"},
        "dispatch_blockers": [
            "requires_archive_build_from_table",
            "requires_exact_cuda_auth_eval_on_built_archive",
        ],
    }


def _compatible_score_table_delta(base_delta: int) -> int:
    if base_delta <= -1:
        return 1
    return -1


def _format0c_source_archive(tmp_path: Path) -> tuple[Path, int, int]:
    from tac.packet_compiler.pr106_sidecar_packet import (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
        decode_pr106_sidecar_packet_dim_delta,
        emit_pr106_sidecar_recode_candidate_archive,
        lossless_pr106_sidecar_recode_candidates,
        parse_pr106_sidecar_packet,
        read_single_stored_member_archive,
    )

    fixture_member = read_single_stored_member_archive(FORMAT07_FIXTURE.read_bytes())
    fixture_packet = parse_pr106_sidecar_packet(fixture_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(fixture_packet)
    candidate = next(
        item
        for item in lossless_pr106_sidecar_recode_candidates(dims, deltas)
        if item.sidecar_format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    )
    _member, archive_bytes = emit_pr106_sidecar_recode_candidate_archive(
        fixture_member,
        fixture_packet,
        candidate,
    )
    source_archive = tmp_path / "format0c_x.zip"
    source_archive.write_bytes(archive_bytes)
    return source_archive, int(dims[0]), int(deltas[0])


def _format07_source_archive(tmp_path: Path) -> tuple[Path, int, int, int]:
    from tac.packet_compiler.pr106_sidecar_packet import (
        PR106_NO_OP_DIM,
        decode_pr106_sidecar_packet_dim_delta,
        parse_pr106_sidecar_packet,
        read_single_stored_member_archive,
    )

    source_archive = tmp_path / "format07_x.zip"
    source_archive.write_bytes(FORMAT07_FIXTURE.read_bytes())
    fixture_member = read_single_stored_member_archive(source_archive.read_bytes())
    fixture_packet = parse_pr106_sidecar_packet(fixture_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(fixture_packet)
    pair_idx = int(np.flatnonzero((dims != PR106_NO_OP_DIM) & (deltas != 0))[0])
    return source_archive, pair_idx, int(dims[pair_idx]), int(deltas[pair_idx])


def test_resolves_nested_kaggle_score_table_layout(tmp_path: Path) -> None:
    module = _load_tool()
    score_root = tmp_path / "download"
    nested = score_root / "pr106_latent_score_table" / "latent_run" / "score_table"
    nested.mkdir(parents=True)
    npy = nested / "score_table.npy"
    manifest = nested / "score_table_manifest.json"
    npy.write_bytes(b"npy")
    write_json(manifest, {"score_claim": False})

    resolved = module.resolve_score_table_artifacts(
        score_table_root=score_root,
        score_table_npy=None,
        score_table_manifest=None,
    )

    assert resolved == (npy.resolve(), manifest.resolve())


def test_rejects_ambiguous_score_table_layout(tmp_path: Path) -> None:
    module = _load_tool()
    for rel in ["score_table", "pr106_latent_score_table/latent_run/score_table"]:
        root = tmp_path / rel
        root.mkdir(parents=True)
        (root / "score_table.npy").write_bytes(b"npy")
        write_json(root / "score_table_manifest.json", {"score_claim": False})

    with pytest.raises(ValueError, match="multiple score-table artifact pairs"):
        module.resolve_score_table_artifacts(
            score_table_root=tmp_path,
            score_table_npy=None,
            score_table_manifest=None,
        )


def test_materializer_runs_builder_and_writes_nonpromotional_manifest(monkeypatch, tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"archive")
    score_dir = tmp_path / "score_table"
    score_dir.mkdir()
    npy = score_dir / "score_table.npy"
    manifest = score_dir / "score_table_manifest.json"
    npy.write_bytes(b"npy")
    write_json(manifest, {"manifest_schema": "pr106_latent_score_table_manifest_v1"})
    output_dir = tmp_path / "out"
    calls: dict[str, object] = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sidecar_archive.zip").write_bytes(b"candidate")
        write_json(
            output_dir / "build_metadata.json",
            {
                "score_claim": False,
                "search_mode": "score_table",
                "score_table": {"score_table_manifest_validated": True},
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="builder ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    payload = module.materialize_candidate(
        source_archive=source_archive,
        output_dir=output_dir,
        score_table_root=tmp_path,
        score_table_npy=npy,
        score_table_manifest=manifest,
        delta_radius=2,
        top_k=600,
        python_executable="python",
    )

    command = calls["command"]
    assert command[:2] == ["python", str(module.BUILDER)]
    assert "--search-mode" in command
    assert command[command.index("--search-mode") + 1] == "score_table"
    assert command[command.index("--score-table-npy") + 1] == str(npy)
    assert command[command.index("--score-table-manifest") + 1] == str(manifest)
    assert command[command.index("--delta-radius") + 1] == "2"
    assert calls["kwargs"]["check"] is True
    assert calls["kwargs"]["timeout"] == 600
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["target_modes"] == ["contest_exact_eval_planning"]
    assert "score_table_manifest_missing_custody_fields" in payload["dispatch_blockers"]
    assert payload["score_table_manifest_audit"]["score_claim"] is False
    assert payload["score_table_manifest_audit"]["ready_for_exact_eval_dispatch"] is False
    assert "source_archive_sha256" in payload["score_table_manifest_audit"]["missing_custody_fields"]
    assert "score_table_manifest_missing_custody_fields" in payload["score_claim_blockers"]
    assert payload["candidate_archive"] == payload["outputs"]["archive"]
    assert payload["outputs"]["archive"]["sha256"]
    written = read_json(output_dir / "materialization_manifest.json")
    assert written["candidate_archive"] == written["outputs"]["archive"]
    assert written["outputs"]["materialization_manifest"]["sha256"]
    assert written["ready_for_exact_eval_dispatch"] is False


def test_materializer_refuses_promotional_builder_metadata(monkeypatch, tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"archive")
    npy = tmp_path / "score_table.npy"
    manifest = tmp_path / "score_table_manifest.json"
    npy.write_bytes(b"npy")
    write_json(manifest, {})
    output_dir = tmp_path / "out"

    def fake_run(command, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sidecar_archive.zip").write_bytes(b"candidate")
        write_json(
            output_dir / "build_metadata.json",
            {
                "score_claim": True,
                "search_mode": "score_table",
                "score_table": {"score_table_manifest_validated": True},
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="score_claim=false"):
        module.materialize_candidate(
            source_archive=source_archive,
            output_dir=output_dir,
            score_table_root=tmp_path,
            score_table_npy=npy,
            score_table_manifest=manifest,
            delta_radius=2,
            top_k=600,
            python_executable="python",
        )


def test_manifest_audit_clears_missing_custody_blocker_for_complete_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    _stored_zip(source_archive)
    npy = tmp_path / "score_table.npy"
    npy.write_bytes(b"npy")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(manifest, _complete_score_table_manifest(source_archive, npy))
    output_dir = tmp_path / "out"

    def fake_run(command, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sidecar_archive.zip").write_bytes(b"candidate")
        write_json(
            output_dir / "build_metadata.json",
            {
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "search_mode": "score_table",
                "dispatch_blockers": ["requires_exact_cuda_auth_eval"],
                "score_table": {"score_table_manifest_validated": True},
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    payload = module.materialize_candidate(
        source_archive=source_archive,
        output_dir=output_dir,
        score_table_root=tmp_path,
        score_table_npy=npy,
        score_table_manifest=manifest,
        delta_radius=2,
        top_k=600,
        python_executable="python",
    )

    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_table_manifest_audit"]["missing_custody_fields"] == []
    assert "score_table_manifest_missing_custody_fields" not in payload["exact_eval_dispatch_blockers"]
    assert "requires_paired_contest_cuda_auth_eval_on_materialized_archive" in payload[
        "dispatch_blockers"
    ]
    assert "requires_paired_contest_cpu_auth_eval_on_materialized_archive" in payload[
        "dispatch_blockers"
    ]
    assert (
        "requires_runtime_decode_apply_proof_for_semantic_packetir_candidate"
        in payload["exact_eval_dispatch_blockers"]
    )


def test_manifest_audit_accepts_x_member_custody(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    _stored_zip(source_archive, payload=b"x-payload", member_name="x")
    npy = tmp_path / "score_table.npy"
    npy.write_bytes(b"npy")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(manifest, _complete_score_table_manifest(source_archive, npy))

    audit = module.audit_score_table_manifest(
        source_archive=source_archive,
        score_table_npy=npy,
        score_table_manifest=manifest,
    )

    assert audit["blockers"] == []
    source = audit["source_archive"]
    assert source["single_member_name"] == "x"
    assert source["manifest_member_name"] == "x"
    assert source["single_member_name_match"] is True
    assert source["single_member_payload_sha256_match"] is True


def test_manifest_audit_blocks_format0c_without_explicit_x_member_name(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive, _base_dim0, _base_delta0 = _format0c_source_archive(tmp_path)
    npy = tmp_path / "score_table.npy"
    npy.write_bytes(b"npy")
    with zipfile.ZipFile(source_archive) as zf:
        payload = zf.read("x")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(
        manifest,
        {
            **_complete_score_table_manifest(source_archive, npy),
            "source_archive_member_name": None,
            "source_archive_member_sha256": hashlib.sha256(payload).hexdigest(),
            "source_zero_bin_sha256": None,
        },
    )

    audit = module.audit_score_table_manifest(
        source_archive=source_archive,
        score_table_npy=npy,
        score_table_manifest=manifest,
    )

    assert "score_table_manifest_packetir_member_name_mismatch_or_missing" in audit[
        "blockers"
    ]


def test_manifest_audit_blocks_format07_without_explicit_member_sha(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive, _pair_idx, _base_dim, _base_delta = _format07_source_archive(tmp_path)
    npy = tmp_path / "score_table.npy"
    npy.write_bytes(b"npy")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(
        manifest,
        {
            **_complete_score_table_manifest(source_archive, npy),
            "source_archive_member_sha256": None,
            "source_zero_bin_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        },
    )

    audit = module.audit_score_table_manifest(
        source_archive=source_archive,
        score_table_npy=npy,
        score_table_manifest=manifest,
    )

    assert "score_table_manifest_packetir_member_sha256_mismatch_or_missing" in audit[
        "blockers"
    ]


def test_format07_materializer_uses_native_packet_ir_not_legacy_builder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tool()
    source_archive, pair_idx, base_dim, base_delta = _format07_source_archive(tmp_path)
    npy = tmp_path / "score_table.npy"
    candidates = module.build_latent_candidate_grid(latent_dim=28, delta_radius=2)
    table = np.full((600, int(candidates.shape[0])), 10.0, dtype=np.float32)
    selected_delta = _compatible_score_table_delta(base_delta)
    selected_index = 1 + base_dim * 4 + {-2: 0, -1: 1, 1: 2, 2: 3}[selected_delta]
    table[pair_idx, selected_index] = 9.0
    np.save(npy, table, allow_pickle=False)
    with zipfile.ZipFile(source_archive) as zf:
        member_name = zf.namelist()[0]
        payload = zf.read(member_name)
    manifest = tmp_path / "score_table_manifest.json"
    write_json(
        manifest,
        {
            **_complete_score_table_manifest(source_archive, npy),
            "source_archive_member_name": member_name,
            "source_archive_member_sha256": hashlib.sha256(payload).hexdigest(),
            "source_zero_bin_sha256": None,
            "source_payload_kind": "pr106_sidecar_packet",
            "delta_radius": 2,
            "candidate_count": int(candidates.shape[0]),
            "score_table_shape": [600, int(candidates.shape[0])],
            "candidate_grid_sha256": module.latent_candidate_grid_npy_sha256(candidates),
            "candidate_grid_npy_sha256": module.latent_candidate_grid_npy_sha256(candidates),
        },
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("legacy builder subprocess must not run for PacketIR")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    output_dir = tmp_path / "out"
    materialization = module.materialize_candidate(
        source_archive=source_archive,
        output_dir=output_dir,
        score_table_root=tmp_path,
        score_table_npy=npy,
        score_table_manifest=manifest,
        delta_radius=2,
        top_k=600,
        python_executable="python",
    )

    assert materialization["builder"]["materialization_engine"] == "format07_packet_ir_native"
    assert materialization["builder"]["command"] == []
    metadata = read_json(output_dir / "build_metadata.json")
    assert metadata["materialization_engine"] == "format07_packet_ir_native"
    assert metadata["source_archive_member_name"] == member_name
    assert metadata["semantic_materialization"]["composed_same_dim_pair_count"] == 1
    from tac.packet_compiler.pr106_sidecar_packet import (
        decode_pr106_sidecar_packet_dim_delta,
        parse_pr106_sidecar_packet,
        read_single_stored_member_archive,
    )

    member = read_single_stored_member_archive((output_dir / "sidecar_archive.zip").read_bytes())
    assert member.name == member_name
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    assert int(dims[pair_idx]) == base_dim
    assert int(deltas[pair_idx]) == base_delta + selected_delta
    assert materialization["source_runtime"]["runtime_source_manifest"]


def test_materializer_refuses_unparsable_packet_ir_legacy_route(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    _stored_zip(source_archive, payload=b"\xfe\xffbad-packet-ir", member_name="0.bin")
    npy = tmp_path / "score_table.npy"
    npy.write_bytes(b"npy")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(manifest, _complete_score_table_manifest(source_archive, npy))

    def fail_run(*_args, **_kwargs):
        raise AssertionError("legacy builder subprocess must not run for bad PacketIR")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="unparsable PR106 PacketIR"):
        module.materialize_candidate(
            source_archive=source_archive,
            output_dir=tmp_path / "out",
            score_table_root=tmp_path,
            score_table_npy=npy,
            score_table_manifest=manifest,
            delta_radius=2,
            top_k=600,
            python_executable="python",
        )


def test_format0c_materializer_uses_native_packet_ir_not_legacy_builder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tool()
    source_archive, base_dim0, base_delta0 = _format0c_source_archive(tmp_path)
    npy = tmp_path / "score_table.npy"
    candidates = module.build_latent_candidate_grid(latent_dim=28, delta_radius=2)
    table = np.full((600, int(candidates.shape[0])), 10.0, dtype=np.float32)
    selected_delta = _compatible_score_table_delta(base_delta0)
    selected_index = 1 + base_dim0 * 4 + {-2: 0, -1: 1, 1: 2, 2: 3}[selected_delta]
    table[0, selected_index] = 9.0
    np.save(npy, table, allow_pickle=False)
    with zipfile.ZipFile(source_archive) as zf:
        payload = zf.read("x")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(
        manifest,
        {
            **_complete_score_table_manifest(source_archive, npy),
            "source_archive_member_name": "x",
            "source_archive_member_sha256": hashlib.sha256(payload).hexdigest(),
            "source_zero_bin_sha256": None,
            "source_payload_kind": "pr106_sidecar_packet",
            "delta_radius": 2,
            "candidate_count": int(candidates.shape[0]),
            "score_table_shape": [600, int(candidates.shape[0])],
            "candidate_grid_sha256": module.latent_candidate_grid_npy_sha256(candidates),
            "candidate_grid_npy_sha256": module.latent_candidate_grid_npy_sha256(candidates),
        },
    )
    output_dir = tmp_path / "out"

    def fail_run(*_args, **_kwargs):
        raise AssertionError("legacy builder subprocess must not run for format0C")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    payload = module.materialize_candidate(
        source_archive=source_archive,
        output_dir=output_dir,
        score_table_root=tmp_path,
        score_table_npy=npy,
        score_table_manifest=manifest,
        delta_radius=2,
        top_k=600,
        python_executable="python",
    )

    assert payload["builder"]["materialization_engine"] == "format0c_packet_ir_native"
    assert payload["builder"]["command"] == []
    metadata = read_json(output_dir / "build_metadata.json")
    assert metadata["materialization_engine"] == "format0c_packet_ir_native"
    runtime_manifest = metadata["source_runtime"]["runtime_source_manifest"]
    assert runtime_manifest["runtime_source_tree_sha256"]
    assert runtime_manifest["runtime_dir"] == "submissions/pr106_latent_sidecar_r2_pr101_grammar"
    assert metadata["source_runtime"]["runtime_dir"] == runtime_manifest["runtime_dir"]
    assert "inflate.sh" in [item["path"] for item in runtime_manifest["files"]]
    assert metadata["score_table"]["score_table_manifest_validated"] is True
    assert metadata["semantic_materialization"]["composed_same_dim_pair_count"] == 1
    from tac.packet_compiler.pr106_sidecar_packet import (
        decode_pr106_sidecar_packet_dim_delta,
        parse_pr106_sidecar_packet,
        read_single_stored_member_archive,
    )

    member = read_single_stored_member_archive((output_dir / "sidecar_archive.zip").read_bytes())
    assert member.name == "x"
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    assert int(dims[0]) == base_dim0
    assert int(deltas[0]) == base_delta0 + selected_delta
    assert "requires_paired_contest_cuda_auth_eval_on_materialized_archive" in payload[
        "dispatch_blockers"
    ]
    assert "requires_paired_contest_cpu_auth_eval_on_materialized_archive" in payload[
        "dispatch_blockers"
    ]
    assert payload["source_runtime"]["runtime_source_manifest"] == runtime_manifest


def test_format0c_materializer_keeps_noop_score_table_as_format0c_noop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tool()
    source_archive, _base_dim0, _base_delta0 = _format0c_source_archive(tmp_path)
    npy = tmp_path / "score_table.npy"
    candidates = module.build_latent_candidate_grid(latent_dim=28, delta_radius=2)
    np.save(npy, np.full((600, int(candidates.shape[0])), 10.0, dtype=np.float32))
    with zipfile.ZipFile(source_archive) as zf:
        source_payload = zf.read("x")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(
        manifest,
        {
            **_complete_score_table_manifest(source_archive, npy),
            "source_archive_member_name": "x",
            "source_archive_member_sha256": hashlib.sha256(source_payload).hexdigest(),
            "source_zero_bin_sha256": None,
            "source_payload_kind": "pr106_sidecar_packet",
            "delta_radius": 2,
            "candidate_count": int(candidates.shape[0]),
            "score_table_shape": [600, int(candidates.shape[0])],
            "candidate_grid_sha256": module.latent_candidate_grid_npy_sha256(candidates),
            "candidate_grid_npy_sha256": module.latent_candidate_grid_npy_sha256(candidates),
        },
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("legacy builder subprocess must not run for format0C")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    output_dir = tmp_path / "out"
    materialization = module.materialize_candidate(
        source_archive=source_archive,
        output_dir=output_dir,
        score_table_root=tmp_path,
        score_table_npy=npy,
        score_table_manifest=manifest,
        delta_radius=2,
        top_k=600,
        python_executable="python",
    )

    metadata = read_json(output_dir / "build_metadata.json")
    member = module.read_single_stored_member_archive(
        (output_dir / "sidecar_archive.zip").read_bytes()
    )
    assert metadata["materialization_engine"] == "format0c_packet_ir_native"
    assert metadata["candidate_sidecar_format_id"] == "0x0C"
    assert metadata["diagnostics"]["unfiltered_strict_improvement_pair_count"] == 0
    assert metadata["semantic_materialization"]["changed_pair_count"] == 0
    assert member.payload == source_payload
    assert materialization["packet_ir_score_affecting_payload_changed"] is False
    assert materialization["packet_ir_charged_archive_bytes_delta"] == 0
    assert (
        module.PACKET_IR_NOOP_DISPATCH_BLOCKER
        in materialization["exact_eval_dispatch_blockers"]
    )
    assert materialization["packet_ir_noop_dispatch_blockers"] == [
        module.PACKET_IR_NOOP_DISPATCH_BLOCKER
    ]


def test_packet_ir_noop_blocker_allows_rate_only_shrink() -> None:
    module = _load_tool()
    build_metadata = {
        "source_archive_bytes": 1000,
        "archive_zip_bytes": 990,
        "packet_ir": {
            "source": {"emitted_payload_sha256": "a" * 64},
            "candidate": {"emitted_payload_sha256": "a" * 64},
        },
    }

    assert module.packet_ir_score_affecting_payload_changed(build_metadata) is False
    assert module.packet_ir_charged_archive_bytes_delta(build_metadata) == -10
    assert module.packet_ir_materialization_noop_blockers(build_metadata) == []


def test_format0c_materializer_routes_incompatible_strict_score_table_dims_to_format0d(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tool()
    source_archive, base_dim0, base_delta0 = _format0c_source_archive(tmp_path)
    npy = tmp_path / "score_table.npy"
    candidates = module.build_latent_candidate_grid(latent_dim=28, delta_radius=2)
    table = np.full((600, int(candidates.shape[0])), 10.0, dtype=np.float32)
    selected_delta = _compatible_score_table_delta(base_delta0)
    compatible_index = 1 + base_dim0 * 4 + {-2: 0, -1: 1, 1: 2, 2: 3}[selected_delta]
    incompatible_dim = (base_dim0 + 1) % 28
    incompatible_index = 1 + incompatible_dim * 4 + {
        -2: 0,
        -1: 1,
        1: 2,
        2: 3,
    }[selected_delta]
    table[0, incompatible_index] = 8.0
    table[0, compatible_index] = 9.0
    np.save(npy, table, allow_pickle=False)
    with zipfile.ZipFile(source_archive) as zf:
        source_payload = zf.read("x")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(
        manifest,
        {
            **_complete_score_table_manifest(source_archive, npy),
            "source_archive_member_name": "x",
            "source_archive_member_sha256": hashlib.sha256(source_payload).hexdigest(),
            "source_zero_bin_sha256": None,
            "source_payload_kind": "pr106_sidecar_packet",
            "delta_radius": 2,
            "candidate_count": int(candidates.shape[0]),
            "score_table_shape": [600, int(candidates.shape[0])],
            "candidate_grid_sha256": module.latent_candidate_grid_npy_sha256(candidates),
            "candidate_grid_npy_sha256": module.latent_candidate_grid_npy_sha256(candidates),
        },
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("legacy builder subprocess must not run for format0C")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    output_dir = tmp_path / "out"
    materialization = module.materialize_candidate(
        source_archive=source_archive,
        output_dir=output_dir,
        score_table_root=tmp_path,
        score_table_npy=npy,
        score_table_manifest=manifest,
        delta_radius=2,
        top_k=600,
        python_executable="python",
    )

    metadata = read_json(output_dir / "build_metadata.json")
    assert metadata["materialization_engine"] == "format0d_packet_ir_native"
    assert metadata["candidate_sidecar_format_id"] == "0x0D"
    assert materialization["candidate_archive"] == materialization["outputs"]["archive"]
    assert materialization["packet_ir_score_affecting_payload_changed"] is True
    assert materialization["packet_ir_charged_archive_bytes_changed"] is True
    assert (
        "requires_runtime_decode_apply_proof_for_semantic_packetir_candidate"
        in materialization["exact_eval_dispatch_blockers"]
    )
    assert metadata["diagnostics"]["pairs_with_incompatible_original_best"] == 1
    assert metadata["diagnostics"]["pairs_with_incompatible_original_strict_best"] == 1
    assert (
        metadata["diagnostics"]["score_table_selection_policy"]
        == "format0d_unfiltered_strict_best_extra_stream"
    )
    assert metadata["semantic_materialization"]["extra_second_dim_pair_count"] == 1
    from tac.packet_compiler.pr106_sidecar_packet import (
        PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA,
        decode_pr106_sidecar_packet_correction_passes,
        parse_pr106_sidecar_packet,
        read_single_stored_member_archive,
    )

    member = read_single_stored_member_archive((output_dir / "sidecar_archive.zip").read_bytes())
    assert hashlib.sha256(member.payload).hexdigest() != hashlib.sha256(
        source_payload
    ).hexdigest()
    packet = parse_pr106_sidecar_packet(member.payload)
    assert packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA
    (base_dims, base_deltas), (extra_dims, extra_deltas) = (
        decode_pr106_sidecar_packet_correction_passes(packet)
    )
    assert int(base_dims[0]) == base_dim0
    assert int(base_deltas[0]) == base_delta0
    assert int(extra_dims[0]) == incompatible_dim
    assert int(extra_deltas[0]) == selected_delta
    assert materialization["builder"]["materialization_engine"] == "format0d_packet_ir_native"


def test_materializer_downgrades_builder_exact_ready_claim(monkeypatch, tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    _stored_zip(source_archive)
    npy = tmp_path / "score_table.npy"
    npy.write_bytes(b"npy")
    manifest = tmp_path / "score_table_manifest.json"
    write_json(manifest, _complete_score_table_manifest(source_archive, npy))
    output_dir = tmp_path / "out"

    def fake_run(command, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sidecar_archive.zip").write_bytes(b"candidate")
        write_json(
            output_dir / "build_metadata.json",
            {
                "score_claim": False,
                "ready_for_exact_eval_dispatch": True,
                "search_mode": "score_table",
                "dispatch_blockers": [],
                "score_table": {"score_table_manifest_validated": True},
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    payload = module.materialize_candidate(
        source_archive=source_archive,
        output_dir=output_dir,
        score_table_root=tmp_path,
        score_table_npy=npy,
        score_table_manifest=manifest,
        delta_radius=2,
        top_k=600,
        python_executable="python",
    )

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "builder_metadata_claims_exact_eval_dispatch_ready" in payload["dispatch_blockers"]
    assert payload["builder_metadata_audit"]["authority_flags"]["ready_for_exact_eval_dispatch"] is True
