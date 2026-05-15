# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path

import pytest

from tac.repo_io import read_json, write_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


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
    assert payload["outputs"]["archive"]["sha256"]
    written = read_json(output_dir / "materialization_manifest.json")
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
    assert "requires_exact_cuda_auth_eval_on_materialized_archive" in payload["dispatch_blockers"]


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
