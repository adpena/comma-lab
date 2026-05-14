# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tac.repo_io import read_json, write_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/materialize_pr106_yshift_score_table_candidate.py",
        "materialize_pr106_yshift_score_table_candidate_test",
    )


def test_resolves_nested_kaggle_yshift_score_table_layout(tmp_path: Path) -> None:
    module = _load_tool()
    score_root = tmp_path / "download"
    nested = score_root / "pr106_yshift_score_table" / "yshift_run" / "score_table"
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


def test_rejects_ambiguous_yshift_score_table_layout(tmp_path: Path) -> None:
    module = _load_tool()
    for rel in ["score_table", "pr106_yshift_score_table/yshift_run/score_table"]:
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


def test_yshift_materializer_runs_builder_and_writes_nonpromotional_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"archive")
    score_dir = tmp_path / "score_table"
    score_dir.mkdir()
    npy = score_dir / "score_table.npy"
    manifest = score_dir / "score_table_manifest.json"
    npy.write_bytes(b"npy")
    write_json(manifest, {"manifest_schema": "pr106_yshift_score_table_manifest_v1"})
    output_dir = tmp_path / "out"
    calls: dict[str, object] = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pr106_yshift_sidechannel_archive.zip").write_bytes(b"candidate")
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
        candidate_radius=3,
        score_step=0.5,
        n_pairs=600,
        python_executable="python",
    )

    command = calls["command"]
    assert command[:2] == ["python", str(module.BUILDER)]
    assert command[command.index("--search-mode") + 1] == "score_table"
    assert command[command.index("--score-table-npy") + 1] == str(npy)
    assert command[command.index("--score-table-manifest") + 1] == str(manifest)
    assert command[command.index("--candidate-radius") + 1] == "3"
    assert command[command.index("--score-step") + 1] == "0.5"
    assert calls["kwargs"]["check"] is True
    assert calls["kwargs"]["timeout"] == 600
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    written = read_json(output_dir / "materialization_manifest.json")
    assert written["outputs"]["materialization_manifest"]["sha256"]


def test_yshift_materializer_refuses_promotional_builder_metadata(monkeypatch, tmp_path: Path) -> None:
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
        (output_dir / "pr106_yshift_sidechannel_archive.zip").write_bytes(b"candidate")
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
            candidate_radius=3,
            score_step=1.0,
            n_pairs=600,
            python_executable="python",
        )
