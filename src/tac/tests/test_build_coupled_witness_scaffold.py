from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.witness_dsl.coupled_witness_state import (
    CodecObjectManifest,
    CoupledWitnessState,
    WitnessCompileConfig,
)
from tools.build_coupled_witness_scaffold import (
    ScaffoldBuildError,
    _read_json,
    _write_staged_file_once,
    build_scaffold,
    run,
)


def _spec(repo: Path) -> dict[str, object]:
    files = {
        "source.bin": b"source",
        "evaluate.py": b"evaluate",
        "segnet.weights": b"segnet",
        "receiver.py": b"receiver",
    }
    for name, payload in files.items():
        (repo / name).write_bytes(payload)
    return {
        "schema": "tac.coupled_witness_scaffold_spec.v1",
        "source_video": {
            "path": "source.bin",
            "artifact_schema": "test.source.v1",
        },
        "evaluator_artifacts": [
            {
                "path": "evaluate.py",
                "artifact_schema": "test.evaluator.v1",
            },
            {
                "path": "segnet.weights",
                "artifact_schema": "test.weights.v1",
            },
        ],
        "receiver_artifacts": [
            {
                "path": "receiver.py",
                "artifact_schema": "test.receiver.v1",
            }
        ],
        "pair_count": 3,
        "pair_order_id": "canonical-contiguous-test.v1",
        "scorer_geometry": {"height": 4, "width": 4},
        "generation": {"seed": 7, "rng_id": "test-rng.v1"},
        "compile": {
            "container_id": "test-container.v1",
            "receiver_contract_id": "unimplemented-test-contract.v0",
            "r_chain_id": "test-r-chain.v1",
            "tie_policy_id": "test-tie-policy.v1",
            "camera_geometry": {"height": 8, "width": 8},
            "decoder_seed": 11,
            "decoder_payload_policy": (
                "counted-source-derived-statistics-only-no-scorer-no-gt-table"
            ),
        },
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }


def test_build_scaffold_binds_live_bytes_and_emits_no_false_authority(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    state, config, codec_object, receipt = build_scaffold(
        spec,
        repo_root=tmp_path,
        spec_file_sha256="a" * 64,
    )

    assert state.streams == ()
    assert state.frozen_space.source_video.sha256 == hashlib.sha256(b"source").hexdigest()
    assert config.stream_policies == ()
    assert codec_object.stage == "C0_IDENTITY_SCAFFOLD"
    assert receipt["archive_emitted"] is False
    assert receipt["decoded_output_emitted"] is False
    assert receipt["score_measured"] is False
    assert receipt["borrowed_candidate_bytes"] == 0


def test_run_writes_canonical_write_once_envelopes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    spec = _spec(repo)
    spec_path = repo / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = repo / "out"

    receipt = run(spec_path=spec_path, output_dir=output, repo_root=repo)

    state = CoupledWitnessState.from_bytes((output / "state.json").read_bytes())
    config = WitnessCompileConfig.from_bytes((output / "compile_config.json").read_bytes())
    codec_object = CodecObjectManifest.from_bytes((output / "codec_object.json").read_bytes())
    assert state.state_sha256 == receipt["state_sha256"]
    assert config.config_sha256 == receipt["compile_config_sha256"]
    assert codec_object.object_sha256 == receipt["codec_object_sha256"]
    with pytest.raises(ScaffoldBuildError, match="already exists"):
        run(spec_path=spec_path, output_dir=output, repo_root=repo)


def test_run_never_overwrites_destination_created_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    spec_path = repo / "spec.json"
    spec_path.write_text(json.dumps(_spec(repo)), encoding="utf-8")
    output = repo / "out"
    injected = False

    def inject_competing_destination(path: Path, payload: bytes) -> None:
        nonlocal injected
        _write_staged_file_once(path, payload)
        if not injected:
            injected = True
            output.mkdir()
            (output / "competing-owner.txt").write_text("do not overwrite", encoding="utf-8")

    monkeypatch.setattr(
        "tools.build_coupled_witness_scaffold._write_staged_file_once",
        inject_competing_destination,
    )
    with pytest.raises(ScaffoldBuildError, match="appeared during publication"):
        run(spec_path=spec_path, output_dir=output, repo_root=repo)

    assert (output / "competing-owner.txt").read_text(encoding="utf-8") == "do not overwrite"
    assert not (repo / ".out.publish.lock").exists()
    assert not tuple(repo.glob(".out.bundle.*"))


def test_run_failure_publishes_no_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    spec_path = repo / "spec.json"
    spec_path.write_text(json.dumps(_spec(repo)), encoding="utf-8")
    output = repo / "out"

    def fail_mid_bundle(path: Path, payload: bytes) -> None:
        if path.name == "compile_config.json":
            raise OSError("injected write failure")
        _write_staged_file_once(path, payload)

    monkeypatch.setattr(
        "tools.build_coupled_witness_scaffold._write_staged_file_once",
        fail_mid_bundle,
    )
    with pytest.raises(ScaffoldBuildError, match="cannot publish scaffold bundle"):
        run(spec_path=spec_path, output_dir=output, repo_root=repo)

    assert not output.exists()
    assert not (repo / ".out.publish.lock").exists()
    assert not tuple(repo.glob(".out.bundle.*"))


def test_run_hashes_and_parses_one_spec_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    spec_path = repo / "spec.json"
    spec_path.write_text(json.dumps(_spec(repo)), encoding="utf-8")
    snapshot = spec_path.read_bytes()
    monkeypatch.setattr(
        "tools.build_coupled_witness_scaffold._read_json",
        lambda _path: (_ for _ in ()).throw(AssertionError("spec reopened")),
    )

    receipt = run(spec_path=spec_path, output_dir=repo / "out", repo_root=repo)

    assert receipt["spec_file_sha256"] == hashlib.sha256(snapshot).hexdigest()


def test_scaffold_refuses_path_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    spec = _spec(repo)
    spec["source_video"] = {
        "path": "../outside.bin",
        "artifact_schema": "test.source.v1",
    }

    with pytest.raises(ScaffoldBuildError, match="repository-relative"):
        build_scaffold(spec, repo_root=repo, spec_file_sha256="b" * 64)


def test_scaffold_spec_reader_refuses_duplicate_keys(tmp_path: Path) -> None:
    spec_path = tmp_path / "duplicate.json"
    spec_path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")

    with pytest.raises(ScaffoldBuildError, match="duplicate JSON key"):
        _read_json(spec_path)
