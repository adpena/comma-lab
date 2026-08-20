from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

import tac.witness_dsl.taskspace_public_auth_eval_closure as closure
from tac.witness_dsl.taskspace_public_auth_eval_closure import (
    AuthClosureCheckpointArtifactV1,
    AuthClosureStageCheckpointV1,
    AuthClosureStageV1,
    PublicAuthClosureError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools/run_taskspace_public_auth_eval_closure.py"
G25_ARCHIVE = (
    REPO_ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "ep725_population_global_recode_v2_20260726_r2"
    / "ep725_population_global_recode_v2.not_a_candidate.zip"
)
LVLS1_RUNTIME = Path("/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py")
SPEC = importlib.util.spec_from_file_location("run_taskspace_public_auth_eval_closure", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _minimal_abi() -> closure.InterpreterDistributionABIClosureV1:
    return closure.InterpreterDistributionABIClosureV1(
        interpreter_implementation="cpython",
        interpreter_version="3.13.0",
        interpreter_cache_tag="cpython-313",
        interpreter_executable_name="python",
        interpreter_executable_realpath="/usr/bin/python",
        interpreter_executable_sha256="0" * 64,
        interpreter_prefix_realpaths=("/usr",),
        interpreter_prefix_tree_sha256=None,
        interpreter_prefix_tree_nbytes=0,
        soabi="cpython-313-x86_64-linux-gnu",
        multiarch="x86_64-linux-gnu",
        platform_system="Linux",
        platform_machine="x86_64",
        distributions=(),
        unresolved_import_roots=(),
    )


def _retained_artifact(run_dir: Path, relative: str, payload: bytes, kind: str) -> AuthClosureCheckpointArtifactV1:
    path = run_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return AuthClosureCheckpointArtifactV1(
        artifact_kind=kind,
        relative_path=relative,
        content_sha256=_sha(payload),
        nbytes=len(payload),
    )


def _checkpoint(
    *,
    run_dir: Path,
    ordinal: int,
    stage: AuthClosureStageV1,
    previous: AuthClosureStageCheckpointV1 | None,
    artifact: AuthClosureCheckpointArtifactV1,
    blockers: tuple[str, ...] = (),
) -> tuple[Path, AuthClosureStageCheckpointV1]:
    cleanup = _retained_artifact(
        run_dir,
        f"receipts/{ordinal:03d}_cleanup.json",
        f"cleanup-{ordinal}".encode("ascii"),
        "cleanup.certification",
    )
    receipt = AuthClosureStageCheckpointV1(
        run_id="g29.resume.test",
        stage=stage,
        stage_ordinal=ordinal,
        previous_checkpoint_sha256=None if previous is None else previous.identity_sha256,
        artifacts=tuple(sorted((artifact, cleanup), key=lambda item: (item.artifact_kind, item.relative_path))),
        blockers=blockers,
        completed=not blockers,
        research_only=True,
        cleanup_certification_sha256=cleanup.content_sha256,
    )
    path = run_dir / "checkpoints" / f"{ordinal:03d}_{stage.value.lower()}.json"
    receipt.write_atomic(path)
    return path, receipt


def test_atomic_bytes_is_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "retained.bin"
    runner._atomic_bytes(path, b"first")
    runner._atomic_bytes(path, b"first")
    with pytest.raises(PublicAuthClosureError, match="refusing to overwrite drifted"):
        runner._atomic_bytes(path, b"second")
    assert path.read_bytes() == b"first"


def test_resume_chain_reopens_every_parent_and_artifact(tmp_path: Path) -> None:
    artifact_zero = _retained_artifact(tmp_path, "receipts/compile.json", b"compile", "compile.receipt")
    _path_zero, checkpoint_zero = _checkpoint(
        run_dir=tmp_path,
        ordinal=0,
        stage=AuthClosureStageV1.COMPILE_PUBLIC_RUNTIME,
        previous=None,
        artifact=artifact_zero,
    )
    artifact_one = _retained_artifact(
        tmp_path,
        "receipts/discovery.json",
        b"discovery",
        "dependency.discovery.receipt",
    )
    path_one, checkpoint_one = _checkpoint(
        run_dir=tmp_path,
        ordinal=1,
        stage=AuthClosureStageV1.DEPENDENCY_DISCOVERY,
        previous=checkpoint_zero,
        artifact=artifact_one,
    )

    chain = runner._verify_resume_chain(
        run_dir=tmp_path,
        resume_path=path_one,
        run_id="g29.resume.test",
    )
    assert tuple(item.identity_sha256 for item in chain) == (
        checkpoint_zero.identity_sha256,
        checkpoint_one.identity_sha256,
    )

    (tmp_path / artifact_one.relative_path).write_bytes(b"tampered")
    with pytest.raises(PublicAuthClosureError, match="retained byte identity"):
        runner._verify_resume_chain(
            run_dir=tmp_path,
            resume_path=path_one,
            run_id="g29.resume.test",
        )


def test_resume_chain_rejects_noncontiguous_history(tmp_path: Path) -> None:
    artifact_zero = _retained_artifact(tmp_path, "receipts/compile.json", b"compile", "compile.receipt")
    _path_zero, checkpoint_zero = _checkpoint(
        run_dir=tmp_path,
        ordinal=0,
        stage=AuthClosureStageV1.COMPILE_PUBLIC_RUNTIME,
        previous=None,
        artifact=artifact_zero,
    )
    artifact_two = _retained_artifact(
        tmp_path,
        "receipts/readiness.json",
        b"readiness",
        "execution.readiness.receipt",
    )
    path_two, _checkpoint_two = _checkpoint(
        run_dir=tmp_path,
        ordinal=2,
        stage=AuthClosureStageV1.EXECUTION_PREFLIGHT,
        previous=checkpoint_zero,
        artifact=artifact_two,
    )
    with pytest.raises(PublicAuthClosureError, match="contiguous zero-based"):
        runner._verify_resume_chain(
            run_dir=tmp_path,
            resume_path=path_two,
            run_id="g29.resume.test",
        )


def test_repeated_blocked_readiness_resume_selects_latest_attempt(tmp_path: Path) -> None:
    artifact_zero = _retained_artifact(tmp_path, "receipts/compile.json", b"compile", "compile.receipt")
    _path_zero, checkpoint_zero = _checkpoint(
        run_dir=tmp_path,
        ordinal=0,
        stage=AuthClosureStageV1.COMPILE_PUBLIC_RUNTIME,
        previous=None,
        artifact=artifact_zero,
    )
    artifact_one = _retained_artifact(
        tmp_path,
        "receipts/readiness-1.json",
        b"readiness-1",
        "execution.readiness.receipt",
    )
    _path_one, checkpoint_one = _checkpoint(
        run_dir=tmp_path,
        ordinal=1,
        stage=AuthClosureStageV1.EXECUTION_PREFLIGHT,
        previous=checkpoint_zero,
        artifact=artifact_one,
        blockers=("BLOCKED",),
    )
    artifact_two = _retained_artifact(
        tmp_path,
        "receipts/readiness-2.json",
        b"readiness-2",
        "execution.readiness.receipt",
    )
    path_two, checkpoint_two = _checkpoint(
        run_dir=tmp_path,
        ordinal=2,
        stage=AuthClosureStageV1.EXECUTION_PREFLIGHT,
        previous=checkpoint_one,
        artifact=artifact_two,
        blockers=("BLOCKED",),
    )
    chain = runner._verify_resume_chain(
        run_dir=tmp_path,
        resume_path=path_two,
        run_id="g29.resume.test",
    )
    assert runner._select_retained_readiness(chain) == checkpoint_two


def test_cleanup_certification_retains_sources_and_declares_no_deletion(tmp_path: Path) -> None:
    source_a = tmp_path / "source.zip"
    source_b = tmp_path / "renderer.py"
    source_a.write_bytes(b"archive")
    source_b.write_bytes(b"renderer")
    runtime = tmp_path / "runtime/inflate.py"
    runtime.parent.mkdir()
    runtime.write_bytes(b"inverse")
    stage_artifact = _retained_artifact(tmp_path, "receipts/stage.json", b"stage", "stage.receipt")
    certification = runner._write_cleanup_certification(
        run_dir=tmp_path,
        run_id="g29.cleanup.test",
        stage=AuthClosureStageV1.COMPILE_PUBLIC_RUNTIME,
        ordinal=0,
        source_paths=(source_a, source_b),
        stage_artifacts=(stage_artifact,),
        invocation_argv=("runner", "--bounded"),
    )
    value = runner.json.loads(certification.read_bytes())
    assert value["large_artifacts_created"] is False
    assert value["deletions_performed"] == []
    assert {row["sha256"] for row in value["source_inputs"]} == {_sha(b"archive"), _sha(b"renderer")}


def test_main_resume_reopens_completed_compile_without_sources_or_recompile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not G25_ARCHIVE.is_file() or not LVLS1_RUNTIME.is_file():
        pytest.skip("real G25 archive or generic LVLS1 runtime is not mounted")
    monkeypatch.setattr(closure, "capture_interpreter_distribution_abi", lambda _roots: _minimal_abi())
    run_dir = tmp_path / "run"
    common = [
        "--repo-root",
        str(REPO_ROOT),
        "--run-dir",
        str(run_dir),
        "--run-id",
        "g29.resume.main",
        "--attest-generic-lineage",
        "--stop-after",
        "compile",
    ]
    assert (
        runner.main(
            [
                *common,
                "--archive",
                str(G25_ARCHIVE),
                "--lvls1-runtime",
                str(LVLS1_RUNTIME),
            ]
        )
        == 0
    )
    checkpoint = run_dir / "checkpoints/000_compile_public_runtime.json"
    assert checkpoint.is_file()
    monkeypatch.setattr(
        runner,
        "compile_lvpg2_public_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("completed compile reran")),
    )
    assert (
        runner.main(
            [
                *common,
                "--archive",
                str(tmp_path / "missing-archive.zip"),
                "--lvls1-runtime",
                str(tmp_path / "missing-renderer.py"),
                "--resume-from",
                str(checkpoint),
            ]
        )
        == 0
    )
