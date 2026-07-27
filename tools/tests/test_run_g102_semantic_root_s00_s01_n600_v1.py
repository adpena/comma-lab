# SPDX-License-Identifier: MIT
"""Fail-closed contract tests for the G102 S00→S01 n600 runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from comma_lab.storage_tiers import StorageTierSpec
from tac import score_geometry
from tac.witness_dsl.dynamic_frontier_target import load_dynamic_frontier_target

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "experiments/run_g102_semantic_root_s00_s01_n600_v1.py"
SPEC = importlib.util.spec_from_file_location("g102_s00_s01_runner_v1", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config(**overrides):
    values = {
        "run_id": "g102-s00-s01-fixture",
        "seed": 1729,
        "requested_bytes": MODULE.MIN_REQUESTED_BYTES,
        "reserve_free_gib": 0.0,
        "compiler_source_path": "src/tac/witness_dsl/taskspace_pfree_semantic_root_v1.py",
        "compiler_source_sha256": "1" * 64,
        "public_codec_section_path": "submissions/robust_current/g102_semantic_root",
        "public_codec_section_sha256": "2" * 64,
        "g46_batch_geometry_audit_path": (
            ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
            "g46_teacher_batch_geometry_audit_v2_20260726.json"
        ),
        "g46_batch_geometry_audit_sha256": "3" * 64,
        "evaluator_source_sha256": MODULE._sha256_file(REPO / MODULE.EVALUATOR_ENTRYPOINT),
        "upstream_snapshot_sha256": "4" * 64,
        "eval_device": "cpu",
    }
    values.update(overrides)
    return MODULE.G102RunnerConfigV1(**values)


def _tier(tmp_path: Path) -> tuple[StorageTierSpec, ...]:
    return (
        StorageTierSpec(
            name="test-storage",
            root=tmp_path,
            priority=0,
            reserve_free_bytes=0,
            allow_create=True,
            allow_local_disk=True,
        ),
    )


def _capability() -> dict[str, object]:
    return {
        "interface_id": MODULE.CAPABILITY_INTERFACE_ID,
        "producer_identity": "fresh_own_lineage_semantic_root_y1_v1",
        "own_lineage": True,
        "p_free": True,
        "full_population_n600": True,
        "label_topology_is_one_factor": True,
        "label_mask_palette_only": False,
        "scorer_native_rgb_appearance": True,
        "chroma_gauge": True,
        "parallax_gauge": True,
        "irreducible_rgb_quotient_seam": True,
        "exact_post_r_seg_closure": True,
        "exact_post_r_pose_closure": True,
        "teacher_quarantined": True,
        "scorer_free_receiver": True,
        "public_codec_section_sha256": "2" * 64,
    }


def _lineage_records(tmp_path: Path) -> tuple[dict[str, object], ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, role in enumerate(sorted(MODULE.CORE_LINEAGE_ROLES)):
        path = tmp_path / f"source_{index:02d}.bin"
        path.write_bytes(f"{role}\n".encode())
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": MODULE._sha256_file(path),
                "role": role,
                "candidate_dependency": role != "PUBLIC_RUNTIME_SOURCE",
                "packaged_in_archive": False,
                "video_derived": role
                in {
                    "FRESH_SOURCE_VIDEO_ENCODER_INPUT",
                    "G46_BATCH16_AUDIT_ENCODER_ONLY",
                    "G46_PRIMARY_RECEIPT_ENCODER_ONLY",
                    "G46_TARGET_LABELS_ENCODER_ONLY",
                },
            }
        )
    return tuple(sorted(rows, key=lambda row: (row["role"], row["path"], row["sha256"])))


def _lineage_manifest(
    *,
    packet: bytes,
    config_sha256: str,
    policy_sha256: str,
    records: tuple[dict[str, object], ...],
) -> bytes:
    body = {
        "schema": MODULE.SOURCE_LINEAGE_SCHEMA,
        "producer_identity": "fresh_own_lineage_semantic_root_y1_v1",
        "packet_sha256": MODULE._sha256(packet),
        "config_sha256": config_sha256,
        "lineage_policy_sha256": policy_sha256,
        "dependency_closure_sha256": MODULE._sha256(MODULE._canonical_json(list(records))),
        "records": list(records),
    }
    body["manifest_sha256"] = MODULE._sha256(MODULE._canonical_json(body))
    return MODULE._canonical_json(body)


def _row(stage_index: int, *, d_seg: float, d_pose: float, archive_bytes: int):
    return MODULE.ExactCompleteArchiveRowV1(
        stage_index=stage_index,
        archive_path=f"/Volumes/fixture/stage_{stage_index:02d}/archive.zip",
        archive_sha256=f"{stage_index + 1:x}" * 64,
        archive_bytes=archive_bytes,
        decoded_raw_sha256="a" * 64,
        d_seg=d_seg,
        d_pose=d_pose,
        score=score_geometry.contest_score(d_seg, d_pose, archive_bytes),
        sample_count=600,
        evaluator_batch_size=16,
        evaluator_source_sha256="b" * 64,
        report_sha256="c" * 64,
    )


def test_s00_checkpoints_storage_seed_geometry_and_refuses_missing_s01(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_load_g46_custody",
        lambda *_args: ({"audit_seal_sha256": "a" * 64}, ()),
    )
    config = _config()
    runner = MODULE.G102SemanticRootS00S01RunnerV1(
        repo_root=REPO,
        config=config,
    )
    run_root, receipt = runner.prepare_s00(tiers=_tier(tmp_path))

    assert receipt["seed"] == config.seed
    assert receipt["pair_count"] == 600
    assert receipt["stage_count"] == 5
    assert receipt["stage_pair_span"] == 120
    assert receipt["evaluator_batch_size"] == 16
    assert receipt["proxy_rows_allowed"] is False
    assert receipt["historical_payload_reused"] is False
    assert receipt["g46_custody"]["audit_seal_sha256"] == "a" * 64
    assert len(receipt["source_lineage_policy_sha256"]) == 64
    assert receipt["s01_ready"] is False
    assert set(receipt["s01_blockers"]) == {
        MODULE.COMPILER_BLOCKER,
        MODULE.PUBLIC_CODEC_BLOCKER,
    }
    assert not (run_root / MODULE.G102State.S01_ROOT_PROGRAM).exists()
    with pytest.raises(MODULE.G102RunnerError, match="S01 refused"):
        runner.run_s01(run_root)


def test_s00_resume_is_byte_identical_and_mutation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_load_g46_custody",
        lambda *_args: ({"audit_seal_sha256": "a" * 64}, ()),
    )
    runner = MODULE.G102SemanticRootS00S01RunnerV1(
        repo_root=REPO,
        config=_config(),
    )
    run_root, first = runner.prepare_s00(tiers=_tier(tmp_path))
    checkpoint = run_root / "S00_CUSTODY/checkpoint.json"
    first_bytes = checkpoint.read_bytes()
    first_stat = checkpoint.stat()

    resumed_root, second = runner.prepare_s00(tiers=_tier(tmp_path))
    assert resumed_root == run_root
    assert second == first
    assert checkpoint.read_bytes() == first_bytes
    assert checkpoint.stat().st_ino == first_stat.st_ino

    mutated = json.loads(first_bytes)
    mutated["seed"] += 1
    checkpoint.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(MODULE.G102RunnerError, match=r"canonical bytes|self seal"):
        runner.prepare_s00(tiers=_tier(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label_mask_palette_only", True),
        ("scorer_native_rgb_appearance", False),
        ("chroma_gauge", False),
        ("parallax_gauge", False),
        ("irreducible_rgb_quotient_seam", False),
        ("exact_post_r_seg_closure", False),
        ("exact_post_r_pose_closure", False),
    ],
)
def test_s01_refuses_label_palette_or_missing_rgb_both_scorer_closure(
    field: str,
    value: object,
) -> None:
    evidence = _capability()
    evidence[field] = value
    with pytest.raises(MODULE.G102RunnerError, match=MODULE.RGB_CLOSURE_BLOCKER):
        MODULE._validate_capability(evidence, expected_codec_sha="2" * 64)


def test_packet_bound_lineage_refuses_different_source_record(
    tmp_path: Path,
) -> None:
    packet = b"semantic-root-packet"
    config_sha256 = "a" * 64
    policy_sha256 = "b" * 64
    required = _lineage_records(tmp_path / "required")
    substitute_root = tmp_path / "substitute"
    substitute_root.mkdir()
    substitute = list(required)
    source_index = next(
        index for index, row in enumerate(substitute) if row["role"] == "FRESH_SOURCE_VIDEO_ENCODER_INPUT"
    )
    other = substitute_root / "fresh_source.bin"
    other.write_bytes(b"different-source")
    substitute[source_index] = {
        **substitute[source_index],
        "path": str(other),
        "bytes": other.stat().st_size,
        "sha256": MODULE._sha256_file(other),
    }
    substitute_rows = tuple(
        sorted(
            substitute,
            key=lambda row: (row["role"], row["path"], row["sha256"]),
        )
    )
    payload = _lineage_manifest(
        packet=packet,
        config_sha256=config_sha256,
        policy_sha256=policy_sha256,
        records=substitute_rows,
    )
    with pytest.raises(MODULE.G102RunnerError, match="omits or changes"):
        MODULE._validate_source_lineage_manifest(
            payload,
            packet=packet,
            config_sha256=config_sha256,
            lineage_policy_sha256=policy_sha256,
            required_records=required,
            repo_root=tmp_path,
        )


def test_lineage_manifest_rejects_forbidden_historical_candidate_dependency(
    tmp_path: Path,
) -> None:
    records = list(_lineage_records(tmp_path))
    forbidden = tmp_path / "g85_payload.bin"
    forbidden.write_bytes(b"historical")
    records.append(
        {
            "path": str(forbidden),
            "bytes": forbidden.stat().st_size,
            "sha256": MODULE._sha256_file(forbidden),
            "role": "OWN_LINEAGE_COMPILER_DEPENDENCY",
            "candidate_dependency": True,
            "packaged_in_archive": False,
            "video_derived": True,
        }
    )
    ordered = tuple(sorted(records, key=lambda row: (row["role"], row["path"], row["sha256"])))
    packet = b"packet"
    with pytest.raises(MODULE.G102RunnerError, match="forbidden historical"):
        MODULE._validate_source_lineage_manifest(
            _lineage_manifest(
                packet=packet,
                config_sha256="a" * 64,
                policy_sha256="b" * 64,
                records=ordered,
            ),
            packet=packet,
            config_sha256="a" * 64,
            lineage_policy_sha256="b" * 64,
            required_records=tuple(records[:-1]),
            repo_root=tmp_path,
        )


def test_complete_row_refuses_proxy_partial_batch_or_forged_score() -> None:
    valid = _row(0, d_seg=0.001, d_pose=0.002, archive_bytes=100_000)
    with pytest.raises(MODULE.G102RunnerError, match="partial, proxy"):
        replace(valid, proxy=True)
    with pytest.raises(MODULE.G102RunnerError, match="partial, proxy"):
        replace(valid, sample_count=120)
    with pytest.raises(MODULE.G102RunnerError, match="partial, proxy"):
        replace(valid, evaluator_batch_size=32)
    with pytest.raises(MODULE.G102RunnerError, match="recomposition"):
        replace(valid, score=valid.score + 0.01)


def test_dynamic_selection_uses_whole_coupled_score_without_component_gates() -> None:
    target = load_dynamic_frontier_target(repo_root=REPO)
    rows = (
        _row(0, d_seg=0.0010, d_pose=0.0001, archive_bytes=130_000),
        _row(1, d_seg=0.0009, d_pose=0.0002, archive_bytes=100_000),
    )
    selection = MODULE.select_coupled_complete_row(rows, target)
    expected = min(rows, key=lambda row: (row.score, row.archive_bytes, row.archive_sha256))
    assert selection["selected_stage_index"] == expected.stage_index
    assert selection["dynamic_target_score"] == target.target_score
    assert selection["independent_component_thresholds_used"] is False
    assert selection["candidate_claim"] is False
    assert selection["score_claim"] is False


def test_five_stage_partition_is_exact_ordered_n600_and_checkpoint_requires_complete_evidence(
    tmp_path: Path,
) -> None:
    stages = tuple(
        tuple(range(index * MODULE.STAGE_PAIR_SPAN, (index + 1) * MODULE.STAGE_PAIR_SPAN))
        for index in range(MODULE.STAGE_COUNT)
    )
    assert tuple(pair for stage in stages for pair in stage) == tuple(range(600))
    archive = b"complete-archive-fixture"
    stage_dir = tmp_path / "stage_00"
    stage_dir.mkdir()
    (stage_dir / "archive.zip").write_bytes(archive)
    row = _row(0, d_seg=0.001, d_pose=0.001, archive_bytes=len(archive))
    config = _config()
    checkpoint = {
        "config_sha256": config.sha256,
        "stage_index": 0,
        "pair_ids": list(stages[0]),
        "previous_checkpoint_sha256": None,
        "population_pair_count": 600,
        "evaluator_batch_size": 16,
        "parse_back_equal": True,
        "double_decode_equal": True,
        "candidate_claim": False,
        "score_claim": False,
        "archive_file": {
            "path": "archive.zip",
            "bytes": len(archive),
            "sha256": MODULE._sha256(archive),
        },
        "exact_complete_archive_row": asdict(row),
        "g17_selected_solution_authority": {
            "canonical_module": "tac.witness_dsl.taskspace_selected_solution_compiler",
            "receipt_path": "g17_whole_object_receipt.json",
            "receipt_sha256": "d" * 64,
        },
    }
    required_records = _lineage_records(tmp_path / "lineage")
    with pytest.raises(MODULE.G102RunnerError, match="resume custody"):
        MODULE._validate_resumed_stage(
            checkpoint,
            config=config,
            repo_root=tmp_path,
            packet=b"packet",
            required_lineage_records=required_records,
            lineage_policy_sha256="e" * 64,
            stage_index=0,
            pair_ids=stages[0],
            previous_checkpoint_sha=None,
            stage_dir=stage_dir,
        )
    checkpoint["double_decode_equal"] = False
    with pytest.raises(MODULE.G102RunnerError, match="resume custody"):
        MODULE._validate_resumed_stage(
            checkpoint,
            config=config,
            repo_root=tmp_path,
            packet=b"packet",
            required_lineage_records=required_records,
            lineage_policy_sha256="e" * 64,
            stage_index=0,
            pair_ids=stages[0],
            previous_checkpoint_sha=None,
            stage_dir=stage_dir,
        )


def test_g17_binding_refuses_noncanonical_state_type() -> None:
    class FakeModule:
        @staticmethod
        def semantic_root_y1_v1_g17_whole_object_state(*_args):
            return object()

    runner = MODULE.G102SemanticRootS00S01RunnerV1(repo_root=REPO, config=_config())
    row = _row(0, d_seg=0.001, d_pose=0.001, archive_bytes=3)
    with pytest.raises(MODULE.G102RunnerError, match="exact canonical G17"):
        runner._bind_g17_authority(
            module=FakeModule(),
            packet=b"packet",
            archive=b"zip",
            row=row,
            report=b"report",
            lineage_manifest=b"lineage",
        )


def test_lying_capability_without_packet_lineage_callable_cannot_enter_s01() -> None:
    methods = {
        name: (lambda *_args: None)
        for name in MODULE.REQUIRED_MODULE_CALLS
        if name != "semantic_root_y1_v1_source_lineage_manifest"
    }
    methods["inflate_semantic_root_y1_v1_archive"] = lambda *_args: None
    fake = SimpleNamespace(**methods)
    with pytest.raises(MODULE.G102RunnerError, match="source_lineage_manifest"):
        MODULE._require_module_interface(fake)


def test_private_only_inflater_is_not_public_entrypoint_authority(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.py").write_text("raise SystemExit('private only')\n")
    with pytest.raises(MODULE.G102RunnerError, match=r"actual inflate\.sh"):
        MODULE._public_runtime_records(tmp_path, "runtime")


def _public_runtime_fixture(tmp_path: Path) -> tuple[Path, str]:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cp "$1/0.bin" "$2/0.raw"\n'
        "echo public-stdout\n"
        "echo public-stderr >&2\n",
        encoding="utf-8",
    )
    records = MODULE._public_runtime_records(tmp_path, "runtime")
    return runtime, MODULE._public_runtime_sha256(records)


def test_actual_public_shell_runs_twice_from_clean_extracted_roots_with_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, runtime_sha256 = _public_runtime_fixture(tmp_path)
    (tmp_path / "upstream").mkdir()
    (tmp_path / MODULE.VIDEO_NAMES_ENTRYPOINT).write_text("0.mkv\n")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("0.bin", b"rgb!")
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(MODULE, "EXPECTED_RAW_BYTES", 4)
    runner = MODULE.G102SemanticRootS00S01RunnerV1(
        repo_root=tmp_path,
        config=_config(
            public_codec_section_path="runtime",
            public_codec_section_sha256=runtime_sha256,
        ),
    )
    raw_a, raw_b = runner._public_double_inflate(
        archive_path=archive,
        runtime_root=runtime,
        stage_dir=stage,
    )
    assert raw_a.read_bytes() == raw_b.read_bytes() == b"rgb!"
    receipt = MODULE._parse_sealed(
        stage / "public_inflate_authority.json",
        schema=MODULE.PUBLIC_INFLATE_RECEIPT_SCHEMA,
        seal_key="receipt_sha256",
    )
    assert receipt["authority_complete"] is True
    assert receipt["private_module_inflater_used_as_authority"] is False
    assert len(receipt["runs"]) == 2
    assert receipt["runs"][0]["cwd"] != receipt["runs"][1]["cwd"]
    for row in receipt["runs"]:
        assert row["argv"][0] == "bash"
        assert row["returncode"] == 0
        assert row["timed_out"] is False
        assert row["stdout"] == "public-stdout\n"
        assert row["stderr"] == "public-stderr\n"
        assert row["elapsed_seconds"] >= 0
        assert row["external_repo_imports_forbidden"] is True
        assert row["environment"]["PYTHONPATH"] == ""
        assert len(row["python_import_guard_sha256"]) == 64
    (stage / "checkpoint.json").write_bytes(b"durable-stage")
    runner._cleanup_public_inflate_scratch(
        stage_dir=stage,
        archive_path=archive,
        runtime_root=runtime,
        reason="test",
    )
    assert not (stage / ".public_inflate_work").exists()
    cleanup = MODULE._parse_sealed(
        stage / "public_inflate_cleanup_certificate.json",
        schema=MODULE.PUBLIC_SCRATCH_CLEANUP_SCHEMA,
        seal_key="certificate_sha256",
    )
    assert cleanup["total_bytes"] >= 8
    assert cleanup["delete_after_certificate"] is True
    assert cleanup["rebuild_inputs"]["archive_sha256"] == MODULE._sha256_file(archive)


def test_public_shell_timeout_preserves_stdout_stderr_argv_and_elapsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, runtime_sha256 = _public_runtime_fixture(tmp_path)
    (tmp_path / "upstream").mkdir()
    (tmp_path / MODULE.VIDEO_NAMES_ENTRYPOINT).write_text("0.mkv\n")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("0.bin", b"x")
    stage = tmp_path / "stage"
    stage.mkdir()

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == MODULE.PUBLIC_INFLATE_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(
            args[0],
            kwargs["timeout"],
            output="partial-out",
            stderr="partial-err",
        )

    monkeypatch.setattr(MODULE.subprocess, "run", timeout)
    runner = MODULE.G102SemanticRootS00S01RunnerV1(
        repo_root=tmp_path,
        config=_config(
            public_codec_section_path="runtime",
            public_codec_section_sha256=runtime_sha256,
        ),
    )
    with pytest.raises(MODULE.G102RunnerError, match=r"actual public inflate\.sh"):
        runner._public_double_inflate(
            archive_path=archive,
            runtime_root=runtime,
            stage_dir=stage,
        )
    receipt = MODULE._parse_sealed(
        stage / "public_inflate_authority.json",
        schema=MODULE.PUBLIC_INFLATE_RECEIPT_SCHEMA,
        seal_key="receipt_sha256",
    )
    run = receipt["runs"][0]
    assert run["timed_out"] is True
    assert run["returncode"] is None
    assert run["stdout"] == "partial-out"
    assert run["stderr"] == "partial-err"
    assert run["argv"][0] == "bash"
    assert run["elapsed_seconds"] >= 0


def test_public_shell_cannot_import_repo_source_outside_clean_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    hidden = tmp_path / "hidden_dep.py"
    hidden.write_text("VALUE = b'leak'\n")
    (runtime / "inflate.py").write_text(
        "import pathlib, sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import hidden_dep\n"
        "pathlib.Path(sys.argv[2]).write_bytes(hidden_dep.VALUE)\n"
    )
    (runtime / "inflate.sh").write_text(
        "#!/usr/bin/env bash\n"
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        '"$PYTHON" "$HERE/inflate.py" "$1/0.bin" "$2/0.raw"\n'
    )
    runtime_sha256 = MODULE._public_runtime_sha256(MODULE._public_runtime_records(tmp_path, "runtime"))
    (tmp_path / "upstream").mkdir()
    (tmp_path / MODULE.VIDEO_NAMES_ENTRYPOINT).write_text("0.mkv\n")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("0.bin", b"x")
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(MODULE, "EXPECTED_RAW_BYTES", 4)
    runner = MODULE.G102SemanticRootS00S01RunnerV1(
        repo_root=tmp_path,
        config=_config(
            public_codec_section_path="runtime",
            public_codec_section_sha256=runtime_sha256,
        ),
    )
    with pytest.raises(MODULE.G102RunnerError, match=r"actual public inflate\.sh"):
        runner._public_double_inflate(
            archive_path=archive,
            runtime_root=runtime,
            stage_dir=stage,
        )
    receipt = MODULE._parse_sealed(
        stage / "public_inflate_authority.json",
        schema=MODULE.PUBLIC_INFLATE_RECEIPT_SCHEMA,
        seal_key="receipt_sha256",
    )
    assert "attempted external repository import" in receipt["runs"][0]["stderr"]


def test_cleanup_resumes_after_interrupted_partial_delete_without_resealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / "stage"
    work = stage / ".public_inflate_work"
    work.mkdir(parents=True)
    (work / "a.raw").write_bytes(b"aaaa")
    (work / "b.raw").write_bytes(b"bbbb")
    (stage / "public_inflate_authority.json").write_bytes(b"public-evidence")
    (stage / "checkpoint.json").write_bytes(b"stage-evidence")
    archive = stage / "archive.zip"
    archive.write_bytes(b"archive")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runner = MODULE.G102SemanticRootS00S01RunnerV1(
        repo_root=tmp_path,
        config=_config(),
    )
    real_rmtree = MODULE.shutil.rmtree

    def partial_delete(path):
        (Path(path) / "a.raw").unlink()
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(MODULE.shutil, "rmtree", partial_delete)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner._cleanup_public_inflate_scratch(
            stage_dir=stage,
            archive_path=archive,
            runtime_root=runtime,
            reason="test",
        )
    certificate = stage / "public_inflate_cleanup_certificate.json"
    original_certificate = certificate.read_bytes()
    assert work.exists()
    assert not (work / "a.raw").exists()
    assert (work / "b.raw").exists()

    monkeypatch.setattr(MODULE.shutil, "rmtree", real_rmtree)
    runner._cleanup_public_inflate_scratch(
        stage_dir=stage,
        archive_path=archive,
        runtime_root=runtime,
        reason="test",
    )
    assert not work.exists()
    assert certificate.read_bytes() == original_certificate
    completion = MODULE._parse_sealed(
        stage / "public_inflate_cleanup_certificate_complete.json",
        schema=MODULE.PUBLIC_SCRATCH_CLEANUP_COMPLETE_SCHEMA,
        seal_key="completion_sha256",
    )
    assert completion["deletion_complete"] is True
    assert completion["certificate_file_sha256"] == MODULE._sha256(original_certificate)


def test_evaluator_failure_preserves_full_process_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "upstream").mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    archive = stage / "archive.zip"
    archive.write_bytes(b"zip")
    raw = stage / "0.raw"
    raw.write_bytes(b"raw")
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            7,
            stdout="evaluator-out-complete",
            stderr="evaluator-err-complete",
        ),
    )
    runner = MODULE.G102SemanticRootS00S01RunnerV1(
        repo_root=tmp_path,
        config=_config(),
    )
    with pytest.raises(MODULE.G102RunnerError, match="failed rc=7"):
        runner._evaluate_public_raw(
            archive_path=archive,
            raw_path=raw,
            stage_index=0,
            stage_dir=stage,
        )
    receipt = MODULE._parse_sealed(
        stage / "evaluator_process.json",
        schema=MODULE.EVALUATOR_PROCESS_SCHEMA,
        seal_key="receipt_sha256",
    )
    assert receipt["returncode"] == 7
    assert receipt["stdout"] == "evaluator-out-complete"
    assert receipt["stderr"] == "evaluator-err-complete"
    assert receipt["authority_complete"] is False


def test_config_is_closed_key_seeded_and_never_accepts_mps(tmp_path: Path) -> None:
    body = asdict(_config())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(body), encoding="utf-8")
    assert MODULE.load_config(config_path) == _config()
    body["invented_threshold"] = 0.5
    config_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(MODULE.G102RunnerError, match="key set"):
        MODULE.load_config(config_path)
    with pytest.raises(MODULE.G102RunnerError, match="MPS"):
        _config(eval_device="mps")


def test_s01_cli_requires_explicit_resume_from() -> None:
    parsed = MODULE._parser().parse_args(["fixture.json", "--run-s01"])
    assert parsed.resume_from is None
    assert "--resume-from" in MODULE._parser().format_help()
