# SPDX-License-Identifier: MIT
"""Smoke tests for experiments/modal_auth_eval.py.

We do NOT invoke Modal in these tests — that requires a Modal account, billing
context, and a live T4. We verify:
  * The module imports cleanly (no syntax errors, no missing imports at parse).
  * The function signatures + Modal decorators are wired correctly.
  * The wrapper is hard-wired to the canonical CUDA contest-auth-eval path.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

import tac.deploy.modal.auth_eval as modal_auth_eval_helpers
from tac.deploy.modal.auth_eval import (
    modal_uploaded_submission_dir_runtime_manifest,
    prepare_modal_auth_eval_request,
    runtime_upload_skip_reason,
    submission_dir_zip_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "experiments" / "modal_auth_eval.py"
CPU_TOOL_PATH = REPO_ROOT / "experiments" / "modal_auth_eval_cpu.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "modal_auth_eval_mod", str(TOOL_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modal_auth_eval_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_cpu_module():
    spec = importlib.util.spec_from_file_location(
        "modal_auth_eval_cpu_mod", str(CPU_TOOL_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modal_auth_eval_cpu_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    pytest.importorskip("modal", reason="modal SDK not installed")
    return _load_module()


def test_module_imports_clean(mod):
    """Module loads without errors when modal is installed."""
    assert hasattr(mod, "run_auth_eval")
    assert hasattr(mod, "main")
    assert hasattr(mod, "app")


def test_modal_app_has_correct_name(mod):
    """The Modal app name is the canonical 'comma-auth-eval' so log scrapers
    and dashboards can find it."""
    assert mod.app.name == "comma-auth-eval"


def test_modal_source_mount_ignore_excludes_generated_bytecode(mod) -> None:
    assert mod.ignore_generated_mount_path(Path("src/tac/tests/__pycache__/x.pyc"))
    assert mod.ignore_generated_mount_path(Path("src/tac/tests/x.pyo"))
    assert mod.ignore_generated_mount_path(Path("src/tac/tests/._x.py"))
    assert not mod.ignore_generated_mount_path(Path("src/tac/tests/test_modal_auth_eval.py"))


def test_modal_auth_eval_requires_pair_group_or_single_axis_waiver(
    mod, tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"archive bytes")
    monkeypatch.setattr(
        mod,
        "claim_modal_auth_eval_dispatch",
        lambda **_kwargs: pytest.fail("claim should not be recorded before pair validation"),
    )

    with pytest.raises(SystemExit, match="paired-by-default"):
        mod.main(
            str(archive),
            str(tmp_path / "out"),
            lane_id="lane_unit_modal_auth_eval_pair_required",  # FAKE_LANE_OK:test-fixture lane_id
            instance_job_id="job_unit_modal_auth_eval_pair_required",
        )


def test_blocking_auth_eval_closes_claim_on_invalid_artifacts() -> None:
    cuda_text = TOOL_PATH.read_text(encoding="utf-8")
    cpu_text = CPU_TOOL_PATH.read_text(encoding="utf-8")

    assert "except ModalArtifactWriteError as exc" in cuda_text
    assert "failed_modal_auth_eval_invalid_artifacts" in cuda_text
    assert "terminal_modal_auth_eval_claim" in cuda_text
    assert "raise SystemExit(5) from exc" in cuda_text
    assert "except ModalArtifactWriteError as exc" in cpu_text
    assert "failed_modal_cpu_auth_eval_invalid_artifacts" in cpu_text
    assert "terminal_modal_auth_eval_claim" in cpu_text
    assert "raise SystemExit(5) from exc" in cpu_text


def test_run_auth_eval_function_signature(mod):
    """The function MUST accept archive_bytes (the upload format) and return
    a dict (so local entrypoint can index .get('score'), etc.)."""
    import inspect
    # Modal wraps the function but preserves signature on .get_raw_f().
    raw = mod.run_auth_eval.get_raw_f() if hasattr(mod.run_auth_eval, "get_raw_f") else mod.run_auth_eval
    sig = inspect.signature(raw)
    assert "archive_bytes" in sig.parameters
    assert "archive_sha256" in sig.parameters
    assert "archive_size_bytes" in sig.parameters
    assert "submission_dir_zip_bytes" in sig.parameters
    assert "submission_dir_zip_sha256" in sig.parameters
    assert "scorer_device" in sig.parameters
    assert "inflate_device_policy" in sig.parameters
    assert "inflate_env_overrides" in sig.parameters
    assert "expected_runtime_tree_sha256" in sig.parameters
    assert "scorer_input_cache_hashes" in sig.parameters
    assert "scorer_input_cache_hash_batch_pairs" in sig.parameters


def test_cuda_remote_unexpected_exception_returns_fail_closed_result(mod, tmp_path, monkeypatch):
    def raise_unexpected(**_kwargs):
        raise ValueError("synthetic runtime envelope failure")

    remote_out = tmp_path / "remote_out"
    remote_root = tmp_path / "remote_root"
    monkeypatch.setattr(mod, "REMOTE_OUT", remote_out)
    monkeypatch.setattr(mod, "REMOTE_WORK_ROOT", remote_root)
    monkeypatch.setattr(mod, "_run_auth_eval_inner", raise_unexpected)

    raw = mod.run_auth_eval.get_raw_f() if hasattr(mod.run_auth_eval, "get_raw_f") else mod.run_auth_eval
    result = raw(b"zip", "a" * 64, 3)

    assert result["passed"] is False
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["returncode"] == 98
    assert result["error_type"] == "ValueError"
    assert "synthetic runtime envelope failure" in result["error"]
    assert "ValueError" in result["traceback"]
    assert "modal_cuda_auth_eval_validation.json" in result["artifacts"]


def test_cpu_remote_unexpected_exception_returns_fail_closed_result(tmp_path, monkeypatch):
    pytest.importorskip("modal", reason="modal SDK not installed")
    cpu_mod = _load_cpu_module()

    def raise_unexpected(**_kwargs):
        raise RuntimeError("synthetic cpu runtime envelope failure")

    remote_out = tmp_path / "remote_cpu_out"
    monkeypatch.setattr(cpu_mod, "REMOTE_OUT", remote_out)
    monkeypatch.setattr(cpu_mod, "_run_auth_eval_inner", raise_unexpected)

    raw = (
        cpu_mod.run_auth_eval_cpu.get_raw_f()
        if hasattr(cpu_mod.run_auth_eval_cpu, "get_raw_f")
        else cpu_mod.run_auth_eval_cpu
    )
    result = raw(b"zip", "b" * 64, 3)

    assert result["passed"] is False
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["returncode"] == 98
    assert result["error_type"] == "RuntimeError"
    assert "synthetic cpu runtime envelope failure" in result["error"]
    assert "RuntimeError" in result["traceback"]
    assert "modal_cpu_auth_eval_validation.json" in result["artifacts"]


def test_cpu_run_auth_eval_signature_and_source_commit_flow(tmp_path, monkeypatch):
    pytest.importorskip("modal", reason="modal SDK not installed")
    cpu_mod = _load_cpu_module()

    import inspect

    raw = (
        cpu_mod.run_auth_eval_cpu.get_raw_f()
        if hasattr(cpu_mod.run_auth_eval_cpu, "get_raw_f")
        else cpu_mod.run_auth_eval_cpu
    )
    sig = inspect.signature(raw)
    assert "source_repo_commit" in sig.parameters
    assert "expected_runtime_tree_sha256" in sig.parameters

    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    expected_hash = "9" * 64
    captured_args = []

    class FakeRemote:
        @staticmethod
        def remote(*args):
            captured_args.append(args)
            return {
                "passed": True,
                "returncode": 0,
                "score_recomputed_from_components": 1.23,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.002,
                "archive_size_bytes": archive.stat().st_size,
                "score_axis": "contest_cpu",
                "promotion_eligible": False,
                "score_claim": True,
                "artifacts": {
                    "contest_auth_eval.json": b"{}\n",
                    "modal_cpu_auth_eval_validation.json": b"{}\n",
                },
            }

    monkeypatch.setattr(cpu_mod, "run_auth_eval_cpu", FakeRemote)
    monkeypatch.setattr(cpu_mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)
    monkeypatch.setattr(cpu_mod, "terminal_modal_auth_eval_claim", lambda **_kwargs: None)

    cpu_mod.main(
        str(archive),
        str(out_dir),
        expected_runtime_tree_sha256=expected_hash,
        lane_id="lane_unit_modal_cpu_auth_eval",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_cpu_auth_eval",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    request = json.loads((out_dir / "modal_cpu_auth_eval_local_request.json").read_text())
    result = json.loads((out_dir / "modal_cpu_auth_eval_result.json").read_text())
    assert request["source_repo_commit"]
    assert request["expected_runtime_tree_sha256"] == expected_hash
    assert result["source_repo_commit"] == request["source_repo_commit"]
    assert result["expected_runtime_tree_sha256"] == expected_hash
    assert captured_args[0][-6] == request["source_repo_commit"]
    assert captured_args[0][-3] == expected_hash
    assert captured_args[0][-2] is False
    assert captured_args[0][-1] == 8


def test_source_uses_literal_cuda_canonical_contest_eval() -> None:
    text = TOOL_PATH.read_text()

    assert "experiments/contest_auth_eval.py" in text
    assert '"--device",' in text
    assert '"cuda",' in text
    assert '"--inflate-device"' in text
    assert '"--inflate-env"' in text
    assert '"--expected-runtime-tree-sha256"' in text
    assert '"--scorer-input-cache-hashes-out"' in text
    assert '"--scorer-input-cache-hash-batch-pairs"' in text
    assert 'DALI_DISABLE_NVML_VALUE = "1"' in text
    assert "REMOTE_PYTHONPATH =" in text
    assert '"DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE' in text
    assert '"PYTHONPATH": REMOTE_PYTHONPATH' in text
    assert 'os.environ["DALI_DISABLE_NVML"] = DALI_DISABLE_NVML_VALUE' in text
    assert 'REMOTE_WORK_ROOT = Path("/root/modal_auth_eval_work")' in text
    assert 'work_dir = REMOTE_WORK_ROOT / "eval_work"' in text
    assert 'work_dir = out_dir / "eval_work"' not in text
    assert "safe_extract_zip(runtime_zip, runtime_root)" in text
    assert "submission_dir_zip_sha256" in text
    assert '"experiments/public_runtime_adapters"' in text
    assert (
        '"experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source"'
        in text
    )
    assert '"--device", "cpu"' not in text
    assert '"/root/submission/inflate_renderer.py"' not in text
    assert "promotion_eligible\": False" in text
    eval_image = text[text.index("eval_image = ("):text.index("def _probe_cuda_environment")]
    assert eval_image.index(".env(") < eval_image.index(".add_local_")


def test_modal_auth_eval_images_include_hard_runtime_entropy_deps() -> None:
    cuda_text = TOOL_PATH.read_text()
    cpu_text = (REPO_ROOT / "experiments" / "modal_auth_eval_cpu.py").read_text()

    for text in (cuda_text, cpu_text):
        assert '"brotli>=1.0"' in text
        assert '"constriction>=0.4,<0.5"' in text
        assert '"pyppmd>=1.3,<2.0"' in text
        assert 'work_dir / "inflated_outputs_manifest.json"' in text
        assert '"--expected-runtime-tree-sha256"' in text
        assert "expected_runtime_tree_sha256" in text
    assert '"PACT_SOURCE_COMMIT": source_repo_commit' in cuda_text
    assert '"PACT_SOURCE_COMMIT": source_repo_commit' in cpu_text
    assert 'REMOTE_WORK_ROOT = Path("/root/modal_auth_eval_work")' in cuda_text
    assert 'REMOTE_WORK_ROOT = Path("/root/modal_auth_eval_cpu_work")' in cpu_text
    assert 'work_dir = out_dir / "eval_work"' not in cuda_text
    assert 'work_dir = out_dir / "eval_work"' not in cpu_text


def test_cuda_artifact_harvest_includes_inflated_output_manifest(mod, tmp_path):
    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    out_dir.mkdir()
    work_dir.mkdir()
    (work_dir / "contest_auth_eval.json").write_text("{}\n")
    (work_dir / "inflated_outputs_manifest.json").write_text(
        '{"aggregate_sha256": "' + ("c" * 64) + '"}\n'
    )
    (work_dir / "scorer_input_cache_hashes.json").write_text(
        '{"hash_only": true, "array_sha256": {}}\n'
    )
    (work_dir / "provenance.json").write_text("{}\n")

    artifacts = mod._collect_artifacts(out_dir, work_dir)

    assert "contest_auth_eval.json" in artifacts
    assert "inflated_outputs_manifest.json" in artifacts
    assert "scorer_input_cache_hashes.json" in artifacts
    assert b'"aggregate_sha256"' in artifacts["inflated_outputs_manifest.json"]


def test_submission_dir_transport_zip_is_deterministic_and_filtered(mod, tmp_path):
    submission_dir = tmp_path / "submission_dir"
    submission_dir.mkdir()
    (submission_dir / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py\n")
    (submission_dir / "inflate.py").write_text("print('ok')\n")
    pycache = submission_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "inflate.cpython-311.pyc").write_bytes(b"pyc")
    (submission_dir / ".DS_Store").write_bytes(b"junk")

    first = submission_dir_zip_bytes(submission_dir)
    second = submission_dir_zip_bytes(submission_dir)

    assert first == second
    zip_path = tmp_path / "transport.zip"
    zip_path.write_bytes(first)
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["inflate.py", "inflate.sh"]
        assert zf.getinfo("inflate.sh").date_time == (1980, 1, 1, 0, 0, 0)


@pytest.mark.parametrize(
    "relative_path",
    [
        ".env",
        "secrets/token.txt",
        "runtime/.hidden",
        "id_rsa",
    ],
)
def test_submission_dir_transport_zip_rejects_hidden_and_secrets(mod, tmp_path, relative_path):
    submission_dir = tmp_path / "submission_dir"
    path = submission_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("do-not-upload")

    with pytest.raises(ValueError, match="refusing"):
        submission_dir_zip_bytes(submission_dir)


def test_submission_dir_transport_zip_rejects_symlinks(mod, tmp_path):
    submission_dir = tmp_path / "submission_dir"
    submission_dir.mkdir()
    secret = tmp_path / "outside_secret.txt"
    secret.write_text("do-not-upload")
    link = submission_dir / "inflate.py"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(ValueError, match="symlink"):
        submission_dir_zip_bytes(submission_dir)


def test_prepare_modal_auth_eval_request_centralizes_upload_shape(tmp_path):
    archive = tmp_path / "candidate archive.zip"
    archive.write_bytes(b"archive bytes")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py\n")
    (runtime / "inflate.py").write_text("print('ok')\n")

    prepared = prepare_modal_auth_eval_request(
        archive=archive,
        output_dir="",
        inflate_sh=runtime / "inflate.sh",
        submission_dir=runtime,
        default_output_root=Path("modal_results"),
        cwd=tmp_path,
    )

    assert prepared.archive_path == archive.resolve()
    assert prepared.archive_bytes == b"archive bytes"
    assert len(prepared.archive_sha256) == 64
    assert prepared.archive_size_bytes == len(b"archive bytes")
    assert prepared.inflate_sh_rel == "inflate.sh"
    assert prepared.submission_dir_path == runtime.resolve()
    assert prepared.submission_dir_zip is not None
    assert len(prepared.submission_dir_zip_sha256 or "") == 64
    assert prepared.output_dir.parent == (tmp_path / "modal_results").resolve()
    assert "candidate_archive" in prepared.output_dir.name


def test_modal_auth_eval_public_api_exports_reusable_custody_helpers() -> None:
    expected = {
        "ModalArtifactWriteError",
        "PreparedModalAuthEvalRequest",
        "UnsafeModalArtifactPath",
        "materialize_modal_artifacts",
        "prepare_modal_auth_eval_request",
        "safe_modal_artifact_path",
    }

    assert expected <= set(modal_auth_eval_helpers.__all__)
    for name in expected:
        assert hasattr(modal_auth_eval_helpers, name)


def test_modal_uploaded_submission_dir_runtime_manifest_uses_remote_shape() -> None:
    local = {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": "/local/static_release_surface",
        "runtime_file_count": 2,
        "files": [
            {
                "relative_path": "inflate.sh",
                "repo_relative_path": "experiments/local/inflate.sh",
                "bytes": 17,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "src/codec.py",
                "repo_relative_path": "experiments/local/src/codec.py",
                "bytes": 31,
                "sha256": "b" * 64,
            },
        ],
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": {
            "schema": "contest_auth_eval_repo_local_tac_import_manifest_v1",
            "runtime_root_name": "static_release_surface",
            "files": [],
        },
        "upstream_evaluate_py": {
            "relative_path": "evaluate.py",
            "bytes": 11,
            "sha256": "c" * 64,
        },
    }

    projected = modal_uploaded_submission_dir_runtime_manifest(local)
    projected_cpu = modal_uploaded_submission_dir_runtime_manifest(
        local,
        remote_submission_dir="/tmp/modal_auth_eval_cpu/submission_dir",
    )

    assert projected["runtime_root"] == "/tmp/modal_auth_eval/submission_dir"
    assert projected["repo_local_tac_import_manifest"]["runtime_root_name"] == "submission_dir"
    assert [
        row["repo_relative_path"] for row in projected["files"]
    ] == [
        "/tmp/modal_auth_eval/submission_dir/inflate.sh",
        "/tmp/modal_auth_eval/submission_dir/src/codec.py",
    ]
    assert len(projected["runtime_tree_sha256"]) == 64
    assert len(projected["runtime_content_tree_sha256"]) == 64
    assert projected["runtime_tree_sha256"] != projected["runtime_content_tree_sha256"]
    assert projected_cpu["runtime_root"] == "/tmp/modal_auth_eval_cpu/submission_dir"
    assert projected_cpu["runtime_tree_sha256"] != projected["runtime_tree_sha256"]
    assert (
        projected_cpu["runtime_content_tree_sha256"]
        == projected["runtime_content_tree_sha256"]
    )


def test_modal_auth_eval_rejects_local_root_runtime_hash_for_uploaded_runtime(
    mod,
    tmp_path,
    monkeypatch,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"archive bytes")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py\n")
    (runtime / "inflate.py").write_text("print('ok')\n")

    remote_hash = "b" * 64
    content_hash = "c" * 64
    monkeypatch.setattr(
        mod,
        "_expected_uploaded_runtime_tree_sha256",
        lambda **_kwargs: (remote_hash, content_hash),
    )
    monkeypatch.setattr(
        mod,
        "claim_modal_auth_eval_dispatch",
        lambda **_kwargs: pytest.fail("claim should not be recorded before hash validation"),
    )

    with pytest.raises(SystemExit) as exc:
        mod.main(
            str(archive),
            str(tmp_path / "out"),
            inflate_sh="inflate.sh",
            submission_dir=str(runtime),
            expected_runtime_tree_sha256="a" * 64,
            lane_id="lane_unit_modal_auth_eval_runtime_hash_guard",  # FAKE_LANE_OK:test-fixture lane_id
            instance_job_id="job_unit_modal_auth_eval_runtime_hash_guard",
        pair_group_id="pair_unit_modal_auth_eval",
        )

    message = str(exc.value)
    assert "uploaded --submission-dir runtime tree" in message
    assert remote_hash in message
    assert content_hash in message


def test_modal_cpu_auth_eval_rejects_local_root_runtime_hash_for_uploaded_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    pytest.importorskip("modal", reason="modal SDK not installed")
    cpu_mod = _load_cpu_module()
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"archive bytes")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py\n")
    (runtime / "inflate.py").write_text("print('ok')\n")

    remote_hash = "d" * 64
    content_hash = "e" * 64
    monkeypatch.setattr(
        cpu_mod,
        "_expected_uploaded_runtime_tree_sha256",
        lambda **_kwargs: (remote_hash, content_hash),
    )
    monkeypatch.setattr(
        cpu_mod,
        "claim_modal_auth_eval_dispatch",
        lambda **_kwargs: pytest.fail("claim should not be recorded before hash validation"),
    )

    with pytest.raises(SystemExit) as exc:
        cpu_mod.main(
            str(archive),
            str(tmp_path / "out"),
            inflate_sh="inflate.sh",
            submission_dir=str(runtime),
            expected_runtime_tree_sha256="f" * 64,
            lane_id="lane_unit_modal_cpu_auth_eval_runtime_hash_guard",  # FAKE_LANE_OK:test-fixture lane_id
            instance_job_id="job_unit_modal_cpu_auth_eval_runtime_hash_guard",
        pair_group_id="pair_unit_modal_auth_eval",
        )

    message = str(exc.value)
    assert "uploaded --submission-dir runtime tree" in message
    assert remote_hash in message
    assert content_hash in message


def test_modal_runtime_upload_skips_host_metadata_files(tmp_path):
    """Runtime zips skip host metadata files before hidden-path validation."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py\n")
    (runtime / "inflate.py").write_text("print('ok')\n")
    (runtime / ".gitignore").write_text("*.pyc\n")
    (runtime / ".gitattributes").write_text("* text=auto\n")

    assert runtime_upload_skip_reason(".gitignore") == "ignored host metadata"
    assert runtime_upload_skip_reason(".gitattributes") == "ignored host metadata"
    blob = submission_dir_zip_bytes(runtime)
    with zipfile.ZipFile(io.BytesIO(blob), mode="r") as zf:
        names = zf.namelist()
    assert ".gitignore" not in names
    assert ".gitattributes" not in names


def test_validate_contest_result_rejects_non_cuda(mod):
    archive_sha = "a" * 64
    payload = {
        "archive_size_bytes": 123,
        "n_samples": 600,
        "score_recomputed_from_components": 1.0,
        "provenance": {
            "device": "cpu",
            "cuda_available": False,
            "archive_sha256": archive_sha,
        },
    }

    errors = mod._validate_contest_result(
        payload,
        expected_archive_sha256=archive_sha,
        expected_archive_size_bytes=123,
    )

    assert any("expected 'cuda'" in error for error in errors)
    assert any("cuda_available" in error for error in errors)


def test_validate_contest_result_accepts_cuda_custody(mod):
    archive_sha = "b" * 64
    payload = {
        "archive_size_bytes": 456,
        "n_samples": 600,
        "score_recomputed_from_components": 0.99,
        "evidence_grade": "contest-CUDA",
        "score_axis": "contest_cuda",
        "exact_cuda_eval_complete": True,
        "provenance": {
            "device": "cuda",
            "cuda_available": True,
            "archive_sha256": archive_sha,
        },
    }

    assert mod._validate_contest_result(
        payload,
        expected_archive_sha256=archive_sha,
        expected_archive_size_bytes=456,
    ) == []


def test_validate_contest_result_accepts_cpu_scorer_on_cuda_host_diagnostic(mod):
    archive_sha = "c" * 64
    payload = {
        "archive_size_bytes": 789,
        "n_samples": 600,
        "score_recomputed_from_components": 0.88,
        "provenance": {
            "device": "cpu",
            "cuda_available": True,
            "archive_sha256": archive_sha,
        },
    }

    assert mod._validate_contest_result(
        payload,
        expected_archive_sha256=archive_sha,
        expected_archive_size_bytes=789,
        expected_device="cpu",
    ) == []


def test_remote_inner_rejects_cpu_scorer_auto_inflate_on_gpu_host(mod):
    result = mod._run_auth_eval_inner(
        archive_bytes=b"zip",
        archive_sha256="a" * 64,
        archive_size_bytes=3,
        inflate_sh_rel="inflate.sh",
        submission_dir_zip_bytes=None,
        submission_dir_zip_sha256=None,
        source_repo_commit="unit",
        inflate_timeout=1,
        evaluate_timeout=1,
        scorer_device="cpu",
        inflate_device_policy="auto",
    )

    assert result["passed"] is False
    assert result["returncode"] == 13
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False


def test_local_request_metadata_is_non_promotable_shape(mod, tmp_path, monkeypatch):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    claim_calls = []
    terminal_calls = []

    class FakeRemote:
        @staticmethod
        def remote(*_args):
            return {
                "passed": True,
                "returncode": 0,
                "score_recomputed_from_components": 1.23,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.002,
                "archive_size_bytes": archive.stat().st_size,
                "promotion_eligible": False,
                "artifacts": {
                    "contest_auth_eval.json": b"{}\n",
                    "modal_cuda_auth_eval_validation.json": b"{}\n",
                },
            }

    monkeypatch.setattr(mod, "run_auth_eval", FakeRemote)
    monkeypatch.setattr(
        mod,
        "claim_modal_auth_eval_dispatch",
        lambda **kwargs: claim_calls.append(kwargs),
    )
    monkeypatch.setattr(
        mod,
        "terminal_modal_auth_eval_claim",
        lambda **kwargs: terminal_calls.append(kwargs),
    )

    mod.main(
        str(archive),
        str(out_dir),
        lane_id="lane_unit_modal_auth_eval",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    request = json.loads((out_dir / "modal_cuda_auth_eval_local_request.json").read_text())
    result = json.loads((out_dir / "modal_cuda_auth_eval_result.json").read_text())
    assert request["score_claim"] is False
    assert request["promotion_eligible"] is False
    assert request["adjudication_required"] is True
    assert request["submission_dir"] is None
    assert request["modal_dispatch_mode"] == "blocking_remote"
    assert result["promotion_eligible"] is False
    assert claim_calls[0]["status"] == "active_modal_auth_eval_running"
    assert terminal_calls[0]["status"] == "completed_modal_auth_eval_recovered"


def test_modal_cuda_inflate_env_request_is_diagnostic_only(mod, tmp_path, monkeypatch):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    captured_args = []

    class FakeRemote:
        @staticmethod
        def remote(*args):
            captured_args.append(args)
            return {
                "passed": True,
                "returncode": 0,
                "score_recomputed_from_components": 1.23,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.002,
                "archive_size_bytes": archive.stat().st_size,
                "promotion_eligible": False,
                "score_claim": False,
                "artifacts": {
                    "contest_auth_eval.json": b"{}\n",
                    "modal_cuda_auth_eval_validation.json": b"{}\n",
                },
            }

    monkeypatch.setattr(mod, "run_auth_eval", FakeRemote)
    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "terminal_modal_auth_eval_claim", lambda **_kwargs: None)

    mod.main(
        str(archive),
        str(out_dir),
        inflate_env="CUDA_VISIBLE_DEVICES=",
        lane_id="lane_unit_modal_auth_eval_diag",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_diag",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    request = json.loads((out_dir / "modal_cuda_auth_eval_local_request.json").read_text())
    result = json.loads((out_dir / "modal_cuda_auth_eval_result.json").read_text())
    assert request["diagnostic_only"] is True
    assert request["inflate_env_overrides"] == ["CUDA_VISIBLE_DEVICES="]
    assert result["inflate_env_overrides"] == ["CUDA_VISIBLE_DEVICES="]
    assert captured_args[0][-6] == "cuda"
    assert captured_args[0][-5] == "auto"
    assert captured_args[0][-4] == ("CUDA_VISIBLE_DEVICES=",)
    assert captured_args[0][-3] == ""
    assert captured_args[0][-2] is False
    assert captured_args[0][-1] == 8


def test_modal_cuda_inflate_device_request_is_diagnostic_only(mod, tmp_path, monkeypatch):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    captured_args = []

    class FakeRemote:
        @staticmethod
        def remote(*args):
            captured_args.append(args)
            return {
                "passed": True,
                "returncode": 0,
                "score_recomputed_from_components": 1.23,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.002,
                "archive_size_bytes": archive.stat().st_size,
                "promotion_eligible": False,
                "score_claim": False,
                "artifacts": {
                    "contest_auth_eval.json": b"{}\n",
                    "modal_cuda_auth_eval_validation.json": b"{}\n",
                },
            }

    monkeypatch.setattr(mod, "run_auth_eval", FakeRemote)
    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "terminal_modal_auth_eval_claim", lambda **_kwargs: None)

    mod.main(
        str(archive),
        str(out_dir),
        inflate_device="cpu",
        lane_id="lane_unit_modal_auth_eval_diag_device",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_diag_device",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    request = json.loads((out_dir / "modal_cuda_auth_eval_local_request.json").read_text())
    result = json.loads((out_dir / "modal_cuda_auth_eval_result.json").read_text())
    assert request["diagnostic_only"] is True
    assert request["inflate_device_policy"] == "cpu"
    assert result["inflate_device_policy"] == "cpu"
    assert captured_args[0][-6] == "cuda"
    assert captured_args[0][-5] == "cpu"
    assert captured_args[0][-4] == ()
    assert captured_args[0][-3] == ""
    assert captured_args[0][-2] is False
    assert captured_args[0][-1] == 8


def test_modal_gpu_host_cpu_scorer_requires_explicit_inflate_device(
    mod, tmp_path, monkeypatch
):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)

    with pytest.raises(SystemExit, match="explicit --inflate-device"):
        mod.main(
            str(archive),
            str(out_dir),
            scorer_device="cpu",
            lane_id="lane_unit_modal_auth_eval_diag_cpu_scorer",  # FAKE_LANE_OK:test-fixture lane_id
            instance_job_id="job_unit_modal_auth_eval_diag_cpu_scorer",
        pair_group_id="pair_unit_modal_auth_eval",
        )


def test_modal_gpu_host_cpu_scorer_cuda_inflate_is_diagnostic_only(
    mod, tmp_path, monkeypatch
):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    captured_args = []

    class FakeRemote:
        @staticmethod
        def remote(*args):
            captured_args.append(args)
            return {
                "passed": True,
                "returncode": 0,
                "score_recomputed_from_components": 1.23,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.002,
                "archive_size_bytes": archive.stat().st_size,
                "promotion_eligible": True,
                "score_claim": True,
                "artifacts": {
                    "contest_auth_eval.json": b"{}\n",
                    "modal_cuda_auth_eval_validation.json": b"{}\n",
                },
            }

    monkeypatch.setattr(mod, "run_auth_eval", FakeRemote)
    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "terminal_modal_auth_eval_claim", lambda **_kwargs: None)

    mod.main(
        str(archive),
        str(out_dir),
        scorer_device="cpu",
        inflate_device="cuda",
        lane_id="lane_unit_modal_auth_eval_cpu_scorer_cuda_inflate",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_cpu_scorer_cuda_inflate",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    request = json.loads((out_dir / "modal_cuda_auth_eval_local_request.json").read_text())
    result = json.loads((out_dir / "modal_cuda_auth_eval_result.json").read_text())
    assert request["diagnostic_only"] is True
    assert request["canonical_path"].endswith("--device cpu")
    assert request["scorer_device"] == "cpu"
    assert request["inflate_device_policy"] == "cuda"
    assert result["scorer_device"] == "cpu"
    assert result["inflate_device_policy"] == "cuda"
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert captured_args[0][-6] == "cpu"
    assert captured_args[0][-5] == "cuda"
    assert captured_args[0][-4] == ()
    assert captured_args[0][-3] == ""
    assert captured_args[0][-2] is False
    assert captured_args[0][-1] == 8


def test_modal_cuda_expected_runtime_hash_flows_to_remote_call(mod, tmp_path, monkeypatch):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    expected_hash = "e" * 64
    captured_args = []

    class FakeRemote:
        @staticmethod
        def remote(*args):
            captured_args.append(args)
            return {
                "passed": True,
                "returncode": 0,
                "score_recomputed_from_components": 1.23,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.002,
                "archive_size_bytes": archive.stat().st_size,
                "promotion_eligible": False,
                "score_claim": False,
                "artifacts": {
                    "contest_auth_eval.json": b"{}\n",
                    "modal_cuda_auth_eval_validation.json": b"{}\n",
                },
            }

    monkeypatch.setattr(mod, "run_auth_eval", FakeRemote)
    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "terminal_modal_auth_eval_claim", lambda **_kwargs: None)

    mod.main(
        str(archive),
        str(out_dir),
        expected_runtime_tree_sha256=expected_hash,
        lane_id="lane_unit_modal_auth_eval_expected_runtime",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_expected_runtime",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    request = json.loads((out_dir / "modal_cuda_auth_eval_local_request.json").read_text())
    result = json.loads((out_dir / "modal_cuda_auth_eval_result.json").read_text())
    assert request["expected_runtime_tree_sha256"] == expected_hash
    assert result["expected_runtime_tree_sha256"] == expected_hash
    assert captured_args[0][-3] == expected_hash
    assert captured_args[0][-2] is False
    assert captured_args[0][-1] == 8


def test_modal_cuda_scorer_input_hash_bridge_flows_to_remote_call(
    mod, tmp_path, monkeypatch
):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    captured_args = []

    class FakeRemote:
        @staticmethod
        def remote(*args):
            captured_args.append(args)
            return {
                "passed": True,
                "returncode": 0,
                "score_recomputed_from_components": 1.23,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.002,
                "archive_size_bytes": archive.stat().st_size,
                "promotion_eligible": False,
                "score_claim": False,
                "artifacts": {
                    "contest_auth_eval.json": b"{}\n",
                    "modal_cuda_auth_eval_validation.json": b"{}\n",
                    "scorer_input_cache_hashes.json": b"{\"hash_only\": true}\n",
                },
            }

    monkeypatch.setattr(mod, "run_auth_eval", FakeRemote)
    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "terminal_modal_auth_eval_claim", lambda **_kwargs: None)

    mod.main(
        str(archive),
        str(out_dir),
        scorer_input_cache_hashes=True,
        scorer_input_cache_hash_batch_pairs=3,
        lane_id="lane_unit_modal_auth_eval_hash_bridge",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_hash_bridge",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    request = json.loads((out_dir / "modal_cuda_auth_eval_local_request.json").read_text())
    result = json.loads((out_dir / "modal_cuda_auth_eval_result.json").read_text())
    assert request["scorer_input_cache_hashes_requested"] is True
    assert request["scorer_input_cache_hash_batch_pairs"] == 3
    assert result["scorer_input_cache_hashes_requested"] is True
    assert result["scorer_input_cache_hash_batch_pairs"] == 3
    assert (out_dir / "scorer_input_cache_hashes.json").is_file()
    assert captured_args[0][-2] is True
    assert captured_args[0][-1] == 3


def test_detached_modal_auth_eval_writes_canonical_spawn_metadata(mod, tmp_path, monkeypatch):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    claim_calls = []

    class FakeSpawn:
        @staticmethod
        def spawn(*_args):
            return type("Call", (), {"object_id": "fc-test-modal-auth"})()

    monkeypatch.setattr(mod, "run_auth_eval", FakeSpawn)
    monkeypatch.setattr(
        mod,
        "claim_modal_auth_eval_dispatch",
        lambda **kwargs: claim_calls.append(kwargs),
    )

    mod.main(
        str(archive),
        str(out_dir),
        detach=True,
        provider_detach_ack=True,
        lane_id="lane_unit_modal_auth_eval",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_detached",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    metadata = json.loads((out_dir / "modal_auth_eval_spawn.json").read_text())
    assert metadata["schema_version"] == "modal_auth_eval_spawn_v1"
    assert metadata["axis"] == "contest_cuda"
    assert metadata["call_id"] == "fc-test-modal-auth"
    assert metadata["result_json_name"] == "modal_cuda_auth_eval_result.json"
    assert metadata["lane_id"] == "lane_unit_modal_auth_eval"  # FAKE_LANE_OK:test-fixture lane_id
    assert (out_dir / "modal_call_id.txt").read_text().strip() == "fc-test-modal-auth"
    assert [call["status"] for call in claim_calls] == [
        "active_modal_auth_eval_spawning",
        "active_modal_auth_eval_spawned",
    ]


def test_detached_modal_auth_eval_marks_inflate_env_as_diagnostic_axis(mod, tmp_path, monkeypatch):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"

    class FakeSpawn:
        @staticmethod
        def spawn(*_args):
            return type("Call", (), {"object_id": "fc-test-modal-auth-diag"})()

    monkeypatch.setattr(mod, "run_auth_eval", FakeSpawn)
    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)

    mod.main(
        str(archive),
        str(out_dir),
        inflate_env="CUDA_VISIBLE_DEVICES=",
        inflate_device="cpu",
        detach=True,
        provider_detach_ack=True,
        lane_id="lane_unit_modal_auth_eval",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_diag_detached",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    metadata = json.loads((out_dir / "modal_auth_eval_spawn.json").read_text())
    assert metadata["axis"] == "diagnostic_cuda"
    assert metadata["diagnostic_only"] is True
    assert metadata["inflate_device_policy"] == "cpu"
    assert metadata["inflate_env_overrides"] == ["CUDA_VISIBLE_DEVICES="]


def test_detached_modal_auth_eval_marks_non_t4_gpu_as_diagnostic_axis(mod, tmp_path, monkeypatch):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"

    class FakeSpawn:
        @staticmethod
        def spawn(*_args):
            return type("Call", (), {"object_id": "fc-test-modal-auth-a100"})()

    monkeypatch.setattr(mod, "run_auth_eval_a100", FakeSpawn)
    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)

    mod.main(
        str(archive),
        str(out_dir),
        gpu="A100",
        detach=True,
        provider_detach_ack=True,
        lane_id="lane_unit_modal_auth_eval",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_a100_detached",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    metadata = json.loads((out_dir / "modal_auth_eval_spawn.json").read_text())
    assert metadata["axis"] == "diagnostic_cuda"
    assert metadata["diagnostic_only"] is True
    assert metadata["non_t4_gpu_diagnostic"] is True


def test_detached_modal_auth_eval_marks_cpu_scorer_as_diagnostic_axis(
    mod, tmp_path, monkeypatch
):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"

    class FakeSpawn:
        @staticmethod
        def spawn(*_args):
            return type("Call", (), {"object_id": "fc-test-modal-auth-cpu-scorer"})()

    monkeypatch.setattr(mod, "run_auth_eval", FakeSpawn)
    monkeypatch.setattr(mod, "claim_modal_auth_eval_dispatch", lambda **_kwargs: None)

    mod.main(
        str(archive),
        str(out_dir),
        scorer_device="cpu",
        inflate_device="cuda",
        detach=True,
        provider_detach_ack=True,
        lane_id="lane_unit_modal_auth_eval",  # FAKE_LANE_OK:test-fixture lane_id
        instance_job_id="job_unit_modal_auth_eval_cpu_scorer_detached",
        pair_group_id="pair_unit_modal_auth_eval",
    )

    metadata = json.loads((out_dir / "modal_auth_eval_spawn.json").read_text())
    assert metadata["axis"] == "diagnostic_cpu"
    assert metadata["diagnostic_only"] is True
    assert metadata["scorer_device"] == "cpu"
    assert metadata["inflate_device_policy"] == "cuda"


def test_detached_modal_auth_eval_requires_provider_detach_ack(mod, tmp_path, monkeypatch):
    archive = tmp_path / "point_004_eps_p2.zip"
    archive.write_bytes(b"archive bytes")
    out_dir = tmp_path / "out"
    claim_calls = []

    monkeypatch.setattr(
        mod,
        "claim_modal_auth_eval_dispatch",
        lambda **kwargs: claim_calls.append(kwargs),
    )

    with pytest.raises(SystemExit, match="provider-level Modal CLI detach"):
        mod.main(
            str(archive),
            str(out_dir),
            detach=True,
            lane_id="lane_unit_modal_auth_eval",  # FAKE_LANE_OK:test-fixture lane_id
            instance_job_id="job_unit_modal_auth_eval_detached",
        pair_group_id="pair_unit_modal_auth_eval",
        )

    assert claim_calls == []
