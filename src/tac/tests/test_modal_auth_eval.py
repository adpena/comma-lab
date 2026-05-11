"""Smoke tests for experiments/modal_auth_eval.py.

We do NOT invoke Modal in these tests — that requires a Modal account, billing
context, and a live T4. We verify:
  * The module imports cleanly (no syntax errors, no missing imports at parse).
  * The function signatures + Modal decorators are wired correctly.
  * The wrapper is hard-wired to the canonical CUDA contest-auth-eval path.
"""
from __future__ import annotations

import json
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

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


def test_source_uses_literal_cuda_canonical_contest_eval() -> None:
    text = TOOL_PATH.read_text()

    assert "experiments/contest_auth_eval.py" in text
    assert '"--device",' in text
    assert '"cuda",' in text
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
    eval_image = text[text.index("eval_image = ("):text.index("def _sha256_bytes")]
    assert eval_image.index(".env(") < eval_image.index(".add_local_")


def test_modal_auth_eval_images_include_hard_runtime_entropy_deps() -> None:
    cuda_text = TOOL_PATH.read_text()
    cpu_text = (REPO_ROOT / "experiments" / "modal_auth_eval_cpu.py").read_text()

    for text in (cuda_text, cpu_text):
        assert '"brotli>=1.0"' in text
        assert '"constriction>=0.4,<0.5"' in text
        assert '"pyppmd>=1.3,<2.0"' in text
        assert 'work_dir / "inflated_outputs_manifest.json"' in text
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
    (work_dir / "provenance.json").write_text("{}\n")

    artifacts = mod._collect_artifacts(out_dir, work_dir)

    assert "contest_auth_eval.json" in artifacts
    assert "inflated_outputs_manifest.json" in artifacts
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

    first = mod.submission_dir_zip_bytes(submission_dir)
    second = mod.submission_dir_zip_bytes(submission_dir)

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
        mod.submission_dir_zip_bytes(submission_dir)


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
        mod.submission_dir_zip_bytes(submission_dir)


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
        lane_id="lane_unit_modal_auth_eval",
        instance_job_id="job_unit_modal_auth_eval",
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
        lane_id="lane_unit_modal_auth_eval",
        instance_job_id="job_unit_modal_auth_eval_detached",
    )

    metadata = json.loads((out_dir / "modal_auth_eval_spawn.json").read_text())
    assert metadata["schema_version"] == "modal_auth_eval_spawn_v1"
    assert metadata["axis"] == "contest_cuda"
    assert metadata["call_id"] == "fc-test-modal-auth"
    assert metadata["result_json_name"] == "modal_cuda_auth_eval_result.json"
    assert metadata["lane_id"] == "lane_unit_modal_auth_eval"
    assert (out_dir / "modal_call_id.txt").read_text().strip() == "fc-test-modal-auth"
    assert [call["status"] for call in claim_calls] == [
        "active_modal_auth_eval_spawning",
        "active_modal_auth_eval_spawned",
    ]


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
            lane_id="lane_unit_modal_auth_eval",
            instance_job_id="job_unit_modal_auth_eval_detached",
        )

    assert claim_calls == []
