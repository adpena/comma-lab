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
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "experiments" / "modal_auth_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "modal_auth_eval_mod", str(TOOL_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modal_auth_eval_mod"] = mod
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


def test_source_uses_literal_cuda_canonical_contest_eval() -> None:
    text = TOOL_PATH.read_text()

    assert "experiments/contest_auth_eval.py" in text
    assert '"--device",' in text
    assert '"cuda",' in text
    assert '"--device", "cpu"' not in text
    assert '"/root/submission/inflate_renderer.py"' not in text
    assert "promotion_eligible\": False" in text


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

    mod.main(str(archive), str(out_dir))

    request = json.loads((out_dir / "modal_cuda_auth_eval_local_request.json").read_text())
    result = json.loads((out_dir / "modal_cuda_auth_eval_result.json").read_text())
    assert request["score_claim"] is False
    assert request["promotion_eligible"] is False
    assert request["adjudication_required"] is True
    assert result["promotion_eligible"] is False
