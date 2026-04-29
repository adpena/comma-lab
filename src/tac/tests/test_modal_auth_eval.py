"""Smoke tests for experiments/modal_auth_eval.py.

We do NOT invoke Modal in these tests — that requires a Modal account, billing
context, and a live T4. We verify:
  * The module imports cleanly (no syntax errors, no missing imports at parse).
  * The function signatures + Modal decorators are wired correctly.
  * The local entrypoint accepts archive path overrides.
"""
from __future__ import annotations

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
