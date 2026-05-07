from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_sensitivity_map_pr106.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_sensitivity_map_pr106", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stub_design_mode_is_never_allowed_for_cuda() -> None:
    module = _load_module()

    assert module._allow_stub_design_mode("cuda", True) is False
    assert module._allow_stub_design_mode("cuda", False) is False
    assert module._allow_stub_design_mode("cpu", False) is False
    assert module._allow_stub_design_mode("cpu", True) is True
