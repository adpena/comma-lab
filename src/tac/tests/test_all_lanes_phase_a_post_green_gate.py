# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "all_lanes_phase_a_post_green_test",
        ALL_LANES,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase_a_post_green_discoverability_gate_passes_current_sources() -> None:
    module = _load_all_lanes_module()

    passed, output = module._run_phase_a_post_green_discoverability_gate()

    assert passed is True
    assert "A5 readiness CLI" in output
    assert "Modal A1 plan/recover" in output
