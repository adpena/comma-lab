from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/probe_costate_trust_region_economics.py"


def _load_tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("_costate_trust_region_probe", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_known_bound_canaries_cover_accept_and_strict_exit() -> None:
    controls = _load_tool()._canaries()
    assert controls["status"] == "PASS"
    assert controls["known_bound_positive"]["status"] == "CERTIFIED_REUSE"
    assert controls["known_bound_negative"]["status"] == "REFRESH"


def test_baseline_counts_rederive_402_and_48_from_primary_receipt() -> None:
    tool = _load_tool()
    receipt = tool._load_bound_receipt(tool.BASELINE_YOPO_RECEIPT, tool.BASELINE_YOPO_SHA256)
    counts = tool._baseline_counts(receipt)
    assert counts == {
        "step_rows": 28,
        "operational_validation_forwards": 402,
        "operational_teacher_forward_backward": 20,
        "measurement_teacher_forward_backward": 28,
        "total_teacher_forward_backward": 48,
    }
