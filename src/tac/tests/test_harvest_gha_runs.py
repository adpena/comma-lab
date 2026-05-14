# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "harvest_gha_runs.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("_harvest_gha_runs_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_harvest_report_parser_recomputes_unrounded_cpu_score(tmp_path: Path) -> None:
    tool = _load_tool()
    report = tmp_path / "report.txt"
    report.write_text(
        """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00003460
  Average SegNet Distortion: 0.00057601
  Submission file size: 178,981 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00476704
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.20
""",
        encoding="utf-8",
    )

    parsed = tool.parse_report(report)

    expected = 100.0 * 0.00057601 + math.sqrt(10.0 * 0.00003460) + 25.0 * 0.00476704
    assert parsed["n_samples"] == 600
    assert parsed["reported_score_display"] == 0.20
    assert parsed["avg_posenet_dist"] == 0.00003460
    assert parsed["avg_segnet_dist"] == 0.00057601
    assert parsed["compression_rate"] == 0.00476704
    assert math.isclose(parsed["canonical_score"], expected, rel_tol=0.0, abs_tol=1e-12)
    assert parsed["score_recomputed_from_components"] == parsed["canonical_score"]
