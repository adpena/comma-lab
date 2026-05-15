# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "tools" / "xray_pair_component_errors.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("xray_pair_component_errors", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_report_is_false_authority_and_sorts_hard_pairs(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"zip")
    rows = [
        module.PairRow(
            pair_idx=0,
            pose_dist=0.0001,
            seg_dist=0.0002,
            pose_score_contribution=0.0316227766,
            seg_score_contribution=0.02,
            component_score_no_rate=0.0516227766,
            frame0_l1=1.0,
            frame1_l1=2.0,
            frame0_changed_fraction=0.1,
            frame1_changed_fraction=0.2,
        ),
        module.PairRow(
            pair_idx=1,
            pose_dist=0.0004,
            seg_dist=0.0001,
            pose_score_contribution=0.0632455532,
            seg_score_contribution=0.01,
            component_score_no_rate=0.0732455532,
            frame0_l1=4.0,
            frame1_l1=3.0,
            frame0_changed_fraction=0.4,
            frame1_changed_fraction=0.3,
        ),
    ]

    report = module.build_report(
        rows=rows,
        inflated_dir=tmp_path / "inflated",
        upstream_dir=tmp_path / "upstream",
        video_names_file=tmp_path / "names.txt",
        device="cpu",
        label="unit",
        top_k=1,
        archive=archive,
    )

    assert report["schema"] == "pair_component_error_xray_v1"
    assert report["score_claim"] is False
    assert report["dispatch_attempted"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["component_summary"]["avg_posenet_dist"] == 0.00025
    assert report["top_pairs"]["combined"][0]["pair_idx"] == 1
    assert report["top_pairs"]["frame1_l1"][0]["pair_idx"] == 1


def test_render_markdown_includes_component_and_pixel_sections(tmp_path: Path) -> None:
    module = _load_module()
    rows = [
        module.PairRow(
            pair_idx=7,
            pose_dist=0.0,
            seg_dist=0.0,
            pose_score_contribution=0.0,
            seg_score_contribution=0.0,
            component_score_no_rate=0.0,
            frame0_l1=0.0,
            frame1_l1=0.0,
            frame0_changed_fraction=0.0,
            frame1_changed_fraction=0.0,
        )
    ]
    report = module.build_report(
        rows=rows,
        inflated_dir=tmp_path / "inflated",
        upstream_dir=tmp_path / "upstream",
        video_names_file=tmp_path / "names.txt",
        device="cpu",
        label="unit",
        top_k=1,
        archive=None,
    )

    text = module.render_markdown(report)

    assert "Pair Component Error XRay" in text
    assert "Component Summary" in text
    assert "Pixel Summary" in text
    assert "score_claim: `false`" in text
