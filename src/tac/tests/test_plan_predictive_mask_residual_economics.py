# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_predictive_mask_residual_economics.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("pmg_residual_economics_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(
    path: Path,
    *,
    candidate_id: str,
    archive_bytes: int,
    disagreement: float,
    payload_bytes: int,
    residual_pixels: int,
    protected_pairs: list[int] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pmg_hotspot_plan": {"candidate_id": candidate_id},
        "output_archive": {
            "bytes": archive_bytes,
            "sha256": f"{candidate_id:0<64}"[:64],
        },
        "pmg_hotspot_cmg3": {
            "payload_bytes": payload_bytes,
            "final_pixel_disagreement_vs_source_fraction": disagreement,
            "residual_pixels_touched": residual_pixels,
        },
    }
    if protected_pairs is not None:
        payload["pmg_hotspot_cmg3"]["pair_protection"] = {
            "protected_pair_indices": protected_pairs,
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_residual_economics_blocks_byte_only_pmg_when_geometry_fails(tmp_path: Path) -> None:
    planner = _load_planner()
    byte_only = _write_manifest(
        tmp_path / "byte_only.json",
        candidate_id="byte_only",
        archive_bytes=120_000,
        disagreement=0.02,
        payload_bytes=60_000,
        residual_pixels=30_000,
    )
    exact = _write_manifest(
        tmp_path / "exact.json",
        candidate_id="exact",
        archive_bytes=640_000,
        disagreement=0.0,
        payload_bytes=570_000,
        residual_pixels=1_800_000,
        protected_pairs=list(range(600)),
    )

    plan = planner.build_plan(
        manifest_paths=[byte_only, exact],
        output_json=tmp_path / "out" / planner.REPORT_NAME,
        repo_root=tmp_path,
        command=["unit-test"],
    )

    assert plan["schema"] == planner.SCHEMA
    assert plan["score_claim"] is False
    assert plan["remote_jobs_dispatched"] is False
    assert plan["decision"] == "residual_rowspan_not_sub024_viable"
    assert plan["pass_counts"]["byte_target_only"] == 1
    assert plan["pass_counts"]["geometry_target_only"] == 1
    assert plan["pass_counts"]["joint_byte_and_geometry"] == 0
    assert plan["best_byte_point"]["candidate_id"] == "byte_only"
    assert plan["best_geometry_point"]["candidate_id"] == "exact"
    assert plan["ranked_next_actions"][0]["action_id"] == "learned_geometry_preserving_mask_decoder"
    assert json.loads((tmp_path / "out" / planner.REPORT_NAME).read_text()) == plan


def test_residual_economics_surfaces_joint_candidate_if_one_exists(tmp_path: Path) -> None:
    planner = _load_planner()
    joint = _write_manifest(
        tmp_path / "joint.json",
        candidate_id="joint",
        archive_bytes=150_000,
        disagreement=0.0005,
        payload_bytes=90_000,
        residual_pixels=120_000,
    )

    plan = planner.build_plan(
        manifest_paths=[joint],
        output_json=tmp_path / "out.json",
        repo_root=tmp_path,
    )

    assert plan["decision"] == "residual_rowspan_has_local_candidate_requiring_exact_cuda"
    assert plan["pass_counts"]["joint_byte_and_geometry"] == 1
    assert plan["ranked_next_actions"][2]["status"] == "requires_claim_and_cuda"


def test_residual_economics_rejects_malformed_manifest(tmp_path: Path) -> None:
    planner = _load_planner()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"output_archive": {}}) + "\n")

    try:
        planner.build_plan(manifest_paths=[bad], output_json=tmp_path / "out.json")
    except planner.PlannerError as exc:
        assert "pmg_hotspot_cmg3" in str(exc)
    else:
        raise AssertionError("malformed PMG manifest was accepted")
