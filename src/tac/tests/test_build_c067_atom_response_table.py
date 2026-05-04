from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_c067_atom_response_table.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_c067_atom_response_table_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _exact_payload(*, bytes_: int, pose: float, seg: float, sha: str, t4: bool = True) -> dict:
    module = _load_module()
    return {
        "archive_size_bytes": bytes_,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": sha,
            "archive_size_bytes": bytes_,
            "device": "cuda",
            "gpu_model": "Tesla T4" if t4 else "NVIDIA L40S",
            "gpu_t4_match": t4,
        },
        "score_recomputed_from_components": (
            100.0 * seg
            + math.sqrt(10.0 * pose)
            + module.RATE_SLOPE_SCORE_PER_BYTE * bytes_
        ),
    }


def test_component_positive_byte_regressive_sjkl_is_not_promoted(tmp_path: Path) -> None:
    module = _load_module()
    baseline = _write_json(
        tmp_path / "base" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=1000, pose=0.001, seg=0.001, sha="a" * 64),
    )
    sjkl = _write_json(
        tmp_path / "exact_eval_sjkl_c067_diag" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=1500, pose=0.000999, seg=0.000999, sha="b" * 64, t4=False),
    )

    table = module.build_c067_atom_response_table(
        root=tmp_path,
        baseline_json=baseline,
        exact_jsons=[sjkl],
        planner_jsons=[],
        include_default_exact_scan=False,
        include_default_planner_scan=False,
    )

    row = next(r for r in table["artifact_rows"] if r["label"] == "exact_eval_sjkl_c067_diag")
    assert row["family"] == "SJ-KL residual"
    assert row["component_status"] == "component_positive"
    assert row["byte_delta_class"] == "byte_regressive"
    assert row["score_delta_vs_baseline"] > 0
    assert row["evidence_grade"] == "B"
    assert row["promotion_eligible"] is False
    assert row["score_claim"] is False
    assert row["next_action"] == "shrink_payload_or_coefficient_only_then_exact_eval"

    family = next(r for r in table["family_rows"] if r["family"] == "SJ-KL residual")
    assert family["status"] == "component_positive_byte_regressive"
    assert family["classification"] == "component_positive_byte_regressive"
    assert family["exact_count"] == 1
    assert family["best_byte_delta"] == 500
    assert family["best_nonrate_delta"] < 0


def test_component_collapse_beats_byte_savings(tmp_path: Path) -> None:
    module = _load_module()
    baseline = _write_json(
        tmp_path / "base" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=1000, pose=0.001, seg=0.001, sha="a" * 64),
    )
    pmg = _write_json(
        tmp_path / "exact_eval_pmg_hotspot_c067" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=500, pose=0.1, seg=0.02, sha="c" * 64),
    )

    table = module.build_c067_atom_response_table(
        root=tmp_path,
        baseline_json=baseline,
        exact_jsons=[pmg],
        planner_jsons=[],
        include_default_exact_scan=False,
        include_default_planner_scan=False,
    )

    row = next(r for r in table["artifact_rows"] if r["label"] == "exact_eval_pmg_hotspot_c067")
    assert row["byte_delta_class"] == "byte_saving"
    assert row["component_status"] == "collapse"
    assert row["collapse_components"] == ["PoseNet", "SegNet"]
    assert row["evidence_grade"] == "A-negative"
    assert row["next_action"] == "do_not_dispatch_without_geometry_escape_and_component_guard"


def test_planner_only_cdo1_keeps_components_unknown(tmp_path: Path) -> None:
    module = _load_module()
    baseline = _write_json(
        tmp_path / "base" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=1000, pose=0.001, seg=0.001, sha="a" * 64),
    )
    planner = _write_json(
        tmp_path / "c067_reversed_base_cdo1_overlay_economics" / "c067_reversed_base_cdo1_overlay_economics.json",
        {
            "score_claim": False,
            "promotion_eligible": False,
            "decision": "byte_headroom_but_geometry_blocked",
            "best_candidates": [
                {
                    "estimated_archive": {
                        "estimated_archive_bytes": 800,
                        "estimated_delta_vs_c067": -200,
                    },
                    "gates": {"residual_geometry_gate": False},
                }
            ],
        },
    )

    table = module.build_c067_atom_response_table(
        root=tmp_path,
        baseline_json=baseline,
        exact_jsons=[],
        planner_jsons=[planner],
        include_default_exact_scan=False,
        include_default_planner_scan=False,
    )

    row = next(r for r in table["artifact_rows"] if r["artifact_kind"] == "planner")
    assert row["family"] == "CDO1 decoded-mask overlay"
    assert row["component_status"] == "unknown"
    assert row["byte_delta_vs_baseline"] == -200
    assert row["byte_delta_kind"] == "estimated_archive_delta"
    assert row["evidence_grade"] == "empirical"
    assert row["score_delta_vs_baseline"] is None
    assert row["next_action"] == "do_not_dispatch_until_residual_geometry_gate_and_runtime_closure"

    markdown = module.render_markdown(table)
    assert "No score claim" in markdown
    assert "CDO1 decoded-mask overlay" in markdown
