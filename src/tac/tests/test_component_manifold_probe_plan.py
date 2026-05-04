from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "build_component_manifold_probe_plan.py"
SPEC = importlib.util.spec_from_file_location("build_component_manifold_probe_plan", MODULE_PATH)
assert SPEC is not None
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def _write_eval(
    path: Path,
    *,
    score: float,
    bytes_: int,
    seg: float,
    pose: float,
    device: str = "cuda",
) -> None:
    payload = {
        "score_recomputed_from_components": score,
        "archive_size_bytes": bytes_,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": ("a" if device == "cuda" else "b") * 64,
            "device": device,
        },
    }
    path.write_text(json.dumps(payload))


def _write_input_plan(path: Path, payload: dict) -> None:
    path.write_text(json.dumps({"schema": "component_manifold_probe_input_v1", **payload}))


def test_component_manifold_probe_computes_curvature_and_synergy(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    minus = tmp_path / "minus.json"
    plus = tmp_path / "plus.json"
    second = tmp_path / "second.json"
    combined = tmp_path / "combined.json"
    _write_eval(baseline, score=1.0, bytes_=600_000, seg=0.004, pose=0.004)
    _write_eval(minus, score=0.99, bytes_=590_000, seg=0.00395, pose=0.0039)
    _write_eval(plus, score=1.03, bytes_=610_000, seg=0.0041, pose=0.0042)
    _write_eval(second, score=0.98, bytes_=580_000, seg=0.0039, pose=0.0038)
    _write_eval(combined, score=0.965, bytes_=570_000, seg=0.0038, pose=0.0037)
    input_plan = tmp_path / "input.json"
    _write_input_plan(
        input_plan,
        {
            "baseline_contest_json": str(baseline),
            "points": [
                {
                    "point_id": "axis_m1",
                    "family": "renderer_fd",
                    "axis_id": "w0",
                    "epsilon": -1.0,
                    "contest_auth_eval_json": str(minus),
                },
                {
                    "point_id": "axis_p1",
                    "family": "renderer_fd",
                    "axis_id": "w0",
                    "epsilon": 1.0,
                    "contest_auth_eval_json": str(plus),
                },
                {
                    "point_id": "second",
                    "family": "pose_bytes",
                    "axis_id": "p0",
                    "epsilon": -1.0,
                    "contest_auth_eval_json": str(second),
                },
                {
                    "point_id": "combined",
                    "family": "stack",
                    "axis_id": "w0_p0",
                    "epsilon": -1.0,
                    "contest_auth_eval_json": str(combined),
                },
            ],
            "interactions": [
                {
                    "interaction_id": "axis_second_stack",
                    "point_ids": ["axis_m1", "second"],
                    "combined_point_id": "combined",
                }
            ],
        },
    )

    output = tmp_path / "probe.json"
    payload = probe.build_component_manifold_probe_plan(
        input_plan=input_plan,
        output_json=output,
    )

    assert output.exists()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["curvature_estimates"][0]["axis_id"] == "w0"
    assert payload["curvature_estimates"][0]["central_score_slope"] == pytest.approx(0.02)
    assert payload["curvature_estimates"][0]["score_curvature_per_epsilon2"] == pytest.approx(0.02)
    assert payload["interactions"][0]["classification"] == "synergy"
    assert {item["point_id"] for item in payload["continuation_candidates"]} == {
        "axis_m1",
        "second",
        "combined",
    }


def test_component_manifold_probe_blocks_collapsed_geometry(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    collapsed = tmp_path / "collapsed.json"
    _write_eval(baseline, score=1.0, bytes_=600_000, seg=0.004, pose=0.004)
    _write_eval(collapsed, score=4.0, bytes_=410_000, seg=0.009, pose=1.3)
    input_plan = tmp_path / "input.json"
    _write_input_plan(
        input_plan,
        {
            "baseline_contest_json": str(baseline),
            "points": [
                {
                    "point_id": "collapsed_alpha",
                    "family": "alpha_amr1",
                    "axis_id": "crf63_pairatom_top10",
                    "epsilon": -1.0,
                    "contest_auth_eval_json": str(collapsed),
                }
            ],
        },
    )

    payload = probe.build_component_manifold_probe_plan(
        input_plan=input_plan,
        output_json=tmp_path / "probe.json",
    )

    assert payload["continuation_candidates"] == []
    assert payload["blocked_points"][0]["point_id"] == "collapsed_alpha"
    assert {item["component"] for item in payload["blocked_points"][0]["violations"]} == {
        "posenet",
        "segnet",
    }
    assert payload["next_dispatch_policy"]["allowed"] is False


def test_component_manifold_probe_rejects_non_cuda_by_default(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    point = tmp_path / "point.json"
    _write_eval(baseline, score=1.0, bytes_=600_000, seg=0.004, pose=0.004)
    _write_eval(point, score=0.99, bytes_=590_000, seg=0.004, pose=0.004, device="cpu")
    input_plan = tmp_path / "input.json"
    _write_input_plan(
        input_plan,
        {
            "baseline_contest_json": str(baseline),
            "points": [
                {
                    "point_id": "cpu_point",
                    "family": "dev",
                    "axis_id": "bad",
                    "contest_auth_eval_json": str(point),
                }
            ],
        },
    )

    with pytest.raises(probe.ComponentManifoldProbeError, match="device must be cuda"):
        probe.build_component_manifold_probe_plan(
            input_plan=input_plan,
            output_json=tmp_path / "probe.json",
        )
