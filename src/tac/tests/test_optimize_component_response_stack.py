# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "optimize_component_response_stack.py"
DENOM = 37_545_489


def _load_module():
    spec = importlib.util.spec_from_file_location("optimize_component_response_stack", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(label: str) -> str:
    return (label.encode("utf-8").hex() * 4)[:64].ljust(64, "0")


def _combined(pose: float, seg: float) -> float:
    return 100.0 * seg + math.sqrt(10.0 * pose)


def _point(*, epsilon: float, pose: float, seg: float, bytes_: int, label: str) -> dict[str, object]:
    values = {"combined": _combined(pose, seg), "posenet": pose, "segnet": seg}
    return {
        "archive": {
            "bytes": bytes_,
            "path": f"{label}.zip",
            "sha256": _sha(f"archive-{label}"),
        },
        "contest_auth_eval_json": {
            "bytes": 1234,
            "path": f"{label}/contest_auth_eval.json",
            "sha256": _sha(f"eval-{label}"),
        },
        "epsilon": epsilon,
        "point_metadata": {
            "archive_bytes": bytes_,
            "archive_sha256": _sha(f"archive-{label}"),
        },
        "values": values,
    }


def _write_curve_bundle(
    root: Path,
    *,
    name: str,
    baseline_label: str = "shared-baseline",
    baseline_pose: float = 0.04,
    baseline_seg: float = 0.01,
    baseline_bytes: int = 1000,
    actions: list[dict[str, float | int]],
    promotion: bool = True,
    device: str = "cuda",
    canonical: bool = True,
) -> Path:
    out = root / name
    out.mkdir()
    points = [
        _point(
            epsilon=0.0,
            pose=baseline_pose,
            seg=baseline_seg,
            bytes_=baseline_bytes,
            label=baseline_label,
        )
    ]
    for index, action in enumerate(actions, start=1):
        points.append(
            point := _point(
                epsilon=float(action["epsilon"]),
                pose=float(action["pose"]),
                seg=float(action["seg"]),
                bytes_=int(action["bytes"]),
                label=f"{name}-eps{index}",
            )
        )
        point_metadata = action.get("point_metadata")
        if isinstance(point_metadata, dict):
            point["point_metadata"] = point_metadata

    curve_paths: dict[str, str] = {}
    for component in ("posenet", "segnet", "combined"):
        curve_points = []
        baseline_value = points[0]["values"][component]  # type: ignore[index]
        for point in points:
            values = point["values"]  # type: ignore[assignment]
            curve_points.append(
                {
                    "all_components": dict(values),  # type: ignore[arg-type]
                    "archive": point["archive"],
                    "baseline": baseline_value,
                    "contest_auth_eval_json": point["contest_auth_eval_json"],
                    "delta": values[component] - baseline_value,  # type: ignore[index,operator]
                    "epsilon": point["epsilon"],
                    "point_metadata": point["point_metadata"],
                    "prediction": {"implemented": True, "relative_error": 0.0},
                    "value": values[component],  # type: ignore[index]
                }
            )
        gates = {
            "coverage_passed": True,
            "finite_values": True,
            "prediction_error_passed": promotion,
            "promotion_gate_passed": promotion,
            "signal_present": True,
            "zero_repro": True,
        }
        curve = {
            "baseline": {
                "archive": points[0]["archive"],
                "contest_auth_eval_json": points[0]["contest_auth_eval_json"],
                "epsilon": 0.0,
                "values": points[0]["values"],
            },
            "canonical_scorer_path": canonical,
            "component": component,
            "component_response_path": "archive_zip_inflate_sh_upstream_evaluate_py",
            "count": len(points),
            "device": device,
            "epsilon_ladder": [point["epsilon"] for point in points],
            "format": "official_component_response_curves_v1",
            "gate_results": gates,
            "official_component_response": True,
            "passed": promotion,
            "points": curve_points,
            "promotion_blockers": [] if promotion else [{"code": "missing_prediction_deltas"}],
            "promotion_eligible": promotion,
            "schema_version": 1,
            "sensitivity_source": "official_archive_finite_difference_component_response",
            "tool": "experiments/profile_component_sensitivity_official.py",
        }
        path = out / f"{component}_official_response_curve.json"
        path.write_text(json.dumps(curve, sort_keys=True) + "\n")
        curve_paths[component] = str(path)

    summary = {
        "baseline_archive": points[0]["archive"],
        "baseline_contest_auth_eval_json": points[0]["contest_auth_eval_json"],
        "device": device,
        "format": "official_component_response_summary_v1",
        "points": points,
        "promotion_eligible": promotion,
        "response_curve_paths": curve_paths,
        "schema_version": 1,
        "tool": "experiments/profile_component_sensitivity_official.py",
    }
    (out / "official_component_response_summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n"
    )
    return out


def test_rejects_invalid_and_prediction_only_evidence(tmp_path: Path) -> None:
    module = _load_module()
    cpu_bundle = _write_curve_bundle(
        tmp_path,
        name="cpu",
        actions=[{"epsilon": 1.0, "pose": 0.039, "seg": 0.009, "bytes": 990}],
        device="cpu",
    )
    with pytest.raises(module.ComponentResponseStackOptimizerError, match="non-CUDA"):
        module.optimize_component_response_stack([cpu_bundle])

    noncanonical_bundle = _write_curve_bundle(
        tmp_path,
        name="noncanonical",
        actions=[{"epsilon": 1.0, "pose": 0.039, "seg": 0.009, "bytes": 990}],
        canonical=False,
    )
    with pytest.raises(module.ComponentResponseStackOptimizerError, match="canonical_scorer_path"):
        module.optimize_component_response_stack([noncanonical_bundle])

    plan = tmp_path / "prediction_only_plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "perturbation": {
                    "auth_eval_required": "cuda",
                    "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
                },
                "points": [
                    {
                        "archive": "point.zip",
                        "epsilon": 1.0,
                        "predicted_delta": {"combined": -0.1, "posenet": -0.001, "segnet": -0.001},
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    with pytest.raises(module.ComponentResponseStackOptimizerError, match="prediction-only"):
        module.optimize_component_response_stack([plan], allow_calibration_inputs=True)


def test_deterministic_optimization_selects_best_feasible_stack(tmp_path: Path) -> None:
    module = _load_module()
    source_a = _write_curve_bundle(
        tmp_path,
        name="source_a",
        actions=[{"epsilon": 1.0, "pose": 0.035, "seg": 0.011, "bytes": 990}],
    )
    source_b = _write_curve_bundle(
        tmp_path,
        name="source_b",
        actions=[{"epsilon": 1.0, "pose": 0.041, "seg": 0.008, "bytes": 1005}],
    )

    first = module.optimize_component_response_stack(
        [source_a, source_b],
        archive_bytes_budget=1000,
        max_posenet_dist=0.05,
        max_segnet_dist=0.012,
        top_k=5,
    )
    second = module.optimize_component_response_stack(
        [source_a, source_b],
        archive_bytes_budget=1000,
        max_posenet_dist=0.05,
        max_segnet_dist=0.012,
        top_k=5,
    )

    assert module._json_bytes(first) == module._json_bytes(second)
    assert first["promotion_eligible"] is False
    assert first["recommendation_promotable"] is False
    assert first["optimization"]["candidate_stacks_considered"] == 3
    best = first["recommendations"][0]
    assert best["feasible_under_dykstra_constraints"] is True
    assert best["action_count"] == 2
    assert best["projected_components"]["archive_bytes"] == 995
    assert best["projected_components"]["posenet_dist"] == pytest.approx(0.036)
    assert best["projected_components"]["segnet_dist"] == pytest.approx(0.009)
    assert best["constraints"]["archive_bytes_budget"]["passed"] is True
    assert best["composability_claim"] is False
    assert best["requires_stacked_exact_eval"] is True


def test_score_formula_accounting_separates_pose_segnet_and_rate(tmp_path: Path) -> None:
    module = _load_module()
    source = _write_curve_bundle(
        tmp_path,
        name="score_terms",
        actions=[{"epsilon": 1.0, "pose": 0.09, "seg": 0.02, "bytes": 1200}],
    )

    result = module.optimize_component_response_stack(
        [source],
        archive_bytes_budget=1300,
        max_posenet_dist=0.10,
        max_segnet_dist=0.03,
        top_k=1,
    )

    candidate = result["recommendations"][0]
    projected = candidate["projected_components"]
    deltas = candidate["score_deltas"]
    baseline = result["baseline"]["score_terms"]
    expected_pose_term = math.sqrt(10.0 * 0.09)
    expected_seg_term = 100.0 * 0.02
    expected_rate_term = 25.0 * 1200 / DENOM

    assert projected["segnet_score_term"] == pytest.approx(expected_seg_term)
    assert projected["posenet_score_term"] == pytest.approx(expected_pose_term)
    assert projected["rate_score_term"] == pytest.approx(expected_rate_term)
    assert projected["score"] == pytest.approx(
        expected_seg_term + expected_pose_term + expected_rate_term
    )
    assert deltas["segnet_score_delta"] == pytest.approx(expected_seg_term - baseline["segnet_score_term"])
    assert deltas["posenet_score_delta"] == pytest.approx(expected_pose_term - baseline["posenet_score_term"])
    assert deltas["rate_score_delta"] == pytest.approx(
        expected_rate_term - baseline["rate_score_term"]
    )


def test_atom_allocation_table_records_risk_and_stack_gates(tmp_path: Path) -> None:
    module = _load_module()
    source = _write_curve_bundle(
        tmp_path,
        name="allocation",
        actions=[
            {
                "epsilon": 1.0,
                "pose": 0.035,
                "seg": 0.01,
                "bytes": 1008,
                "point_metadata": {
                    "atom_id": "pose_atom",
                    "atom_family": "pose",
                    "charged_bytes": 4,
                },
            },
            {
                "epsilon": 2.0,
                "pose": 0.041,
                "seg": 0.008,
                "bytes": 1020,
                "point_metadata": {
                    "atom_id": "mask_atom",
                    "atom_family": "mask",
                    "charged_bytes": 30,
                    "synergizes_with": ["mask_grammar_atoms"],
                },
            },
        ],
    )

    result = module.optimize_component_response_stack(
        [source],
        archive_bytes_budget=1100,
        max_posenet_dist=0.05,
        max_segnet_dist=0.02,
        top_k=2,
    )

    table = result["atom_allocation_table"]
    atoms = table["ranked_atoms"]
    assert atoms[0]["atom_id"] == "pose_atom"
    assert atoms[0]["family"] == "pose"
    assert atoms[0]["interaction_risk"] == "low"
    assert atoms[0]["waterfill_positive_ev"] is True
    mask = next(atom for atom in atoms if atom["atom_id"] == "mask_atom")
    assert mask["interaction_risk"] == "high"
    assert "posenet_regresses" in mask["interaction_risk_reasons"]
    assert mask["synergizes_with"] == ["mask_grammar_atoms"]
    gates = {gate["gate"] for gate in table["exact_eval_stack_gate_recommendations"]}
    assert {"canonical_cuda_auth_eval", "component_antagonism_review"} <= gates
    assert table["score_claim"] is False
