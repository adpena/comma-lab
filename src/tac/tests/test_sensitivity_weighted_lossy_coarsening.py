# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

from tac.component_sensitivity_artifact import write_component_sensitivity_manifest
from tac.sensitivity_map import save_sensitivity_map

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "sensitivity_weighted_lossy_coarsening.py"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _load_tool():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "sensitivity_weighted_lossy_coarsening_under_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_inputs(tmp_path: Path) -> tuple[Path, Path]:
    state_dict = tmp_path / "state_dict.pt"
    sensitivity_map = tmp_path / "sensitivity.pt"
    torch.save(
        {
            "stem.weight": torch.linspace(-1.0, 1.0, steps=16).reshape(4, 4),
            "blocks.0.weight": torch.tensor(
                [[4.0, -4.0, 3.0, -3.0], [1.0, -1.0, 0.5, -0.5]],
                dtype=torch.float32,
            ),
        },
        state_dict,
    )
    save_sensitivity_map(
        sensitivity_map,
        {
            "stem.weight": torch.tensor([10.0, 10.0, 10.0, 10.0]),
            "blocks.0.weight": torch.tensor([1.0, 1.0]),
        },
        metadata={
            "device": "cuda",
            "source": "unit-certified-shape",
        },
    )
    return state_dict, sensitivity_map


def _pair_record(pair_index: int) -> dict[str, int]:
    return {
        "video": 0,
        "pair_index": pair_index,
        "t": 2 * pair_index,
        "t1": 2 * pair_index + 1,
    }


def _component_certification(component: str) -> dict[str, object]:
    return {
        "format": "component_sensitivity_map_certification_v1",
        "component": component,
        "device": "cuda",
        "official_component_response": True,
        "canonical_scorer_path": True,
        "promotion_eligible": True,
        "source_map_sha256": SHA_A,
        "official_response_curve_sha256": SHA_B,
        "stability_sha256": SHA_C,
        "sample_plan_sha256": SHA_D,
        "baseline_archive_sha256": SHA_A,
        "baseline_archive_bytes": 686635,
        "contest_auth_eval_json_sha256": SHA_B,
        "prediction_deltas_sha256": SHA_C,
        "perturbation_basis_sha256": SHA_D,
        "review_packet_sha256": SHA_A,
        "review_clean_passes": 3,
        "review_unresolved_blockers": [],
        "response_gate_results": {
            "finite_values": True,
            "coverage_passed": True,
            "zero_repro": True,
            "zero_repro_error": 0.0,
            "signal_present": True,
            "observed_delta_max": 0.01,
            "prediction_error_passed": True,
            "max_relative_prediction_error": 0.02,
            "promotion_gate_passed": True,
        },
        "stability_gate_results": {
            "passed": True,
            "cv_max": 0.04,
            "spearman_min": 0.96,
            "top_decile_overlap_min": 0.91,
        },
    }


def _component_map(component: str, sha256: str) -> dict[str, object]:
    return {
        "path": f"{component}_certified_sensitivity_map.pt",
        "bytes": 123,
        "sha256": sha256,
        "scorer_target": component,
        "map_format": "tac_score_sensitivity_map_v1",
        "certification": _component_certification(component),
        "tensor": {"dtype": "float32", "shape": [2], "numel": 2},
    }


def _response_curve(component: str, sha256: str) -> dict[str, object]:
    readouts = {
        "posenet": "official_pose_mse",
        "segnet": "official_argmax_disagreement",
        "combined": "official_component_formula",
    }
    return {
        "path": f"{component}_curve.json",
        "bytes": 456,
        "sha256": sha256,
        "count": 5,
        "holdout_error": 0.02,
        "official_component_response": True,
        "passed": True,
        "gate_results": {
            "finite_values": True,
            "coverage_passed": True,
            "zero_repro": True,
            "zero_repro_error": 0.0,
            "signal_present": True,
            "observed_delta_max": 0.01,
            "prediction_error_passed": True,
            "max_relative_prediction_error": 0.02,
            "promotion_gate_passed": True,
        },
        "gate_spec": {
            "zero_repro_tolerance": 1e-7,
            "holdout_error_max": 0.05,
            "spearman_min": 0.3,
        },
        "promotion_blockers": [],
        "component_readout": readouts[component],
        "response_kind": "symmetric",
        "epsilon_ladder": [-0.001, 0.0, 0.001],
    }


def _write_component_sensitivity_manifest(path: Path, *, combined_map_sha256: str) -> Path:
    manifest = {
        "schema_version": 1,
        "format": "component_sensitivity_v1",
        "device": "cuda",
        "promotion_eligible": True,
        "evidence_grade": "A",
        "inputs": {
            "checkpoint": {"path": "checkpoint.bin", "bytes": 1, "sha256": SHA_A},
            "video": {"path": "0.mkv", "bytes": 1, "sha256": SHA_B},
            "upstream": {"path": "upstream", "bytes": 1, "sha256": SHA_C},
        },
        "sample_plan": {
            "calibration_pairs": [_pair_record(idx) for idx in range(480)],
            "holdout_pairs": [_pair_record(idx) for idx in range(480, 600)],
            "split_seed": 123,
            "split_hash": SHA_D,
        },
        "component_maps": {
            "posenet": _component_map("posenet", SHA_A),
            "segnet": _component_map("segnet", SHA_B),
            "combined": _component_map("combined", combined_map_sha256),
        },
        "stability": {
            "cv": {"posenet": 0.04, "segnet": 0.05, "combined": 0.03},
            "rank": {"posenet": 0.98, "segnet": 0.97, "combined": 0.96},
            "top_k": {
                "posenet": {"k": 16, "overlap": 0.91},
                "segnet": {"k": 16, "overlap": 0.89},
                "combined": {"k": 16, "overlap": 0.93},
            },
            "thresholds": {
                "cv_max": 0.35,
                "spearman_min": 0.3,
                "top_decile_overlap_min": 0.5,
            },
            "passed": True,
        },
        "response_curves": {
            "posenet": _response_curve("posenet", SHA_A),
            "segnet": _response_curve("segnet", SHA_B),
            "combined": _response_curve("combined", SHA_C),
        },
        "contest_eval": {
            "archive_bytes": 37_000_000,
            "archive_sha256": SHA_A,
            "contest_auth_eval_json": {
                "path": "contest_auth_eval.json",
                "bytes": 1,
                "sha256": SHA_B,
            },
            "device": "cuda",
            "n_samples": 600,
        },
    }
    write_component_sensitivity_manifest(path, manifest)
    return path


def test_tool_builds_cpu_only_no_score_manifest(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    monkeypatch.setattr(
        tool,
        "FIXED_STATE_SCHEMA",
        (("stem.weight", (4, 4)), ("blocks.0.weight", (2, 4))),
    )
    state_dict, sensitivity_map = _write_tiny_inputs(tmp_path)
    output_json = tmp_path / "manifest.json"

    rc = tool.main(
        [
            "--state-dict",
            str(state_dict),
            "--sensitivity-map",
            str(sensitivity_map),
            "--rms-budget",
            "0.25",
            "--max-K",
            "4",
            "--output",
            str(output_json),
        ]
    )

    assert rc == 0
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["schema"] == tool.SCHEMA_VERSION
    assert manifest["cpu_only"] is True
    assert manifest["remote_dispatch_allowed"] is False
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["weighted_k_allocations"][0]["selected_Ks"]
    assert manifest["implementation"]["allocator_module"] == (
        "tac.optimization.lagrangian_per_tensor_allocation"
    )
    assert manifest["implementation"]["byte_closed_packet_ladder_builder"] == (
        "tools/build_a2_sensitivity_weighted_pr101_packet.py"
    )
    assert manifest["packet_ladder_builder"]["selected_k_schedule_field"] == (
        "weighted_k_allocations[].selected_Ks"
    )


def test_component_sensitivity_manifest_embeds_certified_a2_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool()
    monkeypatch.setattr(
        tool,
        "FIXED_STATE_SCHEMA",
        (("stem.weight", (4, 4)), ("blocks.0.weight", (2, 4))),
    )
    state_dict, sensitivity_map = _write_tiny_inputs(tmp_path)
    sensitivity_sha = hashlib.sha256(sensitivity_map.read_bytes()).hexdigest()
    component_manifest = _write_component_sensitivity_manifest(
        tmp_path / "component_sensitivity_v1.json",
        combined_map_sha256=sensitivity_sha,
    )
    output_json = tmp_path / "manifest.json"

    rc = tool.main(
        [
            "--state-dict",
            str(state_dict),
            "--sensitivity-map",
            str(sensitivity_map),
            "--component-sensitivity-manifest",
            str(component_manifest),
            "--rms-budget",
            "0.25",
            "--max-K",
            "4",
            "--output",
            str(output_json),
        ]
    )

    assert rc == 0
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    binding = manifest["sensitivity_artifact"]["certified_sensitivity_binding"]
    assert binding["status"] == "passed"
    assert binding["source"] == "component_sensitivity_v1.combined"
    assert binding["a2_sensitivity_map_sha256"] == sensitivity_sha
    assert (
        "score_sensitivity_artifact_must_be_certified_before_promotion"
        not in manifest["dispatch_blockers"]
    )


def test_tool_rejects_stub_sensitivity_unless_explicitly_allowed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool()
    monkeypatch.setattr(
        tool,
        "FIXED_STATE_SCHEMA",
        (("stem.weight", (4, 4)), ("blocks.0.weight", (2, 4))),
    )
    state_dict, sensitivity_map = _write_tiny_inputs(tmp_path)
    save_sensitivity_map(
        sensitivity_map,
        {
            "stem.weight": torch.ones(4),
            "blocks.0.weight": torch.ones(2),
        },
        metadata={"device": "cpu", "is_stub": True},
    )
    output_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--state-dict",
            str(state_dict),
            "--sensitivity-map",
            str(sensitivity_map),
            "--rms-budget",
            "0.25",
            "--max-K",
            "2",
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 2
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_fail_closed"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "diagnostic/stub sensitivity rejected" in manifest["reason"]


def test_tool_dry_run_writes_fail_closed_manifest(tmp_path: Path) -> None:
    tool = _load_tool()
    output_json = tmp_path / "dry.json"

    rc = tool.main(["--dry-run", "--output", str(output_json)])

    assert rc == 0
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run_fail_closed"
    assert manifest["dispatch_attempted"] is False
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
