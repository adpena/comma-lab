from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.analysis.lapose_lite_inputs import inputs_from_pair_metric_payload
from tac.analysis.lapose_motion_atoms import LaposeMotionAtomError

REPO = Path(__file__).resolve().parents[3]


def test_inputs_from_pair_metric_payload_builds_deterministic_records() -> None:
    manifest = inputs_from_pair_metric_payload(
        _payload(),
        source_path="pair_metrics.json",
        source_sha256="d" * 64,
        max_pairs=2,
    )

    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["evidence_grade"] == "empirical_cuda_pair_metric_telemetry"
    assert (
        manifest["paper_reference"]["implementation_alignment"]
        == "inspired_planning_only_not_paper_faithful_model"
    )
    assert "pair_metrics_are_not_score_authority" in manifest["dispatch_blockers"]
    assert "lapose_lite_is_not_paper_faithful_lapose_model" in manifest["dispatch_blockers"]
    assert manifest["selected_pair_count"] == 2
    assert manifest["latent_actions"][0]["pair_index"] == 3
    assert manifest["latent_actions"][0]["hard_pair_rank"] == 0
    assert len(manifest["latent_actions"][0]["latent_action"]) == 10
    assert manifest["pair_opportunities"][0]["hard_pair_rank"] == 0
    assert manifest["pair_opportunities"][0]["hard_pair_support"] == [3]
    assert manifest["pair_opportunities"][0]["geometry_priors"] == [
        "scorer_pair_metric",
        "pair_metric_hardness",
    ]
    assert manifest["pair_opportunities"][0]["openpilot_priors"] == []
    assert manifest["pair_opportunities"][0]["source_sha256"] == "d" * 64


def test_inputs_from_pair_metric_payload_fails_closed_on_cpu_or_shape_mismatch() -> None:
    bad = dict(_payload())
    bad["device"] = "mps"
    with pytest.raises(LaposeMotionAtomError, match="CUDA"):
        inputs_from_pair_metric_payload(bad, source_path="pair_metrics.json")

    bad = dict(_payload())
    bad["per_pair_seg_dist"] = [0.1]
    with pytest.raises(LaposeMotionAtomError, match="match n_pairs"):
        inputs_from_pair_metric_payload(bad, source_path="pair_metrics.json")


def test_build_lapose_lite_inputs_from_pair_metrics_cli(tmp_path: Path) -> None:
    metrics = tmp_path / "pair_metrics.json"
    out = tmp_path / "lapose_lite.json"
    metrics.write_text(json.dumps(_payload()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_lapose_lite_inputs_from_pair_metrics.py"),
            "--pair-metrics-json",
            str(metrics),
            "--max-pairs",
            "2",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["selected_pair_count"] == 2
    assert payload["feature_contract"]["name"] == "lapose_lite_pair_metric_v1"


def test_inputs_from_pair_metric_payload_only_emits_sourced_openpilot_priors() -> None:
    payload = _payload()
    payload["openpilot_priors"] = ["ego_motion", "vanishing_point"]

    manifest = inputs_from_pair_metric_payload(
        payload,
        source_path="non_lane_w_pair_metrics.json",
        source_sha256="e" * 64,
        max_pairs=1,
    )

    assert manifest["source_lane"] == "lane_w_hard_pair"
    assert manifest["pair_opportunities"][0]["openpilot_priors"] == [
        "ego_motion",
        "vanishing_point",
    ]


def test_inputs_from_pair_metric_payload_preserves_class_support() -> None:
    payload = _payload()
    payload["per_pair_class_support"] = [
        [0],
        [1, 3],
        [],
        [2, 3, 3],
        [4],
    ]

    manifest = inputs_from_pair_metric_payload(
        payload,
        source_path="pair_metrics_with_classes.json",
        source_sha256="f" * 64,
        max_pairs=2,
    )

    assert manifest["class_support_contract"] == {
        "field": "per_pair_class_support",
        "num_classes": 5,
        "source": "payload.per_pair_class_support",
    }
    assert manifest["pair_opportunities"][0]["pair_index"] == 3
    assert manifest["pair_opportunities"][0]["class_support"] == [2, 3]
    assert manifest["pair_opportunities"][1]["pair_index"] == 1
    assert manifest["pair_opportunities"][1]["class_support"] == [1, 3]


def test_inputs_from_pair_metric_payload_rejects_bad_class_support() -> None:
    payload = _payload()
    payload["per_pair_class_support"] = [[0], [1], [2], [3], [5]]

    with pytest.raises(LaposeMotionAtomError, match=r"outside 0\.\.4"):
        inputs_from_pair_metric_payload(payload, source_path="pair_metrics.json")


def test_canonical_and_compatibility_imports_match() -> None:
    from tac import lapose_foveation_atoms as compat_foveation_atoms
    from tac import lapose_lite_inputs as compat_lapose_lite
    from tac import lapose_motion_atoms as compat_motion_atoms
    from tac import lapose_motion_evidence as compat_motion_evidence
    from tac import meta_lagrangian_allocator as compat_allocator
    from tac.analysis import lapose_foveation_atoms as canonical_foveation_atoms
    from tac.analysis import lapose_lite_inputs as canonical_lapose_lite
    from tac.analysis import lapose_motion_atoms as canonical_motion_atoms
    from tac.analysis import lapose_motion_evidence as canonical_motion_evidence
    from tac.optimization import meta_lagrangian_allocator as canonical_allocator

    assert (
        compat_lapose_lite.inputs_from_pair_metric_payload
        is canonical_lapose_lite.inputs_from_pair_metric_payload
    )
    assert (
        compat_foveation_atoms.build_foveation_transport_atom_manifest
        is canonical_foveation_atoms.build_foveation_transport_atom_manifest
    )
    assert compat_motion_atoms.build_motion_atom_manifest is canonical_motion_atoms.build_motion_atom_manifest
    assert (
        compat_motion_evidence.records_from_component_response
        is canonical_motion_evidence.records_from_component_response
    )
    assert compat_allocator.build_atom_ledger is canonical_allocator.build_atom_ledger


def _payload() -> dict:
    return {
        "schema_version": 1,
        "device": "cuda",
        "lane": "lane_w_hard_pair",
        "n_pairs": 5,
        "hardest_pair_indices": [3, 1, 4],
        "per_pair_pose_dist": [0.1, 0.2, 0.15, 0.5, 0.3],
        "per_pair_seg_dist": [0.01, 0.02, 0.015, 0.04, 0.025],
        "per_pair_contrib": [1.0, 2.0, 1.5, 5.0, 3.0],
    }
