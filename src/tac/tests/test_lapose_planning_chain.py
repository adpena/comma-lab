from __future__ import annotations

from tac.analysis.lapose_lite_inputs import inputs_from_pair_metric_payload
from tac.analysis.lapose_motion_atoms import build_motion_atom_manifest
from tac.analysis.lapose_motion_evidence import records_from_component_response


def test_pair_metrics_to_motion_atoms_to_lagrangian_chain_is_planning_only() -> None:
    pair_metrics = {
        "schema_version": 1,
        "device": "cuda",
        "lane": "generic_component_trace",
        "n_pairs": 4,
        "hardest_pair_indices": [2, 1],
        "per_pair_pose_dist": [0.1, 0.2, 0.8, 0.3],
        "per_pair_seg_dist": [0.01, 0.04, 0.08, 0.03],
        "per_pair_contrib": [1.0, 4.0, 9.0, 3.0],
    }
    lapose_inputs = inputs_from_pair_metric_payload(
        pair_metrics,
        source_path="generic_pair_metrics.json",
        source_sha256="1" * 64,
    )
    component_response = {
        "schema_version": 1,
        "score_claim": False,
        "device": "cuda",
        "promotion_eligible": False,
        "baseline_archive": {"bytes": 1000, "sha256": "2" * 64},
        "points": [
            {
                "epsilon": 0.0,
                "archive": {"bytes": 1000},
                "values": {"combined": 1.0, "segnet": 0.01, "posenet": 0.02},
            },
            {
                "epsilon": 0.5,
                "archive": {"bytes": 1012},
                "values": {"combined": 0.9, "segnet": 0.009, "posenet": 0.019},
            },
        ],
    }

    records = records_from_component_response(
        component_response,
        latent_actions=lapose_inputs["latent_actions"],
        pair_opportunities=lapose_inputs["pair_opportunities"],
        evidence_source_path="component_response.json",
        evidence_source_sha256="3" * 64,
    )
    manifest = build_motion_atom_manifest(
        records["records"],
        base_pose_dist=0.02,
        source="generic_chain_fixture",
    )

    assert lapose_inputs["score_claim"] is False
    assert records["allocation"]["allocation_inference"] is True
    assert manifest["paper_reference"]["arxiv"] == "2604.27448"
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["atom_ledger"]["ready_for_exact_eval_dispatch"] is False
    first_row = manifest["atom_ledger"]["rows"][0]
    assert first_row["pair_support"]
    assert first_row["hard_pair_support"]
    assert first_row["allocation_inference"] is True
    assert first_row["rankable"] is False
    assert "allocated_global_response_not_rankable" in first_row["dispatch_blockers"]
    assert first_row["evidence_source_sha256"] == "3" * 64
    assert first_row["source_archive_sha256"] == "2" * 64
