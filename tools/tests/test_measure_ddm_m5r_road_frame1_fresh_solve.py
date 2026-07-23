# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / ".omx/research/ddm_m5r_road_frame1_fresh_solve_20260723T115443Z"
AXIS = "[macOS-CPU frozen-scorer advisory]"


def test_committed_receipt_has_receiver_closed_custody() -> None:
    receipt = json.loads((RUN / "receipt.json").read_bytes())
    selected = (RUN / "selected.not_a_candidate.zip.receipt-bytes").read_bytes()
    endpoint = receipt["full_n600_selected_endpoint"]

    assert receipt["score_claim"] is False
    assert receipt["evidence_axis"] == AXIS
    assert receipt["certification"]["pointer_moved"] is False
    assert receipt["certification"]["main_landing_review_required"] is True
    assert endpoint["all_batches_checkpointed"] is True
    assert endpoint["pair_count"] == 600
    assert endpoint["archive_sha256"] == hashlib.sha256(selected).hexdigest()
    assert endpoint["strict_joint_objective_admitted"] is False


def test_batch_reduction_exactly_rederives_endpoint() -> None:
    receipt = json.loads((RUN / "receipt.json").read_bytes())
    endpoint = receipt["full_n600_selected_endpoint"]
    batches = [
        json.loads(path.read_bytes())
        for path in sorted((RUN / "stage_checkpoints/04_selected_n600").glob("*.json"))
    ]
    assert len(batches) == 38
    assert {row["candidate_archive_sha256"] for row in batches} == {
        endpoint["archive_sha256"]
    }

    class_names = tuple(endpoint["per_stratum"])
    totals = {
        name: sum(int(row["per_stratum_candidate_errors"][name]) for row in batches)
        for name in class_names
    }
    assert totals == {
        name: int(endpoint["per_stratum"][name]["candidate_errors"])
        for name in class_names
    }
    sites = 600 * 384 * 512
    d_seg = sum(totals.values()) / sites
    pose_sse = sum(float(row["candidate_pose_squared_error_sum"]) for row in batches)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in batches)
    d_pose = pose_sse / pose_coordinates
    objective = (
        100.0 * d_seg
        + math.sqrt(10.0 * d_pose)
        + 25.0 * int(endpoint["archive_bytes"]) / 37_545_489
    )
    assert d_seg == endpoint["d_seg"]
    assert math.isclose(d_pose, endpoint["d_pose"], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        objective, endpoint["joint_objective"], rel_tol=0.0, abs_tol=1e-12
    )


def test_negative_stays_instance_scoped_and_does_not_narrow_366() -> None:
    receipt = json.loads((RUN / "receipt.json").read_bytes())
    parameterization = receipt["parameterization"]
    scope = receipt["catalog_366_true_scope_update"]

    assert parameterization["receiver_effective_count"] == 368
    assert parameterization["receiver_effective_groups"] == {
        "island_worldsheet": 326,
        "lane_program": 24,
        "shared_template_dof": 18,
    }
    assert parameterization["exact_singleton_compiler_refusals"] == 2
    assert receipt["certification"]["negative_family_claim"] is False
    assert receipt["certification"]["verdict_scope"].startswith("INSTANCE:")
    assert scope["updated_interval_errors"] == scope["prior_interval_errors"]
    assert scope["numeric_certified_residual"] is None
