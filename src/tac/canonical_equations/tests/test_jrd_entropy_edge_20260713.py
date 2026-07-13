from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.jrd_entropy_edge_20260713 import (
    EQUATION_ID,
    build_jrd_component_safe_entropy_edge_stop_v1,
    component_safe_posthoc_gain_bytes,
    kkt_mass_initialization,
    populate_jrd_component_safe_entropy_edge_stop_v1,
)
from tac.witness_dsl.jrd_priors import (
    JrdReusablePriorPolicy,
    N600PriorConfirmation,
    PriorState,
)

_REPO = Path(__file__).resolve().parents[4]
_CURVES = (
    _REPO / "experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/section_precision_response_curves.json"
)
_HARVEST_RECEIPT = _REPO / ".omx/research/jrd_reusable_priors_harvest_20260713.json"


def test_component_safe_gain_and_mass_prior_are_exact() -> None:
    assert component_safe_posthoc_gain_bytes(177_169, []) == 0
    assert component_safe_posthoc_gain_bytes(177_169, [177_169]) == 0
    assert component_safe_posthoc_gain_bytes(177_169, [177_165]) == 4
    mass = kkt_mass_initialization(2_446, 226_512)
    assert mass["bias"] == pytest.approx(0.010683182068326942)
    assert mass["weight"] == pytest.approx(0.989316817931673)
    assert sum(mass.values()) == pytest.approx(1.0)


def test_source_json_rederives_screen_counts_without_memo_trust() -> None:
    payload = json.loads(_CURVES.read_text())
    baseline = payload["baseline"]
    zero_dseg = [row for row in payload["rows"] if row["d_seg"] == baseline["d_seg"]]
    pose_only = [row for row in zero_dseg if row["d_pose"] > baseline["d_pose"]]
    shrinking = [row for row in zero_dseg if row["archive_zip_bytes"] < baseline["archive_zip_bytes"]]
    safe_shrinking = [row for row in shrinking if row["d_pose"] <= baseline["d_pose"]]

    assert len(payload["rows"]) == 448
    assert len(zero_dseg) == 33
    assert len(pose_only) == 29
    assert len(shrinking) == 22
    assert len(safe_shrinking) == 0
    assert sorted({row["section"] for row in zero_dseg}) == [
        "refine.1.bias",
        "rgb_0.bias",
        "rgb_0.weight",
    ]
    receipt = json.loads(_HARVEST_RECEIPT.read_text())
    assert receipt["n1_screen"]["response_rows"] == len(payload["rows"])
    assert receipt["n1_screen"]["zero_delta_dseg_rows"] == len(zero_dseg)
    assert receipt["n1_screen"]["zero_delta_dseg_positive_dpose_rows"] == len(pose_only)
    assert receipt["n1_screen"]["zero_delta_dseg_byte_shrinking_rows"] == len(shrinking)
    assert receipt["n1_screen"]["component_safe_and_byte_shrinking_rows"] == 0


def test_n1_policy_refuses_actuation_until_full_n600_authority() -> None:
    policy = JrdReusablePriorPolicy()
    dormant = policy.compile_warm_start()
    assert dormant["state"] == PriorState.DORMANT_N1_SCREEN.value
    assert dormant["precision_actuation"] == "REFUSED_PENDING_N600_CONFIRMATION"
    assert dormant["live_trainer_argv"] == []

    incomplete = N600PriorConfirmation(
        eval_pairs=599,
        real_gt=True,
        numpy_fp32_bit_identical=True,
        exact_r=True,
        separate_dseg_dpose=True,
        exact_archive_bytes=True,
        positive_repeat_noise_floor_zero=True,
        archive_sha256="a" * 64,
        receipt_path="experiments/results/jrd_n599/receipt.json",
        receipt_sha256="c" * 64,
    )
    assert policy.state(incomplete) is PriorState.DORMANT_N1_SCREEN

    complete = N600PriorConfirmation(
        eval_pairs=600,
        real_gt=True,
        numpy_fp32_bit_identical=True,
        exact_r=True,
        separate_dseg_dpose=True,
        exact_archive_bytes=True,
        positive_repeat_noise_floor_zero=True,
        archive_sha256="b" * 64,
        receipt_path="experiments/results/jrd_n600/receipt.json",
        receipt_sha256="d" * 64,
    )
    assert policy.state(complete) is PriorState.ACTIVE_N600_CONFIRMED
    assert policy.compile_warm_start(complete)["activation_receipt"] == {
        "path": "experiments/results/jrd_n600/receipt.json",
        "sha256": "d" * 64,
        "archive_sha256": "b" * 64,
    }


def test_search_route_stops_only_the_exact_exhausted_formulation() -> None:
    policy = JrdReusablePriorPolicy()
    assert (
        policy.search_family_route(
            same_pr110_archive=True,
            family="uniform",
            witness_integrated_training=False,
        )
        == "STOP_EXACT_PR110_POSTHOC_RERUN"
    )
    assert (
        policy.search_family_route(
            same_pr110_archive=True,
            family="learned_conditional",
            witness_integrated_training=True,
        )
        == "ROUTE_TO_N600_WITNESS_MEASUREMENT"
    )


def test_equation_encodes_n1_firewall_and_scoped_null() -> None:
    equation = build_jrd_component_safe_entropy_edge_stop_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.empirical_anchors[0].inputs["eval_pairs"] == 1
    assert equation.empirical_anchors[0].empirical_output["component_safe_and_byte_shrinking_rows"] == 0
    assert equation.empirical_anchors[1].inputs["eval_pairs"] == 600
    assert equation.empirical_anchors[1].empirical_output["archive_bytes_saved"] == 0
    assert "FORMULATION x INSTANCE" in equation.domain_of_validity["measured_negative_scope"]


def test_population_is_append_only(tmp_path) -> None:
    path = tmp_path / "equations.jsonl"
    lock_path = tmp_path / "equations.lock"
    populate_jrd_component_safe_entropy_edge_stop_v1(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="jrd-priors",
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["equation_id"] == EQUATION_ID
