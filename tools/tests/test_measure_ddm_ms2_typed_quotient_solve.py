from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

TOOL = Path("tools/measure_ddm_ms2_typed_quotient_solve.py")
SPEC = importlib.util.spec_from_file_location("measure_ddm_ms2_typed_quotient_solve", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FINAL_RECEIPT = Path(
    ".omx/research/ddm_ms2_typed_quotient_solve_20260724_receipt.json"
)
FINAL_RECEIPT_SHA256 = (
    "04060edf9834b661f12a9794e50ceadf7dd4ab114baf55a15555537abc71e419"
)


def _receipt() -> dict:
    return MODULE.build_receipt(finished_at_utc="2026-07-24T03:00:00Z")


def test_receipt_fails_closed_without_faking_n600_measurement() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "ddm_ms2_typed_quotient_solve_repo_receipt.v1"
    assert receipt["verdict"] == "BLOCKED_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE"
    assert receipt["score_claim"] is False
    assert receipt["promotion_eligible"] is False
    assert receipt["pointer"] == "0.1910828242 [contest-CPU]"
    assert receipt["pointer_moved"] is False
    assert receipt["measurement"] == {
        "finished_at_utc": "2026-07-24T03:00:00Z",
        "required_pair_count": 600,
        "measured_pair_count": 0,
        "torch_threads_required": 4,
        "torch_invoked": False,
        "receiver_invoked": False,
        "r_operator_invoked": False,
        "frozen_scorer_invoked": False,
        "status": "NOT_RUN_FAIL_CLOSED_BEFORE_CANDIDATE",
        "reason": receipt["blockers"],
    }
    assert not any(receipt["actuation"].values())
    assert "family" in receipt["verdict_scope"]
    assert receipt["main_landing_review_required"] is True


def test_predecessor_and_pf2_custody_are_explicit_and_scoped() -> None:
    receipt = _receipt()
    assert receipt["schema_authority_note"]["landed_predecessor_reality"].endswith(
        "executable headline/typing helpers are v1"
    )
    ms1 = receipt["settled_ms1_facts_quoted_not_rerun"]
    assert ms1["best_conditional_coder"] == {
        "bytes": 731_622_325,
        "fraction_saved": 0.01744088062351428,
        "times_target_archive": 4734.684094380161,
        "target_archive_bytes": 154_524,
    }
    assert ms1["identity_euclidean_full_kernel_cvp_control"]["proposal_wins"] == 0
    assert ms1["unchanged_member_oracle"]["pair_count"] == 600
    assert ms1["unchanged_member_oracle"]["d_seg"] == 0.0001519690619574653
    assert ms1["unchanged_member_oracle"]["d_pose"] == 0.00010184327939026322

    pf2 = receipt["input_custody"]["pf2_main_landing_handoff"]
    assert pf2["status"] == "MAIN_LANDED_CITED_NOT_LOCALLY_REHASHED"
    assert pf2["main_commit_short"] == "b8c81edec2"
    assert pf2["reported_bucket_count"] == 1200
    assert pf2["class_pair_count"] == 10
    assert pf2["measured_event_mass"] == 4_011_236
    assert pf2["largest_pair_bucket"] == {
        "class_pair": "Road-Undrivable",
        "event_mass": 1_280_501,
    }
    assert pf2["receipt_sha256"] == (
        "85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73"
    )
    assert pf2["invalidated_identity_metric_receipt_sha256"] == (
        "e48468898a64935168ae863d55efd4e38e0cb27ea18278998a1901bc858229fd"
    )
    assert pf2["metric_rerun_blocker_id"] == (
        "PF2_METRIC_ACTIVE_THREE_FORMULATION_ADJUDICATION_INCOMPLETE"
    )
    assert pf2["safe_use"].startswith("CITE_TYPED_ATLAS_FOR_MAIN_COMPOSITION")


def test_five_typings_and_dual_rows_remain_unavailable_not_imputed() -> None:
    receipt = _receipt()
    declarations = {
        name: row["declaration"] for name, row in receipt["five_typings"].items()
    }
    assert declarations == {
        "quotient_coordinates_only": False,
        "scorer_metric_active": False,
        "alternating_typed_subproblems": False,
        "typed_blocks_active": False,
        "per_dimension_quanta_active": False,
    }
    assert [row["type"] for row in receipt["representation_rows"]] == [
        "SKELETON",
        "CONNECTION",
        "FIBER",
        "GAUGE",
        "RESIDUAL",
    ]
    assert receipt["per_block_dual_exchange_rows"] == []
    slots = receipt["unavailable_pair_slots"]
    assert len(slots) == 10
    assert all(slot["availability"] == "UNAVAILABLE" for slot in slots)
    assert all(slot["pooling"] == "FORBIDDEN" for slot in slots)
    for slot in slots:
        assert slot["measured_flip_mass"] is None
        assert slot["kkt_dual"] is None
        assert slot["score_gain_per_byte"] is None
        assert slot["exact_delta_bytes"] is None
    assert receipt["dual_policy"]["imputation_forbidden"] is True
    assert receipt["train_decision_table_feed"]["status"] == "BLOCKED_CUSTODY_NO_DUAL_ROWS"


def test_directives_and_triality_are_durable_rows() -> None:
    receipt = _receipt()
    sources = {row["source"]: row for row in receipt["directive_consumption"]}
    assert sources["inbox:2026-07-24T02:27:12Z"]["status"] == "CONSUMED_P0"
    assert sources["inbox:2026-07-24T02:28:21Z"]["status"] == "CONSUMED_P0"
    assert sources["inbox:2026-07-24T03:02:01Z"]["status"] == (
        "CONSUMED_AS_TYPED_PRIOR_SOURCE_CUSTODY_OWED"
    )
    assert sources["inbox:2026-07-24T03:13:37Z"]["status"] == (
        "CONSUMED_MAIN_LANDING_CITATION"
    )
    assert sources["RD1 dual feed"]["status"] == "NOT_PRESENT_NOT_IMPUTED"
    ws1 = receipt["ws1_pose_serving_fiber_prior"]
    assert ws1["pose_serving_fiber_reclassified"] == 8
    assert ws1["used_as_measured_block_row"] is False
    typed = receipt["five_typings"]["typed_blocks_active"]
    assert typed["canonical_contract_module"] == (
        "tac.optimization.ddm_min_description_contract"
    )
    assert typed["parallel_enum_defined"] is False
    assert typed["quoted_pf2_class_pair_coverage"] == 10
    assert typed["locally_composed_class_pair_coverage"] == 0
    assert receipt["named_downstream_consumers"][1] == {
        "consumer": "pf2r metric-active three-formulation rerun",
        "blocker_id": (
            "PF2_METRIC_ACTIVE_THREE_FORMULATION_ADJUDICATION_INCOMPLETE"
        ),
        "status": "BLOCKED_MISSING_MS2_METRIC_CUSTODY_ROWS",
    }
    assert len(receipt["canonical_equations"]["equation_ids"]) == 5
    assert receipt["triality"]["equations"] == receipt["canonical_equations"]["equation_ids"]
    assert receipt["triality"]["dag"].endswith("_DAG_FEED_20260724.md")


def test_final_receipt_bytes_are_immutable_and_fail_closed() -> None:
    raw = FINAL_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FINAL_RECEIPT_SHA256
    receipt = json.loads(raw)
    assert receipt["measurement"]["finished_at_utc"] == "2026-07-24T03:18:00Z"
    assert receipt["measurement"]["measured_pair_count"] == 0
    assert receipt["campaign_headline"]["headline_eligible"] is False
    assert receipt["per_block_dual_exchange_rows"] == []
