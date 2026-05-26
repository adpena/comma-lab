# SPDX-License-Identifier: MIT

from tac.optimization.entropy_position import (
    AFTER_ENTROPY_CODER,
    AT_ENTROPY_CODER,
    BEFORE_ENTROPY_CODER,
    META_ENTROPY_POSITION,
    UNKNOWN_ENTROPY_POSITION,
    classify_entropy_position,
)
from tac.optimization.materializer_feedback import (
    materializer_observation_feedback_rows,
)


def test_classifies_materializer_positions_by_entropy_stage() -> None:
    assert classify_entropy_position("tensor_factorize_v1")[
        "entropy_position_class"
    ] == BEFORE_ENTROPY_CODER
    assert classify_entropy_position("archive_section_entropy_recode_v1")[
        "entropy_position_class"
    ] == AT_ENTROPY_CODER
    assert classify_entropy_position("packet_member_zip_header_elide_v1")[
        "entropy_position_class"
    ] == AFTER_ENTROPY_CODER


def test_context_can_route_inverse_scorer_to_posenet_position() -> None:
    row = classify_entropy_position(
        "inverse_scorer_cell_candidate_v1",
        payload_context={"scorer_component": "PoseNet"},
    )

    assert row["entropy_position_id"] == "P19"
    assert row["entropy_position_class"] == BEFORE_ENTROPY_CODER
    assert row["downstream_entropy_coder_rerun_recommended"] is True


def test_higher_order_markov_routes_to_p14() -> None:
    direct = classify_entropy_position("higher_order_markov_selector_recode_v1")
    family = classify_entropy_position(
        "selector_stream_context_recode_v1",
        operation_family="fec8_static_second_order_markov2",
    )

    assert direct["entropy_position_id"] == "P14"
    assert direct["entropy_position_class"] == AT_ENTROPY_CODER
    assert family["entropy_position_id"] == "P14"
    assert family["entropy_position_name"] == "higher_order_context"


def test_meta_and_unknown_rows_fail_closed_for_position_bonus() -> None:
    meta = classify_entropy_position("inverse_steganalysis_high_level_operation_set_v1")
    unknown = classify_entropy_position("new_future_materializer_v99")

    assert meta["entropy_position_class"] == META_ENTROPY_POSITION
    assert meta["score_claim"] is False
    assert unknown["entropy_position_class"] == UNKNOWN_ENTROPY_POSITION
    assert unknown["entropy_position_composition_rule"] == "do_not_apply_position_bonus"
    assert unknown["ready_for_exact_eval_dispatch"] is False


def test_materializer_feedback_rows_carry_entropy_position_fields() -> None:
    rows = materializer_observation_feedback_rows(
        {
            "target_kind": "archive_section_entropy_recode_v1",
            "materializer_id": "archive_section_entropy_recode_adapter",
            "receiver_contract_kind": "family_agnostic_archive_section_entropy_recode",
            "receiver_contract_satisfied": True,
            "source_archive": {"sha256": "a" * 64, "bytes": 1000},
            "candidate_archive": {"sha256": "b" * 64, "bytes": 900},
            "serialized_archive_delta": {
                "realized_saved_bytes": 100,
                "source_archive_bytes": 1000,
                "candidate_archive_bytes": 900,
                "savings_realized": True,
            },
        }
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["entropy_position_id"] == "P9"
    assert row["entropy_position_class"] == AT_ENTROPY_CODER
    assert row["entropy_position"]["entropy_position_name"] == "codebook_entropy"
    assert row["score_claim"] is False
