from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_five_type_correspondence_20260724 import (
    EQUATION_ID,
    build_ddm_g2_five_type_correspondence_v1,
    g2_bucket_to_stream_types,
)
from tac.canonical_equations.evaluators import get_evaluator, has_evaluator
from tac.optimization.ddm_dr2b_tolerance_costate import frequency_band_admission
from tac.optimization.ddm_lambda_continuation_frontier import (
    CodedStream,
    MeasuredDescription,
    typed_dimension_dual_report,
)
from tac.optimization.ddm_min_description_contract import (
    TYPED_STREAM_SCHEMA,
    LayerHome,
    MinimumDescriptionContractError,
    StreamType,
    TypedStreamTag,
    build_minimum_description_headline,
)

REPO = Path(__file__).resolve().parents[2]
REPRICE = REPO / ".omx/research/ddm_ts1_g1_typed_stream_reprice_20260724.json"


def _tag(
    *,
    stream_type: StreamType = StreamType.FIBER,
    counted_bytes: int = 120,
) -> TypedStreamTag:
    return TypedStreamTag(
        type=stream_type,
        layer_home=LayerHome.L2_CHART,
        evaluate_py_recursion_level_cited="L2_chart -> L5_verdict",
        counted_bytes=counted_bytes,
        free_receiver_code=True,
    )


def _headline(**updates: object) -> dict:
    values = {
        "stored_problem_bytes": 100,
        "stored_problem_sha256": "a" * 64,
        "exception_bytes": 20,
        "exception_sha256": "b" * 64,
        "realized_d_seg": 0.001,
        "realized_d_pose": 0.0001,
        "stored_problem_own_lineage": True,
        "donor_conditioned": False,
        "expansion_receiver_closed": True,
        "pose_tube_active": True,
        "realized_uint8_r_frozen_scorers": True,
        "quotient_coordinates_only": True,
        "scorer_metric_active": True,
        "alternating_typed_subproblems": True,
        "typed_blocks_active": True,
        "per_dimension_quanta_active": True,
        "typed_stream_tags": (_tag(),),
    }
    values.update(updates)
    return build_minimum_description_headline(**values)  # type: ignore[arg-type]


def test_stream_type_vocabulary_is_exactly_the_five_type_contract() -> None:
    assert [member.value for member in StreamType] == [
        "SKELETON",
        "CONNECTION",
        "FIBER",
        "GAUGE",
        "RESIDUAL",
    ]


def test_layer_home_vocabulary_is_exactly_the_recursive_stack() -> None:
    assert [member.value for member in LayerHome] == [
        "L1_program",
        "L2_chart",
        "L3_raster",
        "L4_scorer_feature",
        "L5_verdict",
    ]


def test_typed_stream_tag_roundtrips_sealed_json() -> None:
    tag = _tag()
    assert tag.to_dict()["schema"] == TYPED_STREAM_SCHEMA
    assert TypedStreamTag.from_dict(tag.to_dict()) == tag


def test_typed_stream_tag_refuses_unknown_keys() -> None:
    malformed = {**_tag().to_dict(), "free_bytes": 120}
    with pytest.raises(MinimumDescriptionContractError, match="keys differ"):
        TypedStreamTag.from_dict(malformed)


def test_typed_stream_tag_refuses_empty_recursion_citation() -> None:
    with pytest.raises(MinimumDescriptionContractError, match="citation"):
        TypedStreamTag(
            type=StreamType.FIBER,
            layer_home=LayerHome.L2_CHART,
            evaluate_py_recursion_level_cited="",
            counted_bytes=1,
            free_receiver_code=False,
        )


def test_gauge_stream_must_be_zero_counted_bytes() -> None:
    with pytest.raises(MinimumDescriptionContractError, match="GAUGE"):
        _tag(stream_type=StreamType.GAUGE, counted_bytes=1)


def test_complete_typed_headline_reconciles_and_is_eligible() -> None:
    row = _headline()
    assert row["headline_eligible"] is True
    assert row["typed_stream_custody"]["reconciled"] is True
    assert row["decision_triple"]["total_counted_bytes"] == 120


def test_missing_tag_is_warn_only_but_headline_is_withheld() -> None:
    row = _headline(typed_stream_tags=None)
    assert row["headline_eligible"] is False
    assert row["typed_stream_custody"]["mode"] == "WARN_ONLY_WITH_HEADLINE_WITHHELD"
    assert "TYPED_STREAM_TAG_CUSTODY_MISSING_WARN_ONLY" in row["blockers"]


def test_strict_flip_refuses_missing_tag() -> None:
    with pytest.raises(MinimumDescriptionContractError, match="strict"):
        _headline(typed_stream_tags=None, strict_typed_stream_tags=True)


def test_untagged_waiver_is_recorded_but_never_authorizes_headline() -> None:
    row = _headline(
        typed_stream_tags=None,
        untagged_stream_waiver="Historical receipt predates the typed schema.",
    )
    assert row["headline_eligible"] is False
    assert "UNTAGGED_STREAM_WAIVER_NONAUTHORIZING" in row["blockers"]
    assert row["typed_stream_custody"]["waiver_authorizes_headline"] is False


def test_short_waiver_is_refused() -> None:
    with pytest.raises(MinimumDescriptionContractError, match="substantive"):
        _headline(typed_stream_tags=None, untagged_stream_waiver="legacy")


def test_typed_byte_mismatch_withholds_headline() -> None:
    row = _headline(typed_stream_tags=(_tag(counted_bytes=119),))
    assert "TYPED_STREAM_COUNTED_BYTES_DO_NOT_RECONCILE" in row["blockers"]
    assert row["decision_triple"]["total_counted_bytes"] is None


def test_dr2b_exact_r_null_rung_is_zero_byte_gauge() -> None:
    row = frequency_band_admission(
        exact_r_transfer_zero=True,
        emitted_description_bytes=0,
    )
    assert row["typed_stream_tag"]["type"] == "GAUGE"
    assert row["typed_stream_tag"]["counted_bytes"] == 0
    assert row["rate_price_admitted"] is False


def test_dr2b_visible_rung_prices_fiber_only() -> None:
    row = frequency_band_admission(
        exact_r_transfer_zero=False,
        emitted_description_bytes=9,
        flip_distance=0.25,
        delta_d_pose=0.01,
    )
    assert row["typed_stream_tag"]["type"] == "FIBER"
    assert row["typed_stream_tag"]["counted_bytes"] == 9
    assert row["rate_price_admitted"] is True


def _description(candidate_id: str, counted_bytes: int, d_seg: float) -> MeasuredDescription:
    return MeasuredDescription(
        candidate_id=candidate_id,
        counted_bytes=counted_bytes,
        d_seg=d_seg,
        d_pose=0.01,
        coded_streams=(
            CodedStream(
                stream_id=f"{candidate_id}.fiber",
                stratum="ALL",
                factor_kind="fiber",
                custody_role="stored_problem",
                counted_bytes=counted_bytes,
                sha256="c" * 64,
                codec="synthetic-test",
                source_path=f"{candidate_id}.bin",
            ),
        ),
        source_artifact=f"{candidate_id}.json",
        source_sha256="d" * 64,
        receiver_closure="measurement_harness_receiver_closed",
    )


def test_rd1_type_column_stays_null_without_dimension_byte_home() -> None:
    report = typed_dimension_dual_report(
        (
            _description("left", 100, 0.02),
            _description("right", 120, 0.01),
        )
    )
    assert report["axes"]["stream_type"] is None
    assert all(row["stream_type"] is None for row in report["bucket_rows"])
    assert all(row["stream_type"] is None for row in report["component_evidence"])


def test_g1_reprice_preserves_real_coder_selected_bytes() -> None:
    receipt = json.loads(REPRICE.read_text(encoding="utf-8"))
    source = json.loads(
        (REPO / ".omx/research/direct_description_g1_grammar_induction_20260722.json").read_text(encoding="utf-8")
    )
    assert receipt["pair_count"] == 600
    assert receipt["answer"]["measured_selected_knee_mispriced_bytes"] == 0
    assert receipt["answer"]["measured_selected_knee_before_counted_bytes"] == 276790
    assert receipt["answer"]["measured_selected_knee_after_counted_bytes"] == 276790
    assert receipt["coder_contract"]["real_coders"] == [
        "brotli_q11",
        "lzma1_raw_1m",
        "zlib9",
    ]
    expected = {
        "Movable": source["ranked_knees"]["Movable"]["lossy_knee"]["counted_bytes"],
        "Lane": source["ranked_knees"]["Lane"]["lossy_knee"]["counted_bytes"],
        "Boundary": source["ranked_knees"]["Boundary"]["lossy_fidelity_knee"]["counted_bytes"],
    }
    assert {row["stratum"]: row["before_counted_bytes"] for row in receipt["per_stratum_totals"]} == expected


def test_g1_reprice_rows_validate_against_single_typed_schema() -> None:
    receipt = json.loads(REPRICE.read_text(encoding="utf-8"))
    for row in receipt["per_stratum_vocabulary_grammar_table"]:
        tag = TypedStreamTag.from_dict(
            {
                "schema": TYPED_STREAM_SCHEMA,
                "type": row["type"],
                "layer_home": row["layer_home"],
                "evaluate_py_recursion_level_cited": row["evaluate_py_recursion_level_cited"],
                "counted_bytes": row["after_counted_bytes"],
                "free_receiver_code": row["free_receiver_code"],
            }
        )
        assert tag.counted_bytes == row["after_counted_bytes"]
    assert {row["type"] for row in receipt["per_stratum_vocabulary_grammar_table"]} == {
        "CONNECTION",
        "FIBER",
        "SKELETON",
    }


@pytest.mark.parametrize(
    ("bucket", "expected"),
    [
        ("scorer-invisible", ("GAUGE",)),
        ("xi-predictable", ("CONNECTION",)),
        ("chart-expressible", ("SKELETON", "FIBER")),
        ("irreducible", ("RESIDUAL",)),
    ],
)
def test_g2_five_type_correspondence(bucket: str, expected: tuple[str, ...]) -> None:
    assert g2_bucket_to_stream_types(bucket) == expected


def test_g2_five_type_correspondence_refuses_unknown_bucket() -> None:
    with pytest.raises(ValueError, match="bucket must be"):
        g2_bucket_to_stream_types("unknown")


def test_g2_equation_carries_preserved_receipt_as_empirical_anchor() -> None:
    equation = build_ddm_g2_five_type_correspondence_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 1
    assert equation.empirical_anchors[0].source_artifact.endswith("aggregate_ledger.json")
    assert equation.domain_of_validity["score_claim"] is False


def test_g2_lawref_evaluator_is_import_registered_and_callable() -> None:
    assert has_evaluator(EQUATION_ID)
    evaluator = get_evaluator(EQUATION_ID)
    assert evaluator({"bucket": "irreducible"}) == ("RESIDUAL",)
    with pytest.raises(ValueError, match="exactly"):
        evaluator({"bucket": "irreducible", "extra": True})
