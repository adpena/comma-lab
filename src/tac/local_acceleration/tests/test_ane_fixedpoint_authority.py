from __future__ import annotations

import pytest

from tac.local_acceleration.ane_fixedpoint_authority import (
    ANEDisposition,
    compile_ane_fixedpoint_ticket,
)


def _r4() -> dict[str, object]:
    return {
        "schema": "ane_unlock_r4_variants.v1",
        "coreml_w8a8_cpu_and_ne": {
            "status": "MEASURED",
            "metrics": {
                "heldout_n12": {
                    "argmax_flips": 1_081_426,
                    "total_px": 2_359_296,
                    "flip_rate": 0.4583680946397567,
                }
            },
        },
    }


def _qdq(bits: int | None) -> dict[str, object]:
    arm = f"w{bits}a{bits}" if bits is not None else None
    return {
        "schema": "dynamic_fixedpoint_scorer_forward_n600.v1",
        "contract": {
            "native_integer_speed_claim": False,
            "activation_scale_mode": "dynamic_exact_absmax",
        },
        "summary": {
            "full_real_n600": bits is not None,
            "minimum_argmax_exact_arm": arm,
            "arms": {arm: {"argmax_exact_admitted": True}} if arm else {},
        },
    }


def _tie_snap(*, exact: bool = True) -> dict[str, object]:
    return {
        "schema": "weight_l1_tie_snap_scorer_n600.v1",
        "contract": {"runtime_label_or_frame_dependent": False},
        "model_manifest": {
            "precision_histogram": {"27": 4, "28": 28, "29": 32, "30": 41, "31": 20},
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "assignment_rule": (
                "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            ),
            "label_or_frame_dependent": False,
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": exact,
            "selected_heldout_exact": exact,
            "selected_full_exact": exact,
            "minimum_calibration_exact_arm": "epsilon_2m19" if exact else None,
        },
    }


def _class_pair_tie_snap(*, exact: bool = True) -> dict[str, object]:
    receipt = _tie_snap(exact=exact)
    receipt["schema"] = "weight_l1_class_pair_tie_snap_scorer_n600.v1"
    receipt["contract"].update(  # type: ignore[union-attr]
        {
            "design_split": [0, 264],
            "second_validation_split": [264, 600],
            "epsilon": 2.0**-19,
            "candidate_winner_class": 4,
            "candidate_runner_class": 0,
            "replacement_class": 0,
            "rule_frozen_before_second_validation_access": True,
            "second_validation_reselection": False,
        }
    )
    receipt["summary"].update(  # type: ignore[union-attr]
        {"design_exact": exact, "second_validation_exact": exact}
    )
    return receipt


def test_incomplete_qdq_blocks_build() -> None:
    ticket = compile_ane_fixedpoint_ticket(qdq_receipt=_qdq(None), settled_r4_receipt=_r4())
    assert ticket.disposition is ANEDisposition.QDQ_RECEIPT_INCOMPLETE
    assert ticket.build_allowed is False


def test_complete_negative_qdq_is_not_mislabeled_incomplete() -> None:
    receipt = _qdq(None)
    receipt["summary"].update(  # type: ignore[union-attr]
        {"status": "MEASURED", "full_real_n600": True}
    )
    ticket = compile_ane_fixedpoint_ticket(
        qdq_receipt=receipt, settled_r4_receipt=_r4()
    )
    assert ticket.disposition is ANEDisposition.NO_EXACT_ARGMAX_QDQ_ARM
    assert ticket.build_allowed is False
    assert "FORMULATION" in ticket.verdict_scope


def test_settled_w8a8_is_not_remeasured_even_if_qdq_w8_passes() -> None:
    ticket = compile_ane_fixedpoint_ticket(qdq_receipt=_qdq(8), settled_r4_receipt=_r4())
    assert ticket.disposition is ANEDisposition.SETTLED_W8A8_FORMULATION_REFUSED
    assert "45.836809%" in ticket.req_r


def test_distinct_higher_bit_arm_is_unrepresentable_on_public_ane_api() -> None:
    ticket = compile_ane_fixedpoint_ticket(
        qdq_receipt=_qdq(22),
        settled_r4_receipt=_r4(),
        formulation_id="interval_certified_w22_mixed_precision_v1",
    )
    assert ticket.disposition is ANEDisposition.PUBLIC_ANE_PRECISION_UNREPRESENTABLE
    assert ticket.required_bits == 22
    assert ticket.required_activation_scale_mode == "dynamic_exact_absmax"
    assert ticket.build_allowed is False


def test_exact_tie_snap_successor_exposes_realized_higher_bit_api_block() -> None:
    ticket = compile_ane_fixedpoint_ticket(
        qdq_receipt=_tie_snap(),
        settled_r4_receipt=_r4(),
        formulation_id="weight_l1_tie_snap_w27_w31_v1",
    )
    assert ticket.disposition is ANEDisposition.PUBLIC_ANE_PRECISION_UNREPRESENTABLE
    assert ticket.required_bits == 27
    assert ticket.required_activation_scale_mode == "dynamic_exact_absmax"
    assert ticket.build_allowed is False


def test_exact_class_pair_successor_exposes_same_public_ane_api_block() -> None:
    ticket = compile_ane_fixedpoint_ticket(
        qdq_receipt=_class_pair_tie_snap(),
        settled_r4_receipt=_r4(),
        formulation_id="weight_l1_class_pair_tie_snap_w27_w31_v1",
    )
    assert ticket.disposition is ANEDisposition.PUBLIC_ANE_PRECISION_UNREPRESENTABLE
    assert ticket.required_bits == 27
    incomplete = compile_ane_fixedpoint_ticket(
        qdq_receipt=_class_pair_tie_snap(exact=False),
        settled_r4_receipt=_r4(),
        formulation_id="weight_l1_class_pair_tie_snap_w27_w31_v1",
    )
    assert incomplete.disposition is ANEDisposition.NO_EXACT_ARGMAX_NUMERICAL_ARM


def test_distinct_w8_reformulation_is_buildable_but_has_no_authority() -> None:
    ticket = compile_ane_fixedpoint_ticket(
        qdq_receipt=_qdq(8),
        settled_r4_receipt=_r4(),
        formulation_id="distinct_w8_error_feedback_v1",
    )
    assert ticket.disposition is ANEDisposition.DISTINCT_W8A8_REFORMULATION_BUILDABLE
    assert ticket.build_allowed is True
    assert ticket.authority_claim is False
    assert ticket.native_integer_speed_claim is False


def test_settled_receipt_custody_fails_closed() -> None:
    receipt = _r4()
    receipt["coreml_w8a8_cpu_and_ne"]["metrics"]["heldout_n12"]["argmax_flips"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="custody"):
        compile_ane_fixedpoint_ticket(qdq_receipt=_qdq(8), settled_r4_receipt=receipt)
