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


def test_incomplete_qdq_blocks_build() -> None:
    ticket = compile_ane_fixedpoint_ticket(qdq_receipt=_qdq(None), settled_r4_receipt=_r4())
    assert ticket.disposition is ANEDisposition.QDQ_RECEIPT_INCOMPLETE
    assert ticket.build_allowed is False


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
