# SPDX-License-Identifier: MIT
"""Fail-closed ANE fixed-point authority admission for Task #494.

Core ML exposes calibrated eight-bit activation quantization, not a public
programmable higher-bit ANE kernel surface.  The existing calibrated W8A8
formulation is already measured and settled.  This module therefore compiles a
new-build ticket only when a *distinct* formulation and a full-n600 numerical
receipt make it logically possible; it never launders an old W8A8 rerun into a
new authority experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ANEDisposition(StrEnum):
    QDQ_RECEIPT_INCOMPLETE = "qdq_receipt_incomplete"
    SETTLED_W8A8_FORMULATION_REFUSED = "settled_w8a8_formulation_refused"
    PUBLIC_ANE_PRECISION_UNREPRESENTABLE = "public_ane_precision_unrepresentable"
    DISTINCT_W8A8_REFORMULATION_BUILDABLE = "distinct_w8a8_reformulation_buildable"


@dataclass(frozen=True)
class ANEFixedPointTicket:
    disposition: ANEDisposition
    build_allowed: bool
    required_bits: int | None
    required_activation_scale_mode: str | None
    public_activation_bits: tuple[int, ...]
    settled_formulation: str
    verdict_scope: str
    req_r: str
    native_integer_speed_claim: bool = False
    authority_claim: bool = False
    score_claim: bool = False
    pointer_moved: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["disposition"] = self.disposition.value
        return payload


def _minimum_exact_arm(receipt: Mapping[str, Any]) -> tuple[int | None, str | None]:
    contract = receipt.get("contract", {})
    summary = receipt.get("summary", {})
    scale_mode = contract.get("activation_scale_mode") or (
        "fixed_calibration"
        if receipt.get("schema") == "fixedpoint_scorer_forward_n600.v2"
        else None
    )
    expected_schema = {
        "fixed_calibration": "fixedpoint_scorer_forward_n600.v2",
        "dynamic_exact_absmax": "dynamic_fixedpoint_scorer_forward_n600.v1",
    }.get(scale_mode)
    if receipt.get("schema") != expected_schema:
        return None, None
    if contract.get("native_integer_speed_claim") is not False:
        return None, None
    if summary.get("full_real_n600") is not True:
        return None, None
    arm = summary.get("minimum_argmax_exact_arm")
    if not isinstance(arm, str) or not arm.startswith("w") or "a" not in arm:
        return None, None
    try:
        bits = int(arm[1 : arm.index("a")])
    except ValueError:
        return None, None
    if summary.get("arms", {}).get(arm, {}).get("argmax_exact_admitted") is not True:
        return None, None
    return bits, str(scale_mode)


def _settled_w8a8_negative(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("schema") != "ane_unlock_r4_variants.v1":
        return False
    row = receipt.get("coreml_w8a8_cpu_and_ne", {})
    heldout = row.get("metrics", {}).get("heldout_n12", {})
    return bool(
        row.get("status") == "MEASURED"
        and int(heldout.get("argmax_flips", -1)) == 1_081_426
        and int(heldout.get("total_px", -1)) == 2_359_296
        and float(heldout.get("flip_rate", -1.0)) > 0.45
    )


def compile_ane_fixedpoint_ticket(
    *,
    qdq_receipt: Mapping[str, Any],
    settled_r4_receipt: Mapping[str, Any],
    formulation_id: str = "coreml_linear_symmetric_per_channel_w8a8_ptq",
) -> ANEFixedPointTicket:
    """Compile an ANE build decision without rerunning a settled formulation."""

    if not _settled_w8a8_negative(settled_r4_receipt):
        raise ValueError("settled #482 W8A8 receipt custody is missing or altered")
    bits, scale_mode = _minimum_exact_arm(qdq_receipt)
    common = {
        "required_bits": bits,
        "required_activation_scale_mode": scale_mode,
        "public_activation_bits": (8,),
        "settled_formulation": "coreml_linear_symmetric_per_channel_w8a8_ptq",
    }
    if bits is None:
        return ANEFixedPointTicket(
            disposition=ANEDisposition.QDQ_RECEIPT_INCOMPLETE,
            build_allowed=False,
            verdict_scope="INSTANCE: full real-n600 fixed-point feasibility receipt",
            req_r="complete exact pair-index/hash custody for 0..599 and select an exact-argmax arm",
            **common,
        )
    if formulation_id == common["settled_formulation"]:
        return ANEFixedPointTicket(
            disposition=ANEDisposition.SETTLED_W8A8_FORMULATION_REFUSED,
            build_allowed=False,
            verdict_scope=(
                "FORMULATION: CoreML 9 linear-symmetric per-channel-weight/per-tensor-activation "
                "W8A8 PTQ on frozen SegNet; not the ANE or fixed-point family"
            ),
            req_r=(
                "name and implement a mathematically distinct correction/precision formulation; "
                "do not rerun the settled 45.836809% held-out-flip arm"
            ),
            **common,
        )
    if bits not in common["public_activation_bits"]:
        return ANEFixedPointTicket(
            disposition=ANEDisposition.PUBLIC_ANE_PRECISION_UNREPRESENTABLE,
            build_allowed=False,
            verdict_scope=(
                f"FORMULATION/API: public CoreML 9 activation quantization exposes int8, while "
                f"the full-n600 exact arm requires {bits} bits"
            ),
            req_r=(
                "a public higher-bit ANE activation-compute surface or a distinct certified "
                "mixed-precision decomposition with proved mostly-ANE placement"
            ),
            **common,
        )
    return ANEFixedPointTicket(
        disposition=ANEDisposition.DISTINCT_W8A8_REFORMULATION_BUILDABLE,
        build_allowed=True,
        verdict_scope=(
            "FORMULATION CANDIDATE: distinct W8A8 graph; still local advisory until n600, "
            "cross-process, placement, latency, and terminal CPU/CUDA gates pass"
        ),
        req_r=(
            "build in certified scratch, prove exact 0..599 argmax/certificate custody, prove "
            "ANE placement, and measure fully charged latency"
        ),
        **common,
    )


__all__ = [
    "ANEDisposition",
    "ANEFixedPointTicket",
    "compile_ane_fixedpoint_ticket",
]
