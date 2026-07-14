# SPDX-License-Identifier: MIT
"""Fail-closed ANE fixed-point authority admission for Task #494.

Core ML exposes calibrated eight-bit activation quantization, not a public
programmable higher-bit ANE kernel surface.  The existing calibrated W8A8
formulation is already measured and settled.  This module therefore compiles a
new-build ticket only when a *distinct* formulation and a full-n600 numerical
receipt make it logically possible; that receipt may be the original QDQ
ladder or its exact-int64/tie-snap successor.  The compiler never launders an
old W8A8 rerun into a new authority experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ANEDisposition(StrEnum):
    QDQ_RECEIPT_INCOMPLETE = "qdq_receipt_incomplete"
    NO_EXACT_ARGMAX_QDQ_ARM = "no_exact_argmax_qdq_arm"
    NO_EXACT_ARGMAX_NUMERICAL_ARM = "no_exact_argmax_numerical_arm"
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
    schema = receipt.get("schema")
    if schema in {
        "weight_l1_tie_snap_scorer_n600.v1",
        "weight_l1_class_pair_tie_snap_scorer_n600.v1",
    }:
        manifest = receipt.get("model_manifest", {})
        histogram = manifest.get("precision_histogram", {})
        realized_bits = sorted(
            int(bits)
            for bits, count in histogram.items()
            if int(count) > 0
        )
        common_exact = bool(
            summary.get("status") == "MEASURED"
            and summary.get("full_real_n600") is True
            and summary.get("argmax_exact_admitted") is True
            and realized_bits
            and int(manifest.get("converted_conv2d_count", -1)) == 125
            and manifest.get("accumulation") == "exact_signed_int64"
            and manifest.get("assignment_rule")
            == "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            and manifest.get("label_or_frame_dependent") is False
            and contract.get("runtime_label_or_frame_dependent") is False
        )
        if schema == "weight_l1_tie_snap_scorer_n600.v1":
            formulation_exact = bool(
                summary.get("selected_heldout_exact") is True
                and summary.get("selected_full_exact") is True
                and summary.get("minimum_calibration_exact_arm")
            )
        else:
            formulation_exact = bool(
                summary.get("design_exact") is True
                and summary.get("second_validation_exact") is True
                and contract.get("design_split") == [0, 264]
                and contract.get("second_validation_split") == [264, 600]
                and contract.get("candidate_winner_class") == 4
                and contract.get("candidate_runner_class") == 0
                and contract.get("replacement_class") == 0
                and float(contract.get("epsilon", -1.0)) == float(2.0**-19)
                and contract.get("rule_frozen_before_second_validation_access") is True
                and contract.get("second_validation_reselection") is False
            )
        if not (common_exact and formulation_exact):
            return None, None
        return realized_bits[0], "dynamic_exact_absmax"
    expected_schema = {
        "fixed_calibration": "fixedpoint_scorer_forward_n600.v2",
        "dynamic_exact_absmax": "dynamic_fixedpoint_scorer_forward_n600.v1",
    }.get(scale_mode)
    if schema != expected_schema:
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
    """Compile an ANE build decision from a numerical receipt without rerunning W8A8."""

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
        summary = qdq_receipt.get("summary", {})
        complete_negative = bool(
            summary.get("status") == "MEASURED"
            and summary.get("full_real_n600") is True
            and (
                summary.get("minimum_argmax_exact_arm") is None
                if qdq_receipt.get("schema")
                in {
                    "fixedpoint_scorer_forward_n600.v2",
                    "dynamic_fixedpoint_scorer_forward_n600.v1",
                }
                else summary.get("argmax_exact_admitted") is not True
            )
        )
        if complete_negative:
            is_qdq = qdq_receipt.get("schema") in {
                "fixedpoint_scorer_forward_n600.v2",
                "dynamic_fixedpoint_scorer_forward_n600.v1",
            }
            return ANEFixedPointTicket(
                disposition=(
                    ANEDisposition.NO_EXACT_ARGMAX_QDQ_ARM
                    if is_qdq
                    else ANEDisposition.NO_EXACT_ARGMAX_NUMERICAL_ARM
                ),
                build_allowed=False,
                verdict_scope=(
                    "FORMULATION: completed full real-n600 numerical ladder has no "
                    "exact-argmax arm; not the fixed-point or ANE family"
                ),
                req_r=(
                    "a completed distinct numerical formulation with an exact-argmax arm "
                    "before any ANE placement build"
                ),
                **common,
            )
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
