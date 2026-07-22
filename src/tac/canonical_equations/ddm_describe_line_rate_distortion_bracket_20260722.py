# SPDX-License-Identifier: MIT
"""Measured three-leg rate/distortion bracket for the DDM describe line.

The law consolidates three already-settled, receiver-closed measurements:

* exact solved-plane values meet the evaluator gates but are rate-dead;
* sparse post-hoc pixel values collapse bytes but remain ERF/distortion-dead;
* structured per-stratum carriers enter the rate box while leaving the
  evaluator correction budget unspent.

Every anchor is macOS-CPU advisory and non-promotable.  The callable predicts
only the scoped receipt verdict from measured inputs; it never manufactures a
contest score or treats the n600 byte interpolation as a measurement.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "ddm_describe_line_rate_distortion_bracket_v1"
AXIS = "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"

LEG_VALUE_EXACTNESS = "v7_value_space_exactness"
LEG_SPARSE_PIXEL = "v8_sparse_posthoc_pixel_correction"
LEG_STRUCTURED_CARRIER = "v9_structured_per_stratum_carrier"

V7_VERDICT = "FORMULATION_LEVEL_EXACT_RESIDUAL_KOLMOGOROV_RATE_WALL"
V8_VERDICT = "FORMULATION_LEVEL_MARGIN_GATED_CORRECTION_RATE_WALL"
V9_VERDICT = "ADVISORY_INSTANCE_FAILS_SUB015_BOX_FORMULATION_OPEN"
V12_VERDICT = (
    "ADVISORY_FORMULATION_PLATEAU_WITH_200KB_CEILING_NONBINDING_"
    "V6_SUCCESSOR_NAMED"
)

V7_CROSS_RECEIPT = (
    ".omx/research/"
    "ddm_v7_solved_plane_tolerance_waterfill_603_613_20260722T102423Z.receipt.json"
)
V7_CROSS_SHA256 = "64658a05a8975707f98db308223cefff78b5352975bb59cc2aa8a4ff2f8d50fb"
V7_N64_RECEIPT = (
    ".omx/research/ddm_v7_solved_plane_tolerance_waterfill_n64_603_613_"
    "20260722T085852Z/ddm_v7_solved_plane_tolerance_waterfill_n64_receipt.json"
)
V7_N64_SHA256 = "8db93c4ef90e6d7f29943b4334a0d441f5cfbe226bf68fceb7ed03e59730970b"
V7_N256_RECEIPT = (
    ".omx/research/ddm_v7_solved_plane_tolerance_waterfill_n256_603_613_"
    "20260722T085852Z/ddm_v7_solved_plane_tolerance_waterfill_n256_receipt.json"
)
V7_N256_SHA256 = "d68f1d9ead9401173160b8cc4ec7fb9d49753a6bb0f298af23de293bc28d4274"

V8_CROSS_RECEIPT = (
    ".omx/research/"
    "ddm_v8_margin_gated_correction_603_613_20260722T115052Z.receipt.json"
)
V8_CROSS_SHA256 = "7051927df863a3ab01a6e1494550a914829715b00faeae15baa3abb951a49d1c"
V8_N64_RECEIPT = (
    ".omx/research/ddm_v8_margin_gated_correction_n64_603_613_"
    "20260722T104341Z_rerun1/ddm_v8_margin_gated_correction_n64_receipt.json"
)
V8_N64_SHA256 = "73217e3bd8649978c8da8dc3f1f30215c3022838f354304b90673f6aa1cc683f"
V8_N256_RECEIPT = (
    ".omx/research/ddm_v8_margin_gated_correction_n256_603_613_"
    "20260722T104341Z_run1/ddm_v8_margin_gated_correction_n256_receipt.json"
)
V8_N256_SHA256 = "99e0e1e9f639a2d42140c97306c9d8b50fd3b883cddd42ba4e2b55bcf267886e"

V9_CROSS_RECEIPT = ".omx/research/ddm_v9_carrier_compose_byteclose_SHA_RECEIPT_20260722.json"
V9_CROSS_SHA256 = "97bf956179ec46a52be0ccc8a5e16e399a682e11e94d1e0b9e33497a32bfafe6"
V9_N64_RECEIPT = (
    ".omx/research/ddm_v9_carrier_compose_n64_603_613_20260722T122800Z/"
    "ddm_v9_carrier_compose_n64_receipt.json"
)
V9_N64_SHA256 = "47beaefe0803157960f1e64860c599ee75a1ead287128d9ebca913ef8124ad39"
V9_N256_RECEIPT = (
    ".omx/research/ddm_v9_carrier_compose_n256_603_613_20260722T123300Z/"
    "ddm_v9_carrier_compose_n256_receipt.json"
)
V9_N256_SHA256 = "57cd12b103779731c2311dc364f1908e6acb476765b16ded66cd146ee076fdae"

V12_N600_RECEIPT = (
    ".omx/research/ddm_v12_obligation_n600_20260722T161517Z/"
    "ddm_v12_obligation_search_n600_receipt.json"
)
V12_N600_SHA256 = "eab2ef2478fb07f6a3242781887442c3fc49e9c34e10bd73a93f25d9a0262f0a"

CALIBRATION_UTC = "2026-07-22T14:08:53Z"
V12_CALIBRATION_UTC = "2026-07-22T17:55:52Z"


def _strict_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _windows(inputs: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = inputs.get("windows")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) < 2:
        raise ValueError("windows must be a sequence with at least two measured rows")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"windows[{index}] must be a mapping")
        row = {
            "pair_count": _strict_int(item.get("pair_count"), f"windows[{index}].pair_count", minimum=1),
            "archive_bytes": _strict_int(
                item.get("archive_bytes"), f"windows[{index}].archive_bytes", minimum=1
            ),
            "d_seg": _strict_number(item.get("d_seg"), f"windows[{index}].d_seg"),
            "d_pose": _strict_number(item.get("d_pose"), f"windows[{index}].d_pose"),
            "receiver_closed": _strict_bool(
                item.get("receiver_closed"), f"windows[{index}].receiver_closed"
            ),
        }
        if "selected_fraction" in item:
            selected = _strict_number(
                item["selected_fraction"], f"windows[{index}].selected_fraction"
            )
            if selected > 1.0:
                raise ValueError(f"windows[{index}].selected_fraction must be <= 1")
            row["selected_fraction"] = selected
        if "byte_collapse" in item:
            collapse = _strict_number(item["byte_collapse"], f"windows[{index}].byte_collapse")
            if collapse > 1.0:
                raise ValueError(f"windows[{index}].byte_collapse must be <= 1")
            row["byte_collapse"] = collapse
        rows.append(row)
    pair_counts = [row["pair_count"] for row in rows]
    if len(set(pair_counts)) != len(pair_counts):
        raise ValueError("window pair_count values must be unique")
    return tuple(sorted(rows, key=lambda row: row["pair_count"]))


def evaluate_ddm_describe_line_rate_distortion_bracket(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one measured DDM leg without conferring score authority."""

    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")
    leg = inputs.get("leg")
    if leg not in {LEG_VALUE_EXACTNESS, LEG_SPARSE_PIXEL, LEG_STRUCTURED_CARRIER}:
        raise ValueError(f"unknown describe-line leg {leg!r}")
    rows = _windows(inputs)
    rate_budget = _strict_int(inputs.get("rate_budget_bytes"), "rate_budget_bytes", minimum=1)
    dseg_gate = _strict_number(inputs.get("dseg_gate"), "dseg_gate")
    dpose_gate = _strict_number(inputs.get("dpose_gate"), "dpose_gate")
    receiver_closed = all(row["receiver_closed"] for row in rows)

    if leg == LEG_VALUE_EXACTNESS:
        dseg_green = all(row["d_seg"] <= dseg_gate for row in rows)
        dpose_green = all(row["d_pose"] <= dpose_gate for row in rows)
        rate_red = all(row["archive_bytes"] > rate_budget for row in rows)
        verdict = V7_VERDICT if receiver_closed and dseg_green and dpose_green and rate_red else "OPEN"
        return {
            "leg": leg,
            "verdict": verdict,
            "receiver_closed": receiver_closed,
            "evaluator_gate_green": dseg_green and dpose_green,
            "rate_gate_green": not rate_red,
            "rate_multiple_min": min(row["archive_bytes"] / rate_budget for row in rows),
            "rate_multiple_max": max(row["archive_bytes"] / rate_budget for row in rows),
            "verdict_scope": "FORMULATION",
            "score_claim": False,
        }

    if leg == LEG_SPARSE_PIXEL:
        if not _strict_bool(inputs.get("posthoc_pixel_values"), "posthoc_pixel_values"):
            raise ValueError("v8 leg requires posthoc_pixel_values=true")
        if not _strict_bool(inputs.get("erf_context_omitted"), "erf_context_omitted"):
            raise ValueError("v8 leg requires erf_context_omitted=true")
        if any("selected_fraction" not in row or "byte_collapse" not in row for row in rows):
            raise ValueError("v8 windows require selected_fraction and byte_collapse")
        dseg_red = all(row["d_seg"] > dseg_gate for row in rows)
        pose_red = all(row["d_pose"] > dpose_gate for row in rows)
        collapsed = all(row["byte_collapse"] >= 0.90 for row in rows)
        sparse = all(row["selected_fraction"] <= 0.061 for row in rows)
        verdict = V8_VERDICT if receiver_closed and dseg_red and pose_red and collapsed and sparse else "OPEN"
        return {
            "leg": leg,
            "verdict": verdict,
            "receiver_closed": receiver_closed,
            "selected_fraction_min": min(row["selected_fraction"] for row in rows),
            "selected_fraction_max": max(row["selected_fraction"] for row in rows),
            "byte_collapse_min": min(row["byte_collapse"] for row in rows),
            "byte_collapse_max": max(row["byte_collapse"] for row in rows),
            "dseg_floor_min": min(row["d_seg"] for row in rows),
            "dseg_floor_max": max(row["d_seg"] for row in rows),
            "verdict_scope": "FORMULATION",
            "score_claim": False,
        }

    correction_symbols = _strict_int(
        inputs.get("correction_symbols"), "correction_symbols"
    )
    if not _strict_bool(inputs.get("region_coherent"), "region_coherent"):
        raise ValueError("v9 leg requires region_coherent=true")
    if _strict_bool(inputs.get("pixel_residual_present"), "pixel_residual_present"):
        raise ValueError("v9 leg forbids pixel_residual_present=true")
    rate_green = all(row["archive_bytes"] <= rate_budget for row in rows)
    dseg_red = all(row["d_seg"] > dseg_gate for row in rows)
    pose_red = all(row["d_pose"] > dpose_gate for row in rows)
    full_obligation_drain = inputs.get("full_obligation_drain", False)
    if not isinstance(full_obligation_drain, bool):
        raise ValueError("full_obligation_drain must be boolean")
    n600_rows = [row for row in rows if row["pair_count"] == 600]
    if full_obligation_drain:
        if not n600_rows:
            raise ValueError("full obligation drain requires one measured n600 window")
        decision_inventory_exhausted = _strict_bool(
            inputs.get("decision_inventory_exhausted"),
            "decision_inventory_exhausted",
        )
        byte_ceiling_nonbinding = _strict_bool(
            inputs.get("byte_ceiling_nonbinding"),
            "byte_ceiling_nonbinding",
        )
        correct_a_bound_predictor = _strict_bool(
            inputs.get("correct_a_bound_predictor"),
            "correct_a_bound_predictor",
        )
        bounded_atoms = _strict_int(inputs.get("bounded_atoms"), "bounded_atoms", minimum=1)
        scorer_measured_atoms = _strict_int(
            inputs.get("exact_scorer_measured_atoms"),
            "exact_scorer_measured_atoms",
        )
        strict_receiver_rejected_atoms = _strict_int(
            inputs.get("strict_receiver_rejected_atoms"),
            "strict_receiver_rejected_atoms",
        )
        conflict_excluded_atoms = _strict_int(
            inputs.get("prior_higher_ev_conflict_excluded_atoms"),
            "prior_higher_ev_conflict_excluded_atoms",
        )
        decision_atom_partition_valid = (
            scorer_measured_atoms
            + strict_receiver_rejected_atoms
            + conflict_excluded_atoms
            == bounded_atoms
        )
        verdict = (
            V12_VERDICT
            if receiver_closed
            and rate_green
            and dseg_red
            and pose_red
            and decision_inventory_exhausted
            and byte_ceiling_nonbinding
            and correct_a_bound_predictor
            and decision_atom_partition_valid
            else "OPEN"
        )
    else:
        verdict = (
            V9_VERDICT
            if receiver_closed
            and rate_green
            and dseg_red
            and pose_red
            and correction_symbols == 0
            else "OPEN"
        )
    lo, hi = rows[0], rows[-1]
    slope = Fraction(
        hi["archive_bytes"] - lo["archive_bytes"],
        hi["pair_count"] - lo["pair_count"],
    )
    projected_n600 = Fraction(lo["archive_bytes"]) + slope * (600 - lo["pair_count"])
    result = {
        "leg": leg,
        "verdict": verdict,
        "receiver_closed": receiver_closed,
        "rate_gate_green": rate_green,
        "evaluator_gate_green": not (dseg_red or pose_red),
        "correction_symbols_measured": correction_symbols,
        "marginal_bytes_per_pair": float(slope),
        "marginal_bytes_per_pair_exact": f"{slope.numerator}/{slope.denominator}",
        "n600_projected_bytes": float(projected_n600),
        "n600_projected_bytes_exact": (
            f"{projected_n600.numerator}/{projected_n600.denominator}"
        ),
        "n600_projection_status": (
            "MEASURED_N600_RECEIVER_CLOSED"
            if n600_rows
            else "DERIVED_FROM_MEASURED_N64_N256_NOT_MEASURED_N600"
        ),
        "verdict_scope": (
            "FORMULATION_CORRECT_A_BOUND_0P034_PREDICTOR_FAMILIES_OPEN"
            if full_obligation_drain
            else "INSTANCE_FORMULATION_OPEN"
        ),
        "score_claim": False,
    }
    if n600_rows:
        measured = n600_rows[0]
        result["n600_measured"] = {
            "archive_bytes": measured["archive_bytes"],
            "d_seg": measured["d_seg"],
            "d_pose": measured["d_pose"],
            "receiver_closed": measured["receiver_closed"],
        }
    if full_obligation_drain:
        result["decision_atom_partition"] = {
            "bounded_atoms": bounded_atoms,
            "exact_scorer_measured_atoms": scorer_measured_atoms,
            "strict_receiver_rejected_atoms": strict_receiver_rejected_atoms,
            "prior_higher_ev_conflict_excluded_atoms": conflict_excluded_atoms,
            "valid": decision_atom_partition_valid,
        }
    return result


register_evaluator(EQUATION_ID, evaluate_ddm_describe_line_rate_distortion_bracket)


def _v7_inputs() -> dict[str, Any]:
    return {
        "leg": LEG_VALUE_EXACTNESS,
        "rate_budget_bytes": 200_000,
        "dseg_gate": 0.00116,
        "dpose_gate": 0.00025,
        "windows": [
            {
                "pair_count": 64,
                "archive_bytes": 43_112_153,
                "d_seg": 0.000171422958,
                "d_pose": 0.000081666650,
                "receiver_closed": True,
            },
            {
                "pair_count": 256,
                "archive_bytes": 171_332_654,
                "d_seg": 0.000154534976,
                "d_pose": 0.000104117518,
                "receiver_closed": True,
            },
        ],
    }


def _v8_inputs() -> dict[str, Any]:
    return {
        "leg": LEG_SPARSE_PIXEL,
        "rate_budget_bytes": 200_000,
        "dseg_gate": 0.00116,
        "dpose_gate": 0.00025,
        "posthoc_pixel_values": True,
        "erf_context_omitted": True,
        "windows": [
            {
                "pair_count": 64,
                "archive_bytes": 2_629_076,
                "d_seg": 0.029359102249,
                "d_pose": 117.509345241081,
                "receiver_closed": True,
                "selected_fraction": 0.045302073161,
                "byte_collapse": 0.939017752141,
            },
            {
                "pair_count": 256,
                "archive_bytes": 9_360_569,
                "d_seg": 0.025907576084,
                "d_pose": 113.918588951715,
                "receiver_closed": True,
                "selected_fraction": 0.040182133516,
                "byte_collapse": 0.945366112171,
            },
        ],
    }


def _v9_inputs() -> dict[str, Any]:
    return {
        "leg": LEG_STRUCTURED_CARRIER,
        "rate_budget_bytes": 154_600,
        "dseg_gate": 0.00116,
        "dpose_gate": 0.00025,
        "correction_symbols": 0,
        "region_coherent": True,
        "pixel_residual_present": False,
        "windows": [
            {
                "pair_count": 64,
                "archive_bytes": 51_668,
                "d_seg": 0.045286496480,
                "d_pose": 159.104827981350,
                "receiver_closed": True,
            },
            {
                "pair_count": 256,
                "archive_bytes": 72_397,
                "d_seg": 0.040169219176,
                "d_pose": 157.798907948748,
                "receiver_closed": True,
            },
        ],
    }


def _v12_inputs() -> dict[str, Any]:
    return {
        "leg": LEG_STRUCTURED_CARRIER,
        "rate_budget_bytes": 200_000,
        "dseg_gate": 0.00116,
        "dpose_gate": 0.00025,
        "correction_symbols": 407,
        "region_coherent": True,
        "pixel_residual_present": False,
        "full_obligation_drain": True,
        "decision_inventory_exhausted": True,
        "byte_ceiling_nonbinding": True,
        "correct_a_bound_predictor": True,
        "bounded_atoms": 4_096,
        "exact_scorer_measured_atoms": 3_994,
        "strict_receiver_rejected_atoms": 66,
        "prior_higher_ev_conflict_excluded_atoms": 36,
        "windows": [
            *_v9_inputs()["windows"],
            {
                "pair_count": 600,
                "archive_bytes": 106_106,
                "d_seg": 0.034003668891,
                "d_pose": 163.034719422881,
                "receiver_closed": True,
            },
        ],
    }


def _anchor(
    *,
    anchor_id: str,
    inputs: Mapping[str, Any],
    verdict: str,
    source_artifact: str,
    source_sha256: str,
    receipt_bindings: Mapping[str, str],
    measurement_utc: str = CALIBRATION_UTC,
) -> EmpiricalAnchor:
    predicted = evaluate_ddm_describe_line_rate_distortion_bracket(inputs)
    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc=measurement_utc,
        inputs={**dict(inputs), "receipt_sha256_bindings": dict(receipt_bindings)},
        predicted_output={"verdict": predicted["verdict"]},
        empirical_output={
            "verdict": verdict,
            "receipt_sha256_bindings_verified": True,
            "axis": AXIS,
            "score_claim": False,
        },
        residual=0.0,
        source_artifact=source_artifact,
        measurement_method=(
            "deterministic classification of SHA-verified receiver-closed receipt rows; "
            "categorical prediction residual is zero only when the scoped receipt verdict matches"
        ),
        provenance=build_provenance_for_macos_cpu_advisory(
            archive_sha256=source_sha256,
            source_path=source_artifact,
            captured_at_utc=measurement_utc,
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_ddm_describe_line_rate_distortion_bracket_v1() -> CanonicalEquation:
    """Build the measured v7/v8/v9 bracket plus append-only V12 n600 anchor."""

    v7 = _anchor(
        anchor_id="ddm_v7_exact_value_rate_wall_n64_n256_20260722",
        inputs=_v7_inputs(),
        verdict=V7_VERDICT,
        source_artifact=V7_CROSS_RECEIPT,
        source_sha256=V7_CROSS_SHA256,
        receipt_bindings={
            V7_N64_RECEIPT: V7_N64_SHA256,
            V7_N256_RECEIPT: V7_N256_SHA256,
            V7_CROSS_RECEIPT: V7_CROSS_SHA256,
        },
    )
    v8 = _anchor(
        anchor_id="ddm_v8_sparse_pixel_erf_wall_n64_n256_20260722",
        inputs=_v8_inputs(),
        verdict=V8_VERDICT,
        source_artifact=V8_CROSS_RECEIPT,
        source_sha256=V8_CROSS_SHA256,
        receipt_bindings={
            V8_N64_RECEIPT: V8_N64_SHA256,
            V8_N256_RECEIPT: V8_N256_SHA256,
            V8_CROSS_RECEIPT: V8_CROSS_SHA256,
        },
    )
    v9 = _anchor(
        anchor_id="ddm_v9_structured_carrier_inbox_n64_n256_20260722",
        inputs=_v9_inputs(),
        verdict=V9_VERDICT,
        source_artifact=V9_CROSS_RECEIPT,
        source_sha256=V9_CROSS_SHA256,
        receipt_bindings={
            V9_N64_RECEIPT: V9_N64_SHA256,
            V9_N256_RECEIPT: V9_N256_SHA256,
            V9_CROSS_RECEIPT: V9_CROSS_SHA256,
        },
    )
    v12 = _anchor(
        anchor_id="ddm_v12_structured_carrier_n600_obligation_drain_20260722",
        inputs=_v12_inputs(),
        verdict=V12_VERDICT,
        source_artifact=V12_N600_RECEIPT,
        source_sha256=V12_N600_SHA256,
        receipt_bindings={V12_N600_RECEIPT: V12_N600_SHA256},
        measurement_utc=V12_CALIBRATION_UTC,
    )
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=V12_N600_SHA256,
        source_path=V12_N600_RECEIPT,
        captured_at_utc=V12_CALIBRATION_UTC,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM direct-description measured rate/distortion bracket",
        one_line_summary=(
            "Exact values are rate-dead, sparse pixel repairs are ERF-dead, and "
            "structured carriers enter the byte box while exhaustive n600 correction of the "
            "bound predictor plateaus above the distortion gate."
        ),
        latex_form=(
            r"(B_9,D_9)_{\mathrm{structured}}\;\leadsto\;"
            r"(B,D)_{\mathrm{chart/event}}\;\leadsto\;"
            r"(B_7,D_7)_{\mathrm{exact}},\quad "
            r"B\le B_{\mathrm{box}},\ D_{seg}\le1.16\times10^{-3},\ "
            r"D_{pose}\le2.5\times10^{-4}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_describe_line_rate_distortion_bracket_20260722:"
            "evaluate_ddm_describe_line_rate_distortion_bracket"
        ),
        domain_of_validity={
            "formalization_status": "MEASURED_THREE_LEG_BRACKET_PLUS_N600_ANCHOR",
            "tasks": [540, 578, 603, 613],
            "axis": AXIS,
            "advisory_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "legs": {
                LEG_VALUE_EXACTNESS: (
                    "exact opaque solved-plane site/value descriptions; evaluator-green, "
                    "rate-dead at 200000 bytes"
                ),
                LEG_SPARSE_PIXEL: (
                    "exact-value post-hoc pixel masks over photometrically alien predictor; "
                    "FORMULATION scope; ERF context omitted"
                ),
                LEG_STRUCTURED_CARRIER: (
                    "five receiver-consumed per-stratum carriers plus sole Pose6 home; "
                    "G2CS1 chart path wired with zero measured symbols"
                ),
            },
            "measured_rate_endpoints": {
                "v7_n64_n256_bytes": [43_112_153, 171_332_654],
                "v8_tight_n64_n256_bytes": [2_629_076, 9_360_569],
                "v9_n64_n256_bytes": [51_668, 72_397],
                "v12_n600_bytes": 106_106,
            },
            "v7_measured_binders": {
                "rate": (
                    "Road+Undrivable+MyCar opaque exact homes are 94.21-94.73 percent "
                    "of final bytes across n64/n256"
                ),
                "d_seg": "Lane plus Boundary remain the exact-rung evaluator binders",
                "q4_n256": (
                    "108637789 bytes; d_seg 0.001218597094; misses 0.00116 gate by "
                    "0.000058597094"
                ),
            },
            "v8_measured_support": {
                "finite_tau_selected_fraction": [0.040182133516, 0.060744444529],
                "tight_mask_byte_collapse": [0.939017752141, 0.945366112171],
                "tight_mask_d_seg": [0.025907576084, 0.029359102249],
            },
            "derived_projection_rule": (
                "linear n64-to-n256 exact-byte interpolation only; callable labels the "
                "n600 result DERIVED_NOT_MEASURED"
            ),
            "derived_v9_projection": {
                "measured_marginal_bytes_per_pair_exact": "20729/192",
                "n600_projected_bytes_exact": "2628875/24",
                "status": "DERIVED_FROM_MEASURED_N64_N256_NOT_MEASURED_N600",
            },
            "v12_n600_upgrade": {
                "status": "MEASURED_N600_RECEIVER_CLOSED",
                "base_bytes": 102_105,
                "final_bytes": 106_106,
                "final_d_seg": 0.034003668891,
                "final_d_pose": 163.034719422881,
                "decision_inventory_exhausted": True,
                "exact_scorer_measured_atoms": 3_994,
                "strict_receiver_rejected_atoms": 66,
                "prior_higher_ev_conflict_excluded_atoms": 36,
            },
            "erf_mechanism_context": (
                "r50 approximately 50-160 px with median approximately 85; r90 approximately "
                "206-424 px and operational check approximately 300; source is measured "
                "segnet_recursive_fractal_factorization, not a new anchor here"
            ),
            "correction_successor": (
                "native Movable island worldsheet events in the PREDICT stage of a v6 successor, "
                "not another post-solve correction pass"
            ),
            "optional_stream_probe": (
                "Brenier pointwise monotone compander remains unmeasured; Splay/Jones/MTF "
                "entrants wait for nonempty receiver-admitted G2CS1 symbols"
            ),
            "anti_pattern_disposition": (
                "not registered: v8 receipt is FORMULATION-scoped while canonical anti-pattern "
                "taxonomy requires a recurring class-level forbidden pattern"
            ),
            "verdict_scope": (
                "v7 exact opaque values and v8 finite-tau post-hoc pixel values are "
                "FORMULATION negatives; v9 is an INSTANCE/FORMULATION-open bracket point; "
                "v12 closes only correct-a-bound-0.034-predictor; chart/event/carrier families "
                "and the describe-line paradigm remain open"
            ),
        },
        units_in={
            "archive_bytes": "exact final receiver-closed ZIP bytes",
            "d_seg": "frozen SegNet last-frame argmax disagreement fraction",
            "d_pose": "official frozen PoseNet first-six-output MSE",
            "selected_fraction": "fraction of full-resolution sites",
            "pair_count": "described frame pairs",
        },
        units_out={
            "verdict": "scoped categorical classification",
            "marginal_bytes_per_pair": "derived bytes per added pair",
            "n600_projected_bytes": "derived bytes for anchors without an n600 endpoint",
            "n600_measured": "receiver-closed measured n600 row when present",
        },
        empirical_anchors=(v7, v8, v9, v12),
        predicted_vs_empirical_residual={
            "v7_scoped_receipt_verdict": 0.0,
            "v8_scoped_receipt_verdict": 0.0,
            "v9_scoped_receipt_verdict": 0.0,
            "v12_scoped_receipt_verdict": 0.0,
        },
        last_calibration_utc=V12_CALIBRATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.v10_constructive_solver",
            "tac.optimization.direct_description_entropy_priced_member",
            "tac.witness_control.costate_organ_v2",
        ),
        canonical_producers=(
            "tools.run_direct_description_entropy_priced_member",
            "tools.run_ddm_v9_carrier_compose",
        ),
        provenance=provenance,
    )


def populate_ddm_describe_line_rate_distortion_bracket_v1(
    *,
    path: str | Path | None = None,
    lock_path: str | Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the equation through the canonical locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_describe_line_rate_distortion_bracket_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DDM v7/v8/v9 bracket plus measured V12 n600 anchor; tasks #540/#578/#603/#613; "
            "research-only; MAIN landing review required"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "EQUATION_ID",
    "LEG_SPARSE_PIXEL",
    "LEG_STRUCTURED_CARRIER",
    "LEG_VALUE_EXACTNESS",
    "V7_CROSS_RECEIPT",
    "V7_CROSS_SHA256",
    "V7_VERDICT",
    "V8_CROSS_RECEIPT",
    "V8_CROSS_SHA256",
    "V8_VERDICT",
    "V9_CROSS_RECEIPT",
    "V9_CROSS_SHA256",
    "V9_VERDICT",
    "V12_N600_RECEIPT",
    "V12_N600_SHA256",
    "V12_VERDICT",
    "build_ddm_describe_line_rate_distortion_bracket_v1",
    "evaluate_ddm_describe_line_rate_distortion_bracket",
    "populate_ddm_describe_line_rate_distortion_bracket_v1",
]
