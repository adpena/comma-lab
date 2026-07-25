# SPDX-License-Identifier: MIT
"""Fail-closed compiler for the DDM GA1 fiber-to-gauge pricing receipt.

GA1 may emit a measured conversion curve only when the sealed DR2b producer
contains a nonempty tolerance grid and the current C1 typed streams have a
same-object receiver/projector/uint8 crosswalk.  The current producers do not
meet those conditions.  This compiler preserves that absence as typed system
intelligence and derives the strongest lawful result still available: an
upper bound from the current LP1 FIBER byte mass.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = "ddm_ga1_gauge_tolerance_ladder_receipt.v1"
CONFIG_SCHEMA: Final = "DDMGA1GaugeToleranceLadderConfigV1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
RATE_SCORE_PER_BYTE: Final = 25.0 / 37_545_489.0

_REQUIRED_INPUTS: Final = frozenset(
    {
        "authority",
        "dr2b",
        "lp1",
        "rd1",
        "rd1_dimension_supplement",
        "resize_full_kernel",
        "uint8_lattice",
        "ms2r_r2",
    }
)


class GA1GaugeToleranceError(ValueError):
    """A GA1 source binding or accounting invariant differs."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for hashes and receipts."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Hash a file without loading large receipts into memory twice."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GA1GaugeToleranceError(message)


def _resolve(repository_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repository_root / path


def _display_path(repository_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repository_root))
    except ValueError:
        return str(path)


def _load_bound_input(
    repository_root: Path,
    binding: Mapping[str, Any],
    *,
    input_id: str,
) -> tuple[Path, bytes]:
    _require(set(binding) >= {"path", "bytes", "sha256"}, f"{input_id}: incomplete binding")
    path = _resolve(repository_root, str(binding["path"]))
    _require(path.is_file(), f"{input_id}: missing input {path}")
    expected_bytes = binding["bytes"]
    _require(type(expected_bytes) is int and expected_bytes >= 0, f"{input_id}: invalid byte count")
    _require(path.stat().st_size == expected_bytes, f"{input_id}: byte count drift")
    expected_sha = str(binding["sha256"])
    _require(re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None, f"{input_id}: invalid SHA-256")
    _require(sha256_file(path) == expected_sha, f"{input_id}: SHA-256 drift")
    return path, path.read_bytes()


def _json_input(
    repository_root: Path,
    binding: Mapping[str, Any],
    *,
    input_id: str,
) -> tuple[Path, dict[str, Any]]:
    path, payload = _load_bound_input(repository_root, binding, input_id=input_id)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise GA1GaugeToleranceError(f"{input_id}: invalid JSON") from exc
    _require(isinstance(value, dict), f"{input_id}: JSON root must be an object")
    expected_schema = binding.get("expected_schema")
    if expected_schema is not None:
        _require(value.get("schema") == expected_schema, f"{input_id}: schema drift")
    return path, value


def _parse_authority_claims(payload: bytes) -> dict[str, Any]:
    text = payload.decode("utf-8")
    exact = re.search(r"exact-solve \(([\d,]+) errors\)", text)
    box = re.search(r"box allowance \(([\d,]+) errors", text)
    falsifier = re.search(r"If <([0-9]+(?:\.[0-9]+)?)% of counted mass", text)
    _require(exact is not None, "authority: exact-solve error claim missing")
    _require(box is not None, "authority: box error claim missing")
    _require(falsifier is not None, "authority: falsifier threshold missing")
    return {
        "exact_error_claim": int(exact.group(1).replace(",", "")),
        "box_error_claim": int(box.group(1).replace(",", "")),
        "falsifier_fraction": float(falsifier.group(1)) / 100.0,
    }


def _current_lp1_fiber_mass(lp1: Mapping[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    waterfill = lp1.get("c1_corrected_waterfill")
    _require(isinstance(waterfill, dict), "LP1: c1_corrected_waterfill missing")
    total = waterfill.get("corrected_measured_allocated_bytes")
    _require(type(total) is int and total > 0, "LP1: corrected allocation missing")
    rows = waterfill.get("rows")
    _require(isinstance(rows, list), "LP1: C1 stream rows missing")

    fiber_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise GA1GaugeToleranceError("LP1: malformed C1 stream row")
        typed_home = row.get("typed_home")
        if (
            row.get("counted_in_corrected_total") is True
            and isinstance(typed_home, dict)
            and typed_home.get("type") == "FIBER"
        ):
            byte_count = row.get("corrected_allocated_bytes")
            _require(type(byte_count) is int and byte_count >= 0, "LP1: invalid FIBER bytes")
            fiber_rows.append(
                {
                    "stream": row.get("stream"),
                    "source_stratum": row.get("stratum"),
                    "counted_bytes": byte_count,
                    "typed_home": dict(typed_home),
                    "survival_authority": row.get("survival_authority"),
                }
            )
    _require(fiber_rows, "LP1: no currently allocated FIBER stream")
    return total, fiber_rows


def _rd1_exact_endpoint(rd1: Mapping[str, Any]) -> tuple[int, tuple[str, ...], Mapping[str, Any]]:
    anchors = rd1.get("anchors")
    _require(isinstance(anchors, dict), "RD1: anchors missing")
    exact = anchors.get("lambda_infinity_exact")
    _require(isinstance(exact, dict), "RD1: exact endpoint missing")
    per_class = exact.get("per_class")
    _require(isinstance(per_class, dict) and per_class, "RD1: exact per-class rows missing")
    errors: list[int] = []
    for class_name, row in per_class.items():
        _require(isinstance(class_name, str) and class_name, "RD1: invalid class identity")
        _require(isinstance(row, dict), f"RD1: malformed {class_name} row")
        error_count = row.get("errors")
        _require(type(error_count) is int and error_count >= 0, f"RD1: invalid {class_name} errors")
        errors.append(error_count)
    return sum(errors), tuple(sorted(per_class)), exact


def _input_custody(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    inputs = config["inputs"]
    for input_id in sorted(inputs):
        binding = inputs[input_id]
        rows.append(
            {
                "id": input_id,
                "path": binding["path"],
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
                "expected_schema": binding.get("expected_schema"),
                "freshness_at_consumption": "SHA_AND_BYTE_REVALIDATED",
            }
        )
    return rows


def compile_ga1_receipt(
    config_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Compile the current GA1 blocker and typed-mass upper-bound receipt."""

    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    try:
        config = json.loads(config_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise GA1GaugeToleranceError("GA1 config is invalid JSON") from exc
    _require(isinstance(config, dict), "GA1 config root must be an object")
    _require(config.get("schema") == CONFIG_SCHEMA, "GA1 config schema drift")
    _require(config.get("research_only") is True, "GA1 must remain research-only")
    _require(config.get("execution_allowed") is False, "GA1 execution authority must remain false")
    _require(config.get("score_claim") is False, "GA1 score claim must remain false")
    _require(config.get("promotion_eligible") is False, "GA1 promotion must remain false")
    inputs = config.get("inputs")
    _require(isinstance(inputs, dict), "GA1 input bindings missing")
    _require(set(inputs) == _REQUIRED_INPUTS, "GA1 input binding roster drift")

    _, authority_payload = _load_bound_input(repository_root, inputs["authority"], input_id="authority")
    authority_claims = _parse_authority_claims(authority_payload)
    _, dr2b = _json_input(repository_root, inputs["dr2b"], input_id="dr2b")
    _, lp1 = _json_input(repository_root, inputs["lp1"], input_id="lp1")
    _, rd1 = _json_input(repository_root, inputs["rd1"], input_id="rd1")
    _, rd1_dims = _json_input(
        repository_root,
        inputs["rd1_dimension_supplement"],
        input_id="rd1_dimension_supplement",
    )
    _, resize = _json_input(repository_root, inputs["resize_full_kernel"], input_id="resize_full_kernel")
    _, uint8 = _json_input(repository_root, inputs["uint8_lattice"], input_id="uint8_lattice")
    _, ms2r = _json_input(repository_root, inputs["ms2r_r2"], input_id="ms2r_r2")

    ladder = dr2b.get("u1_lossy_tolerance_ladder")
    rerace = dr2b.get("mode_at_tolerance_rerace")
    _require(isinstance(ladder, dict) and isinstance(rerace, dict), "DR2b ladder surfaces missing")
    priced_rungs = ladder.get("priced_sdwl1_rungs")
    lossy_rows = rerace.get("lossy_rows")
    _require(isinstance(priced_rungs, list), "DR2b priced rung field malformed")
    _require(isinstance(lossy_rows, list), "DR2b lossy row field malformed")

    total_counted_bytes, fiber_rows = _current_lp1_fiber_mass(lp1)
    fiber_bytes = sum(row["counted_bytes"] for row in fiber_rows)
    _require(0 < fiber_bytes <= total_counted_bytes, "LP1: FIBER mass is outside allocation")
    max_fraction = fiber_bytes / total_counted_bytes

    rd1_errors, class_names, exact_endpoint = _rd1_exact_endpoint(rd1)
    box_errors = ms2r.get("homotopy", {}).get("solve", {}).get("allowed_errors")
    _require(type(box_errors) is int and box_errors >= 0, "MS2R-R2: box endpoint missing")
    _require(
        box_errors == authority_claims["box_error_claim"],
        "MS2R-R2 box endpoint differs from delegated authority",
    )

    dimension_duals = rd1_dims.get("dimension_duals")
    effective_quantum = rd1_dims.get("effective_quantum_tolerance")
    _require(
        isinstance(dimension_duals, dict) and isinstance(effective_quantum, dict),
        "RD1 dimension supplement missing",
    )
    actionable_duals = dimension_duals.get("actionable_bucket_count")
    priced_quanta = effective_quantum.get("priced_bucket_count")
    _require(type(actionable_duals) is int and actionable_duals >= 0, "RD1 dual count invalid")
    _require(type(priced_quanta) is int and priced_quanta >= 0, "RD1 quantum count invalid")

    coverage = resize.get("coverage")
    reachability = resize.get("uint8_reachability")
    _require(isinstance(coverage, dict) and isinstance(reachability, dict), "#580 receipt incomplete")
    uint8_authority = uint8.get("authority")
    exact_search = uint8.get("aggregate", {}).get("exact_search")
    _require(isinstance(uint8_authority, dict) and isinstance(exact_search, dict), "#532 receipt incomplete")

    has_grid = bool(priced_rungs) and bool(lossy_rows)
    has_current_crosswalk = False
    curve_admitted = has_grid and has_current_crosswalk and actionable_duals > 0 and priced_quanta > 0
    _require(not curve_admitted, "GA1 current compiler unexpectedly reached curve admission")

    falsifier_fraction = authority_claims["falsifier_fraction"]
    dominated_by_upper_bound = max_fraction < falsifier_fraction
    _require(dominated_by_upper_bound, "GA1 typed-mass upper bound no longer triggers falsifier")

    sense_rows = []
    for class_name in class_names:
        for stream in fiber_rows:
            sense_rows.append(
                {
                    "schema": "ddm_ga1_costate_sense_row.v1",
                    "class_name": class_name,
                    "class_identity_source": (
                        "RD1 exact endpoint per_class key; no numeric scorer-channel index consumed"
                    ),
                    "stream": stream["stream"],
                    "source_stratum": stream["source_stratum"],
                    "shared_stream_counted_bytes": stream["counted_bytes"],
                    "not_additive_across_class_strata": True,
                    "tolerance_rung": None,
                    "bytes_convertible": None,
                    "realized_d_seg_cost": None,
                    "joint_exchange_rate": None,
                    "state": "SENSE_BLOCKED_MISSING_DR2B_GRID_AND_CURRENT_C1_PROJECTOR_CROSSWALK",
                }
            )

    blockers = [
        {
            "id": "DR2B_RUNG_GRID_ABSENT",
            "evidence": {
                "status": ladder.get("status"),
                "priced_sdwl1_rung_count": len(priced_rungs),
                "lossy_row_count": len(lossy_rows),
                "crosswalk_status": ladder.get("sdwl1_crosswalk", {}).get("status"),
            },
            "required_closure": (
                "append a SHA-bound nonempty DR2b tolerance grid and same-coordinate "
                "SDWL1-to-current-C1 receiver crosswalk; do not substitute a new grid"
            ),
        },
        {
            "id": "CURRENT_C1_STREAM_TO_PROJECTOR_CROSSWALK_ABSENT",
            "evidence": {
                "current_fiber_stream_count": len(fiber_rows),
                "current_fiber_bytes": fiber_bytes,
                "projector_class_or_margin_stratum": reachability.get("class_or_margin_stratum"),
            },
            "required_closure": (
                "measure each current C1 FIBER stream through #580 range/gauge projection, "
                "receiver parse-back, uint8 lattice realization, exact R, and argmax"
            ),
        },
        {
            "id": "RD1_TYPED_PRICES_UNACTIONABLE",
            "evidence": {
                "actionable_dimension_duals": actionable_duals,
                "priced_effective_quanta": priced_quanta,
                "dual_blocker": dimension_duals.get("blocker"),
                "quantum_blocker": effective_quantum.get("blocker"),
            },
            "required_closure": (
                "bind joint candidate-delta x G4-class x dimension-rate homes and "
                "per-dimension receiver uint8 absolute-step histograms"
            ),
        },
        {
            "id": "EXACT_ENDPOINT_SNAPSHOT_DRIFT",
            "evidence": {
                "delegated_authority_error_count": authority_claims["exact_error_claim"],
                "fresh_rd1_per_class_error_sum": rd1_errors,
                "absolute_difference": abs(authority_claims["exact_error_claim"] - rd1_errors),
            },
            "required_closure": "MAIN selects and SHA-pins one exact endpoint before curve measurement",
        },
    ]

    return {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "lane_id": "ddm_ga1_gauge_tolerance_ladder",
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": config["pointer"],
        "pointer_moved": False,
        "main_landing_review_required": True,
        "verdict": ("CURVE_BLOCKED_SOURCE_CUSTODY; CURRENT_C1_GAUGE_LEVER_DOMINATED_BY_TYPED_MASS_UPPER_BOUND"),
        "verdict_scope": (
            "INSTANCE: SHA-bound LP1 current 134,211-byte allocation and the sealed "
            "DR2b/RD1/#580/#532 producer state only; gauge families and future C1 "
            "stream retypings remain open"
        ),
        "typed_config": {
            "path": _display_path(repository_root, config_path),
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
            "schema": CONFIG_SCHEMA,
        },
        "input_custody": _input_custody(config),
        "source_state": {
            "dr2b_grid": {
                "admitted": has_grid,
                "priced_sdwl1_rung_count": len(priced_rungs),
                "lossy_row_count": len(lossy_rows),
                "status": ladder.get("status"),
            },
            "endpoint_custody": {
                "delegated_exact_errors": authority_claims["exact_error_claim"],
                "fresh_rd1_exact_errors": rd1_errors,
                "fresh_rd1_d_seg": exact_endpoint.get("d_seg"),
                "box_errors": box_errors,
                "exact_endpoint_consistent": authority_claims["exact_error_claim"] == rd1_errors,
            },
            "resize_full_kernel": {
                "full_nullity_fraction": coverage.get("full_nullity_fraction"),
                "uint8_feasible_basis_fraction_lower_bound": reachability.get("feasible_basis_fraction_lower_bound"),
                "is_fixture_only": True,
                "current_c1_stream_crosswalk": False,
            },
            "uint8_lattice": {
                "certified_exact_frames": exact_search.get("certified_exact_frames"),
                "subset_non_promotable": uint8_authority.get("subset_non_promotable"),
                "current_c1_stream_crosswalk": False,
            },
        },
        "curve_admission": {
            "admitted": curve_admitted,
            "curve_rows": [],
            "functional_form_status": "NOT_APPLICABLE_NO_MEASURED_CURVE",
            "measured_table_status": "NOT_EMITTED_MISSING_CANONICAL_RUNG_GRID",
            "blockers": blockers,
        },
        "current_c1_typed_mass_upper_bound": {
            "epistemic_status": "DERIVED_FROM_SHA_BOUND_LP1_ALLOCATION",
            "total_current_counted_bytes": total_counted_bytes,
            "eligible_current_fiber_bytes": fiber_bytes,
            "eligible_current_fiber_rows": fiber_rows,
            "maximum_convertible_bytes_even_if_all_fiber_is_gauge": fiber_bytes,
            "maximum_convertible_fraction_of_current_counted_mass": max_fraction,
            "maximum_rate_only_score_gain_if_zero_distortion_cost": (fiber_bytes * RATE_SCORE_PER_BYTE),
            "falsifier_threshold_fraction": falsifier_fraction,
            "falsifier_triggered_by_upper_bound": dominated_by_upper_bound,
            "disposition": "DOMINATED_INSTANCE_CURRENT_LP1_COMPOSITION",
        },
        "costate_sense": {
            "schema": "ddm_ga1_costate_sense.v1",
            "row_count": len(sense_rows),
            "rows": sense_rows,
            "allocation_bytes": 0,
            "state": "SENSE_BLOCKED_NO_CURVE",
        },
        "rd1_crosscheck": {
            "status": "NOT_RUN_NO_ADMITTED_CURVE",
            "actionable_dimension_duals": actionable_duals,
            "priced_effective_quanta": priced_quanta,
            "over_2x_disagreements": [],
            "pooled_aggregate_controls_not_transferred": True,
        },
        "equation_status": {
            "status": "NO_NEW_EQUATION_NO_MEASURED_CURVE",
            "lawful_bound": ("convertible_fraction <= current_LP1_FIBER_bytes / current_LP1_counted_bytes"),
        },
        "stores_consulted": config["stores_consulted"],
    }


__all__ = [
    "CONFIG_SCHEMA",
    "EVIDENCE_AXIS",
    "RATE_SCORE_PER_BYTE",
    "SCHEMA",
    "GA1GaugeToleranceError",
    "canonical_json_bytes",
    "compile_ga1_receipt",
    "sha256_file",
]
