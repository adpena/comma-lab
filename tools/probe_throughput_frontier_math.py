#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile Task #494 math laws against sealed n600 throughput receipts.

This probe is deliberately a receipt consumer.  It never loads a scorer,
reruns a settled n600 experiment, acquires Metal, or treats a synthetic fixture
as scientific evidence.  Missing host receipts remain explicit owed inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tac.local_acceleration.throughput_frontier_math import (
    PrecisionLayer,
    PrecisionOption,
    crt_reduction_certificate,
    fixed_width_reduction_certificate,
    number_system_disposition,
    solve_discrete_precision_waterfill,
    tropical_argmax_ordered_reduce,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "throughput_frontier_math_probe.v1"
STAGE_SCHEMA = "throughput_frontier_math_stage.v1"
EXPECTED_GT_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_SEGNET_WEIGHTS_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
FIXEDPOINT_PRODUCER_SOURCES = {
    "probe_sha256": REPO_ROOT / "tools/probe_fixedpoint_scorer_forward_n600.py",
    "module_sha256": REPO_ROOT / "src/tac/local_acceleration/calibrated_fixedpoint_scorer.py",
}
FULL_R_PRODUCER_SOURCES = {
    "probe": REPO_ROOT / "tools/probe_pythagorean_exact_arithmetic_bitident.py",
    "fused_r_reference": REPO_ROOT / "src/tac/local_acceleration/metal_fused_r_operator.py",
}
FULL_R_MEMBERS = ("gt_f0.npy", "gt_f1.npy")
FULL_R_VARIANTS = ("float_atomic", "fixed_q15_int32_atomic")
EXPECTED_FULL_R_STATIC_PROOF_SHA256 = "ebbaf1cb271674b8024d3eb7ea6193149ff8a41f2d8627f72ffa9d4d15d38379"
EXPECTED_FULL_R_ERROR_BOUND_SHA256 = "131b7a0b9340aae35aeb4cf24d47d6b79b5745335a062f85db5a8bd20d4ccacf"
HOST_COMMAND_SOURCE = REPO_ROOT / "tools/run_throughput_frontier_math_host.command"
STATIC_SCHEMAS = {
    "verdict_wallclock_n96": "frozen_scorer_verdict_wallclock.v1",
    "pythagorean_one_axis": "pythagorean_exact_arithmetic_bitident_probe.v1",
    "tile_halo_n600": "cheapen_real95_tile_halo_exactness.v1",
    "sparse_adjoint_n600": "p0_sparse_adjoint_costate_vjp.v1",
}
STAGE_FILENAMES = (
    "stage_00_custody.json",
    "stage_01_exact_number_system.json",
    "stage_02_argmax_certificate.json",
    "stage_03_discrete_waterfill.json",
    "stage_04_support_closure.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_json_bytes(raw: bytes, *, path: Path) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return _load_json_bytes(path.read_bytes(), path=path)


def _read_json_source(path: Path, *, required: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Hash and parse one immutable byte snapshot to avoid receipt TOCTOU."""

    descriptor: dict[str, Any] = {
        "path": str(path),
        "required": required,
        "exists": False,
    }
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if required:
            raise
        descriptor.update({"bytes": None, "sha256": None, "schema": None})
        return descriptor, None
    else:
        descriptor["exists"] = True
        try:
            payload = _load_json_bytes(raw, path=path)
        except (json.JSONDecodeError, ValueError) as error:
            if required:
                raise
            descriptor.update(
                {
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "schema": None,
                    "parse_error": type(error).__name__,
                }
            )
            # Optional host receipts must fail closed without preventing the
            # independent forward receipt from being synthesized.
            return descriptor, {"__parse_error__": type(error).__name__}
        descriptor.update(
            {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "schema": payload.get("schema", payload.get("schema_version", "UNKNOWN")),
            }
        )
        return descriptor, payload


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_static_sources(payloads: dict[str, dict[str, Any] | None]) -> None:
    for name, expected_schema in STATIC_SCHEMAS.items():
        payload = payloads[name]
        if payload is None or payload.get("schema") != expected_schema:
            raise ValueError(f"{name} must use schema {expected_schema}")

    verdict = payloads["verdict_wallclock_n96"]
    assert verdict is not None
    if (
        verdict.get("num_pairs") != 96
        or verdict.get("axis") != "[macOS-CPU-torch 1-thread advisory wall-clock] NON-PROMOTABLE"
        or verdict.get("torch_threads") != 1
        or verdict.get("score_claim") is not False
        or verdict.get("promotable") is not False
        or verdict.get("means_only") is not True
    ):
        raise ValueError("authority-verdict wall-clock receipt lacks measured n96 means custody")
    required_finite = (
        "combined_verdict_s_total",
        "per_pair_verdict_s",
        "seg_fraction_of_verdict",
        "pose_fraction_of_verdict",
        "extrapolated_n600_verdict_s",
        "extrapolated_n600_verdict_min",
    )
    if not all(
        isinstance(verdict.get(name), (int, float))
        and not isinstance(verdict.get(name), bool)
        and math.isfinite(float(verdict[name]))
        and float(verdict[name]) > 0.0
        for name in required_finite
    ):
        raise ValueError("authority-verdict wall-clock receipt contains non-positive timing/share fields")
    if not all(0.0 <= float(verdict[name]) <= 1.0 for name in ("seg_fraction_of_verdict", "pose_fraction_of_verdict")):
        raise ValueError("authority-verdict component shares must be probabilities")
    if not math.isclose(
        float(verdict["seg_fraction_of_verdict"]) + float(verdict["pose_fraction_of_verdict"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("authority-verdict component shares do not sum to one")
    if not math.isclose(
        float(verdict["extrapolated_n600_verdict_s"]),
        600.0 * float(verdict["per_pair_verdict_s"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        float(verdict["extrapolated_n600_verdict_min"]),
        float(verdict["extrapolated_n600_verdict_s"]) / 60.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("authority-verdict n600 projection is not derived from per-pair timing")

    pythagorean = payloads["pythagorean_one_axis"]
    assert pythagorean is not None
    if pythagorean.get("summary", {}).get("complete") is not True:
        raise ValueError("settled Pythagorean source is not complete")
    tile = payloads["tile_halo_n600"]
    assert tile is not None
    exact_tile = tile.get("exact_tile_contract")
    coverage = tile.get("n600_real_coverage")
    if not isinstance(exact_tile, dict) or (
        exact_tile.get("exact_dependency") != "FULL_FRAME_GLOBAL"
        or exact_tile.get("exact_source_area_fraction") != 1.0
        or exact_tile.get("ideal_exact_speedup_upper_bound") != 1.0
        or exact_tile.get("squeeze_excite_blocks") != 23
        or exact_tile.get("local_halo_px") != 685
        or exact_tile.get("local_receptive_field_px") != 1311
        or exact_tile.get("verdict") != "NO_GO"
        or not isinstance(exact_tile.get("verdict_scope"), str)
        or not exact_tile["verdict_scope"]
    ):
        raise ValueError("tile-halo source lacks the load-bearing full-frame closure contract")
    if not isinstance(coverage, dict) or coverage.get("n_pairs") != 600:
        raise ValueError("tile-halo source lacks exact n600 coverage")
    for name in ("boundary_area_fraction", "boundary_flip_mass_share"):
        value = coverage.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"tile-halo source has invalid {name}")
    if not _is_sha256(coverage.get("pair_ids_sha256")):
        raise ValueError("tile-halo source lacks pair-id custody")
    sparse = payloads["sparse_adjoint_n600"]
    assert sparse is not None
    if not isinstance(sparse.get("structural_exactness"), dict):
        raise ValueError("sparse-adjoint source lacks structural_exactness")


def _argmax_rows_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: int(item["pair_index"])):
        digest.update(f"{int(row['pair_index'])}:{row['candidate_argmax_sha256']}\n".encode("ascii"))
    return digest.hexdigest()


def _validate_fixedpoint_receipt(
    receipt: dict[str, Any] | None,
    *,
    producer_source_hashes: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Validate full-n600 QDQ custody without promoting it to integer authority."""

    if receipt is None:
        return {
            "status": "OWED_FIXEDPOINT_FORWARD_RECEIPT",
            "valid_full_n600": False,
            "admitted_arm": None,
            "failures": ["receipt missing"],
        }

    failures: list[str] = []
    contract = receipt.get("contract")
    custody = receipt.get("custody")
    summary = receipt.get("summary")
    arms = receipt.get("arms")
    fixedpoint_schema = receipt.get("schema")
    if fixedpoint_schema not in {
        "fixedpoint_scorer_forward_n600.v1",
        "fixedpoint_scorer_forward_n600.v2",
    }:
        failures.append("schema")
    if receipt.get("score_claim") is not False:
        failures.append("score_claim_false")
    if receipt.get("promotion_eligible") is not False:
        failures.append("promotion_eligible_false")
    if receipt.get("pointer_moved") is not False:
        failures.append("pointer_moved_false")
    if receipt.get("completed") is not True:
        failures.append("completed_true")
    if not isinstance(contract, dict):
        failures.append("contract_object")
        contract = {}
    if (
        contract.get("pair_start") != 0
        or contract.get("pair_count") != 600
        or contract.get("native_integer_speed_claim") is not False
        or contract.get("accumulation") != "QDQ emulation with fp32 Conv2d/Linear accumulation"
        or contract.get("calibration_split") != [0, 120]
        or contract.get("heldout_split") != [120, 600]
        or contract.get("threads") != {"interop": 1, "intraop": 1}
    ):
        failures.append("contract_full_n600_qdq_only")
    if fixedpoint_schema == "fixedpoint_scorer_forward_n600.v2" and (
        contract.get("activation_scale_mode") != "fixed_calibration"
        or contract.get("dynamic_scale_order_invariance") is not None
    ):
        failures.append("contract_v2_fixed_calibration_only")
    if not isinstance(custody, dict):
        failures.append("custody_object")
        custody = {}
    if custody.get("gt_cache_sha256") != EXPECTED_GT_CACHE_SHA256:
        failures.append("gt_cache_sha256")
    if custody.get("segnet_weights_sha256") != EXPECTED_SEGNET_WEIGHTS_SHA256:
        failures.append("segnet_weights_sha256")
    if producer_source_hashes is None:
        producer_source_hashes = {
            name: _sha256_file(path) if path.is_file() else None for name, path in FIXEDPOINT_PRODUCER_SOURCES.items()
        }
    for name in ("probe_sha256", "module_sha256"):
        if not _is_sha256(custody.get(name)):
            failures.append(name)
            continue
        if custody.get(name) != producer_source_hashes.get(name):
            failures.append(f"{name}_current_source_match")
    if not _is_sha256(custody.get("calibration_digest")):
        failures.append("calibration_digest")
    if not _is_sha256(receipt.get("fingerprint")):
        failures.append("contract_fingerprint")
    if not isinstance(summary, dict):
        failures.append("summary_object")
        summary = {}
    if summary.get("status") != "MEASURED" or summary.get("full_real_n600") is not True:
        failures.append("summary_measured_full_n600")
    if not isinstance(arms, dict):
        failures.append("arms_object")
        arms = {}
    declared_specs = contract.get("arms")
    if not isinstance(declared_specs, list) or not declared_specs:
        failures.append("contract_arms")
        declared_specs = []
    declared_names = [str(spec.get("name")) for spec in declared_specs if isinstance(spec, dict)]
    if len(declared_names) != len(declared_specs) or len(set(declared_names)) != len(declared_names):
        failures.append("contract_arm_names")
    spec_by_name = {str(spec.get("name")): spec for spec in declared_specs if isinstance(spec, dict)}
    if any(
        isinstance(spec.get("bits"), bool)
        or not isinstance(spec.get("bits"), int)
        or spec["bits"] < 1
        or not isinstance(spec.get("mixed_head_fp32"), bool)
        for spec in spec_by_name.values()
    ):
        failures.append("contract_arm_specs")
    fingerprint_specs = [
        spec for spec in declared_specs if isinstance(spec, dict) and spec.get("name") != "fp32_control"
    ]
    if len(fingerprint_specs) != len(declared_specs) - 1:
        failures.append("producer_contract_fp32_control_shape")
    else:
        # Producer v1 computes its fingerprint before prepending the fp32
        # control spec, then persists the mutated contract.  Reconstruct that
        # exact historical payload so live resumable receipts stay valid.
        fingerprint_contract = {**contract, "arms": fingerprint_specs}
        expected_fingerprint = _canonical_sha256({"contract": fingerprint_contract, "custody": custody})
        if receipt.get("fingerprint") != expected_fingerprint:
            failures.append("producer_contract_fingerprint")
    if set(arms) != set(declared_names):
        failures.append("arm_set")

    summary_arms = summary.get("arms")
    if not isinstance(summary_arms, dict):
        failures.append("summary_arms_object")
        summary_arms = {}
    arm_digests: dict[str, str] = {}
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name in declared_names:
        state = arms.get(name)
        arm_summary = summary_arms.get(name)
        if not isinstance(state, dict) or not isinstance(state.get("segnet_rows"), list):
            failures.append(f"{name}.segnet_rows")
            continue
        rows = state["segnet_rows"]
        indices = [row.get("pair_index") if isinstance(row, dict) else None for row in rows]
        hashes_valid = all(
            isinstance(row, dict)
            and isinstance(row.get("pair_index"), int)
            and not isinstance(row.get("pair_index"), bool)
            and _is_sha256(row.get("candidate_argmax_sha256"))
            and row.get("split") == ("calibration" if int(row["pair_index"]) < 120 else "heldout")
            and isinstance(row.get("flips"), int)
            and not isinstance(row.get("flips"), bool)
            and row["flips"] >= 0
            for row in rows
        )
        indices_valid = all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
        if len(rows) != 600 or not indices_valid or sorted(indices) != list(range(600)) or not hashes_valid:
            failures.append(f"{name}.exact_pair_rows")
            continue
        digest = _argmax_rows_digest(rows)
        arm_digests[name] = digest
        rows_by_name[name] = rows
        if not isinstance(arm_summary, dict) or arm_summary.get("status") != "MEASURED":
            failures.append(f"{name}.summary_measured")
            continue
        segnet = arm_summary.get("segnet")
        full = segnet.get("full") if isinstance(segnet, dict) else None
        heldout = segnet.get("heldout") if isinstance(segnet, dict) else None
        if (
            not isinstance(full, dict)
            or full.get("pairs") != 600
            or full.get("argmax_corpus_sha256") != digest
            or not isinstance(heldout, dict)
            or heldout.get("pairs") != 480
            or heldout.get("argmax_corpus_sha256")
            != _argmax_rows_digest([row for row in rows if int(row["pair_index"]) >= 120])
        ):
            failures.append(f"{name}.segnet_summary_custody")

    recomputed_quantized_admissions: list[tuple[int, bool, str]] = []
    control_rows = rows_by_name.get("fp32_control")
    if control_rows is None:
        failures.append("fp32_control_exact_rows")
        control_hashes: dict[int, str] = {}
    else:
        control_hashes = {int(row["pair_index"]): str(row["candidate_argmax_sha256"]) for row in control_rows}
    for name in declared_names:
        rows = rows_by_name.get(name)
        arm_summary = summary_arms.get(name)
        if rows is None or not isinstance(arm_summary, dict) or len(control_hashes) != 600:
            continue
        row_exactness = [str(row["candidate_argmax_sha256"]) == control_hashes[int(row["pair_index"])] for row in rows]
        row_flip_consistent = all(
            (int(row["flips"]) == 0) == exact for row, exact in zip(rows, row_exactness, strict=True)
        )
        if not row_flip_consistent:
            failures.append(f"{name}.row_flip_hash_consistency")
        full_exact = all(row_exactness)
        heldout_exact = all(
            exact for row, exact in zip(rows, row_exactness, strict=True) if int(row["pair_index"]) >= 120
        )
        segnet = arm_summary.get("segnet")
        full = segnet.get("full") if isinstance(segnet, dict) else None
        heldout = segnet.get("heldout") if isinstance(segnet, dict) else None
        if not isinstance(full, dict) or not isinstance(heldout, dict):
            continue
        full_flips = sum(int(row["flips"]) for row in rows)
        heldout_flips = sum(int(row["flips"]) for row in rows if int(row["pair_index"]) >= 120)
        if (
            full.get("flips") != full_flips
            or heldout.get("flips") != heldout_flips
            or full.get("argmax_exact_gate") is not full_exact
            or heldout.get("argmax_exact_gate") is not heldout_exact
        ):
            failures.append(f"{name}.summary_flip_hash_consistency")
        exact_admitted = full_exact and heldout_exact
        if arm_summary.get("argmax_exact_admitted") is not exact_admitted:
            failures.append(f"{name}.argmax_admission_consistency")
        spec = spec_by_name[name]
        spec_bits = spec.get("bits")
        spec_mixed = spec.get("mixed_head_fp32")
        if isinstance(spec_bits, bool) or not isinstance(spec_bits, int) or not isinstance(spec_mixed, bool):
            continue
        if name != "fp32_control" and spec_bits < 32 and exact_admitted:
            recomputed_quantized_admissions.append((spec_bits, spec_mixed, name))

    admitted = summary.get("minimum_argmax_exact_arm")
    recomputed_admitted = min(recomputed_quantized_admissions)[2] if recomputed_quantized_admissions else None
    if admitted != recomputed_admitted:
        failures.append("minimum_quantized_argmax_arm_consistency")
    expected_rung2_verdict = (
        "ARGMAX_FIXEDPOINT_FEASIBLE" if recomputed_admitted is not None else "NO_ADMITTED_PRECISION_IN_LADDER"
    )
    if summary.get("rung2_verdict") != expected_rung2_verdict:
        failures.append("rung2_verdict_consistency")

    if failures:
        completion_failure_names = {
            "completed_true",
            "summary_object",
            "summary_arms_object",
            "summary_measured_full_n600",
            "fp32_control_exact_rows",
            "minimum_quantized_argmax_arm_consistency",
            "rung2_verdict_consistency",
        }
        completion_only_failures = all(
            failure in completion_failure_names or failure.endswith(".exact_pair_rows") for failure in failures
        )
        every_arm_is_partial = bool(declared_names) and all(
            isinstance(arms.get(name), dict)
            and isinstance(arms[name].get("segnet_rows"), list)
            and len(arms[name]["segnet_rows"]) < 600
            for name in declared_names
        )
        partial_counts = {
            len(arms[name]["segnet_rows"])
            for name in declared_names
            if isinstance(arms.get(name), dict) and isinstance(arms[name].get("segnet_rows"), list)
        }
        partial_prefix_custody_valid = every_arm_is_partial and len(partial_counts) == 1
        if partial_prefix_custody_valid:
            expected_count = next(iter(partial_counts))
            for name in declared_names:
                rows = arms[name]["segnet_rows"]
                if len(rows) != expected_count:
                    partial_prefix_custody_valid = False
                    break
                for pair_index, row in enumerate(rows):
                    if (
                        not isinstance(row, dict)
                        or row.get("pair_index") != pair_index
                        or row.get("split") != ("calibration" if pair_index < 120 else "heldout")
                        or not _is_sha256(row.get("candidate_argmax_sha256"))
                        or isinstance(row.get("flips"), bool)
                        or not isinstance(row.get("flips"), int)
                        or row["flips"] < 0
                    ):
                        partial_prefix_custody_valid = False
                        break
                if not partial_prefix_custody_valid:
                    break
        partial_summary_valid = False
        if partial_prefix_custody_valid:
            expected_count = next(iter(partial_counts))
            partial_summary_valid = (
                summary.get("status") == "INCOMPLETE"
                and summary.get("full_real_n600") is False
                and summary.get("minimum_argmax_exact_arm") is None
                and summary.get("rung2_verdict") == "INCOMPLETE"
                and all(
                    isinstance(summary_arms.get(name), dict)
                    and summary_arms[name].get("status") == "INCOMPLETE"
                    and summary_arms[name].get("pairs") == expected_count
                    and summary_arms[name].get("unique_pair_indices") == expected_count
                    for name in declared_names
                )
            )
        every_arm_has_complete_rows = bool(declared_names) and all(
            isinstance(arms.get(name), dict)
            and isinstance(arms[name].get("segnet_rows"), list)
            and len(arms[name]["segnet_rows"]) == 600
            for name in declared_names
        )
        resumable_progress_shape = (
            partial_prefix_custody_valid and partial_summary_valid and completion_only_failures
        ) or (every_arm_has_complete_rows and set(failures) == {"completed_true"})
        completion_owed = (
            fixedpoint_schema
            in {
                "fixedpoint_scorer_forward_n600.v1",
                "fixedpoint_scorer_forward_n600.v2",
            }
            and (
                fixedpoint_schema == "fixedpoint_scorer_forward_n600.v1"
                or (
                    contract.get("activation_scale_mode") == "fixed_calibration"
                    and contract.get("dynamic_scale_order_invariance") is None
                )
            )
            and receipt.get("score_claim") is False
            and receipt.get("promotion_eligible") is False
            and receipt.get("pointer_moved") is False
            and receipt.get("completed") is not True
            and contract.get("pair_start") == 0
            and contract.get("pair_count") == 600
            and contract.get("native_integer_speed_claim") is False
            and contract.get("accumulation") == "QDQ emulation with fp32 Conv2d/Linear accumulation"
            and contract.get("calibration_split") == [0, 120]
            and contract.get("heldout_split") == [120, 600]
            and contract.get("threads") == {"interop": 1, "intraop": 1}
            and custody.get("gt_cache_sha256") == EXPECTED_GT_CACHE_SHA256
            and custody.get("segnet_weights_sha256") == EXPECTED_SEGNET_WEIGHTS_SHA256
            and all(_is_sha256(custody.get(name)) for name in ("probe_sha256", "module_sha256"))
            and _is_sha256(custody.get("calibration_digest"))
            and _is_sha256(receipt.get("fingerprint"))
            and isinstance(summary, dict)
            and isinstance(arms, dict)
            and bool(declared_names)
            and set(arms) == set(declared_names)
            and resumable_progress_shape
        )
        return {
            "status": (
                "OWED_FIXEDPOINT_FORWARD_COMPLETION"
                if completion_owed
                else "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"
            ),
            "valid_full_n600": False,
            "admitted_arm": None,
            "failures": sorted(set(failures)),
            "native_integer_speed_claim": False,
            "completed_pair_count_by_arm": {
                name: len(state.get("segnet_rows", []))
                for name, state in arms.items()
                if isinstance(state, dict) and isinstance(state.get("segnet_rows"), list)
            },
        }
    return {
        "status": (
            "FORWARD_QDQ_FEASIBILITY_ADMITTED__INTEGER_BACKEND_OWED"
            if recomputed_admitted is not None
            else "FORWARD_QDQ_FEASIBILITY_NO_ADMITTED_PRECISION__INTEGER_BACKEND_OWED"
        ),
        "valid_full_n600": True,
        "admitted_arm": recomputed_admitted,
        "argmax_corpus_sha256_by_arm": arm_digests,
        "native_integer_speed_claim": False,
        "pose_reported_separately": bool(contract.get("include_pose")),
        "failures": [],
    }


def _is_nonnegative_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _derive_full_r_authority_summary(rows: list[dict[str, Any]], *, derived_bound: float) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["pair_index"]),
            FULL_R_MEMBERS.index(str(row["member"])),
        ),
    )
    float_digest = hashlib.sha256()
    integer_digest = hashlib.sha256()
    for row in ordered:
        float_digest.update(f"{row['pair_index']}:{row['member']}:{row['float_output_sha256']}\n".encode("ascii"))
        integer_digest.update(f"{row['pair_index']}:{row['member']}:{row['integer_output_sha256']}\n".encode("ascii"))
    elements = sum(int(row["error_elements"]) for row in ordered)
    sum_squared = sum(float(row["sum_squared_error"]) for row in ordered)
    maximum = max((float(row["max_abs_error"]) for row in ordered), default=0.0)
    exact_coverage = len(ordered) == 1200 and all(
        (int(row["pair_index"]), str(row["member"]))
        == (index // len(FULL_R_MEMBERS), FULL_R_MEMBERS[index % len(FULL_R_MEMBERS)])
        for index, row in enumerate(ordered)
    )
    return {
        "status": "MEASURED" if len(ordered) == 1200 else "INCOMPLETE",
        "frames": len(ordered),
        "expected_frames": 1200,
        "coverage_exact": exact_coverage,
        "float_corpus_sha256": float_digest.hexdigest() if ordered else None,
        "integer_corpus_sha256": integer_digest.hexdigest() if ordered else None,
        "dequantized_max_abs_error_vs_numpy_fp32": maximum,
        "dequantized_rmse_vs_numpy_fp32": (math.sqrt(sum_squared / elements) if elements else None),
        "derived_max_abs_error_bound": derived_bound,
        "within_derived_bound": bool(len(ordered) == 1200 and maximum <= derived_bound),
    }


def _validate_full_r_numpy_state(
    rows: Any,
    *,
    static_integer_proof: list[dict[str, Any]],
    derived_bound: float,
) -> tuple[bool, dict[str, Any] | None]:
    if not isinstance(rows, list) or len(rows) > 1200:
        return False, None
    for index, row in enumerate(rows):
        expected_key = (
            index // len(FULL_R_MEMBERS),
            FULL_R_MEMBERS[index % len(FULL_R_MEMBERS)],
        )
        if (
            not isinstance(row, dict)
            or (
                row.get("pair_index"),
                row.get("member"),
            )
            != expected_key
        ):
            return False, None
        if not all(
            _is_sha256(row.get(field))
            for field in (
                "input_frame_sha256",
                "cotangent_sha256",
                "clip_mask_sha256",
                "float_output_sha256",
                "integer_output_sha256",
            )
        ):
            return False, None
        error_elements = row.get("error_elements")
        max_abs_error = row.get("max_abs_error")
        sum_squared_error = row.get("sum_squared_error")
        if (
            isinstance(error_elements, bool)
            or not isinstance(error_elements, int)
            or error_elements <= 0
            or not _is_nonnegative_finite_number(max_abs_error)
            or not _is_nonnegative_finite_number(sum_squared_error)
        ):
            return False, None
        squared_max = float(max_abs_error) ** 2
        if not math.isfinite(squared_max):
            return False, None
        squared_tolerance = 1e-9 * max(1.0, squared_max * error_elements, float(sum_squared_error))
        if (
            float(sum_squared_error) + squared_tolerance < squared_max
            or float(sum_squared_error) > squared_max * error_elements + squared_tolerance
        ):
            return False, None

        actual_bounds = row.get("stage_actual_max_sum_abs_contributions")
        actual_minimum_bits = row.get("stage_actual_minimum_signed_accumulator_bits")
        if (
            not isinstance(actual_bounds, list)
            or not isinstance(actual_minimum_bits, list)
            or len(actual_bounds) != len(static_integer_proof)
            or len(actual_minimum_bits) != len(static_integer_proof)
        ):
            return False, None
        for actual_bound, actual_bits, static_proof in zip(
            actual_bounds,
            actual_minimum_bits,
            static_integer_proof,
            strict=True,
        ):
            if (
                isinstance(actual_bound, bool)
                or not isinstance(actual_bound, int)
                or actual_bound < 0
                or isinstance(actual_bits, bool)
                or not isinstance(actual_bits, int)
            ):
                return False, None
            certificate = fixed_width_reduction_certificate(
                max_abs_term=actual_bound,
                fan_in=1,
                accumulator_bits=32,
            )
            if (
                actual_bits != certificate["minimum_signed_bits"]
                or certificate["no_overflow"] is not True
                or actual_bound > static_proof["max_abs_accumulator_bound"]
                or actual_bound > static_proof["int32_safe_limit_after_rounding_headroom"]
            ):
                return False, None
    return True, _derive_full_r_authority_summary(rows, derived_bound=derived_bound)


def _validate_full_r_trials(
    trials: Any,
    *,
    static_integer_proof: list[dict[str, Any]],
    integer_corpus_sha256: str,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    if not isinstance(trials, dict) or set(trials) != set(FULL_R_VARIANTS):
        return False, {}
    summaries: dict[str, dict[str, Any]] = {}
    prior_variant_terminal = True
    for variant in FULL_R_VARIANTS:
        rows = trials.get(variant)
        if not isinstance(rows, list) or len(rows) > 10:
            return False, {}
        if variant != FULL_R_VARIANTS[0] and rows and not prior_variant_terminal:
            return False, {}
        measured: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        for trial_index, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or row.get("variant") != variant
                or row.get("trial_index") != trial_index
                or isinstance(row.get("trial_index"), bool)
            ):
                return False, {}
            if row.get("status") == "BLOCKED_NOT_MEASURED":
                if (
                    trial_index != len(rows) - 1
                    or blockers
                    or not isinstance(row.get("blocker"), str)
                    or not row["blocker"]
                    or not isinstance(row.get("verdict_scope"), str)
                    or not row["verdict_scope"]
                ):
                    return False, {}
                blockers.append(row)
                continue
            static_proofs = row.get("static_overflow_proofs")
            if (
                row.get("status") != "MEASURED"
                or row.get("frames") != 1200
                or row.get("pairs") != 600
                or row.get("pair_start") != 0
                or row.get("members") != list(FULL_R_MEMBERS)
                or not _is_sha256(row.get("corpus_sha256"))
                or not isinstance(row.get("device"), str)
                or not row["device"]
                or not isinstance(row.get("mlx_version"), str)
                or not row["mlx_version"]
                or not _is_nonnegative_finite_number(row.get("elapsed_seconds"))
                or float(row["elapsed_seconds"]) <= 0.0
                or (variant == "fixed_q15_int32_atomic" and static_proofs != static_integer_proof)
                or (variant == "float_atomic" and static_proofs is not None)
            ):
                return False, {}
            measured.append(row)
        hashes = [str(row["corpus_sha256"]) for row in measured]
        entry: dict[str, Any] = {
            "attempts": len(rows),
            "measured_processes": len(measured),
            "expected_processes": 10,
            "all_full_coverage": bool(measured),
            "unique_corpus_hashes": len(set(hashes)),
            "cross_process_identical": bool(hashes and len(set(hashes)) == 1),
            "hashes": hashes,
            "blockers": blockers,
        }
        if variant == "fixed_q15_int32_atomic":
            entry["exact_numpy_int_corpus_parity"] = bool(
                hashes and all(value == integer_corpus_sha256 for value in hashes)
            )
        summaries[variant] = entry
        prior_variant_terminal = len(measured) == 10 or bool(blockers)
    return True, summaries


def _derive_full_r_summary(
    authority: dict[str, Any],
    variant_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    float_summary = variant_summaries["float_atomic"]
    fixed_summary = variant_summaries["fixed_q15_int32_atomic"]
    blocked = bool(float_summary["blockers"] or fixed_summary["blockers"])
    complete = bool(
        authority["status"] == "MEASURED"
        and authority["coverage_exact"]
        and float_summary["measured_processes"] == 10
        and fixed_summary["measured_processes"] == 10
    )
    float_diverges = bool(complete and float_summary["unique_corpus_hashes"] > 1)
    integer_holds = bool(
        complete
        and fixed_summary["cross_process_identical"]
        and fixed_summary["exact_numpy_int_corpus_parity"]
        and authority["within_derived_bound"]
    )
    if blocked:
        verdict = "BLOCKED_NOT_MEASURED"
        verdict_scope = "ENVIRONMENT: no evaluated Metal device in attempted child process"
    elif not complete:
        verdict = "INCOMPLETE"
        verdict_scope = "INSTANCE: full-R real-n600 receipt coverage/process count"
    elif float_diverges and integer_holds:
        verdict = "REAL-L70-LEVER-FULL-R-N600"
        verdict_scope = (
            "n600 INSTANCE: real 0.mkv gt_f0+gt_f1, four-axis render-R VJP, "
            "Q15/int32 atomic with Q7/Q5 state schedule, this MLX/Metal host"
        )
    elif not integer_holds:
        verdict = "FULL-R-INTEGER-FORMULATION-NO-GO"
        verdict_scope = "FORMULATION: Q15/int32 atomic plus Q7/Q5 boundary schedule; not the integer family"
    else:
        verdict = "FLOAT-WALL-NOT-REPRODUCED-FULL-R"
        verdict_scope = "INSTANCE: this full-R real-n600 corpus/host/process set"
    return {
        **variant_summaries,
        "authority": authority,
        "complete": complete,
        "decisive_positive": bool(float_diverges and integer_holds),
        "overall_verdict": verdict,
        "verdict_scope": verdict_scope,
    }


def _validate_full_r_receipt(
    receipt: dict[str, Any] | None,
    *,
    producer_source_hashes: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Validate every persisted state of the orthogonal full-R producer."""

    if receipt is None:
        return {
            "status": "OWED_FULL_R_TRAINING_REPRO_RECEIPT",
            "valid_contract": False,
            "complete": False,
            "failures": ["receipt missing"],
        }
    failures: list[str] = []
    contract = receipt.get("contract")
    custody = receipt.get("source_custody")
    if receipt.get("schema") != "pythagorean_exact_arithmetic_full_r_n600.v2":
        failures.append("schema")
    if (
        receipt.get("lane_id") != "throughput_authority_ladder"
        or receipt.get("task_id") != 494
        or receipt.get("axis") != "[macOS-MLX research-signal; NumPy-fp32/int32 authority; non-promotable MEANS]"
        or receipt.get("score_claim") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("pointer_moved") is not False
        or receipt.get("training") is not False
        or receipt.get("paid_dispatch") is not False
        or receipt.get("live_run_mutation") is not False
    ):
        failures.append("nonpromotable_means")
    if not isinstance(contract, dict):
        failures.append("contract_object")
        contract = {}
    if (
        contract.get("scope") != "full-r-n600"
        or contract.get("pair_start") != 0
        or contract.get("pair_count") != 600
        or contract.get("frames") != 1200
        or contract.get("members") != list(FULL_R_MEMBERS)
        or contract.get("n_processes_per_variant") != 10
        or contract.get("gt_cache_sha256") != EXPECTED_GT_CACHE_SHA256
        or not isinstance(contract.get("gt_cache"), str)
        or not contract["gt_cache"]
        or contract.get("q_weight_bits") != 15
        or contract.get("state_bits_by_boundary") != [7, 7, 7, 5, 5]
        or contract.get("signed_requantization")
        != "nearest; exact half away from zero; integer division; no signed shift"
        or contract.get("cotangent")
        != ("bilinear-down(real uint8 0.mkv frame,874x1164->384x512); clip(rint(value-127.5),-127,127)")
        or contract.get("chain")
        != [
            "down_w_transpose_512_to_1164",
            "down_h_transpose_384_to_874",
            "up_w_transpose_1164_to_512",
            "up_h_transpose_874_to_384",
        ]
    ):
        failures.append("contract_real_n600")
    if not isinstance(custody, dict):
        failures.append("source_custody_object")
        custody = {}
    if producer_source_hashes is None:
        producer_source_hashes = {
            name: _sha256_file(path) if path.is_file() else None for name, path in FULL_R_PRODUCER_SOURCES.items()
        }
    for name in FULL_R_PRODUCER_SOURCES:
        entry = custody.get(name)
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or not _is_sha256(entry.get("sha256"))
            or entry.get("sha256") != producer_source_hashes.get(name)
        ):
            failures.append(f"source_custody_{name}")
    gt_entry = custody.get("gt_cache")
    if (
        not isinstance(gt_entry, dict)
        or not isinstance(gt_entry.get("path"), str)
        or not gt_entry["path"]
        or gt_entry.get("sha256") != EXPECTED_GT_CACHE_SHA256
    ):
        failures.append("source_custody_gt_cache")
    if not _is_sha256(receipt.get("contract_fingerprint")):
        failures.append("contract_fingerprint")
    elif receipt.get("contract_fingerprint") != _canonical_sha256(contract):
        failures.append("producer_contract_fingerprint")

    static_integer_proof = receipt.get("static_integer_proof")
    derived_error_bound = receipt.get("derived_integer_error_bound")
    proof_valid = (
        isinstance(static_integer_proof, list)
        and _canonical_sha256(static_integer_proof) == EXPECTED_FULL_R_STATIC_PROOF_SHA256
        and isinstance(derived_error_bound, dict)
        and _canonical_sha256(derived_error_bound) == EXPECTED_FULL_R_ERROR_BOUND_SHA256
        and _is_nonnegative_finite_number(derived_error_bound.get("final_max_abs_error_bound"))
    )
    if not proof_valid:
        failures.append("static_integer_bound_contract")
        static_integer_proof = []
        derived_bound = 0.0
    else:
        derived_bound = float(derived_error_bound["final_max_abs_error_bound"])

    completed_raw = receipt.get("completed")
    if completed_raw is not None and not isinstance(completed_raw, bool):
        failures.append("completed_boolean")
    numpy_authority = receipt.get("numpy_authority")
    numpy_rows = numpy_authority.get("rows") if isinstance(numpy_authority, dict) else None
    numpy_valid, expected_authority = _validate_full_r_numpy_state(
        numpy_rows,
        static_integer_proof=static_integer_proof,
        derived_bound=derived_bound,
    )
    if not isinstance(numpy_authority, dict) or not numpy_valid or expected_authority is None:
        failures.append("numpy_authority_progress")
        expected_authority = {}
        numpy_rows = []
    else:
        recorded_authority = numpy_authority.get("summary")
        if numpy_rows and recorded_authority != expected_authority:
            failures.append("numpy_authority_summary")
        if not numpy_rows and recorded_authority is not None:
            failures.append("numpy_authority_summary")

    raw_summary = receipt.get("summary")
    trials = receipt.get("trials")
    if raw_summary is None:
        trials_empty = (
            isinstance(trials, dict)
            and set(trials) == set(FULL_R_VARIANTS)
            and all(trials[variant] == [] for variant in FULL_R_VARIANTS)
        )
        if completed_raw is True:
            failures.append("completed_summary_consistency")
        if not trials_empty:
            failures.append("trials_before_summary")
        if not failures:
            return {
                "status": "OWED_FULL_R_TRAINING_REPRO_COMPLETION",
                "valid_contract": True,
                "complete": False,
                "completed_numpy_authority_frames": len(numpy_rows),
                "failures": [],
            }
    elif not isinstance(raw_summary, dict):
        failures.append("summary_object")
    else:
        if len(numpy_rows) != 1200 or expected_authority.get("coverage_exact") is not True:
            failures.append("summary_requires_full_numpy_authority")
        trials_valid, variant_summaries = _validate_full_r_trials(
            trials,
            static_integer_proof=static_integer_proof,
            integer_corpus_sha256=str(expected_authority.get("integer_corpus_sha256", "")),
        )
        if not trials_valid:
            failures.append("trial_progress")
        elif expected_authority:
            expected_summary = _derive_full_r_summary(expected_authority, variant_summaries)
            if raw_summary != expected_summary:
                failures.append("summary_full_r_rederived_custody")
            if (completed_raw is True) is not (expected_summary["complete"] is True):
                failures.append("completed_summary_consistency")

    if failures:
        return {
            "status": "BLOCKED_INVALID_FULL_R_TRAINING_REPRO_RECEIPT",
            "valid_contract": False,
            "complete": False,
            "failures": sorted(set(failures)),
        }
    assert isinstance(raw_summary, dict)
    complete = raw_summary["complete"] is True
    return {
        "status": ("FULL_R_TRAINING_REPRO_MEASURED" if complete else "OWED_FULL_R_TRAINING_REPRO_COMPLETION"),
        "valid_contract": True,
        "complete": complete,
        "decisive_positive": bool(raw_summary["decisive_positive"]),
        "overall_verdict": raw_summary["overall_verdict"],
        "verdict_scope": raw_summary["verdict_scope"],
        "failures": [],
    }


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_stage(
    path: Path,
    *,
    stage: str,
    fingerprint: str,
    payload: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    payload_sha256 = _canonical_sha256(payload)
    envelope = {
        "schema": STAGE_SCHEMA,
        "stage": stage,
        "fingerprint": fingerprint,
        "payload_sha256": payload_sha256,
        "payload": payload,
    }
    if resume and path.is_file():
        existing = _load_json(path)
        if set(existing) != {"schema", "stage", "fingerprint", "payload_sha256", "payload"}:
            raise RuntimeError(f"resume stage envelope keys mismatch at {path}")
        if existing.get("schema") != STAGE_SCHEMA:
            raise RuntimeError(f"resume stage envelope schema mismatch at {path}")
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError(f"resume fingerprint drift at {path}")
        if existing.get("stage") != stage:
            raise RuntimeError(f"resume stage mismatch at {path}")
        if existing.get("payload_sha256") != _canonical_sha256(existing.get("payload")):
            raise RuntimeError(f"resume payload hash mismatch at {path}")
        if existing.get("payload_sha256") != payload_sha256:
            raise RuntimeError(f"resume deterministic-payload drift at {path}")
        return existing
    _atomic_write_json(path, envelope)
    return envelope


def _normalized_precision_allocation(receipt: dict[str, Any]) -> dict[str, Any]:
    """Consume only an explicit normalized bound/cost table; never guess fields."""

    manifest = receipt.get("frontier_math_precision_manifest")
    if not isinstance(manifest, dict):
        return {
            "status": "OWED_MEASURED_LAYER_BOUND_COST_TABLE",
            "reason": (
                "receipt lacks frontier_math_precision_manifest; QDQ max error and latency "
                "fields are not silently coerced into rigorous layer bounds"
            ),
        }
    if manifest.get("schema") != "throughput_frontier_math_precision_manifest.v1":
        raise ValueError("unsupported frontier_math_precision_manifest schema")
    bound_kind = manifest.get("bound_kind")
    if bound_kind not in {
        "rigorous_classwise_interval",
        "retrospective_n600_observed",
    }:
        raise ValueError("precision manifest must declare a supported bound_kind")
    if bound_kind == "rigorous_classwise_interval":
        return {
            "status": "OWED_EXTERNAL_RIGOROUS_BOUND_VALIDATOR",
            "bound_kind": bound_kind,
            "unseen_input_certificate": False,
            "reason": (
                "a bound_kind label is not a proof; this consumer has no independent interval "
                "propagation/proof-object validator"
            ),
        }
    layers_raw = manifest.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ValueError("precision manifest layers must be a non-empty list")
    error_budget = manifest.get("error_budget")
    if (
        isinstance(error_budget, bool)
        or not isinstance(error_budget, (int, float))
        or not math.isfinite(float(error_budget))
        or float(error_budget) < 0.0
    ):
        raise ValueError("precision manifest error_budget must be finite and non-negative")
    layers: list[PrecisionLayer] = []
    layer_names: set[str] = set()
    for layer in layers_raw:
        if not isinstance(layer, dict) or not isinstance(layer.get("options"), list):
            raise ValueError("each precision layer must declare options")
        name = layer.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("each precision layer name must be a non-empty string")
        if name in layer_names:
            raise ValueError("precision layer names must be unique")
        layer_names.add(name)
        layers.append(
            PrecisionLayer(
                name=name,
                options=tuple(
                    PrecisionOption(
                        bits=option["bits"],
                        error_bound=option["error_bound"],
                        measured_cost=option["measured_cost"],
                        mode=option.get("mode", "integer"),
                    )
                    for option in layer["options"]
                ),
            )
        )
    allocation = solve_discrete_precision_waterfill(tuple(layers), error_budget=float(error_budget))
    return {
        "status": "SOLVED_CORPUS_SCOPED",
        "bound_kind": bound_kind,
        "unseen_input_certificate": False,
        "error_budget": float(error_budget),
        "choices": [
            {"layer": layer.name, "bits": option.bits, "mode": option.mode}
            for layer, option in zip(layers, allocation.choices, strict=True)
        ],
        "total_error_bound": allocation.total_error_bound,
        "total_measured_cost": allocation.total_measured_cost,
        "pareto_state_count": allocation.pareto_state_count,
    }


def _assert_output_lifecycle(output_dir: Path, *, resume: bool) -> None:
    existing_stage_names = {path.name for path in output_dir.glob("stage_*.json") if path.is_file()}
    unexpected = existing_stage_names - set(STAGE_FILENAMES)
    if unexpected:
        raise RuntimeError(f"unexpected stale stage files: {sorted(unexpected)}")
    managed_existing = existing_stage_names | (
        {"measurement_receipt.json"} if (output_dir / "measurement_receipt.json").is_file() else set()
    )
    if managed_existing and not resume:
        raise RuntimeError("managed output already exists; use --resume or a fresh output directory")


def _write_final_receipt(path: Path, receipt: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    if resume and path.is_file():
        existing = _load_json(path)
        if existing.get("fingerprint") != receipt.get("fingerprint"):
            raise RuntimeError(f"resume final-receipt fingerprint drift at {path}")
        if _canonical_sha256(existing) != _canonical_sha256(receipt):
            raise RuntimeError(f"resume final-receipt deterministic-payload drift at {path}")
        return existing
    _atomic_write_json(path, receipt)
    return receipt


def _build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    source_specs = {
        "verdict_wallclock_n96": (args.verdict_wallclock_receipt, True),
        "pythagorean_one_axis": (args.pythagorean_receipt, True),
        "tile_halo_n600": (args.tile_halo_receipt, True),
        "sparse_adjoint_n600": (args.sparse_adjoint_receipt, True),
        "fixedpoint_forward_n600": (args.fixedpoint_receipt, False),
        "full_r_adjoint_n600": (args.full_r_receipt, False),
    }
    sources: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any] | None] = {}
    for name, (path, required) in source_specs.items():
        sources[name], payloads[name] = _read_json_source(path, required=required)
    _validate_static_sources(payloads)

    git_head = _git_head()
    fixedpoint_producer_hashes = {
        key: _sha256_file(path) if path.is_file() else None for key, path in FIXEDPOINT_PRODUCER_SOURCES.items()
    }
    full_r_producer_hashes = {
        key: _sha256_file(path) if path.is_file() else None for key, path in FULL_R_PRODUCER_SOURCES.items()
    }
    fingerprint_payload = {
        "schema": SCHEMA,
        "tool_sha256": _sha256_file(Path(__file__)),
        "math_module_sha256": _sha256_file(REPO_ROOT / "src/tac/local_acceleration/throughput_frontier_math.py"),
        "git_head": git_head,
        "n_pairs": args.n_pairs,
        # Paths are retained in the stage-00 custody payload, so the complete
        # immutable descriptors must participate in the content address too.
        # Otherwise identical bytes at different original paths alias and a
        # legitimate resume is rejected as deterministic payload drift.
        "sources": sources,
        "fixedpoint_producer_sources": fixedpoint_producer_hashes,
        "full_r_producer_sources": full_r_producer_hashes,
        "host_command_sha256": (_sha256_file(HOST_COMMAND_SOURCE) if HOST_COMMAND_SOURCE.is_file() else None),
    }
    fingerprint = _canonical_sha256(fingerprint_payload)
    explicit_output_dir = getattr(args, "output_dir", None)
    output_root = getattr(args, "output_root", None)
    if (explicit_output_dir is None) == (output_root is None):
        raise ValueError("provide exactly one of output_dir or output_root")
    output_dir = Path(explicit_output_dir) if explicit_output_dir is not None else Path(output_root) / fingerprint
    _assert_output_lifecycle(output_dir, resume=args.resume)
    verdict_wallclock = payloads["verdict_wallclock_n96"]
    pythagorean = payloads["pythagorean_one_axis"]
    tile = payloads["tile_halo_n600"]
    sparse = payloads["sparse_adjoint_n600"]
    fixedpoint = payloads["fixedpoint_forward_n600"]
    full_r = payloads["full_r_adjoint_n600"]
    assert verdict_wallclock is not None
    assert pythagorean is not None
    assert tile is not None
    assert sparse is not None
    fixedpoint_validation = _validate_fixedpoint_receipt(fixedpoint, producer_source_hashes=fixedpoint_producer_hashes)
    full_r_validation = _validate_full_r_receipt(full_r, producer_source_hashes=full_r_producer_hashes)

    custody_payload = {
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "pointer_moved": False,
        "n_pairs_required": args.n_pairs,
        "expected_gt_cache_sha256": EXPECTED_GT_CACHE_SHA256,
        "sources": sources,
        "required_static_sources_complete": all(value["exists"] for value in sources.values() if value["required"]),
        "fixedpoint_forward_validation": fixedpoint_validation,
        "full_r_training_repro_validation": full_r_validation,
        "full_r_is_orthogonal_to_forward_completion": True,
    }
    _write_stage(
        output_dir / "stage_00_custody.json",
        stage="00_custody",
        fingerprint=fingerprint,
        payload=custody_payload,
        resume=args.resume,
    )

    static_contract = pythagorean.get("numpy_static_contract", {})
    bound = static_contract.get("max_abs_integer_accumulator")
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
        raise ValueError("pythagorean receipt lacks a valid integer accumulator bound")
    fixed_certificate = fixed_width_reduction_certificate(max_abs_term=bound, fan_in=1, accumulator_bits=32)
    crt_certificate = crt_reduction_certificate(max_abs_term=bound, fan_in=1, moduli=(4093, 8191))
    exact_payload = {
        "source_verdict": pythagorean.get("summary", {}).get("overall_verdict"),
        "source_verdict_scope": pythagorean.get("summary", {}).get("verdict_scope"),
        "measured_sum_abs_bound": bound,
        "fixed_width_certificate": fixed_certificate,
        "crt_certificate": crt_certificate,
        "number_system_disposition": number_system_disposition(max_abs_term=bound, fan_in=1, crt_moduli=(4093, 8191)),
        "minimal_structure_claim": (
            "exact commutative accumulation for every admissible partial sum, no overflow "
            "or saturation, followed by one deterministic finalization"
        ),
        "scope_guard": (
            "this certifies reorder invariance of the declared quantized reduction only; "
            "NumPy-fp32 equivalence and argmax preservation require a separate error certificate"
        ),
    }
    _write_stage(
        output_dir / "stage_01_exact_number_system.json",
        stage="01_exact_number_system",
        fingerprint=fingerprint,
        payload=exact_payload,
        resume=args.resume,
    )

    tie_candidates = (7.0, 7.0, 3.0, -2.0)
    tie_results = {
        tropical_argmax_ordered_reduce(tie_candidates, order)
        for order in itertools.permutations(range(len(tie_candidates)))
    }
    argmax_payload = {
        "theorem": "winner a is certified iff L_a > max_{c!=a} U_c",
        "symmetric_uniform_corollary": "top1_minus_top2 > 2*epsilon",
        "strict_tie_policy": "equality is uncertified",
        "tropical_head_carrier": "lexicographic max over (value,-class_index)",
        "tropical_tie_fixture": {
            "candidates": list(tie_candidates),
            "permutations_checked": math.factorial(len(tie_candidates)),
            "unique_results": [list(result) for result in sorted(tie_results)],
            "canonical_tie_winner_is_smallest_class_index": tie_results == {(7.0, 0)},
            "evidence_scope": "algebraic unit fixture; not n600 scientific evidence",
        },
        "fixedpoint_receipt_present": fixedpoint is not None,
        "fixedpoint_validation": fixedpoint_validation,
        "n600_custody_observed": fixedpoint_validation["valid_full_n600"],
        "certificate_status": fixedpoint_validation["status"],
        "unseen_input_certificate": False,
        "scope_guard": (
            "an observed n600 maximum logit error can certify only the measured corpus/rung; "
            "it is not a sound unseen-input or per-layer interval bound"
        ),
        "synthetic_fixtures_are_scientific_evidence": False,
    }
    _write_stage(
        output_dir / "stage_02_argmax_certificate.json",
        stage="02_argmax_certificate",
        fingerprint=fingerprint,
        payload=argmax_payload,
        resume=args.resume,
    )

    waterfill_payload = (
        _normalized_precision_allocation(fixedpoint)
        if fixedpoint_validation["valid_full_n600"] and fixedpoint is not None
        else {
            "status": fixedpoint_validation["status"],
            "reason": "no valid full-n600 per-layer precision/error/cost frontier is available",
        }
    )
    waterfill_payload["continuous_law"] = (
        "min sum_l c_l*b_l subject to sum_l a_l*2^-b_l <= epsilon; interior b_l=log2(a_l*sum_j(c_j)/(epsilon*c_l))"
    )
    waterfill_payload["hardware_scope_guard"] = (
        "continuous bits are a lower bound/initializer; actual selection uses discrete measured costs"
    )
    _write_stage(
        output_dir / "stage_03_discrete_waterfill.json",
        stage="03_discrete_waterfill",
        fingerprint=fingerprint,
        payload=waterfill_payload,
        resume=args.resume,
    )

    exact_tile = tile.get("exact_tile_contract", {})
    n600 = tile.get("n600_real_coverage", {})
    structural = sparse.get("structural_exactness", {})
    support_payload = {
        "tile_receipt_n_pairs": n600.get("n_pairs"),
        "boundary_area_fraction_n600": n600.get("boundary_area_fraction"),
        "boundary_flip_mass_share_n600": n600.get("boundary_flip_mass_share"),
        "exact_dependency": exact_tile.get("exact_dependency"),
        "exact_source_area_fraction": exact_tile.get("exact_source_area_fraction"),
        "exact_forward_speedup_upper_bound": exact_tile.get("ideal_exact_speedup_upper_bound"),
        "global_squeeze_excite_count": exact_tile.get("squeeze_excite_blocks"),
        "exact_halo_pixels": exact_tile.get("local_halo_px"),
        "exact_sparse_backward_speedup": structural.get("exact_sparse_backward_speedup_x"),
        "verdict": "NO_GO_EXACT_SPATIAL_TEACHER_SKIP",
        "verdict_scope": exact_tile.get("verdict_scope"),
        "family_not_killed": (
            "separatrix sparsity remains admissible for cotangent sparsity, custom sparse "
            "adjoints, local/distilled students, or explicitly approximate cached-SE formulations"
        ),
        "n16_97pct_claim_promoted_to_n600": False,
    }
    _write_stage(
        output_dir / "stage_04_support_closure.json",
        stage="04_support_closure",
        fingerprint=fingerprint,
        payload=support_payload,
        resume=args.resume,
    )

    stage_paths = [output_dir / name for name in STAGE_FILENAMES]
    missing_stages = [path.name for path in stage_paths if not path.is_file()]
    if missing_stages:
        raise RuntimeError(f"managed stage set incomplete: {missing_stages}")
    receipt = {
        "schema": SCHEMA,
        "lane_id": "lane_throughput_frontier_math_20260713",
        "task_cluster": 494,
        "git_head_at_probe": git_head,
        "fingerprint": fingerprint,
        "content_addressed_output_dir": str(output_dir),
        "axis": "[research-only receipt synthesis; no Metal execution; no score authority]",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "pointer_moved": False,
        "training_performed": False,
        "paid_dispatch": False,
        "live_run_mutated": False,
        "n_pairs_required": args.n_pairs,
        "fixedpoint_forward_validation": fixedpoint_validation,
        "full_r_training_repro_validation": full_r_validation,
        "overall_status": fixedpoint_validation["status"],
        "integer_gpu_ane_backend_authority_complete": False,
        "integer_gpu_ane_backend_authority_owed": True,
        "stage_files": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size} for path in stage_paths
        },
        "ranked_build_next": [
            "BUILD/MEASURE calibrated fixed-point SegNet forward over real n600 with strict argmax certificates and synchronized GPU/ANE latency",
            "BUILD a deterministic tropical class-selection head only after upstream fixed-point logits carry a valid error certificate",
            (
                "BUILD/MEASURE exact-forward or distilled-forward alternatives against the "
                f"derived {float(verdict_wallclock['extrapolated_n600_verdict_min']):.6g}-minute "
                "n600 authority-verdict wall"
            ),
            (
                "BUILD a separate continuous-output PoseNet error/score certificate for the "
                f"measured {100.0 * float(verdict_wallclock['pose_fraction_of_verdict']):.6g}pct "
                "n96 verdict share"
            ),
            "KEEP full-R integer adjoint as training-reproducibility infrastructure, not the primary authority-verdict throughput lever",
        ],
        "single_highest_ev_measurement": (
            "real n600 calibrated fixed-point SegNet forward rung: aggregate and worst-pair "
            "argmax flips/uncertified fraction plus synchronized integer GPU/ANE residency and latency"
        ),
        "premise_status": (
            "CORRECTED_BY_MAIN_VERDICT_TIMER: authority verdict is forward-only. MEASURED n96 "
            f"combined={float(verdict_wallclock['combined_verdict_s_total']):.6g}s, "
            f"per_pair={float(verdict_wallclock['per_pair_verdict_s']):.6g}s, "
            f"SegNet share={float(verdict_wallclock['seg_fraction_of_verdict']):.6g}, "
            f"PoseNet share={float(verdict_wallclock['pose_fraction_of_verdict']):.6g}; "
            "DERIVED linear n600 extrapolation="
            f"{float(verdict_wallclock['extrapolated_n600_verdict_s']):.6g}s="
            f"{float(verdict_wallclock['extrapolated_n600_verdict_min']):.6g}min. "
            "The earlier fast in-loop MLX "
            "backward-heavy slice is not the throughput bottleneck and its routing is retracted."
        ),
        "measured_authority_verdict_axis": verdict_wallclock["axis"],
        "measured_authority_verdict_torch_threads": verdict_wallclock["torch_threads"],
        "measured_authority_verdict_n_pairs": verdict_wallclock["num_pairs"],
        "measured_authority_verdict_combined_seconds": verdict_wallclock["combined_verdict_s_total"],
        "measured_authority_verdict_seconds_per_pair": verdict_wallclock["per_pair_verdict_s"],
        "measured_authority_verdict_component_share_n96": {
            "segnet_forward": verdict_wallclock["seg_fraction_of_verdict"],
            "posenet_forward": verdict_wallclock["pose_fraction_of_verdict"],
        },
        "derived_linear_extrapolation_n600_seconds": verdict_wallclock["extrapolated_n600_verdict_s"],
        "derived_linear_extrapolation_n600_minutes": verdict_wallclock["extrapolated_n600_verdict_min"],
        "full_r_training_repro_status": full_r_validation["status"],
    }
    return _write_final_receipt(output_dir / "measurement_receipt.json", receipt, resume=args.resume)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verdict-wallclock-receipt",
        type=Path,
        default=REPO_ROOT / ".omx/research/frozen_scorer_verdict_wallclock_n96_20260714.json",
    )
    parser.add_argument(
        "--pythagorean-receipt",
        type=Path,
        default=REPO_ROOT / ".omx/research/pythagorean_exact_arithmetic_bitident_probe_20260713.json",
    )
    parser.add_argument("--tile-halo-receipt", type=Path, required=True)
    parser.add_argument("--sparse-adjoint-receipt", type=Path, required=True)
    parser.add_argument("--fixedpoint-receipt", type=Path, required=True)
    parser.add_argument("--full-r-receipt", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=600)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path)
    output.add_argument(
        "--output-root",
        type=Path,
        help="derive the run directory from the in-process immutable input fingerprint",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.n_pairs != 600:
        raise SystemExit("authority receipt synthesis requires --n-pairs 600")
    receipt = _build_receipt(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
