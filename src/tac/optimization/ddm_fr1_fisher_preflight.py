# SPDX-License-Identifier: MIT
"""Fail-closed custody preflight for the DDM FR1 Fisher actuator comparison.

The #583 ordering is an encoder-side ranking.  It does not, by itself, define
an executable corrected-inner-Jacobian perturbation.  This module verifies the
ordering, its correction-status record, and the independent base-curve
parents before any frozen-scorer work is allowed to start.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

import brotli

PREFLIGHT_SCHEMA: Final = "ddm_fr1_fisher_actuator_base_curves_preflight.v1"
CONFIG_SCHEMA: Final = "DDMFR1FisherActuatorBaseCurvesConfigV1"
ORDERING_SCHEMA: Final = "r1b5_fisher_ev_ordering_jsonl.v1"
INNER_STATUS_SCHEMA: Final = "m1_band_inner_jacobian_secant_qp_status.v1"
EXPECTED_COLUMNS: Final = (
    "pair",
    "row",
    "col",
    "linear_index",
    "target_class",
    "realized_class",
    "necessity_edge_tier",
    "resize_support_taps",
    "top1_top2_margin",
    "fisher_trace",
    "target_realized_head_pair_norm",
    "flip_distance",
    "vjp_native_arrangement_match",
    "vjp_local_lipschitz",
    "vjp_unit_pullback_rgb",
)
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


class FR1PreflightError(ValueError):
    """A required snapshot is malformed, drifted, or internally inconsistent."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve(repo_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=True)


def _read_bound(
    repo_root: Path,
    binding: dict[str, Any],
    *,
    label: str,
) -> tuple[Path, bytes]:
    if not {"path", "sha256"} <= set(binding):
        raise FR1PreflightError(f"{label} binding lacks path or sha256")
    path = _resolve(repo_root, str(binding["path"]))
    payload = path.read_bytes()
    actual = _sha256(payload)
    if actual != binding["sha256"]:
        raise FR1PreflightError(f"{label} sha256 drift: expected {binding['sha256']}, got {actual}")
    expected_bytes = binding.get("bytes")
    if expected_bytes is not None and len(payload) != int(expected_bytes):
        raise FR1PreflightError(f"{label} byte drift: expected {expected_bytes}, got {len(payload)}")
    return path, payload


def _json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FR1PreflightError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FR1PreflightError(f"{label} must be a JSON object")
    return value


def _bound_receipt(
    repo_root: Path,
    binding: dict[str, Any],
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path, payload = _read_bound(repo_root, binding, label=label)
    return path, _json(payload, label=label)


def _top_ordering_row(
    payload: bytes,
    *,
    candidate_count: int,
) -> tuple[dict[str, Any], int]:
    try:
        lines = brotli.decompress(payload).splitlines()
    except brotli.error as exc:
        raise FR1PreflightError("Fisher ordering is not valid Brotli") from exc
    if len(lines) != candidate_count + 1:
        raise FR1PreflightError(f"Fisher ordering line count {len(lines)} != {candidate_count + 1}")
    header = _json(lines[0], label="Fisher ordering header")
    if (
        header.get("schema") != ORDERING_SCHEMA
        or int(header.get("candidate_count", -1)) != candidate_count
        or tuple(header.get("columns", ())) != EXPECTED_COLUMNS
    ):
        raise FR1PreflightError("Fisher ordering header contract drifted")
    try:
        row = json.loads(lines[1])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FR1PreflightError("Fisher rank-1 row is not valid JSON") from exc
    if not isinstance(row, list) or len(row) != len(EXPECTED_COLUMNS):
        raise FR1PreflightError("Fisher rank-1 row geometry drifted")
    values = dict(zip(EXPECTED_COLUMNS, row, strict=True))
    target = int(values["target_class"])
    realized = int(values["realized_class"])
    if not 0 <= target < len(CLASS_NAMES) or not 0 <= realized < len(CLASS_NAMES):
        raise FR1PreflightError("Fisher rank-1 row class id escaped the palette")
    actuator_id = (
        f"pdw1_fisher_rank_00001_pair_{int(values['pair']):04d}"
        f"_cell_{int(values['row']):03d}_{int(values['col']):04d}"
        f"_{CLASS_NAMES[target].lower()}_from_{CLASS_NAMES[realized].lower()}"
    )
    values.update(
        {
            "actuator_id": actuator_id,
            "rank": 1,
            "target_class_name": CLASS_NAMES[target],
            "realized_class_name": CLASS_NAMES[realized],
            "status": "RANKED_CANDIDATE_NOT_EXECUTABLE_ACTUATOR",
        }
    )
    return values, len(lines) - 1


def _find_curve(receipt: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    rows = receipt.get("curve")
    if not isinstance(rows, list):
        raise FR1PreflightError("V19C receipt lacks a curve list")
    matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise FR1PreflightError(f"V19C receipt expected one {candidate_id!r} row, got {len(matches)}")
    return matches[0]


def _assert_numeric(row: dict[str, Any], expected: dict[str, Any], *, label: str) -> None:
    for key, value in expected.items():
        if row.get(key) != value:
            raise FR1PreflightError(f"{label}.{key} drift: expected {value!r}, got {row.get(key)!r}")


def build_preflight_receipt(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify all charter inputs and return a typed readiness/blocker receipt."""

    if config.get("schema") != CONFIG_SCHEMA:
        raise FR1PreflightError("FR1 config schema drifted")
    root = repo_root.resolve(strict=True)

    ordering_path, ordering_payload = _read_bound(root, config["ordering"], label="#583 Fisher ordering")
    top, observed_count = _top_ordering_row(
        ordering_payload,
        candidate_count=int(config["ordering"]["candidate_count"]),
    )
    fisher_receipt_path, fisher_receipt = _bound_receipt(root, config["fisher_receipt"], label="#583 Fisher receipt")
    if (
        fisher_receipt.get("schema") != "r1b5_fisher_ev_and_resize_coupling_audit.v1"
        or fisher_receipt.get("ordering_artifact", {}).get("sha256") != config["ordering"]["sha256"]
        or int(fisher_receipt.get("ranking", {}).get("candidate_count", -1)) != observed_count
    ):
        raise FR1PreflightError("#583 Fisher receipt does not bind the ordering")

    band_path, band = _bound_receipt(root, config["band_manifest"], label="M1 band manifest")
    record_binding = (
        band.get("custody", {}).get("ev_selection", {}).get("artifact_records", {}).get("inner_jacobian_secant_qp")
    )
    if not isinstance(record_binding, dict):
        raise FR1PreflightError("M1 band manifest lacks inner-Jacobian status binding")
    record_path = (band_path.parent / str(record_binding["path"])).resolve(strict=True)
    record_payload = record_path.read_bytes()
    if _sha256(record_payload) != record_binding.get("sha256"):
        raise FR1PreflightError("inner-Jacobian status binding drifted")
    inner = _json(record_payload, label="inner-Jacobian status")
    if inner.get("schema") != INNER_STATUS_SCHEMA:
        raise FR1PreflightError("inner-Jacobian status schema drifted")

    blockers: list[str] = []
    if inner.get("first_order_vjp") != "MEASURED_REAL_N600":
        blockers.append("FR1_FIRST_ORDER_VJP_N600_CUSTODY_ABSENT")
    if inner.get("realized_backbone_secants") != "MEASURED_RECEIVER_CLOSED":
        blockers.append("FR1_CORRECTED_INNER_JACOBIAN_REALIZED_SECANTS_ABSENT")
    if inner.get("qp_receiver_closure") != "MEASURED_RECEIVER_CLOSED":
        blockers.append("FR1_CORRECTED_INNER_JACOBIAN_QP_RECEIVER_CLOSURE_ABSENT")
    if inner.get("formalization") != "FORMALIZED_EXECUTABLE":
        blockers.append("FR1_CORRECTED_INNER_JACOBIAN_FORMALIZATION_PENDING")
    fisher_blockers = fisher_receipt.get("blockers", [])
    if "PER_CANDIDATE_EXACT_PREFIX_BYTE_MARGINAL_ABSENT" in fisher_blockers:
        blockers.append("FR1_PER_CANDIDATE_EXACT_PREFIX_BYTE_MARGINAL_ABSENT")

    runtime_path, runtime_payload = _read_bound(
        root, config["runtime_sensitivity"], label="DDM runtime sensitivity API"
    )
    runtime_text = runtime_payload.decode("utf-8")
    if "DDMRuntimePerturbationV1" not in runtime_text:
        raise FR1PreflightError("DDM runtime sensitivity API contract drifted")
    ordering_streams = {"pair", "row", "col", "vjp_unit_pullback_rgb"}
    runtime_streams = {
        "base/chart.anchors",
        "base/chart.gradients",
        "base/chart.residuals",
        "semantic/composed",
    }
    if ordering_streams.isdisjoint(runtime_streams):
        blockers.append("FR1_RANK1_TO_DDM_RUNTIME_PERTURBATION_BRIDGE_ABSENT")

    _, menu_config = _bound_receipt(root, config["v19c"]["menu_config"], label="V19C menu config")
    _, v19c_receipt = _bound_receipt(root, config["v19c"]["receipt"], label="V19C base receipt")
    _, v19c_archive = _read_bound(root, config["v19c"]["archive"], label="V19C base archive")
    v19c_row = _find_curve(v19c_receipt, str(config["v19c"]["candidate_id"]))
    _assert_numeric(v19c_row, config["v19c"]["expected_row"], label="V19C base row")
    if menu_config.get("v19c_archive_sha256") != _sha256(v19c_archive) or int(
        menu_config.get("v19c_archive_bytes", -1)
    ) != len(v19c_archive):
        raise FR1PreflightError("V19C menu config does not bind the base archive")

    _, ws1_receipt = _bound_receipt(root, config["ws1"]["receipt"], label="WS1 endpoint receipt")
    ws1_row = ws1_receipt.get("warm_start_candidates", {}).get("W_seg")
    if not isinstance(ws1_row, dict):
        raise FR1PreflightError("WS1 receipt lacks W_seg endpoint")
    _assert_numeric(ws1_row, config["ws1"]["expected_row"], label="WS1 W_seg row")
    recompile = ws1_receipt.get("seg_lexicographic_rerank", {}).get("receiver_recompile_status", {})
    ws1_endpoint_is_state = bool(ws1_row.get("archive_path") and ws1_row.get("archive_sha256"))

    ws2_paths = sorted(root.glob(str(config["ws2_receipt_glob"])))
    ws2_receipts = []
    for path in ws2_paths:
        payload = path.read_bytes()
        receipt = _json(payload, label=f"WS2 receipt {path}")
        wseg = receipt.get("warm_start_candidates", {}).get("W_seg", {})
        materialized = bool(isinstance(wseg, dict) and wseg.get("archive_path") and wseg.get("archive_sha256"))
        archive_valid = False
        archive_validation_error = None
        if materialized:
            try:
                archive_path = _resolve(root, str(wseg["archive_path"]))
                archive_payload = archive_path.read_bytes()
                if _sha256(archive_payload) != wseg["archive_sha256"]:
                    raise FR1PreflightError("materialized W_seg archive sha256 drifted")
                if wseg.get("archive_bytes") is not None and len(archive_payload) != int(wseg["archive_bytes"]):
                    raise FR1PreflightError("materialized W_seg archive bytes drifted")
                archive_valid = True
            except (FR1PreflightError, OSError, ValueError) as exc:
                archive_validation_error = str(exc)
        launchable = bool(
            receipt.get("execution_allowed") and receipt.get("score_claim") is False and materialized and archive_valid
        )
        ws2_receipts.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "materialized_W_seg": materialized,
                "archive_valid": archive_valid,
                "archive_validation_error": archive_validation_error,
                "launchable": launchable,
            }
        )
    ws2_launchable = sum(bool(row["launchable"]) for row in ws2_receipts)

    return {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "evidence_axis": config["evidence_axis"],
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "main_landing_review_required": True,
        "pointer": config["pointer"],
        "pointer_moved": False,
        "top_ranked_candidate": {
            **top,
            "ordering_path": str(ordering_path),
            "ordering_sha256": config["ordering"]["sha256"],
            "ordering_candidate_count": observed_count,
            "fisher_receipt_path": str(fisher_receipt_path),
            "corrected_inner_jacobian_executable": not blockers,
        },
        "corrected_inner_jacobian_custody": {
            "path": str(record_path),
            "sha256": record_binding["sha256"],
            **inner,
        },
        "runtime_bridge": {
            "path": str(runtime_path),
            "sha256": config["runtime_sensitivity"]["sha256"],
            "ordering_coordinate_domain": sorted(ordering_streams),
            "runtime_coordinate_domain": sorted(runtime_streams),
            "typed_bridge_present": False,
        },
        "base_curves": {
            "v19c_endpoint": {
                **v19c_row,
                "parent_archive_sha256": _sha256(v19c_archive),
                "parent_archive_bytes": len(v19c_archive),
                "state_materialized": True,
            },
            "ws1_W_seg": {
                **ws1_row,
                "endpoint_is_state": ws1_endpoint_is_state,
                "endpoint_state_caveat": (
                    "W_seg is a scored endpoint over the recompiled base; the receipt "
                    "does not bind a materialized W_seg archive."
                ),
                "recompiled_base_archive_sha256": recompile.get("archive_sha256"),
            },
            "ws2": {
                "observed_receipt_count": len(ws2_receipts),
                "launchable_materialized_W_seg_count": ws2_launchable,
                "receipts": ws2_receipts,
                "fallback": (
                    "MATERIALIZED_WS2_W_SEG" if ws2_launchable else "WS1_ENDPOINT_WITH_ENDPOINT_NOT_STATE_CAVEAT"
                ),
            },
        },
        "measurement": {
            "heavy_phase_started": False,
            "governor_memory_admission": "NOT_RUN_NO_HEAVY_PHASE",
            "rows": [],
            "delta_errors_per_class": "NOT_MEASURED",
            "delta_d_seg": "NOT_MEASURED",
            "delta_d_pose": "NOT_MEASURED",
            "delta_bytes": "NOT_MEASURED",
            "joint_delta_S": "NOT_MEASURED",
            "base_dependence_hypothesis": "NO_VERDICT_DATA_CUSTODY",
        },
        "blockers": blockers,
        "execution_allowed": not blockers,
        "verdict": (
            "READY_FOR_INDEPENDENT_BASE_CURVE_MEASUREMENT"
            if not blockers
            else "BLOCKED_PREMEASUREMENT_CORRECTED_INNER_JACOBIAN_ACTUATOR_NOT_EXECUTABLE"
        ),
        "verdict_scope": (
            "INSTANCE:CUSTODY of the #583 rank-1 candidate, corrected-inner-Jacobian "
            "receiver binding, and named V19C/WS1/WS2 parents only; no actuator, "
            "base-curve, formulation, family, or paradigm efficacy verdict"
        ),
    }


__all__ = [
    "CONFIG_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "FR1PreflightError",
    "build_preflight_receipt",
]
