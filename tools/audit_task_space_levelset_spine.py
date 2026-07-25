#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit the from-scratch task-space level-set constructive-inverse spine.

This tool composes already-landed measurement receipts.  It does not synthesize
an archive and it deliberately refuses to turn component measurements into a
full-partition or score claim.  Its job is to bind the source-only pedigree,
replay the Rust receiver parity gate, and state exactly which S0--S4 gates are
closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "task_space_levelset_constructive_spine_receipt.v1"
AXIS = "[macOS-CPU advisory]"
RATE_DENOMINATOR = 37_545_489
REPO = Path(__file__).resolve().parents[1]
DEFAULT_FRONTIER_POINTER = REPO / ".omx" / "state" / "canonical_frontier_pointer.json"


class SpineAuditError(RuntimeError):
    """A required custody or scientific contract is absent or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_json(payload_bytes: bytes, *, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpineAuditError(f"cannot load JSON receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SpineAuditError(f"receipt root must be an object: {path}")
    return payload


def load_json_snapshot(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload_bytes = path.read_bytes()
    except OSError as exc:
        raise SpineAuditError(f"cannot load JSON receipt {path}: {exc}") from exc
    return decode_json(payload_bytes, path=path), payload_bytes


def load_json(path: Path) -> dict[str, Any]:
    return load_json_snapshot(path)[0]


def load_effective_frontier(path: Path) -> dict[str, Any]:
    """Load the dynamic competitive target without implying archive custody."""

    pointer, pointer_bytes = load_json_snapshot(path)
    effective = pointer.get("effective_frontier")
    require(isinstance(effective, dict), "canonical pointer lacks effective_frontier")
    try:
        score = float(effective["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SpineAuditError("canonical pointer effective_frontier.score is invalid") from exc
    require(math.isfinite(score) and score > 0.0, "canonical effective frontier must be finite and positive")
    for field in ("axis", "custody", "evidence_grade", "source", "source_kind"):
        require(
            isinstance(effective.get(field), str) and bool(effective[field].strip()),
            f"canonical effective frontier lacks {field}",
        )
    return {
        "score": score,
        "axis": effective["axis"],
        "custody": effective["custody"],
        "evidence_grade": effective["evidence_grade"],
        "score_precision": effective.get("score_precision"),
        "source": effective["source"],
        "source_kind": effective["source_kind"],
        "snapshot_at_utc": effective.get("snapshot_at_utc"),
        "pointer_path": str(path),
        "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpineAuditError(message)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_rust_parity(runtime_rs: Path) -> dict[str, Any]:
    command = ["cargo", "test", "-p", "tac-levelset-inflate", "--", "--nocapture"]
    completed = subprocess.run(
        command,
        cwd=runtime_rs,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = completed.stdout + completed.stderr
    require(completed.returncode == 0, "Rust level-set receiver parity suite failed")
    require(
        "lane_coverage_parity ... ok" in output
        and "lane_coverage_negative_control_bit_flip_breaks_parity ... ok" in output,
        "Rust AA-SDF parity or its negative control did not execute",
    )
    require(
        "range_decode_parity ... ok" in output and "xi_column_delta_parity ... ok" in output,
        "Rust arithmetic/xi decoder parity did not execute",
    )
    crate = runtime_rs / "crates" / "tac-levelset-inflate"
    manifest = crate / "golden_vectors" / "levelset_lane_coverage_v1.json"
    fixture = crate / "golden_vectors" / "levelset_lane_coverage_v1_band.bin"
    rust_source = crate / "src" / "lane_coverage.rs"
    parity_test = crate / "tests" / "golden_vector_parity.rs"
    for path in (manifest, fixture, rust_source, parity_test):
        require(path.is_file(), f"Rust parity input missing: {path}")
    return {
        "status": "PASS",
        "command": command,
        "returncode": completed.returncode,
        "required_tests": [
            "lane_coverage_parity",
            "lane_coverage_negative_control_bit_flip_breaks_parity",
            "range_decode_parity",
            "xi_column_delta_parity",
        ],
        "sha256": {
            "golden_manifest": sha256_file(manifest),
            "golden_band_fixture": sha256_file(fixture),
            "rust_lane_coverage": sha256_file(rust_source),
            "golden_parity_test": sha256_file(parity_test),
        },
        "scope": "real n96 lane coefficients for AA-SDF golden parity; decoder primitive proof, not an n600 full-partition receiver proof",
    }


def _durable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def _identity_from_bytes(path: Path, payload_bytes: bytes) -> dict[str, Any]:
    return {
        "path": _durable_path(path),
        "bytes": len(payload_bytes),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }


def _snapshot_file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as exc:
        raise SpineAuditError(f"cannot snapshot artifact {path}: {exc}") from exc
    return {
        "path": _durable_path(path),
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _artifact(identity: dict[str, Any], lineage: str, use: str) -> dict[str, Any]:
    return {
        **identity,
        "content_lineage": lineage,
        "consumed_as": use,
    }


def _resolved_same(left: str, right: Path) -> bool:
    return Path(left).expanduser().resolve() == right.expanduser().resolve()


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    require(completed.returncode == 0, "cannot resolve git HEAD for receipt custody")
    value = completed.stdout.strip()
    require(len(value) == 40, "git HEAD is not a full SHA-1")
    return value


def build_receipt(
    *,
    source_video: Path,
    upstream_root: Path,
    gt_cache: Path,
    tie_receipt_path: Path,
    m2_receipt_path: Path,
    lane_receipt_path: Path,
    g1g3_receipt_path: Path,
    aa_receipt_path: Path,
    r1b7_receipt_path: Path,
    rust_parity: dict[str, Any],
    lane_id: str,
    frontier_pointer_path: Path = DEFAULT_FRONTIER_POINTER,
    s2_partition_seed_receipt_path: Path | None = None,
    s2_lane_curve_receipt_path: Path | None = None,
) -> dict[str, Any]:
    required_files = [
        source_video,
        gt_cache,
        tie_receipt_path,
        m2_receipt_path,
        lane_receipt_path,
        g1g3_receipt_path,
        aa_receipt_path,
        r1b7_receipt_path,
        frontier_pointer_path,
    ]
    if s2_partition_seed_receipt_path is not None:
        required_files.append(s2_partition_seed_receipt_path)
    if s2_lane_curve_receipt_path is not None:
        required_files.append(s2_lane_curve_receipt_path)
    modules = upstream_root / "modules.py"
    segnet = upstream_root / "models" / "segnet.safetensors"
    posenet = upstream_root / "models" / "posenet.safetensors"
    required_files.extend((modules, segnet, posenet))
    for path in required_files:
        require(path.is_file(), f"required input missing: {path}")

    source_video_identity = _snapshot_file_identity(source_video)
    gt_cache_identity = _snapshot_file_identity(gt_cache)
    modules_identity = _snapshot_file_identity(modules)
    segnet_identity = _snapshot_file_identity(segnet)
    posenet_identity = _snapshot_file_identity(posenet)
    cache_sha = gt_cache_identity["sha256"]
    tie, tie_bytes = load_json_snapshot(tie_receipt_path)
    m2, m2_bytes = load_json_snapshot(m2_receipt_path)
    lane, lane_bytes_snapshot = load_json_snapshot(lane_receipt_path)
    g1g3, g1g3_bytes = load_json_snapshot(g1g3_receipt_path)
    aa, aa_bytes = load_json_snapshot(aa_receipt_path)
    r1b7, r1b7_bytes = load_json_snapshot(r1b7_receipt_path)
    tie_identity = _identity_from_bytes(tie_receipt_path, tie_bytes)
    m2_identity = _identity_from_bytes(m2_receipt_path, m2_bytes)
    lane_identity = _identity_from_bytes(lane_receipt_path, lane_bytes_snapshot)
    g1g3_identity = _identity_from_bytes(g1g3_receipt_path, g1g3_bytes)
    aa_identity = _identity_from_bytes(aa_receipt_path, aa_bytes)
    r1b7_identity = _identity_from_bytes(r1b7_receipt_path, r1b7_bytes)
    effective_frontier = load_effective_frontier(frontier_pointer_path)
    if s2_partition_seed_receipt_path is None:
        finite_seed = None
        finite_seed_identity = None
    else:
        finite_seed, finite_seed_bytes = load_json_snapshot(
            s2_partition_seed_receipt_path
        )
        finite_seed_identity = _identity_from_bytes(
            s2_partition_seed_receipt_path,
            finite_seed_bytes,
        )
    if s2_lane_curve_receipt_path is None:
        lane_curve = None
        lane_curve_identity = None
    else:
        lane_curve, lane_curve_bytes = load_json_snapshot(s2_lane_curve_receipt_path)
        lane_curve_identity = _identity_from_bytes(
            s2_lane_curve_receipt_path,
            lane_curve_bytes,
        )

    require(tie.get("score_claim") is False, "tie-aware receipt crossed score firewall")
    fidelity = tie.get("input_fidelity", {})
    require(fidelity.get("n_pairs") == 600, "tie-aware receipt is not n600")
    require(fidelity.get("all_fp32_exact") is True, "canonical support-fill is not fp32 exact")
    require(fidelity.get("total_nonzero_values") == 0, "canonical support-fill has nonzero errors")
    require(float(fidelity.get("max_abs_over_pairs", -1.0)) == 0.0, "support-fill max error is nonzero")

    require(m2.get("schema") == "m2_live_target_selection_receipt.v1", "M2 schema drift")
    require(m2.get("score_claim") is False, "M2 receipt crossed score firewall")
    require(m2["source_custody"]["gt_cache_sha256"] == cache_sha, "M2 cache hash mismatch")
    require(m2["source_custody"]["pair_count"] == 600, "M2 receipt is not n600")
    require(float(m2["candidate"]["d_seg"]) == 0.0, "M2 direct row d_seg is not exact")
    require(float(m2["candidate"]["d_pose"]) == 0.0, "M2 direct row d_pose is not exact")
    require(
        m2["geometry_decomposition"]["resize_numerator_mismatches"] == 0,
        "M2 exact resize numerator gate failed",
    )

    require(lane.get("n_pairs") == 600, "lane descriptor receipt is not n600")
    require(_resolved_same(lane["gt_cache"], gt_cache), "lane receipt names a different GT cache")
    require(lane.get("correspondence_lossless_vs_sort") is True, "coherent slot codec is not lossless to fitted lanes")
    coherent = next(
        (row for row in lane.get("rows", []) if row.get("variant") == "coherent_slot_none"),
        None,
    )
    require(coherent is not None, "coherent_slot_none row missing")
    lane_bytes = int(coherent["brotli_bytes"])
    expected_rate = 25.0 * lane_bytes / RATE_DENOMINATOR
    require(abs(float(coherent["rate_term"]) - expected_rate) < 1e-15, "lane rate arithmetic drift")

    require(
        g1g3.get("schema") == "g1_worldsheet_g3_cellcode_measurements.v1",
        "G1/G3 schema drift",
    )
    require(g1g3.get("score_claim") is False, "G1/G3 receipt crossed score firewall")
    require(g1g3["cache"]["sha256"] == cache_sha, "G1/G3 cache hash mismatch")
    g1 = g1g3["g1"]
    g3 = g1g3["g3"]
    require(
        g1["verdict_scope"].startswith("single global ground-plane-homography"),
        "G1 formulation scope drift",
    )
    require(
        g3["alphabet"]["headers_and_finite_coder_overhead_included"] is False, "G3 unexpectedly claims finite coding"
    )
    require(g3["alphabet"]["site_location_cost_included"] is False, "G3 unexpectedly includes site locations")

    require(aa.get("n_pairs") == 600, "AA-SDF observation receipt is not n600")
    require("NON-PROMOTABLE" in aa.get("axis_tag", ""), "AA-SDF authority firewall drift")
    require(r1b7.get("schema") == "r1b7_uint8_survival_carrier_measurement.v1", "r1b7 schema drift")
    require(
        r1b7["integer_aware"]["search"]["new_hard_crossing_site_count"] == 0,
        "r1b7 adjudicated crossing count drift",
    )
    if finite_seed is not None:
        require(
            finite_seed.get("schema") == "s2_partition_seed_measurement.v1",
            "S2 finite partition seed schema drift",
        )
        require(finite_seed.get("score_claim") is False, "S2 finite seed crossed score firewall")
        require(finite_seed.get("n_pairs") == 600, "S2 finite seed is not n600")
        require(
            finite_seed["gt_cache"]["sha256"] == cache_sha,
            "S2 finite seed cache hash mismatch",
        )
        require(
            finite_seed["finite_packet"]["event_count"] == g3["all_flips"]["flip_count"],
            "S2 finite seed event count does not close the G3 inventory",
        )
        require(
            finite_seed["finite_packet"]["double_encode_byte_identical"] is True
            and finite_seed["finite_packet"]["parse_back_event_identity"] is True,
            "S2 finite seed lacks deterministic exact parse-back",
        )
        require(
            finite_seed["finite_packet"]["stored_plane_value_bytes"] == 0,
            "S2 finite seed improperly stores plane values",
        )
        require(
            finite_seed["semantic_detection"]["luma_consulted"] is False,
            "S2 finite seed used forbidden luma class sorting",
        )
        require(
            finite_seed["content_lineage"]["inherited_bytes_in_candidate"] == 0,
            "S2 finite seed imported inherited candidate bytes",
        )
        finite_packet_path = Path(finite_seed["finite_packet"]["path"])
        require(finite_packet_path.is_file(), "S2 finite packet bytes are not in custody")
        finite_packet_identity = _snapshot_file_identity(finite_packet_path)
        require(
            finite_packet_identity["bytes"] == finite_seed["finite_packet"]["packet_bytes"],
            "S2 finite packet byte count mismatch",
        )
        require(
            finite_packet_identity["sha256"] == finite_seed["finite_packet"]["packet_sha256"],
            "S2 finite packet SHA mismatch",
        )
    else:
        finite_packet_path = None
        finite_packet_identity = None
    if lane_curve is not None:
        require(lane_curve.get("schema") == "s2_lane_true_mask_curve.v1", "S2 Lane curve schema drift")
        require(lane_curve.get("score_claim") is False, "S2 Lane curve crossed score firewall")
        require(lane_curve["gt_cache"]["sha256"] == cache_sha, "S2 Lane curve cache hash mismatch")
        require(lane_curve["fit"]["n_pairs"] == 600, "S2 Lane curve is not n600")
        require(
            lane_curve["semantic_detection"]["luma_consulted"] is False,
            "S2 Lane curve used forbidden luma class sorting",
        )
        if finite_seed is not None:
            require(
                lane_curve["semantic_detection"]["semantic_class_ids"]
                == finite_seed["semantic_detection"]["semantic_class_ids"],
                "S2 Lane and finite-seed semantic detection disagree",
            )
        require(len(lane_curve["curve"]) >= 2, "S2 Lane curve lacks a finite comparison")
        require(
            lane_curve["curvelet_shearlet_residual"]["status"]
            == "BLOCKED_TARGET_BOUNDARY_INVERSE_CUSTODY",
            "S2 Lane curve improperly claims genuine residual closure",
        )
        require(
            lane_curve["content_lineage"]["inherited_bytes_in_candidate"] == 0,
            "S2 Lane curve imported inherited candidate bytes",
        )
        genuine_proof_path = Path(
            lane_curve["curvelet_shearlet_residual"]["genuine_structural_proof_path"]
        )
        if not genuine_proof_path.is_absolute():
            genuine_proof_path = REPO / genuine_proof_path
        require(genuine_proof_path.is_file(), "genuine frame structural proof is absent")
        genuine_proof_identity = _snapshot_file_identity(genuine_proof_path)
        require(
            genuine_proof_identity["sha256"]
            == lane_curve["curvelet_shearlet_residual"][
                "genuine_structural_proof_sha256"
            ],
            "genuine frame structural proof SHA mismatch",
        )
    else:
        genuine_proof_identity = None
    require(rust_parity.get("status") == "PASS", "Rust parity attestation is not PASS")

    source_custody = {
        "video": _artifact(
            source_video_identity,
            "source-video-derived",
            "S0 sole video input",
        ),
        "frozen_upstream": [
            _artifact(
                modules_identity,
                "upstream-frozen-scorer-code",
                "S0 frozen preprocessing/model definitions",
            ),
            _artifact(
                segnet_identity,
                "upstream-frozen-scorer-weights",
                "S0 SegNet authority",
            ),
            _artifact(
                posenet_identity,
                "upstream-frozen-scorer-weights",
                "S0 PoseNet authority",
            ),
        ],
        "target_cache": _artifact(
            gt_cache_identity,
            "source-video-derived our-build",
            "S0 n600 target custody",
        ),
        "cache_binding": {
            "m2_sha_match": True,
            "g1g3_sha_match": True,
            "lane_path_match": True,
            "fresh_full_cache_rebuild_this_pass": False,
            "scope": "existing canonical cache re-hashed and cross-bound; build log does not itself pin source/module hashes",
        },
    }

    consumed = [
        _artifact(
            tie_identity,
            "our support-fill solve on the exact-plane receiver; law-only consumption",
            "S1 canonical support-fill exactness; no receiver/archive bytes consumed",
        ),
        _artifact(m2_identity, "source-video-derived our-solve", "S1 source-target exact direct row"),
        _artifact(lane_identity, "source-video-derived our-solve", "S2 coherent lane-chart rate"),
        _artifact(
            g1g3_identity,
            "source-video-derived our-measurement",
            "S2 transport and cell-code priors",
        ),
        _artifact(
            aa_identity,
            "source-video-derived our-measurement",
            "S2 n600 AA-SDF fidelity curve",
        ),
        _artifact(
            r1b7_identity,
            "mixed inherited-base experiment; law-only consumption",
            "S3 fixed-magnitude/no-sub-step lesson only; all archive/payload bytes excluded",
        ),
    ]
    if finite_seed is not None and s2_partition_seed_receipt_path is not None:
        if finite_seed_identity is None:  # pragma: no cover - guarded above
            raise SpineAuditError("S2 finite seed receipt identity was not resolved")
        consumed.append(
            _artifact(
                finite_seed_identity,
                "source-video-derived our finite coder",
                "S2 counted G3 site and cell-identity seed with exact parse-back",
            )
        )
        if finite_packet_path is None or finite_packet_identity is None:  # pragma: no cover
            raise SpineAuditError("S2 finite packet custody was not resolved")
        consumed.append(
            _artifact(
                finite_packet_identity,
                "source-video-derived our finite coder",
                "S2 exact counted packet bytes",
            )
        )
    if lane_curve is not None and s2_lane_curve_receipt_path is not None:
        if lane_curve_identity is None:  # pragma: no cover - guarded above
            raise SpineAuditError("S2 Lane curve receipt identity was not resolved")
        consumed.append(
            _artifact(
                lane_curve_identity,
                "source-video-derived our Lane-chart measurement",
                "S2 finite Lane chart bytes versus true-mask fidelity curve",
            )
        )
        if genuine_proof_identity is None:  # pragma: no cover - guarded above
            raise SpineAuditError("S2 genuine structural proof identity was not resolved")
        consumed.append(
            _artifact(
                genuine_proof_identity,
                "our structural law proof; no candidate bytes consumed",
                "S2 genuine directional-frame implementation custody",
            )
        )

    g1_transition = g1["aggregate"]["by_transition"]
    g3_best = g3["best_measured_prior"]
    g3_bytes = float(g3["all_flips"]["ideal_bytes"][g3_best])
    stages = {
        "S0_true_targets": {
            "status": "VERIFIED_EXISTING_N600_SOURCE_DERIVATION",
            "pair_count": 600,
            "only_semantic_inputs": ["upstream/videos/0.mkv", "upstream/modules.py", "frozen scorer weights"],
            "note": source_custody["cache_binding"]["scope"],
        },
        "S1_canonical_support_fill_and_exact_direct_row": {
            "status": "MEASURED_TWO_DISTINCT_EXACTNESS_ROWS",
            "canonical_support_fill": {
                "target": "rounded uint8 scorer plane Y",
                "fp32_nonzero_values": 0,
                "fp32_total_values": 600 * 384 * 512,
                "max_abs": 0.0,
                "warning": "the receipt verdict prose contradicts its exact numeric fields; numeric fields were re-derived and used",
            },
            "source_target_direct_realization": {
                "archive_bytes": int(m2["candidate"]["archive_bytes"]),
                "d_seg": 0.0,
                "d_pose": 0.0,
                "score_claim": False,
                "scope": "official macOS-CPU advisory row; direct per-camera realization, not a compact descriptor",
            },
        },
        "S2_task_space_level_set_witness": {
            "status": "PARTIAL_MEASURED_NOT_PARTITION_POSE_COMPLETE",
            "division_of_labor": {
                "shape": "level-set charts describe the five-class partition; lane polynomial is the lane SDF zero-set",
                "values": "inverse preimage pins scorer-bearing values inside the selected cells",
                "realization": "S3 must jointly preserve chart and values on the uint8 lattice",
            },
            "lane_chart": {
                "coherent_slot_brotli_bytes": lane_bytes,
                "rate_term": expected_rate,
                "fitted_lines": int(lane["fit_stats"]["total_lines"]),
                "band_recall_mean": float(lane["fit_stats"]["band_recall_mean"]),
                "fidelity_scope": "lossless to fitted lane parameters, not lossless to the Lane argmax mask",
                "true_mask_curve": (
                    {
                        "status": "MEASURED_POLYNOMIAL_CONTROL_GENUINE_RESIDUAL_OPEN",
                        "rows": lane_curve["curve"],
                        "residual_status": lane_curve["curvelet_shearlet_residual"][
                            "status"
                        ],
                        "scope": lane_curve["verdict_scope"],
                    }
                    if lane_curve is not None
                    else {"status": "NOT_PROVIDED"}
                ),
            },
            "worldsheet_transport": {
                "equation_id": "worldsheet_transport_residual_event_rate_v1",
                "within_pair_median_px": float(g1_transition["within_pair"]["median_of_transition_medians_px"]),
                "cross_pair_median_px": float(g1_transition["cross_pair"]["median_of_transition_medians_px"]),
                "within_pair_event_gt4_fraction": float(g1_transition["within_pair"]["event_fraction_gt_px"]["4"]),
                "cross_pair_event_gt4_fraction": float(g1_transition["cross_pair"]["event_fraction_gt_px"]["4"]),
                "formulation_caveat": g1["verdict_scope"],
            },
            "precision_waterfill": {
                "equation_ids": [
                    "frozen_scorer_fisher_curvature_margin_colocation_v1",
                    "fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
                    "segnet_head_rank4_linear_flipdist_v1",
                    "witness_measured_reverse_waterfill_v1",
                ],
                "status": "REGISTERED_LAWS_NOT_EXECUTED_AS_A_COMPLETE_S2_BIT_ALLOCATOR_IN_THIS_ROW",
            },
            "morse_smale_cell_prior": {
                "equation_id": "argmax_cell_identity_ideal_bytes_v1",
                "flip_count": int(g3["all_flips"]["flip_count"]),
                "best_prior": g3_best,
                "ideal_bytes": g3_bytes,
                "not_counted": ["site locations", "headers", "finite coder overhead"],
                "fidelity_scope": "ideal conditional cell-identity floor, not a receiver-closed stream",
                "finite_seed": (
                    {
                        "status": "MEASURED_FINITE_PARSEBACK_COMPLETE",
                        "packet_bytes": int(finite_seed["finite_packet"]["packet_bytes"]),
                        "packet_sha256": finite_seed["finite_packet"]["packet_sha256"],
                        "event_count": int(finite_seed["finite_packet"]["event_count"]),
                        "sites_headers_coder_crc_counted": True,
                        "stored_plane_value_bytes": 0,
                        "semantic_class_ids": finite_seed["semantic_detection"][
                            "semantic_class_ids"
                        ],
                        "self_detection_method": finite_seed["semantic_detection"]["method"],
                        "scope": finite_seed["verdict_scope"],
                    }
                    if finite_seed is not None
                    else {"status": "NOT_PROVIDED"}
                ),
            },
            "aa_sdf_renderer": {
                "equation_id": "aa_sdf_observation_footprint_render_dseg_v1",
                "n600_grid384_d_seg": float(aa["render_grid_curve"]["384"]["real"]["aa"]["d_seg"]["mean"]),
                "rust_oracle_parity": rust_parity,
            },
            "curvelet_residual_chart": {
                "status": "REQUIRED_NOT_COMPOSED",
                "equation_id": "shearlet_nterm_upper_bounds_task_rate_v1",
                "scope": "use only where polynomial/ground charts leave measured residual; no Fourier candidate basis",
            },
            "range_and_blind_geometry": {
                "full_linear_nullity_fraction": float(m2["geometry_decomposition"]["full_linear_nullity_fraction"]),
                "implemented_integer_exact_null_mask_fraction": float(
                    m2["geometry_decomposition"]["implemented_integer_exact_null_mask_fraction"]
                ),
                "blind_coordinates_per_frame": 230_904,
                "blind_fraction": 0.22696926089315625,
                "scope": "generic fill is free only for a camera-resolution payload; direct saving is zero for a pure generator",
            },
            "missing_closure": [
                "baseline five-class chart/predictor composed with the measured finite cell-event seed",
                "counted pose/xi stream bound to the same n600 partition row",
                "curvelet residual chart composed where polynomial/ground charts fail",
                "one parse-back that combines charts, cell events, values, range(A), and generic blind fill",
                "n600 through-R hard-oracle fidelity for that combined descriptor",
            ],
        },
        "S3_integer_aware_realization": {
            "status": "COMPONENTS_PRESENT_NOT_COMPOSED_WITH_S2",
            "lattice_components": [
                "tac.optimization.uint8_lattice_feasibility",
                "tac.optimization.tie_aware_preimage",
            ],
            "fixed_magnitude_lesson": {
                "receipt_verdict": r1b7["verdict"],
                "new_hard_crossings": 0,
                "scope": r1b7["verdict_scope"],
                "candidate_bytes_consumed": False,
            },
            "gate": "no sub-step writes; admit only fixed-magnitude integer writes through a fresh hard oracle",
        },
        "S4_strict_archive_n600_receiver": {
            "status": "NOT_BUILT_FOR_THIS_FROM_SCRATCH_COMPOSITION",
            "archive_bytes": None,
            "per_class_d_seg": None,
            "d_pose": None,
            "score": None,
            "pointer_moved": False,
        },
    }

    return {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "lane_id": lane_id,
        "authority_axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "competitive_target": effective_frontier,
        "pointer_delta": 0.0,
        "verdict": "PARTIAL_CONSTRUCTIVE_SPINE_MEASURED_S2_AND_S3_NOT_COMPOSED_S4_ABSENT",
        "verdict_scope": "current from-scratch source-only composition; component gaps do not close the task-space level-set witness family",
        "source_custody": source_custody,
        "consumed_artifacts": consumed,
        "borrowed_substrate_accounting": {
            "borrowed_candidate_archives": [],
            "borrowed_candidate_payloads": [],
            "inherited_bytes_in_candidate": 0,
            "lineage_submission_eligible": True,
            "candidate_submission_eligible": False,
            "note": "all candidate-bearing rows are source-derived/our-solve; r1b7 is law-only and every inherited archive/payload byte is excluded",
        },
        "stages": stages,
        "admission": {
            "s0_s1_ground_truth_spine": True,
            "s2_lane_true_mask_curve_measured": lane_curve is not None,
            "s2_finite_cell_event_seed": finite_seed is not None,
            "s2_complete_partition_pose_description": False,
            "s3_integer_realization_composed": False,
            "s4_receiver_closed_archive": False,
            "score_or_pointer_authority": False,
        },
        "runtime": {
            "git_head": git_head(),
            "auditor_sha256": sha256_file(Path(__file__)),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--tie-receipt", type=Path, required=True)
    parser.add_argument("--m2-receipt", type=Path, required=True)
    parser.add_argument("--lane-receipt", type=Path, required=True)
    parser.add_argument("--g1g3-receipt", type=Path, required=True)
    parser.add_argument("--aa-receipt", type=Path, required=True)
    parser.add_argument("--r1b7-receipt", type=Path, required=True)
    parser.add_argument("--s2-partition-seed-receipt", type=Path)
    parser.add_argument("--s2-lane-curve-receipt", type=Path)
    parser.add_argument("--runtime-rs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frontier-pointer",
        type=Path,
        default=DEFAULT_FRONTIER_POINTER,
        help="canonical dynamic competitive pointer (never a candidate/archive parent)",
    )
    parser.add_argument("--lane-id", default="joint_planes_direct_strike")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rust = run_rust_parity(args.runtime_rs.resolve())
    receipt = build_receipt(
        source_video=args.source_video.resolve(),
        upstream_root=args.upstream_root.resolve(),
        gt_cache=args.gt_cache.resolve(),
        tie_receipt_path=args.tie_receipt.resolve(),
        m2_receipt_path=args.m2_receipt.resolve(),
        lane_receipt_path=args.lane_receipt.resolve(),
        g1g3_receipt_path=args.g1g3_receipt.resolve(),
        aa_receipt_path=args.aa_receipt.resolve(),
        r1b7_receipt_path=args.r1b7_receipt.resolve(),
        s2_partition_seed_receipt_path=(
            None
            if args.s2_partition_seed_receipt is None
            else args.s2_partition_seed_receipt.resolve()
        ),
        s2_lane_curve_receipt_path=(
            None
            if args.s2_lane_curve_receipt is None
            else args.s2_lane_curve_receipt.resolve()
        ),
        rust_parity=rust,
        lane_id=args.lane_id,
        frontier_pointer_path=args.frontier_pointer.resolve(),
    )
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
