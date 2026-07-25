#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the fresh batch32 MS2R-R3 BOX finite-family solve.

This stage supersedes the stale 14/7-oracle R3 attempt without mutating it.  It
recomputes C1 q1/q4/q8 scorer endpoints at batch32, recomputes real
Brotli-Q11 predictor-record rates, solves the exact q4/q8 byte minimum under
the 136,839-error box, and preserves the still-missing RD1 cell duals as NULL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_metric_custody_bundle import load_metric_custody_bundle  # noqa: E402
from tac.optimization.ddm_min_description_contract import (  # noqa: E402
    LayerHome,
    StreamType,
    TypedStreamTag,
    build_minimum_description_headline,
)
from tac.optimization.ddm_ms2r_r3_box_tolerance_solve import (  # noqa: E402
    CLASS_NAMES,
    backfill_rd1_cells_null_preserving,
    build_binary_dual_diagnostics,
)
from tac.optimization.ddm_ms2r_tolerance_capped_solve_r2 import (  # noqa: E402
    quantize_uint8_half_up,
    solve_binary_pair_lattice,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.scorer_value_oracle import ScorerValueOracle  # noqa: E402
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    DDMV14RealizationFidelityConfigV1,
    _forward,
    _load_models,
)
from tools.run_ddm_ms2r_tolerance_capped_solve_r2 import (  # noqa: E402
    _artifact,
    _bound,
    _materialize_candidate,
    _measure_rate,
    _publish_json,
    _race_candidate_streams,
    _read_json,
    _source_batch,
)

RUN_ID: Final = "ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z"
LANE_ID: Final = "ddm_ms2r_r3_box_tolerance_solve"
SCHEMA: Final = "ddm_ms2r_r3_box_tolerance_solve_receipt.v1"
CONFIG_PATH: Final = REPO / ".omx/research/configs/ddm_ms2r_r3_box_tolerance_solve_20260725.json"
RECEIPT_ROOT: Final = REPO / ".omx/research" / RUN_ID
RECEIPT_PATH: Final = RECEIPT_ROOT / "receipt.json"
SCORED_PIXELS: Final = 600 * 384 * 512
POINTER: Final = "0.1910828242 [contest-CPU]"
AXIS: Final = "[macOS-CPU advisory]"


class R3BoxRunError(ValueError):
    """A typed boundary, sealed input, or resumable checkpoint differs."""


class R3BoxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: str = Field(alias="schema", serialization_alias="schema")
    run_id: str
    lane_id: str
    authority_path: str
    authority_bytes: StrictInt
    authority_sha256: str
    c1_root: str
    c1_archive_path: str
    c1_archive_sha256: str
    scorer_config_path: str
    scorer_config_sha256: str
    upstream_root: str
    target_cache_path: str
    target_cache_bytes: StrictInt
    target_cache_sha256: str
    bundle_complete_path: str
    bundle_complete_sha256: str
    rd1_duals_path: str
    rd1_duals_sha256: str
    ev1_receipt_path: str
    ev1_receipt_sha256: str
    j8f_receipt_path: str
    j8f_receipt_sha256: str
    pc1_receipt_path: str
    pc1_receipt_sha256: str
    bulk_root: str
    pair_count: StrictInt
    source_chunk_pairs: StrictInt
    scorer_batch_size: StrictInt
    scorer_threads: StrictInt
    rate_workers: StrictInt
    minimum_free_bytes: StrictInt
    seed: StrictInt
    allowed_errors: StrictInt
    charter_rounded_q1_errors: StrictInt
    expected_measured_q1_errors: StrictInt
    research_only: StrictBool
    local_measurement_allowed: StrictBool
    external_execution_allowed: StrictBool
    score_claim: StrictBool
    main_review_required: StrictBool
    receipt_timestamp_utc: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[R3BoxConfig, str]:
    payload = path.read_bytes()
    config = R3BoxConfig.model_validate_json(payload)
    if (
        config.schema_name != "DDMMS2RR3BoxToleranceSolveConfigV1"
        or config.run_id != RUN_ID
        or config.lane_id != LANE_ID
        or config.pair_count != 600
        or config.source_chunk_pairs != 12
        or config.scorer_batch_size != 32
        or config.scorer_threads != 4
        or config.seed != 1234
        or config.allowed_errors != 136_839
        or config.charter_rounded_q1_errors != 17_931
        or config.expected_measured_q1_errors != 17_927
        or config.research_only is not True
        or config.local_measurement_allowed is not True
        or config.external_execution_allowed is not False
        or config.score_claim is not False
        or config.main_review_required is not True
    ):
        raise R3BoxRunError("fresh R3 BOX typed execution boundary differs")
    return config, hashlib.sha256(payload).hexdigest()


def _score_terms(*, errors: int, d_pose: float, archive_bytes: int) -> dict[str, float]:
    d_seg = errors / SCORED_PIXELS
    seg_term = 100.0 * d_seg
    pose_term = math.sqrt(10.0 * d_pose)
    rate_term = 25.0 * archive_bytes / 37_545_489
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "objective": seg_term + pose_term + rate_term,
    }


def _validate_scorer_batch(
    value: Mapping[str, Any],
    *,
    start: int,
    count: int,
) -> list[dict[str, Any]]:
    rows = value.get("rows")
    if (
        value.get("schema") != "ddm_ms2r_r3_scorer_batch32.v1"
        or value.get("pair_range") != [start, start + count]
        or not isinstance(rows, list)
        or len(rows) != count
        or [row.get("pair_id") for row in rows] != list(range(start, start + count))
    ):
        raise R3BoxRunError(f"resumed scorer batch differs at {start}")
    return [dict(row) for row in rows]


def _measure_scorers(config: R3BoxConfig, bulk: Path) -> dict[str, Any]:
    checkpoint = bulk / "stage_checkpoints/02_scorers/scorer_measurement.json"
    if checkpoint.exists():
        value = _read_json(checkpoint)
        if (
            value.get("batch_size") != 32
            or value.get("q1_exact_control", {}).get("errors")
            != config.expected_measured_q1_errors
        ):
            raise R3BoxRunError("resumed fresh batch32 scorer measurement differs")
        return value
    scorer_path = _bound(REPO / config.scorer_config_path, config.scorer_config_sha256)
    sealed = DDMV14RealizationFidelityConfigV1.model_validate_json(scorer_path.read_bytes())
    runtime = sealed.model_copy(
        update={
            "upstream_root": config.upstream_root,
            "scorer_batch_size": config.scorer_batch_size,
        }
    )
    target = _bound(Path(config.target_cache_path), config.target_cache_sha256)
    if target.stat().st_size != config.target_cache_bytes:
        raise R3BoxRunError("target cache byte custody differs")
    labels_all = open_stored_npy_memmap(target, "lstars")
    poses_all = open_stored_npy_memmap(target, "gt_poses")
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    rows: list[dict[str, Any] | None] = [None] * config.pair_count
    stratum_sites = dict.fromkeys(CLASS_NAMES, 0)
    pending: list[tuple[int, int, Path]] = []
    for start in range(0, config.pair_count, config.scorer_batch_size):
        count = min(config.scorer_batch_size, config.pair_count - start)
        labels = np.asarray(labels_all[start : start + count], dtype=np.uint8)
        for class_id, name in enumerate(CLASS_NAMES):
            stratum_sites[name] += int(np.count_nonzero(labels == class_id))
        path = bulk / "stage_checkpoints/02_scorers/batches" / f"batch-{start:04d}.json"
        if path.exists():
            for row in _validate_scorer_batch(_read_json(path), start=start, count=count):
                rows[int(row["pair_id"])] = row
        else:
            pending.append((start, count, path))
    verification_probes = 0
    segnet, posenet, scorer_custody = _load_models(runtime)
    if pending:
        for start, count, path in pending:
            y0, y1 = _source_batch(Path(config.c1_root), start, count)
            labels = np.asarray(labels_all[start : start + count], dtype=np.uint8)
            poses = np.asarray(poses_all[start : start + count], dtype=np.float64)
            batch_rows = [{"pair_id": start + local} for local in range(count)]
            for step in (1, 4, 8):
                q0 = y0 if step == 1 else quantize_uint8_half_up(y0, step)
                q1 = y1 if step == 1 else quantize_uint8_half_up(y1, step)
                camera = np.empty((count, 2, 874, 1164, 3), dtype=np.uint8)
                for local in range(count):
                    camera[local, 0] = realize_factor2_uint8_scorer_plane(operator, q0[local])
                    camera[local, 1] = realize_factor2_uint8_scorer_plane(operator, q1[local])
                for plane_id, target_plane in ((0, q0[0]), (1, q1[0])):
                    if not verify_factor2_uint8_scorer_plane(
                        operator, camera[0, plane_id], target_plane
                    ).certified_exact:
                        raise R3BoxRunError("uint8 resize preimage verification failed")
                    verification_probes += 1
                cells, predicted_pose = _forward(segnet, posenet, camera)
                differences = cells != labels
                pair_errors = np.count_nonzero(differences, axis=(1, 2))
                pose_sse = np.square(predicted_pose - poses).sum(axis=1, dtype=np.float64)
                for local, row in enumerate(batch_rows):
                    row[f"q{step}_errors"] = int(pair_errors[local])
                    row[f"q{step}_pose_sse"] = float(pose_sse[local])
                    row[f"q{step}_stratum_errors"] = {
                        name: int(np.count_nonzero(differences[local] & (labels[local] == class_id)))
                        for class_id, name in enumerate(CLASS_NAMES)
                    }
            _publish_json(
                path,
                {
                    "schema": "ddm_ms2r_r3_scorer_batch32.v1",
                    "pair_range": [start, start + count],
                    "rows": batch_rows,
                    "score_claim": False,
                },
            )
            for row in batch_rows:
                rows[int(row["pair_id"])] = row
    if any(row is None for row in rows):
        raise R3BoxRunError("fresh scorer batch set is incomplete")
    completed = [row for row in rows if row is not None]
    q1_errors = sum(int(row["q1_errors"]) for row in completed)
    if q1_errors != config.expected_measured_q1_errors:
        raise R3BoxRunError(
            "fresh batch32 C1 exact replay differs from "
            f"{config.expected_measured_q1_errors}: {q1_errors}"
        )
    value = {
        "schema": "ddm_ms2r_r3_scorer_measurement.v1",
        "pair_count": config.pair_count,
        "batch_size": config.scorer_batch_size,
        "threads": config.scorer_threads,
        "rows": completed,
        "stratum_sites": stratum_sites,
        "q1_exact_control": {
            "errors": q1_errors,
            "d_seg": q1_errors / SCORED_PIXELS,
            "d_pose": sum(float(row["q1_pose_sse"]) for row in completed) / (config.pair_count * 6),
            "charter_rounded_error_count": config.charter_rounded_q1_errors,
            "rounding_difference_errors": config.charter_rounded_q1_errors - q1_errors,
            "status": "MEASURED_EXACT; 17,931_IS_ROUNDED_1.52E-4_ARITHMETIC",
        },
        "uint8_factor2_exact_probe_count_this_invocation": verification_probes,
        "all_candidate_planes_realized_by_exact_constructor": True,
        "target_cache": _artifact(target, "frozen n600 labels and Pose6"),
        "scorer_custody": scorer_custody,
        "evidence_axis": AXIS,
        "score_claim": False,
    }
    _publish_json(checkpoint, value)
    return value


def _solve(
    config: R3BoxConfig,
    rate: Mapping[str, Any],
    scorers: Mapping[str, Any],
    bulk: Path,
) -> dict[str, Any]:
    checkpoint = bulk / "stage_checkpoints/03_solve/exact_binary_solve.json"
    if checkpoint.exists():
        return _read_json(checkpoint)
    rate_rows = rate.get("rows")
    scorer_rows = scorers.get("rows")
    if not isinstance(rate_rows, list) or not isinstance(scorer_rows, list):
        raise R3BoxRunError("rate or scorer rows are absent")
    rows: list[dict[str, Any]] = []
    for rate_row, scorer_row in zip(rate_rows, scorer_rows, strict=True):
        if rate_row["pair_id"] != scorer_row["pair_id"]:
            raise R3BoxRunError("rate/scorer pair identity differs")
        rows.append({**rate_row, **scorer_row})
    result = solve_binary_pair_lattice(rows, allowed_errors=config.allowed_errors)
    selected = result["selected_steps"]
    result["realized_pose_sse"] = sum(
        float(row[f"q{step}_pose_sse"]) for row, step in zip(rows, selected, strict=True)
    )
    result["realized_d_pose"] = result["realized_pose_sse"] / (config.pair_count * 6)
    result["realized_stratum_errors"] = {
        name: sum(
            int(row[f"q{step}_stratum_errors"][name])
            for row, step in zip(rows, selected, strict=True)
        )
        for name in CLASS_NAMES
    }
    result["binary_dual_diagnostics"] = build_binary_dual_diagnostics(rows, selected)
    result["objective_scope"] = (
        "Exact minimum real coded predictor-record bytes subject to the Seg BOX. "
        "Fixed receiver/container overhead preserves the ordering; the full "
        "Seg/Pose/rate functional is evaluated on the selected exact object."
    )
    _publish_json(checkpoint, result)
    return result


def _validate_controls(config: R3BoxConfig) -> dict[str, Any]:
    j8f_path = _bound(Path(config.j8f_receipt_path), config.j8f_receipt_sha256)
    j8f = _read_json(j8f_path)
    stages = j8f.get("application", {}).get("stage_receipts")
    projected = j8f.get("range_gauge_projected_arm", {})
    verdict = projected.get("verdict", {})
    if (
        j8f.get("schema") != "ddm_j8f_counted_application_smoke.v1"
        or j8f.get("score_claim") is not False
        or j8f.get("pointer") != POINTER
        or not isinstance(stages, list)
        or len(stages) != 12
        or projected.get("archive", {}).get("bytes") != 138_804
    ):
        raise R3BoxRunError("J8F counted-application custody differs")
    pc1_path = _bound(REPO / config.pc1_receipt_path, config.pc1_receipt_sha256)
    pc1 = _read_json(pc1_path)
    admission = pc1.get("admission", {})
    if (
        pc1.get("schema") != "ddm_pc1_pose_stream_admission.v1"
        or pc1.get("score_claim") is not False
        or admission.get("admitted") is not True
        or admission.get("n600_batch32_measured") is not True
        or admission.get("descent_was_run") is not False
        or admission.get("tube_claim") is not False
    ):
        raise R3BoxRunError("PC1 admitted pose-stream custody differs")
    return {
        "j8f": {
            "source": _artifact(j8f_path, "J8F counted application receipt"),
            "preserved_stage_count": len(stages),
            "projected_endpoint": {
                "errors": round(float(verdict["d_seg"]) * SCORED_PIXELS),
                "d_seg": verdict["d_seg"],
                "d_pose": verdict["d_pose"],
                "archive_bytes": projected["archive"]["bytes"],
            },
            "box_eligible": float(verdict["d_seg"]) <= 0.00116,
            "composition_status": (
                "COUNTED_OPERATOR_CONSUMED_AS_NONBOX_CONTROL; NO C1 PAIR-COORDINATE FOREIGN KEYS"
            ),
        },
        "pc1": {
            "source": _artifact(pc1_path, "PC1 pose stream admission receipt"),
            "admitted": True,
            "descent_was_run": False,
            "tube_claim": False,
            "parents": {
                name: {
                    "d_seg": value["parent_endpoint"]["d_seg"],
                    "d_pose": value["parent_endpoint"]["d_pose"],
                    "archive_bytes": value["parent_endpoint"]["archive_bytes"],
                    "box_eligible": value["parent_endpoint"]["d_seg"] <= 0.00116,
                }
                for name, value in pc1["parents"].items()
            },
            "composition_status": "ADMITTED_INACTIVE_ZERO_HOME; NO DESCENT ENDPOINT TO COMPOSE",
        },
    }


def _rd1_backfill(config: R3BoxConfig) -> dict[str, Any]:
    source_path = _bound(REPO / config.rd1_duals_path, config.rd1_duals_sha256)
    source = _read_json(source_path)
    rows = source.get("dimension_duals", {}).get("bucket_rows")
    if not isinstance(rows, list):
        raise R3BoxRunError("RD1 source cube is absent")
    ev1_path = _bound(REPO / config.ev1_receipt_path, config.ev1_receipt_sha256)
    ev1 = _read_json(ev1_path)
    ev1_rows = ev1.get("rd1_evidence", {}).get("bucket_rows")
    pair_join = ev1.get("v19_pair_join", {})
    if (
        ev1.get("schema") != "ddm_ev1_campaign_evidence_join_receipt.v1"
        or ev1.get("pair_count") != 600
        or not isinstance(ev1_rows, list)
        or len(pair_join.get("rows", [])) != 600
        or pair_join.get("shared_rate_home", {}).get("per_pair_allocation") is not None
    ):
        raise R3BoxRunError("EV1 600-pair/162-home custody differs")
    value = backfill_rd1_cells_null_preserving(rows, ev1_rows=ev1_rows)
    value["source"] = _artifact(source_path, "RD1 162-cell source cube")
    value["ev1_source"] = _artifact(
        ev1_path, "EV1 600-pair joins and 162 accounting homes"
    )
    value["ev1_pair_rate_allocation_status"] = (
        "NULL_PER_PAIR_ALLOCATION; ACCOUNTING_HOMES_NOT_C1_COORDINATE_FOREIGN_KEYS"
    )
    return value


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    bulk = Path(config.bulk_root)
    bulk.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(bulk).free < config.minimum_free_bytes:
        raise R3BoxRunError("SSD-first preflight has insufficient free bytes")
    bundle_path = _bound(REPO / config.bundle_complete_path, config.bundle_complete_sha256)
    bundle = load_metric_custody_bundle(bundle_path, repository_root=REPO, require_complete=True)
    authority_path = _bound(Path(config.authority_path), config.authority_sha256)
    if authority_path.stat().st_size != config.authority_bytes:
        raise R3BoxRunError("delegated authority byte custody differs")
    coverage = ScorerValueOracle(REPO).coverage_report(verify=True)
    if (
        coverage.get("row_count") != 21
        or coverage.get("counts") != {"WRAPPED": 21, "TYPED-GAP": 0}
        or coverage.get("stale_advisory_count") != 0
    ):
        raise R3BoxRunError("live scorer-value oracle is not fresh 21/0")
    _bound(Path(config.c1_archive_path), config.c1_archive_sha256)
    controls = _validate_controls(config)
    rate = _measure_rate(config, bulk)
    scorers = _measure_scorers(config, bulk)
    solve = _solve(config, rate, scorers, bulk)
    candidate = _materialize_candidate(config, solve, bulk)
    coder_race = _race_candidate_streams(config, solve, candidate, bulk)
    terms = _score_terms(
        errors=int(solve["realized_errors"]),
        d_pose=float(solve["realized_d_pose"]),
        archive_bytes=int(candidate["archive"]["bytes"]),
    )
    if int(solve["realized_errors"]) > config.allowed_errors:
        raise R3BoxRunError("materialized solve left the Seg BOX")
    per_stratum = {
        name: {
            "errors": int(solve["realized_stratum_errors"][name]),
            "sites": int(scorers["stratum_sites"][name]),
            "d_seg": (
                int(solve["realized_stratum_errors"][name])
                / int(scorers["stratum_sites"][name])
            ),
        }
        for name in CLASS_NAMES
    }
    rd1 = _rd1_backfill(config)
    dual_path = RECEIPT_ROOT / "rd1_162_dual_backfill.json"
    _publish_json(dual_path, rd1)
    headline = build_minimum_description_headline(
        stored_problem_bytes=int(candidate["archive"]["bytes"]),
        stored_problem_sha256=str(candidate["archive"]["sha256"]),
        exception_bytes=0,
        exception_sha256=hashlib.sha256(b"").hexdigest(),
        realized_d_seg=terms["d_seg"],
        realized_d_pose=terms["d_pose"],
        stored_problem_own_lineage=True,
        donor_conditioned=False,
        expansion_receiver_closed=True,
        pose_tube_active=False,
        realized_uint8_r_frozen_scorers=True,
        quotient_coordinates_only=True,
        scorer_metric_active=True,
        alternating_typed_subproblems=False,
        typed_blocks_active=False,
        per_dimension_quanta_active=False,
        typed_stream_tags=(
            TypedStreamTag(
                type=StreamType.FIBER,
                layer_home=LayerHome.L3_RASTER,
                evaluate_py_recursion_level_cited=(
                    "L3 C1 quotient raster -> L4 frozen scorers -> L5 advisory verdict"
                ),
                counted_bytes=int(candidate["archive"]["bytes"]),
                free_receiver_code=True,
            ),
            TypedStreamTag(
                type=StreamType.RESIDUAL,
                layer_home=LayerHome.L5_VERDICT,
                evaluate_py_recursion_level_cited="L5 no separate exception stream",
                counted_bytes=0,
                free_receiver_code=True,
            ),
        ),
        strict_typed_stream_tags=True,
        metric_custody_bundle_path=bundle_path,
        metric_custody_repository_root=REPO,
    )
    control_rows = [
        {
            "candidate": "J8F_RANGE_GAUGE_PROJECTED",
            **controls["j8f"]["projected_endpoint"],
            "box_eligible": controls["j8f"]["box_eligible"],
        },
        *[
            {"candidate": f"PC1_{name}_PARENT", **value}
            for name, value in controls["pc1"]["parents"].items()
        ],
    ]
    for row in control_rows:
        row["objective_terms"] = _score_terms(
            errors=int(row.get("errors", round(float(row["d_seg"]) * SCORED_PIXELS))),
            d_pose=float(row["d_pose"]),
            archive_bytes=int(row["archive_bytes"]),
        )
    receipt = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "finished_at_utc": config.receipt_timestamp_utc,
        "verdict": (
            "MEASURED_BATCH32_RECEIVER_CLOSED_Q4_Q8_CHEAPEST_FINITE_FAMILY_BOX_MEMBER; "
            "FULL_TYPED_162_CELL_WATERFILL_BLOCKED"
        ),
        "verdict_scope": (
            "INSTANCE: n600 C1 exact scorer-quotient per-pair q4/q8 family, "
            "real Brotli-Q11 predictor records, exact uint8/R frozen scorers at "
            "batch32 on macOS CPU advisory. Exact byte minimum inside 136839 "
            "errors for this finite family only; no contest score, promotion, "
            "J8F-to-C1 composition, PC1 tube, or full Fisher/G4 claim."
        ),
        "authority": {
            "axis": AXIS,
            "score_claim": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "promotion_eligible": False,
            "research_only": True,
            "main_review_required": True,
        },
        "typed_config": {
            "path": str(config_path.relative_to(REPO)),
            "sha256": config_sha256,
        },
        "delegated_authority": _artifact(
            authority_path, "delegated task authority"
        ),
        "implementation_custody": {
            "sources": [
                _artifact(
                    REPO
                    / "src/tac/optimization/ddm_ms2r_r3_box_tolerance_solve.py",
                    "pure R3 BOX dual diagnostics",
                ),
                _artifact(
                    REPO / "tools/run_ddm_ms2r_r3_box_tolerance_solve.py",
                    "resumable R3 BOX runner",
                ),
                _artifact(
                    REPO / "tests/test_ddm_ms2r_r3_box_tolerance_solve.py",
                    "focused R3 BOX regressions",
                ),
            ],
            "post_edit_shas_recorded": True,
        },
        "fresh_stage": {
            "supersedes_old_oracle_shape": "14 WRAPPED / 7 TYPED-GAP",
            "old_stage_mutated": False,
            "delegated_exact_count_premise_falsification": {
                "delegated_rounded_count": config.charter_rounded_q1_errors,
                "fresh_measured_exact_count": scorers["q1_exact_control"]["errors"],
                "difference_errors": (
                    config.charter_rounded_q1_errors
                    - int(scorers["q1_exact_control"]["errors"])
                ),
                "disposition": (
                    "17,931 is rounded 1.52e-4 arithmetic; exact n600 frozen-scorer "
                    "custody remains 17,927 at batch32"
                ),
            },
            "live_oracle_coverage": coverage,
            "strict_ms3_bundle": {
                "path": str(bundle_path.relative_to(REPO)),
                "bundle_id": bundle.bundle_id,
                "complete": bundle.complete,
                "sha256": config.bundle_complete_sha256,
            },
        },
        "solve": solve,
        "candidate": candidate,
        "coder_race": coder_race,
        "objective_terms": terms,
        "per_stratum": per_stratum,
        "candidate_comparison": [
            {
                "candidate": "MS2R_R3_C1_Q4_Q8",
                "errors": solve["realized_errors"],
                "d_seg": terms["d_seg"],
                "d_pose": terms["d_pose"],
                "archive_bytes": candidate["archive"]["bytes"],
                "box_eligible": True,
                "objective_terms": terms,
                "selection_status": "CHEAPEST_EXACT_MEMBER_OF_FINITE_Q4_Q8_BOX_FAMILY",
            },
            *control_rows,
        ],
        "consumed_controls": controls,
        "ev1_accounting_home_consumption": {
            "source": rd1["ev1_source"],
            "pair_join_count": 600,
            "accounting_home_count": rd1["ev1_accounting_home_cell_count"],
            "nonzero_accounting_byte_cells": rd1[
                "ev1_nonzero_accounting_byte_cell_count"
            ],
            "nonzero_distortion_cells": rd1[
                "ev1_nonzero_distortion_cell_count"
            ],
            "beneficial_diagnostic_accounting_slopes": rd1[
                "ev1_beneficial_accounting_slope_count"
            ],
            "finite_per_dimension_duals": rd1[
                "finite_per_dimension_dual_count"
            ],
            "per_pair_rate_allocation_status": rd1[
                "ev1_pair_rate_allocation_status"
            ],
        },
        "rd1_162_backfill": _artifact(dual_path, "null-preserving RD1 162-cell backfill"),
        "minimum_description_headline": headline,
        "remaining_exact_blocker": (
            "J8F counted applications do not carry C1 pair-coordinate foreign keys, "
            "and C1 predictor records do not own stratum x visibility x G4 byte "
            "coordinates. PC1 has no descent or active tube. Therefore 162 "
            "cellwise lambdas and a mixed receiver cannot be lawfully solved."
        ),
        "resumability": {
            "per_rate_chunk": True,
            "per_scorer_batch": True,
            "solve_checkpoint": True,
            "candidate_checkpoint": True,
            "coder_stream_checkpoint": True,
            "all_stage_checkpoints_preserved": True,
            "bulk_root": str(bulk),
        },
        "pointer_delta": "NONE",
        "score_claim": False,
    }
    _publish_json(RECEIPT_PATH, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    receipt = run(args.config.resolve())
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
