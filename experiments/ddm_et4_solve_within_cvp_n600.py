#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM ET4 n600 solve-within + DK1 CVP composition runner.

This is the fire-order-2 continuation of ET3.  It keeps the full-form SW1
solve-within and DK1 CVP/Babai mechanism, runs it over the 600-pair population,
and persists the realized frame_1 deltas for a counted overlay archive.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "experiments", REPO / "upstream"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from ddm_et2_projected_phase_field import (  # noqa: E402
    BASELINE_ARCHIVE_SHA256,
    BASELINE_BYTES,
    BASELINE_D_POSE,
    BASELINE_D_SEG,
    BASELINE_S,
    forward,
    load_models,
    raw_memmap,
    score_from_components,
)
from ddm_et3_solve_within_cvp_phase_field import (  # noqa: E402
    AXIS,
    SCORE_CLAIM,
    PROMOTION_ELIGIBLE,
    cap_receipt_from_diagnostics,
    git_head,
    jsonable,
    parse_cap_ladder,
    realize_cvp_delta,
    score_pair,
    sha256_file,
    solve_within_null_basis_delta,
    write_json_atomic,
)
from ddm_et1_ph1_block16_on_our_vehicle import translate_blocks  # noqa: E402
from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    CLASS_NAMES,
    N_CLASSES,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_pose_null_constrained_paint import snap_band_to_blocks, yuv6_shift  # noqa: E402
from ddm_sq1_stage_decomposition_and_solved_paint import confusion, resize_to_scorer  # noqa: E402
from ddm_sw1_null_basis_phase_solve import (  # noqa: E402
    block_mask_from_band,
    coeffs_from_delta_euclidean,
    metric_weights_from_saliency,
    null_coordinate_basis,
    pose_constraint_matrix,
)
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.margin_saliency_map import compute_margin_saliency_map  # noqa: E402
from tac.submission_chain import (  # noqa: E402
    ChainPaths,
    SubmissionChainError,
    run_inflate,
    run_upstream_evaluate,
    stage_submission,
)

import ddm_et4_overlay_codec as overlay_codec  # noqa: E402


RUN_SCHEMA = "ddm_et4_solve_within_cvp_n600.v1"
DEFAULT_BULK_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_et4_20260806")
DEFAULT_RECEIPT_DIR = REPO / ".omx/research/ddm_et4_20260806"
DEFAULT_BASE_RUNTIME = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit")
DEFAULT_PARENT_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/"
    "candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes"
)
DEFAULT_PARENT_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/"
    "submission/inflated/0.raw"
)
DEFAULT_PARENT_ARGMAX = Path(
    "/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/"
    "parent_tq1c_argmax_n600.npy"
)
DEFAULT_PARENT_SCORE = Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/aggregate.json")
DEFAULT_OFFSETS = Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/tq1c_block16_offsets.npy")
DEFAULT_PHASE_FIELD = Path(
    "/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/phase_field_rederive_summary.json"
)
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_GT_MKV = REPO / "upstream/videos/0.mkv"
DEFAULT_SW1_SUMMARY = REPO / ".omx/research/ddm_sw1_20260806/sw1_null_basis_summary.json"

BASE_RUNTIME_FILES = (
    "ddm_r7_token_coder.py",
    "ddm_tr1_runtime.py",
    "pfs1_warp_receiver.py",
    "repair_entropy_coder_runtime_adapters.py",
    "ddm_ix2_archive_container.py",
)
STAGED_RUNTIME_FILES = (
    "inflate.sh",
    "inflate_runner.py",
    "ddm_et4_overlay_codec.py",
    "base_inflate_runner.py",
    *BASE_RUNTIME_FILES,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=jsonable, allow_nan=False))
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def class_error_counts(cmat: np.ndarray) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    class_items = CLASS_NAMES.items() if hasattr(CLASS_NAMES, "items") else enumerate(CLASS_NAMES)
    for class_id, class_name in class_items:
        sites = int(cmat[class_id].sum())
        correct = int(cmat[class_id, class_id])
        rows[str(class_name)] = {"errors": sites - correct, "sites": sites}
    return rows


def vo2_element_grade_vector(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """VO2-shaped 10-element form grading for the ET4 composed chain."""

    full_population = len(rows) == N_PAIRS_TOTAL
    cap_stops_present = all(
        row.get("solve_within", {})
        .get("selected_attempt", {})
        .get("cap_stop_receipt", {})
        .get("stop_reason")
        for row in rows
    )
    cache_controls_pass = all(
        row.get("controls", {}).get("C2_parent_argmax_matches_cache") is True
        and row.get("controls", {}).get("C3_gt_argmax_matches_cache") is True
        for row in rows
    )
    block_coverage_full = all(
        int(
            row.get("cvp_realized", {})
            .get("realizer_receipt", {})
            .get("aggregate", {})
            .get("blocks_realized", -1)
        )
        == int(
            row.get("cvp_realized", {})
            .get("realizer_receipt", {})
            .get("aggregate", {})
            .get("exact_declared_scope_count", -2)
        )
        for row in rows
    )
    elements = [
        {
            "name": "initialization",
            "form_grade": "OPTIMAL-RECEIPT",
            "note": "frontier parent archive sha/bytes are verified before any pair runs",
        },
        {
            "name": "proposal_step_rule",
            "form_grade": "OPTIMAL-RECEIPT",
            "note": "SW1 c-space delta=Nc solve-within step; no reduced/project-after form",
        },
        {
            "name": "stopping_rule",
            "form_grade": "OPTIMAL-RECEIPT" if cap_stops_present else "UNKNOWN",
            "note": "every completed row carries a selected CapStopReceipt; cap_bound is not convergence evidence",
        },
        {
            "name": "metric_inner_product",
            "form_grade": "OPTIMAL-RECEIPT",
            "note": (
                "chartered SW1 score-metric objective using margin saliency weights; no global MS4D metric "
                "claim beyond this chain"
            ),
            "lambda_saliency": float(args.lambda_saliency),
            "outside_weight": float(args.outside_weight),
            "saliency_clip": float(args.saliency_clip),
        },
        {
            "name": "subset_sampling",
            "form_grade": "OPTIMAL-RECEIPT" if full_population else "NAIVE-NAMED",
            "note": (
                "full n600 population complete"
                if full_population
                else "current checkpoint is a prefix timing slice; n=8 banks nothing"
            ),
            "completed_rows": len(rows),
            "population": N_PAIRS_TOTAL,
        },
        {
            "name": "realization",
            "form_grade": "OPTIMAL-RECEIPT" if block_coverage_full else "UNKNOWN",
            "note": "DK1 CVP/Babai finite kept-scope realizer; no global MIQP claim",
        },
        {
            "name": "projection_constraint_handling",
            "form_grade": "OPTIMAL-RECEIPT",
            "note": "pose-null basis is enforced inside the solve, not projected after the fact",
        },
        {
            "name": "tie_breaks",
            "form_grade": "UNKNOWN",
            "note": "local candidate enumeration order is deterministic but no global tie theorem is claimed",
        },
        {
            "name": "seed_determinism",
            "form_grade": "OPTIMAL-RECEIPT",
            "note": "no RNG is used by ET4 row ordering or the overlay codec",
        },
        {
            "name": "caches_staleness",
            "form_grade": "OPTIMAL-RECEIPT" if cache_controls_pass else "UNKNOWN",
            "note": "completed rows verify decoded parent argmax and GT argmax against caches",
        },
    ]
    return {
        "schema": "ddm_vo2_r2_element_decomposition.row.v1",
        "instrument_id": "ddm_et4_solve_within_cvp_n600_composed_chain",
        "path": "experiments/ddm_et4_solve_within_cvp_n600.py",
        "provenance_family": "et4",
        "status": "R2_GRADED_CHECKPOINT" if not full_population else "R2_GRADED_N600",
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": PROMOTION_ELIGIBLE,
        "elements": elements,
        "calibration_lineage": [
            {"kind": "receipt_or_source", "ref": ".omx/research/ddm_sw1_20260806/RECEIPT.md"},
            {"kind": "receipt_or_source", "ref": ".omx/research/ddm_dk1_20260806/RECEIPT.md"},
            {"kind": "receipt_or_source", "ref": ".omx/research/ddm_rw2_20260806/RECEIPT.md"},
            {"kind": "receipt_or_source", "ref": ".omx/research/ddm_vo2_20260806/R2_ELEMENT_DECOMPOSITION.jsonl"},
        ],
        "fire_order": "complete n600, byte-close archive through tac.submission_chain, then measure realized S",
    }


def storage_preflight(args: argparse.Namespace) -> dict[str, Any]:
    args.bulk_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.bulk_dir)
    required = int(args.min_free_gb * (1024**3))
    return {
        "schema": "ddm_et4_storage_preflight.v1",
        "bulk_dir": str(args.bulk_dir),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "min_free_bytes": required,
        "passes": bool(usage.free >= required),
        "parent_raw_bytes": args.parent_raw.stat().st_size if args.parent_raw.exists() else None,
        "notes": [
            "bulk and candidate artifacts stay on SSD",
            "no /tmp evidence path is used",
        ],
    }


def memory_preflight() -> dict[str, Any]:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        vm = psutil.virtual_memory()
        return {
            "schema": "ddm_et4_memory_preflight.v1",
            "rss_bytes": int(proc.memory_info().rss),
            "available_bytes": int(vm.available),
            "source": "psutil",
        }
    except Exception as exc:  # pragma: no cover - host dependent
        return {
            "schema": "ddm_et4_memory_preflight.v1",
            "rss_bytes": None,
            "available_bytes": None,
            "source": f"unavailable:{type(exc).__name__}:{exc}",
        }


def pair_list(args: argparse.Namespace) -> list[int]:
    stop = min(N_PAIRS_TOTAL, int(args.pair_stop))
    start = max(0, int(args.pair_start))
    if start > stop:
        raise RuntimeError(f"bad pair range {start}:{stop}")
    pairs = list(range(start, stop))
    if args.limit:
        pairs = pairs[: int(args.limit)]
    return pairs


def patch_path(args: argparse.Namespace, pair: int) -> Path:
    return args.bulk_dir / "patch_records" / f"pair_{int(pair):04d}.npz"


def save_patch_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.savez_compressed(
        tmp,
        pair=np.asarray([int(record["pair"])], dtype=np.uint16),
        nnz=np.asarray([int(record["nnz"])], dtype=np.uint32),
        indices=np.asarray(record["indices"], dtype="<u4"),
        deltas_i16=np.asarray(record["deltas_i16"], dtype="<i2"),
    )
    generated = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    generated.replace(path)


def load_patch_record(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "pair": int(data["pair"][0]),
            "nnz": int(data["nnz"][0]),
            "indices": np.asarray(data["indices"], dtype="<u4"),
            "deltas_i16": np.asarray(data["deltas_i16"], dtype="<i2"),
        }


def aggregate_rows(rows: list[dict[str, Any]], *, archive_bytes: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "ddm_et4_aggregate.v1",
        "n_rows": len(rows),
        "population": N_PAIRS_TOTAL,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": PROMOTION_ELIGIBLE,
        "baseline": {
            "S": BASELINE_S,
            "d_seg": BASELINE_D_SEG,
            "d_pose": BASELINE_D_POSE,
            "archive_bytes": BASELINE_BYTES,
            "archive_sha256": BASELINE_ARCHIVE_SHA256,
        },
    }
    if not rows:
        return out
    before = int(sum(row["flips_before"] for row in rows))
    after = int(sum(row["cvp_realized"]["flips_after"] for row in rows))
    denom = int(sum(row["label_ceiling_net_fixed"] for row in rows))
    pose_before = float(sum(float(row["d_pose_before"]) for row in rows) / len(rows))
    pose_after = float(sum(float(row["cvp_realized"]["d_pose_after"]) for row in rows) / len(rows))
    ratios = np.asarray([float(row["cvp_realized"]["d_pose_ratio"]) for row in rows], dtype=np.float64)
    c_before = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    c_after = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    cap_counter: Counter[str] = Counter()
    exact_scope = 0
    blocks = 0
    for row in rows:
        c_before += np.asarray(row["C_before"], dtype=np.int64)
        c_after += np.asarray(row["cvp_realized"]["C_after"], dtype=np.int64)
        receipt = row["solve_within"]["selected_attempt"]["cap_stop_receipt"]
        cap_counter[str(receipt["stop_reason"])] += 1
        agg = row["cvp_realized"]["realizer_receipt"]["aggregate"]
        exact_scope += int(agg.get("exact_declared_scope_count", 0))
        blocks += int(agg.get("blocks_realized", 0))
    sites = len(rows) * SEG_H * SEG_W
    d_seg_before = before / sites
    d_seg_after = after / sites
    measured_archive_bytes = int(archive_bytes) if archive_bytes is not None else None
    measured_s = (
        score_from_components(d_seg_after, pose_after, measured_archive_bytes)
        if measured_archive_bytes is not None and len(rows) == N_PAIRS_TOTAL
        else None
    )
    out.update(
        {
            "flips_before": before,
            "flips_after": after,
            "net_flip_reduction": before - after,
            "label_ceiling_net_fixed": denom,
            "eta": (before - after) / denom if denom else None,
            "d_seg_before_completed_scope": d_seg_before,
            "d_seg_after_completed_scope": d_seg_after,
            "d_pose_before_completed_scope": pose_before,
            "d_pose_after_completed_scope": pose_after,
            "pose_ratio_min": float(ratios.min()),
            "pose_ratio_p25": float(np.quantile(ratios, 0.25)),
            "pose_ratio_median": float(np.median(ratios)),
            "pose_ratio_mean": float(ratios.mean()),
            "pose_ratio_p75": float(np.quantile(ratios, 0.75)),
            "pose_ratio_max": float(ratios.max()),
            "archive_bytes": measured_archive_bytes,
            "S": measured_s,
            "dS_vs_named_baseline": (measured_s - BASELINE_S) if measured_s is not None else None,
            "delta_d_seg_vs_named_baseline": (d_seg_after - BASELINE_D_SEG) if len(rows) == N_PAIRS_TOTAL else None,
            "delta_d_pose_vs_named_baseline": (pose_after - BASELINE_D_POSE) if len(rows) == N_PAIRS_TOTAL else None,
            "per_class_before": class_error_counts(c_before),
            "per_class_after": class_error_counts(c_after),
            "per_class_delta_errors": {
                name: class_error_counts(c_after)[name]["errors"] - class_error_counts(c_before)[name]["errors"]
                for name in class_error_counts(c_after)
            },
            "cap_stop_counts": dict(sorted(cap_counter.items())),
            "dk1_blocks_realized": blocks,
            "dk1_exact_declared_scope_blocks": exact_scope,
            "full_population_complete": len(rows) == N_PAIRS_TOTAL,
        }
    )
    return out


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    parent_archive_sha: str,
    storage: dict[str, Any],
    memory: dict[str, Any],
    archive_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    archive_bytes = None
    if archive_receipt is not None:
        archive_bytes = int(archive_receipt["archive"]["archive_bytes"])
    pairs = pair_list(args)
    completed = sorted(int(row["pair"]) for row in rows)
    return {
        "schema": RUN_SCHEMA,
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": PROMOTION_ELIGIBLE,
        "charter": {
            "path": ".omx/tmp/codex_runs/et4_prompt.md",
            "common_contract": ".omx/tmp/codex_runs/_common_contract.md",
            "main_override_recorded_verbatim": (
                "et3 measured eta 0.3562364 (2.083x bar 0.1710048742) with pose ratios "
                "0.8128/1.0031/1.1284 (min/med/max) and correctly withheld under its "
                "pre-registered per-pair-max guard. MAIN override BY S-ARITHMETIC "
                "(m67 pace-vs-direction, m52 never-binary-judgment): net seg+rate win ~= "
                "0.3562*0.18039 - 0.0308 ~= -0.0335 S vs worst-case pose cost (max ratio "
                "applied to ALL pairs) ~= +0.0053 S — 6x margin even at the impossible tail; "
                "the n600 MEAN pose ratio is the real quantity and THIS RUN MEASURES IT. "
                "Subset caveat honored: the n=32 set's pose behavior may not project "
                "(m96 axis law) — which is an argument FOR the measurement, not against it."
            ),
        },
        "selection": {
            "mode": "full_population_ordered_pair_ids" if len(pairs) == N_PAIRS_TOTAL else "bounded_chunk",
            "requested_pairs": pairs,
            "requested_count": len(pairs),
            "population": N_PAIRS_TOTAL,
            "chunk_max_contract": 120,
        },
        "parent": {
            "archive": str(args.parent_archive),
            "archive_sha256": parent_archive_sha,
            "archive_bytes": int(args.parent_archive.stat().st_size),
            "expected_sha256": BASELINE_ARCHIVE_SHA256,
            "not_tq1c_base": "tq1c_base" not in str(args.parent_archive),
            "baseline_S": BASELINE_S,
            "baseline_d_seg": BASELINE_D_SEG,
            "baseline_d_pose": BASELINE_D_POSE,
        },
        "mechanism": {
            "solve": "SW1 solve-within null-basis, delta=Nc, c-space objective",
            "realizer": "DK1 CVP/Babai private-support integer realizer, full requested mask per pair",
            "runner_precedent": "ET3 commit 6484a51de6",
            "no_reduced_form": True,
        },
        "vo2_element_grade_vector": vo2_element_grade_vector(args, rows),
        "storage_preflight": storage,
        "memory_preflight": memory,
        "completed_pairs": completed,
        "remaining_pairs_in_request": [pair for pair in pairs if pair not in set(completed)],
        "rows_path": str(args.rows_path),
        "patch_dir": str(args.bulk_dir / "patch_records"),
        "aggregate": aggregate_rows(rows, archive_bytes=archive_bytes),
        "archive": archive_receipt,
        "recall_evidence": {
            "sources_read": [
                ".omx/tmp/codex_runs/et4_prompt.md",
                ".omx/tmp/codex_runs/_common_contract.md",
                "PROGRAM.md",
                "CLAUDE.md",
                "AGENTS.md",
                "docs/operating_manual_craft_handoff.md",
                ".omx/state/main_hot_state.md",
                ".omx/research/ddm_et3_20260806/RECEIPT.md",
                ".omx/research/ddm_sw1_20260806/RECEIPT.md",
                ".omx/research/ddm_dk1_20260806/RECEIPT.md",
                ".omx/research/ddm_rw2_20260806/RECEIPT.md",
                ".omx/research/ddm_vo2_20260806/RECEIPT.md",
                "experiments/ddm_et3_solve_within_cvp_phase_field.py",
                "experiments/ddm_tq1_optimal_token_edit.py",
                "src/tac/submission_chain.py",
            ],
            "queries": [
                "et4|solve-within+CVP|DK1 CVP|phase field sidecar|image-domain correction|overlay|delta sidecar",
                "phase_b_realized|candidate_archives|move_0023|current_offsets|block16_offsets|receipt-bytes",
            ],
            "beyond_charter_findings_that_changed_plan": [
                "ET3 did not persist receiver-consumable frame deltas; ET4 adds per-pair patch records before any byte-close claim.",
                "No existing ET4 sidecar grammar was found in the searched scope; ET4 uses an explicit counted overlay grammar rather than pretending image-domain deltas are IX2 token edits.",
                "TQ1 proves the parent archive restages through the IX2 payload and the qo1 runtime decoder; ET4 wraps that decoder and applies counted frame_1 patches.",
                "RW2/VO2 constrain the DK1 claim to bounded finite kept-set optimality, so ET4 carries exact_declared_scope counts and no global MIQP claim.",
            ],
        },
        "boundaries": [
            "macOS CPU advisory only unless run_upstream_evaluate returns a contest axis on contest hardware",
            "score_claim=false in all local receipts",
            "no pointer promotion; MAIN adjudicates pointer updates",
            "DK1 CVP exactness is declared finite kept-scope only",
            "archive score remains absent until all 600 patches exist and the staged archive is evaluated",
        ],
    }


def write_summaries(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    write_json_atomic(args.summary_path, summary)
    write_json_atomic(args.receipt_summary_path, summary)


def read_parent_payload(parent_archive: Path) -> bytes:
    with zipfile.ZipFile(parent_archive) as zf:
        names = zf.namelist()
        if names != ["0.bin"]:
            raise RuntimeError(f"parent archive member list differs: {names}")
        return zf.read("0.bin")


def prepare_runtime_source(args: argparse.Namespace) -> dict[str, Any]:
    runtime_dir = args.bulk_dir / "runtime_src"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []

    def copy_file(src: Path, dst_name: str) -> None:
        dst = runtime_dir / dst_name
        shutil.copy2(src, dst)
        if dst.name.endswith(".sh"):
            dst.chmod(0o755)
        copied.append(
            {
                "name": dst_name,
                "source": str(src),
                "sha256": sha256_file(dst),
                "bytes": dst.stat().st_size,
            }
        )

    copy_file(REPO / "experiments/ddm_et4_overlay_inflate.sh", "inflate.sh")
    copy_file(REPO / "experiments/ddm_et4_overlay_inflate_runner.py", "inflate_runner.py")
    copy_file(REPO / "experiments/ddm_et4_overlay_codec.py", "ddm_et4_overlay_codec.py")
    copy_file(args.base_runtime / "inflate_runner.py", "base_inflate_runner.py")
    for name in BASE_RUNTIME_FILES:
        copy_file(args.base_runtime / name, name)
    return {
        "schema": "ddm_et4_runtime_src_receipt.v1",
        "runtime_src": str(runtime_dir),
        "runtime_files": copied,
    }


def build_archive(args: argparse.Namespace, rows: list[dict[str, Any]], parent_archive_sha: str) -> dict[str, Any]:
    if len(rows) != N_PAIRS_TOTAL:
        raise RuntimeError(f"cannot byte-close ET4 archive before n600 completion: {len(rows)}/600 rows")
    patch_paths = [patch_path(args, pair) for pair in range(N_PAIRS_TOTAL)]
    missing = [str(path) for path in patch_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot byte-close ET4 archive; missing patch records: {missing[:8]}")
    patch_records = [load_patch_record(path) for path in patch_paths]
    compressed_patch, patch_receipt = overlay_codec.encode_patch_records(
        patch_records,
        quality=args.patch_brotli_quality,
    )
    parent_payload = read_parent_payload(args.parent_archive)
    metadata = {
        "schema": "ddm_et4_overlay_metadata.v1",
        "parent_archive_sha256": parent_archive_sha,
        "parent_archive_bytes": int(args.parent_archive.stat().st_size),
        "parent_payload_sha256": overlay_codec.sha256_bytes(parent_payload),
        "rows_path": str(args.rows_path),
        "rows_sha256": sha256_file(args.rows_path),
        "axis": AXIS,
        "score_claim": False,
    }
    payload, payload_receipt = overlay_codec.encode_overlay_payload(
        parent_payload=parent_payload,
        compressed_patch=compressed_patch,
        metadata=metadata,
    )
    runtime_receipt = prepare_runtime_source(args)
    submission_dir = args.bulk_dir / "submission"
    archive_path = stage_submission(
        payload,
        dest=submission_dir,
        runtime_src=runtime_receipt["runtime_src"],
        runtime_files=STAGED_RUNTIME_FILES,
        payload_member="0.bin",
    )
    archive_sha = sha256_file(archive_path)
    archive_receipt: dict[str, Any] = {
        "schema": "ddm_et4_byteclose_archive_receipt.v1",
        "archive": {
            "path": str(archive_path),
            "archive_bytes": int(archive_path.stat().st_size),
            "archive_sha256": archive_sha,
        },
        "payload": payload_receipt,
        "patch": patch_receipt,
        "runtime": runtime_receipt,
        "submission_dir": str(submission_dir),
        "score_claim": False,
        "axis": AXIS,
    }
    if args.run_inflate:
        chain_paths = ChainPaths.from_env(
            repo_root=REPO,
            runtime_src=runtime_receipt["runtime_src"],
            work_dir=args.bulk_dir / "submission_chain_work",
        )
        missing = chain_paths.preflight(require_eval_inputs=True)
        if missing:
            raise RuntimeError(f"submission chain preflight missing paths: {missing}")
        inflated_dir = submission_dir / "inflated"
        inflate_result = run_inflate(
            submission_dir,
            archive_dir=submission_dir,
            out_dir=inflated_dir,
            video_names_file=chain_paths.video_names_file,
            timeout=args.inflate_timeout_s,
        )
        archive_receipt["inflate"] = inflate_result.__dict__
    if args.run_evaluate:
        if "inflate" not in archive_receipt:
            raise RuntimeError("--run-evaluate requires --run-inflate")
        chain_paths = ChainPaths.from_env(repo_root=REPO, work_dir=args.bulk_dir / "submission_chain_work")
        eval_result = run_upstream_evaluate(
            submission_dir,
            upstream_dir=chain_paths.upstream_dir,
            videos_dir=chain_paths.videos_dir,
            video_names_file=chain_paths.video_names_file,
            archive_bytes=int(archive_path.stat().st_size),
            device=args.eval_device,
            batch_size=args.eval_batch_size,
            num_threads=args.eval_threads,
            timeout=args.eval_timeout_s,
            report_path=submission_dir / "et4_evaluate_report.txt",
        )
        archive_receipt["evaluate"] = eval_result.__dict__
    write_json_atomic(args.bulk_dir / "byteclose_archive_receipt.json", archive_receipt)
    write_json_atomic(args.receipt_dir / "byteclose_archive_receipt.json", archive_receipt)
    return archive_receipt


def execute(args: argparse.Namespace) -> int:
    args.rows_path = args.bulk_dir / "et4_solve_within_cvp_rows.jsonl"
    args.summary_path = args.bulk_dir / "et4_solve_within_cvp_summary.json"
    args.receipt_summary_path = args.receipt_dir / "et4_solve_within_cvp_summary.json"
    args.bulk_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    storage = storage_preflight(args)
    if not storage["passes"]:
        raise RuntimeError(f"storage preflight failed: {storage}")
    memory = memory_preflight()
    parent_archive_sha = sha256_file(args.parent_archive)
    if parent_archive_sha != BASELINE_ARCHIVE_SHA256:
        raise RuntimeError(f"parent archive SHA drifted: {parent_archive_sha}")
    if "tq1c_base" in str(args.parent_archive):
        raise RuntimeError("ET4 parent must be b35e756829 frontier parent, not tq1c_base")
    if int(args.parent_archive.stat().st_size) != BASELINE_BYTES:
        raise RuntimeError(f"parent archive bytes drifted: {args.parent_archive.stat().st_size}")

    rows = load_jsonl(args.rows_path) if args.resume else []
    done = {int(row["pair"]) for row in rows}
    for pair in done:
        if not patch_path(args, pair).exists():
            raise RuntimeError(f"row for pair {pair} exists but patch record is missing")
    summary = build_summary(
        args,
        rows,
        parent_archive_sha=parent_archive_sha,
        storage=storage,
        memory=memory,
    )
    write_summaries(args, summary)

    pairs = pair_list(args)
    todo = [pair for pair in pairs if pair not in done]
    print(
        f"[et4] ready rows={len(rows)} remaining={len(todo)} request={pairs[:4]}...{pairs[-4:] if pairs else []}",
        flush=True,
    )
    if args.prepare_only:
        return 0

    raw = raw_memmap(args.parent_raw)
    parent_lstars = np.load(args.parent_argmax, mmap_mode="r")
    current_offsets = np.load(args.current_offsets, mmap_mode="r")
    gt_labels = open_stored_npy_memmap(args.gt_cache, "lstars")
    parent_score = json.loads(args.parent_score.read_text())
    basis_np, basis_cert = null_coordinate_basis()
    basis_t = torch.from_numpy(basis_np.astype(np.float32))
    constraint_np = pose_constraint_matrix()
    segnet, posenet, _scorer_custody = load_models(args.upstream_root, threads=args.threads)

    wanted: set[int] = set()
    for pair in todo:
        wanted.update({seq_len * pair, seq_len * pair + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted) if wanted else {}
    caps = parse_cap_ladder(args.cap_ladder, fallback=args.steps)

    for pair in todo:
        started = time.time()
        dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]]).astype(np.uint8)
        cells, pose_base = forward(segnet, posenet, dec[None])
        gt_cells, pose_gt_all = forward(segnet, posenet, gt[None])
        lstar = cells[0]
        lgt = gt_cells[0]
        pose_gt = pose_gt_all[0]
        cached_parent = np.asarray(parent_lstars[pair])
        cached_gt = np.asarray(gt_labels[pair], dtype=np.uint8)
        if not np.array_equal(lstar, cached_parent):
            raise RuntimeError(f"C2 failed for pair {pair}: decoded parent argmax != cached parent")
        if not np.array_equal(lgt, cached_gt):
            raise RuntimeError(f"C3 failed for pair {pair}: canonical GT decode argmax != GT cache")

        target = translate_blocks(lstar, np.asarray(current_offsets[pair]), args.block)
        band = target != lstar
        snapped = snap_band_to_blocks(band)
        block_mask = block_mask_from_band(snapped)
        flips0_map = lstar != lgt
        flips0 = int(flips0_map.sum())
        label_after = target != lgt
        label_ceiling_net_fixed = flips0 - int(label_after.sum())
        d_pose_before = float(np.square(pose_base[0] - pose_gt).sum() / 6.0)
        base = resize_to_scorer(dec[1])
        base_sc_u8 = torch.round(base)[0].permute(1, 2, 0).numpy().astype(np.uint8)

        with torch.enable_grad():
            sal = compute_margin_saliency_map(
                segnet,
                torch.from_numpy(np.ascontiguousarray(dec[1])).permute(2, 0, 1).float(),
                flip_pixel_mask=torch.from_numpy(snapped.astype(bool)),
            )
        weights, weight_stats = metric_weights_from_saliency(
            sal.saliency.cpu().numpy(),
            snapped,
            lambda_saliency=args.lambda_saliency,
            outside_weight=args.outside_weight,
            clip=args.saliency_clip,
        )

        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for cap in caps:
            delta_hwc, diag = solve_within_null_basis_delta(
                segnet,
                dec[1],
                gt[1],
                target,
                block_mask,
                weights,
                basis_t,
                constraint_np,
                steps=int(cap),
                lr=args.lr,
                eval_every=args.eval_every,
                convergence_patience_evals=args.convergence_patience_evals,
                convergence_min_improvement=args.convergence_min_improvement,
            )
            selected = diag["selected"]
            cap_receipt = cap_receipt_from_diagnostics(diag, cap=int(cap))
            attempt = {
                "cap": int(cap),
                "proxy_phase_target_flips": int(selected["best_proxy_phase_target_flips"]),
                "selected": selected,
                "cap_stop_receipt": cap_receipt,
            }
            attempts.append(attempt)
            if best is None or attempt["proxy_phase_target_flips"] < best["attempt"]["proxy_phase_target_flips"]:
                best = {"attempt": attempt, "delta_hwc": delta_hwc, "diagnostics": diag}
            if cap_receipt["stop_reason"] == "converged":
                break
        if best is None:
            raise RuntimeError("cap ladder produced no solve-within result")

        cam_cvp, cvp_receipt = realize_cvp_delta(
            camera_frame=dec[1],
            target_delta_hwc=best["delta_hwc"],
            block_mask=block_mask,
            metric_weights_hw=weights,
            cvp_tap_radius=args.cvp_tap_radius,
            cvp_max_channel_candidates=args.cvp_max_channel_candidates,
            cvp_max_pixel_candidates=args.cvp_max_pixel_candidates,
            cvp_max_combinations=args.cvp_max_combinations,
        )
        cvp_lam, _cvp_pose, scored = score_pair(
            segnet=segnet,
            posenet=posenet,
            dec_f0=dec[0],
            cam_f1=cam_cvp,
            pose_gt=pose_gt,
            lgt=lgt,
            flips0_map=flips0_map,
        )
        d_pose_after = float(scored["d_pose_after"])
        pose_sse_delta = float((d_pose_after - d_pose_before) * 6.0)
        seg_delta_s = (int(scored["flips_after"]) - flips0) * (
            100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)
        )
        pose_delta_s = math.sqrt(
            10.0 * (float(parent_score["d_pose"]) + pose_sse_delta / (N_PAIRS_TOTAL * 6))
        ) - math.sqrt(10.0 * float(parent_score["d_pose"]))
        cvp_scorer_u8 = torch.round(resize_to_scorer(cam_cvp))[0].permute(1, 2, 0).numpy().astype(np.uint8)
        patch_record = overlay_codec.frame1_delta_record(pair, dec[1], cam_cvp)
        save_patch_record(patch_path(args, pair), patch_record)
        rec = {
            "schema": "ddm_et4_solve_within_cvp_pair.v1",
            "pair": int(pair),
            "flips_before": flips0,
            "label_ceiling_flips_left": int(label_after.sum()),
            "label_ceiling_net_fixed": label_ceiling_net_fixed,
            "label_ceiling_fixed": int((flips0_map & ~label_after).sum()),
            "label_ceiling_broken": int(((~flips0_map) & label_after).sum()),
            "band_px": int(band.sum()),
            "band_snapped_px": int(snapped.sum()),
            "band_snap_tax": float(snapped.sum() / max(1, band.sum())),
            "d_pose_before": d_pose_before,
            "C_before": confusion(lgt, lstar).tolist(),
            "metric_weights": weight_stats,
            "controls": {
                "C2_parent_argmax_matches_cache": True,
                "C3_gt_argmax_matches_cache": True,
                "offset_shape": list(np.asarray(current_offsets[pair]).shape),
                "null_basis_max_abs_A_times_N": basis_cert["max_abs_A_times_N"],
            },
            "solve_within": {
                "cap_ladder_attempts": attempts,
                "selected_attempt": best["attempt"],
                "diagnostics": best["diagnostics"],
            },
            "cvp_realized": {
                **scored,
                "eta_realized": (
                    (flips0 - int(scored["flips_after"])) / label_ceiling_net_fixed
                    if label_ceiling_net_fixed
                    else None
                ),
                "d_pose_before": d_pose_before,
                "d_pose_ratio": d_pose_after / d_pose_before if d_pose_before else None,
                "pose_sse_delta": pose_sse_delta,
                "seg_delta_S_no_rate": seg_delta_s,
                "pose_delta_S_against_parent": pose_delta_s,
                "joint_delta_S_no_rate_against_parent_pose": seg_delta_s + pose_delta_s,
                "changed_scorer_pixels": int((cvp_scorer_u8 != base_sc_u8).any(axis=2).sum()),
                "changed_scorer_channel_values": int((cvp_scorer_u8 != base_sc_u8).sum()),
                "yuv6_residual": yuv6_shift(base_sc_u8, cvp_scorer_u8),
                "realizer_receipt": cvp_receipt,
                "C_after": confusion(lgt, cvp_lam).tolist(),
            },
            "patch_record": {
                "path": str(patch_path(args, pair)),
                "nnz": int(patch_record["nnz"]),
                "before_sha256": patch_record["before_sha256"],
                "after_sha256": patch_record["after_sha256"],
                "delta_index_sha256": patch_record["delta_index_sha256"],
                "delta_value_sha256": patch_record["delta_value_sha256"],
            },
            "elapsed_s": time.time() - started,
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
            "promotion_eligible": PROMOTION_ELIGIBLE,
        }
        rows.append(rec)
        append_jsonl(args.rows_path, rec)
        summary = build_summary(
            args,
            rows,
            parent_archive_sha=parent_archive_sha,
            storage=storage,
            memory=memory,
        )
        write_summaries(args, summary)
        agg = summary["aggregate"]
        print(
            f"[et4] pair {pair:03d} rows={len(rows):03d} "
            f"eta_pair={rec['cvp_realized']['eta_realized']:+.4f} "
            f"pose={rec['cvp_realized']['d_pose_ratio']:.4f}x "
            f"agg_eta={agg.get('eta')} elapsed={rec['elapsed_s']:.1f}s",
            flush=True,
        )

    archive_receipt = None
    if args.build_archive:
        archive_receipt = build_archive(args, rows, parent_archive_sha)
    final = build_summary(
        args,
        rows,
        parent_archive_sha=parent_archive_sha,
        storage=storage,
        memory=memory,
        archive_receipt=archive_receipt,
    )
    write_summaries(args, final)
    print(json.dumps(final["aggregate"], indent=1, default=jsonable), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-raw", type=Path, default=DEFAULT_PARENT_RAW)
    ap.add_argument("--parent-archive", type=Path, default=DEFAULT_PARENT_ARCHIVE)
    ap.add_argument("--parent-argmax", type=Path, default=DEFAULT_PARENT_ARGMAX)
    ap.add_argument("--parent-score", type=Path, default=DEFAULT_PARENT_SCORE)
    ap.add_argument("--current-offsets", type=Path, default=DEFAULT_OFFSETS)
    ap.add_argument("--phase-field-summary", type=Path, default=DEFAULT_PHASE_FIELD)
    ap.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    ap.add_argument("--gt-mkv", type=Path, default=DEFAULT_GT_MKV)
    ap.add_argument("--upstream-root", type=Path, default=REPO / "upstream")
    ap.add_argument("--sw1-summary", type=Path, default=DEFAULT_SW1_SUMMARY)
    ap.add_argument("--base-runtime", type=Path, default=DEFAULT_BASE_RUNTIME)
    ap.add_argument("--bulk-dir", type=Path, default=DEFAULT_BULK_DIR)
    ap.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--cap-ladder", default="15")
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--convergence-patience-evals", type=int, default=0)
    ap.add_argument("--convergence-min-improvement", type=int, default=1)
    ap.add_argument("--lambda-saliency", type=float, default=1.0)
    ap.add_argument("--outside-weight", type=float, default=0.02)
    ap.add_argument("--saliency-clip", type=float, default=20.0)
    ap.add_argument("--cvp-tap-radius", type=int, default=0)
    ap.add_argument("--cvp-max-channel-candidates", type=int, default=9)
    ap.add_argument("--cvp-max-pixel-candidates", type=int, default=16)
    ap.add_argument("--cvp-max-combinations", type=int, default=250000)
    ap.add_argument("--pair-start", type=int, default=0)
    ap.add_argument("--pair-stop", type=int, default=N_PAIRS_TOTAL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--patch-brotli-quality", type=int, default=11)
    ap.add_argument("--build-archive", action="store_true")
    ap.add_argument("--run-inflate", action="store_true")
    ap.add_argument("--run-evaluate", action="store_true")
    ap.add_argument("--inflate-timeout-s", type=int, default=3600)
    ap.add_argument("--eval-timeout-s", type=int, default=24 * 3600)
    ap.add_argument("--eval-device", default="cpu")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--eval-threads", type=int, default=2)
    args = ap.parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if not (0 <= args.pair_start <= args.pair_stop <= N_PAIRS_TOTAL):
        raise SystemExit("--pair-start/--pair-stop must be inside 0..600")
    if args.pair_stop - args.pair_start > 120 and not args.limit and not args.build_archive:
        raise SystemExit("common contract chunk <=120: use --pair-stop within 120 pairs or --limit")
    if not (0 <= args.patch_brotli_quality <= 11):
        raise SystemExit("--patch-brotli-quality must be in [0,11]")
    return args


def main(argv: list[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
