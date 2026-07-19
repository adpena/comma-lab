#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the d_seg>0 range-payload secant on bounded real-cache chunks.

Each measurement invocation is capped at twelve pairs and checkpoints every
pair/operating-point row.  Two receipts compose into the preregistered n24
curve.  This tool is local advisory measurement only; it never evaluates or
changes a contest pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _path in (REPO, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.joint_seg_pose_rate import (  # noqa: E402
    JointSolveError,
    generated_fill_predictor,
    solve_measured_waterfill,
)
from tac.optimization.seg_secant_rd_curve import (  # noqa: E402
    BREAK_EVEN_BYTES_PER_DSEG,
    OperatingPoint,
    SegSecantError,
    adjacent_seg_secants,
    default_operating_points,
    margin_ordered_abandonment,
    measure_parseback_payload,
    summarize_per_class,
    truncate_preimage_residual_precision,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tools.measure_joint_seg_pose_rate import (  # noqa: E402
    _hard_verdict,
    _load_cache,
    _load_scorers,
)
from tools.measure_uint8_lattice_feasibility import (  # noqa: E402
    _sha256_file,
    _stat_tree_snapshot,
)

SCHEMA = "seg_secant_rd_curve_chunk.v1"
COMPOSED_SCHEMA = "seg_secant_rd_curve_composed.v1"
STATE_SCHEMA = "seg_secant_rd_curve_state.v1"
STAGE_SCHEMA = "seg_secant_rd_curve_pair_point_stage.v1"
MAX_SUBSET = 12
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
SEED = 20260719
POSE_CROSSOVER = 2.5e-4
POINTER = "0.1910828242 [contest-CPU Linux x86_64] UNMOVED"
DEFAULT_CACHE = Path(
    "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
)
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
SACRED = Path(
    "/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z"
)
DEFAULT_EVIDENCE_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/evidence/seg_secant_20260719"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(_canonical(value) + b"\n")
    os.replace(tmp, path)


def _write_once_json(path: Path, value: Any) -> dict[str, Any]:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SegSecantError(f"immutable stage differs on resume: {path}")
    else:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise SegSecantError(f"immutable stage appeared concurrently: {path}") from exc
        finally:
            tmp.unlink(missing_ok=True)
    return {"path": str(path.resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _pair_ids(explicit: Sequence[int] | None) -> list[int]:
    pairs = [] if explicit is None else [int(value) for value in explicit]
    if not pairs or len(pairs) > MAX_SUBSET or len(set(pairs)) != len(pairs):
        raise SegSecantError(f"pair selection must have 1..{MAX_SUBSET} unique ids")
    if any(pair < 0 or pair >= 600 for pair in pairs):
        raise SegSecantError("pair ids must be in [0,600)")
    return pairs


def _ownership_indices(operator: DisjointResizeOperator) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray([support.indices for support in operator.row_supports], dtype=np.int64)
    cols = np.asarray([support.indices for support in operator.col_supports], dtype=np.int64)
    if rows.shape != (SCORER_HW[0], 2) or cols.shape != (SCORER_HW[1], 2):
        raise SegSecantError("canonical resize ownership is not disjoint factor-2 geometry")
    return rows, cols


def _load_margin_custody(
    manifest_path: Path,
    pairs: Sequence[int],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    document = manifest_path / "manifest.json" if manifest_path.is_dir() else manifest_path
    try:
        manifest = json.loads(document.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SegSecantError("VJP custody manifest is unreadable") from exc
    manifest_pair_ids = [int(value) for value in manifest.get("pair_ids", [])]
    selected_in_manifest_order = [pair_id for pair_id in manifest_pair_ids if pair_id in pairs]
    if selected_in_manifest_order != list(pairs):
        raise SegSecantError("VJP custody does not contain selected pair ids in order")
    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, list) or len(sidecars) != len(manifest_pair_ids):
        raise SegSecantError("VJP custody sidecar list is incomplete")
    rows: dict[int, dict[str, np.ndarray]] = {}
    refs: list[dict[str, Any]] = []
    for reference in sidecars:
        pair_id = int(reference.get("pair_id", -1))
        if pair_id not in pairs:
            continue
        path = Path(str(reference.get("path", ""))).resolve()
        if not path.is_file():
            raise SegSecantError("VJP custody sidecar path/pair mismatch")
        if path.stat().st_size != reference.get("bytes") or _sha256_file(path) != reference.get("sha256"):
            raise SegSecantError("VJP custody sidecar byte hash mismatch")
        try:
            with np.load(path, allow_pickle=False) as data:
                native_margin = np.asarray(data["native_margin"], dtype=np.float32).copy()
                cached_margin = np.asarray(data["cached_margin"], dtype=np.float32).copy()
                winner = np.asarray(data["winner"]).copy()
                embedded_pair = int(np.asarray(data["pair_id"]).reshape(()))
        except (OSError, ValueError, KeyError) as exc:
            raise SegSecantError("VJP margin custody arrays are unreadable") from exc
        tensor_hashes = reference.get("tensor_hashes", {})
        if (
            embedded_pair != pair_id
            or native_margin.shape != SCORER_HW
            or cached_margin.shape != SCORER_HW
            or winner.shape != SCORER_HW
            or _sha256_array(native_margin) != tensor_hashes.get("native_margin")
            or _sha256_array(cached_margin) != tensor_hashes.get("cached_margin")
            or _sha256_array(winner) != tensor_hashes.get("winner")
        ):
            raise SegSecantError("VJP margin tensor custody mismatch")
        if not np.isfinite(native_margin).all() or np.any(native_margin < 0):
            raise SegSecantError("VJP native margins leave their valid domain")
        rows[pair_id] = {
            "native_margin": native_margin,
            "cached_margin": cached_margin,
            "winner": winner.astype(np.int64),
        }
        refs.append(
            {
                "pair_id": pair_id,
                "path": str(path),
                "bytes": int(reference["bytes"]),
                "sha256": str(reference["sha256"]),
                "native_margin_sha256": str(tensor_hashes["native_margin"]),
            }
        )
    if list(rows) != list(pairs):
        raise SegSecantError("VJP custody selected sidecar coverage is incomplete")
    return rows, {
        "manifest": str(document.resolve()),
        "manifest_sha256": _sha256_file(document),
        "active_arrangement": manifest.get("active_arrangement"),
        "sidecars": refs,
    }


def _validate_resume_row(row: Mapping[str, Any], config_sha256: str) -> None:
    stage = row.get("stage")
    if not isinstance(stage, Mapping):
        raise SegSecantError("resume row lacks immutable stage reference")
    path = Path(str(stage.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != stage.get("bytes")
        or _sha256_file(path) != stage.get("sha256")
    ):
        raise SegSecantError("resume stage byte custody mismatch")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != STAGE_SCHEMA
        or payload.get("config_sha256") != config_sha256
        or payload.get("pair_id") != row.get("pair_id")
        or payload.get("point_id") != row.get("point_id")
    ):
        raise SegSecantError("resume stage identity mismatch")


def _aggregate_class(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pixels": 0, "mismatches": 0})
    for row in rows:
        for class_id, values in row["per_class_d_seg"].items():
            counts[class_id]["pixels"] += int(values["pixels"])
            counts[class_id]["mismatches"] += int(values["mismatches"])
    return {
        class_id: {
            **values,
            "d_seg_conditional": (
                None if values["pixels"] == 0 else values["mismatches"] / values["pixels"]
            ),
        }
        for class_id, values in sorted(counts.items(), key=lambda item: int(item[0]))
    }


def _summarize_point(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    point = dict(first["operating_point"])
    return {
        **point,
        "pair_count": len(rows),
        "unique_pair_ids": sorted(int(row["pair_id"]) for row in rows),
        "brotli_q11_bytes_per_pair": float(
            np.mean([row["rate"]["brotli_q11_bytes"] for row in rows])
        ),
        "zstd_19_bytes_per_pair": float(
            np.mean([row["rate"]["zstd_19_bytes"] for row in rows])
        ),
        "d_seg": float(np.mean([row["hard_oracle"]["d_seg"] for row in rows])),
        "d_pose": float(np.mean([row["hard_oracle"]["d_pose"] for row in rows])),
        "seg_mismatched_pixels": int(
            sum(row["hard_oracle"]["seg_mismatched_pixels"] for row in rows)
        ),
        "per_class_d_seg": _aggregate_class(rows),
        "repair_count": int(sum(row["repair_count"] for row in rows)),
        "reject_count": int(sum(row["reject_count"] for row in rows)),
        "pose_inactive_count": int(sum(row["pose_constraint"]["inactive"] for row in rows)),
        "pose_violation_count": int(sum(row["pose_constraint"]["violated"] for row in rows)),
        "pose_constraint_tau": POSE_CROSSOVER,
        "bytes_scope": "mean two-frame signed-int32 range residual payload; no archive/header claim",
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    state = args.state.resolve()
    stage_dir = args.stage_dir.resolve()
    cache = args.cache.resolve()
    upstream = args.upstream.resolve()
    custody_manifest = args.vjp_manifest.resolve()
    for path in (output, state, stage_dir):
        if str(path).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
            raise SegSecantError("durable evidence paths may not use a temporary directory")
    if output.exists():
        raise SegSecantError(f"receipt already exists: {output}")
    pairs = _pair_ids(args.pair_indices)
    points = default_operating_points()
    sacred_before = _stat_tree_snapshot(SACRED)
    fields = _load_cache(cache)
    custody, custody_ref = _load_margin_custody(custody_manifest, pairs)
    config = {
        "schema": STATE_SCHEMA,
        "seed": SEED,
        "pairs": pairs,
        "operating_points": [point.as_dict() for point in points],
        "cache": str(cache),
        "cache_sha256": _sha256_file(cache),
        "vjp_custody": custody_ref,
        "upstream": str(upstream),
        "cpu_threads": int(args.cpu_threads),
        "tool_sha256": _sha256_file(Path(__file__).resolve()),
        "module_sha256": _sha256_file(SRC / "tac/optimization/seg_secant_rd_curve.py"),
        "composed_hard_oracle_tool_sha256": _sha256_file(
            REPO / "tools/measure_joint_seg_pose_rate.py"
        ),
        "predictor": "generated piecewise-constant fill of counted scorer-plane source description",
        "frame0_policy": "source frame retained for all points",
        "pose_constraint_tau": POSE_CROSSOVER,
    }
    config_sha = hashlib.sha256(_canonical(config)).hexdigest()
    rows: list[dict[str, Any]] = []
    if args.resume:
        if not state.is_file():
            raise SegSecantError("--resume requires an existing state file")
        loaded = json.loads(state.read_text())
        if loaded.get("config_sha256") != config_sha:
            raise SegSecantError("resume config/custody hash mismatch")
        rows = list(loaded.get("rows", []))
        for row in rows:
            _validate_resume_row(row, config_sha)
    elif state.exists() or (stage_dir.exists() and any(stage_dir.iterdir())):
        raise SegSecantError("preserved state/stages exist; use --resume or new paths")
    else:
        _atomic_json(
            state,
            {"schema": STATE_SCHEMA, "config_sha256": config_sha, "config": config, "rows": []},
        )

    completed = {(int(row["pair_id"]), str(row["point_id"])) for row in rows}
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0],
        camera_w=CAMERA_HW[1],
        scorer_h=SCORER_HW[0],
        scorer_w=SCORER_HW[1],
    )
    ownership_rows, ownership_cols = _ownership_indices(operator)
    segnet, posenet, torch = _load_scorers(upstream, args.cpu_threads)

    for pair_id in pairs:
        source0 = np.asarray(fields["gt_f0"][pair_id], dtype=np.uint8).copy()
        source1 = np.asarray(fields["gt_f1"][pair_id], dtype=np.uint8).copy()
        labels = np.asarray(fields["lstars"][pair_id], dtype=np.int64).copy()
        cached_margins = np.asarray(fields["margins"][pair_id], dtype=np.float32).copy()
        target_pose = np.asarray(fields["gt_poses"][pair_id], dtype=np.float64).copy()
        source_verdict, source_winner, _source_rival = _hard_verdict(
            segnet, posenet, torch, source0, source1, labels, target_pose
        )
        margin_custody = custody[pair_id]
        if not np.array_equal(margin_custody["cached_margin"], cached_margins):
            raise SegSecantError(f"pair {pair_id} VJP cached margin differs from real cache")
        if not np.array_equal(margin_custody["winner"], source_winner):
            raise SegSecantError(f"pair {pair_id} VJP winner differs from frozen source oracle")
        n0, den0 = operator.apply_numerators(source0)
        n1, den1 = operator.apply_numerators(source1)
        if den0 != den1:
            raise SegSecantError("source frame numerator denominators differ")
        y0 = n0.astype(np.float64) / den0
        y1 = n1.astype(np.float64) / den1
        predictor0 = generated_fill_predictor(operator, y0)
        predictor1 = generated_fill_predictor(operator, y1)
        predictor_num0, predictor_den0 = operator.apply_numerators(predictor0)
        predictor_num1, predictor_den1 = operator.apply_numerators(predictor1)
        if predictor_den0 != den0 or predictor_den1 != den1:
            raise SegSecantError("predictor/source numerator denominator mismatch")
        frame0_rate = measure_parseback_payload(n0, predictor_num0)

        all_points = (OperatingPoint("source_reference", "reference", "none", 0), *points)
        for point in all_points:
            key = (pair_id, point.point_id)
            if key in completed:
                continue
            if point.family == "reference":
                candidate1 = source1.copy()
                transform = {"identity_reference": True}
                verdict = source_verdict
                predicted = source_winner
            elif point.family == "margin_abandonment":
                candidate1, transform = margin_ordered_abandonment(
                    source1,
                    predictor1,
                    margin_custody["native_margin"],
                    row_indices=ownership_rows,
                    col_indices=ownership_cols,
                    threshold=float(point.parameter_value),
                )
                verdict, predicted, _rival = _hard_verdict(
                    segnet, posenet, torch, source0, candidate1, labels, target_pose
                )
            elif point.family == "precision_truncation":
                candidate1, transform = truncate_preimage_residual_precision(
                    source1,
                    predictor1,
                    drop_low_bits=int(point.parameter_value),
                )
                verdict, predicted, _rival = _hard_verdict(
                    segnet, posenet, torch, source0, candidate1, labels, target_pose
                )
            else:  # pragma: no cover - fixed preregistered table
                raise SegSecantError(f"unknown point family: {point.family}")
            chosen_num1, chosen_den1 = operator.apply_numerators(candidate1)
            if chosen_den1 != den1:
                raise SegSecantError("candidate/source numerator denominator mismatch")
            frame1_rate = measure_parseback_payload(chosen_num1, predictor_num1)
            hard_d_pose = float(verdict["d_pose"])
            stage_payload = {
                "schema": STAGE_SCHEMA,
                "config_sha256": config_sha,
                "pair_id": pair_id,
                "point_id": point.point_id,
                "operating_point": point.as_dict(),
                "transform": transform,
                "hard_oracle": verdict,
                "per_class_d_seg": summarize_per_class(labels, predicted),
                "pose_constraint": {
                    "tau": POSE_CROSSOVER,
                    "inactive": hard_d_pose < POSE_CROSSOVER,
                    "active": hard_d_pose == POSE_CROSSOVER,
                    "violated": hard_d_pose > POSE_CROSSOVER,
                    "slack": POSE_CROSSOVER - hard_d_pose,
                    "verdict_scope": "this pair and operating point only",
                },
                "rate": {
                    "frame0": frame0_rate,
                    "frame1": frame1_rate,
                    "brotli_q11_bytes": int(
                        frame0_rate["brotli_q11"]["bytes"]
                        + frame1_rate["brotli_q11"]["bytes"]
                    ),
                    "zstd_19_bytes": int(
                        frame0_rate["zstd_19"]["bytes"]
                        + frame1_rate["zstd_19"]["bytes"]
                    ),
                },
                "repair_count": 0,
                "reject_count": 0,
                "repair_reject_scope": (
                    "direct reachable preimage transform; no proposal filtering or hard-zero-Seg repair"
                ),
                "source_frame_sha256": _sha256_array(source1),
                "predictor_frame_sha256": _sha256_array(predictor1),
                "candidate_frame_sha256": _sha256_array(candidate1),
                "chosen_numerators_sha256": _sha256_array(chosen_num1),
                "reconstruction": (
                    "deterministic from frozen real cache, VJP margin custody, generated predictor, "
                    "and point config; camera frames are rebuildable and not persisted"
                ),
            }
            stage_path = stage_dir / point.point_id / f"pair_{pair_id:04d}.json"
            stage = _write_once_json(stage_path, stage_payload)
            row = {
                "pair_id": pair_id,
                "point_id": point.point_id,
                "operating_point": point.as_dict(),
                "hard_oracle": verdict,
                "per_class_d_seg": stage_payload["per_class_d_seg"],
                "pose_constraint": stage_payload["pose_constraint"],
                "rate": {
                    "brotli_q11_bytes": stage_payload["rate"]["brotli_q11_bytes"],
                    "zstd_19_bytes": stage_payload["rate"]["zstd_19_bytes"],
                },
                "repair_count": 0,
                "reject_count": 0,
                "stage": stage,
            }
            rows.append(row)
            _atomic_json(
                state,
                {
                    "schema": STATE_SCHEMA,
                    "config_sha256": config_sha,
                    "config": config,
                    "rows": rows,
                },
            )

    if _stat_tree_snapshot(SACRED) != sacred_before:
        raise SegSecantError("sacred result tree changed during measurement")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["point_id"])].append(row)
    measured_points = [
        _summarize_point(grouped[point.point_id])
        for point in (OperatingPoint("source_reference", "reference", "none", 0), *points)
    ]
    receipt = {
        "schema": SCHEMA,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": f"[{platform.system()}-{platform.machine()} CPU advisory real-cache] NON-PROMOTABLE",
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "verdict_scope": (
                "selected real n600-cache pairs; native-float32 CPU-Torch hard oracle; "
                "range-coordinate payload only; no receiver archive or contest-axis claim"
            ),
        },
        "config": config,
        "config_sha256": config_sha,
        "pair_count": len(pairs),
        "observation_count": len(rows),
        "measured_points": measured_points,
        "rows": rows,
        "resumability": {
            "state": str(state),
            "stage_dir": str(stage_dir),
            "checkpoint_interval": "every pair/operating-point",
            "all_stage_checkpoints_preserved": True,
        },
        "sacred_tree_unchanged": True,
    }
    _atomic_json(output, receipt)
    return receipt


def compose(receipts: Sequence[Path], output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise SegSecantError(f"composed output already exists: {output}")
    documents = [json.loads(path.resolve().read_text()) for path in receipts]
    if len(documents) < 2 or any(document.get("schema") != SCHEMA for document in documents):
        raise SegSecantError("composition requires at least two Seg-secant chunk receipts")
    for document in documents:
        for row in document.get("rows", []):
            _validate_resume_row(row, str(document.get("config_sha256", "")))
    rows = [row for document in documents for row in document["rows"]]
    identities = {(int(row["pair_id"]), str(row["point_id"])) for row in rows}
    if len(identities) != len(rows):
        raise SegSecantError("duplicate pair/point observation in composition")
    config_points = documents[0]["config"]["operating_points"]
    if any(document["config"]["operating_points"] != config_points for document in documents[1:]):
        raise SegSecantError("chunk operating-point grids differ")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["point_id"])].append(row)
    point_order = ["source_reference", *[point["point_id"] for point in config_points]]
    measured_points = [_summarize_point(grouped[point_id]) for point_id in point_order]
    unique_pairs = sorted({int(row["pair_id"]) for row in rows})
    if any(point["pair_count"] != len(unique_pairs) for point in measured_points):
        raise SegSecantError("composed curve lacks common-corpus coverage")
    delivered = [point for point in measured_points if point["family"] != "reference"]
    brotli_secants = adjacent_seg_secants(
        measured_points, codec_key="brotli_q11_bytes_per_pair"
    )
    zstd_secants = adjacent_seg_secants(
        measured_points, codec_key="zstd_19_bytes_per_pair"
    )
    seg_curve = [
        {"bytes": point["brotli_q11_bytes_per_pair"], "distortion": point["d_seg"]}
        for point in measured_points
    ]
    pose_curve = [
        {"bytes": point["brotli_q11_bytes_per_pair"], "distortion": point["d_pose"]}
        for point in measured_points
    ]
    waterfill = solve_measured_waterfill(seg_curve, pose_curve)
    pose_violations = sum(point["pose_violation_count"] for point in delivered)
    pose_observations = len(unique_pairs) * len(delivered)
    pose_inactive = sum(point["pose_inactive_count"] for point in delivered)
    result = {
        "schema": COMPOSED_SCHEMA,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": documents[0]["axis"],
        "authority": documents[0]["authority"],
        "source_receipts": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}
            for path in receipts
        ],
        "unique_pair_count": len(unique_pairs),
        "unique_pair_ids": unique_pairs,
        "observation_count": len(rows),
        "delivered_positive_point_count": sum(point["d_seg"] > 0 for point in delivered),
        "all_delivered_points_have_positive_d_seg": all(point["d_seg"] > 0 for point in delivered),
        "measured_points": measured_points,
        "adjacent_seg_secants": {
            "break_even_bytes_per_unit_d_seg": BREAK_EVEN_BYTES_PER_DSEG,
            "break_even_bytes_per_1e_minus_6_d_seg": BREAK_EVEN_BYTES_PER_DSEG * 1e-6,
            "objective_sign": (
                "for a move to higher d_seg and fewer bytes, accept the distortion iff "
                "bytes_saved/delta_d_seg exceeds break-even"
            ),
            "brotli_q11": brotli_secants,
            "zstd_19": zstd_secants,
        },
        "waterfill": waterfill,
        "pose_constraint": {
            "tau": POSE_CROSSOVER,
            "delivered_pair_point_observations": pose_observations,
            "inactive_count": pose_inactive,
            "violation_count": pose_violations,
            "all_inactive": pose_inactive == pose_observations,
            "verdict_scope": "delivered measured pair/point observations only",
        },
        "repair_reject": {
            "repair_count": sum(point["repair_count"] for point in delivered),
            "reject_count": sum(point["reject_count"] for point in delivered),
            "scope": "direct reachable preimage transforms; no zero-Seg admission filter",
        },
        "labels": {
            "MEASURED": [
                "Brotli-Q11 and zstd-19 parse-back payload bytes",
                "native-float32 frozen CPU-Torch d_seg/d_pose",
                "per-class mismatch counts",
            ],
            "DERIVED": [
                "adjacent delta-bytes/delta-d_seg secants",
                "two-term break-even and score deltas",
            ],
        },
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pair-indices", nargs="+", type=int)
    parser.add_argument("--vjp-manifest", type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.compose:
            result = compose(args.compose, args.output)
        else:
            if args.state is None or args.stage_dir is None or args.vjp_manifest is None:
                raise SegSecantError(
                    "measurement requires --state, --stage-dir, and --vjp-manifest"
                )
            result = measure(args)
    except (SegSecantError, JointSolveError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"output": str(args.output.resolve()), "schema": result["schema"]}, sort_keys=True))


if __name__ == "__main__":
    main()
