#!/usr/bin/env python3
"""Measure chart/object-first MDL member search on the exact v10 resize fibre.

This is a local macOS-CPU advisory tool.  It preserves one atomic stage per
pair, resumes by revalidating those stages, and never mutates the source raw
video or the canonical frontier pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.mdl_polytope_member import (  # noqa: E402
    DEFAULT_CALIBRATION_ROWS,
    MdlPolytopeMemberSolver,
    ProxyCalibration,
    fit_proxy,
    lawref_manifest,
    zlib9_bytes,
)
from tac.optimization.predict_project_schema import (  # noqa: E402
    parse_constraint_seed,
    serialize_constraint_seed,
)

SCHEMA = "mdl_polytope_member_measurement.v1"
STAGE_SCHEMA = "mdl_polytope_member_pair_stage.v2"
AXIS = "[macOS-CPU advisory] NON-PROMOTABLE"
POINTER = "0.1910828242 [contest-CPU] UNMOVED"
SEED = 1234
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
PAIR_COUNT = 600
RAW_BYTES = PAIR_COUNT * 2 * CAMERA_HW[0] * CAMERA_HW[1] * 3
DEFAULT_RAW = Path(
    "/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/"
    "capstone_submission/inflated/0.raw"
)
DEFAULT_TARGETS = Path(
    "/Users/adpena/Projects/pact/experiments/results/ot_offset_n600_modal_20260709/"
    "gt_n600_lstars_slim.npz"
)
DEFAULT_SEED = Path(
    "/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/seeds/"
    "seed_compose_b2_loose.ppcs"
)
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/evidence/mdl_member_20260721")
EXPECTED_RAW_SHA256 = "31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b"
EXPECTED_TARGETS_SHA256 = "e41718a047b77b9072828e72f8cbffa0f5ef7ddf462c7ef4329d997ace89de50"
EXPECTED_SEED_SHA256 = "a21dde38128bed7ff62860ef005b994b74202e0bd00a37d1df8824ee325e856b"


class MeasurementError(RuntimeError):
    """Fail-closed input, resume, scorer, or custody error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(_canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"invalid JSON stage {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"JSON stage {path} is not an object")
    return value


def _check_file(path: Path, expected_bytes: int | None, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise MeasurementError(f"required input is missing: {path}")
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise MeasurementError(f"input byte mismatch for {path}: {size} != {expected_bytes}")
    digest = _sha256(path)
    if digest != expected_sha256:
        raise MeasurementError(f"input SHA mismatch for {path}: {digest} != {expected_sha256}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def _storage_preflight(output_root: Path, max_pairs: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    required = max(64 << 20, max_pairs * 256 << 10)
    if usage.free < required:
        raise MeasurementError(
            f"storage preflight refused: {usage.free} free bytes < {required} required"
        )
    return {
        "path": str(output_root),
        "free_bytes": usage.free,
        "required_bytes": required,
        "passed": True,
        "policy": "atomic JSON stages; no copied source raw; changed frames retained only if needed",
    }


def _load_scorer(upstream: Path, threads: int) -> tuple[Any, Any, dict[str, Any]]:
    if threads < 1 or not (upstream / "modules.py").is_file():
        raise MeasurementError("native CPU-Torch scorer custody is unavailable")
    sys.path.insert(0, str(upstream))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    net = DistortionNet().eval().to("cpu")
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, torch, {
        "implementation": "upstream.modules.DistortionNet.native_cpu_torch",
        "modules_path": str(upstream / "modules.py"),
        "modules_sha256": _sha256(upstream / "modules.py"),
        "segnet_weights_path": str(Path(segnet_sd_path)),
        "segnet_weights_sha256": _sha256(Path(segnet_sd_path)),
        "posenet_weights_path": str(Path(posenet_sd_path)),
        "posenet_weights_sha256": _sha256(Path(posenet_sd_path)),
        "threads": threads,
        "seed": SEED,
        "deterministic_algorithms": True,
    }


def _oracle_batch32(
    net: Any,
    torch: Any,
    pairs: np.ndarray,
    target_cells: np.ndarray,
    target_poses: np.ndarray,
) -> list[dict[str, Any]]:
    if pairs.shape != (32, 2, *CAMERA_HW, 3):
        raise MeasurementError("hard oracle requires canonical batch32 geometry")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(pairs))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
    )
    with torch.inference_mode():
        logits = net.segnet(net.segnet.preprocess_input(tensor))
        argmax = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        pose_output = net.posenet(net.posenet.preprocess_input(tensor))
        pose_tensor = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        poses6 = pose_tensor[:, :6].cpu().numpy().astype(np.float64)
    return [
        {
            "d_seg": float(np.mean(argmax[index] != target_cells[index])),
            "d_pose": float(np.mean((poses6[index] - target_poses[index]) ** 2)),
            "argmax_sha256": _array_sha256(argmax[index]),
            "pose6": poses6[index].tolist(),
        }
        for index in range(32)
    ]


def _level_rows(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "level": level.name,
            "coder_bytes": level.coder_bytes,
            "delta_bytes_vs_previous": level.delta_bytes_vs_previous,
            "selected_groups": level.selected_groups,
            "exact_numerators_equal": level.exact_numerators_equal,
        }
        for level in result.levels
    ]


def _tile_decomposition(
    solver: MdlPolytopeMemberSolver,
    canonical: np.ndarray,
    selected: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_stratum: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sy, sx, cy, cx in solver.tile_slices():
        tile_labels = labels[sy, sx]
        class_id = str(solver.tile_class(tile_labels))
        stratum = solver.tile_stratum(tile_labels)
        base_bytes = zlib9_bytes(canonical[cy, cx])
        selected_bytes = zlib9_bytes(selected[cy, cx])
        for bucket in (by_class[class_id], by_stratum[stratum]):
            bucket["canonical_tile_zlib9_bytes"] += base_bytes
            bucket["selected_tile_zlib9_bytes"] += selected_bytes
            bucket["delta_bytes"] += selected_bytes - base_bytes
            bucket["tile_count"] += 1
    return {
        "per_class": {key: dict(value) for key, value in sorted(by_class.items())},
        "per_stratum": {key: dict(value) for key, value in sorted(by_stratum.items())},
        "scope": "independent tile zlib-9 decomposition; not additive to full-stream coder",
    }


def _calibrate(
    solver: MdlPolytopeMemberSolver,
    raw: np.memmap,
    output_root: Path,
) -> ProxyCalibration:
    path = output_root / "proxy_calibration.json"
    if path.is_file():
        row = _load_json(path)
        if row.get("schema") != "mdl_polytope_proxy_calibration.v1":
            raise MeasurementError("proxy calibration schema drift")
        return ProxyCalibration.from_dict(row["calibration"])
    frame0 = solver.canonicalize(np.asarray(raw[0]))
    frame1 = solver.canonicalize(np.asarray(raw[1]))
    rows0 = solver.calibration_rows(
        solver.generate_candidates(frame0, temporal=frame1),
        temporal=frame1,
        max_rows=DEFAULT_CALIBRATION_ROWS // 2,
    )
    rows1 = solver.calibration_rows(
        solver.generate_candidates(frame1, temporal=frame0),
        temporal=frame0,
        max_rows=DEFAULT_CALIBRATION_ROWS - DEFAULT_CALIBRATION_ROWS // 2,
    )
    calibration = fit_proxy(
        np.concatenate((rows0[0], rows1[0]), axis=0),
        np.concatenate((rows0[1], rows1[1]), axis=0),
    )
    _atomic_json(
        path,
        {
            "schema": "mdl_polytope_proxy_calibration.v1",
            "calibration": calibration.to_dict(),
            "sample_metadata": rows0[2] + rows1[2],
            "authority": {"axis": AXIS, "score_claim": False},
        },
    )
    return calibration


def _stage_config_sha256(
    *, raw_sha256: str, targets_sha256: str, calibration: ProxyCalibration
) -> str:
    payload = {
        "schema": STAGE_SCHEMA,
        "raw_sha256": raw_sha256,
        "targets_sha256": targets_sha256,
        "calibration": calibration.to_dict(),
        "lawrefs": lawref_manifest(),
        "seed": SEED,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _measure_pair(
    *,
    pair_index: int,
    raw: np.memmap,
    labels: np.ndarray,
    solver: MdlPolytopeMemberSolver,
    calibration: ProxyCalibration,
    config_sha256: str,
    stages: Path,
    selected_frames: Path,
) -> dict[str, Any]:
    path = stages / f"pair_{pair_index:04d}.json"
    if path.is_file():
        row = _load_json(path)
        if (
            row.get("schema") != STAGE_SCHEMA
            or row.get("pair_index") != pair_index
            or row.get("config_sha256") != config_sha256
        ):
            raise MeasurementError(f"resume stage drift: {path}")
        return row
    source0 = np.asarray(raw[2 * pair_index])
    source1 = np.asarray(raw[2 * pair_index + 1])
    canonical0 = solver.canonicalize(source0)
    canonical1 = solver.canonicalize(source1)
    if not np.array_equal(source0, canonical0) or not np.array_equal(source1, canonical1):
        raise MeasurementError("source receiver output is not literal canonical support fill")
    started = time.monotonic()
    result0 = solver.solve(
        canonical0,
        temporal=canonical1,
        labels=labels,
        calibration=calibration,
    )
    result1 = solver.solve(
        canonical1,
        temporal=canonical0,
        labels=labels,
        calibration=calibration,
    )
    selected_equal = bool(
        np.array_equal(result0.selected, canonical0)
        and np.array_equal(result1.selected, canonical1)
    )
    selected_path = selected_frames / f"pair_{pair_index:04d}.npz"
    if not selected_equal:
        selected_frames.mkdir(parents=True, exist_ok=True)
        temporary = selected_path.with_name(f".{selected_path.name}.tmp.{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, frame0=result0.selected, frame1=result1.selected)
        os.replace(temporary, selected_path)
    row = {
        "schema": STAGE_SCHEMA,
        "pair_index": pair_index,
        "config_sha256": config_sha256,
        "frame0_levels": _level_rows(result0),
        "frame1_levels": _level_rows(result1),
        "selected_equals_canonical": selected_equal,
        "changed_values": int(
            np.count_nonzero(result0.selected != canonical0)
            + np.count_nonzero(result1.selected != canonical1)
        ),
        "exact_resize_numerators_equal": (
            result0.exact_numerators_equal and result1.exact_numerators_equal
        ),
        "selected_frame_payload": (
            None
            if selected_equal
            else {
                "path": str(selected_path),
                "bytes": selected_path.stat().st_size,
                "sha256": _sha256(selected_path),
            }
        ),
        "decomposition_frame1": _tile_decomposition(
            solver, canonical1, result1.selected, labels
        ),
        "elapsed_seconds": time.monotonic() - started,
        "resumability": {
            "complete_pair_stage": True,
            "selected_reconstruction": (
                "canonical source bytes" if selected_equal else "deterministic solver replay required"
            ),
        },
    }
    _atomic_json(path, row)
    return row


def _load_selected_pair(
    row: Mapping[str, Any], raw: np.memmap, pair_index: int
) -> np.ndarray:
    if row["selected_equals_canonical"]:
        return np.stack((np.asarray(raw[2 * pair_index]), np.asarray(raw[2 * pair_index + 1])))
    payload = row.get("selected_frame_payload")
    if not isinstance(payload, dict):
        raise MeasurementError("changed member lacks preserved selected-frame payload")
    path = Path(payload["path"])
    if _sha256(path) != payload["sha256"]:
        raise MeasurementError("selected-frame payload SHA drift")
    values = np.load(path, allow_pickle=False)
    pair = np.stack((values["frame0"], values["frame1"]))
    if pair.shape != (2, *CAMERA_HW, 3) or pair.dtype != np.uint8:
        raise MeasurementError("selected-frame payload geometry drift")
    return pair


def _run_hard_oracle(
    *,
    net: Any,
    torch: Any,
    raw: np.memmap,
    rows: Sequence[Mapping[str, Any]],
    lstars: np.ndarray,
    poses: np.ndarray,
    max_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    oracle_pairs = ((max_pairs + 31) // 32) * 32
    for start in range(0, oracle_pairs, 32):
        canonical_batch = np.stack(
            [
                np.stack((np.asarray(raw[2 * index]), np.asarray(raw[2 * index + 1])))
                for index in range(start, start + 32)
            ]
        )
        canonical_batch_rows = _oracle_batch32(
            net,
            torch,
            canonical_batch,
            lstars[start : start + 32],
            poses[start : start + 32],
        )
        changed = any(
            not rows[index]["selected_equals_canonical"]
            for index in range(start, min(start + 32, max_pairs))
        )
        if changed:
            selected_batch = canonical_batch.copy()
            for index in range(start, min(start + 32, max_pairs)):
                selected_batch[index - start] = _load_selected_pair(rows[index], raw, index)
            selected_batch_rows = _oracle_batch32(
                net,
                torch,
                selected_batch,
                lstars[start : start + 32],
                poses[start : start + 32],
            )
        else:
            selected_batch_rows = [dict(value) for value in canonical_batch_rows]
        canonical_rows.extend(canonical_batch_rows)
        selected_rows.extend(selected_batch_rows)
    canonical_rows = canonical_rows[:max_pairs]
    selected_rows = selected_rows[:max_pairs]
    for canonical, selected in zip(canonical_rows, selected_rows, strict=True):
        if canonical["argmax_sha256"] != selected["argmax_sha256"] or canonical["pose6"] != selected["pose6"]:
            raise MeasurementError("member selection drifted frozen scorer outputs")
    return canonical_rows, selected_rows


def _sum_levels(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for level in row[key]:
            name = str(level["level"])
            totals[name]["coder_bytes"] += int(level["coder_bytes"])
            totals[name]["delta_bytes_vs_previous"] += int(level["delta_bytes_vs_previous"])
            totals[name]["selected_groups"] += int(level["selected_groups"])
    order = ("canonical", "chart", "object_class_stratum", "pixel_tile_residual")
    return [{"level": name, **dict(totals[name])} for name in order]


def _merge_decomposition(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for group, values in row["decomposition_frame1"][key].items():
            for metric, value in values.items():
                totals[group][metric] += int(value)
    return {group: dict(values) for group, values in sorted(totals.items())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_pairs not in (16, 64):
        raise MeasurementError("delegated gate permits only n16 or n64")
    raw_custody = _check_file(args.raw, RAW_BYTES, EXPECTED_RAW_SHA256)
    targets_custody = _check_file(args.targets, None, EXPECTED_TARGETS_SHA256)
    seed_custody = _check_file(args.seed, None, EXPECTED_SEED_SHA256)
    storage = _storage_preflight(args.output_root, args.max_pairs)
    seed_bytes = args.seed.read_bytes()
    parsed_seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(parsed_seed) != seed_bytes:
        raise MeasurementError("#557 seed parse/serialize roundtrip drift")
    seed_zlib9 = len(zlib.compress(seed_bytes, level=9))
    targets = np.load(args.targets, allow_pickle=False)
    if int(np.asarray(targets["n_pairs"]).reshape(())) != PAIR_COUNT:
        raise MeasurementError("target cache is not n600")
    oracle_pairs = ((args.max_pairs + 31) // 32) * 32
    lstars = np.asarray(targets["lstars"][:oracle_pairs], dtype=np.uint8)
    poses = np.asarray(targets["gt_poses"][:oracle_pairs], dtype=np.float64)
    if lstars.shape != (oracle_pairs, *SCORER_HW) or poses.shape != (oracle_pairs, 6):
        raise MeasurementError("target cache geometry mismatch")
    raw = np.memmap(
        args.raw,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT * 2, *CAMERA_HW, 3),
    )
    solver = MdlPolytopeMemberSolver()
    calibration = _calibrate(solver, raw, args.output_root)
    config_sha256 = _stage_config_sha256(
        raw_sha256=raw_custody["sha256"],
        targets_sha256=targets_custody["sha256"],
        calibration=calibration,
    )
    net, torch, scorer_custody = _load_scorer(args.upstream, args.threads)
    stages = args.output_root / "stages"
    selected_frames = args.output_root / "selected_frames"
    rows = [
        _measure_pair(
            pair_index=index,
            raw=raw,
            labels=lstars[index],
            solver=solver,
            calibration=calibration,
            config_sha256=config_sha256,
            stages=stages,
            selected_frames=selected_frames,
        )
        for index in range(args.max_pairs)
    ]
    canonical_oracle, selected_oracle = _run_hard_oracle(
        net=net,
        torch=torch,
        raw=raw,
        rows=rows,
        lstars=lstars,
        poses=poses,
        max_pairs=args.max_pairs,
    )
    frame0_levels = _sum_levels(rows, "frame0_levels")
    frame1_levels = _sum_levels(rows, "frame1_levels")
    canonical_member_bytes = frame0_levels[0]["coder_bytes"] + frame1_levels[0]["coder_bytes"]
    selected_member_bytes = frame0_levels[-1]["coder_bytes"] + frame1_levels[-1]["coder_bytes"]
    cut_fraction = (canonical_member_bytes - selected_member_bytes) / canonical_member_bytes
    calibration_payload = _canonical_json_bytes(calibration.to_dict())
    seed_plus_calibration_zlib9 = len(zlib.compress(seed_bytes + calibration_payload, level=9))
    dseg_canonical = float(np.mean([row["d_seg"] for row in canonical_oracle]))
    dseg_selected = float(np.mean([row["d_seg"] for row in selected_oracle]))
    dpose_canonical = float(np.mean([row["d_pose"] for row in canonical_oracle]))
    dpose_selected = float(np.mean([row["d_pose"] for row in selected_oracle]))
    receipt = {
        "schema": SCHEMA,
        "task_id": "mdl_polytope_member_solve",
        "lane_id": "mdl_polytope_member_solve_20260721",
        "research_only": True,
        "completed_prefix": args.max_pairs,
        "config_sha256": config_sha256,
        "D1_proxy_calibration": {
            **calibration.to_dict(),
            "minimum_delegated_tiles_met": calibration.row_count >= 20,
            "actual_coder": "zlib-9 over uint8 camera tile bytes",
            "features": (
                "piecewise const breaks, gradient sparsity, exact-kernel move magnitude, "
                "unwarped within-pair temporal residual (xi-consistency proxy only), "
                "occupancy, alphabet size"
            ),
            "proxy_limitation": (
                "temporal_xi_l1 is an unwarped paired-frame L1 feature; it does not "
                "instantiate or measure an explicit SE(3) xi warp"
            ),
        },
        "D2_exact_member_selection": {
            "search_order": ["chart", "object_class_stratum", "pixel_tile_residual"],
            "coordinate_method": "rounded full-kernel pseudoinverse plus dyadic bounded backtracking",
            "integer_resize_exact_pairs": sum(row["exact_resize_numerators_equal"] for row in rows),
            "zero_radius_pose_output_tube_equal_pairs": sum(
                canonical["pose6"] == selected["pose6"]
                for canonical, selected in zip(canonical_oracle, selected_oracle, strict=True)
            ),
            "hard_oracle_seed": SEED,
            "canonical_mean_d_seg": dseg_canonical,
            "selected_mean_d_seg": dseg_selected,
            "canonical_mean_d_pose": dpose_canonical,
            "selected_mean_d_pose": dpose_selected,
            "identical_distortion": dseg_canonical == dseg_selected and dpose_canonical == dpose_selected,
            "selected_equals_canonical_pairs": sum(row["selected_equals_canonical"] for row in rows),
            "batch_geometry": 32,
        },
        "D3_same_coder_comparison": {
            "frame0_levels": frame0_levels,
            "frame1_levels": frame1_levels,
            "canonical_member_zlib9_bytes": canonical_member_bytes,
            "selected_member_zlib9_bytes": selected_member_bytes,
            "member_byte_cut_fraction": cut_fraction,
            "level_attribution": {
                "chart_delta_bytes": frame0_levels[1]["delta_bytes_vs_previous"]
                + frame1_levels[1]["delta_bytes_vs_previous"],
                "object_delta_bytes": frame0_levels[2]["delta_bytes_vs_previous"]
                + frame1_levels[2]["delta_bytes_vs_previous"],
                "pixel_tile_residual_delta_bytes": frame0_levels[3]["delta_bytes_vs_previous"]
                + frame1_levels[3]["delta_bytes_vs_previous"],
            },
            "per_class_frame1": _merge_decomposition(rows, "per_class"),
            "per_stratum_frame1": _merge_decomposition(rows, "per_stratum"),
            "decomposition_scope": (
                "frame1 SegNet-authority tiles only; independent tile zlib-9 totals are "
                "diagnostic and are not additive to the whole-frame coder"
            ),
            "seed_coder": {
                "serializer": "serialize_constraint_seed exact roundtrip",
                "seed_raw_bytes": len(seed_bytes),
                "seed_zlib9_bytes": seed_zlib9,
                "seed_plus_video_derived_calibration_zlib9_bytes": seed_plus_calibration_zlib9,
                "member_policy_overhead_bytes": seed_plus_calibration_zlib9 - seed_zlib9,
                "target_154600_met_by_seed_only": seed_zlib9 <= 154_600,
                "target_121KiB_met_by_seed_only": seed_zlib9 <= 121 * 1024,
                "scope": (
                    "seed coder is the counted description surface; raw-member zlib is diagnostic. "
                    "The member policy does not remove any seed field, so its calibration is overhead."
                ),
            },
        },
        "D4_n600_estimate_and_rate_feed": {
            "threshold_cut_fraction": 0.15,
            "activated": cut_fraction > 0.15,
            "n600_estimate": (
                {
                    "canonical_member_zlib9_bytes": canonical_member_bytes * PAIR_COUNT / args.max_pairs,
                    "selected_member_zlib9_bytes": selected_member_bytes * PAIR_COUNT / args.max_pairs,
                    "label": "ESTIMATE",
                }
                if cut_fraction > 0.15 and args.max_pairs == 64
                else None
            ),
            "feed_541": "ELIGIBLE_PENDING_MAIN" if cut_fraction > 0.15 else "NOT_ROUTED_THRESHOLD_NOT_MET",
        },
        "reuse_manifest": {
            "#49/S12": "tac.optimization.resize_null_preimage coder objective lineage",
            "#580": "FullResizeKernel complete 80.674% nullity plus exact numerator verification",
            "#547/#549": "current exact two-plane canonical support-fill raw and DisjointResizeOperator",
            "#557": "parse_constraint_seed + serialize_constraint_seed + zlib-9 counted baseline",
            "margin_survival": "settled fixed positive-margin alphabet; no remeasurement or RGB rewrite",
            "#586": "lattice enumeration used only as deterministic proposal ordering; no global optimum claim",
            "new_code_failed_search_justification": (
                "#580 selects one whole-frame preference and has no calibrated per-tile proxy, "
                "chart/object-first hierarchy, class/stratum decomposition, or resumable n16/n64 oracle receipt."
            ),
        },
        "lawrefs": lawref_manifest(),
        "input_custody": {"raw": raw_custody, "targets": targets_custody, "seed": seed_custody},
        "scorer_custody": scorer_custody,
        "storage": storage,
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "argv": sys.argv,
            "stages_preserved": len(rows),
            "stage_root": str(stages),
        },
        "verdict": (
            "MEASURED_MEMBER_RATE_CUT_ABOVE_15_PERCENT"
            if cut_fraction > 0.15
            else "MEASURED_NO_MEMBER_RATE_CUT_RETAIN_CANONICAL"
        ),
        "verdict_scope": (
            "n16/n64 prefix; exact current canonical support-fill; bounded chart/object-first "
            "full-kernel candidate family; zlib-9 diagnostic member coder and #557 seed coder. "
            "This is not a family-dead or global MDL optimum verdict."
        ),
        "untested_optimal_form_queue": [
            "direct polynomial chart-coefficient and event-symbol optimization in the counted seed",
            "explicit SE(3) xi-warped temporal chart with measured pose-tube admission",
            "joint scorer-plane packet and member entropy optimization under the real archive coder",
            "curvelet/shearlet residual symbols only after charged chart/object terms are exhausted",
        ],
        "authority": {
            "axis": AXIS,
            "pointer": POINTER,
            "pointer_moved": False,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
        },
    }
    receipt_path = args.output_root / f"receipt_n{args.max_pairs}.json"
    _atomic_json(receipt_path, receipt)
    _atomic_json(args.output_root / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pairs", type=int, choices=(16, 64), required=True)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    default_upstream = REPO / "upstream"
    if not default_upstream.is_dir():
        default_upstream = Path("/Users/adpena/Projects/pact/upstream")
    parser.add_argument("--upstream", type=Path, default=default_upstream)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.raw = args.raw.resolve()
    args.targets = args.targets.resolve()
    args.seed = args.seed.resolve()
    args.output_root = args.output_root.resolve()
    args.upstream = args.upstream.resolve()
    receipt = run(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
