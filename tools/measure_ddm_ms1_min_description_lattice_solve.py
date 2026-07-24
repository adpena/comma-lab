#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the full-n600 minimum-description exact-resize lattice solve.

This tool is local-only and advisory. It compares three decoder origins with a
real zlib-9 context coder:

* A: the current member against zero (raw bytes);
* B: causal previous-selected-frame prediction;
* C: B for frame0 and a counted full-SE(3) xi warp of frame0 for frame1.

The saturated exact-integer resize kernel proposes members for B and C. A
proposal is admitted per frame only on strict real-coder improvement, then the
whole continuous stream is rescored and can reject the collection globally.
Every pair is an atomic resumable stage. Bulky changed-frame payloads and SENSE
telemetry stay on the SSD; the repository receipt is a compact hash-bound index.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math import warp_real_luma_frame0 as g1_warp  # noqa: E402
from tac.optimization.ddm_lattice_costate_sense import (  # noqa: E402
    build_lattice_sense_pair,
    factorize_lattice_sense,
    write_sense_jsonl_atomic,
)
from tac.optimization.mdl_polytope_member import (  # noqa: E402
    MdlPolytopeMemberSolver,
    lawref_manifest,
    modular_uint8_residual,
    reconstruct_modular_uint8,
    zlib9_bytes,
)
from tac.optimization.predict_project_receiver import (  # noqa: E402
    counted_full_screw_xi_series,
)

SCHEMA: Final = "ddm_min_description_lattice_solve_receipt.v1"
STAGE_SCHEMA: Final = "ddm_min_description_lattice_pair_stage.v1"
RUN_ID: Final = "ddm_ms1_min_description_lattice_solve_20260723T233549Z"
LANE_ID: Final = "lane_ddm_ms1_min_description_lattice_solve_20260723"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU] UNMOVED"
SEED: Final = 1234
PAIR_COUNT: Final = 600
CAMERA_HW: Final = (874, 1164)
SCORER_HW: Final = (384, 512)
RAW_BYTES: Final = PAIR_COUNT * 2 * CAMERA_HW[0] * CAMERA_HW[1] * 3
EXPECTED_RAW_SHA256: Final = "31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b"
EXPECTED_TARGETS_SHA256: Final = (
    "e41718a047b77b9072828e72f8cbffa0f5ef7ddf462c7ef4329d997ace89de50"
)
DEFAULT_RAW: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/"
    "capstone_submission/inflated/0.raw"
)
DEFAULT_TARGETS: Final = Path(
    "/Users/adpena/Projects/pact/experiments/results/ot_offset_n600_modal_20260709/"
    "gt_n600_lstars_slim.npz"
)
DEFAULT_UPSTREAM: Final = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/"
    "ddm_ms1_min_description_lattice_solve_20260723_final"
)
DEFAULT_RECEIPT: Final = REPO / ".omx" / "research" / RUN_ID / "receipt.json"
TRANSLATION_SCALE: Final = 0.16
ROTATION_SCALE: Final = 1.0
GROUND_PITCH_RAD: Final = -0.05
ACTIVE_MARGIN_TOLERANCE: Final = 1.0e-6


class MeasurementError(RuntimeError):
    """Fail-closed source, stage, coder, scorer, or custody error."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_json(value) + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"JSON payload is not an object: {path}")
    return value


def _check_file(path: Path, expected_bytes: int | None, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise MeasurementError(f"required input is missing: {path}")
    observed_bytes = path.stat().st_size
    observed_sha256 = _sha256_file(path)
    if expected_bytes is not None and observed_bytes != expected_bytes:
        raise MeasurementError(
            f"input byte mismatch for {path}: {observed_bytes} != {expected_bytes}"
        )
    if observed_sha256 != expected_sha256:
        raise MeasurementError(
            f"input SHA mismatch for {path}: {observed_sha256} != {expected_sha256}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def _storage_preflight(output_root: Path) -> dict[str, Any]:
    resolved = output_root.resolve()
    allowed = Path("/Volumes/VertigoDataTier/pact")
    if resolved != allowed and allowed not in resolved.parents:
        raise MeasurementError(f"bulk evidence must remain below {allowed}")
    resolved.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(resolved).free
    required = 2 << 30
    if free < required:
        raise MeasurementError(f"storage preflight refused: {free} < {required}")
    row = {
        "status": "PASS",
        "path": str(resolved),
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "cleanup": (
            "atomic scratch is success/failure-cleaned; pair stages and changed-frame "
            "payloads are preserved; source bytes are never copied, moved, or deleted"
        ),
    }
    _atomic_json(resolved / "storage_preflight.json", row)
    return row


def _load_scorer(upstream: Path, threads: int) -> tuple[Any, Any, dict[str, Any]]:
    modules_path = upstream / "modules.py"
    evaluate_path = upstream / "evaluate.py"
    if threads < 1 or not modules_path.is_file() or not evaluate_path.is_file():
        raise MeasurementError("frozen native CPU-Torch scorer custody is unavailable")
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
        "modules_path": str(modules_path.resolve()),
        "modules_sha256": _sha256_file(modules_path),
        "evaluate_path": str(evaluate_path.resolve()),
        "evaluate_sha256": _sha256_file(evaluate_path),
        "segnet_weights_path": str(Path(segnet_sd_path).resolve()),
        "segnet_weights_sha256": _sha256_file(Path(segnet_sd_path)),
        "posenet_weights_path": str(Path(posenet_sd_path).resolve()),
        "posenet_weights_sha256": _sha256_file(Path(posenet_sd_path)),
        "device": "cpu",
        "threads": threads,
        "batch_geometry": 32,
        "last_batch_padding": "repeat final pair to 32, then trim; eval-mode outputs are sample-local",
        "seed": SEED,
        "deterministic_algorithms": True,
    }


def _score_batch32(
    net: Any,
    torch: Any,
    pairs: np.ndarray,
    target_cells: np.ndarray,
    target_poses: np.ndarray,
) -> list[dict[str, Any]]:
    if pairs.shape != (32, 2, *CAMERA_HW, 3):
        raise MeasurementError("frozen scorer requires batch32 receiver geometry")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(pairs))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
    )
    with torch.inference_mode():
        logits_tensor = net.segnet(net.segnet.preprocess_input(tensor))
        values, indices = logits_tensor.topk(2, dim=1)
        argmax = indices[:, 0].cpu().numpy().astype(np.uint8)
        margins = (values[:, 0] - values[:, 1]).cpu().numpy().astype(np.float32)
        pose_output = net.posenet(net.posenet.preprocess_input(tensor))
        pose_tensor = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose_tensor[:, :6].cpu().numpy().astype(np.float64)
    return [
        {
            "d_seg": float(np.mean(argmax[index] != target_cells[index])),
            "d_pose": float(np.mean((pose6[index] - target_poses[index]) ** 2)),
            "argmax": argmax[index],
            "argmax_sha256": _sha256_array(argmax[index]),
            "winner_rival_margins": margins[index],
            "margin_sha256": _sha256_array(margins[index]),
            "pose6": pose6[index],
            "pose6_sha256": _sha256_array(pose6[index]),
        }
        for index in range(32)
    ]


def _warp_origin(
    frame0: np.ndarray,
    xi: np.ndarray,
    geom: Any,
    solver: MdlPolytopeMemberSolver,
) -> np.ndarray:
    numerators, denominator = solver.kernel.operator.apply_numerators(frame0)
    if np.any(numerators % denominator):
        raise MeasurementError("xi origin source does not have an exact scorer plane")
    scorer_plane = (numerators // denominator).astype(np.uint8)
    warped = g1_warp.warp_frame0_native_numpy(scorer_plane, xi, geom)
    warped_uint8 = np.clip(np.rint(warped), 0, 255).astype(np.uint8)
    return solver.kernel.operator.realize_factor2_uint8(warped_uint8)


def _save_selected(path: Path, frame0: np.ndarray, frame1: np.ndarray) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, frame0=frame0, frame1=frame1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _load_selected(
    row: Mapping[str, Any],
    origin_name: str,
    raw: np.memmap,
    pair_index: int,
) -> np.ndarray:
    origin_row = row["origins"][origin_name]
    if origin_row["selected_equals_canonical"]:
        return np.stack(
            (
                np.asarray(raw[2 * pair_index]),
                np.asarray(raw[2 * pair_index + 1]),
            )
        )
    payload = origin_row.get("selected_payload")
    if not isinstance(payload, Mapping):
        raise MeasurementError(f"{origin_name} changed member lacks preserved payload")
    path = Path(str(payload["path"]))
    if not path.is_file() or _sha256_file(path) != payload["sha256"]:
        raise MeasurementError(f"{origin_name} selected payload custody drift")
    with np.load(path, allow_pickle=False) as values:
        pair = np.stack((values["frame0"], values["frame1"]))
    if pair.shape != (2, *CAMERA_HW, 3) or pair.dtype != np.uint8:
        raise MeasurementError(f"{origin_name} selected payload geometry drift")
    return pair


def _pair_stage(
    *,
    pair_index: int,
    raw: np.memmap,
    xi: np.ndarray,
    geom: Any,
    solver: MdlPolytopeMemberSolver,
    stage_dir: Path,
    selected_dir: Path,
    config_sha256: str,
    previous_b: np.ndarray,
    previous_c: np.ndarray,
    prune_c: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    path = stage_dir / f"pair_{pair_index:04d}.json"
    if path.is_file():
        row = _load_json(path)
        if (
            row.get("schema") != STAGE_SCHEMA
            or row.get("pair_index") != pair_index
            or row.get("config_sha256") != config_sha256
        ):
            raise MeasurementError(f"resume stage drift: {path}")
        selected_b = _load_selected(row, "B_previous_frame", raw, pair_index)
        selected_c = _load_selected(row, "C_xi_motion", raw, pair_index)
        return row, selected_b[1], selected_c[1]

    canonical = np.stack(
        (
            np.asarray(raw[2 * pair_index]),
            np.asarray(raw[2 * pair_index + 1]),
        )
    )
    started = time.monotonic()

    origin_b0 = previous_b
    solve_b0 = solver.solve_against_origin(canonical[0], origin=origin_b0)
    origin_b1 = solve_b0.selected
    solve_b1 = solver.solve_against_origin(canonical[1], origin=origin_b1)
    selected_b = np.stack((solve_b0.selected, solve_b1.selected))

    if prune_c:
        solve_c0 = None
        solve_c1 = None
        selected_c = canonical.copy()
    else:
        origin_c0 = previous_c
        solve_c0 = (
            solve_b0
            if np.array_equal(origin_c0, origin_b0)
            else solver.solve_against_origin(canonical[0], origin=origin_c0)
        )
        origin_c1 = _warp_origin(solve_c0.selected, xi[pair_index], geom, solver)
        solve_c1 = solver.solve_against_origin(canonical[1], origin=origin_c1)
        selected_c = np.stack((solve_c0.selected, solve_c1.selected))

    def origin_stage(
        name: str,
        selected: np.ndarray,
        frame0: Any,
        frame1: Any,
    ) -> dict[str, Any]:
        selected_equal = bool(np.array_equal(selected, canonical))
        payload = None
        if not selected_equal:
            payload = _save_selected(
                selected_dir / name / f"pair_{pair_index:04d}.npz",
                selected[0],
                selected[1],
            )
        return {
            "frame0": frame0.to_dict(),
            "frame1": frame1.to_dict(),
            "selected_equals_canonical": selected_equal,
            "changed_values": int(np.count_nonzero(selected != canonical)),
            "selected_pair_sha256": _sha256_array(selected),
            "selected_payload": payload,
        }

    row = {
        "schema": STAGE_SCHEMA,
        "pair_index": pair_index,
        "config_sha256": config_sha256,
        "canonical_pair_sha256": _sha256_array(canonical),
        "origins": {
            "B_previous_frame": origin_stage(
                "B_previous_frame", selected_b, solve_b0, solve_b1
            ),
            "C_xi_motion": (
                {
                    "frame0": {
                        "status": "PRUNED_BY_N16_REAL_CODER_GATE",
                        "reason": "all n16 C proposals lost and C canonical residual exceeded A",
                    },
                    "frame1": {
                        "status": "PRUNED_BY_N16_REAL_CODER_GATE",
                        "reason": "all n16 C proposals lost and C canonical residual exceeded A",
                    },
                    "selected_equals_canonical": True,
                    "changed_values": 0,
                    "selected_pair_sha256": _sha256_array(canonical),
                    "selected_payload": None,
                    "verdict_scope": (
                        "C saturated-local-CVP proposal only; C canonical origin remains "
                        "measured through full n600"
                    ),
                }
                if prune_c
                else origin_stage("C_xi_motion", selected_c, solve_c0, solve_c1)
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
        "resumability": {
            "complete_pair_stage": True,
            "previous_state": "selected frame1 reconstructed from source or SHA-bound NPZ",
            "stage_loss_bound": "at most one pair",
        },
    }
    _atomic_json(path, row)
    return row, selected_b[1], selected_c[1]


def _continuous_streams(
    *,
    raw: np.memmap,
    rows: Sequence[Mapping[str, Any]],
    xi: np.ndarray,
    geom: Any,
    solver: MdlPolytopeMemberSolver,
) -> dict[str, Any]:
    compressors = {
        name: zlib.compressobj(level=9)
        for name in (
            "A_raw",
            "B_canonical",
            "B_stage_selected",
            "C_canonical",
            "C_stage_selected",
        )
    }
    sizes = dict.fromkeys(compressors, 0)
    prior_canonical = np.zeros((*CAMERA_HW, 3), dtype=np.uint8)
    prior_b = prior_canonical.copy()
    prior_c = prior_canonical.copy()

    def add(name: str, value: np.ndarray) -> None:
        sizes[name] += len(compressors[name].compress(np.ascontiguousarray(value).tobytes()))

    for pair_index, row in enumerate(rows):
        canonical = np.stack(
            (
                np.asarray(raw[2 * pair_index]),
                np.asarray(raw[2 * pair_index + 1]),
            )
        )
        selected_b = _load_selected(row, "B_previous_frame", raw, pair_index)
        selected_c = _load_selected(row, "C_xi_motion", raw, pair_index)

        add("A_raw", canonical)
        add("B_canonical", modular_uint8_residual(canonical[0], prior_canonical))
        add("B_canonical", modular_uint8_residual(canonical[1], canonical[0]))
        add("B_stage_selected", modular_uint8_residual(selected_b[0], prior_b))
        add("B_stage_selected", modular_uint8_residual(selected_b[1], selected_b[0]))

        canonical_c1 = _warp_origin(canonical[0], xi[pair_index], geom, solver)
        selected_c1 = (
            canonical_c1
            if np.array_equal(selected_c[0], canonical[0])
            else _warp_origin(selected_c[0], xi[pair_index], geom, solver)
        )
        add("C_canonical", modular_uint8_residual(canonical[0], prior_canonical))
        add("C_canonical", modular_uint8_residual(canonical[1], canonical_c1))
        add("C_stage_selected", modular_uint8_residual(selected_c[0], prior_c))
        add("C_stage_selected", modular_uint8_residual(selected_c[1], selected_c1))

        prior_canonical = canonical[1]
        prior_b = selected_b[1]
        prior_c = selected_c[1]
    for name, compressor in compressors.items():
        sizes[name] += len(compressor.flush())
    return {
        "coder": "one continuous zlib-9 stream over all 1200 frames in pair order",
        "bytes": sizes,
        "global_admission": {
            "B_previous_frame": (
                "stage_selected"
                if sizes["B_stage_selected"] < sizes["B_canonical"]
                else "canonical_global_tie_break"
            ),
            "C_xi_motion": (
                "stage_selected"
                if sizes["C_stage_selected"] < sizes["C_canonical"]
                else "canonical_global_tie_break"
            ),
        },
    }


def _c_prune_gate(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) < 16:
        return False
    prefix = rows[:16]
    frame_rows = [
        row["origins"]["C_xi_motion"][frame]
        for row in prefix
        for frame in ("frame0", "frame1")
    ]
    return all(
        item["proposed_residual_bytes"] >= item["canonical_residual_bytes"]
        for item in frame_rows
    ) and sum(item["canonical_residual_bytes"] for item in frame_rows) >= sum(
        item["canonical_member_bytes"] for item in frame_rows
    )


def _selected_for_global_origin(
    *,
    row: Mapping[str, Any],
    origin_name: str,
    admission: str,
    raw: np.memmap,
    pair_index: int,
) -> np.ndarray:
    if origin_name == "A_zero":
        return np.stack(
            (
                np.asarray(raw[2 * pair_index]),
                np.asarray(raw[2 * pair_index + 1]),
            )
        )
    if admission == "stage_selected":
        return _load_selected(row, origin_name, raw, pair_index)
    return np.stack(
        (
            np.asarray(raw[2 * pair_index]),
            np.asarray(raw[2 * pair_index + 1]),
        )
    )


def _score_and_sense(
    *,
    net: Any,
    torch: Any,
    raw: np.memmap,
    rows: Sequence[Mapping[str, Any]],
    lstars: np.ndarray,
    poses: np.ndarray,
    origin_name: str,
    admission: str,
    xi: np.ndarray,
    geom: Any,
    solver: MdlPolytopeMemberSolver,
    sense_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical_oracle: list[dict[str, Any]] = []
    selected_oracle: list[dict[str, Any]] = []
    selected_pairs: list[np.ndarray] = []
    for start in range(0, PAIR_COUNT, 32):
        indices = list(range(start, min(start + 32, PAIR_COUNT)))
        padded = indices + [indices[-1]] * (32 - len(indices))
        canonical_batch = np.stack(
            [
                np.stack((np.asarray(raw[2 * index]), np.asarray(raw[2 * index + 1])))
                for index in padded
            ]
        )
        selected_batch = np.stack(
            [
                _selected_for_global_origin(
                    row=rows[index],
                    origin_name=origin_name,
                    admission=admission,
                    raw=raw,
                    pair_index=index,
                )
                for index in padded
            ]
        )
        target_cells = lstars[padded]
        target_poses = poses[padded]
        canonical_rows = _score_batch32(
            net, torch, canonical_batch, target_cells, target_poses
        )
        if np.array_equal(selected_batch, canonical_batch):
            selected_rows = canonical_rows
        else:
            selected_rows = _score_batch32(
                net, torch, selected_batch, target_cells, target_poses
            )
        canonical_oracle.extend(canonical_rows[: len(indices)])
        selected_oracle.extend(selected_rows[: len(indices)])
        selected_pairs.extend(list(selected_batch[: len(indices)]))

    for pair_index, (canonical, selected) in enumerate(
        zip(canonical_oracle, selected_oracle, strict=True)
    ):
        if (
            canonical["argmax_sha256"] != selected["argmax_sha256"]
            or canonical["pose6_sha256"] != selected["pose6_sha256"]
        ):
            raise MeasurementError(
                f"pair {pair_index}: selected member escaped zero-radius frozen-scorer tube"
            )

    basis_summary = solver.basis_norm_summary()
    sense_rows: list[dict[str, Any]] = []
    previous = np.zeros((*CAMERA_HW, 3), dtype=np.uint8)
    for pair_index, selected in enumerate(selected_pairs):
        if origin_name == "A_zero":
            origin = np.zeros_like(selected)
        elif origin_name == "B_previous_frame":
            origin = np.stack((previous, selected[0]))
        else:
            origin = np.stack(
                (previous, _warp_origin(selected[0], xi[pair_index], geom, solver))
            )
        canonical_pair_bytes = zlib9_bytes(
            np.stack(
                (
                    np.asarray(raw[2 * pair_index]),
                    np.asarray(raw[2 * pair_index + 1]),
                )
            )
        )
        residual = modular_uint8_residual(selected, origin)
        if not np.array_equal(reconstruct_modular_uint8(origin, residual), selected):
            raise MeasurementError("pair residual failed exact parse-back")
        sense_rows.append(
            build_lattice_sense_pair(
                pair_id=pair_index,
                selected=selected,
                origin=origin,
                labels=selected_oracle[pair_index]["argmax"],
                winner_rival_margins=selected_oracle[pair_index][
                    "winner_rival_margins"
                ],
                canonical_member_bytes=canonical_pair_bytes,
                selected_residual_bytes=zlib9_bytes(residual),
                active_tolerance=ACTIVE_MARGIN_TOLERANCE,
                basis_norms=basis_summary,
                local_facet_dimensions=solver.local_facet_dimensions(selected[1]),
                duals=None,
            ).to_dict()
        )
        previous = selected[1]
    write_sense_jsonl_atomic(sense_rows, sense_path)

    def mean(key: str, values: Sequence[Mapping[str, Any]]) -> float:
        return float(np.mean([float(value[key]) for value in values]))

    return (
        {
            "canonical_mean_d_seg": mean("d_seg", canonical_oracle),
            "selected_mean_d_seg": mean("d_seg", selected_oracle),
            "canonical_mean_d_pose": mean("d_pose", canonical_oracle),
            "selected_mean_d_pose": mean("d_pose", selected_oracle),
            "argmax_identical_pairs": sum(
                first["argmax_sha256"] == second["argmax_sha256"]
                for first, second in zip(canonical_oracle, selected_oracle, strict=True)
            ),
            "pose6_bit_identical_pairs": sum(
                first["pose6_sha256"] == second["pose6_sha256"]
                for first, second in zip(canonical_oracle, selected_oracle, strict=True)
            ),
            "batch_geometry": 32,
            "pair_count": PAIR_COUNT,
            "pose_tube": "zero-radius exact fp64 output equality on same deterministic CPU scorer",
        },
        sense_rows,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.pairs != PAIR_COUNT:
        raise MeasurementError("delegated authority requires the full n600 run")
    started = time.monotonic()
    raw_custody = _check_file(args.raw, RAW_BYTES, EXPECTED_RAW_SHA256)
    targets_custody = _check_file(args.targets, None, EXPECTED_TARGETS_SHA256)
    storage = _storage_preflight(args.output_root)
    targets = np.load(args.targets, allow_pickle=False)
    if int(np.asarray(targets["n_pairs"]).reshape(())) != PAIR_COUNT:
        raise MeasurementError("target cache is not n600")
    lstars = np.asarray(targets["lstars"], dtype=np.uint8)
    poses = np.asarray(targets["gt_poses"], dtype=np.float64)
    if lstars.shape != (PAIR_COUNT, *SCORER_HW) or poses.shape != (PAIR_COUNT, 6):
        raise MeasurementError("target cache geometry mismatch")
    raw = np.memmap(
        args.raw,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT * 2, *CAMERA_HW, 3),
    )
    xi, xi_custody = counted_full_screw_xi_series(
        poses,
        translation_scale=TRANSLATION_SCALE,
        rotation_scale=ROTATION_SCALE,
        pitch_rad=GROUND_PITCH_RAD,
        source_sha256=targets_custody["sha256"],
    )
    pose_payload = poses.astype("<f8", copy=False).tobytes()
    xi_origin_payload_bytes = len(zlib.compress(pose_payload, level=9))
    geom = g1_warp.GroundHomographyGeom.eon(
        native_hw=SCORER_HW,
        pitch=GROUND_PITCH_RAD,
    )
    solver = MdlPolytopeMemberSolver()
    implementation = {
        str(path.relative_to(REPO)): _sha256_file(path)
        for path in (
            Path(__file__).resolve(),
            REPO / "src/tac/optimization/mdl_polytope_member.py",
            REPO / "src/tac/optimization/ddm_lattice_costate_sense.py",
            REPO / "src/tac/optimization/resize_full_kernel.py",
            REPO / "src/tac/optimization/predict_project_receiver.py",
            REPO / "src/tac/boundary_math/warp_real_luma_frame0.py",
            REPO / "src/tac/lie/_se3_numpy.py",
        )
    }
    config = {
        "schema": STAGE_SCHEMA,
        "run_id": RUN_ID,
        "raw_sha256": raw_custody["sha256"],
        "targets_sha256": targets_custody["sha256"],
        "seed": SEED,
        "origins": {
            "A": "zero origin; current uint8 member",
            "B": "causal previous-selected-frame; pair frame1 predicts from selected frame0",
            "C": (
                "B for pair frame0; counted pose6 -> full SE(3) xi -> deterministic "
                "ground-homography warp of the exact scorer plane -> exact factor-2 "
                "camera realization for pair frame1"
            ),
        },
        "xi": {
            "translation_scale": TRANSLATION_SCALE,
            "rotation_scale": ROTATION_SCALE,
            "pitch_rad": GROUND_PITCH_RAD,
            "pose_payload_dtype": "<f8",
            "pose_payload_sha256": _sha256_bytes(pose_payload),
        },
        "active_margin_tolerance": ACTIVE_MARGIN_TOLERANCE,
        "c_proposal_prune_gate": (
            "after n16, prune only if every C local-CVP proposal is non-improving and "
            "summed C canonical residual bytes are no smaller than A raw bytes; continue "
            "full-n600 C canonical-origin measurement"
        ),
        "lawrefs": lawref_manifest(),
        "implementation": implementation,
    }
    config_sha256 = _sha256_bytes(_canonical_json(config))
    stage_dir = args.output_root / "stages"
    selected_dir = args.output_root / "selected"
    rows: list[dict[str, Any]] = []
    previous_b = np.zeros((*CAMERA_HW, 3), dtype=np.uint8)
    previous_c = previous_b.copy()
    for pair_index in range(PAIR_COUNT):
        row, previous_b, previous_c = _pair_stage(
            pair_index=pair_index,
            raw=raw,
            xi=xi,
            geom=geom,
            solver=solver,
            stage_dir=stage_dir,
            selected_dir=selected_dir,
            config_sha256=config_sha256,
            previous_b=previous_b,
            previous_c=previous_c,
            prune_c=_c_prune_gate(rows),
        )
        rows.append(row)
        if (pair_index + 1) % 16 == 0 or pair_index + 1 == PAIR_COUNT:
            print(
                json.dumps(
                    {
                        "milestone": "pair_stage",
                        "completed": pair_index + 1,
                        "pairs": PAIR_COUNT,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    streams = _continuous_streams(
        raw=raw,
        rows=rows,
        xi=xi,
        geom=geom,
        solver=solver,
    )
    b_bytes = min(
        streams["bytes"]["B_canonical"],
        streams["bytes"]["B_stage_selected"],
    )
    c_residual_bytes = min(
        streams["bytes"]["C_canonical"],
        streams["bytes"]["C_stage_selected"],
    )
    c_bytes = c_residual_bytes + xi_origin_payload_bytes
    a_bytes = streams["bytes"]["A_raw"]
    winner_origin = min(
        ("A_zero", a_bytes),
        ("B_previous_frame", b_bytes),
        ("C_xi_motion", c_bytes),
        key=lambda value: (value[1], value[0]),
    )[0]
    sense_origin = winner_origin
    if winner_origin == "A_zero":
        sense_admission = "canonical_global_tie_break"
    else:
        sense_admission = streams["global_admission"][sense_origin]

    net, torch, scorer_custody = _load_scorer(args.upstream, args.threads)
    sense_path = args.output_root / "sense" / "pair_rows.jsonl"
    scorer, sense_rows = _score_and_sense(
        net=net,
        torch=torch,
        raw=raw,
        rows=rows,
        lstars=lstars,
        poses=poses,
        origin_name=sense_origin,
        admission=sense_admission,
        xi=xi,
        geom=geom,
        solver=solver,
        sense_path=sense_path,
    )
    factorization = factorize_lattice_sense(
        sense_rows,
        coder_noise_floor_bytes=1,
    )
    factor_path = args.output_root / "sense" / "factorization.json"
    _atomic_json(factor_path, factorization)
    sense_custody = {
        "pair_jsonl": {
            "path": str(sense_path.resolve()),
            "bytes": sense_path.stat().st_size,
            "sha256": _sha256_file(sense_path),
        },
        "factorization": {
            "path": str(factor_path.resolve()),
            "bytes": factor_path.stat().st_size,
            "sha256": _sha256_file(factor_path),
        },
        "coder_noise_floor_bytes": 1,
        "noise_floor_basis": (
            "MEASURED deterministic zlib repeat variability is 0 bytes; integer coder "
            "resolution is therefore the conservative 1-byte factor admission floor"
        ),
    }
    changed = {
        origin: sum(
            not row["origins"][origin]["selected_equals_canonical"] for row in rows
        )
        for origin in ("B_previous_frame", "C_xi_motion")
    }
    receipt = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "task_id": "ddm_ms1_min_description_lattice_solve",
        "lane_id": LANE_ID,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "axis": AXIS,
        "verdict_scope": (
            "full n600 instance of saturated-local-CVP member proposals under zlib-9 "
            "and the three named origins; not a global lattice optimum or family negative"
        ),
        "config_sha256": config_sha256,
        "inputs": {"raw": raw_custody, "targets": targets_custody},
        "storage": {
            key: value
            for key, value in storage.items()
            if key != "observed_free_bytes"
        },
        "implementation_sha256": implementation,
        "resumability": {
            "pair_stage_count": len(rows),
            "complete_pair_stages": len(rows) == PAIR_COUNT,
            "stage_schema": STAGE_SCHEMA,
            "stage_directory": str(stage_dir.resolve()),
            "all_preserved": True,
        },
        "lattice": {
            "objective": "min bytes_code(x - x_origin mod 256)",
            "gauge_fix": (
                "origin fixes the affine representative; saturated ker(A) coordinates "
                "are proposed by exact integer CVP and never credited as free payload"
            ),
            "basis": solver.basis_norm_summary(),
            "member_admission": (
                "strict per-frame residual zlib-9 decrease, followed by strict continuous-"
                "stream global decrease; deterministic canonical tie-break at both levels"
            ),
            "changed_pairs": changed,
            "C_local_CVP_pruned_after_n16": _c_prune_gate(rows),
            "shadow_prices": "UNAVAILABLE; bounded integer projection exposes no KKT multipliers",
        },
        "origins": {
            "A_zero": {
                "residual_bytes": a_bytes,
                "origin_payload_bytes": 0,
                "total_counted_bytes": a_bytes,
                "bytes_over_A": 1.0,
            },
            "B_previous_frame": {
                "canonical_residual_bytes": streams["bytes"]["B_canonical"],
                "stage_selected_residual_bytes": streams["bytes"]["B_stage_selected"],
                "global_admission": streams["global_admission"]["B_previous_frame"],
                "residual_bytes": b_bytes,
                "origin_payload_bytes": 0,
                "total_counted_bytes": b_bytes,
                "bytes_over_A": b_bytes / a_bytes,
            },
            "C_xi_motion": {
                "canonical_residual_bytes": streams["bytes"]["C_canonical"],
                "stage_selected_residual_bytes": streams["bytes"]["C_stage_selected"],
                "global_admission": streams["global_admission"]["C_xi_motion"],
                "residual_bytes": c_residual_bytes,
                "origin_payload_bytes": xi_origin_payload_bytes,
                "origin_payload_coder": "zlib-9 over exact <f8 stored pose6 bytes",
                "total_counted_bytes": c_bytes,
                "bytes_over_A": c_bytes / a_bytes,
                "xi_custody": xi_custody,
            },
        },
        "winner_origin": winner_origin,
        "continuous_coder": streams,
        "frozen_scorer": {**scorer_custody, **scorer},
        "sense": sense_custody,
        "factorization_summary": {
            "pair_count": factorization["pair_count"],
            "admitted_factor_count": factorization["admitted_factor_count"],
            "routes": factorization["routes"],
            "matrix_sha256": factorization["matrix_sha256"],
        },
        "byte_partition": {
            "FREE": (
                "generic modular addition, saturated integer kernel construction, "
                "previous-frame predictor, and SE(3) warp implementation"
            ),
            "NULL": "unselected ker(A) coordinates receive zero byte credit",
            "COUNTED": (
                "continuous residual zlib stream plus exact pose6 payload for C"
            ),
        },
        "prior_blockers": {
            "m6": "SUB015_NOT_REACHED_Y13_RESIDUAL22632",
            "m6_detail": (
                "same-member coder exhausted at 13 bytes; remaining 22632 bytes "
                "requires member selection"
            ),
            "m5r": "SUBSET_PROPOSAL_NOT_ADMITTED_AT_FULL_N600",
            "m5r_detail": "tiny 368-DOF lift instance negative; full-rank family remained open",
        },
        "duration_seconds": time.monotonic() - started,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "next_measurement": {
            "if_B_or_C_wins": (
                "replace zlib diagnostic with receiver-closed contest runtime coder and "
                "run paired contest CPU/CUDA evaluation on exact archive bytes"
            ),
            "if_no_member_move": (
                "retain origin result, then test a bounded exact lattice sieve/branch-and-"
                "bound before any family verdict; do not repeat the same local CVP"
            ),
        },
    }
    _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main() -> int:
    try:
        run(_parser().parse_args())
    except MeasurementError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
