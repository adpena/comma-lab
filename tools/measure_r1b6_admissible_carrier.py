#!/usr/bin/env python3
"""Measure exact singleton preimages through the R1b4 receiver and hard oracle.

This is an encoder-side measurement tool.  It uses source bytes only to choose
between two exact same-rounded-bin uint8 preimages, writes the selected bytes
into the counted R1K1 replay, then delegates all actuation to R1b4.  The
receiver performs no search and contains no scorer or source table.

Authority: [macOS-CPU advisory], score_claim=false, pointer unmoved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from functools import reduce
from itertools import pairwise
from math import gcd
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
for candidate in (REPO, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tac.boundary_math.integer_plane_emitter import (  # noqa: E402
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PLANE_COUNT,
    RGB_CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
)
from tac.boundary_math.r1b4_section_receiver import (  # noqa: E402
    ReplayWrite,
    build_r1b4_archive,
    decode_r1b4_archive,
    encode_replay_payload,
    seal_output_assertion,
    sha256_file,
)
from tac.boundary_math.windowed_curvelet_frame import (  # noqa: E402
    WindowedCurveletConfig,
)
from tac.canonical_equations.day_consolidation_laws_20260720 import (  # noqa: E402
    breakeven_bytes,
)
from tac.optimization.boundary_coordinate_joint_solve import (  # noqa: E402
    BoundaryCoordinatePacket,
    FrameFamily,
    encode_boundary_packet,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    BlockSolveStatus,
    DisjointResizeOperator,
    solve_bounded_integer_block,
)

PAIR_COUNT: Final = 600
SEED: Final = 1234
BATCH_SIZE: Final = 16
OLD_REALIZED_RECOVERY_S: Final = 0.0012332316583976016
OLD_BREAK_EVEN_BYTES: Final = 1852.091296201751
FISHER_SHA256: Final = "765457d424eaf1de7e05ed8703853175ef415bd3f19fb00137a74a29de52ae00"
CONTROL_ARCHIVE_BYTES: Final = 94_344

DEFAULT_BASE: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m1_c2_glue_rebuild_20260719/pre_archive.zip"
)
DEFAULT_DECODER: Final = Path(
    "/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py"
)
DEFAULT_XI0: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b3_producers_20260720T185300Z/xi0.xi0"
)
DEFAULT_TARGET: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m2_live_target_selection_20260720T1528Z/inflated/0.raw"
)
DEFAULT_UPSTREAM: Final = Path(
    "/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709"
)
DEFAULT_FISHER: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/fisher_ev/"
    "fisher_ev_ordering_38077.jsonl.br"
)


class R1B6MeasurementError(RuntimeError):
    """Malformed custody or a failed exact receiver-bound measurement."""


class R1B6SingletonInfeasibleError(R1B6MeasurementError):
    """The requested signed singleton has no admissible exact uint8 preimage."""


def _custody(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise R1B6MeasurementError(f"receipt overwrite refused: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    partial = resolved.with_name(f".{resolved.name}.tmp.{os.getpid()}")
    if partial.exists():
        raise R1B6MeasurementError(f"stale receipt temporary requires review: {partial}")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, resolved)
    directory_fd = os.open(resolved.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _zero_boundary_payload() -> bytes:
    packet = BoundaryCoordinatePacket(
        family=FrameFamily.WINDOWED_CURVELET,
        frame_config=asdict(WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)),
        scorer_height=SCORER_HEIGHT,
        scorer_width=SCORER_WIDTH,
        atom_indices=np.asarray([0], dtype=np.uint32),
        coefficients=np.zeros((PAIR_COUNT, 1, RGB_CHANNELS), dtype=np.int8),
        scales=np.ones(PAIR_COUNT, dtype=np.float16),
    )
    return encode_boundary_packet(packet)


def _raw_memmap(path: Path, *, pairs: int) -> np.memmap:
    expected = pairs * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS
    if path.stat().st_size != expected:
        raise R1B6MeasurementError(f"raw bytes {path.stat().st_size} != {expected}")
    return np.memmap(
        path,
        mode="r",
        dtype=np.uint8,
        shape=(pairs, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS),
    )


def _load_fisher_rows(path: Path, *, pair_cap: int, max_sites: int) -> list[list[Any]]:
    if sha256_file(path) != FISHER_SHA256:
        raise R1B6MeasurementError("Fisher ordering SHA-256 drifted")
    lines = brotli.decompress(path.read_bytes()).decode("utf-8").splitlines()
    if len(lines) != 38_078:
        raise R1B6MeasurementError("Fisher ordering row count drifted")
    header = json.loads(lines[0])
    if header.get("schema") != "r1b5_fisher_ev_ordering_jsonl.v1":
        raise R1B6MeasurementError("Fisher ordering schema drifted")
    rows = [json.loads(line) for line in lines[1:]]
    selected = [row for row in rows if int(row[0]) < pair_cap][:max_sites]
    if not selected:
        raise R1B6MeasurementError("Fisher prefix contains no selected cells")
    if any(int(first[6]) > int(second[6]) for first, second in pairwise(selected)):
        raise R1B6MeasurementError("Fisher prefix violated necessity-tier order")
    return selected


def _signed_rounding_block(
    operator: DisjointResizeOperator,
    rounded_rgb: np.ndarray,
    row: int,
    col: int,
    sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one exact same-rounded-bin uint8 endpoint encoder-side."""

    if sign not in (-1, 1):
        raise R1B6MeasurementError("rounding sign must be -1 or +1")
    row_support = operator.row_supports[row]
    col_support = operator.col_supports[col]
    coefficients = np.outer(row_support.numerators, col_support.numerators).astype(
        np.int64
    ).reshape(-1)
    denominator = int(row_support.denominator) * int(col_support.denominator)
    common_gcd = reduce(gcd, (int(value) for value in coefficients))
    half_step = ((denominator - 1) // 2 // common_gcd) * common_gcd
    if half_step <= 0:
        raise R1B6MeasurementError("resize cell has no signed rounding-bin interior")
    block = np.empty(
        (len(row_support.indices), len(col_support.indices), RGB_CHANNELS), dtype=np.uint8
    )
    numerators = np.empty(RGB_CHANNELS, dtype=np.int64)
    for channel, rounded in enumerate(np.asarray(rounded_rgb, dtype=np.int64)):
        target_integer = int(rounded) * denominator + sign * half_step
        if not 0 <= target_integer <= 255 * denominator:
            raise R1B6SingletonInfeasibleError("signed target escaped uint8 gamut")
        solved = solve_bounded_integer_block(
            coefficients.tolist(),
            denominator,
            target_integer / denominator,
            target_integer=target_integer,
            preferred=np.full(len(coefficients), int(rounded), dtype=np.float64),
            max_nodes=4096,
        )
        if solved.status != BlockSolveStatus.FEASIBLE_EXACT or not solved.exact_target_rational:
            raise R1B6SingletonInfeasibleError("singleton preimage was not exact")
        values = np.asarray(solved.values, dtype=np.uint8).reshape(
            len(row_support.indices), len(col_support.indices)
        )
        if int(np.dot(coefficients, values.reshape(-1).astype(np.int64))) != target_integer:
            raise R1B6MeasurementError("singleton exact numerator verification failed")
        if (target_integer + denominator // 2) // denominator != int(rounded):
            raise R1B6MeasurementError("singleton changed the rounded scorer byte")
        block[:, :, channel] = values
        numerators[channel] = target_integer
    return block, numerators


def _source_closest_block(
    operator: DisjointResizeOperator,
    rounded_rgb: np.ndarray,
    source_frame: np.ndarray,
    row: int,
    col: int,
) -> tuple[int, np.ndarray, np.ndarray, int]:
    row_support = operator.row_supports[row]
    col_support = operator.col_supports[col]
    source = source_frame[
        np.ix_(row_support.indices, col_support.indices, range(RGB_CHANNELS))
    ].astype(np.int32)
    choices: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for sign in (-1, 1):
        try:
            block, numerators = _signed_rounding_block(
                operator, rounded_rgb, row, col, sign
            )
        except R1B6SingletonInfeasibleError:
            continue
        distance = int(np.sum((block.astype(np.int32) - source) ** 2, dtype=np.int64))
        choices.append((distance, sign, block, numerators))
    if not choices:
        raise R1B6SingletonInfeasibleError(
            "neither singleton sign has an exact uint8 preimage"
        )
    distance, sign, block, numerators = min(choices, key=lambda value: (value[0], value[1]))
    return sign, block, numerators, distance


def _replay_for_rows(
    *,
    baseline_raw: Path,
    target_raw: Path,
    rows: list[list[Any]],
    pair_cap: int,
) -> tuple[bytes, dict[str, Any]]:
    baseline = _raw_memmap(baseline_raw, pairs=pair_cap)
    target = _raw_memmap(target_raw, pairs=PAIR_COUNT)
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )
    writes: list[ReplayWrite] = []
    signs: dict[str, int] = {"-1": 0, "1": 0}
    distances = 0
    numerator_digest = hashlib.sha256()
    selected_indices: list[int] = []
    infeasible_indices: list[int] = []
    for fisher_row in rows:
        pair, row, col, linear_index = map(int, fisher_row[:4])
        row_support = operator.row_supports[row]
        col_support = operator.col_supports[col]
        # R1b4's zero-boundary realization is scorer-byte exact; recover the
        # owned rounded descriptor from the exact factor-2 projection.
        coefficients = np.outer(row_support.numerators, col_support.numerators).astype(
            np.int64
        )
        denominator = int(row_support.denominator) * int(col_support.denominator)
        block_now = baseline[pair, 1][
            np.ix_(row_support.indices, col_support.indices, range(RGB_CHANNELS))
        ].astype(np.int64)
        rounded = np.rint(
            np.sum(block_now * coefficients[:, :, None], axis=(0, 1), dtype=np.int64)
            / denominator
        ).clip(0, 255).astype(np.uint8)
        try:
            sign, block, numerators, distance = _source_closest_block(
                operator, rounded, target[pair, 1], row, col
            )
        except R1B6SingletonInfeasibleError:
            infeasible_indices.append(linear_index)
            continue
        signs[str(sign)] += 1
        distances += distance
        selected_indices.append(linear_index)
        numerator_digest.update(np.asarray([linear_index, *numerators], dtype="<i8").tobytes())
        for local_y, camera_y in enumerate(row_support.indices):
            for local_x, camera_x in enumerate(col_support.indices):
                for channel in range(RGB_CHANNELS):
                    writes.append(
                        ReplayWrite(
                            pair,
                            1,
                            int(camera_y),
                            int(camera_x),
                            channel,
                            int(block[local_y, local_x, channel]),
                        )
                    )
    writes.sort()
    if not selected_indices:
        raise R1B6MeasurementError("every requested singleton endpoint was infeasible")
    selected_index_set = set(selected_indices)
    payload = encode_replay_payload(writes)
    return payload, {
        "requested_site_count": len(rows),
        "selected_site_count": len(selected_indices),
        "infeasible_site_count": len(infeasible_indices),
        "road_lane_sites": sum(
            int(row[6]) == 0 and int(row[3]) in selected_index_set for row in rows
        ),
        "other_edge_sites": sum(
            int(row[6]) == 1 and int(row[3]) in selected_index_set for row in rows
        ),
        "nonedge_sites": sum(
            int(row[6]) == 2 and int(row[3]) in selected_index_set for row in rows
        ),
        "replay_write_count": len(writes),
        "replay_payload_bytes": len(payload),
        "sign_histogram": signs,
        "source_distance_sum": distances,
        "selected_linear_indices_sha256": hashlib.sha256(
            np.asarray(selected_indices, dtype="<u8").tobytes()
        ).hexdigest(),
        "infeasible_linear_indices_sha256": hashlib.sha256(
            np.asarray(infeasible_indices, dtype="<u8").tobytes()
        ).hexdigest(),
        "exact_numerator_custody_sha256": numerator_digest.hexdigest(),
        "encoder_search_only": True,
        "receiver_search_invocations": 0,
    }


def _load_model(upstream: Path):
    sys.path.insert(0, str(upstream))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = DistortionNet().eval().to("cpu")
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    return torch, model


def _hard_measure(
    *, target_raw: Path, rows: dict[str, Path], pair_cap: int, upstream: Path
) -> dict[str, Any]:
    torch, model = _load_model(upstream)
    target = _raw_memmap(target_raw, pairs=PAIR_COUNT)
    candidates = {name: _raw_memmap(path, pairs=pair_cap) for name, path in rows.items()}
    accumulators = {
        name: {"flips": 0, "pose_squared_error": []} for name in candidates
    }
    for start in range(0, pair_cap, BATCH_SIZE):
        stop = min(pair_cap, start + BATCH_SIZE)
        target_batch = torch.from_numpy(np.array(target[start:stop], copy=True))
        with torch.inference_mode():
            target_pose, target_logits = model(target_batch)
        target_labels = target_logits.argmax(dim=1)
        target_pose6 = target_pose["pose"][:, :6].cpu().numpy().astype(np.float64)
        for name, candidate in candidates.items():
            candidate_batch = torch.from_numpy(np.array(candidate[start:stop], copy=True))
            with torch.inference_mode():
                candidate_pose, candidate_logits = model(candidate_batch)
            accumulators[name]["flips"] += int(
                torch.count_nonzero(target_labels != candidate_logits.argmax(dim=1)).item()
            )
            candidate_pose6 = candidate_pose["pose"][:, :6].cpu().numpy().astype(np.float64)
            accumulators[name]["pose_squared_error"].append(
                (target_pose6 - candidate_pose6) ** 2
            )
    result: dict[str, Any] = {}
    pixels = pair_cap * SCORER_HEIGHT * SCORER_WIDTH
    for name, accumulator in accumulators.items():
        d_seg = accumulator["flips"] / pixels
        d_pose = float(np.concatenate(accumulator["pose_squared_error"]).mean())
        result[name] = {
            "flip_count": accumulator["flips"],
            "d_seg": d_seg,
            "d_pose": d_pose,
            "seg_component": 100.0 * d_seg,
            "pose_component": math.sqrt(10.0 * d_pose),
            "nonrate_score": 100.0 * d_seg + math.sqrt(10.0 * d_pose),
        }
    return result


def _decode_sealed_arm(
    *,
    label: str,
    base_archive: Path,
    base_decoder: Path,
    boundary_payload: bytes,
    replay_payload: bytes,
    xi0_payload: bytes,
    source_hashes: dict[str, str],
    root: Path,
    pair_cap: int,
    workers: int,
) -> dict[str, Any]:
    unsealed = root / f"{label}_unsealed.zip"
    discovery_raw = root / f"{label}_discovery.raw"
    discovery_receipt = root / f"{label}_discovery_decode.json"
    sealed = root / f"{label}_sealed.zip"
    final_raw = root / f"{label}_sealed.raw"
    final_receipt = root / f"{label}_sealed_decode.json"
    build = build_r1b4_archive(
        base_archive=base_archive,
        boundary_payload=boundary_payload,
        replay_payload=replay_payload,
        xi0_payload=xi0_payload,
        source_manifest_hashes=source_hashes,
        output=unsealed,
        artifact_role="receiver_smoke_only",
        pair_cap=pair_cap,
    )
    first = decode_r1b4_archive(
        archive=unsealed,
        base_decoder=base_decoder,
        scratch_root=root / f"{label}_scratch_first",
        output_raw=discovery_raw,
        receipt_path=discovery_receipt,
        workers=workers,
        allow_unsealed_discovery=True,
    )
    seal = seal_output_assertion(unsealed, decoded_path=discovery_raw, output=sealed)
    second = decode_r1b4_archive(
        archive=sealed,
        base_decoder=base_decoder,
        scratch_root=root / f"{label}_scratch_second",
        output_raw=final_raw,
        receipt_path=final_receipt,
        workers=workers,
    )
    if sha256_file(discovery_raw) != sha256_file(final_raw):
        raise R1B6MeasurementError(f"{label} receiver double decode drifted")
    return {
        "build": build,
        "seal": seal,
        "discovery_decode": first,
        "sealed_decode": second,
        "sealed_archive": _custody(sealed),
        "discovery_raw": discovery_raw,
        "final_raw": final_raw,
    }


def execute(args: argparse.Namespace) -> int:
    started = time.monotonic()
    if args.batch_size != BATCH_SIZE or not 2 <= args.pair_cap <= PAIR_COUNT:
        raise R1B6MeasurementError("batch-size must be 16 and pair-cap must be in [2,600]")
    if args.max_sites <= 0:
        raise R1B6MeasurementError("max-sites must be positive")
    root = args.artifact_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise R1B6MeasurementError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    raw_bytes = args.pair_cap * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS
    free_bytes = shutil.disk_usage(root).free
    required = 6 * raw_bytes + 2 * 1024**3
    if free_bytes < required:
        raise R1B6MeasurementError(
            f"storage preflight refused: free={free_bytes} required={required}"
        )
    base = args.base_archive.expanduser().resolve(strict=True)
    decoder = args.base_decoder.expanduser().resolve(strict=True)
    xi0 = args.xi0.expanduser().resolve(strict=True)
    target = args.target_raw.expanduser().resolve(strict=True)
    fisher = args.fisher_ordering.expanduser().resolve(strict=True)
    upstream = args.upstream.expanduser().resolve(strict=True)
    rows = _load_fisher_rows(fisher, pair_cap=args.pair_cap, max_sites=args.max_sites)
    boundary = _zero_boundary_payload()
    xi0_payload = xi0.read_bytes()
    source_hashes = {
        "fisher_ordering": sha256_file(fisher),
        "target_raw": sha256_file(target),
    }
    baseline = _decode_sealed_arm(
        label="baseline",
        base_archive=base,
        base_decoder=decoder,
        boundary_payload=boundary,
        replay_payload=encode_replay_payload(()),
        xi0_payload=xi0_payload,
        source_hashes=source_hashes,
        root=root,
        pair_cap=args.pair_cap,
        workers=args.decode_workers,
    )
    replay, replay_measurement = _replay_for_rows(
        baseline_raw=baseline["final_raw"],
        target_raw=target,
        rows=rows,
        pair_cap=args.pair_cap,
    )
    candidate = _decode_sealed_arm(
        label="candidate",
        base_archive=base,
        base_decoder=decoder,
        boundary_payload=boundary,
        replay_payload=replay,
        xi0_payload=xi0_payload,
        source_hashes=source_hashes,
        root=root,
        pair_cap=args.pair_cap,
        workers=args.decode_workers,
    )
    hard = _hard_measure(
        target_raw=target,
        rows={"baseline": baseline["final_raw"], "candidate": candidate["final_raw"]},
        pair_cap=args.pair_cap,
        upstream=upstream,
    )
    realized = hard["baseline"]["nonrate_score"] - hard["candidate"]["nonrate_score"]
    seg_recovery = hard["baseline"]["seg_component"] - hard["candidate"]["seg_component"]
    scheduled = replay_measurement["selected_site_count"] * 100.0 / (
        args.pair_cap * SCORER_HEIGHT * SCORER_WIDTH
    )
    realization_fraction = seg_recovery / scheduled if scheduled else 0.0
    break_even = breakeven_bytes(max(0.0, realized))
    archive_delta = candidate["sealed_archive"]["bytes"] - CONTROL_ARCHIVE_BYTES
    verdict = (
        "MEASURED_PREFIX_RECEIVER_BOUND_SINGLETON_POSITIVE"
        if realized > 0.0
        else "MEASURED_PREFIX_RECEIVER_BOUND_SINGLETON_NONPOSITIVE"
    )
    cleanup_rows = []
    for arm in (baseline, candidate):
        for key in ("discovery_raw", "final_raw"):
            path = arm[key]
            cleanup_rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "reason": "success-only receiver/scorer scratch reproducible from sealed archive",
                    "delete_after_receipt_fsync": not args.preserve_raw,
                }
            )
    receipt = {
        "schema": "r1b6_admissible_carrier_prefix_measurement.v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "verdict": verdict,
        "verdict_scope": (
            f"exact Fisher-ordered prefix n{args.pair_cap} with at most {args.max_sites} sites, "
            "absolute-write R1K1 receiver replay, R1b4 sealed double decode, and hard macOS "
            "CPU Torch only; not n600, not compact-binary-v2, not production rank4 custody, "
            "not a contest score, and no boundary/full-kernel family negative"
        ),
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        },
        "runtime": {
            "argv": sys.argv,
            "platform": platform.platform(),
            "python": sys.version,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "pair_cap": args.pair_cap,
            "decode_workers": args.decode_workers,
            "elapsed_seconds": time.monotonic() - started,
        },
        "storage_preflight": {
            "artifact_root": str(root),
            "free_bytes": free_bytes,
            "required_free_bytes": required,
            "expected_one_raw_bytes": raw_bytes,
            "ok": True,
        },
        "inputs": {
            "base_archive": _custody(base),
            "base_decoder": _custody(decoder),
            "xi0": _custody(xi0),
            "target_raw": _custody(target),
            "fisher_ordering": _custody(fisher),
        },
        "selection_and_replay": replay_measurement,
        "receiver": {
            "baseline_sealed_archive": baseline["sealed_archive"],
            "candidate_sealed_archive": candidate["sealed_archive"],
            "candidate_archive_delta_bytes_vs_94344_control": archive_delta,
            "candidate_under_old_break_even": archive_delta <= OLD_BREAK_EVEN_BYTES,
            "baseline_decode_seconds": baseline["sealed_decode"]["decode_seconds"],
            "candidate_decode_seconds": candidate["sealed_decode"]["decode_seconds"],
            "decode_gate_seconds": 1800.0,
            "decode_gate_pass": candidate["sealed_decode"]["decode_seconds"] <= 1800.0,
            "deterministic_double_decode": True,
            "search_invocations": 0,
        },
        "hard_oracle": hard,
        "realization": {
            "scheduled_prefix_seg_recovery_s": scheduled,
            "realized_prefix_seg_recovery_s": seg_recovery,
            "realized_prefix_combined_nonrate_recovery_s": realized,
            "seg_realization_fraction": realization_fraction,
            "old_anchor": {
                "realized_recovery_s": OLD_REALIZED_RECOVERY_S,
                "break_even_bytes": OLD_BREAK_EVEN_BYTES,
                "realization_fraction": 0.09462121664378247,
            },
            "new_prefix_break_even_bytes": break_even,
            "canonical_refinement_eligible": args.pair_cap == PAIR_COUNT,
            "equation": "B=max(0,Delta_S_realized)*37545489/25",
        },
        "cleanup": {
            "schema": "certified_rebuildable_scratch_cleanup.v1",
            "rows": cleanup_rows,
            "preserve_raw": args.preserve_raw,
        },
    }
    _atomic_json(args.output, receipt)
    if not args.preserve_raw:
        for row in cleanup_rows:
            Path(row["path"]).unlink()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-archive", type=Path, default=DEFAULT_BASE)
    result.add_argument("--base-decoder", type=Path, default=DEFAULT_DECODER)
    result.add_argument("--xi0", type=Path, default=DEFAULT_XI0)
    result.add_argument("--target-raw", type=Path, default=DEFAULT_TARGET)
    result.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    result.add_argument("--fisher-ordering", type=Path, default=DEFAULT_FISHER)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--pair-cap", type=int, default=16)
    result.add_argument("--max-sites", type=int, default=512)
    result.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    result.add_argument("--decode-workers", type=int, default=4)
    result.add_argument("--preserve-raw", action="store_true")
    return result


def main() -> None:
    raise SystemExit(execute(parser().parse_args()))


if __name__ == "__main__":
    main()
