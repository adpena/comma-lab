#!/usr/bin/env python3
"""Calibrate the R1b5 xi0 coordinate-to-frame0 warp on a held-out prefix.

This is a macOS-CPU advisory measurement.  The fit is conditioned on the exact
source frame1 (the Seg-correct state); a separate receiver-transfer measurement
uses the decoded control frame1 and requires per-pair Seg invariance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.integer_plane_emitter import (  # noqa: E402
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PLANE_COUNT,
    RGB_CHANNELS,
)
from tac.boundary_math.integer_plane_emitter_byte_close import (  # noqa: E402
    decode_counted_archive,
)
from tac.boundary_math.r1b4_section_receiver import _translate_frame0, sha256_file  # noqa: E402
from tac.optimization.r1b3_producer_preflight import decode_xi0_payload  # noqa: E402

DEFAULT_BASE: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m1_c2_glue_rebuild_20260719/pre_archive.zip"
)
DEFAULT_DECODER: Final = Path(
    "/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py"
)
DEFAULT_XI0: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b3_producers_20260720T185300Z/xi0.xi0"
)
DEFAULT_CACHE: Final = Path(
    "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
)
DEFAULT_UPSTREAM: Final = Path(
    "/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709"
)
DEFAULT_ARTIFACT_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/xi0_calibration"
)
PAIR_CAP: Final = 8
TRAIN_INDICES: Final = (0, 1, 2, 3)
HELDOUT_INDICES: Final = (4, 5, 6, 7)
SHIFTS: Final = np.arange(-16, 17, dtype=np.int16)
SEED: Final = 1234


class Xi0CalibrationError(RuntimeError):
    """Fail-closed xi0 calibration error."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _custody(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Xi0CalibrationError(f"receipt overwrite refused: {path}")
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _policy_shifts(values: np.ndarray, policy: Mapping[str, float | str]) -> np.ndarray:
    kind = str(policy["kind"])
    if kind == "control":
        raw = np.zeros(values.shape, dtype=np.float64)
    elif kind == "existing_affine":
        raw = (values.astype(np.float64) - 31.0) * 1.0
    elif kind == "scalar":
        raw = values.astype(np.float64) * float(policy["gain"])
    elif kind == "affine":
        raw = (values.astype(np.float64) - float(policy["center"])) * float(policy["gain"])
    else:
        raise Xi0CalibrationError(f"unknown xi0 policy kind: {kind}")
    return np.clip(np.rint(raw), -16, 16).astype(np.int16)


def _fit_scalar(values: np.ndarray, losses: np.ndarray, train_indices: tuple[int, ...]) -> dict[str, Any]:
    """Fit shift=round(gain*xi0) using training rows only."""

    best: tuple[tuple[float, float, float], float, np.ndarray] | None = None
    for gain_index in range(-150, 151):
        gain = gain_index / 200.0
        shifts = _policy_shifts(values, {"kind": "scalar", "gain": gain})
        loss = float(np.mean([losses[index, int(shifts[index]) + 16] for index in train_indices]))
        key = (loss, abs(gain), gain)
        if best is None or key < best[0]:
            best = (key, gain, shifts)
    assert best is not None
    return {
        "kind": "scalar",
        "gain": best[1],
        "maximum_absolute_pixels": 16,
        "train_mean_d_pose": best[0][0],
        "all_shifts": best[2].astype(int).tolist(),
    }


def _fit_affine(values: np.ndarray, losses: np.ndarray, train_indices: tuple[int, ...]) -> dict[str, Any]:
    """Fit shift=round((xi0-center)*gain) using training rows only."""

    best: tuple[tuple[float, float, float, float, float], float, float, np.ndarray] | None = None
    for center_index in range(80, 169):
        center = center_index / 4.0
        for gain_index in range(-160, 161):
            gain = gain_index / 40.0
            shifts = _policy_shifts(
                values, {"kind": "affine", "center": center, "gain": gain}
            )
            loss = float(
                np.mean([losses[index, int(shifts[index]) + 16] for index in train_indices])
            )
            key = (loss, abs(gain), abs(center - 31.0), gain, center)
            if best is None or key < best[0]:
                best = (key, center, gain, shifts)
    assert best is not None
    return {
        "kind": "affine",
        "center": best[1],
        "gain": best[2],
        "pixels_per_unit": best[2],
        "maximum_absolute_pixels": 16,
        "train_mean_d_pose": best[0][0],
        "all_shifts": best[3].astype(int).tolist(),
    }


def _split_metrics(
    losses: np.ndarray,
    dim0_losses: np.ndarray,
    shifts: np.ndarray,
    indices: tuple[int, ...],
) -> dict[str, Any]:
    rows = []
    for index in indices:
        shift_index = int(shifts[index]) + 16
        d_pose = float(losses[index, shift_index])
        dim0 = float(dim0_losses[index, shift_index])
        rows.append(
            {
                "pair_index": index,
                "shift_pixels": int(shifts[index]),
                "d_pose": d_pose,
                "pose_dim0_squared_error": dim0,
                "pose_dim0_error_share": 0.0 if d_pose == 0.0 else dim0 / (6.0 * d_pose),
            }
        )
    return {
        "pair_indices": list(indices),
        "mean_d_pose": float(np.mean([row["d_pose"] for row in rows])),
        "mean_pose_dim0_squared_error": float(
            np.mean([row["pose_dim0_squared_error"] for row in rows])
        ),
        "per_pair": rows,
    }


def _pose_outputs(model: Any, torch: Any, batch: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(batch))
    with torch.inference_mode():
        posenet_input, _ = model.preprocess_input(tensor)
        outputs = model.posenet(posenet_input)["pose"][:, :6]
    return outputs.detach().cpu().numpy().astype(np.float64)


def _full_outputs(model: Any, torch: Any, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.ascontiguousarray(batch))
    with torch.inference_mode():
        pose, seg = model(tensor)
    return (
        pose["pose"][:, :6].detach().cpu().numpy().astype(np.float64),
        seg.detach().cpu().numpy(),
    )


def _translate_batch(frame0: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    translated = np.empty_like(frame0)
    for index, shift in enumerate(shifts):
        result, realized = _translate_frame0(
            frame0[index],
            float(shift),
            {"center": 0.0, "pixels_per_unit": 1.0, "maximum_absolute_pixels": 16},
        )
        if realized != int(shift):
            raise Xi0CalibrationError("receiver translation did not realize calibrated shift")
        translated[index] = result
    return translated


def _storage_preflight(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    raw_bytes = PAIR_CAP * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS
    required = raw_bytes * 3 + (128 << 20)
    usage = shutil.disk_usage(root)
    result = {
        "tier": "VertigoDataTier" if str(root).startswith("/Volumes/VertigoDataTier/") else "other",
        "decoded_raw_bytes": raw_bytes,
        "required_free_bytes": required,
        "free_bytes": usage.free,
        "ok": usage.free >= required,
    }
    if result["tier"] != "VertigoDataTier" or not result["ok"]:
        raise Xi0CalibrationError("xi0 calibration storage preflight failed closed")
    return result


def _load_gt_pose(cache: Path) -> np.ndarray:
    with zipfile.ZipFile(cache, "r") as archive, archive.open("gt_poses.npy", "r") as handle:
        poses = np.load(handle, allow_pickle=False)
    if poses.shape != (600, 6) or not np.all(np.isfinite(poses)):
        raise Xi0CalibrationError("GT pose custody geometry/values invalid")
    return np.asarray(poses[:PAIR_CAP], dtype=np.float64)


def execute(args: argparse.Namespace) -> int:
    started = time.monotonic()
    root = args.artifact_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise Xi0CalibrationError(f"artifact root is not empty: {root}")
    storage = _storage_preflight(root)
    base = args.base_archive.expanduser().resolve(strict=True)
    decoder = args.base_decoder.expanduser().resolve(strict=True)
    xi0_path = args.xi0.expanduser().resolve(strict=True)
    cache = args.cache.expanduser().resolve(strict=True)
    upstream = args.upstream.expanduser().resolve(strict=True)

    from tools.measure_c2_integer_plane_emitter import _load_distortion_net, _load_real_cache

    cache_fields, cache_sha = _load_real_cache(cache)
    gt_pose_cache = _load_gt_pose(cache)
    xi0_values = decode_xi0_payload(xi0_path.read_bytes()).astype(np.float64)[:PAIR_CAP]
    if not np.allclose(xi0_values, gt_pose_cache[:, 0], rtol=0.0, atol=2e-2):
        raise Xi0CalibrationError("xi0 payload is not the cache PoseNet coordinate-zero target")
    model, torch, scorer_hashes = _load_distortion_net(upstream, args.cpu_threads)

    with tempfile.TemporaryDirectory(prefix="r1b5_xi0_", dir=root) as temp_name:
        temp = Path(temp_name)
        control_raw = temp / "control.raw"
        decode = decode_counted_archive(
            archive=base,
            base_decoder=decoder,
            scratch_root=temp / "decode_scratch",
            pair_cap=PAIR_CAP,
            output_raw=control_raw,
            workers=args.decode_workers,
        )
        raw_custody = _custody(control_raw)
        shape = (PAIR_CAP, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS)
        control_map = np.memmap(control_raw, mode="r", dtype=np.uint8, shape=shape)
        control = np.asarray(control_map).copy()
        del control_map

        gt_f0 = np.stack(
            [np.asarray(cache_fields["gt_f0"][index]) for index in range(PAIR_CAP)]
        )
        gt_f1 = np.stack(
            [np.asarray(cache_fields["gt_f1"][index]) for index in range(PAIR_CAP)]
        )
        source_batch = np.stack((gt_f0, gt_f1), axis=1)
        source_pose = _pose_outputs(model, torch, source_batch)
        cache_pose_max_abs = float(np.max(np.abs(source_pose - gt_pose_cache)))

        losses = np.empty((PAIR_CAP, SHIFTS.size), dtype=np.float64)
        dim0_losses = np.empty_like(losses)
        for shift_index, shift in enumerate(SHIFTS):
            shifted = _translate_batch(
                control[:, 0], np.full(PAIR_CAP, int(shift), dtype=np.int16)
            )
            conditioned = np.stack((shifted, gt_f1), axis=1)
            predicted = _pose_outputs(model, torch, conditioned)
            squared = np.square(predicted - source_pose)
            losses[:, shift_index] = np.mean(squared, axis=1)
            dim0_losses[:, shift_index] = squared[:, 0]

        scalar = _fit_scalar(xi0_values, losses, TRAIN_INDICES)
        affine = _fit_affine(xi0_values, losses, TRAIN_INDICES)
        policies: dict[str, dict[str, Any]] = {
            "control": {"kind": "control"},
            "existing_affine": {
                "kind": "existing_affine",
                "center": 31.0,
                "gain": 1.0,
                "pixels_per_unit": 1.0,
                "maximum_absolute_pixels": 16,
            },
            "fitted_scalar": scalar,
            "fitted_affine": affine,
        }
        conditioned_rows: dict[str, Any] = {}
        for name, policy in policies.items():
            shifts = _policy_shifts(xi0_values, policy)
            conditioned_rows[name] = {
                "policy": policy,
                "train": _split_metrics(losses, dim0_losses, shifts, TRAIN_INDICES),
                "heldout": _split_metrics(losses, dim0_losses, shifts, HELDOUT_INDICES),
            }

        source_pose_full, source_seg = _full_outputs(model, torch, source_batch)
        receiver_rows: dict[str, Any] = {}
        control_seg: np.ndarray | None = None
        for name, policy in policies.items():
            shifts = _policy_shifts(xi0_values, policy)
            shifted = _translate_batch(control[:, 0], shifts)
            receiver_batch = np.stack((shifted, control[:, 1]), axis=1)
            predicted_pose, predicted_seg = _full_outputs(model, torch, receiver_batch)
            squared = np.square(predicted_pose - source_pose_full)
            argmax = np.argmax(predicted_seg, axis=1)
            source_argmax = np.argmax(source_seg, axis=1)
            d_seg = np.mean(argmax != source_argmax, axis=(1, 2))
            if control_seg is None:
                control_seg = predicted_seg.copy()
                control_d_seg = d_seg.copy()
            elif not np.array_equal(predicted_seg, control_seg) or not np.array_equal(
                d_seg, control_d_seg
            ):
                raise Xi0CalibrationError(
                    f"xi0 policy {name} changed per-pair Seg output despite frame1 invariance"
                )
            per_pair = []
            for index in range(PAIR_CAP):
                d_pose = float(np.mean(squared[index]))
                dim0 = float(squared[index, 0])
                per_pair.append(
                    {
                        "pair_index": index,
                        "shift_pixels": int(shifts[index]),
                        "d_pose": d_pose,
                        "d_seg": float(d_seg[index]),
                        "pose_dim0_squared_error": dim0,
                        "pose_dim0_error_share": 0.0 if d_pose == 0.0 else dim0 / (6.0 * d_pose),
                    }
                )
            receiver_rows[name] = {
                "policy": policy,
                "mean_d_pose": float(np.mean([row["d_pose"] for row in per_pair])),
                "mean_d_seg": float(np.mean([row["d_seg"] for row in per_pair])),
                "per_pair": per_pair,
            }

        heldout_rank = sorted(
            (conditioned_rows[name]["heldout"]["mean_d_pose"], name)
            for name in ("existing_affine", "fitted_scalar", "fitted_affine")
        )
        selected_name = heldout_rank[0][1]
        control_heldout = conditioned_rows["control"]["heldout"]["mean_d_pose"]
        selected_heldout = conditioned_rows[selected_name]["heldout"]["mean_d_pose"]
        calibration_improves = selected_heldout < control_heldout
        cleanup = {
            **raw_custody,
            "reason": "success-only capped decode reproducible from hash-bound archive and decoder",
            "deleted_by_temporary_directory": True,
            "score_claim": False,
        }

    result = {
        "schema": "r1b5_xi0_coordinate_warp_calibration.v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "verdict": (
            "MEASURED_PREFIX_CALIBRATION_PRODUCTION_ADOPTION_BLOCKED_ON_SEG_CORRECT_N600_STATE"
            if calibration_improves
            else "MEASURED_PREFIX_CALIBRATION_NO_HELDOUT_IMPROVEMENT"
        ),
        "verdict_scope": (
            "exact prefix n8 macOS CPU advisory; fit on pairs 0..3, held out pairs 4..7; "
            "conditioned frame1 is exact source; receiver transfer uses decoded control frame1; "
            "not n600, not contest CPU/CUDA, no score or promotion claim"
        ),
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "seed": SEED,
            "batch_geometry": "fixed B8 for every scorer call; consumes #570 but is not n600 B16+B8",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.19108 [contest-CPU] UNMOVED",
        },
        "conditioning": {
            "seg_correct_fit_state": "exact GT frame1 held fixed for all 33 shifts",
            "receiver_transfer_state": "decoded control frame1 held byte-identical",
            "application_order": ["seg_boundary_state", "xi0_frame0_warp"],
            "per_pair_d_seg_invariance_hard_asserted": True,
        },
        "scorer_native_lenses": {
            "channel": "exact upstream RGB->bilinear->YUV6 PoseNet preprocess",
            "pose_coordinate": "public PoseNet output dim0 target; frame0 actuator only",
            "seg_factorization": "upstream SegNet reads frame1 only",
            "secant": "33 realized integer horizontal shifts through exact R1b4 edge-replicated actuator",
            "derivative_claim": False,
        },
        "split": {"train": list(TRAIN_INDICES), "heldout": list(HELDOUT_INDICES)},
        "inputs": {
            "base_archive": _custody(base),
            "base_decoder": _custody(decoder),
            "xi0": _custody(xi0_path),
            "cache": _custody(cache),
            "upstream": str(upstream),
            "scorer_hashes": scorer_hashes,
            "cache_sha256": cache_sha,
        },
        "storage_preflight": storage,
        "decode": decode,
        "cleanup": cleanup,
        "xi0_values": xi0_values.tolist(),
        "cache_pose_vs_fresh_b8_max_abs": cache_pose_max_abs,
        "shift_grid": SHIFTS.astype(int).tolist(),
        "conditioned_loss_table": {
            "d_pose": losses.tolist(),
            "pose_dim0_squared_error": dim0_losses.tolist(),
        },
        "policies": conditioned_rows,
        "receiver_transfer": receiver_rows,
        "heldout_rank": [name for _, name in heldout_rank],
        "selected_prefix_policy": selected_name if calibration_improves else None,
        "production_policy": None,
        "production_blockers": [
            "SEG_CORRECT_N600_BOUNDARY_STATE_ABSENT",
            "XI0_N600_HELDOUT_HARD_ORACLE_ABSENT",
            "CONTEST_CPU_CUDA_PARITY_ABSENT",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "source": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "tool_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        },
    }
    _atomic_json(root / "receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--xi0", type=Path, default=DEFAULT_XI0)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    raise SystemExit(execute(_parser().parse_args()))
