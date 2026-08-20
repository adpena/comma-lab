#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure a real n2 task-space P/G/A object on the frozen macOS CPU scorer.

The result is a component-level advisory receipt only.  It is not an n600
submission evaluation, does not authorize promotion, and never treats the
current competitive pointer as a score produced by this bounded measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import stat
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
UPSTREAM = REPO / "upstream"
for search_path in (SRC, UPSTREAM):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.witness_dsl.dynamic_frontier_target import (  # noqa: E402
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (  # noqa: E402
    EP725_RUNTIME_BYTES,
    EP725_RUNTIME_SHA256,
    inspect_ep725_source,
)
from tac.witness_dsl.taskspace_monolithic_pga_receiver import (  # noqa: E402
    receive_ep725_taskspace_monolithic_pga_archive,
)
from tac.witness_dsl.taskspace_outer_archive_codec import parse_taskspace_outer_archive  # noqa: E402

SCHEMA: Final = "tac.taskspace_pga_n2_macos_cpu_advisory_measurement.v2"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
SEED: Final = 1234
PAIR_COUNT: Final = 2
TARGET_CACHE_PATH: Final = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
TARGET_CACHE_BYTES: Final = 5_078_017_610
TARGET_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
TARGET_LABELS_U8_SHA256: Final = "6a9ee68a5d1ec8ec53653216d53b7406575530a2d1abf608d27547e779c6d474"
TARGET_POSES_F64_SHA256: Final = "5e9fa18c432367bd9661a8fd24bb19ed3a9e2a4a5747344e268cce092b03a2e3"
TARGET_F0_U8_SHA256: Final = "7c60022730f753bfe345e61a24e5d2c1db46df4419a2e62e79dd4d8b30dc9a3c"
TARGET_F1_U8_SHA256: Final = "86b55333a50cd971dbb70f6ccba5ec9cfec2b840b92ed6e9382a4087e36fc02b"
MODULES_SHA256: Final = "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa"
POSENET_SHA256: Final = "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"
SEGNET_SHA256: Final = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
SPINE_PATH: Final = REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/spine_refresh.json"
SPINE_SHA256: Final = "56b8039eba75426b3ee0ae4cb988fe1847a5f648b9f9d455cf5fc01911d19563"
SUPERSEDED_V1_PATH: Final = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "ep725_n2_causal_pga_control_macos_cpu_advisory_measurement.json"
)
SUPERSEDED_V1_SHA256: Final = "18d5230c73626db1ad45da1c65c1212b3744d1fed421adfbdd265975410f9643"
IMPLEMENTATION_PATHS: Final = (
    "src/tac/boundary_math/power_diagram_witness.py",
    "src/tac/witness_dsl/dynamic_frontier_target.py",
    "src/tac/witness_dsl/ep725_levelset_predictor_adapter.py",
    "src/tac/witness_dsl/taskspace_monolithic_pga_receiver.py",
    "src/tac/witness_dsl/taskspace_outer_archive_codec.py",
    "tools/measure_taskspace_pga_n2_macos_cpu.py",
)
DEFAULT_ARCHIVE: Final = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "ep725_n2_causal_pga_control.not_a_candidate.zip"
)
DEFAULT_OUTPUT: Final = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "ep725_n2_causal_pga_control_macos_cpu_advisory_measurement_v2.json"
)


class TaskspacePGAN2MeasurementError(RuntimeError):
    """Archive, target, scorer, determinism, or durable-write proof failed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TaskspacePGAN2MeasurementError("measurement receipt is not finite canonical ASCII JSON") from exc


def _read_stable_regular(path: Path, *, expected_bytes: int | None = None) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise TaskspacePGAN2MeasurementError(f"measurement input is not a regular file: {path}")
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskspacePGAN2MeasurementError(f"cannot read measurement input: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after or len(payload) != before.st_size:
        raise TaskspacePGAN2MeasurementError(f"measurement input changed while reading: {path}")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise TaskspacePGAN2MeasurementError(f"measurement input byte count changed: {path}")
    return payload


def _verify_small_scorer_custody() -> dict[str, Any]:
    paths = {
        "modules": (UPSTREAM / "modules.py", MODULES_SHA256),
        "posenet": (UPSTREAM / "models/posenet.safetensors", POSENET_SHA256),
        "segnet": (UPSTREAM / "models/segnet.safetensors", SEGNET_SHA256),
    }
    rows: dict[str, Any] = {}
    for name, (path, expected_sha256) in paths.items():
        payload = _read_stable_regular(path)
        if _sha256(payload) != expected_sha256:
            raise TaskspacePGAN2MeasurementError(f"frozen {name} custody changed")
        rows[name] = {"path": os.fspath(path), "bytes": len(payload), "sha256": expected_sha256}
    return rows


def _git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TaskspacePGAN2MeasurementError("cannot resolve git HEAD for measurement custody") from exc
    head = completed.stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise TaskspacePGAN2MeasurementError("git HEAD is not a lowercase 40-character SHA-1")
    return head


def _implementation_custody() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_path in IMPLEMENTATION_PATHS:
        payload = _read_stable_regular(REPO / relative_path)
        rows.append({"path": relative_path, "bytes": len(payload), "sha256": _sha256(payload)})
    return rows


def _superseded_v1_custody() -> dict[str, Any]:
    payload = _read_stable_regular(SUPERSEDED_V1_PATH)
    observed_sha256 = _sha256(payload)
    if observed_sha256 != SUPERSEDED_V1_SHA256:
        raise TaskspacePGAN2MeasurementError("superseded V1 receipt custody changed")
    return {
        "path": os.fspath(SUPERSEDED_V1_PATH),
        "bytes": len(payload),
        "sha256": observed_sha256,
        "schema": "tac.taskspace_pga_n2_macos_cpu_advisory_measurement.v1",
        "reason": (
            "V1 measured valid component distances but did not reopen its cited constructive spine "
            "and omitted git plus implementation-file custody; V2 adds those proofs without changing V1"
        ),
    }


def _load_target() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    spine_payload = _read_stable_regular(SPINE_PATH)
    if _sha256(spine_payload) != SPINE_SHA256:
        raise TaskspacePGAN2MeasurementError("constructive spine custody changed")
    try:
        spine = json.loads(spine_payload.decode("utf-8"))
        spine_target = spine["source_custody"]["target_cache"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TaskspacePGAN2MeasurementError("constructive spine target-cache row is unavailable") from exc
    expected_spine_target = {
        "bytes": TARGET_CACHE_BYTES,
        "consumed_as": "S0 n600 target custody",
        "content_lineage": "source-video-derived our-build",
        "path": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        "sha256": TARGET_CACHE_SHA256,
    }
    if spine_target != expected_spine_target:
        raise TaskspacePGAN2MeasurementError("constructive spine target-cache binding changed")
    try:
        metadata = TARGET_CACHE_PATH.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskspacePGAN2MeasurementError("frozen target cache is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != TARGET_CACHE_BYTES:
        raise TaskspacePGAN2MeasurementError("frozen target cache identity/size changed")
    labels = np.ascontiguousarray(open_stored_npy_memmap(TARGET_CACHE_PATH, "lstars")[:PAIR_COUNT], dtype=np.uint8)
    poses = np.ascontiguousarray(open_stored_npy_memmap(TARGET_CACHE_PATH, "gt_poses")[:PAIR_COUNT])
    f0 = np.ascontiguousarray(open_stored_npy_memmap(TARGET_CACHE_PATH, "gt_f0")[:PAIR_COUNT])
    f1 = np.ascontiguousarray(open_stored_npy_memmap(TARGET_CACHE_PATH, "gt_f1")[:PAIR_COUNT])
    observed = {
        "labels": _array_sha256(labels),
        "poses": _array_sha256(poses),
        "f0": _array_sha256(f0),
        "f1": _array_sha256(f1),
    }
    expected = {
        "labels": TARGET_LABELS_U8_SHA256,
        "poses": TARGET_POSES_F64_SHA256,
        "f0": TARGET_F0_U8_SHA256,
        "f1": TARGET_F1_U8_SHA256,
    }
    if observed != expected:
        raise TaskspacePGAN2MeasurementError("frozen n2 target slice custody changed")
    frames = np.ascontiguousarray(np.stack((f0, f1), axis=1), dtype=np.uint8)
    return (
        frames,
        labels,
        poses,
        {
            "path": os.fspath(TARGET_CACHE_PATH),
            "bytes": TARGET_CACHE_BYTES,
            "sha256": TARGET_CACHE_SHA256,
            "content_lineage": expected_spine_target["content_lineage"],
            "constructive_spine_path": os.fspath(SPINE_PATH),
            "constructive_spine_sha256": SPINE_SHA256,
            "constructive_spine_target_cache_row_exact": True,
            "fresh_full_cache_rehash_this_measurement": False,
            "full_cache_hash_claim_source": "pinned constructive-spine receipt reopened and exact row verified",
            "n2_member_sha256": expected,
        },
    )


def _score_exact_cpu(
    target_frames: np.ndarray,
    target_labels: np.ndarray,
    target_poses: np.ndarray,
    candidate_frames: np.ndarray,
) -> dict[str, Any]:
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    model = DistortionNet().eval().to(device)
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    target_tensor = torch.from_numpy(np.ascontiguousarray(target_frames).copy()).to(device)
    candidate_tensor = torch.from_numpy(np.ascontiguousarray(candidate_frames).copy()).to(device)
    with torch.inference_mode():
        target_pose_out, target_seg_out = model(target_tensor)
        candidate_pose_a, candidate_seg_a = model(candidate_tensor)
        candidate_pose_b, candidate_seg_b = model(candidate_tensor)
    for key in target_pose_out:
        if not torch.equal(candidate_pose_a[key], candidate_pose_b[key]):
            raise TaskspacePGAN2MeasurementError("PoseNet changed on deterministic candidate replay")
    if not torch.equal(candidate_seg_a, candidate_seg_b):
        raise TaskspacePGAN2MeasurementError("SegNet changed on deterministic candidate replay")
    target_pose6 = target_pose_out["pose"][..., :6]
    candidate_pose6 = candidate_pose_a["pose"][..., :6]
    target_argmax = target_seg_out.argmax(dim=1)
    candidate_argmax = candidate_seg_a.argmax(dim=1)
    target_pose_numpy = np.ascontiguousarray(target_pose6.cpu().numpy())
    target_argmax_numpy = np.ascontiguousarray(target_argmax.cpu().numpy(), dtype=np.uint8)
    if not np.array_equal(target_argmax_numpy, target_labels):
        raise TaskspacePGAN2MeasurementError("fresh frozen SegNet target differs from cached lstars")
    pose_cache_max_abs = float(np.max(np.abs(target_pose_numpy.astype(np.float64) - target_poses)))
    pose_cache_scale = max(1.0, float(np.max(np.abs(target_poses))))
    pose_cache_atol = 2.0 * float(np.finfo(np.float32).eps) * pose_cache_scale
    if pose_cache_max_abs > pose_cache_atol:
        raise TaskspacePGAN2MeasurementError(
            "fresh frozen PoseNet target differs from cache beyond two scaled fp32 epsilons: "
            f"observed={pose_cache_max_abs}, bound={pose_cache_atol}"
        )
    per_pair_pose = (candidate_pose6 - target_pose6).pow(2).mean(dim=1)
    per_pair_seg = (candidate_argmax != target_argmax).float().mean(dim=(1, 2))
    d_pose = float(per_pair_pose.mean().item())
    d_seg = float(per_pair_seg.mean().item())
    if not math.isfinite(d_pose) or not math.isfinite(d_seg) or d_pose < 0.0 or not 0.0 <= d_seg <= 1.0:
        raise TaskspacePGAN2MeasurementError("frozen scorer emitted invalid component distances")
    return {
        "d_pose": d_pose,
        "d_seg": d_seg,
        "per_pair_d_pose": [float(value) for value in per_pair_pose.cpu().tolist()],
        "per_pair_d_seg": [float(value) for value in per_pair_seg.cpu().tolist()],
        "target_pose6_f32_sha256": _array_sha256(target_pose_numpy),
        "target_labels_u8_sha256": _array_sha256(target_argmax_numpy),
        "target_pose_cache_max_abs": pose_cache_max_abs,
        "target_pose_cache_atol": pose_cache_atol,
        "target_pose_cache_parity_within_two_scaled_fp32_epsilons": True,
        "candidate_pose6_f32_sha256": _array_sha256(np.ascontiguousarray(candidate_pose6.cpu().numpy())),
        "candidate_labels_u8_sha256": _array_sha256(
            np.ascontiguousarray(candidate_argmax.cpu().numpy(), dtype=np.uint8)
        ),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": "cpu",
        "seed": SEED,
        "double_forward_exact": True,
    }


def measure(archive_path: Path) -> bytes:
    """Receive twice, score twice, and return one canonical advisory receipt."""

    frontier = load_dynamic_frontier_target(repo_root=REPO)
    verify_dynamic_frontier_target_snapshot(frontier)
    archive_bytes = _read_stable_regular(archive_path)
    parsed = parse_taskspace_outer_archive(archive_bytes)
    source = inspect_ep725_source()
    if len(source.runtime) != EP725_RUNTIME_BYTES or _sha256(source.runtime) != EP725_RUNTIME_SHA256:
        raise TaskspacePGAN2MeasurementError("explicit ep725 runtime custody changed")
    receive_kwargs = {
        "predictor_runtime": source.runtime,
        "pair_count": PAIR_COUNT,
        "timeout_seconds": 180.0,
        "expected_encoding": parsed.encoding,
        "expected_archive_sha256": parsed.archive_sha256,
        "expected_member_sha256": parsed.member_sha256,
    }
    first = receive_ep725_taskspace_monolithic_pga_archive(archive_bytes, **receive_kwargs)
    second = receive_ep725_taskspace_monolithic_pga_archive(archive_bytes, **receive_kwargs)
    if first.receipt != second.receipt or not np.array_equal(
        first.chronological_camera_frames,
        second.chronological_camera_frames,
    ):
        raise TaskspacePGAN2MeasurementError("receiver output changed across exact replay")
    candidate = np.ascontiguousarray(first.chronological_camera_frames, dtype=np.uint8)
    target, target_labels, target_poses, target_custody = _load_target()
    scorer_custody = _verify_small_scorer_custody()
    measurement = _score_exact_cpu(target, target_labels, target_poses, candidate)
    git_head_before_landing = _git_head()
    implementation_custody = _implementation_custody()
    supersedes_receipt = _superseded_v1_custody()
    verify_dynamic_frontier_target_snapshot(frontier)
    payload = {
        "schema": SCHEMA,
        "axis": AXIS,
        "scope": "real ep725 n2 component measurement; not an n600 evaluation",
        "git_head_before_landing": git_head_before_landing,
        "implementation_custody": implementation_custody,
        "supersedes_receipt": supersedes_receipt,
        "competitive_target": asdict(frontier),
        "archive": {
            "path": os.fspath(archive_path),
            "bytes": len(archive_bytes),
            "sha256": _sha256(archive_bytes),
            "encoding": parsed.encoding.value,
            "member_bytes": parsed.member_nbytes,
            "member_sha256": parsed.member_sha256,
        },
        "receiver": {
            "receipt": first.receipt.as_dict(),
            "receipt_sha256": first.receipt.receipt_sha256,
            "decoded_output_bytes": candidate.nbytes,
            "decoded_output_sha256": _array_sha256(candidate),
            "double_receive_exact": True,
            "source_archive_read_for_decode": False,
            "explicit_runtime_sha256": EP725_RUNTIME_SHA256,
        },
        "target_custody": target_custody,
        "scorer_custody": scorer_custody,
        "measurement": measurement,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "truth": {
            "component_distances_measured": True,
            "n600_evaluation": False,
            "authoritative_contest_cpu_evaluation": False,
            "authoritative_contest_cuda_evaluation": False,
            "score_claim": False,
            "candidate_archive_eligible": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "research_only": True,
        },
    }
    return _canonical_json(payload) + b"\n"


def parse_measurement_receipt(payload: bytes) -> dict[str, Any]:
    """Strict parse/re-emit with closed top-level schema and authority labels."""

    if type(payload) is not bytes or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise TaskspacePGAN2MeasurementError("measurement receipt requires exactly one terminal newline")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskspacePGAN2MeasurementError(f"measurement receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload[:-1].decode("ascii"), object_pairs_hook=unique_pairs)
    except TaskspacePGAN2MeasurementError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspacePGAN2MeasurementError("measurement receipt is not strict ASCII JSON") from exc
    expected_keys = {
        "schema",
        "axis",
        "scope",
        "git_head_before_landing",
        "implementation_custody",
        "supersedes_receipt",
        "competitive_target",
        "archive",
        "receiver",
        "target_custody",
        "scorer_custody",
        "measurement",
        "runtime",
        "truth",
    }
    if type(value) is not dict or set(value) != expected_keys or value.get("schema") != SCHEMA:
        raise TaskspacePGAN2MeasurementError("measurement receipt top-level schema is not closed V2")
    if value.get("axis") != AXIS or value.get("truth") != {
        "authoritative_contest_cpu_evaluation": False,
        "authoritative_contest_cuda_evaluation": False,
        "candidate_archive_eligible": False,
        "component_distances_measured": True,
        "n600_evaluation": False,
        "pointer_moved": False,
        "promotion_eligible": False,
        "research_only": True,
        "score_claim": False,
    }:
        raise TaskspacePGAN2MeasurementError("measurement authority labels became permissive")
    if _canonical_json(value) + b"\n" != payload:
        raise TaskspacePGAN2MeasurementError("measurement receipt is not canonical on parse-back")
    return value


def write_once_or_equal(path: Path, payload: bytes) -> None:
    """Persist exact receipt bytes without overwriting historical evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_stable_regular(path) != payload:
            raise TaskspacePGAN2MeasurementError(f"refusing to overwrite different receipt: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
    except FileExistsError:
        if _read_stable_regular(path) != payload:
            raise TaskspacePGAN2MeasurementError(f"receipt race produced different bytes: {path}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = measure(args.archive)
    parsed = parse_measurement_receipt(receipt)
    write_once_or_equal(args.output, receipt)
    summary = {
        "axis": parsed["axis"],
        "archive_sha256": parsed["archive"]["sha256"],
        "d_pose": parsed["measurement"]["d_pose"],
        "d_seg": parsed["measurement"]["d_seg"],
        "output": os.fspath(args.output),
        "receipt_bytes": len(receipt),
        "receipt_sha256": _sha256(receipt),
        "score_claim": False,
    }
    print(_canonical_json(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
