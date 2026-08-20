#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure exact n2 P/G chronology controls on the frozen macOS CPU scorer.

This is a component-level diagnostic.  It batches several ephemeral decoder
surfaces through one frozen scorer instance so the P, G/Y1, and Y0 chronology
debts can be localized without persisting dense frames or claiming an n600
contest score.
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
TOOLS = REPO / "tools"
for search_path in (SRC, UPSTREAM, TOOLS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import materialize_taskspace_pga_n2_receipt as materializer  # noqa: E402
import measure_taskspace_pga_n2_macos_cpu as baseline_measure  # noqa: E402

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.witness_dsl.bounded_target_g_encoder import (  # noqa: E402
    FrozenTargetSliceCustodyV1,
    compile_bounded_target_g_v2,
)
from tac.witness_dsl.dynamic_frontier_target import (  # noqa: E402
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (  # noqa: E402
    EP725_MEMBER_BYTES,
    EP725_MEMBER_SHA256,
    EP725_RUNTIME_BYTES,
    EP725_RUNTIME_SHA256,
    decode_ep725_counted_member_ephemeral_surface,
    inspect_ep725_source,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (  # noqa: E402
    overlay_g_on_predictor_camera_y1,
)

SCHEMA: Final = "tac.taskspace_n2_stage_ablation_macos_cpu.v1"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
SEED: Final = 1234
PAIR_COUNT: Final = 2
BASELINE_RECEIPT_PATH: Final = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_n2_causal_pga_control_receipt.json"
)
BASELINE_RECEIPT_SHA256: Final = "9868d98a86389ad255b1f49f22c76556a155b7394b4691fdb603e4e400ac7e30"
DEFAULT_OUTPUT: Final = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "ep725_n2_taskspace_stage_ablation_macos_cpu_advisory.json"
)
VARIANT_ORDER: Final = (
    "p_only",
    "p_plus_exact_semantic_g_overlay",
    "p0_plus_exact_target_y1",
    "exact_target_y0_plus_g_overlay_y1",
    "exact_target_y0_plus_predictor_y1",
    "exact_target_control",
)
IMPLEMENTATION_PATHS: Final = (
    "src/tac/witness_dsl/bounded_target_g_encoder.py",
    "src/tac/witness_dsl/ep725_levelset_predictor_adapter.py",
    "src/tac/witness_dsl/predictor_preserving_taskspace_overlay.py",
    "tools/materialize_taskspace_pga_n2_receipt.py",
    "tools/measure_taskspace_pga_n2_macos_cpu.py",
    "tools/measure_taskspace_n2_stage_ablation.py",
)


class TaskspaceN2StageAblationError(RuntimeError):
    """Ablation custody, deterministic scoring, or receipt closure failed."""


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
        raise TaskspaceN2StageAblationError("ablation receipt is not finite canonical ASCII JSON") from exc


def _read_stable_regular(path: Path) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise TaskspaceN2StageAblationError(f"custody input is not a regular file: {path}")
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskspaceN2StageAblationError(f"cannot read custody input: {path}") from exc
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_id != after_id or len(payload) != before.st_size:
        raise TaskspaceN2StageAblationError(f"custody input changed while reading: {path}")
    return payload


def _git_head() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TaskspaceN2StageAblationError("git HEAD is unavailable") from exc
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise TaskspaceN2StageAblationError("git HEAD is not canonical SHA-1 hex")
    return value


def _implementation_custody() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_path in IMPLEMENTATION_PATHS:
        payload = _read_stable_regular(REPO / relative_path)
        rows.append({"path": relative_path, "bytes": len(payload), "sha256": _sha256(payload)})
    return rows


def _score_variants(
    target_frames: np.ndarray,
    target_labels: np.ndarray,
    target_poses: np.ndarray,
    variants: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    if tuple(variants) != VARIANT_ORDER:
        raise TaskspaceN2StageAblationError("variant order escaped the preregistered closed set")
    expected_shape = target_frames.shape
    if any(value.dtype != np.uint8 or value.shape != expected_shape for value in variants.values()):
        raise TaskspaceN2StageAblationError("every ablation variant must match exact target uint8 ABI")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = DistortionNet().eval().to(torch.device("cpu"))
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    target_tensor = torch.from_numpy(np.ascontiguousarray(target_frames).copy())
    joined = np.concatenate(tuple(variants.values()), axis=0)
    joined_tensor = torch.from_numpy(np.ascontiguousarray(joined).copy())
    with torch.inference_mode():
        target_pose_out, target_seg_out = model(target_tensor)
        candidate_pose_a, candidate_seg_a = model(joined_tensor)
        candidate_pose_b, candidate_seg_b = model(joined_tensor)
    if any(not torch.equal(candidate_pose_a[key], candidate_pose_b[key]) for key in candidate_pose_a):
        raise TaskspaceN2StageAblationError("PoseNet variant batch changed on deterministic replay")
    if not torch.equal(candidate_seg_a, candidate_seg_b):
        raise TaskspaceN2StageAblationError("SegNet variant batch changed on deterministic replay")
    target_pose6 = target_pose_out["pose"][..., :6]
    target_argmax = target_seg_out.argmax(dim=1)
    target_pose_numpy = np.ascontiguousarray(target_pose6.cpu().numpy())
    target_argmax_numpy = np.ascontiguousarray(target_argmax.cpu().numpy(), dtype=np.uint8)
    if not np.array_equal(target_argmax_numpy, target_labels):
        raise TaskspaceN2StageAblationError("fresh target SegNet labels differ from frozen cache")
    pose_cache_max_abs = float(np.max(np.abs(target_pose_numpy.astype(np.float64) - target_poses)))
    pose_cache_scale = max(1.0, float(np.max(np.abs(target_poses))))
    pose_cache_atol = 2.0 * float(np.finfo(np.float32).eps) * pose_cache_scale
    if pose_cache_max_abs > pose_cache_atol:
        raise TaskspaceN2StageAblationError("fresh target PoseNet output differs from frozen cache")

    results: dict[str, dict[str, Any]] = {}
    candidate_pose6 = candidate_pose_a["pose"][..., :6]
    candidate_argmax = candidate_seg_a.argmax(dim=1)
    for variant_index, name in enumerate(VARIANT_ORDER):
        start = variant_index * PAIR_COUNT
        stop = start + PAIR_COUNT
        pose = candidate_pose6[start:stop]
        labels = candidate_argmax[start:stop]
        per_pair_pose = (pose - target_pose6).pow(2).mean(dim=1)
        per_pair_seg = (labels != target_argmax).float().mean(dim=(1, 2))
        d_pose = float(per_pair_pose.mean().item())
        d_seg = float(per_pair_seg.mean().item())
        if not math.isfinite(d_pose) or not 0.0 <= d_seg <= 1.0:
            raise TaskspaceN2StageAblationError("frozen scorer emitted invalid ablation distance")
        results[name] = {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "per_pair_d_seg": [float(value) for value in per_pair_seg.cpu().tolist()],
            "per_pair_d_pose": [float(value) for value in per_pair_pose.cpu().tolist()],
            "candidate_frames_sha256": _array_sha256(variants[name]),
            "candidate_labels_sha256": _array_sha256(np.ascontiguousarray(labels.cpu().numpy(), dtype=np.uint8)),
            "candidate_pose6_sha256": _array_sha256(np.ascontiguousarray(pose.cpu().numpy())),
        }
    baseline = results["p_plus_exact_semantic_g_overlay"]
    for row in results.values():
        row["delta_d_seg_vs_p_plus_g"] = row["d_seg"] - baseline["d_seg"]
        row["delta_d_pose_vs_p_plus_g"] = row["d_pose"] - baseline["d_pose"]
    scorer = {
        "target_labels_sha256": _array_sha256(target_argmax_numpy),
        "target_pose6_sha256": _array_sha256(target_pose_numpy),
        "target_pose_cache_max_abs": pose_cache_max_abs,
        "target_pose_cache_atol": pose_cache_atol,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "double_forward_exact": True,
    }
    return results, scorer


def measure() -> bytes:
    """Construct ephemeral stage controls, score once in a batch, and receipt."""

    frontier = load_dynamic_frontier_target(repo_root=REPO)
    verify_dynamic_frontier_target_snapshot(frontier)
    baseline_receipt_bytes = _read_stable_regular(BASELINE_RECEIPT_PATH)
    if _sha256(baseline_receipt_bytes) != BASELINE_RECEIPT_SHA256:
        raise TaskspaceN2StageAblationError("baseline materialization receipt custody changed")
    baseline_receipt = materializer.parse_materialization_receipt(baseline_receipt_bytes)
    target_frames, target_labels, target_poses, target_custody = baseline_measure._load_target()
    scorer_custody = baseline_measure._verify_small_scorer_custody()

    source = inspect_ep725_source()
    if (
        len(source.member) != EP725_MEMBER_BYTES
        or _sha256(source.member) != EP725_MEMBER_SHA256
        or len(source.runtime) != EP725_RUNTIME_BYTES
        or _sha256(source.runtime) != EP725_RUNTIME_SHA256
    ):
        raise TaskspaceN2StageAblationError("ep725 source member/runtime custody changed")
    causal = decode_ep725_counted_member_ephemeral_surface(
        source.member,
        shipped_runtime=source.runtime,
        pair_count=PAIR_COUNT,
        timeout_seconds=180.0,
    )
    cache_path, _cache_custody = materializer._load_target_cache_path()
    compile_target_labels = np.ascontiguousarray(
        open_stored_npy_memmap(cache_path, "lstars")[:PAIR_COUNT],
        dtype=np.uint8,
    )
    if not np.array_equal(compile_target_labels, target_labels):
        raise TaskspaceN2StageAblationError("G compile target differs from scorer target custody")
    profile, profile_custody = materializer._load_realization_profile()
    compiled_g = compile_bounded_target_g_v2(
        causal.predictor_state,
        compile_target_labels,
        target_custody=FrozenTargetSliceCustodyV1(
            cache_sha256=materializer.TARGET_CACHE_SHA256,
            member_name="lstars",
            source_pair_ids=causal.predictor_state.source_pair_ids,
            target_labels_sha256=_array_sha256(compile_target_labels),
        ),
        realization_profile=profile,
    )
    overlay = overlay_g_on_predictor_camera_y1(
        causal.frame1_camera,
        causal.predictor_state.labels,
        compiled_g.compiled.decoded,
    )
    p_only = np.ascontiguousarray(causal.ephemeral_surface.chronological_camera_frames, dtype=np.uint8)
    predictor_y0 = causal.chronological_camera_frames[:, 0]
    p_plus_g = np.ascontiguousarray(np.stack((predictor_y0, overlay.camera_y1), axis=1))
    p_plus_g_frame_hashes = [_sha256(memoryview(frame).cast("B")) for pair_frames in p_plus_g for frame in pair_frames]
    if p_plus_g_frame_hashes != baseline_receipt["receiver"]["receipt"]["factor2_camera_frame_sha256_chronological"]:
        raise TaskspaceN2StageAblationError("ephemeral P+G control differs from durable baseline receiver")
    variants = {
        "p_only": p_only,
        "p_plus_exact_semantic_g_overlay": p_plus_g,
        "p0_plus_exact_target_y1": np.ascontiguousarray(np.stack((predictor_y0, target_frames[:, 1]), axis=1)),
        "exact_target_y0_plus_g_overlay_y1": np.ascontiguousarray(
            np.stack((target_frames[:, 0], overlay.camera_y1), axis=1)
        ),
        "exact_target_y0_plus_predictor_y1": np.ascontiguousarray(
            np.stack((target_frames[:, 0], causal.frame1_camera), axis=1)
        ),
        "exact_target_control": np.ascontiguousarray(target_frames.copy()),
    }
    results, scorer_runtime = _score_variants(target_frames, target_labels, target_poses, variants)
    verify_dynamic_frontier_target_snapshot(frontier)
    body = {
        "schema": SCHEMA,
        "axis": AXIS,
        "scope": "real ep725 n2 stage-component ablation; not an n600 evaluation",
        "git_head_before_landing": _git_head(),
        "competitive_target": asdict(frontier),
        "baseline_materialization": {
            "path": os.fspath(BASELINE_RECEIPT_PATH),
            "bytes": len(baseline_receipt_bytes),
            "sha256": BASELINE_RECEIPT_SHA256,
            "archive_sha256": baseline_receipt["whole_object"]["selected_archive_sha256"],
            "receiver_receipt_sha256": baseline_receipt["receiver"]["receipt_sha256"],
        },
        "target_custody": target_custody,
        "scorer_custody": scorer_custody,
        "realization_profile_custody": profile_custody,
        "semantic_control": {
            "predictor_labels_sha256": causal.causal_receipt.labels_sha256,
            "compiled_g_packet_sha256": compiled_g.compiled.receipt.packet_sha256,
            "compiled_g_labels_sha256": _array_sha256(compiled_g.compiled.decoded.labels),
            "target_labels_sha256": _array_sha256(target_labels),
            "semantic_debt_before_cells": compiled_g.receipt.debt_before_cells,
            "semantic_debt_after_cells": compiled_g.receipt.debt_after_cells,
            "exact_semantic_target_reconstructed": True,
        },
        "variant_order": list(VARIANT_ORDER),
        "variants": results,
        "scorer_runtime": scorer_runtime,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "implementation_custody": _implementation_custody(),
        "truth": {
            "component_distances_measured": True,
            "dense_frames_persisted": False,
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
    return _canonical_json(body) + b"\n"


def parse_stage_ablation_receipt(payload: bytes) -> dict[str, Any]:
    """Strict closed-schema parse and canonical byte re-emission."""

    if type(payload) is not bytes or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise TaskspaceN2StageAblationError("receipt requires exactly one terminal newline")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskspaceN2StageAblationError(f"receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload[:-1].decode("ascii"), object_pairs_hook=unique_pairs)
    except TaskspaceN2StageAblationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspaceN2StageAblationError("receipt is not strict ASCII JSON") from exc
    expected_keys = {
        "schema",
        "axis",
        "scope",
        "git_head_before_landing",
        "competitive_target",
        "baseline_materialization",
        "target_custody",
        "scorer_custody",
        "realization_profile_custody",
        "semantic_control",
        "variant_order",
        "variants",
        "scorer_runtime",
        "runtime",
        "implementation_custody",
        "truth",
    }
    if type(value) is not dict or set(value) != expected_keys or value.get("schema") != SCHEMA:
        raise TaskspaceN2StageAblationError("receipt top-level schema is not closed V1")
    if value.get("axis") != AXIS or value.get("variant_order") != list(VARIANT_ORDER):
        raise TaskspaceN2StageAblationError("receipt axis or preregistered variant order changed")
    if set(value.get("variants", {})) != set(VARIANT_ORDER):
        raise TaskspaceN2StageAblationError("receipt variant universe changed")
    truth = value.get("truth")
    if type(truth) is not dict or truth != {
        "authoritative_contest_cpu_evaluation": False,
        "authoritative_contest_cuda_evaluation": False,
        "candidate_archive_eligible": False,
        "component_distances_measured": True,
        "dense_frames_persisted": False,
        "n600_evaluation": False,
        "pointer_moved": False,
        "promotion_eligible": False,
        "research_only": True,
        "score_claim": False,
    }:
        raise TaskspaceN2StageAblationError("receipt authority labels became permissive")
    if _canonical_json(value) + b"\n" != payload:
        raise TaskspaceN2StageAblationError("receipt is not canonical on parse-back")
    return value


def write_once_or_equal(path: Path, payload: bytes) -> None:
    """Write one durable receipt without overwriting historical evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_stable_regular(path) != payload:
            raise TaskspaceN2StageAblationError(f"refusing to overwrite different receipt: {path}")
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
            raise TaskspaceN2StageAblationError(f"receipt race produced different bytes: {path}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = measure()
    parsed = parse_stage_ablation_receipt(receipt)
    write_once_or_equal(args.output, receipt)
    print(
        _canonical_json(
            {
                "axis": parsed["axis"],
                "output": os.fspath(args.output),
                "receipt_bytes": len(receipt),
                "receipt_sha256": _sha256(receipt),
                "variants": {
                    name: {
                        "d_seg": parsed["variants"][name]["d_seg"],
                        "d_pose": parsed["variants"][name]["d_pose"],
                    }
                    for name in VARIANT_ORDER
                },
                "score_claim": False,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
