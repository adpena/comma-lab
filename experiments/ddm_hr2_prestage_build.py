# SPDX-License-Identifier: MIT
"""Materialize HR2's scorer-free real-frame pre-stage receipts.

Every RGB, gradient, typed-program, binding, and memory-refusal output is
retained under the requested SSD root.  This script imports no scorer or model
module and makes no SegNet/PoseNet/score claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tac.differentiable_eval_roundtrip import (
    CAMERA_HW,
    SCORER_HW,
    CameraLiftKernel,
    EvalRoundTripOrdering,
    apply_camera_uint8_lift_during_training,
    apply_eval_roundtrip_during_training,
)
from tac.witness_dsl.hr1_prestage import (
    Hr1Arm,
    Hr1PrestageError,
    atomic_write_json,
    build_hr1_binding_manifest,
    compile_memory_configuration,
    make_four_arm_race_programs,
    make_shape_only_memory_configuration,
    payload_manifest_for_tree,
    stream_sha256,
)

DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_hr2_prestage_build_20260811/retained")
ALLOWED_OUTPUT_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
MIN_FREE_BYTES = 256 * 1024 * 1024


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_root(path: Path) -> Path:
    resolved = path.resolve()
    if not any(_under(resolved, root.resolve()) for root in ALLOWED_OUTPUT_ROOTS):
        raise Hr1PrestageError("HR2 retained output must live on the configured SSD tier")
    if resolved.exists() and any(resolved.iterdir()):
        raise Hr1PrestageError(f"refusing to overwrite non-empty retained root: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(resolved)
    free_bytes = free.f_bavail * free.f_frsize
    if free_bytes < MIN_FREE_BYTES:
        raise Hr1PrestageError(
            f"storage preflight REFUSE: {free_bytes} free bytes < {MIN_FREE_BYTES} required"
        )
    return resolved


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_save_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise Hr1PrestageError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest, size = stream_sha256(path)
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _explicit_camera_receiver(work: torch.Tensor, kernel: CameraLiftKernel) -> torch.Tensor:
    return F.interpolate(
        work,
        size=CAMERA_HW,
        mode=kernel.value,
        align_corners=False,
    ).clamp(0.0, 255.0).round()


def _explicit_camera_ste_roundtrip(work: torch.Tensor, kernel: CameraLiftKernel) -> torch.Tensor:
    lifted = F.interpolate(
        work,
        size=CAMERA_HW,
        mode=kernel.value,
        align_corners=False,
    )
    clamped = lifted.clamp(0.0, 255.0)
    camera = clamped + (clamped.round() - clamped).detach()
    return F.interpolate(camera, size=SCORER_HW, mode="bilinear", align_corners=False)


def _tensor_max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.maximum(actual.abs().max(), expected.abs().max()).clamp_min(1e-12)
    return float((actual - expected).abs().max() / scale)


def _decode_real_frames(video: Path, count: int):
    import av
    upstream_root = Path(__file__).resolve().parents[1] / "upstream"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    from frame_utils import yuv420_to_rgb

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index >= count:
                break
            yield index, yuv420_to_rgb(frame)


def _max_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(observed if sys.platform == "darwin" else observed * 1024)


def _materialize_pixel_controls(
    *,
    video: Path,
    frame_count: int,
    output_root: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []
    total_channel_argmax_changes = 0
    decoded = 0
    gradient_weight_saved = False
    for frame_index, camera_hwc in _decode_real_frames(video, frame_count):
        decoded += 1
        prefix = f"frame_{frame_index:04d}"
        camera_np = camera_hwc.cpu().numpy()
        records.append(_atomic_save_npy(output_root / f"{prefix}_source_camera_u8.npy", camera_np))
        camera = camera_hwc.permute(2, 0, 1).unsqueeze(0).float()
        work = F.interpolate(camera, size=SCORER_HW, mode="bilinear", align_corners=False)
        work_np = work[0].permute(1, 2, 0).cpu().numpy()
        records.append(_atomic_save_npy(output_root / f"{prefix}_work_rgb_f32.npy", work_np))

        frame_receipt: dict[str, Any] = {"frame_index": frame_index, "kernels": {}}
        for kernel in CameraLiftKernel:
            helper = apply_camera_uint8_lift_during_training(work, lift_kernel=kernel)
            reference = _explicit_camera_receiver(work, kernel)
            helper_u8 = helper[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            reference_u8 = reference[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            helper_record = _atomic_save_npy(
                output_root / f"{prefix}_{kernel.value}_training_camera_u8.npy",
                helper_u8,
            )
            reference_record = _atomic_save_npy(
                output_root / f"{prefix}_{kernel.value}_public_receiver_camera_u8.npy",
                reference_u8,
            )
            records.extend((helper_record, reference_record))
            if not np.array_equal(helper_u8, reference_u8):
                raise AssertionError(f"camera-byte positive control failed for {kernel.value}")

            actual_input = work.detach().clone().requires_grad_(True)
            reference_input = work.detach().clone().requires_grad_(True)
            weight = torch.linspace(
                0.25,
                1.25,
                SCORER_HW[0] * SCORER_HW[1],
                dtype=work.dtype,
            ).reshape(1, 1, *SCORER_HW)
            if not gradient_weight_saved:
                records.append(
                    _atomic_save_npy(
                        output_root / "gradient_spatial_weight_f32.npy",
                        weight[0, 0].cpu().numpy(),
                    )
                )
                gradient_weight_saved = True
            actual_out = apply_eval_roundtrip_during_training(
                actual_input,
                ordering=EvalRoundTripOrdering.CAMERA_UINT8,
                lift_kernel=kernel,
            )
            reference_out = _explicit_camera_ste_roundtrip(reference_input, kernel)
            (actual_out * weight).sum().backward()
            (reference_out * weight).sum().backward()
            assert actual_input.grad is not None and reference_input.grad is not None
            actual_grad = actual_input.grad[0].permute(1, 2, 0).cpu().numpy()
            reference_grad = reference_input.grad[0].permute(1, 2, 0).cpu().numpy()
            records.append(
                _atomic_save_npy(
                    output_root / f"{prefix}_{kernel.value}_training_input_grad_f32.npy",
                    actual_grad,
                )
            )
            records.append(
                _atomic_save_npy(
                    output_root / f"{prefix}_{kernel.value}_cpu_reference_input_grad_f32.npy",
                    reference_grad,
                )
            )
            max_rel = _tensor_max_relative_error(actual_input.grad, reference_input.grad)
            if max_rel > 1e-7:
                raise AssertionError(
                    f"per-tensor gradient max-relative error {max_rel} exceeds 1e-7 for {kernel.value}"
                )
            frame_receipt["kernels"][kernel.value] = {
                "camera_bytes_equal": True,
                "camera_values_compared": int(reference_u8.size),
                "gradient_tensor_max_relative_error": max_rel,
                "gradient_values_compared": int(reference_grad.size),
            }
            del helper, reference, actual_input, reference_input, actual_out, reference_out
            gc.collect()

        legacy = apply_eval_roundtrip_during_training(work)
        corrected = apply_eval_roundtrip_during_training(
            work,
            ordering=EvalRoundTripOrdering.CAMERA_UINT8,
            lift_kernel=CameraLiftKernel.BICUBIC,
        )
        legacy_np = legacy[0].permute(1, 2, 0).cpu().numpy()
        corrected_np = corrected[0].permute(1, 2, 0).cpu().numpy()
        legacy_argmax = legacy.argmax(1)[0].to(torch.uint8).cpu().numpy()
        corrected_argmax = corrected.argmax(1)[0].to(torch.uint8).cpu().numpy()
        records.extend(
            (
                _atomic_save_npy(output_root / f"{prefix}_legacy_scorer_rgb_f32.npy", legacy_np),
                _atomic_save_npy(output_root / f"{prefix}_camera_uint8_scorer_rgb_f32.npy", corrected_np),
                _atomic_save_npy(output_root / f"{prefix}_legacy_rgb_channel_argmax_u8.npy", legacy_argmax),
                _atomic_save_npy(output_root / f"{prefix}_camera_uint8_rgb_channel_argmax_u8.npy", corrected_argmax),
            )
        )
        channel_changes = int(np.count_nonzero(legacy_argmax != corrected_argmax))
        value_changes = int(np.count_nonzero(legacy_np != corrected_np))
        total_channel_argmax_changes += channel_changes
        frame_receipt.update(
            {
                "legacy_vs_camera_value_changes": value_changes,
                "legacy_vs_camera_rgb_channel_argmax_changes": channel_changes,
                "argmax_scope": "RGB channel only; scorer-free; no SegNet conclusion",
            }
        )
        per_frame.append(frame_receipt)
        del camera, work, legacy, corrected
        gc.collect()
    if decoded != frame_count:
        raise Hr1PrestageError(f"requested {frame_count} real frames but decoded {decoded}")
    if total_channel_argmax_changes == 0:
        raise AssertionError("prior-law RGB-channel argmax proxy was identical on every real frame")
    return {
        "axis": "[scorer-free pixel control]",
        "frame_count": decoded,
        "camera_hw": list(CAMERA_HW),
        "scorer_hw": list(SCORER_HW),
        "per_frame": per_frame,
        "total_rgb_channel_argmax_changes": total_channel_argmax_changes,
        "retained_array_records": records,
        "segnet_loaded": False,
        "posenet_loaded": False,
        "score_claim": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if type(args.frames) is not int or not 1 <= args.frames <= 8:
        raise Hr1PrestageError("--frames must be an exact integer in [1,8]")
    repo_root = Path(__file__).resolve().parents[1]
    video = args.video if args.video.is_absolute() else repo_root / args.video
    if not video.is_file():
        raise Hr1PrestageError(f"canonical real video not found: {video}")
    output_root = _validate_output_root(args.output_root)

    pixel_control = _materialize_pixel_controls(
        video=video,
        frame_count=args.frames,
        output_root=output_root,
    )
    atomic_write_json(output_root / "20_PIXEL_CONTROL.json", pixel_control)

    bindings = build_hr1_binding_manifest(repo_root)
    atomic_write_json(output_root / "30_CONTENT_BINDINGS.json", bindings)

    programs = [program.compile().to_dict() for program in make_four_arm_race_programs()]
    atomic_write_json(
        output_root / "40_TYPED_PROGRAMS.json",
        {
            "schema_version": "hr1_four_arm_programs.v1",
            "execution_allowed": False,
            "programs": programs,
        },
    )

    memory_decisions = []
    for arm in Hr1Arm:
        config = make_shape_only_memory_configuration(arm)
        decision = compile_memory_configuration(config)
        memory_decisions.append(
            {"configuration": config.to_dict(), "decision": decision.to_dict()}
        )
        if decision.disposition.value != "REFUSE":
            raise AssertionError(f"pre-stage memory compiler unexpectedly passed {arm.value}")
    atomic_write_json(
        output_root / "50_MEMORY_REFUSALS.json",
        {
            "schema_version": "hr1_shape_only_memory_compiler.v1",
            "projection_claim": False,
            "decisions": memory_decisions,
        },
    )

    source_bindings = {}
    for role, source in (
        ("runner", Path(__file__)),
        ("roundtrip_helper", repo_root / "src/tac/differentiable_eval_roundtrip.py"),
        ("prestage_module", repo_root / "src/tac/witness_dsl/hr1_prestage.py"),
    ):
        digest, size = stream_sha256(source)
        source_bindings[role] = {"path": str(source.resolve()), "bytes": size, "sha256": digest}
    measured_max_rss = _max_rss_bytes()
    result = {
        "schema_version": "ddm_hr2_prestage_build_result.v1",
        "axis": "[scorer-free pixel/apparatus control]",
        "command": [sys.executable, *sys.argv],
        "source_bindings": source_bindings,
        "video_path": str(video.resolve()),
        "video_sha256": stream_sha256(video)[0],
        "output_root": str(output_root),
        "max_rss_bytes": measured_max_rss,
        "rss_cap_bytes": 1024 * 1024 * 1024,
        "rss_cap_pass": measured_max_rss <= 1024 * 1024 * 1024,
        "pixel_control_pass": True,
        "all_programs_execution_allowed": False,
        "all_memory_configurations_refused": True,
        "scorer_forward_count": 0,
        "model_weight_load_count": 0,
        "modal_dispatch_count": 0,
        "score_claim": False,
    }
    if not result["rss_cap_pass"]:
        raise Hr1PrestageError(
            f"measured max RSS {result['max_rss_bytes']} exceeded {result['rss_cap_bytes']}"
        )
    atomic_write_json(output_root / "90_RESULT.json", result)
    tree_manifest = payload_manifest_for_tree(
        output_root,
        exclude_names={"99_TREE_MANIFEST.json"},
    )
    tree_record = atomic_write_json(output_root / "99_TREE_MANIFEST.json", tree_manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_root": str(output_root),
                "tree_manifest_sha256": tree_record.sha256,
                "max_rss_bytes": result["max_rss_bytes"],
                "rgb_channel_argmax_changes": pixel_control["total_rgb_channel_argmax_changes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
