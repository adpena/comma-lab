#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare public PR95 inflate output against direct MLX full-frame render."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    FALSE_AUTHORITY,
    HNeRVDecoderMLX,
    load_pytorch_state_dict_into_mlx,
    parse_pr95_public_archive_zip,
    pr95_mlx_conv2d_accumulation_overrides_from_items,
    pr95_mlx_conv2d_accumulation_overrides_from_preset,
    require_mlx,
    validate_pr95_mlx_conv2d_accumulation_mode,
)
from tac.local_acceleration.pr95_hnerv_mlx_contract import (  # noqa: E402
    PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER,
    PR95_FULL_FRAME_INFLATE_PARITY_FAILED_BLOCKER,
)
from tac.repo_io import write_json_artifact  # noqa: E402

DEFAULT_INFLATE_SH = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/inflate.sh"
)
CAMERA_H = 874
CAMERA_W = 1164
RGB_CHANNELS = 3
SCHEMA = "pr95_hnerv_public_full_frame_inflate_parity_proof.v1"
CHANNEL_NAMES = ("r", "g", "b")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes_iter(chunks: list[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def _expected_raw_bytes(meta: dict[str, Any]) -> int:
    return int(meta["n_pairs"]) * 2 * CAMERA_H * CAMERA_W * RGB_CHANNELS


def _runtime_files(inflate_sh: Path) -> list[dict[str, Any]]:
    submission_dir = inflate_sh.resolve().parent
    paths = [
        inflate_sh,
        submission_dir / "inflate.py",
        submission_dir / "src/model.py",
        submission_dir / "src/codec.py",
    ]
    return [
        {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _mlx_device(name: str) -> Any:
    import mlx.core as mx

    if name == "cpu":
        return mx.cpu
    if name == "gpu":
        return mx.gpu
    raise ValueError(f"unsupported MLX device {name!r}")


def _load_public_pr95_decoder_cls(model_path: Path) -> Any:
    if not model_path.is_file():
        raise FileNotFoundError(f"public PR95 source model.py not found: {model_path}")
    spec = importlib.util.spec_from_file_location(
        f"public_pr95_hnerv_model_{abs(hash(model_path.resolve()))}",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import public PR95 model.py: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        return module.HNeRVDecoder
    except AttributeError as exc:
        raise RuntimeError(f"{model_path} does not define HNeRVDecoder") from exc


def _run_public_inflate(
    *,
    archive_zip: Path,
    inflate_sh: Path,
    member_name: str,
    file_base: str,
    work_dir: Path,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    data_dir = work_dir / "data"
    raw_dir = work_dir / "raw"
    file_list = work_dir / "file_list.txt"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip) as zf:
        member_bytes = zf.read(member_name)
    (data_dir / f"{file_base}.bin").write_bytes(member_bytes)
    file_list.write_text(f"{file_base}.mkv\n", encoding="utf-8")
    env = os.environ.copy()
    venv_bin = REPO_ROOT / ".venv/bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    proc = subprocess.run(
        [
            "bash",
            inflate_sh.as_posix(),
            data_dir.as_posix(),
            raw_dir.as_posix(),
            file_list.as_posix(),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return proc, raw_dir / f"{file_base}.raw"


def _render_mlx_camera_frame_bytes(
    *,
    archive_zip: Path,
    member_name: str,
    mlx_device: str,
    conv2d_accumulation_mode: str,
    conv2d_accumulation_overrides: dict[str, str],
    chunk_pairs: int,
) -> list[bytes]:
    require_mlx()
    import mlx.core as mx
    import torch
    import torch.nn.functional as F

    packet = parse_pr95_public_archive_zip(archive_zip, member_name=member_name)
    meta = packet.meta
    eval_h, eval_w = [int(dim) for dim in meta["eval_size"]]

    chunks: list[bytes] = []
    previous_device = mx.default_device()
    mx.set_default_device(_mlx_device(mlx_device))
    try:
        model = HNeRVDecoderMLX(
            latent_dim=int(meta["latent_dim"]),
            base_channels=int(meta["base_channels"]),
            eval_size=(eval_h, eval_w),
            conv2d_accumulation_mode=conv2d_accumulation_mode,
            conv2d_accumulation_overrides=conv2d_accumulation_overrides,
        )
        load_pytorch_state_dict_into_mlx(model, packet.state_dict)
        for start in range(0, int(packet.latents.shape[0]), chunk_pairs):
            stop = min(start + chunk_pairs, int(packet.latents.shape[0]))
            latent_chunk = mx.array(packet.latents[start:stop].astype(np.float32))
            decoded = model(latent_chunk)
            mx.eval(decoded)
            decoded_np = np.asarray(decoded, dtype=np.float32)
            flat = decoded_np.reshape((stop - start) * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                torch.from_numpy(flat),
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            frames = (
                up.clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            chunks.append(frames.tobytes())
    finally:
        mx.set_default_device(previous_device)
    return chunks


def _render_torch_camera_frame_bytes(
    *,
    archive_zip: Path,
    member_name: str,
    model_path: Path,
    chunk_pairs: int,
) -> list[bytes]:
    import torch
    import torch.nn.functional as F

    packet = parse_pr95_public_archive_zip(archive_zip, member_name=member_name)
    meta = packet.meta
    eval_h, eval_w = [int(dim) for dim in meta["eval_size"]]
    decoder_cls = _load_public_pr95_decoder_cls(model_path)
    torch_state_dict = {
        name: torch.from_numpy(value.astype(np.float32, copy=True))
        for name, value in packet.state_dict.items()
    }
    model = decoder_cls(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=(eval_h, eval_w),
    ).eval()
    model.load_state_dict(torch_state_dict)

    chunks: list[bytes] = []
    with torch.no_grad():
        for start in range(0, int(packet.latents.shape[0]), chunk_pairs):
            stop = min(start + chunk_pairs, int(packet.latents.shape[0]))
            latent_chunk = torch.from_numpy(
                packet.latents[start:stop].astype(np.float32, copy=True)
            )
            decoded = model(latent_chunk).detach()
            flat = decoded.reshape((stop - start) * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            frames = (
                up.clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            chunks.append(frames.tobytes())
    return chunks


def _frame_byte_stride() -> int:
    return CAMERA_H * CAMERA_W * RGB_CHANNELS


def _unravel_raw_index(index: int) -> dict[str, int | str]:
    frame_index, rem = divmod(int(index), _frame_byte_stride())
    y, rem = divmod(rem, CAMERA_W * RGB_CHANNELS)
    x, channel = divmod(rem, RGB_CHANNELS)
    channel_i = int(channel)
    return {
        "flat_index": int(index),
        "frame_index": int(frame_index),
        "y": int(y),
        "x": int(x),
        "channel": channel_i,
        "channel_name": CHANNEL_NAMES[channel_i],
        "boundary_distance_pixels": int(
            min(y, x, CAMERA_H - 1 - y, CAMERA_W - 1 - x)
        ),
    }


def _mismatch_row(
    *,
    index: int,
    public: np.ndarray,
    mlx: np.ndarray,
) -> dict[str, Any]:
    row = _unravel_raw_index(index)
    public_value = int(public[index])
    mlx_value = int(mlx[index])
    signed_delta = int(mlx_value - public_value)
    row.update(
        {
            "public_uint8": public_value,
            "mlx_uint8": mlx_value,
            "signed_delta_mlx_minus_public": signed_delta,
            "abs_delta_uint8": abs(signed_delta),
        }
    )
    return row


def _diff_atlas(
    *,
    public: np.ndarray,
    mlx: np.ndarray,
    changed: np.ndarray,
    max_samples: int,
) -> dict[str, Any]:
    changed_indices = np.flatnonzero(changed)
    changed_count = int(changed_indices.size)
    n_frames = int(public.size // _frame_byte_stride())
    changed_nhwc = changed.reshape(n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS)
    diff_i16 = mlx.astype(np.int16) - public.astype(np.int16)
    changed_delta = diff_i16[changed]
    per_frame = changed_nhwc.sum(axis=(1, 2, 3)).astype(np.int64)
    per_channel = changed_nhwc.sum(axis=(0, 1, 2)).astype(np.int64)
    samples = [
        _mismatch_row(index=int(index), public=public, mlx=mlx)
        for index in changed_indices[:max_samples]
    ]
    if changed_count:
        ys = (changed_indices % _frame_byte_stride()) // (CAMERA_W * RGB_CHANNELS)
        xs = (changed_indices % (CAMERA_W * RGB_CHANNELS)) // RGB_CHANNELS
        distances = np.minimum.reduce(
            [ys, xs, CAMERA_H - 1 - ys, CAMERA_W - 1 - xs]
        )
        boundary_buckets = {
            "edge_0": int(np.count_nonzero(distances == 0)),
            "near_edge_1_2": int(
                np.count_nonzero((distances >= 1) & (distances <= 2))
            ),
            "near_edge_3_8": int(
                np.count_nonzero((distances >= 3) & (distances <= 8))
            ),
            "interior_ge9": int(np.count_nonzero(distances >= 9)),
        }
    else:
        boundary_buckets = {
            "edge_0": 0,
            "near_edge_1_2": 0,
            "near_edge_3_8": 0,
            "interior_ge9": 0,
        }
    signed_delta_histogram = {
        str(int(value)): int(count)
        for value, count in zip(
            *np.unique(changed_delta.astype(np.int16), return_counts=True),
            strict=True,
        )
    }
    return {
        "raw_layout": "frames_nhwc_uint8",
        "frame_shape_nhwc": [n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS],
        "per_frame_changed_byte_count": [int(value) for value in per_frame],
        "per_channel_changed_byte_count": {
            name: int(per_channel[index])
            for index, name in enumerate(CHANNEL_NAMES)
        },
        "boundary_distance_changed_byte_count": boundary_buckets,
        "signed_delta_histogram_mlx_minus_public": signed_delta_histogram,
        "first_mismatch": samples[0] if samples else None,
        "mismatch_samples": samples,
        "mismatch_sample_count": len(samples),
        "mismatch_sample_cap": int(max_samples),
    }


def _diff_stats(
    public_raw: bytes,
    mlx_raw_chunks: list[bytes],
    *,
    max_samples: int,
) -> dict[str, Any]:
    public = np.frombuffer(public_raw, dtype=np.uint8)
    mlx = np.frombuffer(b"".join(mlx_raw_chunks), dtype=np.uint8)
    if public.shape != mlx.shape:
        return {
            "same_shape": False,
            "public_count": int(public.size),
            "mlx_count": int(mlx.size),
            "byte_exact": False,
        }
    diff = np.abs(public.astype(np.int16) - mlx.astype(np.int16))
    changed = diff != 0
    stats: dict[str, Any] = {
        "same_shape": True,
        "public_count": int(public.size),
        "mlx_count": int(mlx.size),
        "byte_exact": bool(not np.any(changed)),
        "changed_byte_count": int(np.count_nonzero(changed)),
        "changed_byte_fraction": float(np.count_nonzero(changed) / max(1, diff.size)),
        "max_abs_uint8": int(diff.max()) if diff.size else 0,
        "mean_abs_uint8": float(diff.mean()) if diff.size else 0.0,
        "p99_abs_uint8": float(np.quantile(diff, 0.99)) if diff.size else 0.0,
        "p999_abs_uint8": float(np.quantile(diff, 0.999)) if diff.size else 0.0,
    }
    if public.size and public.size % _frame_byte_stride() == 0:
        stats.update(
            _diff_atlas(
                public=public,
                mlx=mlx,
                changed=changed,
                max_samples=max_samples,
            )
        )
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--inflate-sh", type=Path, default=DEFAULT_INFLATE_SH)
    parser.add_argument(
        "--torch-reference-model-py",
        type=Path,
        help=(
            "Public PR95 model.py for direct PyTorch full-frame reconstruction. "
            "Defaults to <inflate.sh parent>/src/model.py."
        ),
    )
    parser.add_argument(
        "--skip-torch-direct-reference",
        action="store_true",
        help="Skip direct PyTorch-vs-public full-frame reference localization.",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--member-name", default="0.bin")
    parser.add_argument("--file-base", default="0")
    parser.add_argument("--mlx-device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--conv2d-accumulation-mode",
        default="optimized",
        choices=("optimized", "fixed_fp32", "kahan_fp32", "fixed_fp64"),
    )
    parser.add_argument(
        "--conv2d-override-preset",
        default="none",
        choices=(
            "none",
            "rgb_heads_kahan_fp32",
            "refine_rgb_heads_kahan_fp32",
            "blocks_kahan_fp32",
            "blocks01_kahan_fp32",
            "blocks02_kahan_fp32",
            "blocks_refine_kahan_fp32",
        ),
    )
    parser.add_argument("--conv2d-override", action="append")
    parser.add_argument("--chunk-pairs", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--max-mismatch-samples",
        type=int,
        default=32,
        help="Maximum localized mismatch rows embedded in the drift atlas.",
    )
    parser.add_argument("--allow-large-output", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-existing-sha256")
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep staged public raw output. Default proof records hashes only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.chunk_pairs < 1:
        raise SystemExit("--chunk-pairs must be >= 1")
    if args.max_mismatch_samples < 0:
        raise SystemExit("--max-mismatch-samples must be >= 0")
    archive_zip = args.archive_zip.resolve()
    inflate_sh = args.inflate_sh.resolve()
    torch_reference_model_py = (
        args.torch_reference_model_py.resolve()
        if args.torch_reference_model_py is not None
        else inflate_sh.parent / "src/model.py"
    )
    output_json = args.output_json.resolve()
    if not inflate_sh.is_file():
        raise SystemExit(f"inflate.sh not found: {inflate_sh}")
    conv_mode = validate_pr95_mlx_conv2d_accumulation_mode(
        args.conv2d_accumulation_mode
    )
    conv_overrides = pr95_mlx_conv2d_accumulation_overrides_from_items(
        args.conv2d_override,
        base=pr95_mlx_conv2d_accumulation_overrides_from_preset(
            args.conv2d_override_preset
        ),
    )
    if args.mlx_device == "gpu" and (
        conv_mode == "fixed_fp64"
        or any(mode == "fixed_fp64" for mode in conv_overrides.values())
    ):
        raise SystemExit("fixed_fp64 Conv2d accumulation is unsupported on MLX GPU")

    packet = parse_pr95_public_archive_zip(archive_zip, member_name=args.member_name)
    expected_bytes = _expected_raw_bytes(packet.meta)
    if expected_bytes > args.max_output_bytes and not args.allow_large_output:
        raise SystemExit(
            f"refusing predicted raw output {expected_bytes} bytes above "
            f"--max-output-bytes {args.max_output_bytes}; pass --allow-large-output"
        )

    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else output_json.parent / "full_frame_inflate_parity_work"
    )
    started = datetime.now(UTC)
    proc, public_raw_path = _run_public_inflate(
        archive_zip=archive_zip,
        inflate_sh=inflate_sh,
        member_name=args.member_name,
        file_base=args.file_base,
        work_dir=work_dir,
        timeout_seconds=args.timeout_seconds,
    )
    public_raw = public_raw_path.read_bytes() if public_raw_path.is_file() else b""
    public_ok = (
        proc.returncode == 0
        and public_raw_path.is_file()
        and len(public_raw) == expected_bytes
    )
    mlx_chunks = _render_mlx_camera_frame_bytes(
        archive_zip=archive_zip,
        member_name=args.member_name,
        mlx_device=args.mlx_device,
        conv2d_accumulation_mode=conv_mode,
        conv2d_accumulation_overrides=conv_overrides,
        chunk_pairs=args.chunk_pairs,
    )
    stats = _diff_stats(
        public_raw,
        mlx_chunks,
        max_samples=args.max_mismatch_samples,
    )
    torch_reference: dict[str, Any] = {
        "enabled": False,
        "model_py": torch_reference_model_py.as_posix(),
    }
    if not args.skip_torch_direct_reference:
        torch_chunks = _render_torch_camera_frame_bytes(
            archive_zip=archive_zip,
            member_name=args.member_name,
            model_path=torch_reference_model_py,
            chunk_pairs=args.chunk_pairs,
        )
        torch_stats = _diff_stats(
            public_raw,
            torch_chunks,
            max_samples=args.max_mismatch_samples,
        )
        torch_reference = {
            "enabled": True,
            "model_py": torch_reference_model_py.as_posix(),
            "raw_bytes": sum(len(chunk) for chunk in torch_chunks),
            "raw_sha256": _sha256_bytes_iter(torch_chunks),
            "diff": torch_stats,
            "byte_exact_with_public_inflate": bool(
                public_ok and torch_stats.get("byte_exact") is True
            ),
        }
    byte_exact = bool(public_ok and stats.get("byte_exact") is True)
    if not public_ok:
        drift_localization_verdict = "public_inflate_failed"
    elif byte_exact:
        drift_localization_verdict = "mlx_full_frame_byte_exact"
    elif torch_reference.get("enabled") and torch_reference.get(
        "byte_exact_with_public_inflate"
    ):
        drift_localization_verdict = "mlx_decoder_or_mlx_bridge_arithmetic_drift"
    elif torch_reference.get("enabled"):
        drift_localization_verdict = "public_runtime_parser_or_torch_reference_mismatch"
    else:
        drift_localization_verdict = "torch_reference_not_run"
    blockers = [
        "full_frame_inflate_parity_is_not_score_authority",
        "requires_exact_cpu_cuda_auth_eval_before_score_claim",
    ]
    if not byte_exact:
        blockers.insert(
            0,
            PR95_FULL_FRAME_INFLATE_PARITY_FAILED_BLOCKER
            if public_ok
            else PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER,
        )
    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "started_utc": started.isoformat(),
        "lane_id": "lane_pr95_hnerv_mlx_reproduction",
        "source_pr": 95,
        "submission": "hnerv_muon",
        "archive_path": archive_zip.as_posix(),
        "archive_bytes": archive_zip.stat().st_size,
        "archive_sha256": _sha256_file(archive_zip),
        "archive_packet": packet.custody_manifest(),
        "inflate_sh": inflate_sh.as_posix(),
        "runtime_files": _runtime_files(inflate_sh),
        "work_dir": work_dir.as_posix() if args.keep_work_dir else None,
        "public_raw_path": public_raw_path.as_posix() if args.keep_work_dir else None,
        "expected_raw_bytes": expected_bytes,
        "public_raw_bytes": len(public_raw),
        "public_raw_sha256": hashlib.sha256(public_raw).hexdigest()
        if public_raw
        else None,
        "mlx_raw_bytes": sum(len(chunk) for chunk in mlx_chunks),
        "mlx_raw_sha256": _sha256_bytes_iter(mlx_chunks),
        "torch_direct_reference": torch_reference,
        "drift_localization_verdict": drift_localization_verdict,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "mlx_device": args.mlx_device,
        "conv2d_accumulation_mode": conv_mode,
        "conv2d_accumulation_overrides": conv_overrides,
        "diff": stats,
        "runtime_consumption_proof_passed": public_ok,
        "runtime_consumption_proven": public_ok,
        "full_frame_inflate_output_parity_claim": byte_exact,
        "full_frame_inflate_parity_satisfied": byte_exact,
        "full_frame_inflate_parity_failed": bool(public_ok and not byte_exact),
        "receiver_contract_satisfied": byte_exact,
        "receiver_proof_present": byte_exact,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": blockers,
        },
        **FALSE_AUTHORITY,
    }
    write_json_artifact(
        output_json,
        payload,
        allow_overwrite=args.allow_overwrite,
        expected_existing_sha256=args.expected_existing_sha256,
    )
    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0 if byte_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
