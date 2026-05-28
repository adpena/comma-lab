#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare public PR95 inflate output against direct MLX full-frame render."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
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


def _diff_stats(public_raw: bytes, mlx_raw_chunks: list[bytes]) -> dict[str, Any]:
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
    return {
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--inflate-sh", type=Path, default=DEFAULT_INFLATE_SH)
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
    archive_zip = args.archive_zip.resolve()
    inflate_sh = args.inflate_sh.resolve()
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
    mlx_chunks = _render_mlx_camera_frame_bytes(
        archive_zip=archive_zip,
        member_name=args.member_name,
        mlx_device=args.mlx_device,
        conv2d_accumulation_mode=conv_mode,
        conv2d_accumulation_overrides=conv_overrides,
        chunk_pairs=args.chunk_pairs,
    )
    stats = _diff_stats(public_raw, mlx_chunks)
    public_ok = (
        proc.returncode == 0
        and public_raw_path.is_file()
        and len(public_raw) == expected_bytes
    )
    byte_exact = bool(public_ok and stats.get("byte_exact") is True)
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
        for path in sorted(work_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if work_dir.exists():
            work_dir.rmdir()
    return 0 if byte_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
