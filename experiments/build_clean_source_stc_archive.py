#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lane STC CLEAN-SOURCE archive builder.

The original Lane STC build (``experiments/build_lane_stc_archive.py``) takes
an existing AV1-encoded ``masks.mkv`` and re-encodes the AV1-DECODED class IDs
through the STC boundary codec. That regresses ~50x because half the input
pixels are AV1 quantization speckle — STC losslessly preserves noise that AV1
itself encodes lossy and efficiently. See
``project_lane_stc_av1_regression_finding_20260429`` (memory) and
``docs/paper/lane_stc_boundary_coding_design_20260429.md``.

This builder skips AV1 entirely:

    upstream/videos/0.mkv  -->  SegNet (compress-time only)  -->  argmax class IDs
                           -->  encode_mask_video_stc        -->  masks.stcb
    Lane A renderer.bin            (unchanged, pulled from anchor)
    Lane A optimized_poses.pt      (unchanged, pulled from anchor)
                           -->  archive_lane_stc_clean.zip   (deterministic zip)

Strict-scorer-rule compliance: SegNet runs ONLY at compress time inside this
encoder. The shipped archive contains NO scorer weights; the inflate path
uses ``decode_mask_video_stc`` which is pure integer/byte parsing.

Usage:
    python experiments/build_clean_source_stc_archive.py \\
        --anchor-archive experiments/results/lane_a_landed/archive_lane_a.zip \\
        --gt-video       upstream/videos/0.mkv \\
        --output         experiments/results/lane_stc_clean/archive_lane_stc_clean.zip \\
        --device         cuda \\
        --boundary-fraction 0.05 \\
        --manifest       experiments/results/lane_stc_clean/manifest.json
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT / "src",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.scorer import extract_gt_masks, load_scorers  # noqa: E402
from tac.repo_io import json_text  # noqa: E402
from tac.stc_boundary_codec import (  # noqa: E402
    decode_mask_video_stc,
    encode_mask_video_stc,
)
from tac.submission_archive import safe_extract_zip  # noqa: E402


def _det_write(zout: zipfile.ZipFile, src: Path, arcname: str) -> None:
    """Deterministic zip write (fixed mtime, ZIP_DEFLATED, level 9).

    Required by Codex R5-r6 #5 / check_archive_builders_use_deterministic_zip.
    """
    info = zipfile.ZipInfo(arcname)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zout.writestr(info, src.read_bytes(), compresslevel=9)


def _decode_gt_video(gt_video: Path) -> list[torch.Tensor]:
    """Decode 0.mkv into a list of (H, W, 3) uint8 RGB tensors."""
    import av as _av

    container = _av.open(str(gt_video))
    frames: list[torch.Tensor] = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(arr))
    container.close()
    return frames


def build_clean_source_stc_archive(
    anchor_archive: Path,
    gt_video: Path,
    output: Path,
    *,
    device: str = "cuda",
    boundary_fraction: float = 0.05,
    batch_size: int = 8,
    upstream_dir: Path | None = None,
) -> dict:
    """Build a Lane STC archive from CLEAN SegNet argmax (no AV1 roundtrip).

    Returns a dict with byte statistics suitable for run_log / manifest:
        anchor_archive_bytes, anchor_masks_bytes, stcb_bytes,
        output_archive_bytes, rate_delta_bytes_masks_layer,
        rate_delta_bytes_archive, predicted_score_delta.
    """
    anchor_archive = Path(anchor_archive)
    gt_video = Path(gt_video)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    upstream_dir = (
        Path(upstream_dir) if upstream_dir is not None else _REPO_ROOT / "upstream"
    )

    if not anchor_archive.exists():
        raise FileNotFoundError(f"anchor archive missing: {anchor_archive}")
    if not gt_video.exists():
        raise FileNotFoundError(f"GT video missing: {gt_video}")
    anchor_archive_bytes = anchor_archive.stat().st_size

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        safe_extract_zip(anchor_archive, td_path)

        renderer = td_path / "renderer.bin"
        if not renderer.exists():
            raise FileNotFoundError(
                f"anchor archive missing renderer.bin: {anchor_archive}"
            )

        anchor_masks_mkv = td_path / "masks.mkv"
        anchor_masks_bytes = (
            anchor_masks_mkv.stat().st_size if anchor_masks_mkv.exists() else -1
        )

        # Stage 1: decode GT video to RGB frames (compress-time only).
        t0 = time.monotonic()
        print(f"[clean-stc] decoding GT video {gt_video}", flush=True)
        gt_frames = _decode_gt_video(gt_video)
        print(
            f"[clean-stc] decoded {len(gt_frames)} frames in "
            f"{time.monotonic() - t0:.1f}s",
            flush=True,
        )

        # Stage 2: SegNet at compress time -> argmax class IDs.
        # NOTE: scorer-at-compress-time is allowed by the strict-scorer-rule.
        # No scorer weights ship in the archive; the inflate path is pure-byte.
        t0 = time.monotonic()
        torch_device = torch.device(device)
        print(
            f"[clean-stc] loading frozen SegNet (device={torch_device}) "
            f"from {upstream_dir}",
            flush=True,
        )
        _, segnet = load_scorers(
            posenet_path=upstream_dir / "models" / "posenet.safetensors",
            segnet_path=upstream_dir / "models" / "segnet.safetensors",
            device=torch_device,
            upstream_dir=upstream_dir,
        )
        segnet.eval()
        print(
            f"[clean-stc] running SegNet argmax (batch_size={batch_size})",
            flush=True,
        )
        masks = extract_gt_masks(gt_frames, segnet, torch_device, batch_size=batch_size)
        # Free GPU memory before encoding (encode is pure CPU).
        del segnet
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
        masks = masks.cpu().contiguous()
        print(
            f"[clean-stc] argmax masks shape={tuple(masks.shape)} in "
            f"{time.monotonic() - t0:.1f}s",
            flush=True,
        )

        # Stage 3: STC encode the CLEAN argmax (no AV1 roundtrip).
        stcb_path = td_path / "masks.stcb"
        t0 = time.monotonic()
        stcb_bytes = encode_mask_video_stc(
            masks,
            stcb_path,
            boundary_fraction=boundary_fraction,
            verify_roundtrip=True,
        )
        print(
            f"[clean-stc] STC encoded {stcb_bytes:,} B "
            f"(boundary_fraction={boundary_fraction}) in "
            f"{time.monotonic() - t0:.1f}s",
            flush=True,
        )

        # Stage 4: independent decode + EXACT compare (load-bearing test).
        decoded = decode_mask_video_stc(stcb_path)
        if not torch.equal(decoded, masks):
            n_diff = int((decoded != masks).sum().item())
            raise RuntimeError(
                f"[clean-stc] STC roundtrip class-ID mismatch: "
                f"{n_diff} pixels differ -- refusing to ship a corrupt archive"
            )

        # Stage 5: deterministic zip with renderer + masks.stcb + poses.
        with zipfile.ZipFile(
            output, "w", zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zout:
            _det_write(zout, renderer, "renderer.bin")
            _det_write(zout, stcb_path, "masks.stcb")
            for poses_name in ("optimized_poses.pt", "poses.pt"):
                p = td_path / poses_name
                if p.exists():
                    _det_write(zout, p, poses_name)
                    break

    output_bytes = output.stat().st_size
    rate_delta_bytes_masks_layer = (
        stcb_bytes - anchor_masks_bytes if anchor_masks_bytes >= 0 else 0
    )
    rate_delta_bytes_archive = output_bytes - anchor_archive_bytes
    # Contest rate weight: 25 * delta_bytes / 37,545,489.
    predicted_score_delta = 25 * rate_delta_bytes_archive / 37_545_489

    return {
        "anchor_archive": str(anchor_archive),
        "anchor_archive_bytes": int(anchor_archive_bytes),
        "anchor_masks_bytes": int(anchor_masks_bytes),
        "stcb_bytes": int(stcb_bytes),
        "output_archive": str(output),
        "output_archive_bytes": int(output_bytes),
        "rate_delta_bytes_masks_layer": int(rate_delta_bytes_masks_layer),
        "rate_delta_bytes_archive": int(rate_delta_bytes_archive),
        "boundary_fraction": float(boundary_fraction),
        "predicted_score_delta": float(predicted_score_delta),
        "device": str(device),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor-archive", type=Path, required=True)
    p.add_argument("--gt-video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            "cuda / mps / cpu. cuda is the contest-equivalent SegNet path; "
            "MPS produces 2x SegNet drift (CLAUDE.md feedback_mps_cuda_drift_critical). "
            "cpu is acceptable for deterministic-bytes builds (slow but exact)."
        ),
    )
    p.add_argument("--boundary-fraction", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=None,
        help="Override upstream/ root (defaults to <repo>/upstream).",
    )
    p.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args(argv)

    info = build_clean_source_stc_archive(
        anchor_archive=args.anchor_archive,
        gt_video=args.gt_video,
        output=args.output,
        device=args.device,
        boundary_fraction=args.boundary_fraction,
        batch_size=args.batch_size,
        upstream_dir=args.upstream_dir,
    )

    print("")
    print("[clean-stc] ===== BYTE MEASUREMENT (load-bearing) =====")
    print(f"[clean-stc]   anchor archive_lane_a.zip = {info['anchor_archive_bytes']:>9,} B")
    print(f"[clean-stc]   anchor masks.mkv (AV1)    = {info['anchor_masks_bytes']:>9,} B")
    print(f"[clean-stc]   clean masks.stcb (STC)    = {info['stcb_bytes']:>9,} B")
    print(
        f"[clean-stc]   delta on mask layer        = "
        f"{info['rate_delta_bytes_masks_layer']:+,} B"
    )
    print(f"[clean-stc]   output archive (zip)      = {info['output_archive_bytes']:>9,} B")
    print(
        f"[clean-stc]   delta on full archive      = "
        f"{info['rate_delta_bytes_archive']:+,} B "
        f"(predicted score delta {info['predicted_score_delta']:+.4f})"
    )
    if info["stcb_bytes"] < info["anchor_masks_bytes"]:
        print("[clean-stc] LOAD-BEARING CLAIM HOLDS: clean STC is smaller than AV1.")
    else:
        print("[clean-stc] LOAD-BEARING CLAIM FAILS: clean STC is NOT smaller than AV1.")

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json_text(info), encoding="utf-8")
        print(f"[clean-stc] manifest -> {args.manifest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
