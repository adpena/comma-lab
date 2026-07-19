# SPDX-License-Identifier: MIT
"""Phase-1 v10 RATE-CRUSH pass 3 — donor lossless image/video coders, exact bytes.

Measures real encoder output bytes for the frozen C1 scorer planes through
JPEG-XL lossless, WebP lossless, and ffmpeg FFV1 / x264rgb / x265 lossless.
Every stream is decoded back and byte-compared to the source plane; a variant
that fails exact roundtrip is recorded as FAILED and excluded.

Axis: [macOS-CPU local rate measurement] NON-PROMOTABLE; score_claim=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

H, W, C = 384, 512, 3
PLANE_BYTES = H * W * C


def _load_chunk(chunks_dir: Path, index: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    manifest = json.loads((chunks_dir / f"chunk-{index:04d}.manifest.json").read_text())
    y0_raw = (chunks_dir / f"chunk-{index:04d}.y0.bin").read_bytes()
    y1_raw = (chunks_dir / f"chunk-{index:04d}.y1.bin").read_bytes()
    if hashlib.sha256(y0_raw).hexdigest() != manifest["y0_sha256"]:
        raise SystemExit(f"chunk {index}: y0 sha custody failure")
    if hashlib.sha256(y1_raw).hexdigest() != manifest["y1_sha256"]:
        raise SystemExit(f"chunk {index}: y1 sha custody failure")
    pair_ids = [int(p) for p in manifest["pair_ids"]]
    n = len(pair_ids)
    y0 = np.frombuffer(y0_raw, dtype=np.uint8).reshape(n, H, W, C)
    y1 = np.frombuffer(y1_raw, dtype=np.uint8).reshape(n, H, W, C)
    return y0, y1, pair_ids


def _write_ppm(path: Path, plane: np.ndarray) -> None:
    header = f"P6\n{W} {H}\n255\n".encode()
    path.write_bytes(header + plane.tobytes())


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _jxl_pair(plane: np.ndarray, work: Path, tag: str, effort: int) -> tuple[int, bool]:
    src = work / f"{tag}.ppm"
    out = work / f"{tag}.jxl"
    dec = work / f"{tag}.dec.ppm"
    _write_ppm(src, plane)
    _run(["/opt/homebrew/bin/cjxl", str(src), str(out), "-d", "0", "-e", str(effort), "--quiet"])
    _run(["/opt/homebrew/bin/djxl", str(out), str(dec), "--quiet"])
    ok = dec.read_bytes().endswith(plane.tobytes())
    return out.stat().st_size, ok


def _webp_pair(plane: np.ndarray, work: Path, tag: str) -> tuple[int, bool]:
    src = work / f"{tag}.ppm"
    out = work / f"{tag}.webp"
    dec = work / f"{tag}.dec.ppm"
    _write_ppm(src, plane)
    _run(["/opt/homebrew/bin/cwebp", "-lossless", "-z", "9", "-exact", str(src), "-o", str(out)])
    _run(["/opt/homebrew/bin/dwebp", str(out), "-ppm", "-o", str(dec)])
    ok = dec.read_bytes().endswith(plane.tobytes())
    return out.stat().st_size, ok


def _ffmpeg_sequence(frames: np.ndarray, work: Path, tag: str, args_out: list[str], out_name: str) -> tuple[int, bool]:
    raw = work / f"{tag}.rgb"
    out = work / out_name
    dec = work / f"{tag}.dec.rgb"
    raw.write_bytes(frames.tobytes())
    base = [
        "/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "20",
        "-i", str(raw),
    ]
    _run(base + args_out + [str(out)])
    _run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
        "-f", "rawvideo", "-pix_fmt", "rgb24", str(dec),
    ])
    ok = dec.read_bytes() == frames.tobytes()
    return out.stat().st_size, ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--chunk-indices", type=int, nargs="+", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--jxl-effort", type=int, default=9)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    all_y0, all_y1, all_ids = [], [], []
    for index in args.chunk_indices:
        y0, y1, pair_ids = _load_chunk(args.chunks_dir, index)
        all_y0.append(y0)
        all_y1.append(y1)
        all_ids.extend(pair_ids)
    y0 = np.concatenate(all_y0)
    y1 = np.concatenate(all_y1)
    n = len(all_ids)

    results: dict[str, dict[str, float | int | bool]] = {}
    with tempfile.TemporaryDirectory(dir="/Volumes/VertigoDataTier/pact/evidence/v10_ratecrush_20260719") as tmp:
        work = Path(tmp)
        for name, encoder in (("jxl", _jxl_pair), ("webp", _webp_pair)):
            total = 0
            all_ok = True
            for i in range(n):
                for label, plane in (("y0", y0[i]), ("y1", y1[i])):
                    if name == "jxl":
                        size, ok = _jxl_pair(plane, work, f"{name}_{i}_{label}", args.jxl_effort)
                    else:
                        size, ok = _webp_pair(plane, work, f"{name}_{i}_{label}")
                    total += size
                    all_ok = all_ok and ok
            results[name] = {"total_bytes": total, "bytes_per_pair": total / n, "roundtrip_exact": all_ok}

        if not args.skip_video:
            # Interleave planes into the true temporal order: y0_0, y1_0, y0_1, ...
            frames = np.empty((2 * n, H, W, C), dtype=np.uint8)
            frames[0::2] = y0
            frames[1::2] = y1
            video_variants = {
                "ffv1": (["-c:v", "ffv1", "-level", "3", "-context", "1", "-g", "600"], "v.mkv"),
                "x264rgb_lossless": (["-c:v", "libx264rgb", "-qp", "0", "-preset", "veryslow"], "v264.mkv"),
                "x265_lossless_rgb": (
                    ["-c:v", "libx265", "-x265-params", "lossless=1", "-preset", "slower", "-pix_fmt", "gbrp"],
                    "v265.mkv",
                ),
            }
            for name, (enc_args, out_name) in video_variants.items():
                try:
                    size, ok = _ffmpeg_sequence(frames, work, name, enc_args, out_name)
                    results[name] = {"total_bytes": size, "bytes_per_pair": size / n, "roundtrip_exact": ok}
                except subprocess.CalledProcessError:
                    results[name] = {"total_bytes": -1, "bytes_per_pair": -1.0, "roundtrip_exact": False}

    receipt = {
        "schema": "v10_ratecrush_rank_donor_coders.v1",
        "axis": "[macOS-CPU local rate measurement] NON-PROMOTABLE",
        "score_claim": False,
        "chunk_indices": args.chunk_indices,
        "pair_count": n,
        "jxl_effort": args.jxl_effort,
        "results": results,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(json.dumps(receipt["results"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
