#!/usr/bin/env python3
"""Build lane_a_brotli archive by stacking Lane A's exact CUDA-built masks.mkv
and optimized_poses.pt with a brotli-compressed renderer.bin.br.

Why direct composition (not build_baseline_archive.py):
  - Lane A's masks.mkv was generated on a Vast.ai 4090 (CUDA SegNet) and saved
    in experiments/results/lane_a_landed/extracted/masks.mkv.
  - We do NOT have CUDA locally; running build_baseline_archive.py with
    --device cpu would re-encode SegNet on CPU, producing different mask bytes
    than the CUDA archive that scored 1.15 — invalidating the comparison.
  - By reusing the EXACT bytes of Lane A's masks.mkv + poses + renderer, the
    only difference between this archive and Lane A's is renderer.bin.br vs
    renderer.bin. Pure rate-side test.

Inputs (all byte-identical to Lane A):
  renderer.bin       : submissions/baseline_dilated_h64_0_90/renderer.bin
                        (matches lane_a/extracted/renderer.bin sha
                         08f12d72...)
  masks.mkv          : experiments/results/lane_a_landed/extracted/masks.mkv
                        (sha c07bd465..., CUDA-built on 4090)
  optimized_poses.pt : experiments/results/lane_a_landed/optimized_poses.pt
                        (sha e0cd8ccb..., matches lane_a archive in-zip)

Output:
  experiments/results/lane_a_brotli/archive.zip
  experiments/results/lane_a_brotli/archive.provenance.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from tac.submission_archive import compress_file_brotli  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    renderer_src = REPO / "submissions" / "baseline_dilated_h64_0_90" / "renderer.bin"
    masks_src = REPO / "experiments" / "results" / "lane_a_landed" / "extracted" / "masks.mkv"
    poses_src = REPO / "experiments" / "results" / "lane_a_landed" / "optimized_poses.pt"

    out_dir = REPO / "experiments" / "results" / "lane_a_brotli"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_archive = out_dir / "archive.zip"

    for label, p in [("renderer", renderer_src), ("masks", masks_src), ("poses", poses_src)]:
        if not p.exists():
            raise SystemExit(f"FATAL: {label} source missing: {p}")

    # Verify SHAs match Lane A's
    expected_shas = {
        "renderer.bin": "08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529",
        "masks.mkv":    "c07bd465b0091282d8d0577e04118c68b3433d50327d993192e3a52e3d343515",
        "optimized_poses.pt": "e0cd8ccb29bb3cd8613285e1633a6466f8a01f5df101d80243ec52d50d2fb85b",
    }
    actual_shas = {
        "renderer.bin": _sha256(renderer_src),
        "masks.mkv":    _sha256(masks_src),
        "optimized_poses.pt": _sha256(poses_src),
    }
    print("[build] Verifying input SHAs match Lane A 1.15 archive components:")
    for name, want in expected_shas.items():
        got = actual_shas[name]
        ok = "OK" if got == want else "MISMATCH"
        print(f"  {name:25s} {got}  [{ok}]")
        if got != want:
            raise SystemExit(
                f"FATAL: {name} sha {got} does not match Lane A archive sha {want}. "
                f"Cannot guarantee byte-perfect rate-side comparison."
            )

    # Stage 1: brotli-compress renderer.bin
    quality = 11
    renderer_br = out_dir / "renderer.bin.br"
    print(f"\n[build] brotli-compressing renderer.bin (quality={quality})...")
    t0 = time.monotonic()
    compress_file_brotli(renderer_src, renderer_br, quality=quality)
    br_size = renderer_br.stat().st_size
    raw_size = renderer_src.stat().st_size
    saved = raw_size - br_size
    rate_delta = saved / 37545489 * 25
    print(
        f"[build] renderer.bin {raw_size:,} -> .br {br_size:,} "
        f"(saved {saved:,} bytes, -{rate_delta:.4f} score) "
        f"in {time.monotonic()-t0:.1f}s"
    )

    # Stage 2: assemble archive
    print(f"\n[build] writing {out_archive}")
    with zipfile.ZipFile(out_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(renderer_br, arcname="renderer.bin.br")
        z.write(masks_src, arcname="masks.mkv")
        z.write(poses_src, arcname="optimized_poses.pt")

    archive_size = out_archive.stat().st_size
    rate_unscaled = archive_size / 37545489
    rate_contribution = 25 * rate_unscaled
    print(f"\n=== Archive built ===")
    print(f"  Path:      {out_archive}")
    print(f"  Bytes:     {archive_size:,}")
    print(f"  SHA-256:   {_sha256(out_archive)}")
    print(f"  Rate (unscaled):    {rate_unscaled:.6f}")
    print(f"  Rate (score):       {rate_contribution:.4f}")
    print(f"  vs Lane A (694045): {archive_size - 694045:+d} bytes "
          f"({(archive_size - 694045) / 694045 * 100:+.2f}%)")

    # Stage 3: provenance
    prov = {
        "schema_version": 1,
        "tool": "experiments/results/lane_a_brotli/build_brotli_from_lane_a.py",
        "lane": "lane_b_alt_brotli_on_lane_a_pose",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "device": "no-segnet (byte-identical reuse of Lane A CUDA-built masks)",
        "use_brotli": True,
        "brotli_quality": quality,
        "archive_path": str(out_archive),
        "archive_size_bytes": archive_size,
        "archive_sha256": _sha256(out_archive),
        "rate_unscaled": rate_unscaled,
        "rate_score_contribution": rate_contribution,
        "lane_a_archive_size_bytes": 694045,
        "lane_a_archive_sha256": "a9921cd3b974ff0a7c37b39e7af22d9b75802f1219fc46aecb6eb8eaa7a08e84",
        "lane_a_final_score": 1.15,
        "predicted_score": round(1.15 - rate_delta, 4),
        "components": {
            "renderer.bin.br": {
                "source": str(renderer_src),
                "size_bytes_uncompressed": raw_size,
                "size_bytes_in_archive": br_size,
                "sha256_uncompressed": actual_shas["renderer.bin"],
                "sha256_compressed": _sha256(renderer_br),
                "compression": f"brotli-q{quality}",
            },
            "masks.mkv": {
                "source": str(masks_src),
                "size_bytes": masks_src.stat().st_size,
                "sha256": actual_shas["masks.mkv"],
                "origin": "Lane A: extracted from "
                          "experiments/results/lane_a_landed/archive_lane_a.zip — "
                          "originally generated on Vast.ai RTX 4090 CUDA SegNet "
                          "at 384x512 CRF=50",
            },
            "optimized_poses.pt": {
                "source": str(poses_src),
                "size_bytes": poses_src.stat().st_size,
                "sha256": actual_shas["optimized_poses.pt"],
                "origin": "Lane A pose-TTO output (committed at "
                          "experiments/results/lane_a_landed/optimized_poses.pt)",
            },
        },
    }
    prov_path = out_archive.with_suffix(".provenance.json")
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"  Provenance: {prov_path}")

    # Cleanup intermediate .br
    renderer_br.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
