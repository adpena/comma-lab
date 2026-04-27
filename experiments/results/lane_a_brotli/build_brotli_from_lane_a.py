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

2026-04-27 codex R5-r6 #5 fix: deterministic zip output.
The previous build called `z.write(path, arcname=...)` for each component,
which embeds the source-file mtime AND permission bits AND OS attribute
into the zip entry. Two consecutive runs produced different archive bytes
(different SHA-256) because the renderer.bin.br was freshly created with
the current wall-clock mtime. To be reproducible we now:
  1. Build each ZipInfo by hand with date_time = `(2026, 4, 27, 0, 0, 0)`.
  2. Force `external_attr = 0o644 << 16` (no exec bit, sane perms).
  3. Force `create_system = 3` (Unix) so attribute decoding is consistent.
  4. Sort entries by arcname before writing.
  5. Optionally compare the resulting archive SHA-256 against an
     `--expected-archive-sha` CLI arg; raise SystemExit on mismatch.
With this in place, repeat invocations produce byte-identical archives.
"""
from __future__ import annotations

import argparse
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

# DETERMINISTIC_ZIP_OK — codex R5-r6 #5: this builder uses the
# deterministic-zip helper below rather than zipfile.ZipFile.write(path).
_DET_ZIP_DATE_TIME = (2026, 4, 27, 0, 0, 0)
_DET_ZIP_EXTERNAL_ATTR = (0o644 & 0xFFFF) << 16  # rw-r--r-- as Unix mode
_DET_ZIP_CREATE_SYSTEM = 3                       # 3 == Unix


def _deterministic_zip_write(
    z: zipfile.ZipFile,
    arcname: str,
    src: Path,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 9,
) -> None:
    """Write ``src`` into ``z`` with FIXED metadata so reruns are byte-identical.

    Use this instead of ``z.write(src, arcname)`` whenever the produced
    archive will be SHA-anchored or rate-compared. The default
    ``zipfile.write`` embeds the source mtime + os-dependent perm bits,
    which drift between hosts/reruns.
    """
    info = zipfile.ZipInfo(filename=arcname, date_time=_DET_ZIP_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = _DET_ZIP_EXTERNAL_ATTR
    info.create_system = _DET_ZIP_CREATE_SYSTEM
    with open(src, "rb") as f:
        data = f.read()
    z.writestr(info, data, compress_type=compress_type, compresslevel=compresslevel)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build lane_a_brotli archive deterministically.",
    )
    parser.add_argument(
        "--expected-archive-sha", default=None,
        help=(
            "If given, fail with SystemExit when the produced archive's "
            "SHA-256 does not match this value. Pin once via codex R5-r6 #5."
        ),
    )
    cli_args = parser.parse_args()

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

    # Stage 2: assemble archive (codex R5-r6 #5: deterministic).
    # Sort entries by arcname so the on-disk order is stable across reruns.
    entries = sorted(
        [
            ("renderer.bin.br", renderer_br),
            ("masks.mkv", masks_src),
            ("optimized_poses.pt", poses_src),
        ],
        key=lambda kv: kv[0],
    )
    print(f"\n[build] writing {out_archive} (deterministic mode)")
    with zipfile.ZipFile(out_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arcname, src in entries:
            _deterministic_zip_write(z, arcname, src)

    archive_size = out_archive.stat().st_size
    archive_sha = _sha256(out_archive)
    rate_unscaled = archive_size / 37545489
    rate_contribution = 25 * rate_unscaled
    print(f"\n=== Archive built (deterministic) ===")
    print(f"  Path:      {out_archive}")
    print(f"  Bytes:     {archive_size:,}")
    print(f"  SHA-256:   {archive_sha}")
    print(f"  Rate (unscaled):    {rate_unscaled:.6f}")
    print(f"  Rate (score):       {rate_contribution:.4f}")
    print(f"  vs Lane A (694045): {archive_size - 694045:+d} bytes "
          f"({(archive_size - 694045) / 694045 * 100:+.2f}%)")

    # codex R5-r6 #5: pinned-SHA gate so silent rebuild drift fails fast.
    if cli_args.expected_archive_sha is not None:
        if archive_sha != cli_args.expected_archive_sha:
            raise SystemExit(
                f"FATAL: produced archive SHA-256 {archive_sha} does not "
                f"match --expected-archive-sha {cli_args.expected_archive_sha}. "
                f"Either an input drifted (renderer.bin / masks.mkv / poses) "
                f"or the deterministic-zip helper produced different bytes."
            )
        print(f"  Expected SHA matched: {archive_sha} [PASS]")

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
        "archive_sha256": archive_sha,
        "deterministic_zip": True,
        "deterministic_zip_date_time": list(_DET_ZIP_DATE_TIME),
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
