#!/usr/bin/env python3
"""Build Lane G v3 + Lane PFP16 stacked archive (no GPU).

Loads Lane G v3's `optimized_poses.pt` (15,620 B fp32 pickle), re-encodes it
through Lane PFP16 (raw fp16 buffer, 7,200 B), and assembles a contest
submission archive that swaps ONLY the pose component (renderer.bin and
masks.mkv are preserved bit-identically from Lane G v3).

The pose file is renamed from `optimized_poses.pt` to `optimized_poses.bin`
inside the archive so the inflate-side loader's content-detection (Branch B
of `tac.submission_archive.load_optimized_poses`) routes it correctly. This
keeps the inflate path dispatch unchanged — Lane PFP16 is a build-time
concept, not a wire format with a magic byte.

Output archive:
    archive_lane_g_v3_pfp16.zip = {
        renderer.bin       (Lane G v3 ASYM, 289,776 B raw, bit-identical),
        masks.mkv          (Lane G v3, 421,483 B raw, bit-identical),
        optimized_poses.bin (Lane PFP16, 7,200 B raw fp16),
    }

Predicted byte change vs Lane G v3 archive (694,074 B):
    raw poses:           15,620 -> 7,200 = -8,420 B (-53.9%) [empirical]
    archive total est:   694,074 -> ~685,654 B (delta ≈ -8,420 B in ZIP_DEFLATED)
    rate term reduction: 25 × 8,420 / 37,545,489 = +0.00561 score [derivation]
    → predicted contest-CUDA score ~1.044 (Lane G v3 baseline 1.05) [derivation]

Caveats:
    [derivation]-grade prediction. Actual contest-CUDA auth eval may differ
    by ±0.005 from PoseNet's intrinsic fp16 precision (well below the noise
    floor). HARD KILL: any auth eval > 1.05 means the fp16 cast somehow
    changed PoseNet behaviour beyond its intrinsic fp16 path — investigate
    immediately.

Strict-scorer-rule compliant: no PoseNet / SegNet load at any stage of
this script; pure pose tensor cast + zipfile assembly.

Cross-references
----------------
* Lane GP v4 KILL: .omx/research/council_lane_gp_v4_design_20260430.md
* PFP16 codec: src/tac/pfp16_codec.py
* PFP16 tests: src/tac/tests/test_pfp16_codec.py
* Inflate dispatch (auto-handles raw fp16 .bin):
  submissions/robust_current/inflate_renderer.py
  (Branch B of tac.submission_archive.load_optimized_poses)
* Companion remote dispatch: scripts/remote_lane_pfp16_stack.sh
* Lane G v3 anchor: experiments/results/lane_g_v3_landed/

Usage
-----
.venv/bin/python experiments/build_lane_g_v3_pfp16_stack.py \\
    --output experiments/results/lane_g_v3_pfp16/archive.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure tac is importable when this script is run as `python experiments/...`.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

LANE_G_V3_DIR = REPO_ROOT / "experiments" / "results" / "lane_g_v3_landed"
LANE_G_V3_RENDERER = LANE_G_V3_DIR / "iter_0" / "renderer.bin"
LANE_G_V3_MASKS = LANE_G_V3_DIR / "iter_0" / "masks.mkv"
LANE_G_V3_POSES = LANE_G_V3_DIR / "iter_0" / "optimized_poses.pt"
LANE_G_V3_ARCHIVE = LANE_G_V3_DIR / "archive_lane_g_v3.zip"


def _sha256(data: bytes | Path) -> str:
    h = hashlib.sha256()
    if isinstance(data, Path):
        with data.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
    else:
        h.update(data)
    return h.hexdigest()


def _verify_anchor() -> dict:
    """Verify all Lane G v3 anchors exist and return their metadata."""
    anchors = {
        "renderer": LANE_G_V3_RENDERER,
        "masks": LANE_G_V3_MASKS,
        "poses": LANE_G_V3_POSES,
        "archive": LANE_G_V3_ARCHIVE,
    }
    meta = {}
    for name, p in anchors.items():
        if not p.exists():
            raise FileNotFoundError(
                f"missing Lane G v3 anchor: {p}; this build is gated on "
                f"Lane G v3 having been landed."
            )
        meta[name] = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": _sha256(p),
        }
    head = LANE_G_V3_RENDERER.read_bytes()[:4]
    if head != b"ASYM":
        raise RuntimeError(
            f"Lane G v3 renderer.bin has unexpected magic {head!r}; "
            f"expected ASYM (Lane G v3 ships AsymmetricPairGenerator)."
        )
    return meta


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Path to write the stacked archive ZIP.",
    )
    p.add_argument(
        "--provenance-json", type=Path, default=None,
        help="Optional path to write a provenance JSON alongside the "
             "archive (sha256s, byte deltas, predicted score).",
    )
    args = p.parse_args()

    print("=== Lane G v3 + Lane PFP16 stacked archive build ===")
    t_start = time.monotonic()

    print("Stage 1: verify Lane G v3 anchors...")
    anchor_meta = _verify_anchor()
    for name, m in anchor_meta.items():
        print(f"  {name}: {m['size_bytes']:,}B sha256={m['sha256'][:12]}...")

    print("Stage 2: load Lane G v3 fp32 poses + cast to PFP16 raw fp16...")
    from tac.submission_archive import load_optimized_poses
    from tac.pfp16_codec import (
        PFP16_FORMAT_SENTINEL,
        encode_pfp16,
        decode_pfp16,
    )
    poses = load_optimized_poses(str(LANE_G_V3_POSES), pose_dim=6)
    if poses.shape != (600, 6):
        raise RuntimeError(
            f"Lane G v3 poses unexpected shape {tuple(poses.shape)}; "
            f"expected (600, 6)."
        )
    pfp16_blob = encode_pfp16(poses)
    pfp16_size = len(pfp16_blob)
    fp32_size = LANE_G_V3_POSES.stat().st_size
    pose_savings_bytes = fp32_size - pfp16_size
    pose_savings_pct = 100.0 * pose_savings_bytes / fp32_size
    print(f"  fp32 pickle={fp32_size:,}B -> PFP16 raw={pfp16_size:,}B "
          f"({pose_savings_bytes:+,}B, {pose_savings_pct:+.2f}%)")

    # Verify roundtrip is below tol BEFORE building archive (defense in depth
    # — encode_pfp16 already checks but we re-verify on the canonical
    # decode_pfp16 path that the inflate-side will use).
    decoded = decode_pfp16(pfp16_blob, pose_dim=6)
    max_err = float((poses - decoded).abs().max().item())
    mean_err = float((poses - decoded).abs().mean().item())
    print(f"  roundtrip: max_abs_err={max_err:.6f}, mean_abs_err={mean_err:.6f}")
    if max_err > 0.06:
        raise RuntimeError(
            f"PFP16 roundtrip max_err {max_err:.6f} exceeds 0.06 tol; "
            f"refusing to ship."
        )

    print("Stage 3: assemble stacked archive...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    renderer_bytes = LANE_G_V3_RENDERER.read_bytes()
    masks_bytes = LANE_G_V3_MASKS.read_bytes()
    # Pose file is RENAMED from optimized_poses.pt → optimized_poses.bin so
    # the inflate-side `load_optimized_poses` routes through Branch B (raw
    # fp16 detection by absence of pickle magic). The inflate path already
    # checks BOTH `optimized_poses.pt` and `optimized_poses.bin`.

    # Use deterministic ZipInfo + writestr (per Codex R5-r6 #5
    # "archive_builders_use_deterministic_zip" preflight check). Fixed
    # timestamp so the byte hash is reproducible across machines.
    fixed_dt = (1980, 1, 1, 0, 0, 0)

    def _zinfo(name: str) -> zipfile.ZipInfo:
        zi = zipfile.ZipInfo(name)
        zi.date_time = fixed_dt
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = (0o644 & 0xFFFF) << 16
        return zi

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(_zinfo("renderer.bin"), renderer_bytes)
        z.writestr(_zinfo("masks.mkv"), masks_bytes)
        z.writestr(_zinfo("optimized_poses.bin"), pfp16_blob)

    archive_size = args.output.stat().st_size
    archive_sha256 = _sha256(args.output)
    lane_g_v3_archive_size = LANE_G_V3_ARCHIVE.stat().st_size
    archive_delta = archive_size - lane_g_v3_archive_size

    print(f"  wrote {args.output} ({archive_size:,}B sha256={archive_sha256[:16]}...)")
    print(f"  Lane G v3 archive baseline: {lane_g_v3_archive_size:,}B")
    print(f"  archive delta: {archive_delta:+,}B")

    # Sanity: must be SMALLER than baseline.
    if archive_size >= lane_g_v3_archive_size:
        raise RuntimeError(
            f"Lane PFP16 stacked archive ({archive_size}B) is not smaller "
            f"than Lane G v3 baseline ({lane_g_v3_archive_size}B); "
            f"compression added overhead. Investigate before shipping."
        )

    rate_delta = 25.0 * archive_delta / 37_545_489
    predicted_score = 1.05 + rate_delta
    print(f"  predicted score: 1.05 + {rate_delta:+.5f} = "
          f"{predicted_score:.4f} [derivation, contest-CUDA pending]")

    elapsed = time.monotonic() - t_start
    print(f"  build took {elapsed:.2f}s")

    if args.provenance_json:
        prov = {
            "format": "lane_g_v3_pfp16_stack_provenance_v1",
            "lane_id": "lane_g_v3_pfp16_stack",
            "build_started_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - elapsed),
            ),
            "build_completed_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
            ),
            "anchors": anchor_meta,
            "stacked_archive": {
                "path": str(args.output),
                "size_bytes": archive_size,
                "sha256": archive_sha256,
            },
            "pose_pfp16": {
                "format_sentinel": PFP16_FORMAT_SENTINEL,
                "input_bytes_fp32_pickle": fp32_size,
                "output_bytes_fp16_raw": pfp16_size,
                "savings_bytes": pose_savings_bytes,
                "savings_pct": pose_savings_pct,
                "max_roundtrip_error": max_err,
                "mean_roundtrip_error": mean_err,
                "n_pairs": int(poses.shape[0]),
                "pose_dim": int(poses.shape[1]),
            },
            "size_delta_vs_lane_g_v3": archive_delta,
            "lane_g_v3_baseline_score": 1.05,
            "predicted_score_derivation": predicted_score,
            "predicted_score_tag": "[derivation]",
            "predicted_score_band": [1.04, 1.05],
            "score_validation_required": "contest-CUDA via experiments/contest_auth_eval.py",
            "strict_scorer_rule": "compliant — no scorer load at any stage",
            "council_provenance": {
                "design_doc": ".omx/research/council_lane_gp_v4_design_20260430.md",
                "rationale": "Hotz successor option from Lane GP v4 KILL — fp16 cast captures the bulk of pose-stream byte budget at zero distortion penalty",
            },
        }
        args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_json.write_text(json.dumps(prov, indent=2))
        print(f"  wrote provenance: {args.provenance_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
