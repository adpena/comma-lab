#!/usr/bin/env python3
"""Build Lane G v3 + Ω-W-V2 stacked archive (no GPU).

Loads Lane G v3's ASYM renderer.bin, re-encodes the eligible Conv2d weights
through Ω-W-V2 (water-fill + arithmetic terminal), and assembles a contest
submission archive that swaps ONLY the renderer.bin component (masks.mkv
and optimized_poses.pt are preserved bit-identically from Lane G v3).

Output archive:
    archive_lane_g_v3_omega_w_v2.zip = {
        renderer.bin (OWV2 magic, ~236KB),
        masks.mkv    (Lane G v3, 421,483B raw),
        optimized_poses.pt (Lane G v3, 15,620B raw),
    }

Predicted byte change vs Lane G v3 archive (694,074B):
    raw renderer.bin:    296,776 -> ~235,660 = -61,116B (-20.59%) [empirical]
    archive total est:   694,074 -> ~633,000B
    rate term reduction: 25 × 61,116 / 37,545,489 = +0.041 score [derivation]
    → predicted contest-CUDA score ~1.01 (Lane G v3 baseline 1.05) [derivation]

Caveats:
    [derivation]-grade prediction. The actual contest-CUDA auth eval may
    differ (PoseNet/SegNet distortion contributions could shift slightly
    because the FP16-encoded protected layers and the arithmetic-decoded
    conv weights have small round-trip error). Hard kill criterion:
    if auth eval > 1.05, the OWV2 round-trip is breaking the renderer's
    score-relevant behaviour — investigate before continuing.

Strict-scorer-rule compliant: no PoseNet / SegNet load at any stage of
this script; pure renderer state_dict round-trip + zipfile assembly.

Cross-references
----------------
* Council chain audit (Part G1): .omx/research/council_chain_integrity_audit_20260430.md
* OWV2 archive module: src/tac/owv2_renderer_archive.py
* Inflate dispatch: submissions/robust_current/inflate_renderer.py (OWV2 case)
* Round-trip tests: src/tac/tests/test_owv2_renderer_archive_inflate.py
* Companion remote dispatch: scripts/remote_lane_omega_w_v2_stack.sh
* Lane G v3 anchor: experiments/results/lane_g_v3_landed/

Usage
-----
.venv/bin/python experiments/build_lane_g_v3_omega_w_v2_stack.py \\
    --output experiments/results/lane_g_v3_omega_w_v2/archive.zip
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
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
                f"Lane G v3 having been landed (commit 232b24ec or earlier "
                f"if the OWV2 commit hasn't yet rebased Lane G v3)."
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write the stacked archive ZIP.")
    p.add_argument("--bit-budget-ratio", type=float, default=0.7,
                   help="OWV2 per-tensor bit-budget as fraction of V1 raw "
                        "byte estimate (default 0.7; matches the Council F "
                        "empirical anchor).")
    p.add_argument("--provenance-json", type=Path, default=None,
                   help="Optional path to write a provenance JSON alongside "
                        "the archive (sha256s, byte deltas, predicted score).")
    args = p.parse_args()

    print("=== Lane G v3 + Ω-W-V2 stacked archive build ===")
    t_start = time.monotonic()

    print("Stage 1: verify Lane G v3 anchors...")
    anchor_meta = _verify_anchor()
    for name, m in anchor_meta.items():
        print(f"  {name}: {m['size_bytes']:,}B sha256={m['sha256'][:12]}…")

    print("Stage 2: load Lane G v3 ASYM renderer.bin...")
    from tac.renderer_export import load_renderer_checkpoint
    model = load_renderer_checkpoint(str(LANE_G_V3_RENDERER))
    n_params = sum(int(p_.numel()) for p_ in model.parameters())
    print(f"  loaded AsymmetricPairGenerator: {n_params:,} params")

    print("Stage 3: re-encode renderer to OWV2 archive format...")
    from tac.owv2_renderer_archive import encode_owv2_archive, is_owv2_archive
    owv2_blob = encode_owv2_archive(
        model=model, bit_budget_ratio=args.bit_budget_ratio,
    )
    if not is_owv2_archive(owv2_blob):
        raise RuntimeError(
            "encode_owv2_archive returned a blob that does not start with "
            "OWV2 magic — this is a tac bug; do not proceed."
        )
    asym_size = LANE_G_V3_RENDERER.stat().st_size
    owv2_size = len(owv2_blob)
    savings_pct = 100.0 * (1.0 - owv2_size / asym_size)
    print(f"  ASYM={asym_size:,}B → OWV2={owv2_size:,}B "
          f"({savings_pct:+.2f}% savings)")

    print("Stage 4: assemble stacked archive...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    masks_bytes = LANE_G_V3_MASKS.read_bytes()
    poses_bytes = LANE_G_V3_POSES.read_bytes()

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
        z.writestr(_zinfo("renderer.bin"), owv2_blob)
        z.writestr(_zinfo("masks.mkv"), masks_bytes)
        z.writestr(_zinfo("optimized_poses.pt"), poses_bytes)

    archive_size = args.output.stat().st_size
    archive_sha256 = _sha256(args.output)
    lane_g_v3_archive_size = LANE_G_V3_ARCHIVE.stat().st_size

    print(f"  wrote {args.output} ({archive_size:,}B sha256={archive_sha256[:16]}…)")
    print(f"  Lane G v3 archive baseline: {lane_g_v3_archive_size:,}B")
    print(f"  archive delta: {archive_size - lane_g_v3_archive_size:+,}B")

    rate_delta = 25.0 * (archive_size - lane_g_v3_archive_size) / 37_545_489
    predicted_score = 1.05 + rate_delta
    print(f"  predicted score: 1.05 + {rate_delta:+.4f} = "
          f"{predicted_score:.4f} [derivation, contest-CUDA pending]")

    elapsed = time.monotonic() - t_start
    print(f"  build took {elapsed:.1f}s")

    if args.provenance_json:
        prov = {
            "format": "lane_g_v3_omega_w_v2_stack_provenance_v1",
            "build_started_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - elapsed),
            ),
            "build_completed_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bit_budget_ratio": args.bit_budget_ratio,
            "anchors": anchor_meta,
            "stacked_archive": {
                "path": str(args.output),
                "size_bytes": archive_size,
                "sha256": archive_sha256,
            },
            "renderer_bin_owv2": {
                "size_bytes": owv2_size,
                "savings_pct_vs_asym": savings_pct,
            },
            "size_delta_vs_lane_g_v3": archive_size - lane_g_v3_archive_size,
            "lane_g_v3_baseline_score": 1.05,
            "predicted_score_derivation": predicted_score,
            "predicted_score_tag": "[derivation]",
            "score_validation_required": "contest-CUDA via experiments/contest_auth_eval.py",
            "strict_scorer_rule": "compliant — no scorer load at any stage",
        }
        args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_json.write_text(json.dumps(prov, indent=2))
        print(f"  wrote provenance: {args.provenance_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
