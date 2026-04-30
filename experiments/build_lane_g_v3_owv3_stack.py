#!/usr/bin/env python3
"""Build Lane G v3 + Ω-W-V3 stacked archive.

This builder swaps only `renderer.bin`; Lane G v3 `masks.mkv` and
`optimized_poses.pt` are preserved bit-identically. The OWV3 renderer payload
is produced from a compress-time sensitivity map. Decode/inflate does not need
the sensitivity map and must not load scorers.

Output archive:

    archive_lane_g_v3_owv3.zip = {
        renderer.bin        (OWV3 magic),
        masks.mkv           (Lane G v3, bit-identical),
        optimized_poses.pt  (Lane G v3, bit-identical),
    }

This script produces implementation/provenance evidence only. Promotion
requires exact CUDA contest eval through `experiments/contest_auth_eval.py`.
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
    anchors = {
        "renderer": LANE_G_V3_RENDERER,
        "masks": LANE_G_V3_MASKS,
        "poses": LANE_G_V3_POSES,
        "archive": LANE_G_V3_ARCHIVE,
    }
    meta = {}
    for name, path in anchors.items():
        if not path.exists():
            raise FileNotFoundError(f"missing Lane G v3 anchor: {path}")
        meta[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    head = LANE_G_V3_RENDERER.read_bytes()[:4]
    if head != b"ASYM":
        raise RuntimeError(
            f"Lane G v3 renderer.bin has unexpected magic {head!r}; expected ASYM"
        )
    return meta


def _zinfo(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o644 & 0xFFFF) << 16
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sensitivity-map", type=Path, required=True,
                        help="OWV3 sensitivity_map.pt from tac.sensitivity_map.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write archive zip.")
    parser.add_argument("--provenance-json", type=Path, default=None,
                        help="Optional provenance JSON path.")
    parser.add_argument("--bit-budget-ratio", type=float, default=0.7)
    parser.add_argument("--protect-threshold", type=float, default=1e-3)
    parser.add_argument("--aggressive-threshold", type=float, default=1e-5)
    parser.add_argument("--allow-non-authoritative", action="store_true",
                        help="Allow non-CUDA sensitivity metadata. Output is smoke-only.")
    parser.add_argument("--no-decode-verify", action="store_true",
                        help="Skip local OWV3 decode sanity check.")
    args = parser.parse_args(argv)

    print("=== Lane G v3 + Ω-W-V3 stacked archive build ===")
    t_start = time.monotonic()

    print("Stage 1: verify Lane G v3 anchors...")
    anchor_meta = _verify_anchor()
    for name, meta in anchor_meta.items():
        print(f"  {name}: {meta['size_bytes']:,}B sha256={meta['sha256'][:12]}...")

    if not args.sensitivity_map.exists():
        raise FileNotFoundError(args.sensitivity_map)

    print("Stage 2: load renderer and sensitivity map...")
    from tac.renderer_export import load_renderer_checkpoint
    from tac.sensitivity_map import (
        load_sensitivity_map,
        require_authoritative_device,
        validate_sensitivity_map_for_model,
    )

    model = load_renderer_checkpoint(str(LANE_G_V3_RENDERER))
    model.eval()
    sensitivities, sensitivity_metadata = load_sensitivity_map(args.sensitivity_map)
    sensitivity_device = sensitivity_metadata.get("source_device") or sensitivity_metadata.get("device")
    if not args.allow_non_authoritative:
        require_authoritative_device(sensitivity_device)
    sens_stats = validate_sensitivity_map_for_model(
        sensitivities,
        model,
        require_all_conv=True,
    )
    print(
        f"  sensitivity: {sens_stats.n_layers} layers, {sens_stats.n_channels} channels, "
        f"range=[{sens_stats.min_value:.6g}, {sens_stats.max_value:.6g}]"
    )

    print("Stage 3: encode OWV3 renderer...")
    from tac.owv3_sensitivity_weighted import (
        decode_owv3_archive,
        encode_owv3_archive,
        is_owv3_archive,
    )

    owv3_blob = encode_owv3_archive(
        model=model,
        sensitivities=sensitivities,
        bit_budget_ratio=args.bit_budget_ratio,
        protect_threshold=args.protect_threshold,
        aggressive_threshold=args.aggressive_threshold,
        require_all_conv_sensitivity=True,
    )
    if not is_owv3_archive(owv3_blob):
        raise RuntimeError("OWV3 encoder returned non-OWV3 blob")
    if not args.no_decode_verify:
        decoded = decode_owv3_archive(data=owv3_blob, device="cpu")
        if set(decoded.state_dict()) != set(model.state_dict()):
            raise RuntimeError("OWV3 decode state_dict keys diverged from source")
    asym_size = LANE_G_V3_RENDERER.stat().st_size
    owv3_size = len(owv3_blob)
    renderer_delta = owv3_size - asym_size
    print(
        f"  ASYM={asym_size:,}B -> OWV3={owv3_size:,}B "
        f"(delta={renderer_delta:+,}B)"
    )

    print("Stage 4: assemble deterministic archive...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(_zinfo("renderer.bin"), owv3_blob)
        z.writestr(_zinfo("masks.mkv"), LANE_G_V3_MASKS.read_bytes())
        z.writestr(_zinfo("optimized_poses.pt"), LANE_G_V3_POSES.read_bytes())

    archive_size = args.output.stat().st_size
    archive_sha256 = _sha256(args.output)
    lane_g_v3_archive_size = LANE_G_V3_ARCHIVE.stat().st_size
    archive_delta = archive_size - lane_g_v3_archive_size
    rate_delta = 25.0 * archive_delta / 37_545_489
    predicted_score = 1.05 + rate_delta
    elapsed = time.monotonic() - t_start

    print(f"  wrote {args.output} ({archive_size:,}B sha256={archive_sha256[:16]}...)")
    print(f"  archive delta vs Lane G v3: {archive_delta:+,}B")
    print(
        f"  predicted rate-only score: 1.05 + {rate_delta:+.5f} = "
        f"{predicted_score:.4f} [derivation, contest-CUDA pending]"
    )
    print(f"  build took {elapsed:.2f}s")

    if args.provenance_json:
        prov = {
            "format": "lane_g_v3_owv3_stack_provenance_v1",
            "lane_id": "lane_g_v3_owv3_stack",
            "build_started_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - elapsed),
            ),
            "build_completed_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "anchors": anchor_meta,
            "sensitivity_map": {
                "path": str(args.sensitivity_map),
                "sha256": _sha256(args.sensitivity_map),
                "metadata": sensitivity_metadata,
                "stats": {
                    "n_layers": sens_stats.n_layers,
                    "n_channels": sens_stats.n_channels,
                    "min_value": sens_stats.min_value,
                    "max_value": sens_stats.max_value,
                },
            },
            "owv3": {
                "renderer_bytes": owv3_size,
                "renderer_delta_vs_asym": renderer_delta,
                "bit_budget_ratio": args.bit_budget_ratio,
                "protect_threshold": args.protect_threshold,
                "aggressive_threshold": args.aggressive_threshold,
                "decode_verified": not args.no_decode_verify,
            },
            "stacked_archive": {
                "path": str(args.output),
                "size_bytes": archive_size,
                "sha256": archive_sha256,
            },
            "size_delta_vs_lane_g_v3": archive_delta,
            "lane_g_v3_baseline_score": 1.05,
            "predicted_score_derivation": predicted_score,
            "predicted_score_tag": "[derivation]",
            "score_validation_required": "contest-CUDA via experiments/contest_auth_eval.py",
            "strict_scorer_rule": "compliant - no scorer load at OWV3 decode/inflate",
            "evidence_label": "implementation-smoke until exact CUDA contest eval",
        }
        args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_json.write_text(json.dumps(prov, indent=2))
        print(f"  wrote provenance: {args.provenance_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
