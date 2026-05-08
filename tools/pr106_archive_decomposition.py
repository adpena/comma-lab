#!/usr/bin/env python3
"""PR106 archive decomposition — closes task #395 + validates the
mask-budget assumption flagged by the 2026-05-08 adversarial review of
PARADIGM-Ω-OPT.

The Round-1 review flagged: the design memo's mask budget estimate of
120-150 KB is unverified. PR106's actual archive is 178 KB total. If
masks really are 120-150 KB then renderer + poses + overhead must fit
in 28-58 KB — which contradicts the renderer-dominant assumption.

This tool actually unzips PR106's archive and decomposes per-component
bytes so the design's predictions can be reality-checked.

Default input: experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip
(178,258 B — the canonical PR101 HNeRV FT microcodec frontier candidate;
PR101's archive in this codebase IS the contest-relevant frontier per
session memory).

CLAUDE.md compliance: pure CPU + zipfile + math; no scorer load; no
contest score claim; output tagged ``[CPU-prep empirical archive
decomposition]``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TOOL_NAME = "tools/pr106_archive_decomposition.py"
SCHEMA_VERSION = "pr106_archive_decomposition.v1"
EVIDENCE_GRADE = "[CPU-prep empirical archive decomposition]"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def categorize_member(name: str) -> str:
    """Map a zip member name to a canonical archive component category."""
    lower = name.lower()
    if "renderer" in lower or lower.endswith(".bin") or "decoder" in lower or "weights" in lower:
        return "renderer"
    if "mask" in lower or lower.endswith(".mkv") or lower.endswith(".av1") or "video" in lower:
        return "masks"
    if "pose" in lower or "trajectory" in lower:
        return "poses"
    if "side" in lower or "scale" in lower or "metadata" in lower or "manifest" in lower:
        return "side_info"
    return "other"


def decompose(archive_path: Path) -> dict:
    """Open the archive zip, list members, sum bytes per category."""
    if not archive_path.is_file():
        raise SystemExit(f"archive not found: {archive_path}")

    archive_size = archive_path.stat().st_size
    archive_sha = sha256_file(archive_path)

    members: list[dict] = []
    by_category: dict[str, dict] = {}
    total_compressed = 0
    total_uncompressed = 0

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            cat = categorize_member(info.filename)
            row = {
                "name": info.filename,
                "category": cat,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "compress_type": info.compress_type,
                "compress_type_name": {
                    0: "STORED", 8: "DEFLATED", 12: "BZIP2", 14: "LZMA"
                }.get(info.compress_type, f"UNKNOWN({info.compress_type})"),
                "crc": f"0x{info.CRC:08x}",
                "ratio": (
                    info.compress_size / info.file_size
                    if info.file_size > 0 else 1.0
                ),
            }
            members.append(row)
            agg = by_category.setdefault(cat, {
                "n_files": 0, "compressed_bytes": 0, "uncompressed_bytes": 0,
                "members": [],
            })
            agg["n_files"] += 1
            agg["compressed_bytes"] += info.compress_size
            agg["uncompressed_bytes"] += info.file_size
            agg["members"].append(info.filename)
            total_compressed += info.compress_size
            total_uncompressed += info.file_size

    overhead_bytes = archive_size - total_compressed

    # Compute category shares
    for cat, agg in by_category.items():
        agg["compressed_share_pct"] = (
            100.0 * agg["compressed_bytes"] / max(archive_size, 1)
        )
        agg["uncompressed_share_pct"] = (
            100.0 * agg["uncompressed_bytes"] / max(total_uncompressed, 1)
        )

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "input_archive": str(archive_path),
        "input_archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "n_members": len(members),
        "total_compressed_bytes": total_compressed,
        "total_uncompressed_bytes": total_uncompressed,
        "zip_overhead_bytes": overhead_bytes,
        "zip_overhead_pct": 100.0 * overhead_bytes / max(archive_size, 1),
        "by_category": by_category,
        "members": members,
    }


def render_summary(manifest: dict) -> str:
    lines: list[str] = []
    lines.append(f"=== PR106 archive decomposition ===")
    lines.append(f"Archive: {manifest['input_archive']}")
    lines.append(f"SHA256:  {manifest['input_archive_sha256'][:16]}...")
    lines.append(f"Total bytes: {manifest['archive_size_bytes']:,}")
    lines.append(f"  compressed members: {manifest['total_compressed_bytes']:,}")
    lines.append(f"  zip overhead:        {manifest['zip_overhead_bytes']:,} "
                 f"({manifest['zip_overhead_pct']:.1f}%)")
    lines.append("")
    lines.append("BY CATEGORY:")
    lines.append(f"  {'category':<14s} | {'n':>3s} | {'compressed':>12s} | "
                 f"{'uncompressed':>14s} | {'share':>7s}")
    cats_sorted = sorted(
        manifest["by_category"].items(),
        key=lambda kv: -kv[1]["compressed_bytes"],
    )
    for cat, agg in cats_sorted:
        lines.append(
            f"  {cat:<14s} | {agg['n_files']:>3} | "
            f"{agg['compressed_bytes']:>12,} | "
            f"{agg['uncompressed_bytes']:>14,} | "
            f"{agg['compressed_share_pct']:>6.1f}%"
        )
    lines.append("")
    lines.append(f"Members ({manifest['n_members']}):")
    for m in sorted(manifest["members"], key=lambda r: -r["compressed_size"]):
        lines.append(
            f"  [{m['category']:<10s}] {m['name']:<40s}  "
            f"{m['compressed_size']:>10,} B ({m['compress_type_name']}, "
            f"ratio={m['ratio']:.3f})"
        )
    lines.append("")
    lines.append("OMEGA-OPT REALITY CHECK:")
    lines.append(f"  Design memo predicts:")
    lines.append(f"    renderer ~150 KB, masks ~120-150 KB, poses ~7 KB, overhead ~16 KB")
    lines.append(f"    Total: ~291 KB (predicted)")
    lines.append(f"  Empirical PR101 archive: {manifest['archive_size_bytes']:,} B")
    renderer_share = manifest["by_category"].get("renderer", {}).get("compressed_bytes", 0)
    masks_share = manifest["by_category"].get("masks", {}).get("compressed_bytes", 0)
    poses_share = manifest["by_category"].get("poses", {}).get("compressed_bytes", 0)
    side_share = manifest["by_category"].get("side_info", {}).get("compressed_bytes", 0)
    other_share = manifest["by_category"].get("other", {}).get("compressed_bytes", 0)
    lines.append(f"  Empirical breakdown:")
    lines.append(f"    renderer:  {renderer_share:>10,} B")
    lines.append(f"    masks:     {masks_share:>10,} B")
    lines.append(f"    poses:     {poses_share:>10,} B")
    lines.append(f"    side_info: {side_share:>10,} B")
    lines.append(f"    other:     {other_share:>10,} B")
    if masks_share == 0:
        lines.append(f"  WARNING: NO MASK MEMBERS DETECTED in this archive — likely")
        lines.append(f"           PR101 microcodec frontier (renderer-only candidate).")
        lines.append(f"           Mask-budget assumption in Ω-OPT memo is UNVERIFIED")
        lines.append(f"           on this substrate. PR106-with-masks archive needs")
        lines.append(f"           separate decomposition.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
        help="Archive zip to decompose (default: PR101 HNeRV FT microcodec frontier 178,258 B)",
    )
    p.add_argument(
        "--also-decompose", type=Path, nargs="*", default=None,
        help="Additional archives to decompose for cross-comparison",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--summary-text", action="store_true",
                   help="Print human-readable summary (default: JSON)")
    args = p.parse_args(argv)

    if not args.archive.is_file():
        raise SystemExit(f"archive not found: {args.archive}")

    manifests: list[dict] = [decompose(args.archive)]
    if args.also_decompose:
        for extra in args.also_decompose:
            if extra.is_file():
                manifests.append(decompose(extra))

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr106_archive_decomposition_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"runs": manifests, "tool": TOOL_NAME, "schema": SCHEMA_VERSION}, indent=2),
        encoding="utf-8",
    )

    if args.summary_text:
        for m in manifests:
            print(render_summary(m))
            print("\n" + "=" * 60 + "\n")
        print(f"manifest: {args.output_json}")
    else:
        print(render_summary(manifests[0]))
        print(f"\n(manifest: {args.output_json})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
