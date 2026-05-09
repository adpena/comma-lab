#!/usr/bin/env python3
"""Per-section entropy density vs encoded-bytes heatmap.

WHEN TO USE: when planning what byte-level section to attack next on a
contest archive. The binary forensics dossier
(`.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`)
documented the per-PR sidecar / decoder layouts; this tool quantifies WHICH
sections still have headroom against the Shannon floor (recoverable bits) and
WHICH are saturated (further codec work is dominated).

WHAT IT REVEALS: a section-by-section table where each row carries
(name, encoded_bytes, raw_byte_entropy_bits, encoded_bits_per_byte,
floor_bits_per_byte, saturation_ratio, recoverable_bytes). A section with
saturation_ratio close to 1.0 has nothing left for entropy coders. A section
with saturation_ratio < 0.6 still has 40%+ headroom — those are the sections
worth attacking.

The "raw byte entropy" is computed as Shannon entropy over the raw byte symbols
inside the encoded section AFTER the contest's encoded stream is extracted.
This is a LOWER bound on the entropy of the underlying signal (the encoded
stream is already losslessly compressed; its Shannon entropy is at most
encoded_bits_per_byte and at least the lossless-coder's noise floor).

For a section that is brotli-encoded INT8 weights, encoded_bits_per_byte is
typically ~6-7 (brotli output is near-uniform). The floor is the empirical
entropy of the SYMBOLS post-decompression — which we don't have without
inflating. This tool's "floor" is therefore an UPPER bound on attainable
compression: if the encoded stream's bytes are themselves near 8 bpb, brotli
cannot recompress them; if they are 5 bpb, a better entropy coder can still
gain roughly 3 bits/byte.

NOT a score claim. NOT a CUDA dispatch input. Tagged
`[diagnostic: archive section entropy heatmap]` per CLAUDE.md.

Output:
  experiments/results/xray_archive_section_entropy_heatmap_<timestamp>/
    heatmap.json                 — machine-readable per-section table
    heatmap.md                   — human-readable markdown (with regen header)
    rebuild_command.txt          — exact CLI to re-run

Usage:
  .venv/bin/python tools/xray_archive_section_entropy_heatmap.py \
      --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
      [--archive ...]   # multi-archive comparison
      [--output-dir experiments/results/xray_archive_section_entropy_heatmap_<ts>/]
      [--label pr101]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "xray_archive_section_entropy_heatmap_v1"
TOOL = "tools/xray_archive_section_entropy_heatmap.py"


def shannon_entropy_bits(byte_string: bytes) -> float:
    """Return Shannon entropy in bits per byte for ``byte_string``.

    A truly random byte stream has entropy 8.0 bits/byte. A maximally
    compressible stream (one repeated byte) has entropy 0.0 bits/byte.
    Brotli/zstd output is typically 7.0-7.95 bits/byte; INT8 quantized
    weights pre-codec are typically 4-6 bits/byte.
    """
    if not byte_string:
        return 0.0
    counts = Counter(byte_string)
    n = len(byte_string)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def section_recoverable_bytes(
    encoded_bytes: int,
    encoded_bits_per_byte: float,
    floor_bits_per_byte: float,
) -> int:
    """Estimate recoverable bytes if floor is achieved.

    recoverable = encoded * (1 - floor / encoded_bpb).
    Returns 0 when encoded_bpb <= floor_bpb (already saturated).
    """
    if encoded_bits_per_byte <= 0.0:
        return 0
    if floor_bits_per_byte >= encoded_bits_per_byte:
        return 0
    ratio = floor_bits_per_byte / encoded_bits_per_byte
    return max(0, int(round(encoded_bytes * (1.0 - ratio))))


def profile_archive_sections(
    archive_path: Path,
    *,
    label: str | None = None,
) -> dict:
    """Return per-section entropy heatmap for a single archive.

    For each ZIP member: read the encoded bytes (the stored bytes inside the
    ZIP, not the inflated contents — entropy of the on-disk archive bytes is
    what determines archive size). Compute Shannon entropy and saturation.
    """
    archive_path = Path(archive_path)
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    file_size = archive_path.stat().st_size

    sections: list[dict] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            with zf.open(info, "r") as f:
                payload = f.read()  # decompressed bytes (post-deflate)
            # Compressed-on-disk size from ZIP central directory:
            compressed_size = info.compress_size
            uncompressed_size = info.file_size
            # Entropy of the *uncompressed* payload. This is what an entropy
            # coder operates on. Brotli/zstd output has entropy near 8 bpb;
            # INT8 weights have entropy 4-6 bpb (recoverable headroom).
            payload_entropy = shannon_entropy_bits(payload)
            # Compression ratio observed in the archive (1.0 = stored,
            # < 1.0 = compressed by the ZIP coder).
            zip_compress_ratio = (
                compressed_size / uncompressed_size
                if uncompressed_size > 0 else 1.0
            )
            # Effective bits/byte the ZIP coder achieved on this member.
            zip_bits_per_byte = 8.0 * zip_compress_ratio
            # Saturation = how close ZIP got to Shannon floor.
            #   saturation_ratio = floor / actual; close to 1.0 means saturated
            saturation_ratio = (
                payload_entropy / zip_bits_per_byte
                if zip_bits_per_byte > 0.0 else 1.0
            )
            recoverable = section_recoverable_bytes(
                encoded_bytes=int(compressed_size),
                encoded_bits_per_byte=zip_bits_per_byte,
                floor_bits_per_byte=payload_entropy,
            )
            sections.append({
                "name": info.filename,
                "uncompressed_bytes": int(uncompressed_size),
                "compressed_bytes": int(compressed_size),
                "compress_method": int(info.compress_type),
                "payload_entropy_bits_per_byte": round(payload_entropy, 4),
                "zip_bits_per_byte": round(zip_bits_per_byte, 4),
                "saturation_ratio": round(min(saturation_ratio, 1.0), 4),
                "recoverable_bytes_if_floor_reached": int(recoverable),
            })

    # Sort sections by recoverable-bytes desc (biggest exploitation target first)
    sections.sort(key=lambda r: -r["recoverable_bytes_if_floor_reached"])

    total_compressed = sum(s["compressed_bytes"] for s in sections)
    total_uncompressed = sum(s["uncompressed_bytes"] for s in sections)
    total_recoverable = sum(s["recoverable_bytes_if_floor_reached"] for s in sections)

    return {
        "label": label or archive_path.stem,
        "archive_path": str(archive_path),
        "archive_sha256": sha,
        "file_size_bytes": int(file_size),
        "total_compressed_bytes": int(total_compressed),
        "total_uncompressed_bytes": int(total_uncompressed),
        "total_recoverable_if_floor_bytes": int(total_recoverable),
        "section_count": len(sections),
        "sections": sections,
    }


def render_markdown(report: dict, regen_header: str) -> str:
    """Render the heatmap as a markdown table with regen header."""
    lines: list[str] = [regen_header, ""]
    lines.append("# Archive section entropy heatmap")
    lines.append("")
    lines.append(
        f"_Schema_: `{report['schema_version']}` · _Generated_: "
        f"`{report['generated_at_utc']}`"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Archive | bytes | sections | Σrecoverable_if_floor |")
    lines.append("|---|---:|---:|---:|")
    for ar in report["archives"]:
        lines.append(
            f"| `{ar['label']}` | {ar['file_size_bytes']:,} | "
            f"{ar['section_count']} | "
            f"{ar['total_recoverable_if_floor_bytes']:,} |"
        )
    lines.append("")
    for ar in report["archives"]:
        lines.append(f"## `{ar['label']}` — sections")
        lines.append("")
        lines.append(
            "| section | comp_bytes | uncomp_bytes | payload_bpb | "
            "zip_bpb | sat_ratio | recover_bytes |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for s in ar["sections"]:
            lines.append(
                f"| `{s['name']}` | {s['compressed_bytes']:,} | "
                f"{s['uncompressed_bytes']:,} | "
                f"{s['payload_entropy_bits_per_byte']} | "
                f"{s['zip_bits_per_byte']} | "
                f"{s['saturation_ratio']} | "
                f"{s['recoverable_bytes_if_floor_reached']:,} |"
            )
        lines.append("")
    lines.append("")
    lines.append(
        "_Tag_: `[diagnostic: archive section entropy heatmap]`. "
        "Recoverable bytes are an UPPER BOUND assuming the entropy coder "
        "exactly hits the payload's Shannon floor. Brotli/zstd output is "
        "near-uniform (entropy ~8 bpb) so saturated sections have 0 "
        "recoverable bytes via re-coding alone — they require a different "
        "transform (different quantization, sparsification, or prior). "
        "Diagnostic only; never a score claim."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Per-archive entropy heatmap: which sections have recoverable bytes "
            "vs which are saturated. Diagnostic only."
        )
    )
    parser.add_argument(
        "--archive",
        action="append",
        required=True,
        help="Archive ZIP (repeat for multi-archive comparison)",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label per archive (repeat in same order as --archive)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write outputs to this dir (default: experiments/results/xray_archive_section_entropy_heatmap_<ts>/)",
    )
    args = parser.parse_args(argv)

    archives = [Path(a) for a in args.archive]
    labels = args.label or [None] * len(archives)
    if len(labels) != len(archives):
        print(
            f"ERROR: --label count ({len(labels)}) != --archive count ({len(archives)})",
            file=sys.stderr,
        )
        return 2

    for ar in archives:
        if not ar.exists():
            print(f"ERROR: archive not found: {ar}", file=sys.stderr)
            return 2

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"xray_archive_section_entropy_heatmap_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_reports = [
        profile_archive_sections(ar, label=lab)
        for ar, lab in zip(archives, labels)
    ]
    state_hash = hashlib.sha256(
        "|".join(r["archive_sha256"] for r in archive_reports).encode()
    ).hexdigest()[:16]
    report = {
        "schema_version": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "from_state_hash": state_hash,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "diagnostic_only",
        "archives": archive_reports,
    }

    out_json = out_dir / "heatmap.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    regen_header = (
        f"<!-- generated_at: {report['generated_at_utc']}, "
        f"from_state_hash: {report['from_state_hash']} -->"
    )
    out_md = out_dir / "heatmap.md"
    out_md.write_text(render_markdown(report, regen_header))

    cli = (
        ".venv/bin/python tools/xray_archive_section_entropy_heatmap.py \\\n  "
        + " \\\n  ".join(f"--archive {shlex.quote(str(a))}" for a in args.archive)
    )
    if args.label:
        cli += " \\\n  " + " \\\n  ".join(
            f"--label {shlex.quote(str(lab))}" for lab in args.label
        )
    (out_dir / "rebuild_command.txt").write_text(cli + "\n")

    print(f"[xray-entropy-heatmap] wrote {out_json}")
    print(f"[xray-entropy-heatmap] wrote {out_md}")
    print(f"[xray-entropy-heatmap] {len(archives)} archives, "
          f"{sum(r['section_count'] for r in archive_reports)} sections, "
          f"Σrecoverable={sum(r['total_recoverable_if_floor_bytes'] for r in archive_reports):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
