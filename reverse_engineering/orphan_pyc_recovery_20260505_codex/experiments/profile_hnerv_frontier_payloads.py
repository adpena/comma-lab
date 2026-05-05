# pyc-recovery: human-reconstructed from experiments/profile_hnerv_frontier_payloads.py.pyc
# This is the canonical main-repo content as of 2026-05-05.
# Recovery spec preserved at: profile_hnerv_frontier_payloads.recovery_spec.json
# Original STUB has been replaced with this canonical version.
#!/usr/bin/env python3
"""Profile byte layout for public HNeRV frontier archives.

This is a deconstruction tool, not a scorer. It reads a contest archive,
extracts the single charged payload member, and emits deterministic section
bytes, SHA-256, and entropy so public-submission ideas can feed repacking and
Lagrangian allocation work without relying on prose notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile


PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387

PR103_SCA_LEN = 56
PR103_BR_LEN = 7_097
PR103_HIST_LEN = 895
PR103_MERGED_AC_LEN = 153_856
PR103_LATENT_META_LEN = 112
PR103_LO_LEN = 15_537
PR103_HI_HIST_LEN = 15


@dataclass(frozen=True)
class SectionProfile:
    name: str
    start: int
    end: int
    bytes: int
    sha256: str
    entropy_bits_per_byte: float


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    counts = Counter(data)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def section(name: str, blob: bytes, start: int, end: int) -> SectionProfile:
    if start < 0 or end < start or end > len(blob):
        raise ValueError(f"bad section {name}: start={start} end={end} len={len(blob)}")
    data = blob[start:end]
    return SectionProfile(
        name=name,
        start=start,
        end=end,
        bytes=len(data),
        sha256=sha256_bytes(data),
        entropy_bits_per_byte=round(entropy_bits_per_byte(data), 6),
    )


def extract_single_member(archive: Path) -> tuple[str, bytes, int, str]:
    archive_blob = archive.read_bytes()
    with ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"{archive} has {len(infos)} file members; expected exactly one")
        member = infos[0]
        payload = zf.read(member.filename)
    return member.filename, payload, len(archive_blob), sha256_bytes(archive_blob)


def infer_profile_kind(kind: str, archive: Path, member_name: str, payload: bytes) -> str:
    if kind != "auto":
        return kind
    text = f"{archive} {member_name}".lower()
    if "pr101" in text or "hnerv_ft_microcodec" in text:
        return "pr101_microcodec"
    if "pr103" in text or "hnerv_lc_ac" in text:
        return "pr103_lc_ac"
    if payload[:1] == b"\xff" and len(payload) >= 4:
        dec_len = int.from_bytes(payload[1:4], "little")
        if 0 < dec_len < len(payload) - 4:
            return "ff_packed_brotli_hnerv"
    if len(payload) == PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN + 607:
        return "pr101_microcodec"
    pr103_min = (
        PR103_SCA_LEN
        + PR103_BR_LEN
        + PR103_HIST_LEN
        + PR103_MERGED_AC_LEN
        + PR103_LATENT_META_LEN
        + PR103_LO_LEN
        + PR103_HI_HIST_LEN
    )
    if len(payload) >= pr103_min:
        return "pr103_lc_ac"
    return "unknown_single_payload"


def profile_payload(kind: str, payload: bytes) -> list[SectionProfile]:
    sections: list[SectionProfile] = []
    if kind == "pr101_microcodec":
        decoder_end = PR101_DECODER_BLOB_LEN
        latent_end = decoder_end + PR101_LATENT_BLOB_LEN
        return [
            section("decoder_compact_brotli_streams", payload, 0, decoder_end),
            section("latents_raw_lzma_delta_u8", payload, decoder_end, latent_end),
            section("sidecar_dim_delta_huffman_enum", payload, latent_end, len(payload)),
        ]
    if kind == "pr103_lc_ac":
        cursor = 0
        for name, size in [
            ("scales_fp16", PR103_SCA_LEN),
            ("non_ac_weights_brotli", PR103_BR_LEN),
            ("ac_histograms_brotli", PR103_HIST_LEN),
            ("merged_range_coded_weights_and_hi_latents", PR103_MERGED_AC_LEN),
            ("latent_min_scale_fp16", PR103_LATENT_META_LEN),
            ("latent_low_bytes_brotli", PR103_LO_LEN),
            ("latent_hi_histogram_brotli", PR103_HI_HIST_LEN),
        ]:
            sections.append(section(name, payload, cursor, cursor + size))
            cursor += size
        sections.append(section("sidecar_corrections_brotli", payload, cursor, len(payload)))
        return sections
    if kind == "ff_packed_brotli_hnerv":
        dec_len = int.from_bytes(payload[1:4], "little")
        return [
            section("packed_header_ff_len24", payload, 0, 4),
            section("decoder_packed_brotli", payload, 4, 4 + dec_len),
            section("latents_and_sidecar_brotli", payload, 4 + dec_len, len(payload)),
        ]
    return [section("opaque_single_payload", payload, 0, len(payload))]


def build_record(archive: Path, kind: str) -> dict[str, object]:
    member_name, payload, archive_bytes, archive_sha = extract_single_member(archive)
    resolved_kind = infer_profile_kind(kind, archive, member_name, payload)
    sections = profile_payload(resolved_kind, payload)
    return {
        "archive": str(archive),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "member_name": member_name,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "kind": resolved_kind,
        "zip_overhead_bytes": archive_bytes - len(payload),
        "sections": [asdict(item) for item in sections],
        "evidence_grade": "forensic_byte_profile",
        "score_claim": False,
    }


def render_markdown(record: dict[str, object]) -> str:
    lines = [
        f"# HNeRV Frontier Payload Profile: {record['archive']}",
        "",
        f"- archive_bytes: `{record['archive_bytes']}`",
        f"- archive_sha256: `{record['archive_sha256']}`",
        f"- zip_member: `{record['member_name']}`",
        f"- member_bytes: `{record['member_bytes']}`",
        f"- member_sha256: `{record['member_sha256']}`",
        f"- inferred_kind: `{record['kind']}`",
        f"- zip_overhead_bytes: `{record['zip_overhead_bytes']}`",
        "",
        "| section | start | end | bytes | entropy b/B | sha256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in record["sections"]:
        lines.append(
            "| {name} | {start} | {end} | {bytes} | {entropy_bits_per_byte:.6f} | `{sha256}` |".format(
                **item
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--kind", default="auto")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    records = [build_record(path, args.kind) for path in args.archives]
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text("\n".join(render_markdown(record) for record in records))
    if not args.json_out and not args.md_out:
        print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
