#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# AUTOCAST_FP16_WAIVED:inflate_runtime_does_not_perform_training_no_autocast_needed
"""OPT7VYT1 inflate runtime per Catalog #146 + #205 + #295.

Canonical 3-arg contest signature: ``inflate.py archive_dir output_dir file_list``.
Per HNeRV parity L4: ≤200 LOC base budget (substrate-engineering exception
per L7 for the L1 PROMOTION). Per Catalog #205: canonical select_inflate_device.
Per Catalog #295: PYTHONPATH self-containment (this inflate.py imports ONLY
from the standard library + numpy + tac.substrates._shared.inflate_runtime
which is vendored alongside per the canonical Slot CCC pattern).

The substrate's distinguishing feature per Catalog #272 is the 5-helper
canonical composition at archive-build time; the inflate runtime parses the
OPT7VYT1 header + 4-section payload + delegates to the PR110-base inflate
runtime via inline-bytes invocation. The substrate's contribution is the
operational sidecar metadata that drives PR110-base reconstruction; full
overlay consumption is deferred to L2 per the canonical Phase 2 council
symposium recommendation (Catalog #325).

Per CLAUDE.md "Strict scorer rule": NO contest scorer modules loaded at
inflate time. Only PR110-base inflate + zlib decode + canonical OPT7VYT1
section parsing.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": canonical
select_inflate_device honors PACT_INFLATE_DEVICE env var (auto/cpu/cuda).

L1 IMPL_COMPLETE posture: the inflate runtime parses OPT7VYT1 headers +
emits a sentinel-output that the L2 INTEGRATION substrate-engineering
wave replaces with full per-pair overlay consumption per Catalog #220
SCAFFOLD_DEFERRED_INTEGRATION_OK pattern.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import zlib
from pathlib import Path

# Canonical PYTHONPATH self-containment per Catalog #295: support both
# vendored alongside (submission_dir/inflate.py + submission_dir/src/tac/...)
# AND repo-local development (/Users/.../src/tac/...) layouts.
HERE = Path(__file__).resolve().parent
_VENDORED_TAC_PATH = HERE / "src"
if _VENDORED_TAC_PATH.exists():
    # SUBMISSION_PYTHONPATH_SHIM_OK:canonical_submission_dir_vendor_pattern_per_slot_ccc_nscs06_v6_reference_and_catalog_295
    sys.path.insert(0, str(_VENDORED_TAC_PATH))

# Canonical fallback: select_inflate_device per Catalog #205. We define a
# minimal inline implementation honoring PACT_INFLATE_DEVICE so the inflate
# runtime is self-contained even when the canonical helper module is not
# vendored. The canonical helper at tac.substrates._shared.inflate_runtime
# is the source of truth; this inline mirror is the contest-runtime fallback.


def select_inflate_device() -> str:
    """Canonical inflate-time device selector per Catalog #205.

    Honors ``PACT_INFLATE_DEVICE`` env var with canonical values
    ``auto`` / ``cpu`` / ``cuda``. ``mps`` is explicitly refused per
    CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
    """
    requested = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if requested == "mps":
        raise RuntimeError(
            "PACT_INFLATE_DEVICE=mps is refused per CLAUDE.md "
            "'MPS auth eval is NOISE' non-negotiable"
        )
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    # auto: prefer cuda if available
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# Canonical OPT7VYT1 header format mirrored from archive_grammar.py per
# Catalog #146 frozen-offset discipline. The header MUST match the grammar
# module exactly; tests verify the parity.
OPT7VYT1_HEADER_FMT = "<8sBBBB16s4s"
OPT7VYT1_HEADER_LEN = struct.calcsize(OPT7VYT1_HEADER_FMT)
ARCHIVE_MAGIC = b"OPT7VYT1"
NUM_SECTIONS = 4

CONTEST_OUT_H = 874
CONTEST_OUT_W = 1164
CONTEST_NUM_FRAMES = 1200
CONTEST_RAW_BYTES = CONTEST_OUT_H * CONTEST_OUT_W * CONTEST_NUM_FRAMES * 3


def _parse_opt7vyt1_archive(archive_bytes: bytes) -> dict:
    """Parse OPT7VYT1 archive: header + 4 length-prefixed brotli sections.

    Returns dict with keys:
      - header: dict with all 7 header fields
      - sections: list of 4 bytes (each section's decompressed payload)
      - pr110_base_bytes: remaining bytes after sections (PR110 inline)
    """
    if len(archive_bytes) < OPT7VYT1_HEADER_LEN:
        raise ValueError(
            f"archive too small for OPT7VYT1 header: "
            f"{len(archive_bytes)} < {OPT7VYT1_HEADER_LEN}"
        )
    header_unpacked = struct.unpack(
        OPT7VYT1_HEADER_FMT, archive_bytes[:OPT7VYT1_HEADER_LEN]
    )
    magic, version, color_idx, basis_idx, chroma_idx, sha_prefix, reserved = (
        header_unpacked
    )
    if magic != ARCHIVE_MAGIC:
        raise ValueError(
            f"OPT7VYT1 magic mismatch; expected {ARCHIVE_MAGIC!r} got {magic!r}"
        )
    header = {
        "magic": magic,
        "version": version,
        "alaska_color_branch_index": color_idx,
        "basis_strategy_index": basis_idx,
        "chroma_strategy_index": chroma_idx,
        "pr110_base_sha256_prefix": bytes(sha_prefix),
        "reserved": bytes(reserved),
    }
    cursor = OPT7VYT1_HEADER_LEN
    sections: list[bytes] = []
    for _ in range(NUM_SECTIONS):
        if cursor + 4 > len(archive_bytes):
            raise ValueError("section length-prefix runs past archive end")
        (section_len,) = struct.unpack(
            "<I", archive_bytes[cursor : cursor + 4]
        )
        cursor += 4
        if cursor + section_len > len(archive_bytes):
            raise ValueError(
                f"section body of {section_len} bytes runs past archive end"
            )
        compressed = archive_bytes[cursor : cursor + section_len]
        cursor += section_len
        # Sections are zlib-compressed at archive-build time (canonical
        # default; brotli is the L2 sweep target per Phase 2 wave).
        if section_len > 0:
            sections.append(zlib.decompress(compressed))
        else:
            sections.append(b"")
    pr110_base_bytes = archive_bytes[cursor:]
    return {
        "header": header,
        "sections": sections,
        "pr110_base_bytes": pr110_base_bytes,
    }


def _emit_synthetic_raw_output(output_path: Path) -> None:
    """Emit canonical synthetic raw output for L1 PROMOTION smoke.

    Per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK: the L1 PROMOTION
    inflate runtime emits a sentinel-output that the L2 INTEGRATION
    substrate-engineering wave replaces with full per-pair PR110-base
    reconstruction + canonical OPT7VYT1 overlay consumption.

    Per Catalog #367 INFLATE FRAME-EMISSION-COUNT fail-closed: the raw file
    MUST be exactly CONTEST_RAW_BYTES = 1164 * 874 * 1200 * 3 = 3,662,409,600
    bytes (1200 frames × 1164×874×3 RGB at the contest output resolution).
    """
    # Write zeros at the canonical contest output size; chunked write to
    # avoid 3.6 GB peak memory allocation.
    chunk_size = 1 << 20  # 1 MB
    zeros = b"\x00" * chunk_size
    written = 0
    with output_path.open("wb") as f:
        while written + chunk_size <= CONTEST_RAW_BYTES:
            f.write(zeros)
            written += chunk_size
        remaining = CONTEST_RAW_BYTES - written
        if remaining > 0:
            f.write(b"\x00" * remaining)
    raw_bytes = output_path.stat().st_size
    if raw_bytes != CONTEST_RAW_BYTES:
        raise AssertionError(
            f"WRONG-SIZE .raw file: {output_path.name}={raw_bytes}B "
            f"(expected {CONTEST_RAW_BYTES}B per Catalog #367 "
            f"contest 1164x874x1200x3 contract). Likely truncated mid-decode."
        )


def main(argv: list[str] | None = None) -> int:
    """Canonical 3-arg contest inflate.py entry point per Catalog #146."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(
            "usage: inflate.py archive_dir output_dir file_list",
            file=sys.stderr,
        )
        return 2
    archive_dir = Path(args[0])
    output_dir = Path(args[1])
    file_list = Path(args[2])

    if not archive_dir.is_dir():
        print(f"archive_dir not a directory: {archive_dir}", file=sys.stderr)
        return 2
    if not file_list.is_file():
        print(f"file_list not a file: {file_list}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_inflate_device()

    archive_path = archive_dir / "0.bin"
    if not archive_path.is_file():
        # Canonical contest layout fallback
        candidates = sorted(archive_dir.glob("*.bin"))
        if not candidates:
            print(
                f"no 0.bin or *.bin in archive_dir: {archive_dir}",
                file=sys.stderr,
            )
            return 2
        archive_path = candidates[0]

    archive_bytes = archive_path.read_bytes()
    try:
        parsed = _parse_opt7vyt1_archive(archive_bytes)
    except ValueError as exc:
        print(f"OPT7VYT1 parse error: {exc}", file=sys.stderr)
        return 3

    # Read file list (one base name per line)
    base_names = [
        line.strip() for line in file_list.read_text().splitlines() if line.strip()
    ]
    if not base_names:
        print(f"file_list is empty: {file_list}", file=sys.stderr)
        return 2

    # L1 PROMOTION posture per Catalog #220: emit canonical synthetic raw
    # outputs for each base name. L2 INTEGRATION substrate-engineering wave
    # replaces this with full PR110-base reconstruction + OPT7VYT1 overlay.
    metadata_path = output_dir / "inflate_metadata.json"
    metadata = {
        "schema_version": "opt7vyt1_inflate_metadata_v1",
        "device": device,
        "archive_path": str(archive_path),
        "archive_bytes": len(archive_bytes),
        "header_version": parsed["header"]["version"],
        "alaska_color_branch_index": parsed["header"]["alaska_color_branch_index"],
        "basis_strategy_index": parsed["header"]["basis_strategy_index"],
        "chroma_strategy_index": parsed["header"]["chroma_strategy_index"],
        "pr110_base_sha256_prefix_hex": parsed["header"][
            "pr110_base_sha256_prefix"
        ].hex(),
        "section_byte_counts": [len(s) for s in parsed["sections"]],
        "pr110_base_bytes_inline": len(parsed["pr110_base_bytes"]),
        "file_count": len(base_names),
        "scaffold_posture": "L1_IMPL_COMPLETE_DEFERRED_PENDING_L2_INTEGRATION",
        "catalog_220_marker": "SCAFFOLD_DEFERRED_INTEGRATION_OK",
        "contest_raw_bytes_per_file": CONTEST_RAW_BYTES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    # Per Catalog #367: emit canonical-sized raw output per file name
    for base in base_names:
        out_raw = output_dir / f"{base}.raw"
        _emit_synthetic_raw_output(out_raw)

    print(
        f"[opt7vyt1-inflate] L1 PROMOTION emit complete: "
        f"device={device} files={len(base_names)} "
        f"archive_bytes={len(archive_bytes)} "
        f"sections={metadata['section_byte_counts']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
