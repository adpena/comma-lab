#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ruff: noqa: I001
"""Compose a pr106_stacked archive from pre-built sister-lane archives.

This is the META-COMPOSITION builder. It takes:
  - 1 PR106 packed archive (anchor; contains 0.bin with 0xFF magic)
  - 0..4 sister-lane archives (each a single-sidechannel wrapped PR106 archive)

and produces a single pr106_stacked archive with all the sidechannel sections
composed into a single 0.bin under the wire format documented in
submissions/pr106_stacked/inflate.py.

Wire format extracted from each sister archive:
  - --latent ARCHIVE  : pr106_latent_sidecar archive (magic 0xFE).
                        Section payload = the appended sidecar_blob (already
                        brotli-compressed in PR100 hnerv_lc_v2 wire form).
  - --yshift ARCHIVE  : pr106_yshift_sidechannel archive (magic 0xFC).
                        Section payload = the appended SC01 brotli'd blob.
  - --lrl1 ARCHIVE    : pr106_lrl1_sidechannel archive (magic 0xFB).
                        Section payload = the appended LR01 brotli'd blob.
  - --wavelet ARCHIVE : hnerv_wavelet_sidechannel archive (magic 0xFA).
                        Section payload = the appended WR01 brotli'd blob.

Each sidechannel input is OPTIONAL. With ZERO sister archives, the output is
a "pure passthrough" pr106_stacked (PR106 + only end-sentinel) which inflates
to byte-identical pixels as plain PR106 inflate — that's the closing-the-loop
test in src/tac/tests/test_pr106_stacked.py.

Sidechannel application order is canonical (latent → yshift → lrl1 → wavelet
consume-only proof) regardless of CLI arg order or wire-section order.

Usage (build the full 3-stack from sister archives, plus pure-passthrough):
    .venv/bin/python experiments/build_pr106_stacked.py \\
        --pr106-archive <pr106_archive.zip> \\
        --latent <pr106_latent_sidecar/sidecar_archive.zip> \\
        --yshift <pr106_yshift_sidechannel_archive.zip> \\
        --lrl1   <pr106_lrl1_sidechannel_archive.zip> \\
        --wavelet <hnerv_wavelet_sidechannel_candidate.zip> \\
        --output-dir experiments/results/lane_pr106_stacked_$(date -u +%Y%m%dT%H%M%SZ)

CPU smoke (zero corrections, byte-identical pixel guarantee):
    Build each sister with --search-mode zero first, then compose. The pixel
    output will match plain PR106 inflate (latent dim=255 sentinel + yshift
    zero-row + lrl1 zero-coef are all true no-ops).

CLAUDE.md MPS-noise rule: --device cpu is acceptable for compose-time scaffold
(this script is a pure-bytes composition; no scorer/decoder forward). The
inflate-time pixel render IS CUDA-required.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import struct
import sys
import time
import zipfile
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    prepend_paths = _tool_bootstrap.prepend_paths
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
prepend_paths(
    REPO_ROOT / "submissions" / "pr106_stacked",
    REPO_ROOT / "submissions" / "pr106_latent_sidecar" / "src",
)

from tac.repo_io import json_text, sha256_file

# Imported AFTER path insertion: pr106_stacked outer parser + section constants.
from inflate import (  # type: ignore[import-not-found]
    STACKED_MAGIC_BYTE,
    SECTION_END,
    SECTION_LATENT,
    SECTION_YSHIFT,
    SECTION_LRL1,
    SECTION_WAVELET,
    parse_stacked_archive,
    decode_latent_blob,
    decode_yshift_blob,
    decode_lrl1_blob,
    decode_wavelet_blob,
)


# Per-sister magic bytes (used to validate input archive shapes)
LATENT_OUTER_MAGIC = 0xFE
LATENT_FORMAT_ID = 0x01
YSHIFT_OUTER_MAGIC = 0xFC
LRL1_OUTER_MAGIC = 0xFB
WAVELET_OUTER_MAGIC = 0xFA
WAVELET_OUTER_VERSION = 1


def _read_single_zip_payload(archive_path: Path) -> tuple[str, bytes]:
    """Read the only member from a single-member ZIP archive."""
    with zipfile.ZipFile(archive_path) as z:
        infos = z.infolist()
        if len(infos) != 1:
            raise ValueError(f"{archive_path}: expected one ZIP member, got {len(infos)}")
        info = infos[0]
        if info.is_dir():
            raise ValueError(f"{archive_path}: single ZIP member is a directory")
        return info.filename, z.read(info.filename)


def _read_zip_bin(archive_path: Path) -> bytes:
    """Read 0.bin from a single-member archive.zip (sister-lane convention)."""
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        if "0.bin" not in names:
            raise ValueError(
                f"{archive_path}: 0.bin missing (members={names})"
            )
        return z.read("0.bin")


def _archive_custody(path: Path | None) -> dict[str, object] | None:
    """Return byte/hash custody for an optional archive path."""
    if path is None:
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def extract_pr106_bytes(pr106_archive: Path) -> bytes:
    """Read the PR106 packed-archive 0.bin verbatim. Must start with 0xFF magic."""
    pr106_bytes = _read_zip_bin(pr106_archive)
    if not pr106_bytes:
        raise ValueError(f"{pr106_archive}: empty 0.bin")
    if pr106_bytes[0] != 0xFF:
        raise ValueError(
            f"{pr106_archive}: 0.bin first byte 0x{pr106_bytes[0]:02X} != 0xFF "
            f"(expected PR106 packed-archive magic)"
        )
    return pr106_bytes


def extract_latent_section_blob(
    latent_archive: Path,
    *,
    expected_pr106_bytes: bytes | None = None,
) -> bytes:
    """Extract the latent sidecar payload (already-brotli-compressed) from a
    pr106_latent_sidecar archive.

    The sister archive layout is:
        magic(0xFE) + format_id(0x01) + uint32 pr106_len + pr106_bytes +
        uint16 sidecar_len + sidecar_bytes (the brotli'd PR100 hnerv_lc_v2 blob)
    """
    bin_bytes = _read_zip_bin(latent_archive)
    if not bin_bytes:
        raise ValueError(f"{latent_archive}: empty 0.bin")
    if bin_bytes[0] != LATENT_OUTER_MAGIC:
        raise ValueError(
            f"{latent_archive}: outer magic 0x{bin_bytes[0]:02X} != "
            f"0x{LATENT_OUTER_MAGIC:02X} (expected pr106_latent_sidecar)"
        )
    if bin_bytes[1] != LATENT_FORMAT_ID:
        raise ValueError(
            f"{latent_archive}: format_id 0x{bin_bytes[1]:02X} != "
            f"0x{LATENT_FORMAT_ID:02X}"
        )
    pos = 2
    (pr106_len,) = struct.unpack_from("<I", bin_bytes, pos)
    pos += 4
    pr106_end = pos + pr106_len
    if pr106_end > len(bin_bytes):
        raise ValueError(f"{latent_archive}: embedded PR106 payload overruns archive")
    if expected_pr106_bytes is not None and bin_bytes[pos:pr106_end] != expected_pr106_bytes:
        raise ValueError(
            f"{latent_archive}: embedded PR106 payload does not match stack anchor"
        )
    pos = pr106_end
    if pos + 2 > len(bin_bytes):
        raise ValueError(f"{latent_archive}: truncated before sidecar_len")
    (sc_len,) = struct.unpack_from("<H", bin_bytes, pos)
    pos += 2
    end = pos + sc_len
    if end != len(bin_bytes):
        raise ValueError(
            f"{latent_archive}: sidecar trailing bytes (pos={end}, "
            f"total={len(bin_bytes)})"
        )
    return bin_bytes[pos:end]  # already brotli-compressed PR100-style blob


def extract_yshift_section_blob(
    yshift_archive: Path,
    *,
    expected_pr106_bytes: bytes | None = None,
) -> bytes:
    """Extract the SC01 brotli'd blob from a pr106_yshift_sidechannel archive.

    Sister archive layout:
        magic(0xFC) + uint24 pr106_len + pr106_bytes + sc_version(1) +
        uint16 sc_len + brotli(SC01 payload)
    """
    bin_bytes = _read_zip_bin(yshift_archive)
    if not bin_bytes:
        raise ValueError(f"{yshift_archive}: empty 0.bin")
    if bin_bytes[0] != YSHIFT_OUTER_MAGIC:
        raise ValueError(
            f"{yshift_archive}: outer magic 0x{bin_bytes[0]:02X} != "
            f"0x{YSHIFT_OUTER_MAGIC:02X} (expected pr106_yshift_sidechannel)"
        )
    pr106_len = int.from_bytes(bin_bytes[1:4], "little")
    pos = 4
    pr106_end = pos + pr106_len
    if pr106_end > len(bin_bytes):
        raise ValueError(f"{yshift_archive}: embedded PR106 payload overruns archive")
    if expected_pr106_bytes is not None and bin_bytes[pos:pr106_end] != expected_pr106_bytes:
        raise ValueError(
            f"{yshift_archive}: embedded PR106 payload does not match stack anchor"
        )
    pos = pr106_end
    if pos >= len(bin_bytes):
        raise ValueError(f"{yshift_archive}: no sidechannel section present")
    sc_version = bin_bytes[pos]
    pos += 1
    if sc_version != 1:
        raise ValueError(
            f"{yshift_archive}: sc_version {sc_version} != 1"
        )
    if pos + 2 > len(bin_bytes):
        raise ValueError(f"{yshift_archive}: truncated before sc_len")
    sc_len = struct.unpack_from("<H", bin_bytes, pos)[0]
    pos += 2
    end = pos + sc_len
    if end != len(bin_bytes):
        raise ValueError(
            f"{yshift_archive}: sc trailing bytes (pos={end}, "
            f"total={len(bin_bytes)})"
        )
    return bin_bytes[pos:end]  # already brotli'd SC01 blob


def extract_lrl1_section_blob(
    lrl1_archive: Path,
    *,
    expected_pr106_bytes: bytes | None = None,
) -> bytes:
    """Extract the LR01 brotli'd blob from a pr106_lrl1_sidechannel archive.

    Sister archive layout (mirrors yshift sister; only outer magic differs):
        magic(0xFB) + uint24 pr106_len + pr106_bytes + sc_version(1) +
        uint16 sc_len + brotli(LR01 payload)
    """
    bin_bytes = _read_zip_bin(lrl1_archive)
    if not bin_bytes:
        raise ValueError(f"{lrl1_archive}: empty 0.bin")
    if bin_bytes[0] != LRL1_OUTER_MAGIC:
        raise ValueError(
            f"{lrl1_archive}: outer magic 0x{bin_bytes[0]:02X} != "
            f"0x{LRL1_OUTER_MAGIC:02X} (expected pr106_lrl1_sidechannel)"
        )
    pr106_len = int.from_bytes(bin_bytes[1:4], "little")
    pos = 4
    pr106_end = pos + pr106_len
    if pr106_end > len(bin_bytes):
        raise ValueError(f"{lrl1_archive}: embedded PR106 payload overruns archive")
    if expected_pr106_bytes is not None and bin_bytes[pos:pr106_end] != expected_pr106_bytes:
        raise ValueError(
            f"{lrl1_archive}: embedded PR106 payload does not match stack anchor"
        )
    pos = pr106_end
    if pos >= len(bin_bytes):
        raise ValueError(f"{lrl1_archive}: no sidechannel section present")
    sc_version = bin_bytes[pos]
    pos += 1
    if sc_version != 1:
        raise ValueError(
            f"{lrl1_archive}: sc_version {sc_version} != 1"
        )
    if pos + 2 > len(bin_bytes):
        raise ValueError(f"{lrl1_archive}: truncated before sc_len")
    sc_len = struct.unpack_from("<H", bin_bytes, pos)[0]
    pos += 2
    end = pos + sc_len
    if end != len(bin_bytes):
        raise ValueError(
            f"{lrl1_archive}: sc trailing bytes (pos={end}, "
            f"total={len(bin_bytes)})"
        )
    return bin_bytes[pos:end]  # already brotli'd LR01 blob


def extract_wavelet_section_blob(
    wavelet_archive: Path,
    *,
    expected_pr106_bytes: bytes | None = None,
) -> bytes:
    """Extract the WR01 brotli'd blob from an hnerv_wavelet_sidechannel archive.

    Sister archive layout:
        magic(0xFA) + version(1) + uint24 pr106_len + pr106_bytes +
        uint32 wr01_len + brotli(WR01 atom payload)
    """
    member_name, bin_bytes = _read_single_zip_payload(wavelet_archive)
    if not bin_bytes:
        raise ValueError(f"{wavelet_archive}: empty {member_name}")
    if len(bin_bytes) < 9:
        raise ValueError(f"{wavelet_archive}: truncated wavelet wrapper")
    if bin_bytes[0] != WAVELET_OUTER_MAGIC:
        raise ValueError(
            f"{wavelet_archive}: outer magic 0x{bin_bytes[0]:02X} != "
            f"0x{WAVELET_OUTER_MAGIC:02X} (expected hnerv_wavelet_sidechannel)"
        )
    if bin_bytes[1] != WAVELET_OUTER_VERSION:
        raise ValueError(
            f"{wavelet_archive}: version {bin_bytes[1]} != {WAVELET_OUTER_VERSION}"
        )
    pr106_len = int.from_bytes(bin_bytes[2:5], "little")
    pos = 5
    pr106_end = pos + pr106_len
    if pr106_end + 4 > len(bin_bytes):
        raise ValueError(f"{wavelet_archive}: truncated before WR01 length")
    pr106_bytes = bin_bytes[pos:pr106_end]
    if expected_pr106_bytes is not None and pr106_bytes != expected_pr106_bytes:
        raise ValueError(
            f"{wavelet_archive}: embedded PR106 payload does not match stack anchor"
        )
    wr01_len = struct.unpack_from("<I", bin_bytes, pr106_end)[0]
    wr01_start = pr106_end + 4
    wr01_end = wr01_start + wr01_len
    if wr01_end != len(bin_bytes):
        raise ValueError(
            f"{wavelet_archive}: WR01 trailing bytes (pos={wr01_end}, "
            f"total={len(bin_bytes)})"
        )
    return bin_bytes[wr01_start:wr01_end]  # already brotli'd WR01 blob


def build_stacked_archive_bytes(
    pr106_bytes: bytes,
    *,
    latent_blob: bytes | None = None,
    yshift_blob: bytes | None = None,
    lrl1_blob: bytes | None = None,
    wavelet_blob: bytes | None = None,
) -> bytes:
    """Compose the pr106_stacked 0.bin layout.

    Sections are written in canonical order (latent → yshift → lrl1 → wavelet) and
    terminated with the 0x00 end-of-sections sentinel. Inflate-time
    application order is also canonical regardless of wire order, so this
    write order is just convention.
    """
    if len(pr106_bytes) >= (1 << 24):
        raise ValueError(
            f"pr106 bytes too large for uint24: {len(pr106_bytes)}"
        )
    out = io.BytesIO()
    out.write(bytes([STACKED_MAGIC_BYTE]))
    out.write(len(pr106_bytes).to_bytes(3, "little"))
    out.write(pr106_bytes)
    for section_id, blob in (
        (SECTION_LATENT, latent_blob),
        (SECTION_YSHIFT, yshift_blob),
        (SECTION_LRL1, lrl1_blob),
        (SECTION_WAVELET, wavelet_blob),
    ):
        if blob is None:
            continue
        if len(blob) >= (1 << 16):
            raise ValueError(
                f"section 0x{section_id:02X} blob too large for uint16: "
                f"{len(blob)}"
            )
        out.write(bytes([section_id]))
        out.write(struct.pack("<H", len(blob)))
        out.write(blob)
    out.write(bytes([SECTION_END]))
    return out.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pr106-archive", type=Path, required=True,
                        help="PR106 packed archive.zip (anchor).")
    parser.add_argument("--latent", type=Path, default=None,
                        help="pr106_latent_sidecar archive.zip (optional).")
    parser.add_argument("--yshift", type=Path, default=None,
                        help="pr106_yshift_sidechannel archive.zip (optional).")
    parser.add_argument("--lrl1", type=Path, default=None,
                        help="pr106_lrl1_sidechannel archive.zip (optional).")
    parser.add_argument("--wavelet", type=Path, default=None,
                        help="hnerv_wavelet_sidechannel archive.zip (optional).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for composed archive + metadata.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Stage A: load PR106 anchor.
    if not args.pr106_archive.is_file():
        sys.exit(f"FATAL: PR106 anchor not found at {args.pr106_archive}")
    pr106_bytes = extract_pr106_bytes(args.pr106_archive)
    pr106_size_bytes = len(pr106_bytes)
    pr106_zip_size = args.pr106_archive.stat().st_size
    print(f"[build-stacked] PR106 0.bin: {pr106_size_bytes} bytes "
          f"(zip {pr106_zip_size} bytes)")

    # Stage B: extract each sidechannel section blob (already brotli'd).
    sections: dict[int, bytes] = {}
    if args.latent is not None:
        if not args.latent.is_file():
            sys.exit(f"FATAL: --latent archive not found at {args.latent}")
        sections[SECTION_LATENT] = extract_latent_section_blob(
            args.latent,
            expected_pr106_bytes=pr106_bytes,
        )
        print(f"[build-stacked] latent section: {len(sections[SECTION_LATENT])} bytes "
              f"(from {args.latent})")
    if args.yshift is not None:
        if not args.yshift.is_file():
            sys.exit(f"FATAL: --yshift archive not found at {args.yshift}")
        sections[SECTION_YSHIFT] = extract_yshift_section_blob(
            args.yshift,
            expected_pr106_bytes=pr106_bytes,
        )
        print(f"[build-stacked] yshift section: {len(sections[SECTION_YSHIFT])} bytes "
              f"(from {args.yshift})")
    if args.lrl1 is not None:
        if not args.lrl1.is_file():
            sys.exit(f"FATAL: --lrl1 archive not found at {args.lrl1}")
        sections[SECTION_LRL1] = extract_lrl1_section_blob(
            args.lrl1,
            expected_pr106_bytes=pr106_bytes,
        )
        print(f"[build-stacked] lrl1 section: {len(sections[SECTION_LRL1])} bytes "
              f"(from {args.lrl1})")
    if args.wavelet is not None:
        if not args.wavelet.is_file():
            sys.exit(f"FATAL: --wavelet archive not found at {args.wavelet}")
        sections[SECTION_WAVELET] = extract_wavelet_section_blob(
            args.wavelet,
            expected_pr106_bytes=pr106_bytes,
        )
        print(f"[build-stacked] wavelet WR01 section: {len(sections[SECTION_WAVELET])} bytes "
              f"(from {args.wavelet})")

    # Stage C: validate each section by parsing back through the inflate-side
    # decoder. This is a roundtrip safety net that catches drift between
    # builder + inflater section parsers.
    if SECTION_LATENT in sections:
        decoded_latent = decode_latent_blob(sections[SECTION_LATENT])
        n_corr = int((decoded_latent["dim"] != 255).sum())
        print(f"[build-stacked] latent verified: n_pairs={decoded_latent['n_pairs']}, "
              f"corrections={n_corr}")
    if SECTION_YSHIFT in sections:
        decoded_yshift = decode_yshift_blob(sections[SECTION_YSHIFT])
        print(f"[build-stacked] yshift verified: n_frames={decoded_yshift['n_frames']}, "
              f"step={decoded_yshift['step']}")
    if SECTION_LRL1 in sections:
        decoded_lrl1 = decode_lrl1_blob(sections[SECTION_LRL1])
        print(f"[build-stacked] lrl1 verified: K={decoded_lrl1['K']}, "
              f"basis={decoded_lrl1['low_h']}x{decoded_lrl1['low_w']}, "
              f"n_frames={decoded_lrl1['n_frames']}")
    if SECTION_WAVELET in sections:
        decoded_wavelet = decode_wavelet_blob(sections[SECTION_WAVELET])
        proof = decoded_wavelet["runtime_consumption_proof"]
        print(
            f"[build-stacked] wavelet WR01 verified: "
            f"sections={decoded_wavelet['section_count']}, "
            f"atoms={proof['decoded_atom_count']}, "
            f"mode={decoded_wavelet['runtime_mode']}"
        )
    else:
        decoded_wavelet = None

    # Stage D: compose the outer archive bytes.
    new_bin = build_stacked_archive_bytes(
        pr106_bytes,
        latent_blob=sections.get(SECTION_LATENT),
        yshift_blob=sections.get(SECTION_YSHIFT),
        lrl1_blob=sections.get(SECTION_LRL1),
        wavelet_blob=sections.get(SECTION_WAVELET),
    )

    # Stage E: roundtrip parse-back (catches builder-vs-inflate drift).
    sd, lat, meta, parsed_sections = parse_stacked_archive(new_bin)
    if set(parsed_sections.keys()) != set(sections.keys()):
        raise RuntimeError(
            f"parse roundtrip section mismatch: built {set(sections.keys())}, "
            f"parsed {set(parsed_sections.keys())}"
        )
    print(f"[build-stacked] parse roundtrip OK: {len(sd)} tensors, "
          f"latents shape={tuple(lat.shape)}, sections={list(parsed_sections.keys())}, "
          f"meta={meta}")

    # Stage F: write archive.zip with deterministic ZIP framing.
    archive_path = args.output_dir / "pr106_stacked_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as z:  # DETERMINISTIC_ZIP_OK
        zi = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, new_bin)
    archive_size = archive_path.stat().st_size
    delta = archive_size - pr106_zip_size
    score_delta_rate_only = 25.0 * delta / 37545489.0
    print(f"[build-stacked] wrote {archive_path}: {archive_size} bytes "
          f"(PR106 zip was {pr106_zip_size}; delta {delta:+d}; "
          f"rate-component score Δ {score_delta_rate_only:+.6f})")

    # Stage G: write build metadata.
    elapsed = time.time() - started
    metadata = {
        "manifest_schema": "pr106_stacked_build_metadata_v2",
        "lane_id": "lane_pr106_stacked",
        "wall_clock_seconds": elapsed,
        "pr106_archive": str(args.pr106_archive),
        "pr106_archive_sha256": sha256_file(args.pr106_archive),
        "pr106_zip_bytes": pr106_zip_size,
        "pr106_bin_bytes": pr106_size_bytes,
        "latent_archive": str(args.latent) if args.latent else None,
        "yshift_archive": str(args.yshift) if args.yshift else None,
        "lrl1_archive": str(args.lrl1) if args.lrl1 else None,
        "wavelet_archive": str(args.wavelet) if args.wavelet else None,
        "input_archives": {
            "pr106": _archive_custody(args.pr106_archive),
            "latent": _archive_custody(args.latent),
            "yshift": _archive_custody(args.yshift),
            "lrl1": _archive_custody(args.lrl1),
            "wavelet": _archive_custody(args.wavelet),
        },
        "section_blob_bytes": {
            "latent": len(sections.get(SECTION_LATENT, b"")) if SECTION_LATENT in sections else None,
            "yshift": len(sections.get(SECTION_YSHIFT, b"")) if SECTION_YSHIFT in sections else None,
            "lrl1": len(sections.get(SECTION_LRL1, b"")) if SECTION_LRL1 in sections else None,
            "wavelet": len(sections.get(SECTION_WAVELET, b"")) if SECTION_WAVELET in sections else None,
        },
        "n_sections": len(sections),
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "archive_zip_bytes": archive_size,
        "output_archive": _archive_custody(archive_path),
        "delta_bytes_vs_pr106_zip": delta,
        "rate_component_score_delta_vs_pr106": score_delta_rate_only,
        "outer_dispatch_magic": f"0x{STACKED_MAGIC_BYTE:02X}",
        "wavelet_runtime_consumption_proof": (
            decoded_wavelet["runtime_consumption_proof"]
            if decoded_wavelet is not None
            else None
        ),
        "wavelet_runtime_mode": (
            decoded_wavelet["runtime_mode"]
            if decoded_wavelet is not None
            else None
        ),
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "compose_time_scaffold_only",
            "requires_sister_sidechannels_to_win_exact_cuda_before_stack_dispatch",
            "requires_archive_manifest_preflight",
            "requires_exact_cuda_auth_eval",
        ] + (
            [
                "wavelet_wr01_runtime_mode_is_explicit_noop",
                "wavelet_wr01_rate_regression_without_distortion_benefit",
                "requires_reviewed_wavelet_apply_transform",
            ]
            if decoded_wavelet is not None
            else []
        ),
        "tag": "[advisory only]",  # composition is byte-side; pixel score still needs CUDA
        "council_status": (
            "PROPOSAL — pre-registered at L1; gated on apogee_intN + 3 sidechannels "
            "all winning empirically (per docs/INDEX_score_aware_sidechannel_thread)"
        ),
        "score_claim": False,
        "next_step": (
            "Compose-time scaffold only. ALL 3 sister sidechannel lanes "
            "(lane_pr106_latent_sidecar + lane_pr106_yshift_sidechannel + "
            "lane_pr106_lrl1_sidechannel + optional WR01 wavelet proof section) "
            "MUST land < 0.20800 [contest-CUDA] "
            "empirically before this stacked archive earns a contest dispatch. "
            "Per tools/sidechannel_stack_predictor.py --bits 5 --all, "
            "int4+full-stack predicted score = 0.163 (-0.046 vs PR106 0.20945)."
        ),
    }
    metadata_path = args.output_dir / "build_metadata.json"
    metadata_path.write_text(json_text(metadata), encoding="utf-8")
    print(f"[build-stacked] wrote {metadata_path}")
    print(f"[build-stacked] DONE in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
