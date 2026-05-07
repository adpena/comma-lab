#!/usr/bin/env python3
"""Repack a PR106-format archive's decoder section using PR101's split-Brotli codec.

Workflow:
  1. Read source archive.zip, extract single-member 0.bin.
  2. Parse PR106's packed-archive layout: 4-byte header (0xFF + u24 dec_len) +
     decoder brotli payload + latents brotli payload.
  3. Decode PR106's decoder section to a torch state_dict.
  4. (Optional) Run :func:`validate_byte_map_savings` Contrarian gate; warn on
     any byte_map that REGRESSES on the source weights.
  5. Re-encode the state_dict via :func:`encode_decoder_compact` (PR101 split-
     Brotli + per-tensor byte maps).
  6. Splice the new decoder section back into the PR106-format payload (header
     bytes are recomputed from the new decoder length).
  7. Write a new single-member archive.zip + manifest.json.

The output is BYTE-FAITHFUL for the latents section (we never touch it) and
schema-compatible with PR101's ``decode_decoder_compact``. The new archive
needs a runtime adapter that recognizes the PR101 decoder format — that is
NOT shipped here (this CLI is a forensic + measurement tool first).

Strict-scorer-rule: this script never loads a scorer. All measurement is
byte-level. Score deltas are tagged ``[predicted]``, never ``[contest-CUDA]``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import zipfile
from pathlib import Path

import torch

# Make the in-tree codec importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    encode_decoder_compact,
    validate_byte_map_savings,
)

# We import PR106's parser directly to ensure we never drift from the upstream
# decode logic.
PR106_SRC_PATH = REPO_ROOT / "experiments" / "results" / (
    "public_pr106_belt_and_suspenders_intake_20260504_codex/source/"
    "submissions/belt_and_suspenders/src"
)
sys.path.insert(0, str(PR106_SRC_PATH.resolve()))

from codec import parse_packed_archive  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


# Score-impact constant: 25 bytes per evaluation frame divided by total bytes.
# Same constant PR106's own audits use.
RATE_CONSTANT_PER_BYTE = 25 / 37_545_489


def _read_archive(path: Path) -> tuple[bytes, str, bytes]:
    """Return (member_name, archive_sha256, payload_bytes) for a single-member zip."""
    archive_bytes = path.read_bytes()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise SystemExit(
                f"expected single-member archive, got {names!r}"
            )
        member_name = names[0]
        payload = z.read(member_name)
    return member_name.encode("utf-8"), archive_sha, payload


def _parse_pr106_packed(payload: bytes) -> tuple[bytes, bytes, bytes]:
    """Return (header_bytes, decoder_brotli, latents_section) from a PR106 packed
    payload. The header is 4 bytes: 0xFF + 3-byte little-endian decoder length."""
    if len(payload) < 4:
        raise SystemExit("payload too short to contain PR106 packed header")
    if payload[0] != 0xFF:
        raise SystemExit(f"unexpected magic byte 0x{payload[0]:02x}, expected 0xFF")
    dec_len = int.from_bytes(payload[1:4], "little")
    if 4 + dec_len > len(payload):
        raise SystemExit("PR106 header decoder length exceeds payload")
    header = payload[:4]
    decoder = payload[4:4 + dec_len]
    latents = payload[4 + dec_len:]
    return header, decoder, latents


def _rebuild_pr106_packed(decoder_brotli: bytes, latents: bytes) -> bytes:
    """Rebuild a PR106 packed payload (0xFF + u24 dec_len + decoder + latents).

    Note: the resulting decoder section will be in PR101 split-brotli format,
    NOT PR106 monolithic-brotli. PR106's parser will FAIL on it; a PR101-aware
    runtime adapter is needed to inflate the result.
    """
    if len(decoder_brotli) >= (1 << 24):
        raise SystemExit(
            f"decoder section too large for u24 length prefix: {len(decoder_brotli)}"
        )
    out = io.BytesIO()
    out.write(b"\xff")
    out.write(len(decoder_brotli).to_bytes(3, "little"))
    out.write(decoder_brotli)
    out.write(latents)
    return out.getvalue()


def _write_single_member_archive(path: Path, member_name: bytes, payload: bytes) -> str:
    """Write a STORED single-member zip; return its sha256."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo(member_name.decode("utf-8"), date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pr101_repacked_archive(
    *,
    source_archive: Path,
    output_dir: Path,
    validate_byte_maps: bool = True,
    brotli_quality: int = 11,
    verbose: bool = True,
) -> dict[str, object]:
    """Repack ``source_archive``'s decoder section via PR101 split-brotli codec.

    Returns the manifest dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_archive = output_dir / "archive.zip"
    manifest_path = output_dir / "manifest.json"

    member_name, source_sha, source_payload = _read_archive(source_archive)
    header, source_decoder, latents = _parse_pr106_packed(source_payload)

    if verbose:
        print(f"[pr101-repack] source archive: {source_archive}")
        print(f"[pr101-repack]   sha256={source_sha[:16]}... bytes={source_archive.stat().st_size}")
        print(f"[pr101-repack]   member={member_name.decode()!r} payload_bytes={len(source_payload)}")
        print(f"[pr101-repack]   PR106 decoder section: {len(source_decoder)} bytes")
        print(f"[pr101-repack]   latents section:        {len(latents)} bytes")

    # Decode the source decoder via PR106's own parser (re-use full payload to
    # avoid re-implementing PR106's header parse).
    state_dict, _latents_tensor, schema = parse_packed_archive(source_payload)
    if verbose:
        n_params = sum(t.numel() for t in state_dict.values())
        print(f"[pr101-repack] decoded state_dict: {len(state_dict)} tensors, {n_params} params")

    # Optional Contrarian gate.
    byte_map_audit: dict[int, dict[str, int]] | None = None
    if validate_byte_maps:
        byte_map_audit = validate_byte_map_savings(state_dict, brotli_quality=brotli_quality)
        if verbose:
            print("[pr101-repack] byte-map savings audit:")
            for idx, info in byte_map_audit.items():
                marker = "WARN" if info["delta_bytes"] > 0 else "OK"
                print(
                    f"  [{marker}] idx={idx} byte_map={info['byte_map']:>6s} "
                    f"with={info['with_map_bytes']:>6d} without={info['without_map_bytes']:>6d} "
                    f"delta={info['delta_bytes']:+d}"
                )

    # Re-encode via PR101 codec.
    new_decoder = encode_decoder_compact(state_dict, brotli_quality=brotli_quality)
    new_payload = _rebuild_pr106_packed(new_decoder, latents)
    new_archive_sha = _write_single_member_archive(output_archive, member_name, new_payload)
    new_archive_size = output_archive.stat().st_size

    decoder_delta = len(new_decoder) - len(source_decoder)
    archive_delta = new_archive_size - source_archive.stat().st_size
    predicted_score_delta = round(archive_delta * RATE_CONSTANT_PER_BYTE, 12)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "tool": "experiments.build_pr101_repacked_archive",
        "score_claim": False,
        "score_evidence_grade": "[predicted]",
        "source_archive_path": str(source_archive),
        "source_archive_sha256": source_sha,
        "source_archive_bytes": source_archive.stat().st_size,
        "source_payload_bytes": len(source_payload),
        "source_member_name": member_name.decode("utf-8"),
        "source_decoder_section_bytes": len(source_decoder),
        "source_latents_section_bytes": len(latents),
        "n_state_dict_tensors": len(state_dict),
        "n_state_dict_params": sum(t.numel() for t in state_dict.values()),
        "schema_meta": schema,
        "brotli_quality": brotli_quality,
        "validate_byte_maps": validate_byte_maps,
        "byte_map_audit": byte_map_audit,
        "output_archive_path": str(output_archive),
        "output_archive_sha256": new_archive_sha,
        "output_archive_bytes": new_archive_size,
        "output_decoder_section_bytes": len(new_decoder),
        "output_latents_section_bytes": len(latents),
        "decoder_delta_bytes": decoder_delta,
        "archive_delta_bytes": archive_delta,
        "predicted_score_delta_tag": "[predicted]",
        "predicted_score_delta_rate_component": predicted_score_delta,
        "runtime_adapter_required": True,
        "runtime_adapter_blockers": [
            "pr101_split_brotli_runtime_adapter_not_yet_integrated",
            "pr106_inflate_will_fail_on_pr101_decoder_format",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(f"[pr101-repack] new decoder section: {len(new_decoder)} bytes")
        print(f"[pr101-repack] decoder delta:       {decoder_delta:+d} bytes")
        print(f"[pr101-repack] archive delta:       {archive_delta:+d} bytes")
        print(
            f"[pr101-repack] predicted score Δ:    {predicted_score_delta:+.6f} "
            "[predicted] (rate component only)"
        )
        print(f"[pr101-repack] wrote {output_archive} (sha256={new_archive_sha[:16]}...)")
        print(f"[pr101-repack] wrote {manifest_path}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Input archive.zip (PR106-format packed payload)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write archive.zip + manifest.json",
    )
    parser.add_argument(
        "--brotli-quality",
        type=int,
        default=11,
        help="Brotli compression level (PR101 ships at 11)",
    )
    parser.add_argument(
        "--validate-byte-maps",
        dest="validate_byte_maps",
        action="store_true",
        default=True,
        help="Run the Contrarian gate audit (default ON)",
    )
    parser.add_argument(
        "--no-validate-byte-maps",
        dest="validate_byte_maps",
        action="store_false",
        help="Skip the Contrarian byte-map audit",
    )
    args = parser.parse_args()

    if not args.source_archive.is_file():
        print(f"ERROR: --source-archive not found: {args.source_archive}", file=sys.stderr)
        return 2

    build_pr101_repacked_archive(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        validate_byte_maps=args.validate_byte_maps,
        brotli_quality=args.brotli_quality,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
