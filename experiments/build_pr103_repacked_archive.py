#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Repack a PR106-format archive's decoder section using PR103's arithmetic codec.

Workflow:
  1. Read source archive.zip, extract single-member 0.bin.
  2. Parse PR106's packed-archive layout: 4-byte header (0xFF + u24 dec_len) +
     decoder brotli payload + latents brotli payload.
  3. Decode PR106's decoder section to a torch state_dict.
  4. (Optional) ``--compose-with-pr101``: first re-encode via PR101 split-Brotli
     codec (Op 1) to capture bit-identical Op-1-decoded weights, then proceed
     with PR103 AC encoding (Op 2). Otherwise: skip Op 1 and encode the
     PR106-decoded state_dict directly via PR103 AC.
  5. (Optional) ``--validate-ac``: Contrarian gate — measure per-tensor AC vs
     brotli on the source weights; log warnings on regressions.
  6. Re-encode via :func:`encode_decoder_ac` (PR103 AC + adaptive lgwin +
     merged RangeEncoder).
  7. Splice the new decoder section back into the PR106-format payload (header
     bytes are recomputed from the new decoder length).
  8. Write a new single-member archive.zip + manifest.json.

The output is BYTE-FAITHFUL for the latents section (we never touch it). The
new archive needs a runtime adapter that recognizes the PR103 decoder
format — that is NOT shipped here (this CLI is a forensic + measurement tool
first, mirroring ``build_pr101_repacked_archive.py``).

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

# Make the in-tree codecs importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    decode_decoder_compact,
    encode_decoder_compact,
)
from tac.pr103_arithmetic_codec import (  # noqa: E402
    encode_decoder_ac,
    validate_ac_savings,
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
# Same constant PR101's audit uses.
RATE_CONSTANT_PER_BYTE = 25 / 37_545_489


def _read_archive(path: Path) -> tuple[bytes, str, bytes]:
    """Return (member_name_bytes, archive_sha256, payload_bytes) for a single-member zip."""
    archive_bytes = path.read_bytes()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise SystemExit(f"expected single-member archive, got {names!r}")
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


def _rebuild_pr106_packed(decoder_blob: bytes, latents: bytes) -> bytes:
    """Rebuild a PR106 packed payload (0xFF + u24 dec_len + decoder + latents).

    Note: the resulting decoder section is the PR103-AC blob, NOT PR106
    monolithic-brotli or PR101 split-brotli. PR106's parser will FAIL on it;
    a PR103-aware runtime adapter is needed to inflate the result.
    """
    if len(decoder_blob) >= (1 << 24):
        raise SystemExit(
            f"decoder section too large for u24 length prefix: {len(decoder_blob)}"
        )
    out = io.BytesIO()
    out.write(b"\xff")
    out.write(len(decoder_blob).to_bytes(3, "little"))
    out.write(decoder_blob)
    out.write(latents)
    return out.getvalue()


def _write_single_member_archive(path: Path, member_name: bytes, payload: bytes) -> str:
    """Write a STORED single-member zip; return its sha256."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo(member_name.decode("utf-8"), date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pr103_repacked_archive(
    *,
    source_archive: Path,
    output_dir: Path,
    validate_ac: bool = True,
    compose_with_pr101: bool = False,
    brotli_quality: int = 11,
    adaptive_lgwin: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """Repack ``source_archive``'s decoder section via PR103 AC codec.

    Returns the manifest dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_archive = output_dir / "archive.zip"
    manifest_path = output_dir / "manifest.json"

    member_name, source_sha, source_payload = _read_archive(source_archive)
    _header, source_decoder, latents = _parse_pr106_packed(source_payload)

    if verbose:
        print(f"[pr103-repack] source archive: {source_archive}")
        print(
            f"[pr103-repack]   sha256={source_sha[:16]}... "
            f"bytes={source_archive.stat().st_size}"
        )
        print(
            f"[pr103-repack]   member={member_name.decode()!r} "
            f"payload_bytes={len(source_payload)}"
        )
        print(f"[pr103-repack]   PR106 decoder section: {len(source_decoder)} bytes")
        print(f"[pr103-repack]   latents section:        {len(latents)} bytes")

    # Decode the source decoder via PR106's own parser.
    state_dict, _latents_tensor, schema = parse_packed_archive(source_payload)
    if verbose:
        n_params = sum(t.numel() for t in state_dict.values())
        print(
            f"[pr103-repack] decoded state_dict: {len(state_dict)} tensors, "
            f"{n_params} params"
        )

    # Optional Op 1 composition: encode → decode via PR101 to capture
    # bit-identical Op-1-decoded weights before applying Op 2.
    op1_decoder_bytes: int | None = None
    if compose_with_pr101:
        op1_blob = encode_decoder_compact(state_dict, brotli_quality=brotli_quality)
        op1_decoder_bytes = len(op1_blob)
        state_dict_for_ac = decode_decoder_compact(op1_blob)
        if verbose:
            print(
                f"[pr103-repack] Op 1 (PR101 split-Brotli): "
                f"{op1_decoder_bytes} bytes (will compose Op 2 on top)"
            )
    else:
        state_dict_for_ac = state_dict

    # Optional Contrarian gate.
    ac_audit: dict[int, dict[str, int]] | None = None
    if validate_ac:
        ac_audit = validate_ac_savings(state_dict_for_ac, brotli_quality=brotli_quality)
        if verbose:
            print("[pr103-repack] AC vs brotli per-tensor audit:")
            for idx, info in ac_audit.items():
                marker = "WARN" if info["delta_bytes"] > 0 else "OK"
                print(
                    f"  [{marker}] idx={idx:2d} ac={info['ac_bytes']:>6d} "
                    f"brotli={info['brotli_bytes']:>6d} "
                    f"delta={info['delta_bytes']:+d} "
                    f"n_symbols={info['n_symbols']:>6d}"
                )
            ac_total = sum(info["ac_bytes"] for info in ac_audit.values())
            br_total = sum(info["brotli_bytes"] for info in ac_audit.values())
            print(
                f"[pr103-repack]   per-tensor AC total: {ac_total} bytes "
                f"vs brotli total: {br_total} bytes "
                f"(delta {ac_total - br_total:+d})"
            )

    # Op 2: re-encode via PR103 AC codec.
    new_decoder = encode_decoder_ac(
        state_dict_for_ac,
        brotli_quality=brotli_quality,
        adaptive_lgwin=adaptive_lgwin,
    )
    new_payload = _rebuild_pr106_packed(new_decoder, latents)
    new_archive_sha = _write_single_member_archive(output_archive, member_name, new_payload)
    new_archive_size = output_archive.stat().st_size

    decoder_delta = len(new_decoder) - len(source_decoder)
    archive_delta = new_archive_size - source_archive.stat().st_size
    predicted_score_delta = round(archive_delta * RATE_CONSTANT_PER_BYTE, 12)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "tool": "experiments.build_pr103_repacked_archive",
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
        "adaptive_lgwin": adaptive_lgwin,
        "compose_with_pr101": compose_with_pr101,
        "op1_decoder_bytes": op1_decoder_bytes,
        "validate_ac": validate_ac,
        "ac_audit": ac_audit,
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
            "pr103_arithmetic_codec_runtime_adapter_not_yet_integrated",
            "pr106_inflate_will_fail_on_pr103_decoder_format",
            (
                "pr103_decoder_section_has_no_section_length_prefixes_so_runtime"
                "_must_carry_hardcoded_lengths_or_a_small_header"
            ),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(f"[pr103-repack] new decoder section: {len(new_decoder)} bytes")
        print(f"[pr103-repack] decoder delta:       {decoder_delta:+d} bytes")
        print(f"[pr103-repack] archive delta:       {archive_delta:+d} bytes")
        if op1_decoder_bytes is not None:
            op2_vs_op1 = len(new_decoder) - op1_decoder_bytes
            print(
                f"[pr103-repack] Op 2 vs Op 1 delta:  {op2_vs_op1:+d} bytes "
                "(negative = Op 2 wins)"
            )
        print(
            f"[pr103-repack] predicted score Δ:    {predicted_score_delta:+.6f} "
            "[predicted] (rate component only, vs source)"
        )
        print(f"[pr103-repack] wrote {output_archive} (sha256={new_archive_sha[:16]}...)")
        print(f"[pr103-repack] wrote {manifest_path}")

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
        help="Brotli compression level (PR103 ships at 11)",
    )
    parser.add_argument(
        "--validate-ac",
        dest="validate_ac",
        action="store_true",
        default=True,
        help="Run the Contrarian per-tensor AC audit (default ON)",
    )
    parser.add_argument(
        "--no-validate-ac",
        dest="validate_ac",
        action="store_false",
        help="Skip the Contrarian per-tensor AC audit",
    )
    parser.add_argument(
        "--compose-with-pr101",
        action="store_true",
        help=(
            "Run Op 1 (PR101 split-Brotli) first, then Op 2 (PR103 AC) on the "
            "Op-1-decoded weights. Default OFF (encodes PR106-decoded weights "
            "directly via Op 2)."
        ),
    )
    parser.add_argument(
        "--no-adaptive-lgwin",
        dest="adaptive_lgwin",
        action="store_false",
        default=True,
        help=(
            "Skip per-stream adaptive lgwin search. Default: search enabled. "
            "Useful for fast smoke tests; loses ~100B savings."
        ),
    )
    args = parser.parse_args()

    if not args.source_archive.is_file():
        print(f"ERROR: --source-archive not found: {args.source_archive}", file=sys.stderr)
        return 2

    build_pr103_repacked_archive(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        validate_ac=args.validate_ac,
        compose_with_pr101=args.compose_with_pr101,
        brotli_quality=args.brotli_quality,
        adaptive_lgwin=args.adaptive_lgwin,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
