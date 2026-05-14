#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR101 archive-substitution surgery: replace the decoder_blob in a PR101
archive.zip with a re-encoded blob from the cathedral's CodecOp wrapper.

This tool closes the actuator gap from the PR106 pose-axis forensic memo:
the cathedral has CodecOps that produce wire-format-compatible blobs, but
to actually dispatch a substituted archive to contest-CUDA we need
byte-level substrate surgery on the existing archive.

The CLI intentionally exposes decoder substitution only. The module also
contains latent/sidecar section splicers for forensic and future actuator work,
but those still require PR101-grammar proof before dispatch use.

Critical scope finding (2026-05-07): PR101 has **no separate pose stream**.
The HNeRV decoder generates pose-aware frames implicitly from latents
(see ``inflate.py:42`` — ``decoder(latents[i:j])``). Therefore:

  - **Op1_PR101SplitBrotli** is wire-format-compatible: same encode/decode
    schema as PR101's native ``decode_decoder_compact``. CAN substitute.
  - **Op_KLPoseStream** targets a stream PR101 doesn't ship. CANNOT
    substitute on PR101 (would need a different substrate like PR100/PR106
    that ships poses_se3 separately).
  - **Op_KLLatent** is built for cathedral/raw-latent substrates, but PR101's
    native latent stream uses LZMA + per-dim min/scale + uint8 codes. The
    cathedral KL pattern is therefore incompatible with stock PR101 runtime
    substitution. CANNOT substitute without a forked inflate.

PR101 archive byte layout (from
``submissions/hnerv_ft_microcodec/src/codec.py:467``):

  archive.zip[member="x"] = decoder_blob[0 : 162_164]
                          + latent_blob[162_164 : 177_551]
                          + sidecar_blob[177_551 : 178_158]

This tool's substitution is byte-slice splicing on the inner blob, then
re-zipping with the same single member name ``x`` (uncompressed, since
the inner blob is already entropy-coded).

Usage::

    .venv/bin/python tools/pr101_archive_substitution_surgery.py \\
        --input-archive experiments/.../public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \\
        --replacement-decoder-blob /tmp/op1_blob.bin \\
        --output-archive experiments/results/<lane>/substituted_archive.zip

CLAUDE.md compliance: strict-scorer-rule (pure CPU + zipfile + bytes); no
/tmp paths persisted in evidence (the example uses /tmp only for an
intermediate blob the operator passes in); no MPS/CUDA. Outputs to
``experiments/results/<lane>/`` per the canonical durable-state convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

# PR101 layout constants (kept independent of upstream source — verified
# against ``hnerv_ft_microcodec/src/codec.py:24-26`` 2026-05-07).
PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_TOTAL_INNER_BYTES_NOMINAL = 178_158  # decoder + latent + 607-byte sidecar
PR101_INNER_MEMBER_NAME = "x"


@dataclass
class SubstitutionReport:
    input_archive: str
    output_archive: str
    input_size_bytes: int
    output_size_bytes: int
    decoder_blob_offset: int
    decoder_blob_input_len: int
    decoder_blob_replacement_len: int
    latent_blob_offset: int
    latent_blob_len: int
    sidecar_blob_offset: int
    sidecar_blob_len: int
    inner_member_name: str
    inner_member_input_size: int
    inner_member_output_size: int
    sha256_input_archive: str
    sha256_output_archive: str
    sha256_input_inner_member: str
    sha256_output_inner_member: str
    sha256_input_decoder_blob: str
    sha256_replacement_decoder_blob: str
    sha256_input_latent_blob: str
    sha256_output_latent_blob: str
    sha256_input_sidecar_blob: str
    sha256_output_sidecar_blob: str
    bytes_delta: int
    notes: list[str]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_inner_blob(archive_path: Path) -> bytes:
    """Extract the single inner blob (member 'x') from a PR101 archive.

    Raises ValueError if the archive doesn't have the expected layout.
    """
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if names != [PR101_INNER_MEMBER_NAME]:
            raise ValueError(
                f"archive {archive_path} has members {names!r}; expected "
                f"['{PR101_INNER_MEMBER_NAME}'] (PR101 layout)"
            )
        with zf.open(PR101_INNER_MEMBER_NAME) as fp:
            return fp.read()


def _split_pr101_inner_blob(blob: bytes) -> tuple[bytes, bytes, bytes]:
    """Split the PR101 inner blob into (decoder, latent, sidecar) bytes.

    Validates the blob is at least DECODER_BLOB_LEN + LATENT_BLOB_LEN bytes;
    sidecar is whatever's left (PR101 default is 607 bytes but it varies).
    """
    if len(blob) < PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN:
        raise ValueError(
            f"inner blob length {len(blob)} < required minimum "
            f"{PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN} for PR101 layout"
        )
    decoder_blob = blob[:PR101_DECODER_BLOB_LEN]
    latent_blob = blob[PR101_DECODER_BLOB_LEN:PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN]
    sidecar_blob = blob[PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN:]
    return decoder_blob, latent_blob, sidecar_blob


def _write_pr101_archive(
    inner_blob: bytes, archive_path: Path
) -> None:
    """Write a PR101-shaped archive: a zip with single member 'x' holding
    the inner blob. Stored uncompressed (the inner blob is already
    entropy-coded; ZIP_DEFLATE on top adds a few bytes of header overhead
    with no compression gain).

    Uses a fixed timestamp (1980-01-01 00:00:00) for determinism; PR101's
    own archives don't strip ZIP timestamps but cathedral discipline is
    deterministic-bytes for byte-faithful reproducibility.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename=PR101_INNER_MEMBER_NAME)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(info, inner_blob)


def substitute_decoder_blob(
    *,
    input_archive: Path,
    replacement_decoder_blob: bytes,
    output_archive: Path,
) -> SubstitutionReport:
    """Substitute the decoder_blob section in a PR101 archive.

    Splices the inner blob: keeps the latent + sidecar sections intact,
    replaces the decoder section with the supplied bytes (which need not
    match the original length; PR101's parse_archive uses fixed offsets,
    so a non-DECODER_BLOB_LEN replacement WILL break inflate unless the
    replacement is exactly DECODER_BLOB_LEN bytes OR the operator also
    forks the inflate to use a different layout).

    Raises ValueError if the replacement length doesn't match
    DECODER_BLOB_LEN — defensive default since silent length-mismatch
    would corrupt downstream parsing.
    """
    notes: list[str] = []
    if len(replacement_decoder_blob) != PR101_DECODER_BLOB_LEN:
        raise ValueError(
            f"replacement_decoder_blob length {len(replacement_decoder_blob)} "
            f"!= DECODER_BLOB_LEN {PR101_DECODER_BLOB_LEN}; PR101's inflate "
            f"uses fixed offsets and would corrupt latent_blob extraction"
        )

    input_blob = _read_inner_blob(input_archive)
    decoder_blob, latent_blob, sidecar_blob = _split_pr101_inner_blob(input_blob)
    new_inner = replacement_decoder_blob + latent_blob + sidecar_blob
    _write_pr101_archive(new_inner, output_archive)

    input_size = input_archive.stat().st_size
    output_size = output_archive.stat().st_size
    return SubstitutionReport(
        input_archive=str(input_archive),
        output_archive=str(output_archive),
        input_size_bytes=input_size,
        output_size_bytes=output_size,
        decoder_blob_offset=0,
        decoder_blob_input_len=len(decoder_blob),
        decoder_blob_replacement_len=len(replacement_decoder_blob),
        latent_blob_offset=PR101_DECODER_BLOB_LEN,
        latent_blob_len=len(latent_blob),
        sidecar_blob_offset=PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN,
        sidecar_blob_len=len(sidecar_blob),
        inner_member_name=PR101_INNER_MEMBER_NAME,
        inner_member_input_size=len(input_blob),
        inner_member_output_size=len(new_inner),
        sha256_input_archive=_sha256(input_archive.read_bytes()),
        sha256_output_archive=_sha256(output_archive.read_bytes()),
        sha256_input_inner_member=_sha256(input_blob),
        sha256_output_inner_member=_sha256(new_inner),
        sha256_input_decoder_blob=_sha256(decoder_blob),
        sha256_replacement_decoder_blob=_sha256(replacement_decoder_blob),
        sha256_input_latent_blob=_sha256(latent_blob),
        sha256_output_latent_blob=_sha256(latent_blob),
        sha256_input_sidecar_blob=_sha256(sidecar_blob),
        sha256_output_sidecar_blob=_sha256(sidecar_blob),
        bytes_delta=output_size - input_size,
        notes=notes,
    )


def substitute_latent_blob(
    *,
    input_archive: Path,
    replacement_latent_blob: bytes,
    output_archive: Path,
) -> SubstitutionReport:
    """Substitute the latent_blob section in a PR101 archive.

    Same defensive contract as substitute_decoder_blob: PR101's
    parse_archive uses fixed offsets, so the replacement length MUST
    equal LATENT_BLOB_LEN (15,387 bytes) — anything else corrupts
    sidecar_blob extraction. Decoder + sidecar sections are preserved
    byte-faithfully.

    Caveat: PR101's decode_latents_compact expects an LZMA stream over
    a specific (per-dim min/scale + temporal-delta uint8) format. Any
    replacement that doesn't satisfy that structure will fail at
    parse-time inside PR101's inflate. The defensive guard here is
    length only; structural correctness is the caller's responsibility.
    """
    if len(replacement_latent_blob) != PR101_LATENT_BLOB_LEN:
        raise ValueError(
            f"replacement_latent_blob length {len(replacement_latent_blob)} "
            f"!= LATENT_BLOB_LEN {PR101_LATENT_BLOB_LEN}; PR101's inflate "
            f"uses fixed offsets and would corrupt sidecar_blob extraction"
        )
    input_blob = _read_inner_blob(input_archive)
    decoder_blob, latent_blob, sidecar_blob = _split_pr101_inner_blob(input_blob)
    new_inner = decoder_blob + replacement_latent_blob + sidecar_blob
    _write_pr101_archive(new_inner, output_archive)
    input_size = input_archive.stat().st_size
    output_size = output_archive.stat().st_size
    return SubstitutionReport(
        input_archive=str(input_archive),
        output_archive=str(output_archive),
        input_size_bytes=input_size,
        output_size_bytes=output_size,
        decoder_blob_offset=0,
        decoder_blob_input_len=len(decoder_blob),
        decoder_blob_replacement_len=len(decoder_blob),
        latent_blob_offset=PR101_DECODER_BLOB_LEN,
        latent_blob_len=len(replacement_latent_blob),
        sidecar_blob_offset=PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN,
        sidecar_blob_len=len(sidecar_blob),
        inner_member_name=PR101_INNER_MEMBER_NAME,
        inner_member_input_size=len(input_blob),
        inner_member_output_size=len(new_inner),
        sha256_input_archive=_sha256(input_archive.read_bytes()),
        sha256_output_archive=_sha256(output_archive.read_bytes()),
        sha256_input_inner_member=_sha256(input_blob),
        sha256_output_inner_member=_sha256(new_inner),
        sha256_input_decoder_blob=_sha256(decoder_blob),
        sha256_replacement_decoder_blob=_sha256(decoder_blob),
        sha256_input_latent_blob=_sha256(latent_blob),
        sha256_output_latent_blob=_sha256(replacement_latent_blob),
        sha256_input_sidecar_blob=_sha256(sidecar_blob),
        sha256_output_sidecar_blob=_sha256(sidecar_blob),
        bytes_delta=output_size - input_size,
        notes=[
            "latent_blob substituted; decoder + sidecar sections preserved.",
            f"input latent sha256={_sha256(latent_blob)}; "
            f"replacement latent sha256={_sha256(replacement_latent_blob)}",
        ],
    )


def substitute_sidecar_blob(
    *,
    input_archive: Path,
    replacement_sidecar_blob: bytes,
    output_archive: Path,
) -> SubstitutionReport:
    """Substitute the sidecar_blob section in a PR101 archive.

    The sidecar is the LAST section, so its length is NOT fixed by
    parse_archive's offset arithmetic — any non-empty replacement is
    valid (PR101 reads ``archive_bytes[DECODER + LATENT:]`` as a tail
    slice). Decoder + latent are preserved byte-faithfully.

    Caveat: PR101's apply_latent_sidecar interprets these bytes as a
    Brotli stream of (dim, delta_x100) pairs with prefix metadata.
    Replacements that don't satisfy that grammar will fail at parse-time.
    The defensive guard here is "non-empty"; structural correctness is
    the caller's responsibility.
    """
    if not replacement_sidecar_blob:
        raise ValueError(
            "replacement_sidecar_blob is empty; PR101's apply_latent_sidecar "
            "requires at least the structural prefix"
        )
    input_blob = _read_inner_blob(input_archive)
    decoder_blob, latent_blob, sidecar_blob = _split_pr101_inner_blob(input_blob)
    new_inner = decoder_blob + latent_blob + replacement_sidecar_blob
    _write_pr101_archive(new_inner, output_archive)
    input_size = input_archive.stat().st_size
    output_size = output_archive.stat().st_size
    return SubstitutionReport(
        input_archive=str(input_archive),
        output_archive=str(output_archive),
        input_size_bytes=input_size,
        output_size_bytes=output_size,
        decoder_blob_offset=0,
        decoder_blob_input_len=len(decoder_blob),
        decoder_blob_replacement_len=len(decoder_blob),
        latent_blob_offset=PR101_DECODER_BLOB_LEN,
        latent_blob_len=len(latent_blob),
        sidecar_blob_offset=PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN,
        sidecar_blob_len=len(replacement_sidecar_blob),
        inner_member_name=PR101_INNER_MEMBER_NAME,
        inner_member_input_size=len(input_blob),
        inner_member_output_size=len(new_inner),
        sha256_input_archive=_sha256(input_archive.read_bytes()),
        sha256_output_archive=_sha256(output_archive.read_bytes()),
        sha256_input_inner_member=_sha256(input_blob),
        sha256_output_inner_member=_sha256(new_inner),
        sha256_input_decoder_blob=_sha256(decoder_blob),
        sha256_replacement_decoder_blob=_sha256(decoder_blob),
        sha256_input_latent_blob=_sha256(latent_blob),
        sha256_output_latent_blob=_sha256(latent_blob),
        sha256_input_sidecar_blob=_sha256(sidecar_blob),
        sha256_output_sidecar_blob=_sha256(replacement_sidecar_blob),
        bytes_delta=output_size - input_size,
        notes=[
            "sidecar_blob substituted; decoder + latent sections preserved.",
            f"input sidecar sha256={_sha256(sidecar_blob)} ({len(sidecar_blob)} B); "
            f"replacement sidecar sha256={_sha256(replacement_sidecar_blob)} "
            f"({len(replacement_sidecar_blob)} B)",
            f"sidecar_size_delta = {len(replacement_sidecar_blob) - len(sidecar_blob):+}",
        ],
    )


def verify_byte_layout(
    archive_path: Path,
) -> dict[str, object]:
    """Diagnostic: load a PR101 archive and report its layout.

    Useful for sanity-checking before substitution + for ad-hoc forensics
    on operator-supplied archives.
    """
    blob = _read_inner_blob(archive_path)
    decoder_blob, latent_blob, sidecar_blob = _split_pr101_inner_blob(blob)
    return {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "inner_blob_size": len(blob),
        "decoder_blob": {
            "offset": 0,
            "len": len(decoder_blob),
            "expected_len": PR101_DECODER_BLOB_LEN,
            "matches_expected": len(decoder_blob) == PR101_DECODER_BLOB_LEN,
            "sha256": _sha256(decoder_blob),
        },
        "latent_blob": {
            "offset": PR101_DECODER_BLOB_LEN,
            "len": len(latent_blob),
            "expected_len": PR101_LATENT_BLOB_LEN,
            "matches_expected": len(latent_blob) == PR101_LATENT_BLOB_LEN,
            "sha256": _sha256(latent_blob),
        },
        "sidecar_blob": {
            "offset": PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN,
            "len": len(sidecar_blob),
            "sha256": _sha256(sidecar_blob),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="Inspect PR101 archive layout")
    p_verify.add_argument("--archive", type=Path, required=True)

    p_subst = sub.add_parser(
        "substitute",
        help="Replace decoder_blob with a re-encoded blob",
    )
    p_subst.add_argument("--input-archive", type=Path, required=True)
    p_subst.add_argument("--replacement-decoder-blob", type=Path, required=True,
                         help="Path to a .bin file with the new decoder_blob bytes")
    p_subst.add_argument("--output-archive", type=Path, required=True)
    p_subst.add_argument("--report", type=Path, default=None,
                         help="Optional JSON path to write the SubstitutionReport")

    args = parser.parse_args(argv)

    if args.cmd == "verify":
        layout = verify_byte_layout(args.archive)
        print(json.dumps(layout, indent=2, sort_keys=True))
        return 0

    if args.cmd == "substitute":
        replacement_bytes = args.replacement_decoder_blob.read_bytes()
        report = substitute_decoder_blob(
            input_archive=args.input_archive,
            replacement_decoder_blob=replacement_bytes,
            output_archive=args.output_archive,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
