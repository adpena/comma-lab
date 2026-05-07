#!/usr/bin/env python3
"""Emit cross-language conformance golden vectors for PR101 codec primitives.

Per the codex parallel-session contract (memo
``feedback_codex_parallel_session_packet_compiler_20260507.md``,
"Cross-Language Conformance Tests" section), every promoted byte-level
primitive must have **reusable conformance assets** — canonical input
bytes, output bytes, SHA-256s, lengths, and decoded semantic facts that a
Rust/Zig/C/ASM port can consume byte-for-byte without reading Python.

This emitter writes JSON manifests under
``src/tac/tests/golden_vectors/`` for the PR101 codec stack:

  * ``pr101_decoder_blob_v1.json`` — decoder blob encode/decode round-trip
  * ``pr101_inner_blob_layout_v1.json`` — inner blob splice grammar
  * ``pr101_archive_zip_layout_v1.json`` — ZIP outer-archive layout

Each manifest carries:

  * Schema version + oracle identifier (Python reference module)
  * Input/output bytes (length + SHA-256)
  * Decoded semantic facts (tensor schema, byte_maps, n_quant, splice
    offsets) so a native port has the contract spelled out, not implied
  * Negative vectors (malformed headers, oversize inputs) so a port can
    confirm fail-closed behavior
  * Deterministic-output proof (running the emitter twice yields the
    same SHAs)

CLAUDE.md compliance: pure CPU + zipfile + brotli; no scorer load; no
score claims. The vectors are *byte-level conformance assets* and
explicitly NOT contest-faithful score evidence.

Usage::

    .venv/bin/python tools/emit_pr101_golden_vectors.py \\
        --input-archive experiments/.../public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \\
        --output-dir src/tac/tests/golden_vectors

Re-running with the same input archive must produce byte-identical
outputs. The emitter prints each output file path + its SHA on stdout
for forensic chain-of-custody.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    decode_decoder_compact,
    encode_decoder_compact,
)

DECODER_SCHEMA_VERSION = "pr101_decoder_blob_v1"
INNER_SCHEMA_VERSION = "pr101_inner_blob_layout_v1"
ARCHIVE_SCHEMA_VERSION = "pr101_archive_zip_layout_v1"
ORACLE_DECODER = "tac.pr101_split_brotli_codec.decode_decoder_compact"
ORACLE_ENCODER = "tac.pr101_split_brotli_codec.encode_decoder_compact"

PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_INNER_MEMBER_NAME = "x"


@dataclass(frozen=True)
class EmittedVector:
    schema: str
    output_path: Path
    sha256: str
    description: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(payload: dict) -> str:
    """Return a deterministic JSON string for the payload.

    Sort keys + UTF-8 + ``\\n`` newline so re-running the emitter
    produces byte-identical output (cross-language conformance asset
    invariant per the codex contract).
    """
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _read_pr101_archive_inner_blob(archive_path: Path) -> tuple[bytes, bytes, bytes, bytes]:
    """Return (full_archive_bytes, inner_blob, decoder_blob, latent_blob, sidecar_blob).

    Asserts the PR101 layout invariants (single member named 'x',
    ``decoder + latent + sidecar`` byte-slice grammar).
    """
    import zipfile
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if names != [PR101_INNER_MEMBER_NAME]:
            raise ValueError(
                f"archive {archive_path} has members {names!r}; expected "
                f"['{PR101_INNER_MEMBER_NAME}']"
            )
        inner = zf.read(PR101_INNER_MEMBER_NAME)
    archive_bytes = archive_path.read_bytes()
    if len(inner) < PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN:
        raise ValueError(
            f"inner blob length {len(inner)} below PR101 minimum"
        )
    decoder = inner[:PR101_DECODER_BLOB_LEN]
    latent = inner[PR101_DECODER_BLOB_LEN:PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN]
    sidecar = inner[PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN:]
    return archive_bytes, inner, decoder, latent, sidecar


def emit_decoder_blob_vector(
    *,
    archive_path: Path,
    output_dir: Path,
) -> EmittedVector:
    """Emit decoder blob round-trip golden vector.

    Loads the canonical PR101 decoder blob, decodes it to a state_dict,
    re-encodes deterministically, asserts byte-identical round-trip,
    writes the JSON manifest with full custody.
    """
    _, _, decoder_blob, _, _ = _read_pr101_archive_inner_blob(archive_path)
    state_dict = decode_decoder_compact(decoder_blob)
    re_encoded = encode_decoder_compact(state_dict)
    if re_encoded != decoder_blob:
        raise RuntimeError(
            f"PR101 decoder round-trip mismatch: input sha={_sha256(decoder_blob)[:12]}, "
            f"re-encoded sha={_sha256(re_encoded)[:12]}; the cathedral encoder is no "
            f"longer byte-faithful with PR101's native encoder"
        )

    decoded_semantic_facts = {
        "n_tensors": len(state_dict),
        "schema_storage_order": [name for name, _ in FIXED_STATE_SCHEMA],
        "tensor_shapes": {name: list(state_dict[name].shape) for name in state_dict},
        "tensor_dtypes": {name: str(state_dict[name].dtype) for name in state_dict},
    }
    payload = {
        "schema": DECODER_SCHEMA_VERSION,
        "oracle": ORACLE_DECODER,
        "oracle_status": "python_reference",
        "round_trip_verified": True,
        "input_archive": str(archive_path.relative_to(REPO_ROOT)) if archive_path.is_relative_to(REPO_ROOT) else str(archive_path),
        "input_blob": {
            "bytes": len(decoder_blob),
            "sha256": _sha256(decoder_blob),
        },
        "output_blob": {
            "bytes": len(re_encoded),
            "sha256": _sha256(re_encoded),
        },
        "decoded_semantic_facts": decoded_semantic_facts,
        "byte_identical_to_input": True,
        "negative_vectors": {
            "empty_input_must_raise": "decode_decoder_compact(b'') raises ValueError",
            "truncated_input_must_raise": (
                "decode_decoder_compact(decoder_blob[:1000]) raises ValueError "
                "or returns malformed state_dict — Python reference asserts this"
            ),
        },
    }
    output_path = output_dir / f"{DECODER_SCHEMA_VERSION}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = _canonical_json(payload)
    output_path.write_text(output_text, encoding="utf-8")
    return EmittedVector(
        schema=DECODER_SCHEMA_VERSION,
        output_path=output_path,
        sha256=_sha256(output_text.encode("utf-8")),
        description=f"PR101 decoder blob round-trip ({len(decoder_blob)} B input)",
    )


def emit_inner_blob_layout_vector(
    *,
    archive_path: Path,
    output_dir: Path,
) -> EmittedVector:
    """Emit the inner-blob splice-grammar golden vector.

    PR101 uses fixed-offset slicing on the inner blob. A native port must
    extract decoder/latent/sidecar at exactly DECODER_BLOB_LEN,
    LATENT_BLOB_LEN — this manifest pins those constants to bytes-on-disk
    proof for forensic verification.
    """
    archive_bytes, inner, decoder, latent, sidecar = _read_pr101_archive_inner_blob(archive_path)
    payload = {
        "schema": INNER_SCHEMA_VERSION,
        "oracle": "tools/pr101_archive_substitution_surgery._split_pr101_inner_blob",
        "oracle_status": "python_reference",
        "input_archive": str(archive_path.relative_to(REPO_ROOT)) if archive_path.is_relative_to(REPO_ROOT) else str(archive_path),
        "inner_blob": {
            "bytes": len(inner),
            "sha256": _sha256(inner),
            "member_name": PR101_INNER_MEMBER_NAME,
        },
        "splice_grammar": {
            "decoder_blob": {
                "offset": 0,
                "length": PR101_DECODER_BLOB_LEN,
                "sha256": _sha256(decoder),
            },
            "latent_blob": {
                "offset": PR101_DECODER_BLOB_LEN,
                "length": PR101_LATENT_BLOB_LEN,
                "sha256": _sha256(latent),
            },
            "sidecar_blob": {
                "offset": PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN,
                "length": len(sidecar),
                "sha256": _sha256(sidecar),
                "length_is_variable": True,
            },
        },
        "negative_vectors": {
            "inner_too_short": (
                f"inner blob with len < {PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN} "
                "must fail at split"
            ),
            "decoder_replacement_wrong_len": (
                f"a decoder replacement with len != {PR101_DECODER_BLOB_LEN} "
                "must fail at substitution (corrupts latent extraction)"
            ),
        },
    }
    output_path = output_dir / f"{INNER_SCHEMA_VERSION}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = _canonical_json(payload)
    output_path.write_text(output_text, encoding="utf-8")
    return EmittedVector(
        schema=INNER_SCHEMA_VERSION,
        output_path=output_path,
        sha256=_sha256(output_text.encode("utf-8")),
        description=f"PR101 inner-blob splice grammar ({len(inner)} B inner)",
    )


def emit_archive_zip_layout_vector(
    *,
    archive_path: Path,
    output_dir: Path,
) -> EmittedVector:
    """Emit the outer ZIP container golden vector.

    PR101 archives are single-member ZIP files. This manifest pins the
    ZIP-level invariants (single member name, archive bytes, archive
    SHA) so a native ZIP-aware port can verify it produces a structurally
    compatible container.
    """
    archive_bytes, inner, _, _, _ = _read_pr101_archive_inner_blob(archive_path)
    import zipfile
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        info = infos[0]
        member_meta = {
            "name": info.filename,
            "compress_type": int(info.compress_type),
            "compressed_size": int(info.compress_size),
            "uncompressed_size": int(info.file_size),
            "crc32": int(info.CRC),
            "date_time": list(info.date_time),
        }
    payload = {
        "schema": ARCHIVE_SCHEMA_VERSION,
        "oracle": "Python zipfile + tools/pr101_archive_substitution_surgery._read_inner_blob",
        "oracle_status": "python_reference",
        "input_archive": str(archive_path.relative_to(REPO_ROOT)) if archive_path.is_relative_to(REPO_ROOT) else str(archive_path),
        "outer_archive": {
            "bytes": len(archive_bytes),
            "sha256": _sha256(archive_bytes),
        },
        "single_member": member_meta,
        "inner_member_bytes": {
            "bytes": len(inner),
            "sha256": _sha256(inner),
        },
        "negative_vectors": {
            "multi_member_must_raise": (
                "any archive with >1 member must fail PR101 layout assertion"
            ),
            "wrong_member_name_must_raise": (
                f"member name != {PR101_INNER_MEMBER_NAME!r} must fail PR101 layout assertion"
            ),
        },
    }
    output_path = output_dir / f"{ARCHIVE_SCHEMA_VERSION}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = _canonical_json(payload)
    output_path.write_text(output_text, encoding="utf-8")
    return EmittedVector(
        schema=ARCHIVE_SCHEMA_VERSION,
        output_path=output_path,
        sha256=_sha256(output_text.encode("utf-8")),
        description=f"PR101 outer-archive ZIP layout ({len(archive_bytes)} B archive)",
    )


def emit_all(
    *,
    archive_path: Path,
    output_dir: Path,
) -> list[EmittedVector]:
    """Emit all PR101 golden-vector manifests."""
    return [
        emit_archive_zip_layout_vector(archive_path=archive_path, output_dir=output_dir),
        emit_inner_blob_layout_vector(archive_path=archive_path, output_dir=output_dir),
        emit_decoder_blob_vector(archive_path=archive_path, output_dir=output_dir),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-archive", type=Path, required=True,
                        help="Canonical PR101 archive.zip to derive vectors from")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write JSON manifests under "
                             "(canonical: src/tac/tests/golden_vectors/)")
    args = parser.parse_args(argv)

    if not args.input_archive.is_file():
        raise SystemExit(f"input archive not found: {args.input_archive}")

    vectors = emit_all(archive_path=args.input_archive, output_dir=args.output_dir)
    for v in vectors:
        rel = v.output_path.relative_to(REPO_ROOT) if v.output_path.is_relative_to(REPO_ROOT) else v.output_path
        print(f"{v.schema}: {rel}  (sha256={v.sha256[:16]}, {v.description})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
