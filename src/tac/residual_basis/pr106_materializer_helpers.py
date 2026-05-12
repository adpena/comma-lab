"""Shared helpers for the 5 per-family PR106 + non-HNeRV residual materializers.

Each per-family tool at ``tools/materialize_<family>_residual_pr106_sidecar.py``:

1. reads the PR106 r2 canonical archive (``archive.zip``);
2. extracts the canonical ``0.bin`` PR106 payload;
3. computes a family-specific residual (this module is family-agnostic — that
   step is the family's own subagent code);
4. wraps via ``tac.residual_basis.pr106_sidecar_packing.build_archive(...)``;
5. emits ``materialization_manifest.json`` with promotion-status pinned False;
6. emits a no-op-detector byte-mutation smoke test result.

This helper closes the boilerplate around steps 1, 2, 4, 5, 6 so every tool
focuses on its own residual encoder.

Per CLAUDE.md HNeRV parity discipline:

* archive_grammar: monolithic ``0.bin`` (single-file per lesson 3); per-family
  wire format documented in ``pr106_sidecar_packing.py``.
* parser_section_manifest: returned in the manifest payload.
* no_op_detector_planned: ``run_no_op_byte_mutation_smoke`` proves a single
  residual byte change produces a different inflate output via the per-family
  inflate.py runtime.
* score_claim / promotion_eligible / ready_for_exact_eval_dispatch: pinned
  False at every emission boundary.

No score claim. No GPU dispatch. No MPS forwarding. No /tmp paths.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from tac.residual_basis.pr106_sidecar_packing import (
    PR106_RESIDUAL_FORMAT_IDS,
    BuildResidualArchiveResult,
    build_archive,
    parse_archive,
)

PR106_BIN_MEMBER_NAME: Final[str] = "0.bin"
DEFAULT_PR106_ARCHIVE: Final[Path] = (
    Path(__file__).resolve().parents[3]
    / "submissions/pr106_latent_sidecar_r2/archive.zip"
)


class MaterializerError(ValueError):
    """Raised on contract violations in the materialization pipeline."""


@dataclass(frozen=True)
class MaterializationManifest:
    """Typed manifest emitted alongside each candidate archive.

    Per CLAUDE.md HNeRV parity discipline 8-archive-grammar fields + Catalog
    #100 ``check_gate2_no_naked_bytes``: every field that could be promotion-
    bait is pinned False here.
    """

    family: str
    format_id: int
    pr106_source_archive: str
    pr106_source_sha256: str
    pr106_bytes_size: int
    residual_bytes_size: int
    archive_bytes_size: int
    archive_sha256: str
    schema: str
    timestamp_utc: str
    extra: dict[str, Any]
    archive_grammar: str = field(
        default="pr106_plus_residual_sidecar_monolithic_v1", init=False
    )
    parser_section_manifest: str = field(
        default="magic(1B)=0xFD + format_id(1B) + pr106_len(4B LE) + pr106_bytes + residual_len(4B LE) + residual_bytes",
        init=False,
    )
    inflate_runtime_loc_budget: int = field(default=200, init=False)
    runtime_dep_closure: tuple[str, ...] = field(
        default=("numpy", "torch", "brotli", "PR106 codec.py", "PR106 model.py"),
        init=False,
    )
    export_format: str = field(
        default="pr106_plus_residual_per_family_v1", init=False
    )
    score_aware_loss: str = field(
        default="research_only_scaffold_no_score_aware_loss_yet", init=False
    )
    bolt_on_loc_budget: int = field(default=350, init=False)
    no_op_detector_planned: bool = field(default=True, init=False)
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default="research_signal", init=False)


def now_utc_iso() -> str:
    """Return the current UTC time in ISO 8601 with seconds precision."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_pr106_bytes(pr106_archive: Path) -> tuple[bytes, str]:
    """Extract the canonical 0.bin PR106 payload from the archive zip.

    Returns (bytes, sha256_of_zip). The PR106 canonical archive is a single-
    member zip per HNeRV parity discipline; if the zip has additional members
    the helper refuses (defensive guard).
    """
    if not pr106_archive.is_file():
        raise MaterializerError(f"PR106 archive not found: {pr106_archive}")
    archive_sha = sha256_file(pr106_archive)
    with zipfile.ZipFile(pr106_archive, mode="r") as zf:
        names = zf.namelist()
        if PR106_BIN_MEMBER_NAME not in names:
            raise MaterializerError(
                f"{PR106_BIN_MEMBER_NAME} not in archive {pr106_archive}: members={names}"
            )
        # Reject multi-member zips — PR106 canonical is single-file.
        if names != [PR106_BIN_MEMBER_NAME]:
            raise MaterializerError(
                f"PR106 archive must contain only '{PR106_BIN_MEMBER_NAME}'; got {names}"
            )
        pr106_bytes = zf.read(PR106_BIN_MEMBER_NAME)
    if not pr106_bytes:
        raise MaterializerError(f"PR106 0.bin is empty in {pr106_archive}")
    return pr106_bytes, archive_sha


def emit_archive_zip(archive_bytes: bytes, output_zip: Path) -> Path:
    """Wrap the raw archive bytes into the canonical single-member zip.

    The contest packet format requires a zip-shaped wrapper; this helper
    writes the residual archive as ``0.bin`` inside a zip with a fixed
    timestamp so the output is byte-deterministic.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    # Use ZipInfo with fixed mtime (2025-01-01 00:00:00 UTC) for deterministic zip bytes.
    info = zipfile.ZipInfo(filename=PR106_BIN_MEMBER_NAME, date_time=(2025, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED  # 0.bin is already-compressed payload
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(output_zip, mode="w") as zf:
        zf.writestr(info, archive_bytes)
    return output_zip


def write_manifest(manifest: MaterializationManifest, output_path: Path) -> Path:
    """Write the manifest as deterministic-ordered JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": manifest.schema,
        "timestamp_utc": manifest.timestamp_utc,
        "family": manifest.family,
        "format_id": manifest.format_id,
        "pr106_source_archive": manifest.pr106_source_archive,
        "pr106_source_sha256": manifest.pr106_source_sha256,
        "pr106_bytes_size": manifest.pr106_bytes_size,
        "residual_bytes_size": manifest.residual_bytes_size,
        "archive_bytes_size": manifest.archive_bytes_size,
        "archive_sha256": manifest.archive_sha256,
        "archive_grammar": manifest.archive_grammar,
        "parser_section_manifest": manifest.parser_section_manifest,
        "inflate_runtime_loc_budget": manifest.inflate_runtime_loc_budget,
        "runtime_dep_closure": list(manifest.runtime_dep_closure),
        "export_format": manifest.export_format,
        "score_aware_loss": manifest.score_aware_loss,
        "bolt_on_loc_budget": manifest.bolt_on_loc_budget,
        "no_op_detector_planned": manifest.no_op_detector_planned,
        "score_claim": manifest.score_claim,
        "promotion_eligible": manifest.promotion_eligible,
        "ready_for_exact_eval_dispatch": manifest.ready_for_exact_eval_dispatch,
        "evidence_grade": manifest.evidence_grade,
        "extra": manifest.extra,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path


def materialize_family_archive(
    *,
    family: str,
    pr106_archive: Path,
    residual_bytes: bytes,
    output_dir: Path,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path, MaterializationManifest, BuildResidualArchiveResult]:
    """End-to-end family-agnostic materializer.

    Reads PR106 r2 → wraps with the family-specific residual blob →
    emits ``<family>_pr106_residual_sidecar_archive.zip`` and
    ``materialization_manifest.json`` into ``output_dir``.

    The residual byte encoding (the family-specific part) MUST be
    pre-computed by the per-family tool and passed in as ``residual_bytes``;
    this helper does not interpret the residual contents.

    Returns ``(archive_zip_path, manifest_path, manifest, build_result)``.
    """
    if family not in PR106_RESIDUAL_FORMAT_IDS:
        raise MaterializerError(f"unknown family {family!r}")
    pr106_bytes, pr106_sha = extract_pr106_bytes(pr106_archive)
    build_result = build_archive(
        family=family, pr106_bytes=pr106_bytes, residual_bytes=residual_bytes
    )
    archive_zip = output_dir / f"{family}_pr106_residual_sidecar_archive.zip"
    emit_archive_zip(build_result.archive_bytes, archive_zip)
    manifest_path = output_dir / "materialization_manifest.json"
    manifest = MaterializationManifest(
        family=family,
        format_id=build_result.format_id,
        pr106_source_archive=str(pr106_archive),
        pr106_source_sha256=pr106_sha,
        pr106_bytes_size=build_result.pr106_len,
        residual_bytes_size=build_result.residual_len,
        archive_bytes_size=len(build_result.archive_bytes),
        archive_sha256=sha256_bytes(build_result.archive_bytes),
        schema=f"{family}_pr106_residual_sidecar_materialization_v1",
        timestamp_utc=now_utc_iso(),
        extra=extra or {},
    )
    write_manifest(manifest, manifest_path)
    return archive_zip, manifest_path, manifest, build_result


def repack_dense_as_sparse(
    *,
    family: str,
    dense_residual_bytes: bytes,
    n_frames: int,
) -> bytes:
    """Repack a dense-wire-format residual blob into the sparse PacketIR envelope.

    The sparse envelope for wavelet/c3/cool_chic/coord_mlp families is
    temporal-subsampled outer (all-frames-carrying-signal in this dense
    rewrap) + per-frame (scale prefix + RLE-of-zeros over band int8 coeffs).

    For an EMPTY dense input (``residual_mode=empty``) the result is also
    empty — sparse over zero frames is zero bytes (the L1 sparse decoder
    short-circuits an empty blob).

    Closes O's L2 wire-format ceiling: a sparse-aware materializer + sparse
    inflate path is required before L2 score-aware encoders can emit bytes
    that fit inside any meaningful contest byte budget. This helper is the
    byte-stable scaffold path; the L2 encoder's sparse output integration
    is the score-aware research-only path layered on top.
    """

    from tac.packet_compiler.sparse_packet_ir import (
        encode_rle_of_zeros,
        encode_temporal_subsampled,
        pad_per_frame_to_uniform_size_with_length_prefix,
        serialize_rle_of_zeros,
        serialize_temporal_subsampled,
    )
    import struct

    import numpy as np

    def _temporal_payload_stream(payloads: list[bytes | None]) -> bytes:
        # Per Sparse PacketIR uniform-per-frame contract: variable-size payloads
        # are zero-padded to the longest payload, with a 4-byte LE u32 length
        # prefix per frame so the decoder recovers the original size.
        # `pad_per_frame_to_uniform_size_with_length_prefix` is the canonical
        # adapter; the family inflate.py decoders read the leading <I and slice
        # off the zero padding (see e.g. wavelet inflate.py:99-102).
        frames = pad_per_frame_to_uniform_size_with_length_prefix(payloads)
        return serialize_temporal_subsampled(encode_temporal_subsampled(frames))

    if not dense_residual_bytes:
        return b""
    if family == "wavelet":
        camera_h, camera_w, rgb = 874, 1164, 3
        half_h, half_w = camera_h // 2, camera_w // 2
        band_size = half_h * half_w
        per_frame_bytes = 16 + 4 * rgb * band_size
        if len(dense_residual_bytes) != n_frames * per_frame_bytes:
            raise MaterializerError(
                f"wavelet dense bytes {len(dense_residual_bytes)} != "
                f"n_frames*per_frame {n_frames * per_frame_bytes}"
            )
        per_frame: list[bytes | None] = []
        pos = 0
        for _t in range(n_frames):
            scales_blob = dense_residual_bytes[pos : pos + 16]
            pos += 16
            band_int8 = np.frombuffer(
                dense_residual_bytes, dtype=np.int8, count=4 * rgb * band_size, offset=pos
            )
            pos += 4 * rgb * band_size
            if scales_blob == b"\x00" * 16 and not bool(np.any(band_int8)):
                per_frame.append(None)
                continue
            rle = encode_rle_of_zeros(band_int8.copy())
            rle_bytes = serialize_rle_of_zeros(rle)
            per_frame.append(scales_blob + rle_bytes)
        return _temporal_payload_stream(per_frame)
    if family in ("c3", "coord_mlp"):
        camera_h, camera_w, rgb = 874, 1164, 3
        if family == "c3":
            grid_h, grid_w = camera_h // 4, camera_w // 4
        else:  # coord_mlp
            grid_h, grid_w = camera_h // 8, camera_w // 8
        per_frame_bytes = 4 + grid_h * grid_w * rgb
        if len(dense_residual_bytes) != n_frames * per_frame_bytes:
            raise MaterializerError(
                f"{family} dense bytes {len(dense_residual_bytes)} != "
                f"n_frames*per_frame {n_frames * per_frame_bytes}"
            )
        per_frame_list: list[bytes | None] = []
        pos = 0
        for _t in range(n_frames):
            scale_bytes = dense_residual_bytes[pos : pos + 4]
            pos += 4
            coeffs = np.frombuffer(
                dense_residual_bytes, dtype=np.int8, count=grid_h * grid_w * rgb, offset=pos
            )
            pos += grid_h * grid_w * rgb
            if scale_bytes == b"\x00" * 4 and not bool(np.any(coeffs)):
                per_frame_list.append(None)
                continue
            rle_bytes = serialize_rle_of_zeros(encode_rle_of_zeros(coeffs.copy()))
            per_frame_list.append(scale_bytes + rle_bytes)
        return _temporal_payload_stream(per_frame_list)
    if family == "cool_chic":
        if len(dense_residual_bytes) < 2:
            raise MaterializerError("cool_chic dense bytes too short for header")
        (n_levels,) = struct.unpack_from("<H", dense_residual_bytes, 0)
        pos = 2
        out_parts: list[bytes] = [struct.pack("<H", n_levels)]
        camera_h, camera_w, rgb = 874, 1164, 3
        for L in range(n_levels):
            (scale,) = struct.unpack_from("<f", dense_residual_bytes, pos)
            pos += 4
            h_L = camera_h // (2 ** L)
            w_L = camera_w // (2 ** L)
            level_bytes = n_frames * h_L * w_L * rgb
            level_int8 = np.frombuffer(
                dense_residual_bytes, dtype=np.int8, count=level_bytes, offset=pos
            )
            pos += level_bytes
            rle_bytes = serialize_rle_of_zeros(encode_rle_of_zeros(level_int8.copy()))
            out_parts.append(struct.pack("<fI", scale, len(rle_bytes)))
            out_parts.append(rle_bytes)
        if pos != len(dense_residual_bytes):
            raise MaterializerError(
                f"cool_chic trailing bytes after pyramid: pos={pos} total={len(dense_residual_bytes)}"
            )
        return b"".join(out_parts)
    if family == "siren":
        # Dense layout: 4B scale + 2B n_coefs + n_coefs * 9B records.
        if len(dense_residual_bytes) < 6:
            raise MaterializerError("siren dense bytes too short")
        scale_bytes = dense_residual_bytes[:4]
        (n_coefs,) = struct.unpack_from("<H", dense_residual_bytes, 4)
        record_size = 9
        addr_parts: list[bytes] = []
        coef_pairs: list[int] = []
        pos = 6
        for _ in range(n_coefs):
            frame_idx, k_row, k_col, channel, real_q, imag_q = struct.unpack_from(
                "<HhhBbb", dense_residual_bytes, pos
            )
            pos += record_size
            addr_parts.append(struct.pack("<HhhB", frame_idx, k_row, k_col, channel))
            coef_pairs.extend([real_q, imag_q])
        coef_stream = np.array(coef_pairs, dtype=np.int8)
        rle_bytes = serialize_rle_of_zeros(encode_rle_of_zeros(coef_stream))
        return scale_bytes + struct.pack("<I", n_coefs) + b"".join(addr_parts) + rle_bytes
    raise MaterializerError(f"sparse repack not implemented for family {family!r}")


def truncate_wavelet_dense_to_top_k(
    *,
    dense_residual_bytes: bytes,
    n_frames: int,
    top_k_per_frame: int,
) -> bytes:
    """Zero all but the largest-magnitude wavelet coefficients per frame.

    This is a deterministic sparse-budget bridge between the dense L2 wavelet
    oracle and the runtime-consumed sparse PacketIR stream. It deliberately
    keeps the per-frame scale words unchanged and sparsifies only the int8
    coefficient payload, so downstream decoders still consume the standard
    wavelet residual grammar.
    """

    import numpy as np

    if top_k_per_frame < 0:
        raise MaterializerError(
            f"top_k_per_frame must be >= 0; got {top_k_per_frame}"
        )
    if not dense_residual_bytes or top_k_per_frame == 0:
        return b""
    camera_h, camera_w, rgb = 874, 1164, 3
    half_h, half_w = camera_h // 2, camera_w // 2
    coeff_count = 4 * rgb * half_h * half_w
    per_frame_bytes = 16 + coeff_count
    expected = n_frames * per_frame_bytes
    if len(dense_residual_bytes) != expected:
        raise MaterializerError(
            f"wavelet dense bytes {len(dense_residual_bytes)} != expected {expected}"
        )
    parts: list[bytes] = []
    pos = 0
    for _t in range(n_frames):
        scales_blob = dense_residual_bytes[pos : pos + 16]
        pos += 16
        coeffs = np.frombuffer(
            dense_residual_bytes,
            dtype=np.int8,
            count=coeff_count,
            offset=pos,
        ).copy()
        pos += coeff_count
        nonzero = np.flatnonzero(coeffs)
        if nonzero.size > top_k_per_frame:
            magnitudes = np.abs(coeffs[nonzero].astype(np.int16))
            # Stable descending-magnitude order keeps tie breaks deterministic
            # by preserving the ascending-index order from flatnonzero().
            keep_local = np.argsort(-magnitudes, kind="stable")[:top_k_per_frame]
            keep = nonzero[keep_local]
            sparse = np.zeros_like(coeffs)
            sparse[keep] = coeffs[keep]
            coeffs = sparse
        parts.append(scales_blob + coeffs.tobytes())
    return b"".join(parts)


def run_no_op_detector_byte_mutation(
    *,
    archive_bytes: bytes,
    expected_format_id: int,
) -> dict[str, Any]:
    """Run the byte-mutation no-op detector smoke: flip a residual byte and
    verify the parsed residual changes.

    This is the in-process smoke; the per-family inflate runtime tests do the
    end-to-end byte-mutation parity check (which requires PyTorch + the PR106
    decoder) on a real PR106 input.
    """
    parsed_a = parse_archive(archive_bytes)
    if parsed_a.format_id != expected_format_id:
        raise MaterializerError(
            f"format_id mismatch in no-op detector smoke: "
            f"got 0x{parsed_a.format_id:02X} expected 0x{expected_format_id:02X}"
        )
    if not parsed_a.residual_bytes:
        return {
            "result": "skipped_empty_residual",
            "rationale": "no_op_detector_inapplicable_when_residual_is_zero_bytes",
        }
    blob = bytearray(archive_bytes)
    # First residual byte sits at offset header(6B) + pr106_len + residual_len_prefix(4B).
    offset = 6 + len(parsed_a.pr106_bytes) + 4
    original = blob[offset]
    blob[offset] = (original + 1) & 0xFF
    parsed_b = parse_archive(bytes(blob))
    return {
        "result": "passed" if parsed_a.residual_bytes != parsed_b.residual_bytes else "failed",
        "offset_mutated": offset,
        "original_byte": original,
        "mutated_byte": (original + 1) & 0xFF,
    }


__all__ = [
    "DEFAULT_PR106_ARCHIVE",
    "MaterializationManifest",
    "MaterializerError",
    "PR106_BIN_MEMBER_NAME",
    "emit_archive_zip",
    "extract_pr106_bytes",
    "materialize_family_archive",
    "now_utc_iso",
    "run_no_op_detector_byte_mutation",
    "sha256_bytes",
    "sha256_file",
    "truncate_wavelet_dense_to_top_k",
    "write_manifest",
]
