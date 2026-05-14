# SPDX-License-Identifier: MIT
"""Runtime-closure helpers for PR103 AC decoder bytes inside a PR106 envelope.

The PR103 arithmetic decoder section is not self-describing: its variable
sections are split by hardcoded lengths in the runtime.  The PR106 archive
envelope, meanwhile, only carries a single decoder length:

    0xff | decoder_len:u24_le | decoder_bytes | pr106_fixed_latents_brotli

This module records and validates the missing PR103 section metadata so a
self-contained inflate runtime can consume the repacked decoder bytes without
guessing.  It is a byte/proof surface only; it does not load scorers.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA
from tac.packet_compiler.pr106_fixed_latent_recode import decode_pr106_fixed_latent_raw
from tac.pr103_arithmetic_codec import (
    AC_TENSOR_INDICES,
    EncodedAcDecoderBlob,
    Pr103ArithmeticCodecError,
    decode_decoder_ac,
    encode_decoder_ac,
)

PR103_PR106_RUNTIME_FORMAT = "pr103_ac_decoder_inside_pr106_ff_packed_v1"
PR106_PACKED_META: dict[str, Any] = {
    "n_pairs": 600,
    "latent_dim": 28,
    "base_channels": 36,
    "eval_size": [384, 512],
}
PR103_SECTION_KEYS = ("br", "hists", "merged_ac", "hi_hist", "ac_fallback")


class Pr103Pr106RuntimeClosureError(ValueError):
    """Raised when PR103/PR106 runtime-closure metadata is unsafe or mismatched."""


@dataclass(frozen=True)
class SingleMemberPayload:
    """Strict single-member ZIP extraction result."""

    member_name: str
    payload: bytes
    archive_bytes: int
    archive_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class Pr106PackedSections:
    """Sections of the PR106 ``0xff`` packed payload envelope."""

    header: bytes
    decoder: bytes
    latents: bytes


@dataclass(frozen=True)
class Pr106DecodedPayload:
    """Decoded PR106 packed source payload."""

    state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    meta: dict[str, Any] = field(default_factory=lambda: dict(PR106_PACKED_META))


@dataclass(frozen=True)
class Pr103Pr106DecodedPayload:
    """Decoded PR103-repacked PR106 payload."""

    state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    meta: dict[str, Any] = field(default_factory=lambda: dict(PR106_PACKED_META))


@dataclass(frozen=True)
class Pr103Pr106RuntimeClosure:
    """Decoder metadata required by a PR103-aware PR106 inflate runtime.

    ``section_lengths`` and ``ac_fallback_set`` are the safety-critical fields:
    changing either changes how the decoder interprets bytes.  The optional
    SHA/byte fields bind the metadata to one exact decoder and latents section.
    """

    section_lengths: dict[str, int]
    ac_fallback_set: tuple[int, ...] = ()
    n_latent_hi_symbols: int = 0
    decoder_section_bytes: int | None = None
    decoder_section_sha256: str | None = None
    latents_section_bytes: int | None = None
    latents_section_sha256: str | None = None
    brotli_quality: int = 11
    adaptive_lgwin: bool = True
    ac_auto_fallback: bool = True
    format: str = PR103_PR106_RUNTIME_FORMAT

    def __post_init__(self) -> None:
        normalized_lengths = _validate_section_lengths(self.section_lengths)
        fallback = _validate_ac_fallback_set(self.ac_fallback_set)
        if int(self.n_latent_hi_symbols) != 0:
            raise Pr103Pr106RuntimeClosureError(
                "PR106 fixed-latents closure requires n_latent_hi_symbols=0"
            )
        if self.format != PR103_PR106_RUNTIME_FORMAT:
            raise Pr103Pr106RuntimeClosureError(
                f"unsupported runtime closure format {self.format!r}"
            )
        object.__setattr__(self, "section_lengths", normalized_lengths)
        object.__setattr__(self, "ac_fallback_set", fallback)
        object.__setattr__(self, "n_latent_hi_symbols", int(self.n_latent_hi_symbols))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "format": self.format,
            "section_lengths": dict(self.section_lengths),
            "ac_fallback_set": list(self.ac_fallback_set),
            "n_latent_hi_symbols": self.n_latent_hi_symbols,
            "decoder_section_bytes": self.decoder_section_bytes,
            "decoder_section_sha256": self.decoder_section_sha256,
            "latents_section_bytes": self.latents_section_bytes,
            "latents_section_sha256": self.latents_section_sha256,
            "brotli_quality": self.brotli_quality,
            "adaptive_lgwin": self.adaptive_lgwin,
            "ac_auto_fallback": self.ac_auto_fallback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pr103Pr106RuntimeClosure:
        if int(data.get("schema_version", 1)) != 1:
            raise Pr103Pr106RuntimeClosureError(
                f"unsupported closure schema_version={data.get('schema_version')!r}"
            )
        if "section_lengths" not in data:
            raise Pr103Pr106RuntimeClosureError("closure missing section_lengths")
        return cls(
            section_lengths=dict(data["section_lengths"]),
            ac_fallback_set=tuple(data.get("ac_fallback_set", ()) or ()),
            n_latent_hi_symbols=int(data.get("n_latent_hi_symbols", 0)),
            decoder_section_bytes=data.get("decoder_section_bytes"),
            decoder_section_sha256=data.get("decoder_section_sha256"),
            latents_section_bytes=data.get("latents_section_bytes"),
            latents_section_sha256=data.get("latents_section_sha256"),
            brotli_quality=int(data.get("brotli_quality", 11)),
            adaptive_lgwin=bool(data.get("adaptive_lgwin", True)),
            ac_auto_fallback=bool(data.get("ac_auto_fallback", True)),
            format=str(data.get("format", PR103_PR106_RUNTIME_FORMAT)),
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_single_member_payload(path: Path) -> SingleMemberPayload:
    """Read a strict single-file archive without extracting to disk."""
    archive_blob = path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise Pr103Pr106RuntimeClosureError(
                f"{path} has {len(infos)} file members; expected exactly one"
            )
        info = infos[0]
        _validate_zip_member_name(info.filename)
        payload = zf.read(info.filename)
    return SingleMemberPayload(
        member_name=info.filename,
        payload=payload,
        archive_bytes=len(archive_blob),
        archive_sha256=sha256_bytes(archive_blob),
        payload_sha256=sha256_bytes(payload),
    )


def split_pr106_packed_payload(payload: bytes) -> Pr106PackedSections:
    """Split PR106's packed envelope and fail closed on malformed lengths."""
    if len(payload) < 4:
        raise Pr103Pr106RuntimeClosureError("payload too short for PR106 packed header")
    if payload[0] != 0xFF:
        raise Pr103Pr106RuntimeClosureError(
            f"expected PR106 packed magic 0xff, got 0x{payload[0]:02x}"
        )
    dec_len = int.from_bytes(payload[1:4], "little")
    if dec_len <= 0:
        raise Pr103Pr106RuntimeClosureError(
            f"invalid PR106 decoder section length {dec_len}"
        )
    if 4 + dec_len >= len(payload):
        raise Pr103Pr106RuntimeClosureError(
            "PR106 decoder section length leaves no fixed-latents section"
        )
    return Pr106PackedSections(
        header=payload[:4],
        decoder=payload[4 : 4 + dec_len],
        latents=payload[4 + dec_len :],
    )


def decode_pr106_packed_payload(payload: bytes) -> Pr106DecodedPayload:
    """Decode a public PR106-style packed payload."""
    sections = split_pr106_packed_payload(payload)
    return Pr106DecodedPayload(
        state_dict=_decode_pr106_packed_decoder(sections.decoder),
        latents=decode_pr106_fixed_latents(sections.latents),
        meta=dict(PR106_PACKED_META),
    )


def decode_pr106_fixed_latents(data: bytes) -> torch.Tensor:
    """Decode PR106's fixed 600x28 latent section."""
    raw = decode_pr106_fixed_latent_raw(data)
    n = int(PR106_PACKED_META["n_pairs"])
    d = int(PR106_PACKED_META["latent_dim"])
    meta_len = d * 4
    total = n * d
    expected = total + meta_len + total
    if len(raw) != expected:
        raise Pr103Pr106RuntimeClosureError(
            f"bad PR106 fixed-latents payload: decompressed len {len(raw)} "
            f"expected {expected}"
        )
    lo = np.frombuffer(raw[:total], dtype=np.uint8).astype(np.uint16)
    mins = torch.from_numpy(
        np.frombuffer(raw[total : total + d * 2], dtype=np.float16).copy()
    ).float()
    scales = torch.from_numpy(
        np.frombuffer(raw[total + d * 2 : total + meta_len], dtype=np.float16).copy()
    ).float()
    hi = np.frombuffer(
        raw[total + meta_len : total + meta_len + total], dtype=np.uint8
    ).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(
        delta_zz % 2 == 0,
        delta_zz.astype(np.int32) // 2,
        -(delta_zz.astype(np.int32) // 2) - 1,
    ).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


def section_lengths_from_layout(layout: EncodedAcDecoderBlob) -> dict[str, int]:
    return {
        "br": sum(len(item) for item in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
        "ac_fallback": len(layout.ac_fallback_blob),
    }


def build_runtime_closure_from_layout(
    layout: EncodedAcDecoderBlob,
    *,
    candidate_payload: bytes | None = None,
    brotli_quality: int = 11,
    adaptive_lgwin: bool = True,
    ac_auto_fallback: bool = True,
) -> Pr103Pr106RuntimeClosure:
    """Build closure metadata from an encoder layout and optional payload bind."""
    decoder = layout.blob
    latents: bytes | None = None
    if candidate_payload is not None:
        sections = split_pr106_packed_payload(candidate_payload)
        decoder = sections.decoder
        latents = sections.latents
        if sections.decoder != layout.blob:
            raise Pr103Pr106RuntimeClosureError(
                "layout decoder bytes do not match candidate payload decoder bytes "
                f"(layout_sha={sha256_bytes(layout.blob)} "
                f"candidate_sha={sha256_bytes(sections.decoder)})"
            )
    return Pr103Pr106RuntimeClosure(
        section_lengths=section_lengths_from_layout(layout),
        ac_fallback_set=tuple(layout.ac_fallback_set),
        n_latent_hi_symbols=0,
        decoder_section_bytes=len(decoder),
        decoder_section_sha256=sha256_bytes(decoder),
        latents_section_bytes=len(latents) if latents is not None else None,
        latents_section_sha256=sha256_bytes(latents) if latents is not None else None,
        brotli_quality=brotli_quality,
        adaptive_lgwin=adaptive_lgwin,
        ac_auto_fallback=ac_auto_fallback,
    )


def derive_runtime_closure_from_pr106_source(
    *,
    source_payload: bytes,
    candidate_payload: bytes,
    brotli_quality: int = 11,
    adaptive_lgwin: bool = True,
    ac_auto_fallback: bool = True,
) -> Pr103Pr106RuntimeClosure:
    """Re-derive PR103 section metadata from source PR106 bytes and bind it.

    This is the closure proof for the PR106 -> PR103 AC repack: decode the
    source PR106 decoder, re-encode through PR103 AC, and require the produced
    decoder bytes to match the candidate archive exactly before emitting the
    section lengths that a runtime would hardcode or carry in a tiny header.
    """
    source = decode_pr106_packed_payload(source_payload)
    layout = encode_decoder_ac(
        source.state_dict,
        brotli_quality=brotli_quality,
        adaptive_lgwin=adaptive_lgwin,
        return_layout=True,
        ac_auto_fallback=ac_auto_fallback,
    )
    if not isinstance(layout, EncodedAcDecoderBlob):
        raise Pr103Pr106RuntimeClosureError("encode_decoder_ac did not return layout")
    return build_runtime_closure_from_layout(
        layout,
        candidate_payload=candidate_payload,
        brotli_quality=brotli_quality,
        adaptive_lgwin=adaptive_lgwin,
        ac_auto_fallback=ac_auto_fallback,
    )


def parse_pr103_repacked_pr106_payload(
    payload: bytes,
    closure: Pr103Pr106RuntimeClosure | dict[str, Any],
) -> Pr103Pr106DecodedPayload:
    """Decode PR103 AC decoder bytes from a PR106 packed envelope.

    The closure is validated against payload byte lengths and SHA-256 values
    before the PR103 decoder sees any data.
    """
    closure_obj = (
        Pr103Pr106RuntimeClosure.from_dict(closure)
        if isinstance(closure, dict)
        else closure
    )
    sections = split_pr106_packed_payload(payload)
    _validate_closure_against_sections(closure_obj, sections)
    try:
        decoded = decode_decoder_ac(
            sections.decoder,
            section_lengths=closure_obj.section_lengths,
            n_latent_hi_symbols=closure_obj.n_latent_hi_symbols,
            ac_fallback_set=closure_obj.ac_fallback_set,
        )
    except Pr103ArithmeticCodecError as exc:
        raise Pr103Pr106RuntimeClosureError(str(exc)) from exc
    return Pr103Pr106DecodedPayload(
        state_dict=decoded.state_dict,
        latents=decode_pr106_fixed_latents(sections.latents),
        meta=dict(PR106_PACKED_META),
    )


def build_runtime_closure_proof_record(
    *,
    source_archive: SingleMemberPayload,
    candidate_archive: SingleMemberPayload,
    closure: Pr103Pr106RuntimeClosure,
    source_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable proof/custody record for a closure."""
    decoded = parse_pr103_repacked_pr106_payload(candidate_archive.payload, closure)
    sections = split_pr106_packed_payload(candidate_archive.payload)
    record: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr103_pr106_runtime_closure",
        "score_claim": False,
        "score_evidence_grade": "empirical_runtime_closure",
        "runtime_closure": closure.to_dict(),
        "source_archive": {
            "member_name": source_archive.member_name,
            "archive_bytes": source_archive.archive_bytes,
            "archive_sha256": source_archive.archive_sha256,
            "payload_bytes": len(source_archive.payload),
            "payload_sha256": source_archive.payload_sha256,
        },
        "candidate_archive": {
            "member_name": candidate_archive.member_name,
            "archive_bytes": candidate_archive.archive_bytes,
            "archive_sha256": candidate_archive.archive_sha256,
            "payload_bytes": len(candidate_archive.payload),
            "payload_sha256": candidate_archive.payload_sha256,
            "decoder_section_bytes": len(sections.decoder),
            "decoder_section_sha256": sha256_bytes(sections.decoder),
            "latents_section_bytes": len(sections.latents),
            "latents_section_sha256": sha256_bytes(sections.latents),
        },
        "decoded_runtime_contract": {
            "state_dict_tensors": len(decoded.state_dict),
            "state_dict_params": int(sum(t.numel() for t in decoded.state_dict.values())),
            "latents_shape": list(decoded.latents.shape),
            "meta": decoded.meta,
        },
        "fail_closed_contract": [
            "decoder_section_sha256_must_match",
            "latents_section_sha256_must_match",
            "section_lengths_sum_must_equal_decoder_bytes",
            "ac_fallback_set_must_match_fallback_section",
            "ac_fallback_set_indices_must_be_pr103_ac_indices",
        ],
        "exact_cuda_remaining_blockers": [
            "copy_or_generate_runtime_adapter_into_submission_tree",
            "ensure constriction/brotli dependency closure inside runtime environment",
            "run pre_submission_compliance_check on final archive/runtime packet",
            "claim dispatch lane before exact CUDA auth eval",
            "run archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        ],
    }
    if source_manifest is not None:
        record["manifest_consistency"] = _manifest_consistency(
            source_manifest, candidate_archive, sections
        )
    return record


def _decode_pr106_packed_decoder(data: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(data)
    packed_schema = sorted(FIXED_STATE_SCHEMA, key=lambda item: -int(np.prod(item[1])))
    pos = 0
    quantized: list[np.ndarray] = []
    for _name, shape in packed_schema:
        size = int(np.prod(shape))
        if pos + size > len(raw):
            raise Pr103Pr106RuntimeClosureError("truncated PR106 packed decoder payload")
        quantized.append(_zigzag_decode_u8(np.frombuffer(raw[pos : pos + size], dtype=np.uint8)))
        pos += size
    scales_pos = pos
    expected = scales_pos + 4 * len(packed_schema)
    if expected != len(raw):
        raise Pr103Pr106RuntimeClosureError(
            f"bad PR106 packed decoder payload: expected raw len {expected}, got {len(raw)}"
        )
    sd: dict[str, torch.Tensor] = {}
    for index, (name, shape) in enumerate(packed_schema):
        scale_start = scales_pos + 4 * index
        scale = float(
            np.frombuffer(raw[scale_start : scale_start + 4], dtype=np.float32)[0]
        )
        sd[name] = torch.from_numpy(quantized[index].astype(np.float32).reshape(shape)) * scale
    return sd


def _zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def _validate_zip_member_name(name: str) -> None:
    if not name or name.startswith("/") or name.startswith("\\"):
        raise Pr103Pr106RuntimeClosureError(f"unsafe zip member name {name!r}")
    parts = Path(name).parts
    if any(part in ("", ".", "..") for part in parts):
        raise Pr103Pr106RuntimeClosureError(f"unsafe zip member name {name!r}")


def _validate_section_lengths(values: dict[str, Any]) -> dict[str, int]:
    missing = set(PR103_SECTION_KEYS) - set(values)
    extra = set(values) - set(PR103_SECTION_KEYS)
    if missing or extra:
        raise Pr103Pr106RuntimeClosureError(
            f"section_lengths keys mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    out: dict[str, int] = {}
    for key in PR103_SECTION_KEYS:
        value = int(values[key])
        if value < 0:
            raise Pr103Pr106RuntimeClosureError(
                f"section length {key!r} must be nonnegative, got {value}"
            )
        out[key] = value
    return out


def _validate_ac_fallback_set(values: tuple[int, ...]) -> tuple[int, ...]:
    fallback = tuple(int(item) for item in values)
    if tuple(sorted(fallback)) != fallback or len(set(fallback)) != len(fallback):
        raise Pr103Pr106RuntimeClosureError(
            f"ac_fallback_set must be sorted and unique, got {fallback!r}"
        )
    invalid = set(fallback) - set(AC_TENSOR_INDICES)
    if invalid:
        raise Pr103Pr106RuntimeClosureError(
            f"ac_fallback_set contains non-PR103 AC indices: {sorted(invalid)}"
        )
    return fallback


def _validate_closure_against_sections(
    closure: Pr103Pr106RuntimeClosure,
    sections: Pr106PackedSections,
) -> None:
    scales_len = len(FIXED_STATE_SCHEMA) * 2
    expected_decoder_len = scales_len + sum(closure.section_lengths.values())
    if expected_decoder_len != len(sections.decoder):
        raise Pr103Pr106RuntimeClosureError(
            f"section_lengths sum ({expected_decoder_len}) != decoder len "
            f"({len(sections.decoder)})"
        )
    fallback_len = closure.section_lengths["ac_fallback"]
    if fallback_len > 0 and not closure.ac_fallback_set:
        raise Pr103Pr106RuntimeClosureError(
            "ac_fallback section non-empty but ac_fallback_set is empty"
        )
    if fallback_len == 0 and closure.ac_fallback_set:
        raise Pr103Pr106RuntimeClosureError(
            f"ac_fallback_set {list(closure.ac_fallback_set)!r} non-empty but "
            "ac_fallback section length is zero"
        )
    if closure.decoder_section_bytes is not None and (
        int(closure.decoder_section_bytes) != len(sections.decoder)
    ):
        raise Pr103Pr106RuntimeClosureError(
            f"decoder byte length mismatch: closure={closure.decoder_section_bytes} "
            f"payload={len(sections.decoder)}"
        )
    if closure.decoder_section_sha256 is not None and (
        closure.decoder_section_sha256 != sha256_bytes(sections.decoder)
    ):
        raise Pr103Pr106RuntimeClosureError("decoder SHA-256 mismatch")
    if closure.latents_section_bytes is not None and (
        int(closure.latents_section_bytes) != len(sections.latents)
    ):
        raise Pr103Pr106RuntimeClosureError(
            f"latents byte length mismatch: closure={closure.latents_section_bytes} "
            f"payload={len(sections.latents)}"
        )
    if closure.latents_section_sha256 is not None and (
        closure.latents_section_sha256 != sha256_bytes(sections.latents)
    ):
        raise Pr103Pr106RuntimeClosureError("latents SHA-256 mismatch")


def _manifest_consistency(
    manifest: dict[str, Any],
    candidate_archive: SingleMemberPayload,
    sections: Pr106PackedSections,
) -> dict[str, Any]:
    checks = {
        "output_archive_sha256": manifest.get("output_archive_sha256")
        == candidate_archive.archive_sha256,
        "output_archive_bytes": manifest.get("output_archive_bytes")
        == candidate_archive.archive_bytes,
        "output_decoder_section_bytes": manifest.get("output_decoder_section_bytes")
        == len(sections.decoder),
        "output_latents_section_bytes": manifest.get("output_latents_section_bytes")
        == len(sections.latents),
        "runtime_adapter_required": manifest.get("runtime_adapter_required") is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_tool": manifest.get("tool"),
        "manifest_score_claim": manifest.get("score_claim"),
        "manifest_score_evidence_grade": manifest.get("score_evidence_grade"),
    }
