"""S2SBS_AR/S2S1 monolithic archive grammar.

Wire format (one ``0.bin`` payload):

    magic                4 bytes   b"S2SB"
    grammar              4 bytes   b"S2S1"
    version              uint8     1
    num_pairs            uint16
    output_height        uint16
    output_width         uint16
    lf_cutoff_h          uint16
    lf_cutoff_w          uint16
    delta_amp_milli_u8   uint16   (delta_amp_uint8 * 1000)
    payload_channel      uint8    (0=R, 1=G, 2=B)
    base_seed            uint32
    payload_bytes_per_pair  uint16
    ecc_rate_milli       uint16   (ecc_rate * 1000)
    payload_len          uint32   (HF byte stream length across all pairs)
    base_blob_len        uint32   (deterministic base manifest length)
    metadata_len         uint32
    payload_sha256       32 bytes (sha256(payload_blob + base_blob + meta_blob))
    payload_blob         binary   per-pair length-prefixed payload bytes
    base_blob            json     base decoder manifest (deterministic seed)
    metadata_blob        json     research-only metadata (no score authority)

Per CLAUDE.md Catalog #100 "no naked bytes": ``score_claim`` /
``promotion_eligible`` / ``ready_for_exact_eval_dispatch`` are FORCED
False on every parsed archive regardless of input metadata. The byte
mutation contract (Catalog #105) is exercised by the dedicated tests.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .architecture import (
    CONTEST_NUM_PAIRS,
    MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT,
    PayloadChannel,
    S2sbsConfig,
)

S2SBS_AR_MAGIC = b"S2SB"
S2S1_GRAMMAR = b"S2S1"
S2S1_SCHEMA_VERSION = 1
# <4s4s B H H H H H H B I H H I I I 32s
S2S1_HEADER_STRUCT = struct.Struct("<4s4sBHHHHHHBIHHIII32s")
PAYLOAD_RECORD_HEADER = struct.Struct("<HH")  # pair_index (u16), payload_len (u16)

_FALSE_AUTHORITY_KEYS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
    "scorer_authority",
)


@dataclass(frozen=True)
class S2sbsArchive:
    """Parsed S2SBS_AR/S2S1 archive with charged byte accounting."""

    config: S2sbsConfig
    payloads: tuple[PayloadChannel, ...]
    base_manifest: dict[str, object]
    metadata: dict[str, object]
    section_lengths: dict[str, int]
    payload_sha256: str
    charged_bytes: int

    @property
    def score_claim(self) -> bool:
        return bool(self.metadata.get("score_claim", False))

    @property
    def total_payload_bytes(self) -> int:
        return sum(len(row.payload) for row in self.payloads)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _base_metadata(extra: Mapping[str, object] | None) -> dict[str, object]:
    incoming = dict(extra or {})
    for key in _FALSE_AUTHORITY_KEYS:
        if incoming.get(key) is True:
            raise ValueError(f"S2S1 metadata must not set {key}=true")
    meta: dict[str, object] = {
        "catalog": 124,
        "archive_grammar": "S2SBS_AR/S2S1 monolithic single-file 0.bin",
        "parser_section_manifest": "parse_archive() -> S2sbsArchive",
        "inflate_runtime_loc_budget": "<=200 LOC",
        "runtime_dep_closure": "torch only; no scorer imports",
        "export_format": "base_payload (LF) + hf_payload (Hermitian-FFT)",
        "score_aware_loss": "alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)",
        "bolt_on_loc_budget": "substrate_engineering L0/L1 scaffold",
        "no_op_detector_planned": "byte mutation must change parsed state and render smoke",
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "scorer_authority": False,
        "charged_bytes_source": "len(s2s1_blob)",
        "audit_memo": ".omx/research/s2sbs_blindspot_audit_20260513.md",
    }
    meta.update(incoming)
    for key in _FALSE_AUTHORITY_KEYS:
        meta[key] = False
    meta["research_only"] = True
    return meta


def _encode_payloads(rows: Sequence[PayloadChannel]) -> bytes:
    seen: set[int] = set()
    out = bytearray()
    for row in sorted(rows, key=lambda item: item.pair_index):
        if row.pair_index in seen:
            raise ValueError(f"duplicate payload pair_index {row.pair_index}")
        seen.add(row.pair_index)
        if len(row.payload) > 0xFFFF:
            raise ValueError(f"payload length {len(row.payload)} exceeds uint16")
        out.extend(PAYLOAD_RECORD_HEADER.pack(int(row.pair_index), len(row.payload)))
        out.extend(row.payload)
    return bytes(out)


def _decode_payloads(blob: bytes) -> tuple[PayloadChannel, ...]:
    rows: list[PayloadChannel] = []
    pos = 0
    while pos < len(blob):
        if pos + PAYLOAD_RECORD_HEADER.size > len(blob):
            raise ValueError("S2S1 payload header truncated")
        pair_index, payload_len = PAYLOAD_RECORD_HEADER.unpack_from(blob, pos)
        pos += PAYLOAD_RECORD_HEADER.size
        if pos + int(payload_len) > len(blob):
            raise ValueError("S2S1 payload body truncated")
        payload = bytes(blob[pos : pos + int(payload_len)])
        pos += int(payload_len)
        rows.append(PayloadChannel(pair_index=int(pair_index), payload=payload))
    return tuple(rows)


def _base_manifest_payload(cfg: S2sbsConfig) -> dict[str, object]:
    return {
        "decoder_kind": "deterministic_base_decoder",
        "base_seed": int(cfg.base_seed),
        "output_pair_contract": "forward(pair_indices)->rgb0,rgb1",
        "score_claim": False,
    }


def pack_archive(
    *,
    config: S2sbsConfig,
    payloads: Sequence[PayloadChannel] = (),
    metadata: Mapping[str, object] | None = None,
) -> bytes:
    """Pack a deterministic S2SBS_AR/S2S1 monolithic archive."""

    for row in payloads:
        if row.pair_index >= config.num_pairs:
            raise ValueError(
                f"payload pair_index {row.pair_index} >= num_pairs {config.num_pairs}"
            )
    payload_blob = _encode_payloads(payloads)
    base_blob = _json_bytes(_base_manifest_payload(config))
    meta_blob = _json_bytes(_base_metadata(metadata))
    digest = hashlib.sha256(payload_blob + base_blob + meta_blob).digest()
    delta_milli = round(float(config.delta_amp_uint8) * 1000.0)
    ecc_milli = round(float(config.ecc_rate) * 1000.0)
    header = S2S1_HEADER_STRUCT.pack(
        S2SBS_AR_MAGIC,
        S2S1_GRAMMAR,
        S2S1_SCHEMA_VERSION,
        int(config.num_pairs),
        int(config.output_height),
        int(config.output_width),
        int(config.hf_blindspot_lf_cutoff_h),
        int(config.hf_blindspot_lf_cutoff_w),
        delta_milli,
        int(config.channel_index),
        int(config.base_seed),
        int(config.payload_bytes_per_pair),
        ecc_milli,
        len(payload_blob),
        len(base_blob),
        len(meta_blob),
        digest,
    )
    return header + payload_blob + base_blob + meta_blob


def parse_archive(blob: bytes) -> S2sbsArchive:
    """Parse S2SBS_AR/S2S1 bytes; fail closed on any malformed section."""

    if len(blob) < S2S1_HEADER_STRUCT.size:
        raise ValueError("S2S1 archive too short for header")
    (
        magic,
        grammar,
        version,
        num_pairs,
        output_height,
        output_width,
        lf_cutoff_h,
        lf_cutoff_w,
        delta_milli,
        channel_idx,
        base_seed,
        payload_bytes_per_pair,
        ecc_milli,
        payload_len,
        base_len,
        meta_len,
        digest,
    ) = S2S1_HEADER_STRUCT.unpack_from(blob, 0)
    if magic != S2SBS_AR_MAGIC:
        raise ValueError(f"bad S2S1 magic: {magic!r}")
    if grammar != S2S1_GRAMMAR:
        raise ValueError(f"bad S2S1 grammar: {grammar!r}")
    if version != S2S1_SCHEMA_VERSION:
        raise ValueError(f"unsupported S2S1 schema version: {version}")
    pos = S2S1_HEADER_STRUCT.size
    expected = pos + int(payload_len) + int(base_len) + int(meta_len)
    if expected != len(blob):
        raise ValueError(f"S2S1 archive size {len(blob)} != header-declared {expected}")
    payload_blob = blob[pos : pos + payload_len]
    pos += payload_len
    base_blob = blob[pos : pos + base_len]
    pos += base_len
    meta_blob = blob[pos : pos + meta_len]

    if hashlib.sha256(payload_blob + base_blob + meta_blob).digest() != digest:
        raise ValueError("S2S1 payload checksum mismatch")

    delta_amp = float(delta_milli) / 1000.0
    ecc_rate = float(ecc_milli) / 1000.0
    # Guard against header rounding overshooting the audit-pinned safety cap.
    if delta_amp > MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT:
        raise ValueError(
            f"S2S1 delta_amp {delta_amp} exceeds joint-safety cap "
            f"{MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT}"
        )
    channel_name = ("R", "G", "B")[int(channel_idx)] if 0 <= int(channel_idx) <= 2 else None
    if channel_name is None:
        raise ValueError(f"S2S1 channel index out of range: {channel_idx}")
    cfg = S2sbsConfig(
        num_pairs=int(num_pairs),
        output_height=int(output_height),
        output_width=int(output_width),
        hf_blindspot_lf_cutoff_h=int(lf_cutoff_h),
        hf_blindspot_lf_cutoff_w=int(lf_cutoff_w),
        delta_amp_uint8=delta_amp,
        payload_channel=channel_name,
        base_seed=int(base_seed),
        payload_bytes_per_pair=int(payload_bytes_per_pair),
        ecc_rate=ecc_rate,
    )
    payloads = _decode_payloads(payload_blob)
    base_manifest = json.loads(base_blob.decode("utf-8"))
    if int(base_manifest.get("base_seed", -1)) != int(cfg.base_seed):
        raise ValueError("S2S1 base_seed manifest mismatch")
    meta = json.loads(meta_blob.decode("utf-8"))
    for key in _FALSE_AUTHORITY_KEYS:
        if meta.get(key) is True:
            raise ValueError(f"S2S1 parsed metadata has forbidden {key}=true")
    meta = _base_metadata(meta)
    return S2sbsArchive(
        config=cfg,
        payloads=payloads,
        base_manifest=base_manifest,
        metadata=meta,
        section_lengths={
            "payloads": int(payload_len),
            "base_manifest": int(base_len),
            "metadata": int(meta_len),
        },
        payload_sha256=digest.hex(),
        charged_bytes=len(blob),
    )


__all__ = [
    "PAYLOAD_RECORD_HEADER",
    "S2S1_GRAMMAR",
    "S2S1_HEADER_STRUCT",
    "S2S1_SCHEMA_VERSION",
    "S2SBS_AR_MAGIC",
    "S2sbsArchive",
    "pack_archive",
    "parse_archive",
]

# Silence usage gates: CONTEST_NUM_PAIRS only used in docstring/derivation.
_ = CONTEST_NUM_PAIRS
