# SPDX-License-Identifier: MIT
"""Deterministic structural inventory for HPM1 mask payloads.

This module deliberately stops at byte-structure custody. It can prove that an
``HPM1`` segment is parsed into header/token/model sections and structurally
re-emitted byte-for-byte, but it does not claim HPAC arithmetic decode,
semantic mask reconstruction, or runtime parity.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

from tac.pr85_bundle import HPM1_HEADER_BYTES, HPM1_MAGIC, parse_hpm1_mask_segment
from tac.repo_io import repo_relative, sha256_bytes, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
KIND = "hpm1_payload_structural_decode_inventory"
HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT = (
    "hpm1_payload_structural_decode_inventory_v1"
)
HPM1_DECODE_REENCODE_BLOCKER_CONTRACT = "hpm1_decode_reencode_blocker_manifest_v1"

_HEADER_FIELD_NAMES = (
    "n_frames",
    "height",
    "width",
    "predictor_count",
    "delta",
    "channels",
    "use_spm",
    "hpac_d_film",
    "tokens_len",
    "hpac_len",
    "ppmd_order",
)


def _path_record(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "bytes": None, "sha256": ""}
    candidate = Path(path)
    exists = candidate.is_file()
    return {
        "path": repo_relative(candidate, REPO_ROOT),
        "exists": exists,
        "bytes": candidate.stat().st_size if exists else None,
        "sha256": sha256_file(candidate) if exists else "",
    }


def _entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for byte in data:
        counts[byte] += 1
    total = float(len(data))
    entropy = 0.0
    for count in counts:
        if not count:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 6)


def _uint32_words(raw: bytes) -> list[int]:
    if len(raw) % 4:
        return []
    return [word[0] for word in struct.iter_unpack("<I", raw)]


def _section(name: str, raw: bytes, *, offset: int, wire_construct: str) -> dict[str, Any]:
    return {
        "name": name,
        "wire_construct": wire_construct,
        "offset": offset,
        "end_offset_exclusive": offset + len(raw),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "entropy_bits_per_byte": _entropy_bits_per_byte(raw),
    }


def _header_fields(raw_header: bytes) -> list[dict[str, Any]]:
    values = struct.unpack_from("<IIIIIIIIIII", raw_header, len(HPM1_MAGIC))
    rows = [
        {
            "name": "magic",
            "offset": 0,
            "bytes": len(HPM1_MAGIC),
            "format": "bytes",
            "value": HPM1_MAGIC.decode("ascii"),
        }
    ]
    for index, (name, value) in enumerate(zip(_HEADER_FIELD_NAMES, values, strict=True)):
        offset = len(HPM1_MAGIC) + index * 4
        rows.append(
            {
                "name": name,
                "offset": offset,
                "bytes": 4,
                "format": "uint32_le",
                "value": int(value),
            }
        )
    return rows


def _common_prefix_bytes(left: bytes, right: bytes) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def _build_hpm1_segment(
    *,
    n_frames: int,
    height: int,
    width: int,
    predictor_count: int,
    delta: int,
    channels: int,
    use_spm: int,
    hpac_d_film: int,
    tokens: bytes,
    hpac: bytes,
    ppmd_order: int,
) -> bytes:
    fields = (
        n_frames,
        height,
        width,
        predictor_count,
        delta,
        channels,
        use_spm,
        hpac_d_film,
        len(tokens),
        len(hpac),
        ppmd_order,
    )
    return HPM1_MAGIC + struct.pack("<IIIIIIIIIII", *map(int, fields)) + tokens + hpac


def _unsupported_wire_constructs() -> list[dict[str, Any]]:
    return [
        {
            "name": "hpac_autoregressive_probability_rows",
            "status": "unsupported_for_parity",
            "wire_construct": (
                "HPACMini logits -> clipped float64 softmax rows -> "
                "constriction.stream.model.Categorical(perfect=False)"
            ),
            "current_evidence": (
                "local prefix replay remains fail-closed before frame-0 decode "
                "completion; this inventory does not generate probability rows"
            ),
            "next_proof": (
                "record first failing/passing probability row with model, "
                "context, dtype, and categorical parameters pinned"
            ),
        },
        {
            "name": "constriction_range_decoder_uint32_queue_replay",
            "status": "unsupported_for_parity",
            "wire_construct": (
                "np.frombuffer(tokens.bin, dtype=np.uint32) consumed by "
                "constriction.stream.queue.RangeDecoder"
            ),
            "current_evidence": (
                "token bytes are inventoried as little-endian uint32 words, "
                "but entropy-model compatibility is not proven"
            ),
            "next_proof": (
                "decode all 600 frames from the exact token bytes and record "
                "decoded token tensor SHA-256"
            ),
        },
        {
            "name": "hpac_context_update_order",
            "status": "unsupported_for_parity",
            "wire_construct": (
                "per-frame/per-group current-token context plus previous-frame "
                "context and optional CausalSPM"
            ),
            "current_evidence": (
                "header proves P/delta/use_spm/hpac_d_film values only; it does "
                "not prove source-identical context mutation"
            ),
            "next_proof": (
                "trace the first two context windows against reference decoded "
                "tokens and prove no drift before byte re-encode"
            ),
        },
        {
            "name": "range_encoder_uint32_reemit",
            "status": "unsupported_for_parity",
            "wire_construct": (
                "semantic tokens -> HPAC probability rows -> constriction range "
                "encoder -> uint32 queue bytes"
            ),
            "current_evidence": (
                "structural HPM1 segment re-emits only by preserving opaque "
                "token bytes; no semantic encoder has reproduced them"
            ),
            "next_proof": (
                "encode decoded tokens back to the exact token stream SHA-256 "
                "and then the exact HPM1 segment SHA-256"
            ),
        },
        {
            "name": "contest_runtime_sidecar_free_hpm1_loader",
            "status": "unsupported_for_parity",
            "wire_construct": (
                "archive member -> inflate runtime HPM1 branch without uncharged "
                "tokens/model sidecars or fallback"
            ),
            "current_evidence": (
                "candidate archive carries the payload bytes, but the runtime "
                "member is still a fail-closed verifier skeleton"
            ),
            "next_proof": (
                "replace the skeleton with a charged runtime loader and prove it "
                "consumes only archive members"
            ),
        },
    ]


def build_hpm1_structural_decode_inventory(
    segment: bytes,
    *,
    payload_member: str = "categorical_payload.bin",
    source_archive: str | Path | None = None,
    source_member: str = "x:mask",
    candidate_archive: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed structural inventory for one HPM1 segment.

    The returned manifest is useful parity-support evidence, not a parity
    report. ``structural_reencode.matches_source_segment`` may be true while
    ``full_decode`` and ``byte_exact_semantic_reencode`` remain false.
    """

    raw_segment = bytes(segment)
    contract = parse_hpm1_mask_segment(raw_segment)
    metadata = dict(contract.metadata)
    token_start = HPM1_HEADER_BYTES
    token_end = token_start + int(metadata["tokens_len"])
    hpac_end = token_end + int(metadata["hpac_len"])
    header = raw_segment[:HPM1_HEADER_BYTES]
    tokens = raw_segment[token_start:token_end]
    hpac = raw_segment[token_end:hpac_end]
    words = _uint32_words(tokens)

    reencoded = _build_hpm1_segment(
        n_frames=int(metadata["n_frames"]),
        height=int(metadata["height"]),
        width=int(metadata["width"]),
        predictor_count=int(metadata["predictor_count"]),
        delta=int(metadata["delta"]),
        channels=int(metadata["channels"]),
        use_spm=int(metadata["use_spm"]),
        hpac_d_film=int(metadata["hpac_d_film"]),
        tokens=tokens,
        hpac=hpac,
        ppmd_order=int(metadata["ppmd_order"]),
    )
    segment_sha = sha256_bytes(raw_segment)
    reencoded_sha = sha256_bytes(reencoded)
    expected_symbol_count = (
        int(metadata["n_frames"]) * int(metadata["height"]) * int(metadata["width"])
    )
    predictor_count = int(metadata["predictor_count"])
    height = int(metadata["height"])
    width = int(metadata["width"])
    patch_rows = height // predictor_count if predictor_count else None
    patch_cols = width // predictor_count if predictor_count else None
    groups_per_frame = (
        (1 + int(metadata["delta"])) * predictor_count - int(metadata["delta"])
        if predictor_count
        else None
    )
    unsupported = _unsupported_wire_constructs()

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "hpm1_structural_decode_inventory_contract": HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "hpm1_structural_byte_inventory",
        "payload_member": payload_member,
        "source_member": source_member,
        "source_archive": _path_record(source_archive),
        "candidate_archive": _path_record(candidate_archive),
        "segment": {
            "codec": "HPM1",
            "bytes": len(raw_segment),
            "sha256": segment_sha,
            "magic": raw_segment[:4].decode("ascii", errors="replace"),
            "header_bytes": HPM1_HEADER_BYTES,
            "tail_bytes": len(raw_segment) - hpac_end,
        },
        "header": {
            "sha256": sha256_bytes(header),
            "fields": _header_fields(header),
            "config": {
                "n_frames": int(metadata["n_frames"]),
                "height": height,
                "width": width,
                "predictor_count": predictor_count,
                "delta": int(metadata["delta"]),
                "channels": int(metadata["channels"]),
                "use_spm": int(metadata["use_spm"]),
                "hpac_d_film": int(metadata["hpac_d_film"]),
                "tokens_len": int(metadata["tokens_len"]),
                "hpac_len": int(metadata["hpac_len"]),
                "ppmd_order": int(metadata["ppmd_order"]),
            },
        },
        "sections": [
            _section("header", header, offset=0, wire_construct="HPM1 fixed header"),
            _section(
                "tokens",
                tokens,
                offset=token_start,
                wire_construct="constriction uint32 range-coded token stream",
            ),
            _section(
                "hpac_ppmd_model",
                hpac,
                offset=token_end,
                wire_construct="PPMd-compressed HPACMini torch state dict",
            ),
        ],
        "token_stream_inventory": {
            "bytes": len(tokens),
            "sha256": sha256_bytes(tokens),
            "uint32_aligned": len(tokens) % 4 == 0,
            "uint32_word_count": len(words),
            "first_words_hex": [f"0x{word:08x}" for word in words[:8]],
            "last_words_hex": [f"0x{word:08x}" for word in words[-8:]],
            "min_word": min(words) if words else None,
            "max_word": max(words) if words else None,
            "zero_word_count": sum(1 for word in words if word == 0),
            "unique_word_count": len(set(words)),
            "range_decoder_input_dtype": "uint32_le",
        },
        "hpac_model_inventory": {
            "bytes": len(hpac),
            "sha256": sha256_bytes(hpac),
            "ppmd_order": int(metadata["ppmd_order"]),
            "runtime_loader_contract": "pyppmd.decompress(..., max_order=4) then torch.load",
            "semantic_state_dict_loaded": False,
            "semantic_state_dict_blocker": "not_loaded_by_structural_inventory",
        },
        "decoded_geometry_contract": {
            "required_full_decode_frames": int(metadata["n_frames"]),
            "decoded_shape_nhw": [
                int(metadata["n_frames"]),
                height,
                width,
            ],
            "expected_decoded_symbol_count": expected_symbol_count,
            "symbol_domain": [0, 4],
            "source_runtime_num_classes": 5,
            "height_width_divisible_by_predictor_count": (
                predictor_count > 0
                and height % predictor_count == 0
                and width % predictor_count == 0
            ),
            "patch_grid": [patch_rows, patch_cols],
            "patches_per_frame": (
                patch_rows * patch_cols
                if patch_rows is not None and patch_cols is not None
                else None
            ),
            "groups_per_frame": groups_per_frame,
        },
        "structural_reencode": {
            "passed": reencoded == raw_segment,
            "matches_source_segment": reencoded == raw_segment,
            "reencoded_segment_bytes": len(reencoded),
            "reencoded_segment_sha256": reencoded_sha,
            "source_segment_sha256": segment_sha,
            "common_prefix_bytes": _common_prefix_bytes(raw_segment, reencoded),
            "scope": "header_plus_opaque_token_and_hpac_section_repack_only",
            "not_semantic_decode_reencode_parity": True,
        },
        "full_decode": {
            "passed": False,
            "frame_count": 0,
            "decoded_masks_sha256": "",
            "refusal_reasons": [
                "hpac_probability_rows_not_byte_closed",
                "constriction_range_decoder_replay_not_proven",
                "hpac_context_update_order_not_proven",
            ],
        },
        "byte_exact_semantic_reencode": {
            "passed": False,
            "byte_exact": False,
            "reencoded_hpm1_sha256": "",
            "refusal_reasons": [
                "full_semantic_decode_missing",
                "range_encoder_uint32_reemit_not_proven",
                "opaque_token_preservation_is_not_decode_reencode_parity",
            ],
        },
        "blocker_manifest": {
            "schema_version": SCHEMA_VERSION,
            "blocker_manifest_contract": HPM1_DECODE_REENCODE_BLOCKER_CONTRACT,
            "status": "blocked_structural_inventory_only",
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_exact_eval_dispatch": False,
            "unsupported_wire_constructs": unsupported,
            "next_required_proof_contract": "pr91_hpm1_decode_reencode_parity_v1",
        },
        "unsupported_wire_constructs": unsupported,
        "next_required_proofs": [
            "load the embedded HPAC model and pin state-dict shape/hash custody",
            "decode all 600 HPM1 frames from the exact token stream on CPU",
            "record decoded token tensor SHA-256 and context-window traces",
            "range-encode decoded tokens back to the exact token stream SHA-256",
            "rebuild the HPM1 segment to the exact source segment SHA-256 from semantic tokens",
            "prove charged runtime loading without sidecars or fallback",
        ],
    }


__all__ = [
    "HPM1_DECODE_REENCODE_BLOCKER_CONTRACT",
    "HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT",
    "KIND",
    "SCHEMA_VERSION",
    "build_hpm1_structural_decode_inventory",
]
