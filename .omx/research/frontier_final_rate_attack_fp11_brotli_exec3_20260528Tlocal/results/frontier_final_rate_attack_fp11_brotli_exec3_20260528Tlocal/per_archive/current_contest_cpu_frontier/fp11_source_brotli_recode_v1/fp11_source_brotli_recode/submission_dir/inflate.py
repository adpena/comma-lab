#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Inflate FEC6/FECa with an archive-charged DQS1 decoder-q selector (V14-V2 substitution)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from codec import (  # type: ignore[import-not-found]
    CONV4_STORAGE_PERMS,
    DECODER_BLOB_LEN,
    DECODER_BYTE_MAPS,
    DECODER_STORAGE_ORDER,
    DECODER_STREAM_ENDS,
    decode_mapped_u8,
    decompress_brotli_streams,
    parse_archive,
)
from frame_selector import PALETTE_MODE_IDS, apply_selector_to_frames, unpack_selector_indices  # type: ignore[import-not-found]
from frame_selector import _blue_tile as selector_blue_tile  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]


CAMERA_H, CAMERA_W = 874, 1164
OUTER_MAGIC = b"FP11"
DQS1_MAGIC = b"DQS1"
DQS1_FRAME_POLICY_BY_CODE = {1: "pair_all_frames", 2: "segnet_last_frame_only"}
DQS1_PAIR_ENCODING_BY_CODE = {0: "raw_u16", 1: "sorted_gap_uleb"}

COMPACT_FAMILY_IDS = {"identity": 0, "rgb_bias": 1, "blue_chroma": 2, "roll": 3}
COMPACT_FAMILY_BY_ID = {value: key for key, value in COMPACT_FAMILY_IDS.items()}
FEC5_FIXED_K8_MODE_IDS = (
    "none",
    "frame0_blue_chroma_amp_3",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_luma_bias_-1",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_luma_bias_-4",
)
FEC5_FIXED_K8_CODE_BITS = ("00", "01", "100", "101", "1100", "1101", "1110", "1111")
FEC5_FIXED_K8_DECODE = {bits: code for code, bits in enumerate(FEC5_FIXED_K8_CODE_BITS)}
# INNOVATION 1: FEC6 K=16 active mode palette over the 31-mode FES1 transform space (NEW BOLT-ON on top of
# PR #101; PR #101 has no per-pair selector mechanism). The per-pair selector picks one of K=16 deterministic
# frame-0 transforms (identity / luma bias / RGB bias / blue chroma amp / 1-pixel roll). K=16 is empirically
# the minimum active palette that retains the top per-pair scorer-targeted transforms while keeping the
# entropy-coded selector cheap. The "vs internal FEC5 K=8 predecessor" comparison is *internal lineage*, not
# PR #101 lineage. Empirically attributable to the contest-CPU delta of -0.000794 vs PR #101.
FEC6_FIXED_K16_MODE_IDS = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)
# INNOVATION 2: fixed-Huffman k=16 codebook on selector indices (NEW BOLT-ON; sister technique to PR #101's
# canonical Huffman for the *latent sidecar*, but applied to a NEW layer — selector indices — with a FIXED
# code, so no per-archive header bytes are spent declaring the code table). The naive 4-bits/pair fixed cost
# would be 4 * 600 = 300 bytes; the fixed Huffman code compacts the 600-pair selector to a 243-byte bitstream
# (1944 bits = 3.24 bits/pair) wrapped in a 6-byte header for a 249-byte wire payload (3.32 bits/pair).
# Code lengths range 2 .. 8 bits; shortest codes assigned to most-frequent modes (00 = "none" most common;
# 01 = "frame0_blue_chroma_amp_3"; 100 = "frame0_rgb_bias_m2_p1_p1"). The selector payload is byte-appended
# inside member `x` *outside* PR #101's Brotli envelope (local FP11 wrapper grammar); it is NOT itself
# Brotli-coded, and the ZIP member `x` is stored uncompressed (compression_type=0 / ZIP_STORED).
FEC6_FIXED_K16_CODE_BITS = (
    "00",
    "1100",
    "01",
    "111010",
    "11010",
    "111011",
    "111100",
    "100",
    "111101",
    "11011",
    "1111110",
    "111110",
    "11111110",
    "101",
    "11100",
    "11111111",
)
FEC6_FIXED_K16_DECODE = {bits: code for code, bits in enumerate(FEC6_FIXED_K16_CODE_BITS)}


def parse_signed_token(token: str) -> int:
    if token.startswith("p"):
        return int(token[1:])
    if token.startswith("m"):
        return -int(token[1:])
    return int(token)


def mode_spec_from_static_mode_id(mode_id: str) -> tuple[str, tuple[int, ...], int]:
    if mode_id == "none":
        return ("identity", (), 0)
    frame_index = 1 if mode_id.startswith("frame1_") else 0
    base = mode_id.replace("frame1_", "frame0_", 1)
    if base.startswith("frame0_luma_bias_"):
        value = int(base.removeprefix("frame0_luma_bias_"))
        return ("rgb_bias", (value, value, value), frame_index)
    if base.startswith("frame0_rgb_bias_"):
        params = tuple(parse_signed_token(part) for part in base.removeprefix("frame0_rgb_bias_").split("_"))
        if len(params) != 3:
            raise ValueError(f"bad RGB compact selector mode {mode_id!r}")
        return ("rgb_bias", params, frame_index)
    if base.startswith("frame0_blue_chroma_amp_"):
        return ("blue_chroma", (int(base.removeprefix("frame0_blue_chroma_amp_")),), frame_index)
    if base.startswith("frame0_roll_dx"):
        suffix = base.removeprefix("frame0_roll_dx")
        dx_token, dy_token = suffix.split("_dy", 1)
        return ("roll", (int(dx_token), int(dy_token)), frame_index)
    raise ValueError(f"unsupported static compact selector mode {mode_id!r}")


def unpack_compact_selector_codes(
    selector_payload: bytes,
) -> tuple[list[int], tuple[tuple[str, tuple[int, ...], int], ...]]:
    if len(selector_payload) < 8:
        raise ValueError("compact selector truncated before header")
    if selector_payload[:4] == b"FEC5":
        n_pairs = struct.unpack_from("<H", selector_payload, 4)[0]
        codes = unpack_fec5_fixed_huffman_codes(selector_payload[6:], n_pairs=n_pairs)
        specs = tuple(mode_spec_from_static_mode_id(mode_id) for mode_id in FEC5_FIXED_K8_MODE_IDS)
        return codes, specs
    if selector_payload[:4] == b"FEC6":
        n_pairs = struct.unpack_from("<H", selector_payload, 4)[0]
        codes = unpack_fec6_fixed_huffman_codes(selector_payload[6:], n_pairs=n_pairs)
        specs = tuple(mode_spec_from_static_mode_id(mode_id) for mode_id in FEC6_FIXED_K16_MODE_IDS)
        return codes, specs
    if selector_payload[:4] == b"FECa":
        # V14-V2: Cascade A FEC10 hybrid adaptive-blend dispatch (per RECOVERY-1 commit 39c76755b)
        from fec10_hybrid_decoder import decode_fec10_hybrid_selector  # noqa: PLC0415
        codes = decode_fec10_hybrid_selector(selector_payload)
        specs = tuple(mode_spec_from_static_mode_id(mode_id) for mode_id in FEC6_FIXED_K16_MODE_IDS)
        return codes, specs
    if selector_payload[:4] not in {b"FEC2", b"FEC3"}:
        raise ValueError(f"compact selector magic mismatch: {selector_payload[:4]!r}")
    magic = selector_payload[:4]
    n_pairs, bits_per_symbol, n_specs = struct.unpack_from("<HBB", selector_payload, 4)
    if not (1 <= bits_per_symbol <= 4):
        raise ValueError(f"unsupported compact selector bit width: {bits_per_symbol}")
    if n_specs > (1 << bits_per_symbol):
        raise ValueError("compact selector palette is wider than encoded symbols")
    pos = 8
    specs: list[tuple[str, tuple[int, ...], int]] = []
    if magic == b"FEC3":
        for _ in range(n_specs):
            if pos + 2 > len(selector_payload):
                raise ValueError("compact selector static palette table truncated")
            tag, value = struct.unpack_from("<BB", selector_payload, pos)
            pos += 2
            if tag == 0:
                if value >= len(PALETTE_MODE_IDS):
                    raise ValueError(f"compact selector static palette index {value} outside runtime palette")
                specs.append(mode_spec_from_static_mode_id(PALETTE_MODE_IDS[value]))
                continue
            if tag != 1:
                raise ValueError(f"unsupported compact selector spec tag {tag}")
            if pos + 5 > len(selector_payload):
                raise ValueError("compact selector dynamic spec table truncated")
            family_id, p0, p1, p2, frame_index = struct.unpack_from("<BbbbB", selector_payload, pos)
            pos += 5
            family = COMPACT_FAMILY_BY_ID.get(int(family_id))
            if family is None:
                raise ValueError(f"unsupported compact selector family id {family_id}")
            if frame_index not in (0, 1):
                raise ValueError(f"unsupported compact selector frame index {frame_index}")
            if family == "identity":
                params: tuple[int, ...] = ()
            elif family == "rgb_bias":
                params = (int(p0), int(p1), int(p2))
            elif family == "blue_chroma":
                params = (int(p0),)
            elif family == "roll":
                params = (int(p0), int(p1))
            specs.append((family, params, int(frame_index)))
    else:
        for _ in range(n_specs):
            if pos + 5 > len(selector_payload):
                raise ValueError("compact selector spec table truncated")
            family_id, p0, p1, p2, frame_index = struct.unpack_from("<BbbbB", selector_payload, pos)
            pos += 5
            family = COMPACT_FAMILY_BY_ID.get(int(family_id))
            if family is None:
                raise ValueError(f"unsupported compact selector family id {family_id}")
            if frame_index not in (0, 1):
                raise ValueError(f"unsupported compact selector frame index {frame_index}")
            if family == "identity":
                params = ()
            elif family == "rgb_bias":
                params = (int(p0), int(p1), int(p2))
            elif family == "blue_chroma":
                params = (int(p0),)
            elif family == "roll":
                params = (int(p0), int(p1))
            specs.append((family, params, int(frame_index)))
    payload = selector_payload[pos:]
    codes: list[int] = []
    bit_pos = 0
    for _ in range(n_pairs):
        code = 0
        for shift in range(bits_per_symbol):
            absolute = bit_pos + shift
            byte_index = absolute // 8
            if byte_index >= len(payload):
                raise ValueError("compact selector bitstream truncated")
            code |= ((payload[byte_index] >> (absolute % 8)) & 1) << shift
        if code >= len(specs):
            raise ValueError(f"compact selector code {code} outside palette")
        codes.append(int(code))
        bit_pos += bits_per_symbol
    if (bit_pos + 7) // 8 != len(payload):
        raise ValueError("compact selector has trailing payload bytes")
    return codes, tuple(specs)


def unpack_fec5_fixed_huffman_codes(payload: bytes, *, n_pairs: int) -> list[int]:
    codes: list[int] = []
    prefix = ""
    bit_pos = 0
    max_bits = len(payload) * 8
    while len(codes) < n_pairs:
        if bit_pos >= max_bits:
            raise ValueError("FEC5 compact selector bitstream truncated")
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        code = FEC5_FIXED_K8_DECODE.get(prefix)
        if code is not None:
            codes.append(int(code))
            prefix = ""
            continue
        if len(prefix) > 4:
            raise ValueError("FEC5 compact selector contains invalid prefix code")
    if prefix:
        raise ValueError("FEC5 compact selector ended mid-symbol")
    for trailing in range(bit_pos, max_bits):
        if (payload[trailing // 8] >> (7 - (trailing % 8))) & 1:
            raise ValueError("FEC5 compact selector has non-zero padding bits")
    return codes


# INNOVATION 2 (decode side): FEC6 fixed-Huffman bitstream decoder. Reads MSB-first, matches longest-prefix
# against FEC6_FIXED_K16_DECODE, refuses trailing non-zero padding bits + invalid prefix > 8 bits. Fully
# deterministic; no on-device search at inflate.
def unpack_fec6_fixed_huffman_codes(payload: bytes, *, n_pairs: int) -> list[int]:
    codes: list[int] = []
    prefix = ""
    bit_pos = 0
    max_bits = len(payload) * 8
    while len(codes) < n_pairs:
        if bit_pos >= max_bits:
            raise ValueError("FEC6 compact selector bitstream truncated")
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        code = FEC6_FIXED_K16_DECODE.get(prefix)
        if code is not None:
            codes.append(int(code))
            prefix = ""
            continue
        if len(prefix) > 8:
            raise ValueError("FEC6 compact selector contains invalid prefix code")
    if prefix:
        raise ValueError("FEC6 compact selector ended mid-symbol")
    for trailing in range(bit_pos, max_bits):
        if (payload[trailing // 8] >> (7 - (trailing % 8))) & 1:
            raise ValueError("FEC6 compact selector has non-zero padding bits")
    return codes


def unpack_pr101_selector(
    selector_payload: bytes,
) -> tuple[str, list[int], tuple[tuple[str, tuple[int, ...], int], ...]]:
    if selector_payload.startswith(b"FES1"):
        return "static", unpack_selector_indices(selector_payload), ()
    if selector_payload.startswith((b"FEC2", b"FEC3", b"FEC5", b"FEC6", b"FECa")):
        # V14-V2: extended dispatch to include FECa (Cascade A FEC10 hybrid; per RECOVERY-1 commit 39c76755b)
        selector_codes, specs = unpack_compact_selector_codes(selector_payload)
        return "compact", selector_codes, specs
    raise ValueError(f"unsupported selector payload magic: {selector_payload[:4]!r}")


def parse_dqs1_payload(payload: bytes) -> dict[str, object]:
    if len(payload) < 11:
        raise ValueError("DQS1 payload truncated")
    if payload[:4] != DQS1_MAGIC:
        raise ValueError(f"DQS1 magic mismatch: {payload[:4]!r}")
    mode_byte = int(payload[4])
    frame_policy_code = mode_byte & 0x0F
    pair_encoding_code = mode_byte >> 4
    frame_policy = DQS1_FRAME_POLICY_BY_CODE.get(frame_policy_code)
    if frame_policy is None:
        raise ValueError(f"unsupported DQS1 frame policy code: {frame_policy_code}")
    pair_encoding = DQS1_PAIR_ENCODING_BY_CODE.get(pair_encoding_code)
    if pair_encoding is None:
        raise ValueError(f"unsupported DQS1 pair encoding code: {pair_encoding_code}")
    storage_index = int(payload[5])
    q_offset = struct.unpack_from("<H", payload, 6)[0]
    delta = struct.unpack_from("<b", payload, 8)[0]
    count = struct.unpack_from("<H", payload, 9)[0]

    def pack_uleb128(value: int) -> bytes:
        if value < 0:
            raise ValueError("DQS1 ULEB value must be non-negative")
        out = bytearray()
        remaining = int(value)
        while True:
            byte = remaining & 0x7F
            remaining >>= 7
            if remaining:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                return bytes(out)

    def unpack_uleb128(offset: int) -> tuple[int, int]:
        value = 0
        shift = 0
        start = offset
        while offset < len(payload):
            byte = payload[offset]
            value |= (byte & 0x7F) << shift
            offset += 1
            if byte & 0x80 == 0:
                if payload[start:offset] != pack_uleb128(value):
                    raise ValueError("noncanonical DQS1 ULEB gap")
                return value, offset
            shift += 7
            if shift > 28:
                raise ValueError("DQS1 ULEB gap is too large")
        raise ValueError("truncated DQS1 ULEB gap")

    if pair_encoding == "raw_u16":
        expected_len = 11 + count * 2
        if len(payload) != expected_len:
            raise ValueError(f"DQS1 payload length mismatch: expected {expected_len}, got {len(payload)}")
        pair_indices = [
            int(struct.unpack_from("<H", payload, 11 + offset * 2)[0])
            for offset in range(count)
        ]
    elif pair_encoding == "sorted_gap_uleb":
        pair_indices = []
        previous = None
        offset = 11
        for _ in range(count):
            value, offset = unpack_uleb128(offset)
            if previous is not None and value <= 0:
                raise ValueError("DQS1 sorted_gap_uleb gap must be positive")
            pair_index = value if previous is None else previous + value
            if pair_index > 65535:
                raise ValueError("DQS1 selected pair outside u16 range")
            pair_indices.append(int(pair_index))
            previous = int(pair_index)
        if offset != len(payload):
            raise ValueError("DQS1 payload has trailing pair-index bytes")
    else:
        raise ValueError(f"unsupported DQS1 pair encoding {pair_encoding!r}")
    if pair_indices != sorted(pair_indices):
        raise ValueError("DQS1 selected pair indices must be sorted")
    if len(set(pair_indices)) != len(pair_indices):
        raise ValueError("DQS1 selected pair indices contain duplicates")
    for pair_index in pair_indices:
        if pair_index >= 600:
            raise ValueError(f"DQS1 selected pair outside FEC6 range: {pair_index}")
    return {
        "frame_policy": frame_policy,
        "frame_policy_code": frame_policy_code,
        "mode_byte": mode_byte,
        "pair_encoding": pair_encoding,
        "pair_encoding_code": pair_encoding_code,
        "storage_index": storage_index,
        "q_offset": q_offset,
        "delta": delta,
        "pair_indices": tuple(pair_indices),
    }


def parse_pr101_frame_selector_archive(
    bin_bytes: bytes,
) -> tuple[bytes, str, list[int], tuple[tuple[str, tuple[int, ...], int], ...], dict[str, object] | None]:
    if len(bin_bytes) < 10:
        raise ValueError("PR101 frame-selector wrapper truncated before header")
    magic = bin_bytes[:4]
    if magic != OUTER_MAGIC:
        raise ValueError(f"PR101 frame-selector magic mismatch: {magic!r}")
    pos = 4
    (source_len,) = struct.unpack_from("<I", bin_bytes, pos)
    pos += 4
    source_payload = bin_bytes[pos : pos + source_len]
    pos += source_len
    if len(source_payload) != source_len:
        raise ValueError("PR101 source payload truncated")
    if pos + 2 > len(bin_bytes):
        raise ValueError("PR101 frame-selector wrapper truncated before selector length")
    (selector_len,) = struct.unpack_from("<H", bin_bytes, pos)
    pos += 2
    selector_payload = bin_bytes[pos : pos + selector_len]
    pos += selector_len
    if len(selector_payload) != selector_len:
        raise ValueError("FES1 selector payload truncated")
    dqs1_packet = None
    if pos != len(bin_bytes):
        dqs1_packet = parse_dqs1_payload(bin_bytes[pos:])
        pos = len(bin_bytes)
    selector_kind, selector_codes, selector_specs = unpack_pr101_selector(selector_payload)
    return source_payload, selector_kind, selector_codes, selector_specs, dqs1_packet


def _tensor_index_for_stored_q_offset(q_offset: int, shape: tuple[int, ...], storage_index: int) -> tuple[int, ...]:
    if len(shape) == 4:
        storage_perm = tuple(int(axis) for axis in CONV4_STORAGE_PERMS[int(storage_index)])
        stored_shape = tuple(shape[axis] for axis in storage_perm)
        stored_index = np.unravel_index(int(q_offset), stored_shape)
        tensor_index = [0] * len(shape)
        for storage_axis, tensor_axis in enumerate(storage_perm):
            tensor_index[int(tensor_axis)] = int(stored_index[storage_axis])
        return tuple(tensor_index)
    return tuple(int(value) for value in np.unravel_index(int(q_offset), shape))


def apply_dqs1_patch_to_decoder_state(
    decoder_sd: dict[str, torch.Tensor],
    source_payload: bytes,
    dqs1_packet: dict[str, object],
    meta: dict[str, object],
) -> dict[str, torch.Tensor]:
    decoder_blob = source_payload[:DECODER_BLOB_LEN]
    raw = decompress_brotli_streams(decoder_blob, len(DECODER_STREAM_ENDS))
    probe = HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    )
    items = list(probe.state_dict().items())
    target_storage_index = int(dqs1_packet["storage_index"])
    q_offset = int(dqs1_packet["q_offset"])
    delta = int(dqs1_packet["delta"])
    pos = 0
    for storage_index in DECODER_STORAGE_ORDER:
        name, tensor = items[int(storage_index)]
        shape = tuple(int(value) for value in tuple(tensor.shape))
        numel = int(tensor.numel())
        q_start = pos
        q_end = q_start + numel
        scale_start = q_end
        scale_end = scale_start + 2
        if scale_end > len(raw):
            raise ValueError("DQS1 decoder raw stream truncated")
        if int(storage_index) != target_storage_index:
            pos = scale_end
            continue
        if q_offset < 0 or q_offset >= numel:
            raise ValueError(f"DQS1 q_offset {q_offset} outside tensor {name} numel={numel}")
        q_bytes = np.frombuffer(raw[q_start:q_end], dtype=np.uint8).copy()
        q_values = decode_mapped_u8(q_bytes, DECODER_BYTE_MAPS.get(int(storage_index), "zig"))
        q_before = int(q_values[q_offset])
        q_after = int(np.clip(q_before + delta, -127, 127))
        if q_after == q_before:
            raise ValueError("DQS1 q-domain mutation is clipped/no-op")
        scale = float(np.frombuffer(raw, dtype=np.float16, count=1, offset=scale_start)[0])
        tensor_index = _tensor_index_for_stored_q_offset(q_offset, shape, int(storage_index))
        out = {key: value.clone() for key, value in decoder_sd.items()}
        patched = out[name].clone()
        patched[tensor_index] = patched[tensor_index] + float(q_after - q_before) * scale
        out[name] = patched
        return out
    raise ValueError(f"DQS1 storage_index {target_storage_index} not found in decoder schema")
def apply_dynamic_mode(frame_chw: torch.Tensor, spec: tuple[str, tuple[int, ...], int]) -> torch.Tensor:
    family, params, _frame_index = spec
    if family == "identity":
        return frame_chw
    out = frame_chw.clone()
    if family == "rgb_bias":
        delta = torch.tensor(params, dtype=out.dtype, device=out.device).view(3, 1, 1)
        return out + delta
    if family == "blue_chroma":
        amp = float(params[0])
        _channels, height, width = out.shape
        tile = selector_blue_tile(height, width, device=out.device, dtype=out.dtype)
        out[0].add_(tile * amp)
        out[2].sub_(tile * amp)
        return out
    if family == "roll":
        dx, dy = int(params[0]), int(params[1])
        return torch.roll(out, shifts=(dy, dx), dims=(1, 2))
    raise ValueError(f"unsupported compact selector family {family!r}")


# INNOVATION 3: offline per-pair selector decision (vs on-device search at inflate). Selector indices are
# precomputed against the SegNet + PoseNet response surface during candidate enumeration. The inflate path
# is fully deterministic: the codes have already been chosen; this function just applies the matching
# frame-0 (or frame-1) transform per pair. No scorer weights loaded at inflate time per strict-scorer-rule.
def apply_pr101_selector_to_frames(
    frames_bchw: torch.Tensor,
    selector_kind: str,
    selector_codes: list[int],
    selector_specs: tuple[tuple[str, tuple[int, ...], int], ...],
    *,
    pair_start: int,
) -> torch.Tensor:
    if selector_kind == "static":
        return apply_selector_to_frames(frames_bchw, selector_codes, pair_start=pair_start)
    if selector_kind != "compact":
        raise ValueError(f"unsupported selector kind {selector_kind!r}")
    if frames_bchw.shape[0] % 2 != 0:
        raise ValueError("compact selector expects complete frame pairs")
    out = frames_bchw.clone()
    n_pairs = frames_bchw.shape[0] // 2
    for offset in range(n_pairs):
        pair_index = pair_start + offset
        if pair_index >= len(selector_codes):
            raise ValueError(f"selector has {len(selector_codes)} entries; missing pair {pair_index}")
        spec = selector_specs[int(selector_codes[pair_index])]
        family, _params, frame_index = spec
        if family == "identity":
            continue
        frame_offset = offset * 2 + int(frame_index)
        out[frame_offset] = apply_dynamic_mode(out[frame_offset], spec)
    return out.clamp_(0.0, 255.0).round_()


def inflate(src_bin: str, dst_raw: str) -> int:
    source_payload, selector_kind, selector_codes, selector_specs, dqs1_packet = parse_pr101_frame_selector_archive(Path(src_bin).read_bytes())
    decoder_sd, latents, meta = parse_archive(source_payload)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # INLINE_DEVICE_FORK_OK:contest_submission_inflate_runtime_byte_stable_per_pr107_apogee_precedent_canonical_select_inflate_device_helper_would_require_vendored_sister_module_inflating_reviewability_loc_budget_further
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    mutated_decoder = None
    selected_pairs: set[int] = set()
    frame_policy = "pair_all_frames"
    if dqs1_packet is not None:
        mutated_sd = apply_dqs1_patch_to_decoder_state(decoder_sd, source_payload, dqs1_packet, meta)
        mutated_decoder = HNeRVDecoder(
            latent_dim=meta["latent_dim"],
            base_channels=meta["base_channels"],
            eval_size=tuple(meta["eval_size"]),
        ).to(device)
        mutated_decoder.load_state_dict(mutated_sd)
        mutated_decoder.eval()
        selected_pairs = {int(value) for value in dqs1_packet["pair_indices"]}
        frame_policy = str(dqs1_packet["frame_policy"])

    latents = latents.to(device)
    n_pairs = int(meta["n_pairs"])
    if len(selector_codes) != n_pairs:
        raise SystemExit(f"selector has {len(selector_codes)} pairs; archive requires exactly {n_pairs}")
    eval_h, eval_w = meta["eval_size"]

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            if mutated_decoder is not None:
                selected_local = [pair - i for pair in range(i, j) if pair in selected_pairs]
                if selected_local:
                    selected_local_tensor = torch.tensor(
                        selected_local,
                        dtype=torch.long,
                        device=decoded.device,
                    )
                    decoded = decoded.clone()
                    mutated_decoded = mutated_decoder(latents[i:j])
                    if frame_policy == "pair_all_frames":
                        decoded[selected_local_tensor] = mutated_decoded[selected_local_tensor]
                    elif frame_policy == "segnet_last_frame_only":
                        decoded[selected_local_tensor, 1] = mutated_decoded[selected_local_tensor, 1]
                    else:
                        raise ValueError(f"unsupported DQS1 frame policy {frame_policy!r}")
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            rounded = up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W).clamp(0, 255).round()
            rounded = apply_pr101_selector_to_frames(
                rounded,
                selector_kind,
                selector_codes,
                selector_specs,
                pair_start=i,
            )
            frames = rounded.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
