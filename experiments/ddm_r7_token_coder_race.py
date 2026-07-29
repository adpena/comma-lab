# SPDX-License-Identifier: MIT
"""Run the DDM R7 lossless token/vehicle stream race on frozen checkpoints.

No scorer is imported or run.  Checkpoints are opened read-only, all reported
coder bytes are materialized frames, and every row must decode back to the
exact input before it is admitted to the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import platform
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for entry in (str(REPO), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.ddm_r7_logistic_mix import (
    decode_logistic_mix,
    encode_logistic_mix,
)
from experiments.ddm_r7_token_coder import (
    CODEC_IDS,
    decode_token_codes,
    encode_token_codes,
    factor_mode_delta,
    frame_accounting,
    pack_nibbles,
)
from tac.optimization import arith_selfcomp_rate_coders as selfcomp
from tac.optimization import ddm_tr1_runtime as tr1
from tac.optimization import repair_entropy_coder_runtime_adapters as repair_adapters
from tac.optimization.arith_selfcomp_rate_coders import (
    decode_bellard_class_mixing,
    decode_g4_decoder_context,
    decode_willems_ctw,
    encode_bellard_class_mixing,
    encode_g4_decoder_context,
    encode_willems_ctw,
)
from tac.optimization.repair_entropy_coder_runtime_adapters import (
    ans_rans_prototype_decode,
    ans_rans_prototype_encode,
)

try:
    import brotli
except ImportError:  # pragma: no cover - explicit unavailable row
    brotli = None  # type: ignore[assignment]


SCHEMA: Final = "ddm_r7_token_coder_race.v1"
POINTER_LOCAL: Final = "0.1910828242 [contest-CPU] UNMOVED"
COMPETITIVE_TARGET: Final = "official PR130 0.172141 [contest-CUDA], displayed 0.172"
RATE_DENOMINATOR: Final = 37_545_489
CEILING_0172: Final = 190_334
CEILING_015: Final = 157_294
GENERIC_MAGIC: Final = b"R7GF"
GENERIC_HEADER: Final = struct.Struct("<4sBQ32s")
GENERIC_CODECS: Final = {
    1: "brotli_q11",
    2: "lzma1",
    3: "r7_lzma2_64k_joint_inspired",
}
POOL_MAGIC: Final = b"R7PL"
POOL_HEADER: Final = struct.Struct("<4sB")
POOL_LENGTH: Final = struct.Struct("<I")


class R7RaceError(ValueError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_custody(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _raw_lzma_encode(payload: bytes) -> bytes:
    return lzma.compress(
        payload,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
    )


def _bounded_lzma_decode(
    payload: bytes,
    *,
    format: int,
    filters: list[dict[str, Any]] | None,
    expected_length: int,
) -> bytes:
    try:
        decoder = lzma.LZMADecompressor(format=format, filters=filters)
        restored = decoder.decompress(payload, max_length=expected_length + 1)
    except lzma.LZMAError as exc:
        raise R7RaceError("invalid or oversized LZMA stream") from exc
    if len(restored) != expected_length or not decoder.eof or decoder.unused_data:
        raise R7RaceError("LZMA stream length or termination differs")
    return restored


def _generic_encode(codec_id: int, raw: bytes) -> bytes:
    if codec_id == 1:
        if brotli is None:
            raise R7RaceError("Brotli dependency unavailable")
        coded = bytes(brotli.compress(raw, quality=11))
    elif codec_id == 2:
        coded = _raw_lzma_encode(raw)
    elif codec_id == 3:
        coded = lzma.compress(
            raw,
            format=lzma.FORMAT_XZ,
            filters=[
                {
                    "id": lzma.FILTER_LZMA2,
                    "dict_size": 64 * 1024,
                    "lc": 0,
                    "lp": 1,
                    "pb": 0,
                    "mode": lzma.MODE_NORMAL,
                    "nice_len": 273,
                    "mf": lzma.MF_BT4,
                    "depth": 0,
                }
            ],
        )
    else:  # pragma: no cover - caller is closed
        raise R7RaceError("generic codec id unsupported")
    return (
        GENERIC_HEADER.pack(
            GENERIC_MAGIC,
            codec_id,
            len(raw),
            hashlib.sha256(raw).digest(),
        )
        + coded
    )


def _generic_decode(frame: bytes) -> bytes:
    if len(frame) <= GENERIC_HEADER.size:
        raise R7RaceError("generic frame truncated")
    magic, codec_id, raw_length, digest = GENERIC_HEADER.unpack_from(frame)
    if magic != GENERIC_MAGIC or codec_id not in GENERIC_CODECS:
        raise R7RaceError("generic frame header differs")
    coded = frame[GENERIC_HEADER.size :]
    if codec_id == 1:
        if brotli is None:
            raise R7RaceError("Brotli dependency unavailable")
        raw = bytes(brotli.decompress(coded))
    elif codec_id == 2:
        raw = _bounded_lzma_decode(
            coded,
            format=lzma.FORMAT_RAW,
            filters=[
                {
                    "id": lzma.FILTER_LZMA1,
                    "preset": 9 | lzma.PRESET_EXTREME,
                }
            ],
            expected_length=raw_length,
        )
    else:
        raw = _bounded_lzma_decode(
            coded,
            format=lzma.FORMAT_XZ,
            filters=None,
            expected_length=raw_length,
        )
    if len(raw) != raw_length or hashlib.sha256(raw).digest() != digest:
        raise R7RaceError("generic frame parse-back differs")
    if _generic_encode(codec_id, raw) != frame:
        raise R7RaceError("generic frame is noncanonical")
    return raw


def _pool_frame(frames: list[bytes]) -> bytes:
    if not frames or len(frames) > 255:
        raise R7RaceError("pool frame count is outside uint8")
    return (
        POOL_HEADER.pack(POOL_MAGIC, len(frames))
        + b"".join(POOL_LENGTH.pack(len(frame)) for frame in frames)
        + b"".join(frames)
    )


def _pool_split(frame: bytes) -> list[bytes]:
    if len(frame) < POOL_HEADER.size:
        raise R7RaceError("pool frame is truncated")
    magic, count = POOL_HEADER.unpack_from(frame)
    if magic != POOL_MAGIC or count == 0:
        raise R7RaceError("pool frame header differs")
    table_end = POOL_HEADER.size + count * POOL_LENGTH.size
    if len(frame) < table_end:
        raise R7RaceError("pool length table is truncated")
    lengths = [POOL_LENGTH.unpack_from(frame, POOL_HEADER.size + index * POOL_LENGTH.size)[0] for index in range(count)]
    if any(length <= 0 for length in lengths) or table_end + sum(lengths) != len(frame):
        raise R7RaceError("pool member lengths do not close")
    frames: list[bytes] = []
    offset = table_end
    for length in lengths:
        frames.append(frame[offset : offset + length])
        offset += length
    if _pool_frame(frames) != frame:
        raise R7RaceError("pool frame is noncanonical")
    return frames


def _measure_pool(
    raws: list[bytes],
    encoders: list[Callable[[bytes], bytes]],
    decoders: list[Callable[[bytes], bytes]],
    *,
    implementations: list[str],
    pr130_lesson: str,
) -> dict[str, Any]:
    if not (len(raws) == len(encoders) == len(decoders) == len(implementations)):
        raise R7RaceError("pool measurement arity differs")
    start = time.monotonic()
    members = [encoder(raw) for raw, encoder in zip(raws, encoders, strict=True)]
    frame = _pool_frame(members)
    encode_seconds = time.monotonic() - start
    start = time.monotonic()
    parsed = _pool_split(frame)
    restored = [decoder(member) for member, decoder in zip(parsed, decoders, strict=True)]
    decode_seconds = time.monotonic() - start
    if restored != raws:
        raise R7RaceError("pool parse-back differs")
    return {
        "available": True,
        "encode_seconds": round(encode_seconds, 6),
        "decode_seconds_including_canonical_reencode": round(
            decode_seconds,
            6,
        ),
        "frame_sha256": _sha256(frame),
        "framed_bytes": len(frame),
        "implementations": implementations,
        "member_count": len(members),
        "member_framed_bytes": [len(member) for member in members],
        "parseback_exact": True,
        "pr130_lesson": pr130_lesson,
        "raw_bytes": sum(len(raw) for raw in raws),
        "raw_sha256": [_sha256(raw) for raw in raws],
    }


def _measure_bytes(
    raw: bytes,
    encoder: Callable[[bytes], bytes],
    decoder: Callable[[bytes], bytes],
    *,
    implementation: str,
    pr130_lesson: str,
) -> dict[str, Any]:
    start = time.monotonic()
    frame = encoder(raw)
    encode_seconds = time.monotonic() - start
    start = time.monotonic()
    restored = decoder(frame)
    decode_seconds = time.monotonic() - start
    if restored != raw:
        raise R7RaceError(f"{implementation} parse-back differs")
    second = encoder(raw)
    if second != frame:
        raise R7RaceError(f"{implementation} is nondeterministic")
    return {
        "available": True,
        "encode_seconds": round(encode_seconds, 6),
        "decode_seconds_including_canonical_reencode": round(decode_seconds, 6),
        "frame_sha256": _sha256(frame),
        "framed_bytes": len(frame),
        "implementation": implementation,
        "parseback_exact": True,
        "pr130_lesson": pr130_lesson,
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
    }


def _byte_coders(include_slow: bool) -> dict[str, tuple[Callable[[bytes], bytes], Callable[[bytes], bytes], str]]:
    rows: dict[str, tuple[Callable[[bytes], bytes], Callable[[bytes], bytes], str]] = {
        "rans_o0_static": (
            ans_rans_prototype_encode,
            ans_rans_prototype_decode,
            "PR130 range backend is reference only; complete owned rANS bytes reraced",
        ),
        "g4_causal_byte_prefix": (
            encode_g4_decoder_context,
            decode_g4_decoder_context,
            "owned previous-byte plus bit-prefix context; not token-lattice left/up G4",
        ),
        "bayes_linear_mix_4expert": (
            encode_bellard_class_mixing,
            decode_bellard_class_mixing,
            "owned Bayesian probability mix control; not the delegated logistic mixer",
        ),
        "logistic_mix_4expert_fixedpoint": (
            encode_logistic_mix,
            decode_logistic_mix,
            "delegated Mahoney-lite family: true fixed-point logit-domain mix with four owned causal experts",
        ),
        "brotli_q11": (
            lambda raw: _generic_encode(1, raw),
            _generic_decode,
            "existing E4 baseline; PR130 did not Brotli-code its token stream",
        ),
        "lzma1_extreme": (
            lambda raw: _generic_encode(2, raw),
            _generic_decode,
            "existing E4 raw-LZMA1 fallback; distinct from PR130 LZMA2 recipe",
        ),
        "r7_lzma2_64k_joint_inspired": (
            lambda raw: _generic_encode(3, raw),
            _generic_decode,
            "R7-owned 64KiB LZMA2 variant inspired by PR130 joint-XZ pooling; no claim of exact filter custody",
        ),
    }
    if include_slow:
        rows["willems_ctw_depth8"] = (
            encode_willems_ctw,
            decode_willems_ctw,
            "universal context-tree candidate; PR130 context lesson tested without its weights",
        )
    return rows


def _token_pr130_lesson(codec: str) -> str:
    if codec in {
        "kt_prev1",
        "kt_o8_prev5_backoff",
        "cae_inspired_identity_inter",
        "smevr",
    }:
        return (
            "PR130 causal past+spatial lesson translated to owned DDM mode/base/lattice "
            "contexts; P64 topology, weights, embeddings, and constants rejected"
        )
    if codec == "huffman_nibble":
        return "PR130 canonical-Huffman lesson reraced with the complete 16-byte DDM code-length table charged"
    if codec.startswith("rans"):
        return "PR130 range backend is reference only; complete owned rANS frame reraced"
    if codec == "lzma1":
        return "existing E4 fallback; PR130 exact XZ/LZMA2 is a separate arm"
    if codec == "brotli11":
        return "existing E4 baseline; PR130 did not Brotli-code its token stream"
    return "no PR130 mechanism transferred"


def _mode_base_and_delta_raw(codes: np.ndarray, levels: int) -> tuple[bytes, bytes]:
    base, delta = factor_mode_delta(codes, levels)
    return pack_nibbles(base), pack_nibbles(delta)


def _renderer_raw(parsed: tr1.ParsedTR1Packet) -> tuple[bytes, bytes, bytes]:
    mask_flat = np.concatenate([mask.astype(np.uint8).reshape(-1) for mask in parsed.masks])
    mask = np.packbits(mask_flat, bitorder="big").tobytes()
    mods = (
        np.concatenate(
            [
                item
                for gain, bias in zip(parsed.gains, parsed.biases, strict=True)
                for item in (
                    np.asarray(gain, dtype=np.float32).reshape(-1),
                    np.asarray(bias, dtype=np.float32).reshape(-1),
                )
            ]
        )
        .astype(">f2")
        .tobytes()
    )
    joint = struct.pack("<II", mask_flat.size, len(mods) // 2) + mask + mods
    return mask, mods, joint


def _token_entropy(codes: np.ndarray, levels: int) -> dict[str, Any]:
    base, delta = factor_mode_delta(codes, levels)
    pair_count, height, width, channels = delta.shape
    frame_values = height * width
    rows: list[dict[str, Any]] = []
    total_unconditional_bits = 0.0
    total_prev_bits = 0.0
    for channel in range(channels):
        values = np.ascontiguousarray(delta[..., channel]).reshape(-1)
        counts = np.bincount(values, minlength=levels)
        probabilities = counts[counts > 0] / values.size
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        total_unconditional_bits += entropy * values.size
        transitions = np.zeros((levels, levels), dtype=np.int64)
        for pair_index in range(1, pair_count):
            previous = values[(pair_index - 1) * frame_values : pair_index * frame_values]
            current = values[pair_index * frame_values : (pair_index + 1) * frame_values]
            np.add.at(transitions, (previous, current), 1)
        conditional_bits = 0.0
        for source in range(levels):
            row = transitions[source]
            total = int(row.sum())
            if total:
                p = row[row > 0] / total
                conditional_bits += float(-(p * np.log2(p)).sum()) * total
        initial = values[:frame_values]
        initial_counts = np.bincount(initial, minlength=levels)
        initial_p = initial_counts[initial_counts > 0] / initial.size
        conditional_bits += float(-(initial_p * np.log2(initial_p)).sum()) * initial.size
        total_prev_bits += conditional_bits
        rows.append(
            {
                "channel": channel,
                "empirical_plugin_h_delta_bits_per_symbol": entropy,
                "empirical_plugin_h_delta_given_previous_bits_per_symbol": (conditional_bits / values.size),
                "p_delta_zero": float(counts[0] / values.size),
            }
        )
    # One checkpoint supplies one token realization and no realized partition
    # member.  The exact universal conditional-entropy lower bound is therefore
    # zero.  The subtraction below is explicitly an assumption-labelled proxy.
    partition_description_assumption_bytes = 173_616.498
    return {
        "channels": rows,
        "mode_base_packed_bytes": (base.size + 1) // 2,
        "mode_base_symbols": base.size,
        "residual_empirical_plugin_unconditional_bytes": (total_unconditional_bits / 8.0),
        "residual_empirical_plugin_prev1_bytes": total_prev_bits / 8.0,
        "argmax_equivalence_conditional_bound": {
            "exact_universal_lower_bound_bytes": 0,
            "identifiability": (
                "NOT_IDENTIFIABLE_FROM_CHECKPOINT: token dump has no realized partition array "
                "and only one realization per checkpoint"
            ),
            "proxy_inequality": (
                "under the assumed first-order source only: H_assumed(R|B,P) >= H_assumed(R|B) - H_assumed(P)"
            ),
            "proxy_assumption": (
                "ASSUME the residual empirical plug-in first-order model is the source "
                "ensemble and H_assumed(P) is no more than pp1 GT-partition KT "
                "173616.498 B; excludes the counted mode base; not a dump theorem or score row"
            ),
            "proxy_assumed_average_fiber_floor_bytes": max(
                0.0,
                total_prev_bits / 8.0 - partition_description_assumption_bytes,
            ),
        },
    }


def _checkpoint_row(
    path: Path,
    *,
    include_slow: bool,
    partial: dict[str, Any] | None = None,
    save_partial: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    before_bytes, before_sha = _file_custody(path)
    compiled = tr1.compile_archive_from_checkpoint(path)
    parsed_archive = tr1.parse_archive(compiled.archive_bytes)
    parsed = parsed_archive.packet
    codes = parsed.token_codes
    levels = int(parsed.selector["token_quant_levels"])
    base_raw, delta_raw = _mode_base_and_delta_raw(codes, levels)
    planes = [pack_nibbles(codes[..., channel]) for channel in range(codes.shape[-1])]
    mask_raw, mods_raw, renderer_joint = _renderer_raw(parsed)
    selector_raw = json.dumps(
        dict(parsed.selector),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata_joint = struct.pack("<I", len(renderer_joint)) + renderer_joint + selector_raw

    working = {} if partial is None else partial
    token_rows = working.setdefault("production_token_frames", {})
    if not isinstance(token_rows, dict):
        raise R7RaceError("resume token rows are malformed")
    for codec in CODEC_IDS:
        if codec in token_rows:
            continue
        start = time.monotonic()
        frame = encode_token_codes(codes, levels=levels, codec=codec)
        encode_seconds = time.monotonic() - start
        start = time.monotonic()
        restored = decode_token_codes(frame)
        decode_seconds = time.monotonic() - start
        if not np.array_equal(restored, codes):
            raise R7RaceError(f"token codec {codec} parse-back differs")
        accounting = frame_accounting(frame)
        token_rows[codec] = {
            "available": True,
            "base_bytes": accounting.base_bytes,
            "decode_seconds_including_canonical_reencode": round(decode_seconds, 6),
            "delta_bytes": accounting.delta_bytes,
            "encode_seconds": round(encode_seconds, 6),
            "frame_sha256": accounting.sha256,
            "framed_bytes": accounting.framed_bytes,
            "implementation": f"experiments.ddm_r7_token_coder:{codec}",
            "parseback_exact": True,
            "pr130_lesson": _token_pr130_lesson(codec),
            "raw_token_bytes": accounting.raw_token_bytes,
            "representation_pool": "mode_delta",
        }
        if save_partial is not None:
            save_partial(working)

    byte_rows = working.setdefault("byte_coders", {})
    if not isinstance(byte_rows, dict):
        raise R7RaceError("resume byte rows are malformed")
    coder_specs = _byte_coders(include_slow)
    for coder_name, (encoder, decoder, pr130_lesson) in coder_specs.items():
        stream_rows = byte_rows.setdefault(coder_name, {})
        if not isinstance(stream_rows, dict):
            raise R7RaceError("resume stream rows are malformed")
        token_plane_group = stream_rows.setdefault(
            "token_planes",
            {
                "non_additive_alternative_to": "mode_delta",
                "planes": [],
            },
        )
        if not isinstance(token_plane_group, dict) or not isinstance(
            token_plane_group.get("planes"),
            list,
        ):
            raise R7RaceError("resume token-plane rows are malformed")
        plane_rows = token_plane_group["planes"]
        if len(plane_rows) > len(planes):
            raise R7RaceError("resume token-plane count exceeds checkpoint")
        for plane in planes[len(plane_rows) :]:
            plane_rows.append(
                _measure_bytes(
                    plane,
                    encoder,
                    decoder,
                    implementation=coder_name,
                    pr130_lesson=pr130_lesson,
                )
            )
            if save_partial is not None:
                save_partial(working)
        if "pool" not in token_plane_group:
            token_plane_group["pool"] = _measure_pool(
                planes,
                [encoder] * len(planes),
                [decoder] * len(planes),
                implementations=[coder_name] * len(planes),
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "mode_base" not in stream_rows:
            stream_rows["mode_base"] = _measure_bytes(
                base_raw,
                encoder,
                decoder,
                implementation=coder_name,
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "mode_delta" not in stream_rows:
            stream_rows["mode_delta"] = _measure_bytes(
                delta_raw,
                encoder,
                decoder,
                implementation=coder_name,
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "mode_delta_pool" not in stream_rows:
            stream_rows["mode_delta_pool"] = _measure_pool(
                [base_raw, delta_raw],
                [encoder, encoder],
                [decoder, decoder],
                implementations=[coder_name, coder_name],
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "supermask" not in stream_rows:
            stream_rows["supermask"] = _measure_bytes(
                mask_raw,
                encoder,
                decoder,
                implementation=coder_name,
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "mods" not in stream_rows:
            stream_rows["mods"] = _measure_bytes(
                mods_raw,
                encoder,
                decoder,
                implementation=coder_name,
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "supermask_mods_joint" not in stream_rows:
            stream_rows["supermask_mods_joint"] = _measure_bytes(
                renderer_joint,
                encoder,
                decoder,
                implementation=coder_name,
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "supermask_mods_split_pool" not in stream_rows:
            stream_rows["supermask_mods_split_pool"] = _measure_pool(
                [mask_raw, mods_raw],
                [encoder, encoder],
                [decoder, decoder],
                implementations=[coder_name, coder_name],
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "selector" not in stream_rows:
            stream_rows["selector"] = _measure_bytes(
                selector_raw,
                encoder,
                decoder,
                implementation=coder_name,
                pr130_lesson=pr130_lesson,
            )
            if save_partial is not None:
                save_partial(working)
        if "supermask_mods_selector_joint" not in stream_rows:
            stream_rows["supermask_mods_selector_joint"] = _measure_bytes(
                metadata_joint,
                encoder,
                decoder,
                implementation=coder_name,
                pr130_lesson=(pr130_lesson + "; PR130 joint-XZ/model-pool lesson tested as a non-additive alternative"),
            )
            if save_partial is not None:
                save_partial(working)

    mixed_rows = working.setdefault("mixed_coder_pools", {})
    if not isinstance(mixed_rows, dict):
        raise R7RaceError("resume mixed-pool rows are malformed")

    def best_for(stream: str, *, plane_index: int | None = None) -> str:
        def framed_bytes(coder_name: str) -> int:
            if plane_index is None:
                row = byte_rows[coder_name][stream]
            else:
                row = byte_rows[coder_name]["token_planes"]["planes"][plane_index]
            return int(row["framed_bytes"])

        return min(byte_rows, key=lambda name: (framed_bytes(name), name))

    def materialize_mixed_pool(
        key: str,
        raws: list[bytes],
        coder_names: list[str],
        lesson: str,
    ) -> None:
        if key in mixed_rows:
            return
        specs = [coder_specs[name] for name in coder_names]
        mixed_rows[key] = _measure_pool(
            raws,
            [spec[0] for spec in specs],
            [spec[1] for spec in specs],
            implementations=coder_names,
            pr130_lesson=lesson,
        )
        if save_partial is not None:
            save_partial(working)

    materialize_mixed_pool(
        "mode_delta",
        [base_raw, delta_raw],
        [best_for("mode_base"), best_for("mode_delta")],
        "per-stream non-additive minimum with exact counted pool wrapper",
    )
    materialize_mixed_pool(
        "token_planes",
        planes,
        [best_for("token_planes", plane_index=index) for index in range(len(planes))],
        "per-plane non-additive minimum with exact counted pool wrapper",
    )
    materialize_mixed_pool(
        "renderer_split",
        [mask_raw, mods_raw],
        [best_for("supermask"), best_for("mods")],
        "independent supermask/mods minima with exact counted pool wrapper",
    )
    materialize_mixed_pool(
        "metadata_renderer_split_selector",
        [mask_raw, mods_raw, selector_raw],
        [best_for("supermask"), best_for("mods"), best_for("selector")],
        "independent supermask/mods/selector minima with exact counted pool wrapper",
    )
    materialize_mixed_pool(
        "metadata_renderer_joint_selector",
        [renderer_joint, selector_raw],
        [best_for("supermask_mods_joint"), best_for("selector")],
        "best joint renderer plus selector with exact counted pool wrapper",
    )

    before_ledger = {item["name"]: int(item["bytes"]) for item in tr1.section_ledger(parsed)}
    old_sections = sum(before_ledger.values())
    fixed_archive_overhead = len(compiled.archive_bytes) - old_sections
    token_rank = sorted(
        (
            {
                "coder": name,
                "framed_bytes": int(row["framed_bytes"]),
                "source": "production_token_frame",
            }
            for name, row in token_rows.items()
        ),
        key=lambda item: (item["framed_bytes"], item["coder"]),
    )
    for name, streams in byte_rows.items():
        for representation, pool_key in (
            ("mode_delta", "mode_delta_pool"),
            ("token_planes", "token_planes"),
        ):
            pool = streams[pool_key] if pool_key == "mode_delta_pool" else streams[pool_key]["pool"]
            token_rank.append(
                {
                    "coder": name,
                    "framed_bytes": int(pool["framed_bytes"]),
                    "source": f"generic_{representation}_counted_pool",
                }
            )
    for representation in ("mode_delta", "token_planes"):
        token_rank.append(
            {
                "coder": "+".join(mixed_rows[representation]["implementations"]),
                "framed_bytes": int(mixed_rows[representation]["framed_bytes"]),
                "source": f"mixed_coder_{representation}_counted_pool",
            }
        )
    token_rank.sort(key=lambda item: (item["framed_bytes"], item["coder"]))
    renderer_rank: list[dict[str, Any]] = []
    for name, streams in byte_rows.items():
        renderer_rank.extend(
            [
                {
                    "coder": name,
                    "framed_bytes": int(streams["supermask_mods_joint"]["framed_bytes"]),
                    "source": "joint_raw_stream",
                },
                {
                    "coder": name,
                    "framed_bytes": int(streams["supermask_mods_split_pool"]["framed_bytes"]),
                    "source": "split_counted_pool",
                },
            ]
        )
    renderer_rank.append(
        {
            "coder": "+".join(mixed_rows["renderer_split"]["implementations"]),
            "framed_bytes": int(mixed_rows["renderer_split"]["framed_bytes"]),
            "source": "mixed_coder_split_counted_pool",
        }
    )
    renderer_rank.sort(key=lambda item: (item["framed_bytes"], item["coder"]))
    selector_rank = sorted(
        (
            {
                "coder": name,
                "framed_bytes": int(streams["selector"]["framed_bytes"]),
            }
            for name, streams in byte_rows.items()
        ),
        key=lambda item: (item["framed_bytes"], item["coder"]),
    )
    metadata_rank = [
        *(
            {
                "coder": name,
                "framed_bytes": int(streams["supermask_mods_selector_joint"]["framed_bytes"]),
                "source": "joint_raw_stream",
            }
            for name, streams in byte_rows.items()
        ),
        {
            "coder": "+".join(mixed_rows["metadata_renderer_split_selector"]["implementations"]),
            "framed_bytes": int(mixed_rows["metadata_renderer_split_selector"]["framed_bytes"]),
            "source": "mixed_coder_three_member_counted_pool",
        },
        {
            "coder": "+".join(mixed_rows["metadata_renderer_joint_selector"]["implementations"]),
            "framed_bytes": int(mixed_rows["metadata_renderer_joint_selector"]["framed_bytes"]),
            "source": "mixed_coder_joint_renderer_plus_selector_counted_pool",
        },
    ]
    metadata_rank.sort(key=lambda item: (item["framed_bytes"], item["coder"]))
    selected_metadata_bytes = metadata_rank[0]["framed_bytes"]

    def rank_stream(stream: str) -> list[dict[str, Any]]:
        return sorted(
            (
                {
                    "coder": name,
                    "framed_bytes": int(streams[stream]["framed_bytes"]),
                }
                for name, streams in byte_rows.items()
            ),
            key=lambda item: (item["framed_bytes"], item["coder"]),
        )

    token_plane_ranks = [
        sorted(
            (
                {
                    "coder": name,
                    "framed_bytes": int(streams["token_planes"]["planes"][plane_index]["framed_bytes"]),
                }
                for name, streams in byte_rows.items()
            ),
            key=lambda item: (item["framed_bytes"], item["coder"]),
        )
        for plane_index in range(len(planes))
    ]
    composed = (
        fixed_archive_overhead + token_rank[0]["framed_bytes"] + selected_metadata_bytes + before_ledger["pose_stub"]
    )
    after_bytes, after_sha = _file_custody(path)
    if (after_bytes, after_sha) != (before_bytes, before_sha):
        raise R7RaceError("checkpoint changed during read-only race")
    return {
        "checkpoint": {
            "bytes": before_bytes,
            "path": str(path),
            "sha256": before_sha,
        },
        "competitive_arithmetic": {
            "ceiling_0.172_bytes": CEILING_0172,
            "ceiling_0.15_bytes": CEILING_015,
            "composed_section_estimate_bytes": composed,
            "deficit_to_0.172_ceiling_bytes": composed - CEILING_0172,
            "deficit_to_0.15_ceiling_bytes": composed - CEILING_015,
            "exact_byteclose_status": "OWED_EG1_INTEGRATION_NOT_AN_ARCHIVE_ROW",
            "fixed_e1_archive_overhead_bytes": fixed_archive_overhead,
            "metadata_pool_selection": metadata_rank[0]["source"],
            "metadata_selected_bytes": selected_metadata_bytes,
            "rate_score_term_at_composed_estimate": 25.0 * composed / RATE_DENOMINATOR,
        },
        "entropy": _token_entropy(codes, levels),
        "existing_e1": {
            "archive_bytes": len(compiled.archive_bytes),
            "archive_sha256": compiled.archive_sha256,
            "section_ledger": tr1.section_ledger(parsed),
        },
        "ranked": {
            "metadata_complete": metadata_rank,
            "mode_base": rank_stream("mode_base"),
            "mode_delta": rank_stream("mode_delta"),
            "mods": rank_stream("mods"),
            "renderer_supermask_mods": renderer_rank,
            "selector": selector_rank,
            "supermask": rank_stream("supermask"),
            "token_planes": token_plane_ranks,
            "token_mode_delta": token_rank,
        },
        "raw_streams": {
            "mode_base": {"bytes": len(base_raw), "sha256": _sha256(base_raw)},
            "mode_delta": {"bytes": len(delta_raw), "sha256": _sha256(delta_raw)},
            "mods": {"bytes": len(mods_raw), "sha256": _sha256(mods_raw)},
            "selector": {"bytes": len(selector_raw), "sha256": _sha256(selector_raw)},
            "supermask": {"bytes": len(mask_raw), "sha256": _sha256(mask_raw)},
            "supermask_mods_joint": {
                "bytes": len(renderer_joint),
                "sha256": _sha256(renderer_joint),
            },
            "supermask_mods_selector_joint": {
                "bytes": len(metadata_joint),
                "sha256": _sha256(metadata_joint),
            },
            "token_planes": [{"bytes": len(plane), "sha256": _sha256(plane)} for plane in planes],
        },
        "rows": {
            "byte_coders": byte_rows,
            "mixed_coder_pools": mixed_rows,
            "production_token_frames": token_rows,
        },
        "selector": dict(parsed.selector),
    }


def _json_payload(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    payload = _json_payload(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _immutable_json(path: Path, value: Any) -> None:
    """Atomically create a stage snapshot, accepting only identical reuse."""

    payload = _json_payload(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise R7RaceError("immutable stage checkpoint already differs")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise R7RaceError("immutable stage checkpoint raced with different bytes") from None
    finally:
        temporary.unlink(missing_ok=True)


def _stage_path(
    resume_path: Path,
    stage_index: int,
    checkpoint_sha256: str,
) -> Path:
    return resume_path.with_name(f"{resume_path.stem}.stage-{stage_index:03d}-{checkpoint_sha256[:12]}.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="stage-atomic progress file; created when absent and resumed when present",
    )
    parser.add_argument(
        "--skip-slow-ctw",
        action="store_true",
        help="emit no CTW row; forbidden for the final R7 receipt",
    )
    args = parser.parse_args()
    started = time.time()
    paths = [path.resolve() for path in args.checkpoint]
    path_labels = [str(path) for path in paths]
    if len(paths) < 2 or len(set(path_labels)) != len(paths):
        raise R7RaceError("race requires at least two distinct early/latest checkpoints")
    evidence_root = (REPO / ".omx" / "research").resolve()
    output_path = args.output.resolve()
    resume_path = args.resume_from.resolve()
    for target in (output_path, resume_path):
        if not target.is_relative_to(evidence_root):
            raise R7RaceError("output and progress must stay under .omx/research")
        if target in paths:
            raise R7RaceError("evidence target aliases an input checkpoint")
    if output_path == resume_path:
        raise R7RaceError("output and progress paths must differ")
    include_slow = not args.skip_slow_ctw
    source_paths = {
        "coder": REPO / "experiments/ddm_r7_token_coder.py",
        "logistic_mix": REPO / "experiments/ddm_r7_logistic_mix.py",
        "race": Path(__file__).resolve(),
        "range_coder": REPO / "src/tac/lossless/range_coder.py",
        "repair_adapters": Path(repair_adapters.__file__).resolve(),
        "repo_io": REPO / "src/tac/repo_io.py",
        "selfcomp_coders": Path(selfcomp.__file__).resolve(),
        "tr1_runtime": Path(tr1.__file__).resolve(),
    }
    source_hashes = {name: _file_custody(path)[1] for name, path in source_paths.items()}
    runtime_fingerprint = {
        "brotli_module_sha256": (None if brotli is None else _file_custody(Path(brotli.__file__).resolve())[1]),
        "brotli_version": (None if brotli is None else getattr(brotli, "__version__", "unknown")),
        "lzma_module_sha256": _file_custody(Path(lzma.__file__).resolve())[1],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python": sys.version,
    }
    checkpoint_custody = {
        label: {
            "bytes": observed[0],
            "sha256": observed[1],
        }
        for label, observed in ((label, _file_custody(path)) for path, label in zip(paths, path_labels, strict=True))
    }
    expected_stage_paths = [
        _stage_path(
            resume_path,
            index,
            str(checkpoint_custody[label]["sha256"]),
        )
        for index, label in enumerate(path_labels, start=1)
    ]
    if output_path in expected_stage_paths or resume_path in expected_stage_paths:
        raise R7RaceError("evidence output aliases a stage checkpoint path")
    if resume_path.exists():
        progress = json.loads(resume_path.read_text())
        if (
            progress.get("schema") != f"{SCHEMA}.progress.v2"
            or progress.get("checkpoint_paths") != path_labels
            or progress.get("checkpoint_custody") != checkpoint_custody
            or progress.get("include_slow_ctw") is not include_slow
            or progress.get("runtime_fingerprint") != runtime_fingerprint
            or progress.get("source_sha256") != source_hashes
        ):
            raise R7RaceError("resume progress contract differs from this exact race")
    else:
        progress = {
            "checkpoint_custody": checkpoint_custody,
            "checkpoint_paths": path_labels,
            "completed": {},
            "created_at_unix": started,
            "include_slow_ctw": include_slow,
            "partials": {},
            "runtime_fingerprint": runtime_fingerprint,
            "schema": f"{SCHEMA}.progress.v2",
            "source_sha256": source_hashes,
            "stage_checkpoints": [],
        }
        _atomic_json(resume_path, progress)
    completed = progress.setdefault("completed", {})
    partials = progress.setdefault("partials", {})
    if not isinstance(completed, dict) or not isinstance(partials, dict):
        raise R7RaceError("resume progress rows are malformed")
    stage_checkpoints = progress.setdefault("stage_checkpoints", [])
    if not isinstance(stage_checkpoints, list):
        raise R7RaceError("resume stage-checkpoint ledger is malformed")
    for item in stage_checkpoints:
        stage_path = Path(str(item["path"]))
        if not stage_path.is_file() or not stage_path.resolve().is_relative_to(evidence_root):
            raise R7RaceError("preserved stage checkpoint is absent")
    rows: list[dict[str, Any]] = []
    for path, label in zip(paths, path_labels, strict=True):
        observed = _file_custody(path)
        expected_custody = checkpoint_custody[label]
        expected = (
            int(expected_custody["bytes"]),
            str(expected_custody["sha256"]),
        )
        if observed != expected:
            raise R7RaceError("checkpoint custody changed after progress contract creation")
        if label in completed:
            row = completed[label]
            completed_expected = (
                int(row["checkpoint"]["bytes"]),
                str(row["checkpoint"]["sha256"]),
            )
            if observed != completed_expected:
                raise R7RaceError("completed checkpoint custody changed before resume")
            rows.append(row)
            continue

        def save_partial(value: dict[str, Any], *, key: str = label) -> None:
            partials[key] = value
            progress["last_stage"] = key
            progress["updated_at_unix"] = time.time()
            _atomic_json(resume_path, progress)

        row = _checkpoint_row(
            path,
            include_slow=include_slow,
            partial=partials.get(label),
            save_partial=save_partial,
        )
        completed[label] = row
        partials.pop(label, None)
        progress["last_completed_stage"] = label
        progress["updated_at_unix"] = time.time()
        stage_index = len(rows) + 1
        stage_path = _stage_path(
            resume_path,
            stage_index,
            str(checkpoint_custody[label]["sha256"]),
        )
        stage_entry = {
            "checkpoint_path": label,
            "path": str(stage_path),
            "stage_index": stage_index,
        }
        stage_checkpoints.append(stage_entry)
        _immutable_json(stage_path, progress)
        _atomic_json(resume_path, progress)
        rows.append(row)
    stage_checkpoint_custody = [
        {
            **item,
            "bytes": _file_custody(Path(str(item["path"])))[0],
            "sha256": _file_custody(Path(str(item["path"])))[1],
        }
        for item in stage_checkpoints
    ]
    receipt = {
        "authority": {
            "competitive_target": COMPETITIVE_TARGET,
            "evidence_axis": "[macOS-CPU advisory, rate-only]",
            "local_custody_pointer": POINTER_LOCAL,
            "pointer_moved": False,
            "promotion_eligible": False,
            "research_only": True,
            "score_claim": False,
        },
        "ceilings": {
            "0.15": CEILING_015,
            "0.172": CEILING_0172,
            "basis": "authority-provided whole-archive ceilings; rows recompute deficits only",
        },
        "checkpoints": rows,
        "environment": runtime_fingerprint,
        "hard_boundaries": {
            "exact_eval_run": False,
            "live_checkpoint_writes": False,
            "n600_scorer_run": False,
            "paid_dispatch": False,
        },
        "pr130_lesson_status": (
            "HARVEST_SIGNAL_ONLY_COMPLETE; transferred math reraced on DDM bytes; "
            "PR topology, weights, embeddings, constants, and archive bytes not adopted"
        ),
        "pr130_lessons": {
            "archive_anchor": (
                "[external official contest-CUDA] PR130 stored member 190952 B "
                "plus 100 B STORED-ZIP overhead = 191052 B archive; "
                "intake evidence only, archive bytes not held by this arm"
            ),
            "raced": [
                "causal decoded-past and spatial contexts via owned KT/CAE/SMEVR/G4/mix",
                "canonical Huffman with complete code-length table charged",
                "R7-owned 64KiB LZMA2 variant inspired by PR130 joint-XZ pooling",
                "integer arithmetic, digest, and canonical parse-back discipline",
            ],
            "rejected_or_parked": [
                "P64/190-group HPAC topology, weights, 600x8 embeddings, and constants: borrowed-poison",
                "direct constriction transfer: float bridge, dependency, and 2197.6s intake decode",
                "exact PR130 LZMA2 filters: source custody absent; no exact-recipe claim",
                "context-order rANS and tANS: not implemented; current rANS rows are o0 only",
                "nonidentity xi-advected MPEG-4 INTER-CAE: charter discrepancy, still owed",
                "Rice temporal coding: no geometric-tailed DDM selector/mod trajectory in this packet",
                "per-row lossy bit depth/normalization: REALIZED-GATE-OWED",
            ],
        },
        "resumability": {
            "checkpoint_custody": checkpoint_custody,
            "checkpoint_count": len(rows),
            "per_coder_and_stream_atomic_progress": True,
            "progress_path": str(resume_path),
            "source_sha256": source_hashes,
            "stage_checkpoint_custody": stage_checkpoint_custody,
            "stage_checkpoints_preserved": len(stage_checkpoint_custody) == len(rows),
        },
        "schema": SCHEMA,
        "wall_seconds": time.time() - started,
    }
    _atomic_json(output_path, receipt)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "checkpoint_rankings": [row["ranked"]["token_mode_delta"] for row in rows],
                "wall_seconds": receipt["wall_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
