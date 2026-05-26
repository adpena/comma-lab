#!/usr/bin/env python3
"""Fail-closed JCSP runtime probe for ``submissions/robust_current``.

This module is intentionally stdlib-only.  It runs from ``inflate.sh`` before
any rendering branch so an archive carrying ``jcsp.bin`` cannot silently fall
through to an unrelated runtime path.  The default production mode only clears
submission-runtime consumption for the narrow real AQ rawvideo contract:
expected ``.raw`` stream names, arithmetic-static AQv1/AQc1 payloads,
nonempty RGB24-aligned bytes, no fixtures, no no-ops, and no extra streams.
Exact CUDA dispatch remains blocked in metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JCSP_RUNTIME_BRIDGE_PROBE_SCHEMA = "jcsp_submission_runtime_bridge_probe_v1"
JCSP_ARCHIVE_MEMBER_NAME = "jcsp.bin"
JCSP_REQUIRED_SUBMISSION_RUNTIME = "submissions/robust_current"
JCSP_RUNTIME_BRIDGE_PATH = "submissions/robust_current/jcsp_runtime_bridge.py"
JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER = (
    "submissions_robust_current_jcsp_bin_consumption_missing"
)
JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER = (
    "jcsp_local_skeleton_not_submission_runtime_container"
)
JCSP_RUNTIME_OUTPUT_CONTRACT_SCHEMA = "jcsp_submission_runtime_output_contract_v1"
JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER = "jcsp_runtime_raw_output_parity_missing"
JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA = (
    "jcsp_runtime_raw_output_parity_contract_v1"
)
JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA = (
    "jcsp_runtime_raw_output_parity_proof_v1"
)
JCSP_RUNTIME_RAW_OUTPUT_CONSUMER_READINESS_SCHEMA = (
    "jcsp_runtime_raw_output_consumer_readiness_v1"
)
JCSP_RUNTIME_RAW_OUTPUT_EMISSION_SCHEMA = "jcsp_runtime_raw_output_emission_v1"
JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE = "jcsp_runtime_bridge_emitted_rawvideo"
JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_SOURCE = (
    "jcsp_runtime_bridge_fixture_raw_passthrough"
)
JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_BLOCKER = (
    "jcsp_fixture_raw_passthrough_not_dispatch_proof"
)
JCSP_RUNTIME_NOOP_PACKET_BLOCKER = "jcsp_zero_stream_noop_packet"
JCSP_RUNTIME_DECODER_ADAPTER_SCHEMA = "jcsp_runtime_decoder_adapter_contract_v1"
JCSP_RUNTIME_DECODED_STREAM_SCHEMA = "jcsp_runtime_decoded_stream_v1"
JCSP_RUNTIME_REAL_DECODER_BLOCKER = "jcsp_real_stream_decoder_adapter_missing"
JCSP_RUNTIME_AQ_RAWVIDEO_ADAPTER_ID = "jcsp_arithmetic_uint8_rawvideo_adapter_v1"
JCSP_RUNTIME_AQ_RAWVIDEO_STREAM_KIND = "arithmetic_static_uint8_rawvideo_v1"
JCSP_RUNTIME_PREFLIGHT_ADAPTER_SCHEMA = "jcsp_runtime_preflight_adapter_v1"
JCSP_RUNTIME_REAL_AQ_PREFLIGHT_ADAPTER_ID = (
    "jcsp_real_aq_rawvideo_runtime_preflight_adapter_v1"
)
JCSP_RUNTIME_AQ_RAWVIDEO_FORMAT_CONTRACT = (
    "JCSP stream codec_kind=0, stream name equals expected .raw path, "
    "payload magic AQv1/AQc1, AQ num_symbols=256, AQ offset=0, decoded "
    "symbols are written as nonempty RGB24 rawvideo bytes with length divisible by 3"
)
JCSP_MAGIC = b"JCSP"
JCSK_MAGIC = b"JCSK"
JCSP_VERSION = 1
JCSK_VERSION = 1
KIND_ARITHMETIC_STATIC = 0
KIND_BALLE_HYPERPRIOR = 1
KIND_RAW_PASSTHROUGH = 2
EXIT_JCSP_MEMBER_REFUSED = 44

AQ_MAGIC = b"AQv1"
AQC_MAGIC = b"AQc1"
AQ_VERSION = 1
AQC_VERSION = 1
AQ_RAWVIDEO_NUM_SYMBOLS = 256
AQ_RAWVIDEO_OFFSET = 0

_AC_PRECISION = 32
_AC_TOP = 1 << _AC_PRECISION
_AC_HALF = _AC_TOP >> 1
_AC_QUARTER = _AC_TOP >> 2
_AC_THREE_QUARTER = _AC_HALF + _AC_QUARTER
_AC_MASK = _AC_TOP - 1

_PAYLOAD_MAGICS_BY_CODEC_KIND: dict[int, tuple[bytes, ...]] = {
    KIND_ARITHMETIC_STATIC: (b"AQv1", b"AQc1"),
    KIND_BALLE_HYPERPRIOR: (b"BHv1",),
}


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_manifest_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("manifest_sha256", None)
    out["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(out))
    return out


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(manifest) + b"\n")


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items))


def _raw_output_for_video_name(video_name: str) -> str:
    text = str(video_name).strip()
    if not text:
        raise ValueError("empty video name")
    if "\x00" in text or text.startswith("/"):
        raise ValueError(f"unsafe video name {text!r}")
    parts = Path(text).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe video name {text!r}")
    stem = text.rsplit(".", 1)[0] if "." in text else text
    return f"{stem}.raw"


def _validate_raw_output_rel_path(raw_output: str) -> str:
    text = str(raw_output).strip()
    if not text:
        raise ValueError("empty raw output path")
    if "\x00" in text or text.startswith("/"):
        raise ValueError(f"unsafe raw output path {text!r}")
    parts = Path(text).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe raw output path {text!r}")
    if not text.endswith(".raw"):
        raise ValueError(f"raw output path must end with .raw: {text!r}")
    return text


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

    def read(self) -> int:
        if self.byte_pos >= len(self.data):
            return 0
        byte = self.data[self.byte_pos]
        bit = (byte >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit


class _ArithmeticDecoder:
    def __init__(self, data: bytes) -> None:
        self.reader = _BitReader(data)
        self.low = 0
        self.high = _AC_MASK
        self.value = 0
        for _ in range(_AC_PRECISION):
            self.value = (self.value << 1) | self.reader.read()

    def get_target(self, total: int) -> int:
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def remove(self, cum_low: int, cum_high: int, total: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + (span * cum_high) // total - 1
        self.low = self.low + (span * cum_low) // total
        while True:
            if self.high < _AC_HALF:
                pass
            elif self.low >= _AC_HALF:
                self.low -= _AC_HALF
                self.high -= _AC_HALF
                self.value -= _AC_HALF
            elif self.low >= _AC_QUARTER and self.high < _AC_THREE_QUARTER:
                self.low -= _AC_QUARTER
                self.high -= _AC_QUARTER
                self.value -= _AC_QUARTER
            else:
                break
            self.low = (self.low << 1) & _AC_MASK
            self.high = ((self.high << 1) | 1) & _AC_MASK
            self.value = ((self.value << 1) | self.reader.read()) & _AC_MASK


def _read_exact(payload: bytes, cursor: int, nbytes: int, label: str) -> tuple[bytes, int]:
    if nbytes < 0 or cursor < 0 or cursor + nbytes > len(payload):
        raise ValueError(
            f"AQ rawvideo payload truncated while reading {label}: "
            f"offset={cursor} need={nbytes} len={len(payload)}"
        )
    return payload[cursor : cursor + nbytes], cursor + nbytes


def _uint32_freq_table(raw: bytes) -> list[int]:
    if len(raw) % 4:
        raise ValueError("AQ rawvideo frequency table byte length is not divisible by 4")
    return [item[0] for item in struct.iter_unpack("<I", raw)]


def _cumulative_table(freq: list[int], *, allow_zero: bool, label: str) -> tuple[list[int], int]:
    if not freq:
        raise ValueError(f"{label} frequency table is empty")
    if allow_zero:
        if any(item < 0 for item in freq):
            raise ValueError(f"{label} frequency table contains negative counts")
    elif any(item <= 0 for item in freq):
        raise ValueError(f"{label} frequency table contains zero-probability symbols")
    cum = [0]
    running = 0
    for count in freq:
        running += int(count)
        cum.append(running)
    if running <= 0:
        raise ValueError(f"{label} frequency table total must be positive")
    return cum, running


def _validate_declared_symbol_count(
    *,
    freq: list[int],
    n_symbols: int,
    allow_zero_freq: bool,
    label: str,
) -> None:
    if n_symbols <= 0:
        raise ValueError(f"{label} n_symbols must be positive")
    total = sum(int(item) for item in freq)
    if allow_zero_freq:
        if total != int(n_symbols):
            raise ValueError(
                f"{label} frequency total {total} must equal n_symbols {n_symbols}"
            )
        return
    if total < int(n_symbols):
        raise ValueError(
            f"{label} frequency total {total} is smaller than n_symbols {n_symbols}"
        )
    zero_protection_overhead = total - int(n_symbols)
    if zero_protection_overhead > len(freq) - 1:
        raise ValueError(
            f"{label} zero-protected frequency overhead {zero_protection_overhead} "
            f"exceeds alphabet slack {len(freq) - 1}"
        )


def _decode_symbols_from_arithmetic_payload(
    *,
    freq: list[int],
    payload: bytes,
    n_symbols: int,
    allow_zero_freq: bool,
    label: str,
) -> bytes:
    cum, total = _cumulative_table(freq, allow_zero=allow_zero_freq, label=label)
    decoder = _ArithmeticDecoder(payload)
    decoded = bytearray()
    observed = [0] * len(freq)
    for index in range(int(n_symbols)):
        target = decoder.get_target(total)
        symbol = bisect_right(cum, target) - 1
        if symbol < 0 or symbol >= len(freq) or int(freq[symbol]) <= 0:
            raise ValueError(
                f"{label} target={target} fell outside a positive-frequency "
                f"symbol at {index}/{n_symbols}"
            )
        decoder.remove(int(cum[symbol]), int(cum[symbol + 1]), total)
        if symbol > 255:
            raise ValueError(f"{label} decoded symbol {symbol} exceeds uint8 range")
        decoded.append(symbol)
        observed[symbol] += 1

    if allow_zero_freq:
        if observed != [int(item) for item in freq]:
            raise ValueError(f"{label} decoded symbol counts do not match sparse frequency table")
    else:
        for observed_count, expected_count in zip(observed, freq, strict=True):
            if observed_count > int(expected_count):
                raise ValueError(f"{label} decoded symbol count exceeds dense frequency table")
            if int(expected_count) > 1 and observed_count != int(expected_count):
                raise ValueError(f"{label} decoded symbol counts do not match dense frequency table")
    return bytes(decoded)


def _decode_aq_uint8_rawvideo(payload: bytes) -> tuple[bytes, str]:
    magic = payload[:4]
    if magic == AQ_MAGIC:
        return _decode_aqv1_uint8_rawvideo(payload), "AQv1"
    if magic == AQC_MAGIC:
        return _decode_aqc1_uint8_rawvideo(payload), "AQc1"
    raise ValueError(f"unsupported AQ rawvideo magic {magic!r}")


def _decode_aqv1_uint8_rawvideo(payload: bytes) -> bytes:
    cursor = 0
    magic, cursor = _read_exact(payload, cursor, 4, "magic")
    if magic != AQ_MAGIC:
        raise ValueError(f"bad AQv1 rawvideo magic {magic!r}")
    raw, cursor = _read_exact(payload, cursor, 2, "version")
    (version,) = struct.unpack("<H", raw)
    if version != AQ_VERSION:
        raise ValueError(f"unsupported AQv1 rawvideo version {version}")
    raw, cursor = _read_exact(payload, cursor, 2, "num_symbols")
    (num_symbols,) = struct.unpack("<H", raw)
    raw, cursor = _read_exact(payload, cursor, 4, "offset")
    (offset,) = struct.unpack("<i", raw)
    if num_symbols != AQ_RAWVIDEO_NUM_SYMBOLS or offset != AQ_RAWVIDEO_OFFSET:
        raise ValueError(
            "AQv1 rawvideo requires num_symbols=256 and offset=0; "
            f"got num_symbols={num_symbols} offset={offset}"
        )
    raw, cursor = _read_exact(payload, cursor, 8, "n_symbols")
    (n_symbols,) = struct.unpack("<Q", raw)
    raw, cursor = _read_exact(payload, cursor, int(num_symbols) * 4, "frequency table")
    freq = _uint32_freq_table(raw)
    raw, cursor = _read_exact(payload, cursor, 8, "payload_size")
    (payload_size,) = struct.unpack("<Q", raw)
    if payload_size <= 0:
        raise ValueError("AQv1 rawvideo payload_size must be positive")
    coded_payload, cursor = _read_exact(payload, cursor, int(payload_size), "payload")
    if cursor != len(payload):
        raise ValueError("AQv1 rawvideo payload has trailing bytes after declared payload")
    _validate_declared_symbol_count(
        freq=freq,
        n_symbols=int(n_symbols),
        allow_zero_freq=False,
        label="AQv1 rawvideo",
    )
    return _decode_symbols_from_arithmetic_payload(
        freq=freq,
        payload=coded_payload,
        n_symbols=int(n_symbols),
        allow_zero_freq=False,
        label="AQv1 rawvideo",
    )


def _decode_aqc1_uint8_rawvideo(payload: bytes) -> bytes:
    cursor = 0
    magic, cursor = _read_exact(payload, cursor, 4, "magic")
    if magic != AQC_MAGIC:
        raise ValueError(f"bad AQc1 rawvideo magic {magic!r}")
    raw, cursor = _read_exact(payload, cursor, 2, "version")
    (version,) = struct.unpack("<H", raw)
    if version != AQC_VERSION:
        raise ValueError(f"unsupported AQc1 rawvideo version {version}")
    raw, cursor = _read_exact(payload, cursor, 2, "num_symbols")
    (num_symbols,) = struct.unpack("<H", raw)
    raw, cursor = _read_exact(payload, cursor, 4, "offset")
    (offset,) = struct.unpack("<i", raw)
    if num_symbols != AQ_RAWVIDEO_NUM_SYMBOLS or offset != AQ_RAWVIDEO_OFFSET:
        raise ValueError(
            "AQc1 rawvideo requires num_symbols=256 and offset=0; "
            f"got num_symbols={num_symbols} offset={offset}"
        )
    raw, cursor = _read_exact(payload, cursor, 8, "n_symbols")
    (n_symbols,) = struct.unpack("<Q", raw)
    raw, cursor = _read_exact(payload, cursor, 2, "n_present")
    (n_present,) = struct.unpack("<H", raw)
    if n_present == 0 or n_present > num_symbols:
        raise ValueError(
            f"AQc1 rawvideo n_present must be in [1, num_symbols], got {n_present}"
        )
    freq = [0] * int(num_symbols)
    seen: set[int] = set()
    for _ in range(int(n_present)):
        raw, cursor = _read_exact(payload, cursor, 2, "symbol")
        (symbol,) = struct.unpack("<H", raw)
        raw, cursor = _read_exact(payload, cursor, 4, "count")
        (count,) = struct.unpack("<I", raw)
        if symbol >= num_symbols:
            raise ValueError(f"AQc1 rawvideo symbol index out of range: {symbol}")
        if symbol in seen:
            raise ValueError(f"AQc1 rawvideo duplicate symbol index: {symbol}")
        if count == 0:
            raise ValueError(f"AQc1 rawvideo symbol {symbol} has zero count")
        seen.add(int(symbol))
        freq[int(symbol)] = int(count)
    raw, cursor = _read_exact(payload, cursor, 8, "payload_size")
    (payload_size,) = struct.unpack("<Q", raw)
    if payload_size <= 0:
        raise ValueError("AQc1 rawvideo payload_size must be positive")
    coded_payload, cursor = _read_exact(payload, cursor, int(payload_size), "payload")
    if cursor != len(payload):
        raise ValueError("AQc1 rawvideo payload has trailing bytes after declared payload")
    _validate_declared_symbol_count(
        freq=freq,
        n_symbols=int(n_symbols),
        allow_zero_freq=True,
        label="AQc1 rawvideo",
    )
    return _decode_symbols_from_arithmetic_payload(
        freq=freq,
        payload=coded_payload,
        n_symbols=int(n_symbols),
        allow_zero_freq=True,
        label="AQc1 rawvideo",
    )


@dataclass(frozen=True)
class DecodedRawOutputStream:
    path: str
    payload: bytes
    source_stream_name: str
    source_codec_kind: int
    candidate_output_source: str
    decoder_adapter_id: str
    decoded_stream_kind: str
    real_decoded_rawvideo: bool
    fixture_passthrough: bool
    dispatch_blockers: tuple[str, ...]

    def manifest_row(self) -> dict[str, Any]:
        return {
            "schema": JCSP_RUNTIME_DECODED_STREAM_SCHEMA,
            "path": self.path,
            "bytes": len(self.payload),
            "sha256": _sha256_bytes(self.payload),
            "source_stream_name": self.source_stream_name,
            "source_codec_kind": int(self.source_codec_kind),
            "candidate_output_source": self.candidate_output_source,
            "decoder_adapter_id": self.decoder_adapter_id,
            "decoded_stream_kind": self.decoded_stream_kind,
            "real_decoded_rawvideo": bool(self.real_decoded_rawvideo),
            "fixture_passthrough": bool(self.fixture_passthrough),
            "dispatch_blockers": list(self.dispatch_blockers),
        }


class RawOutputDecoderAdapter:
    adapter_id = "jcsp_raw_output_decoder_adapter"
    adapter_kind = "abstract"
    candidate_output_source = JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE
    emits_real_rawvideo = False
    fixture_passthrough = False
    non_production = False

    def contract(self, expected_raw_outputs: Iterable[str]) -> dict[str, Any]:
        expected = list(expected_raw_outputs)
        dispatch_blockers = [JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER]
        if not self.emits_real_rawvideo:
            dispatch_blockers.append(JCSP_RUNTIME_REAL_DECODER_BLOCKER)
        if self.fixture_passthrough:
            dispatch_blockers.append(JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_BLOCKER)
        return {
            "schema": JCSP_RUNTIME_DECODER_ADAPTER_SCHEMA,
            "score_claim": False,
            "dispatch_attempted": False,
            "adapter_id": self.adapter_id,
            "adapter_kind": self.adapter_kind,
            "candidate_output_source": self.candidate_output_source,
            "required_candidate_output_source_for_dispatch": (
                JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE
            ),
            "emits_real_decoded_rawvideo": bool(self.emits_real_rawvideo),
            "fixture_passthrough": bool(self.fixture_passthrough),
            "non_production": bool(self.non_production),
            "expected_raw_outputs": expected,
            "expected_raw_output_count": len(expected),
            "ready_for_output_parity": False,
            "ready_for_submission_runtime_consumption": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": _dedupe(dispatch_blockers),
        }

    def decode_stream(
        self,
        stream: Mapping[str, Any],
        *,
        expected_raw_outputs: set[str],
    ) -> tuple[DecodedRawOutputStream | None, str | None]:
        raise NotImplementedError("raw output decoder adapter is not implemented")


class ArithmeticUint8RawVideoDecoderAdapter(RawOutputDecoderAdapter):
    adapter_id = JCSP_RUNTIME_AQ_RAWVIDEO_ADAPTER_ID
    adapter_kind = JCSP_RUNTIME_AQ_RAWVIDEO_STREAM_KIND
    candidate_output_source = JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE
    emits_real_rawvideo = True
    fixture_passthrough = False
    non_production = False

    def contract(self, expected_raw_outputs: Iterable[str]) -> dict[str, Any]:
        payload = super().contract(expected_raw_outputs)
        payload.update(
            {
                "accepted_codec_kind": KIND_ARITHMETIC_STATIC,
                "accepted_payload_magics": ["AQv1", "AQc1"],
                "stream_name_contract": (
                    "stream name must exactly equal expected .raw relative path"
                ),
                "stream_payload_contract": JCSP_RUNTIME_AQ_RAWVIDEO_FORMAT_CONTRACT,
                "decoded_stream_kind": JCSP_RUNTIME_AQ_RAWVIDEO_STREAM_KIND,
                "ready_for_submission_runtime_consumption": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
        payload["dispatch_blockers"] = _dedupe(
            [
                JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
                *payload["dispatch_blockers"],
                "exact_cuda_auth_eval_missing",
            ]
        )
        return payload

    def decode_stream(
        self,
        stream: Mapping[str, Any],
        *,
        expected_raw_outputs: set[str],
    ) -> tuple[DecodedRawOutputStream | None, str | None]:
        stream_name = str(stream.get("name", ""))
        try:
            rel = _validate_raw_output_rel_path(stream_name)
        except ValueError:
            return None, "jcsp_aq_rawvideo_stream_name_not_raw_output_path"
        if rel not in expected_raw_outputs:
            return None, "jcsp_aq_rawvideo_unexpected_raw_stream"
        if int(stream.get("codec_kind", -1)) != KIND_ARITHMETIC_STATIC:
            return None, "jcsp_aq_rawvideo_stream_not_arithmetic_static"
        raw_payload = stream.get("payload", b"")
        if not isinstance(raw_payload, (bytes, bytearray, memoryview)):
            return None, "jcsp_aq_rawvideo_payload_not_bytes"
        try:
            decoded, payload_magic = _decode_aq_uint8_rawvideo(bytes(raw_payload))
        except ValueError:
            return None, "jcsp_aq_rawvideo_decode_failed"
        if not decoded:
            return None, "jcsp_aq_rawvideo_decoded_rawvideo_empty"
        if len(decoded) % 3:
            return None, "jcsp_aq_rawvideo_decoded_bytes_not_rgb24_aligned"
        return (
            DecodedRawOutputStream(
                path=rel,
                payload=decoded,
                source_stream_name=stream_name,
                source_codec_kind=int(stream["codec_kind"]),
                candidate_output_source=self.candidate_output_source,
                decoder_adapter_id=self.adapter_id,
                decoded_stream_kind=f"{self.adapter_kind}:{payload_magic}",
                real_decoded_rawvideo=True,
                fixture_passthrough=False,
                dispatch_blockers=(JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,),
            ),
            None,
        )


class FixtureRawPassthroughDecoderAdapter(RawOutputDecoderAdapter):
    adapter_id = "jcsp_fixture_raw_passthrough_adapter_v1"
    adapter_kind = "fixture_raw_passthrough"
    candidate_output_source = JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_SOURCE
    emits_real_rawvideo = False
    fixture_passthrough = True
    non_production = True

    def contract(self, expected_raw_outputs: Iterable[str]) -> dict[str, Any]:
        payload = super().contract(expected_raw_outputs)
        payload.update(
            {
                "accepted_codec_kind": KIND_RAW_PASSTHROUGH,
                "stream_name_contract": (
                    "stream name must exactly equal expected .raw relative path"
                ),
                "stream_payload_contract": (
                    "payload bytes are written verbatim to that .raw path"
                ),
            }
        )
        payload["dispatch_blockers"] = _dedupe(
            [
                JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_BLOCKER,
                JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
                JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                *payload["dispatch_blockers"],
                "exact_cuda_auth_eval_missing",
            ]
        )
        return payload

    def decode_stream(
        self,
        stream: Mapping[str, Any],
        *,
        expected_raw_outputs: set[str],
    ) -> tuple[DecodedRawOutputStream | None, str | None]:
        stream_name = str(stream.get("name", ""))
        try:
            rel = _validate_raw_output_rel_path(stream_name)
        except ValueError:
            return None, "jcsp_fixture_stream_name_not_raw_output_path"
        if int(stream.get("codec_kind", -1)) != KIND_RAW_PASSTHROUGH:
            return None, "jcsp_fixture_stream_not_raw_passthrough"
        if rel not in expected_raw_outputs:
            return None, "jcsp_fixture_unexpected_raw_stream"
        return (
            DecodedRawOutputStream(
                path=rel,
                payload=bytes(stream.get("payload", b"")),
                source_stream_name=stream_name,
                source_codec_kind=int(stream["codec_kind"]),
                candidate_output_source=self.candidate_output_source,
                decoder_adapter_id=self.adapter_id,
                decoded_stream_kind=self.adapter_kind,
                real_decoded_rawvideo=False,
                fixture_passthrough=True,
                dispatch_blockers=(
                    JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_BLOCKER,
                    JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                ),
            ),
            None,
        )


def _decoder_interface_contract(
    expected_raw_outputs: Iterable[str],
) -> dict[str, Any]:
    expected = list(expected_raw_outputs)
    real_adapter = ArithmeticUint8RawVideoDecoderAdapter()
    fixture_adapter = FixtureRawPassthroughDecoderAdapter()
    return {
        "schema": JCSP_RUNTIME_DECODER_ADAPTER_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "purpose": "runtime boundary between decoded JCSP streams and rawvideo output",
        "required_candidate_output_source_for_dispatch": (
            JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE
        ),
        "production_decoder_adapter_wired": True,
        "real_decoder_adapter_available": True,
        "real_decoder_adapter": real_adapter.contract(expected),
        "fixture_passthrough_adapter_available": True,
        "fixture_passthrough_is_non_dispatch_proof": True,
        "expected_raw_outputs": expected,
        "expected_raw_output_count": len(expected),
        "fixture_adapter": fixture_adapter.contract(expected),
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
            "exact_cuda_auth_eval_missing",
        ],
    }


def _expected_raw_outputs_from_names_file(
    video_names_file: str | Path | None,
) -> tuple[list[str], str | None]:
    if video_names_file is None:
        return [], None
    try:
        text = Path(video_names_file).read_text(encoding="utf-8")
        outputs = [
            _raw_output_for_video_name(line)
            for line in text.splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [], str(exc)
    return _dedupe(outputs), None


def _observe_existing_raw_outputs(
    inflated_dir: str | Path | None,
    expected_raw_outputs: list[str],
) -> list[dict[str, Any]]:
    if inflated_dir is None:
        return []
    root = Path(inflated_dir)
    rows: list[dict[str, Any]] = []
    for rel in expected_raw_outputs:
        rel = _validate_raw_output_rel_path(rel)
        path = root / rel
        exists = path.exists()
        row: dict[str, Any] = {
            "path": rel,
            "exists": exists,
            "is_file": path.is_file(),
            "bytes": None,
            "sha256": None,
            "sha256_status": "not_hashed_pre_dispatch_probe",
            "parity_proof_source": "preexisting_raw_output_unproven",
        }
        if path.is_file():
            try:
                row["bytes"] = int(path.stat().st_size)
            except OSError:
                row["bytes"] = None
        rows.append(row)
    return rows


def _raw_output_parity_contract(
    *,
    expected_raw_outputs: list[str],
    existing_raw_output_count: int,
    names_error: str | None,
) -> dict[str, Any]:
    expected_known = names_error is None and bool(expected_raw_outputs)
    return {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA,
        "required_proof_schema": JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "expected_raw_outputs_known": expected_known,
        "expected_raw_output_count": len(expected_raw_outputs),
        "expected_raw_outputs_sha256": _sha256_bytes(
            _canonical_json_bytes({"expected_raw_outputs": expected_raw_outputs})
        ),
        "required_candidate_output_source": JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE,
        "required_reference_output_source": (
            "contest_reference_runtime_or_byte_custody_baseline"
        ),
        "preexisting_raw_outputs_are_not_parity_proof": True,
        "existing_raw_output_count_at_probe": int(existing_raw_output_count),
        "required_per_output_fields": [
            "path",
            "candidate_exists",
            "candidate_bytes",
            "candidate_sha256",
            "reference_exists",
            "reference_bytes",
            "reference_sha256",
            "byte_exact_match",
        ],
        "acceptance_conditions": [
            "jcsp_stream_consumer_decodes_jcsp_streams",
            "bridge_emits_exactly_expected_raw_outputs",
            "candidate_outputs_are_from_current_bridge_run",
            "reference_outputs_are_from_contest_runtime_or_custody_baseline",
            "all_candidate_sha256_values_match_reference_sha256_values",
            "parity_proof_manifest_uses_required_schema",
        ],
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "dispatch_blocker": JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
    }


def _raw_output_file_identity(
    root: str | Path,
    rel: str,
    *,
    role: str,
) -> dict[str, Any]:
    path = Path(root) / rel
    exists = path.exists()
    is_file = path.is_file()
    row: dict[str, Any] = {
        f"{role}_exists": exists,
        f"{role}_is_file": is_file,
        f"{role}_bytes": None,
        f"{role}_sha256": None,
    }
    if is_file:
        row[f"{role}_bytes"] = int(path.stat().st_size)
        row[f"{role}_sha256"] = _sha256_file(path)
    return row


def prove_jcsp_runtime_raw_output_parity(
    expected_raw_outputs: Iterable[str],
    *,
    candidate_raw_dir: str | Path,
    reference_raw_dir: str | Path,
    candidate_outputs_emitted_by_bridge: bool,
    candidate_output_source: str | None = None,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic byte-exact raw-output parity proof.

    This helper is intentionally independent from the current inflate probe.
    The current bridge does not emit raw outputs, so ``probe_jcsp_runtime_bridge``
    still refuses every present ``jcsp.bin``.  The future JCSP stream consumer
    can call this only after it writes the contest ``.raw`` files for the
    current run.
    """

    expected = _dedupe(
        [_validate_raw_output_rel_path(str(item)) for item in expected_raw_outputs]
    )
    source = (
        str(candidate_output_source)
        if candidate_output_source is not None
        else (
            JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE
            if candidate_outputs_emitted_by_bridge
            else "preexisting_or_unproven_raw_files"
        )
    )
    source_is_real_bridge = bool(
        candidate_outputs_emitted_by_bridge
        and source == JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE
    )
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for rel in expected:
        row = {"path": rel}
        row.update(_raw_output_file_identity(candidate_raw_dir, rel, role="candidate"))
        row.update(_raw_output_file_identity(reference_raw_dir, rel, role="reference"))
        row["byte_exact_match"] = bool(
            row["candidate_is_file"]
            and row["reference_is_file"]
            and row["candidate_bytes"] == row["reference_bytes"]
            and row["candidate_sha256"] == row["reference_sha256"]
        )
        if not row["candidate_is_file"]:
            blockers.append("jcsp_candidate_raw_output_missing")
        if not row["reference_is_file"]:
            blockers.append("jcsp_reference_raw_output_missing")
        if row["candidate_is_file"] and row["reference_is_file"] and not row[
            "byte_exact_match"
        ]:
            blockers.append("jcsp_raw_output_sha256_mismatch")
        rows.append(row)

    if not expected:
        blockers.append("jcsp_expected_raw_outputs_missing")
    if not source_is_real_bridge:
        blockers.append("jcsp_candidate_outputs_not_emitted_by_bridge")
        if candidate_outputs_emitted_by_bridge:
            blockers.append(JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_BLOCKER)

    all_candidate_present = bool(expected) and all(
        row["candidate_is_file"] for row in rows
    )
    all_reference_present = bool(expected) and all(
        row["reference_is_file"] for row in rows
    )
    byte_exact = bool(expected) and all(row["byte_exact_match"] for row in rows)
    ready_for_output_parity = bool(
        source_is_real_bridge
        and all_candidate_present
        and all_reference_present
        and byte_exact
    )
    manifest: dict[str, Any] = {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_outputs_emitted_by_bridge": bool(
            candidate_outputs_emitted_by_bridge
        ),
        "candidate_output_source": source,
        "candidate_outputs_from_real_bridge_rawvideo": source_is_real_bridge,
        "fixture_raw_output_emission": (
            source == JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_SOURCE
        ),
        "reference_output_source": "contest_reference_runtime_or_byte_custody_baseline",
        "candidate_raw_dir": str(Path(candidate_raw_dir)),
        "reference_raw_dir": str(Path(reference_raw_dir)),
        "expected_raw_outputs": expected,
        "expected_raw_output_count": len(expected),
        "output_count": len(rows),
        "all_candidate_outputs_present": all_candidate_present,
        "all_reference_outputs_present": all_reference_present,
        "byte_exact_raw_output_parity": byte_exact,
        "ready_for_output_parity": ready_for_output_parity,
        "ready_for_submission_runtime_consumption": ready_for_output_parity,
        "ready_for_exact_eval_dispatch": False,
        "outputs": rows,
        "dispatch_blockers": (
            []
            if ready_for_output_parity
            else _dedupe([JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER, *blockers])
        ),
    }
    manifest = _with_manifest_sha256(manifest)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def _minimal_fixture_stream_path_contract(
    expected_raw_outputs: Iterable[str],
) -> dict[str, Any]:
    expected = list(expected_raw_outputs)
    adapter = FixtureRawPassthroughDecoderAdapter()
    return {
        "schema": "jcsp_minimal_fixture_stream_path_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "purpose": "local_raw_output_emission_fixture_only",
        "container_magic": "JCSP",
        "container_version": JCSP_VERSION,
        "stream_codec_kind": KIND_RAW_PASSTHROUGH,
        "stream_name_contract": (
            "stream name must exactly equal expected .raw relative path"
        ),
        "stream_payload_contract": (
            "payload bytes are written verbatim to that .raw path"
        ),
        "expected_raw_outputs": expected,
        "expected_raw_output_count": len(expected),
        "emitter": "emit_jcsp_fixture_raw_outputs",
        "candidate_output_source": JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_SOURCE,
        "decoder_adapter": adapter.contract(expected),
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_BLOCKER,
            JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
            "exact_cuda_auth_eval_missing",
        ],
    }


def emit_jcsp_fixture_raw_outputs(
    archive_dir: str | Path,
    *,
    expected_raw_outputs: Iterable[str],
    output_dir: str | Path,
    reference_raw_dir: str | Path | None = None,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    manifest_json: str | Path | None = None,
    parity_manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Emit deterministic fixture ``.raw`` files from raw-passthrough streams.

    This is a local bridge fixture, not the contest JCSP decoder.  The only
    accepted stream path is ``KIND_RAW_PASSTHROUGH`` with a stream name exactly
    matching an expected relative ``.raw`` output.  Arithmetic and hyperprior
    streams still fail closed; fixture parity remains non-dispatchable unless a
    later real decoder emits ``JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE`` outputs.
    """

    expected = _dedupe(
        [_validate_raw_output_rel_path(str(item)) for item in expected_raw_outputs]
    )
    output_root = Path(output_dir)
    archive_root = Path(archive_dir)
    member_path = archive_root / member_name
    blockers: list[str] = []
    emitted_rows: list[dict[str, Any]] = []
    parity_proof: dict[str, Any] | None = None
    parsed_summary: dict[str, Any] | None = None
    decoder_adapter = FixtureRawPassthroughDecoderAdapter()

    if not expected:
        blockers.append("jcsp_expected_raw_outputs_missing")
    if not member_path.exists():
        blockers.append("jcsp_member_missing_for_fixture_raw_output_emission")
    elif not member_path.is_file():
        blockers.append("jcsp_member_path_not_regular_file")

    decoded_streams_by_name: dict[str, DecodedRawOutputStream] = {}
    if not blockers:
        blob = member_path.read_bytes()
        if len(blob) < 4 or blob[:4] != JCSP_MAGIC:
            blockers.append("jcsp_fixture_emitter_requires_real_jcsp_member")
        else:
            try:
                parsed = _parse_real_jcsp_container(blob, include_payload=True)
            except ValueError as exc:
                blockers.append("jcsp_runtime_probe_parse_failed")
                parsed_summary = {"parse_error": str(exc)}
            else:
                parsed_summary = {
                    "container_magic": parsed["container_magic"],
                    "container_version": parsed["container_version"],
                    "stream_count": parsed["stream_count"],
                    "noop_fixture": parsed["noop_fixture"],
                }
                if parsed["noop_fixture"]:
                    blockers.append(JCSP_RUNTIME_NOOP_PACKET_BLOCKER)
                expected_set = set(expected)
                for stream in parsed["streams"]:
                    decoded, blocker = decoder_adapter.decode_stream(
                        stream,
                        expected_raw_outputs=expected_set,
                    )
                    if blocker is not None:
                        blockers.append(blocker)
                        continue
                    if decoded is None:
                        blockers.append("jcsp_fixture_decoder_returned_no_stream")
                        continue
                    rel = decoded.path
                    if rel in decoded_streams_by_name:
                        blockers.append("jcsp_fixture_duplicate_raw_stream_name")
                        continue
                    decoded_streams_by_name[rel] = decoded

                stream_set = set(decoded_streams_by_name)
                if expected_set - stream_set:
                    blockers.append("jcsp_fixture_missing_expected_raw_stream")
                if stream_set - expected_set:
                    blockers.append("jcsp_fixture_unexpected_raw_stream")
                for rel in expected:
                    if (output_root / rel).exists():
                        blockers.append(
                            "jcsp_fixture_raw_output_target_already_exists"
                        )

    blockers = _dedupe(blockers)
    if not blockers:
        for rel in expected:
            decoded = decoded_streams_by_name[rel]
            path = output_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(decoded.payload)
            emitted_rows.append(decoded.manifest_row())

    if reference_raw_dir is not None and emitted_rows:
        parity_proof = prove_jcsp_runtime_raw_output_parity(
            expected,
            candidate_raw_dir=output_root,
            reference_raw_dir=reference_raw_dir,
            candidate_outputs_emitted_by_bridge=True,
            candidate_output_source=JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_SOURCE,
            manifest_json=parity_manifest_json,
        )

    dispatch_blockers = _dedupe(
        [
            JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_BLOCKER,
            JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
            *blockers,
            *(
                parity_proof.get("dispatch_blockers", [])
                if parity_proof is not None
                else []
            ),
            "exact_cuda_auth_eval_missing",
        ]
    )
    byte_exact_parity = bool(
        parity_proof is not None
        and parity_proof.get("byte_exact_raw_output_parity") is True
    )
    manifest: dict[str, Any] = {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_EMISSION_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_output_source": JCSP_RUNTIME_FIXTURE_RAW_OUTPUT_SOURCE,
        "candidate_outputs_from_real_bridge_rawvideo": False,
        "member_name": member_name,
        "member_present": member_path.exists(),
        "expected_raw_outputs": expected,
        "expected_raw_output_count": len(expected),
        "candidate_raw_dir": str(output_root),
        "reference_raw_dir": (
            str(Path(reference_raw_dir)) if reference_raw_dir is not None else None
        ),
        "decoder_interface": _decoder_interface_contract(expected),
        "decoder_adapter": decoder_adapter.contract(expected),
        "minimal_fixture_stream_path": _minimal_fixture_stream_path_contract(expected),
        "parsed_container": parsed_summary,
        "raw_output_emission_attempted": not blockers,
        "fixture_raw_outputs_emitted": bool(emitted_rows),
        "bridge_emits_contest_raw_outputs": False,
        "real_decoded_rawvideo_stream_count": sum(
            1 for row in emitted_rows if row["real_decoded_rawvideo"]
        ),
        "fixture_passthrough_stream_count": sum(
            1 for row in emitted_rows if row["fixture_passthrough"]
        ),
        "emitted_raw_output_count": len(emitted_rows),
        "emitted_raw_outputs": emitted_rows,
        "output_parity_checked": parity_proof is not None,
        "output_parity_artifact": (
            str(Path(parity_manifest_json))
            if parity_manifest_json is not None
            else None
        ),
        "raw_output_parity_proof": parity_proof,
        "byte_exact_fixture_raw_output_parity": byte_exact_parity,
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": dispatch_blockers,
    }
    manifest = _with_manifest_sha256(manifest)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def _parsed_container_summary(parsed: Mapping[str, Any]) -> dict[str, Any]:
    stream_rows: list[dict[str, Any]] = []
    for stream in parsed.get("streams", []):
        if not isinstance(stream, Mapping):
            continue
        stream_rows.append(
            {
                key: value
                for key, value in stream.items()
                if key != "payload"
            }
        )
    return {
        "container_magic": parsed.get("container_magic"),
        "container_version": parsed.get("container_version"),
        "stream_count": parsed.get("stream_count"),
        "streams": stream_rows,
        "noop_fixture": parsed.get("noop_fixture"),
    }


def plan_jcsp_real_raw_output_emission(
    archive_dir: str | Path,
    *,
    expected_raw_outputs: Iterable[str],
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Plan the narrow real JCSP AQ/rawvideo consumer without writing files.

    The plan proves which contest ``.raw`` members the current bridge would
    emit, or records deterministic refusal blockers.  It decodes in memory,
    imports no scorer code, reads no sidecars, writes no raw outputs, and never
    clears exact-eval dispatch.
    """

    expected = _dedupe(
        [_validate_raw_output_rel_path(str(item)) for item in expected_raw_outputs]
    )
    archive_root = Path(archive_dir)
    member_path = archive_root / member_name
    blockers: list[str] = []
    decode_failures: list[dict[str, Any]] = []
    parsed_summary: dict[str, Any] | None = None
    decoder_adapter = ArithmeticUint8RawVideoDecoderAdapter()

    if not expected:
        blockers.append("jcsp_expected_raw_outputs_missing")
    if not member_path.exists():
        blockers.append("jcsp_member_missing_for_real_raw_output_readiness")
    elif not member_path.is_file():
        blockers.append("jcsp_member_path_not_regular_file")

    decoded_streams_by_name: dict[str, DecodedRawOutputStream] = {}
    if not blockers:
        blob = member_path.read_bytes()
        if len(blob) < 4 or blob[:4] != JCSP_MAGIC:
            blockers.append("jcsp_real_readiness_requires_real_jcsp_member")
        else:
            try:
                parsed = _parse_real_jcsp_container(blob, include_payload=True)
            except ValueError as exc:
                blockers.append("jcsp_runtime_probe_parse_failed")
                parsed_summary = {"parse_error": str(exc)}
            else:
                parsed_summary = _parsed_container_summary(parsed)
                if parsed["noop_fixture"]:
                    blockers.append(JCSP_RUNTIME_NOOP_PACKET_BLOCKER)
                expected_set = set(expected)
                for stream in parsed["streams"]:
                    decoded, blocker = decoder_adapter.decode_stream(
                        stream,
                        expected_raw_outputs=expected_set,
                    )
                    if blocker is not None:
                        blockers.append(blocker)
                        decode_failures.append(
                            {
                                "stream_name": stream.get("name"),
                                "codec_kind": stream.get("codec_kind"),
                                "payload_magic": stream.get("payload_magic"),
                                "blocker": blocker,
                            }
                        )
                        continue
                    if decoded is None:
                        blocker = "jcsp_aq_rawvideo_decoder_returned_no_stream"
                        blockers.append(blocker)
                        decode_failures.append(
                            {
                                "stream_name": stream.get("name"),
                                "codec_kind": stream.get("codec_kind"),
                                "payload_magic": stream.get("payload_magic"),
                                "blocker": blocker,
                            }
                        )
                        continue
                    rel = decoded.path
                    if rel in decoded_streams_by_name:
                        blocker = "jcsp_aq_rawvideo_duplicate_raw_stream_name"
                        blockers.append(blocker)
                        decode_failures.append(
                            {
                                "stream_name": stream.get("name"),
                                "codec_kind": stream.get("codec_kind"),
                                "payload_magic": stream.get("payload_magic"),
                                "blocker": blocker,
                            }
                        )
                        continue
                    decoded_streams_by_name[rel] = decoded

                stream_set = set(decoded_streams_by_name)
                if expected_set - stream_set:
                    blockers.append("jcsp_aq_rawvideo_missing_expected_raw_stream")
                if stream_set - expected_set:
                    blockers.append("jcsp_aq_rawvideo_unexpected_raw_stream")

    blockers = _dedupe(blockers)
    would_emit_rows: list[dict[str, Any]] = []
    if not blockers:
        would_emit_rows = [
            decoded_streams_by_name[rel].manifest_row() for rel in expected
        ]
    ready_for_raw_output_emission = bool(expected and would_emit_rows and not blockers)
    dispatch_blockers = _dedupe(
        [
            *blockers,
            *(
                ["jcsp_raw_output_emission_not_run"]
                if ready_for_raw_output_emission
                else []
            ),
            JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
            "exact_cuda_auth_eval_missing",
        ]
    )
    manifest: dict[str, Any] = {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_CONSUMER_READINESS_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "runtime_action": "plan_real_aq_rawvideo_output_emission",
        "no_scorer_at_inflate": True,
        "score_affecting_sidecars_allowed": False,
        "raw_output_files_written": False,
        "candidate_output_source": JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE,
        "member_name": member_name,
        "member_present": member_path.exists(),
        "detects_required_member": member_path.exists(),
        "decodes_required_member": ready_for_raw_output_emission,
        "consumes_required_member": False,
        "expected_raw_outputs": expected,
        "expected_raw_output_count": len(expected),
        "decoder_adapter": decoder_adapter.contract(expected),
        "parsed_container": parsed_summary,
        "would_emit_contest_raw_outputs": ready_for_raw_output_emission,
        "would_emit_raw_output_count": len(would_emit_rows),
        "would_emit_raw_outputs": would_emit_rows,
        "decode_failures": decode_failures,
        "ready_for_raw_output_emission": ready_for_raw_output_emission,
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": dispatch_blockers,
    }
    manifest = _with_manifest_sha256(manifest)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def emit_jcsp_real_raw_outputs(
    archive_dir: str | Path,
    *,
    expected_raw_outputs: Iterable[str],
    output_dir: str | Path,
    reference_raw_dir: str | Path | None = None,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    manifest_json: str | Path | None = None,
    parity_manifest_json: str | Path | None = None,
    require_output_parity_for_runtime_consumption: bool = True,
) -> dict[str, Any]:
    """Decode the narrow real JCSP AQ/rawvideo stream contract.

    Supported stream kind:
    ``codec_kind=KIND_ARITHMETIC_STATIC`` with AQv1/AQc1 payload bytes whose
    AQ header declares ``num_symbols=256`` and ``offset=0``. The stream name
    must exactly match an expected ``.raw`` relative output path. The decoded
    symbols must be nonempty RGB24 bytes, then are written as uint8 rawvideo.
    All other JCSP stream kinds fail closed with manifest blockers.
    """

    expected = _dedupe(
        [_validate_raw_output_rel_path(str(item)) for item in expected_raw_outputs]
    )
    output_root = Path(output_dir)
    archive_root = Path(archive_dir)
    member_path = archive_root / member_name
    blockers: list[str] = []
    emitted_rows: list[dict[str, Any]] = []
    decode_failures: list[dict[str, Any]] = []
    parity_proof: dict[str, Any] | None = None
    parsed_summary: dict[str, Any] | None = None
    decoder_adapter = ArithmeticUint8RawVideoDecoderAdapter()

    if not expected:
        blockers.append("jcsp_expected_raw_outputs_missing")
    if not member_path.exists():
        blockers.append("jcsp_member_missing_for_real_raw_output_emission")
    elif not member_path.is_file():
        blockers.append("jcsp_member_path_not_regular_file")

    decoded_streams_by_name: dict[str, DecodedRawOutputStream] = {}
    if not blockers:
        blob = member_path.read_bytes()
        if len(blob) < 4 or blob[:4] != JCSP_MAGIC:
            blockers.append("jcsp_real_emitter_requires_real_jcsp_member")
        else:
            try:
                parsed = _parse_real_jcsp_container(blob, include_payload=True)
            except ValueError as exc:
                blockers.append("jcsp_runtime_probe_parse_failed")
                parsed_summary = {"parse_error": str(exc)}
            else:
                parsed_summary = _parsed_container_summary(parsed)
                if parsed["noop_fixture"]:
                    blockers.append(JCSP_RUNTIME_NOOP_PACKET_BLOCKER)
                expected_set = set(expected)
                for stream in parsed["streams"]:
                    decoded, blocker = decoder_adapter.decode_stream(
                        stream,
                        expected_raw_outputs=expected_set,
                    )
                    if blocker is not None:
                        blockers.append(blocker)
                        decode_failures.append(
                            {
                                "stream_name": stream.get("name"),
                                "codec_kind": stream.get("codec_kind"),
                                "payload_magic": stream.get("payload_magic"),
                                "blocker": blocker,
                            }
                        )
                        continue
                    if decoded is None:
                        blocker = "jcsp_aq_rawvideo_decoder_returned_no_stream"
                        blockers.append(blocker)
                        decode_failures.append(
                            {
                                "stream_name": stream.get("name"),
                                "codec_kind": stream.get("codec_kind"),
                                "payload_magic": stream.get("payload_magic"),
                                "blocker": blocker,
                            }
                        )
                        continue
                    rel = decoded.path
                    if rel in decoded_streams_by_name:
                        blockers.append("jcsp_aq_rawvideo_duplicate_raw_stream_name")
                        continue
                    decoded_streams_by_name[rel] = decoded

                stream_set = set(decoded_streams_by_name)
                expected_set = set(expected)
                if expected_set - stream_set:
                    blockers.append("jcsp_aq_rawvideo_missing_expected_raw_stream")
                if stream_set - expected_set:
                    blockers.append("jcsp_aq_rawvideo_unexpected_raw_stream")
                for rel in expected:
                    if (output_root / rel).exists():
                        blockers.append("jcsp_aq_rawvideo_output_target_already_exists")

    blockers = _dedupe(blockers)
    if not blockers:
        for rel in expected:
            decoded = decoded_streams_by_name[rel]
            path = output_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(decoded.payload)
            emitted_rows.append(decoded.manifest_row())

    if reference_raw_dir is not None and emitted_rows:
        parity_proof = prove_jcsp_runtime_raw_output_parity(
            expected,
            candidate_raw_dir=output_root,
            reference_raw_dir=reference_raw_dir,
            candidate_outputs_emitted_by_bridge=True,
            candidate_output_source=JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE,
            manifest_json=parity_manifest_json,
        )

    ready_for_output_parity = bool(
        parity_proof is not None
        and parity_proof.get("ready_for_output_parity") is True
    )
    raw_contract_consumed = bool(expected and emitted_rows and not blockers)
    ready_for_submission_runtime_consumption = (
        ready_for_output_parity
        if require_output_parity_for_runtime_consumption
        else raw_contract_consumed
    )
    parity_blockers = (
        parity_proof.get("dispatch_blockers", [])
        if parity_proof is not None
        else (
            [JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER]
            if require_output_parity_for_runtime_consumption
            else []
        )
    )
    dispatch_blockers = _dedupe(
        [
            *blockers,
            *parity_blockers,
            *(
                []
                if ready_for_submission_runtime_consumption
                else [JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER]
            ),
            "exact_cuda_auth_eval_missing",
        ]
    )
    byte_exact_parity = bool(
        parity_proof is not None
        and parity_proof.get("byte_exact_raw_output_parity") is True
    )
    manifest: dict[str, Any] = {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_EMISSION_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "runtime_action": "emit_real_aq_rawvideo_outputs",
        "candidate_output_source": JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE,
        "candidate_outputs_from_real_bridge_rawvideo": bool(emitted_rows),
        "member_name": member_name,
        "member_present": member_path.exists(),
        "consumes_required_member": bool(emitted_rows),
        "expected_raw_outputs": expected,
        "expected_raw_output_count": len(expected),
        "candidate_raw_dir": str(output_root),
        "reference_raw_dir": (
            str(Path(reference_raw_dir)) if reference_raw_dir is not None else None
        ),
        "decoder_interface": _decoder_interface_contract(expected),
        "decoder_adapter": decoder_adapter.contract(expected),
        "parsed_container": parsed_summary,
        "raw_output_emission_attempted": not blockers,
        "real_raw_outputs_emitted": bool(emitted_rows),
        "fixture_raw_outputs_emitted": False,
        "bridge_emits_contest_raw_outputs": bool(emitted_rows),
        "real_decoded_rawvideo_stream_count": sum(
            1 for row in emitted_rows if row["real_decoded_rawvideo"]
        ),
        "fixture_passthrough_stream_count": sum(
            1 for row in emitted_rows if row["fixture_passthrough"]
        ),
        "emitted_raw_output_count": len(emitted_rows),
        "emitted_raw_outputs": emitted_rows,
        "decode_failures": decode_failures,
        "output_parity_checked": parity_proof is not None,
        "output_parity_artifact": (
            str(Path(parity_manifest_json))
            if parity_manifest_json is not None
            else None
        ),
        "raw_output_parity_proof": parity_proof,
        "byte_exact_raw_output_parity": byte_exact_parity,
        "raw_output_parity_required_for_runtime_consumption": bool(
            require_output_parity_for_runtime_consumption
        ),
        "ready_for_output_parity": ready_for_output_parity,
        "ready_for_submission_runtime_consumption": (
            ready_for_submission_runtime_consumption
        ),
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": dispatch_blockers,
    }
    manifest = _with_manifest_sha256(manifest)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def consume_jcsp_real_raw_outputs(
    archive_dir: str | Path,
    *,
    video_names_file: str | Path | None,
    output_dir: str | Path,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Production JCSP runtime consumer for real AQ/rawvideo streams.

    This mode is intentionally narrow: every expected contest ``.raw`` output
    must be present as exactly one real AQv1/AQc1 arithmetic-static JCSP stream
    named by its relative ``.raw`` path.  Zero-stream no-ops, missing streams,
    extra streams, raw passthrough fixtures, Ballé streams, and stale output
    files all fail closed.  The bridge writes rawvideo bytes and clears only
    submission-runtime consumption; exact CUDA dispatch remains blocked.
    """

    expected_raw_outputs, names_error = _expected_raw_outputs_from_names_file(
        video_names_file
    )
    manifest = emit_jcsp_real_raw_outputs(
        archive_dir,
        expected_raw_outputs=expected_raw_outputs,
        output_dir=output_dir,
        member_name=member_name,
        manifest_json=None,
        parity_manifest_json=None,
        require_output_parity_for_runtime_consumption=False,
    )
    consume_blockers: list[str] = []
    if video_names_file is None:
        consume_blockers.append("jcsp_video_names_file_missing")
    if names_error is not None:
        consume_blockers.append("jcsp_video_names_file_parse_failed")
    if consume_blockers:
        manifest["ready_for_submission_runtime_consumption"] = False
        manifest["ready_for_exact_eval_dispatch"] = False

    manifest.update(
        {
            "runtime_action": "consume_real_aq_rawvideo_outputs",
            "production_bridge_mode": "consume-real-raw-outputs",
            "video_names_file": (
                str(Path(video_names_file)) if video_names_file is not None else None
            ),
            "video_names_file_parse_error": names_error,
            "no_scorer_at_inflate": True,
            "score_affecting_sidecars_allowed": False,
            "exact_eval_dispatch_blocked": True,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": _dedupe(
                [
                    *manifest.get("dispatch_blockers", []),
                    *consume_blockers,
                    "exact_cuda_auth_eval_missing",
                ]
            ),
        }
    )
    manifest = _with_manifest_sha256(manifest)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def emit_jcsp_real_aq_rawvideo_runtime_preflight(
    archive_dir: str | Path,
    *,
    video_names_file: str | Path | None,
    output_dir: str | Path,
    reference_raw_dir: str | Path | None = None,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    manifest_json: str | Path | None = None,
    parity_manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Fail-closed CLI adapter for the narrow real AQ/rawvideo bridge.

    This is the runtime-facing preflight step: derive expected contest ``.raw``
    paths from the names file, decode real AQv1/AQc1 JCSP streams into
    ``output_dir``, optionally prove byte parity against a reference raw tree,
    and still keep exact eval dispatch blocked.  It imports no scorer modules
    and makes no score claim.
    """

    expected_raw_outputs, names_error = _expected_raw_outputs_from_names_file(
        video_names_file
    )
    manifest = emit_jcsp_real_raw_outputs(
        archive_dir,
        expected_raw_outputs=expected_raw_outputs,
        output_dir=output_dir,
        reference_raw_dir=reference_raw_dir,
        member_name=member_name,
        parity_manifest_json=parity_manifest_json,
    )
    preflight_blockers: list[str] = []
    if video_names_file is None:
        preflight_blockers.append("jcsp_video_names_file_missing")
    if names_error is not None:
        preflight_blockers.append("jcsp_video_names_file_parse_failed")

    if preflight_blockers:
        manifest["ready_for_output_parity"] = False
        manifest["ready_for_submission_runtime_consumption"] = False
        manifest["ready_for_exact_eval_dispatch"] = False

    manifest.update(
        {
            "runtime_action": "emit_real_aq_rawvideo_runtime_preflight_fail_closed",
            "runtime_preflight_adapter": {
                "schema": JCSP_RUNTIME_PREFLIGHT_ADAPTER_SCHEMA,
                "adapter_id": JCSP_RUNTIME_REAL_AQ_PREFLIGHT_ADAPTER_ID,
                "score_claim": False,
                "dispatch_attempted": False,
                "no_scorer_at_inflate": True,
                "decoder_adapter_id": JCSP_RUNTIME_AQ_RAWVIDEO_ADAPTER_ID,
                "candidate_output_source": JCSP_RUNTIME_REAL_RAW_OUTPUT_SOURCE,
                "required_reference_for_submission_runtime_consumption": (
                    "byte-exact raw output parity proof"
                ),
                "ready_for_exact_eval_dispatch": False,
            },
            "video_names_file": (
                str(Path(video_names_file)) if video_names_file is not None else None
            ),
            "video_names_file_parse_error": names_error,
            "no_scorer_at_inflate": True,
            "exact_eval_dispatch_blocked": True,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": _dedupe(
                [
                    *manifest.get("dispatch_blockers", []),
                    *preflight_blockers,
                    "exact_cuda_auth_eval_missing",
                ]
            ),
        }
    )
    manifest = _with_manifest_sha256(manifest)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def _contest_output_contract(
    *,
    member_present: bool,
    inflated_dir: str | Path | None,
    video_names_file: str | Path | None,
) -> dict[str, Any]:
    expected_raw_outputs, names_error = _expected_raw_outputs_from_names_file(
        video_names_file,
    )
    observed = _observe_existing_raw_outputs(inflated_dir, expected_raw_outputs)
    existing_count = sum(1 for row in observed if row["exists"])
    parity_contract = _raw_output_parity_contract(
        expected_raw_outputs=expected_raw_outputs,
        existing_raw_output_count=existing_count,
        names_error=names_error,
    )
    blockers: list[str] = []
    if member_present:
        blockers.extend(
            [
                JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
                "jcsp_runtime_probe_does_not_emit_raw_outputs",
                "jcsp_raw_output_emission_missing",
            ]
        )
        if names_error is not None:
            blockers.append("jcsp_video_names_file_parse_failed")
        if existing_count:
            blockers.append("jcsp_existing_raw_outputs_unproven")
    return {
        "schema": JCSP_RUNTIME_OUTPUT_CONTRACT_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "required_when_jcsp_member_present": member_present,
        "contest_output_format": "one uint8 RGB rawvideo .raw per names_file entry",
        "video_names_file_observed": video_names_file is not None,
        "video_names_file_parse_error": names_error,
        "inflated_dir_observed": inflated_dir is not None,
        "expected_raw_outputs": expected_raw_outputs,
        "expected_raw_output_count": len(expected_raw_outputs),
        "existing_raw_outputs_at_probe": observed,
        "existing_raw_output_count": existing_count,
        "decoder_interface": _decoder_interface_contract(expected_raw_outputs),
        "raw_output_parity_contract": parity_contract,
        "minimal_fixture_stream_path": _minimal_fixture_stream_path_contract(
            expected_raw_outputs
        ),
        "required_raw_output_parity_proof_schema": (
            JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA
        ),
        "bridge_emits_contest_raw_outputs": False,
        "raw_output_emission_attempted": False,
        "output_parity_checked": False,
        "output_parity_artifact": None,
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "dispatch_blocker": JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
        "dispatch_blockers": _dedupe(blockers),
    }


def _require_available(blob: bytes, cursor: int, n_bytes: int, context: str) -> None:
    if n_bytes < 0 or cursor < 0 or cursor + n_bytes > len(blob):
        raise ValueError(
            f"truncated {context} at offset {cursor}; need {n_bytes} bytes, "
            f"blob len={len(blob)}"
        )


def _payload_magic_for_kind(
    *,
    codec_kind: int,
    payload: bytes,
    stream_name: str,
) -> str:
    if codec_kind == KIND_RAW_PASSTHROUGH:
        if not payload:
            raise ValueError(f"stream {stream_name!r} raw payload is empty")
        return payload[:4].decode("ascii", errors="replace")
    allowed = _PAYLOAD_MAGICS_BY_CODEC_KIND.get(int(codec_kind))
    if allowed is None:
        raise ValueError(f"stream {stream_name!r} has invalid codec_kind {codec_kind}")
    if len(payload) < 4:
        raise ValueError(
            f"stream {stream_name!r} payload is too small for codec magic"
        )
    magic = payload[:4]
    if magic not in allowed:
        allowed_text = ", ".join(repr(item) for item in allowed)
        raise ValueError(
            f"stream {stream_name!r} payload magic {magic!r} is incompatible "
            f"with codec_kind {codec_kind}; expected one of {allowed_text}"
        )
    return magic.decode("ascii", errors="replace")


def _parse_real_jcsp_container(
    blob: bytes,
    *,
    include_payload: bool = False,
) -> dict[str, Any]:
    _require_available(blob, 0, 7, "JCSP header")
    if blob[:4] != JCSP_MAGIC:
        raise ValueError(f"bad JCSP magic {blob[:4]!r}")
    cursor = 4
    (version,) = struct.unpack_from("<H", blob, cursor)
    cursor += 2
    if version != JCSP_VERSION:
        raise ValueError(f"unsupported JCSP version {version}")
    (stream_count,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1

    streams: list[dict[str, Any]] = []
    names: list[str] = []
    for index in range(int(stream_count)):
        _require_available(blob, cursor, 1, f"stream {index} name length")
        (name_len,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        if name_len <= 0:
            raise ValueError(f"stream {index} has empty name")
        _require_available(blob, cursor, name_len, f"stream {index} name")
        name = blob[cursor : cursor + name_len].decode("utf-8", errors="strict")
        cursor += name_len
        if "\x00" in name:
            raise ValueError(f"stream {index} name contains NUL")
        if name in names:
            raise ValueError(f"duplicate stream name {name!r}")
        names.append(name)

        _require_available(blob, cursor, 1, f"stream {name!r} codec kind")
        (codec_kind,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        if codec_kind not in (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
        ):
            raise ValueError(f"stream {name!r} has invalid codec_kind {codec_kind}")

        _require_available(blob, cursor, 4, f"stream {name!r} ADMM target")
        (admm_bytes_target,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} actual bytes")
        (actual_bytes,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} score delta")
        (score_delta_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} marginal")
        (marginal_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} payload length")
        (payload_len,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        if int(actual_bytes) != int(payload_len):
            raise ValueError(
                f"stream {name!r} actual_bytes={actual_bytes} does not match "
                f"payload_len={payload_len}"
            )
        _require_available(blob, cursor, payload_len, f"stream {name!r} payload")
        payload = blob[cursor : cursor + payload_len]
        cursor += payload_len
        payload_magic = _payload_magic_for_kind(
            codec_kind=int(codec_kind),
            payload=payload,
            stream_name=name,
        )
        row: dict[str, Any] = {
            "index": index,
            "name": name,
            "codec_kind": int(codec_kind),
            "admm_bytes_target": int(admm_bytes_target),
            "actual_bytes": int(actual_bytes),
            "score_delta_milli": int(score_delta_milli),
            "marginal_milli": int(marginal_milli),
            "payload_magic": payload_magic,
            "payload_sha256": _sha256_bytes(payload),
        }
        if include_payload:
            row["payload"] = payload
        streams.append(row)

    _require_available(blob, cursor, 4, "JCSP KKT residual")
    (kkt_residual_milli,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    _require_available(blob, cursor, 4, "JCSP iteration count")
    (iters,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    _require_available(blob, cursor, 1, "JCSP converged flag")
    (converged_raw,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if converged_raw not in (0, 1):
        raise ValueError(f"invalid JCSP converged flag {converged_raw}")
    if cursor != len(blob):
        raise ValueError(
            f"trailing bytes after JCSP container: cursor={cursor}, len={len(blob)}"
        )
    return {
        "container_magic": "JCSP",
        "container_version": int(version),
        "stream_count": int(stream_count),
        "streams": streams,
        "waterline_kkt_residual_milli": int(kkt_residual_milli),
        "iters": int(iters),
        "converged": bool(converged_raw),
        "noop_fixture": int(stream_count) == 0,
    }


def _probe_local_skeleton_container(blob: bytes) -> dict[str, Any]:
    details: dict[str, Any] = {
        "container_magic": "JCSK",
        "refused_preview_member": True,
    }
    if len(blob) < 10:
        details["preview_parse_error"] = "truncated JCSK header"
        return details
    (version,) = struct.unpack_from("<H", blob, 4)
    (body_len,) = struct.unpack_from("<I", blob, 6)
    body_start = 10
    body_end = body_start + int(body_len)
    details["container_version"] = int(version)
    details["declared_body_bytes"] = int(body_len)
    if version != JCSK_VERSION:
        details["preview_parse_error"] = f"unsupported JCSK version {version}"
        return details
    if body_end != len(blob):
        details["preview_parse_error"] = (
            f"JCSK body length mismatch declared={body_len} "
            f"actual={len(blob) - body_start}"
        )
        return details
    try:
        manifest = json.loads(
            blob[body_start:body_end].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_pairs,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        details["preview_parse_error"] = str(exc)
        return details
    if not isinstance(manifest, Mapping):
        details["preview_parse_error"] = "JCSK manifest is not a mapping"
        return details
    details["preview_manifest_schema"] = str(manifest.get("schema", ""))
    details["preview_manifest_sha256"] = str(manifest.get("manifest_sha256", ""))
    try:
        details["preview_stream_count"] = int(manifest.get("stream_count", -1))
    except (TypeError, ValueError):
        details["preview_parse_error"] = "JCSK manifest stream_count is invalid"
    return details


def probe_jcsp_runtime_bridge(
    archive_dir: str | Path,
    *,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    inflated_dir: str | Path | None = None,
    video_names_file: str | Path | None = None,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Probe ``archive_dir/member_name`` and return a deterministic contract.

    A present JCSP member is never treated as dispatch-ready by this tranche.
    Real ``JCSP`` bytes are parsed and then refused because the probe path does
    not emit raw outputs or prove parity.  Local ``JCSK`` preview bytes are
    refused before runtime-loader readiness.
    """

    archive_root = Path(archive_dir)
    member_path = archive_root / member_name
    member_exists = member_path.exists()
    output_contract = _contest_output_contract(
        member_present=member_exists,
        inflated_dir=inflated_dir,
        video_names_file=video_names_file,
    )
    base: dict[str, Any] = {
        "schema": JCSP_RUNTIME_BRIDGE_PROBE_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "required_submission_runtime": JCSP_REQUIRED_SUBMISSION_RUNTIME,
        "runtime_bridge_path": JCSP_RUNTIME_BRIDGE_PATH,
        "member_name": member_name,
        "member_present": member_exists,
        "detects_required_member": member_exists,
        "detected_real_jcsp_member": False,
        "refused_preview_member": False,
        "ready_for_runtime_probe": True,
        "ready_for_runtime_loader": False,
        "consumes_required_member": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_action": "no_jcsp_member_present",
        "contest_output_contract": output_contract,
        "dispatch_blockers": [],
    }
    if not member_path.exists():
        manifest = _with_manifest_sha256(base)
        if manifest_json is not None:
            _write_manifest(Path(manifest_json), manifest)
        return manifest
    if not member_path.is_file():
        base.update(
            {
                "runtime_action": "refuse_non_file_jcsp_member_path",
                "refusal_reason": "jcsp member path is not a regular file",
                "dispatch_blockers": [
                    "jcsp_member_path_not_regular_file",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    *output_contract["dispatch_blockers"],
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
        manifest = _with_manifest_sha256(base)
        if manifest_json is not None:
            _write_manifest(Path(manifest_json), manifest)
        return manifest

    blob = member_path.read_bytes()
    base.update(
        {
            "member_bytes": len(blob),
            "member_sha256": _sha256_bytes(blob),
            "member_prefix_hex": blob[:16].hex(),
        }
    )
    if len(blob) < 4:
        base.update(
            {
                "runtime_action": "refuse_invalid_jcsp_member",
                "refusal_reason": "jcsp member is too small for magic",
                "dispatch_blockers": [
                    "jcsp_member_too_small_for_magic",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    *output_contract["dispatch_blockers"],
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
    elif blob[:4] == JCSK_MAGIC:
        details = _probe_local_skeleton_container(blob)
        base.update(details)
        base.update(
            {
                "runtime_action": "refuse_jcsk_preview_member",
                "refusal_reason": (
                    "jcsp.bin contains local JCSK preview bytes, not the "
                    "runtime JCSP container"
                ),
                "dispatch_blockers": [
                    JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    *output_contract["dispatch_blockers"],
                    "strict_preflight_proof_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
    elif blob[:4] == JCSP_MAGIC:
        try:
            parsed = _parse_real_jcsp_container(blob)
        except ValueError as exc:
            base.update(
                {
                    "container_magic": "JCSP",
                    "runtime_action": "refuse_invalid_jcsp_container",
                    "refusal_reason": str(exc),
                    "dispatch_blockers": [
                        "jcsp_runtime_probe_parse_failed",
                        JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                        *output_contract["dispatch_blockers"],
                        "exact_cuda_auth_eval_missing",
                    ],
                }
            )
        else:
            base.update(parsed)
            if parsed["noop_fixture"]:
                base.update(
                    {
                        "detected_real_jcsp_member": True,
                        "ready_for_runtime_loader": False,
                        "runtime_action": "refuse_zero_stream_jcsp_noop_packet",
                        "refusal_reason": (
                            "real JCSP container has zero streams; refusing "
                            "no-op packet before runtime consumption"
                        ),
                        "dispatch_blockers": [
                            JCSP_RUNTIME_NOOP_PACKET_BLOCKER,
                            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                            *output_contract["dispatch_blockers"],
                            "exact_cuda_auth_eval_missing",
                        ],
                    }
                )
            else:
                base.update(
                    {
                        "detected_real_jcsp_member": True,
                        "ready_for_runtime_loader": True,
                        "runtime_action": (
                            "refuse_until_jcsp_raw_output_emission_and_parity"
                        ),
                        "refusal_reason": (
                            "real JCSP container parsed and a narrow AQ rawvideo "
                            "adapter exists, but the probe path does not emit raw "
                            "outputs or prove parity"
                        ),
                        "dispatch_blockers": [
                            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                            "jcsp_runtime_preflight_adapter_not_requested",
                            *output_contract["dispatch_blockers"],
                            "exact_cuda_auth_eval_missing",
                        ],
                    }
                )
    else:
        base.update(
            {
                "container_magic": blob[:4].decode("ascii", errors="replace"),
                "runtime_action": "refuse_unknown_jcsp_member_magic",
                "refusal_reason": (
                    f"unknown jcsp.bin magic {blob[:4]!r}; expected "
                    f"{JCSP_MAGIC!r} or refused preview {JCSK_MAGIC!r}"
                ),
                "dispatch_blockers": [
                    "jcsp_unknown_member_magic",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    *output_contract["dispatch_blockers"],
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )

    base["dispatch_blockers"] = _dedupe(base["dispatch_blockers"])
    manifest = _with_manifest_sha256(base)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe robust_current JCSP runtime bridge state"
    )
    parser.add_argument("archive_dir", help="inflater archive directory")
    parser.add_argument(
        "--mode",
        choices=(
            "probe",
            "plan-real-raw-outputs",
            "emit-real-raw-outputs",
            "consume-real-raw-outputs",
        ),
        default="probe",
        help=(
            "probe only, plan the real AQ rawvideo consumer without writes, "
            "run the fail-closed real AQ rawvideo preflight emitter, or consume "
            "real AQ rawvideo streams into contest .raw outputs"
        ),
    )
    parser.add_argument(
        "--member-name",
        default=JCSP_ARCHIVE_MEMBER_NAME,
        help="JCSP member filename inside archive_dir",
    )
    parser.add_argument(
        "--manifest-json",
        required=True,
        help="path to write deterministic probe manifest JSON",
    )
    parser.add_argument(
        "--inflated-dir",
        default=None,
        help="inflated output directory used to inspect pre-existing raw outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "raw output directory for emit-real-raw-outputs and "
            "consume-real-raw-outputs modes; defaults to --inflated-dir"
        ),
    )
    parser.add_argument(
        "--reference-raw-dir",
        default=None,
        help="optional reference rawvideo directory for byte parity proof",
    )
    parser.add_argument(
        "--parity-manifest-json",
        default=None,
        help="optional path to write raw output parity proof JSON",
    )
    parser.add_argument(
        "--video-names-file",
        default=None,
        help="contest video names file used to derive required .raw outputs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.mode == "consume-real-raw-outputs":
        output_dir = args.output_dir or args.inflated_dir
        if output_dir is None:
            print(
                "[jcsp-runtime-bridge] FATAL: --mode consume-real-raw-outputs "
                "requires --output-dir or --inflated-dir",
                file=sys.stderr,
            )
            return 2
        manifest = consume_jcsp_real_raw_outputs(
            args.archive_dir,
            member_name=args.member_name,
            video_names_file=args.video_names_file,
            output_dir=output_dir,
            manifest_json=args.manifest_json,
        )
        if (
            manifest.get("ready_for_submission_runtime_consumption") is True
            and manifest.get("real_raw_outputs_emitted") is True
            and manifest.get("candidate_outputs_from_real_bridge_rawvideo") is True
        ):
            print(
                "[jcsp-runtime-bridge] consumed real AQ rawvideo JCSP streams; "
                f"manifest: {args.manifest_json}",
                file=sys.stderr,
            )
            return 0
        print(
            "[jcsp-runtime-bridge] wrote deterministic real AQ rawvideo "
            f"production manifest: {args.manifest_json}",
            file=sys.stderr,
        )
        print(
            "[jcsp-runtime-bridge] FATAL: real AQ rawvideo production "
            "consumption refused; blockers="
            f"{','.join(manifest.get('dispatch_blockers', []))}",
            file=sys.stderr,
        )
        return EXIT_JCSP_MEMBER_REFUSED

    if args.mode == "plan-real-raw-outputs":
        expected_raw_outputs, names_error = _expected_raw_outputs_from_names_file(
            args.video_names_file
        )
        manifest = plan_jcsp_real_raw_output_emission(
            args.archive_dir,
            member_name=args.member_name,
            expected_raw_outputs=expected_raw_outputs,
            manifest_json=args.manifest_json,
        )
        if names_error is not None:
            manifest["ready_for_raw_output_emission"] = False
            manifest["ready_for_output_parity"] = False
            manifest["ready_for_submission_runtime_consumption"] = False
            manifest["ready_for_exact_eval_dispatch"] = False
            manifest["video_names_file_parse_error"] = names_error
            manifest["dispatch_blockers"] = _dedupe(
                [
                    *manifest.get("dispatch_blockers", []),
                    "jcsp_video_names_file_parse_failed",
                    "exact_cuda_auth_eval_missing",
                ]
            )
            manifest = _with_manifest_sha256(manifest)
            _write_manifest(Path(args.manifest_json), manifest)
        if not manifest["member_present"]:
            return 0
        print(
            "[jcsp-runtime-bridge] wrote deterministic real AQ rawvideo "
            f"consumer readiness manifest: {args.manifest_json}",
            file=sys.stderr,
        )
        print(
            "[jcsp-runtime-bridge] FATAL: real AQ rawvideo consumer plan "
            "does not run branch dispatch; blockers="
            f"{','.join(manifest.get('dispatch_blockers', []))}",
            file=sys.stderr,
        )
        return EXIT_JCSP_MEMBER_REFUSED

    if args.mode == "emit-real-raw-outputs":
        output_dir = args.output_dir or args.inflated_dir
        if output_dir is None:
            print(
                "[jcsp-runtime-bridge] FATAL: --mode emit-real-raw-outputs "
                "requires --output-dir or --inflated-dir",
                file=sys.stderr,
            )
            return 2
        manifest = emit_jcsp_real_aq_rawvideo_runtime_preflight(
            args.archive_dir,
            member_name=args.member_name,
            video_names_file=args.video_names_file,
            output_dir=output_dir,
            reference_raw_dir=args.reference_raw_dir,
            manifest_json=args.manifest_json,
            parity_manifest_json=args.parity_manifest_json,
        )
        if not manifest["member_present"]:
            return 0
        print(
            "[jcsp-runtime-bridge] wrote deterministic real AQ rawvideo "
            f"preflight manifest: {args.manifest_json}",
            file=sys.stderr,
        )
        print(
            "[jcsp-runtime-bridge] FATAL: real AQ rawvideo preflight remains "
            "closed to exact eval dispatch; blockers="
            f"{','.join(manifest.get('dispatch_blockers', []))}",
            file=sys.stderr,
        )
        return EXIT_JCSP_MEMBER_REFUSED

    manifest = probe_jcsp_runtime_bridge(
        args.archive_dir,
        member_name=args.member_name,
        inflated_dir=args.inflated_dir,
        video_names_file=args.video_names_file,
        manifest_json=args.manifest_json,
    )
    if not manifest["member_present"]:
        return 0
    print(
        "[jcsp-runtime-bridge] wrote deterministic probe manifest: "
        f"{args.manifest_json}",
        file=sys.stderr,
    )
    print(
        "[jcsp-runtime-bridge] FATAL: "
        f"{manifest.get('refusal_reason', 'jcsp member refused')}",
        file=sys.stderr,
    )
    return EXIT_JCSP_MEMBER_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
