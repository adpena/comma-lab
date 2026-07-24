# SPDX-License-Identifier: MIT
"""Exact, bounded solved-value records and lossless prices for DDM #669c.

This module prices semantic records extracted from the SHA-bound IS1 solved
planes.  It does not call ``evaluate.py``, emit an archive, or claim a score.
Large scorer planes are consumed one pair at a time and never persisted.
"""

from __future__ import annotations

import json
import lzma
import os
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tac.lossless.range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.solve_diff_operator_mining import (
    AXIS,
    POINTER,
    SolveDiffMiningConfigV1,
    _load_production_inputs,
    _open_production_inputs,
    load_sha256_checked_bytes,
    realize_solve_camera,
    sha256_file,
)

SCHEMA = "ddm_dm1_solved_value_pricing.v1"
ROW_SCHEMA = "ddm_dm1_solved_value_row.v1"
CONFIG_SCHEMA = "ddm_dm1_solved_value_pricing_config.v1"
ROW_MAGIC = b"DM1ROW1\0"
CODEC_MAGIC = b"DM1COD1\0"
JOINT_MAGIC = b"DM1JNT1\0"
CODECS = ("zlib9", "lzma9", "context_arithmetic")
_CODEC_IDS = {name: index for index, name in enumerate(CODECS)}
_ID_CODECS = {index: name for name, index in _CODEC_IDS.items()}
_ROW_HEADER = struct.Struct("<8sBHBBBBH")
_COUNT_AND_BYTES = struct.Struct("<I")
_CODEC_HEADER = struct.Struct("<8sBQQ32s")
_JOINT_COUNT = struct.Struct("<8sH")
_SHA_BYTES = 32
_MAX_RANGE_TOTAL = 32_768
_REPO = Path(__file__).resolve().parents[3]


class SolvedValuePricingError(ValueError):
    """Fail-closed custody, semantic-record, or parseback error."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def checked_json(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    raw = load_sha256_checked_bytes(path, expected_sha256)
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise SolvedValuePricingError(f"{path} must contain a JSON object")
    return value


def _encode_varint(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SolvedValuePricingError("varint value must be a nonnegative integer")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _decode_varint(raw: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(raw):
            raise SolvedValuePricingError("truncated varint")
        byte = raw[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise SolvedValuePricingError("overlong varint")


def support_sha256(flat_indices: np.ndarray) -> str:
    indices = np.asarray(flat_indices, dtype="<u4")
    if indices.ndim != 1 or len(indices) == 0:
        raise SolvedValuePricingError("row support must be a nonempty 1D array")
    if np.any(indices[1:] <= indices[:-1]):
        raise SolvedValuePricingError("row support indices must be strictly increasing")
    return sha256(indices.tobytes()).hexdigest()


def _class_pair_from_bucket(bucket_id: str) -> tuple[str, str]:
    stem = bucket_id.split("__", maxsplit=1)[0]
    parts = stem.split("_")
    if len(parts) != 2 or any(not value for value in parts):
        raise SolvedValuePricingError(f"cannot parse class pair from bucket {bucket_id!r}")
    return parts[0], parts[1]


def _stratum_from_bucket(bucket_id: str) -> Literal["boundary", "cell"]:
    parts = bucket_id.split("__")
    if len(parts) != 3 or parts[1] not in {"boundary", "cell"}:
        raise SolvedValuePricingError(f"cannot parse stratum from bucket {bucket_id!r}")
    return parts[1]  # type: ignore[return-value]


@dataclass(frozen=True)
class SolvedValueRecord:
    """One canonical semantic record before lossless coding."""

    pair_id: int
    bucket_id: str
    class_left: int
    class_right: int
    stream_type: StreamType
    layer_home: LayerHome
    support_sha256: str
    winners: bytes
    margin_relations: bytes
    flat_indices: tuple[int, ...] = ()

    @property
    def stratum(self) -> Literal["boundary", "cell"]:
        return _stratum_from_bucket(self.bucket_id)

    @property
    def support_count(self) -> int:
        return len(self.winners)

    def __post_init__(self) -> None:
        if not 0 <= self.pair_id < 600:
            raise SolvedValuePricingError("pair_id must be in [0,600)")
        if not self.bucket_id or len(self.bucket_id.encode()) > 65_535:
            raise SolvedValuePricingError("bucket_id must be nonempty and bounded")
        if not 0 <= self.class_left <= 255 or not 0 <= self.class_right <= 255:
            raise SolvedValuePricingError("class ids must be uint8")
        if self.class_left == self.class_right:
            raise SolvedValuePricingError("class pair must contain distinct classes")
        if len(self.support_sha256) != 64:
            raise SolvedValuePricingError("support SHA-256 must be lowercase hex")
        try:
            bytes.fromhex(self.support_sha256)
        except ValueError as error:
            raise SolvedValuePricingError("support SHA-256 must be lowercase hex") from error
        if not self.winners or any(value not in (0, 1, 2) for value in self.winners):
            raise SolvedValuePricingError("winner symbols must be nonempty and in {0,1,2}")
        if len(self.margin_relations) != len(self.winners):
            raise SolvedValuePricingError("margin relation count must equal winner count")
        if any(value not in (0, 1, 2) for value in self.margin_relations):
            raise SolvedValuePricingError("margin relations must be in {0,1,2}")
        if self.stratum == "boundary":
            if self.stream_type is not StreamType.SKELETON:
                raise SolvedValuePricingError("boundary records must be SKELETON")
            if self.layer_home is not LayerHome.L4_SCORER_FEATURE:
                raise SolvedValuePricingError("boundary records must be homed at deepest L4")
            if len(self.flat_indices) != len(self.winners):
                raise SolvedValuePricingError("boundary records must carry exact placement")
            indices = np.asarray(self.flat_indices, dtype=np.int64)
            if np.any(indices[1:] <= indices[:-1]):
                raise SolvedValuePricingError("boundary placement must be strictly increasing")
            if support_sha256(indices.astype("<u4")) != self.support_sha256:
                raise SolvedValuePricingError("boundary placement SHA-256 mismatch")
        else:
            if self.stream_type is not StreamType.FIBER:
                raise SolvedValuePricingError("cell records must be FIBER")
            if self.layer_home is not LayerHome.L4_SCORER_FEATURE:
                raise SolvedValuePricingError("cell records must be homed at L4")
            if self.flat_indices:
                raise SolvedValuePricingError(
                    "cell record must reference, not recount, existing support"
                )

    def encode(self) -> bytes:
        bucket = self.bucket_id.encode()
        body = bytearray(
            _ROW_HEADER.pack(
                ROW_MAGIC,
                0 if self.stratum == "boundary" else 1,
                self.pair_id,
                self.class_left,
                self.class_right,
                list(StreamType).index(self.stream_type),
                list(LayerHome).index(self.layer_home),
                len(bucket),
            )
        )
        body.extend(bucket)
        body.extend(_COUNT_AND_BYTES.pack(self.support_count))
        body.extend(bytes.fromhex(self.support_sha256))
        if self.stratum == "boundary":
            previous = 0
            for index in self.flat_indices:
                body.extend(_encode_varint(index - previous))
                previous = index
        body.extend(self.winners)
        body.extend(self.margin_relations)
        return bytes(body)

    @classmethod
    def decode(
        cls,
        raw: bytes,
        *,
        external_cell_support: np.ndarray | None = None,
    ) -> SolvedValueRecord:
        if len(raw) < _ROW_HEADER.size + _COUNT_AND_BYTES.size + _SHA_BYTES:
            raise SolvedValuePricingError("row record is truncated")
        (
            magic,
            kind,
            pair_id,
            class_left,
            class_right,
            stream_index,
            layer_index,
            bucket_bytes,
        ) = _ROW_HEADER.unpack_from(raw)
        if magic != ROW_MAGIC or kind not in (0, 1):
            raise SolvedValuePricingError("row magic/kind mismatch")
        try:
            stream_type = list(StreamType)[stream_index]
            layer_home = list(LayerHome)[layer_index]
        except IndexError as error:
            raise SolvedValuePricingError("row stream/layer enum index invalid") from error
        offset = _ROW_HEADER.size
        bucket_end = offset + bucket_bytes
        if bucket_end > len(raw):
            raise SolvedValuePricingError("row bucket is truncated")
        try:
            bucket_id = raw[offset:bucket_end].decode()
        except UnicodeDecodeError as error:
            raise SolvedValuePricingError("row bucket is not UTF-8") from error
        offset = bucket_end
        if offset + _COUNT_AND_BYTES.size + _SHA_BYTES > len(raw):
            raise SolvedValuePricingError("row support header is truncated")
        support_count = _COUNT_AND_BYTES.unpack_from(raw, offset)[0]
        offset += _COUNT_AND_BYTES.size
        support_hash = raw[offset : offset + _SHA_BYTES].hex()
        offset += _SHA_BYTES
        indices: tuple[int, ...] = ()
        if kind == 0:
            parsed: list[int] = []
            previous = 0
            for _ in range(support_count):
                delta, offset = _decode_varint(raw, offset)
                current = previous + delta
                parsed.append(current)
                previous = current
            indices = tuple(parsed)
        payload_end = offset + 2 * support_count
        if payload_end != len(raw):
            raise SolvedValuePricingError("row payload length/trailing bytes mismatch")
        winners = raw[offset : offset + support_count]
        relations = raw[offset + support_count : payload_end]
        record = cls(
            pair_id=pair_id,
            bucket_id=bucket_id,
            class_left=class_left,
            class_right=class_right,
            stream_type=stream_type,
            layer_home=layer_home,
            support_sha256=support_hash,
            winners=winners,
            margin_relations=relations,
            flat_indices=indices,
        )
        if kind == 1:
            if external_cell_support is None:
                raise SolvedValuePricingError("cell parseback requires SHA-verified support")
            if support_sha256(external_cell_support) != support_hash:
                raise SolvedValuePricingError("external cell support SHA-256 mismatch")
            if len(external_cell_support) != support_count:
                raise SolvedValuePricingError("external cell support count mismatch")
        if record.encode() != raw:
            raise SolvedValuePricingError("row record is not canonical")
        return record


def _adaptive_frequencies() -> list[list[int]]:
    return [[1] * 256 for _ in range(256)]


def _rescale(counts: list[int]) -> None:
    if sum(counts) >= _MAX_RANGE_TOTAL:
        counts[:] = [max(1, (value + 1) // 2) for value in counts]


def encode_context_arithmetic(raw: bytes) -> bytes:
    if not raw:
        raise SolvedValuePricingError("context arithmetic input must be nonempty")
    contexts = _adaptive_frequencies()
    encoder = RangeEncoder()
    previous = 0
    for symbol in raw:
        counts = contexts[previous]
        cumulative, total = cumulative_frequencies(counts)
        encoder.encode(symbol=symbol, cumulative=cumulative, total=total)
        counts[symbol] += 1
        _rescale(counts)
        previous = symbol
    return encoder.finish()


def decode_context_arithmetic(encoded: bytes, raw_length: int) -> bytes:
    if raw_length <= 0:
        raise SolvedValuePricingError("context arithmetic raw length must be positive")
    contexts = _adaptive_frequencies()
    decoder = RangeDecoder(encoded)
    out = bytearray()
    previous = 0
    for _ in range(raw_length):
        counts = contexts[previous]
        cumulative, total = cumulative_frequencies(counts)
        target = decoder.target(total)
        symbol = int(np.searchsorted(cumulative, target, side="right") - 1)
        if not 0 <= symbol < 256:
            raise SolvedValuePricingError("context arithmetic symbol is invalid")
        decoder.update(
            low_count=cumulative[symbol],
            high_count=cumulative[symbol + 1],
            total=total,
        )
        out.append(symbol)
        counts[symbol] += 1
        _rescale(counts)
        previous = symbol
    return bytes(out)


def _encode_codec_payload(raw: bytes, codec: str) -> bytes:
    if codec == "zlib9":
        return zlib.compress(raw, level=9)
    if codec == "lzma9":
        return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9)
    if codec == "context_arithmetic":
        return encode_context_arithmetic(raw)
    raise SolvedValuePricingError(f"unknown codec {codec!r}")


def encode_codec(raw: bytes, codec: str) -> bytes:
    payload = _encode_codec_payload(raw, codec)
    return _CODEC_HEADER.pack(
        CODEC_MAGIC,
        _CODEC_IDS[codec],
        len(raw),
        len(payload),
        sha256(raw).digest(),
    ) + payload


def decode_codec(container: bytes) -> tuple[str, bytes]:
    if len(container) < _CODEC_HEADER.size:
        raise SolvedValuePricingError("codec container is truncated")
    magic, codec_id, raw_length, payload_length, raw_hash = _CODEC_HEADER.unpack_from(container)
    if magic != CODEC_MAGIC or codec_id not in _ID_CODECS:
        raise SolvedValuePricingError("codec container magic/id mismatch")
    if len(container) != _CODEC_HEADER.size + payload_length:
        raise SolvedValuePricingError("codec container payload length mismatch")
    payload = container[_CODEC_HEADER.size :]
    codec = _ID_CODECS[codec_id]
    try:
        if codec == "zlib9":
            raw = zlib.decompress(payload)
        elif codec == "lzma9":
            raw = lzma.decompress(payload, format=lzma.FORMAT_XZ)
        else:
            raw = decode_context_arithmetic(payload, raw_length)
    except (zlib.error, lzma.LZMAError, ValueError) as error:
        raise SolvedValuePricingError("codec container failed exact decode") from error
    if len(raw) != raw_length or sha256(raw).digest() != raw_hash:
        raise SolvedValuePricingError("codec container exact parseback hash mismatch")
    if _encode_codec_payload(raw, codec) != payload:
        raise SolvedValuePricingError("codec container is not canonical")
    return codec, raw


def encode_joint_raw(row_raws: Sequence[bytes]) -> bytes:
    if len(row_raws) != 25:
        raise SolvedValuePricingError("joint record must contain exactly 25 rows")
    out = bytearray(_JOINT_COUNT.pack(JOINT_MAGIC, len(row_raws)))
    for raw in row_raws:
        out.extend(struct.pack("<I", len(raw)))
        out.extend(raw)
    return bytes(out)


def decode_joint_raw(raw: bytes) -> tuple[bytes, ...]:
    if len(raw) < _JOINT_COUNT.size:
        raise SolvedValuePricingError("joint raw record is truncated")
    magic, count = _JOINT_COUNT.unpack_from(raw)
    if magic != JOINT_MAGIC or count != 25:
        raise SolvedValuePricingError("joint raw magic/count mismatch")
    offset = _JOINT_COUNT.size
    rows: list[bytes] = []
    for _ in range(count):
        if offset + 4 > len(raw):
            raise SolvedValuePricingError("joint row length is truncated")
        length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        end = offset + length
        if end > len(raw):
            raise SolvedValuePricingError("joint row is truncated")
        rows.append(raw[offset:end])
        offset = end
    if offset != len(raw):
        raise SolvedValuePricingError("joint record has trailing bytes")
    return tuple(rows)


def price_raw(raw: bytes) -> tuple[dict[str, dict[str, Any]], str]:
    prices: dict[str, dict[str, Any]] = {}
    for codec in CODECS:
        container = encode_codec(raw, codec)
        decoded_codec, decoded = decode_codec(container)
        if decoded_codec != codec or decoded != raw:
            raise SolvedValuePricingError(f"{codec} exact parseback failed")
        prices[codec] = {
            "container_bytes": len(container),
            "container_sha256": sha256_bytes(container),
            "payload_bytes": len(container) - _CODEC_HEADER.size,
            "parseback_exact": True,
        }
    winner = min(CODECS, key=lambda name: (prices[name]["container_bytes"], CODECS.index(name)))
    return prices, winner


def typed_home(stratum: str, counted_bytes: int) -> TypedStreamTag:
    """Re-home solved information at the deepest layer that still changes L5."""
    if stratum == "boundary":
        stream = StreamType.SKELETON
        citation = (
            "L4 frozen SegNet last-frame logits own the exact interface choice; "
            "L3 raster is the owed realization surface, not the information home"
        )
    elif stratum == "cell":
        stream = StreamType.FIBER
        citation = (
            "L4 frozen SegNet last-frame logits own the within-support class/margin choice"
        )
    else:
        raise SolvedValuePricingError(f"unknown stratum {stratum!r}")
    return TypedStreamTag(
        type=stream,
        layer_home=LayerHome.L4_SCORER_FEATURE,
        evaluate_py_recursion_level_cited=citation,
        counted_bytes=counted_bytes,
        free_receiver_code=True,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SolvedValuePricingError(f"refusing to overwrite unequal artifact {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _segnet_logits(segnet: Any, camera_pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    tensor = (
        torch.from_numpy(np.asarray(camera_pair)[None])
        .permute(0, 1, 4, 2, 3)
        .to(dtype=torch.float32)
    )
    with torch.inference_mode():
        realized = segnet.preprocess_input(tensor)
        logits = segnet(realized)
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise SolvedValuePricingError(f"unexpected SegNet logits shape {tuple(logits.shape)}")
    realized_last = realized[0].permute(1, 2, 0).cpu().numpy()
    return logits[0].cpu().numpy().astype(np.float32), realized_last


def _event_support(
    index_npz: Any,
    bucket_id: str,
    array_key: str,
    pair_id: int,
) -> np.ndarray:
    key = array_key
    if key not in index_npz:
        raise SolvedValuePricingError(f"PF2 event index lacks {key}")
    event_ids = np.asarray(index_npz[key], dtype=np.int64)
    pixels_per_pair = 384 * 512
    selected = event_ids[event_ids // pixels_per_pair == pair_id]
    flat = np.unique(selected % pixels_per_pair)
    if len(flat) == 0:
        raise SolvedValuePricingError(f"PF2 support empty for pair={pair_id} bucket={bucket_id}")
    if np.any(flat < 0) or np.any(flat >= pixels_per_pair):
        raise SolvedValuePricingError("PF2 support escapes scorer plane")
    return flat.astype(np.uint32)


def _winner_symbols(
    logits: np.ndarray,
    flat_indices: np.ndarray,
    class_left: int,
    class_right: int,
) -> tuple[bytes, bytes, dict[str, Any]]:
    flat_logits = logits.reshape(logits.shape[0], -1)
    argmax = np.argmax(flat_logits[:, flat_indices], axis=0)
    winners = np.where(
        argmax == class_left,
        0,
        np.where(argmax == class_right, 1, 2),
    ).astype(np.uint8)
    pairwise = (
        flat_logits[class_left, flat_indices] - flat_logits[class_right, flat_indices]
    ).astype(np.float64)
    relations = np.where(pairwise > 0.0, 0, np.where(pairwise < 0.0, 1, 2)).astype(np.uint8)
    summary = {
        "actual_argmax_left": int(np.count_nonzero(winners == 0)),
        "actual_argmax_right": int(np.count_nonzero(winners == 1)),
        "actual_argmax_other": int(np.count_nonzero(winners == 2)),
        "pairwise_margin_positive": int(np.count_nonzero(relations == 0)),
        "pairwise_margin_negative": int(np.count_nonzero(relations == 1)),
        "pairwise_margin_zero": int(np.count_nonzero(relations == 2)),
        "pairwise_margin_min": float(np.min(pairwise)),
        "pairwise_margin_max": float(np.max(pairwise)),
        "pairwise_margin_mean": float(np.mean(pairwise)),
        "pairwise_margin_sha256": sha256(
            np.asarray(pairwise, dtype="<f4").tobytes()
        ).hexdigest(),
    }
    return winners.tobytes(), relations.tobytes(), summary


def _class_lookup() -> tuple[str, ...]:
    from tac.witness_control.factorized_adjoint import CLASS_NAMES

    return tuple(name.lower().replace(" ", "") for name in CLASS_NAMES)


def _class_id(name: str) -> int:
    normalized = name.lower().replace("-", "").replace(" ", "")
    aliases = {"mycar": "mycar"}
    normalized = aliases.get(normalized, normalized)
    values = _class_lookup()
    if normalized not in values:
        raise SolvedValuePricingError(f"class {name!r} absent from canonical order {values!r}")
    return values.index(normalized)


def _validate_false_authority(config: Mapping[str, Any]) -> None:
    required = {
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise SolvedValuePricingError(f"config {key} must equal {expected!r}")


def _next_measurement(stratum: str) -> str:
    if stratum == "boundary":
        return (
            "Build a receiver-closed L3 RGB preimage for this exact L4 interface record, "
            "then remeasure exact parseback and joint Seg/Pose trust-region survival."
        )
    return (
        "Condition an L3 RGB preimage on the SHA-bound existing support for this L4 fiber, "
        "then remeasure exact parseback and joint Seg/Pose trust-region survival."
    )


def _implementation_custody() -> dict[str, dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        _REPO / "tools/measure_ddm_dm1_solved_value_pricing.py",
        _REPO / "src/tac/lossless/range_coder.py",
        _REPO / "src/tac/optimization/ddm_min_description_contract.py",
        _REPO / "src/tac/optimization/resize_full_kernel.py",
        _REPO / "src/tac/optimization/solve_diff_operator_mining.py",
        _REPO / "src/tac/scorer.py",
    )
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise SolvedValuePricingError(f"implementation custody path is absent: {path}")
        relative = str(path.relative_to(_REPO))
        output[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def materialize(config_path: str | Path, output_dir: str | Path) -> Mapping[str, Any]:
    """Materialize and price exactly the registered 25 rows."""
    config_raw = Path(config_path).read_bytes()
    config = json.loads(config_raw)
    if config.get("schema") != CONFIG_SCHEMA:
        raise SolvedValuePricingError("pricing config schema mismatch")
    _validate_false_authority(config)
    if sha256_file(config["demand_ledger_path"]) != config["demand_ledger_sha256"]:
        raise SolvedValuePricingError("demand ledger SHA-256 mismatch")
    demand = checked_json(config["demand_ledger_path"], config["demand_ledger_sha256"])
    if demand.get("row_count") != 25 or len(demand.get("rows", ())) != 25:
        raise SolvedValuePricingError("demand ledger must bind exactly 25 rows")
    if sha256_file(config["rg3_summary_path"]) != config["rg3_summary_sha256"]:
        raise SolvedValuePricingError("RG3 summary SHA-256 mismatch")
    checked_json(config["rg3_summary_path"], config["rg3_summary_sha256"])
    if sha256_file(config["pf2_index_receipt_path"]) != config["pf2_index_receipt_sha256"]:
        raise SolvedValuePricingError("PF2 index receipt SHA-256 mismatch")
    index_receipt = checked_json(
        config["pf2_index_receipt_path"], config["pf2_index_receipt_sha256"]
    )
    if index_receipt.get("index_sha256") != config["pf2_event_index_sha256"]:
        raise SolvedValuePricingError("PF2 receipt-to-index binding mismatch")
    bucket_arrays = index_receipt.get("bucket_arrays")
    if not isinstance(bucket_arrays, Mapping):
        raise SolvedValuePricingError("PF2 receipt bucket-array mapping missing")
    if sha256_file(config["pf2_event_index_path"]) != config["pf2_event_index_sha256"]:
        raise SolvedValuePricingError("PF2 event index SHA-256 mismatch")
    for path_key, sha_key in (
        ("segnet_weights_path", "segnet_weights_sha256"),
        ("upstream_modules_path", "upstream_modules_sha256"),
        ("source_config_path", "source_config_sha256"),
    ):
        if sha256_file(config[path_key]) != config[sha_key]:
            raise SolvedValuePricingError(f"{path_key} SHA-256 mismatch")
    source_config = SolveDiffMiningConfigV1.model_validate_json(
        Path(config["source_config_path"]).read_bytes()
    )
    kernel = FullResizeKernel.build()
    context = _open_production_inputs(source_config)

    import torch

    from tac.scorer import load_default_segnet

    torch.set_num_threads(int(config.get("torch_threads", 4)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    segnet = load_default_segnet(config["upstream_dir"], device="cpu")

    demand_rows = list(demand["rows"])
    row_positions: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
    for position, row in enumerate(demand_rows):
        row_positions.setdefault(int(row["pair_id"]), []).append((position, row))
    output_rows: list[Mapping[str, Any] | None] = [None] * 25
    raw_rows: list[bytes | None] = [None] * 25
    source_chunk_hashes: dict[str, str] = {}
    with np.load(config["pf2_event_index_path"], allow_pickle=False) as event_index:
        for pair_id in sorted(row_positions):
            chunk = _load_production_inputs(context, source_config, [pair_id], kernel)
            source_chunk_hashes.update(chunk.source_hashes)
            solved_pair = chunk.solved_planes[0]
            camera = np.stack(
                [
                    realize_solve_camera(solved_pair[frame], kernel)
                    for frame in range(2)
                ],
                axis=0,
            )
            logits, realized_last = _segnet_logits(segnet, camera)
            rounded = np.clip(np.rint(realized_last), 0, 255).astype(np.uint8)
            if not np.array_equal(rounded, solved_pair[1]):
                raise SolvedValuePricingError(
                    f"pair {pair_id} camera preimage failed exact solved-plane roundtrip"
                )
            roundtrip_max_abs = float(
                np.max(np.abs(realized_last.astype(np.float64) - solved_pair[1]))
            )
            for position, row in row_positions[pair_id]:
                bucket_id = str(row["bucket_id"])
                stratum = _stratum_from_bucket(bucket_id)
                left_name, right_name = _class_pair_from_bucket(bucket_id)
                class_left, class_right = _class_id(left_name), _class_id(right_name)
                array_key = bucket_arrays.get(bucket_id)
                if not isinstance(array_key, str):
                    raise SolvedValuePricingError(
                        f"PF2 receipt lacks bucket-array binding for {bucket_id}"
                    )
                flat = _event_support(event_index, bucket_id, array_key, pair_id)
                support_hash = support_sha256(flat)
                winners, relations, margin_summary = _winner_symbols(
                    logits, flat, class_left, class_right
                )
                record = SolvedValueRecord(
                    pair_id=pair_id,
                    bucket_id=bucket_id,
                    class_left=class_left,
                    class_right=class_right,
                    stream_type=(
                        StreamType.SKELETON if stratum == "boundary" else StreamType.FIBER
                    ),
                    layer_home=LayerHome.L4_SCORER_FEATURE,
                    support_sha256=support_hash,
                    winners=winners,
                    margin_relations=relations,
                    flat_indices=(
                        tuple(int(value) for value in flat)
                        if stratum == "boundary"
                        else ()
                    ),
                )
                raw = record.encode()
                parsed = SolvedValueRecord.decode(
                    raw,
                    external_cell_support=flat if stratum == "cell" else None,
                )
                if parsed != record:
                    raise SolvedValuePricingError("semantic row parseback mismatch")
                prices, winning_codec = price_raw(raw)
                tag = typed_home(stratum, prices[winning_codec]["container_bytes"])
                output_rows[position] = {
                    "schema": ROW_SCHEMA,
                    "row_index": position,
                    "pair_id": pair_id,
                    "bucket_id": bucket_id,
                    "stratum": stratum,
                    "oracle_content": row["oracle_content"],
                    "source_rg3_family": row["source_rg3_family"],
                    "source_candidate": {
                        "stream_type": row["candidate_stream_type"],
                        "layer_home": row["candidate_layer_home"],
                    },
                    "adjudicated_typed_home": tag.to_dict(),
                    "candidate_disposition": (
                        "CORRECTED: SKELETON retained; L3 is realization, deepest "
                        "score-visible information home is L4."
                        if stratum == "boundary"
                        else "CONFIRMED: within-support choice is FIBER/L4."
                    ),
                    "wrong_candidates": (
                        [
                            "SKELETON/L3 as information home: too shallow; L3 is owed realization",
                            "CONNECTION: no same-bucket xi-adjacent row exists in the 25-row ledger",
                            "GAUGE: argmax/pairwise choice changes the L5 Seg verdict",
                        ]
                        if stratum == "boundary"
                        else [
                            "SKELETON/L3: support placement is external existing context",
                            "CONNECTION: no same-bucket xi-adjacent row exists in the 25-row ledger",
                            "GAUGE: the within-cell choice changes the L5 Seg verdict",
                        ]
                    ),
                    "class_left": left_name,
                    "class_left_id": class_left,
                    "class_right": right_name,
                    "class_right_id": class_right,
                    "support": {
                        "count": len(flat),
                        "sha256_uint32le": support_hash,
                        "coordinate_payload": (
                            "counted_exact_delta_varint" if stratum == "boundary"
                            else "external_sha_bound_pf2_context_not_recounted"
                        ),
                    },
                    "solved_value": margin_summary,
                    "semantic_record": {
                        "raw_bytes": len(raw),
                        "raw_sha256": sha256_bytes(raw),
                        "parseback_exact": True,
                    },
                    "independent_prices": prices,
                    "winning_codec": winning_codec,
                    "exact_counted_bytes": prices[winning_codec]["container_bytes"],
                    "roundtrip": {
                        "solved_plane_uint8_exact": True,
                        "pre_round_max_abs": roundtrip_max_abs,
                        "semantic_record_exact": True,
                    },
                    "next_measurement": _next_measurement(stratum),
                    "verdict_scope": (
                        "Exact semantic-record price on the registered PF2 support under the "
                        "frozen SegNet last-frame axis only. Not an L3 receiver/archive price; "
                        "Pose, full-video, contest-CPU/CUDA, and score movement are unmeasured."
                    ),
                }
                raw_rows[position] = raw
            del camera, logits, realized_last, solved_pair, chunk

    if any(row is None for row in output_rows) or any(raw is None for raw in raw_rows):
        raise SolvedValuePricingError("not all 25 demand rows were materialized")
    exact_rows = [row for row in output_rows if row is not None]
    exact_raws = [raw for raw in raw_rows if raw is not None]
    joint_raw = encode_joint_raw(exact_raws)
    if decode_joint_raw(joint_raw) != tuple(exact_raws):
        raise SolvedValuePricingError("joint raw exact parseback failed")
    joint_prices, joint_winner = price_raw(joint_raw)

    like_for_like_adjacent: list[dict[str, Any]] = []
    by_bucket: dict[str, list[Mapping[str, Any]]] = {}
    for row in exact_rows:
        by_bucket.setdefault(str(row["bucket_id"]), []).append(row)
    for bucket_id, rows in by_bucket.items():
        ordered = sorted(rows, key=lambda row: int(row["pair_id"]))
        for left, right in pairwise(ordered):
            gap = int(right["pair_id"]) - int(left["pair_id"])
            if gap == 1:
                like_for_like_adjacent.append(
                    {
                        "bucket_id": bucket_id,
                        "left_pair_id": left["pair_id"],
                        "right_pair_id": right["pair_id"],
                        "pair_gap": gap,
                    }
                )
    if like_for_like_adjacent:
        raise SolvedValuePricingError(
            "xi-adjacent candidates require a separately preregistered relation payload"
        )

    result = {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "source_commit": config["source_commit"],
        "config_path": str(config_path),
        "config_sha256": sha256_bytes(config_raw),
        "custody": {
            "demand_ledger_path": config["demand_ledger_path"],
            "demand_ledger_sha256": config["demand_ledger_sha256"],
            "rg3_summary_path": config["rg3_summary_path"],
            "rg3_summary_sha256": config["rg3_summary_sha256"],
            "pf2_event_index_path": config["pf2_event_index_path"],
            "pf2_event_index_sha256": config["pf2_event_index_sha256"],
            "pf2_index_receipt_path": config["pf2_index_receipt_path"],
            "pf2_index_receipt_sha256": config["pf2_index_receipt_sha256"],
            "source_config_path": config["source_config_path"],
            "source_config_sha256": config["source_config_sha256"],
            "solved_planes_receipt_path": source_config.solved_planes_receipt_path,
            "solved_planes_receipt_sha256": source_config.solved_planes_receipt_sha256,
            "segnet_weights_path": config["segnet_weights_path"],
            "segnet_weights_sha256": config["segnet_weights_sha256"],
            "upstream_modules_path": config["upstream_modules_path"],
            "upstream_modules_sha256": config["upstream_modules_sha256"],
            "streamed_solved_chunk_hashes": dict(sorted(source_chunk_hashes.items())),
            "implementation": _implementation_custody(),
        },
        "row_count": 25,
        "boundary_rows": sum(row["stratum"] == "boundary" for row in exact_rows),
        "cell_rows": sum(row["stratum"] == "cell" for row in exact_rows),
        "rows": exact_rows,
        "independent": {
            "sum_winning_row_container_bytes": sum(
                int(row["exact_counted_bytes"]) for row in exact_rows
            ),
            "sum_by_fixed_codec": {
                codec: sum(
                    int(row["independent_prices"][codec]["container_bytes"])
                    for row in exact_rows
                )
                for codec in CODECS
            },
        },
        "joint_shared_context": {
            "raw_bytes": len(joint_raw),
            "raw_sha256": sha256_bytes(joint_raw),
            "prices": joint_prices,
            "winning_codec": joint_winner,
            "exact_counted_bytes": joint_prices[joint_winner]["container_bytes"],
            "all_25_rows_parseback_exact": True,
        },
        "xi_adjacent_predictability": {
            "definition": "same bucket_id and consecutive pair ids (delta xi = 1)",
            "eligible_comparators": like_for_like_adjacent,
            "eligible_count": len(like_for_like_adjacent),
            "connection_price_bytes": None,
            "verdict": (
                "NULL: the registered 25 rows contain no like-for-like xi-adjacent pair; "
                "CONNECTION re-homing is not measured."
            ),
            "next_measurement": (
                "Register and solve at least one same-bucket consecutive-pair support before "
                "pricing a CONNECTION relation; do not infer it from nonmatching buckets."
            ),
        },
        "context_only": {
            "slack_reference": "#613",
            "tangent_reference_bytes": 154_522,
            "new_box_arithmetic_performed": False,
            "new_nonadditive_pool_claimed": False,
            "interpretation": (
                "These semantic-record bytes are not receiver-closed archive bytes and are "
                "therefore context, not a deduction from #613 slack or the 154,522B tangent."
            ),
        },
        "rehome_verdict": {
            "boundary": "SKELETON/L4 information home; L3 remains the owed realization surface.",
            "cell": "FIBER/L4 confirmed when the SHA-bound support context is available.",
            "asymmetry": (
                "SKELETON pays placement plus choice; FIBER pays only the choice because "
                "existing support placement is referenced by SHA."
            ),
        },
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
        "main_review_required": True,
        "verdict_scope": (
            "25-row, exact-PF2-support, local frozen-SegNet semantic pricing only. No "
            "receiver-closed RGB generator, Pose verdict, exact evaluator replay, archive, "
            "contest-axis score, promotion, or frontier mutation."
        ),
    }
    payload = canonical_json_bytes(result)
    receipt_path = Path(output_dir) / "ddm_dm1_25_row_solved_value_pricing_receipt.json"
    _atomic_write(receipt_path, payload)
    manifest = {
        "schema": "ddm_dm1_solved_value_pricing_manifest.v1",
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_bytes(payload),
        "receipt_bytes": len(payload),
        "large_artifacts_created": False,
        "auto_cleanup": "not_applicable_bounded_records_only",
        "score_claim": False,
        "pointer_moved": False,
        "main_review_required": True,
    }
    _atomic_write(
        Path(output_dir) / "manifest.json",
        canonical_json_bytes(manifest),
    )
    return result


__all__ = [
    "CODECS",
    "CONFIG_SCHEMA",
    "SolvedValuePricingError",
    "SolvedValueRecord",
    "decode_codec",
    "decode_context_arithmetic",
    "decode_joint_raw",
    "encode_codec",
    "encode_context_arithmetic",
    "encode_joint_raw",
    "materialize",
    "price_raw",
    "support_sha256",
    "typed_home",
]
