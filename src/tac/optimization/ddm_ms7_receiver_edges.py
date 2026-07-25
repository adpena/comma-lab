# SPDX-License-Identifier: MIT
"""Custody-bound R0 reach pricing and PF3 coder race for DDM MS7.

This module keeps three distinct claims separate:

* R0 allocates each terminal PF2 block a *derived* share of its pair's G3
  distortion mass, weighted by the block's exact share of pair flips.
* DM4 supplies the only already-measured guaranteed-reach price for every
  terminal block.  Unbuilt T-residual and unmeasured dynamic-RG3 prices remain
  ``None``.
* A coder race prices one exact receiver object.  Every admitted coder row
  must decode byte-for-byte to that same object.

The scorer measurement itself lives in ``tools/run_ddm_ms7_receiver_edges.py``
so this reusable core stays importable without loading Torch or the 4.7-GiB
target cache.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import shutil
import struct
import subprocess
import tempfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724 import (
    RATE_DENOMINATOR_BYTES,
)
from tac.optimization.arith_selfcomp_rate_coders import (
    RateCoderError,
    byte_context_frame_accounting,
    decode_bellard_class_mixing,
    decode_brotli_q11,
    decode_g4_decoder_context,
    decode_spatial_context_arithmetic,
    decode_spatial_context_constriction,
    decode_willems_ctw,
    encode_bellard_class_mixing,
    encode_brotli_q11,
    encode_g4_decoder_context,
    encode_spatial_context_arithmetic,
    encode_spatial_context_constriction,
    encode_willems_ctw,
    spatial_context_frame_accounting,
)

R0_SCHEMA: Final = "ddm_ms7_r0_25_bucket_reach_table.v1"
CODER_RACE_SCHEMA: Final = "ddm_ms7_same_object_coder_race.v1"
COUNTED_STREAM_RACE_SCHEMA: Final = "ddm_cc2_counted_stream_context_coder_race.v1"
POINTER: Final = "0.1910828242 [contest-CPU]"
EXPECTED_ROWS: Final = 25
EXPECTED_ATLAS_ROWS: Final = 600
_CODED_MAGIC: Final = b"D7CR1"
_CODED_HEADER: Final = struct.Struct(">5sBQ32s")
_CODEC_IDS: Final = {
    "ZLIB9": 1,
    "RAW_LZMA1": 2,
    "ORDER1_CONTEXT_ARITHMETIC": 3,
    "E4_BROTLI_Q11": 4,
    "CONSTRICTION_ORDER1_CONTEXT_ANS": 5,
    "ZSTD19_TRAINED_DICTIONARY": 6,
}
_ID_CODECS: Final = {value: key for key, value in _CODEC_IDS.items()}


class MS7ReceiverEdgesError(ValueError):
    """A source, identity join, or exact coder parse-back differs."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MS7ReceiverEdgesError(f"bound JSON is absent or not regular: {path}")
    if sha256_file(path) != expected_sha256:
        raise MS7ReceiverEdgesError(f"bound JSON SHA-256 differs: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MS7ReceiverEdgesError(f"bound JSON is not an object: {path}")
    return value


def read_bound_atlas(path: Path, expected_sha256: str) -> dict[int, dict[str, Any]]:
    """Load the SHA-bound G3 JSONL atlas with exact 0..599 identity."""

    if not path.is_file() or path.is_symlink():
        raise MS7ReceiverEdgesError(f"G3 atlas is absent or not regular: {path}")
    if sha256_file(path) != expected_sha256:
        raise MS7ReceiverEdgesError("G3 atlas SHA-256 differs")
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            value = json.loads(raw)
            if (
                not isinstance(value, dict)
                or value.get("schema") != "ddm_g3_score_atlas_pair.v1"
                or isinstance(value.get("pair_index"), bool)
                or not isinstance(value.get("pair_index"), int)
            ):
                raise MS7ReceiverEdgesError(f"G3 atlas row {line_number} is malformed")
            pair_index = int(value["pair_index"])
            if pair_index in rows:
                raise MS7ReceiverEdgesError("G3 atlas pair identity is duplicated")
            rows[pair_index] = value
    if set(rows) != set(range(EXPECTED_ATLAS_ROWS)):
        raise MS7ReceiverEdgesError("G3 atlas must contain exact pair identity 0..599")
    return rows


def _row_identity(row: Mapping[str, Any]) -> tuple[int, str]:
    pair = row.get("pair_id")
    bucket = row.get("bucket_id")
    if (
        isinstance(pair, bool)
        or not isinstance(pair, int)
        or not 0 <= pair < EXPECTED_ATLAS_ROWS
        or not isinstance(bucket, str)
        or not bucket
    ):
        raise MS7ReceiverEdgesError("terminal-row pair/bucket identity is malformed")
    return pair, bucket


def build_r0_reach_table(
    *,
    direct_metric: Mapping[str, Any],
    dm4_receipt: Mapping[str, Any],
    atlas: Mapping[int, Mapping[str, Any]],
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Join the exact 25 terminal rows to G3 mass and measured DM4 prices.

    ``flip_weighted_S_leverage`` is deliberately a derived allocation, not a
    measured candidate delta:

    ``pair_distortion_score_mass * support_count / pair_flip_count``.

    The R0 ignore decision is scoped to the currently measured guaranteed-reach
    menu.  R1 and R2 prices stay NULL, so the table does not pretend that an
    unmeasured cheaper representation is impossible.
    """

    blocks = direct_metric.get("direct_blocks")
    dm4_rows = dm4_receipt.get("rows")
    if (
        direct_metric.get("schema") != "ddm_seg_metric_custody.direct_scorer_intrinsic.v2"
        or not isinstance(blocks, list)
        or len(blocks) != EXPECTED_ROWS
    ):
        raise MS7ReceiverEdgesError("MS4D direct metric does not contain exact 25 blocks")
    if (
        dm4_receipt.get("schema") != "ddm_dm4_targeted_realization_cures.v1"
        or not isinstance(dm4_rows, list)
        or len(dm4_rows) != EXPECTED_ROWS
        or dm4_receipt.get("row_count") != EXPECTED_ROWS
    ):
        raise MS7ReceiverEdgesError("DM4 receipt does not contain exact 25 price rows")
    dm4_by_identity = {_row_identity(row): row for row in dm4_rows if isinstance(row, Mapping)}
    identities = [_row_identity(row) for row in blocks if isinstance(row, Mapping)]
    if len(identities) != EXPECTED_ROWS or len(set(identities)) != EXPECTED_ROWS:
        raise MS7ReceiverEdgesError("MS4D terminal identities are absent or duplicated")
    if set(dm4_by_identity) != set(identities):
        raise MS7ReceiverEdgesError("MS4D and DM4 terminal identities differ")

    rows: list[dict[str, Any]] = []
    for row_index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise MS7ReceiverEdgesError("MS4D terminal block is malformed")
        pair_id, bucket_id = _row_identity(block)
        support_count = block.get("support_count")
        if isinstance(support_count, bool) or not isinstance(support_count, int) or support_count <= 0:
            raise MS7ReceiverEdgesError("MS4D support count must be a positive integer")
        atlas_row = atlas.get(pair_id)
        if not isinstance(atlas_row, Mapping):
            raise MS7ReceiverEdgesError("terminal pair is absent from the G3 atlas")
        segmentation = atlas_row.get("segmentation")
        score_mass = atlas_row.get("score_mass")
        if not isinstance(segmentation, Mapping) or not isinstance(score_mass, Mapping):
            raise MS7ReceiverEdgesError("G3 pair lacks segmentation or score mass")
        pair_flips = segmentation.get("flip_count")
        pair_mass = score_mass.get("distortion_score_mass")
        if (
            isinstance(pair_flips, bool)
            or not isinstance(pair_flips, int)
            or pair_flips <= 0
            or not isinstance(pair_mass, (int, float))
            or not math.isfinite(float(pair_mass))
            or float(pair_mass) < 0.0
            or support_count > pair_flips
        ):
            raise MS7ReceiverEdgesError("G3 pair mass/flip custody is invalid")
        dm4 = dm4_by_identity[(pair_id, bucket_id)]
        rgb_record = dm4.get("rgb_record")
        price = rgb_record.get("exact_counted_bytes") if isinstance(rgb_record, Mapping) else None
        if (
            isinstance(price, bool)
            or not isinstance(price, int)
            or price <= 0
            or rgb_record.get("parseback_exact") is not True
        ):
            raise MS7ReceiverEdgesError("DM4 row lacks an exact positive counted-byte price")

        event_mass = support_count / pair_flips
        leverage = event_mass * float(pair_mass)
        rate_bound = 25.0 * price / RATE_DENOMINATOR_BYTES
        pays = leverage >= rate_bound
        rows.append(
            {
                "row_index": row_index,
                "bucket_id": bucket_id,
                "pair_id": pair_id,
                "support_count": support_count,
                "pair_flip_count": pair_flips,
                "event_mass": event_mass,
                "event_mass_definition": "support_count / g3_pair_flip_count",
                "pair_distortion_score_mass": float(pair_mass),
                "flip_weighted_S_leverage": leverage,
                "flip_weighted_S_leverage_authority": ("DERIVED_G3_PAIR_DISTORTION_MASS_TIMES_EXACT_EVENT_MASS"),
                "reach_prices": {
                    "R1_DYNAMIC_EXISTING_COORDINATE_BYTES": None,
                    "R2_T_RESIDUAL_BYTES": None,
                    "R3_DM4_CORRECTED_J_SHEARLET_BYTES": price,
                },
                "cheapest_reach_family": ("R3_DM4_DIRECT_SPARSE_PIXEL_CORRECTED_J_SHEARLET"),
                "cheapest_reach_price_bound_bytes": price,
                "cheapest_reach_price_bound_authority": (
                    "MEASURED_EXACT_DM4_CONSTRUCTIVE_GUARANTEED_REACH_UPPER_BOUND"
                ),
                "cheapest_reach_rate_score_bound": rate_bound,
                "mass_pays_cheapest_measured_guaranteed_reach": pays,
                "mass_minus_rate_score_bound": leverage - rate_bound,
                "verdict": ("MASS_PAYS_MEASURED_GUARANTEED_REACH" if pays else "UNREACHABLE-AND-IGNORED"),
                "verdict_scope": (
                    "INSTANCE exact terminal row x CURRENT_MEASURED_GUARANTEED_REACH_MENU; "
                    "R1 dynamic and R2 T-residual prices remain NULL"
                ),
            }
        )
    paying = [row for row in rows if row["mass_pays_cheapest_measured_guaranteed_reach"]]
    return {
        "schema": R0_SCHEMA,
        "pointer": POINTER,
        "pointer_moved": False,
        "row_count": len(rows),
        "mass_paying_row_count": len(paying),
        "unreachable_and_ignored_row_count": len(rows) - len(paying),
        "r1_execution_subset": [{"pair_id": row["pair_id"], "bucket_id": row["bucket_id"]} for row in paying],
        "sources": dict(sources),
        "rows": rows,
        "r1_execution_disposition": (
            "AUTHORIZED_ONLY_FOR_R0_MASS_PAYING_SUBSET" if paying else "NOT_RUN_EMPTY_R0_MASS_PAYING_SUBSET"
        ),
        "r2_execution_disposition": "NOT_RUN_R1_DID_NOT_LEAVE_A_MASS_PAYING_FAILURE",
        "r3_execution_disposition": "NOT_RUN_NO_ROW_PAYS_MEASURED_R3_BOUND",
        "score_claim": False,
        "research_only": True,
        "main_review_required": True,
    }


def _linear_signed_array(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.uint8).view(np.int8).reshape(1, 1, len(raw), 1)


def _frame_coded(codec: str, encoded: bytes, raw: bytes) -> bytes:
    return (
        _CODED_HEADER.pack(
            _CODED_MAGIC,
            _CODEC_IDS[codec],
            len(raw),
            hashlib.sha256(raw).digest(),
        )
        + encoded
    )


def _zstandard_module() -> Any | None:
    try:
        import zstandard as zstd  # type: ignore[import-not-found]
    except ImportError:
        return None
    return zstd


def _dictionary_samples(raw: bytes) -> tuple[bytes, ...]:
    """Derive deterministic counted training samples from the owned object.

    The dictionary is video-derived, so its bytes are embedded in the coded
    frame and counted.  Splitting the byte object is only a fallback for this
    first one-object race; callers with parsed member streams may replace it
    in a later schema rather than pretending a flat byte context is G4.
    """

    width = max(64, min(4096, math.ceil(len(raw) / 32)))
    return tuple(raw[start : start + width] for start in range(0, len(raw), width))


def _encode_zstd_dictionary(raw: bytes) -> bytes:
    zstd = _zstandard_module()
    samples = _dictionary_samples(raw)
    if len(samples) < 8:
        raise MS7ReceiverEdgesError("zstd dictionary training requires at least eight samples")
    dictionary_size = min(8192, max(1024, len(raw) // 16))
    if zstd is not None:
        try:
            trained = zstd.train_dictionary(dictionary_size, list(samples))
            dictionary = trained.as_bytes()
            compressed = zstd.ZstdCompressor(
                level=19,
                dict_data=zstd.ZstdCompressionDict(dictionary),
            ).compress(raw)
        except zstd.ZstdError as exc:
            raise MS7ReceiverEdgesError("zstd dictionary training/encode failed") from exc
    else:
        executable = shutil.which("zstd")
        if executable is None:
            raise MS7ReceiverEdgesError("zstandard module and zstd CLI are unavailable")
        with tempfile.TemporaryDirectory(prefix="ddm_ms7_zstd_train_") as temporary:
            root = Path(temporary)
            sample_paths = []
            for index, sample in enumerate(samples):
                path = root / f"sample_{index:04d}.bin"
                path.write_bytes(sample)
                sample_paths.append(path)
            dictionary_path = root / "dictionary.zstd"
            trained = subprocess.run(
                [
                    executable,
                    "--train",
                    *(str(path) for path in sample_paths),
                    f"--maxdict={dictionary_size}",
                    "-o",
                    str(dictionary_path),
                ],
                capture_output=True,
                check=False,
            )
            if trained.returncode != 0 or not dictionary_path.is_file():
                raise MS7ReceiverEdgesError(f"zstd CLI dictionary training failed with exit {trained.returncode}")
            dictionary = dictionary_path.read_bytes()
            encoded = subprocess.run(
                [
                    executable,
                    "-19",
                    "-D",
                    str(dictionary_path),
                    "--stdout",
                    "--no-progress",
                ],
                input=raw,
                capture_output=True,
                check=False,
            )
            if encoded.returncode != 0:
                raise MS7ReceiverEdgesError(f"zstd CLI dictionary encode failed with exit {encoded.returncode}")
            compressed = encoded.stdout
    return struct.pack(">I", len(dictionary)) + dictionary + compressed


def _decode_zstd_dictionary(payload: bytes) -> bytes:
    zstd = _zstandard_module()
    if len(payload) < 4:
        raise MS7ReceiverEdgesError("zstd dictionary frame is truncated")
    (dictionary_size,) = struct.unpack_from(">I", payload)
    if dictionary_size <= 0 or len(payload) <= 4 + dictionary_size:
        raise MS7ReceiverEdgesError("zstd dictionary frame lengths are invalid")
    dictionary = payload[4 : 4 + dictionary_size]
    compressed = payload[4 + dictionary_size :]
    if zstd is not None:
        try:
            return bytes(
                zstd.ZstdDecompressor(dict_data=zstd.ZstdCompressionDict(dictionary)).decompress(
                    compressed, allow_extra_data=False
                )
            )
        except zstd.ZstdError as exc:
            raise MS7ReceiverEdgesError("zstd dictionary decode failed") from exc
    executable = shutil.which("zstd")
    if executable is None:
        raise MS7ReceiverEdgesError("zstandard module and zstd CLI are unavailable")
    with tempfile.TemporaryDirectory(prefix="ddm_ms7_zstd_decode_") as temporary:
        dictionary_path = Path(temporary) / "dictionary.zstd"
        dictionary_path.write_bytes(dictionary)
        decoded = subprocess.run(
            [
                executable,
                "-d",
                "-D",
                str(dictionary_path),
                "--stdout",
                "--no-progress",
            ],
            input=compressed,
            capture_output=True,
            check=False,
        )
        if decoded.returncode != 0:
            raise MS7ReceiverEdgesError(f"zstd CLI dictionary decode failed with exit {decoded.returncode}")
        return decoded.stdout


def decode_coded_receiver_object(frame: bytes) -> bytes:
    """Decode a strict MS7 compressed receiver-object frame."""

    if len(frame) < _CODED_HEADER.size:
        raise MS7ReceiverEdgesError("coded receiver object is truncated")
    magic, codec_id, raw_length, digest = _CODED_HEADER.unpack_from(frame)
    codec = _ID_CODECS.get(codec_id)
    if magic != _CODED_MAGIC or codec is None:
        raise MS7ReceiverEdgesError("coded receiver object header is invalid")
    payload = frame[_CODED_HEADER.size :]
    try:
        if codec == "ZLIB9":
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload) + decoder.flush()
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise MS7ReceiverEdgesError("ZLIB9 frame is truncated or has a trailer")
        elif codec == "RAW_LZMA1":
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
            raw = decoder.decompress(payload)
            if not decoder.eof or decoder.unused_data:
                raise MS7ReceiverEdgesError("LZMA frame is truncated or has a trailer")
        elif codec == "ORDER1_CONTEXT_ARITHMETIC":
            array = decode_spatial_context_arithmetic(payload)
            raw = np.ascontiguousarray(array).view(np.uint8).tobytes()
        elif codec == "CONSTRICTION_ORDER1_CONTEXT_ANS":
            array = decode_spatial_context_constriction(payload)
            raw = np.ascontiguousarray(array).view(np.uint8).tobytes()
        elif codec == "ZSTD19_TRAINED_DICTIONARY":
            raw = _decode_zstd_dictionary(payload)
        elif codec == "E4_BROTLI_Q11":
            raw = decode_brotli_q11(payload)
        else:  # pragma: no cover - protected by the exact enum above
            raise MS7ReceiverEdgesError("coded receiver object codec is unknown")
    except (zlib.error, lzma.LZMAError, RateCoderError) as exc:
        raise MS7ReceiverEdgesError(f"{codec} receiver-object decode failed") from exc
    if len(raw) != raw_length or hashlib.sha256(raw).digest() != digest:
        raise MS7ReceiverEdgesError("coded receiver-object parse-back differs")
    return raw


def _race_row(
    *,
    codec: str,
    raw: bytes,
    encoded: bytes,
    implementation: str,
) -> tuple[dict[str, Any], bytes]:
    frame = _frame_coded(codec, encoded, raw)
    if decode_coded_receiver_object(frame) != raw:
        raise MS7ReceiverEdgesError(f"{codec} parse-back differs")
    return (
        {
            "codec": codec,
            "available": True,
            "framed_bytes": len(frame),
            "frame_sha256": sha256_bytes(frame),
            "parseback_exact": True,
            "implementation": implementation,
        },
        frame,
    )


def race_same_receiver_object(raw: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Run the mandatory real coder menu on one byte-identical object.

    G4 is explicitly unavailable for a flat archive object because no
    receiver-derived spatial payload home exists.  It remains NULL rather than
    being relabeled as the order-1 byte stream.
    """

    raw = bytes(raw)
    if not raw:
        raise MS7ReceiverEdgesError("coder race requires a nonempty receiver object")
    rows: list[dict[str, Any]] = [
        {
            "codec": "RAW_COMPACT",
            "available": True,
            "framed_bytes": len(raw),
            "frame_sha256": sha256_bytes(raw),
            "parseback_exact": True,
            "implementation": "identity receiver-object bytes",
        }
    ]
    frames: dict[str, bytes] = {"RAW_COMPACT": raw}
    encoded_rows = (
        ("ZLIB9", zlib.compress(raw, level=9), "stdlib:zlib level=9"),
        (
            "RAW_LZMA1",
            lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME),
            "stdlib:lzma XZ LZMA2 preset=9|extreme",
        ),
        (
            "ORDER1_CONTEXT_ARITHMETIC",
            encode_spatial_context_arithmetic(_linear_signed_array(raw)),
            "repository range coder; 1xN left/up signed-byte context",
        ),
    )
    for codec, encoded, implementation in encoded_rows:
        row, frame = _race_row(
            codec=codec,
            raw=raw,
            encoded=encoded,
            implementation=implementation,
        )
        rows.append(row)
        frames[codec] = frame
    try:
        row, frame = _race_row(
            codec="E4_BROTLI_Q11",
            raw=raw,
            encoded=encode_brotli_q11(raw),
            implementation="optional:brotli quality=11",
        )
        rows.append(row)
        frames["E4_BROTLI_Q11"] = frame
    except RateCoderError as exc:
        rows.append(
            {
                "codec": "E4_BROTLI_Q11",
                "available": False,
                "framed_bytes": None,
                "frame_sha256": None,
                "parseback_exact": False,
                "implementation": "optional:brotli quality=11",
                "unavailable_reason": str(exc),
            }
        )
    try:
        constriction_encoded = encode_spatial_context_constriction(_linear_signed_array(raw))
        # A second materialization is the cheap deterministic encoder gate.
        if encode_spatial_context_constriction(_linear_signed_array(raw)) != constriction_encoded:
            raise MS7ReceiverEdgesError("constriction encoder is nondeterministic")
        row, frame = _race_row(
            codec="CONSTRICTION_ORDER1_CONTEXT_ANS",
            raw=raw,
            encoded=constriction_encoded,
            implementation=("optional:constriction Rust range/ANS; counted adaptive left/up signed-byte context model"),
        )
        rows.append(row)
        frames["CONSTRICTION_ORDER1_CONTEXT_ANS"] = frame
    except (RateCoderError, MS7ReceiverEdgesError) as exc:
        rows.append(
            {
                "codec": "CONSTRICTION_ORDER1_CONTEXT_ANS",
                "available": False,
                "framed_bytes": None,
                "frame_sha256": None,
                "parseback_exact": False,
                "implementation": "optional:constriction Rust range/ANS",
                "unavailable_reason": str(exc),
            }
        )
    try:
        zstd_encoded = _encode_zstd_dictionary(raw)
        if _encode_zstd_dictionary(raw) != zstd_encoded:
            raise MS7ReceiverEdgesError("zstd dictionary encoder is nondeterministic")
        row, frame = _race_row(
            codec="ZSTD19_TRAINED_DICTIONARY",
            raw=raw,
            encoded=zstd_encoded,
            implementation=("optional:zstandard level=19; trained dictionary bytes counted in frame"),
        )
        rows.append(row)
        frames["ZSTD19_TRAINED_DICTIONARY"] = frame
    except MS7ReceiverEdgesError as exc:
        rows.append(
            {
                "codec": "ZSTD19_TRAINED_DICTIONARY",
                "available": False,
                "framed_bytes": None,
                "frame_sha256": None,
                "parseback_exact": False,
                "implementation": "optional:zstandard level=19 trained dictionary",
                "unavailable_reason": str(exc),
            }
        )
    rows.append(
        {
            "codec": "G4_FREE_DECODER_DERIVED_SPATIAL_CONTEXT",
            "available": False,
            "framed_bytes": None,
            "frame_sha256": None,
            "parseback_exact": False,
            "implementation": None,
            "unavailable_reason": ("NULL_NO_RECEIVER_DERIVED_SPATIAL_PAYLOAD_HOME_FOR_FLAT_RG3_ARCHIVE_OBJECT"),
        }
    )
    eligible = [
        row for row in rows if row["available"] and row["parseback_exact"] and isinstance(row["framed_bytes"], int)
    ]
    winner = min(eligible, key=lambda row: (int(row["framed_bytes"]), str(row["codec"])))
    return (
        {
            "schema": CODER_RACE_SCHEMA,
            "same_object_raw_bytes": len(raw),
            "same_object_raw_sha256": sha256_bytes(raw),
            "coder_payload_owner": "DDM_MS7_SAME_OBJECT_RECEIVER_ARCHIVE_PAYLOAD",
            "rows": rows,
            "winner": {
                "codec": winner["codec"],
                "framed_bytes": winner["framed_bytes"],
                "frame_sha256": winner["frame_sha256"],
                "parseback_exact": True,
            },
            "score_claim": False,
        },
        frames,
    )


def race_counted_stream_contexts(raw: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Race the exact CC2 five-arm menu on one counted stream.

    Every non-raw arm is a complete self-delimiting frame.  The tiny ARM/IFCE
    row counts its video-derived static model table.  G4, CTW, and the
    Bellard-class Bayesian mixer carry zero model bytes because their complete
    state is deterministically reconstructed from the already-decoded prefix.
    """

    raw = bytes(raw)
    rows: list[dict[str, Any]] = [
        {
            "codec": "RAW_CURRENT",
            "available": True,
            "framed_bytes": len(raw),
            "header_bytes": 0,
            "model_parameter_bytes": 0,
            "coded_payload_bytes": len(raw),
            "frame_sha256": sha256_bytes(raw),
            "parseback_exact": True,
            "implementation": "identity counted stream bytes",
            "decoder_model_authority": "NONE",
        }
    ]
    frames = {"RAW_CURRENT": raw}

    arm = encode_spatial_context_arithmetic(_linear_signed_array(raw))
    if decode_spatial_context_arithmetic(arm).reshape(-1).view(np.uint8).tobytes() != raw:
        raise MS7ReceiverEdgesError("counted tiny ARM/IFCE parse-back differs")
    arm_accounting = spatial_context_frame_accounting(arm)
    rows.append(
        {
            "codec": "COUNTED_TINY_ARM_IFCE",
            "available": True,
            **arm_accounting,
            "frame_sha256": sha256_bytes(arm),
            "parseback_exact": True,
            "implementation": ("repository range coder; counted static left/up signed-byte sign-magnitude model"),
            "decoder_model_authority": "VIDEO_DERIVED_MODEL_TABLE_FULLY_COUNTED",
        }
    )
    frames["COUNTED_TINY_ARM_IFCE"] = arm

    decoder_derived = (
        (
            "G4_FREE_DECODER_CONTEXT",
            encode_g4_decoder_context,
            decode_g4_decoder_context,
            "causal previous-byte/bit-prefix adaptive arithmetic model",
        ),
        (
            "WILLEMS_CTW",
            encode_willems_ctw,
            decode_willems_ctw,
            "binary depth-8 Willems context-tree weighting with KT estimators",
        ),
        (
            "BELLARD_CLASS_MIXING",
            encode_bellard_class_mixing,
            decode_bellard_class_mixing,
            "online Bayesian mixture of four decoder-derived KT experts",
        ),
    )
    for codec, encoder, decoder, implementation in decoder_derived:
        frame = encoder(raw)
        if decoder(frame) != raw:
            raise MS7ReceiverEdgesError(f"{codec} parse-back differs")
        # A second complete materialization guards deterministic decode-time
        # model evolution and prevents accidental hidden mutable state.
        if encoder(raw) != frame:
            raise MS7ReceiverEdgesError(f"{codec} encoder is nondeterministic")
        rows.append(
            {
                "codec": codec,
                "available": True,
                **byte_context_frame_accounting(frame),
                "frame_sha256": sha256_bytes(frame),
                "parseback_exact": True,
                "implementation": implementation,
                "decoder_model_authority": (
                    "GENERIC_STATE_DERIVED_ONLY_FROM_ALREADY_DECODED_PREFIX_ZERO_COUNTED_PARAMETERS"
                ),
            }
        )
        frames[codec] = frame

    if [row["codec"] for row in rows] != [
        "RAW_CURRENT",
        "COUNTED_TINY_ARM_IFCE",
        "G4_FREE_DECODER_CONTEXT",
        "WILLEMS_CTW",
        "BELLARD_CLASS_MIXING",
    ]:
        raise MS7ReceiverEdgesError("CC2 coder menu differs from the pre-registered five arms")
    winner = min(rows, key=lambda row: (int(row["framed_bytes"]), str(row["codec"])))
    return (
        {
            "schema": COUNTED_STREAM_RACE_SCHEMA,
            "same_object_raw_bytes": len(raw),
            "same_object_raw_sha256": sha256_bytes(raw),
            "rows": rows,
            "winner": {
                "codec": winner["codec"],
                "framed_bytes": winner["framed_bytes"],
                "frame_sha256": winner["frame_sha256"],
                "parseback_exact": True,
            },
            "negative_verdict_scope": (
                "ONE_ALREADY_COUNTED_STREAM_INSTANCE_ONLY; A LOSING ARM DOES NOT KILL "
                "THE CODER FAMILY OR A DIFFERENT STREAM GEOMETRY"
            ),
            "score_claim": False,
        },
        frames,
    )


__all__ = [
    "CODER_RACE_SCHEMA",
    "COUNTED_STREAM_RACE_SCHEMA",
    "POINTER",
    "R0_SCHEMA",
    "MS7ReceiverEdgesError",
    "build_r0_reach_table",
    "decode_coded_receiver_object",
    "race_counted_stream_contexts",
    "race_same_receiver_object",
    "read_bound_atlas",
    "read_bound_json",
    "sha256_bytes",
    "sha256_file",
]
