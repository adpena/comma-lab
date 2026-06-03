# SPDX-License-Identifier: MIT
"""Receiver-visible SNeRV archive packet bundling.

This module is intentionally small: it gives SNeRV a deterministic archive
section grammar for the receiver-facing byte streams the advisory already
charges. It does not load scorers and it does not claim full inflate readiness.

Sections are bundled under one header so downstream work can stop treating LF
codes, decoder bytes, and compact L-infinity step maps as disconnected blobs.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tac.analysis.snerv_step_map_coder import decode_step_maps
from tac.codec.receiver_integer_plane_codec import (
    SPATIAL_DELTA_ZIGZAG_LEB128_CODEC,
    ReceiverIntegerPlaneCodecError,
    canonical_int64_raw,
    decode_spatial_delta_zigzag_leb128_planes,
    encode_spatial_delta_zigzag_leb128_planes,
)
from tac.substrates._shared.int_stream_codec import (
    pack_fixed_width_uints,
    unpack_fixed_width_uints,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    DEFAULT_SNERV_MODEL_SIZE,
    HfGenerationDecoder,
    SnervFrameCode,
    SnervModelSizeConfig,
    decode_frame,
    dequantize_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.dwt import (
    WaveletPyramid,
    idwt2_multilevel,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (
    SNERV_LF_PAYLOAD_INTN_CODEC_PROOF as _SNERV_LF_PAYLOAD_INTN_CODEC_PROOF,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (
    SnervLfPayloadCodecError,
    decode_lf_quant_payload_v2,
    encode_lf_quant_payload_v2,
    inspect_lf_quant_payload_v2,
    is_lf_quant_payload_v2,
)
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    FALSE_AUTHORITY,
    OFFICIAL_SNERV_HFR_SOURCE_CONTRACT,
    OFFICIAL_SNERV_HFR_SOURCE_SHA,
    OfficialConv2dNchw,
    OfficialHfrConvBlock,
    OfficialHfrHeads,
)
from tac.substrates.snerv_inverse_steg_carrier.official_mfu import (
    OFFICIAL_SNERV_MFU_SOURCE,
    OFFICIAL_SNERV_T_MFU_SOURCE,
    OfficialConvTranspose2dNchw,
    OfficialResidualBlockNoBN,
    OfficialResidualBlocksWithInputConv,
    OfficialSnervMfu,
    OfficialSnervMfuSpec,
)
from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    OFFICIAL_SNERV_T_SOURCE_SHA,
    OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT,
    prepare_official_tub_graph_inputs,
)

SNERV_ARCHIVE_SCHEMA = "snerv_inverse_steg_archive.v1"
SNERV_ARCHIVE_MAGIC = b"SNAR1"
SNERV_LF_QUANT_MAGIC = b"SNQL1"
SNERV_DECODER_MAGIC = b"SNDC1"
SNERV_LF_PAYLOAD_INTN_CODEC_PROOF = _SNERV_LF_PAYLOAD_INTN_CODEC_PROOF
HEADER_LEN_FMT = "<I"
SECTION_ORDER = ("metadata_payload", "lf_payload", "decoder_payload", "step_map_packet")
DECODER_SUBBANDS = ("LH", "HL", "HH")
LF_QUANT_PAYLOAD_SCHEMA_V1 = "snerv_lf_quant_payload.v1"
LF_QUANT_PAYLOAD_SCHEMA_V2 = "snerv_lf_quant_payload.v2"
LF_QUANT_CODEC_INT64_LZMA = "int64_lzma"
LF_QUANT_CODEC_SPATIAL_DELTA_LEB128_LZMA = SPATIAL_DELTA_ZIGZAG_LEB128_CODEC
DECODER_PAYLOAD_V1_SCHEMA = "snerv_decoder_payload.v1"
DECODER_PAYLOAD_V2_SCHEMA = "snerv_decoder_payload.v2"
DECODER_PAYLOAD_V3_SCHEMA = "snerv_decoder_payload.v3"
DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA = (
    "snerv_decoder_payload.official_mfu_hfr_tub.v1"
)
DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_PROOF_SCHEMA = (
    "snerv_decoder_payload.official_mfu_hfr_tub.receiver_runtime_proof.v1"
)
DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SELF_CONSISTENCY_SCHEMA = (
    "snerv_decoder_payload.official_mfu_hfr_tub.receiver_self_consistency.v1"
)
DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_SCHEMA = (
    "snerv_decoder_payload.official_mfu_hfr_tub.source_forward_replay.v1"
)
DECODER_PAYLOAD_LEGACY_CODEC = "float32_lzma"
DECODER_PAYLOAD_MIXED_CODEC = "mixed_magnitude_symmetric"
DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_CODEC = "official_numpy_float64_lzma"
OFFICIAL_MFU_HFR_TUB_REQUIRED_TENSOR_KEYS: tuple[str, ...] = (
    "mfu.upsample_mid.weight",
    "mfu.upsample_mid.bias",
    "mfu.rb_mid.input_conv.weight",
    "mfu.rb_mid.input_conv.bias",
    "mfu.upsample_high.weight",
    "mfu.upsample_high.bias",
    "mfu.rb_high.input_conv.weight",
    "mfu.rb_high.input_conv.bias",
    "hfr.lh.conv1.weight",
    "hfr.lh.conv1.bias",
    "hfr.lh.conv2.weight",
    "hfr.lh.conv2.bias",
    "hfr.hl.conv1.weight",
    "hfr.hl.conv1.bias",
    "hfr.hl.conv2.weight",
    "hfr.hl.conv2.bias",
    "hfr.hh.conv1.weight",
    "hfr.hh.conv1.bias",
    "hfr.hh.conv2.weight",
    "hfr.hh.conv2.bias",
    "inputs.mfu.low",
    "inputs.mfu.skip_mid",
    "inputs.mfu.skip_high",
    "inputs.tub.current",
    "inputs.tub.previous",
    "inputs.tub.next_frame",
)
DECODER_PAYLOAD_QUANTIZED_CODECS = {
    "int8_symmetric": 8,
    "int4_symmetric": 4,
    "int2_symmetric": 2,
}
DECODER_SCALE_DTYPE_TO_NUMPY = {
    "float16_le": np.dtype("<f2"),
    "float32_le": np.dtype("<f4"),
}
DECODER_FP16_MAX_FINITE = float(np.finfo(np.float16).max)
DECODER_PAYLOAD_MIXED_MODE_TO_CODE = {
    "zero": 0,
    "int2": 1,
    "int4": 2,
    "int8": 3,
    "fp16": 4,
    "fp32": 5,
}
DECODER_PAYLOAD_MIXED_CODE_TO_MODE = {
    code: mode for mode, code in DECODER_PAYLOAD_MIXED_MODE_TO_CODE.items()
}


class SnervArchiveError(ValueError):
    """Raised when the SNeRV receiver archive packet is malformed."""


@dataclass(frozen=True)
class SnervArchivePacket:
    """A bundled receiver-visible SNeRV archive packet."""

    packet: bytes
    schema: str
    section_order: tuple[str, ...]
    section_bytes: dict[str, int]
    section_sha256: dict[str, str]
    metadata: dict[str, Any]
    header_bytes: int
    total_bytes: int
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["packet"] = {"bytes": len(self.packet), "sha256": _sha256(self.packet)}
        return d


@dataclass(frozen=True)
class DecodedSnervArchive:
    """Decoded SNeRV archive sections and metadata."""

    schema: str
    section_order: tuple[str, ...]
    sections: dict[str, bytes]
    metadata: dict[str, Any]
    packet_sha256: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def decode_step_maps(self) -> list[np.ndarray]:
        """Decode compact step maps from the bundled receiver packet."""

        return decode_step_maps(self.sections["step_map_packet"])

    def decode_lf_zero_points(self) -> np.ndarray:
        """Decode LF zero-point metadata from the bundled receiver packet."""

        expected = self.metadata.get("lf_plane_count")
        return decode_lf_metadata_payload(
            self.sections["metadata_payload"],
            expected_count=int(expected) if expected is not None else None,
        )

    def decode_lf_quant_planes(self) -> list[np.ndarray]:
        """Decode LF quantized coefficient planes from the bundled receiver packet."""

        return decode_lf_quant_payload(self.sections["lf_payload"])

    def decode_decoder(self) -> HfGenerationDecoder:
        """Decode the shared HF generator from the bundled receiver packet."""

        return decode_decoder_payload(self.sections["decoder_payload"])

    def decode_official_mfu_hfr_tub_payload(self) -> OfficialMfuHfrTubReceiverPayload:
        """Decode receiver-bound official MFU/HFR/TUB primitive payload bytes."""

        return decode_official_mfu_hfr_tub_decoder_payload(
            self.sections["decoder_payload"]
        )

    def execute_official_mfu_hfr_tub_payload(self) -> dict[str, Any]:
        """Run official MFU/HFR/TUB primitives from archived decoder bytes."""

        return execute_official_mfu_hfr_tub_decoder_payload(
            self.sections["decoder_payload"]
        )

    def decode_frame_planes(self, *, clip_to_uint8_range: bool = True) -> list[np.ndarray]:
        """Decode receiver-visible LF planes into ordered reconstructed frames.

        This is the scorer-free inflate primitive for the SNAR1 packet: it consumes
        only archived LF quant planes, archived zero-points, archived compact step
        maps, and the archived HF decoder. The return value is a flat plane list in
        archive order: pair-major, frame-major, channel-major when metadata includes
        the full-frame grouping fields.
        """

        return decode_snerv_archive_frame_planes_from_decoded(
            self,
            clip_to_uint8_range=clip_to_uint8_range,
        )

    def decode_frames(self, *, clip_to_uint8_range: bool = True) -> np.ndarray:
        """Decode a full receiver frame tensor ``(pairs, 2, 3, H, W)`` from SNAR1."""

        return decode_snerv_archive_frames_from_decoded(
            self,
            clip_to_uint8_range=clip_to_uint8_range,
        )


@dataclass(frozen=True)
class OfficialMfuHfrTubReceiverPayload:
    """Receiver-bound official MFU/HFR/TUB payload decoded from SNAR1 bytes."""

    header: dict[str, Any]
    tensors: dict[str, np.ndarray]
    payload_sha256: str
    payload_bytes: int
    score_claim: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    @property
    def schema(self) -> str:
        return str(self.header["schema"])

    def build_mfu(self) -> OfficialSnervMfu:
        """Hydrate official MFU executable primitives from decoded tensors."""

        spec = _official_mfu_spec_from_header(self.header)
        return OfficialSnervMfu(
            spec=spec,
            upsample_mid=OfficialConvTranspose2dNchw(
                self.tensors["mfu.upsample_mid.weight"],
                self.tensors["mfu.upsample_mid.bias"],
                stride=spec.mid_stride,
            ),
            rb_mid=_official_rb_from_tensors(
                self.tensors,
                prefix="mfu.rb_mid",
                num_blocks=spec.num_blocks,
            ),
            upsample_high=OfficialConvTranspose2dNchw(
                self.tensors["mfu.upsample_high.weight"],
                self.tensors["mfu.upsample_high.bias"],
                stride=spec.high_stride,
            ),
            rb_high=_official_rb_from_tensors(
                self.tensors,
                prefix="mfu.rb_high",
                num_blocks=spec.num_blocks,
            ),
        )

    def build_hfr_heads(self) -> OfficialHfrHeads:
        """Hydrate official HFR executable primitives from decoded tensors."""

        return OfficialHfrHeads(
            lh_head=_official_hfr_head_from_tensors(self.tensors, prefix="hfr.lh"),
            hl_head=_official_hfr_head_from_tensors(self.tensors, prefix="hfr.hl"),
            hh_head=_official_hfr_head_from_tensors(self.tensors, prefix="hfr.hh"),
        )

    def mfu_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return archived official MFU input bundle."""

        return (
            self.tensors["inputs.mfu.low"],
            self.tensors["inputs.mfu.skip_mid"],
            self.tensors["inputs.mfu.skip_high"],
        )

    def tub_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return archived official TUB source frame triplet."""

        return (
            self.tensors["inputs.tub.current"],
            self.tensors["inputs.tub.previous"],
            self.tensors["inputs.tub.next_frame"],
        )

    def execute(self) -> dict[str, Any]:
        """Execute receiver-side official primitives and return hashed proof."""

        low, skip_mid, skip_high = self.mfu_inputs()
        current, previous, next_frame = self.tub_inputs()
        mfu_out, hfr_out, tub_out, output_tensors = _execute_official_mfu_hfr_tub_forward(
            mfu=self.build_mfu(),
            hfr_heads=self.build_hfr_heads(),
            low=low,
            skip_mid=skip_mid,
            skip_high=skip_high,
            tub_current=current,
            tub_previous=previous,
            tub_next_frame=next_frame,
            tub_config=dict(self.header.get("tub_config") or {}),
        )
        output_rows, output_bundle_sha256 = _official_receiver_self_consistency_output_manifest(
            output_tensors
        )
        self_consistency_reference = _validate_official_receiver_self_consistency_reference(
            self.header,
            output_rows=output_rows,
            output_bundle_sha256=output_bundle_sha256,
        )
        return {
            "schema": DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_PROOF_SCHEMA,
            "payload_schema": self.schema,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": int(self.payload_bytes),
            "receiver_bound_official_primitive_payload": True,
            "receiver_export_bound": True,
            "receiver_runtime_decode_proven": True,
            "executed_components": {
                "official_mfu": True,
                "official_hfr": True,
                "official_tub": True,
            },
            "source_contracts": dict(self.header.get("source_contracts") or {}),
            "tensor_count": len(self.tensors),
            "tensor_manifest_sha256": _sha256(
                json.dumps(
                    self.header.get("tensor_manifest", []),
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ),
            "mfu_output": mfu_out.as_jsonable_metadata(),
            "hfr_output": hfr_out.as_jsonable(),
            "tub_output": tub_out.as_jsonable_metadata(),
            "output_tensors": output_rows,
            "output_bundle_sha256": output_bundle_sha256,
            "receiver_export_self_consistency_verified": True,
            "receiver_self_consistency_reference_sha256": _json_sha256(
                self_consistency_reference
            ),
            "source_forward_replay_bound": False,
            "source_forward_replay_verified": False,
            "source_forward_replay_authority": False,
            "contest_scorer_authority": False,
            **FALSE_AUTHORITY,
        }

    def decode_frame_planes(self, *, clip_to_uint8_range: bool = True) -> list[np.ndarray]:
        """Render official MFU/HFR payload outputs into receiver frame planes.

        This frame-producing official-payload bridge uses archived MFU tensors
        to generate the LL/pyr output, archived HFR heads to generate LH/HL/HH
        detail planes, and one-level Haar synthesis for every batch element.
        """

        low, skip_mid, skip_high = self.mfu_inputs()
        mfu_out = self.build_mfu().forward(low, skip_mid, skip_high)
        hfr_out = self.build_hfr_heads().forward(mfu_out.pyr_out)
        planes = _official_mfu_hfr_frame_planes(
            mfu_out.pyr_out,
            hfr_out.yh_out,
            clip_to_uint8_range=clip_to_uint8_range,
        )
        return planes

    def decode_frames(self, *, clip_to_uint8_range: bool = True) -> np.ndarray:
        """Render official MFU/HFR payload as ``(1, 1, C, H, W)`` frames."""

        planes = self.decode_frame_planes(clip_to_uint8_range=clip_to_uint8_range)
        if not planes:
            raise SnervArchiveError("official payload produced no frame planes")
        shape = planes[0].shape
        if any(plane.shape != shape for plane in planes):
            raise SnervArchiveError("official payload produced ragged frame planes")
        return np.stack(planes, axis=0)[np.newaxis, np.newaxis, :, :, :].astype(
            np.float32
        )

    def as_jsonable(self) -> dict[str, Any]:
        """Return payload metadata without embedding tensor bytes."""

        return {
            "schema": self.schema,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": int(self.payload_bytes),
            "header": dict(self.header),
            "tensor_count": len(self.tensors),
            **FALSE_AUTHORITY,
        }


def _execute_official_mfu_hfr_tub_forward(
    *,
    mfu: OfficialSnervMfu,
    hfr_heads: OfficialHfrHeads,
    low: np.ndarray,
    skip_mid: np.ndarray,
    skip_high: np.ndarray,
    tub_current: np.ndarray,
    tub_previous: np.ndarray,
    tub_next_frame: np.ndarray,
    tub_config: Mapping[str, Any],
) -> tuple[Any, Any, Any, dict[str, np.ndarray]]:
    mfu_out = mfu.forward(low, skip_mid, skip_high)
    hfr_out = hfr_heads.forward(mfu_out.pyr_out)
    temporal_shape = tub_config.get("temporal_encoder_output_shape")
    fc_hw = tub_config.get("fc_hw")
    decoder_shape = tub_config.get("output2_decoder_output_shape")
    tub_out = prepare_official_tub_graph_inputs(
        tub_current,
        tub_previous,
        tub_next_frame,
        temporal_encoder_output_shape=(
            tuple(int(v) for v in temporal_shape) if temporal_shape is not None else None
        ),
        fc_hw=tuple(int(v) for v in fc_hw) if fc_hw is not None else None,
        output2_decoder_output_shape=(
            tuple(int(v) for v in decoder_shape) if decoder_shape is not None else None
        ),
    )
    output_tensors = {
        "mfu.pyr_out": mfu_out.pyr_out,
        "hfr.yh_out": hfr_out.yh_out,
        "tub.normalized_lf": tub_out.normalized_lf,
        "tub.prev_lowpass_over_2": tub_out.prev_lowpass_over_2,
        "tub.next_lowpass_over_2": tub_out.next_lowpass_over_2,
    }
    return mfu_out, hfr_out, tub_out, output_tensors


def _official_mfu_hfr_frame_planes(
    pyr_out: np.ndarray,
    yh_out: np.ndarray,
    *,
    clip_to_uint8_range: bool,
) -> list[np.ndarray]:
    ll = np.asarray(pyr_out, dtype=np.float64)
    yh = np.asarray(yh_out, dtype=np.float64)
    if ll.ndim != 4:
        raise SnervArchiveError(f"official MFU pyr_out must be NCHW, got {ll.shape}")
    if yh.ndim != 5 or int(yh.shape[2]) != 3:
        raise SnervArchiveError(
            f"official HFR yh_out must be (N,C,3,H,W), got {yh.shape}"
        )
    if int(ll.shape[0]) != int(yh.shape[0]):
        raise SnervArchiveError(
            f"official MFU batch {ll.shape[0]} != HFR batch {yh.shape[0]}"
        )
    if tuple(int(v) for v in ll.shape[-2:]) != tuple(int(v) for v in yh.shape[-2:]):
        raise SnervArchiveError(
            f"official LL shape {ll.shape[-2:]} != HFR detail shape {yh.shape[-2:]}"
        )
    detail_channels = int(yh.shape[1])
    if int(ll.shape[1]) not in (1, detail_channels):
        raise SnervArchiveError(
            "official MFU pyr_out channels must be 1 or match HFR channels "
            f"({ll.shape[1]} vs {detail_channels})"
        )
    planes: list[np.ndarray] = []
    h, w = int(ll.shape[-2]), int(ll.shape[-1])
    for batch in range(int(ll.shape[0])):
        for channel in range(detail_channels):
            ll_channel = 0 if int(ll.shape[1]) == 1 else channel
            pyramid = WaveletPyramid(
                coeffs=[
                    ll[batch, ll_channel],
                    (
                        yh[batch, channel, 0],
                        yh[batch, channel, 1],
                        yh[batch, channel, 2],
                    ),
                ],
                levels=1,
                wavelet="haar",
                orig_hw=(2 * h, 2 * w),
                padded_hw=(2 * h, 2 * w),
            )
            plane = idwt2_multilevel(pyramid)
            if clip_to_uint8_range:
                plane = np.clip(plane, 0.0, 255.0)
            planes.append(np.asarray(plane, dtype=np.float32))
    return planes


def _build_official_receiver_self_consistency_reference(
    *,
    mfu: OfficialSnervMfu,
    hfr_heads: OfficialHfrHeads,
    low: np.ndarray,
    skip_mid: np.ndarray,
    skip_high: np.ndarray,
    tub_current: np.ndarray,
    tub_previous: np.ndarray,
    tub_next_frame: np.ndarray,
    tub_config: Mapping[str, Any],
) -> dict[str, Any]:
    _mfu_out, _hfr_out, _tub_out, output_tensors = _execute_official_mfu_hfr_tub_forward(
        mfu=mfu,
        hfr_heads=hfr_heads,
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high,
        tub_current=tub_current,
        tub_previous=tub_previous,
        tub_next_frame=tub_next_frame,
        tub_config=tub_config,
    )
    output_rows, output_bundle_sha256 = _official_receiver_self_consistency_output_manifest(
        output_tensors
    )
    return {
        "schema": DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SELF_CONSISTENCY_SCHEMA,
        "backend": "official_numpy_export_vs_receiver_replay",
        "receiver_export_payload_bound": True,
        "receiver_export_self_consistency_verified": True,
        "source_forward_replay_verified_by_export": False,
        "output_tensors": output_rows,
        "output_bundle_sha256": output_bundle_sha256,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _official_receiver_self_consistency_output_manifest(
    output_tensors: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], str]:
    output_rows = [
        _tensor_manifest_row(name, np.asarray(array, dtype="<f8"))
        for name, array in output_tensors.items()
    ]
    output_bundle_sha256 = _sha256(
        b"".join(
            np.ascontiguousarray(np.asarray(array, dtype="<f8")).tobytes()
            for array in output_tensors.values()
        )
    )
    return output_rows, output_bundle_sha256


def pack_snerv_archive(
    *,
    metadata_payload: bytes,
    lf_payload: bytes,
    decoder_payload: bytes,
    step_map_packet: bytes,
    metadata: dict[str, Any] | None = None,
) -> SnervArchivePacket:
    """Bundle receiver-visible SNeRV sections into one deterministic packet."""

    sections = {
        "metadata_payload": bytes(metadata_payload),
        "lf_payload": bytes(lf_payload),
        "decoder_payload": bytes(decoder_payload),
        "step_map_packet": bytes(step_map_packet),
    }
    _validate_sections(sections)
    clean_metadata = _jsonable_metadata(metadata or {})
    cursor = 0
    section_headers = []
    payload_parts = []
    section_bytes: dict[str, int] = {}
    section_sha256: dict[str, str] = {}
    for name in SECTION_ORDER:
        blob = sections[name]
        section_headers.append(
            {
                "name": name,
                "offset": cursor,
                "bytes": len(blob),
                "sha256": _sha256(blob),
            }
        )
        payload_parts.append(blob)
        section_bytes[name] = len(blob)
        section_sha256[name] = _sha256(blob)
        cursor += len(blob)
    header = {
        "schema": SNERV_ARCHIVE_SCHEMA,
        "section_order": list(SECTION_ORDER),
        "sections": section_headers,
        "metadata": clean_metadata,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    header_bytes_raw = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    packet = (
        SNERV_ARCHIVE_MAGIC
        + struct.pack(HEADER_LEN_FMT, len(header_bytes_raw))
        + header_bytes_raw
        + b"".join(payload_parts)
    )
    return SnervArchivePacket(
        packet=packet,
        schema=SNERV_ARCHIVE_SCHEMA,
        section_order=SECTION_ORDER,
        section_bytes=section_bytes,
        section_sha256=section_sha256,
        metadata=clean_metadata,
        header_bytes=len(SNERV_ARCHIVE_MAGIC)
        + struct.calcsize(HEADER_LEN_FMT)
        + len(header_bytes_raw),
        total_bytes=len(packet),
    )


def unpack_snerv_archive(packet: bytes) -> DecodedSnervArchive:
    """Decode and validate a bundled SNeRV archive packet."""

    packet = bytes(packet)
    if not packet.startswith(SNERV_ARCHIVE_MAGIC):
        raise SnervArchiveError("bad SNeRV archive magic")
    offset = len(SNERV_ARCHIVE_MAGIC)
    if len(packet) < offset + struct.calcsize(HEADER_LEN_FMT):
        raise SnervArchiveError("truncated SNeRV archive header")
    (header_len,) = struct.unpack(
        HEADER_LEN_FMT,
        packet[offset : offset + struct.calcsize(HEADER_LEN_FMT)],
    )
    offset += struct.calcsize(HEADER_LEN_FMT)
    header_end = offset + header_len
    if header_end > len(packet):
        raise SnervArchiveError("declared SNeRV archive header exceeds packet size")
    header = json.loads(packet[offset:header_end].decode("utf-8"))
    if header.get("schema") != SNERV_ARCHIVE_SCHEMA:
        raise SnervArchiveError(f"unsupported SNeRV archive schema: {header.get('schema')!r}")
    section_order = tuple(str(v) for v in header.get("section_order", []))
    if section_order != SECTION_ORDER:
        raise SnervArchiveError(f"unexpected section order: {section_order!r}")
    payload = packet[header_end:]
    sections: dict[str, bytes] = {}
    expected_offset = 0
    seen: set[str] = set()
    for row in header.get("sections", []):
        name = str(row["name"])
        if name not in SECTION_ORDER:
            raise SnervArchiveError(f"unknown SNeRV archive section: {name!r}")
        if name in seen:
            raise SnervArchiveError(f"duplicate SNeRV archive section: {name!r}")
        start = int(row["offset"])
        end = start + int(row["bytes"])
        if start != expected_offset:
            raise SnervArchiveError(
                f"SNeRV archive section {name!r} offset {start} != expected {expected_offset}"
            )
        if start < 0 or end > len(payload):
            raise SnervArchiveError(f"SNeRV archive section {name!r} out of range")
        blob = payload[start:end]
        expected_sha = str(row["sha256"])
        if _sha256(blob) != expected_sha:
            raise SnervArchiveError(f"SNeRV archive section {name!r} sha256 mismatch")
        sections[name] = blob
        seen.add(name)
        expected_offset = end
    if expected_offset != len(payload):
        raise SnervArchiveError("SNeRV archive has unreferenced trailing payload bytes")
    if tuple(sections.keys()) != SECTION_ORDER:
        raise SnervArchiveError("SNeRV archive missing required sections")
    return DecodedSnervArchive(
        schema=SNERV_ARCHIVE_SCHEMA,
        section_order=SECTION_ORDER,
        sections=sections,
        metadata=dict(header.get("metadata", {})),
        packet_sha256=_sha256(packet),
    )


def decode_snerv_archive_step_maps(packet: bytes) -> list[np.ndarray]:
    """Convenience helper for receiver-side step-map decode proof."""

    return unpack_snerv_archive(packet).decode_step_maps()


def decode_snerv_archive_frame_planes(
    packet: bytes,
    *,
    clip_to_uint8_range: bool = True,
) -> list[np.ndarray]:
    """Decode all archived SNeRV frame planes without scorer/torch imports."""

    return unpack_snerv_archive(packet).decode_frame_planes(
        clip_to_uint8_range=clip_to_uint8_range,
    )


def decode_snerv_archive_frames(
    packet: bytes,
    *,
    clip_to_uint8_range: bool = True,
) -> np.ndarray:
    """Decode a full ``(n_pairs, 2, 3, H, W)`` receiver tensor from SNAR1 bytes."""

    return unpack_snerv_archive(packet).decode_frames(
        clip_to_uint8_range=clip_to_uint8_range,
    )


def decode_snerv_archive_frame_planes_from_decoded(
    decoded: DecodedSnervArchive,
    *,
    clip_to_uint8_range: bool = True,
) -> list[np.ndarray]:
    """Decode archived LF planes into receiver frames from an unpacked archive."""

    if is_official_mfu_hfr_tub_decoder_payload(decoded.sections["decoder_payload"]):
        return decoded.decode_official_mfu_hfr_tub_payload().decode_frame_planes(
            clip_to_uint8_range=clip_to_uint8_range,
        )

    metadata = decoded.metadata
    levels = _metadata_int(metadata, "levels", minimum=1)
    wavelet = _metadata_str(metadata, "wavelet")
    orig_hw = _metadata_hw(metadata)
    lf_planes = decoded.decode_lf_quant_planes()
    zeros = decoded.decode_lf_zero_points()
    step_maps = decoded.decode_step_maps()
    decoder = decoded.decode_decoder()
    _validate_replay_counts(lf_planes, zeros, step_maps)

    codes: list[SnervFrameCode] = []
    decoded_lfs: list[np.ndarray] = []
    for idx, (q, zero, steps) in enumerate(zip(lf_planes, zeros, step_maps, strict=True)):
        if q.shape != steps.shape:
            raise SnervArchiveError(
                f"receiver replay plane {idx} LF shape {q.shape} != step shape {steps.shape}"
            )
        code = SnervFrameCode(
            lf_quant=q,
            lf_scale=1.0,
            lf_zero=float(zero),
            lf_shape=tuple(int(v) for v in q.shape),
            levels=levels,
            wavelet=wavelet,
            orig_hw=orig_hw,
            per_element_steps=steps,
        )
        codes.append(code)
        decoded_lfs.append(
            dequantize_lf(
                q,
                1.0,
                float(zero),
                per_element_steps=steps,
            )
        )

    temporal_group_count = 1
    if int(decoder.model_size.temporal_context) > 0:
        temporal_group_count = _metadata_int(metadata, "channels", default=1, minimum=1)

    out: list[np.ndarray] = []
    for idx, code in enumerate(codes):
        lf_sequence = None
        sequence_index = None
        if int(decoder.model_size.temporal_context) > 0:
            group = idx % temporal_group_count
            lf_sequence = decoded_lfs[group::temporal_group_count]
            sequence_index = idx // temporal_group_count
        frame = decode_frame(
            code,
            decoder,
            lf_sequence=lf_sequence,
            sequence_index=sequence_index,
        )
        if clip_to_uint8_range:
            frame = np.clip(frame, 0.0, 255.0)
        out.append(np.asarray(frame, dtype=np.float32))
    return out


def decode_snerv_archive_frames_from_decoded(
    decoded: DecodedSnervArchive,
    *,
    clip_to_uint8_range: bool = True,
) -> np.ndarray:
    """Decode and group receiver frames as ``(n_pairs, 2, 3, H, W)``."""

    metadata = decoded.metadata
    n_pairs = _metadata_int(metadata, "n_pairs", minimum=1)
    frames_per_pair = _metadata_int(metadata, "frames_per_pair", default=2, minimum=1)
    channels = _metadata_int(metadata, "channels", default=3, minimum=1)
    h, w = _metadata_hw(metadata)
    planes = decode_snerv_archive_frame_planes_from_decoded(
        decoded,
        clip_to_uint8_range=clip_to_uint8_range,
    )
    expected = n_pairs * frames_per_pair * channels
    if len(planes) != expected:
        raise SnervArchiveError(
            f"receiver replay decoded {len(planes)} planes, expected {expected} "
            f"from n_pairs={n_pairs}, frames_per_pair={frames_per_pair}, channels={channels}"
        )
    arr = np.stack(planes, axis=0)
    return arr.reshape(n_pairs, frames_per_pair, channels, h, w).astype(np.float32)


def encode_lf_metadata_payload(
    *,
    lf_zero_points: list[float] | np.ndarray,
) -> bytes:
    """Encode LF dequant metadata as compact receiver-visible bytes."""

    zeros = np.asarray(lf_zero_points, dtype="<f4").reshape(-1)
    if zeros.size == 0:
        raise SnervArchiveError("lf_zero_points must be non-empty")
    if not np.all(np.isfinite(zeros)):
        raise SnervArchiveError("lf_zero_points must be finite")
    return zeros.tobytes()


def encode_lf_quant_payload(
    lf_quant_planes: list[np.ndarray],
    *,
    codec: str = LF_QUANT_CODEC_INT64_LZMA,
) -> bytes:
    """Encode quantized LF planes as deterministic scorer-free receiver bytes."""

    arrays = [_validate_lf_quant_plane(a) for a in lf_quant_planes]
    if not arrays:
        raise SnervArchiveError("lf_quant_planes must be non-empty")
    normalized = str(codec).strip().lower()
    if normalized in {"v1", "legacy", "int64", "int64_lzma"}:
        return _encode_lf_quant_payload_int64_lzma(arrays)
    if normalized in {
        "v2",
        "spatial_delta",
        "spatial_delta_zigzag_leb128",
        LF_QUANT_CODEC_SPATIAL_DELTA_LEB128_LZMA,
    }:
        return _encode_lf_quant_payload_spatial_delta_leb128_lzma(arrays)
    if normalized == "auto":
        candidates = (
            _encode_lf_quant_payload_int64_lzma(arrays),
            _encode_lf_quant_payload_spatial_delta_leb128_lzma(arrays),
        )
        return min(candidates, key=len)
    try:
        return encode_lf_quant_payload_v2(arrays, mode=normalized)
    except SnervLfPayloadCodecError as exc:
        raise SnervArchiveError(str(exc)) from exc
    raise SnervArchiveError(f"unsupported LF quant payload codec: {codec!r}")


def _encode_lf_quant_payload_int64_lzma(arrays: list[np.ndarray]) -> bytes:
    raw = canonical_int64_raw(arrays)
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": LF_QUANT_PAYLOAD_SCHEMA_V1,
        "codec": LF_QUANT_CODEC_INT64_LZMA,
        "dtype": "int64_le",
        "shapes": [list(a.shape) for a in arrays],
        "raw_sha256": _sha256(raw),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    return _pack_subpacket(SNERV_LF_QUANT_MAGIC, header, compressed)


def _encode_lf_quant_payload_spatial_delta_leb128_lzma(
    arrays: list[np.ndarray],
) -> bytes:
    payload = encode_spatial_delta_zigzag_leb128_planes(arrays)
    compressed = lzma.compress(
        payload.raw,
        format=lzma.FORMAT_XZ,
        preset=9 | lzma.PRESET_EXTREME,
    )
    header = {
        "schema": LF_QUANT_PAYLOAD_SCHEMA_V2,
        **payload.header,
        "compressed_bytes": len(compressed),
    }
    return _pack_subpacket(SNERV_LF_QUANT_MAGIC, header, compressed)


def decode_lf_quant_payload(payload: bytes) -> list[np.ndarray]:
    """Decode LF quantized coefficient planes from receiver payload bytes."""

    if is_lf_quant_payload_v2(payload):
        try:
            return decode_lf_quant_payload_v2(payload)
        except SnervLfPayloadCodecError as exc:
            raise SnervArchiveError(str(exc)) from exc
    header, compressed = _unpack_lf_quant_subpacket(payload)
    schema = str(header.get("schema"))
    codec = str(header.get("codec", LF_QUANT_CODEC_INT64_LZMA))
    if schema == LF_QUANT_PAYLOAD_SCHEMA_V2:
        if codec != LF_QUANT_CODEC_SPATIAL_DELTA_LEB128_LZMA:
            raise SnervArchiveError(f"unsupported LF quant payload codec: {codec!r}")
        return _decode_lf_quant_payload_spatial_delta_leb128_lzma(header, compressed)
    if schema != LF_QUANT_PAYLOAD_SCHEMA_V1:
        raise SnervArchiveError(f"unsupported subpacket schema: {schema!r}")
    if codec != LF_QUANT_CODEC_INT64_LZMA:
        raise SnervArchiveError(f"unsupported LF quant payload codec: {codec!r}")
    raw = lzma.decompress(compressed)
    if len(raw) != int(header["raw_bytes"]):
        raise SnervArchiveError("LF quant payload raw byte count mismatch")
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervArchiveError("LF quant payload raw sha256 mismatch")
    out = []
    cursor = 0
    for shape in header["shapes"]:
        out_shape = tuple(int(v) for v in shape)
        count = int(np.prod(out_shape))
        nbytes = count * np.dtype("<i8").itemsize
        arr = np.frombuffer(raw[cursor : cursor + nbytes], dtype="<i8").copy()
        if arr.size != count:
            raise SnervArchiveError("LF quant payload ended inside a plane")
        out.append(arr.reshape(out_shape))
        cursor += nbytes
    if cursor != len(raw):
        raise SnervArchiveError("LF quant payload has unused raw bytes")
    return out


def inspect_lf_quant_payload_header(payload: bytes) -> dict[str, Any]:
    """Return validated LF payload header metadata without decoding planes."""

    if is_lf_quant_payload_v2(payload):
        try:
            report = inspect_lf_quant_payload_v2(payload).as_jsonable()
        except SnervLfPayloadCodecError as exc:
            raise SnervArchiveError(str(exc)) from exc
        report["payload_bytes"] = int(report.get("payload_bytes") or 0)
        report["section_bytes"] = len(payload)
        return report
    header, body = _unpack_lf_quant_subpacket(payload)
    out = dict(header)
    out["payload_bytes"] = len(body)
    out["section_bytes"] = len(payload)
    return out


def _decode_lf_quant_payload_spatial_delta_leb128_lzma(
    header: dict[str, Any],
    compressed: bytes,
) -> list[np.ndarray]:
    try:
        raw = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise SnervArchiveError("LF quant payload decompression failed") from exc
    if len(raw) != int(header["raw_bytes"]):
        raise SnervArchiveError("LF quant payload raw byte count mismatch")
    try:
        return decode_spatial_delta_zigzag_leb128_planes(raw, header=header)
    except ReceiverIntegerPlaneCodecError as exc:
        raise SnervArchiveError(str(exc)) from exc


def _encode_decoder_scale_payload(scales: Sequence[float]) -> tuple[bytes, str]:
    arr64 = np.asarray(scales, dtype=np.float64)
    if not np.all(np.isfinite(arr64)):
        raise SnervArchiveError("decoder quantized scales must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        arr16 = arr64.astype("<f2")
    use_fp16 = np.all(np.isfinite(arr16))
    if use_fp16 and arr64.size:
        positive = arr64 > 0.0
        use_fp16 = bool(np.all(arr16[positive] > 0.0))
    if use_fp16:
        return arr16.tobytes(), "float16_le"
    arr32 = arr64.astype("<f4")
    if not np.all(np.isfinite(arr32)):
        raise SnervArchiveError("decoder quantized scales exceed float32 range")
    if arr64.size:
        positive = arr64 > 0.0
        if not np.all(arr32[positive] > 0.0):
            raise SnervArchiveError("decoder quantized scales underflow float32")
    return arr32.tobytes(), "float32_le"


def _validate_finite_decoder_values(values: np.ndarray, *, context: str) -> np.ndarray:
    arr64 = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr64)):
        raise SnervArchiveError(f"{context} must be finite")
    return arr64


def _decoder_float32_payload(values: np.ndarray, *, context: str) -> bytes:
    arr64 = _validate_finite_decoder_values(values, context=context)
    max_abs = float(np.max(np.abs(arr64))) if arr64.size else 0.0
    if max_abs > float(np.finfo(np.float32).max):
        raise SnervArchiveError(f"{context} exceeds float32 receiver range")
    return arr64.astype("<f4").tobytes()


def _decoder_scale_dtype(header: Mapping[str, Any]) -> np.dtype:
    raw = str(header.get("scale_dtype", "float16_le"))
    dtype = DECODER_SCALE_DTYPE_TO_NUMPY.get(raw)
    if dtype is None:
        raise SnervArchiveError(f"unsupported decoder scale dtype: {raw!r}")
    return dtype


def encode_decoder_payload(
    decoder: HfGenerationDecoder,
    *,
    codec: str = DECODER_PAYLOAD_LEGACY_CODEC,
    mixed_modes: Sequence[str] | None = None,
) -> bytes:
    """Encode the shared HF decoder as deterministic scorer-free receiver bytes."""

    levels, values, model_size = _decoder_to_flat_values(decoder)
    raw = _decoder_float32_payload(values, context="decoder raw reference")
    if not raw:
        raise SnervArchiveError("decoder payload must be non-empty")
    normalized = str(codec).strip().lower()
    if normalized in {
        DECODER_PAYLOAD_LEGACY_CODEC,
        "fp32_lzma",
        "float32",
        "legacy",
    }:
        if mixed_modes is not None:
            raise SnervArchiveError("mixed decoder modes require mixed codec")
        return _encode_decoder_payload_v1(
            levels=levels,
            raw=raw,
            model_size=model_size,
        )
    if normalized in DECODER_PAYLOAD_QUANTIZED_CODECS:
        if mixed_modes is not None:
            raise SnervArchiveError("mixed decoder modes require mixed codec")
        return _encode_decoder_payload_quantized(
            levels=levels,
            values=values,
            model_size=model_size,
            bits=DECODER_PAYLOAD_QUANTIZED_CODECS[normalized],
            codec=normalized,
            raw_reference=raw,
        )
    if normalized in {
        DECODER_PAYLOAD_MIXED_CODEC,
        "mixed_per_kernel_symmetric",
        "mixed_symmetric",
    }:
        return _encode_decoder_payload_mixed(
            levels=levels,
            values=values,
            model_size=model_size,
            raw_reference=raw,
            explicit_modes=mixed_modes,
        )
    raise SnervArchiveError(f"unsupported decoder payload codec: {codec!r}")


def encode_official_mfu_hfr_tub_decoder_payload(
    *,
    mfu: OfficialSnervMfu,
    hfr_heads: OfficialHfrHeads,
    low: np.ndarray,
    skip_mid: np.ndarray,
    skip_high: np.ndarray,
    tub_current: np.ndarray,
    tub_previous: np.ndarray,
    tub_next_frame: np.ndarray,
    temporal_encoder_output_shape: tuple[int, int, int, int] | None = None,
    fc_hw: tuple[int, int] | None = None,
    output2_decoder_output_shape: tuple[int, int, int, int] | None = None,
) -> bytes:
    """Encode executable official MFU/HFR/TUB receiver primitive bytes.

    This is not a contest-score claim. It is the receiver/runtime custody
    surface for official primitive tensors and inputs: the decoder section can
    now carry bytes that are decoded into the official NumPy primitives instead
    of only carrying the local linear HF surrogate.
    """

    tensors = _official_payload_tensor_dict(
        mfu=mfu,
        hfr_heads=hfr_heads,
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high,
        tub_current=tub_current,
        tub_previous=tub_previous,
        tub_next_frame=tub_next_frame,
    )
    tensor_manifest, raw = _pack_tensor_manifest(tensors)
    compressed = lzma.compress(
        raw,
        format=lzma.FORMAT_XZ,
        preset=9 | lzma.PRESET_EXTREME,
    )
    spec = mfu.spec
    tub_config = {
        "temporal_encoder_output_shape": (
            [int(v) for v in temporal_encoder_output_shape]
            if temporal_encoder_output_shape is not None
            else None
        ),
        "fc_hw": [int(v) for v in fc_hw] if fc_hw is not None else None,
        "output2_decoder_output_shape": (
            [int(v) for v in output2_decoder_output_shape]
            if output2_decoder_output_shape is not None
            else None
        ),
    }
    self_consistency_reference = _build_official_receiver_self_consistency_reference(
        mfu=mfu,
        hfr_heads=hfr_heads,
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high,
        tub_current=tub_current,
        tub_previous=tub_previous,
        tub_next_frame=tub_next_frame,
        tub_config=tub_config,
    )
    header = {
        "schema": DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
        "codec": DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_CODEC,
        "dtype": "float64_le",
        "mfu_spec": {
            "low_channels": int(spec.low_channels),
            "mid_channels": int(spec.mid_channels),
            "high_channels": int(spec.high_channels),
            "mid_stride": int(spec.mid_stride),
            "high_stride": int(spec.high_stride),
            "num_blocks": int(spec.num_blocks),
            "source": str(spec.source),
        },
        "hfr_in_channels": int(hfr_heads.in_channels),
        "tub_config": tub_config,
        "source_contracts": {
            "mfu": str(spec.source),
            "mfu_non_temporal": OFFICIAL_SNERV_MFU_SOURCE,
            "mfu_temporal": OFFICIAL_SNERV_T_MFU_SOURCE,
            "hfr": OFFICIAL_SNERV_HFR_SOURCE_CONTRACT,
            "hfr_source_sha": OFFICIAL_SNERV_HFR_SOURCE_SHA,
            "tub": OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT,
            "tub_source_sha": OFFICIAL_SNERV_T_SOURCE_SHA,
        },
        "tensor_manifest": tensor_manifest,
        "tensor_count": len(tensor_manifest),
        "raw_tensor_bytes": len(raw),
        "raw_tensor_sha256": _sha256(raw),
        "compressed_bytes": len(compressed),
        "compressed_sha256": _sha256(compressed),
        "receiver_self_consistency_reference": self_consistency_reference,
        "receiver_self_consistency_reference_sha256": _json_sha256(
            self_consistency_reference
        ),
        "receiver_export_payload_bound": True,
        "receiver_export_self_consistency_verified": True,
        "source_forward_replay_bound_by_export": False,
        "receiver_runtime_decode_proven_by_payload": False,
        "source_forward_replay_authority": False,
        **FALSE_AUTHORITY,
    }
    return _pack_subpacket(SNERV_DECODER_MAGIC, header, compressed)


def _encode_decoder_payload_v1(
    *,
    levels: int,
    raw: bytes,
    model_size: SnervModelSizeConfig,
) -> bytes:
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": DECODER_PAYLOAD_V1_SCHEMA,
        "levels": levels,
        "subbands": list(DECODER_SUBBANDS),
        "kernel_shape": _decoder_kernel_shape_header(model_size),
        "feature_count": int(model_size.feature_count),
        "model_size_config": model_size.as_jsonable(),
        "dtype": "float32_le",
        "raw_sha256": _sha256(raw),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    return _pack_subpacket(SNERV_DECODER_MAGIC, header, compressed)


def _encode_decoder_payload_quantized(
    *,
    levels: int,
    values: np.ndarray,
    model_size: SnervModelSizeConfig,
    bits: int,
    codec: str,
    raw_reference: bytes,
) -> bytes:
    qmax = (1 << (int(bits) - 1)) - 1
    if qmax < 1:
        raise SnervArchiveError(f"invalid decoder quantizer bits: {bits}")
    feature_count = int(model_size.feature_count)
    value_groups = values.reshape(levels * len(DECODER_SUBBANDS), feature_count)
    scales = []
    unsigned_parts = []
    max_abs_error = 0.0
    mean_abs_errors = []
    for group in value_groups:
        group64 = _validate_finite_decoder_values(
            group,
            context="decoder quantized group",
        )
        max_abs = float(np.max(np.abs(group64))) if group64.size else 0.0
        scale = 1.0 if max_abs == 0.0 else max_abs / float(qmax)
        q_signed = np.round(group64 / scale).clip(-qmax, qmax).astype(np.int64)
        dequant = q_signed.astype(np.float64) * scale
        err = np.abs(dequant - group64)
        max_abs_error = max(max_abs_error, float(np.max(err)) if err.size else 0.0)
        mean_abs_errors.append(float(np.mean(err)) if err.size else 0.0)
        scales.append(scale)
        unsigned_parts.append((q_signed + qmax).astype(np.int64))
    q_unsigned = np.concatenate(unsigned_parts) if unsigned_parts else np.zeros(0)
    packed_q = pack_fixed_width_uints(q_unsigned, bits=bits)
    scale_payload, scale_dtype = _encode_decoder_scale_payload(scales)
    raw_payload = scale_payload + packed_q
    header = {
        "schema": DECODER_PAYLOAD_V2_SCHEMA,
        "levels": levels,
        "subbands": list(DECODER_SUBBANDS),
        "kernel_shape": _decoder_kernel_shape_header(model_size),
        "feature_count": feature_count,
        "model_size_config": model_size.as_jsonable(),
        "codec": codec,
        "bits_per_weight": int(bits),
        "quantizer": "symmetric_per_kernel_adaptive_scale",
        "q_offset": int(qmax),
        "scale_dtype": scale_dtype,
        "scale_count": len(scales),
        "scale_bytes": len(scale_payload),
        "packed_q_bytes": len(packed_q),
        "value_count": int(values.size),
        "raw_reference_sha256": _sha256(raw_reference),
        "raw_reference_bytes": len(raw_reference),
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(np.mean(mean_abs_errors)) if mean_abs_errors else 0.0,
        "payload_sha256": _sha256(raw_payload),
        "payload_bytes": len(raw_payload),
    }
    return _pack_subpacket(SNERV_DECODER_MAGIC, header, raw_payload)


def _encode_decoder_payload_mixed(
    *,
    levels: int,
    values: np.ndarray,
    model_size: SnervModelSizeConfig,
    raw_reference: bytes,
    explicit_modes: Sequence[str] | None = None,
) -> bytes:
    feature_count = int(model_size.feature_count)
    value_groups = values.reshape(levels * len(DECODER_SUBBANDS), feature_count)
    mode_plan: tuple[str, ...] | None = None
    if explicit_modes is not None:
        mode_plan = tuple(_normalize_mixed_decoder_kernel_mode(v) for v in explicit_modes)
        if len(mode_plan) != len(value_groups):
            raise SnervArchiveError(
                f"decoder mixed mode count {len(mode_plan)} != expected "
                f"{len(value_groups)}"
            )
    mode_codes: list[int] = []
    scales = []
    q_parts: list[bytes] = []
    fp16_parts: list[bytes] = []
    fp32_parts: list[bytes] = []
    max_abs_error = 0.0
    mean_abs_errors = []
    histogram = dict.fromkeys(DECODER_PAYLOAD_MIXED_MODE_TO_CODE, 0)
    for idx, group in enumerate(value_groups):
        group64 = _validate_finite_decoder_values(
            group,
            context="decoder mixed group",
        )
        mode = (
            mode_plan[idx]
            if mode_plan is not None
            else _select_mixed_decoder_kernel_mode(group64)
        )
        histogram[mode] += 1
        mode_codes.append(DECODER_PAYLOAD_MIXED_MODE_TO_CODE[mode])
        if mode == "zero":
            dequant = np.zeros_like(group64, dtype=np.float64)
        elif mode == "fp16":
            max_abs = float(np.max(np.abs(group64))) if group64.size else 0.0
            if max_abs > DECODER_FP16_MAX_FINITE:
                raise SnervArchiveError(
                    "decoder mixed fp16 group exceeds float16 receiver range"
                )
            payload = np.asarray(group64, dtype="<f2").tobytes()
            fp16_parts.append(payload)
            dequant = np.frombuffer(payload, dtype="<f2").astype(np.float64)
        elif mode == "fp32":
            payload = _decoder_float32_payload(
                group64,
                context="decoder mixed fp32 group",
            )
            fp32_parts.append(payload)
            dequant = np.frombuffer(payload, dtype="<f4").astype(np.float64)
        else:
            bits = int(mode.removeprefix("int"))
            qmax = (1 << (bits - 1)) - 1
            max_abs = float(np.max(np.abs(group64))) if group64.size else 0.0
            scale = 1.0 if max_abs == 0.0 else max_abs / float(qmax)
            q_signed = np.round(group64 / scale).clip(-qmax, qmax).astype(np.int64)
            q_parts.append(pack_fixed_width_uints(q_signed + qmax, bits=bits))
            scales.append(scale)
            dequant = q_signed.astype(np.float64) * scale
        err = np.abs(dequant - group64)
        max_abs_error = max(max_abs_error, float(np.max(err)) if err.size else 0.0)
        mean_abs_errors.append(float(np.mean(err)) if err.size else 0.0)
    mode_code_payload = pack_fixed_width_uints(mode_codes, bits=3)
    scale_payload, scale_dtype = _encode_decoder_scale_payload(scales)
    q_payload = b"".join(q_parts)
    fp16_payload = b"".join(fp16_parts)
    fp32_payload = b"".join(fp32_parts)
    raw_payload = (
        mode_code_payload
        + scale_payload
        + q_payload
        + fp16_payload
        + fp32_payload
    )
    header = {
        "schema": DECODER_PAYLOAD_V3_SCHEMA,
        "levels": levels,
        "subbands": list(DECODER_SUBBANDS),
        "kernel_shape": _decoder_kernel_shape_header(model_size),
        "feature_count": feature_count,
        "model_size_config": model_size.as_jsonable(),
        "codec": DECODER_PAYLOAD_MIXED_CODEC,
        "quantizer": "mixed_per_kernel_zero_int2_int4_int8_fp16_fp32",
        "mode_assignment_source": (
            "explicit" if mode_plan is not None else "magnitude_heuristic"
        ),
        "mode_code_bits": 3,
        "mode_codebook": dict(DECODER_PAYLOAD_MIXED_MODE_TO_CODE),
        "mode_histogram": histogram,
        "mode_count": len(mode_codes),
        "mode_code_bytes": len(mode_code_payload),
        "scale_dtype": scale_dtype,
        "scale_count": len(scales),
        "scale_bytes": len(scale_payload),
        "packed_q_bytes": len(q_payload),
        "fp16_value_bytes": len(fp16_payload),
        "fp32_value_bytes": len(fp32_payload),
        "value_count": int(values.size),
        "raw_reference_sha256": _sha256(raw_reference),
        "raw_reference_bytes": len(raw_reference),
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(np.mean(mean_abs_errors)) if mean_abs_errors else 0.0,
        "payload_sha256": _sha256(raw_payload),
        "payload_bytes": len(raw_payload),
    }
    return _pack_subpacket(SNERV_DECODER_MAGIC, header, raw_payload)


def decode_decoder_payload(payload: bytes) -> HfGenerationDecoder:
    """Decode the shared HF decoder from receiver payload bytes."""

    header, compressed = _unpack_subpacket(
        payload,
        magic=SNERV_DECODER_MAGIC,
        schema=(
            DECODER_PAYLOAD_V1_SCHEMA,
            DECODER_PAYLOAD_V2_SCHEMA,
            DECODER_PAYLOAD_V3_SCHEMA,
            DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
        ),
    )
    if header["schema"] == DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA:
        raise SnervArchiveError(
            "official MFU/HFR/TUB payload requires "
            "decode_official_mfu_hfr_tub_decoder_payload"
        )
    levels = int(header["levels"])
    if header["schema"] == DECODER_PAYLOAD_V3_SCHEMA:
        return _decode_decoder_payload_mixed(header, compressed)
    if header["schema"] == DECODER_PAYLOAD_V2_SCHEMA:
        return _decode_decoder_payload_quantized(header, compressed)
    raw = lzma.decompress(compressed)
    if len(raw) != int(header["raw_bytes"]):
        raise SnervArchiveError("decoder payload raw byte count mismatch")
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervArchiveError("decoder payload raw sha256 mismatch")
    values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    model_size = _model_size_from_decoder_header(header)
    values = _strip_legacy_decoder_output_affine_tail(
        header=header,
        values=values,
        levels=levels,
        model_size=model_size,
    )
    return _decoder_from_flat_values(
        levels=levels,
        values=values,
        model_size=model_size,
    )


def inspect_decoder_payload_header(payload: bytes) -> dict[str, Any]:
    """Return validated decoder payload header metadata without decoding weights."""

    header, _payload = _unpack_subpacket(
        payload,
        magic=SNERV_DECODER_MAGIC,
        schema=(
            DECODER_PAYLOAD_V1_SCHEMA,
            DECODER_PAYLOAD_V2_SCHEMA,
            DECODER_PAYLOAD_V3_SCHEMA,
            DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
        ),
    )
    return dict(header)


def is_official_mfu_hfr_tub_decoder_payload(payload: bytes) -> bool:
    """Return true when ``payload`` is the official primitive decoder schema."""

    try:
        header = inspect_decoder_payload_header(payload)
    except SnervArchiveError:
        return False
    return header.get("schema") == DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA


def decode_official_mfu_hfr_tub_decoder_payload(
    payload: bytes,
) -> OfficialMfuHfrTubReceiverPayload:
    """Decode official MFU/HFR/TUB receiver primitive payload bytes."""

    header, compressed = _unpack_subpacket(
        payload,
        magic=SNERV_DECODER_MAGIC,
        schema=DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
    )
    if str(header.get("codec")) != DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_CODEC:
        raise SnervArchiveError(
            f"unsupported official primitive payload codec: {header.get('codec')!r}"
        )
    if _sha256(compressed) != str(header["compressed_sha256"]):
        raise SnervArchiveError("official primitive payload compressed sha256 mismatch")
    try:
        raw = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise SnervArchiveError("official primitive payload decompression failed") from exc
    if len(raw) != int(header["raw_tensor_bytes"]):
        raise SnervArchiveError("official primitive payload raw byte count mismatch")
    if _sha256(raw) != str(header["raw_tensor_sha256"]):
        raise SnervArchiveError("official primitive payload raw sha256 mismatch")
    tensors = _unpack_tensor_manifest(raw, header.get("tensor_manifest") or [])
    payload_obj = OfficialMfuHfrTubReceiverPayload(
        header=dict(header),
        tensors=tensors,
        payload_sha256=_sha256(bytes(payload)),
        payload_bytes=len(bytes(payload)),
    )
    _validate_official_payload_exec_surfaces(payload_obj)
    return payload_obj


def execute_official_mfu_hfr_tub_decoder_payload(payload: bytes) -> dict[str, Any]:
    """Decode and execute official MFU/HFR/TUB primitive bytes."""

    return decode_official_mfu_hfr_tub_decoder_payload(payload).execute()


def _official_payload_tensor_dict(
    *,
    mfu: OfficialSnervMfu,
    hfr_heads: OfficialHfrHeads,
    low: np.ndarray,
    skip_mid: np.ndarray,
    skip_high: np.ndarray,
    tub_current: np.ndarray,
    tub_previous: np.ndarray,
    tub_next_frame: np.ndarray,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "mfu.upsample_mid.weight": mfu.upsample_mid.weight,
        "mfu.upsample_high.weight": mfu.upsample_high.weight,
        "inputs.mfu.low": low,
        "inputs.mfu.skip_mid": skip_mid,
        "inputs.mfu.skip_high": skip_high,
        "inputs.tub.current": tub_current,
        "inputs.tub.previous": tub_previous,
        "inputs.tub.next_frame": tub_next_frame,
    }
    _store_optional_bias(
        tensors,
        "mfu.upsample_mid.bias",
        mfu.upsample_mid.bias,
    )
    _store_optional_bias(
        tensors,
        "mfu.upsample_high.bias",
        mfu.upsample_high.bias,
    )
    _store_rb_tensors(tensors, "mfu.rb_mid", mfu.rb_mid)
    _store_rb_tensors(tensors, "mfu.rb_high", mfu.rb_high)
    _store_hfr_head_tensors(tensors, "hfr.lh", hfr_heads.lh_head)
    _store_hfr_head_tensors(tensors, "hfr.hl", hfr_heads.hl_head)
    _store_hfr_head_tensors(tensors, "hfr.hh", hfr_heads.hh_head)
    return tensors


def _store_optional_bias(
    tensors: dict[str, np.ndarray],
    key: str,
    value: np.ndarray | None,
) -> None:
    tensors[key] = (
        np.zeros((0,), dtype=np.float64)
        if value is None
        else np.asarray(value, dtype=np.float64)
    )


def _store_conv2d_tensors(
    tensors: dict[str, np.ndarray],
    prefix: str,
    conv: OfficialConv2dNchw,
) -> None:
    tensors[f"{prefix}.weight"] = conv.weight
    _store_optional_bias(tensors, f"{prefix}.bias", conv.bias)


def _store_rb_tensors(
    tensors: dict[str, np.ndarray],
    prefix: str,
    block: OfficialResidualBlocksWithInputConv,
) -> None:
    _store_conv2d_tensors(tensors, f"{prefix}.input_conv", block.input_conv)
    for idx, residual in enumerate(block.residual_blocks):
        _store_conv2d_tensors(tensors, f"{prefix}.block{idx}.conv1", residual.conv1)
        _store_conv2d_tensors(tensors, f"{prefix}.block{idx}.conv2", residual.conv2)


def _store_hfr_head_tensors(
    tensors: dict[str, np.ndarray],
    prefix: str,
    head: OfficialHfrConvBlock,
) -> None:
    _store_conv2d_tensors(tensors, f"{prefix}.conv1", head.conv1)
    _store_conv2d_tensors(tensors, f"{prefix}.conv2", head.conv2)


def _pack_tensor_manifest(
    tensors: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], bytes]:
    rows: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        arr = np.asarray(tensors[name], dtype="<f8")
        if not np.all(np.isfinite(arr)):
            raise SnervArchiveError(f"official primitive tensor {name!r} is non-finite")
        contiguous = np.ascontiguousarray(arr, dtype="<f8")
        blob = contiguous.tobytes()
        rows.append(
            {
                "name": str(name),
                "dtype": "float64_le",
                "shape": [int(v) for v in contiguous.shape],
                "offset": int(offset),
                "nbytes": len(blob),
                "sha256": _sha256(blob),
            }
        )
        chunks.append(blob)
        offset += len(blob)
    return rows, b"".join(chunks)


def _unpack_tensor_manifest(
    raw: bytes,
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    cursor = 0
    for row in manifest:
        name = str(row.get("name") or "")
        if not name:
            raise SnervArchiveError("official primitive tensor manifest missing name")
        if name in tensors:
            raise SnervArchiveError(f"duplicate official primitive tensor {name!r}")
        if row.get("dtype") != "float64_le":
            raise SnervArchiveError(f"unsupported official primitive dtype for {name!r}")
        offset = int(row.get("offset", -1))
        nbytes = int(row.get("nbytes", -1))
        if offset != cursor or nbytes < 0:
            raise SnervArchiveError("official primitive tensor manifest is noncontiguous")
        blob = raw[offset : offset + nbytes]
        if len(blob) != nbytes:
            raise SnervArchiveError(f"official primitive tensor {name!r} truncated")
        if _sha256(blob) != str(row.get("sha256")):
            raise SnervArchiveError(f"official primitive tensor {name!r} sha256 mismatch")
        shape = tuple(int(v) for v in row.get("shape") or ())
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype("<f8").itemsize
        if expected != nbytes:
            raise SnervArchiveError(f"official primitive tensor {name!r} shape/byte mismatch")
        tensors[name] = np.frombuffer(blob, dtype="<f8").reshape(shape).copy()
        cursor += nbytes
    if cursor != len(raw):
        raise SnervArchiveError("official primitive tensor manifest left trailing bytes")
    return tensors


def _official_mfu_spec_from_header(header: Mapping[str, Any]) -> OfficialSnervMfuSpec:
    raw = header.get("mfu_spec")
    if not isinstance(raw, Mapping):
        raise SnervArchiveError("official primitive payload missing mfu_spec")
    return OfficialSnervMfuSpec(
        low_channels=int(raw["low_channels"]),
        mid_channels=int(raw["mid_channels"]),
        high_channels=int(raw["high_channels"]),
        mid_stride=int(raw["mid_stride"]),
        high_stride=int(raw["high_stride"]),
        num_blocks=int(raw["num_blocks"]),
        source=str(raw.get("source", OFFICIAL_SNERV_MFU_SOURCE)),
    )


def _official_rb_from_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    prefix: str,
    num_blocks: int,
) -> OfficialResidualBlocksWithInputConv:
    return OfficialResidualBlocksWithInputConv(
        input_conv=OfficialConv2dNchw(
            tensors[f"{prefix}.input_conv.weight"],
            _optional_tensor(tensors, f"{prefix}.input_conv.bias"),
            padding=1,
        ),
        residual_blocks=tuple(
            OfficialResidualBlockNoBN(
                conv1=OfficialConv2dNchw(
                    tensors[f"{prefix}.block{idx}.conv1.weight"],
                    _optional_tensor(tensors, f"{prefix}.block{idx}.conv1.bias"),
                    padding=1,
                ),
                conv2=OfficialConv2dNchw(
                    tensors[f"{prefix}.block{idx}.conv2.weight"],
                    _optional_tensor(tensors, f"{prefix}.block{idx}.conv2.bias"),
                    padding=1,
                ),
            )
            for idx in range(int(num_blocks))
        ),
    )


def _official_hfr_head_from_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    prefix: str,
) -> OfficialHfrConvBlock:
    return OfficialHfrConvBlock(
        conv1=OfficialConv2dNchw(
            tensors[f"{prefix}.conv1.weight"],
            _optional_tensor(tensors, f"{prefix}.conv1.bias"),
            padding=0,
        ),
        conv2=OfficialConv2dNchw(
            tensors[f"{prefix}.conv2.weight"],
            _optional_tensor(tensors, f"{prefix}.conv2.bias"),
            padding=1,
        ),
    )


def _optional_tensor(
    tensors: Mapping[str, np.ndarray],
    key: str,
) -> np.ndarray | None:
    arr = np.asarray(tensors[key], dtype=np.float64)
    return None if arr.size == 0 else arr


def _validate_official_payload_exec_surfaces(
    payload: OfficialMfuHfrTubReceiverPayload,
) -> None:
    required = {
        "inputs.mfu.low",
        "inputs.mfu.skip_mid",
        "inputs.mfu.skip_high",
        "inputs.tub.current",
        "inputs.tub.previous",
        "inputs.tub.next_frame",
    }
    missing = sorted(required.difference(payload.tensors))
    if missing:
        raise SnervArchiveError(
            "official primitive payload missing tensors: " + ", ".join(missing)
        )
    payload.build_mfu()
    payload.build_hfr_heads()
    payload.mfu_inputs()
    payload.tub_inputs()


def _tensor_manifest_row(name: str, array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    blob = arr.tobytes()
    return {
        "name": str(name),
        "dtype": "float64_le",
        "shape": [int(v) for v in arr.shape],
        "bytes": len(blob),
        "sha256": _sha256(blob),
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
    }


def _decode_decoder_payload_mixed(
    header: dict[str, Any],
    payload: bytes,
) -> HfGenerationDecoder:
    if _sha256(payload) != str(header["payload_sha256"]):
        raise SnervArchiveError("decoder mixed payload sha256 mismatch")
    levels = int(header["levels"])
    model_size = _model_size_from_decoder_header(header)
    feature_count = int(model_size.feature_count)
    group_count = levels * len(DECODER_SUBBANDS)
    if int(header["mode_count"]) != group_count:
        raise SnervArchiveError("decoder mixed mode count mismatch")
    mode_code_bytes = int(header["mode_code_bytes"])
    scale_bytes = int(header["scale_bytes"])
    scale_dtype = _decoder_scale_dtype(header)
    scale_count = int(header.get("scale_count", 0))
    if scale_bytes != scale_count * scale_dtype.itemsize:
        raise SnervArchiveError("decoder mixed scale byte count mismatch")
    q_bytes = int(header["packed_q_bytes"])
    fp16_bytes = int(header["fp16_value_bytes"])
    fp32_bytes = int(header.get("fp32_value_bytes", 0))
    expected_payload_bytes = (
        mode_code_bytes + scale_bytes + q_bytes + fp16_bytes + fp32_bytes
    )
    if len(payload) != expected_payload_bytes:
        raise SnervArchiveError("decoder mixed payload byte count mismatch")
    mode_code_payload = payload[:mode_code_bytes]
    scale_payload = payload[mode_code_bytes : mode_code_bytes + scale_bytes]
    q_payload = payload[
        mode_code_bytes + scale_bytes : mode_code_bytes + scale_bytes + q_bytes
    ]
    float_payload = payload[mode_code_bytes + scale_bytes + q_bytes :]
    fp16_payload = float_payload[:fp16_bytes]
    fp32_payload = float_payload[fp16_bytes : fp16_bytes + fp32_bytes]
    mode_codes = unpack_fixed_width_uints(
        mode_code_payload,
        bits=int(header["mode_code_bits"]),
        count=group_count,
    )
    scales = np.frombuffer(scale_payload, dtype=scale_dtype).astype(np.float64)
    if not np.all(np.isfinite(scales)):
        raise SnervArchiveError("decoder mixed scale payload contains non-finite values")
    values = np.zeros(int(header["value_count"]), dtype=np.float64)
    scale_cursor = 0
    q_cursor = 0
    fp16_cursor = 0
    fp32_cursor = 0
    for group_idx, raw_code in enumerate(mode_codes.tolist()):
        mode = DECODER_PAYLOAD_MIXED_CODE_TO_MODE.get(int(raw_code))
        if mode is None:
            raise SnervArchiveError(f"unknown decoder mixed mode code: {raw_code}")
        start = group_idx * feature_count
        stop = start + feature_count
        if mode == "zero":
            continue
        if mode == "fp16":
            nbytes = feature_count * np.dtype("<f2").itemsize
            segment = fp16_payload[fp16_cursor : fp16_cursor + nbytes]
            if len(segment) != nbytes:
                raise SnervArchiveError("decoder mixed fp16 payload truncated")
            values[start:stop] = np.frombuffer(segment, dtype="<f2").astype(np.float64)
            fp16_cursor += nbytes
            continue
        if mode == "fp32":
            nbytes = feature_count * np.dtype("<f4").itemsize
            segment = fp32_payload[fp32_cursor : fp32_cursor + nbytes]
            if len(segment) != nbytes:
                raise SnervArchiveError("decoder mixed fp32 payload truncated")
            values[start:stop] = np.frombuffer(segment, dtype="<f4").astype(np.float64)
            fp32_cursor += nbytes
            continue
        bits = int(mode.removeprefix("int"))
        qmax = (1 << (bits - 1)) - 1
        nbytes = (feature_count * bits + 7) // 8
        segment = q_payload[q_cursor : q_cursor + nbytes]
        if len(segment) != nbytes:
            raise SnervArchiveError("decoder mixed q payload truncated")
        if scale_cursor >= scales.size:
            raise SnervArchiveError("decoder mixed scale payload truncated")
        q_unsigned = unpack_fixed_width_uints(
            segment,
            bits=bits,
            count=feature_count,
        )
        q_signed = q_unsigned.astype(np.int64) - qmax
        values[start:stop] = q_signed.astype(np.float64) * float(scales[scale_cursor])
        scale_cursor += 1
        q_cursor += nbytes
    if scale_cursor != scales.size:
        raise SnervArchiveError("decoder mixed payload has unused scales")
    if q_cursor != len(q_payload):
        raise SnervArchiveError("decoder mixed payload has unused q bytes")
    if fp16_cursor != len(fp16_payload):
        raise SnervArchiveError("decoder mixed payload has unused fp16 bytes")
    if fp32_cursor != len(fp32_payload):
        raise SnervArchiveError("decoder mixed payload has unused fp32 bytes")
    return _decoder_from_flat_values(
        levels=levels,
        values=values,
        model_size=model_size,
    )


def _select_mixed_decoder_kernel_mode(group: np.ndarray) -> str:
    max_abs = float(np.max(np.abs(group))) if group.size else 0.0
    if max_abs <= 1e-12:
        return "zero"
    if max_abs > DECODER_FP16_MAX_FINITE:
        return "fp32"
    if max_abs >= 0.125:
        return "fp16"
    if max_abs >= 0.05:
        return "int8"
    if max_abs >= 0.015:
        return "int4"
    return "int2"


def _normalize_mixed_decoder_kernel_mode(raw: str) -> str:
    mode = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "0": "zero",
        "none": "zero",
        "z": "zero",
        "i2": "int2",
        "2": "int2",
        "int2_symmetric": "int2",
        "i4": "int4",
        "4": "int4",
        "int4_symmetric": "int4",
        "i8": "int8",
        "8": "int8",
        "int8_symmetric": "int8",
        "float16": "fp16",
        "f16": "fp16",
        "half": "fp16",
        "float32": "fp32",
        "f32": "fp32",
        "single": "fp32",
        "fp32_protect": "fp32",
    }
    mode = aliases.get(mode, mode)
    if mode not in DECODER_PAYLOAD_MIXED_MODE_TO_CODE:
        raise SnervArchiveError(f"unsupported decoder mixed mode: {raw!r}")
    return mode


def _decode_decoder_payload_quantized(
    header: dict[str, Any],
    payload: bytes,
) -> HfGenerationDecoder:
    if _sha256(payload) != str(header["payload_sha256"]):
        raise SnervArchiveError("decoder quantized payload sha256 mismatch")
    levels = int(header["levels"])
    model_size = _model_size_from_decoder_header(header)
    feature_count = int(model_size.feature_count)
    bits = int(header["bits_per_weight"])
    offset = int(header["q_offset"])
    scale_bytes = int(header["scale_bytes"])
    scale_count = int(header["scale_count"])
    value_count = int(header["value_count"])
    if scale_count != levels * len(DECODER_SUBBANDS):
        raise SnervArchiveError("decoder quantized scale count mismatch")
    scale_dtype = _decoder_scale_dtype(header)
    if scale_bytes != scale_count * scale_dtype.itemsize:
        raise SnervArchiveError("decoder quantized scale byte count mismatch")
    if len(payload) < scale_bytes:
        raise SnervArchiveError("decoder quantized payload too short")
    scale_payload = payload[:scale_bytes]
    packed_q = payload[scale_bytes:]
    if len(packed_q) != int(header["packed_q_bytes"]):
        raise SnervArchiveError("decoder quantized q byte count mismatch")
    scales = np.frombuffer(scale_payload, dtype=scale_dtype).astype(np.float64)
    if not np.all(np.isfinite(scales)):
        raise SnervArchiveError(
            "decoder quantized scale payload contains non-finite values"
        )
    q_unsigned = unpack_fixed_width_uints(packed_q, bits=bits, count=value_count)
    q_signed = q_unsigned.astype(np.int64) - offset
    values = q_signed.astype(np.float64)
    for idx, scale in enumerate(scales):
        start = idx * feature_count
        stop = start + feature_count
        values[start:stop] *= float(scale)
    return _decoder_from_flat_values(
        levels=levels,
        values=values,
        model_size=model_size,
    )


def _official_payload_tensor_dict(
    *,
    mfu: OfficialSnervMfu,
    hfr_heads: OfficialHfrHeads,
    low: np.ndarray,
    skip_mid: np.ndarray,
    skip_high: np.ndarray,
    tub_current: np.ndarray,
    tub_previous: np.ndarray,
    tub_next_frame: np.ndarray,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "mfu.upsample_mid.weight": mfu.upsample_mid.weight,
        "mfu.upsample_mid.bias": _required_bias(mfu.upsample_mid.bias, "mfu.upsample_mid.bias"),
        "mfu.upsample_high.weight": mfu.upsample_high.weight,
        "mfu.upsample_high.bias": _required_bias(mfu.upsample_high.bias, "mfu.upsample_high.bias"),
        "inputs.mfu.low": low,
        "inputs.mfu.skip_mid": skip_mid,
        "inputs.mfu.skip_high": skip_high,
        "inputs.tub.current": tub_current,
        "inputs.tub.previous": tub_previous,
        "inputs.tub.next_frame": tub_next_frame,
    }
    tensors.update(_official_rb_to_tensors(mfu.rb_mid, prefix="mfu.rb_mid"))
    tensors.update(_official_rb_to_tensors(mfu.rb_high, prefix="mfu.rb_high"))
    tensors.update(_official_hfr_heads_to_tensors(hfr_heads))
    return tensors


def _official_rb_to_tensors(
    rb: OfficialResidualBlocksWithInputConv,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    tensors = {
        f"{prefix}.input_conv.weight": rb.input_conv.weight,
        f"{prefix}.input_conv.bias": _required_bias(
            rb.input_conv.bias,
            f"{prefix}.input_conv.bias",
        ),
    }
    for idx, block in enumerate(rb.residual_blocks):
        base = f"{prefix}.block{idx}"
        tensors[f"{base}.conv1.weight"] = block.conv1.weight
        tensors[f"{base}.conv1.bias"] = _required_bias(
            block.conv1.bias,
            f"{base}.conv1.bias",
        )
        tensors[f"{base}.conv2.weight"] = block.conv2.weight
        tensors[f"{base}.conv2.bias"] = _required_bias(
            block.conv2.bias,
            f"{base}.conv2.bias",
        )
    return tensors


def _official_hfr_heads_to_tensors(heads: OfficialHfrHeads) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for name, head in (
        ("lh", heads.lh_head),
        ("hl", heads.hl_head),
        ("hh", heads.hh_head),
    ):
        base = f"hfr.{name}"
        tensors[f"{base}.conv1.weight"] = head.conv1.weight
        tensors[f"{base}.conv1.bias"] = _required_bias(
            head.conv1.bias,
            f"{base}.conv1.bias",
        )
        tensors[f"{base}.conv2.weight"] = head.conv2.weight
        tensors[f"{base}.conv2.bias"] = _required_bias(
            head.conv2.bias,
            f"{base}.conv2.bias",
        )
    return tensors


def _required_bias(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise SnervArchiveError(f"official primitive payload missing required bias {name!r}")
    return value


def _pack_tensor_manifest(
    tensors: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], bytes]:
    if not tensors:
        raise SnervArchiveError("official primitive payload requires tensors")
    manifest = []
    raw_parts = []
    for name in sorted(tensors):
        arr = _canonical_float64_tensor(tensors[name], name=name)
        blob = arr.tobytes()
        manifest.append(_tensor_manifest_row(name, arr))
        raw_parts.append(blob)
    return manifest, b"".join(raw_parts)


def _unpack_tensor_manifest(
    raw: bytes,
    manifest: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    if not manifest:
        raise SnervArchiveError("official primitive tensor manifest is empty")
    cursor = 0
    seen: set[str] = set()
    out: dict[str, np.ndarray] = {}
    for row in manifest:
        name = str(row.get("name"))
        if not name:
            raise SnervArchiveError("official primitive tensor manifest missing name")
        if name in seen:
            raise SnervArchiveError(f"duplicate official primitive tensor {name!r}")
        shape = tuple(int(v) for v in row.get("shape", ()))
        if not shape or any(v <= 0 for v in shape):
            raise SnervArchiveError(f"official primitive tensor {name!r} has bad shape")
        if str(row.get("dtype")) != "float64_le":
            raise SnervArchiveError(
                f"official primitive tensor {name!r} has unsupported dtype"
            )
        nbytes = int(np.prod(shape)) * np.dtype("<f8").itemsize
        if int(row.get("bytes", -1)) != nbytes:
            raise SnervArchiveError(
                f"official primitive tensor {name!r} byte count mismatch"
            )
        segment = raw[cursor : cursor + nbytes]
        if len(segment) != nbytes:
            raise SnervArchiveError(f"official primitive tensor {name!r} is truncated")
        if _sha256(segment) != str(row.get("sha256")):
            raise SnervArchiveError(
                f"official primitive tensor {name!r} sha256 mismatch"
            )
        arr = np.frombuffer(segment, dtype="<f8").copy().reshape(shape)
        if not np.all(np.isfinite(arr)):
            raise SnervArchiveError(
                f"official primitive tensor {name!r} contains non-finite values"
            )
        out[name] = arr
        seen.add(name)
        cursor += nbytes
    if cursor != len(raw):
        raise SnervArchiveError("official primitive payload has unused raw tensor bytes")
    return out


def _tensor_manifest_row(name: str, arr: np.ndarray) -> dict[str, Any]:
    canonical = _canonical_float64_tensor(arr, name=name)
    blob = canonical.tobytes()
    return {
        "name": str(name),
        "shape": [int(v) for v in canonical.shape],
        "dtype": "float64_le",
        "bytes": len(blob),
        "sha256": _sha256(blob),
    }


def _canonical_float64_tensor(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype="<f8")
    if arr.size == 0:
        raise SnervArchiveError(f"official primitive tensor {name!r} is empty")
    if not np.all(np.isfinite(arr)):
        raise SnervArchiveError(
            f"official primitive tensor {name!r} contains non-finite values"
        )
    return np.ascontiguousarray(arr, dtype="<f8")


def _official_mfu_spec_from_header(header: dict[str, Any]) -> OfficialSnervMfuSpec:
    raw = header.get("mfu_spec")
    if not isinstance(raw, dict):
        raise SnervArchiveError("official primitive payload missing mfu_spec")
    return OfficialSnervMfuSpec(
        low_channels=int(raw["low_channels"]),
        mid_channels=int(raw["mid_channels"]),
        high_channels=int(raw["high_channels"]),
        mid_stride=int(raw["mid_stride"]),
        high_stride=int(raw["high_stride"]),
        num_blocks=int(raw["num_blocks"]),
        source=str(raw.get("source") or OFFICIAL_SNERV_MFU_SOURCE),
    )


def _official_rb_from_tensors(
    tensors: dict[str, np.ndarray],
    *,
    prefix: str,
    num_blocks: int,
) -> OfficialResidualBlocksWithInputConv:
    return OfficialResidualBlocksWithInputConv(
        input_conv=OfficialConv2dNchw(
            _tensor(tensors, f"{prefix}.input_conv.weight"),
            _tensor(tensors, f"{prefix}.input_conv.bias"),
            padding=1,
        ),
        residual_blocks=tuple(
            OfficialResidualBlockNoBN(
                conv1=OfficialConv2dNchw(
                    _tensor(tensors, f"{prefix}.block{idx}.conv1.weight"),
                    _tensor(tensors, f"{prefix}.block{idx}.conv1.bias"),
                    padding=1,
                ),
                conv2=OfficialConv2dNchw(
                    _tensor(tensors, f"{prefix}.block{idx}.conv2.weight"),
                    _tensor(tensors, f"{prefix}.block{idx}.conv2.bias"),
                    padding=1,
                ),
            )
            for idx in range(int(num_blocks))
        ),
    )


def _official_hfr_head_from_tensors(
    tensors: dict[str, np.ndarray],
    *,
    prefix: str,
) -> OfficialHfrConvBlock:
    return OfficialHfrConvBlock(
        conv1=OfficialConv2dNchw(
            _tensor(tensors, f"{prefix}.conv1.weight"),
            _tensor(tensors, f"{prefix}.conv1.bias"),
            padding=0,
        ),
        conv2=OfficialConv2dNchw(
            _tensor(tensors, f"{prefix}.conv2.weight"),
            _tensor(tensors, f"{prefix}.conv2.bias"),
            padding=1,
        ),
    )


def _tensor(tensors: dict[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return tensors[name]
    except KeyError as exc:
        raise SnervArchiveError(
            f"official primitive payload missing tensor {name!r}"
        ) from exc


def _validate_official_payload_exec_surfaces(
    payload: OfficialMfuHfrTubReceiverPayload,
) -> None:
    payload.build_mfu()
    payload.build_hfr_heads()
    payload.mfu_inputs()
    payload.tub_inputs()
    _official_receiver_self_consistency_reference_from_header(payload.header)


def _official_receiver_self_consistency_reference_from_header(
    header: Mapping[str, Any],
) -> dict[str, Any]:
    reference = header.get("receiver_self_consistency_reference")
    if not isinstance(reference, Mapping):
        raise SnervArchiveError(
            "official primitive payload missing receiver self-consistency reference"
        )
    reference = dict(reference)
    if (
        reference.get("schema")
        != DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SELF_CONSISTENCY_SCHEMA
    ):
        raise SnervArchiveError(
            "official primitive payload receiver self-consistency schema mismatch"
        )
    if reference.get("receiver_export_payload_bound") is not True:
        raise SnervArchiveError(
            "official primitive payload receiver self-consistency export flag missing"
        )
    if reference.get("receiver_export_self_consistency_verified") is not True:
        raise SnervArchiveError(
            "official primitive payload receiver export self-consistency proof missing"
        )
    if reference.get("source_forward_replay_verified_by_export") is not False:
        raise SnervArchiveError(
            "official primitive payload receiver self-consistency must not claim source-forward replay"
        )
    if not _looks_like_sha256(reference.get("output_bundle_sha256")):
        raise SnervArchiveError(
            "official primitive payload receiver self-consistency output sha256 missing"
        )
    output_rows = reference.get("output_tensors")
    if not isinstance(output_rows, Sequence) or isinstance(output_rows, (str, bytes)):
        raise SnervArchiveError(
            "official primitive payload receiver self-consistency tensor manifest missing"
        )
    if not output_rows:
        raise SnervArchiveError(
            "official primitive payload receiver self-consistency tensor manifest is empty"
        )
    for row in output_rows:
        if not isinstance(row, Mapping):
            raise SnervArchiveError(
                "official primitive payload receiver self-consistency tensor row invalid"
            )
        if not str(row.get("name") or ""):
            raise SnervArchiveError(
                "official primitive payload receiver self-consistency tensor row missing name"
            )
        if str(row.get("dtype") or "") != "float64_le":
            raise SnervArchiveError(
                "official primitive payload receiver self-consistency tensor dtype invalid"
            )
        if int(row.get("bytes", 0)) <= 0:
            raise SnervArchiveError(
                "official primitive payload receiver self-consistency tensor bytes invalid"
            )
        if not _looks_like_sha256(row.get("sha256")):
            raise SnervArchiveError(
                "official primitive payload receiver self-consistency tensor sha256 missing"
            )
    expected_reference_sha = header.get("receiver_self_consistency_reference_sha256")
    if not _looks_like_sha256(expected_reference_sha):
        raise SnervArchiveError(
            "official primitive payload receiver self-consistency reference sha256 missing"
        )
    if _json_sha256(reference) != str(expected_reference_sha):
        raise SnervArchiveError(
            "official primitive payload receiver self-consistency reference sha256 mismatch"
        )
    return reference


def _validate_official_receiver_self_consistency_reference(
    header: Mapping[str, Any],
    *,
    output_rows: Sequence[Mapping[str, Any]],
    output_bundle_sha256: str,
) -> dict[str, Any]:
    reference = _official_receiver_self_consistency_reference_from_header(header)
    if reference.get("output_bundle_sha256") != output_bundle_sha256:
        raise SnervArchiveError(
            "official primitive receiver self-consistency output bundle sha256 mismatch"
        )
    if _json_sha256(reference.get("output_tensors")) != _json_sha256(
        list(output_rows)
    ):
        raise SnervArchiveError(
            "official primitive receiver self-consistency tensor manifest mismatch"
        )
    return reference


def _looks_like_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _decoder_to_flat_values(
    decoder: HfGenerationDecoder,
) -> tuple[int, np.ndarray, SnervModelSizeConfig]:
    levels = int(decoder.levels)
    model_size = decoder.model_size
    feature_count = int(model_size.feature_count)
    arrays = []
    for lvl in range(levels):
        level = decoder.kernels.get(lvl)
        if not isinstance(level, dict):
            raise SnervArchiveError(f"decoder missing level {lvl}")
        for subband in DECODER_SUBBANDS:
            kernel = np.asarray(level.get(subband), dtype=np.float64)
            if kernel.size != feature_count:
                raise SnervArchiveError(
                    f"decoder kernel {lvl}/{subband} has {kernel.size} values, "
                    f"expected {feature_count}"
                )
            if not np.all(np.isfinite(kernel)):
                raise SnervArchiveError(f"decoder kernel {lvl}/{subband} is non-finite")
            arrays.append(kernel.reshape(-1))
    values = np.concatenate(arrays).astype(np.float64) if arrays else np.zeros(0)
    return levels, values, model_size


def _decoder_from_flat_values(
    *,
    levels: int,
    values: np.ndarray,
    model_size: SnervModelSizeConfig | None = None,
) -> HfGenerationDecoder:
    model_size = model_size or DEFAULT_SNERV_MODEL_SIZE
    feature_count = int(model_size.feature_count)
    expected = levels * len(DECODER_SUBBANDS) * feature_count
    if values.size != expected:
        raise SnervArchiveError(
            f"decoder payload has {values.size} values, expected {expected}"
        )
    kernels: dict[int, dict[str, np.ndarray]] = {}
    cursor = 0
    for lvl in range(levels):
        kernels[lvl] = {}
        for subband in DECODER_SUBBANDS:
            kernels[lvl][subband] = values[
                cursor : cursor + feature_count
            ].reshape(_decoder_kernel_storage_shape(model_size))
            cursor += feature_count
    return HfGenerationDecoder(
        kernels=kernels,
        levels=levels,
        model_size=model_size,
    )


def _strip_legacy_decoder_output_affine_tail(
    *,
    header: dict[str, Any],
    values: np.ndarray,
    levels: int,
    model_size: SnervModelSizeConfig,
) -> np.ndarray:
    """Trim legacy v1 decoder affine scalars from raw kernel payloads.

    Some older SNAR1 packets encoded kernel weights followed by receiver-visible
    output-affine scalars while declaring those scalars in ``output_affine``.
    The kernel decoder must not treat the affine tail as extra HF weights.
    """

    expected = int(levels) * len(DECODER_SUBBANDS) * int(model_size.feature_count)
    if int(values.size) == expected:
        return values
    output_affine = header.get("output_affine")
    if isinstance(output_affine, dict):
        affine_count = _legacy_output_affine_value_count(output_affine)
        if affine_count > 0 and int(values.size) == expected + affine_count:
            return values[:expected]
    return values


def _legacy_output_affine_value_count(output_affine: dict[str, Any]) -> int:
    count = int(output_affine.get("count", 0))
    mode = str(output_affine.get("mode", "")).strip().lower()
    if mode == "scalar" and {"scale", "bias"}.issubset(output_affine):
        return 2 * max(count, 1)
    return count


def _decoder_kernel_shape_header(model_size: SnervModelSizeConfig) -> list[int]:
    if model_size == DEFAULT_SNERV_MODEL_SIZE:
        return [3, 3]
    return [int(model_size.feature_count)]


def _decoder_kernel_storage_shape(model_size: SnervModelSizeConfig) -> tuple[int, ...]:
    if model_size == DEFAULT_SNERV_MODEL_SIZE:
        return (3, 3)
    return (int(model_size.feature_count),)


def _model_size_from_decoder_header(header: dict[str, Any]) -> SnervModelSizeConfig:
    raw = header.get("model_size_config")
    if isinstance(raw, dict):
        return SnervModelSizeConfig(
            fc_dim=int(raw.get("fc_dim", raw.get("feature_count", 9))),
            emb_size=int(raw.get("emb_size", 0)),
            patch_radius=int(raw.get("patch_radius", 1)),
            mfu_scales=tuple(int(v) for v in raw.get("mfu_scales", (1, 2, 4))),
            hfr_gain=float(raw.get("hfr_gain", 0.0)),
            temporal_context=int(raw.get("temporal_context", 0)),
            temporal_mode=str(raw.get("temporal_mode", "delta")),
            adapter=str(raw.get("adapter", "snerv_fc_dim_emb_size_adapter_v1")),
        )
    feature_count = int(header.get("feature_count", 9))
    if feature_count != 9:
        return SnervModelSizeConfig(
            fc_dim=feature_count,
            emb_size=0,
            patch_radius=1,
        )
    return DEFAULT_SNERV_MODEL_SIZE


def decode_lf_metadata_payload(
    payload: bytes,
    *,
    expected_count: int | None = None,
) -> np.ndarray:
    """Decode compact LF zero-point metadata payload."""

    if len(payload) % 4:
        raise SnervArchiveError("LF metadata payload byte count is not float32-aligned")
    zeros = np.frombuffer(payload, dtype="<f4").copy()
    if expected_count is not None and zeros.size != expected_count:
        raise SnervArchiveError(
            f"decoded {zeros.size} LF zero-points, expected {expected_count}"
        )
    if zeros.size == 0:
        raise SnervArchiveError("LF metadata payload is empty")
    if not np.all(np.isfinite(zeros)):
        raise SnervArchiveError("LF metadata payload contains non-finite values")
    return zeros


def _validate_sections(sections: dict[str, bytes]) -> None:
    for name in SECTION_ORDER:
        blob = sections.get(name)
        if not isinstance(blob, bytes) or not blob:
            raise SnervArchiveError(f"section {name!r} must be non-empty bytes")


def _validate_lf_quant_plane(plane: np.ndarray) -> np.ndarray:
    arr = np.asarray(plane)
    if arr.size == 0:
        raise SnervArchiveError("LF quant planes must be non-empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise SnervArchiveError("LF quant planes must contain integers")
    return arr.astype("<i8", copy=False)


def _validate_replay_counts(
    lf_planes: list[np.ndarray],
    zeros: np.ndarray,
    step_maps: list[np.ndarray],
) -> None:
    if len(lf_planes) != int(zeros.size):
        raise SnervArchiveError(
            f"receiver replay LF plane count {len(lf_planes)} != zero-point count {zeros.size}"
        )
    if len(lf_planes) != len(step_maps):
        raise SnervArchiveError(
            f"receiver replay LF plane count {len(lf_planes)} != step-map count {len(step_maps)}"
        )
    if not lf_planes:
        raise SnervArchiveError("receiver replay requires at least one LF plane")


def _metadata_int(
    metadata: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    if key not in metadata:
        if default is None:
            raise SnervArchiveError(f"receiver replay metadata missing {key!r}")
        value = default
    else:
        value = metadata[key]
    if isinstance(value, bool):
        raise SnervArchiveError(f"receiver replay metadata {key!r} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise SnervArchiveError(
            f"receiver replay metadata {key!r} must be an integer"
        ) from exc
    if minimum is not None and out < minimum:
        raise SnervArchiveError(
            f"receiver replay metadata {key!r}={out} must be >= {minimum}"
        )
    return out


def _metadata_str(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise SnervArchiveError(f"receiver replay metadata missing string {key!r}")
    return value


def _metadata_hw(metadata: dict[str, Any]) -> tuple[int, int]:
    value = metadata.get("carrier_hw", metadata.get("orig_hw"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SnervArchiveError(
            "receiver replay metadata missing 2-element 'carrier_hw'/'orig_hw'"
        )
    h, w = int(value[0]), int(value[1])
    if h <= 0 or w <= 0:
        raise SnervArchiveError("receiver replay metadata height/width must be positive")
    return h, w


def _pack_subpacket(magic: bytes, header: dict[str, Any], payload: bytes) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return magic + struct.pack(HEADER_LEN_FMT, len(header_bytes)) + header_bytes + payload


def _unpack_subpacket(
    packet: bytes,
    *,
    magic: bytes,
    schema: str | tuple[str, ...],
) -> tuple[dict[str, Any], bytes]:
    packet = bytes(packet)
    if not packet.startswith(magic):
        raise SnervArchiveError(f"bad subpacket magic for {schema}")
    offset = len(magic)
    if len(packet) < offset + struct.calcsize(HEADER_LEN_FMT):
        raise SnervArchiveError(f"truncated subpacket header for {schema}")
    (header_len,) = struct.unpack(
        HEADER_LEN_FMT,
        packet[offset : offset + struct.calcsize(HEADER_LEN_FMT)],
    )
    offset += struct.calcsize(HEADER_LEN_FMT)
    header_end = offset + header_len
    if header_end > len(packet):
        raise SnervArchiveError(f"declared subpacket header exceeds bytes for {schema}")
    header = json.loads(packet[offset:header_end].decode("utf-8"))
    allowed = (schema,) if isinstance(schema, str) else tuple(schema)
    if header.get("schema") not in allowed:
        raise SnervArchiveError(f"unsupported subpacket schema: {header.get('schema')!r}")
    return dict(header), packet[header_end:]


def _unpack_lf_quant_subpacket(packet: bytes) -> tuple[dict[str, Any], bytes]:
    return _unpack_subpacket(
        packet,
        magic=SNERV_LF_QUANT_MAGIC,
        schema=(LF_QUANT_PAYLOAD_SCHEMA_V1, LF_QUANT_PAYLOAD_SCHEMA_V2),
    )


def _jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(metadata, sort_keys=True)
    except TypeError as exc:
        raise SnervArchiveError("metadata must be JSON-serializable") from exc
    return dict(metadata)


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
