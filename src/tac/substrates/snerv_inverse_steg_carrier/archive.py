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
    OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS,
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
    OFFICIAL_SNERV_T_TUB_SOURCE_FORWARD_BLOCKERS,
    official_output2_fusion_numpy,
    prepare_official_tub_graph_inputs,
)

SNERV_ARCHIVE_SCHEMA = "snerv_inverse_steg_archive.v1"
SNERV_ARCHIVE_SCHEMA_V2 = "snerv_inverse_steg_archive.snar2.v1"
SNERV_ARCHIVE_MAGIC = b"SNAR1"
SNERV_ARCHIVE_MAGIC_V2 = b"SNAR2"
SNERV_LF_QUANT_MAGIC = b"SNQL1"
SNERV_DECODER_MAGIC = b"SNDC1"
SNERV_LF_PAYLOAD_INTN_CODEC_PROOF = _SNERV_LF_PAYLOAD_INTN_CODEC_PROOF
HEADER_LEN_FMT = "<I"
SECTION_ORDER = ("metadata_payload", "lf_payload", "decoder_payload", "step_map_packet")
SNAR2_VERSION = 1
SNAR2_SECTION_HASH_BYTES = 8
_SNAR2_HEADER_FMT = "<5sBBBBHBBIBBHH4I4Q"
SNAR2_HEADER_BYTES = struct.calcsize(_SNAR2_HEADER_FMT)
_SNAR2_WAVELET_TO_CODE = {
    "haar": 1,
    "db1": 1,
    "db2": 2,
}
_SNAR2_CODE_TO_WAVELET = {
    1: "haar",
    2: "db2",
}
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
OFFICIAL_SKIP_HIGH_CODEC_FULL = "full_float64"
OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN = "shared_mean_float64"
OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN = "channel_mean_float64"
OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN = "scalar_mean_float64"
OFFICIAL_TUB_INPUT_CODEC_FULL = "full_float64"
OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC = "unused_synthetic_float64"
OFFICIAL_MFU_HFR_TUB_BASE_SOURCE_FORWARD_BLOCKERS: tuple[str, ...] = (
    *OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS,
    "official_hfr_weight_tensor_mapping_not_loaded",
    "full_official_hfr_forward_artifact_not_emitted",
    *OFFICIAL_SNERV_T_TUB_SOURCE_FORWARD_BLOCKERS,
)
OFFICIAL_SKIP_HIGH_MODE_TO_CODEC = {
    "full": OFFICIAL_SKIP_HIGH_CODEC_FULL,
    "full_float64": OFFICIAL_SKIP_HIGH_CODEC_FULL,
    OFFICIAL_SKIP_HIGH_CODEC_FULL: OFFICIAL_SKIP_HIGH_CODEC_FULL,
    "shared": OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN,
    "shared_mean": OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN,
    OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN: OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN,
    "channel": OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN,
    "channel_mean": OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN,
    OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN: OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN,
    "scalar": OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN,
    "scalar_mean": OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN,
    "global_mean": OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN,
    OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN: OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN,
}
OFFICIAL_TUB_INPUT_MODE_TO_CODEC = {
    "full": OFFICIAL_TUB_INPUT_CODEC_FULL,
    "full_float64": OFFICIAL_TUB_INPUT_CODEC_FULL,
    OFFICIAL_TUB_INPUT_CODEC_FULL: OFFICIAL_TUB_INPUT_CODEC_FULL,
    "unused_zero": OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC,
    "unused_zeros": OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC,
    "unused_synthetic": OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC,
    "unused_synthetic_float64": OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC,
    OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC: OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC,
}
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
DECODER_PAYLOAD_AUTO_ALIASES = frozenset({"auto", "portfolio", "portfolio_auto"})
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


def resolve_decoder_payload_codec(codec: str | None) -> str:
    """Resolve launch-time decoder codec controls to a receiver codec.

    LF payloads have a true portfolio codec.  Decoder payloads currently expose
    the mixed per-kernel codec as the receiver-safe adaptive policy, so generic
    launch controls such as ``portfolio_auto`` must resolve before metadata and
    bytes are packed.
    """

    normalized = str(DECODER_PAYLOAD_LEGACY_CODEC if codec is None else codec).strip().lower()
    if normalized in DECODER_PAYLOAD_AUTO_ALIASES:
        return DECODER_PAYLOAD_MIXED_CODEC
    if normalized in {DECODER_PAYLOAD_LEGACY_CODEC, "fp32_lzma", "float32", "legacy"}:
        return DECODER_PAYLOAD_LEGACY_CODEC
    if normalized in DECODER_PAYLOAD_QUANTIZED_CODECS:
        return normalized
    if normalized in {DECODER_PAYLOAD_MIXED_CODEC, "mixed_per_kernel_symmetric", "mixed_symmetric"}:
        return DECODER_PAYLOAD_MIXED_CODEC
    raise SnervArchiveError(f"unsupported decoder payload codec: {codec!r}")


@dataclass(frozen=True)
class SnervArchivePacket:
    """A bundled receiver-visible SNeRV archive packet."""

    packet: bytes
    schema: str
    section_order: tuple[str, ...]
    section_bytes: dict[str, int]
    section_sha256: dict[str, str]
    section_reports: dict[str, Any]
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

    def decode_frame_planes(
        self,
        *,
        clip_to_uint8_range: bool = True,
        frame_plane_indices: Sequence[int] | None = None,
    ) -> list[np.ndarray]:
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
            frame_plane_indices=frame_plane_indices,
        )

    def decode_frames(self, *, clip_to_uint8_range: bool = True) -> np.ndarray:
        """Decode a full receiver frame tensor ``(pairs, 2, 3, H, W)`` from SNAR1."""

        return decode_snerv_archive_frames_from_decoded(
            self,
            clip_to_uint8_range=clip_to_uint8_range,
        )

    def decode_pair_frames(
        self,
        pair_indices: Sequence[int],
        *,
        clip_to_uint8_range: bool = True,
    ) -> np.ndarray:
        """Decode selected pair tensors without rendering unselected frames."""

        return decode_snerv_archive_pair_frames_from_decoded(
            self,
            pair_indices,
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

    def tub_output2_inputs(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return optional archived TUB ``output_2`` fusion tensors."""

        has_temporal = "tub.temporal_encoder_concat" in self.tensors
        has_raw = "tub.output2_raw" in self.tensors
        if has_temporal != has_raw:
            raise SnervArchiveError(
                "official TUB output2 payload must include both "
                "tub.temporal_encoder_concat and tub.output2_raw"
            )
        if not has_temporal:
            return None
        return (
            self.tensors["tub.temporal_encoder_concat"],
            self.tensors["tub.output2_raw"],
        )

    def execute(self) -> dict[str, Any]:
        """Execute receiver-side official primitives and return hashed proof."""

        low, skip_mid, skip_high = self.mfu_inputs()
        current, previous, next_frame = self.tub_inputs()
        output2_inputs = self.tub_output2_inputs()
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
            tub_temporal_encoder_concat=(
                output2_inputs[0] if output2_inputs is not None else None
            ),
            tub_output2_raw=output2_inputs[1] if output2_inputs is not None else None,
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
                "official_tub_output2_fusion": output2_inputs is not None,
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
            "source_forward_blockers": list(
                self.header.get("source_forward_blockers") or ()
            ),
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
        output2_inputs = self.tub_output2_inputs()
        if output2_inputs is not None:
            fc_hw = self.header.get("tub_config", {}).get("fc_hw")
            if fc_hw is None:
                raise SnervArchiveError("official TUB output2 frame decode requires fc_hw")
            fusion = official_output2_fusion_numpy(
                output2_inputs[0],
                output2_inputs[1],
                fc_hw=tuple(int(v) for v in fc_hw),
            )
            frames = _official_planes_to_frames(
                planes,
                frame_count=int(mfu_out.pyr_out.shape[0]),
                channels=int(hfr_out.yh_out.shape[1]),
            )
            frames = _apply_official_tub_output2_frame_residual(
                frames,
                fusion.output2_fused,
                clip_to_uint8_range=clip_to_uint8_range,
            )
            return [np.asarray(frames[frame, channel], dtype=np.float32) for frame in range(frames.shape[0]) for channel in range(frames.shape[1])]
        return planes

    def decode_frames(self, *, clip_to_uint8_range: bool = True) -> np.ndarray:
        """Render official MFU/HFR payload as flat ``(frames, C, H, W)`` frames.

        The official decoder payload does not carry archive-level pair grouping;
        only SNAR metadata can authoritatively reshape into
        ``(n_pairs, frames_per_pair, C, H, W)``. Keeping the direct payload API
        flat prevents callers from accidentally treating receiver self-consistency
        replay as contest frame-pair authority.
        """

        low, skip_mid, skip_high = self.mfu_inputs()
        mfu_out = self.build_mfu().forward(low, skip_mid, skip_high)
        hfr_out = self.build_hfr_heads().forward(mfu_out.pyr_out)
        planes = _official_mfu_hfr_frame_planes(
            mfu_out.pyr_out,
            hfr_out.yh_out,
            clip_to_uint8_range=clip_to_uint8_range,
        )
        if not planes:
            raise SnervArchiveError("official payload produced no frame planes")
        shape = planes[0].shape
        if any(plane.shape != shape for plane in planes):
            raise SnervArchiveError("official payload produced ragged frame planes")
        frame_count = int(mfu_out.pyr_out.shape[0])
        channels = int(hfr_out.yh_out.shape[1])
        expected = frame_count * channels
        if len(planes) != expected:
            raise SnervArchiveError(
                f"official payload produced {len(planes)} planes, expected {expected} "
                f"from frames={frame_count}, channels={channels}"
            )
        frames = _official_planes_to_frames(
            planes,
            frame_count=frame_count,
            channels=channels,
        )
        output2_inputs = self.tub_output2_inputs()
        if output2_inputs is not None:
            fc_hw = self.header.get("tub_config", {}).get("fc_hw")
            if fc_hw is None:
                raise SnervArchiveError("official TUB output2 frame decode requires fc_hw")
            fusion = official_output2_fusion_numpy(
                output2_inputs[0],
                output2_inputs[1],
                fc_hw=tuple(int(v) for v in fc_hw),
            )
            frames = _apply_official_tub_output2_frame_residual(
                frames,
                fusion.output2_fused,
                clip_to_uint8_range=clip_to_uint8_range,
            )
        return frames.astype(np.float32)

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
    tub_temporal_encoder_concat: np.ndarray | None = None,
    tub_output2_raw: np.ndarray | None = None,
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
    if (tub_temporal_encoder_concat is None) != (tub_output2_raw is None):
        raise SnervArchiveError(
            "official TUB output2 receiver replay requires both temporal concat "
            "and raw output2 decoder tensors"
        )
    if tub_temporal_encoder_concat is not None and tub_output2_raw is not None:
        if fc_hw is None:
            raise SnervArchiveError("official TUB output2 receiver replay requires fc_hw")
        fusion = official_output2_fusion_numpy(
            tub_temporal_encoder_concat,
            tub_output2_raw,
            fc_hw=tuple(int(v) for v in fc_hw),
        )
        output_tensors["tub.output2_decoder_input"] = fusion.decoder_input
        output_tensors["tub.output2_fused"] = fusion.output2_fused
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


def _official_planes_to_frames(
    planes: Sequence[np.ndarray],
    *,
    frame_count: int,
    channels: int,
) -> np.ndarray:
    if not planes:
        raise SnervArchiveError("official payload produced no frame planes")
    shape = planes[0].shape
    if any(plane.shape != shape for plane in planes):
        raise SnervArchiveError("official payload produced ragged frame planes")
    expected = int(frame_count) * int(channels)
    if len(planes) != expected:
        raise SnervArchiveError(
            f"official payload produced {len(planes)} planes, expected {expected} "
            f"from frames={frame_count}, channels={channels}"
        )
    h, w = (int(v) for v in shape)
    return np.stack(planes, axis=0).reshape(int(frame_count), int(channels), h, w).astype(
        np.float32
    )


def _apply_official_tub_output2_frame_residual(
    frames: np.ndarray,
    output2_fused: np.ndarray,
    *,
    clip_to_uint8_range: bool,
) -> np.ndarray:
    """Project stored official ``output_2`` bytes into receiver-visible frames.

    This is receiver-payload authority only: it proves the archived output_2
    bytes affect rendered frames.  It is not full trained source-forward
    authority, because the learned temporal decoder weights remain separately
    blocked until an upstream checkpoint replay closes them.
    """

    out = np.asarray(output2_fused, dtype=np.float32)
    frame_array = np.asarray(frames, dtype=np.float32)
    if out.ndim != 4:
        raise SnervArchiveError(f"official TUB output2 fused tensor must be NCHW, got {out.shape}")
    frame_count, channels, h, w = (int(v) for v in frame_array.shape)
    if int(out.shape[0]) <= 0 or int(out.shape[1]) <= 0:
        raise SnervArchiveError("official TUB output2 fused tensor must be non-empty")
    residual = np.zeros_like(frame_array, dtype=np.float32)
    for frame in range(frame_count):
        src_frame = out[min(frame, int(out.shape[0]) - 1)]
        for channel in range(channels):
            src = src_frame[channel % int(out.shape[1])]
            resized = _nearest_resize_2d(src, h, w)
            residual[frame, channel] = resized
    mixed = frame_array + residual
    if clip_to_uint8_range:
        mixed = np.clip(mixed, 0.0, 255.0)
    return np.asarray(mixed, dtype=np.float32)


def _nearest_resize_2d(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src = np.asarray(array, dtype=np.float32)
    if src.ndim != 2:
        raise SnervArchiveError(f"official TUB output2 channel must be HW, got {src.shape}")
    src_h, src_w = (int(v) for v in src.shape)
    if src_h <= 0 or src_w <= 0 or target_h <= 0 or target_w <= 0:
        raise SnervArchiveError("official TUB output2 resize requires positive shapes")
    y_idx = np.minimum((np.arange(target_h) * src_h) // target_h, src_h - 1)
    x_idx = np.minimum((np.arange(target_w) * src_w) // target_w, src_w - 1)
    return np.asarray(src[y_idx[:, None], x_idx[None, :]], dtype=np.float32)


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
    source_forward_blockers: Sequence[str],
    tub_temporal_encoder_concat: np.ndarray | None = None,
    tub_output2_raw: np.ndarray | None = None,
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
        tub_temporal_encoder_concat=tub_temporal_encoder_concat,
        tub_output2_raw=tub_output2_raw,
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
        "source_forward_blockers": list(source_forward_blockers),
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
    section_reports = _receiver_section_reports(
        sections,
        section_bytes=section_bytes,
        section_sha256=section_sha256,
    )
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
        section_reports=section_reports,
        metadata=clean_metadata,
        header_bytes=len(SNERV_ARCHIVE_MAGIC)
        + struct.calcsize(HEADER_LEN_FMT)
        + len(header_bytes_raw),
        total_bytes=len(packet),
    )


def pack_snerv_archive_snar2(
    *,
    metadata_payload: bytes,
    lf_payload: bytes,
    decoder_payload: bytes,
    step_map_packet: bytes,
    metadata: dict[str, Any] | None = None,
) -> SnervArchivePacket:
    """Bundle SNeRV sections with a fixed binary SNAR2 header.

    SNAR2 removes the human-readable outer JSON grammar from SNAR1.  The wire
    header carries only fixed-order section lengths, compact receiver metadata
    integers, and short section hash prefixes.  Full SHA-256 provenance remains
    in surrounding reports, not in the packet header.
    """

    sections = {
        "metadata_payload": bytes(metadata_payload),
        "lf_payload": bytes(lf_payload),
        "decoder_payload": bytes(decoder_payload),
        "step_map_packet": bytes(step_map_packet),
    }
    _validate_sections(sections)
    clean_metadata = _jsonable_metadata(metadata or {})
    fields = _snar2_metadata_fields(clean_metadata)
    payload_parts = [sections[name] for name in SECTION_ORDER]
    section_bytes = {name: len(sections[name]) for name in SECTION_ORDER}
    section_sha256 = {name: _sha256(sections[name]) for name in SECTION_ORDER}
    section_reports = _receiver_section_reports(
        sections,
        section_bytes=section_bytes,
        section_sha256=section_sha256,
    )
    section_lengths = [section_bytes[name] for name in SECTION_ORDER]
    for name, length in zip(SECTION_ORDER, section_lengths, strict=True):
        _snar2_u32(f"{name}_section_length", int(length))
    hash_prefixes = [_sha256_prefix64(sections[name]) for name in SECTION_ORDER]
    header = struct.pack(
        _SNAR2_HEADER_FMT,
        SNERV_ARCHIVE_MAGIC_V2,
        SNAR2_VERSION,
        0,
        int(fields["wavelet_code"]),
        len(SECTION_ORDER),
        int(fields["n_pairs"]),
        int(fields["frames_per_pair"]),
        int(fields["channels"]),
        int(fields["lf_plane_count"]),
        int(fields["levels"]),
        int(fields["metadata_flags"]),
        int(fields["height"]),
        int(fields["width"]),
        *section_lengths,
        *hash_prefixes,
    )
    packet = header + b"".join(payload_parts)
    return SnervArchivePacket(
        packet=packet,
        schema=SNERV_ARCHIVE_SCHEMA_V2,
        section_order=SECTION_ORDER,
        section_bytes=section_bytes,
        section_sha256=section_sha256,
        section_reports=section_reports,
        metadata=_snar2_metadata_from_fields(fields),
        header_bytes=SNAR2_HEADER_BYTES,
        total_bytes=len(packet),
    )


def _receiver_section_reports(
    sections: Mapping[str, bytes],
    *,
    section_bytes: Mapping[str, int],
    section_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Return receiver-byte accounting reports without changing archive bytes."""

    reports: dict[str, Any] = {}
    lf_payload = bytes(sections.get("lf_payload", b""))
    try:
        lf_report = inspect_lf_quant_payload_header(lf_payload)
        lf_status = "receiver_visible_lf_payload_accounting_verified"
        lf_blockers: list[str] = []
    except SnervArchiveError as exc:
        lf_report = {
            "schema": "snerv_lf_payload_codec_report.blocked.v1",
            "error": f"{type(exc).__name__}:{exc}",
        }
        lf_status = "blocked_lf_payload_accounting_not_inspectable"
        lf_blockers = ["snerv_lf_payload_accounting_not_inspectable"]
    lf_report = _jsonable_metadata(
        {
            **dict(lf_report),
            "report_status": lf_status,
            "section_name": "lf_payload",
            "section_bytes": int(section_bytes.get("lf_payload", len(lf_payload))),
            "section_sha256": str(section_sha256.get("lf_payload") or _sha256(lf_payload)),
            "blockers": lf_blockers,
            **FALSE_AUTHORITY,
        }
    )
    reports["lf_payload_codec_report"] = lf_report
    return reports


def unpack_snerv_archive(packet: bytes) -> DecodedSnervArchive:
    """Decode and validate a bundled SNeRV archive packet."""

    packet = bytes(packet)
    if packet.startswith(SNERV_ARCHIVE_MAGIC):
        return _unpack_snerv_archive_v1(packet)
    if packet.startswith(SNERV_ARCHIVE_MAGIC_V2):
        return _unpack_snerv_archive_v2(packet)
    raise SnervArchiveError("bad SNeRV archive magic")


def _unpack_snerv_archive_v1(packet: bytes) -> DecodedSnervArchive:
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


def _unpack_snerv_archive_v2(packet: bytes) -> DecodedSnervArchive:
    if len(packet) < SNAR2_HEADER_BYTES:
        raise SnervArchiveError("truncated SNAR2 fixed header")
    unpacked = struct.unpack(_SNAR2_HEADER_FMT, packet[:SNAR2_HEADER_BYTES])
    (
        magic,
        version,
        flags,
        wavelet_code,
        section_count,
        n_pairs,
        frames_per_pair,
        channels,
        lf_plane_count,
        levels,
        metadata_flags,
        height,
        width,
        *tail,
    ) = unpacked
    if magic != SNERV_ARCHIVE_MAGIC_V2:
        raise SnervArchiveError("bad SNAR2 archive magic")
    if int(version) != SNAR2_VERSION:
        raise SnervArchiveError(f"unsupported SNAR2 version: {version!r}")
    if int(flags) != 0:
        raise SnervArchiveError(f"unsupported SNAR2 flags: {flags!r}")
    if int(section_count) != len(SECTION_ORDER):
        raise SnervArchiveError(
            f"SNAR2 section_count {section_count} != expected {len(SECTION_ORDER)}"
        )
    section_lengths = [int(value) for value in tail[: len(SECTION_ORDER)]]
    hash_prefixes = [int(value) for value in tail[len(SECTION_ORDER) :]]
    if len(hash_prefixes) != len(SECTION_ORDER):
        raise SnervArchiveError("SNAR2 fixed header hash table malformed")
    if any(length <= 0 for length in section_lengths):
        raise SnervArchiveError("SNAR2 section lengths must be positive")
    payload = packet[SNAR2_HEADER_BYTES:]
    if sum(section_lengths) != len(payload):
        raise SnervArchiveError("SNAR2 section lengths do not cover payload exactly")
    sections: dict[str, bytes] = {}
    cursor = 0
    for name, length, expected_prefix in zip(
        SECTION_ORDER,
        section_lengths,
        hash_prefixes,
        strict=True,
    ):
        blob = payload[cursor : cursor + int(length)]
        if _sha256_prefix64(blob) != int(expected_prefix):
            raise SnervArchiveError(f"SNAR2 section {name!r} sha256-prefix mismatch")
        sections[name] = blob
        cursor += int(length)
    metadata = _snar2_metadata_from_fields(
        {
            "wavelet_code": int(wavelet_code),
            "n_pairs": int(n_pairs),
            "frames_per_pair": int(frames_per_pair),
            "channels": int(channels),
            "lf_plane_count": int(lf_plane_count),
            "levels": int(levels),
            "metadata_flags": int(metadata_flags),
            "height": int(height),
            "width": int(width),
        }
    )
    return DecodedSnervArchive(
        schema=SNERV_ARCHIVE_SCHEMA_V2,
        section_order=SECTION_ORDER,
        sections=sections,
        metadata=metadata,
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


def decode_snerv_archive_pair_frames(
    packet: bytes,
    pair_indices: Sequence[int],
    *,
    clip_to_uint8_range: bool = True,
) -> np.ndarray:
    """Decode selected receiver pairs as ``(len(pair_indices), 2, 3, H, W)``."""

    return unpack_snerv_archive(packet).decode_pair_frames(
        pair_indices,
        clip_to_uint8_range=clip_to_uint8_range,
    )


def decode_snerv_archive_frame_planes_from_decoded(
    decoded: DecodedSnervArchive,
    *,
    clip_to_uint8_range: bool = True,
    frame_plane_indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Decode archived LF planes into receiver frames from an unpacked archive."""

    if is_official_mfu_hfr_tub_decoder_payload(decoded.sections["decoder_payload"]):
        if frame_plane_indices is not None:
            raise SnervArchiveError(
                "selected-frame decode is not supported for official MFU/HFR/TUB "
                "proof payloads"
            )
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
    selected = _validate_frame_plane_indices(
        frame_plane_indices,
        plane_count=len(lf_planes),
    )

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

    decode_indices = range(len(codes)) if selected is None else selected
    out: list[np.ndarray] = []
    for idx in decode_indices:
        code = codes[idx]
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


def decode_snerv_archive_pair_frames_from_decoded(
    decoded: DecodedSnervArchive,
    pair_indices: Sequence[int],
    *,
    clip_to_uint8_range: bool = True,
) -> np.ndarray:
    """Decode selected receiver pairs without reconstructing all pair frames."""

    metadata = decoded.metadata
    n_pairs = _metadata_int(metadata, "n_pairs", minimum=1)
    frames_per_pair = _metadata_int(metadata, "frames_per_pair", default=2, minimum=1)
    channels = _metadata_int(metadata, "channels", default=3, minimum=1)
    h, w = _metadata_hw(metadata)
    clean_pair_indices = _validate_pair_indices(pair_indices, n_pairs=n_pairs)
    frame_plane_indices = _frame_plane_indices_for_pairs(
        clean_pair_indices,
        frames_per_pair=frames_per_pair,
        channels=channels,
    )
    planes = decode_snerv_archive_frame_planes_from_decoded(
        decoded,
        clip_to_uint8_range=clip_to_uint8_range,
        frame_plane_indices=frame_plane_indices,
    )
    expected = len(clean_pair_indices) * frames_per_pair * channels
    if len(planes) != expected:
        raise SnervArchiveError(
            f"receiver replay decoded {len(planes)} selected planes, expected {expected}"
        )
    arr = np.stack(planes, axis=0)
    return arr.reshape(len(clean_pair_indices), frames_per_pair, channels, h, w).astype(
        np.float32
    )


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


def _validate_pair_indices(pair_indices: Sequence[int], *, n_pairs: int) -> list[int]:
    out = [int(idx) for idx in pair_indices]
    if not out:
        raise SnervArchiveError("pair_indices must be non-empty")
    invalid = [idx for idx in out if idx < 0 or idx >= int(n_pairs)]
    if invalid:
        preview = invalid[:8]
        suffix = "" if len(invalid) <= 8 else f" ... +{len(invalid) - 8} more"
        raise SnervArchiveError(
            f"pair_indices outside [0,{int(n_pairs)}): {preview}{suffix}"
        )
    return out


def _frame_plane_indices_for_pairs(
    pair_indices: Sequence[int],
    *,
    frames_per_pair: int,
    channels: int,
) -> list[int]:
    stride = int(frames_per_pair) * int(channels)
    return [
        int(pair_index) * stride + int(frame_index) * int(channels) + int(channel)
        for pair_index in pair_indices
        for frame_index in range(int(frames_per_pair))
        for channel in range(int(channels))
    ]


def _validate_frame_plane_indices(
    frame_plane_indices: Sequence[int] | None,
    *,
    plane_count: int,
) -> list[int] | None:
    if frame_plane_indices is None:
        return None
    out = [int(idx) for idx in frame_plane_indices]
    if not out:
        raise SnervArchiveError("frame_plane_indices must be non-empty when provided")
    invalid = [idx for idx in out if idx < 0 or idx >= int(plane_count)]
    if invalid:
        preview = invalid[:8]
        suffix = "" if len(invalid) <= 8 else f" ... +{len(invalid) - 8} more"
        raise SnervArchiveError(
            f"frame_plane_indices outside [0,{int(plane_count)}): {preview}{suffix}"
        )
    return out


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
            encode_lf_quant_payload_v2(arrays, mode="portfolio_auto"),
        )
        return min(candidates, key=len)
    if normalized.startswith("v2:"):
        mode, wrapper = _parse_lf_payload_v2_codec_label(normalized)
        try:
            return encode_lf_quant_payload_v2(arrays, mode=mode, wrapper=wrapper)
        except SnervLfPayloadCodecError as exc:
            raise SnervArchiveError(str(exc)) from exc
    try:
        return encode_lf_quant_payload_v2(arrays, mode=normalized)
    except SnervLfPayloadCodecError as exc:
        raise SnervArchiveError(str(exc)) from exc
    raise SnervArchiveError(f"unsupported LF quant payload codec: {codec!r}")


def _parse_lf_payload_v2_codec_label(codec: str) -> tuple[str, str]:
    """Parse receiver-selected LF labels back into v2 encoder controls."""

    parts = str(codec).strip().lower().split(":")
    if len(parts) == 3 and parts[0] == "v2":
        _prefix, mode, wrapper = parts
    elif len(parts) == 4 and parts[:2] == ["v2", "portfolio"]:
        _prefix, _portfolio, mode, wrapper = parts
    else:
        raise SnervArchiveError(f"unsupported LF v2 selected codec label: {codec!r}")
    if mode.startswith("unknown") or wrapper.startswith("unknown"):
        raise SnervArchiveError(
            f"LF v2 selected codec label is not reproducible: {codec!r}"
        )
    return mode, wrapper


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
    normalized = resolve_decoder_payload_codec(codec)
    if normalized == DECODER_PAYLOAD_LEGACY_CODEC:
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
    if normalized == DECODER_PAYLOAD_MIXED_CODEC:
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
    tub_temporal_encoder_concat: np.ndarray | None = None,
    tub_output2_raw: np.ndarray | None = None,
    store_tub_output2_for_receiver_proof: bool = False,
    skip_high_codec: str | None = None,
    skip_high_source_shape: tuple[int, int, int, int] | None = None,
    tub_input_codec: str | None = None,
) -> bytes:
    """Encode executable official MFU/HFR/TUB receiver primitive bytes.

    This is not a contest-score claim. It is the receiver/runtime custody
    surface for official primitive tensors and inputs: the decoder section can
    now carry bytes that are decoded into the official NumPy primitives instead
    of only carrying the local linear HF surrogate.
    """

    skip_high_plan = _official_skip_high_storage_plan(
        skip_high,
        codec=skip_high_codec,
        source_shape=skip_high_source_shape,
    )
    tub_input_plan = _official_tub_input_storage_plan(
        current=tub_current,
        previous=tub_previous,
        next_frame=tub_next_frame,
        codec=tub_input_codec,
    )
    output2_plan = _official_tub_output2_storage_plan(
        temporal_encoder_concat=tub_temporal_encoder_concat,
        output2_raw=tub_output2_raw,
        fc_hw=fc_hw,
        temporal_encoder_output_shape=temporal_encoder_output_shape,
        output2_decoder_output_shape=output2_decoder_output_shape,
        store_for_receiver_proof=bool(store_tub_output2_for_receiver_proof),
    )
    source_forward_blockers = _official_mfu_hfr_tub_source_forward_blockers(
        skip_high_plan["metadata"],
        tub_input_plan["metadata"],
        output2_plan["metadata"],
    )
    tensors = _official_payload_tensor_dict(
        mfu=mfu,
        hfr_heads=hfr_heads,
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high_plan["stored"],
        tub_current=tub_input_plan["stored"]["current"],
        tub_previous=tub_input_plan["stored"]["previous"],
        tub_next_frame=tub_input_plan["stored"]["next_frame"],
        tub_temporal_encoder_concat=output2_plan["stored"].get(
            "temporal_encoder_concat"
        ),
        tub_output2_raw=output2_plan["stored"].get("output2_raw"),
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
            [int(v) for v in output2_plan["temporal_encoder_output_shape"]]
            if output2_plan["temporal_encoder_output_shape"] is not None
            else None
        ),
        "fc_hw": [int(v) for v in fc_hw] if fc_hw is not None else None,
        "output2_decoder_output_shape": (
            [int(v) for v in output2_plan["output2_decoder_output_shape"]]
            if output2_plan["output2_decoder_output_shape"] is not None
            else None
        ),
    }
    self_consistency_reference = _build_official_receiver_self_consistency_reference(
        mfu=mfu,
        hfr_heads=hfr_heads,
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high_plan["effective"],
        tub_current=tub_input_plan["effective"]["current"],
        tub_previous=tub_input_plan["effective"]["previous"],
        tub_next_frame=tub_input_plan["effective"]["next_frame"],
        tub_config=tub_config,
        source_forward_blockers=source_forward_blockers,
        tub_temporal_encoder_concat=output2_plan["effective"].get(
            "temporal_encoder_concat"
        ),
        tub_output2_raw=output2_plan["effective"].get("output2_raw"),
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
        "skip_high_storage": skip_high_plan["metadata"],
        "tub_input_storage": tub_input_plan["metadata"],
        "tub_output2_storage": output2_plan["metadata"],
        "receiver_self_consistency_reference": self_consistency_reference,
        "receiver_self_consistency_reference_sha256": _json_sha256(
            self_consistency_reference
        ),
        "receiver_export_payload_bound": True,
        "receiver_export_self_consistency_verified": True,
        "source_forward_replay_bound_by_export": False,
        "source_forward_blockers": list(source_forward_blockers),
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
    tensors = _expand_official_skip_high_storage(header, tensors)
    tensors = _expand_official_tub_input_storage(header, tensors)
    payload_obj = OfficialMfuHfrTubReceiverPayload(
        header=dict(header),
        tensors=tensors,
        payload_sha256=_sha256(bytes(payload)),
        payload_bytes=len(bytes(payload)),
    )
    _validate_official_payload_exec_surfaces(payload_obj)
    return payload_obj


def _normalize_official_skip_high_codec(codec: str | None) -> str:
    raw = "full" if codec is None else str(codec).strip().lower()
    try:
        return OFFICIAL_SKIP_HIGH_MODE_TO_CODEC[raw]
    except KeyError as exc:
        raise SnervArchiveError(f"unsupported official skip_high codec: {codec!r}") from exc


def _normalize_official_tub_input_codec(codec: str | None) -> str:
    raw = "full" if codec is None else str(codec).strip().lower()
    try:
        return OFFICIAL_TUB_INPUT_MODE_TO_CODEC[raw]
    except KeyError as exc:
        raise SnervArchiveError(f"unsupported official tub input codec: {codec!r}") from exc


def _official_skip_high_storage_plan(
    skip_high: np.ndarray,
    *,
    codec: str | None,
    source_shape: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    arr = _canonical_float64_tensor(skip_high, name="inputs.mfu.skip_high")
    if arr.ndim != 4:
        raise SnervArchiveError(
            f"official skip_high must be NCHW, got {tuple(arr.shape)}"
        )
    normalized = _normalize_official_skip_high_codec(codec)
    declared_source_shape = (
        None
        if source_shape is None
        else tuple(int(value) for value in source_shape)
    )
    if declared_source_shape is not None and (
        len(declared_source_shape) != 4 or any(value <= 0 for value in declared_source_shape)
    ):
        raise SnervArchiveError(
            f"official skip_high source_shape must be positive NCHW, got {declared_source_shape}"
        )
    if normalized == OFFICIAL_SKIP_HIGH_CODEC_FULL and declared_source_shape is not None:
        if tuple(arr.shape) != declared_source_shape:
            raise SnervArchiveError(
                "official full skip_high payload shape must match source_shape; "
                f"payload={tuple(arr.shape)} source_shape={declared_source_shape}"
            )
        source_shape_tuple = declared_source_shape
    elif declared_source_shape is None:
        source_shape_tuple = tuple(int(v) for v in arr.shape)
    else:
        source_shape_tuple = declared_source_shape

    compact_expected_tail = {
        OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN: tuple(source_shape_tuple[1:]),
        OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN: (int(source_shape_tuple[1]), 1, 1),
        OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN: (1, 1, 1),
    }
    compact_payload = (
        normalized in compact_expected_tail
        and declared_source_shape is not None
        and int(arr.shape[0]) == 1
        and tuple(arr.shape[1:]) == compact_expected_tail[normalized]
    )
    if normalized == OFFICIAL_SKIP_HIGH_CODEC_FULL:
        stored = arr
        effective = arr
    elif compact_payload:
        stored = arr
        effective = np.broadcast_to(stored, source_shape_tuple).copy()
    elif normalized == OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN:
        stored = np.mean(arr, axis=0, keepdims=True, dtype=np.float64)
        effective = np.broadcast_to(stored, source_shape_tuple).copy()
    elif normalized == OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN:
        stored = np.mean(arr, axis=(0, 2, 3), keepdims=True, dtype=np.float64)
        effective = np.broadcast_to(stored, source_shape_tuple).copy()
    elif normalized == OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN:
        stored = np.asarray([[[[float(np.mean(arr, dtype=np.float64))]]]], dtype=np.float64)
        effective = np.broadcast_to(stored, source_shape_tuple).copy()
    else:  # pragma: no cover - guarded by normalizer
        raise SnervArchiveError(f"unsupported official skip_high codec: {codec!r}")
    stored = _canonical_float64_tensor(stored, name="inputs.mfu.skip_high")
    effective = _canonical_float64_tensor(effective, name="inputs.mfu.skip_high.expanded")
    full_raw_bytes = int(np.prod(source_shape_tuple)) * np.dtype("<f8").itemsize
    stored_raw_bytes = int(stored.size) * np.dtype("<f8").itemsize
    blockers = _official_skip_high_source_forward_blockers(
        codec=normalized,
        compact_payload=compact_payload,
    )
    return {
        "stored": stored,
        "effective": effective,
        "metadata": {
            "schema": "snerv_official_skip_high_storage.v1",
            "codec": normalized,
            "source_shape": [int(v) for v in source_shape_tuple],
            "stored_shape": [int(v) for v in stored.shape],
            "effective_shape": [int(v) for v in effective.shape],
            "source_raw_bytes": full_raw_bytes,
            "stored_raw_bytes": stored_raw_bytes,
            "raw_byte_savings": full_raw_bytes - stored_raw_bytes,
            "encoder_consumed_compact_train_state": bool(compact_payload),
            "receiver_expands_skip_high": normalized
            != OFFICIAL_SKIP_HIGH_CODEC_FULL,
            "lossless_relative_to_source_skip_high": normalized
            == OFFICIAL_SKIP_HIGH_CODEC_FULL,
            "train_time_tied_state_required_for_exact_compact_export": normalized
            != OFFICIAL_SKIP_HIGH_CODEC_FULL,
            "source_forward_blockers": blockers,
            **FALSE_AUTHORITY,
        },
    }


def _official_skip_high_source_forward_blockers(
    *,
    codec: str,
    compact_payload: bool,
) -> list[str]:
    if codec == OFFICIAL_SKIP_HIGH_CODEC_FULL:
        return []
    if compact_payload:
        return []
    return ["snerv_compact_skip_high_train_state_not_bound"]


def _expand_official_skip_high_storage(
    header: Mapping[str, Any],
    tensors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    storage = header.get("skip_high_storage")
    if storage is None:
        return tensors
    if not isinstance(storage, Mapping):
        raise SnervArchiveError("official skip_high storage metadata must be an object")
    codec = _normalize_official_skip_high_codec(str(storage.get("codec", "full")))
    skip_high = _tensor(tensors, "inputs.mfu.skip_high")
    source_shape = tuple(int(v) for v in storage.get("source_shape") or ())
    stored_shape = tuple(int(v) for v in storage.get("stored_shape") or ())
    if stored_shape and tuple(skip_high.shape) != stored_shape:
        raise SnervArchiveError(
            "official compact skip_high stored shape mismatch; "
            f"manifest={tuple(skip_high.shape)} header={stored_shape}"
        )
    if codec == OFFICIAL_SKIP_HIGH_CODEC_FULL:
        if source_shape and tuple(skip_high.shape) != source_shape:
            raise SnervArchiveError(
                "official full skip_high source shape mismatch; "
                f"manifest={tuple(skip_high.shape)} header={source_shape}"
            )
        return tensors
    if codec not in {
        OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN,
        OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN,
        OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN,
    }:
        raise SnervArchiveError(f"unsupported official skip_high codec: {codec!r}")
    if len(source_shape) != 4 or any(v <= 0 for v in source_shape):
        raise SnervArchiveError("official shared skip_high source shape is invalid")
    if skip_high.ndim != 4 or int(skip_high.shape[0]) != 1:
        raise SnervArchiveError(
            "official shared skip_high payload must store exactly one NCHW frame"
        )
    expected_tail = {
        OFFICIAL_SKIP_HIGH_CODEC_SHARED_MEAN: tuple(source_shape[1:]),
        OFFICIAL_SKIP_HIGH_CODEC_CHANNEL_MEAN: (int(source_shape[1]), 1, 1),
        OFFICIAL_SKIP_HIGH_CODEC_SCALAR_MEAN: (1, 1, 1),
    }[codec]
    if tuple(skip_high.shape[1:]) != expected_tail:
        raise SnervArchiveError(
            "official shared skip_high channel/spatial shape mismatch"
        )
    expanded = np.broadcast_to(skip_high, source_shape).astype(np.float64, copy=True)
    out = dict(tensors)
    out["inputs.mfu.skip_high"] = expanded
    return out


def _official_tub_input_storage_plan(
    *,
    current: np.ndarray,
    previous: np.ndarray,
    next_frame: np.ndarray,
    codec: str | None,
) -> dict[str, Any]:
    source = {
        "current": _canonical_float64_tensor(current, name="inputs.tub.current"),
        "previous": _canonical_float64_tensor(previous, name="inputs.tub.previous"),
        "next_frame": _canonical_float64_tensor(next_frame, name="inputs.tub.next_frame"),
    }
    normalized = _normalize_official_tub_input_codec(codec)
    if normalized == OFFICIAL_TUB_INPUT_CODEC_FULL:
        stored = source
        effective = source
    elif normalized == OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC:
        stored = {
            key: _unused_tub_storage_placeholder()
            for key, value in source.items()
        }
        effective = {
            key: _unused_tub_placeholder(value.shape)
            for key, value in source.items()
        }
    else:  # pragma: no cover - guarded by normalizer
        raise SnervArchiveError(f"unsupported official tub input codec: {codec!r}")
    source_raw_bytes = sum(
        int(value.size) * np.dtype("<f8").itemsize for value in source.values()
    )
    stored_raw_bytes = sum(
        int(value.size) * np.dtype("<f8").itemsize for value in stored.values()
    )
    return {
        "stored": stored,
        "effective": effective,
        "metadata": {
            "schema": "snerv_official_tub_input_storage.v1",
            "codec": normalized,
            "source_shapes": {
                key: [int(v) for v in value.shape] for key, value in source.items()
            },
            "stored_shapes": {
                key: [int(v) for v in value.shape] for key, value in stored.items()
            },
            "source_raw_bytes": int(source_raw_bytes),
            "stored_raw_bytes": int(stored_raw_bytes),
            "raw_byte_savings": int(source_raw_bytes - stored_raw_bytes),
            "receiver_expands_tub_inputs": (
                normalized == OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC
            ),
            "receiver_frame_synthesis_uses_tub_inputs": False,
            "lossless_relative_to_source_tub_inputs": (
                normalized == OFFICIAL_TUB_INPUT_CODEC_FULL
            ),
            "source_forward_replay_authority": False,
            "contest_scorer_authority": False,
            "source_forward_blockers": (
                []
                if normalized == OFFICIAL_TUB_INPUT_CODEC_FULL
                else ["snerv_unused_tub_inputs_synthetic_not_source_forward_replay"]
            ),
            **FALSE_AUTHORITY,
        },
    }


def _official_tub_output2_storage_plan(
    *,
    temporal_encoder_concat: np.ndarray | None,
    output2_raw: np.ndarray | None,
    fc_hw: tuple[int, int] | None,
    temporal_encoder_output_shape: tuple[int, int, int, int] | None,
    output2_decoder_output_shape: tuple[int, int, int, int] | None,
    store_for_receiver_proof: bool,
) -> dict[str, Any]:
    if (temporal_encoder_concat is None) != (output2_raw is None):
        raise SnervArchiveError(
            "official TUB output2 payload requires both temporal_encoder_concat "
            "and output2_raw"
        )
    if temporal_encoder_concat is None or output2_raw is None:
        return {
            "stored": {},
            "effective": {},
            "temporal_encoder_output_shape": temporal_encoder_output_shape,
            "output2_decoder_output_shape": output2_decoder_output_shape,
            "metadata": {
                "schema": "snerv_official_tub_output2_storage.v1",
                "stored": False,
                "source_payload_present": False,
                "proof_only_elided_from_selected_runtime_packet": False,
                "proof_only_false_authority_metadata": False,
                "receiver_executes_output2_fusion_from_payload": False,
                "receiver_frame_decode_consumes_output2": False,
                "train_time_loss_coupled": False,
                "scored_pixel_render_bound": False,
                "score_lagrangian_admission": "not_present",
                "score_lagrangian_action": "none",
                "source_raw_bytes": 0,
                "stored_raw_bytes": 0,
                "raw_byte_savings": 0,
                "source_forward_replay_authority": False,
                "contest_scorer_authority": False,
                **FALSE_AUTHORITY,
            },
        }
    if fc_hw is None:
        raise SnervArchiveError("official TUB output2 payload requires fc_hw")
    temporal = _canonical_float64_tensor(
        temporal_encoder_concat,
        name="tub.temporal_encoder_concat",
    )
    raw = _canonical_float64_tensor(output2_raw, name="tub.output2_raw")
    temporal_shape = tuple(int(v) for v in temporal.shape)
    raw_shape = tuple(int(v) for v in raw.shape)
    if temporal_encoder_output_shape is not None:
        expected = tuple(int(v) for v in temporal_encoder_output_shape)
        if temporal_shape != expected:
            raise SnervArchiveError(
                "official TUB temporal encoder output shape mismatch; "
                f"tensor={temporal_shape} config={expected}"
            )
    if output2_decoder_output_shape is not None:
        expected = tuple(int(v) for v in output2_decoder_output_shape)
        if raw_shape != expected:
            raise SnervArchiveError(
                "official TUB output2 raw shape mismatch; "
                f"tensor={raw_shape} config={expected}"
            )
    # Validate the exact receiver algebra at export time before bytes are stored.
    fusion = official_output2_fusion_numpy(temporal, raw, fc_hw=fc_hw)
    source_raw_bytes = int(temporal.size + raw.size) * np.dtype("<f8").itemsize
    should_store = bool(store_for_receiver_proof)
    stored = (
        {
            "temporal_encoder_concat": temporal,
            "output2_raw": raw,
        }
        if should_store
        else {}
    )
    effective = dict(stored)
    stored_raw_bytes = source_raw_bytes if should_store else 0
    return {
        "stored": stored,
        "effective": effective,
        "temporal_encoder_output_shape": temporal_shape,
        "output2_decoder_output_shape": raw_shape,
        "metadata": {
            "schema": "snerv_official_tub_output2_storage.v1",
            "stored": should_store,
            "source_payload_present": True,
            "proof_only_elided_from_selected_runtime_packet": not should_store,
            "proof_only_false_authority_metadata": not should_store,
            "storage_policy": (
                "store_for_receiver_proof"
                if should_store
                else "elide_until_receiver_frame_decode_bound"
            ),
            "receiver_executes_output2_fusion_from_payload": should_store,
            "receiver_frame_decode_consumes_output2": should_store,
            "train_time_loss_coupled": False,
            "scored_pixel_render_bound": should_store,
            "score_lagrangian_admission": (
                "receiver_frame_decode_bound_proof_only_false_authority"
                if should_store
                else "elided_non_score_causal_payload"
            ),
            "score_lagrangian_action": (
                "keep_only_for_receiver_proof_until_trained_source_forward_parity"
                if should_store
                else "elide_for_score_candidate_or_implement_source_faithful_tub_decoder"
            ),
            "tensor_names": [
                "tub.temporal_encoder_concat",
                "tub.output2_raw",
            ],
            "temporal_encoder_output_shape": [int(v) for v in temporal_shape],
            "output2_decoder_output_shape": [int(v) for v in raw_shape],
            "fc_hw": [int(v) for v in fc_hw],
            "output2_decoder_input_shape": [
                int(v) for v in fusion.decoder_input.shape
            ],
            "output2_fused_shape": [int(v) for v in fusion.output2_fused.shape],
            "temporal_encoder_concat_sha256": _sha256(temporal.tobytes()),
            "output2_raw_sha256": _sha256(raw.tobytes()),
            "output2_decoder_input_sha256": _sha256(
                np.asarray(fusion.decoder_input, dtype="<f8").tobytes()
            ),
            "output2_fused_sha256": _sha256(
                np.asarray(fusion.output2_fused, dtype="<f8").tobytes()
            ),
            "source_raw_bytes": source_raw_bytes,
            "stored_raw_bytes": stored_raw_bytes,
            "raw_byte_savings": source_raw_bytes - stored_raw_bytes,
            "source_forward_replay_authority": False,
            "contest_scorer_authority": False,
            "source_forward_blockers": (
                []
                if should_store
                else ["snerv_tub_output2_source_payload_elided_from_runtime_packet"]
            ),
            **FALSE_AUTHORITY,
        },
    }


def _official_mfu_hfr_tub_source_forward_blockers(
    skip_high_storage: Mapping[str, Any],
    tub_input_storage: Mapping[str, Any],
    tub_output2_storage: Mapping[str, Any],
) -> tuple[str, ...]:
    blockers: list[str] = list(OFFICIAL_MFU_HFR_TUB_BASE_SOURCE_FORWARD_BLOCKERS)
    for storage in (skip_high_storage, tub_input_storage, tub_output2_storage):
        raw = storage.get("source_forward_blockers") or ()
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise SnervArchiveError("official source-forward blockers must be a list")
        blockers.extend(str(value) for value in raw)
    return tuple(dict.fromkeys(blockers))


def _expand_official_tub_input_storage(
    header: Mapping[str, Any],
    tensors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    storage = header.get("tub_input_storage")
    if storage is None:
        return tensors
    if not isinstance(storage, Mapping):
        raise SnervArchiveError("official TUB input storage metadata must be an object")
    codec = _normalize_official_tub_input_codec(str(storage.get("codec", "full")))
    source_shapes = storage.get("source_shapes") or {}
    stored_shapes = storage.get("stored_shapes") or {}
    if not isinstance(source_shapes, Mapping) or not isinstance(stored_shapes, Mapping):
        raise SnervArchiveError("official TUB input storage shapes must be objects")

    out = dict(tensors)
    for key in ("current", "previous", "next_frame"):
        tensor_name = f"inputs.tub.{key}"
        value = _tensor(tensors, tensor_name)
        source_shape = tuple(int(v) for v in (source_shapes.get(key) or ()))
        stored_shape = tuple(int(v) for v in (stored_shapes.get(key) or ()))
        if stored_shape and tuple(value.shape) != stored_shape:
            raise SnervArchiveError(
                "official compact TUB stored shape mismatch; "
                f"{tensor_name} manifest={tuple(value.shape)} header={stored_shape}"
            )
        if codec == OFFICIAL_TUB_INPUT_CODEC_FULL:
            if source_shape and tuple(value.shape) != source_shape:
                raise SnervArchiveError(
                    "official full TUB source shape mismatch; "
                    f"{tensor_name} manifest={tuple(value.shape)} header={source_shape}"
                )
            continue
        if codec != OFFICIAL_TUB_INPUT_CODEC_UNUSED_SYNTHETIC:
            raise SnervArchiveError(f"unsupported official TUB input codec: {codec!r}")
        if len(source_shape) != 3 or any(v <= 0 for v in source_shape):
            raise SnervArchiveError(
                f"official synthetic TUB source shape is invalid for {tensor_name}"
            )
        if tuple(value.shape) != (2,) or not np.array_equal(
            value.astype(np.float64, copy=False),
            _unused_tub_storage_placeholder(),
        ):
            raise SnervArchiveError(
                f"official synthetic TUB placeholder is invalid for {tensor_name}"
            )
        out[tensor_name] = _unused_tub_placeholder(source_shape)
    return out


def _unused_tub_storage_placeholder() -> np.ndarray:
    """Return the tiny deterministic sentinel stored for unused TUB inputs."""

    return np.asarray([1.0, -1.0], dtype=np.float64)


def _unused_tub_placeholder(shape: tuple[int, ...]) -> np.ndarray:
    """Expand a deterministic non-constant placeholder for unused TUB inputs."""

    out = np.zeros(tuple(int(v) for v in shape), dtype=np.float64)
    flat = out.reshape(-1)
    if flat.size == 0:
        return out
    flat[0] = 1.0
    if flat.size > 1:
        flat[-1] = -1.0
    return out


def execute_official_mfu_hfr_tub_decoder_payload(payload: bytes) -> dict[str, Any]:
    """Decode and execute official MFU/HFR/TUB primitive bytes."""

    return decode_official_mfu_hfr_tub_decoder_payload(payload).execute()


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
    tub_temporal_encoder_concat: np.ndarray | None = None,
    tub_output2_raw: np.ndarray | None = None,
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
    if (tub_temporal_encoder_concat is None) != (tub_output2_raw is None):
        raise SnervArchiveError(
            "official TUB output2 tensor dictionary requires both temporal concat "
            "and output2 raw tensors"
        )
    if tub_temporal_encoder_concat is not None and tub_output2_raw is not None:
        tensors["tub.temporal_encoder_concat"] = tub_temporal_encoder_concat
        tensors["tub.output2_raw"] = tub_output2_raw
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
    payload.tub_output2_inputs()
    _validate_official_tub_output2_storage(payload.header, payload.tensors)
    _official_receiver_self_consistency_reference_from_header(payload.header)


def _validate_official_tub_output2_storage(
    header: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
) -> None:
    storage = header.get("tub_output2_storage")
    if storage is None:
        return
    if not isinstance(storage, Mapping):
        raise SnervArchiveError("official TUB output2 storage metadata must be an object")
    has_temporal = "tub.temporal_encoder_concat" in tensors
    has_raw = "tub.output2_raw" in tensors
    if has_temporal != has_raw:
        raise SnervArchiveError(
            "official TUB output2 storage has incomplete tensor pair"
        )
    stored = bool(storage.get("stored"))
    if stored != bool(has_temporal):
        raise SnervArchiveError(
            "official TUB output2 storage metadata/tensor presence mismatch"
        )
    if not stored:
        return
    temporal = np.asarray(tensors["tub.temporal_encoder_concat"])
    raw = np.asarray(tensors["tub.output2_raw"])
    if tuple(temporal.shape) != tuple(
        int(v) for v in storage.get("temporal_encoder_output_shape") or ()
    ):
        raise SnervArchiveError("official TUB output2 temporal shape metadata mismatch")
    if tuple(raw.shape) != tuple(
        int(v) for v in storage.get("output2_decoder_output_shape") or ()
    ):
        raise SnervArchiveError("official TUB output2 raw shape metadata mismatch")


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
    blockers = reference.get("source_forward_blockers")
    if blockers is not None:
        if isinstance(blockers, (str, bytes)) or not isinstance(blockers, Sequence):
            raise SnervArchiveError(
                "official primitive payload source-forward blockers must be a list"
            )
        if not blockers:
            raise SnervArchiveError(
                "official primitive payload must preserve source-forward blockers"
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
            official_skip_high_mode=str(raw.get("official_skip_high_mode", "full")),
            official_tub_output2_store_for_receiver_proof=bool(
                raw.get("official_tub_output2_store_for_receiver_proof", False)
            ),
            official_tub_output2_export_mode=str(
                raw.get("official_tub_output2_export_mode", "auto_elide")
            ),
            official_tub_output2_store_for_receiver_proof_requested=bool(
                raw.get(
                    "official_tub_output2_store_for_receiver_proof_requested",
                    raw.get("official_tub_output2_store_for_receiver_proof", False),
                )
            ),
            fc_dim_source=str(raw.get("fc_dim_source", "decoder_header")),
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


def _snar2_metadata_fields(metadata: Mapping[str, Any]) -> dict[str, int]:
    clean_metadata = dict(metadata)
    if "carrier_hw" in clean_metadata and "orig_hw" in clean_metadata:
        carrier_hw = _metadata_hw_value("carrier_hw", clean_metadata["carrier_hw"])
        orig_hw = _metadata_hw_value("orig_hw", clean_metadata["orig_hw"])
        if carrier_hw != orig_hw:
            raise SnervArchiveError(
                "SNAR2 fixed header cannot encode distinct carrier_hw and orig_hw"
            )
    n_pairs = _metadata_int(clean_metadata, "n_pairs", minimum=1)
    frames_per_pair = _metadata_int(
        clean_metadata,
        "frames_per_pair",
        default=2,
        minimum=1,
    )
    channels = _metadata_int(clean_metadata, "channels", default=3, minimum=1)
    lf_plane_count = _metadata_int(clean_metadata, "lf_plane_count", minimum=1)
    levels = _metadata_int(clean_metadata, "levels", minimum=1)
    wavelet = _metadata_str(clean_metadata, "wavelet").strip().lower()
    if wavelet not in _SNAR2_WAVELET_TO_CODE:
        raise SnervArchiveError(f"SNAR2 unsupported compact wavelet code: {wavelet!r}")
    height, width = _metadata_hw(dict(metadata))
    _snar2_u16("n_pairs", n_pairs)
    _snar2_u8("frames_per_pair", frames_per_pair)
    _snar2_u8("channels", channels)
    _snar2_u32("lf_plane_count", lf_plane_count)
    _snar2_u8("levels", levels)
    _snar2_u16("height", height)
    _snar2_u16("width", width)
    return {
        "n_pairs": n_pairs,
        "frames_per_pair": frames_per_pair,
        "channels": channels,
        "lf_plane_count": lf_plane_count,
        "levels": levels,
        "wavelet_code": int(_SNAR2_WAVELET_TO_CODE[wavelet]),
        "metadata_flags": 0,
        "height": height,
        "width": width,
    }


def _snar2_metadata_from_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    wavelet_code = int(fields["wavelet_code"])
    wavelet = _SNAR2_CODE_TO_WAVELET.get(wavelet_code)
    if wavelet is None:
        raise SnervArchiveError(f"unsupported SNAR2 wavelet code: {wavelet_code!r}")
    n_pairs = int(fields["n_pairs"])
    frames_per_pair = int(fields["frames_per_pair"])
    channels = int(fields["channels"])
    lf_plane_count = int(fields["lf_plane_count"])
    levels = int(fields["levels"])
    metadata_flags = int(fields["metadata_flags"])
    height = int(fields["height"])
    width = int(fields["width"])
    if metadata_flags != 0:
        raise SnervArchiveError(f"unsupported SNAR2 metadata_flags: {metadata_flags!r}")
    _snar2_u16("n_pairs", n_pairs)
    _snar2_u8("frames_per_pair", frames_per_pair)
    _snar2_u8("channels", channels)
    _snar2_u32("lf_plane_count", lf_plane_count)
    _snar2_u8("levels", levels)
    _snar2_u16("height", height)
    _snar2_u16("width", width)
    for name, value in (
        ("n_pairs", n_pairs),
        ("frames_per_pair", frames_per_pair),
        ("channels", channels),
        ("lf_plane_count", lf_plane_count),
        ("levels", levels),
        ("height", height),
        ("width", width),
    ):
        if int(value) < 1:
            raise SnervArchiveError(f"SNAR2 {name}={value} must be >= 1")
    return {
        "n_pairs": n_pairs,
        "frames_per_pair": frames_per_pair,
        "channels": channels,
        "lf_plane_count": lf_plane_count,
        "levels": levels,
        "wavelet": wavelet,
        "carrier_hw": [height, width],
        "orig_hw": [height, width],
    }


def _snar2_u8(name: str, value: int) -> None:
    if int(value) < 0 or int(value) > 0xFF:
        raise SnervArchiveError(f"SNAR2 {name}={value} out of u8 range")


def _snar2_u16(name: str, value: int) -> None:
    if int(value) < 0 or int(value) > 0xFFFF:
        raise SnervArchiveError(f"SNAR2 {name}={value} out of u16 range")


def _snar2_u32(name: str, value: int) -> None:
    if int(value) < 0 or int(value) > 0xFFFFFFFF:
        raise SnervArchiveError(f"SNAR2 {name}={value} out of u32 range")


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
    return _metadata_hw_value("'carrier_hw'/'orig_hw'", value)


def _metadata_hw_value(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SnervArchiveError(f"receiver replay metadata missing 2-element {name}")
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


def _sha256_prefix64(blob: bytes) -> int:
    return int.from_bytes(hashlib.sha256(blob).digest()[:SNAR2_SECTION_HASH_BYTES], "little")


def _json_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
