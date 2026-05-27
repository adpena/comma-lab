# SPDX-License-Identifier: MIT
"""Parser-section manifests for monolithic public HNeRV packets.

The public HNeRV-family frontier archives are usually one charged ZIP member.
This helper records the parser-proven sections inside that member: archive
identity, member identity, byte offsets, lengths, section names, and hashes.
It is custody/audit infrastructure only; it never emits a score claim.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
)
from tac.hnerv_pr103_lc_ac_schema import (
    PUBLIC_PR103_LAYOUT,
    HnervPr103LcAcSchemaError,
    parse_pr103_lc_ac_payload,
)
from tac.optimization.decoder_q_selective_runtime_packet import (
    PACKET_MAGIC as DQS1_MAGIC_PREFIX,
)
from tac.optimization.decoder_q_selective_runtime_packet import (
    DecoderQSelectiveRuntimePacketError,
    unpack_dqs1_payload,
)
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_MAGIC,
    parse_pr106_sidecar_packet,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes

SCHEMA_VERSION = 1
TOOL_NAME = "tac.analysis.hnerv_packet_sections"
MANIFEST_SCHEMA = "tac_hnerv_packet_section_manifest.v1"
BATCH_SCHEMA = "tac_hnerv_packet_section_manifest_batch.v1"

PARSER_AUTO = "auto"
PARSER_PR101 = "pr101_microcodec_fixed"
PARSER_PR101_FEC6 = "pr101_fec6_fixed_huffman_selector"
PARSER_PR101_FEC6_SHORT = "pr101_fec6"
PARSER_PR103 = "pr103_lc_ac"
PARSER_PR106 = "pr106_ff_packed_hnerv"
PARSER_A2K1 = "a2k1_variable_decoder_pr101"
PARSER_A5FC = "a5fc_frame_conditional_pr101"
PARSER_CPLX1 = "cplx1_op1_byte_maps"
PARSER_IBPS1 = "ibps1_mdl_ibps"
PARSER_IBPS1_SHORT = "ibps1"
PARSER_D1POLY1 = "d1poly1_segnet_margin_polytope"
PARSER_D1POLY1_SHORT = "d1poly1"
PARSER_WZF01 = "wzf01_wyner_ziv_frame_0"
PARSER_WZF01_SHORT = "wzf01"
PARSER_DP1 = "dp1_pretrained_driving_prior"
PARSER_DP1_SHORT = "dp1"
PARSER_C1WMFV1 = "c1wmfv1_world_model_foveation"
PARSER_C1WMFV1_SHORT = "c1wmfv1"
PARSER_WZ1 = "wz1_wyner_ziv_cooperative_receiver"
PARSER_WZ1_SHORT = "wz1"
PARSER_Z4CR1 = "z4cr1_cooperative_receiver_loss"
PARSER_Z4CR1_SHORT = "z4cr1"
PARSER_Z5PCWM1 = "z5pcwm1_predictive_coding_world_model"
PARSER_Z5PCWM1_SHORT = "z5pcwm1"
PARSER_TT5L = "tt5l_time_traveler_l5_autonomy"
PARSER_TT5L_SHORT = "tt5l"
PARSER_ALIASES = {
    PARSER_PR101_FEC6_SHORT: PARSER_PR101_FEC6,
    "fec6_fixed_huffman_k16": PARSER_PR101_FEC6,
    "pr101_frame_exploit_selector": PARSER_PR101_FEC6,
    PARSER_IBPS1_SHORT: PARSER_IBPS1,
    "c6_e4_mdl_ibps": PARSER_IBPS1,
    "mdl_ibps": PARSER_IBPS1,
    PARSER_D1POLY1_SHORT: PARSER_D1POLY1,
    "d1_segnet_margin_polytope": PARSER_D1POLY1,
    "d1_polytope": PARSER_D1POLY1,
    PARSER_WZF01_SHORT: PARSER_WZF01,
    "d4_wyner_ziv_frame_0": PARSER_WZF01,
    "wyner_ziv_frame_0": PARSER_WZF01,
    PARSER_DP1_SHORT: PARSER_DP1,
    "pretrained_driving_prior": PARSER_DP1,
    "driving_prior": PARSER_DP1,
    PARSER_C1WMFV1_SHORT: PARSER_C1WMFV1,
    "c1_world_model_foveation": PARSER_C1WMFV1,
    "world_model_foveation": PARSER_C1WMFV1,
    PARSER_WZ1_SHORT: PARSER_WZ1,
    "wyner_ziv_cooperative_receiver": PARSER_WZ1,
    "cooperative_receiver": PARSER_WZ1,
    PARSER_Z4CR1_SHORT: PARSER_Z4CR1,
    "z4_cooperative_receiver_loss": PARSER_Z4CR1,
    "cooperative_receiver_loss": PARSER_Z4CR1,
    PARSER_Z5PCWM1_SHORT: PARSER_Z5PCWM1,
    "z5_predictive_coding_world_model": PARSER_Z5PCWM1,
    "predictive_coding_world_model": PARSER_Z5PCWM1,
    PARSER_TT5L_SHORT: PARSER_TT5L,
    "time_traveler_l5_autonomy": PARSER_TT5L,
    "time_traveler": PARSER_TT5L,
}
PARSER_CHOICES = (
    PARSER_AUTO,
    PARSER_PR101,
    PARSER_PR101_FEC6,
    PARSER_PR101_FEC6_SHORT,
    PARSER_PR103,
    PARSER_PR106,
    PARSER_A2K1,
    PARSER_A5FC,
    PARSER_CPLX1,
    PARSER_IBPS1,
    PARSER_IBPS1_SHORT,
    PARSER_D1POLY1,
    PARSER_D1POLY1_SHORT,
    PARSER_WZF01,
    PARSER_WZF01_SHORT,
    PARSER_DP1,
    PARSER_DP1_SHORT,
    PARSER_C1WMFV1,
    PARSER_C1WMFV1_SHORT,
    PARSER_WZ1,
    PARSER_WZ1_SHORT,
    PARSER_Z4CR1,
    PARSER_Z4CR1_SHORT,
    PARSER_Z5PCWM1,
    PARSER_Z5PCWM1_SHORT,
    PARSER_TT5L,
    PARSER_TT5L_SHORT,
)

PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_TOTAL_KNOWN_LEN = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN + 607
A2K1_MAGIC = b"A2K1"
A2K1_HEADER_LEN = 8
A5FC_MAGIC = b"A5FC"
A5FC_HEADER_LEN = 20
CPLX1_MAGIC = b"CPLX"
CPLX1_HEADER_LEN = 8
CPLX1_BYTE_MAP_LEN_FIELD_LEN = 2

PR101_SECTION_ROLES = {
    "decoder_compact_brotli_streams": "decoder_weight_stream",
    "latents_raw_lzma_delta_u8": "latent_stream",
    "sidecar_dim_delta_huffman_enum": "sidecar_or_correction_stream",
}
PR101_FEC6_SECTION_ROLES = {
    "fp11_magic": "control_or_metadata",
    "source_len_u32le": "control_or_metadata",
    "source_decoder_compact_brotli_streams": "decoder_weight_stream",
    "source_latents_raw_lzma_delta_u8": "latent_stream",
    "source_sidecar_dim_delta_huffman_enum": "sidecar_or_correction_stream",
    "selector_len_u16le": "control_or_metadata",
    "selector_fec6_fixed_huffman_k16_header": "entropy_model_or_range_stream",
    "selector_fec6_fixed_huffman_k16_bitstream": "sidecar_or_correction_stream",
    "selector_fec8_markov_header": "entropy_model_or_range_stream",
    "selector_fec8_markov_bitstream": "sidecar_or_correction_stream",
    "selector_fec10_hybrid_adaptive_blend_header": "entropy_model_or_range_stream",
    "selector_fec10_hybrid_adaptive_blend_bitstream": "sidecar_or_correction_stream",
    "selector_dqs1_selective_runtime_tail": "sidecar_or_correction_stream",
}
PR103_SECTION_ROLES = {
    "scales_fp16": "control_or_metadata",
    "non_ac_weights_brotli": "decoder_weight_stream",
    "ac_histograms_brotli": "entropy_model_or_range_stream",
    "merged_range_coded_weights_and_hi_latents": "entropy_model_or_range_stream",
    "latent_min_scale_fp16": "control_or_metadata",
    "latent_low_bytes_brotli": "latent_stream",
    "latent_hi_histogram_brotli": "entropy_model_or_range_stream",
    "sidecar_corrections_brotli": "sidecar_or_correction_stream",
}
PR106_SECTION_ROLES = {
    "packed_header_ff_len24": "control_or_metadata",
    "decoder_packed_brotli": "decoder_weight_stream",
    "latents_and_sidecar_brotli": "latent_stream",
}
A2K1_SECTION_ROLES = {
    "a2k1_magic": "control_or_metadata",
    "decoder_len_u32le": "control_or_metadata",
    "decoder_blob": "decoder_weight_stream",
    "latent_blob": "latent_stream",
    "sidecar_blob": "sidecar_or_correction_stream",
}
A5FC_SECTION_ROLES = {
    "a5fc_magic": "control_or_metadata",
    "decoder_len_u32le": "control_or_metadata",
    "latent_meta_len_u32le": "control_or_metadata",
    "q_bits_sideinfo_len_u32le": "control_or_metadata",
    "latent_wire_len_u32le": "control_or_metadata",
    "decoder_blob": "decoder_weight_stream",
    "latent_min_scale_fp16": "control_or_metadata",
    "q_bits_sideinfo_3bit": "entropy_model_or_range_stream",
    "latent_wire_variable_width": "latent_stream",
    "sidecar_blob": "sidecar_or_correction_stream",
}
CPLX1_SECTION_ROLES = {
    "cplx_magic": "control_or_metadata",
    "decoder_section_len_u32le": "control_or_metadata",
    "byte_maps_json_len_u16le": "control_or_metadata",
    "byte_maps_json": "decoder_byte_map_metadata",
    "op1_inner_blob": "decoder_weight_stream",
    "latent_blob": "latent_stream",
    "sidecar_blob": "sidecar_or_correction_stream",
}
IBPS1_MAGIC_PREFIX = b"IBPS"
D1POLY1_MAGIC_PREFIX = b"D1PY"
WZF01_MAGIC_PREFIX = b"WZF\x01"
DP1_MAGIC_PREFIX = b"DP1\x00"  # DP1_COMPOSITION_OK:parser-dispatch-magic-constant-not-runtime-composition-call-FIX-WAVE-R1-out-of-scope-cleanup
C1WMFV1_MAGIC_PREFIX = b"WMF\x01"
WZ1_MAGIC_PREFIX = b"WZ1\x00"
Z4CR1_MAGIC_PREFIX = b"Z4CR"
Z5PCWM1_MAGIC_PREFIX = b"Z5WM"
TT5L_MAGIC_PREFIX = b"TT5L"
FP11_MAGIC_PREFIX = b"FP11"
FEC6_MAGIC_PREFIX = b"FEC6"
FEC8_MAGIC_PREFIX = b"FEC8"
FECA_MAGIC_PREFIX = b"FECa"


class HnervPacketSectionManifestError(ValueError):
    """Raised when a packet-section manifest cannot be emitted or validated."""


def build_packet_section_manifest(
    archive_path: str | Path,
    *,
    label: str | None = None,
    parser: str = PARSER_AUTO,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Emit and validate one no-score parser-section manifest."""

    manifest = _emit_packet_section_manifest(
        archive_path,
        label=label,
        parser=parser,
        repo_root=repo_root,
    )
    blockers = validate_packet_section_manifest(manifest, repo_root=repo_root)
    manifest["parser_section_gate"] = _gate(blockers)
    return manifest


def build_packet_section_manifest_batch(
    archives: Iterable[tuple[str, str | Path, str]],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Emit a deterministic no-score batch manifest for labeled archives."""

    records = [
        build_packet_section_manifest(path, label=label, parser=parser, repo_root=repo_root)
        for label, path, parser in archives
    ]
    blockers: list[str] = []
    for record in records:
        label = str(record.get("label") or "unknown")
        for blocker in validate_packet_section_manifest(record, repo_root=repo_root):
            blockers.append(f"{label}:{blocker}")
    if not records:
        blockers.append("batch_missing_records")
    return {
        "schema": BATCH_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "score_claim": False,
        "score_evidence_grade": "invalid_no_score",
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "parser_section_gate": _gate(blockers),
        "records": records,
    }


def validate_packet_section_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    recompute_archive: bool = True,
) -> list[str]:
    """Return fail-closed blockers for one parser-section manifest."""

    blockers = _validate_manifest_shape(manifest)
    if not recompute_archive:
        return blockers
    archive_path = _archive_path_from_manifest(manifest, repo_root=repo_root)
    if archive_path is None:
        return [*blockers, "archive_path_missing"]
    if not archive_path.is_file():
        return [*blockers, "archive_path_missing_on_disk"]
    parser_name = _parser_name_from_manifest(manifest)
    if parser_name is None:
        return [*blockers, "parser_name_missing"]
    try:
        rebuilt = _emit_packet_section_manifest(
            archive_path,
            label=str(manifest.get("label") or ""),
            parser=parser_name,
            repo_root=repo_root,
        )
    except (HnervPacketSectionManifestError, HnervLowlevelPackError, HnervPr103LcAcSchemaError) as exc:
        return [*blockers, f"archive_reparse_failed:{exc}"]
    blockers.extend(_compare_rebuilt_manifest(manifest, rebuilt))
    return blockers


def validate_packet_section_manifest_batch(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Return fail-closed blockers for a batch or single manifest payload."""

    if payload.get("schema") == MANIFEST_SCHEMA:
        return validate_packet_section_manifest(payload, repo_root=repo_root)
    blockers: list[str] = []
    if payload.get("schema") != BATCH_SCHEMA:
        blockers.append("batch_schema_mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("batch_schema_version_mismatch")
    if payload.get("score_claim") is not False:
        blockers.append("batch_score_claim_not_false")
    if payload.get("score_evidence_grade") != "invalid_no_score":
        blockers.append("batch_score_evidence_grade_not_invalid_no_score")
    if payload.get("dispatch_attempted") is not False:
        blockers.append("batch_dispatch_attempted_not_false")
    if payload.get("gpu_required") is not False:
        blockers.append("batch_gpu_required_not_false")
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("batch_ready_for_exact_eval_dispatch_not_false")
    blockers.extend(_validate_parser_section_gate(payload.get("parser_section_gate"), "batch"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return [*blockers, "batch_records_missing"]
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            blockers.append(f"record_not_object:{index}")
            continue
        label = str(record.get("label") or index)
        for blocker in validate_packet_section_manifest(record, repo_root=repo_root):
            blockers.append(f"{label}:{blocker}")
    return blockers


def render_manifest_summary(payload: Mapping[str, Any]) -> str:
    """Render a compact text summary for CLI output."""

    if payload.get("schema") == BATCH_SCHEMA:
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        lines = [
            "HNeRV packet-section manifest batch",
            f"score_claim: {payload.get('score_claim')}",
            f"parser_section_gate_ready: {_gate_ready(payload)}",
            f"records: {len(records)}",
        ]
        for record in records:
            if isinstance(record, Mapping):
                lines.append(_record_summary_line(record))
        return "\n".join(lines) + "\n"
    return "\n".join(
        [
            "HNeRV packet-section manifest",
            f"score_claim: {payload.get('score_claim')}",
            f"parser_section_gate_ready: {_gate_ready(payload)}",
            _record_summary_line(payload),
        ]
    ) + "\n"


def dumps_manifest(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON for a manifest or batch."""

    return json_text(payload)


def _emit_packet_section_manifest(
    archive_path: str | Path,
    *,
    label: str | None,
    parser: str,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    parser = _normalize_parser_name(parser)
    if parser not in PARSER_CHOICES:
        raise HnervPacketSectionManifestError(f"unknown parser {parser!r}")
    archive = Path(archive_path)
    single = read_strict_single_member_zip(archive)
    member_meta = _zip_member_metadata(archive)
    payload_sha256 = sha256_bytes(single.payload)
    parser_name = _infer_parser(
        parser,
        archive_path=archive,
        label=label or "",
        member_name=single.member_name,
        payload=single.payload,
    )
    parser_input, wrapper_meta = _parser_input_payload(
        parser_name=parser_name,
        member_payload=single.payload,
    )
    sections = _parse_sections(parser_name, parser_input)
    coverage = _coverage_record(sections, payload_bytes=len(parser_input))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "score_claim": False,
        "score_evidence_grade": "invalid_no_score",
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "label": label or archive.stem,
        "archive": {
            "path": repo_relative(archive, repo_root or Path.cwd()),
            "bytes": single.archive_bytes,
            "sha256": single.archive_sha256,
        },
        "member": {
            "name": single.member_name,
            "bytes": single.member_bytes,
            "sha256": payload_sha256,
            "zip_compress_type": member_meta["compress_type"],
            "zip_compress_size": member_meta["compress_size"],
            "zip_crc": member_meta["crc"],
        },
        "parser": {
            "name": parser_name,
            "requested": parser,
            "confidence": _parser_confidence(parser_name),
        },
        "parser_input": {
            "kind": "pr106_sidecar_inner_payload" if wrapper_meta else "member_payload",
            "bytes": len(parser_input),
            "sha256": sha256_bytes(parser_input),
            "offset_base": "parser_input",
        },
        "parser_section_manifest": _gate3_parser_section_manifest(sections),
        "sections": sections,
        "coverage": coverage,
        "parser_section_gate": _gate(_validate_manifest_shape_without_gate(sections, coverage)),
        "notes": [
            "parser-section custody only",
            "no component score, score movement, or dispatch authorization is claimed",
        ],
    }
    if wrapper_meta is not None:
        manifest["pr106_sidecar_wrapper"] = wrapper_meta
    return manifest


def _zip_member_metadata(archive: Path) -> dict[str, int]:
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise HnervPacketSectionManifestError(f"expected one ZIP member, got {len(infos)}")
        info = infos[0]
        return {
            "compress_type": int(info.compress_type),
            "compress_size": int(info.compress_size),
            "crc": int(info.CRC),
        }


def _infer_parser(
    requested: str,
    *,
    archive_path: Path,
    label: str,
    member_name: str,
    payload: bytes,
) -> str:
    requested = _normalize_parser_name(requested)
    if requested != PARSER_AUTO:
        return requested
    text = f"{label} {archive_path.as_posix()} {member_name}".lower()
    if (
        "pr101_fec6" in text
        or "fec6_fixed_huffman" in text
        or "frame_exploit_selector" in text
    ):
        return PARSER_PR101_FEC6
    if "pr106" in text or "belt_and_suspenders" in text:
        return PARSER_PR106
    if "pr103" in text or "hnerv_lc_ac" in text:
        return PARSER_PR103
    if "a2k1" in text or "a2_lossy" in text or "lossy_coarsening" in text:
        return PARSER_A2K1
    if "a5fc" in text or "frame_conditional" in text:
        return PARSER_A5FC
    if "cplx1" in text or "cplx" in text or "op1_finalizer" in text:
        return PARSER_CPLX1
    if "ibps1" in text or "c6_e4_mdl_ibps" in text or "mdl_ibps" in text:
        return PARSER_IBPS1
    if "d1poly1" in text or "d1_segnet_margin_polytope" in text or "d1_polytope" in text:
        return PARSER_D1POLY1
    if "wzf01" in text or "d4_wyner_ziv_frame_0" in text or "wyner_ziv_frame_0" in text:
        return PARSER_WZF01
    if "dp1_" in text or "pretrained_driving_prior" in text or "driving_prior" in text:
        return PARSER_DP1
    if "c1wmfv1" in text or "c1_world_model_foveation" in text or "world_model_foveation" in text:
        return PARSER_C1WMFV1
    if (
        "wyner_ziv_cooperative_receiver" in text
        or ("cooperative_receiver" in text and "z4" not in text)
        or "wz1_" in text
    ):
        return PARSER_WZ1
    if "z4cr1" in text or "z4_cooperative_receiver_loss" in text or "cooperative_receiver_loss" in text:
        return PARSER_Z4CR1
    if (
        "z5pcwm1" in text
        or "z5_predictive_coding_world_model" in text
        or "predictive_coding_world_model" in text
    ):
        return PARSER_Z5PCWM1
    if "tt5l" in text or "time_traveler_l5_autonomy" in text or "time_traveler" in text:
        return PARSER_TT5L
    if payload.startswith(A2K1_MAGIC):
        return PARSER_A2K1
    if payload.startswith(A5FC_MAGIC):
        return PARSER_A5FC
    if payload.startswith(CPLX1_MAGIC):
        return PARSER_CPLX1
    if payload.startswith(IBPS1_MAGIC_PREFIX):
        return PARSER_IBPS1
    if payload.startswith(D1POLY1_MAGIC_PREFIX):
        return PARSER_D1POLY1
    if payload.startswith(WZF01_MAGIC_PREFIX):
        return PARSER_WZF01
    if payload.startswith(DP1_MAGIC_PREFIX):
        return PARSER_DP1
    if payload.startswith(C1WMFV1_MAGIC_PREFIX):
        return PARSER_C1WMFV1
    if payload.startswith(WZ1_MAGIC_PREFIX):
        return PARSER_WZ1
    if payload.startswith(Z4CR1_MAGIC_PREFIX):
        return PARSER_Z4CR1
    if payload.startswith(Z5PCWM1_MAGIC_PREFIX):
        return PARSER_Z5PCWM1
    if payload.startswith(TT5L_MAGIC_PREFIX):
        return PARSER_TT5L
    if payload.startswith(FP11_MAGIC_PREFIX):
        return PARSER_PR101_FEC6
    if "pr101" in text or "hnerv_ft_microcodec" in text:
        return PARSER_PR101
    if len(payload) >= 4 and payload[0] == 0xFF:
        return PARSER_PR106
    if len(payload) >= 8 and payload[0] == PR106_SIDECAR_MAGIC:
        return PARSER_PR106
    if len(payload) == PR101_TOTAL_KNOWN_LEN:
        return PARSER_PR101
    if len(payload) >= PUBLIC_PR103_LAYOUT.fixed_bytes:
        return PARSER_PR103
    raise HnervPacketSectionManifestError("could not infer HNeRV packet parser")


def _normalize_parser_name(parser_name: str) -> str:
    return PARSER_ALIASES.get(parser_name, parser_name)


def _parser_input_payload(
    *,
    parser_name: str,
    member_payload: bytes,
) -> tuple[bytes, dict[str, Any] | None]:
    if parser_name != PARSER_PR106:
        return member_payload, None
    if not member_payload or member_payload[0] != PR106_SIDECAR_MAGIC:
        return member_payload, None
    try:
        packet = parse_pr106_sidecar_packet(member_payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(
            f"PR106 sidecar wrapper parse failed: {exc}"
        ) from exc
    framing_meta = packet.framing_meta or b""
    wrapper_meta = {
        "kind": "pr106_sidecar_wrapper",
        "magic": f"0x{PR106_SIDECAR_MAGIC:02X}",
        "format_id": f"0x{packet.format_id:02X}",
        "sidecar_kind": packet.sidecar_kind,
        "outer_member_bytes": len(member_payload),
        "outer_member_sha256": sha256_bytes(member_payload),
        "inner_pr106_bytes": len(packet.pr106_bytes),
        "inner_pr106_sha256": sha256_bytes(packet.pr106_bytes),
        "sidecar_payload_bytes": len(packet.sidecar_payload),
        "sidecar_payload_sha256": sha256_bytes(packet.sidecar_payload),
        "framing_meta_bytes": len(framing_meta),
        "framing_meta_sha256": sha256_bytes(framing_meta) if framing_meta else None,
        "score_claim": False,
        "dispatch_attempted": False,
    }
    return packet.pr106_bytes, wrapper_meta


def _parse_sections(parser_name: str, payload: bytes) -> list[dict[str, Any]]:
    if parser_name == PARSER_PR101:
        return _parse_pr101_sections(payload)
    if parser_name == PARSER_PR101_FEC6:
        return _parse_pr101_fec6_sections(payload)
    if parser_name == PARSER_PR103:
        return _parse_pr103_sections(payload)
    if parser_name == PARSER_PR106:
        return _parse_pr106_sections(payload)
    if parser_name == PARSER_A2K1:
        return _parse_a2k1_sections(payload)
    if parser_name == PARSER_A5FC:
        return _parse_a5fc_sections(payload)
    if parser_name == PARSER_CPLX1:
        return _parse_cplx1_sections(payload)
    if parser_name == PARSER_IBPS1:
        return _parse_ibps1_sections(payload)
    if parser_name == PARSER_D1POLY1:
        return _parse_d1poly1_sections(payload)
    if parser_name == PARSER_WZF01:
        return _parse_wzf01_sections(payload)
    if parser_name == PARSER_DP1:
        return _parse_dp1_sections(payload)
    if parser_name == PARSER_C1WMFV1:
        return _parse_c1wmfv1_sections(payload)
    if parser_name == PARSER_WZ1:
        return _parse_wz1_sections(payload)
    if parser_name == PARSER_Z4CR1:
        return _parse_z4cr1_sections(payload)
    if parser_name == PARSER_Z5PCWM1:
        return _parse_z5pcwm1_sections(payload)
    if parser_name == PARSER_TT5L:
        return _parse_tt5l_sections(payload)
    raise HnervPacketSectionManifestError(f"unknown parser {parser_name!r}")


def _parse_pr101_sections(payload: bytes) -> list[dict[str, Any]]:
    minimum = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    if len(payload) < minimum:
        raise HnervPacketSectionManifestError(
            f"PR101 payload too short: expected at least {minimum} bytes, got {len(payload)}"
        )
    specs = [
        ("decoder_compact_brotli_streams", 0, PR101_DECODER_BLOB_LEN),
        ("latents_raw_lzma_delta_u8", PR101_DECODER_BLOB_LEN, minimum),
        ("sidecar_dim_delta_huffman_enum", minimum, len(payload)),
    ]
    return [
        _section_record(
            index,
            name=name,
            offset=start,
            data=payload[start:end],
            role=PR101_SECTION_ROLES[name],
        )
        for index, (name, start, end) in enumerate(specs)
    ]


def _parse_pr101_fec6_sections(payload: bytes) -> list[dict[str, Any]]:
    if len(payload) < 8:
        raise HnervPacketSectionManifestError(
            f"PR101 FEC6 payload too short for FP11 header: got {len(payload)} bytes"
        )
    if payload[:4] != FP11_MAGIC_PREFIX:
        raise HnervPacketSectionManifestError("PR101 FEC6 payload missing FP11 magic")
    source_len = int.from_bytes(payload[4:8], "little")
    source_start = 8
    source_end = source_start + source_len
    selector_len_start = source_end
    selector_start = selector_len_start + 2
    if source_len < PR101_TOTAL_KNOWN_LEN:
        raise HnervPacketSectionManifestError(
            f"PR101 FEC6 source payload too short: expected at least {PR101_TOTAL_KNOWN_LEN}, got {source_len}"
        )
    if selector_start > len(payload):
        raise HnervPacketSectionManifestError(
            f"PR101 FEC6 payload truncated before selector length: source_len={source_len}"
        )
    selector_len = int.from_bytes(payload[selector_len_start:selector_start], "little")
    selector_end = selector_start + selector_len
    dqs1_tail = b""
    if selector_end != len(payload):
        dqs1_tail = payload[selector_end:]
        if not dqs1_tail.startswith(DQS1_MAGIC_PREFIX):
            raise HnervPacketSectionManifestError(
                "PR101 FEC6 selector length mismatch: "
                f"selector_len={selector_len} payload_bytes={len(payload)}"
            )
        try:
            unpack_dqs1_payload(dqs1_tail)
        except DecoderQSelectiveRuntimePacketError as exc:
            raise HnervPacketSectionManifestError(
                f"PR101 FEC6 DQS1 tail parse failed: {exc}"
            ) from exc
    if selector_end > len(payload):
        raise HnervPacketSectionManifestError(
            f"PR101 FEC6 selector length mismatch: selector_len={selector_len} payload_bytes={len(payload)}"
        )
    if selector_len < 6:
        raise HnervPacketSectionManifestError(
            f"PR101 FEC6 selector payload too short: {selector_len}"
        )
    selector_magic = payload[selector_start : selector_start + 4]
    selector_header_len: int
    selector_header_name: str
    selector_bitstream_name: str
    if selector_magic == FEC6_MAGIC_PREFIX:
        selector_header_len = 6
        selector_header_name = "selector_fec6_fixed_huffman_k16_header"
        selector_bitstream_name = "selector_fec6_fixed_huffman_k16_bitstream"
        n_pairs = int.from_bytes(
            payload[selector_start + 4 : selector_start + 6],
            "little",
        )
    elif selector_magic == FEC8_MAGIC_PREFIX:
        selector_header_len = 8
        selector_header_name = "selector_fec8_markov_header"
        selector_bitstream_name = "selector_fec8_markov_bitstream"
        if selector_len < selector_header_len:
            raise HnervPacketSectionManifestError(
                f"PR101 FEC8 selector payload too short: {selector_len}"
            )
        n_pairs = int.from_bytes(
            payload[selector_start + 6 : selector_start + 8],
            "little",
        )
    elif selector_magic == FECA_MAGIC_PREFIX:
        selector_header_len = 8
        selector_header_name = "selector_fec10_hybrid_adaptive_blend_header"
        selector_bitstream_name = "selector_fec10_hybrid_adaptive_blend_bitstream"
        if selector_len < selector_header_len:
            raise HnervPacketSectionManifestError(
                f"PR101 FEC10 selector payload too short: {selector_len}"
            )
        n_pairs = int.from_bytes(
            payload[selector_start + 6 : selector_start + 8],
            "little",
        )
    else:
        raise HnervPacketSectionManifestError(
            f"PR101 frame selector magic mismatch: {selector_magic!r}"
        )
    if n_pairs != 600:
        raise HnervPacketSectionManifestError(
            f"PR101 frame selector n_pairs mismatch: expected 600, got {n_pairs}"
        )
    decoder_start = source_start
    latent_start = decoder_start + PR101_DECODER_BLOB_LEN
    source_sidecar_start = latent_start + PR101_LATENT_BLOB_LEN
    specs = [
        ("fp11_magic", 0, payload[:4]),
        ("source_len_u32le", 4, payload[4:8]),
        (
            "source_decoder_compact_brotli_streams",
            decoder_start,
            payload[decoder_start:latent_start],
        ),
        (
            "source_latents_raw_lzma_delta_u8",
            latent_start,
            payload[latent_start:source_sidecar_start],
        ),
        (
            "source_sidecar_dim_delta_huffman_enum",
            source_sidecar_start,
            payload[source_sidecar_start:source_end],
        ),
        (
            "selector_len_u16le",
            selector_len_start,
            payload[selector_len_start:selector_start],
        ),
        (
            selector_header_name,
            selector_start,
            payload[selector_start : selector_start + selector_header_len],
        ),
        (
            selector_bitstream_name,
            selector_start + selector_header_len,
            payload[selector_start + selector_header_len : selector_end],
        ),
    ]
    if dqs1_tail:
        specs.append(
            (
                "selector_dqs1_selective_runtime_tail",
                selector_end,
                dqs1_tail,
            )
        )
    return [
        _section_record(
            index,
            name=name,
            offset=offset,
            data=data,
            role=PR101_FEC6_SECTION_ROLES[name],
        )
        for index, (name, offset, data) in enumerate(specs)
    ]


def _parse_pr103_sections(payload: bytes) -> list[dict[str, Any]]:
    parsed = parse_pr103_lc_ac_payload(payload)
    records = []
    for index, section in enumerate(parsed.sections):
        records.append(
            _section_record(
                index,
                name=section.name,
                offset=section.start,
                data=section.data,
                role=PR103_SECTION_ROLES.get(section.name, "opaque_payload_stream"),
            )
        )
    return records


def _parse_pr106_sections(payload: bytes) -> list[dict[str, Any]]:
    packed = parse_ff_packed_brotli_hnerv(payload)
    decoder_offset = len(packed.header)
    tail_offset = decoder_offset + len(packed.decoder_packed_brotli)
    specs = [
        ("packed_header_ff_len24", 0, packed.header),
        ("decoder_packed_brotli", decoder_offset, packed.decoder_packed_brotli),
        ("latents_and_sidecar_brotli", tail_offset, packed.latents_and_sidecar_brotli),
    ]
    return [
        _section_record(
            index,
            name=name,
            offset=offset,
            data=data,
            role=PR106_SECTION_ROLES[name],
        )
        for index, (name, offset, data) in enumerate(specs)
    ]


def _parse_a2k1_sections(payload: bytes) -> list[dict[str, Any]]:
    if len(payload) < A2K1_HEADER_LEN + PR101_LATENT_BLOB_LEN:
        raise HnervPacketSectionManifestError(
            f"A2K1 payload too short for header+latent window: got {len(payload)} bytes"
        )
    if payload[:4] != A2K1_MAGIC:
        raise HnervPacketSectionManifestError("A2K1 payload missing magic")
    decoder_len = int.from_bytes(payload[4:A2K1_HEADER_LEN], "little")
    if decoder_len <= 0:
        raise HnervPacketSectionManifestError(f"A2K1 decoder length invalid: {decoder_len}")
    decoder_start = A2K1_HEADER_LEN
    decoder_end = decoder_start + decoder_len
    latent_start = decoder_end
    latent_end = latent_start + PR101_LATENT_BLOB_LEN
    if latent_end >= len(payload):
        raise HnervPacketSectionManifestError(
            "A2K1 decoder length truncates latent/sidecar window: "
            f"decoder_len={decoder_len} payload_bytes={len(payload)}"
        )
    specs = [
        ("a2k1_magic", 0, payload[:4]),
        ("decoder_len_u32le", 4, payload[4:A2K1_HEADER_LEN]),
        ("decoder_blob", decoder_start, payload[decoder_start:decoder_end]),
        ("latent_blob", latent_start, payload[latent_start:latent_end]),
        ("sidecar_blob", latent_end, payload[latent_end:]),
    ]
    return [
        _section_record(
            index,
            name=name,
            offset=offset,
            data=data,
            role=A2K1_SECTION_ROLES[name],
        )
        for index, (name, offset, data) in enumerate(specs)
    ]


def _parse_a5fc_sections(payload: bytes) -> list[dict[str, Any]]:
    if len(payload) < A5FC_HEADER_LEN:
        raise HnervPacketSectionManifestError(
            f"A5FC payload too short for header: got {len(payload)} bytes"
        )
    if payload[:4] != A5FC_MAGIC:
        raise HnervPacketSectionManifestError("A5FC payload missing magic")
    decoder_len = int.from_bytes(payload[4:8], "little")
    latent_meta_len = int.from_bytes(payload[8:12], "little")
    q_bits_sideinfo_len = int.from_bytes(payload[12:16], "little")
    latent_wire_len = int.from_bytes(payload[16:20], "little")
    if decoder_len <= 0:
        raise HnervPacketSectionManifestError(
            f"A5FC decoder length invalid: {decoder_len}"
        )
    if latent_meta_len <= 0:
        raise HnervPacketSectionManifestError(
            f"A5FC latent meta length invalid: {latent_meta_len}"
        )
    if q_bits_sideinfo_len <= 0:
        raise HnervPacketSectionManifestError(
            f"A5FC q-bit side-info length invalid: {q_bits_sideinfo_len}"
        )
    if latent_wire_len <= 0:
        raise HnervPacketSectionManifestError(
            f"A5FC latent wire length invalid: {latent_wire_len}"
        )
    decoder_start = A5FC_HEADER_LEN
    decoder_end = decoder_start + decoder_len
    latent_meta_start = decoder_end
    latent_meta_end = latent_meta_start + latent_meta_len
    q_bits_start = latent_meta_end
    q_bits_end = q_bits_start + q_bits_sideinfo_len
    latent_wire_start = q_bits_end
    latent_wire_end = latent_wire_start + latent_wire_len
    if latent_wire_end >= len(payload):
        raise HnervPacketSectionManifestError(
            "A5FC lengths leave no sidecar section: "
            f"payload_bytes={len(payload)} latent_wire_end={latent_wire_end}"
        )
    specs = [
        ("a5fc_magic", 0, payload[:4]),
        ("decoder_len_u32le", 4, payload[4:8]),
        ("latent_meta_len_u32le", 8, payload[8:12]),
        ("q_bits_sideinfo_len_u32le", 12, payload[12:16]),
        ("latent_wire_len_u32le", 16, payload[16:20]),
        ("decoder_blob", decoder_start, payload[decoder_start:decoder_end]),
        (
            "latent_min_scale_fp16",
            latent_meta_start,
            payload[latent_meta_start:latent_meta_end],
        ),
        (
            "q_bits_sideinfo_3bit",
            q_bits_start,
            payload[q_bits_start:q_bits_end],
        ),
        (
            "latent_wire_variable_width",
            latent_wire_start,
            payload[latent_wire_start:latent_wire_end],
        ),
        ("sidecar_blob", latent_wire_end, payload[latent_wire_end:]),
    ]
    return [
        _section_record(
            index,
            name=name,
            offset=offset,
            data=data,
            role=A5FC_SECTION_ROLES[name],
        )
        for index, (name, offset, data) in enumerate(specs)
    ]


def _parse_cplx1_sections(payload: bytes) -> list[dict[str, Any]]:
    minimum = CPLX1_HEADER_LEN + CPLX1_BYTE_MAP_LEN_FIELD_LEN + PR101_LATENT_BLOB_LEN
    if len(payload) < minimum:
        raise HnervPacketSectionManifestError(
            f"CPLX1 payload too short for header+latent window: got {len(payload)} bytes"
        )
    if payload[:4] != CPLX1_MAGIC:
        raise HnervPacketSectionManifestError("CPLX1 payload missing CPLX magic")
    section_total = int.from_bytes(payload[4:CPLX1_HEADER_LEN], "little")
    if section_total < CPLX1_HEADER_LEN + CPLX1_BYTE_MAP_LEN_FIELD_LEN:
        raise HnervPacketSectionManifestError(
            f"CPLX1 decoder section length invalid: {section_total}"
        )
    if section_total > len(payload) - PR101_LATENT_BLOB_LEN:
        raise HnervPacketSectionManifestError(
            "CPLX1 decoder section length leaves no complete PR101 latent window: "
            f"section_total={section_total} payload_bytes={len(payload)}"
        )
    json_len = int.from_bytes(payload[8:10], "little")
    json_start = 10
    json_end = json_start + json_len
    if json_end > section_total:
        raise HnervPacketSectionManifestError(
            f"CPLX1 byte_maps JSON length overflows decoder section: {json_len}"
        )
    op1_start = json_end
    if op1_start >= section_total:
        raise HnervPacketSectionManifestError("CPLX1 op1_inner_blob is empty")
    byte_maps_json = payload[json_start:json_end]
    try:
        byte_maps = json.loads(byte_maps_json.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HnervPacketSectionManifestError("CPLX1 byte_maps JSON is invalid") from exc
    if not isinstance(byte_maps, dict):
        raise HnervPacketSectionManifestError("CPLX1 byte_maps JSON must be an object")
    try:
        {int(key): str(value) for key, value in byte_maps.items()}
    except (TypeError, ValueError) as exc:
        raise HnervPacketSectionManifestError("CPLX1 byte_maps JSON keys must be ints") from exc
    latent_start = section_total
    latent_end = latent_start + PR101_LATENT_BLOB_LEN
    if latent_end >= len(payload):
        raise HnervPacketSectionManifestError("CPLX1 sidecar section is empty")
    specs = [
        ("cplx_magic", 0, payload[:4]),
        ("decoder_section_len_u32le", 4, payload[4:CPLX1_HEADER_LEN]),
        ("byte_maps_json_len_u16le", 8, payload[8:10]),
        ("byte_maps_json", json_start, byte_maps_json),
        ("op1_inner_blob", op1_start, payload[op1_start:section_total]),
        ("latent_blob", latent_start, payload[latent_start:latent_end]),
        ("sidecar_blob", latent_end, payload[latent_end:]),
    ]
    return [
        _section_record(
            index,
            name=name,
            offset=offset,
            data=data,
            role=CPLX1_SECTION_ROLES[name],
        )
        for index, (name, offset, data) in enumerate(specs)
    ]


def _parse_ibps1_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.c6_e4_mdl_ibps.archive import (
            IBPS1_SECTION_ROLES,
            parse_ibps1_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"IBPS1 canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_ibps1_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"IBPS1 parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=IBPS1_SECTION_ROLES[name],
            )
        )
    return records


def _parse_d1poly1_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.d1_segnet_margin_polytope.archive import (
            D1POLY1_SECTION_ROLES,
            parse_d1poly1_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"D1POLY1 canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_d1poly1_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"D1POLY1 parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=D1POLY1_SECTION_ROLES[name],
            )
        )
    return records


def _parse_wzf01_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.d4_wyner_ziv_frame_0.archive import (
            WZF01_SECTION_ROLES,
            parse_wzf01_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"WZF01 canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_wzf01_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"WZF01 parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=WZF01_SECTION_ROLES[name],
            )
        )
    return records


def _parse_dp1_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.pretrained_driving_prior.archive import (
            DP1_SECTION_ROLES,
            parse_dp1_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"DP1 canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_dp1_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"DP1 parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=DP1_SECTION_ROLES[name],
            )
        )
    return records


def _parse_c1wmfv1_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.c1_world_model_foveation.archive import (
            C1WMFV1_SECTION_ROLES,
            parse_c1wmfv1_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"C1WMFV1 canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_c1wmfv1_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"C1WMFV1 parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=C1WMFV1_SECTION_ROLES[name],
            )
        )
    return records


def _parse_wz1_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.wyner_ziv_cooperative_receiver.archive import (
            WZ1_SECTION_ROLES,
            parse_wz1_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"WZ1 canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_wz1_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"WZ1 parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=WZ1_SECTION_ROLES[name],
            )
        )
    return records


def _parse_z4cr1_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.z4_cooperative_receiver_loss.archive import (
            Z4CR1_SECTION_ROLES,
            parse_z4cr1_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"Z4CR1 canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_z4cr1_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"Z4CR1 parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=Z4CR1_SECTION_ROLES[name],
            )
        )
    return records


def _parse_z5pcwm1_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.z5_predictive_coding_world_model.archive import (
            Z5PCWM1_SECTION_ROLES,
            parse_z5pcwm1_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"Z5PCWM1 canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_z5pcwm1_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"Z5PCWM1 parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=Z5PCWM1_SECTION_ROLES[name],
            )
        )
    return records


def _parse_tt5l_sections(payload: bytes) -> list[dict[str, Any]]:
    try:
        from tac.substrates.time_traveler_l5_autonomy.archive import (
            TT5L_SECTION_ROLES,
            parse_tt5l_archive_bytes,
        )
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise HnervPacketSectionManifestError(
            f"TT5L canonical parser import failed: {exc}"
        ) from exc
    try:
        section_map = parse_tt5l_archive_bytes(payload)
    except ValueError as exc:
        raise HnervPacketSectionManifestError(f"TT5L parse failed: {exc}") from exc
    records = []
    for index, (name, (offset, length)) in enumerate(section_map.items()):
        records.append(
            _section_record(
                index,
                name=name,
                offset=offset,
                data=payload[offset : offset + length],
                role=TT5L_SECTION_ROLES[name],
            )
        )
    return records


def _section_record(index: int, *, name: str, offset: int, data: bytes, role: str) -> dict[str, Any]:
    end = offset + len(data)
    return {
        "index": index,
        "name": name,
        "offset": offset,
        "start": offset,
        "end": end,
        "length": len(data),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "entropy_bits_per_byte": _byte_entropy_bits_per_byte(data),
        "optimization_role": role,
        "score_claim": False,
    }


def _byte_entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    total = len(data)
    counts = Counter(data)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round(float(entropy), 12)


def _gate3_parser_section_manifest(sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = [str(section["name"]) for section in sections]
    offsets = [int(section["offset"]) for section in sections]
    lengths = [int(section["length"]) for section in sections]
    hashes = [str(section["sha256"]) for section in sections]
    entropy = [float(section.get("entropy_bits_per_byte", 0.0)) for section in sections]
    return {
        "offsets": offsets,
        "lengths": lengths,
        "section_names": names,
        "section_sha256s": hashes,
        "entropy_estimates": entropy,
        "old_new_section_boundaries": {
            name: {"old": [offset, offset + length], "new": [offset, offset + length]}
            for name, offset, length in zip(names, offsets, lengths, strict=True)
        },
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _coverage_record(sections: Sequence[Mapping[str, Any]], *, payload_bytes: int) -> dict[str, Any]:
    contiguous = True
    cursor = 0
    total = 0
    for section in sections:
        offset = int(section.get("offset") or 0)
        length = int(section.get("length") or 0)
        if offset != cursor or length < 0:
            contiguous = False
        cursor = offset + length
        total += length
    return {
        "payload_bytes": payload_bytes,
        "section_bytes": total,
        "covers_payload": total == payload_bytes and cursor == payload_bytes,
        "contiguous": contiguous and cursor == payload_bytes,
        "section_count": len(sections),
    }


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> list[str]:
    blockers = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        blockers.append("schema_mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        blockers.append("schema_version_mismatch")
    if manifest.get("score_claim") is not False:
        blockers.append("score_claim_not_false")
    if manifest.get("score_evidence_grade") != "invalid_no_score":
        blockers.append("score_evidence_grade_not_invalid_no_score")
    if manifest.get("dispatch_attempted") is not False:
        blockers.append("dispatch_attempted_not_false")
    if manifest.get("gpu_required") is not False:
        blockers.append("gpu_required_not_false")
    if manifest.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("ready_for_exact_eval_dispatch_not_false")
    blockers.extend(_validate_parser_section_gate(manifest.get("parser_section_gate"), "manifest"))
    archive = manifest.get("archive")
    member = manifest.get("member")
    parser = manifest.get("parser")
    sections = manifest.get("sections")
    coverage = manifest.get("coverage")
    if not isinstance(archive, Mapping):
        blockers.append("archive_identity_missing")
    else:
        if not isinstance(archive.get("bytes"), int) or int(archive.get("bytes") or 0) <= 0:
            blockers.append("archive_bytes_invalid")
        if not _is_sha256(archive.get("sha256")):
            blockers.append("archive_sha256_invalid")
    if not isinstance(member, Mapping):
        blockers.append("member_identity_missing")
    else:
        if not isinstance(member.get("name"), str) or not member.get("name"):
            blockers.append("member_name_missing")
        if not isinstance(member.get("bytes"), int) or int(member.get("bytes") or 0) <= 0:
            blockers.append("member_bytes_invalid")
        if not _is_sha256(member.get("sha256")):
            blockers.append("member_sha256_invalid")
    if not isinstance(parser, Mapping) or parser.get("name") not in PARSER_CHOICES[1:]:
        blockers.append("parser_name_invalid")
    if not isinstance(sections, list) or not sections:
        blockers.append("sections_missing")
        sections = []
    if not isinstance(coverage, Mapping):
        blockers.append("coverage_missing")
        coverage = {}
    blockers.extend(_validate_manifest_shape_without_gate(sections, coverage))
    parser_section_manifest = manifest.get("parser_section_manifest")
    if parser_section_manifest is not None:
        blockers.extend(_validate_gate3_parser_section_manifest(parser_section_manifest, sections))
    return blockers


def _validate_parser_section_gate(gate: Any, prefix: str) -> list[str]:
    blockers: list[str] = []
    if not isinstance(gate, Mapping):
        return [f"{prefix}_parser_section_gate_missing"]
    if gate.get("score_claim") is not False:
        blockers.append(f"{prefix}_parser_section_gate_score_claim_not_false")
    if gate.get("dispatch_attempted") is not False:
        blockers.append(f"{prefix}_parser_section_gate_dispatch_attempted_not_false")
    if not isinstance(gate.get("ready"), bool):
        blockers.append(f"{prefix}_parser_section_gate_ready_not_bool")
    if not isinstance(gate.get("blockers"), list):
        blockers.append(f"{prefix}_parser_section_gate_blockers_not_list")
    return blockers


def _validate_manifest_shape_without_gate(
    sections: Sequence[Any],
    coverage: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    cursor = 0
    names: set[str] = set()
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            blockers.append(f"section_not_object:{index}")
            continue
        name = str(section.get("name") or "")
        if not name:
            blockers.append(f"section_name_missing:{index}")
        if name in names:
            blockers.append(f"section_name_duplicate:{name}")
        names.add(name)
        if section.get("index") != index:
            blockers.append(f"section_index_mismatch:{name or index}")
        offset = section.get("offset")
        length = section.get("length")
        end = section.get("end")
        if not isinstance(offset, int) or offset < 0:
            blockers.append(f"section_offset_invalid:{name or index}")
            offset = cursor
        if not isinstance(length, int) or length < 0:
            blockers.append(f"section_length_invalid:{name or index}")
            length = 0
        if section.get("bytes") != length:
            blockers.append(f"section_bytes_length_mismatch:{name or index}")
        if section.get("start") != offset:
            blockers.append(f"section_start_offset_mismatch:{name or index}")
        if end != offset + length:
            blockers.append(f"section_end_mismatch:{name or index}")
        if offset != cursor:
            blockers.append(f"section_not_contiguous:{name or index}")
        cursor = offset + length
        if not _is_sha256(section.get("sha256")):
            blockers.append(f"section_sha256_invalid:{name or index}")
        if section.get("score_claim") is not False:
            blockers.append(f"section_score_claim_not_false:{name or index}")
    if coverage.get("covers_payload") is not True:
        blockers.append("coverage_does_not_cover_payload")
    if coverage.get("contiguous") is not True:
        blockers.append("coverage_not_contiguous")
    if coverage.get("section_count") != len(sections):
        blockers.append("coverage_section_count_mismatch")
    if isinstance(coverage.get("payload_bytes"), int) and coverage.get("payload_bytes") != cursor:
        blockers.append("coverage_payload_bytes_mismatch")
    if isinstance(coverage.get("section_bytes"), int) and coverage.get("section_bytes") != cursor:
        blockers.append("coverage_section_bytes_mismatch")
    return blockers


def _validate_gate3_parser_section_manifest(
    payload: Any,
    sections: Sequence[Any],
) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["parser_section_manifest_not_object"]
    if not sections or not all(isinstance(section, Mapping) for section in sections):
        return []
    expected = _gate3_parser_section_manifest(sections)  # type: ignore[arg-type]
    blockers: list[str] = []
    for key in (
        "offsets",
        "lengths",
        "section_names",
        "section_sha256s",
        "entropy_estimates",
        "old_new_section_boundaries",
    ):
        if payload.get(key) != expected[key]:
            blockers.append(f"parser_section_manifest_{key}_does_not_match_sections")
    if payload.get("score_claim") is not False:
        blockers.append("parser_section_manifest_score_claim_not_false")
    if payload.get("dispatch_attempted") is not False:
        blockers.append("parser_section_manifest_dispatch_attempted_not_false")
    return blockers


def _archive_path_from_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path | None,
) -> Path | None:
    archive = manifest.get("archive")
    if not isinstance(archive, Mapping) or not isinstance(archive.get("path"), str):
        return None
    path = Path(str(archive["path"]))
    if path.is_absolute():
        return path
    if repo_root is not None:
        return Path(repo_root) / path
    return path


def _parser_name_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    parser = manifest.get("parser")
    if not isinstance(parser, Mapping):
        return None
    name = parser.get("name")
    return str(name) if name in PARSER_CHOICES[1:] else None


def _compare_rebuilt_manifest(manifest: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> list[str]:
    blockers = []
    for key in ("archive", "member", "parser", "sections", "coverage"):
        expected = rebuilt.get(key)
        actual = manifest.get(key)
        if key == "parser" and isinstance(expected, Mapping) and isinstance(actual, Mapping):
            expected = {"name": expected.get("name")}
            actual = {"name": actual.get("name")}
        if actual != expected:
            blockers.append(f"{key}_does_not_match_archive")
    return blockers


def _gate(blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "name": "parser-section gate",
        "ready": not blockers,
        "blockers": list(blockers),
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _parser_confidence(parser_name: str) -> str:
    if parser_name == PARSER_PR101:
        return "fixed public PR101 offsets"
    if parser_name == PARSER_PR101_FEC6:
        return "FP11 wrapper plus PR101-family frame selector payload"
    if parser_name == PARSER_PR103:
        return "existing PR103 lc_ac parser"
    if parser_name == PARSER_PR106:
        return "0xff header plus 24-bit decoder length"
    if parser_name == PARSER_A2K1:
        return "A2K1 magic plus u32 decoder length and PR101 latent window"
    if parser_name == PARSER_A5FC:
        return "A5FC magic plus side-info and variable-width PR101 latent wire"
    if parser_name == PARSER_CPLX1:
        return "CPLX magic plus decoder section length and byte_maps JSON"
    if parser_name == PARSER_IBPS1:
        return "IBPS1 magic plus canonical C6 MDL-IBPS section parser"
    if parser_name == PARSER_D1POLY1:
        return "D1POLY1 magic plus canonical D1 SegNet margin polytope section parser"
    if parser_name == PARSER_WZF01:
        return "WZF01 magic plus canonical D4 Wyner-Ziv frame-0 section parser"
    if parser_name == PARSER_DP1:
        return "DP1 magic plus canonical pre-trained driving prior section parser"
    if parser_name == PARSER_C1WMFV1:
        return "C1WMFV1 magic plus canonical world-model-foveation section parser"
    if parser_name == PARSER_WZ1:
        return "WZ1 magic plus canonical Wyner-Ziv cooperative-receiver section parser"
    if parser_name == PARSER_Z4CR1:
        return "Z4CR1 magic plus canonical cooperative-receiver-loss section parser"
    if parser_name == PARSER_Z5PCWM1:
        return "Z5PCWM1 magic plus canonical predictive-coding-world-model section parser"
    if parser_name == PARSER_TT5L:
        return "TT5L magic plus canonical Time-Traveler L5 autonomy section parser"
    return "unknown"


def _gate_ready(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("parser_section_gate")
    return isinstance(gate, Mapping) and gate.get("ready") is True


def _record_summary_line(record: Mapping[str, Any]) -> str:
    archive = record.get("archive") if isinstance(record.get("archive"), Mapping) else {}
    member = record.get("member") if isinstance(record.get("member"), Mapping) else {}
    parser = record.get("parser") if isinstance(record.get("parser"), Mapping) else {}
    sections = record.get("sections") if isinstance(record.get("sections"), list) else []
    return (
        f"- {record.get('label')}: parser={parser.get('name')} "
        f"archive_sha256={archive.get('sha256')} member={member.get('name')} "
        f"member_sha256={member.get('sha256')} sections={len(sections)}"
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


__all__ = [
    "A2K1_MAGIC",
    "A5FC_MAGIC",
    "BATCH_SCHEMA",
    "C1WMFV1_MAGIC_PREFIX",
    "CPLX1_MAGIC",
    "D1POLY1_MAGIC_PREFIX",
    "DP1_MAGIC_PREFIX",
    "FEC6_MAGIC_PREFIX",
    "FEC8_MAGIC_PREFIX",
    "FECA_MAGIC_PREFIX",
    "FP11_MAGIC_PREFIX",
    "IBPS1_MAGIC_PREFIX",
    "MANIFEST_SCHEMA",
    "PARSER_A2K1",
    "PARSER_A5FC",
    "PARSER_AUTO",
    "PARSER_C1WMFV1",
    "PARSER_CHOICES",
    "PARSER_CPLX1",
    "PARSER_D1POLY1",
    "PARSER_DP1",
    "PARSER_IBPS1",
    "PARSER_PR101",
    "PARSER_PR101_FEC6",
    "PARSER_PR103",
    "PARSER_PR106",
    "PARSER_TT5L",
    "PARSER_WZ1",
    "PARSER_WZF01",
    "PARSER_Z4CR1",
    "PARSER_Z5PCWM1",
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "TT5L_MAGIC_PREFIX",
    "WZ1_MAGIC_PREFIX",
    "WZF01_MAGIC_PREFIX",
    "Z4CR1_MAGIC_PREFIX",
    "Z5PCWM1_MAGIC_PREFIX",
    "HnervPacketSectionManifestError",
    "build_packet_section_manifest",
    "build_packet_section_manifest_batch",
    "dumps_manifest",
    "render_manifest_summary",
    "validate_packet_section_manifest",
    "validate_packet_section_manifest_batch",
]
