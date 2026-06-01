# SPDX-License-Identifier: MIT
"""Common representation spine for compact video receiver families.

The spine is a charged-section projection, not score authority.  PR95/HNeRV,
RNeRV/PACT-NeRV, VQ/Tree/Hi/SR-NeRV, C3/Cool-Chic, implicit bases, and
procedural priors can all expose their bytes through the same HPRC section
contract so acquisition compares marginal contest-score value per byte instead
of family-specific folklore.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from tac.hnerv_lowlevel_packer import read_packed_archive_view, read_strict_single_member_zip
from tac.repo_io import sha256_file
from tac.substrates.hprc.archive import (
    HprcPacketConfig,
    HprcSectionKind,
    pack_hprc_packet,
    parse_hprc_packet,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.hprc.pr95_adapter import parse_pr95_hnerv_payload

HPRC_REPRESENTATION_SPINE_SCHEMA = "hprc_representation_spine_packet.v1"
HPRC_REPRESENTATION_SPINE_MANIFEST_SCHEMA = "hprc_representation_spine_manifest.v1"
HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA = "hprc_representation_spine_projection.v1"

PVQ_MAGIC = b"PVQ\x00"
PVQ_HEADER_FMT = "<4sBHHHIIII"
PVQ_HEADER_SIZE = struct.calcsize(PVQ_HEADER_FMT)
PACT_NERV_LEN_PREFIXED_HEADER_FMT = "<4sBHHBIIII"
PACT_NERV_LEN_PREFIXED_HEADER_SIZE = struct.calcsize(PACT_NERV_LEN_PREFIXED_HEADER_FMT)


class HprcRepresentationFamily(StrEnum):
    """Compact representation families normalized onto HPRC sections."""

    PR95_HNERV = "pr95_hnerv"
    HNERV_PACKED = "hnerv_packed"
    RNERV = "rnerv"
    PACT_NERV = "pact_nerv"
    PACT_NERV_VQ = "pact_nerv_vq"
    TREE_NERV = "tree_nerv"
    HI_NERV = "hi_nerv"
    SR_NERV = "sr_nerv"
    VQ_NERV = "vq_nerv"
    SIREN_IMPLICIT = "siren_implicit"
    FINER_IMPLICIT = "finer_implicit"
    C3_COOL_CHIC = "c3_cool_chic"
    PROCEDURAL_DRIVING_PRIOR = "procedural_driving_prior"


REPRESENTATION_FAMILY_IDS: dict[HprcRepresentationFamily, int] = {
    HprcRepresentationFamily.PR95_HNERV: 95,
    HprcRepresentationFamily.HNERV_PACKED: 101,
    HprcRepresentationFamily.RNERV: 130,
    HprcRepresentationFamily.PACT_NERV: 140,
    HprcRepresentationFamily.PACT_NERV_VQ: 141,
    HprcRepresentationFamily.TREE_NERV: 142,
    HprcRepresentationFamily.HI_NERV: 143,
    HprcRepresentationFamily.SR_NERV: 144,
    HprcRepresentationFamily.VQ_NERV: 145,
    HprcRepresentationFamily.SIREN_IMPLICIT: 160,
    HprcRepresentationFamily.FINER_IMPLICIT: 161,
    HprcRepresentationFamily.C3_COOL_CHIC: 170,
    HprcRepresentationFamily.PROCEDURAL_DRIVING_PRIOR: 180,
}

_SECTION_ROLES: dict[HprcSectionKind, str] = {
    HprcSectionKind.DECODER_QW: "charged_decoder_or_program_weights",
    HprcSectionKind.LATENTS_RC: "charged_latent_stream",
    HprcSectionKind.CODEBOOKS_Q: "charged_codebooks_or_atoms",
    HprcSectionKind.SELECTORS_RC: "charged_indices_modes_or_temporal_policy",
    HprcSectionKind.RESIDUAL_RC: "charged_scorer_priced_residual_tokens",
    HprcSectionKind.RDO_PLAN: "charged_allocation_hints_no_authority",
    HprcSectionKind.RECEIVER_STATE: "charged_decode_constants_or_headers",
    HprcSectionKind.MANIFEST_JSON: "charged_provenance_manifest",
}


class HprcRepresentationSpineError(ValueError):
    """Raised when a representation cannot be projected into the spine."""


@dataclass(frozen=True)
class HprcRepresentationSpinePacket:
    """HPRC packet bytes plus normalized byte-value metadata."""

    family: HprcRepresentationFamily
    hprc_bin: bytes
    manifest: dict[str, Any]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.hprc_bin).hexdigest()


def build_representation_spine_packet(
    *,
    family: HprcRepresentationFamily | str,
    decoder_blob: bytes | bytearray | memoryview = b"",
    latents_blob: bytes | bytearray | memoryview = b"",
    codebooks_blob: bytes | bytearray | memoryview = b"",
    selectors_blob: bytes | bytearray | memoryview = b"",
    residual_blob: bytes | bytearray | memoryview = b"",
    receiver_state_blob: bytes | bytearray | memoryview = b"",
    rdo_plan: dict[str, Any] | None = None,
    manifest_extra: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    config: HprcPacketConfig | None = None,
) -> HprcRepresentationSpinePacket:
    """Build a canonical HPRC section projection for one representation.

    Empty blobs are omitted.  The output is useful for byte accounting,
    acquisition, and adapter handoff; it is not a score claim and does not
    imply the target family uses HPRC as its final runtime decoder.
    """

    fam = _coerce_family(family)
    section_payloads: dict[HprcSectionKind, bytes] = {}
    _add_if_present(section_payloads, HprcSectionKind.DECODER_QW, decoder_blob)
    _add_if_present(section_payloads, HprcSectionKind.LATENTS_RC, latents_blob)
    _add_if_present(section_payloads, HprcSectionKind.CODEBOOKS_Q, codebooks_blob)
    _add_if_present(section_payloads, HprcSectionKind.SELECTORS_RC, selectors_blob)
    _add_if_present(section_payloads, HprcSectionKind.RESIDUAL_RC, residual_blob)
    _add_if_present(section_payloads, HprcSectionKind.RECEIVER_STATE, receiver_state_blob)
    if not section_payloads:
        raise HprcRepresentationSpineError("representation spine needs at least one payload section")

    plan_payload = _json_bytes(
        {
            "schema": "hprc_representation_spine_rdo_plan.v1",
            "family": fam.value,
            "authority": "allocation_hint_only",
            "acceptance_rule": "delta_nonrate + 25*delta_archive_bytes/N < 0 after replay",
            "requires_receiver_proof_before_promotion": True,
            "requires_exact_cpu_cuda_before_score_authority": True,
            **(rdo_plan or {}),
            **FALSE_AUTHORITY,
        }
    )
    section_payloads[HprcSectionKind.RDO_PLAN] = plan_payload

    cfg = config or HprcPacketConfig(decoder_family_id=REPRESENTATION_FAMILY_IDS[fam])
    manifest_payload = _representation_manifest_payload(
        family=fam,
        config=cfg,
        payload_sections=section_payloads,
        source=source or {},
        manifest_extra=manifest_extra or {},
    )
    section_payloads[HprcSectionKind.MANIFEST_JSON] = _json_bytes(manifest_payload)

    hprc_bin = pack_hprc_packet(section_payloads, config=cfg)
    parsed = parse_hprc_packet(hprc_bin)
    packet_manifest = parsed.manifest()
    packet_manifest["representation_spine"] = {
        **manifest_payload,
        "hprc_bin_bytes": len(hprc_bin),
        "hprc_bin_sha256": hashlib.sha256(hprc_bin).hexdigest(),
    }
    packet_manifest.update(FALSE_AUTHORITY)
    return HprcRepresentationSpinePacket(family=fam, hprc_bin=hprc_bin, manifest=packet_manifest)


def build_pr95_hnerv_spine_from_archive(
    archive_zip: str | Path,
) -> HprcRepresentationSpinePacket:
    """Project PR95's u32-section HNeRV packet into the common spine."""

    archive = Path(archive_zip).expanduser().resolve(strict=False)
    view = read_strict_single_member_zip(archive)
    parsed = parse_pr95_hnerv_payload(view.payload)
    return build_representation_spine_packet(
        family=HprcRepresentationFamily.PR95_HNERV,
        decoder_blob=_expect_bytes(parsed["decoder_blob"], "decoder_blob"),
        latents_blob=_expect_bytes(parsed["latents_blob"], "latents_blob"),
        receiver_state_blob=_expect_bytes(parsed["meta_blob"], "meta_blob"),
        source=_source_archive_row(archive, member_name=view.member_name, member_bytes=view.member_bytes),
        manifest_extra={
            "source_payload_kind": "pr95_u32_prefixed_hnerv",
            "role": "frontier_scale_control_base_renderer",
        },
    )


def build_packed_hnerv_spine_from_archive(
    archive_zip: str | Path,
) -> HprcRepresentationSpinePacket:
    """Project PR101/PR106-style packed HNeRV archives into the spine."""

    archive = Path(archive_zip).expanduser().resolve(strict=False)
    view = read_packed_archive_view(archive)
    state = _json_bytes(
        {
            "schema": "hprc_packed_hnerv_receiver_state.v1",
            "payload_kind": view.payload_kind,
            "header_format": view.packed.header_format,
            "header_sha256": hashlib.sha256(view.packed.header).hexdigest(),
            "header_bytes": len(view.packed.header),
            "decoder_brotli_stream_count": view.decoder_brotli_stream_count,
            "hnerv_payload_start": int(view.hnerv_payload_start),
        }
    )
    return build_representation_spine_packet(
        family=HprcRepresentationFamily.HNERV_PACKED,
        decoder_blob=view.packed.decoder_packed_brotli,
        latents_blob=view.packed.latents_and_sidecar_brotli,
        receiver_state_blob=state,
        source=_source_archive_row(
            archive,
            member_name=view.archive.member_name,
            member_bytes=view.archive.member_bytes,
        ),
        manifest_extra={
            "source_payload_kind": view.payload_kind,
            "role": "packed_hnerv_base_renderer_or_control",
            "repackable_sections": list(view.repackable_sections),
        },
    )


def build_pact_nerv_vq_spine_from_archive_payload(
    archive_payload: bytes | bytearray | memoryview,
    *,
    source: dict[str, Any] | None = None,
    manifest_extra: dict[str, Any] | None = None,
) -> HprcRepresentationSpinePacket:
    """Project PVQ bytes into decoder/codebook/selector/receiver-state sections."""

    payload = bytes(archive_payload)
    parts = _split_pact_nerv_vq_payload(payload)
    state = _json_bytes(
        {
            "schema": "hprc_pact_nerv_vq_receiver_state.v1",
            "pvq_header": parts["header"],
            "meta_sha256": hashlib.sha256(parts["meta_blob"]).hexdigest(),
            "meta_bytes": len(parts["meta_blob"]),
        }
    )
    return build_representation_spine_packet(
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder_blob=parts["decoder_blob"],
        codebooks_blob=parts["codebook_blob"],
        selectors_blob=parts["indices_blob"],
        receiver_state_blob=state,
        source=source or {},
        manifest_extra={
            "source_payload_kind": "pact_nerv_vq_pvq",
            "role": "vq_codebook_base_renderer",
            "latent_dim": parts["header"]["latent_dim"],
            "num_pairs": parts["header"]["num_pairs"],
            "codebook_size": parts["header"]["codebook_size"],
            **(manifest_extra or {}),
        },
    )


def build_pact_nerv_vq_spine_from_archive(
    archive_zip: str | Path,
) -> HprcRepresentationSpinePacket:
    """Read a PVQ archive.zip and project its charged ``0.bin`` into the spine."""

    archive = Path(archive_zip).expanduser().resolve(strict=False)
    payload, member_name, member_bytes = _read_archive_member(archive, preferred_member="0.bin")
    return build_pact_nerv_vq_spine_from_archive_payload(
        payload,
        source=_source_archive_row(archive, member_name=member_name, member_bytes=member_bytes),
    )


def build_pact_nerv_len_prefixed_spine_from_archive_payload(
    archive_payload: bytes | bytearray | memoryview,
    *,
    payload_kind: str,
    expected_magic: bytes,
    side_channel_kind: str,
    family: HprcRepresentationFamily | str = HprcRepresentationFamily.PACT_NERV,
    source: dict[str, Any] | None = None,
    manifest_extra: dict[str, Any] | None = None,
) -> HprcRepresentationSpinePacket:
    """Project PACT-NeRV 26-byte-header packets into the common spine.

    This covers the IA3 and selector-family archives whose grammar is:
    header, decoder blob, latent blob, side-channel blob, meta blob.  The
    side channel is charged as selectors/modes/conditioning, so acquisition can
    price it independently from decoder weights and per-pair latents.
    """

    payload = bytes(archive_payload)
    parts = _split_pact_nerv_len_prefixed_payload(
        payload,
        expected_magic=expected_magic,
        side_channel_kind=side_channel_kind,
    )
    state = _json_bytes(
        {
            "schema": "hprc_pact_nerv_len_prefixed_receiver_state.v1",
            "payload_kind": payload_kind,
            "header": parts["header"],
            "meta_sha256": hashlib.sha256(parts["meta_blob"]).hexdigest(),
            "meta_bytes": len(parts["meta_blob"]),
            "side_channel_kind": side_channel_kind,
        }
    )
    return build_representation_spine_packet(
        family=family,
        decoder_blob=parts["decoder_blob"],
        latents_blob=parts["latents_blob"],
        selectors_blob=parts["side_blob"],
        receiver_state_blob=state,
        source=source or {},
        manifest_extra={
            "source_payload_kind": payload_kind,
            "role": "pact_nerv_learned_receiver_or_selector_policy",
            "payload_magic": parts["header"]["magic"],
            "latent_dim": parts["header"]["latent_dim"],
            "num_pairs": parts["header"]["num_pairs"],
            "mode_or_palette_or_pose_dim": parts["header"]["mode_or_palette_or_pose_dim"],
            "side_channel_kind": side_channel_kind,
            **(manifest_extra or {}),
        },
    )


def build_pact_nerv_len_prefixed_spine_from_archive(
    archive_zip: str | Path,
) -> HprcRepresentationSpinePacket:
    """Read a PACT-NeRV IA3/selector archive.zip and project ``0.bin``."""

    archive = Path(archive_zip).expanduser().resolve(strict=False)
    payload, member_name, member_bytes = _read_archive_member(archive, preferred_member="0.bin")
    if len(payload) < 4:
        raise HprcRepresentationSpineError("PACT-NeRV archive payload shorter than magic")
    magic = payload[:4]
    if magic not in _PACT_NERV_MAGIC_PROJECTIONS:
        raise HprcRepresentationSpineError(
            f"unsupported PACT-NeRV len-prefixed magic: {magic!r}"
        )
    payload_kind, side_channel_kind = _PACT_NERV_MAGIC_PROJECTIONS[magic]
    return build_pact_nerv_len_prefixed_spine_from_archive_payload(
        payload,
        payload_kind=payload_kind,
        expected_magic=magic,
        side_channel_kind=side_channel_kind,
        source=_source_archive_row(archive, member_name=member_name, member_bytes=member_bytes),
    )


def build_generic_neural_spine_packet(
    *,
    family: HprcRepresentationFamily | str,
    decoder_blob: bytes | bytearray | memoryview,
    latents_blob: bytes | bytearray | memoryview = b"",
    codebooks_blob: bytes | bytearray | memoryview = b"",
    selectors_blob: bytes | bytearray | memoryview = b"",
    residual_blob: bytes | bytearray | memoryview = b"",
    receiver_state_blob: bytes | bytearray | memoryview = b"",
    manifest_extra: dict[str, Any] | None = None,
) -> HprcRepresentationSpinePacket:
    """Generic emitter for RNeRV/PACT-NeRV/C3/implicit/procedural experiments."""

    return build_representation_spine_packet(
        family=family,
        decoder_blob=decoder_blob,
        latents_blob=latents_blob,
        codebooks_blob=codebooks_blob,
        selectors_blob=selectors_blob,
        residual_blob=residual_blob,
        receiver_state_blob=receiver_state_blob,
        manifest_extra={
            "source_payload_kind": "generic_section_blobs",
            "role": "new_substrate_adapter_spine_entry",
            **(manifest_extra or {}),
        },
    )


def write_representation_spine_projection(
    *,
    output_dir: str | Path,
    spine: HprcRepresentationSpinePacket,
    basename: str = "hprc_representation_spine",
) -> dict[str, Any]:
    """Write spine bytes plus manifest for downstream acquisition."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    bin_path = out / f"{basename}.bin"
    manifest_path = out / f"{basename}_manifest.json"
    bin_path.write_bytes(spine.hprc_bin)
    payload = {
        "schema": HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
        "family": spine.family.value,
        "hprc_bin_path": bin_path.as_posix(),
        "hprc_bin_bytes": len(spine.hprc_bin),
        "hprc_bin_sha256": spine.sha256,
        "manifest": spine.manifest,
        "byte_value_contract": {
            "score_value_per_byte_rule": "delta_nonrate + 25*delta_archive_bytes/N < 0",
            "compare_sections_under_receiver_proof": True,
            "exact_axis_required_for_score_authority": True,
        },
        **FALSE_AUTHORITY,
    }
    manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {**payload, "manifest_path": manifest_path.as_posix()}


def _split_pact_nerv_vq_payload(payload: bytes) -> dict[str, Any]:
    if len(payload) < PVQ_HEADER_SIZE:
        raise HprcRepresentationSpineError("PVQ payload shorter than header")
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        codebook_size,
        decoder_len,
        codebook_len,
        indices_len,
        meta_len,
    ) = struct.unpack(PVQ_HEADER_FMT, payload[:PVQ_HEADER_SIZE])
    if magic != PVQ_MAGIC:
        raise HprcRepresentationSpineError(f"bad PVQ magic: {magic!r}")
    expected_codebook_len = int(codebook_size) * int(latent_dim) * 2
    expected_indices_len = int(num_pairs) * 2
    if int(codebook_len) != expected_codebook_len:
        raise HprcRepresentationSpineError(
            f"PVQ codebook_len {codebook_len} != {expected_codebook_len}"
        )
    if int(indices_len) != expected_indices_len:
        raise HprcRepresentationSpineError(
            f"PVQ indices_len {indices_len} != {expected_indices_len}"
        )
    end_decoder = PVQ_HEADER_SIZE + int(decoder_len)
    end_codebook = end_decoder + int(codebook_len)
    end_indices = end_codebook + int(indices_len)
    end_meta = end_indices + int(meta_len)
    if end_meta != len(payload):
        raise HprcRepresentationSpineError(
            f"PVQ payload bytes {len(payload)} != header-declared {end_meta}"
        )
    return {
        "header": {
            "version": int(version),
            "latent_dim": int(latent_dim),
            "num_pairs": int(num_pairs),
            "codebook_size": int(codebook_size),
            "decoder_len": int(decoder_len),
            "codebook_len": int(codebook_len),
            "indices_len": int(indices_len),
            "meta_len": int(meta_len),
        },
        "decoder_blob": payload[PVQ_HEADER_SIZE:end_decoder],
        "codebook_blob": payload[end_decoder:end_codebook],
        "indices_blob": payload[end_codebook:end_indices],
        "meta_blob": payload[end_indices:end_meta],
    }


def _split_pact_nerv_len_prefixed_payload(
    payload: bytes,
    *,
    expected_magic: bytes,
    side_channel_kind: str,
) -> dict[str, Any]:
    if len(payload) < PACT_NERV_LEN_PREFIXED_HEADER_SIZE:
        raise HprcRepresentationSpineError("PACT-NeRV payload shorter than header")
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        mode_or_palette_or_pose_dim,
        decoder_len,
        latent_len,
        side_len,
        meta_len,
    ) = struct.unpack(
        PACT_NERV_LEN_PREFIXED_HEADER_FMT,
        payload[:PACT_NERV_LEN_PREFIXED_HEADER_SIZE],
    )
    if magic != expected_magic:
        raise HprcRepresentationSpineError(
            f"bad PACT-NeRV magic: {magic!r}; expected {expected_magic!r}"
        )
    expected_latent_len = int(num_pairs) * int(latent_dim) * 2
    if int(latent_len) != expected_latent_len:
        raise HprcRepresentationSpineError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_len}"
        )
    end_decoder = PACT_NERV_LEN_PREFIXED_HEADER_SIZE + int(decoder_len)
    end_latents = end_decoder + int(latent_len)
    end_side = end_latents + int(side_len)
    end_meta = end_side + int(meta_len)
    if end_meta != len(payload):
        raise HprcRepresentationSpineError(
            f"PACT-NeRV payload bytes {len(payload)} != header-declared {end_meta}"
        )
    return {
        "header": {
            "magic": magic.decode("ascii", errors="replace"),
            "version": int(version),
            "latent_dim": int(latent_dim),
            "num_pairs": int(num_pairs),
            "mode_or_palette_or_pose_dim": int(mode_or_palette_or_pose_dim),
            "decoder_len": int(decoder_len),
            "latent_len": int(latent_len),
            "side_len": int(side_len),
            "meta_len": int(meta_len),
            "side_channel_kind": side_channel_kind,
        },
        "decoder_blob": payload[PACT_NERV_LEN_PREFIXED_HEADER_SIZE:end_decoder],
        "latents_blob": payload[end_decoder:end_latents],
        "side_blob": payload[end_latents:end_side],
        "meta_blob": payload[end_side:end_meta],
    }


_PACT_NERV_MAGIC_PROJECTIONS: dict[bytes, tuple[str, str]] = {
    b"PIA3": ("pact_nerv_ia3_pia3", "ego_pose_conditioning"),
    b"PSV2": ("pact_nerv_selector_v2_psv2", "arithmetic_selector_k16"),
    b"PSV3": ("pact_nerv_selector_v3_psv3", "rice_golomb_selector"),
    b"PSV4": ("pact_nerv_selector_v4_psv4", "rle_selector"),
}


def _representation_manifest_payload(
    *,
    family: HprcRepresentationFamily,
    config: HprcPacketConfig,
    payload_sections: dict[HprcSectionKind, bytes],
    source: dict[str, Any],
    manifest_extra: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        {
            "id": int(kind),
            "name": kind.name.lower(),
            "role": _SECTION_ROLES[kind],
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for kind, payload in sorted(payload_sections.items(), key=lambda item: int(item[0]))
    ]
    payload_bytes = sum(int(row["bytes"]) for row in rows)
    return {
        "schema": HPRC_REPRESENTATION_SPINE_MANIFEST_SCHEMA,
        "family": family.value,
        "decoder_family_id": REPRESENTATION_FAMILY_IDS[family],
        "config": config.as_dict(),
        "source": source,
        "sections": rows,
        "byte_accounting": {
            "payload_section_bytes": payload_bytes,
            "section_count": len(rows),
            "score_authority": False,
            "archive_zip_authority": False,
            "note": "spine projection bytes are acquisition input until a receiver-proven archive consumes them",
        },
        "all_candidate_families_share_this_schema": [
            item.value for item in HprcRepresentationFamily
        ],
        "exact_promotion_requirements": [
            "archive_zip_contains_charged_bytes",
            "inflate_sh_consumes_bytes_decode_only",
            "receiver_proof_passed",
            "contest_cpu_cuda_exact_eval_before_score_claim",
        ],
        "manifest_extra": manifest_extra,
        **FALSE_AUTHORITY,
    }


def _source_archive_row(archive: Path, *, member_name: str, member_bytes: int) -> dict[str, Any]:
    return {
        "kind": "archive_zip",
        "path": archive.as_posix(),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "member_name": member_name,
        "member_bytes": int(member_bytes),
    }


def _read_archive_member(
    archive: Path,
    *,
    preferred_member: str,
) -> tuple[bytes, str, int]:
    with zipfile.ZipFile(archive, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if preferred_member in names:
            payload = zf.read(preferred_member)
            return payload, preferred_member, len(payload)
        if len(names) == 1:
            name = names[0]
            payload = zf.read(name)
            return payload, name, len(payload)
    raise HprcRepresentationSpineError(
        f"archive {archive} has no {preferred_member!r} member and is not single-member"
    )


def _add_if_present(
    sections: dict[HprcSectionKind, bytes],
    kind: HprcSectionKind,
    payload: bytes | bytearray | memoryview,
) -> None:
    data = bytes(payload)
    if data:
        sections[kind] = data


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _coerce_family(family: HprcRepresentationFamily | str) -> HprcRepresentationFamily:
    if isinstance(family, HprcRepresentationFamily):
        return family
    try:
        return HprcRepresentationFamily(str(family))
    except ValueError as exc:
        raise HprcRepresentationSpineError(f"unknown representation family: {family!r}") from exc


def _expect_bytes(value: Any, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise HprcRepresentationSpineError(f"{label} must be bytes")
    return bytes(value)


__all__ = [
    "HPRC_REPRESENTATION_SPINE_MANIFEST_SCHEMA",
    "HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA",
    "HPRC_REPRESENTATION_SPINE_SCHEMA",
    "PACT_NERV_LEN_PREFIXED_HEADER_FMT",
    "PACT_NERV_LEN_PREFIXED_HEADER_SIZE",
    "REPRESENTATION_FAMILY_IDS",
    "HprcRepresentationFamily",
    "HprcRepresentationSpineError",
    "HprcRepresentationSpinePacket",
    "build_generic_neural_spine_packet",
    "build_packed_hnerv_spine_from_archive",
    "build_pact_nerv_len_prefixed_spine_from_archive",
    "build_pact_nerv_len_prefixed_spine_from_archive_payload",
    "build_pact_nerv_vq_spine_from_archive",
    "build_pact_nerv_vq_spine_from_archive_payload",
    "build_pr95_hnerv_spine_from_archive",
    "build_representation_spine_packet",
    "write_representation_spine_projection",
]
