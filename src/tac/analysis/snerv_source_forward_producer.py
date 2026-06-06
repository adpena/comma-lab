# SPDX-License-Identifier: MIT
"""Producer helpers for SNeRV SourceForwardProof rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from tac.analysis.snerv_source_forward_proof import (
    SOURCE_FORWARD_SURFACES,
    build_snerv_source_forward_proof_action_effect,
    build_snerv_source_forward_surface_provenance,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    build_snerv_archive_payload_bitflip_falsification,
    unpack_snerv_archive,
)


def build_snerv_source_forward_proof_from_archive_packet(
    *,
    action_id: str,
    archive_packet: bytes,
    pair_ids: Sequence[int],
    official_torch_tensors: Mapping[str, Any] | None = None,
    pact_mlx_tensors: Mapping[str, Any] | None = None,
    capture_pact_mlx_from_archive: bool = False,
    scorer_tensors_by_surface: Mapping[str, Mapping[str, Any]] | None = None,
    scorer_deltas: Mapping[str, Any] | None = None,
    bitflip_section: str = "decoder_payload",
    bitflip_offset: int = 0,
    bitflip_mask: int = 1,
    tolerance_by_tensor: Mapping[str, float] | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a SourceForwardProof row from a charged SNeRV archive packet.

    This helper is allowed to fail closed.  It binds the two archive-derived
    surfaces immediately and leaves official Torch, Pact MLX, and scorer tensors
    as explicit missing-tensor blockers until their real producers fill them.
    """

    decoded = unpack_snerv_archive(archive_packet)
    receiver_surfaces = decoded.source_forward_receiver_tensor_surfaces(pair_ids)
    pact_mlx_capture: dict[str, Any] | None = None
    if capture_pact_mlx_from_archive:
        if pact_mlx_tensors is not None:
            raise ValueError(
                "pact_mlx_tensors and capture_pact_mlx_from_archive are mutually exclusive"
            )
        pact_mlx_capture = build_pact_mlx_primitive_tensors_from_archive_packet(
            archive_packet=archive_packet,
            pair_ids=pair_ids,
            portable_tub_tensors={
                key: value
                for key, value in receiver_surfaces["surface_tensors"][
                    "archive_parseback"
                ].items()
                if key in {"tub_in", "tub_out"}
            },
        )
        pact_mlx_tensors = pact_mlx_capture["tensors"]
    tensors_by_surface: dict[str, dict[str, Any]] = {
        surface: {}
        for surface in SOURCE_FORWARD_SURFACES
    }
    tensors_by_surface["official_torch"].update(dict(official_torch_tensors or {}))
    tensors_by_surface["pact_mlx"].update(dict(pact_mlx_tensors or {}))
    tensors_by_surface["archive_parseback"].update(
        dict(receiver_surfaces["surface_tensors"]["archive_parseback"])
    )
    tensors_by_surface["numpy_receiver"].update(
        dict(receiver_surfaces["surface_tensors"]["numpy_receiver"])
    )
    for surface, tensors in dict(scorer_tensors_by_surface or {}).items():
        if surface in tensors_by_surface:
            tensors_by_surface[surface].update(dict(tensors))

    bitflip = build_snerv_archive_payload_bitflip_falsification(
        archive_packet,
        bitflip_section=bitflip_section,
        bit_offset=int(bitflip_offset),
        bit_mask=int(bitflip_mask),
    )
    payload_section_hashes = {
        name: _section_sha256(section)
        for name, section in decoded.sections.items()
    }
    provenance = build_snerv_source_forward_surface_provenance(
        pair_ids=pair_ids,
        archive_sha256=decoded.packet_sha256,
        producer_by_surface={
            "official_torch": "official_torch_source_forward_producer",
            "pact_mlx": "pact_mlx_source_forward_producer",
            "archive_parseback": "snerv_archive_parseback_receiver_tensor_surfaces",
            "numpy_receiver": "snerv_numpy_receiver_tensor_surfaces",
        },
        backend_by_surface={
            "official_torch": "torch",
            "pact_mlx": "mlx",
            "archive_parseback": "archive_parseback",
            "numpy_receiver": "numpy_receiver",
        },
    )
    row = build_snerv_source_forward_proof_action_effect(
        action_id=action_id,
        archive_sha256=decoded.packet_sha256,
        archive_bytes=len(bytes(archive_packet)),
        payload_section_hashes=payload_section_hashes,
        pair_ids=pair_ids,
        tensors_by_surface=tensors_by_surface,
        scorer_deltas=dict(scorer_deltas or {}),
        destructive_payload_bit_flip=bitflip,
        surface_provenance=provenance,
        tolerance_by_tensor=tolerance_by_tensor,
        generated_utc=generated_utc,
    )
    row["producer_status"] = {
        "schema": "snerv_source_forward_producer_status.v1",
        "archive_receiver_surfaces_bound": True,
        "parseback_receiver_rgb_uint8_equal": bool(
            receiver_surfaces["parseback_receiver_rgb_uint8_equal"]
        ),
        "archive_receiver_missing_action_effect_tensor_names": list(
            receiver_surfaces["missing_action_effect_tensor_names"]
        ),
        "official_torch_supplied_tensor_count": len(dict(official_torch_tensors or {})),
        "pact_mlx_supplied_tensor_count": len(dict(pact_mlx_tensors or {})),
        "pact_mlx_captured_from_archive": bool(pact_mlx_capture is not None),
        "pact_mlx_capture_schema": (
            pact_mlx_capture.get("schema") if pact_mlx_capture is not None else None
        ),
        "pact_mlx_capture_missing_action_effect_tensor_names": (
            list(pact_mlx_capture.get("missing_action_effect_tensor_names") or ())
            if pact_mlx_capture is not None
            else []
        ),
        "pact_mlx_capture_tub_prep_backend": (
            pact_mlx_capture.get("tub_prep_backend")
            if pact_mlx_capture is not None
            else None
        ),
        "scorer_surface_count": len(dict(scorer_tensors_by_surface or {})),
    }
    return row


def build_pact_mlx_primitive_tensors_from_archive_packet(
    *,
    archive_packet: bytes,
    pair_ids: Sequence[int],
    portable_tub_tensors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture Pact MLX MFU/HFR/output2 primitive tensors from archive bytes."""

    decoded = unpack_snerv_archive(archive_packet)
    payload = decoded.decode_official_mfu_hfr_tub_payload()
    low, skip_mid, skip_high = payload.mfu_inputs()
    current, previous, next_frame = payload.tub_inputs()
    output2_inputs = payload.tub_output2_inputs()
    frames = decoded.decode_pair_frames(pair_ids, clip_to_uint8_range=True)
    if frames.ndim != 5:
        raise ValueError(
            "decoded SNeRV archive pair frames must be shaped (pairs,2,3,H,W); "
            f"got {tuple(frames.shape)}"
        )
    from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
        SnervMlxOfficialMfuHfrTubScoreRenderer,
    )

    tub_config = dict(payload.header.get("tub_config") or {})
    renderer = SnervMlxOfficialMfuHfrTubScoreRenderer(
        mfu=payload.build_mfu(),
        hfr_heads=payload.build_hfr_heads(),
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high,
        output_hw=(int(frames.shape[-2]), int(frames.shape[-1])),
        tub_current=current,
        tub_previous=previous,
        tub_next_frame=next_frame,
        tub_temporal_encoder_concat=(
            output2_inputs[0] if output2_inputs is not None else None
        ),
        tub_output2_raw=output2_inputs[1] if output2_inputs is not None else None,
        tub_output2_fc_hw=(
            tuple(int(value) for value in tub_config["fc_hw"])
            if "fc_hw" in tub_config
            else None
        ),
    )
    return renderer.source_forward_primitive_tensor_bundle(
        pair_ids=pair_ids,
        clip_to_uint8_range=True,
        portable_tub_tensors={
            key: value
            for key, value in dict(portable_tub_tensors or {}).items()
            if key in {"tub_in", "tub_out"}
        },
    )


def _section_sha256(section: bytes) -> str:
    return sha256(bytes(section)).hexdigest()


__all__ = [
    "build_pact_mlx_primitive_tensors_from_archive_packet",
    "build_snerv_source_forward_proof_from_archive_packet",
]
