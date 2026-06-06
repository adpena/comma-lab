# SPDX-License-Identifier: MIT
"""Producer helpers for SNeRV SourceForwardProof rows."""

from __future__ import annotations

import math
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


def build_torch_scorer_source_forward_surface(
    *,
    candidate_pairs_nchw255: Any,
    reference_pairs_nchw255: Any,
    posenet: Any,
    segnet: Any,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    """Capture official scorer tensors and metrics for one SourceForward surface."""

    import torch

    dev = torch.device(device)
    candidate = _torch_pair_tensor_nchw255(
        candidate_pairs_nchw255,
        device=dev,
        name="candidate_pairs_nchw255",
    )
    reference = _torch_pair_tensor_nchw255(
        reference_pairs_nchw255,
        device=dev,
        name="reference_pairs_nchw255",
    )
    if tuple(candidate.shape) != tuple(reference.shape):
        raise ValueError(
            "candidate/reference scorer pair tensors must have identical shape; "
            f"got {tuple(candidate.shape)} vs {tuple(reference.shape)}"
        )
    posenet = posenet.to(dev) if hasattr(posenet, "to") else posenet
    segnet = segnet.to(dev) if hasattr(segnet, "to") else segnet
    if hasattr(posenet, "eval"):
        posenet.eval()
    if hasattr(segnet, "eval"):
        segnet.eval()
    with torch.no_grad():
        pos_in = posenet.preprocess_input(candidate)
        ref_pos_in = posenet.preprocess_input(reference)
        pos_out = _pose_output_tensor(posenet(pos_in))
        ref_pos_out = _pose_output_tensor(posenet(ref_pos_in))
        pose = pos_out[..., :6]
        ref_pose = ref_pos_out[..., :6]
        d_pose = float(torch.mean((pose - ref_pose) ** 2).detach().cpu().item())

        seg_in = segnet.preprocess_input(candidate)
        ref_seg_in = segnet.preprocess_input(reference)
        seg_logits = _tensor_from_model_output(segnet(seg_in), key="logits")
        ref_seg_logits = _tensor_from_model_output(segnet(ref_seg_in), key="logits")
        seg_argmax = torch.argmax(seg_logits, dim=1)
        ref_seg_argmax = torch.argmax(ref_seg_logits, dim=1)
        disagree = (seg_argmax != ref_seg_argmax).to(torch.float32)
        d_seg = float(torch.mean(disagree).detach().cpu().item())
    return {
        "schema": "snerv_torch_scorer_source_forward_surface.v1",
        "tensors": {
            "segnet_input": _torch_to_numpy(seg_in),
            "posenet_input": _torch_to_numpy(pos_in),
            "segnet_logits": _torch_to_numpy(seg_logits),
            "segnet_argmax": _torch_to_numpy(seg_argmax),
            "posenet_output": _torch_to_numpy(pose),
        },
        "metrics": {
            "d_seg": d_seg,
            "d_pose": d_pose,
        },
        "surface_scorer_capture_authority": "real_torch_scorer_forward_capture",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_snerv_scorer_deltas_from_surface_metrics(
    metrics_by_surface: Mapping[str, Mapping[str, Any]],
    *,
    score_surface: str = "numpy_receiver",
    reference_surface: str = "official_torch",
    tolerance_by_field: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the scorer-delta row consumed by SourceForwardProof validation."""

    metrics = {
        str(surface): {
            "d_seg": float(values["d_seg"]),
            "d_pose": float(values["d_pose"]),
        }
        for surface, values in dict(metrics_by_surface).items()
    }
    selected = metrics.get(str(score_surface))
    reference = metrics.get(str(reference_surface))
    if selected is None:
        raise ValueError(f"score_surface {score_surface!r} missing from scorer metrics")
    if reference is None:
        raise ValueError(
            f"reference_surface {reference_surface!r} missing from scorer metrics"
        )
    delta_score_nonrate = (
        100.0 * (float(selected["d_seg"]) - float(reference["d_seg"]))
        + math.sqrt(max(0.0, 10.0 * float(selected["d_pose"])))
        - math.sqrt(max(0.0, 10.0 * float(reference["d_pose"])))
    )
    return {
        "d_seg": float(selected["d_seg"]),
        "d_pose": float(selected["d_pose"]),
        "delta_score_nonrate": float(delta_score_nonrate),
        "by_surface": metrics,
        "tolerance_by_field": {
            str(key): float(value)
            for key, value in dict(tolerance_by_field or {}).items()
        },
    }


def _section_sha256(section: bytes) -> str:
    return sha256(bytes(section)).hexdigest()


def _torch_pair_tensor_nchw255(value: Any, *, device: Any, name: str) -> Any:
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 5 or int(tensor.shape[1]) != 2 or int(tensor.shape[2]) != 3:
        raise ValueError(
            f"{name} must be shaped (pairs,2,3,H,W) in [0,255]; "
            f"got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


def _pose_output_tensor(output: Any) -> Any:
    if isinstance(output, Mapping):
        if "pose" not in output:
            raise ValueError("PoseNet output mapping must contain key 'pose'")
        return _tensor_from_model_output(output["pose"], key="pose")
    return _tensor_from_model_output(output, key="pose")


def _tensor_from_model_output(output: Any, *, key: str) -> Any:
    import torch

    if isinstance(output, Mapping):
        if key in output:
            output = output[key]
        elif len(output) == 1:
            output = next(iter(output.values()))
        else:
            raise ValueError(f"model output mapping does not contain key {key!r}")
    tensor = output if isinstance(output, torch.Tensor) else torch.as_tensor(output)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"model output {key!r} contains non-finite values")
    return tensor


def _torch_to_numpy(tensor: Any) -> Any:
    return tensor.detach().cpu().numpy().copy()


__all__ = [
    "build_pact_mlx_primitive_tensors_from_archive_packet",
    "build_snerv_scorer_deltas_from_surface_metrics",
    "build_snerv_source_forward_proof_from_archive_packet",
    "build_torch_scorer_source_forward_surface",
]
