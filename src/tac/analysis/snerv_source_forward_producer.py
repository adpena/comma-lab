# SPDX-License-Identifier: MIT
"""Producer helpers for SNeRV SourceForwardProof rows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

import numpy as np

from tac.analysis.snerv_source_forward_proof import (
    DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS,
    REPARAMETERIZED_RENAME_REQUIRED,
    SOURCE_FORWARD_SURFACES,
    SOURCE_FORWARD_TENSOR_NAMES,
    SOURCE_IDENTICAL,
    build_snerv_source_forward_proof_action_effect,
    build_snerv_source_forward_surface_provenance,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    build_snerv_archive_payload_bitflip_falsification,
    unpack_snerv_archive,
)

SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA = "snerv_output2_boundary_verdict.v1"
SOURCE_GRAPH_UNPROVEN = "SOURCE_GRAPH_UNPROVEN"


def build_snerv_source_forward_proof_from_archive_packet(
    *,
    action_id: str,
    archive_packet: bytes,
    pair_ids: Sequence[int],
    official_torch_tensors: Mapping[str, Any] | None = None,
    official_torch_capture_manifest: Mapping[str, Any] | None = None,
    capture_official_torch_from_archive: bool = False,
    capture_official_torch_from_upstream_fixture: bool = False,
    official_snerv_repo_dir: str | None = None,
    official_torch_train_one_step: bool = False,
    official_torch_checkpoint_state_dict: Mapping[str, Any] | None = None,
    official_torch_checkpoint_state_dict_path: str | None = None,
    official_torch_checkpoint_state_dict_kind: str = (
        "official_trained_checkpoint_state_dict"
    ),
    pact_mlx_tensors: Mapping[str, Any] | None = None,
    capture_pact_mlx_from_archive: bool = False,
    scorer_tensors_by_surface: Mapping[str, Mapping[str, Any]] | None = None,
    scorer_deltas: Mapping[str, Any] | None = None,
    capture_torch_scorer_from_rgb: bool = False,
    reference_pairs_nchw255: Any | None = None,
    posenet: Any | None = None,
    segnet: Any | None = None,
    scorer_device: str | Any = "cpu",
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
    official_torch_capture: dict[str, Any] | None = None
    official_torch_upstream_capture: dict[str, Any] | None = None
    official_torch_manifest_status = (
        validate_snerv_official_torch_upstream_capture_manifest(
            official_torch_capture_manifest,
            pair_ids=pair_ids,
            tensor_names=official_torch_tensors.keys()
            if official_torch_tensors is not None
            else (),
        )
    )
    official_torch_upstream_capture_status: dict[str, Any] = {
        "schema": "snerv_official_torch_upstream_capture_status.v1",
        "verdict": None,
        "strict_state_dict_load_required": True,
        "source_graph_unproven": False,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if capture_official_torch_from_archive:
        if official_torch_tensors is not None:
            raise ValueError(
                "official_torch_tensors and capture_official_torch_from_archive "
                "are mutually exclusive"
            )
        official_torch_capture = (
            build_official_torch_primitive_tensors_from_archive_packet(
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
        )
        official_torch_tensors = official_torch_capture["tensors"]
        official_torch_manifest_status = (
            validate_snerv_official_torch_upstream_capture_manifest(
                official_torch_capture_manifest,
                pair_ids=pair_ids,
                tensor_names=official_torch_tensors.keys(),
            )
        )
    if capture_official_torch_from_upstream_fixture:
        if official_torch_tensors is not None:
            raise ValueError(
                "official_torch_tensors and "
                "capture_official_torch_from_upstream_fixture are mutually exclusive"
            )
        if capture_official_torch_from_archive:
            raise ValueError(
                "capture_official_torch_from_archive and "
                "capture_official_torch_from_upstream_fixture are mutually exclusive"
            )
        try:
            official_torch_upstream_capture = (
                build_official_torch_upstream_fixture_tensors(
                    official_repo_dir=official_snerv_repo_dir,
                    train_one_step=official_torch_train_one_step,
                    official_trained_checkpoint_state_dict=(
                        official_torch_checkpoint_state_dict
                    ),
                    official_trained_checkpoint_state_dict_path=(
                        official_torch_checkpoint_state_dict_path
                    ),
                    official_trained_checkpoint_state_dict_kind=(
                        official_torch_checkpoint_state_dict_kind
                    ),
                )
            )
            official_torch_tensors = official_torch_upstream_capture["tensors"]
            official_torch_capture_manifest = (
                build_snerv_official_torch_upstream_capture_manifest(
                    pair_ids=pair_ids,
                    tensor_names=official_torch_tensors.keys(),
                    model_source_sha256=official_torch_upstream_capture.get(
                        "model_source_sha256"
                    ),
                    checkpoint_sha256=official_torch_upstream_capture.get(
                        "checkpoint_sha256"
                    ),
                    state_dict_sha256=official_torch_upstream_capture.get(
                        "state_dict_sha256"
                    ),
                    decoder_len=official_torch_upstream_capture.get("decoder_len"),
                )
            )
            official_torch_manifest_status = (
                validate_snerv_official_torch_upstream_capture_manifest(
                    official_torch_capture_manifest,
                    pair_ids=pair_ids,
                    tensor_names=official_torch_tensors.keys(),
                )
            )
            official_torch_upstream_capture_status = {
                **official_torch_upstream_capture_status,
                "verdict": "SOURCE_GRAPH_CAPTURED",
                "source_graph_unproven": False,
                "blockers": [],
            }
        except Exception as exc:
            official_torch_upstream_capture = {
                "schema": "snerv_official_torch_upstream_capture_failed.v1",
                "verdict": SOURCE_GRAPH_UNPROVEN,
                "source_graph_unproven": True,
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "blockers": [
                    "snerv_upstream_source_graph_unproven",
                    f"snerv_upstream_source_capture_failed:{type(exc).__name__}",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
            official_torch_tensors = {}
            official_torch_capture_manifest = None
            official_torch_manifest_status = (
                validate_snerv_official_torch_upstream_capture_manifest(
                    None,
                    pair_ids=pair_ids,
                    tensor_names=(),
                )
            )
            official_torch_upstream_capture_status = dict(official_torch_upstream_capture)
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
                if key in {"coord_time_embedding", "tub_in", "tub_out"}
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
    scorer_capture_by_surface: dict[str, dict[str, Any]] = {}
    if capture_torch_scorer_from_rgb:
        if scorer_tensors_by_surface is not None:
            raise ValueError(
                "scorer_tensors_by_surface and capture_torch_scorer_from_rgb are mutually exclusive"
            )
        if scorer_deltas is not None:
            raise ValueError(
                "scorer_deltas and capture_torch_scorer_from_rgb are mutually exclusive"
            )
        if reference_pairs_nchw255 is None or posenet is None or segnet is None:
            raise ValueError(
                "capture_torch_scorer_from_rgb requires reference_pairs_nchw255, "
                "posenet, and segnet"
            )
        for surface in SOURCE_FORWARD_SURFACES:
            rgb = _surface_rgb_pair_tensor_for_scorer(tensors_by_surface[surface])
            if rgb is None:
                continue
            scorer_capture = build_torch_scorer_source_forward_surface(
                candidate_pairs_nchw255=rgb,
                reference_pairs_nchw255=reference_pairs_nchw255,
                posenet=posenet,
                segnet=segnet,
                device=scorer_device,
            )
            scorer_capture_by_surface[surface] = scorer_capture
            tensors_by_surface[surface].update(dict(scorer_capture["tensors"]))
        if scorer_capture_by_surface:
            scorer_deltas = build_snerv_scorer_deltas_from_surface_metrics(
                {
                    surface: capture["metrics"]
                    for surface, capture in scorer_capture_by_surface.items()
                },
                allow_missing_reference=True,
            )

    output2_boundary = build_snerv_output2_boundary_verdict(
        tensors_by_surface=tensors_by_surface,
        archive_decoder_header=decoded.decode_official_mfu_hfr_tub_payload().header,
        tolerance=float(dict(tolerance_by_tensor or {}).get("output_2", 0.0)),
    )
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
        tensor_capture_authority_by_surface={
            "official_torch": _official_torch_tensor_capture_authority(
                official_torch_tensors=official_torch_tensors,
                receiver_bound_capture=official_torch_capture is not None,
                manifest_status=official_torch_manifest_status,
            )
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
        output2_boundary_verdict=output2_boundary,
        surface_provenance=provenance,
        tolerance_by_tensor=tolerance_by_tensor,
        generated_utc=generated_utc,
    )
    if official_torch_upstream_capture_status.get("source_graph_unproven") is True:
        row["blockers"] = _ordered_unique(
            [
                *row.get("blockers", ()),
                *official_torch_upstream_capture_status.get("blockers", ()),
            ]
        )
        row["passed"] = False
        row["source_forward_replay_bound"] = False
        row["source_forward_replay_verified"] = False
        row["source_forward_replay_authority"] = False
        row["full_stack_source_forward_replay_proven"] = False
        row["launch_gate_clearable"] = False
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
        "official_torch_upstream_capture_manifest": (
            dict(official_torch_capture_manifest or {})
            if official_torch_capture_manifest is not None
            else None
        ),
        "official_torch_upstream_capture_manifest_status": (
            official_torch_manifest_status
        ),
        "official_torch_upstream_capture_manifest_passed": bool(
            official_torch_manifest_status["passed"]
        ),
        "official_torch_captured_from_archive": bool(
            official_torch_capture is not None
        ),
        "official_torch_captured_from_upstream_fixture": bool(
            official_torch_upstream_capture is not None
        ),
        "official_torch_upstream_capture_schema": (
            official_torch_upstream_capture.get("schema")
            if official_torch_upstream_capture is not None
            else None
        ),
        "official_torch_upstream_capture_source_scope": (
            official_torch_upstream_capture.get("source_scope")
            if official_torch_upstream_capture is not None
            else None
        ),
        "official_torch_upstream_capture_status": (
            official_torch_upstream_capture_status
        ),
        "official_torch_capture_schema": (
            official_torch_capture.get("schema")
            if official_torch_capture is not None
            else None
        ),
        "official_torch_capture_authority_blocked": bool(
            official_torch_capture is not None
        ),
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
        "torch_scorer_captured_from_rgb": bool(scorer_capture_by_surface),
        "torch_scorer_capture_surface_count": len(scorer_capture_by_surface),
        "torch_scorer_capture_surfaces": sorted(scorer_capture_by_surface),
        "scorer_surface_count": len(dict(scorer_tensors_by_surface or {}))
        + len(scorer_capture_by_surface),
        "output2_boundary_verdict": output2_boundary,
    }
    return row


def build_snerv_output2_boundary_verdict(
    *,
    tensors_by_surface: Mapping[str, Mapping[str, Any]],
    archive_decoder_header: Mapping[str, Any],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Classify whether ``output_2`` is one causal tensor or a boundary gap.

    This is intentionally stricter than RGB parity.  A receiver may produce the
    correct uint8 frames while ``output_2`` is absent, reparameterized, or merely
    a frame-shaped residual side channel.  Long-run SNeRV source-forward launch
    needs that distinction as a typed proof field, not an English note.
    """

    surfaces = {
        surface: dict(tensors_by_surface.get(surface) or {})
        for surface in SOURCE_FORWARD_SURFACES
    }
    has_output2 = {
        surface: "output_2" in tensors for surface, tensors in surfaces.items()
    }
    shapes = {
        surface: list(np.asarray(tensors["output_2"]).shape)
        for surface, tensors in surfaces.items()
        if "output_2" in tensors
    }
    storage_raw = dict(archive_decoder_header).get("tub_output2_storage")
    storage = dict(storage_raw) if isinstance(storage_raw, Mapping) else {}
    receiver_consumes_output2 = bool(
        storage.get("receiver_frame_decode_consumes_output2")
    )
    receiver_shape_matches = bool(storage.get("receiver_output2_frame_shape_match"))
    source_payload_present = bool(storage.get("source_payload_present"))
    stored = bool(storage.get("stored"))
    official_has = bool(has_output2["official_torch"])
    receiver_has = bool(
        has_output2["archive_parseback"] or has_output2["numpy_receiver"]
    )
    all_have = all(has_output2.values())

    blockers: list[str] = []
    verdict = DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
    required_next_step = (
        "capture_upstream_output2_and_receiver_output2_in_same_coordinate_system"
    )
    if all_have:
        deltas = {
            surface: _max_abs_delta_or_none(
                surfaces["official_torch"]["output_2"],
                surfaces[surface]["output_2"],
            )
            for surface in SOURCE_FORWARD_SURFACES
            if surface != "official_torch"
        }
        if any(value is None for value in deltas.values()):
            verdict = DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
            blockers.append("snerv_output2_shape_mismatch_across_surfaces")
            blockers.append("snerv_output2_adapter_would_be_required")
            required_next_step = "drop_output2_and_store_mfu_hfr_tub_lf_hf_basis"
        elif any(float(value) > float(tolerance) for value in deltas.values()):
            verdict = REPARAMETERIZED_RENAME_REQUIRED
            blockers.append("snerv_output2_value_mismatch_across_surfaces")
            required_next_step = "rename_receiver_state_or_rederive_output2_from_basis"
        elif not receiver_consumes_output2:
            verdict = REPARAMETERIZED_RENAME_REQUIRED
            blockers.append("snerv_output2_tensor_present_but_not_receiver_consumed")
            required_next_step = "rename_receiver_side_output2_or_drop_stored_output2"
        else:
            verdict = SOURCE_IDENTICAL
            required_next_step = "output2_boundary_closed"
    elif not any(has_output2.values()):
        verdict = DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
        blockers.append("snerv_output2_not_in_selected_source_forward_basis")
        required_next_step = "store_lf_hf_mfu_hfr_tub_pair_adapter_and_derive_output2"
    elif receiver_has and not official_has:
        verdict = REPARAMETERIZED_RENAME_REQUIRED
        blockers.append("snerv_receiver_output2_present_without_upstream_output2")
        required_next_step = "rename_receiver_side_output2_or_capture_upstream_output2"
    elif official_has and not receiver_has:
        verdict = DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
        blockers.append("snerv_upstream_output2_not_receiver_bound")
        required_next_step = "derive_output2_from_mfu_hfr_tub_basis_or_elide_payload"
    else:
        verdict = DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
        blockers.append("snerv_output2_partial_surface_set")
        required_next_step = "drop_output2_until_all_authorities_share_causal_basis"

    if stored and source_payload_present and not receiver_shape_matches:
        if verdict == SOURCE_IDENTICAL:
            verdict = DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
        blockers.append("snerv_output2_stored_but_receiver_shape_mismatch")
        blockers.append("snerv_output2_adapter_would_be_required")
        required_next_step = "stop_calling_frame_shaped_payload_source_output2"

    return {
        "schema": SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
        "verdict": verdict,
        "passed": verdict == SOURCE_IDENTICAL,
        "has_output2_by_surface": has_output2,
        "output2_shapes_by_surface": shapes,
        "archive_tub_output2_storage": {
            "stored": stored,
            "source_payload_present": source_payload_present,
            "receiver_executes_output2_fusion_from_payload": bool(
                storage.get("receiver_executes_output2_fusion_from_payload")
            ),
            "receiver_frame_decode_consumes_output2": receiver_consumes_output2,
            "receiver_output2_frame_shape_match": receiver_shape_matches,
            "receiver_frame_decode_binding_status": storage.get(
                "receiver_frame_decode_binding_status"
            ),
            "score_lagrangian_admission": storage.get("score_lagrangian_admission"),
            "score_lagrangian_action": storage.get("score_lagrangian_action"),
        },
        "minimal_causal_basis_recommendation": (
            []
            if verdict == SOURCE_IDENTICAL
            else [
                "lf_carrier",
                "hf_carrier",
                "mfu_state",
                "hfr_state",
                "tub_temporal_state",
                "pair_adapter",
                "derive_output_2",
            ]
        ),
        "allowed_verdicts": [
            SOURCE_IDENTICAL,
            REPARAMETERIZED_RENAME_REQUIRED,
            DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS,
        ],
        "blockers": _ordered_unique(blockers),
        "required_next_step": required_next_step,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_MANIFEST_SCHEMA = (
    "snerv_official_torch_upstream_capture_manifest.v1"
)
SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_AUTHORITY = (
    "upstream_snerv_t_forward_source_graph"
)
SNERV_OFFICIAL_TORCH_UPSTREAM_SOURCE_LINES = "model/snerv_t.py:125-184"


def build_snerv_official_torch_upstream_capture_manifest(
    *,
    pair_ids: Sequence[int],
    tensor_names: Sequence[str],
    official_repo_dir: str | None = None,
    model_source_sha256: str | None = None,
    checkpoint_sha256: str | None = None,
    state_dict_sha256: str | None = None,
    decoder_len: int | None = None,
    capture_authority: str = SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_AUTHORITY,
) -> dict[str, Any]:
    """Build the manifest required for official Torch SourceForward authority.

    The manifest is intentionally source-bound: a caller must provide checkpoint
    and state-dict hashes from the actual upstream replay, not just receiver
    tensors that look numerically plausible.
    """

    if model_source_sha256 is None and official_repo_dir is not None:
        from pathlib import Path

        source_path = Path(official_repo_dir) / "model" / "snerv_t.py"
        if source_path.is_file():
            model_source_sha256 = sha256(source_path.read_bytes()).hexdigest()
    return {
        "schema": SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_MANIFEST_SCHEMA,
        "capture_authority": str(capture_authority),
        "model_class": "SNeRV_T",
        "model_source_path": "model/snerv_t.py",
        "model_source_lines": SNERV_OFFICIAL_TORCH_UPSTREAM_SOURCE_LINES,
        "model_source_sha256": model_source_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "state_dict_sha256": state_dict_sha256,
        "decoder_len": None if decoder_len is None else int(decoder_len),
        "pair_ids": [int(value) for value in pair_ids],
        "tensor_names": sorted(str(name) for name in tensor_names),
        "upstream_forward_replay_verified": True,
        "receiver_bound_capture": False,
        "source_forward_replay_authority": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def validate_snerv_official_torch_upstream_capture_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    pair_ids: Sequence[int],
    tensor_names: Sequence[str],
) -> dict[str, Any]:
    """Fail closed unless official Torch tensors are tied to upstream replay."""

    blockers: list[str] = []
    if manifest is None:
        return {
            "passed": False,
            "blockers": [
                "snerv_official_torch_upstream_capture_manifest_missing"
            ],
        }
    row = dict(manifest)
    if row.get("schema") != SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_MANIFEST_SCHEMA:
        blockers.append("snerv_official_torch_upstream_capture_manifest_schema_invalid")
    if (
        str(row.get("capture_authority") or "")
        != SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_AUTHORITY
    ):
        blockers.append(
            "snerv_official_torch_upstream_capture_authority_not_source_graph"
        )
    if row.get("upstream_forward_replay_verified") is not True:
        blockers.append("snerv_official_torch_upstream_forward_replay_not_verified")
    if row.get("receiver_bound_capture") is True:
        blockers.append("snerv_official_torch_capture_is_receiver_bound")
    if row.get("source_forward_replay_authority") is not True:
        blockers.append("snerv_official_torch_source_forward_authority_false")
    if row.get("model_class") != "SNeRV_T":
        blockers.append("snerv_official_torch_model_class_not_snerv_t")
    if row.get("model_source_path") != "model/snerv_t.py":
        blockers.append("snerv_official_torch_model_source_path_invalid")
    if row.get("model_source_lines") != SNERV_OFFICIAL_TORCH_UPSTREAM_SOURCE_LINES:
        blockers.append("snerv_official_torch_model_source_lines_invalid")
    for field in ("model_source_sha256", "checkpoint_sha256", "state_dict_sha256"):
        if not _looks_like_sha256(row.get(field)):
            blockers.append(f"snerv_official_torch_{field}_invalid")
    manifest_pair_ids = _normalize_pair_ids(row.get("pair_ids"))
    expected_pair_ids = _normalize_pair_ids(pair_ids)
    if manifest_pair_ids != expected_pair_ids:
        blockers.append("snerv_official_torch_pair_ids_mismatch")
    manifest_tensor_names = {str(name) for name in row.get("tensor_names") or ()}
    supplied_tensor_names = {str(name) for name in tensor_names}
    missing_from_manifest = sorted(supplied_tensor_names - manifest_tensor_names)
    if missing_from_manifest:
        blockers.append(
            "snerv_official_torch_manifest_missing_supplied_tensors:"
            + ",".join(missing_from_manifest)
        )
    unknown = sorted(manifest_tensor_names - set(SOURCE_FORWARD_TENSOR_NAMES))
    if unknown:
        blockers.append(
            "snerv_official_torch_manifest_unknown_tensors:" + ",".join(unknown)
        )
    return {
        "passed": not _ordered_unique(blockers),
        "blockers": _ordered_unique(blockers),
    }


def _official_torch_tensor_capture_authority(
    *,
    official_torch_tensors: Mapping[str, Any] | None,
    receiver_bound_capture: bool,
    manifest_status: Mapping[str, Any],
) -> str:
    if receiver_bound_capture:
        return "receiver_bound_torch_ops_not_upstream_source_forward"
    if official_torch_tensors is None:
        return "missing_metadata_only_official_torch_surface"
    if manifest_status.get("passed") is True:
        return SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_AUTHORITY
    return "metadata_only_missing_upstream_source_manifest"


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
            if key in {"coord_time_embedding", "tub_in", "tub_out"}
        },
    )


def build_official_torch_primitive_tensors_from_archive_packet(
    *,
    archive_packet: bytes,
    pair_ids: Sequence[int],
    portable_tub_tensors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture receiver-bound official MFU/HFR/output2 tensors with Torch ops.

    This is a diagnostic parity reducer, not upstream trained-checkpoint source
    authority.  The producer marks its provenance fail-closed when inserted as
    the ``official_torch`` SourceForwardProof surface.
    """

    import torch

    decoded = unpack_snerv_archive(archive_packet)
    payload = decoded.decode_official_mfu_hfr_tub_payload()
    low, skip_mid, skip_high = payload.mfu_inputs()
    output2_inputs = payload.tub_output2_inputs()
    frames = decoded.decode_pair_frames(pair_ids, clip_to_uint8_range=True)
    if frames.ndim != 5:
        raise ValueError(
            "decoded SNeRV archive pair frames must be shaped (pairs,2,3,H,W); "
            f"got {tuple(frames.shape)}"
        )
    mfu = payload.build_mfu()
    hfr = payload.build_hfr_heads()
    selected_pair_ids = [int(value) for value in pair_ids]
    if not selected_pair_ids:
        raise ValueError("official Torch primitive tensor capture needs pair ids")
    frame_indices = _pair_ids_to_frame_indices(selected_pair_ids, int(frames.shape[0]))
    low_t = _torch_take_frames(low, frame_indices)
    skip_mid_t = _torch_take_frames(skip_mid, frame_indices)
    skip_high_t = _torch_take_frames(skip_high, frame_indices)
    mfu_out = _torch_mfu_forward(
        mfu,
        low_t,
        skip_mid_t,
        skip_high_t,
    )
    lh = _torch_hfr_head_forward(hfr.lh_head, mfu_out["pyr_out"])
    hl = _torch_hfr_head_forward(hfr.hl_head, mfu_out["pyr_out"])
    hh = _torch_hfr_head_forward(hfr.hh_head, mfu_out["pyr_out"])
    hfr_out = torch.stack((lh, hl, hh), dim=2)
    recon = _torch_haar_idwt2_level_nchw(mfu_out["pyr_out"], lh, hl, hh)
    output2_fused = None
    if output2_inputs is not None:
        tub_config = dict(payload.header.get("tub_config") or {})
        if "fc_hw" not in tub_config:
            raise ValueError("official Torch output2 capture requires fc_hw")
        _decoder_input, output2_fused = _torch_output2_fusion(
            output2_inputs[0],
            output2_inputs[1],
            fc_hw=tuple(int(value) for value in tub_config["fc_hw"]),
        )
        residual = output2_fused[frame_indices]
        if tuple(residual.shape) == tuple(recon.shape):
            recon = torch.clamp(recon + residual, 0.0, 255.0)
    h, w = (int(frames.shape[-2]), int(frames.shape[-1]))
    pair = recon.reshape((len(selected_pair_ids), 2, 3, h, w))
    pair255 = torch.clamp(pair, 0.0, 255.0)
    tensors: dict[str, Any] = {
        "mfu_in": _pack_tensor_group(
            ("low", _torch_to_numpy(low_t)),
            ("skip_mid", _torch_to_numpy(skip_mid_t)),
            ("skip_high", _torch_to_numpy(skip_high_t)),
        ),
        "mfu_out": _pack_tensor_group(
            ("up1", _torch_to_numpy(mfu_out["up1"])),
            ("cat_mid", _torch_to_numpy(mfu_out["cat_mid"])),
            ("unet1", _torch_to_numpy(mfu_out["unet1"])),
            ("unet1_up", _torch_to_numpy(mfu_out["unet1_up"])),
            ("cat_high", _torch_to_numpy(mfu_out["cat_high"])),
            ("pyr_out", _torch_to_numpy(mfu_out["pyr_out"])),
        ),
        "hfr_in": _torch_to_numpy(mfu_out["pyr_out"]),
        "hfr_out": _torch_to_numpy(hfr_out),
        "rgb_pair_float": _torch_to_numpy(pair255),
        "rgb_pair_uint8": np.clip(
            np.rint(_torch_to_numpy(pair255)),
            0,
            255,
        ).astype(np.uint8),
    }
    if output2_fused is not None:
        tensors["output_2"] = _torch_to_numpy(output2_fused)
    supplied_portable = dict(portable_tub_tensors or {})
    for name in ("tub_in", "tub_out"):
        if name in supplied_portable:
            tensors[name] = np.asarray(supplied_portable[name], dtype=np.float32)
    return {
        "schema": "snerv_official_torch_receiver_bound_primitive_tensor_bundle.v1",
        "surface": "official_torch",
        "pair_ids": selected_pair_ids,
        "tensor_names": sorted(tensors),
        "tensors": tensors,
        "mfu_hfr_output2_rgb_backend": "torch",
        "source_forward_replay_authority": False,
        "authority_blocker": "receiver_bound_torch_ops_not_upstream_source_forward",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_official_torch_upstream_fixture_tensors(
    *,
    official_repo_dir: str | None = None,
    train_one_step: bool = False,
    official_trained_checkpoint_state_dict: Mapping[str, Any] | None = None,
    official_trained_checkpoint_state_dict_path: str | None = None,
    official_trained_checkpoint_state_dict_kind: str = (
        "official_trained_checkpoint_state_dict"
    ),
) -> dict[str, Any]:
    """Capture official Torch tensors from pinned upstream ``SNeRV_T.forward``."""

    from tac.analysis.snerv_official_tub_source_forward_replay import (
        DEFAULT_OFFICIAL_SNERV_REPO,
        build_snerv_official_tub_source_forward_tensor_bundle,
    )

    return build_snerv_official_tub_source_forward_tensor_bundle(
        official_repo_dir=official_repo_dir or DEFAULT_OFFICIAL_SNERV_REPO,
        train_one_step=train_one_step,
        official_trained_checkpoint_state_dict=official_trained_checkpoint_state_dict,
        official_trained_checkpoint_state_dict_path=(
            official_trained_checkpoint_state_dict_path
        ),
        official_trained_checkpoint_state_dict_kind=(
            official_trained_checkpoint_state_dict_kind
        ),
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
    allow_missing_reference: bool = False,
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
        if not allow_missing_reference:
            raise ValueError(
                f"reference_surface {reference_surface!r} missing from scorer metrics"
            )
        delta_score_nonrate = 0.0
    else:
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


def _surface_rgb_pair_tensor_for_scorer(surface_tensors: Mapping[str, Any]) -> Any | None:
    tensors = dict(surface_tensors)
    if "rgb_pair_uint8" in tensors:
        return tensors["rgb_pair_uint8"]
    if "rgb_pair_float" in tensors:
        return tensors["rgb_pair_float"]
    return None


def _max_abs_delta_or_none(left: Any, right: Any) -> float | None:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return None
    delta = np.abs(left_arr - right_arr)
    if not np.all(np.isfinite(delta)):
        return None
    return float(np.max(delta)) if delta.size else 0.0


def _pair_ids_to_frame_indices(pair_ids: Sequence[int], pair_count: int) -> list[int]:
    out: list[int] = []
    for pair_id in pair_ids:
        idx = int(pair_id)
        if idx < 0 or idx >= int(pair_count):
            raise ValueError(
                f"pair id {idx} outside available range [0,{int(pair_count)})"
            )
        out.extend((2 * idx, 2 * idx + 1))
    return out


def _torch_take_frames(value: Any, frame_indices: Sequence[int]) -> Any:
    import torch

    arr = torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32)
    return arr[torch.as_tensor(list(frame_indices), dtype=torch.long)].contiguous()


def _torch_conv2d(conv: Any, x: Any) -> Any:
    import torch.nn.functional as F

    weight = _torch_weight(conv.weight)
    bias = None if conv.bias is None else _torch_bias(conv.bias)
    return F.conv2d(
        x,
        weight,
        bias=bias,
        stride=int(conv.stride),
        padding=int(conv.padding),
    )


def _torch_conv_transpose2d(conv: Any, x: Any) -> Any:
    import torch.nn.functional as F

    weight = _torch_weight(conv.weight)
    bias = None if conv.bias is None else _torch_bias(conv.bias)
    return F.conv_transpose2d(
        x,
        weight,
        bias=bias,
        stride=tuple(int(value) for value in conv.stride),
        padding=tuple(int(value) for value in conv.padding),
        output_padding=tuple(int(value) for value in conv.output_padding),
        dilation=tuple(int(value) for value in conv.dilation),
        groups=int(conv.groups),
    )


def _torch_residual_block_forward(block: Any, x: Any) -> Any:
    import torch.nn.functional as F

    hidden = F.leaky_relu(_torch_conv2d(block.conv1, x), negative_slope=0.1)
    return x + _torch_conv2d(block.conv2, hidden)


def _torch_residual_blocks_forward(blocks: Any, x: Any) -> Any:
    out = _torch_conv2d(blocks.input_conv, x)
    for block in blocks.residual_blocks:
        out = _torch_residual_block_forward(block, out)
    return out


def _torch_mfu_forward(mfu: Any, low: Any, skip_mid: Any, skip_high: Any) -> dict[str, Any]:
    up1 = _torch_conv_transpose2d(mfu.upsample_mid, low)
    cat_mid = _torch_cat_channels((up1, skip_mid))
    unet1 = _torch_residual_blocks_forward(mfu.rb_mid, cat_mid)
    unet1_up = _torch_conv_transpose2d(mfu.upsample_high, unet1)
    cat_high = _torch_cat_channels((unet1_up, skip_high))
    pyr_out = _torch_residual_blocks_forward(mfu.rb_high, cat_high)
    return {
        "up1": up1,
        "cat_mid": cat_mid,
        "unet1": unet1,
        "unet1_up": unet1_up,
        "cat_high": cat_high,
        "pyr_out": pyr_out,
    }


def _torch_hfr_head_forward(head: Any, pyr_out: Any) -> Any:
    import torch.nn.functional as F

    hidden = F.leaky_relu(_torch_conv2d(head.conv1, pyr_out), negative_slope=0.1)
    return _torch_conv2d(head.conv2, hidden)


def _torch_cat_channels(values: Sequence[Any]) -> Any:
    import torch

    return torch.cat(tuple(values), dim=1)


def _torch_haar_idwt2_level_nchw(ll: Any, lh: Any, hl: Any, hh: Any) -> Any:
    import torch

    detail_channels = int(lh.shape[1])
    ll_channels = int(ll.shape[1])
    if ll_channels not in (1, detail_channels):
        raise ValueError(
            "official Torch MFU pyr_out channels must be 1 or match HFR detail "
            f"channels ({ll_channels} vs {detail_channels})"
        )
    if ll_channels == 1 and detail_channels != 1:
        ll = ll.expand(int(ll.shape[0]), detail_channels, int(ll.shape[2]), int(ll.shape[3]))
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    n, c_count, h, w = (
        int(ll.shape[0]),
        int(ll.shape[1]),
        int(ll.shape[2]),
        int(ll.shape[3]),
    )
    row0 = torch.stack((a, b), dim=-1).reshape((n, c_count, h, w * 2))
    row1 = torch.stack((c, d), dim=-1).reshape((n, c_count, h, w * 2))
    return torch.stack((row0, row1), dim=-2).reshape((n, c_count, h * 2, w * 2))


def _torch_output2_fusion(
    temporal_encoder_concat: Any,
    decoder_output: Any,
    *,
    fc_hw: tuple[int, int],
) -> tuple[Any, Any]:
    import torch

    temporal = torch.as_tensor(
        np.asarray(temporal_encoder_concat, dtype=np.float32),
        dtype=torch.float32,
    )
    raw = torch.as_tensor(
        np.asarray(decoder_output, dtype=np.float32),
        dtype=torch.float32,
    )
    emb_ch = int(temporal.shape[1]) // 2
    decoder_input = torch.cat((temporal[:, :emb_ch], temporal[:, emb_ch:]), dim=0)
    fc_h, fc_w = (int(fc_hw[0]), int(fc_hw[1]))
    out_n, _out_c, out_h, out_w = (
        int(raw.shape[0]),
        int(raw.shape[1]),
        int(raw.shape[2]),
        int(raw.shape[3]),
    )
    fused = (
        raw.reshape((out_n, -1, fc_h, fc_w, out_h, out_w))
        .permute((0, 1, 4, 2, 5, 3))
        .reshape((out_n, -1, fc_h * out_h, fc_w * out_w))
    )
    return decoder_input, fused


def _torch_weight(value: Any) -> Any:
    import torch

    return torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32)


def _torch_bias(value: Any) -> Any:
    import torch

    return torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32)


def _pack_tensor_group(*items: tuple[str, Any]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for _name, value in items:
        arr = np.asarray(value, dtype=np.float32)
        header = np.asarray(
            [float(arr.ndim), *[float(dim) for dim in arr.shape], float(arr.size)],
            dtype=np.float32,
        )
        arrays.append(header.reshape(-1))
        arrays.append(arr.reshape(-1))
    if not arrays:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(arrays).astype(np.float32, copy=False)


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


def _looks_like_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _normalize_pair_ids(value: Any) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    if not value:
        return []
    try:
        out = [int(item) for item in value]
    except (TypeError, ValueError):
        return []
    if any(item < 0 for item in out):
        return []
    return out


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


__all__ = [
    "DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS",
    "REPARAMETERIZED_RENAME_REQUIRED",
    "SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_AUTHORITY",
    "SNERV_OFFICIAL_TORCH_UPSTREAM_CAPTURE_MANIFEST_SCHEMA",
    "SNERV_OFFICIAL_TORCH_UPSTREAM_SOURCE_LINES",
    "SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA",
    "SOURCE_GRAPH_UNPROVEN",
    "SOURCE_IDENTICAL",
    "build_official_torch_primitive_tensors_from_archive_packet",
    "build_official_torch_upstream_fixture_tensors",
    "build_pact_mlx_primitive_tensors_from_archive_packet",
    "build_snerv_official_torch_upstream_capture_manifest",
    "build_snerv_output2_boundary_verdict",
    "build_snerv_scorer_deltas_from_surface_metrics",
    "build_snerv_source_forward_proof_from_archive_packet",
    "build_torch_scorer_source_forward_surface",
    "validate_snerv_official_torch_upstream_capture_manifest",
]
