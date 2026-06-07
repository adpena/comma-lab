# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

import numpy as np

from tac.analysis.snerv_source_forward_proof import (
    SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
    SOURCE_FORWARD_SURFACES,
    SOURCE_IDENTICAL,
    build_snerv_payload_bitflip_falsification,
    build_snerv_source_forward_proof_action_effect,
    build_snerv_source_forward_surface_provenance,
)


def valid_snerv_source_forward_action_effect(
    *,
    action_id: str = "a" * 64,
    archive_sha256: str = "1" * 64,
    archive_bytes: int = 12345,
    pair_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Return a small valid SNeRV source-forward ActionEffect fixture."""

    pair_ids = [0] if pair_ids is None else list(pair_ids)
    return build_snerv_source_forward_proof_action_effect(
        action_id=action_id,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        payload_section_hashes={
            "lf_payload": "a" * 64,
            "decoder_payload": "2" * 64,
            "output_2": "5" * 64,
        },
        pair_ids=pair_ids,
        tensors_by_surface=_snerv_tensor_surfaces(),
        scorer_deltas=_snerv_scorer_deltas(),
        destructive_payload_bit_flip=build_snerv_payload_bitflip_falsification(
            bitflip_section="decoder_payload.output_2",
            baseline_section_sha256="2" * 64,
            mutated_section_sha256="3" * 64,
            proof_passed_after_bitflip=False,
            first_failed_tensor="output_2",
            first_failed_surface="archive_parseback",
            bit_offset=17,
            bit_mask=1,
        ),
        output2_boundary_verdict=_source_identical_output2_verdict(),
        surface_provenance=build_snerv_source_forward_surface_provenance(
            pair_ids=pair_ids,
            archive_sha256=archive_sha256,
            extra_by_surface={"official_torch": _official_torch_lineage()},
        ),
    )


def _snerv_tensor_surfaces() -> dict[str, dict[str, np.ndarray]]:
    base = {
        "coord_time_embedding": np.array([[0.0, 1.0]], dtype=np.float64),
        "mfu_in": np.zeros((1, 1, 2, 2), dtype=np.float64),
        "mfu_out": np.ones((1, 1, 2, 2), dtype=np.float64),
        "hfr_in": np.ones((1, 1, 2, 2), dtype=np.float64),
        "hfr_out": np.full((1, 1, 2, 2), 2.0, dtype=np.float64),
        "tub_in": np.full((1, 1, 2, 2), 3.0, dtype=np.float64),
        "tub_out": np.full((1, 1, 2, 2), 4.0, dtype=np.float64),
        "output_2": np.full((1, 1, 2, 2), 5.0, dtype=np.float64),
        "rgb_pair_float": np.zeros((1, 2, 3, 2, 2), dtype=np.float64),
        "rgb_pair_uint8": np.zeros((1, 2, 3, 2, 2), dtype=np.uint8),
        "segnet_input": np.zeros((1, 3, 2, 2), dtype=np.float64),
        "posenet_input": np.zeros((1, 6, 2, 2), dtype=np.float64),
        "segnet_logits": np.zeros((1, 4, 2, 2), dtype=np.float64),
        "segnet_argmax": np.zeros((1, 2, 2), dtype=np.int64),
        "posenet_output": np.zeros((1, 12), dtype=np.float64),
    }
    return {
        surface: {name: value.copy() for name, value in base.items()}
        for surface in SOURCE_FORWARD_SURFACES
    }


def _snerv_scorer_deltas() -> dict[str, Any]:
    return {
        "d_seg": 0.0,
        "d_pose": 0.0,
        "delta_score_nonrate": 0.0,
        "by_surface": {
            surface: {"d_seg": 0.0, "d_pose": 0.0}
            for surface in SOURCE_FORWARD_SURFACES
        },
    }


def _official_torch_lineage() -> dict[str, str]:
    return {
        "trained_checkpoint_lineage": "official_trained_checkpoint_state_dict",
        "checkpoint_sha256": "6" * 64,
        "state_dict_sha256": "7" * 64,
        "model_source_sha256": "8" * 64,
        "source_scope": "official_trained_checkpoint",
        "capture_origin": "official_upstream_trained_checkpoint",
    }


def _source_identical_output2_verdict() -> dict[str, Any]:
    return {
        "schema": SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
        "verdict": SOURCE_IDENTICAL,
        "passed": True,
        "has_output2_by_surface": dict.fromkeys(SOURCE_FORWARD_SURFACES, True),
        "output2_shapes_by_surface": {
            surface: [1, 1, 2, 2] for surface in SOURCE_FORWARD_SURFACES
        },
        "archive_tub_output2_storage": {
            "section": "decoder_payload.output_2",
            "sha256": "5" * 64,
            "bytes": 64,
        },
        "minimal_causal_basis_recommendation": [
            "keep_output2_source_forward_bound"
        ],
        "blockers": [],
        "required_next_step": "output2_boundary_closed",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
