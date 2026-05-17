# SPDX-License-Identifier: MIT
"""Canonical example boost stages for the tac.boosting namespace.

Three minimal stages that exercise the decorator + composition API:

  1. ``raw_decoder``: passthrough seed stage; emits ``frames_v0`` from
     the seed state's ``seed_frames`` key.
  2. ``cascade_pose_residual_v1``: depth-1 additive pose residual cascade;
     consumes ``frames_v0`` + ``predicted_distortion_v0``, emits
     ``frames_v1`` + ``residual_correction_v1``.
  3. ``cascade_seg_residual_v1``: depth-1 additive seg residual cascade;
     consumes ``frames_v1`` + ``predicted_distortion_v1``, emits
     ``frames_v2`` + ``residual_correction_v2``.

The example correction values are TOY (zeros / small constants) so the
stages are testable without GPU. Real consumers replace the body with
substrate-specific per-pair correction tensors.

Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — every claim in
the docstrings is backed by an executable body.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.boosting.contract import BoostStageContract
from tac.boosting.decorator import boost_stage


@boost_stage(
    BoostStageContract(
        id="raw_decoder",
        parent_stage_id=None,
        stage_phase="compress",
        description=(
            "Passthrough seed stage; emits frames_v0 from the seed state's "
            "seed_frames key. Used as the parent of every cascade."
        ),
        consumes=frozenset({"seed_frames"}),
        emits=frozenset({"frames_v0", "predicted_distortion_v0"}),
        correction_kind="replace",
        correction_resolution="global",
        deterministic=True,
        scorer_free=True,
        sensitivity_weighted=False,
        max_bytes_added=0,
        merge_policy="last_writer_wins",
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_pareto_constraint="rate_distortion_v1",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="cathedral_autopilot_v1",
        hook_continual_learning_anchor_kind="boosting_stage_outcomes_v1",
        hook_probe_disambiguator=None,
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": (
                "Passthrough stage emits zero residual; sensitivity weighting "
                "is meaningless at the seed."
            ),
            "hook_bit_allocator_class": (
                "Passthrough emits zero bytes; bit allocation is undefined."
            ),
            "hook_probe_disambiguator": (
                "Seed-passthrough has a single canonical interpretation."
            ),
        },
        lane_id="lane_tac_boosting_namespace_decorator_api_20260517",
        design_memo=(
            ".omx/research/"
            "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
        ),
        canonical_vs_unique_decision=(
            "ADOPT_CANONICAL: seed stage is identical across substrates "
            "(emits raw frames; zero residual)."
        ),
    )
)
def raw_decoder(state: Mapping[str, Any], *, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Seed stage: emits frames_v0 unchanged + zero predicted distortion."""
    seed = state.get("seed_frames")
    if seed is None:
        raise ValueError(
            "raw_decoder requires seed_frames in state; got "
            f"{sorted(state.keys())}"
        )
    return {
        "frames_v0": seed,
        "predicted_distortion_v0": 0.0,
    }


@boost_stage(
    BoostStageContract(
        id="cascade_pose_residual_v1",
        parent_stage_id="raw_decoder",
        stage_phase="compress",
        description=(
            "Depth-1 additive pose residual cascade. Consumes frames_v0 + "
            "predicted_distortion_v0; emits frames_v1 + residual_correction_v1."
        ),
        consumes=frozenset({"frames_v0", "predicted_distortion_v0"}),
        emits=frozenset({"frames_v1", "residual_correction_v1"}),
        correction_kind="additive",
        correction_resolution="per_pair",
        deterministic=True,
        scorer_free=True,
        sensitivity_weighted=False,
        max_bytes_added=512,
        merge_policy="last_writer_wins",
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_pareto_constraint="rate_distortion_v1",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="cathedral_autopilot_v1",
        hook_continual_learning_anchor_kind="boosting_stage_outcomes_v1",
        hook_probe_disambiguator=None,
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": (
                "Example stage applies a TOY constant correction; production "
                "consumers will set sensitivity_weighted=True and consume "
                "the master gradient."
            ),
            "hook_bit_allocator_class": (
                "Example stage emits a fixed 512-byte correction; downstream "
                "codec owns bit allocation per Catalog #272."
            ),
            "hook_probe_disambiguator": (
                "Residual cascade canonical per PR106 format0d anchor."
            ),
        },
        lane_id="lane_tac_boosting_namespace_decorator_api_20260517",
        design_memo=(
            ".omx/research/"
            "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
        ),
        canonical_vs_unique_decision=(
            "ADOPT_CANONICAL: cascade ordering + emit versioning shared "
            "across all cascade stages."
        ),
    )
)
def cascade_pose_residual_v1(
    state: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Toy depth-1 pose residual cascade.

    Emits frames_v1 = frames_v0 (zero residual) + a small constant
    predicted_distortion_v1. Real consumers compute a per-pair pose
    correction tensor and add it to frames_v0.
    """
    frames_v0 = state["frames_v0"]
    predicted_distortion_v0 = state["predicted_distortion_v0"]
    # Toy implementation: zero residual
    return {
        "frames_v1": frames_v0,
        "residual_correction_v1": 0.0,
        "predicted_distortion_v1": predicted_distortion_v0 * 0.5,
    }


@boost_stage(
    BoostStageContract(
        id="cascade_seg_residual_v1",
        parent_stage_id="cascade_pose_residual_v1",
        stage_phase="compress",
        description=(
            "Depth-1 additive seg residual cascade. Consumes frames_v1 + "
            "predicted_distortion_v1; emits frames_v2 + residual_correction_v2."
        ),
        consumes=frozenset({"frames_v1", "predicted_distortion_v1"}),
        emits=frozenset({"frames_v2", "residual_correction_v2"}),
        correction_kind="additive",
        correction_resolution="per_pair",
        deterministic=True,
        scorer_free=True,
        sensitivity_weighted=False,
        max_bytes_added=512,
        merge_policy="last_writer_wins",
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_pareto_constraint="rate_distortion_v1",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="cathedral_autopilot_v1",
        hook_continual_learning_anchor_kind="boosting_stage_outcomes_v1",
        hook_probe_disambiguator=None,
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": (
                "Example stage applies a TOY constant correction; production "
                "consumers will set sensitivity_weighted=True."
            ),
            "hook_bit_allocator_class": (
                "Example stage emits a fixed 512-byte correction."
            ),
            "hook_probe_disambiguator": (
                "Residual cascade canonical per PR106 format0d anchor."
            ),
        },
        lane_id="lane_tac_boosting_namespace_decorator_api_20260517",
        design_memo=(
            ".omx/research/"
            "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
        ),
        canonical_vs_unique_decision=(
            "ADOPT_CANONICAL: cascade ordering canonical."
        ),
    )
)
def cascade_seg_residual_v1(
    state: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Toy depth-1 seg residual cascade.

    Emits frames_v2 = frames_v1 (zero residual). Real consumers compute
    a per-pair seg correction tensor and add it to frames_v1.
    """
    frames_v1 = state["frames_v1"]
    predicted_distortion_v1 = state["predicted_distortion_v1"]
    return {
        "frames_v2": frames_v1,
        "residual_correction_v2": 0.0,
        "predicted_distortion_v2": predicted_distortion_v1 * 0.5,
    }
