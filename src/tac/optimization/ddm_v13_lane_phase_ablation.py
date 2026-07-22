# SPDX-License-Identifier: MIT
"""Typed contract for the receiver-closed DDM v13 Lane phase-symbol ablation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.optimization.direct_description_carrier_compose import LaneDriftKnotV1
from tac.optimization.direct_description_minimizer import rfc8785_canonicalize

CONFIG_SCHEMA = "DDMV13LanePhaseAblationConfigV1"
RESULT_SCHEMA = "ddm_v13_lane_phase_ablation_receipt.v1"


class DDMV13LanePhaseAblationConfigV1(BaseModel):
    """Strict local-only config for one n64 or n600 phase ablation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DDMV13LanePhaseAblationConfigV1"] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: StrictStr
    pair_start: StrictInt = Field(ge=0, le=599)
    pair_count: Literal[64, 600]
    v13_config_path: StrictStr
    v13_config_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    v13_receipt_path: StrictStr
    v13_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(default=4, ge=1, le=8)
    max_measurements_per_invocation: Literal[1] = 1
    phase_policy: Literal["xi_phase_predict_then_sparse_per_pair_phase_symbols_camera_resolution_before_R"] = (
        "xi_phase_predict_then_sparse_per_pair_phase_symbols_camera_resolution_before_R"
    )
    successor_scope: Literal["raw_phase_ablation_before_anisotropic_ar1_whitening"] = (
        "raw_phase_ablation_before_anisotropic_ar1_whitening"
    )
    ranker: Literal["frozen_scorer_flip_margin_not_pixel_energy"] = "frozen_scorer_flip_margin_not_pixel_energy"
    seed: Literal[1234] = 1234
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _window(self) -> DDMV13LanePhaseAblationConfigV1:
        if self.pair_count == 600 and self.pair_start != 0:
            raise ValueError("n600 phase ablation must cover exact pairs [0,600)")
        if self.pair_count == 64 and self.pair_start + self.pair_count > 600:
            raise ValueError("n64 phase ablation escapes the scorer cache")
        return self

    def typed_config_hash(self) -> str:
        import hashlib

        payload = rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))
        return hashlib.sha256(payload).hexdigest()


def phase_only_knots(rows: tuple[LaneDriftKnotV1, ...]) -> tuple[LaneDriftKnotV1, ...]:
    """Retain only counted per-pair phase symbols; discard geometry/width fields."""

    return tuple(
        LaneDriftKnotV1(
            line_index=row.line_index,
            pair_index=row.pair_index,
            phase_delta_q8=row.phase_delta_q8,
        )
        for row in rows
        if row.phase_delta_q8
    )


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "DDMV13LanePhaseAblationConfigV1",
    "phase_only_knots",
]
