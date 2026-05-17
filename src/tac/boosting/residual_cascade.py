# SPDX-License-Identifier: MIT
"""ResidualCascadeBuilder — generalize PR106 format0d 2-pass additive
correction to N-pass cascade per spec §I.6.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§I.6 "Residual cascade — single decoder, no cascade → decoder_2 corrects
decoder_1's residual; decoder_3 corrects decoder_2's; each stage adds
bytes ∝ remaining distortion".

The empirical anchor (PV-5):
  submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575
  decode_format0d_sidecar returns 4 arrays (base_dim, base_delta_q,
  extra_dim, extra_delta_q). The wire format is:
    base_payload || struct.pack('<H', extra_len) || extra_payload || meta_6

The cascade builder is a STAGE-COMPOSITION abstraction (it produces a
list of BoostStageContract objects, one per cascade depth, that the
pipeline runs sequentially). The per-stage codec is delegated to the
caller's stage function — the cascade builder ensures byte-deterministic
ORDERING + correct PARENT_STAGE_ID linking + cascade-aware
``correction_kind`` declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.boosting.contract import BoostStageContract

__all__ = [
    "ResidualCascadeBuilder",
    "ResidualCascadeStageSpec",
]


@dataclass(frozen=True)
class ResidualCascadeStageSpec:
    """Specification for a single stage in a residual cascade.

    Frozen so cascade composition is structurally immutable. The
    ``parent_stage_id`` is auto-linked by the builder to enforce the
    cascade ordering invariant (stage N consumes stage N-1's residual).
    """

    stage_id: str
    correction_resolution: str = "per_pair"
    max_bytes_added: int | None = None
    sensitivity_weighted: bool = False
    lane_id: str | None = None


class ResidualCascadeBuilder:
    """Builder for an N-pass additive residual cascade.

    The PR106 format0d 2-pass pattern generalizes to depth N as::

        decoder_1: emits frames_v1 from seed state
        decoder_2: consumes (frames_v1, predicted_distortion_v1)
                   emits frames_v2 + residual_correction_v2
        ...
        decoder_N: consumes (frames_v{N-1}, predicted_distortion_v{N-1})
                   emits frames_vN + residual_correction_vN

    Each stage's emit key is versioned (``_v<N>``) so the pipeline's
    ambiguous-emit detector accepts the cascade as structurally
    unambiguous (each version is consumed by exactly one downstream stage).

    Usage::

        builder = ResidualCascadeBuilder(
            root_stage_id="raw_decoder",
            depth=3,
            stage_specs=[
                ResidualCascadeStageSpec(stage_id="cascade_pose_residual_1"),
                ResidualCascadeStageSpec(stage_id="cascade_pose_residual_2"),
                ResidualCascadeStageSpec(stage_id="cascade_pose_residual_3"),
            ],
        )
        contracts = builder.build_contracts()
        # Caller now decorates the corresponding stage functions with
        # @boost_stage(contracts[i])
    """

    def __init__(
        self,
        *,
        root_stage_id: str,
        depth: int,
        stage_specs: list[ResidualCascadeStageSpec],
        lane_id: str | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError(
                f"ResidualCascadeBuilder depth={depth} must be >= 1"
            )
        if len(stage_specs) != depth:
            raise ValueError(
                f"ResidualCascadeBuilder requires {depth} stage_specs; "
                f"got {len(stage_specs)}"
            )
        if not isinstance(root_stage_id, str) or not root_stage_id.strip():
            raise ValueError(
                f"root_stage_id={root_stage_id!r} must be a non-empty string"
            )
        seen_ids: set[str] = {root_stage_id}
        for spec in stage_specs:
            if spec.stage_id in seen_ids:
                raise ValueError(
                    f"Duplicate stage_id in cascade: {spec.stage_id!r}"
                )
            seen_ids.add(spec.stage_id)

        self.root_stage_id = root_stage_id
        self.depth = depth
        self.stage_specs = list(stage_specs)
        self.lane_id = lane_id

    def build_contracts(self) -> list[BoostStageContract]:
        """Build the list of BoostStageContract for each cascade depth.

        Each stage's:
          - ``parent_stage_id`` is the prior stage's id (or root_stage_id
            for stage 1).
          - ``consumes`` is ``{"frames_v<i-1>", "predicted_distortion_v<i-1>"}``
            (or ``{"frames_v0"}`` for stage 1, consuming the seed).
          - ``emits`` is ``{"frames_v<i>", "residual_correction_v<i>"}``.
          - ``correction_kind`` is ``"additive"`` (the canonical residual
            cascade pattern).
        """
        contracts: list[BoostStageContract] = []
        prior_stage_id = self.root_stage_id
        prior_version = 0
        for i, spec in enumerate(self.stage_specs, start=1):
            consumes = (
                frozenset({f"frames_v{prior_version}"})
                if i == 1
                else frozenset(
                    {
                        f"frames_v{prior_version}",
                        f"predicted_distortion_v{prior_version}",
                    }
                )
            )
            emits = frozenset(
                {
                    f"frames_v{i}",
                    f"residual_correction_v{i}",
                }
            )
            # Per the spec §I.6: each stage's hook_probe_disambiguator is
            # None with explicit rationale that a residual cascade has a
            # canonical additive interpretation (no 2+ defensible alts).
            contract = BoostStageContract(
                id=spec.stage_id,
                parent_stage_id=prior_stage_id,
                stage_phase="compress",
                description=(
                    f"Residual cascade stage {i}/{self.depth}; consumes "
                    f"prior stage's frames_v{prior_version} + predicted "
                    f"distortion, emits additive correction to frames_v{i}."
                ),
                consumes=consumes,
                emits=emits,
                correction_kind="additive",
                correction_resolution=spec.correction_resolution,
                deterministic=True,
                scorer_free=True,
                sensitivity_weighted=spec.sensitivity_weighted,
                max_bytes_added=spec.max_bytes_added,
                merge_policy="last_writer_wins",
                hook_sensitivity_contribution=(
                    "master_gradient_v1"
                    if spec.sensitivity_weighted
                    else "not_applicable_with_rationale"
                ),
                hook_pareto_constraint="rate_distortion_v1",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="cathedral_autopilot_v1",
                hook_continual_learning_anchor_kind="boosting_stage_outcomes_v1",
                hook_probe_disambiguator=None,
                hook_not_applicable_rationale={
                    "hook_probe_disambiguator": (
                        "Residual cascade has a single canonical additive "
                        "interpretation per PR106 format0d empirical anchor "
                        "(submissions/pr106_latent_sidecar_r2_pr101_grammar/"
                        "inflate.py:549-575). No 2+ defensible alts."
                    ),
                    **(
                        {}
                        if spec.sensitivity_weighted
                        else {
                            "hook_sensitivity_contribution": (
                                "Stage does not consume master_gradient; "
                                "operator opted for fixed per-pair correction."
                            )
                        }
                    ),
                    "hook_bit_allocator_class": (
                        "Residual cascade emits bytes additively; the codec "
                        "stage downstream owns bit allocation per Catalog "
                        "#272 distinguishing-feature integration contract."
                    ),
                },
                lane_id=spec.lane_id or self.lane_id,
                design_memo=(
                    ".omx/research/"
                    "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
                ),
                canonical_vs_unique_decision=(
                    "ADOPT_CANONICAL: BoostStageContract template per spec "
                    "§5.3 — cascade ordering + emit versioning is the same "
                    "across all cascade stages (no substrate-specific reason "
                    "to fork)."
                ),
            )
            contracts.append(contract)
            prior_stage_id = spec.stage_id
            prior_version = i
        return contracts
