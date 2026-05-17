# SPDX-License-Identifier: MIT
"""PerPairDecoderEnsembleSelector — M decoders × per-pair selector.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§I.6 "Per-pair decoder ensemble — one decoder for all 600 pairs → M
decoder variants in archive; per-pair selector picks the best at inflate
via deterministic image-domain criterion".

The selector is structurally similar to fec6's K=16 mode-selector but
operates on DECODERS instead of palette modes. At compress time the
builder produces an archive grammar that holds M decoder variants + a
per-pair index byte stream. At inflate time the selector uses a
deterministic image-domain criterion (e.g. local variance, NOT a scorer)
to pick which decoder applies to each pair.

Per CLAUDE.md "Strict scorer rule": inflate-time selector function MUST
NOT load PoseNet/SegNet. The selector criterion is a deterministic
image-domain proxy (e.g. local variance, gradient magnitude, color
histogram diversity) that runs scorer-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.boosting.contract import BoostStageContract

__all__ = [
    "PerPairDecoderEnsembleSelector",
    "PerPairDecoderEnsembleSpec",
]


@dataclass(frozen=True)
class PerPairDecoderEnsembleSpec:
    """Specification for an M-decoder per-pair selector ensemble."""

    stage_id: str
    num_decoders: int  # M in the spec
    selector_criterion: str  # e.g. "local_variance" / "gradient_magnitude"
    per_pair_index_bits: int = 4  # bits per pair to select among M decoders
    decoder_archive_keys: tuple[str, ...] = ()  # archive byte streams
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.num_decoders < 2:
            raise ValueError(
                f"PerPairDecoderEnsembleSpec num_decoders={self.num_decoders} "
                f"must be >= 2 (an ensemble of one decoder is just the base)."
            )
        if self.per_pair_index_bits < 1:
            raise ValueError(
                f"per_pair_index_bits={self.per_pair_index_bits} must be >= 1"
            )
        # Selector criterion must be a known scorer-free image-domain proxy
        legal_criteria = {
            "local_variance",
            "gradient_magnitude",
            "color_histogram_diversity",
            "edge_density",
            "deterministic_motion_magnitude",
            "custom_image_domain",  # caller provides callable elsewhere
        }
        if self.selector_criterion not in legal_criteria:
            raise ValueError(
                f"selector_criterion={self.selector_criterion!r} not in "
                f"{sorted(legal_criteria)}. Per CLAUDE.md 'Strict scorer rule' "
                f"the criterion must be a deterministic image-domain proxy "
                f"(NOT a scorer forward)."
            )


class PerPairDecoderEnsembleSelector:
    """Build a per-pair decoder-ensemble selector stage contract.

    The builder produces a single BoostStageContract that:
      - has ``stage_phase='inflate'`` (selector runs at inflate time)
      - has ``scorer_free=True`` (criterion is image-domain only)
      - has ``correction_kind='replace'`` (per-pair selector replaces the
        decoder, not adds to it)
      - has ``correction_resolution='per_pair'``
      - declares the M decoder archive keys + the per-pair-index stream as
        consumes, and emits ``frames_v_selected``

    Sister of ResidualCascadeBuilder. ResidualCascadeBuilder cascades
    decoders depth-wise (each refines the prior); PerPairDecoderEnsembleSelector
    picks ONE of M decoders per pair (mutually exclusive selection).
    """

    def __init__(self, *, spec: PerPairDecoderEnsembleSpec) -> None:
        self.spec = spec

    def build_contract(self) -> BoostStageContract:
        s = self.spec
        # Build consumes: M decoder archive keys + per-pair index stream +
        # selector criterion input
        consumes = frozenset(
            (*s.decoder_archive_keys, f"per_pair_decoder_index_{s.num_decoders}", "frames_v0")
        )
        return BoostStageContract(
            id=s.stage_id,
            parent_stage_id=None,
            stage_phase="inflate",
            description=(
                f"Per-pair decoder ensemble selector (M={s.num_decoders} "
                f"decoders, criterion={s.selector_criterion!r}). At inflate "
                f"time picks one of M decoders per pair using the deterministic "
                f"image-domain criterion (scorer-free per CLAUDE.md strict-"
                f"scorer rule)."
            ),
            consumes=consumes,
            emits=frozenset({"frames_v_selected"}),
            correction_kind="replace",
            correction_resolution="per_pair",
            deterministic=True,
            scorer_free=True,
            sensitivity_weighted=False,
            max_bytes_added=(
                # 600 pairs * per_pair_index_bits bits, rounded up to bytes
                ((600 * s.per_pair_index_bits) + 7) // 8
            ),
            merge_policy="last_writer_wins",
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_pareto_constraint="rate_distortion_v1",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind="boosting_stage_outcomes_v1",
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": (
                    "Per-pair decoder selection is bounded by available "
                    "M decoder variants in the archive; master gradient does "
                    "not change the discrete selection set."
                ),
                "hook_bit_allocator_class": (
                    "Per-pair index stream uses uniform bit allocation per "
                    "pair (per_pair_index_bits constant across the video). "
                    "Adaptive allocation handled by sister codec stage."
                ),
                "hook_probe_disambiguator": (
                    f"Per-pair decoder selection has a single canonical "
                    f"interpretation (deterministic image-domain criterion "
                    f"per CLAUDE.md strict-scorer rule); the criterion "
                    f"{s.selector_criterion!r} is fixed at compress time."
                ),
            },
            lane_id=s.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
            ),
            canonical_vs_unique_decision=(
                "FORK_BECAUSE_PRINCIPLED_MISMATCH: The selector criterion "
                "is substrate-specific (different substrates expose different "
                "image-domain signals — fec6 uses K=16 palette diversity, "
                "PR106 uses latent_dim variance). The CONTRACT is canonical "
                "but the selector_criterion field admits substrate-optimal "
                "engineering per the operating mode."
            ),
        )
