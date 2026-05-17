# SPDX-License-Identifier: MIT
"""ModeEnsembleDispatch — K modes × M decoders product space per spec §I.6.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§I.6 "Mode ensemble — K=16 modes per pair (one decoder evaluated K ways)
→ K=16 modes × M=8 decoders = 128 per-pair candidates".

This is the COMPOSITION of fec6's K-mode selector × PerPairDecoderEnsembleSelector's
M-decoder selector. Per pair, the dispatcher evaluates K*M candidates
and selects the best via a deterministic image-domain criterion (NOT a
scorer, per CLAUDE.md strict-scorer rule).

The product-space size grows as K*M; at K=16 M=8 = 128 per-pair candidates,
which at 600 pairs = 76800 candidate evaluations. The dispatcher delegates
the per-candidate scoring to a caller-provided callable — the contract
itself only declares the structural product shape + the index encoding
(7 bits per pair to encode log2(128)=7 bits).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

from tac.boosting.contract import BoostStageContract

__all__ = [
    "ModeEnsembleDispatch",
    "ModeEnsembleDispatchSpec",
]


@dataclass(frozen=True)
class ModeEnsembleDispatchSpec:
    """Specification for a K-mode × M-decoder product-space dispatcher."""

    stage_id: str
    num_modes: int  # K in the spec
    num_decoders: int  # M in the spec
    selector_criterion: str
    decoder_archive_keys: tuple[str, ...] = ()
    mode_palette_archive_key: str = "mode_palette"
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.num_modes < 2:
            raise ValueError(
                f"num_modes={self.num_modes} must be >= 2"
            )
        if self.num_decoders < 2:
            raise ValueError(
                f"num_decoders={self.num_decoders} must be >= 2"
            )
        legal_criteria = {
            "local_variance",
            "gradient_magnitude",
            "color_histogram_diversity",
            "edge_density",
            "deterministic_motion_magnitude",
            "joint_mode_decoder_score_proxy",  # mode-aware image-domain proxy
            "custom_image_domain",
        }
        if self.selector_criterion not in legal_criteria:
            raise ValueError(
                f"selector_criterion={self.selector_criterion!r} not in "
                f"{sorted(legal_criteria)}"
            )

    @property
    def product_space_size(self) -> int:
        """K × M; the number of per-pair candidate evaluations."""
        return self.num_modes * self.num_decoders

    @property
    def per_pair_index_bits(self) -> int:
        """Bits needed to encode the per-pair (mode, decoder) selection."""
        return ceil(log2(self.product_space_size))


class ModeEnsembleDispatch:
    """Build a K-mode × M-decoder dispatch stage contract.

    The dispatcher produces a single BoostStageContract that:
      - has ``stage_phase='inflate'`` (dispatch runs at inflate time)
      - has ``scorer_free=True`` (criterion is image-domain only)
      - has ``correction_kind='replace'`` (selected (mode, decoder) replaces
        the seed per pair)
      - has ``correction_resolution='per_pair'``
      - declares the M decoder archive keys + mode palette + per-pair
        index stream as consumes, and emits ``frames_v_dispatched``

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": the dispatcher does
    NOT inherit from PerPairDecoderEnsembleSelector. The product-space
    semantics + index-bit budget + per-pair candidate evaluation are
    distinct enough that subclassing would suppress the substrate-optimal
    engineering. The two builders share the BoostStageContract dataclass
    but generate different contracts.
    """

    def __init__(self, *, spec: ModeEnsembleDispatchSpec) -> None:
        self.spec = spec

    def build_contract(self) -> BoostStageContract:
        s = self.spec
        consumes = frozenset(
            (
                *s.decoder_archive_keys,
                s.mode_palette_archive_key,
                f"per_pair_mode_decoder_index_{s.num_modes}x{s.num_decoders}",
                "frames_v0",
            )
        )
        return BoostStageContract(
            id=s.stage_id,
            parent_stage_id=None,
            stage_phase="inflate",
            description=(
                f"K={s.num_modes}-mode x M={s.num_decoders}-decoder ensemble "
                f"dispatch (product space K*M={s.product_space_size}, "
                f"{s.per_pair_index_bits} bits per pair). Selector criterion: "
                f"{s.selector_criterion!r} (scorer-free deterministic image-"
                f"domain proxy per CLAUDE.md strict-scorer rule)."
            ),
            consumes=consumes,
            emits=frozenset({"frames_v_dispatched"}),
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
                    "Per-pair dispatch is bounded by the discrete K*M "
                    "candidate set; master gradient does not change the set."
                ),
                "hook_bit_allocator_class": (
                    "Per-pair index uses uniform bit budget "
                    f"({s.per_pair_index_bits} bits per pair); adaptive "
                    "allocation is handled by the codec stage."
                ),
                "hook_probe_disambiguator": (
                    "Mode-decoder dispatch has a single canonical "
                    "interpretation (deterministic image-domain criterion "
                    f"{s.selector_criterion!r}); the criterion is fixed at "
                    "compress time."
                ),
            },
            lane_id=s.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
            ),
            canonical_vs_unique_decision=(
                "FORK_BECAUSE_PRINCIPLED_MISMATCH: The product-space "
                "dispatcher is structurally distinct from a single-decoder "
                "ensemble (the per-pair index encodes BOTH a mode and a "
                "decoder selection; the canonical bit budget grows as "
                "log2(K*M) instead of log2(M)). Forking from "
                "PerPairDecoderEnsembleSelector preserves substrate-optimal "
                "engineering per the operating mode."
            ),
        )
