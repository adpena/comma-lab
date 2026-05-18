# SPDX-License-Identifier: MIT
"""ScorerWeightsAsSharedPrior — extract sensitivity-weighted features from
frozen PoseNet/SegNet weights and use them as a shared prior at compress
+ inflate (Atick-Redlich cooperative-receiver framing).

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§J row 5 (decoder architecture per pair — per-pair-class architecture) +
the operator's WUNDERKIND-VISIONARY task #713 (Atick-Redlich
cooperative-receiver) + CLAUDE.md "HNeRV / leaderboard-implementation
parity discipline" L8 (scorer-preprocess MUST be gradient-reachable).

The Wyner-Ziv 1976 theorem applied to the contest:
  - X = the per-pair RGB frames the encoder is compressing
  - Y = the frozen scorer weights (PoseNet + SegNet)
  - The decoder has full access to Y by contest contract (PoseNet + SegNet
    weights are baked into the contest scorer at inflate-evaluation time;
    however the strict-scorer-rule FORBIDS loading them at INFLATE).

So strictly speaking, the contest decoder's inflate.py does NOT load the
scorer (Catalog "strict scorer rule" non-negotiable). But the COMPRESS-time
encoder DOES have access to the same scorer weights — and any feature it
extracts from them at compress time can be baked as a constant table into
inflate.py (per Catalog #146 contest_one_video_replay) up to the
inflate-runtime byte budget.

This baker is the canonical case of:
  - stage_phase="both"
  - side_info_source="scorer_weights"
  - scorer_free=False (loads scorer at compress; strictly forbidden at
    inflate per CLAUDE.md "Strict scorer rule" — but a precomputed
    constant table CAN be baked into inflate.py)
  - inflate_runtime_bytes_added > 0 (the precomputed feature table)
  - archive_bytes_added = 0 (no per-pair payload; the prior is in the
    inflate.py constant)

The unique-per-method engineering surface is the FEATURE-EXTRACTION
function the substrate-specific decorated baker provides. For example:
  - SegNet stem-layer activation statistics on a held-out validation set
  - PoseNet feature-map principal components (top-K eigenvectors)
  - Per-class scorer-activation centroids (for K-means lookup at inflate)

The infrastructure (builder + spec + contract) is canonical; the
feature-extraction loss/objective is substrate-specific.

Per CLAUDE.md "Forbidden /tmp paths" + Catalog #146: the precomputed
feature table goes into a numpy constant baked at compress time into the
inflate.py source code — not into a sidecar file. This keeps the
strict-scorer-rule contract honored (inflate.py does NOT load scorer
weights; it has compile-time constants derived from them).
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.side_information.contract import (
    SideInfoBakerContract,
)

__all__ = [
    "LEGAL_FEATURE_EXTRACTION_KIND",
    "ScorerWeightsAsSharedPrior",
    "ScorerWeightsAsSharedPriorSpec",
]

LEGAL_FEATURE_EXTRACTION_KIND: frozenset[str] = frozenset(
    {
        # SegNet stem-layer mean activations on a held-out batch.
        "segnet_stem_activation_mean",
        # SegNet per-class activation centroids (K-means in feature space).
        "segnet_per_class_centroids",
        # PoseNet feature-map principal components (top-K eigenvectors).
        "posenet_feature_pca",
        # PoseNet pose-head linear weights (top-K projections).
        "posenet_pose_head_projection",
        # Generic operator-attested feature extraction (any other shape).
        "custom",
    }
)


@dataclass(frozen=True)
class ScorerWeightsAsSharedPriorSpec:
    """Specification for a scorer-weights-as-shared-prior baker.

    Frozen so spec composition is structurally immutable. The feature-
    extraction kind + table size + seed are pinned at decoration time for
    byte-stable reproducibility per Catalog #158.

    The ``inflate_runtime_bytes_added`` field controls how many bytes the
    baked feature table contributes to inflate.py. Operators should keep
    this small (≤ 1-2 KB; HNeRV parity L4 ≤ 100 LOC inflate budget bound).
    """

    baker_id: str
    feature_extraction_kind: str  # one of LEGAL_FEATURE_EXTRACTION_KIND
    inflate_runtime_bytes_added: int = 256
    # Wyner-Ziv correlation I(X; scorer_features) / H(X); operator's
    # estimate per Atick-Redlich cooperative-receiver framing.
    wyner_ziv_correlation_estimate: float | None = None
    seed: int = 42
    correction_resolution: str = "global"
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.feature_extraction_kind not in LEGAL_FEATURE_EXTRACTION_KIND:
            raise ValueError(
                f"feature_extraction_kind={self.feature_extraction_kind!r} "
                f"not in {sorted(LEGAL_FEATURE_EXTRACTION_KIND)}"
            )
        if self.inflate_runtime_bytes_added < 0:
            raise ValueError(
                f"inflate_runtime_bytes_added="
                f"{self.inflate_runtime_bytes_added} must be >= 0"
            )
        if self.wyner_ziv_correlation_estimate is not None:
            if not 0.0 <= self.wyner_ziv_correlation_estimate <= 1.0:
                raise ValueError(
                    f"wyner_ziv_correlation_estimate="
                    f"{self.wyner_ziv_correlation_estimate} must be in "
                    f"[0.0, 1.0]"
                )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class ScorerWeightsAsSharedPrior:
    """Builder for a scorer-weights-as-shared-prior baker contract.

    The canonical compress-time + inflate-time cycle:
      1. COMPRESS: load PoseNet + SegNet weights (allowed at compress);
         apply ``feature_extraction_kind`` to derive a small constant
         table (~256 bytes); bake table into inflate.py as a Python
         literal constant.
      2. INFLATE: inflate.py uses the constant table directly — NEVER
         loads scorer weights. This honors the strict-scorer-rule.

    Per CLAUDE.md HNeRV parity discipline lesson 4 + Catalog #146
    ``contest_one_video_replay``: precomputed constants are admissible
    even when the inflate.py code-size budget is tight, as long as they
    are a deterministic byte-stable function of the contest scorer
    weights and the public training set.

    Usage::

        from tac.side_information import (
            ScorerWeightsAsSharedPrior, ScorerWeightsAsSharedPriorSpec,
            side_info_baker,
        )

        spec = ScorerWeightsAsSharedPriorSpec(
            baker_id="segnet_per_class_centroids_k16",
            feature_extraction_kind="segnet_per_class_centroids",
            inflate_runtime_bytes_added=256,  # 16 centroids * 16 dims
            wyner_ziv_correlation_estimate=0.18,
            seed=42,
            correction_resolution="per_class",
            description=(
                "K=16 per-class SegNet centroids; inflate.py uses them as "
                "a quantization codebook for low-bit per-region detail."
            ),
            lane_id="lane_segnet_centroid_prior_20260601",
        )
        contract = ScorerWeightsAsSharedPrior(spec=spec).build_contract()

        @side_info_baker(contract)
        def segnet_per_class_centroids_k16(
            state, *, policy, seed=42, master_gradient=None,
        ):
            # Substrate-specific feature extraction:
            #   1. Load SegNet at compress (allowed)
            #   2. Forward over held-out batch
            #   3. K-means cluster the stem-layer activations
            #   4. Return centroid table as bytes
            ...
            return {
                "side_info_centroids_v1": centroid_bytes,
                "inflate_runtime_bytes_added": len(centroid_bytes),
            }

    The builder does NOT execute the feature extraction — it produces the
    CONTRACT. The decorated function IS the substrate-specific extraction.
    This separation is the canonical "infrastructure vs engineering" split
    per CLAUDE.md HNeRV parity discipline L7.
    """

    def __init__(self, *, spec: ScorerWeightsAsSharedPriorSpec) -> None:
        if not isinstance(spec, ScorerWeightsAsSharedPriorSpec):
            raise TypeError(
                f"spec must be ScorerWeightsAsSharedPriorSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SideInfoBakerContract:
        """Build the SideInfoBakerContract for this scorer-prior baker.

        Emits the canonical pattern:
          - stage_phase="both" (compress: feature extract; inflate:
            constant lookup)
          - correction_kind="shared_prior_bake"
          - side_info_source="scorer_weights"
          - side_info_reproducible=True (the prior is derivable from
            public scorer weights + contest video)
          - scorer_free=False (compress loads scorer; inflate does NOT —
            the constant table IS the inflate-side artifact)
          - archive_bytes_added=0 (the prior is in inflate.py constant,
            not the contest archive)
          - inflate_runtime_bytes_added=spec.inflate_runtime_bytes_added
        """
        return SideInfoBakerContract(
            id=self.spec.baker_id,
            parent_baker_id=None,
            stage_phase="both",
            description=(
                self.spec.description
                or (
                    f"ScorerWeightsAsSharedPrior; extraction="
                    f"{self.spec.feature_extraction_kind!r}; "
                    f"inflate_runtime_bytes_added="
                    f"{self.spec.inflate_runtime_bytes_added}; "
                    f"seed={self.spec.seed}."
                )
            ),
            consumes=frozenset({"scorer_weights", "held_out_batch"}),
            emits=frozenset({"side_info_scorer_prior_v1"}),
            correction_kind="shared_prior_bake",
            correction_resolution=self.spec.correction_resolution,
            side_info_source="scorer_weights",
            side_info_reproducible=True,
            requires_canonical_comma2k19_cache=False,
            wyner_ziv_correlation_estimate=(
                self.spec.wyner_ziv_correlation_estimate
            ),
            deterministic=True,
            scorer_free=False,  # compress-time loads scorer
            archive_bytes_added=0,
            inflate_runtime_bytes_added=(
                self.spec.inflate_runtime_bytes_added
            ),
            seed=self.spec.seed,
            merge_policy="last_writer_wins",
            hook_sensitivity_contribution="scorer_weights_shared_prior_v1",
            hook_pareto_constraint="wyner_ziv_rate_distortion_v1",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind=(
                "side_information_baker_outcomes_v1"
            ),
            hook_probe_disambiguator=(
                "tools/probe_scorer_prior_feature_extraction_disambiguator.py"
            ),
            hook_not_applicable_rationale={
                "hook_bit_allocator_class": (
                    "Shared-prior bake emits a fixed-size constant; bit "
                    "allocation is undefined (the prior IS its own "
                    "allocation)."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the bake-at-compress + lookup-at-"
                "inflate cycle infrastructure; FORK_BECAUSE_PRINCIPLED_"
                "MISMATCH for the feature-extraction function (which is "
                "substrate-specific and provided by the decorated baker). "
                "Per CLAUDE.md 'Strict scorer rule' the scorer load is "
                "ONLY at compress; inflate uses the precomputed constant."
            ),
        )
