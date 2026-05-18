# SPDX-License-Identifier: MIT
"""DashcamDomainPrior — domain-specific priors for dashcam video (road
texture, lane line statistics, sky/horizon split) distilled at compress
time from public dashcam datasets and shipped as inflate.py constants.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§J row 6 (per-pair-class architecture — different decoder for highway /
urban / parking) + Atick-Redlich cooperative-receiver framing applied to
the dashcam domain specifically.

The Wyner-Ziv 1976 theorem applied to dashcam-specific priors:
  - X = per-pair RGB frames of the comma.ai contest video
  - Y = domain-specific dashcam priors (road-texture statistics, lane-line
    edge densities, sky/horizon vertical split distribution, taillight
    color centroids, etc.)
  - The decoder reconstructs the same priors from the published source
    citation; the priors ARE the side information.

Acceptable public sources (publicly-reproducible per contest rules):
  - Comma2k19 (MIT-licensed; routed via Catalog #213 canonical cache)
  - BDD100K (BDD-Berkeley-100K driving dataset; CC-BY-NC-SA-4.0)
  - Cityscapes (free for academic use; published with license attribution)
  - KITTI (CC-BY-NC-SA-3.0; published with license attribution)
  - nuScenes (CC-BY-NC-SA-4.0 with research-only restrictions)

Per CLAUDE.md "Public Disclosure Hygiene": every prior MUST cite the
specific public source. The ``source_dataset`` + ``license_tag`` fields
are the reproducibility contract that distinguishes DashcamDomainPrior
from NonReproducibleSideInfoViolation.

This baker is the canonical case of:
  - stage_phase="both"
  - side_info_source="dashcam_domain"
  - side_info_reproducible=True (every prior carries source dataset +
    license)
  - scorer_free=True
  - archive_bytes_added=0
  - inflate_runtime_bytes_added=prior_size (typical: 64 B - 4 KB)

The unique-per-method engineering surface is WHICH dashcam-specific
PRIOR (road-texture / lane-line / sky-horizon / taillight / etc.) and
HOW it's distilled (K-means / histogram / PCA / etc.). Each substrate
chooses.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.side_information.contract import (
    SideInfoBakerContract,
)

__all__ = [
    "DashcamDomainPrior",
    "DashcamDomainPriorSpec",
    "LEGAL_DASHCAM_PRIOR_KIND",
    "LEGAL_SOURCE_DATASET",
]

LEGAL_DASHCAM_PRIOR_KIND: frozenset[str] = frozenset(
    {
        # Road-texture luminance histogram (16-32 bin histogram).
        "road_texture_histogram",
        # Lane-line edge density per pixel column (vertical bands).
        "lane_line_edge_density",
        # Sky/horizon vertical-split prior (where the horizon line lies).
        "sky_horizon_split_prior",
        # Taillight color centroids (red/orange spectrum K-means).
        "taillight_color_centroids",
        # Per-class road-marking template patches.
        "road_marking_templates",
        # Per-time-of-day brightness scale factors (day/night/dusk).
        "time_of_day_brightness_factors",
        # Other operator-attested dashcam-domain prior.
        "custom",
    }
)

LEGAL_SOURCE_DATASET: frozenset[str] = frozenset(
    {
        "comma2k19",      # MIT — routed via Catalog #213 canonical cache
        "bdd100k",        # CC-BY-NC-SA-4.0
        "cityscapes",     # academic-only with attribution
        "kitti",          # CC-BY-NC-SA-3.0
        "nuscenes",       # CC-BY-NC-SA-4.0
        "argoverse",      # CC-BY-NC-SA-4.0
        "waymo_open",     # custom academic license
        "multiple",       # weighted combination of public sources
    }
)


@dataclass(frozen=True)
class DashcamDomainPriorSpec:
    """Specification for a dashcam-domain-prior baker.

    Frozen so spec composition is structurally immutable. The prior kind
    + source dataset + size are pinned at decoration time for byte-stable
    reproducibility per Catalog #158.

    The ``source_dataset`` MUST be one of LEGAL_SOURCE_DATASET; this is
    the reproducibility contract that distinguishes DashcamDomainPrior
    from NonReproducibleSideInfoViolation.
    """

    baker_id: str
    prior_kind: str  # one of LEGAL_DASHCAM_PRIOR_KIND
    source_dataset: str  # one of LEGAL_SOURCE_DATASET
    inflate_runtime_bytes_added: int
    license_tag: str = "CC-BY-NC-SA-4.0"  # default for most dashcam datasets
    wyner_ziv_correlation_estimate: float | None = None
    seed: int = 42
    correction_resolution: str = "per_class"
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.prior_kind not in LEGAL_DASHCAM_PRIOR_KIND:
            raise ValueError(
                f"prior_kind={self.prior_kind!r} not in "
                f"{sorted(LEGAL_DASHCAM_PRIOR_KIND)}"
            )
        if self.source_dataset not in LEGAL_SOURCE_DATASET:
            raise ValueError(
                f"source_dataset={self.source_dataset!r} not in "
                f"{sorted(LEGAL_SOURCE_DATASET)}"
            )
        if self.inflate_runtime_bytes_added < 1:
            raise ValueError(
                f"inflate_runtime_bytes_added="
                f"{self.inflate_runtime_bytes_added} must be >= 1"
            )
        if (
            not isinstance(self.license_tag, str)
            or not self.license_tag.strip()
        ):
            raise ValueError(
                f"license_tag={self.license_tag!r} must be a non-empty "
                f"string"
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


class DashcamDomainPrior:
    """Builder for a dashcam-domain-prior baker contract.

    The canonical compress-time + inflate-time cycle:
      1. COMPRESS:
         a. Resolve the source dataset (Comma2k19 via Catalog #213 cache;
            others via operator-provisioned read-only mount with explicit
            license attribution).
         b. Apply ``prior_kind`` distillation (histogram / K-means / PCA
            / template matching) on the dataset frames.
         c. Bake the prior as a numpy literal constant in inflate.py.
      2. INFLATE: inflate.py uses the constant for per-region / per-class
         / per-time-of-day quantization or reconstruction priors.

    When source_dataset="comma2k19" the baker SHOULD route through
    ``Comma2k19LocalCache.fetch_chunk(...)`` per Catalog #213. The
    contract's ``requires_canonical_comma2k19_cache`` field is set
    accordingly at build time.

    Usage::

        from tac.side_information import (
            DashcamDomainPrior, DashcamDomainPriorSpec, side_info_baker,
        )

        spec = DashcamDomainPriorSpec(
            baker_id="comma2k19_sky_horizon_split_prior",
            prior_kind="sky_horizon_split_prior",
            source_dataset="comma2k19",
            inflate_runtime_bytes_added=384,  # 96 vertical bands * 4 bytes
            license_tag="MIT",  # Comma2k19 is MIT
            wyner_ziv_correlation_estimate=0.28,
            seed=42,
            correction_resolution="per_class",
            description=(
                "Per-row sky/horizon split prior distilled from Comma2k19; "
                "inflate.py uses the prior to allocate per-row detail "
                "(sky region = low-detail; road region = high-detail)."
            ),
            lane_id="lane_dashcam_sky_horizon_prior_20260601",
        )
        contract = DashcamDomainPrior(spec=spec).build_contract()

        @side_info_baker(contract)
        def comma2k19_sky_horizon_split_prior(state, *, policy, seed=42):
            # Substrate-specific dashcam-prior distillation:
            #   1. cache = Comma2k19LocalCache()  # canonical
            #   2. frames = decode chunks via canonical Comma2k19FrameIterator
            #   3. compute per-row luminance histogram
            #   4. find horizon row (where luminance transitions sky→road)
            #   5. return per-row split prior as bytes
            ...
            return {
                "side_info_dashcam_prior_v1": prior_bytes,
                "inflate_runtime_bytes_added": len(prior_bytes),
                "source_dataset": "comma2k19",
                "license_tag": "MIT",
            }
    """

    def __init__(self, *, spec: DashcamDomainPriorSpec) -> None:
        if not isinstance(spec, DashcamDomainPriorSpec):
            raise TypeError(
                f"spec must be DashcamDomainPriorSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SideInfoBakerContract:
        """Build the SideInfoBakerContract for this dashcam-prior baker.

        Emits the canonical pattern:
          - stage_phase="both"
          - correction_kind="shared_prior_bake"
          - side_info_source="dashcam_domain"
          - side_info_reproducible=True (every prior cites source_dataset)
          - requires_canonical_comma2k19_cache=True when source_dataset
            is comma2k19 (forces import-time verification per Catalog
            #213)
          - scorer_free=True (priors derived from public datasets, NOT
            from the contest scorer)
          - archive_bytes_added=0
          - inflate_runtime_bytes_added=spec.inflate_runtime_bytes_added
        """
        requires_comma2k19_cache = self.spec.source_dataset == "comma2k19"
        return SideInfoBakerContract(
            id=self.spec.baker_id,
            parent_baker_id=None,
            stage_phase="both",
            description=(
                self.spec.description
                or (
                    f"DashcamDomainPrior; kind={self.spec.prior_kind!r}; "
                    f"source={self.spec.source_dataset!r}; "
                    f"inflate_runtime_bytes_added="
                    f"{self.spec.inflate_runtime_bytes_added}; "
                    f"license_tag={self.spec.license_tag!r}; "
                    f"seed={self.spec.seed}."
                )
            ),
            consumes=frozenset({"dashcam_dataset_frames"}),
            emits=frozenset(
                {
                    "side_info_dashcam_prior_v1",
                    "source_dataset",
                    "license_tag",
                }
            ),
            correction_kind="shared_prior_bake",
            correction_resolution=self.spec.correction_resolution,
            side_info_source="dashcam_domain",
            side_info_reproducible=True,
            requires_canonical_comma2k19_cache=requires_comma2k19_cache,
            wyner_ziv_correlation_estimate=(
                self.spec.wyner_ziv_correlation_estimate
            ),
            deterministic=True,
            scorer_free=True,
            archive_bytes_added=0,
            inflate_runtime_bytes_added=(
                self.spec.inflate_runtime_bytes_added
            ),
            seed=self.spec.seed,
            merge_policy="last_writer_wins",
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_pareto_constraint="wyner_ziv_rate_distortion_v1",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind=(
                "side_information_baker_outcomes_v1"
            ),
            hook_probe_disambiguator=(
                "tools/probe_dashcam_prior_kind_disambiguator.py"
            ),
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": (
                    "Dashcam-domain priors are unsupervised statistics; "
                    "master_gradient does not directly inform them (the "
                    "substrate-specific baker MAY weight inputs by "
                    "master_gradient but the default does not)."
                ),
                "hook_bit_allocator_class": (
                    "Prior is a fixed-size constant; bit allocation is "
                    "undefined (the prior informs per-region detail "
                    "allocation downstream but does not allocate bits "
                    "itself)."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the source-dataset routing (Catalog "
                "#213 for Comma2k19, operator-provisioned for others), "
                "the bake-then-lookup infrastructure, and the license "
                "attribution propagation. FORK_BECAUSE_PRINCIPLED_MISMATCH "
                "for the per-prior-kind distillation (road-texture vs "
                "lane-line vs sky-horizon vs taillight each need domain-"
                "specific feature extraction tuned to the prior's "
                "intended consumer at inflate)."
            ),
        )
