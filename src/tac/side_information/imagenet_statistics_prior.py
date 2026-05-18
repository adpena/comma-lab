# SPDX-License-Identifier: MIT
"""ImageNetStatisticsPrior — use publicly-reproducible ImageNet / Kinetics
statistics as a shared prior at compress + inflate.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§J + CLAUDE.md "Public Disclosure Hygiene" + Wyner-Ziv 1976: this baker
uses statistics derived from public datasets (ImageNet, Kinetics,
Open-Images, etc.) as a shared prior. Public reproducibility = LEGAL
side-info per contest rules.

The Wyner-Ziv 1976 theorem applied:
  - X = the per-pair RGB frames being compressed
  - Y = published image-dataset statistics (means, covariances, class
    distributions, frequency-of-occurrence tables)
  - The decoder reconstructs the same statistics from the published
    source citation; the statistics IS the side information.

Typical statistics tables (all published, reproducible):
  - ImageNet RGB mean: [0.485, 0.456, 0.406] (3 floats = 12 bytes)
  - ImageNet RGB std: [0.229, 0.224, 0.225] (3 floats = 12 bytes)
  - Per-class frequency for K=1000 ImageNet classes (1000 floats = 4 KB)
  - Per-region brightness distribution percentiles (~100 floats = 400 B)
  - Per-channel covariance matrix (3x3 = 9 floats = 36 bytes)

This baker is the canonical case of:
  - stage_phase="both" (compress: derive normalization / statistics;
    inflate: use baked constants)
  - side_info_source="imagenet_statistics"
  - side_info_reproducible=True (every statistic must cite a published
    source URL the verifier can reproduce)
  - scorer_free=True (statistics are independent of the contest scorer)
  - archive_bytes_added=0 (statistics in inflate.py constant)
  - inflate_runtime_bytes_added=statistic_size

The unique-per-method engineering surface is WHICH STATISTIC and HOW it's
applied at inflate (per-pixel normalization, per-region quantization,
per-class prior probabilities, etc.). Each substrate chooses.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.side_information.contract import (
    SideInfoBakerContract,
)

__all__ = [
    "ImageNetStatisticsPrior",
    "ImageNetStatisticsPriorSpec",
    "LEGAL_STATISTIC_KIND",
]

LEGAL_STATISTIC_KIND: frozenset[str] = frozenset(
    {
        # Per-channel RGB means + stds (12-24 bytes).
        "rgb_mean_std",
        # Per-channel covariance matrix (3x3 = 36 bytes).
        "rgb_covariance",
        # Per-class frequency table for ImageNet K=1000 (~4 KB).
        "imagenet_class_frequency",
        # Per-class frequency table for COCO K=80 (~320 B).
        "coco_class_frequency",
        # Per-region brightness percentile bands.
        "brightness_percentile_bands",
        # Per-channel luminance distribution histograms.
        "luminance_histograms",
        # Other operator-attested public statistic.
        "custom",
    }
)


@dataclass(frozen=True)
class ImageNetStatisticsPriorSpec:
    """Specification for an ImageNet-statistics-prior baker.

    Frozen so spec composition is structurally immutable. The statistic
    kind + size + source URL are pinned at decoration time for byte-stable
    reproducibility per Catalog #158.

    The ``source_url`` field MUST be a publicly-accessible URL the
    verifier can fetch to reproduce the statistic. This is the structural
    reproducibility check that distinguishes ImageNetStatisticsPrior from
    NonReproducibleSideInfoViolation.
    """

    baker_id: str
    statistic_kind: str  # one of LEGAL_STATISTIC_KIND
    inflate_runtime_bytes_added: int  # size of the precomputed statistic
    source_url: str  # publicly-accessible reproducibility URL
    license_tag: str = "public_domain"
    wyner_ziv_correlation_estimate: float | None = None
    seed: int = 42
    correction_resolution: str = "global"
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.statistic_kind not in LEGAL_STATISTIC_KIND:
            raise ValueError(
                f"statistic_kind={self.statistic_kind!r} not in "
                f"{sorted(LEGAL_STATISTIC_KIND)}"
            )
        if self.inflate_runtime_bytes_added < 1:
            raise ValueError(
                f"inflate_runtime_bytes_added="
                f"{self.inflate_runtime_bytes_added} must be >= 1"
            )
        if (
            not isinstance(self.source_url, str)
            or not self.source_url.strip()
        ):
            raise ValueError(
                f"source_url={self.source_url!r} must be a non-empty string "
                f"(publicly-accessible URL for reproducibility per "
                f"NonReproducibleSideInfoViolation discipline)"
            )
        if not self.source_url.startswith(("http://", "https://")):
            raise ValueError(
                f"source_url={self.source_url!r} must start with http:// "
                f"or https:// (must be publicly-accessible URL)"
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


class ImageNetStatisticsPrior:
    """Builder for an ImageNet-statistics-prior baker contract.

    The canonical compress-time + inflate-time cycle:
      1. COMPRESS:
         a. Fetch / derive the named statistic from the published source
            (e.g. PyTorch torchvision ImageNet mean/std constants).
         b. Bake the statistic as a numpy literal constant in inflate.py.
      2. INFLATE: inflate.py uses the constant for per-pixel
         normalization, per-class priors, brightness band lookup, etc.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #210
    (DP1 codebook provenance metadata): the source_url + license_tag are
    REQUIRED fields so a downstream auditor can verify the statistic was
    derived from a publicly-reproducible source.

    Usage::

        from tac.side_information import (
            ImageNetStatisticsPrior, ImageNetStatisticsPriorSpec,
            side_info_baker,
        )

        spec = ImageNetStatisticsPriorSpec(
            baker_id="imagenet_rgb_mean_std",
            statistic_kind="rgb_mean_std",
            inflate_runtime_bytes_added=24,  # 6 floats * 4 bytes
            source_url="https://pytorch.org/vision/main/models.html",
            license_tag="BSD-3-Clause",
            wyner_ziv_correlation_estimate=0.05,
            seed=42,
            correction_resolution="per_pixel",
            description=(
                "ImageNet RGB mean/std for per-pixel normalization at "
                "inflate."
            ),
            lane_id="lane_imagenet_normalization_prior_20260601",
        )
        contract = ImageNetStatisticsPrior(spec=spec).build_contract()

        @side_info_baker(contract)
        def imagenet_rgb_mean_std(state, *, policy, seed=42):
            # Substrate-specific statistic baking:
            mean_bytes = b"\\x00..."  # ImageNet RGB mean as 12 bytes
            std_bytes = b"\\x00..."   # ImageNet RGB std as 12 bytes
            return {
                "side_info_imagenet_mean_std_v1": mean_bytes + std_bytes,
                "inflate_runtime_bytes_added": 24,
                "source_url": (
                    "https://pytorch.org/vision/main/models.html"
                ),
                "license_tag": "BSD-3-Clause",
            }
    """

    def __init__(self, *, spec: ImageNetStatisticsPriorSpec) -> None:
        if not isinstance(spec, ImageNetStatisticsPriorSpec):
            raise TypeError(
                f"spec must be ImageNetStatisticsPriorSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SideInfoBakerContract:
        """Build the SideInfoBakerContract for this ImageNet-statistics baker.

        Emits the canonical pattern:
          - stage_phase="both"
          - correction_kind="shared_prior_bake"
          - side_info_source="imagenet_statistics"
          - side_info_reproducible=True (every statistic carries a
            published source_url)
          - scorer_free=True (statistics independent of contest scorer)
          - archive_bytes_added=0 (statistic in inflate.py constant)
          - inflate_runtime_bytes_added=spec.inflate_runtime_bytes_added
        """
        return SideInfoBakerContract(
            id=self.spec.baker_id,
            parent_baker_id=None,
            stage_phase="both",
            description=(
                self.spec.description
                or (
                    f"ImageNetStatisticsPrior; kind="
                    f"{self.spec.statistic_kind!r}; "
                    f"inflate_runtime_bytes_added="
                    f"{self.spec.inflate_runtime_bytes_added}; "
                    f"source_url={self.spec.source_url!r}; "
                    f"license_tag={self.spec.license_tag!r}; "
                    f"seed={self.spec.seed}."
                )
            ),
            consumes=frozenset({"imagenet_published_statistics"}),
            emits=frozenset(
                {
                    "side_info_statistic_v1",
                    "source_url",
                    "license_tag",
                }
            ),
            correction_kind="shared_prior_bake",
            correction_resolution=self.spec.correction_resolution,
            side_info_source="imagenet_statistics",
            side_info_reproducible=True,
            requires_canonical_comma2k19_cache=False,
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
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": (
                    "Published statistics are fixed prior; master_gradient "
                    "does not modify them."
                ),
                "hook_bit_allocator_class": (
                    "Statistic is a fixed-size constant; bit allocation "
                    "is undefined."
                ),
                "hook_probe_disambiguator": (
                    "Published statistic has single canonical "
                    "interpretation (it IS the published value)."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the bake-published-statistic + lookup-"
                "at-inflate cycle; FORK_BECAUSE_PRINCIPLED_MISMATCH for "
                "which statistic to bake (rgb_mean_std for normalization vs "
                "class_frequency for K-class priors vs brightness bands "
                "for per-region detail each serve different reconstruction "
                "objectives). The source_url + license_tag REQUIRED fields "
                "(structurally enforced in __post_init__) are the "
                "reproducibility contract that distinguishes this baker "
                "from a non-reproducible private-dataset prior."
            ),
        )
