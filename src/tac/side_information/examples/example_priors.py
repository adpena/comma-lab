# SPDX-License-Identifier: MIT
"""Canonical example side-information bakers — one per builder.

Five minimal bakers that exercise the decorator + composition API:

  1. ``scorer_segnet_centroids_example``       : ScorerWeightsAsSharedPrior
  2. ``comma2k19_chroma_palette_example``      : Comma2k19DerivedPriorPalette
  3. ``imagenet_rgb_mean_std_example``         : ImageNetStatisticsPrior
  4. ``dashcam_sky_horizon_example``           : DashcamDomainPrior
  5. ``wz_residual_linear_predictor_example``  : WynerZivResidualEncoder

The example bodies are TOY (fixed bytes / no real distillation / no real
encoding) so the bakers are testable without GPU or external datasets.
Real consumers replace the body with substrate-specific logic.

Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — every claim in the
docstrings is backed by an executable body.
"""

from __future__ import annotations

from typing import Any, Mapping

from tac.side_information.comma2k19_derived_prior_palette import (
    Comma2k19DerivedPriorPalette,
    Comma2k19DerivedPriorPaletteSpec,
)
from tac.side_information.dashcam_domain_prior import (
    DashcamDomainPrior,
    DashcamDomainPriorSpec,
)
from tac.side_information.decorator import side_info_baker
from tac.side_information.imagenet_statistics_prior import (
    ImageNetStatisticsPrior,
    ImageNetStatisticsPriorSpec,
)
from tac.side_information.scorer_weights_as_shared_prior import (
    ScorerWeightsAsSharedPrior,
    ScorerWeightsAsSharedPriorSpec,
)
from tac.side_information.wyner_ziv_residual_encoder import (
    WynerZivResidualEncoder,
    WynerZivResidualEncoderSpec,
)


# Lane id shared across the example bakers for provenance.
_EXAMPLE_LANE_ID = (
    "lane_tac_side_information_namespace_decorator_api_20260517"
)
_EXAMPLE_DESIGN_MEMO = (
    ".omx/research/"
    "meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md"
)


# ---------------------------------------------------------------------------
# 1. ScorerWeightsAsSharedPrior sample
# ---------------------------------------------------------------------------

_scorer_segnet_centroids_example_contract = ScorerWeightsAsSharedPrior(
    spec=ScorerWeightsAsSharedPriorSpec(
        baker_id="scorer_segnet_centroids_example",
        feature_extraction_kind="segnet_per_class_centroids",
        inflate_runtime_bytes_added=256,  # 16 centroids * 16 dims
        wyner_ziv_correlation_estimate=0.18,
        seed=42,
        correction_resolution="per_class",
        description="Toy SegNet per-class centroid prior example.",
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@side_info_baker(_scorer_segnet_centroids_example_contract)
def scorer_segnet_centroids_example(
    state: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    seed: int = 42,
    master_gradient: Any | None = None,
) -> dict[str, Any]:
    """Toy SegNet per-class centroid prior. Real consumers load the SegNet
    + extract per-class activation centroids; this example returns a
    fixed 256-byte placeholder.
    """
    centroid_bytes = b"\x00" * 256
    return {
        "side_info_scorer_prior_v1": centroid_bytes,
        "inflate_runtime_bytes_added": len(centroid_bytes),
    }


# ---------------------------------------------------------------------------
# 2. Comma2k19DerivedPriorPalette sample
# ---------------------------------------------------------------------------

_comma2k19_chroma_palette_example_contract = Comma2k19DerivedPriorPalette(
    spec=Comma2k19DerivedPriorPaletteSpec(
        baker_id="comma2k19_chroma_palette_example",
        palette_kind="chroma_anchors",
        num_palette_entries=16,
        bytes_per_entry=2,  # UV pair
        num_distillation_frames=1200,
        wyner_ziv_correlation_estimate=0.42,
        seed=42,
        correction_resolution="global",
        description=(
            "Toy K=16 UV chroma anchor palette example "
            "(returns zero palette without real distillation)."
        ),
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@side_info_baker(_comma2k19_chroma_palette_example_contract)
def comma2k19_chroma_palette_example(
    state: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Toy K=16 UV chroma palette. Real consumers route through
    ``Comma2k19LocalCache.fetch_chunk("example_1")``, decode frames via
    Comma2k19FrameIterator, UV-space K-means cluster, and emit the
    cluster centers. This example returns 32 zero bytes.
    """
    palette_bytes = b"\x00" * 32  # 16 entries * 2 bytes
    return {
        "side_info_palette_v1": palette_bytes,
        "inflate_runtime_bytes_added": len(palette_bytes),
        "license_tags": ["MIT"],
        "source_url": "https://github.com/commaai/comma2k19",
    }


# ---------------------------------------------------------------------------
# 3. ImageNetStatisticsPrior sample
# ---------------------------------------------------------------------------

_imagenet_rgb_mean_std_example_contract = ImageNetStatisticsPrior(
    spec=ImageNetStatisticsPriorSpec(
        baker_id="imagenet_rgb_mean_std_example",
        statistic_kind="rgb_mean_std",
        inflate_runtime_bytes_added=24,  # 6 floats * 4 bytes
        source_url="https://pytorch.org/vision/main/models.html",
        license_tag="BSD-3-Clause",
        wyner_ziv_correlation_estimate=0.05,
        seed=42,
        correction_resolution="per_pixel",
        description="Toy ImageNet RGB mean/std example.",
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@side_info_baker(_imagenet_rgb_mean_std_example_contract)
def imagenet_rgb_mean_std_example(
    state: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Toy ImageNet RGB mean/std baker. Real consumers bake the
    canonical PyTorch torchvision means/stds as 24-byte float32 constant.
    This example emits zero bytes as a placeholder.
    """
    statistic_bytes = b"\x00" * 24
    return {
        "side_info_statistic_v1": statistic_bytes,
        "inflate_runtime_bytes_added": len(statistic_bytes),
        "source_url": "https://pytorch.org/vision/main/models.html",
        "license_tag": "BSD-3-Clause",
    }


# ---------------------------------------------------------------------------
# 4. DashcamDomainPrior sample (NON-Comma2k19 source to avoid
#    requires_canonical_comma2k19_cache when the helper is unavailable
#    in some test environments)
# ---------------------------------------------------------------------------

_dashcam_sky_horizon_example_contract = DashcamDomainPrior(
    spec=DashcamDomainPriorSpec(
        baker_id="dashcam_sky_horizon_example",
        prior_kind="sky_horizon_split_prior",
        source_dataset="bdd100k",
        inflate_runtime_bytes_added=384,  # 96 vertical bands * 4 bytes
        license_tag="CC-BY-NC-SA-4.0",
        wyner_ziv_correlation_estimate=0.28,
        seed=42,
        correction_resolution="per_class",
        description=(
            "Toy BDD100K sky/horizon split prior example "
            "(returns fixed placeholder bytes)."
        ),
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@side_info_baker(_dashcam_sky_horizon_example_contract)
def dashcam_sky_horizon_example(
    state: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Toy BDD100K sky/horizon split prior baker. Real consumers fetch
    BDD100K frames, compute per-row luminance histograms, find the
    horizon row, and emit per-row split prior bytes. This example
    returns 384 zero bytes.
    """
    prior_bytes = b"\x00" * 384
    return {
        "side_info_dashcam_prior_v1": prior_bytes,
        "inflate_runtime_bytes_added": len(prior_bytes),
        "source_dataset": "bdd100k",
        "license_tag": "CC-BY-NC-SA-4.0",
    }


# ---------------------------------------------------------------------------
# 5. WynerZivResidualEncoder sample (consumes the imagenet baker as the
#    shared prior so the example pipeline can be tested without external
#    data)
# ---------------------------------------------------------------------------

_wz_residual_linear_predictor_example_contract = WynerZivResidualEncoder(
    spec=WynerZivResidualEncoderSpec(
        baker_id="wz_residual_linear_predictor_example",
        shared_prior_baker_id="imagenet_rgb_mean_std_example",
        reconstruction_fn="linear_predictor",
        residual_code="arithmetic",
        archive_bytes_added=2048,
        inflate_runtime_bytes_added=512,
        wyner_ziv_correlation_estimate=0.55,
        sensitivity_weighted=False,
        seed=42,
        correction_resolution="per_pair",
        description=(
            "Toy Wyner-Ziv residual encoder example using imagenet "
            "rgb_mean_std as the shared prior + linear predictor + "
            "arithmetic residual code."
        ),
        lane_id=_EXAMPLE_LANE_ID,
    )
).build_contract()


@side_info_baker(_wz_residual_linear_predictor_example_contract)
def wz_residual_linear_predictor_example(
    state: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Toy Wyner-Ziv residual encoder. Real consumers compute
    f(Y) = linear_predictor(shared_prior), then encode X - f(Y) via
    arithmetic code. This example returns 2048 zero bytes.
    """
    residual_bytes = b"\x00" * 2048
    return {
        "archive_residual_bytes_v1": residual_bytes,
        "archive_bytes_added": len(residual_bytes),
    }
