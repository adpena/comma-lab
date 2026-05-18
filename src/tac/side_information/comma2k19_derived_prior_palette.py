# SPDX-License-Identifier: MIT
"""Comma2k19DerivedPriorPalette — distill a small palette/dictionary from
the MIT-licensed Comma2k19 dashcam dataset at compress time and bake it
as a numpy constant into inflate.py per Catalog #146
contest_one_video_replay.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§J row 7 + CLAUDE.md "Substrate retirement discipline" lesson 2 (Comma2k19
dataset is canonically loaded via
``tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache``;
STRICT preflight Catalog #213 enforces this) + the parent-prompt
five-builder enumeration: this baker is the canonical case of:

  - side_info_source="comma2k19_distilled"
  - requires_canonical_comma2k19_cache=True
  - stage_phase="both" (compress: distill from Comma2k19 chunks; inflate:
    use baked palette constant)
  - archive_bytes_added=0 (palette is in inflate.py, not archive)
  - inflate_runtime_bytes_added=palette_size (typical: 16-256 entries
    × 1-12 bytes each ⇒ 16 bytes to 3 KB)

The Wyner-Ziv 1976 source-coding-with-side-information theorem applied:
  - X = the per-pair RGB frames being compressed
  - Y = a small palette of "typical" colors/chroma anchors/edge templates
    distilled from publicly-available dashcam data
  - The decoder can quantize / map each pixel against the palette without
    any extra archive bytes; the palette IS the side information.

Per CLAUDE.md "Public Disclosure Hygiene" + the Catalog #213 canonical
helper: the palette MUST be derivable from publicly-available data
(Comma2k19 is MIT-licensed; license_tags="MIT" attribution is propagated
through provenance per Catalog #210). The baker SHOULD NOT inadvertently
encode the contest video into the palette — the canonical helper's
``check_no_contest_video_leakage_in_distillation_callers`` (Catalog #209)
sister gate refuses callers that mix contest video into distillation
inputs.

The unique-per-method engineering surface is the DISTILLATION function
(K-means on chroma values, clustering on edge templates, statistics on
road-texture patches, etc.). Each substrate may choose a different
distillation kind.
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.side_information.contract import (
    SideInfoBakerContract,
)

__all__ = [
    "Comma2k19DerivedPriorPalette",
    "Comma2k19DerivedPriorPaletteSpec",
    "LEGAL_PALETTE_KIND",
]

LEGAL_PALETTE_KIND: frozenset[str] = frozenset(
    {
        # K=16 to K=256 chroma-anchor palette (UV-space K-means).
        "chroma_anchors",
        # K=8 to K=32 RGB-triplet palette (RGB-space K-means).
        "rgb_triplets",
        # K=16 to K=64 edge-template patches (e.g. 4x4 sobel-edge codes).
        "edge_templates",
        # K=16 to K=32 road-texture patches (16x16 luminance patches).
        "road_texture_patches",
        # K=8 to K=32 sky/horizon split codes (per-row luminance bands).
        "sky_horizon_split",
        # Other operator-attested palette kind (must reference public source).
        "custom",
    }
)


@dataclass(frozen=True)
class Comma2k19DerivedPriorPaletteSpec:
    """Specification for a Comma2k19-derived-prior-palette baker.

    Frozen so spec composition is structurally immutable. The palette kind
    + size + seed are pinned at decoration time for byte-stable
    reproducibility per Catalog #158.

    The cache directory is resolved by the canonical
    ``Comma2k19LocalCache.default_cache_dir()`` helper if None; this keeps
    the worker filesystem layout entirely in the canonical helper's hands.
    """

    baker_id: str
    palette_kind: str  # one of LEGAL_PALETTE_KIND
    num_palette_entries: int
    bytes_per_entry: int = 3  # default: RGB-triplet 3 bytes
    num_distillation_frames: int = 1200  # 60 sec × 20 Hz Comma2k19 Example_1
    wyner_ziv_correlation_estimate: float | None = None
    seed: int = 42
    correction_resolution: str = "global"
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.palette_kind not in LEGAL_PALETTE_KIND:
            raise ValueError(
                f"palette_kind={self.palette_kind!r} not in "
                f"{sorted(LEGAL_PALETTE_KIND)}"
            )
        if self.num_palette_entries < 2:
            raise ValueError(
                f"num_palette_entries={self.num_palette_entries} must be "
                f">= 2 (K=1 palette is degenerate)"
            )
        if self.bytes_per_entry < 1:
            raise ValueError(
                f"bytes_per_entry={self.bytes_per_entry} must be >= 1"
            )
        if self.num_distillation_frames < 1:
            raise ValueError(
                f"num_distillation_frames={self.num_distillation_frames} "
                f"must be >= 1"
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

    @property
    def total_palette_bytes(self) -> int:
        """Total bytes the palette will occupy in inflate.py."""
        return self.num_palette_entries * self.bytes_per_entry


class Comma2k19DerivedPriorPalette:
    """Builder for a Comma2k19-derived-prior-palette baker contract.

    The canonical compress-time + inflate-time cycle:
      1. COMPRESS:
         a. Resolve Comma2k19 chunks via the canonical
            ``Comma2k19LocalCache.fetch_chunk(...)`` helper (per Catalog
            #213); SHA-256 verifies integrity; MIT license_tags propagated.
         b. Decode N frames via the Comma2k19FrameIterator (per Catalog
            #209 sister discipline — does NOT mix contest video).
         c. Apply ``palette_kind`` clustering on the frames to derive
            K=num_palette_entries cluster centers.
         d. Bake the centers as a Python literal constant in inflate.py.
      2. INFLATE: inflate.py reads the constant + uses it as a
         quantization codebook (the decoder maps each pixel to the
         nearest palette entry).

    Usage::

        from tac.side_information import (
            Comma2k19DerivedPriorPalette,
            Comma2k19DerivedPriorPaletteSpec,
            side_info_baker,
        )

        spec = Comma2k19DerivedPriorPaletteSpec(
            baker_id="comma2k19_chroma_palette_k16",
            palette_kind="chroma_anchors",
            num_palette_entries=16,
            bytes_per_entry=2,  # UV-pair 2 bytes
            num_distillation_frames=1200,
            wyner_ziv_correlation_estimate=0.42,
            seed=42,
            correction_resolution="global",
            description=(
                "K=16 UV chroma anchors distilled from Comma2k19 Example_1 "
                "via UV-space K-means; baked into inflate.py as 32-byte "
                "constant."
            ),
            lane_id="lane_comma2k19_chroma_palette_20260601",
        )
        contract = Comma2k19DerivedPriorPalette(spec=spec).build_contract()

        @side_info_baker(contract)
        def comma2k19_chroma_palette_k16(state, *, policy, seed=42):
            # Substrate-specific palette distillation:
            #   1. cache = Comma2k19LocalCache()  # canonical helper
            #   2. chunk_path = cache.fetch_chunk("example_1")  # auto-download
            #   3. frames = list(Comma2k19FrameIterator(chunk_path,
            #          num_frames=spec.num_distillation_frames))
            #   4. uv = chroma_from_rgb(frames)
            #   5. palette = kmeans(uv, K=spec.num_palette_entries, seed=seed)
            #   6. return {"side_info_palette_v1": palette_bytes, ...}
            ...
            return {
                "side_info_palette_v1": palette_bytes,
                "inflate_runtime_bytes_added": len(palette_bytes),
                "license_tags": ["MIT"],  # propagated from canonical cache
                "source_url": (
                    "https://github.com/commaai/comma2k19"
                ),
            }

    Per CLAUDE.md Catalog #213 STRICT preflight: every Comma2k19 download
    MUST route through ``Comma2k19LocalCache.fetch_chunk(...)``. Bare URL
    fetches are refused at preflight time.
    """

    def __init__(self, *, spec: Comma2k19DerivedPriorPaletteSpec) -> None:
        if not isinstance(spec, Comma2k19DerivedPriorPaletteSpec):
            raise TypeError(
                f"spec must be Comma2k19DerivedPriorPaletteSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SideInfoBakerContract:
        """Build the SideInfoBakerContract for this palette baker.

        Emits the canonical pattern:
          - stage_phase="both"
          - correction_kind="palette_distillation"
          - side_info_source="comma2k19_distilled"
          - side_info_reproducible=True (MIT-licensed Comma2k19 +
            published distillation seed)
          - requires_canonical_comma2k19_cache=True (forces import-time
            verification of the canonical helper per Catalog #213)
          - scorer_free=True (palette distillation does NOT load the
            contest scorer)
          - archive_bytes_added=0 (palette baked into inflate.py constant,
            not contest archive)
          - inflate_runtime_bytes_added=spec.total_palette_bytes
        """
        return SideInfoBakerContract(
            id=self.spec.baker_id,
            parent_baker_id=None,
            stage_phase="both",
            description=(
                self.spec.description
                or (
                    f"Comma2k19DerivedPriorPalette; kind="
                    f"{self.spec.palette_kind!r}; K="
                    f"{self.spec.num_palette_entries}; bytes_per_entry="
                    f"{self.spec.bytes_per_entry}; total_palette_bytes="
                    f"{self.spec.total_palette_bytes}; seed={self.spec.seed}."
                )
            ),
            consumes=frozenset({"comma2k19_chunks"}),
            emits=frozenset(
                {
                    "side_info_palette_v1",
                    "license_tags",
                    "source_url",
                }
            ),
            correction_kind="palette_distillation",
            correction_resolution=self.spec.correction_resolution,
            side_info_source="comma2k19_distilled",
            side_info_reproducible=True,
            requires_canonical_comma2k19_cache=True,
            wyner_ziv_correlation_estimate=(
                self.spec.wyner_ziv_correlation_estimate
            ),
            deterministic=True,
            scorer_free=True,
            archive_bytes_added=0,
            inflate_runtime_bytes_added=self.spec.total_palette_bytes,
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
                "tools/probe_palette_kind_disambiguator.py"
            ),
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": (
                    "Palette distillation is unsupervised K-means; "
                    "master_gradient does not directly inform cluster "
                    "centers (the substrate-specific decorated baker MAY "
                    "weight inputs by master_gradient but the default "
                    "infrastructure does not)."
                ),
                "hook_bit_allocator_class": (
                    "Palette is a fixed-size constant; bit allocation is "
                    "undefined (the codebook IS its own allocation)."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the Comma2k19-cache routing (Catalog "
                "#213), the K-means distillation infrastructure, and the "
                "MIT-license attribution propagation. "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the per-palette-"
                "kind clustering function (chroma-anchors vs RGB-triplets "
                "vs edge-templates vs road-texture-patches each need "
                "domain-specific feature extraction)."
            ),
        )
