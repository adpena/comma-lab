# SPDX-License-Identifier: MIT
"""tac.substrates.hi_nerv — Hierarchical NeRV (substrate L0 SKETCH).

Per-frame implicit renderer with a 3-scale latent pyramid; coarse-to-fine
decoding combines latents from multiple resolutions before the final RGB
heads. The substrate's distinctive prior is multi-resolution
representation — coarse latents capture low-frequency / global structure,
fine latents add high-frequency detail.

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Council design memo:
    .omx/research/grand_council_fields_medal_substrate_design_20260512.md

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 33-byte header) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS for base codecs (torch + brotli); optional latent arithmetic codec also requires constriction |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~580 total) |
| L8 eval-roundtrip + diff yuv6 | PLANNED (wired at L1 SCAFFOLD) |
| L9 runtime closure | PASS for base codecs (torch + brotli); optional latent arithmetic codec declares constriction |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (HIV1)
    parser_section_manifest:   parse_archive() -> 7 sections
                               (HEADER + 3 latent pyramid + 3 decoder + meta)
                               * 3 latent_pyramid sections (scales 0,1,2)
                               * 4 decoder sections (coarse_dec, mid_dec, fine_dec, head)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:       torch, brotli; optional constriction for hi-ac latents
    export_format:             brotli-compressed pyramid decoder state_dict
                               + int16 per-scale latents + utf8-json meta
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~580 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke
"""

from .architecture import (
    HinervConfig,
    HinervSubstrate,
    expected_decoder_state_shapes,
    validate_decoder_state_dict,
)
from .archive import (
    HINERV_ARCHIVE_SECTION_TELEMETRY_SCHEMA,
    HinervArchive,
    HinervArchiveSections,
    build_archive_section_telemetry,
    pack_archive,
    parse_archive,
    repack_archive_decoder_codec,
    split_archive_sections,
)
from .archive_candidate import (
    export_hi_nerv_mlx_archive,
    export_hi_nerv_mlx_archive_bound_candidate_package,
    pack_archive_from_exported_state_dict,
)
from .bitstream import (
    HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE,
    HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA,
    HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF,
    apply_decoder_pruning,
    apply_decoder_quant_noise,
    measure_hi_nerv_decoder_bitstream_roundtrip,
    prepare_hi_nerv_decoder_bitstream_state,
    select_hi_nerv_bitstream_codec_by_scorer_waterfill,
)
from .official_grid import (
    HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF,
    HINERV_OFFICIAL_GRID_TRILINEAR3D_SOURCE_CONTRACT,
    OfficialGridTrilinear3D,
    OfficialGridTrilinear3DError,
    official_grid_trilinear3d_forward,
)
from .official_patch import (
    HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF,
    HINERV_OFFICIAL_PATCH_INDEX_SOURCE_CONTRACT,
    OfficialPatchIndexError,
    OfficialPixelIndex3D,
    official_compute_pixel_idx_3d,
    official_flat_patch_index_to_thw,
    official_patch_index_contract,
    official_patch_to_video,
    official_video_to_patch,
    official_vidx_to_pidx,
)
from .score_aware_loss import HinervScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "HINERV_ARCHIVE_SECTION_TELEMETRY_SCHEMA",
    "HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF",
    "HINERV_OFFICIAL_GRID_TRILINEAR3D_SOURCE_CONTRACT",
    "HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF",
    "HINERV_OFFICIAL_PATCH_INDEX_SOURCE_CONTRACT",
    "HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE",
    "HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA",
    "HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF",
    "HinervArchive",
    "HinervArchiveSections",
    "HinervConfig",
    "HinervScoreAwareLoss",
    "HinervSubstrate",
    "OfficialGridTrilinear3D",
    "OfficialGridTrilinear3DError",
    "OfficialPatchIndexError",
    "OfficialPixelIndex3D",
    "ScoreAwareLossWeights",
    "apply_decoder_pruning",
    "apply_decoder_quant_noise",
    "build_archive_section_telemetry",
    "expected_decoder_state_shapes",
    "export_hi_nerv_mlx_archive",
    "export_hi_nerv_mlx_archive_bound_candidate_package",
    "measure_hi_nerv_decoder_bitstream_roundtrip",
    "official_compute_pixel_idx_3d",
    "official_flat_patch_index_to_thw",
    "official_grid_trilinear3d_forward",
    "official_patch_index_contract",
    "official_patch_to_video",
    "official_video_to_patch",
    "official_vidx_to_pidx",
    "pack_archive",
    "pack_archive_from_exported_state_dict",
    "parse_archive",
    "prepare_hi_nerv_decoder_bitstream_state",
    "repack_archive_decoder_codec",
    "select_hi_nerv_bitstream_codec_by_scorer_waterfill",
    "split_archive_sections",
    "validate_decoder_state_dict",
]
