# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:l0_scaffold_landed_20260526_path_3_i_faiss_ivf_pq_residual_codec_per_pair_rgb_residual_stacking_on_pr110_fec6_frontier_NEW_substrate_design_per_phase_2_path_b_REDIRECT_meta_layer_register_substrate_decorator_pending_phase_2_council_symposium_per_catalog_325_and_substrate_contract_canonical_helper_adoption_when_faisspq1_archive_grammar_per_pair_pq_codebook_and_codeword_stream_field_is_added_to_contract_schema
"""faiss_ivf_pq_residual — Faiss IVF-PQ residual codec for PR110 fec6 stacking (L0 SCAFFOLD).

Path 3 candidate #I per operator directive 2026-05-26 verbatim:
*"Never simply extend unless a rigorous adversarial cargo cult pass has
been done first"*

NEW operator binding directive 2026-05-26 directive #3 verbatim:
*"we also need adversarial review against all landing recursive for math
and scientific and engineering rigor and for MLX drift minimization and
portability via numpy"*

This substrate is a NEW substrate-design FRESH posture per PHASE 2
substrate-design decision memo (Path b REDIRECT per Catalog #290). V1+V4+V8
prior Faiss work (`src/tac/substrates/v8_learned_compression_faiss/` +
`.omx/research/v1_faiss_v4_probe_plus_v8_design_landed_20260519.md` +
sister memos) is research INPUT only — that work targets the ATW V2-1
side-info channel surface (categorical SegNet softmax). This substrate
targets the per-pair RGB residual codec surface (continuous RGB error)
stacking on PR110 fec6 frontier (canonical sha 6bae0201).

Paradigm anchor: Jégou-Douze-Schmid 2011 *Product quantization for nearest
neighbor search* applied to per-pair RGB residual quantization. PQ
decomposes the per-tile residual vector into M sub-quantizers each with
ksub codewords; per-tile encoding = M log2(ksub) bits; codebook is shared
across all 600 pairs (NOT per-pair). The codec primitive is vector
quantization — paradigm-distinct from sister candidates:
- A=DreamerV3 RSSM: categorical latent dynamics (G×K group-categorical)
- E=BoostNeRV PR110 residual: iterative boosting (gradient-residual learner)
- G=NIRVANA cascading NeRV: hierarchical wavelet-pyramid cascade
- F=Z8 hierarchical predictive coding: canonical quadruple
- K=COIN++ INR: meta-learned MLP-per-coordinate

Architecture (L0 SCAFFOLD):

    PR110 fec6 frontier reconstruction (per-pair frame, 384×512 RGB float)
       |
       v
    PER-PAIR RESIDUAL = frame_gt - frame_pr110_reconstructed (NHWC float32)
       |
       v
    SPATIAL TILING (per-pair → tiles_per_pair tiles of TILE_H × TILE_W × 3)
       |
       v
    PQ ENCODING (per-tile vector → M codewords via codebook nearest-neighbor)
       |       |
       |       +--> codebook (shared; ~3-5KB brotli-compressed)
       v
    PER-PAIR CODEWORD STREAM (~30 byte/pair raw → brotli-compressed)
       |
       v
    ARCHIVE: FAISSPQ1 header + codebook_blob + codeword_blob + meta_blob

Inflate-time path:
    PQ decode per pair → tile reassemble → per-pair RGB residual
       |
       v
    PR110 fec6 reconstruction + per-pair residual = corrected RGB
       |
       v
    Bicubic upsample to camera 874×1164 + uint8 cast

Archive grammar (FAISSPQ1):
    29-byte header: magic b"FQP1\\x00" (5) + version u8 (1) + M u8 (1) +
        KSUB u16 (2) + TILE_H u16 (2) + TILE_W u16 (2) + TILES_PER_PAIR u16 (2) +
        NUM_PAIRS u16 (2) + CODEBOOK_BLOB_LEN u32 (4) +
        CODEWORD_BLOB_LEN u32 (4) + META_BLOB_LEN u32 (4)
    CODEBOOK_BLOB_LEN bytes: brotli(q=9) compressed PQ codebook
        (M × ksub × (tile_dim/M) float32)
    CODEWORD_BLOB_LEN bytes: brotli(q=9) compressed per-pair codeword stream
        (NUM_PAIRS × TILES_PER_PAIR × M codeword indices)
    META_BLOB_LEN bytes: sorted-keys JSON utf-8 (residual_scale, num_pairs,
        eval_hw, tile_size, ...)

CLAUDE.md compliance:
- No silent device defaults (MLX explicit; PyTorch export path uses canonical
  ``tac.substrates._shared.inflate_runtime.select_inflate_device`` per Catalog #205)
- No scorer load at inflate time (only PQ codebook gather + tile reassemble +
  bilinear upsample + residual addition + uint8 cast)
- No /tmp paths in persisted artifacts (Catalog #113 forbidden-path)
- Every file reviewable in 30 seconds per HNeRV parity L12
- ``_full_main`` raises NotImplementedError per Catalog #240 L0 SCAFFOLD posture
- numpy reference implementation per axis 3 portability (operator directive #3)
- MLX drift minimization per axis 2 (operator directive #3): canonical
  helpers `bilinear_resize2x_align_corners_false_nhwc` per sister A=DreamerV3
  forensic; codebook gather is float-deterministic; PQ encoding is integer-deterministic
"""

from tac.substrates.faiss_ivf_pq_residual.archive import (  # noqa: F401
    FAISSPQ1_HEADER_FMT,
    FAISSPQ1_HEADER_SIZE,
    FAISSPQ1_MAGIC,
    FAISSPQ1_SCHEMA_VERSION,
    FaissIVFPQResidualArchive,
    FaissIVFPQResidualArchiveError,
    build_archive_bytes,
    parse_archive,
)
from tac.substrates.faiss_ivf_pq_residual.mlx_renderer import (
    EVAL_HW,
    FaissIVFPQResidualConfig,
    _ensure_mlx_available,
    _full_main,
    estimate_archive_bytes,
    estimate_per_pair_codeword_bytes_raw,
)

__all__ = [
    "EVAL_HW",
    "FAISSPQ1_HEADER_FMT",
    "FAISSPQ1_HEADER_SIZE",
    "FAISSPQ1_MAGIC",
    "FAISSPQ1_SCHEMA_VERSION",
    "FaissIVFPQResidualArchive",
    "FaissIVFPQResidualArchiveError",
    "FaissIVFPQResidualConfig",
    "_ensure_mlx_available",
    "_full_main",
    "build_archive_bytes",
    "estimate_archive_bytes",
    "estimate_per_pair_codeword_bytes_raw",
    "parse_archive",
]
