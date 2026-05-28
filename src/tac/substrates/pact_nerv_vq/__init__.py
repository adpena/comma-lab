# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_vq — Pact-NeRV-VQ (L0 SKETCH).

Group 2 variant #7 of PACT-NERV-ULTIMATE (research memo
`.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md`).
Mid-LOC apparatus-aligned variant per the WAVE-3-NERV-LITERATURE-L0-RESCOPED
canonical pattern.

Literature anchor: van den Oord, Vinyals, Kavukcuoglu 2017 *"Neural Discrete
Representation Learning"* (VQ-VAE), arXiv:1711.00937. Canonical OSS reference
repo: `lucidrains/vector-quantize-pytorch`. Aaron van den Oord is the inner
council seat per CLAUDE.md "Grand Council (advisory)" roster.

The distinguishing primitive: vector-quantization of per-pair latents through
a learnable codebook + EMA codebook update (van den Oord 1711.00937 §3.2) +
commitment loss (§3.1). Discrete latents shrink the int16-quantized latent
bytes per-pair to log2(codebook_size) bits each, providing a rate-axis lever
the IA3 / DistilledScorer variants do not have.

Status: **L1 MLX-LOCAL** (research_only=true). Local MLX artifacts are
advisory only until paired contest CPU/CUDA evidence lands.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (PVQ)
    parser_section_manifest:   parse_archive() -> 5 sections (header +
                               decoder_blob + codebook_blob + indices_blob +
                               meta_blob)
    inflate_runtime_loc_budget: ≤ 150 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli weights; uint16 codebook indices +
                               fp16 codebook vectors
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
                               + commitment_weight * ||z_e - sg(z_q)||^2
    bolt_on_loc_budget:        ~300 LOC (substrate_engineering tag; task-cap)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation

Cargo-cult audit per Catalog #303:
- VQ codebook + EMA = HARD-EARNED (van den Oord 1711.00937 + inner council seat).
- Commitment loss weight 0.25 = HARD-EARNED (van den Oord §3.1 canonical).
- Codebook size 512 + codebook dim 24 = CARGO-CULTED at L0 (alternatives:
  RVQ residual VQ; FSQ finite-scalar Mentzer 2309.15505; Stage 1 sweep).
- Per-pair single-token quantization = CARGO-CULTED at L0 (alternative:
  per-pair sequence of tokens; Stage 1 ablation).
"""

from .architecture import (
    PactNervVqConfig,
    PactNervVqSubstrate,
    VectorQuantizerEMA,
)
from .archive import (
    PVQ_HEADER_SIZE,
    PVQ_MAGIC,
    PVQ_SCHEMA_VERSION,
    PactNervVqArchive,
    pack_archive,
    parse_archive,
)
from .archive_candidate import (
    export_pact_nerv_vq_mlx_archive,
    pack_archive_from_exported_state_dict,
    vq_meta_from_config,
)
from .score_aware_loss import PactNervVqScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "PVQ_HEADER_SIZE",
    "PVQ_MAGIC",
    "PVQ_SCHEMA_VERSION",
    "PactNervVqArchive",
    "PactNervVqConfig",
    "PactNervVqScoreAwareLoss",
    "PactNervVqSubstrate",
    "ScoreAwareLossWeights",
    "VectorQuantizerEMA",
    "export_pact_nerv_vq_mlx_archive",
    "pack_archive",
    "pack_archive_from_exported_state_dict",
    "parse_archive",
    "vq_meta_from_config",
]
