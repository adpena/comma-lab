"""tac.substrates.self_compress_nn — δ Self-Compress NN (L0 SKETCH).

The Fields-medal grand council 2026-05-12 DEFERRED-pending-criterion scaffold
per the "Multiple contenders → multiple paths" non-negotiable. The α substrate
``sane_hnerv`` shipped PRIMARY (10/10); β/γ/δ are parallel scaffolds with
explicit reactivation criteria (NO KILL per CLAUDE.md "KILL is LAST RESORT").

Council design memo:
    .omx/research/grand_council_fields_medal_substrate_design_20260512.md

Council δ design position (from §4.1 + §4.2 + §8):

    δ-candidate: "Self-Compress NN" (PARADIGM-δεζ Track #307; MDL-optimized
    weight clustering during training). The model self-compresses during
    training via MDL-driven weight clustering (van den Oord persistent
    codebook EMA pattern). Inference-time decompression reconstructs full
    weights from cluster indices + codebook.

Per the council §4.2 13-lessons compliance table, δ has L2 NEEDS-WORK
(archive grammar requires a per-layer codebook+indices section) and L4 FAIL
(inflate decode adds ~50 LOC over α's 80). Both are explicit waivers below.

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (Lagrangian wires SegNet/PoseNet on decompressed weights) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training;
|                                  | the NEEDS-WORK in council §4.2 reflected this scaffold not yet existing) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar; SCV1 magic) |
| L4 inflate <= 100 LOC, <= 2 deps | WAIVED <= 200 LOC (council §4.2 FAIL note:
|                                  | per-layer cluster_index decode + codebook lookup adds ~50 LOC over α) |
| L5 full RGB renderer | PASS (NOT a mask codec; full RGB renderer) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
|                              + lambda_mdl*codebook_rate) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~650 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; per-layer codebook lookup is in-file) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke planned + tested) |
| L12 single-LOC review discipline | PASS (architecture ≤ 300 LOC; inflate ≤ 200 LOC) |
| L13 KILL last resort | PASS (DEFERRED-pending-research per CLAUDE.md) |

Catalog #124 archive-grammar 8 fields (declared at design-time):
    archive_grammar:           monolithic_single_file_0.bin_with_codebook_indices
                               (MAGIC|VER|NLAYERS|NPAIRS|LDIM|KCB|DV|
                                CODEBOOK|LAYER_META|INDICES|LATENTS|META —
                                35-byte header + 5 payload sections)
    parser_section_manifest:   parse_archive() returns SelfCompressNnArchive
                               with (codebook, layer_cluster_indices,
                                     layer_meta, latents, meta)
                               where codebook is (K, D_cb) shared across layers,
                               layer_cluster_indices is list of int16 tensors
                               (one per quantized weight tensor), and
                               layer_meta lists target shapes + tensor names.
    inflate_runtime_loc_budget: <= 200 LOC (waiver — codebook decode adds ~50 LOC);
                               target ~170 LOC
    runtime_dep_closure:       torch, brotli (codebook lookup is plain tensor
                               indexing; NO extra deps beyond α's set)
    export_format:             utf-8 JSON layer_meta + brotli-compressed
                               codebook (fp16) + raw int16 cluster_indices
                               + raw int16 latents + utf-8 JSON meta
    score_aware_loss:          L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ))
                                 + λ_mdl·MDL_codebook_rate(θ)
                               where d_seg + d_pose come from
                               tac.differentiable_eval_roundtrip on the
                               decompressed (codebook-quantized) weight
                               substrate.
    bolt_on_loc_budget:        substrate ~280 + archive ~200 + inflate ~170
                               + loss ~110 = ~760 LOC
                               (lane_class=substrate_engineering per L7)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof +
                               executable byte-mutation smoke in
                               tests/test_self_compress_nn_roundtrip.py
                               (mutates codebook AND cluster_indices)

Reactivation criteria (per CLAUDE.md "KILL is LAST RESORT"; council §8):
    Reactivate IFF SC++ Stage 1 lands an empirical anchor with
    >= -5% bytes saving on a score-aware-trained substrate. Until that
    condition is met this substrate stays L0 SKETCH with research_only=true.

This module IS an L0 SKETCH landing. The training ``_full_main`` entry-point
is NOT wired here; it ships in a separate post-SC++ Stage 1-anchor follow-up
subagent.
"""

from .archive import (
    SelfCompressNnArchive,
    pack_archive,
    parse_archive,
)
from .architecture import (
    SelfCompressNnConfig,
    SelfCompressNnSubstrate,
)
from .score_aware_loss import (
    SelfCompressNnScoreAwareLoss,
    SelfCompressNnScoreAwareLossWeights,
)

__all__ = [
    "SelfCompressNnArchive",
    "SelfCompressNnSubstrate",
    "SelfCompressNnConfig",
    "SelfCompressNnScoreAwareLoss",
    "SelfCompressNnScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
