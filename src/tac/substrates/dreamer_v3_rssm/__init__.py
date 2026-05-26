# SPDX-License-Identifier: MIT
"""tac.substrates.dreamer_v3_rssm — DreamerV3 RSSM categorical posterior MLX-local L0 SCAFFOLD.

Path 3 substrate-class-shift candidate adjudicated by the 2026-05-19 T3 grand
council per-substrate symposium
(``.omx/research/council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519.md``)
with verdict PROCEED_WITH_REVISIONS + 6 binding op-routables. This L0 SCAFFOLD
is the canonical first landing of op-routable #2 (Path B2 design memo +
substrate scaffold) at $0 cost (MLX-local; no paid CUDA).

Canonical equation: ``categorical_posterior_capacity_vs_continuous_gaussian_v1``
(registered in ``.omx/state/canonical_equations_registry.jsonl`` per Catalog
#344; derivation memo at
``.omx/research/dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z.md``).

Core architectural claim (substrate-CLASS shift from continuous-Gaussian IB)
---------------------------------------------------------------------------

Replace C6 IBPS v1's continuous-Gaussian 24-dim latent (effective ~50 bits per
sample) with a categorical posterior at ``G=24`` groups × ``K=256`` categories
(``H(T) = G * log2(K) = 192 bits/sample``; ~4× capacity headroom). The
categorical alphabet with uniform max-entropy prior cannot collapse to a single
mode (the structural failure mode of C6 IBPS v1 SegNet-collapse @ 105.15
contest-CUDA per `c6_e4_mdl_ibps` substrate landing).

Per Hafner 2024 DreamerV3 + vdOord VQ-VAE 2017 + sister C6 IBPS v2 PATH B2
symposium binding from PR95Author + Hassabis + Schmidhuber + Shannon canonical
positions: discrete-posterior at small-neural-architecture scale (~50K params)
IS the canonical contest-winning pattern bridge.

MLX-local-iteration enablement (today's anchor 2026-05-26)
----------------------------------------------------------

Per ``.omx/research/mlx_candidate_contest_equivalence_gate_landed_20260526.md``
+ corrected ``.omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md``:
the corrected empirical anchor ``|S_MLX - S_PyTorch| = 0.000011`` is **72×
smaller** than PR110 frontier delta 0.000789, so MLX-local iteration is
contest-grade at every score-granularity (including frontier-tightening).
This substrate's L0 scaffold lives in MLX to enable $0 iteration BEFORE any
paid CUDA dispatch is authorized.

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` (RSSMC1 magic;
  substrate-engineering scope per HNeRV parity L7 waiver)
- ``parser_section_manifest``: RSSMC1_HEADER + DECODER_BLOB + CATEGORICAL_LOGITS_BLOB
  + CATEGORY_INDICES_BLOB + META_BLOB
- ``inflate_runtime_loc_budget``: ≤200 LOC substrate-engineering waiver
  (categorical dequant + decoder forward + bilinear → contest HW)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 ≤2 deps;
  MLX kept OUT of inflate runtime; inflate.py is PyTorch-canonical per
  contest reference scorer)
- ``export_format``: RSSMC1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: pending Path B2 trainer landing; canonical helper at
  ``tac.substrates._shared.score_aware_common.score_pair_components`` per
  Catalog #164 (this L0 scaffold uses MSE proxy on RGB output for MLX-local
  convergence smoke; score-aware loss routed in PyTorch port via canonical
  helper)
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity
  L7 waiver; substrate replaces continuous Gaussian with categorical posterior
  + decoder + Gumbel-Softmax reparametrization composition exceeds ≤350 LOC
  bolt-on cap)
- ``no_op_detector_planned``: byte-mutation gate per Catalog #139 + #105
  + #272 distinguishing-feature contract; per-pair category index mutation
  IS distinguishing-feature (canonical disambiguator vs C6's per-pair float
  latent mutation)

target_modes: ``contest_one_video_replay``, ``contest_generalized``, ``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY";
  L0 scaffold pending Path B2 PyTorch port + Modal smoke per symposium
  op-routable #3)
canary_status: ``independent_substrate`` (substrate-CLASS shift from HNeRV-family;
  parallel sister candidate to Z7-Mamba-2 + NSCS06 v8 chroma_lut + Z6 PC)

axis_tag: ``[macOS-MLX research-signal]`` per CLAUDE.md "MLX portable-local-substrate
  authority"
score_claim: false
promotable: false
ready_for_exact_eval_dispatch: false

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Two defensible interpretations of "what dominates ΔS at the categorical posterior":

1. **K-capacity hypothesis** — the categorical alphabet size K (per group) dominates;
   K=256 (8 bits/group) optimally bridges the C6 continuous-Gaussian baseline.
2. **G-groups hypothesis** — the number of independent categorical groups G
   dominates; G=24 matches the C6 baseline's effective bottleneck.

Per the canonical equation ``categorical_posterior_capacity_vs_continuous_gaussian_v1``
the joint parameterization is ``H(T) = G * log2(K)``; both axes contribute
multiplicatively. The probe IS the Blahut-Arimoto sweep per sister equation
``categorical_blahut_arimoto_rate_distortion_v1`` over (G, K) ∈
{(8, 16), (16, 32), (24, 256), (32, 32)} configurations to triangulate which
axis dominates at the contest's PoseNet+SegNet feedback. Implemented as
``tools/probe_dreamer_v3_rssm_g_k_sweep_disambiguator.py`` (planned per
symposium op-routable #1a; MLX-local-prescreen at $0 wall-clock).

Cross-substrate composability (Catalog #305 facet 5)
----------------------------------------------------

Sister substrate convergence per symposium Step 7 + DD sister symposium:
- Path B2 (this) — substrate latent surface (categorical posterior over RSSM state)
- V1 Faiss V8 — side-info channel surface (categorical posterior over SegNet
  softmax histograms)
- NSCS06 v8 hybrid_class_shift_path_C — chroma residual surface (entropy
  bottleneck over residual stream)

All 3 substrates use discrete-posterior strategy at small-neural-architecture
scale (~50K params) per Catalog #303 + Hafner 2024 + vdOord 2017 lineage.
Cross-substrate composition (DreamerV3 RSSM + V1 Faiss V8 + NSCS06 v8 hybrid_path_C)
IS the canonical 3-substrate-frontier-breaking-ensemble per DD aggregate
predicted band ``[0.187, 0.205]``.
"""

from __future__ import annotations

# Re-export canonical public API surfaces so callers route through
# ``from tac.substrates.dreamer_v3_rssm import ...`` instead of internal modules.

from tac.substrates.dreamer_v3_rssm.archive import (
    RSSMC1_HEADER_FMT,
    RSSMC1_HEADER_SIZE,
    RSSMC1_MAGIC,
    RSSMC1_SCHEMA_VERSION,
    DreamerV3RSSMArchive,
    pack_archive,
    parse_archive,
    parse_rssmc1_archive_bytes,
)
from tac.substrates.dreamer_v3_rssm.module import (
    DEFAULT_G,
    DEFAULT_K,
    EVAL_HW,
    NUM_PAIRS,
    DreamerV3RSSMConfig,
    DreamerV3RSSMSubstrateMLX,
    gumbel_softmax_sample,
    rssmc_decoder_param_count,
)

# Catalog #124 8-field representation-lane declaration (canonical tokens for
# the AST walker per the gate's regex set). DO NOT remove without operator
# review per Catalog #229 premise verification.
ARCHIVE_GRAMMAR_FIELDS: dict[str, str] = {
    "archive_grammar": "monolithic_single_file_0bin_rssmc1",
    "parser_section_manifest": (
        "rssmc1_header + decoder_blob + categorical_logits_blob "
        "+ category_indices_blob + meta_blob"
    ),
    "inflate_runtime_loc_budget": (
        "le_200_loc_substrate_engineering_waiver_per_hnerv_parity_l7"
    ),
    "runtime_dep_closure": "torch_brotli_only_per_hnerv_parity_l4",
    "export_format": "rssmc1_monolithic_single_zip_member_0bin",
    "score_aware_loss": (
        "pending_path_b2_trainer_canonical_helper_score_pair_components_catalog_164"
    ),
    "bolt_on_loc_budget": (
        "substrate_engineering_lane_class_per_hnerv_parity_l7_waiver"
    ),
    "no_op_detector_planned": (
        "byte_mutation_gate_catalog_139_plus_105_plus_272_distinguishing_feature_"
        "per_pair_category_index"
    ),
}

# Canonical equation references per Catalog #344 (registered in
# .omx/state/canonical_equations_registry.jsonl).
CANONICAL_EQUATION_IDS: tuple[str, ...] = (
    "categorical_posterior_capacity_vs_continuous_gaussian_v1",
    "categorical_blahut_arimoto_rate_distortion_v1",
)


__all__ = [
    "ARCHIVE_GRAMMAR_FIELDS",
    "CANONICAL_EQUATION_IDS",
    "DEFAULT_G",
    "DEFAULT_K",
    "DreamerV3RSSMArchive",
    "DreamerV3RSSMConfig",
    "DreamerV3RSSMSubstrateMLX",
    "EVAL_HW",
    "NUM_PAIRS",
    "RSSMC1_HEADER_FMT",
    "RSSMC1_HEADER_SIZE",
    "RSSMC1_MAGIC",
    "RSSMC1_SCHEMA_VERSION",
    "gumbel_softmax_sample",
    "pack_archive",
    "parse_archive",
    "parse_rssmc1_archive_bytes",
    "rssmc_decoder_param_count",
]
