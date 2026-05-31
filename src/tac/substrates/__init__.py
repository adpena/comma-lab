# SPDX-License-Identifier: MIT
"""tac.substrates — score-aware representation/codec substrates.

A "substrate" in pact terminology is a representation that simultaneously
binds:

1. Score-aware training loop (gradient-through-SegNet/PoseNet on contest video)
2. Monolithic single-file ``0.bin`` archive grammar with fixed offsets
3. ``inflate.py`` runtime <= 100 LOC consuming the archive bytes
4. Eval-roundtrip-aware loss with differentiable yuv6 (per CLAUDE.md
   "eval_roundtrip — non-negotiable" + PR #95/#106 yuv6 monkey-patch contract)
5. EMA shadow checkpoint (per CLAUDE.md "EMA — non-negotiable")
6. Catalog #124 STRICT archive-grammar 8 fields declared at design time

Each substrate ships as an isolated subpackage with the structure::

    src/tac/substrates/<name>/
        __init__.py
        architecture.py     # the score-aware substrate model class
        archive.py          # the monolithic 0.bin builder/parser
        inflate.py          # <= 100 LOC inflate consumer
        score_aware_loss.py # the Lagrangian
        tests/
            test_<name>_roundtrip.py  # Catalog #91 encoder/decoder roundtrip

Lane registration::

    python tools/lane_maturity.py add-lane lane_substrate_<name>_20260512 \
        --name "<title>" --phase 2

Designed by the Fields-medal grand council 2026-05-12; see::

    .omx/research/grand_council_fields_medal_substrate_design_20260512.md

Substrate implementation traditions (CANON-1.A taxonomy, 2026-05-12)
=====================================================================

There are TWO co-existing substrate implementation traditions in this
codebase. Both are CANONICAL; neither subsumes the other yet. The
classification was formalized in
``.omx/research/substrate_tradition_taxonomy_20260512.md`` and originates
in the ``.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md``
CANON-1.A finding.

TRADITION 1 — ``src/tac/substrates/<name>/`` (THIS PACKAGE)
-----------------------------------------------------------
**Status**: L0 SKETCH, ``research_only=true`` per Catalog #124 declaration.
**Discipline**: Catalog #124 STRICT 8-archive-grammar fields at design time,
the 13 HNeRV parity discipline lessons, 3-clean-pass grand-council
adversarial review, mandatory ``score_aware_loss.py`` + ``inflate.py`` <=
100 LOC + monolithic ``0.bin``.
**Composition**: each substrate is an isolated subpackage; the structure
forces every new representation/codec lane through the design-time gate.
**Migration target**: at first ``<= 0.21 [contest-CUDA]`` empirical anchor,
the substrate SUBSUMES its sibling under TRADITION 2 (the older
``<name>_as_renderer.py`` is archived to
``.omx/research/historical_substrates/`` with provenance).

TRADITION 2 — ``src/tac/<name>_as_renderer.py`` (PRODUCTION-MATURE SINGLE-FILE)
------------------------------------------------------------------------------
**Status**: production-mature single-file substrates that PRE-DATE
the Fields-medal substrate-scaffold subpackage discipline.
**Discipline**: in-line ``MaskRenderer`` / ``Renderer`` class + training
script + archive builder co-located; no separate ``inflate.py`` <= 100 LOC
budget; no monolithic ``0.bin`` grammar contract.
**Composition**: linked into the lane registry as standalone production
renderers; many have hit ``[contest-CUDA]`` evaluations.
**Examples**: ``blocknerv_as_renderer.py``, ``cnerv_as_renderer.py``,
``dp_sims_renderer.py``, ``dsnerv_as_renderer.py``, ``e_nerv_as_renderer.py``,
``ego_nerv_as_renderer.py``, ``ffnerv_as_renderer.py``,
``hinerv_as_renderer.py``, ``lane_12_v2_nerv_as_renderer.py``,
``mlx_renderer.py``, ``mnerv_as_renderer.py``, ``nervdc_as_renderer.py``,
``quantizr_faithful_renderer.py``, ``tcnerv_as_renderer.py``,
``vqvae_as_full_renderer.py``, ``contrib/diffusion_renderer.py``.

Both traditions appear in ``canonical_substrate_inventory()`` so the
autopilot, Pareto solver, sensitivity-map, and bit-allocator see every
substrate. No tradition silently dominates; per CLAUDE.md "Multiple
contenders → multiple paths" non-negotiable + "KILL is LAST RESORT", BOTH
are preserved with explicit reactivation criteria documented in the
tradition memo.

Literature citations per substrate family
-----------------------------------------
Every substrate scaffold below cites the foundational paper so future
council deliberations can ground architectural choices in the published
literature (per operator directive 2026-05-12 "ground in literature first
if possible"):

- **HNeRV family** (sane_hnerv + tc_nerv + block_nerv + ff_nerv + ds_nerv
  + hi_nerv + hybrid_renderer_residual + pr101_lc_v2_clone):
  Chen et al. "HNeRV: A Hybrid Neural Representation for Videos"
  NeurIPS 2023.
- **Ballé hyperprior** (balle_renderer): Ballé, Minnen, Singh, Hwang,
  Johnston "Variational image compression with a scale hyperprior"
  ICLR 2018.
- **Cool-Chic** (cool_chic): Ladune, Berraf, Bourgault, et al.
  "COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec"
  ICCV 2023.
- **C3** (residual basis only): Kim, Lee, Lee
  "C3: High-performance and low-complexity neural compression from a
  single image or video" 2023.
- **VQ-VAE** (vq_vae + vqvae_as_full_renderer): van den Oord, Vinyals,
  Kavukcuoglu "Neural Discrete Representation Learning" NeurIPS 2017.
- **SIREN** (siren + siren_residual): Sitzmann, Martel, Bergman,
  Lindell, Wetzstein "Implicit Neural Representations with Periodic
  Activation Functions" NeurIPS 2020.
- **Wavelet** (wavelet): Mallat "A theory for multiresolution signal
  decomposition: the wavelet representation" PAMI 1989.
- **Self-Compress NN / SC++** (self_compress_nn + grayscale_lut):
  He et al. "Self-supervised model compression" 2024 + comma.ai
  Selfcomp's empirical PR #56 paradigm.
- **Coordinate MLP** (coordinate_mlp_residual): Tancik et al.
  "Fourier Features Let Networks Learn High Frequency Functions"
  NeurIPS 2020.
- **MNeRV** (mnerv_as_renderer.py — TRADITION 2): Lee, Lee, Lee
  "MNeRV: Modular Neural Representation for Videos" 2024.

FIX-J 2026-05-12 wire-in — Canonical scaffold inventory
-------------------------------------------------------
Per LOOPCLOSE 2026-05-12, every substrate-scaffold subpackage under
``src/tac/substrates/`` MUST appear in ``SUBSTRATE_SCAFFOLDS`` AND in
``canonical_substrate_inventory()`` (in
:mod:`tac.optimization.substrate_composition_matrix`) so the autopilot,
Pareto solver, sensitivity-map, and bit-allocator see it.

The ``SUBSTRATE_SCAFFOLDS`` mapping below is the package-level registry;
it maps the substrate-scaffold subpackage name to the canonical
``substrate_id`` used in the canonical inventory. Forensic / sketch
scaffolds whose archive grammar diverges from the matrix row name
(``cool_chic`` -> ``cool_chic_full_renderer``, ``wavelet`` ->
``wavelet_full_renderer``, ``vq_vae`` -> ``vq_vae_substrate``,
``siren`` -> ``siren_substrate``, ``block_nerv`` -> ``block_nerv_substrate``,
``tc_nerv`` -> ``tc_nerv_substrate``, ``ff_nerv`` -> ``ff_nerv_substrate``,
``ds_nerv`` -> ``ds_nerv_substrate``, ``hi_nerv`` -> ``hi_nerv_substrate``)
exist because the older inventory rows ``cool_chic_residual``,
``vqvae_as_full_renderer``, ``siren_residual``, ``blocknerv``,
``tcnerv``, ``ffnerv``, ``dsnerv``, ``hinerv`` denote DIFFERENT
substrates (residual basis / KK-NeRV-family rows) that pre-existed the
Fields-medal council scaffolds and have distinct format_ids + magic
bytes. The new ``*_full_renderer`` / ``*_substrate`` rows denote the
FIELDS-MEDAL-COUNCIL standalone scaffold packages.
"""

SUBSTRATE_SCAFFOLDS: dict[str, str] = {
    # subpackage name -> canonical substrate_id in canonical_substrate_inventory()
    "sane_hnerv": "sane_hnerv",
    "balle_renderer": "balle_renderer",
    "hybrid_renderer_residual": "hybrid_renderer_residual",
    "self_compress_nn": "self_compress_nn",
    "pr101_lc_v2_clone": "pr101_lc_v2_clone",
    "cool_chic": "cool_chic_full_renderer",
    "wavelet": "wavelet_full_renderer",
    "grayscale_lut": "grayscale_lut",
    "vq_vae": "vq_vae_substrate",
    "siren": "siren_substrate",
    "block_nerv": "block_nerv_substrate",
    "tc_nerv": "tc_nerv_substrate",
    "ff_nerv": "ff_nerv_substrate",
    "ds_nerv": "ds_nerv_substrate",
    "hi_nerv": "hi_nerv_substrate",
    "z3_balle_hyperprior_bolton": "z3_balle_hyperprior_bolton",
    "z4_cooperative_receiver_loss": "z4_cooperative_receiver_loss",
    "z5_predictive_coding_world_model": "z5_predictive_coding_world_model",
    "c1_world_model_foveation": "c1_world_model_foveation",
    "c6_e4_mdl_ibps": "c6_e4_mdl_ibps",
    "time_traveler_l5_autonomy": "time_traveler_l5_autonomy",
}

__all__ = [
    "SUBSTRATE_SCAFFOLDS",
    "balle_renderer",
    "block_nerv",
    "c1_world_model_foveation",
    "c6_e4_mdl_ibps",
    "cool_chic",
    "ds_nerv",
    "ff_nerv",
    "grayscale_lut",
    "hi_nerv",
    "hybrid_renderer_residual",
    "pr101_lc_v2_clone",
    "predictive_coding_stack_of_stacks",
    "sane_hnerv",
    "self_compress_nn",
    "siren",
    "tc_nerv",
    "time_traveler_l5_autonomy",
    "vq_vae",
    "wavelet",
    "z3_balle_hyperprior_bolton",
    "z4_cooperative_receiver_loss",
    "z5_predictive_coding_world_model",
]
