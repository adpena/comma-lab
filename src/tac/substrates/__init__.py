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
}

__all__ = [
    "SUBSTRATE_SCAFFOLDS",
    "balle_renderer",
    "block_nerv",
    "cool_chic",
    "ds_nerv",
    "ff_nerv",
    "grayscale_lut",
    "hi_nerv",
    "hybrid_renderer_residual",
    "pr101_lc_v2_clone",
    "sane_hnerv",
    "self_compress_nn",
    "siren",
    "tc_nerv",
    "vq_vae",
    "wavelet",
]
