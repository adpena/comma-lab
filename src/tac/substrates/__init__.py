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
"""

__all__ = ["sane_hnerv"]
