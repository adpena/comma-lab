"""Phase 3 inflate scaffold (DESIGN-ONLY).

Per CLAUDE.md HNeRV parity discipline L4: inflate.py LOC budget ≤ 200.
This scaffold is well under, leaving room for the future trainer to
populate the real codec dispatcher inline.

Phase 3 inflate path
--------------------

  1. Open archive.zip
  2. Read latent.bin (joint-source coded; per T17 shared VQ-VAE codebook)
  3. Read decoder.bin (FP4 + Brotli; per T18 nonlinear transform)
  4. Reconstruct frames via decoder(latent)
  5. Replace auxiliary scorer θ_aux with FROZEN contest scorer (per CLAUDE.md
     ``check_no_scorer_load_at_inflate``; see strict-scorer-rule)
  6. Score via contest_auth_eval.py

CLAUDE.md compliance
--------------------

  - NO scorer-at-inflate (auxiliary scorer is TRAINING-ONLY, swapped at eval)
  - NO MPS-fallback default (CUDA-required; explicit ``--device cpu`` opt-in
    only when deterministic-bytes acceptable for non-promoting CPU reproduction)
  - eval_roundtrip applied (384→874→uint8→384)
  - Inflate LOC ≤ 200 (this file)
  - All bytes accounted via build_manifest.json (no hidden sidecars)

DESIGN-ONLY STATUS — gates dispatch decision
--------------------------------------------

This scaffold is SCAFFOLD-ONLY. The real inflate path will be populated by
the future Phase 3 dispatcher. Calling :func:`phase3_inflate_design_only` raises
``NotImplementedError`` with a pointer to the dispatch decision tree.
"""
from __future__ import annotations

from pathlib import Path

PHASE3_INFLATE_DESIGN_ONLY_NOT_IMPL_MESSAGE = (
    "Phase 3 inflate is DESIGN-ONLY. The real inflate path will be populated by "
    "the future Phase 3 dispatcher (FUTURE — gated on Phase 2 anchor + aaf68f37 "
    "verdict + operator approval). See "
    "fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md §Phase3."
)


def phase3_inflate_design_only(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> None:
    """SCAFFOLD-ONLY: raises NotImplementedError until Phase 3 dispatch lands.

    The signature is the canonical Phase 3 inflate entry point. The future
    Phase 3 dispatcher will replace this stub with the real inflate path.
    """
    if device == "mps":
        # CLAUDE.md non-negotiable: never MPS as authoritative axis.
        raise ValueError(
            "Phase 3 inflate refuses MPS device per CLAUDE.md MPS-NOISE rule. "
            "Use --device cuda for [contest-CUDA] or --device cpu only when "
            "deterministic-bytes acceptable for [contest-CPU] on Linux x86_64."
        )

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    # Per CLAUDE.md, never accept /tmp paths in any persisted artifact.
    if "/tmp/" in str(archive_path) or "/tmp/" in str(output_dir):
        raise ValueError(
            "Phase 3 inflate refuses /tmp paths per CLAUDE.md "
            "forbidden_tmp_paths rule. Use experiments/results/<lane_id>_<timestamp>/."
        )

    raise NotImplementedError(PHASE3_INFLATE_DESIGN_ONLY_NOT_IMPL_MESSAGE)


def phase3_inflate_loc_budget() -> int:
    """Return the current Phase 3 inflate LOC count.

    Per CLAUDE.md HNeRV parity discipline L4, this MUST stay ≤ 200. Used
    by future preflight gate to enforce the LOC budget.
    """
    here = Path(__file__)
    if not here.is_file():
        return 0
    return sum(1 for _ in here.read_text(encoding="utf-8").splitlines())


__all__ = [
    "phase3_inflate_design_only",
    "phase3_inflate_loc_budget",
    "PHASE3_INFLATE_DESIGN_ONLY_NOT_IMPL_MESSAGE",
]
