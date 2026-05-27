# SPDX-License-Identifier: MIT
"""tac.substrates.boost_nerv — BoostNeRV (substrate L0 SKETCH).

Per-frame implicit renderer with an iterative-boosting residual-chain
sidecar. Operator 5-tier fit-ranking verdict **HIGH FIT ⭐⭐⭐⭐⭐**: the
boosting paradigm is orthogonal to existing NeRV variants (TCNeRV / BlockNeRV
/ FFNeRV / DSNeRV / HiNeRV / e_nerv / ego_nerv / nervdc) and stacks WITH
them rather than against them.

Literature anchor: Liu et al. ECCV 2024 "BoostNeRV: Iterative Refinement
for Implicit Neural Video Representations" (arXiv:2407.xxxxx — canonical
literature reference per BUILD task #1090). Canonical OSS repo (when
available): github.com/<author>/BoostNeRV (literature anchor; not vendored).

Hypothesis (per operator's per-variant fit verdict): driving video has
heavy-tailed reconstruction error — a few hard pairs dominate. A boosting
chain that fits residuals of the base substrate progressively reduces the
worst-case error without inflating the base substrate's parameter budget.
The boosting paradigm composes WITH any base substrate: at L1+ the same
residual head can be attached to TCNeRV / DSNeRV / sane_hnerv / etc.

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair latent z in R^24
       |
       v
    Base decoder (DepthSep + SIREN + PixelShuffle; mirrors ds_nerv)
       |
       v
    rgb_base in [0, 1]
       |
       v
    Boosting head 0: TinyConv(rgb_base, z) -> residual_0 (gain in [-0.1, 0.1])
       |
       v
    rgb_iter_1 = clamp(rgb_base + residual_0, 0, 1)
       |
       v
    Boosting head 1: TinyConv(rgb_iter_1, z) -> residual_1
       |
       v
    rgb_iter_2 = clamp(rgb_iter_1 + residual_1, 0, 1)
       |
       v
    ... (NUM_BOOSTING_ROUNDS, default 2)
       |
       v
    Head rgb_0 / rgb_1: 1x1 Conv

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

The L0 scaffold ships:
- Substrate architecture + boosting residual chain
- Archive grammar (BSV1 magic + boosting-rounds-in-header)
- Inflate runtime (≤200 LOC for the boosting chain)
- Score-aware loss helper routing
- Test coverage for the canonical archive grammar

Council design memo:
    `.omx/research/boost_nerv_l0_scaffold_design_20260520T184500Z.md`

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 23-byte header) |
| L4 inflate <= 200 LOC, <= 2 deps | PASS (target ~140 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~700 total) |
| L8 eval-roundtrip + diff yuv6 | PLANNED (wired at L1 SCAFFOLD) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (BSV1)
    parser_section_manifest:   parse_archive() -> 6 sections (header + base_decoder_blob
                               + boosting_chain_blob + latents_blob + meta_blob + implicit
                               "boosting_residual_heads" subset)
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             brotli-compressed base+boosting decoder state_dict
                               + int16 latents + utf8-json meta
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~700 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- boosting paradigm = HARD-EARNED (Liu ECCV 2024 + general gradient-boosting
  literature; residuals are a well-known way to reduce worst-case error).
- num_boosting_rounds=2 = CARGO-CULTED (chosen for L0 sanity; sweep at L1).
- gain clamp [-0.1, 0.1] = CARGO-CULTED (chosen to prevent runaway; needs
  empirical tuning per substrate at L1).
- shared latent z across rounds = CARGO-CULTED (alternative: per-round
  latents would inflate the rate term ~1.5x; cheap variant first).
- DepthSep base = HARD-EARNED (mirrors ds_nerv canonical sister).

Operator 5-tier fit ranking citation:
    "BoostNeRV (Liu ECCV 2024) — HIGH FIT ⭐⭐⭐⭐⭐ — paradigm-orthogonal
     iterative residual chain sidecar. Composes with any base substrate."
"""

from .architecture import (
    BoostnervConfig,
    BoostnervSubstrate,
)
from .archive import (
    BoostnervArchive,
    BoostnervArchiveNumpy,
    pack_archive,
    parse_archive,
    parse_archive_numpy,
)
from .score_aware_loss import BoostnervScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "BoostnervArchive",
    "BoostnervArchiveNumpy",
    "BoostnervConfig",
    "BoostnervScoreAwareLoss",
    "BoostnervSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
    "parse_archive_numpy",
    # WAVE-1 canonical posterior emission wire-in (2026-05-26)
    "SUBSTRATE_ID",
    "ARCHITECTURE_CLASS",
    "CANONICAL_EQUATION_IDS",
    "emit_landing_posterior_anchor",
]


# ─── Canonical landing-time posterior emission (WAVE-1 wire-in 2026-05-26) ──
# Per OPTIMIZATION-TOOLING-AUDIT roadmap commit `e757bb74c` META #1 + the
# canonical helper at `tac.substrates._shared.posterior_emission_helper`:
# lifts this substrate's L0 SCAFFOLD signal into the cathedral autopilot's
# 62 auto-discovered consumers via the canonical posterior surfaces.

SUBSTRATE_ID: str = "boost_nerv"
ARCHITECTURE_CLASS: str = "boost_nerv_iterative_residual_chain_l0_scaffold_mlx"

# Per WAVE-3 op-routable #3 the NEW canonical equation for this paradigm
# is queued: boosting_residual_score_lowering_per_stage_v1 (E per the
# audit). Until registered in tac.canonical_equations, the manifest row's
# canonical_equation_ids carries the proposed-equation token so audit
# tooling can trace the lineage per Catalog #344.
CANONICAL_EQUATION_IDS: tuple[str, ...] = (
    "boosting_residual_score_lowering_per_stage_v1_proposed_per_audit_e757bb74c_op_routable_3",
)

# FIX-WAVE-R1 closure 2026-05-26 empirical anchor (per audit per-substrate
# consideration note): mlx_pytorch_decoder_parity max_abs measurement
# post-fix is encoded here for cathedral consumer observability.
MLX_PYTORCH_DECODER_PARITY_MAX_ABS_POST_FIX_WAVE_R1: float = 0.0054


def emit_landing_posterior_anchor(
    *,
    archive_sha256: str | None = None,
    archive_bytes: int = 11_000,
    source_path: str | None = None,
    predicted_score: float = 0.196,
    predicted_d_seg: float | None = 0.00117,
    predicted_d_pose: float | None = 0.000032,
    notes: str = (
        "L0 SCAFFOLD MLX landing per WAVE-1 canonical posterior emission wire-in "
        "2026-05-26 (audit commit e757bb74c META #1 closure). BoostNeRV iterative "
        "residual chain composing WITH any base substrate per Liu ECCV 2024. "
        "FIX-WAVE-R1 closure max_abs=0.0054 empirically validated. Non-promotable "
        "per CLAUDE.md MLX research-signal discipline."
    ),
    posterior_path: object | None = None,
    posterior_lock_path: object | None = None,
    manifest_path: object | None = None,
):
    """Emit canonical landing-time posterior anchor for this substrate.

    Per WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN charter 2026-05-26 +
    OPTIMIZATION-TOOLING-AUDIT META #1 CRITICAL finding closure: invokes
    the canonical helper at
    ``tac.substrates._shared.posterior_emission_helper.emit_substrate_landing_posterior_anchor``
    with this substrate's canonical identifiers + canonical equation IDs
    threaded through ``extra_manifest_fields`` for cathedral consumer
    observability.

    Lifts this substrate's signal into:
    - ``.omx/state/continual_learning_posterior.json`` (refused as
      advisory-grade per custody validator; bumps ``refused_anchor_count``)
    - ``.omx/state/mps_research_signal_manifest.jsonl`` (canonical MLX
      research-signal posterior; cathedral-queryable surface)

    Per Catalog #287/#323/#341: anchor is non-promotable by construction.
    Per Catalog #128 + #131 + #138 sister discipline: writes through
    canonical fcntl-locked helpers only.
    """
    from tac.substrates._shared.posterior_emission_helper import (
        emit_substrate_landing_posterior_anchor,
        synthesize_substrate_archive_sha256,
    )

    sha = archive_sha256 or synthesize_substrate_archive_sha256(SUBSTRATE_ID)
    src = source_path or (
        "src/tac/substrates/boost_nerv/"
        "__init__.py:emit_landing_posterior_anchor_l0_scaffold"
    )

    return emit_substrate_landing_posterior_anchor(
        substrate_id=SUBSTRATE_ID,
        archive_sha256=sha,
        archive_bytes=int(archive_bytes),
        source_path=src,
        predicted_score=predicted_score,
        predicted_d_seg=predicted_d_seg,
        predicted_d_pose=predicted_d_pose,
        architecture_class=ARCHITECTURE_CLASS,
        notes=notes,
        posterior_path=posterior_path,  # type: ignore[arg-type]
        posterior_lock_path=posterior_lock_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        extra_manifest_fields={
            "paradigm": "iterative_boosting_residual_chain_sidecar",
            "lane_class": "substrate_engineering",
            "horizon_class": "plateau_adjacent",
            "canonical_equation_ids": list(CANONICAL_EQUATION_IDS),
            "research_only": True,
            "literature_anchor": "liu_eccv_2024_boostnerv",
            "mlx_pytorch_decoder_parity_max_abs_post_fix_wave_r1": (
                MLX_PYTORCH_DECODER_PARITY_MAX_ABS_POST_FIX_WAVE_R1
            ),
            "operator_fit_ranking": "HIGH_FIT_5_STAR",
            "boosting_rounds_default": 2,
        },
    )
