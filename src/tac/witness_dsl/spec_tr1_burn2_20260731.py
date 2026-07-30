"""ddm_b2b — burn-2 program factories: the QA86 config-corrections + QA83 head race +
QA84 grammar race, composed on the QA24 5-piece base (SoT levers in
``spec_tr1_renderer_20260728``).

These are BURN-2-READY programs (never the live burn): MAIN composes + fires them at the
post-burn boundary (or the mid-run resume, on operator GO). Every race arm is
pre-registered at MATCHED counted bytes with a falsifier; the byte-matching itself is a
burn-2 tuning step (measure ``total_counted_bytes``, adjust code_width/grid to match).

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED. Config-generation only;
score_claim=False; every row that a burn produces is [macOS-CPU advisory] until byte-closed.
"""

from __future__ import annotations

from dataclasses import replace

from tac.witness_dsl.qa84_rowband_grammar_20260731 import (
    RowBandGrammar,
    default_flip_band_grammar,
)
from tac.witness_dsl.spec_tr1_renderer_20260728 import (
    TR1RendererProgramV1,
    lever_a1_gate,
    lever_basin_handoff,
    lever_byte_ledger_coder,
    lever_desc_level_roundtrip,
    lever_ema_decay,
    lever_lotto,
    lever_rate_in_loss,
    lever_renderer_capacity,
    lever_renderer_head,
    lever_seg_margin_weight,
    lever_seg_physics,
    lever_token_grid,
    lever_token_init,
    lever_token_quant_anneal,
    lever_token_rowband,
    lever_token_temporal,
    lever_variant,
    lever_window,
    qa24_composed_burn_program,
)

# --- the S-exact exchange rate (contest scoring: rate term = 25*bytes/37_545_489) --------
S_RATE_NUM: int = 25
S_RATE_ARCHIVE_DENOM: int = 37_545_489


def burn_geometry_n_counted_tokens(*, num_pairs: int = 600, keep_cells: int = 384,
                                   code_width: int = 4, shared_base: bool = True) -> int:
    """The COUNTED token count for the burn geometry (shared_base keep-set): the base is
    coded once (keep_cells*code_width) + the per-frame delta stream
    (num_pairs*keep_cells*code_width). The rate-in-loss surrogate's S value scales with this."""
    delta = num_pairs * keep_cells * code_width
    base = keep_cells * code_width if shared_base else 0
    return int(base + delta)


def derive_w_rate_exchange_rate(n_counted_tokens: int) -> tuple[float, str]:
    """QA86(d) / census T19: DERIVE the S-commensurate rate-in-loss weight from the exact
    exchange rate. The surrogate ``token_rate_term`` returns MEAN bits/token; reducing the
    mean by 1 bit/token saves ``n_counted_tokens/8`` bytes, whose S value is
    ``(25/37_545_489) * n/8``. Matching w_seg=100 (which IS S-exact: 100*d_seg is the S seg
    term), the S-commensurate weight is that per-unit-surrogate S value.

    Returns (w_rate, provenance). The SURROGATE<->exact-bytes map is approximate (soft
    entropy != realized SMEVR bytes), so this is a DERIVED-ESTIMATE anchoring the burn-2
    rate A/B (QA86a), never an asserted optimum.
    """
    bytes_per_unit_surrogate = n_counted_tokens / 8.0
    w = (S_RATE_NUM / S_RATE_ARCHIVE_DENOM) * bytes_per_unit_surrogate
    prov = (f"DERIVED-ESTIMATE w_rate = (25/37_545_489) * n/8 = {w:.6f} for "
            f"n_counted_tokens={n_counted_tokens}; S-commensurate with w_seg=100 (S-exact). "
            f"SURROGATE<->exact-bytes approximate => QA86a rate A/B measures 0.05 vs {w:.4f}.")
    return float(w), prov


#: The derived w_rate for the live burn geometry (keep-384, c4, 600 pairs, shared_base).
BURN_DERIVED_W_RATE, BURN_DERIVED_W_RATE_PROV = derive_w_rate_exchange_rate(
    burn_geometry_n_counted_tokens())


def _append_levers(base: TR1RendererProgramV1, *extra) -> TR1RendererProgramV1:
    """Compose extra levers onto a base program (later levers WIN = theta* composition)."""
    prog = replace(base, levers=tuple(base.levers) + tuple(extra))
    prog.validate()  # never-invent-flags fail-closed
    return prog


# ---------------------------------------------------------------------------
# QA86(a) — the entropy-vs-SMEVR-matched rate-surrogate A/B (sg1 §3.4 skipped race).
# ---------------------------------------------------------------------------
def qa86_rate_surrogate_race_programs(
    variant: str, out_dir: str, mask_path: str, *, w_rate: float = 0.05,
    epochs: int = 400, max_wall_minutes: float = 480.0, gt_cache: str | None = None,
) -> dict[str, TR1RendererProgramV1]:
    """The QA86(a) rate-surrogate A/B: {entropy (coder-mismatched control) ·
    smevr_surrogate (temporal-delta, matches the shipped SMEVR event/value split)} at
    MATCHED budget (same mask/grid/w_rate; SMEVR byte ledger on both). Falsifier: the
    smevr_surrogate arm's realized SMEVR archive bytes not lower than the entropy arm at
    matched d_seg => the coder-matched surrogate does not bind => keep entropy (or fall to
    post-hoc coding)."""
    def _arm(rate_model: str) -> TR1RendererProgramV1:
        return _append_levers(
            qa24_composed_burn_program(
                variant, out_dir, mask_path, epochs=epochs,
                max_wall_minutes=max_wall_minutes, w_rate=w_rate, rate_model=rate_model,
                gt_cache=gt_cache),
            lever_byte_ledger_coder("smevr"))
    return {"A_entropy": _arm("entropy"), "B_smevr_surrogate": _arm("smevr_surrogate")}


# ---------------------------------------------------------------------------
# QA86(c) — the MID-RUN resume config (corrects the EMA clamp at a stage-boundary resume).
# ---------------------------------------------------------------------------
def qa86_mid_run_resume_program(
    variant: str, out_dir: str, mask_path: str, resume_from: str, *,
    ema_decay: float, epochs: int = 400, max_wall_minutes: float = 480.0,
    w_rate: float = 0.05, rate_model: str = "entropy", gt_cache: str | None = None,
) -> TR1RendererProgramV1:
    """QA86(c) MID-RUN FIX: resume the LIVE burn from a stage-boundary checkpoint with the
    EMA clamp corrected. ``ema_decay`` is passed EXPLICITLY (auditable: MAIN sees exactly
    what fires) = the run-geometry-derived value the fixed derive now yields (0.99986667 for
    U=30,000). The SMEVR byte ledger is engaged.

    MAIN fires this ONLY on operator GO. TRADEOFF to surface: the live shadow was warmed
    under 0.9995 (warmup 4,000 steps); switching to 0.999867 slows FORWARD averaging
    (warmup 15,038). It changes NO trained/shipped byte (EMA is the inference-shadow only);
    it changes the endpoint EMA-shadow the post-burn re-solve reads. A byte-continuous
    resume that declines the change passes --ema-decay 0.9995 + --byte-ledger-coder zlib."""
    base = qa24_composed_burn_program(
        variant, out_dir, mask_path, epochs=epochs, max_wall_minutes=max_wall_minutes,
        w_rate=w_rate, rate_model=rate_model, gt_cache=gt_cache, resume_from=resume_from)
    return _append_levers(base, lever_ema_decay(ema_decay), lever_byte_ledger_coder("smevr"))


# ---------------------------------------------------------------------------
# QA83 — the output-space factorization head race (census §4.1).
# ---------------------------------------------------------------------------
def qa83_head_race_programs(
    variant: str, out_dir: str, mask_path: str, *, epochs: int = 400,
    max_wall_minutes: float = 480.0, w_rate: float = 0.05, gt_cache: str | None = None,
    slack_gain: float = 0.05,
) -> dict[str, TR1RendererProgramV1]:
    """QA83 §4.1 head A/B/C at MATCHED counted bytes: {A rgb c4 (control = burn endpoint) ·
    B class_field k=1 + lift · C class_field_photo k=2 (+margin-slack luma photometric)}. The
    lift is rule-118-FREE code; only the k-channel token field is counted, so B/C free head
    bytes (fund more tokens OR fewer token bytes at held d_seg). Byte-matching is a burn-2
    tuning step (measure total_counted_bytes; the §4.1 form is c4->c2 code_width). SMEVR
    ledger on all arms. Falsifier: B/C endpoint d_seg no better than A at matched bytes =>
    factorized-output closes at INSTANCE + the v14 static-dict negative extends to trained
    forms."""
    def _arm(mode: str) -> TR1RendererProgramV1:
        return _append_levers(
            qa24_composed_burn_program(
                variant, out_dir, mask_path, epochs=epochs,
                max_wall_minutes=max_wall_minutes, w_rate=w_rate, gt_cache=gt_cache),
            lever_renderer_head(mode, slack_gain), lever_byte_ledger_coder("smevr"))
    return {"A_rgb": _arm("rgb"), "B_class_field": _arm("class_field"),
            "C_class_field_photo": _arm("class_field_photo")}


# ---------------------------------------------------------------------------
# QA84 — the variable-size cell tiling grammar race (census §4.2).
# ---------------------------------------------------------------------------
def _rowband_arm_program(variant: str, out_dir: str, grammar: RowBandGrammar, *,
                         epochs: int, max_wall_minutes: float, w_rate: float,
                         margin_temp: float, gt_cache: str | None) -> TR1RendererProgramV1:
    """The row-band arm: D8 base + row-band tie + the shared d_seg pieces (margin-weight,
    rate-in-loss, lattice-anneal, solve_project init, basin-handoff) — NO D16 cell_mask (the
    tie IS the cell structure). Composed-S is verdict-only (omitted; instrument, not a byte)."""
    levers = [
        lever_variant(variant),
        lever_token_grid(8, grammar.code_width),   # D8 FINE base (row-band needs it)
        lever_renderer_capacity(24),
        lever_desc_level_roundtrip(16, "round"),
        lever_token_temporal("shared_base"),
        lever_seg_physics("ce", 100.0),
        lever_token_init("solve_project"),
        lever_basin_handoff("on"),
        lever_a1_gate(10),
        lever_window(epochs, max_wall_minutes, batch_pairs=8, lr=2e-3),
        lever_seg_margin_weight(margin_temp),
        lever_token_quant_anneal("at_knee"),
        lever_rate_in_loss(w_rate),
        lever_token_rowband(grammar),
        lever_byte_ledger_coder("smevr"),
    ]
    if variant == "lotto":
        levers.append(lever_lotto(118, 0.5))
    prog = TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=out_dir,
                                gt_cache=gt_cache, full_confirm=True)
    prog.validate()
    return prog


def qa84_grammar_race_programs(
    variant: str, out_dir: str, mask_path: str, *, grammar: RowBandGrammar | None = None,
    epochs: int = 400, max_wall_minutes: float = 480.0, w_rate: float = 0.05,
    margin_temp: float = 1.0, gt_cache: str | None = None,
) -> dict[str, object]:
    """QA84 §4.2 grammar race at MATCHED counted bytes from birth: {A uniform D16+drop50
    (control = the qa24 base) · B row-band D8/D16 (the op1 GATE-PASSED foveation lane, 72.1%%
    flip mass rows 160-240)}. Quadtree D8-at-annulus is the NAMED further arm (census: pays only
    if in-band azimuthal sparsity is real — QA74 g4). Byte-matching (D8-rowband vs D16-drop50) is
    a burn-2 tuning step (measure total_counted_bytes). SMEVR ledger on both; raster wire order
    (QA85). Falsifier: no matched-bytes d_seg win => spatial uniformity survives at INSTANCE;
    row-band >= quadtree => the separable approximation suffices."""
    g = grammar or default_flip_band_grammar()
    control = _append_levers(
        qa24_composed_burn_program(
            variant, out_dir, mask_path, epochs=epochs, max_wall_minutes=max_wall_minutes,
            w_rate=w_rate, margin_temp=margin_temp, gt_cache=gt_cache),
        lever_byte_ledger_coder("smevr"))
    rowband = _rowband_arm_program(
        variant, out_dir, g, epochs=epochs, max_wall_minutes=max_wall_minutes,
        w_rate=w_rate, margin_temp=margin_temp, gt_cache=gt_cache)
    return {"A_uniform_D16_drop50": control, "B_rowband_D8": rowband, "grammar": g,
            "quadtree_named_further_arm": "D8-at-annulus-cells (census §4.2; pays iff in-band "
            "azimuthal sparsity real, QA74 g4 custody) — not built this unit"}


# ---------------------------------------------------------------------------
# The fully-corrected burn-2 BASE (all QA86 corrections applied; derived w_rate option).
# ---------------------------------------------------------------------------
def burn2_corrected_base_program(
    variant: str, out_dir: str, mask_path: str, *, epochs: int = 400,
    max_wall_minutes: float = 480.0, use_derived_w_rate: bool = False,
    rate_model: str = "entropy", gt_cache: str | None = None,
    resume_from: str | None = None,
) -> TR1RendererProgramV1:
    """The QA24 5-piece base with ALL QA86 corrections engaged: SMEVR byte ledger (T5),
    EMA clamp fixed by the trainer derive (T6, no explicit override needed on a fresh burn),
    w_rate = derived-estimate (T19) if ``use_derived_w_rate`` else the live 0.05, margin_temp
    provenance carried. The default aim of burn-2; head/grammar levers are A/B'd separately."""
    w_rate = BURN_DERIVED_W_RATE if use_derived_w_rate else 0.05
    base = qa24_composed_burn_program(
        variant, out_dir, mask_path, epochs=epochs, max_wall_minutes=max_wall_minutes,
        w_rate=w_rate, rate_model=rate_model, gt_cache=gt_cache, resume_from=resume_from)
    return _append_levers(base, lever_byte_ledger_coder("smevr"))
