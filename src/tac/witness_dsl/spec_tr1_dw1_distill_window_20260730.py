"""ddm_dw1 — QA75 distill-WINDOW matched A/B/C programs (DSL, SoT).

The fork discriminator: does the QA74 25.58x amortization gap close by distilling the
renderer to the FEASIBLE b2b SegNet field (QA75 leads) or is it optimization/capacity-
limited (QA24 form fixes lead)?  Three MATCHED bounded governed windows resumed from the
SAME E2 endpoint (stage_seg_trunk_tau_final, epoch 400), differing by EXACTLY the distill /
head-relax term:

  * Window B (control)  = the STRONGEST honest continuation: the burn's FULL endpoint config
    (margin-weighted CE + rate-in-loss + SMEVR ledger + event schedule), distill OFF.
  * Window A (distill)  = B + the QA75 solve-frame distill term (mini-race-winning form).
  * Window C (chart)    = A + head-range-relax linear (off-RGB output-chart probe; MAIN charter).
    ADVISORY-NON-DEPLOYABLE (a head change breaks the E1 receiver arch tr1_lotto_combined_ema_v1).

Matched-config discipline (guard 3): all three share ONE base; A-vs-B differs by exactly the
distill flags, C-vs-A by exactly --head-range-relax.  The argv-diff law (warm_start_resume)
enforces it; the three diffs are pasted in the memo.

ema-decay is OMITTED (all three DERIVE the same window-geometry decay => matched); the burn
pinned it for its 400-ep geometry (a deliberate window delta per the warm-start resume law).

score_claim=false; advisory [macOS-CPU/MLX]; pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

from tac.witness_dsl.spec_tr1_renderer_20260728 import (
    TR1RendererProgramV1,
    lever_a1_gate,
    lever_basin_handoff,
    lever_byte_ledger_coder,
    lever_desc_level_roundtrip,
    lever_head_range_relax,
    lever_lotto,
    lever_rate_in_loss,
    lever_renderer_capacity,
    lever_seg_margin_weight,
    lever_seg_physics,
    lever_solve_frame_distill,
    lever_token_cell_mask,
    lever_token_grid,
    lever_token_init,
    lever_token_quant_anneal,
    lever_token_temporal,
    lever_variant,
    lever_window,
)

# E2 endpoint provenance (ddm_bc1 burn, stage_seg_trunk_tau_final; meta::epoch = 400).
E2_RESUME_EPOCH = 400


def _matched_base_levers(*, mask_path: str, window_epochs: int, max_wall_minutes: float,
                         a1_gate_every: int) -> list:
    """The base config SHARED by Windows A/B/C = the burn's endpoint config with the intended
    window deltas (basin-handoff OFF for a fixed-length window; a1-gate cadence for slope
    samples; the bounded window epochs).  epochs = resume_epoch + 1 + window_epochs so the
    trainer's range(start_epoch, epochs) runs exactly ``window_epochs`` epochs from E2."""
    epochs = E2_RESUME_EPOCH + 1 + window_epochs
    return [
        lever_variant("lotto"),
        lever_token_grid(16, 4),
        lever_renderer_capacity(24),
        lever_desc_level_roundtrip(16, "round"),
        lever_token_temporal("shared_base"),
        lever_seg_physics("ce", 100.0, 1.0, 1.0),      # on resume re-anchors to tau (#517-twin)
        lever_token_init("solve_project"),             # SKIPPED on resume; matches burn config
        lever_basin_handoff("off"),                    # window delta: run the full fixed window
        lever_a1_gate(a1_gate_every),                  # window delta: dense slope samples
        lever_window(epochs, max_wall_minutes, batch_pairs=8, lr=2e-3),
        lever_token_cell_mask(mask_path),              # matches burn (structural cell mask)
        lever_seg_margin_weight(1.0),                  # matches burn (margin-weighted CE)
        lever_token_quant_anneal("at_knee"),           # matches burn (STE re-engaged past knee)
        lever_rate_in_loss(0.05, "entropy"),           # matches burn (rate-in-loss)
        lever_byte_ledger_coder("smevr"),              # matches burn (SMEVR byte reporting; guard 5)
        lever_lotto(118, 0.5),                         # matches burn (lotto variant)
    ]


def dw1_window_program(kind: str, out_dir: str, *, mask_path: str, gt_cache: str,
                       resume_from: str, distill_field_cache: str,
                       distill_form: str = "kd_logits", distill_weight: float = 100.0,
                       distill_temp: float = 2.0, distill_attack_temp: float = 0.0,
                       window_epochs: int = 60, max_wall_minutes: float = 75.0,
                       a1_gate_every: int = 5) -> TR1RendererProgramV1:
    """Build one matched window. kind ∈ {control | distill | distill_head_relax}.
    full_confirm=True => the trainer runs the n600 realized confirm at the final stage exit
    (both endpoints get a real byte-closed n600 d_seg, guard 5/verdict step 4b)."""
    if kind not in ("control", "distill", "distill_head_relax"):
        raise ValueError("kind is control|distill|distill_head_relax")
    levers = _matched_base_levers(mask_path=mask_path, window_epochs=window_epochs,
                                  max_wall_minutes=max_wall_minutes, a1_gate_every=a1_gate_every)
    if kind in ("distill", "distill_head_relax"):
        levers.append(lever_solve_frame_distill(
            distill_field_cache, form=distill_form, weight=distill_weight,
            temp=distill_temp, attack_temp=distill_attack_temp))
    if kind == "distill_head_relax":
        levers.append(lever_head_range_relax("linear"))
    return TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=out_dir,
                                gt_cache=gt_cache, resume_from=resume_from, full_confirm=True)
