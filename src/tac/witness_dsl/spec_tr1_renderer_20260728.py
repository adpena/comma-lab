"""ddm_tb1 — SPEC_tr1 renderer DSL: typed Lever factories + program compile (SoT).

Config for the tr1 trained partition→pixel renderer is DSL-COMPILED, never ad-hoc
argv (CLAUDE.md "The DSL HOLDS every designed lever"). Every SPEC S4.1 lever lands
here as a ``Lever`` factory (the canonical ``tac.witness_dsl.curriculum_dsl.Lever``
dataclass: name + flag ``overrides``); ``TR1RendererProgramV1.compile_trainer_argv``
merges lever overrides (later levers win = theta* composition) into the trainer's
argv, and ``validate()`` FAIL-CLOSES on any flag the trainer's argparse does not
declare (never-invent-flags), by AST-scanning the trainer source's ``add_argument``
calls — no import of MLX needed at validation time.

Trainer: ``experiments/train_tr1_partition_renderer_mlx.py`` (this tree).
Evidence axis: config-generation only; score_claim=False.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.witness_dsl.curriculum_dsl import Lever

TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"


# ---------------------------------------------------------------------------
# Lever factories (SPEC S4.1, amended by the 2026-07-28 recall directive forces).
# ---------------------------------------------------------------------------
def lever_variant(variant: str) -> Lever:
    if variant not in ("plain", "lotto"):
        raise ValueError(f"variant must be plain|lotto, got {variant!r}")
    return Lever(name=f"tr1_variant_{variant}", overrides={"--variant": variant},
                 notes="A2 race arm: plain-conv vs G1-LOTTO supermask under matched counted bytes")


def lever_token_grid(downsample: int = 16, code_width: int = 4) -> Lever:
    if downsample not in (8, 16, 32):
        # ddm_gd4: 32 ADMITTED (mt1 §5 #1 rate row = 101,636 B archive, 23.72% of the gap;
        # gd3 §6.1 confirmed the trainer is ds=32-ready and the argparse/DSL menus were the
        # whole blocker). 384/32=12, 512/32=16 are integer; D=12 stays EXCLUDED (512/12 is not).
        raise ValueError("grid downsample raced over {8,16,32} (D=12 excluded: 512/12 "
                         "non-integer lattice — tb1 memo deviation from SPEC S1.2)")
    if code_width not in (2, 4, 6):
        raise ValueError("code width raced over {2,4,6} per SPEC S1.2")
    return Lever(name=f"tr1_token_grid_D{downsample}_c{code_width}",
                 overrides={"--grid-downsample": str(downsample),
                            "--code-width": str(code_width)},
                 notes="grid pitch vs bytes; ERF-bounded (r50~85px); Pareto-raced")


def lever_renderer_capacity(width: int = 24) -> Lever:
    return Lever(name=f"tr1_renderer_w{width}", overrides={"--renderer-width": str(width)},
                 notes="G3 capacity; topology derives from D (conv0 + per-up conv + head)")


def lever_desc_level_roundtrip(quant_levels: int = 16, ste: str = "round") -> Lever:
    if ste not in ("round", "dither"):
        raise ValueError("token STE raced over round|dither (asymmetry force)")
    return Lever(name=f"tr1_token_quant_L{quant_levels}_{ste}",
                 overrides={"--token-quant-levels": str(quant_levels), "--token-ste": ste},
                 notes="S2.1 fd2-wall STE across the description lattice; STE variant RACED")


def lever_token_temporal(mode: str = "shared_base") -> Lever:
    if mode not in ("shared_base", "independent"):
        raise ValueError("token temporal mode is shared_base|independent")
    return Lever(name=f"tr1_token_temporal_{mode}", overrides={"--token-temporal-mode": mode},
                 notes="Einstein d_cov/d_gauge force: identity-xi shared base vs A/B control")


def lever_lotto(seed: int = 118, mask_density_init: float = 0.5) -> Lever:
    return Lever(name=f"tr1_lotto_seed{seed}",
                 overrides={"--lotto-seed": str(seed),
                            "--lotto-mask-density-init": str(mask_density_init)},
                 notes="rule-118: PRNG expansion FREE; seed+density COUNTED in selector ledger")


def lever_seg_physics(form_start: str = "ce", w_seg: float = 100.0,
                      class_weight_lane: float = 1.0, margin_target: float = 1.0) -> Lever:
    return Lever(name=f"tr1_seg_{form_start}",
                 overrides={"--seg-form-start": form_start, "--w-seg": str(w_seg),
                            "--class-weight-lane": str(class_weight_lane),
                            "--margin-target": str(margin_target)},
                 notes="scorer-in-loop seg trunk; pose TERMINAL (#383) — no pose flag exists "
                       "on this trainer by design; margin_hinge = step-native raced form; "
                       "class_weight_lane = sn1 sided-asymmetry lever")


def lever_token_init(mode: str = "zero") -> Lever:
    if mode not in ("zero", "solve_project"):
        raise ValueError("token init mode is zero|solve_project")
    return Lever(name=f"tr1_token_init_{mode}",
                 overrides={"--token-init-mode": mode},
                 notes="lv1 B solve-init: eu1 teacher-as-init-oracle — v3 ANALYTIC "
                       "projection of the materializable solution-set member (GT frame_1 "
                       "at the render plane, area-mean downsample) into token space as "
                       "base+delta before the scorer loop; zero = tb1 gauge-hygiene "
                       "control (ker(A) stays at the zero lattice point). Adopt iff "
                       "strictly better realized d_seg at matched epoch (pre-registered "
                       "lv1 B rule; v1 joint-L2 + v2 token-gradient formulations "
                       "MEASURED inadmissible — custody in the lv1 memo)")


def lever_basin_handoff(mode: str = "off") -> Lever:
    if mode not in ("off", "on"):
        raise ValueError("basin handoff is off|on")
    return Lever(name=f"tr1_basin_handoff_{mode}",
                 overrides={"--basin-handoff": mode},
                 notes="operator x2 2026-07-28 burn schedule rule: TRAIN ONLY TO CONDITION "
                       "— on basin-entry (TerminalSolve §16.1 validity: quadratic crawl + "
                       "topology stable + no transitions + zero-alarm COUPLED_DESCENT "
                       "window) STOP permanently and hand off to the solve executors "
                       "(#423 quadratic_basin_finisher_probe GN/CG + eg1 E3 QDBS rail "
                       "cf7172e747 + #383 terminal pose; v19 realized acceptance vs the "
                       "handoff full-confirm baseline). Saddle/grokking #216/#475 "
                       "disambiguation: stalled solve + still-descending training => "
                       "resume training, re-arm doubled window. SHORTENS the sealed "
                       "wall-clock (train-least realized)")


def lever_a1_gate(gate_every: int = 5) -> Lever:
    return Lever(name=f"tr1_a1_gate_every{gate_every}", overrides={"--gate-every": str(gate_every)},
                 notes="A1 (fd2 binding transfer lesson): realized-argmax gate cadence")


def lever_window(epochs: int, max_wall_minutes: float, batch_pairs: int = 8,
                 lr: float = 2e-3) -> Lever:
    return Lever(name=f"tr1_window_ep{epochs}",
                 overrides={"--epochs": str(epochs),
                            "--max-wall-minutes": str(max_wall_minutes),
                            "--batch-pairs": str(batch_pairs), "--lr": str(lr)},
                 notes="bounded governed window; checkpoint-on-exit (P0 resumability)")


# ---------------------------------------------------------------------------
# QA83 / QA86 burn-2 levers (census §4.1 / §2 T-rows; each with a pre-registered
# falsifier + provenance rung — no bare constants).
# ---------------------------------------------------------------------------
def lever_renderer_head(mode: str = "rgb", slack_gain: float = 0.05) -> Lever:
    """QA83 (census §4.1) OUTPUT-SPACE FACTORIZATION: the renderer head output space.
    'rgb' = 3-ch RGB via sigmoid*255 (control = burn endpoint); 'class_field' = k=1 class
    scalar c(x) -> FIXED monotone gray lift (the 1-luma-channel ur-instance; comma10k
    class_values on one luma axis); 'class_field_photo' = k=2 (class + margin-slack-confined
    luma photometric channel). The lift is rule-118-FREE decoder code; only the k-channel
    token field is counted. v14 FORMULATION-negative (static-dict painting) does NOT cover
    this TRAINED-renderer form (census §4.1 conditional-validity note)."""
    if mode not in ("rgb", "class_field", "class_field_photo"):
        raise ValueError("renderer_head_mode is rgb|class_field|class_field_photo")
    ov = {"--renderer-head-mode": mode}
    if mode == "class_field_photo":
        ov["--head-photo-slack-gain"] = str(slack_gain)
    return Lever(
        name=f"tr1_renderer_head_{mode}", overrides=ov,
        notes="QA83 §4.1 factorized output head; lift=comma10k luma anchors "
              "(41/76/90/124/161), margin-optimal RGB refinement = named compress-time scorer "
              "step; falsifier: class_field/photo endpoint d_seg >= rgb at MATCHED counted bytes "
              "=> factorized-output closes at INSTANCE + v14 negative extends to trained forms",
        constant_manifest={
            "--head-photo-slack-gain": {
                "value": slack_gain, "rung": "DERIVED-CONSERVATIVE-BOUND",
                "provenance": "QA80 band-lemma margin-slack budget d=|m|/||dw||; 0.05*255~=13/255 "
                              "luma is a conservative fixed bound (bottom-margin-decile excluded "
                              "=> ~zero seg flips); the EXACT per-pixel budget from QA80's "
                              "flip-distance field is the named scorer refinement, not a constant"},
        } if mode == "class_field_photo" else {},
    )


def lever_ema_decay(value: float) -> Lever:
    """QA86(c) EXPLICIT EMA decay (for the mid-run resume config): pin --ema-decay so the
    sealed ticket records exactly the run-geometry-derived value the trainer's fixed
    derive_ema_decay now yields (default None => derive). Use for an auditable resume where
    MAIN must see the exact decay that fires; a fresh burn-2 can omit it (derive is fixed)."""
    return Lever(name=f"tr1_ema_decay_{value:.8f}".rstrip("0"),
                 overrides={"--ema-decay": repr(float(value))},
                 notes="QA86(c) explicit EMA decay = run-geometry law d=1-2/(phi*U), phi=0.5; the "
                       "[0.9,0.9995] tiny-smoke clamp UPPER cap that bound over the derived "
                       "0.999867 is REMOVED (trainer derive_ema_decay run-geometry ceiling)",
                 constant_manifest={
                     "--ema-decay": {
                         "value": float(value), "rung": "DERIVED (ema_decay_run_geometry_v1)",
                         "provenance": "d=1-2/(phi*U), phi=0.5, U=updates_per_run; registered LawRef "
                                       "ema_decay_run_geometry_v1 mode decay_from_warmup_fraction"},
                 })


def lever_token_rowband(grammar) -> Lever:
    """QA84 (census §4.2) VARIABLE-CELL GRAMMAR: pass a RowBandGrammar (D8 base, bulk 2x2 tie =>
    D16-effective, op1 flip-band free at D8). The spec is embedded inline (rule-118-free decoder
    side-info; only ~130 B counted). Compose with lever_token_grid(8, c) — the row-band arm
    needs the FINE (D8) base. gr1 nested-rungs DOMINATED is INSTANCE-scoped (solved-token
    post-hoc); the trained-renderer from-birth form is uncovered. Raster wire order UNCHANGED
    (QA85 Hilbert receipt stands). Falsifier: no matched-bytes d_seg win vs uniform D16 =>
    spatial uniformity survives at INSTANCE on this vehicle."""
    # constant_refs (=lawrefs) custody the LOAD-BEARING measured flip-mass (0.721) via the
    # registered anchor rowband_flip_mass_foveation_band_v1 (MAIN drift-detector EQUATIONS leg).
    from tac.witness_dsl.lawref import LADDER_MEASURED_ANCHOR, InputRef, LawRef

    flip_mass_ref = LawRef(
        equation_id="rowband_flip_mass_foveation_band_v1",
        inputs={"value": InputRef.literal(
            0.721, "op1 foveation gate QA74 per-render-row flip-mass typing (rows 160-240)",
            config_tags={"vehicle": "tr1_renderer"})},
        ladder_class=LADDER_MEASURED_ANCHOR)
    return Lever(
        name="tr1_token_rowband", overrides={"--token-rowband-spec": grammar.spec_json()},
        notes=f"QA84 §4.2 row-band foveation; DOF {grammar.dof_summary()}; falsifier: no "
              "matched-bytes d_seg win vs uniform D16 => uniformity survives at INSTANCE; "
              "row-band >= quadtree => the separable approximation suffices, quadtree closes",
        lawrefs={"--token-rowband-spec": flip_mass_ref},
        constant_manifest={
            "--token-rowband-spec": {
                "value": grammar.spec_json(), "rung": "MEASURED_ANCHOR (op1 foveation gate, QA74)",
                "equation_id": "rowband_flip_mass_foveation_band_v1",
                "provenance": "band rows from FLIP_BAND_RENDER_ROWS = 72.1-72.7%% flip mass "
                              "(op1 gate PASSED >=50%%; custodied by rowband_flip_mass_"
                              "foveation_band_v1 EmpiricalAnchor); D ~ (flip-density)^-alpha "
                              "separable approximation; not a bare constant — a measured field"},
        })


def lever_byte_ledger_coder(coder: str = "smevr") -> Lever:
    """QA86(b) / census T5: coder used to PRICE the token stream for stage/telemetry
    decisions. 'smevr' (default) = the SHIPPED r7 coder (decisions match the archive;
    Hilbert-race receipt: SMEVR-2D reproduces the stored member bytes EXACTLY); 'zlib' =
    the legacy temporal-delta surrogate (decision-noise vs shipped bytes; kept for a
    byte-continuous live-burn resume). NEVER changes trained/shipped bytes."""
    if coder not in ("smevr", "zlib"):
        raise ValueError("byte_ledger_coder is smevr|zlib")
    return Lever(name=f"tr1_byte_ledger_coder_{coder}",
                 overrides={"--byte-ledger-coder": coder},
                 notes="QA86(b) census T5: decisions priced by the shipped SMEVR coder; falsifier "
                       "N/A (observability/decision-quality fix, not a score lever)")


# ---------------------------------------------------------------------------
# QA24 5-piece composed re-burn levers (sg1 §3; each with a pre-registered falsifier).
# ---------------------------------------------------------------------------
def lever_token_cell_mask(mask_path: str) -> Lever:
    """§3.1 COARSE-FROM-BIRTH: a (grid_h,grid_w) bool .npy (True = KEEP). The derived
    gr1 cell_drop50 keep-384 (99.61% flip mass; sg1 §2). Inactive cells are zeroed in the
    token field (no gradient, excluded from the coded stream). Not a config tweak: the
    uniform lever_token_grid only supports D in {8,16}; this is a SELECTIVE 384-cell mask."""
    return Lever(name="tr1_token_cell_mask", overrides={"--token-cell-mask": str(mask_path)},
                 notes="§3.1 coarse-from-birth keep-set (gr1 cell_drop50 -0.098 post-hoc bound; "
                       "falsifier: re-burn endpoint d_seg >= 0.004310 at matched bytes => "
                       "coarse-from-birth closes at INSTANCE, solve-distillation leads)")


def lever_seg_margin_weight(temp: float = 1.0) -> Lever:
    """§3.2 boundary-annulus form fix: 100% of realized flips are in the bottom GT-margin
    decile (sg1 §1.3) yet the burn loss is uniform. Reweights the per-pixel seg loss toward
    the small-margin boundary annulus across ALL classes (distinct from class_weight_lane,
    which lv1 §D.4 rejected at 2.0). RACE vs the uniform arm."""
    return Lever(name="tr1_seg_margin_weight",
                 overrides={"--margin-weighted-loss": "on", "--margin-weight-temp": str(temp)},
                 notes="§3.2 inverse-margin reweight (make_loss_fn margin_weighted); falsifier: "
                       "margin-weighted endpoint d_seg >= uniform at matched epoch => close at "
                       "INSTANCE (does not help THIS renderer)",
                 constant_manifest={
                     "--margin-weight-temp": {
                         "value": temp, "rung": "RACED-NOT-ASSERTED (census T8/QA86d)",
                         "provenance": "FORM derived from the Fisher-margin law (curvature<->(-margin) "
                                       "Pearson 0.978) => w=1/(1+m/temp); the SCALE temp is a bare "
                                       "default with NO provenance rung. Burn-2 duty-to-measure: sweep "
                                       "{0.3, 1, 3} (half-weight at m=temp in logit-margin units). Not "
                                       "a constant asserted as optimal — a tracked race slot"},
                 })


def lever_rate_in_loss(w_rate: float, rate_model: str = "entropy") -> Lever:
    """§3.4 rate-in-loss (stl1 row-8 LAW, first application to the renderer burn): a
    differentiable token code-length/entropy surrogate added to the seg loss. The explicit
    form of the §3.3(b) redistribution co-benefit. 'entropy' = marginal soft-histogram of the
    quantized token lattice; 'smevr_surrogate' = temporal-delta soft-histogram (zlib-on-delta
    coder surrogate)."""
    if rate_model not in ("entropy", "smevr_surrogate"):
        raise ValueError("rate_model must be entropy|smevr_surrogate")
    return Lever(name=f"tr1_rate_in_loss_{rate_model}",
                 overrides={"--w-rate": str(w_rate), "--rate-model": rate_model},
                 notes="§3.4 stl1 row-8 rate-in-loss LAW; falsifier: rate-in-loss arm archive "
                       "bytes not lower than distortion-only at matched d_seg => law does not "
                       "bind this payload, fall back to post-hoc coding (gr1/lv1 stack)",
                 constant_manifest={
                     "--w-rate": {
                         "value": w_rate, "rung": "DERIVED-ESTIMATE (census T19/QA86d)",
                         "provenance": "S-exact exchange rate 25/37,545,489 (S/byte); the "
                                       "S-commensurate in-loss weight (matching w_seg=100 which IS "
                                       "S-exact) = (25/37,545,489) * n_counted_delta_tokens/8 ~= 0.077 "
                                       "for the burn geometry (n~923k tokens). The live 0.05 is ~65%% "
                                       "of derived; SURROGATE<->exact-bytes map is approximate => the "
                                       "burn-2 rate A/B (QA86a) MEASURES bytes/d_seg at 0.05 vs derived. "
                                       "See spec_tr1_burn2.derive_w_rate_exchange_rate"},
                     "--rate-model": {
                         "value": rate_model, "rung": "UNRACED (QA86a OWED)",
                         "provenance": "CORRECTED 2026-08-01 (receipt d619ec4ede). The prior rung said "
                                       "'RACED (QA86a)' while its own text said sg1 §3.4 SKIPPED the "
                                       "race -- a rung asserting a measurement that was never taken "
                                       "(gd1 T4 also classes it GENERIC-CHOSEN-UNRACED). The race is "
                                       "OWED, not done. Prior text also claimed smevr_surrogate "
                                       "'matches the shipped SMEVR event/value split'; it does NOT -- "
                                       "SMEVR factors against the per-cell temporal MODE "
                                       "(factor_mode_delta) while the surrogate uses CONSECUTIVE-FRAME "
                                       "deltas. It APPROXIMATES the temporal structure; it does not "
                                       "match the split. MEASURED rank-fidelity vs real shipped SMEVR "
                                       "bytes on the burn-4 parent lineage (r1c ep504->640, n=70 "
                                       "fields): live 'entropy' rho = -0.7235 [-0.943,-0.227] "
                                       "(ANTI-correlated -- a marginal histogram is invariant under "
                                       "temporal permutation, so it goes blind once the field stops "
                                       "FILLING and starts REARRANGING, which is where burn-4 sits); "
                                       "'smevr_surrogate' rho = +0.7412. Scope: FORMULATION, "
                                       "regime-scoped (entropy tracks at rho=+1.00 in the filling "
                                       "regime) -- NOT a family kill. Fixing rate_model is logically "
                                       "PRIOR to raising w_rate, whose derivation assumes a "
                                       "surrogate-to-bytes premise 'entropy' does not satisfy."},
                 })


def lever_token_quant_anneal(mode: str = "at_knee") -> Lever:
    """§3.3(a) lattice annealing / staged quantizer (PR95 DYNAMICS only, never the recipe):
    'off' engages the token STE from birth (tb1 control); 'at_knee' keeps float tokens (no
    STE) until the CE->tau knee EVENT, then engages the STE — find the basin in float, refine
    on the shipped lattice (possibly-original 'lattice annealing', ms2 tolerance-homotopy
    lineage)."""
    if mode not in ("off", "at_knee"):
        raise ValueError("token_quant_anneal must be off|at_knee")
    return Lever(name=f"tr1_token_quant_anneal_{mode}",
                 overrides={"--token-quant-anneal": mode},
                 notes="§3.3(a) staged quantizer engagement; falsifier: annealed-STE endpoint "
                       "d_seg >= from-birth-STE at matched bytes => no basin benefit, close")


def lever_composed_s_verdict(subset: int, subset_ids_path: str | None = None,
                             delta_ref_path: str | None = None) -> Lever:
    """§3.5 QA77-lite: >0 = at stage exits run the bounded terminal pose+photometric solve on
    this many pairs and record COMPOSED S (100*d_seg + sqrt(10*d_pose) + rate) so stage/
    endpoint decisions see the co9 sky/hood-freeze pose cost (Knee-A externality). VERDICT-
    level only (never differentiated through). REQUIRED by §2's pose caveat when the grid drops
    sky/hood. ``subset_ids_path`` (MAIN QA66 signal): .npy of pair indices = the pose-mass TAIL
    (top-17 = 74.3%) for max signal/sec; None = head. Solver = the PROVEN warp-constrained
    FD-Jacobian LM-GN (eg1/tt1 approach; MAIN steer) — the razor-sharp realized pose landscape
    needs damped GN, not first-order descent; convergence is smoke-gated before fire."""
    ov = {"--composed-s-gate-subset": str(subset)}
    if subset_ids_path is not None:
        ov["--composed-s-subset-ids"] = str(subset_ids_path)
    if delta_ref_path is not None:  # ADOPTED directional-delta (MAIN Option A)
        ov["--composed-s-delta-ref"] = str(delta_ref_path)
    return Lever(name="tr1_composed_s_verdict", overrides=ov,
                 notes="§3.5 co9 Knee-A pricing; QA66 pose-tail subset; falsifier: composed-S "
                       "never diverges from seg-only at stage exits => free insurance that "
                       "changed no decision (record; keep it — the correct instrument)")


def lever_jd1_joint_pose_finish(
    *,
    w_pose: float,
    start_epoch: int = 0,
    engage_on: str = "post_knee",
    seg_hold_weight: float = 0.0,
    seg_hold_floor_source: str = "off",
    seg_hold_floor: float = 0.0,
    seg_hold_margin: float = 0.0,
) -> Lever:
    """JD1: TR1 joint pose-finish after the seg/constrain boundary.

    This is the TR1-side consumer for the #383/R1 pose-finish physics: after the configured
    engagement predicate fires, the trainer loads ``gt_poses`` from the frozen GT cache and sends
    those 6D targets through ``make_loss_fn`` with ``compute_pose=True``.  The seg-hold term is a
    real hinge on the same seg proxy used by the trunk, so the pose descent cannot silently erase
    the stage-2 seg floor.  It is still a duty-to-measure lever: this factory makes the build
    reachable and sealed; it asserts no d_seg/d_pose improvement.
    """
    if w_pose <= 0.0:
        raise ValueError("JD1 joint pose-finish requires w_pose > 0")
    if start_epoch < 0:
        raise ValueError("start_epoch must be >= 0")
    if engage_on not in ("post_knee", "start_epoch"):
        raise ValueError("engage_on must be post_knee|start_epoch")
    if engage_on == "start_epoch" and start_epoch <= 0:
        raise ValueError("start_epoch engagement requires a positive start_epoch")
    if seg_hold_weight < 0.0 or seg_hold_floor < 0.0 or seg_hold_margin < 0.0:
        raise ValueError("seg-hold weight/floor/margin must be non-negative")
    if seg_hold_floor_source not in (
        "off", "last_pre_pose_epoch_loss", "checkpoint_tail_ep_loss", "explicit",
    ):
        raise ValueError("seg_hold_floor_source is off|last_pre_pose_epoch_loss|"
                         "checkpoint_tail_ep_loss|explicit")
    if seg_hold_weight > 0.0 and seg_hold_floor_source == "off":
        raise ValueError("seg_hold_weight requires a non-off floor source")
    if seg_hold_floor_source == "explicit" and seg_hold_floor <= 0.0:
        raise ValueError("explicit floor source requires seg_hold_floor > 0")
    overrides = {
        "--jd1-pose-finish-mode": "joint_loss",
        "--jd1-pose-finish-engage-on": engage_on,
        "--jd1-pose-finish-start-epoch": str(int(start_epoch)),
        "--jd1-w-pose": repr(float(w_pose)),
    }
    if seg_hold_weight > 0.0:
        overrides.update({
            "--jd1-seg-hold-weight": repr(float(seg_hold_weight)),
            "--jd1-seg-hold-floor-source": seg_hold_floor_source,
            "--jd1-seg-hold-margin": repr(float(seg_hold_margin)),
        })
        if seg_hold_floor_source == "explicit":
            overrides["--jd1-seg-hold-floor"] = repr(float(seg_hold_floor))
    return Lever(
        name="tr1_jd1_joint_pose_finish",
        overrides=overrides,
        notes="JD1 TR1 joint pose-finish: consumes gt_poses via make_loss_fn/PoseNet after "
              "post-knee constrain, with optional seg-hold hinge. Default trainer flags are "
              "off, so this is a sealed duty-to-measure arm, not a score claim.",
        constant_manifest={
            "--jd1-w-pose": {
                "value": float(w_pose),
                "rung": "RACED-NOT-ASSERTED (terminal pose-finish weight)",
                "provenance": "TR1 consumes the existing score-domain sqrt(10*d_pose) pose term; "
                              "the first live value is a race start, not an optimum."},
            "--jd1-seg-hold-weight": {
                "value": float(seg_hold_weight),
                "rung": "RACED-NOT-ASSERTED (stage-2 constrain strength)",
                "provenance": "Hinge on the latched seg proxy floor; prevents pose descent from "
                              "silently spending the seg win. Strength requires post-TP1 A/B."},
        })


def lever_solve_frame_distill(field_cache: str, *, form: str = "kd_logits",
                              weight: float = 100.0, temp: float = 2.0,
                              attack_temp: float = 0.0) -> Lever:
    """QA75 solve-frame distillation (ddm_dw1): finish the renderer against the b2b SegNet
    FIELD on the EXACT C1 solve frames (a precomputed scorer response, realized d_seg ~1.52e-4)
    instead of / alongside argmax-CE vs GT. The teacher's soft logits carry dark knowledge the
    hard GT labels lack; distilling to the FEASIBLE teacher is the clean capacity-vs-objective
    arbiter for the QA74 25.58x amortization gap. Forms {kd_logits | margin_field | argmax_ce} and
    the attack-set weighting are RACED (own-optimum law); the winner is passed here.

    Falsifier (preregistered, QA75 ledger row): distilled endpoint NOT clearly better than the CE
    control at matched budget => the amortization gap is NOT distillation-curable (optimization/
    capacity leads, class-change leg strengthens); distill slope clearly better => burn-3
    distill-opening GO. NON-DEPLOYABLE nothing here (rgb head, E1-decodable); Window C (head relax)
    carries the non-deployable flag, not this lever."""
    if form not in ("kd_logits", "margin_field", "argmax_ce"):
        raise ValueError("distill form is kd_logits|margin_field|argmax_ce")
    return Lever(
        name=f"tr1_solve_frame_distill_{form}",
        overrides={"--distill-field-cache": str(field_cache),
                   "--distill-form": form, "--distill-weight": str(weight),
                   "--distill-temp": str(temp), "--distill-attack-temp": str(attack_temp)},
        notes="QA75 solve-frame distill; teacher = b2b SegNet field on the EXACT C1 solve frames "
              "(feasible margins by construction, unlike GT); falsifier: distilled not clearly "
              "better than CE at matched budget => gap not distillation-curable (QA24 form fixes "
              "lead) else burn-3 distill-opening GO",
        constant_manifest={
            "--distill-temp": {
                "value": temp, "rung": "CANONICAL (Quantizr/PR95 kl_on_logits T=2.0; Hinton 2015)",
                "provenance": "project-canonical KD temperature; the SegNet distillation "
                              "temperature used across the contest (CLAUDE.md Quantizr SegNet "
                              "kl_on_logits T=2.0). Not a bare constant — the KD lineage default"},
            "--distill-weight": {
                "value": weight, "rung": "DERIVED = w_seg (S-exact d_seg weight 100)",
                "provenance": "the distill term is a d_seg surrogate on the FEASIBLE teacher, so it "
                              "shares the S-exact seg weight (100 = the d_seg coefficient in S). "
                              "Own-optimum: comparable early gnorm to the seg term (raced, not "
                              "asserted); a per-form gnorm check confirms it is not under/over-driven"},
            "--distill-attack-temp": {
                "value": attack_temp, "rung": "RACED (ddm_dw1 mini-race dimension)",
                "provenance": "Fisher-margin law (curvature<->(-margin) Pearson 0.978): emphasise "
                              "the low-GT-margin boundary annulus (QA74 attack set, 100% of flips) "
                              "via exp(-m/temp); 0=uniform. RACED against attack-weighted, never "
                              "assumed optimal — uniform KD wastes gradient on the dark Fisher interior"},
            "--distill-form": {
                "value": form, "rung": "RACED (ddm_dw1 loss-form mini-race winner)",
                "provenance": "own-optimum law: kd_logits (Hinton dark knowledge) vs margin_field "
                              "(the field margin IS the flip-distance currency) vs argmax_ce (hard "
                              "teacher label, margin-weighted); winner selected at ITS optimum on n96"},
            "--distill-field-cache": {
                "value": str(field_cache), "rung": "MEASURED_ANCHOR (b2b field pass)",
                "provenance": "concatenated teacher distill-logit cache (P,5,384,512) f16 built by "
                              "tools/ddm_dw1_build_distill_field_cache.py over the b2b field "
                              "(ddm_b2b_qa75_field_20260730); each pair sha-verified vs the field manifest"},
        })


def lever_head_range_relax(mode: str = "linear") -> Lever:
    """Window C (ddm_dw1, MAIN charter 2026-07-30): the OFF-RGB output-chart race. 'linear' adds a
    warm-start-EQUIVALENT trainable per-channel output residual (init 0 => head == sigmoid*255 at
    ep0) that de-saturates the rgb head so gradients reach out-of-chart (dark) pixels — the direct
    test of whether the sigmoid*255 rgb chart binds (pj1 dark-reachability floor 67.95). ADVISORY-
    NON-DEPLOYABLE: a head change breaks the E1 receiver (arch tr1_lotto_combined_ema_v1); the
    slope is the decision signal for whether a receiver arch rev is worth building, NOT a deployable
    row. Falsifier: C-vs-A slope split at/below noise => the rgb chart is NOT the binding constraint
    (tokens/loss already task-space); a clear C>A split => the chart binds, receiver-rev is on."""
    if mode not in ("off", "linear"):
        raise ValueError("head_range_relax is off|linear")
    return Lever(name=f"tr1_head_range_relax_{mode}",
                 overrides={"--head-range-relax": mode},
                 notes="Window C off-RGB output-chart probe; ADVISORY-NON-DEPLOYABLE (breaks the "
                       "E1 receiver); warm-start-equivalent (init-0 residual); falsifier: C-vs-A "
                       "slope split <= noise => rgb chart does not bind")


def lever_token_quant_margin_coupling(field_custody: str, *, min_levels: int = 0) -> Lever:
    """ax1 §2a margin-coupled token quant (ddm_pa1b #793; folded from the ax1 DESIGN-stub): per-cell
    EFFECTIVE quant precision allocated by the MEASURED QA80 exact flip-distance field.  The
    segnet-fractal flip-distance law d=|m|/‖Δw‖ says quant noise is seg-safe where the field is
    slack and dangerous where it is tight (100% of realized flips are in the bottom GT-margin
    decile) — so spend precision where the field is tight, coarsen where slack.  Realized as a
    FIXED per-cell level map (non-trainable => byte-identical resume) stored in the SAME uint8
    lattice; coarse cells snap to a sublattice → SMEVR codes them with lower entropy (the bytes
    materialize through the SHIPPED coder, identical mechanism to the QA84 rowband tie).  Pool-A
    member: COMPETES with rowband + delta-sparsity for the same counted bytes (joint race only,
    never stack-claimed — non-additive-pools LAW).  Falsifier: matched-SMEVR-byte A/B no d_seg win
    ⇒ instance-close.  The allocation LAW is a rank transform of the field's own flip-mass — NO
    bare α/β constant."""
    ov = {"--token-quant-margin-coupling": "on", "--token-quant-coupling-field": str(field_custody)}
    if min_levels:
        ov["--token-quant-coupling-min-levels"] = str(min_levels)
    return Lever(
        name="tr1_token_quant_margin_coupling", overrides=ov,
        notes="ax1 §2a per-cell precision by QA80 flip-distance mass (Pool A); falsifier: "
              "matched-SMEVR-byte A/B no d_seg win => instance-close",
        constant_manifest={
            "--token-quant-coupling-field": {
                "value": str(field_custody), "rung": "MEASURED_ANCHOR (QA80 field pass, zb1 item-1)",
                "provenance": "exact flip-distance field, frozen CPU-torch SegNet over 600 burn "
                              "frame1s, 600/600 per-pair sha custody (ddm_zb1_qa80_field_20260730); "
                              "the allocation law = rank transform of the field's own flip-mass "
                              "order statistic (segnet-fractal d=|m|/||dw|| flip-distance law) — "
                              "not a bare constant, a measured field"},
            "--token-quant-coupling-min-levels": {
                "value": int(min_levels), "rung": "DERIVED (quant_levels//4 lattice-floor when 0)",
                "provenance": "coarse-floor endpoint of the per-cell level ladder; 0 => derive "
                              "base_levels//4 (a power-of-two-friendly coarse floor); base endpoint "
                              "= token_quant_levels (the config's raced level count, not asserted)"},
        })


def lever_delta_group_sparsity(weight: float, *, engage: str = "after_base_stability",
                               weight_field: str = "uniform") -> Lever:
    """ax1 §4a/§5 delta group-sparsity (ddm_pa1b #793; folded from the ax1 DESIGN-stub): group-L2
    (group-lasso) shrinkage on the per-pair token deltas.  op1 P2 measured 98.806% image-stationary
    flip mass, but the trainer has NO delta-shrinkage force — SMEVR exploits stationarity only at
    coding time; this shrinks whole-cell deltas at the SOURCE so bytes fall through the coder's
    zero-delta runs.  §7 ordering: engage AFTER the base-stability EVENT (the CE→tau knee; shrinking
    deltas against a moving base is noise) — event-driven, never epoch-hardcoded.  gc10 F2: this is
    the TRAIN-side twin of the export-side ν null-snap; ``engage='from_step_0'`` is the ν-snap
    warm-start holder (a burn warm-started from the snapped export keeps bytes low).  §5:
    ``weight_field='xi_informed'`` relaxes shrinkage on dynamic (lane/movable) cells and tightens
    on the static mass (DERIVED from the QA80 winner-class field).  Pool-A member.  Loss term only
    ⇒ NO trainable param ⇒ byte-identical resume.  Falsifier: d_seg cost at matched SMEVR bytes ⇒
    instance-close."""
    if engage not in ("after_base_stability", "from_step_0"):
        raise ValueError("engage is after_base_stability|from_step_0")
    if weight_field not in ("uniform", "xi_informed"):
        raise ValueError("weight_field is uniform|xi_informed")
    return Lever(
        name="tr1_delta_group_sparsity",
        overrides={"--token-delta-group-sparsity": "on", "--delta-sparsity-weight": str(weight),
                   "--delta-sparsity-engage": engage, "--delta-sparsity-weight-field": weight_field},
        notes="ax1 §4a group-L2 delta shrinkage (Pool A; gc10 F2 train-side twin of the ν snap); "
              "engage-after-base-stability event (§7); falsifier: d_seg cost at matched bytes => "
              "instance-close",
        constant_manifest={
            "--delta-sparsity-weight": {
                "value": float(weight), "rung": "RACED-NOT-ASSERTED (Pool-A matched-bytes race)",
                "provenance": "the shrinkage strength is a Pool-A race slot swept at matched SMEVR "
                              "bytes (v19b +0.0805 synergy precedent => measure JOINTLY, per-lever "
                              "stack-claims refused); not a constant asserted as optimal"},
            "--delta-sparsity-weight-field": {
                "value": weight_field, "rung": "DERIVED (ax1 §5 ego-motion prior)",
                "provenance": "xi_informed relax map = the QA80 winner-class dynamic fraction "
                              "(lane/movable move; static mass does not) — DERIVED from the measured "
                              "field, not a hand-drawn mask"},
        })


def lever_telemetry_v9_port(state: str = "on") -> Lever:
    """ddm_tp1 (#804) v9-line confound-cure TELEMETRY PORT (vh1 row 7; burn-4 §3.1 prereq 1).

    ``state='on'`` emits ADDITIVE read-only rows to telemetry.jsonl — per-term ``loss_terms``
    (#304), ``term_domination`` + ``term_inert`` alarms (#321), a #404 positive-control
    sentinel, and canonical ``lever_engage`` companions (Q7).  Burn-4's F1–F4 halt rules need
    these SIGNALS to EXIST (the guards are only as good as the telemetry that trips them).

    This is OBSERVABILITY, not a score-affecting lever: it is a DSL Lever ONLY so the DSL holds
    every trainer flag (never-invent-flags / config-orphan law) and so a sealed ticket can turn
    it on explicitly.  It defaults ``off`` at the TRAINER argparse for the sealed-r1c-lineage
    BYTE-IDENTITY guarantee (the flag is threaded via args, never TR1Config, and new rows go to
    tlog/JSONL only, never the checkpoint-baked telemetry_tail => trained/checkpoint bytes are
    flag-invariant).  This is the default-off-is-orphan reconciliation's SECOND case, recorded:
    score-neutral telemetry is off here NOT by orphaning but because a sealed live run demands
    trained-byte invariance — a controller/ticket turns it on on-demand.  No falsifier (read-only
    telemetry never moves S)."""
    if state not in ("off", "on"):
        raise ValueError("telemetry_v9_port state is off|on")
    return Lever(
        name=f"tr1_telemetry_v9_port_{state}",
        overrides={"--telemetry-v9-port": state},
        notes="ddm_tp1 v9 telemetry port (loss_terms #304 + term_domination/term_inert #321 + "
              "#404 positive-control + Q7 lever_engage). Observability-as-Lever: default-off for "
              "sealed-lineage byte-identity, controller/ticket turns on on-demand (read-only; "
              "no falsifier — telemetry never moves S)")


def lever_reset_operator(arm: str = "B") -> Lever:
    """ddm_bp1 (#824) BOUNDARY RESET RACE arm selector — the DSL half of the
    ``TR1ResetOperatorWiring`` charter (grade: BUILT-ELSEWHERE-UNWIRED-HERE).

    The trainer zeroes BOTH Adam moments at every window boundary (fresh ``optim.Adam``; all six
    ``save_checkpoint`` sites pass ``opt_state_flat={}``), i.e. ``tac.optimization.reset_operator``
    knobs are pinned at ``what='both', to='zero', structure='uniform'``.  The one free knob is
    ``bias_correction``, so exactly two pre-registered arms are REACHABLE without new plumbing:

      * ``'B'``  — ``ARM_B_ZERO_RESET``, ``bias_correction=False``.  This IS MLX's own
        ``optim.Adam`` default (VERIFIED from the installed signature), so the control arm trains
        BIT-IDENTICALLY to every pre-#824 run.
      * ``'Bprime'`` — ``ARM_BPRIME_BIAS_CORRECTED``, ``bias_correction=True``.  Removes the
        post-reset over-step ``eta(t)=(1-b1^t)/sqrt(1-b2^t)`` (eta(1)=3.162, max eta(12)=6.569),
        worth 1212.57 excess sign-steps = 16.168 epochs of free displacement per boundary at
        75 steps/epoch, 81.7% of it inside the first 13 epochs (MEASURED at the converged
        n=20k sum; the reset_operator docstring's ~1203/~16.0/82% is a shorter-window read of
        the same quantity).

    Arms A and C were OUT OF SCOPE for #824 by MEASURED verdict, not by preference:
    ``ResetOperatorConfig.requires_persistence`` is True for both, and the persistence plumbing was
    doubly dead — ``opt_flat`` had ONE repo-wide hit (the ``load_checkpoint`` return) that nothing
    read, and nothing wrote it.  C was a BUILD, not a port; it must not gate this race.

    **SUPERSEDED 2026-08-03 (ddm_op2 OP2-1) — arm C is now BUILT.**  The #824 scoping was correct
    *as a scoping of that race* and is preserved verbatim above; what changed is that the BUILD
    became load-bearing.  ``ddm_gd5`` §3.6 MEASURED the omission's price on a live from-scratch
    ds=32 run: the LIVE training signal jumps 1.912 -> 14.846 across a window boundary and takes
    ~17 epochs to return, against this arm's own 16.167-epoch prediction from a different channel.
    In 30-minute windows that is ~218 of a 666-epoch budget (33%) spent re-converging a
    deliberately reset optimizer — plausibly why the incumbent lineage only ever existed as a
    continuation.  ``--persist-optimizer-state on`` (args-only, DEFAULT OFF so ``config_hash``,
    ``ema_decay`` and every checkpoint stay flag-invariant) saves and restores the moments;
    MEASURED positive control: a restored boundary reproduces an uninterrupted run to max abs
    param diff **0.0**, while the reset path diverges 5.5e-2.  Arm C therefore still must not GATE
    this race — B vs B' is unchanged — but it is no longer unavailable.

    Falsifier (pre-registered): if the B' arm's boundary jump matches B's within the gate's own
    single-pixel resolution, the eta(t) impulse is NOT the mechanism behind the measured restart
    descent (R1-C: the two restarts sum 168.6% of the ep644->945 net while the 61 training
    intervals sum -68.6%; p~=0.0225 on the raw basis that matches that effect size) and the
    remaining boundary legs lead instead.  There are exactly TWO other legs, both MEASURED
    (round-2 correction): the Adam MOMENT reset itself (which bias_correction rescales but does
    not remove) and the EMA DECAY-VALUE change.  The EMA SHADOW is NOT re-anchored — the trainer
    loads it from the checkpoint (``ema = st['ema']``), so it is continuous across boundaries;
    do not cite a shadow re-anchor.  Separating those two needs arm C plus a decay-hold arm, NOT
    a bigger B' window.  A COLLAPSED jump under B' also means a boundary-state endpoint pick is
    NOT safe.

    SEAL-BLOCKING INVARIANT (round-2, MEASURED at ``train_tr1`` ``global_step = 0 if resume_from
    is None else ema_warmup_updates`` feeding ``gate_basis = 'ema_shadow' if global_step >=
    ema_warmup_updates else 'live_ema_warmup'``): a RESUMED run reports the ``ema_shadow`` basis
    from its first post-resume gate, while a FRESH run reads ``live_ema_warmup`` for its first
    U/2 updates.  **A fresh arm and a resumed arm are not read on the same instrument and their
    comparison is void.**  Both arms must be resumed from the SAME checkpoint, or both fresh;
    :func:`bp1_boundary_reset_race_program` takes a required ``resume_from`` shared by both, and
    the ticket builder refuses if the two arms' resume targets differ.
    """
    if arm not in ("B", "Bprime"):
        raise ValueError("reachable tr1 reset arms are B|Bprime; A and C require optimizer-state "
                         "persistence, which this trainer does not have (see docstring)")
    return Lever(
        name=f"tr1_reset_arm_{arm}",
        overrides={"--adam-bias-correction": "on" if arm == "Bprime" else "off"},
        notes="#824 reset-race arm selector (tac.optimization.reset_operator ARM_B_ZERO_RESET / "
              "ARM_BPRIME_BIAS_CORRECTED); on|off rather than store_true so the compiled argv "
              "carries a VALUED flag (the stray-True seal break); falsifier in the factory "
              "docstring — jump collapses => the eta(t) impulse WAS the mechanism",
        constant_manifest={
            "--adam-bias-correction": {
                "value": "on" if arm == "Bprime" else "off",
                "rung": "DERIVED (closed form, independently MEASURED against the real optimizer)",
                "provenance": "eta(t)=(1-b1^t)/sqrt(1-b2^t) at MLX Adam's default betas "
                              "[0.9, 0.999]; tac.optimization.reset_operator."
                              "effective_lr_multiplier / cumulative_excess_sign_steps. Verified "
                              "empirically: optim.Adam(lr) == optim.Adam(lr, "
                              "bias_correction=False) step-for-step, and the corrected/"
                              "uncorrected ratio equals 1/eta(t) to 6 digits "
                              "(test_ddm_bp1_boundary_reset_race.py). Not a bare switch"},
        })


def lever_boundary_probe(state: str = "on") -> Lever:
    """ddm_bp1 (#824) BOUNDARY INSTRUMENT — the positive control + fail-closed EMA-basis guard.

    OBSERVABILITY-as-Lever (the telemetry_v9_port precedent): args-only in the trainer, never
    ``TR1Config``, so trained/checkpoint bytes are flag-invariant and BOTH arms set it identically
    => it cannot confound the race.  ``'on'`` adds

      (a) a POSITIVE-CONTROL re-gate at the resume epoch BEFORE any training — the restored state
          must reproduce the parent checkpoint's last realized-gate reading to within HALF a
          single-pixel quantum (``0.5/(n_gate*384*512)``, DERIVED, not a bare tolerance).  If the
          canary is invisible the instrument is untrusted and NO #824 verdict is admissible
          (CLAUDE.md L3 verdict-clearance); and
      (b) a FAIL-CLOSED refusal when the parent and child resolved DIFFERENT ``ema_decay``.  The
          realized gate reads the EMA SHADOW, and ``derive_ema_decay`` consumes
          ``epochs*(num_pairs//batch_pairs)`` — so an ``--epochs`` change ALONE moves the
          shadow's averaging length underneath the measurement (the burn ran U=49,950/60,450/
          70,950 => a different decay at EVERY boundary; gd1: "the shadow lengthens
          166->202->236 ep").  Pair this lever with :func:`lever_ema_decay` so the basis is pinned.

    The FREE half of the instrument (per-gate interval decomposition + the ``boundary_jump`` row)
    is NOT behind this flag: it is derived from values already computed, so per the "score-neutral
    observability is not gate-able" rule it defaults ON.  Only the re-gate costs compute.
    No falsifier — read-only instrumentation never moves S.
    """
    if state not in ("off", "on"):
        raise ValueError("boundary_probe state is off|on")
    return Lever(
        name=f"tr1_boundary_probe_{state}",
        overrides={"--boundary-probe": state},
        notes="#824 boundary instrument: resume positive-control re-gate + fail-closed EMA-basis "
              "guard; identical in both arms => never a confound; read-only => no falsifier")


def bp1_boundary_reset_race_program(
    arm: str, out_dir: str, resume_from: str, *, ema_decay: float,
    epochs: int, max_wall_minutes: float, parent_levers: tuple[Lever, ...],
    gt_cache: str | None = None) -> TR1RendererProgramV1:
    """#824 arm A (``arm='B'``) vs arm B' (``arm='Bprime'``): BYTE-IDENTICAL except one flag.

    Built by COMPOSING the parent burn's own sealed levers (``parent_levers``, taken verbatim from
    the burn-4 ticket the arms resume from) with exactly three additions, all identical across the
    two arms except the first flag's value:

      1. :func:`lever_reset_operator` — the ONE flag that differs (``--adam-bias-correction``).
      2. :func:`lever_ema_decay` — pins ``--ema-decay`` EXPLICITLY so ``derive_ema_decay`` is
         bypassed on both arms (the R1-C third-reset confound: the derivation consumes
         ``--epochs`` and would move the gate's own EMA basis between arms or windows).
      3. :func:`lever_boundary_probe` — the instrument, on in both arms.

    ``lever_window`` is re-emitted from ``epochs``/``max_wall_minutes`` so the two arms also carry
    IDENTICAL geometry — belt and braces with (2): either alone fixes the basis, both together
    make the fix independent of which mechanism a reader trusts.
    """
    superseded = ("tr1_window_ep", "tr1_ema_decay", "tr1_boundary_probe", "tr1_reset_arm")
    levers = [lv for lv in parent_levers if not lv.name.startswith(superseded)]
    levers.append(lever_window(epochs, max_wall_minutes, batch_pairs=8, lr=2e-3))
    levers.append(lever_ema_decay(ema_decay))
    levers.append(lever_boundary_probe("on"))
    levers.append(lever_reset_operator(arm))
    return TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=out_dir,
                                gt_cache=gt_cache, resume_from=resume_from, full_confirm=True)


def qa24_composed_burn_program(variant: str, out_dir: str, mask_path: str, *,
                               epochs: int = 400, max_wall_minutes: float = 480.0,
                               w_rate: float = 0.05, rate_model: str = "entropy",
                               margin_temp: float = 1.0, composed_s_subset: int = 16,
                               composed_s_subset_ids: str | None = None,
                               composed_s_delta_ref: str | None = None,
                               gt_cache: str | None = None,
                               resume_from: str | None = None) -> TR1RendererProgramV1:
    """The QA24 5-piece COMPOSED seg re-burn (sg1 §3): the sealed T3 skeleton + solve_project
    init + basin-handoff on + the FIVE composed pieces (coarse-grid mask · margin-weight ·
    QAT lattice-anneal · rate-in-loss · composed-S verdict). ATOMIC: the sealed ticket depends
    on all pieces. §3.5 uses the PROVEN warp-pose6 FD-LM-GN solver (eg1/tt1); firing is gated on
    a bc1 convergence smoke (fire iff the tail-subset composed-S converges trustworthily)."""
    levers = [
        lever_variant(variant),
        lever_token_grid(16, 4),
        lever_renderer_capacity(24),
        lever_desc_level_roundtrip(16, "round"),
        lever_token_temporal("shared_base"),
        lever_seg_physics("ce", 100.0),
        lever_token_init("solve_project"),
        lever_basin_handoff("on"),
        lever_a1_gate(10),
        lever_window(epochs, max_wall_minutes, batch_pairs=8, lr=2e-3),
        # ---- the 5 composed pieces ----
        lever_token_cell_mask(mask_path),
        lever_seg_margin_weight(margin_temp),
        lever_token_quant_anneal("at_knee"),
        lever_rate_in_loss(w_rate, rate_model),
        lever_composed_s_verdict(composed_s_subset, composed_s_subset_ids,
                                 composed_s_delta_ref),
    ]
    if variant == "lotto":
        levers.append(lever_lotto(118, 0.5))
    return TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=out_dir,
                                gt_cache=gt_cache, resume_from=resume_from,
                                full_confirm=True)


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TR1RendererProgramV1:
    levers: tuple[Lever, ...]
    num_pairs: int
    out_dir: str
    seed: int = 0
    gt_cache: str | None = None
    resume_from: str | None = None
    full_confirm: bool = False

    def merged_overrides(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for lever in self.levers:
            merged.update({str(k): str(v) for k, v in lever.overrides.items()})
        return merged

    def compile_trainer_argv(self) -> list[str]:
        argv: list[str] = [TRAINER_RELPATH,
                           "--num-pairs", str(self.num_pairs),
                           "--out-dir", self.out_dir,
                           "--seed", str(self.seed)]
        for k, v in sorted(self.merged_overrides().items()):
            # store_true master switches (e.g. --lane-guard, lg1 #808): a bool True
            # override stringifies to "True"; argparse action="store_true" takes NO
            # value, so emit the bare flag ("True") / omit entirely ("False"). No
            # value-taking trainer flag uses the literal strings True/False (choices
            # are on/off etc.), so this cannot swallow a real value. Regression:
            # test_lane_guard.py compile round-trip (no stray True token).
            if v == "True":
                argv.append(k)
            elif v == "False":
                continue
            else:
                argv.extend([k, v])
        if self.gt_cache:
            argv.extend(["--gt-cache", self.gt_cache])
        if self.resume_from:
            argv.extend(["--resume-from", self.resume_from])
        if self.full_confirm:
            argv.append("--full-confirm")
        self.validate()
        return argv

    def validate(self, trainer_path: Path | None = None) -> None:
        """FAIL-CLOSED never-invent-flags: every emitted flag must exist in the
        trainer's argparse (AST scan of ``add_argument`` string literals)."""
        declared = trainer_declared_flags(trainer_path)
        emitted = set(self.merged_overrides())
        emitted |= {"--num-pairs", "--out-dir", "--seed"}
        if self.gt_cache:
            emitted.add("--gt-cache")
        if self.resume_from:
            emitted.add("--resume-from")
        if self.full_confirm:
            emitted.add("--full-confirm")
        invented = sorted(emitted - declared)
        if invented:
            raise ValueError(
                f"TR1 DSL validate FAIL-CLOSED (never-invent-flags): {invented} not "
                f"declared by {TRAINER_RELPATH} argparse; declared={sorted(declared)}")

    def sealed_ticket(self) -> dict[str, Any]:
        """The sealed DSL ticket for a governed T2 window (committed before launch)."""
        argv = self.compile_trainer_argv()
        payload = {
            "schema": "ddm_tb1_tr1_sealed_ticket.v1",
            "trainer": TRAINER_RELPATH,
            "argv": argv,
            "levers": [{"name": lv.name, "overrides": dict(lv.overrides), "notes": lv.notes}
                       for lv in self.levers],
            "score_claim": False,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["ticket_hash"] = hashlib.sha256(blob).hexdigest()
        return payload


def trainer_declared_flags(trainer_path: Path | None = None) -> set[str]:
    if trainer_path is None:
        # this module may be imported from a linked worktree OR from MAIN — resolve
        # the trainer RELATIVE TO THIS FILE's tree (shared-venv hijack guard).
        trainer_path = Path(__file__).resolve().parents[3] / TRAINER_RELPATH
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def default_t3_long_burn_program(variant: str, out_dir: str, *, epochs: int = 400,
                                 max_wall_minutes: float = 480.0,
                                 gt_cache: str | None = None,
                                 resume_from: str | None = None) -> TR1RendererProgramV1:
    """The T3 sealed long-burn skeleton (READY_TO_FIRE_UNDER_STANDING_GO — fires from
    MAIN only, never from a build arm). Event-driven schedule inside the trainer;
    resumable-from-disk; per-stage EMA-shadow checkpoints; A1 stage-exit gates."""
    levers = [
        lever_variant(variant),
        lever_token_grid(16, 4),
        lever_renderer_capacity(24),
        lever_desc_level_roundtrip(16, "round"),
        lever_token_temporal("shared_base"),
        lever_seg_physics("ce", 100.0),
        lever_a1_gate(10),
        lever_window(epochs, max_wall_minutes, batch_pairs=8, lr=2e-3),
    ]
    if variant == "lotto":
        levers.append(lever_lotto(118, 0.5))
    return TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=out_dir,
                                gt_cache=gt_cache, resume_from=resume_from,
                                full_confirm=True)


def lever_lane_guard_lambda(budget_s: float = 0.0, eta: float = 0.0,
                            step_cap: float = 0.0, lambda_max: float = 5.0) -> Lever:
    """ddm_lg1 (#808) piece 1 — the primal-dual Lane constraint (default-OFF; byte path
    preserved when the flag is absent). Holds realized Lane per-class error <= the ep641
    endpoint budget via a bounded dual multiplier updated at GATE cadence (caps-law: dual
    variables move at constraint-evaluation cadence with a capped step, never per-step).
    0.0 sentinels => DERIVED at build (tac.optimization.lane_guard: budget 0.12589 = xp1
    base_lane_S_units; eta = lambda_target/(n_gates*erosion_s) ~66.2; cap 0.1). Falsifier:
    burn-4 with the guard shows the SAME Lane-pool growth (+0.00151 S class) as the
    unprotected rung-1 continuation => the constraint form (loss-weight dual) is too weak
    on this vehicle and the #208 containment-projection lift is the successor."""
    from tac.witness_dsl.lawref import LADDER_MEASURED_ANCHOR, InputRef, LawRef

    budget_ref = LawRef(
        equation_id="dsl_custodied_scalar_identity_v1",
        inputs={"value": InputRef.literal(
            0.12589, "xp1 (#806) ep641 endpoint base_lane_S_units — "
            "/Volumes/VertigoDataTier/pact/ddm_xp1_20260731/xp1_verdict.json; ckpt sha "
            "40553db8be98215a67205d3670aa15d9b9edbe2322380ce169d8448af670f2db",
            config_tags={"vehicle": "tr1_renderer"})},
        ladder_class=LADDER_MEASURED_ANCHOR)
    return Lever(
        name="tr1_lane_guard_lambda",
        overrides={"--lane-guard": True, "--lane-guard-budget-s": str(budget_s),
                   "--lane-guard-eta": str(eta),
                   "--lane-guard-lambda-step-cap": str(step_cap),
                   "--lane-guard-lambda-max": str(lambda_max)},
        notes="lg1 piece 1: dual-ascent Lane budget constraint; realized g from the a1 "
              "gate's EXISTING argmax (zero new scorer passes); complementarity lambda*g "
              "telemetry (#549 KKTDiagnostics-aligned)",
        lawrefs={"--lane-guard-budget-s": budget_ref},
        constant_manifest={
            "--lane-guard-budget-s": {
                "value": budget_s or 0.12589,
                "rung": "MEASURED_ANCHOR (xp1 ep641 endpoint Lane per-class S)",
                "equation_id": "dsl_custodied_scalar_identity_v1",
                "provenance": "0.0 sentinel => LANE_BUDGET_S_UNITS (xp1_verdict.json "
                              "base_per_class_S_units[1]); error definition matches qa92 "
                              "_per_class_flip_counts exactly"},
            "--lane-guard-eta": {
                "value": eta or 66.2251655629139,
                "rung": "DERIVED_AT_CONFIG (derive_eta_lambda)",
                "provenance": "eta = lambda_target/(n_gates_to_engage * erosion_s) = "
                              "1.0/(10 * 0.00151); erosion_s = xp1 MEASURED unprotected "
                              "rung-1 Lane erosion (+0.00151 S)"},
        })


def lever_lane_guard_born(weight: float) -> Lever:
    """ddm_lg1 (#808) piece 2 — born-lane protection (default-OFF). Extra loss weight on the
    currently-WON Lane support (gt==Lane & realized==Lane, refreshed at gate cadence from the
    a1 gate's realized argmax), scaled by the MEASURED Lane head-sensitivity ratio 1.19607
    (mean of the four Lane-pair rank-4 head normals 4.007/3.953/3.862/3.748 over the all-pair
    mean — segnet fractal memo §2). The #725 per-channel refinement (Lane strata dominated by
    pre-head channels 2/9/6; ch9 alone 30% of Lane-Undrivable capacity) is the DEFERRED
    Fisher-anchor successor (render params do not expose scorer channels; the scalar ratio is
    the honest render-side projection)."""
    return Lever(
        name="tr1_lane_guard_born",
        overrides={"--lane-guard": True, "--lane-guard-born-weight": str(weight)},
        notes="lg1 piece 2: born-lane support protection; weight RACED (no measured "
              "own-optimum yet); sensitivity scale MEASURED (fractal memo §2 head normals)",
        constant_manifest={
            "--lane-guard-born-weight": {
                "value": weight, "rung": "RACED (no measured optimum; race at engage)",
                "provenance": "multiplied in-code by LANE_HEAD_SENSITIVITY_RATIO 1.19607 = "
                              "mean(4 Lane-pair ||dw||)/mean(10 pair ||dw||), MEASURED "
                              "segnet_recursive_fractal_factorization_20260715.md §2"},
        })


def lever_lane_guard_margin_floor(weight: float, pct: float = 10.0) -> Lever:
    """ddm_lg1 (#808) piece 3 — low-margin Lane emphasis in the head-hyperplane metric
    (default-OFF). Hinge deficit relu(1 - m/floor) on GT-Lane pixels; floor DERIVED at the
    first gate as the pct-th percentile of the run's own QA80 margin field restricted to
    GT-Lane (never a bare constant; the pct=10 default is the MEASURED '100% of realized
    flips live in the bottom GT-margin decile' law — sg1 §1.3 / ax1 pool-A header). QA80
    n600 field anchors for cross-check: q05 med 0.4302, q50 med 1.8181 (ddm_zb1 custody)."""
    return Lever(
        name="tr1_lane_guard_margin_floor",
        overrides={"--lane-guard": True, "--lane-guard-margin-floor-weight": str(weight)},
        notes="lg1 piece 3: margin floor per flip-prone Lane pixel, d=|m|/||dw|| closed-form "
              "lineage (rank-4 head); floor derived per-run at first gate (pct=10)",
        constant_manifest={
            "--lane-guard-margin-floor-weight": {
                "value": weight, "rung": "RACED (no measured optimum; race at engage)",
                "provenance": f"floor = percentile_{pct}(QA80 margin | gt==Lane) derived "
                              "in-run (derive_margin_floor); pct=10 from the MEASURED "
                              "bottom-decile flip law (sg1 §1.3); NOT a bare constant"},
        })


def lever_lane_guard_ratchet(horizon_gates: int = 0) -> Lever:
    """ddm_bs2 (#871) — the BUDGET SCHEDULE that makes piece 1 able to engage at all.

    MEASURED DEFECT (re-derived at source from the burn-4 primary telemetry, 64
    ``lane_guard`` gate rows over windows 01-03): ``lambda_lane == 0.0`` on 64/64 gates and
    ``g < 0`` on 64/64, with ``budget_s_units`` taking exactly ONE value (0.12589) while
    realized Lane descended 0.122438 -> 0.072225.  A budget pinned at the STARTING level is
    slack at gate 1 and gets monotonically slacker as the primal wins, so the multiplier is
    projected to zero forever — correct KKT on a mis-specified problem.  Concretely it
    licensed the primal to give back ALL 0.050213 S-units of won Lane before piece 1 could
    respond.  This lever replaces the constant with

        budget(t) = min( budget(t-1),  mean(last m realized) + k*sigma )

    a monotone non-increasing ratchet.  ``m`` = the dual's own integration time (matched
    bandwidth, no loop resonance); ``sigma`` MEASURED online by the trend-agnostic
    first-difference estimator (burn-4: 0.00142148 S, MAD twin 0.00146636 agreeing to 3.2%);
    ``k`` CALIBRATED so noise alone cannot move the dual by more than one ``lambda_step_cap``
    over the horizon.  The ratchet target is never an extrapolation — it is a level the run
    has already held for ``m`` gates — so the constraint is feasible by construction.  It
    also removes a second MEASURED defect: comparing a 36-of-600-pair gate estimate against
    a 600-pair-measured constant carries gd1's +3.34% Lane design error, whereas a ratchet
    compares the fixed-subset estimator only to itself and cancels the offset.

    CONTROLS (both ran; the negative one FAILED first and forced the calibrator):
      * NEGATIVE (200 stationary null trials x 64 gates at the measured sigma) — with the
        analytic k the guard engaged on 36.20% of gates, 200/200 trials: a thrash generator.
        With the calibrated k, E[max lambda] = 0.0657 <= the 0.1 step cap, 6.35% of gates.
      * POSITIVE (descent then genuine erosion) — detects +0.005 S and above; +0.002 S sits
        under the deadband and is correctly ignored.  The legacy constant budget detects
        NOTHING up to +0.050 S (1 gate of 64, lambda 0.001).
    Falsifier: a run where the ratcheted guard engages while every GT-referenced Lane cost
    is falling => the ratchet is tracking optimizer noise, not erosion, and the deadband
    calibration (not the ratchet form) is refuted."""
    return Lever(
        name="tr1_lane_guard_ratchet",
        overrides={"--lane-guard": True, "--lane-guard-ratchet": True,
                   "--lane-guard-ratchet-horizon": str(int(horizon_gates))},
        notes="bs2 #871: monotone Lane-budget ratchet; sigma MEASURED online, k CALIBRATED "
              "against the null (E[max lambda | no erosion] <= lambda_step_cap); guard also "
              "self-reports inertness when the ratchet is OFF (inertness_alarm)",
        constant_manifest={
            "--lane-guard-ratchet-horizon": {
                "value": int(horizon_gates),
                "rung": "DERIVED_AT_CONFIG (0 => self-derive max(mean_gates, gates_seen))",
                "provenance": "deadband horizon in gates; k grows ~sqrt(2 ln W), the "
                              "Bonferroni-like scaling, so a longer horizon widens the "
                              "deadband. 0 self-derives from gates seen; pass the run's "
                              "planned gate count for a stationary deadband."},
        })


def default_t1_smoke_program(variant: str, out_dir: str, *, num_pairs: int = 24,
                             epochs: int = 60, max_wall_minutes: float = 75.0,
                             gt_cache: str | None = None) -> TR1RendererProgramV1:
    """The pre-registered T1 smoke config per variant (both arms of the A2 race)."""
    levers = [
        lever_variant(variant),
        lever_token_grid(16, 4),
        lever_renderer_capacity(24),
        lever_desc_level_roundtrip(16, "round"),
        lever_token_temporal("shared_base"),
        lever_seg_physics("ce", 100.0),
        lever_a1_gate(5),
        lever_window(epochs, max_wall_minutes, batch_pairs=8, lr=2e-3),
    ]
    if variant == "lotto":
        levers.append(lever_lotto(118, 0.5))
    return TR1RendererProgramV1(levers=tuple(levers), num_pairs=num_pairs,
                                out_dir=out_dir, gt_cache=gt_cache)
