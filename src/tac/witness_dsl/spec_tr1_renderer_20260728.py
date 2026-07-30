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
    if downsample not in (8, 16):
        raise ValueError("grid downsample raced over {8,16} (D=12 excluded: 512/12 "
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
                         "value": rate_model, "rung": "RACED (QA86a)",
                         "provenance": "sg1 §3.4 skipped 'race-if-cheap'; entropy=marginal surrogate "
                                       "(coder-mismatched), smevr_surrogate=temporal-delta (matches "
                                       "the shipped SMEVR event/value split). Burn-2 A/B."},
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
