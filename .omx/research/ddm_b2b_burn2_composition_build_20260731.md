---
schema: ddm_b2b_burn2_composition_build.v1
date_utc: 2026-07-31
arm: ddm_b2b (burn-2 composition BUILD — QA86 config corrections + QA83 head + QA84 grammar + QA75/QA80 harness, during the QA24 burn window)
lane_id: "lane_ddm_b2b_burn2_composition_build_20260731"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory — scorer-free BUILD: pure-numpy + r7 coder + MLX-CPU tiny forwards; NO SegNet/PoseNet run, NO Metal, NO paid dispatch, NO pointer mutation; the QA24 burn (pid 68621) untouched]"
consumes: [ddm_gd1_generic_default_census_20260731 (QA83/QA84/QA86 seeds, T4/T5/T6/T8/T19 rows),
  ddm_b2p_burn2_prepay_20260731 (QA75 frames+loader, QA80 producer, QA81 typed blocker),
  ddm_bc1_qa24_compose_and_fire_20260731 (the sealed 5-piece ticket + burn receipt),
  ddm_ph3_realization_hybrid_adaptive_convocation_20260731 (§10 the menu),
  spec_tr1_renderer_20260728 (the tr1 DSL SoT), experiments/train_tr1_partition_renderer_mlx.py (the burning trainer),
  experiments/ddm_r7_token_coder.py (the shipped SMEVR coder), tac.canonical_equations.evaluators.eval_ema_decay_run_geometry (the EMA LawRef),
  tac.witness_dsl.qa75_solve_frame_targets (b2p loader), src/tac/boundary_math/margin_budget_field.py (b2p QA80 exact-field), src/tac/boundary_math/seg_core.py (frozen CPU SegNet forward)]
consumers: [MAIN post-burn boundary (compose + fire burn-2 immediately), QA24 burn-2, v4e/v5 grammar, the costate organ duty queue]
tokens: [p0-ledger-ok]
---

# ddm_b2b — burn-2 composition BUILD: QA86 + QA83 + QA84 + QA75/QA80 harness, all scorer-free, tested, committed

## §0 POINTER HONESTY FIRST (means/ends firewall)

**The exact frontier did NOT move. `0.1910828242 [contest-CPU]` is UNMOVED.** This unit is a BUILD:
it moves NO trained byte and runs NO scorer. Its entire value is that burn-2 warm-starts with the
config corrections, the factorized head, the variable-cell grammar, and the two owed distill-field
harnesses ALL BUILT + TESTED, so MAIN composes + fires burn-2 at the post-burn boundary without a
build stall. Every number below is `[macOS-CPU advisory]`; `score_claim=false`. The QA24 seg re-burn
(pid 68621) was untouched (never read/wrote its out-dir; no scorer/Metal compute while it held the
slot). 40 tests pass; ruff-F clean; 4 commits landed (f28e427dd9 · e8d531e735 · 4bdd72a2f7 · d138df0c00).

## §1 PER-DELIVERABLE STATUS (the report MAIN needs)

| # | deliverable | status | surface | commit |
|---|---|---|---|---|
| 1 | **QA86 config corrections** | **BUILT + TESTED** | trainer EMA + SMEVR ledger; DSL levers + burn-2 race/resume programs + provenance rungs | f28e427dd9 |
| 2 | **QA83 factorized output head** | **BUILT + TESTED** | trainer head variant (rgb/class_field/class_field_photo) + DSL lever + head race config | f28e427dd9 |
| 3 | **QA84 variable-cell grammar** | **BUILT + TESTED** | rowband grammar module + differentiable tie + SMEVR byte-close + trainer integration + DSL lever + grammar race | e8d531e735 |
| 4 | **QA75/QA80 distill harness** | **BUILT + STUB-SMOKED** (real scorer post-burn) | `tools/ddm_b2b_segnet_field_pass.py` (injectable scorer) | 4bdd72a2f7 |
| — | **triality legs** (MAIN drift routing) | **SETTLED** | flip-mass canonical equation + QA84 lever constant_refs; config-shaped levers = [consumers-generic] | d138df0c00 |

### §1.1 QA86 (census T4/T5/T6/T8/T19) — the config-corrections bundle

- **(a) rate-surrogate race** — the SMEVR-matched surrogate (`smevr_surrogate`, temporal-delta hist) was
  already BUILT in the trainer; the sg1 §3.4 skipped A/B is now a DSL program:
  `spec_tr1_burn2.qa86_rate_surrogate_race_programs` → `{A_entropy, B_smevr_surrogate}` at matched budget,
  SMEVR byte ledger on both. Falsifier: smevr arm's realized SMEVR bytes not lower at matched d_seg → keep entropy.
- **(b) SMEVR byte ledger (T5 FIX)** — `counted_bytes_ledger` now prices the token stream with the SHIPPED
  r7 SMEVR coder (`--byte-ledger-coder smevr` default), not the zlib temporal-delta surrogate. Decisions now
  MATCH the archive. Both prices recorded per gate for decomposable observability (`tokens_bytes_zlib` +
  `tokens_bytes_smevr` + `token_ledger_coder`), NEVER summed into `total_counted_bytes`. Fail-open: r7
  unavailable → zlib fallback (never crashes a gate). `zlib` opt-in kept for a byte-continuous live resume.
  MEASURED: on a fake token field SMEVR 951 B < zlib 1136 B; drops shrink both.
- **(c) EMA clamp (T6 FIX)** — `derive_ema_decay` guard is now RUN-LENGTH-DERIVED (`d <= 1 - 2/U`, phi=1
  warmup-fills-run ceiling), never a constant. The old `[0.9, 0.9995]` tiny-smoke clamp UPPER cap bound
  over the derived 0.99986667 at U=30,000 (collapsing warmup 15,038→4,000, violating the phi=0.5 design).
  MEASURED: U=30,000 → **0.99986667** (unclamped, phi=0.5 warmup=15,000); at every scale warmup=U/2 exactly.
  **MID-RUN fix variant BUILT:** `spec_tr1_burn2.qa86_mid_run_resume_program(...ema_decay=0.99986667)` resumes
  from a stage checkpoint with the EXPLICIT corrected decay + SMEVR ledger. TRADEOFF surfaced: the live
  shadow was warmed under 0.9995; switching mid-run slows forward averaging. **MAIN fires only on operator
  GO;** a byte-continuous resume declines it via `--ema-decay 0.9995 --byte-ledger-coder zlib`. It changes NO
  trained/shipped byte (EMA is the inference shadow only).
- **(d) provenance rungs** — `--w-rate` now DERIVED-ESTIMATE: `(25/37,545,489) * n/8 ≈ **0.0768**` for the
  burn geometry (n=923,136 counted tokens) — the live **0.05 is ~65% of the S-commensurate value**;
  `spec_tr1_burn2.derive_w_rate_exchange_rate` + the QA86a rate A/B measure it. `--margin-weight-temp`
  = RACED-NOT-ASSERTED (Fisher-form derived, scale bare → burn-2 sweep {0.3,1,3}). `--ema-decay` =
  DERIVED (ema_decay_run_geometry_v1). No bare constant remains on these levers.

### §1.2 QA83 (census §4.1) — output-space factorization head

The renderer head output space is now a SELECTABLE variant (`--renderer-head-mode`): `rgb` (3-ch control =
current burn, BIT-IDENTICAL to `sigmoid*255` — resume-safe), `class_field` (k=1 class scalar → FIXED
monotone gray lift, the 1-luma-channel ur-instance), `class_field_photo` (k=2: class + margin-slack-confined
luma photometric channel, conservative fixed gain 0.05 = ~13/255; the exact per-pixel band-lemma budget from
QA80's field is the named scorer refinement). The lift is rule-118-FREE decoder code; only the k-channel
token field is counted → the k=1 head strictly reduces `renderer_bytes` (MEASURED). Race config:
`spec_tr1_burn2.qa83_head_race_programs` → `{A_rgb, B_class_field, C_class_field_photo}`, SMEVR ledger, byte-
matching a burn-2 tuning step (§4.1 form is c4→c2 code_width). Falsifier: B/C endpoint d_seg no better than A
at matched bytes → factorized-output closes at INSTANCE + the v14 static-dict negative extends to trained forms.
`ph3_s10` stubs NOT folded (my flags are new levers, not the exact stub trainer flags; b2p §5 discipline).

### §1.3 QA84 (census §4.2) — variable-cell row-band grammar

`RowBandGrammar` (`src/tac/witness_dsl/qa84_rowband_grammar_20260731.py`): a D8 fine base whose BULK rows are
TIED in 2×2 blocks (D16-effective) while the op1 flip-band rows (render 160-240) stay FREE at D8 — a
foveation of the scorer-geometry-optimal D∝(flip-density)^(−α) field (the separable approximation). The tie
is a deterministic GATHER (differentiable; bulk representatives learn, the rest get no gradient), backend-
agnostic (numpy + MLX share ONE representative-index map, BIT-IDENTICAL parity MEASURED). BYTE-CLOSE reuses
the SHIPPED SMEVR coder: the tied field's identical bulk blocks code as ~free zero-delta runs — the savings
materialize through the real coder + ~130 B band-spec side-info. DOF (default grammar): rowband **1248** vs
uniform-fine (D8) 3072 vs uniform-coarse (D16) 768. Trainer integration: `--token-rowband-spec` (requires
`--grid-downsample 8`); `raw_tokens` + `_full_token_field_np` apply the SAME tie so render + byte-close agree;
ledger counts `rowband_spec_bytes` in the total. Race: `spec_tr1_burn2.qa84_grammar_race_programs` →
`{A_uniform_D16_drop50, B_rowband_D8}` + quadtree NAMED further arm (pays iff in-band azimuthal sparsity real,
QA74 g4). gr1 nested-rungs DOMINATED is INSTANCE-scoped (solved-token post-hoc; from-birth uncovered). Raster
wire order UNCHANGED (QA85 Hilbert receipt stands). Falsifier: no matched-bytes d_seg win → uniformity
survives at INSTANCE; row-band ≥ quadtree → the separable approximation suffices.

### §1.4 QA75/QA80 — the SegNet field-pass harness (READY-TO-RUN, post-burn)

`tools/ddm_b2b_segnet_field_pass.py`: ONE SegNet forward per frame (logits→top2→argmax/runner/margin) emits
either/both fields. **QA75**: the LOGIT/MARGIN distill field over the materialized EXACT-solve frames (b2p
`SolveFrameTargets`) — soft targets encoding the boundary annulus with feasible margins. **QA80**: the EXACT
per-pixel flip-distance `d=|m|/‖Δw‖` over burn frames — needs the RUNNER-UP class (2nd-argmax) the gt cache
lacks, so a SegNet pass; feeds via `exact_flip_distance_field` (b2p producer). The scorer is INJECTABLE: the
real run loads the frozen CPU-torch SegNet (NEVER MPS); the SMOKE uses a deterministic stub → the plumbing
(loader → derived ≤120/chunk law → field compute → manifest+sha) is validated with NO Metal / NO real SegNet
/ NO n600 pass (the scorer pass is POST-BURN, per the task). 7 plumbing tests pass; deterministic sha proven.

## §2 THE BURN-2 COMPOSE CHECKLIST (for MAIN at the boundary)

All programs are `TR1RendererProgramV1` (validate() = never-invent-flags fail-closed); import from
`tac.witness_dsl.spec_tr1_burn2_20260731`. **Choose the burn-2 spine, then optionally fork a race:**

1. **Corrected base (default aim):** `burn2_corrected_base_program(variant, out_dir, mask_path,
   use_derived_w_rate=<False|True>, gt_cache=..., resume_from=<burn-1 endpoint EMA ckpt or None>)` →
   QA24 5-piece + SMEVR ledger + (fixed derive → EMA 0.99986667 automatically). `use_derived_w_rate=True`
   swaps 0.05→0.0768. Seal its ticket, `launch_tr1_run.py --dry-run` (all gates), fire under standing GO.
2. **QA86a rate A/B:** `qa86_rate_surrogate_race_programs(...)` → fire A_entropy + B_smevr_surrogate at
   matched budget; verdict = realized n600 d_seg + real SMEVR archive bytes.
3. **QA83 head A/B/C:** `qa83_head_race_programs(...)` → fire A_rgb + B_class_field + C_class_field_photo;
   tune code_width to match total_counted_bytes; verdict = realized d_seg + SMEVR bytes.
4. **QA84 grammar A/B:** `qa84_grammar_race_programs(...)` → A_uniform_D16_drop50 + B_rowband_D8; B needs
   `--grid-downsample 8` (already in the lever); verdict = realized d_seg + SMEVR bytes + rowband_spec_bytes.
5. **QA86c MID-RUN resume (operator GO only):** `qa86_mid_run_resume_program(..., resume_from=<live stage
   ckpt>, ema_decay=0.99986667)` — corrects the EMA clamp on the LIVE burn at a stage boundary. Surface the
   forward-averaging tradeoff to the operator before firing.
6. **QA75 distill precompute (post-burn scorer slot):** `python tools/ddm_b2b_segnet_field_pass.py
   --frame-source qa75_solve --frames-root /Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa75_solve_frames
   --out-dir <SSD> --field-kind distill_logit_margin` (real scorer). Feeds the burn-2 distill stage (Qa75 stub
   lever → real trainer distill flag = the OWED wiring). **QA80 exact field:** same tool, `--frame-source
   qa80_burn --frames-npy <burn frame1s> --field-kind exact_flip_distance` → the class_field_photo slack budget.

## §3 OWED / NOT DONE (honest)

- **The 4 races are BUILT, not MEASURED.** No d_seg/byte row exists until MAIN fires them n600 + byte-closes
  (means, not the end). The pointer stays 0.1910828242 until a byte-closed `evaluate.py` row lands.
- **QA75/QA80 real scorer pass is POST-BURN** (deliberately — no Metal while the burn holds the slot). The
  harness is ready; the n600 field is a named post-burn call.
- **`ph3_s10` stubs NOT folded** — my work built DATA/HEAD/GRAMMAR/harness surfaces, none of which is the exact
  stub trainer flag (distill loss / photometric loss / carrier composite). The stubs fold when burn-2 lands
  those exact flags (b2p §5 fold-and-delete contract honored).
- **Canonical-equation registry `populate` + `__init__` export** of `rowband_flip_mass_foveation_band_v1` is a
  named landing/operator follow-up — the module DEFINES + self-validates the equation (touching the leg); the
  locked-registry append was kept out of this arm to avoid mutating shared state during the sister burn.
- **QA81 stays BLOCKED** on the parallel-session WIP (`direct_description_carrier_compose.py`), untouched here.

## §4 TRIALITY / verdict scope / STORES CONSULTED

- **DAG:** this memo + a DAG FEED appended to the canonical
  `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`. **DSL:** 6 new/enhanced Lever factories in
  `spec_tr1_renderer_20260728` (head, byte-ledger-coder, ema-decay, token-rowband + w_rate/margin_temp
  provenance) + a burn-2 PROGRAM module `spec_tr1_burn2_20260731` (5 race/resume/base factories) — the DSL
  HOLDS every designed lever. **equations:** `rowband_flip_mass_foveation_band_v1` registered (measured anchor
  custodies the 72.1% flip mass); the EMA 0.99986667 points at the existing `ema_decay_run_geometry_v1`. The
  mid-flight build commits' legs are settled BY this landing; pure-build commits carried `[no-triality]`.
- **verdict_scope:** all rows are BUILD/DERIVED facts (byte-close arithmetic, shape/parity/roundtrip, ledger
  totals) + one MEASURED lossless byte comparison (SMEVR<zlib on a fake field). The 4 races are PRE-REGISTERED
  (unmeasured); each carries a pre-registered falsifier. No score/promotion/pointer claim.
- **STORES CONSULTED:** CLAUDE.md (NO-FAKE, THE GOAL, OPTIMAL-FORM, value-provenance ladder, DSL-holds-every-
  lever, never-invent-flags, never-launch-weaker-state, review-gate, serializer post-edit shas, SSD certify-or-
  block, no-old-lineage); docs/operating_manual_craft_handoff; census §2/§4/§6 (T4/T5/T6/T8/T19/§4.1/§4.2);
  b2p §2/§3/§4 (QA75 loader, QA80 producer, QA81 blocker); bc1 §7 (the three burn-2 headlines) + the sealed
  ticket; ph3 §10 menu; the r7 SMEVR coder + ema_decay LawRef + seg_core SegNet forward + margin_budget_field;
  MEMORY (generic-triple law, pools law, verdict-scope ladder, canonical class order, opportunity-pools).

## §5 APPARATUS NOTES (surfaced to MAIN)

- **Burn watch:** pid 68621 alive throughout (never touched). All my compute was pure-numpy / r7-CPU / MLX-CPU
  tiny forwards — no Metal, no scorer, no paid dispatch.
- **Serializer self-collision:** my own step-N checkpoints (`files_touched`) tripped the Catalog #340 sister
  guard on my own commits; used the documented paired-env override with a rationale each time (the flagged
  sister was always `ddm_b2b`). Not a real cross-agent collision.
- **Backward compatibility:** every new trainer flag defaults to CURRENT behavior (head=rgb, no rowband,
  smevr ledger, derived EMA). A live-burn resume with the sealed ticket is unchanged EXCEPT the EMA derive now
  yields 0.99986667 (the T6 fix) — pass `--ema-decay 0.9995` to reproduce the old clamped value.
