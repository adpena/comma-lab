---
schema: ddm_sg1_segnet_typing_and_reburn.v1
date_utc: 2026-07-31
arm: ddm_sg1 (SegNet: QA74 residual typing + QA24 re-burn config derivation + QA75 prep)
lane_id: "lane_ddm_sg1_segnet_typing_and_reburn_20260731"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory — renderer argmax realized through the real render+R+SegNet (bit-identical to gr1's verdict); exact-solve concede from the ms2r_r3 receipt; NO paid dispatch, NO scorer promotion, NO pointer mutation]"
consumes: [ddm_ph3_realization_hybrid_adaptive_convocation_20260731 (§8 + §9 + 3 operator corrections),
  ddm_ja1_joint_atlas_waterfill_20260731, ddm_gr1_granularity_rerace_20260730, ddm_lv1_capstone_leverage_and_burn_20260728,
  ddm_tb1_renderer_build_20260728, SPEC_tr1 (spec_tr1_renderer_20260728 DSL), ddm_ms2r_r3 exact/box solve receipts,
  segnet_recursive_fractal_factorization_20260715, frozen_scorer_exact_factorization_20260715]
consumers: [QA24 burn config (MAIN atomic build + fire), QA75 solve-distillation stage, the costate SENSE surface, v4e/v5 composition]
tokens: [no-triality, p0-ledger-ok, magnitude-ok]
---

# ddm_sg1 — QA74 typed the SegNet residual · QA24 config DERIVED (not fired) · QA75 scaffolded

## §0 POINTER HONESTY FIRST (operating manual §7 — the END first)

**The exact frontier did NOT move. `0.1910828242 [contest-CPU]` is UNMOVED.** This unit produced
(a) a MEASURED typed decomposition of the renderer's d_seg residual (the QA74 $0 deliverable), (b) a
DERIVED + validated QA24 re-burn config, and (c) a QA75 solve-distillation stage scaffold. **The QA24
heavy re-burn did NOT fire this session** — see §7 for why (the binding operator corrections + MAIN's
ph3 §9 gate the fire behind a 5-piece composed config build that must be atomic, and firing the
un-composed config would violate correction-2's "never launch a weaker state" + the OPTIMAL-FORM
non-negotiable). Every number below is `[macOS-CPU advisory]`; `score_claim=false`. This is MEANS: no
byte-closed exact contest row exists from this arm.

Operating row: v4c measured S 0.992972 (seg 0.431 · pose 0.322 · rate 0.240). Renderer endpoint d_seg
**0.00388778** (n600, re-measured this arm, matches gr1's 0.00389011 within 2.3e-6).

## §1 QA74 — THE TYPED RESIDUAL (MEASURED; the pose-collapse playbook applied to seg)

**Method:** rendered all 600 pairs through gr1's exact render→R→uint8→frozen-CPU-SegNet path
(`experiments/ddm_sg1_residual_typing.py`, committed 6648db9a6e; chunk 120; n600 mean d_seg 0.00388778
reproduces gr1 → the argmax path is validated). Per-pixel realized argmax vs GT `lstars`; typed by
{GT class · GT-margin depth · spatial stationarity · token cell · renderer-vs-EXACT-solve}.

**Reference correction applied (operator 07-31 teacher correction):** the decisive "solve" column uses
the **EXACT C1 solve** (17,927 err = d_seg 1.52e-4; r6cal SHA-bound settled control), NOT the box solve
(136,839 = 0.00116). The exact solve is a compress-time TEACHER (never ships; strict-scorer bars only
DECODE-time loads). Its per-class concede floor is read directly from the ms2r_r3
`scorer_measurement.json` `q1_stratum_errors` receipt — no 409MB inflate needed.

### §1.1 THE DECISIVE COLUMN — ≥96% amortization gap (the headline)

| | renderer errors | exact-solve concede | amortization gap (attack) | gap % | renderer err-rate in class |
|---|---:|---:|---:|---:|---:|
| **Lane** | **177,631** | 2,556 | **175,075** | 98.6% | **25.72%** |
| Road | 141,985 | 7,833 | 134,152 | 94.5% | 0.52% |
| Movable | 69,158 | 1,346 | 67,812 | 98.1% | 4.74% |
| Undrivable | 58,907 | 3,622 | 55,285 | 93.9% | 0.10% |
| MyCar | 10,940 | 2,570 | 8,370 | 76.5% | 0.037% |
| **TOTAL** | **458,621** | **17,927** | **≥440,694** | **≥96.1%** | — |

**≥96.1% of the renderer's residual is AMORTIZATION GAP (attackable); ≤3.9% is the exact-solve
realization floor (concede).** Amortization gap = renderer d_seg 0.00389 / exact-solve d_seg 1.52e-4 =
**25.58×** (the operator's corrected ~25.6×, NOT the box-solve 3.35×). The 96.1% is a **LOWER bound** on
the attack fraction (containment caveat: the per-class subtraction assumes exact-solve errors ⊆ renderer
errors; if any concede pixel is a renderer-correct pixel, the true attack fraction is higher). Per-pixel
overlap is the owed refinement (needs the exact-solve argmax = a 409MB inflate); the aggregate is
decisive and the direction robust. **Schmidhuber's frame confirmed at 25.6×: SegNet's remaining descent
is CLOSING THE AMORTIZATION GAP, not more bytes (measured saturated) or more capacity.**

### §1.2 LANE VERDICT (Tao's question — MEASURED-answered)

**Lane is RENDERER-REACH-limited, NOT SegNet-stride-limited.** renderer Lane errors 177,631 vs
exact-solve Lane concede 2,556 = **69.5×**. The 77%-skip stride-2 stem limit is present in BOTH scorers,
but the exact solve proves Lane is 98.6% attackable → QA24/distillation CAN cure it (renderer reach), it
is NOT conceded to the SegNet resolution floor. **Lane is THE priority:** it is the single largest error
class (38.7% of all flips) at a **25.7% error rate within Lane pixels** (vs Road 0.52%) — a 50× error
concentration. Movable (thin/small, 4.7% rate) is the sister thin-structure problem. The bulk classes
(Road/Undrivable/MyCar, <0.5% rate) are already well-handled.

> **CORRECTION to ph3 §8's cited distribution.** §8 quoted "flips ~50% Road / 19% Lane / 13% Undrivable"
> (an ancestor-vehicle characterization). The MEASURED tr1-renderer residual is **Lane-DOMINATED**:
> Lane 38.7% · Road 31.0% · Movable 15.1% · Undrivable 12.8% · MyCar 2.4%. The burn must prioritize Lane,
> not Road.

### §1.3 THE BOUNDARY-ANNULUS FINDING (the decisive form signal)

**100.0% of the 458,621 renderer flips sit in the BOTTOM GT-margin decile** (margin < 3.94; flip-margin
p10/p50/p90 = 0.030 / 0.190 / 0.594). The renderer NEVER flips a confidently-classified GT pixel; the
residual lives ENTIRELY on the codim-1 separatrix (the small-margin boundary annulus). **This is the
single most important form signal in this arm** — it says the loss should concentrate its gradient on
the boundary annulus (margin-weighting), not spend it on the 99.6% deep-interior pixels that never flip.
The burn currently does the opposite (§4).

### §1.4 SPATIAL STATIONARITY + CELLS (the QA24 grid aim)

- **Row band:** 80.0% of flip mass in SegNet rows 160-240 (the horizon band; matches op1's 72.1%
  foveation). Hood (rows 290+): only 1.01% — MyCar/hood is well-handled (IoU 0.994 confirmed).
- **Pixel-recurrence:** 0 pixels flip in ≥50% of pairs (max recurrence 22.8%). The flip SET is transient
  at the pixel level (content-driven) but concentrated in a STATIONARY image REGION (the horizon band).
  Reconciles op1's "98.8% image-stationary" = the REGION is stationary, individual pixels are not → the
  per-cell capacity must handle temporal variation (the delta), not just a static base.
- **Cells (grid 24×32, downsample 16 → cell = SegNet (r//16, c//16), exact alignment):** the top
  flip-mass cells cluster at grid rows 10-14 (SegNet 160-240), cols 3-27. **The 384 cells that gr1's
  cell_drop50 KEEPS (top-|g|-sum half) carry 99.61% of the total flip mass** — the amortization gap
  concentrates exactly on the cells the coarse grid protects.

## §2 QA24 GRID — DERIVED + VALIDATED (the coarse-from-birth keep-set)

The from-birth coarse grid = **keep the 384 cells cell_drop50 keeps** (top-half |g|-sum, gr1's ranking),
**drop the low-|g| 384 from birth.** Validation this arm:
- kept-384 flip-mass fraction = **0.9961** (the dropped 384 cells carry 0.39% of flips → seg-free to drop).
- |g|-rank vs flip-mass-rank top-384 Jaccard = 0.631 (moderate; the |g| set nonetheless captures 99.6%
  of flip mass because every high-flip cell is high-|g|).
- Kept cells = **grid rows 5-19** (SegNet 80-320: the drivable/horizon band), fully dense rows 10-18;
  **dropped = sky (grid rows 0-4) + lower-hood (rows 20-23)**. Artifacts: `qa24_grid_keep_mask_50.npy`,
  `qa24_combined_priority.npy` on SSD `ddm_sg1_20260731/`.

> **POSE CAVEAT (co9 bidirectional law — binding on the config, §7 QA77-lite cures it):** dropping
> sky+hood FREEZES far-field content, which the co9 law prices on the POSE axis (Knee-A: sky/hood freezes
> cost pose). The QA24 grid therefore MUST run with QA77-lite composed-S stage verdicts (ph3 §9.2) so the
> burn's stage decisions see the pose cost, not just seg — otherwise the grid re-commits the exact Knee-A
> externality gr1 flagged. This is why MAIN's §9.2 is CONFIRMED into the frozen config (§7).

**Falsifier (Contrarian, binding):** QA24's from-birth claim rests on gr1's post-hoc −0.098 bound. If the
re-burn endpoint d_seg ≥ cell_drop50's 0.004310 at matched bytes → coarse-from-birth closes at INSTANCE,
solve-distillation (QA75) becomes the lead rung.

## §3 QA24 CONFIG — THE 5 COMPOSED PIECES (derived; provenance + falsifier each)

The QA24 burn is NOT a re-run of the T3 ticket. Per the three operator corrections + MAIN ph3 §9, the
config composes FIVE pieces beyond the sealed T3 skeleton (variant=lotto · D16/c4 · w24 · L16/round ·
shared_base · ce→tau · solve_project · basin-handoff on · 400ep / 480min). Each below names the exact
trainer surface it touches, its provenance, and its falsifier. **The pieces are ATOMIC — the sealed
ticket + fire depend on all five; MAIN builds them as one composed config (§7).**

### §3.1 COARSE-GRID FROM BIRTH (the QA24 essence — NEW trainer feature)
- **Surface:** `build_module` token init (zero the 384 inactive cells' delta from birth) + `render_frame`
  (inactive cells contribute 0) + the byte-close (exclude inactive cells from SMEVR coding). NOT a config
  tweak: `lever_token_grid` only supports uniform D∈{8,16}; the derived grid is a SELECTIVE 384-cell mask.
- **New DSL lever needed:** `lever_token_cell_mask(mask_path)` → `--token-cell-mask <npy>` (the
  `qa24_grid_keep_mask_50.npy` derived in §2). Fail-closed validate as usual.
- **Provenance:** gr1 cell_drop50 (−0.098 post-hoc, n600 byte-closed) + this arm's 99.61% flip-mass
  coverage of the kept set.
- **Falsifier:** §2 above.

### §3.2 MARGIN-WEIGHTED LOSS (the #1 form fix — correction 2)
- **MEASURED motivation:** §1.3 — 100% of flips at small GT-margin, yet the burn loss is built with
  `margin_weighted=False` (train_tr1 line 776). The loss spends its budget on the 99.6% never-flipping
  deep-interior pixels. The boundary-reweight lever family (`margin_weighted` / `focal_gamma` /
  `fisher_density_weight`) is ENTIRELY default-off in the burn.
- **Surface:** reconstruct `make_loss_fn(..., margin_weighted=True, margin_weight_temp=τ)` in the tr1
  trainer (the witness make_loss_fn already implements it — no new loss code, just wire the flag).
- **New DSL lever:** `lever_seg_margin_weight(temp)` → `--margin-weighted-loss` + `--margin-weight-temp`.
- **Provenance:** this arm's boundary-annulus finding + CLAUDE.md witness-line "margin/UNIWARD" lever +
  the make_loss_fn focal/Fisher-density family (documented "prefer one at a time; the A/B decides").
- **Falsifier (RACE, don't adopt):** margin-weighted burn arm vs uniform arm; if margin-weighted endpoint
  d_seg ≥ uniform at matched epoch → margin-weighting does not help THIS renderer, record + close
  (INSTANCE). NOTE: distinct from `class_weight_lane=2.0` which lv1 §D.4 REJECTED (class-weighting
  inflates one class at a bulk price; margin-weighting targets the boundary annulus across ALL classes —
  a different lever that may pay where class-weighting failed).

### §3.3 QAT-DYNAMICS AS SCHED EVENTS (correction 3 — DERIVED, not copied)
The transferable PR95 QAT DYNAMICS (never the recipe), mapped to our lattice:
- **(a) staged quantizer engagement / LATTICE ANNEALING.** Currently the token STE quantizer (L16) is on
  from birth (`quantized_tokens`). PR95's dynamic: smooth optimization finds the basin FIRST, the
  quantizer engages MID-training, refinement continues ON-lattice. **Our derivation (possibly original —
  "lattice annealing"):** anneal the token lattice fine→L16 as a sched1 EVENT at the CE→tau knee (our ms2
  tolerance-homotopy lineage), OR engage the STE at the knee (float tokens before, quantized after). This
  lets the renderer find the basin in float, then refine on the shipped lattice. Falsifier: annealed-STE
  endpoint d_seg ≥ from-birth-STE at matched bytes → no basin benefit, close.
- **(b) REDISTRIBUTION → coder-friendly fields (RATE co-benefit).** Training through a quantizer clumps
  the learned distribution at lattice levels = lower token entropy = fewer coded bytes. This is the #110
  latent-structure-regularizer reborn AND it COMPOSES with §3.4 rate-in-loss (the explicit form of the
  same effect). Falsifier: measure token entropy (SMEVR bytes) at matched d_seg with vs without the
  in-loop quantizer/rate term; if no byte reduction → redistribution is not paying, drop.
- **(c) sigma-noise = quantization simulation.** ALREADY PARTIAL via the STE + the dither field
  (`token_ste=dither`). Derivation: the dither STE IS the sigma-noise quantization simulation. No new
  lever; note that `--token-ste dither` is the raced form of this dynamic (HELD-never-fired per lv1).
  Falsifier: dither vs round arm on (d_seg, bytes) — the standing raced flag.

### §3.4 RATE-IN-LOSS (MAIN ph3 §9.1 — stl1 row-8 LAW, first application to the renderer burn)
- **CONFIRMED into the frozen config.** The burn currently optimizes DISTORTION ONLY; rate is 97.8% of the
  archive and every prior post-hoc coding loss (the 3× white-field losers) hit exactly this seam. Add a
  token code-length/entropy surrogate term to the loss (a simple per-token entropy model over the SMEVR
  contexts, or a differentiable entropy estimate of the quantized token histogram).
- **Surface:** the tr1 loss (add `w_rate · Ĥ(quantized_tokens)` to the seg loss); new DSL lever
  `lever_rate_in_loss(w_rate)` → `--w-rate` + `--rate-model {entropy,smevr_surrogate}`.
- **Composition:** COMPOSES with §3.3(b) — rate-in-loss is the explicit form of the redistribution
  co-benefit. Race the two arms if cheap; else rate-in-loss ON as default per the law (MAIN's steer).
- **Falsifier:** rate-in-loss arm vs distortion-only at matched d_seg; if archive bytes not lower → the
  law does not bind this payload (record), fall back to post-hoc coding (the gr1/lv1 stack).

### §3.5 QA77-LITE COMPOSED-S STAGE VERDICTS (MAIN ph3 §9.2 — kills the Knee-A externality)
- **CONFIRMED into the frozen config (REQUIRED by §2's pose caveat).** At sched1 events / stage exits,
  run the CHEAP terminal solves (6-dim GN/pair pose + 2-param photometric, bounded subset) so stage
  decisions + endpoint acceptance see COMPOSED S, not raw seg. VERDICT-level only — do NOT differentiate
  through it (that is v6/FULL). ~free.
- **Surface:** extend `realized_gate` / the stage-exit predicate to compute composed S = 100·d_seg +
  √(10·d_pose_solved) + rate on the gate subset (reuse the tt1 twin analytic gradient for the pose solve
  per §6). New DSL lever `lever_composed_s_verdict(subset)` → `--composed-s-gate-subset`.
- **Provenance:** co9 bidirectional pose law + gr1's pose caveat + this arm's grid dropping sky/hood.
- **Falsifier:** if composed-S never diverges from seg-only at the stage exits (the grid's pose cost is
  negligible) → the lite verdict is free insurance that changed no decision (record; keep it — it is the
  correct instrument regardless).

## §4 FORM AUDIT (correction 2 — the un-taken iteration levers, per OPTIMAL-FORM)

Correction 2 ("our renderer was never really fully iterated"): the 0.00389 plateau is **LIFTED-TRAINER
form**, not a capacity/target wall (the amortization gap 25.6× + the exact solve's 1.52e-4 prove the
target is far below). The NSCS06 v6→v7 precedent (one cargo-cult-unwind = 44%) applies. The audit:

| lever | current | finding | action |
|---|---|---|---|
| **margin-weighting** | OFF | **#1 cargo-cult**: 100% flips at small margin, loss uniform | FOLD into burn (§3.2), RACE |
| focal_gamma / fisher-density | OFF | sister boundary-reweighters (the make_loss family) | candidate 2nd-arm; one-at-a-time |
| rate-in-loss | OFF | distortion-only; rate is 97.8% (§3.4) | FOLD (§3.4) |
| activation | GELU | lv1 v1 saw GELU-dead basin (L2 pretrain); conv renderer std | UN-TAKEN — own A/B (risky), post-burn |
| capacity taper (w24) | uniform | Lane/thin-structure is the problem; capacity is not boundary-routed | UN-TAKEN — boundary-capacity-routing, post-burn |
| class_weight_lane | 1.0 | lv1 §D.4 REJECTED 2.0 (bulk price) | keep 1.0; margin-weight is the correct thin-structure lever |
| solve_project init | ON | already the lv1 v3 GT-frame projection | keep (composes with the coarse grid) |
| token STE | round | dither = the sigma-noise/QAT dynamic (§3.3c), held-never-fired | RACE dither vs round |

**Un-taken iteration levers for the post-burn round (listed, not folded — each needs its own A/B):**
activation (GELU → step-native/other), boundary-capacity-routing (renderer width concentrated on the
horizon band per §1.4), focal/Fisher-density boundary reweighters, thin-lane dedicated carrier (cb1
lineage) or #149 pre-R sub-pixel Lane placement (Lane is 38.7% of flips — the highest-leverage single
structural lever, but a v5-structural change, not a burn fold).

## §5 STAGE 3 — QA75 SOLVE-DISTILLATION STAGE (scaffold; research_only, blocker named)

The solve-distillation finishing stage (pn1-S5 revived): finish the renderer to MATCH the exact-solve on
the QA74 attack set (§1.1), instead of fighting argmax-CE against GT. Sidesteps the CE-vs-argmax wall.
**Scaffold committed** as a stage config (`ddm_sg1_qa75_solve_distill_stage.json`) — the loss + the
attack-set target set + the MATERIALIZATION BLOCKER + the falsifier. **HONEST BLOCKER (research_only):**
the exact-solve FRAMES/labels are NOT materialized as arrays (409MB archive-form; lv1's materialization
blocker). QA75 as a full solve-distillation is BLOCKED on inflating the exact-solve archive → per-pixel
target labels/frames. The scaffold names this as the owed step. The MATERIALIZABLE proxy (the GT-frame
projection = the existing `solve_project`) is already the init; the distinct QA75 value (the solve's
R-REALIZABLE representation on the 17,927-floor-excluded attack pixels) requires the inflate. **Falsifier
(§8):** distilled endpoint ≤ CE endpoint at matched budget → the gap is not target-limited (distillation
is the mechanism); distilled ≈ CE → the gap is optimization-limited (the burn form fixes are the mechanism).

## §6 tt1 CONSUMED (MAIN ph3 §9.3) — pose scope REDUCED

tt1 FIRED-POSITIVE (realized gradient-TTO −0.0630 ΔS in 13.6 min on frozen v4c). Consequences folded:
(a) the burn does NOT over-invest in pose polish — seg descent + §3.5 composed-S verdicts suffice; TTO is
the standing v4e TERMINAL stage and finishes pose post-hoc at −0.95 S/hr. (b) The post-burn re-solve
chain adopts the tt1 twin analytic gradient (`experiments/ddm_tt1_twin.py`) instead of coarse-FD GN —
same solve, strictly better convergence, and it is the pose solver for §3.5's composed-S verdicts too.

## §7 STAGE 2 DISPOSITION — DERIVED + STAGED; the burn did NOT fire (honest)

**The QA24 burn did NOT fire this session, and the pointer is UNMOVED.** Stated plainly per the
means/ends firewall. Why (composing the STANDING GO with the binding corrections per operating manual §1):

1. The STANDING GO was granted on ph3 §8's QA24. **Three operator corrections + MAIN's ph3 §9 then
   arrived (all binding, all "before the config freezes / re-derive before launch")** and each ADDS a
   genuine piece to the config: coarse-grid mask (new trainer feature), margin-weighting (form fix),
   QAT-dynamics events, rate-in-loss (new loss term), composed-S verdicts (new verdict path). Correction 2
   is explicit: "never launch a weaker state."
2. **Firing the un-composed T3 config would be a re-run, not QA24** (no coarse grid, no form fixes) — a
   weaker state, which correction 2 forbids, and the "dispatch-at-lifted-form" trap the OPTIMAL-FORM
   non-negotiable extincts.
3. **The 5 pieces are ATOMIC** (the sealed ticket + byte-close depend on all five) and touch the
   resume-sensitive trainer + DSL across multiple surfaces. Half-wiring them under time pressure violates
   the P0 "never half-wire under resume risk" non-negotiable. The honest composition: DERIVE all five
   precisely (§3, done) + hand MAIN the atomic build + fire path, rather than fire a rushed/faked config.

**MAIN fire path (the governed chain is READY):** the launcher `tools/launch_tr1_run.py` is built and
real (G1 DSL-recompile hash + G2 hijack guard + G3 memory preflight 12.8 GiB×2 + G4 scorer slot + G5
detached receipt; `--dry-run` validates all gates without firing). The build TODO, in order:
1. Build the 3 new DSL levers (§3.1 cell-mask, §3.2 margin-weight, §3.4 rate-in-loss) + wire their flags
   into the tr1 trainer argparse + consumers; build the §3.1 coarse-grid mask feature (build_module +
   render + byte-close) + §3.5 composed-S verdict path (realized_gate extension, reusing ddm_tt1_twin).
   Adopt §3.3 QAT dynamics as sched events. Each with its unit test + falsifier pre-registered.
2. Seal a NEW `ddm_tb1_tr1_sealed_ticket.v1` via `TR1RendererProgramV1(...).sealed_ticket()` with the
   composed levers + solve_project + basin-handoff on + the derived grid mask.
3. `launch_tr1_run.py --ticket <new> --out-dir <SSD> --dry-run` → confirm all 5 gates PASS.
4. Fire from MAIN (the reaper kills in-session detached children at ~5-6 min per lv1 §B; the burn runs as
   MAIN's Monitor-supervised wall-capped `--resume-from` ratchets, P0 resumable).

**MAIN's ph3 §9 confirm (asked):** items 1 (rate-in-loss) AND 2 (QA77-lite composed-S) BOTH make the
frozen config — §3.4 and §3.5. §9's selector-in-loop / support-gating / FULL bilevel are v6-design and
are NOT scoped into the burn (no scope-creep).

## §8 INVALIDATION BUDGET (charter honesty) — the post-burn re-solve chain (MAIN's next charter)

The QA24 burn (when fired) invalidates the pose/photometric/selector streams (it is a from-scratch base;
the ja1 order-of-operations DAG: token_base change → invalidates pose+photo+selector). The proven
post-burn re-solve chain, budgeted (measured wall-clock from the v4b/v4c receipts):
- pose re-solve (full-600) ~3.5 h — **now adopts the tt1 twin analytic gradient** (§6; strictly better
  than coarse-FD GN, measured on the already-solved 250).
- photometric re-fit ~35 min.
- gate + byte-close + composed-S accept.
The capacity pool runs PARALLEL to v4d per ja1 (not competing for the byte budget). The §3.5 composed-S
verdicts price the sky/hood-freeze pose cost DURING the burn, so the post-burn re-solve is a refinement,
not a rescue.

## §9 TRIALITY / verdict scope / STORES CONSULTED

- **DAG:** this memo + the SSD receipts (`ddm_sg1_20260731/`: sg1_typing_receipt.json, qa24 grid masks,
  argmax chunks) + the DAG FEED. **DSL/equations:** research-only; the QA24 config levers are DERIVED
  specs (§3), un-built → the DSL is NOT yet touched (no drift: the derivation is the design, the build is
  MAIN's atomic landing). `[no-triality]` for this memo (typing + derivation arm; no canonical-equation
  or DSL surface changed yet).
- **verdict_scope: FORMULATION/INSTANCE** — the typed residual is MEASURED on the tr1 renderer endpoint
  (this vehicle, n600, single realization); the ≥96% amortization gap + Lane 69.5× + boundary-annulus
  100% are robust MEASURED facts; the grid is validated; the 5 config pieces are DERIVED hypotheses each
  with a pre-registered falsifier (RACE, not adopt).
- **#404:** advisory; the typing RANKS the burn aim, the burn's own gates VERIFY.
- **STORES CONSULTED:** CLAUDE.md; AGENTS.md; docs/operating_manual_craft_handoff.md; ph3 §8+§9 + the 3
  operator corrections + MAIN ph3 §9 message; ja1 table (QA24 capacity pool, parallel); gr1 (cell_drop50
  −0.098, the grid source); lv1 (solve-init A/B ADOPTED = solve_project; the reaper; the materialization
  blocker); tb1 + SPEC_tr1 DSL (the burn config + governed launcher); ms2r_r3 exact/box solve receipts
  (the concede floor); segnet_recursive_fractal (rank-4 head + margins); the shared-venv-hijack memory
  (fixed this arm: eg1 worktree had editable-installed tac into the shared venv — restored to main src).

## §10 APPARATUS NOTE (surfaced to MAIN, not silently fixed)

On arm start, `import tac` resolved to `.omx/tmp/codex_worktrees/ddm_eg1_endgame_chain_20260729T000937Z/
src` (the eg1 arm editable-installed tac from ITS worktree into the SHARED `.venv` — the
shared-venv-hijack class). Restored to main src via `uv pip install -e . --no-deps` (verified
`tac.__file__` → `/Users/adpena/Projects/pact/src`). Already-running processes (v4d PID 92494) unaffected
(their modules loaded at start). **This hazard persisted ~1 day and would have made any custody-sensitive
dispatch from the shared venv run stale eg1 src** — the OWED two-landing guard (a preflight/spawn-time
scan of `.venv/**/*.pth` for `codex_worktrees` targets) is still un-built; MAIN should land it.
