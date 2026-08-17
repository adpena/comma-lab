# sub015 DAG FEED — exponential-linear witness warm-start

UTC: 2026-07-14T17:28:43Z  
Lane: `lane_exp_linear_reparam_warmstart_20260714`  
Contract: `PAPER_WARM_START_FROM_DIVERGENCE`  
`research_only=true`

## Executable dependency graph

```text
[paper/source custody]
          |
          v
[paper-vs-witness divergence fork]
          |
          v
[actual ep650 checkpoint + real GT + feature-state hashes]
          |
          v
[anchored mismatch inverse: exact W/rate/d_seg identity] ---- MEASURED
          |
          v
[deterministic 2x2 MLX: AdamW, Muon, SEL+AdamW, SEL+Muon] -- BLOCKED: Metal custody
          |
          v
[CPU/numpy through-R batch32 verdict at each boundary]
          |
          +---------------------+
          |                     |
          v                     v
[matched-d_seg step gate]  [int8+Brotli + distribution gate]
          |                     |
          +----------+----------+
                     v
        [SEL-over-Muon additive/redundant verdict]
                     |
                     v
    [typed DSL + canonical equation legs after drain] ------ HELD
                     |
                     v
       [resume/fold/parse-back receiver closure]
```

## Node state

| Node | State | Receipt / blocker |
|---|---|---|
| paper/source custody | complete | arXiv `2607.09967` |
| divergence fork | complete | findings memo |
| real artifact custody | complete | checkpoint, GT, and feature-state SHA-256 in common-start JSON |
| anchored conversion | complete | exact effective identity, exact 62,087-byte identity, exact through-R d_seg identity |
| probe implementation | complete | `tools/probe_exp_linear_reparam_warmstart_mlx.py` plus unit tests |
| 2×2 execution | blocked | headless MLX cannot load a Metal device, including MLX CPU selection |
| matched-step verdict | waiting | requires all four traces |
| terminal rate verdict | waiting | requires all four traces |
| DSL/equation implementation | held | contested shared trees; V9 provenance owner retains custody |
| promotion / heavy launch | forbidden | advisory containment and no measured landing |

## Preregistered transitions

1. Execute/resume all four arms to step 24 with CPU/numpy batch32 d_seg every two steps.
2. Require that each non-SEL control's terminal d_seg strictly improves from the common start, then set that terminal value as its matched target.
3. Record first step at or below target. `fewer_steps` requires a strict step reduction and equal-or-better terminal d_seg.
4. Classify `ADDITIVE_ON_THIS_SMOKE` only when SEL+Muon passes step 3 over Muon.
5. Compare exact int8+Brotli bytes and magnitude statistics at each arm's first matched-basin row; retain fixed-step terminal deltas as secondary telemetry.
6. If both fewer-step and rate/objective admission gates pass, route the held typed lever to the provenance owner after the shared-tree drain.

## Canonical consumers after a measured anchor

- Sensitivity map: treatment effect on matched d_seg step count and exact blob bytes.
- Pareto surface: `(steps_to_target, d_seg, blob_bytes)` with axis and formulation token.
- Bit allocator: terminal magnitude/entropy delta by parameter group.
- Cathedral/autopilot: admit only a typed stage-boundary action with resume and fold receipts.
- Continual learning: append the scoped contrast, including a negative or blocker token; do not infer across formulation or hardware axes.
- Probe disambiguator: keep `fixed` and `annealed` scale-LR schedules as explicit modes if the fixed treatment is negative.

No consumer is wired from the present blocker because doing so would turn an unmeasured prior into false authority.

## Triality / pointer delta

- DSL leg: spec present, implementation **HELD**.
- DAG leg: this file.
- Equation leg: `ΔW≈-ηJ_fJ_fᵀ∇_W L`, with local metric `(J_fJ_fᵀ)⁻¹`; canonical code **HELD**.
- Pointer delta: **NONE**.

## FEED-islandbirth (2026-07-15) — island-birth is NOT a visible saddle-node in $0 frozen data; DERIVED per-class birth-weight ∝ (P/A)_c

`[macOS advisory] NON-PROMOTABLE` — pointer 0.19108 UNMOVED (means). Memo:
`.omx/research/island_birth_saddle_node_hysteresis_measurement_20260715.md`.

SIGNAL (source memory `curriculum_is_continuation_instabilities_are_bifurcations_20260714` + Fork Dynamics
item-5): is witness island-birth a saddle-node with hysteresis → compute λ_c to set the curriculum's birth-λ?
DIAGNOSTIC ($0, read-only): occupancy order parameter (Lane/Movable **islands/pair** = connected components,
+ area% + presence%) measured on frozen render-through-R argmax maps vs `gt_n96`.
- **Continuation (curriculum stages CE299→Tau599→l7725→Muon900/925, n96): SMOOTH, no fold.** Lane islands/pair
  flat ~16 (GT 21.3); Movable saturated at CE ~3.0 (GT 3.1); Lane area monotone 0.349→0.522% (GT 0.589).
  Both classes 100% pair-presence at every stage — no global birth event.
- **Cross-config (mod32cap n16, perclass_baseline n600): all cluster at one sub-GT plateau** (Lane ~15–16,
  Movable ~2.6–3.0). The "mod32cap ZERO islands" (L2/L3) is NOT reproduced at component level — present-but-
  deficient, not absent. No absent↔present branch pair ⇒ no hysteresis loop in frozen data.
RESPONSE (honest): **hysteresis UNMEASURABLE at $0 by construction** — it's a training-flow property; a static
frozen-checkpoint λ-sweep is algebraically reversible f(λ) → would be a surrogate FAKE (NO-FAKE class 8).
Only measured bistability hint = #300 seed-absorption (n=2, subcritical-threshold-consistent, not a loop).
Normal-form fit: too coarse / monotone → λ_c NOT extractable. **DERIVED reduced-order lever:** birth-balance
group `λ_c ≡ W_birth/(δ·(P/A)_c)`; MEASURED GT `(P/A)_Lane/(P/A)_Movable = 0.760/0.086 = 8.9` ⇒ per-class
birth-weight `∝ (P/A)_c` → **Lane ≈ 8.9× Movable** (isoperimetric drain; cruder 1/A_GT gives 2.6×). Absolute
λ_c/δ needs the flow sweep. NEXT (operator-GO, not $0): resume EMA-BEST, quasi-static W_birth UP/DOWN ramp,
log islands/pair per epoch (n96) → measured λ_c ± hysteresis margin, δ = W_birth*/(P/A)_c at the up-fold.
Triality: memo (DAG leg) + lever `W_birth,c ∝ (P/A)_c` owed as a DSL Lever (register + duty-to-measure).
saddle-node **NOT CONFIRMED** (frozen smooth; hysteresis unmeasurable $0) · concrete lever = per-class
birth-weight ∝ (P/A)_c, **Lane ≈ 8.9× Movable**, MEASURED GT geometry n96.

## FEED-fractal503 (2026-07-15) [triality: DAG leg here · eqs N/A composes fullstack_unique_home_assignment_v1 · DSL N/A design-only]
Task #503 RECURSIVE-FRACTAL-OPTIMAL representation (DESIGN + $0 ranking, pointer UNMOVED 0.19108/0.18804-bank, NO launch). Memo: recursive_fractal_optimal_representation_design_503_20260715.md. SYNTHESIS over #392(P1-P12)/#398(clause-A/B home map)/#502(curvelet GO)/island-birth. ONE recursion: **store the GENERATOR at each dim, DERIVE the finer dim as evaluation/warp/residual, store nothing a coarser generator+warp produces** — the fractal-nesting composition law = the double-count cure (same ξ warps FRAME intercept + glues PAIR + IS d_pose, stored once read thrice; boundary=tie-locus of CLASS generators, never stored; pixel=argmax readout). Per-dim optimal reps: PIXEL=argmax readout (naive task-low-rank REFUTED, palette rank 15/15 codex) · CLASS=Laguerre per-class carriers store-generators (#284/v8, Movable 6289B MEASURED) · BOUNDARY=oriented curvelet annulus (#502 GO 1.7-2.0× capacity, 41× anisotropic n600 MEASURED) · FRAME=keyframe+ξ-warp (horizon-poly+ξ 4.7KB=0.0032S, 88× below naive MEASURED) · PAIR=se(3) screw ξ (banked d_pose 0.001610 L68) · EPOCH=continuation + per-class birth-weight ∝(P/A)_c (Lane 8.9× Movable DERIVED). **#1 BUILD = BOUNDARY oriented curvelet frame (#502)**: gap is 100% d_seg (L68), 26.8%(n600) flip mass in 4.7% oriented annulus (L66), oriented=1.7-2.0× more rate-efficient there — the only fresh $0 MEASURED GO on the exact pointer-blocking quantity (realized d_seg through-R = the build, OWED). GAPS: EPOCH λ_c/δ + the composed launch = live-run/operator-GO; CLASS decoupling + CHROMA legibility + curvelet-vs-shearlet family = $0-probe-owed. MEANS — pointer moves only through byte-closed n600 exact row.

## FEED-20260817d — FOURTH POINTER MOVE (rr4) + the determinization wave + the micro-edit ENGINE

1. **POINTER MOVED (largest micro-era move): S 0.15959729295498598 → 0.15853325034789678
   @ 181,161 B [contest-CUDA T4 n600]**, archive sha 35ac2b9b…, ΔS −0.0010640426070892, −1,598 B.
   All 3 pre-registered rr4 falsifiers hit EXACTLY (seg 0.00029611 == base, pose 6.88e-06 == base,
   S to 17 digits). Mechanism sealed: rr2's −1,598 B HPAC-context token recode was REAL; the S 27.83
   refusal was STAGING INFIDELITY (fired bytes ≠ proved bytes), NOT device-scoped probabilities —
   the HPAC student is an integer lattice, device-exact (identical corrected_quantized_logit sha on
   T4-CUDA and macOS-CPU). Anchor via posterior_update_locked; sub-0.15 gap 0.00853, pose term
   ≈0.0083 of it. Memo: ddm_rr4_t4_verdict_pointer_move_20260817.md.
2. **Determinization wave (operator ×2 "no more by-hand")**: tools/fire_modal_auth_eval.py
   (65e15db4e9) = the ONLY Modal fire path (sanitize→local validators→computed shas→phantom-claim
   close→fixed template→auto-armed detached poller→manifest); E1–E11 error ledger
   ddm_er1_error_class_ledger_and_determinization_20260817.md; memory
   hand_assembled_dispatch_is_the_error_factory_20260817.
3. **oq1 orphan drain**: 437/437 dispositioned; ZERO datable rows postdate the 08-06 PR130 intake —
   the backlog predates the live vehicle, LIVE-FRONTIER survivors ≈ 0 (3-arm concordant).
4. **tc1 TR1 verdict (charter falsifier FIRED)**: TR1's binding term is its realized d_seg floor —
   a ZERO-byte archive still scores 2.49× the frontier; class routes to the live vehicle (F4).
   Transfer products: free-archive test · solve_project antagonism (−28.9% d_seg, +32/53% B) ·
   dw1-scoped distill negative · EMA-warmup contamination (decay .997 @1upd/ep ⇒ warmup ep~1318) ·
   62/116 off-flag census.
5. **LIVE**: ddm_rv2 frontier adversarial review ROUND 1 (operator-ordered; rr4 custody ·
   recode-operator composability · bank-recompile · next-mover ranking w/ #1089/#1091
   verify-at-source) + ddm_me1 MICRO-EDIT ENGINE (operator 08-17 "build and or train a tool to
   explore and optimize and apply all micro edits that lower score" — generate/LOPO-rank/evaluate/
   compose-under-joint-remeasure/recompile-vs-live-coder/emit; realized acceptance only).
   hg1 arm_b hinge endpoint ~23:20Z (MAIN adjudicates §8).
6. **Codex wall re-verified STANDING** (operator believed lifted; direct smoke refused with reset
   Aug 20 08:32) — routing stays Opus/MAIN.

STORES CONSULTED: MODAL_REMOTE_RESULT.json (rr4 row) · ddm_rr4_cuda_prob_reencode_20260817.md §6/§8 ·
continual_learning posterior (hv1 precedent row) · ddm_oq1_drain_dispositions_20260817.json ·
tc1 memos 99ad5cb5d5/7763583f6d · er1 ledger · me1 charter.

## FEED-20260817e — rv2 review round 1: row SOUND, 10 findings, the exact routing arithmetic

1. **The rr4 frontier row SURVIVED adversarial round 1** (ddm_rv2, 813153ca1d): S re-derived to
   17 digits, archive independently hashed (35ac2b9b…, single stored member 1a6b40cc…), decode
   falsifier resolved in favor; all 10 findings are custody/narration/apparatus. Counter 0/3.
2. **FO-1 EXECUTED by MAIN ($0, before the ~24h TTL)**: all 8 discarded eval artifacts recovered
   from the Modal result cache (contest_auth_eval.json 28,937 B · report.txt 664 B + 6 more,
   shas in RECOVERY_MANIFEST.json; SSD mirror in endpoint_closure/returned_artifacts_fo1) —
   the frontier row is now --contest-final-eligible. Root cause F1: modal_endpoint_close.py
   drops str-typed artifacts (measure-and-discard class, payload-law violation) — FO-2
   two-landing fix owed.
3. **F3 correction applied**: "bank needs recompile" = registered falsified premise (instance) —
   qs2/re1 already inside the row via mc36; me1's first job voided + re-aimed at source-verify
   then the three-way prior race (recompile leg = identity baseline).
4. **F5 spend correction**: Modal $18.62/$20, headroom $1.38 ≈ 8 T4 rows (MAIN's ~$7 was 2.7×
   low); operator cap ruling OWED; CPU-axis row (~$0.40 ≈ 29% of headroom) held for adjudication
   w/ the PR135/138 GPU-eval field-norm fact.
5. **THE EXACT ROUTING ARITHMETIC (first time exact at this base)**: pose→0 saves 0.00829458 <
   gap 0.00853325 (short 2.4e-4) — pose ALONE cannot close sub-0.15; seg needs −28.8% of d_seg;
   rate needs −12,815 B (archive ≤168,346 B); pose marginal 6.03× seg. Every prior route memo
   used the stale gap — re-derive before allocating.
6. **Next-movers ranked from receipts**: hg1 arm_b hinge (live, ~23:20Z; m_safe derived 0.03918
   vs default 1.0 = 25.5× too big, 97.65% of hinge gradient wasted — the case is STRONGER than
   its own memo claimed) · ce1/cw1 aligned objective (EF3000 −2,286 flips below init, first
   descent in 10 runs; S reach unmeasured, needs byte-closed repack) · ra2+ra1 ~278 B
   ΔS −1.851e-4 $0 (gate measured vacuous — retire + fire). tc1 ratified over oq1 on
   boundary_gated_token_code_width (DECLARED-ONLY, instance).

STORES CONSULTED: ddm_rv2_frontier_adversarial_review_r1_20260817.md · MODAL_REMOTE_RESULT.json ·
RECOVERY_MANIFEST.json · bu1 registration rows · billing authority tool output (per rv2 F5).

## FEED-20260817f — THE GESTALT NAMED (operator-prompted synthesis)

The two-week arc synthesized (ddm_gestalt_operator_algebra_synthesis_20260817.md): **the contest
has resolved into an OPERATOR ALGEBRA on a frozen semantic object** — every pointer move was an
operator (SELECT/EDIT/RECODE), never descent; the field converged independently (PR138 opal =
our rr4 in different clothes); zero-counted-byte mechanisms dominate (rule-118 lived); no single
axis closes (pose→0 short 2.4e-4, exact); models rank / reality accepts; correction propagation
(not discovery) is the binding apparatus constraint. Grokking read: plateau→representation-change
→4 moves in 4 days. Pre-registered prediction (falsifiable): next rate win = opal-class online
prior (me1 leg iii); next seg win = retrain-FOR-operators (aligned objective/editability); pose =
joint/nonlinear only. If a plain longer burn or static coder race moves the pointer next, the
gestalt is WRONG in a named way. 23rd convocation spawned to adversarially test + extend.
