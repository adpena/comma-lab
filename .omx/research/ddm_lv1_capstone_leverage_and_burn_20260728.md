# ddm_lv1 — FINAL-CAPSTONE LEVERAGE: v9/v10+ audit × solve-init A/B × token-stack race (CAE-first) × burn prep+fire

**Date:** 2026-07-28 · **Arm:** `ddm_lv1_20260728` · **Charter:** lv1 (operator ×2 07-28: full-corpus
leverage + "anything else like LOTTO… What about CAE?") + coordinator steer (token
factorization/adaptive-quant/truncation in the fc1/oc1 stage order).
**Evidence axis:** `[macOS-CPU/MLX advisory]` throughout — `score_claim=false · promotion_eligible=false`.

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Nothing here is a score. The burn is a
MEANS to the byte-close→R6 exact row; the competitive bar is `effective_frontier 0.172` (PR130); the GOAL
bar is `min(0.15, 0.172)`.

## STORES CONSULTED (recall-first, multi-pass)

CLAUDE.md + AGENTS.md · lv1 charter · tb1 memo + sealed T3 ticket (`007d8eacf402…`, code `17166ee9c4`)
· eu1 memo (teacher-to-packet: solved object = "existence witness and initialization oracle"; G1 rails)
· ms2r_r3 receipts (C1/BOX settled controls; `stage_checkpoints/04_candidate/0.bin` 277.7M archive-form)
· v9 training target capsule (`taskspace_v9_training_target_capsule_n600_20260727/20_aggregate/`, 382MB:
seg_labels_u8 + margins + pose6) · pp1 direct-partition coder (`experiments/ddm_pp1_direct_partition_coder.py`
— the proven closed-form-KT==real-AC machinery + o4/o6/o8+prev5 templates) · cc2 coder-race receipts
(`ddm_cc2_coder_races_20260725T035900Z.done`) · op1 receipts (98.806% image-stationary flip mass; row
foveation gate 72.1% rows 160–240) · fc1/oc1 stage order + r2s auth-weighted quantize/sparse machinery
(steer-cited) · gt cache `mlx_fleet_gt_cache/gt_n600.npz` · MEMORY.md current-state.

## A — v9/v10+ LEVERAGE AUDIT (completeness pass; 25 rows, receipts grep-verified)

Tally: **CONSUMED 7 · RACED 1 · OWED 9 · SUPERSEDED 8 · NOT-FOUND 0.**
Top-3 OWED by leverage: (1) **#574 INTER-CAE × cc2 × g1 × g4 → the token-coder build** (four owed lines
converge on the ONE binding rate axis ~530KB → G4 ≤130KB) — consumed by Phase C below; (2) **#543 →
E4/WS1 exporter byte-close** (the only route to a pointer move) — named burn-alongside owed item;
(3) **#597 predict→project → solve-init tokens** — measured by Phase B below.

Full 25-row table: `[[lv1 Phase A table]]` (appended at the end of this memo, §A-TABLE).

## B — SOLVE-INIT TOKENS A/B (the headline candidate; pre-registered)

**Custody verdict (charter "verify custody paths"):** the q1-lineage exact-solve FRAMES are NOT
materializable as arrays — C1/BOX exist only as 277.7M archive-form candidates (ms2r_r3
`stage_checkpoints/04_candidate/0.bin`; the ms2rp routing card's finite-price MATERIALIZATION blocker);
the v9 target capsule holds labels/margins (score-space), not frames. The canonical MATERIALIZABLE
solution-set member in the trainer's own data path is the **GT frame itself** (d_seg ≡ 0 through R by
construction) — used as the projection source per the min-S-over-SOLUTION-SET objective.

**Three formulations, two measured dead (negative→cure adjacency; verdict_scope FORMULATION each):**
- **v1 JOINT tokens+renderer L2 pretrain → GELU-DEAD MEAN-IMAGE BASIN.** The L2 fit reaches the
  mean-image fixed point by sending every GELU pre-activation dead (gelu'==0 in fp32): pretrain l2
  frozen to 9 digits from pretrain_epoch 2; downstream scorer loop DEAD ON ARRIVAL (gnorm ~1e-9,
  ep_loss flat 164.85). Caught in-flight by the liveness instruments; killed at ~2 min. Custody:
  `b_solveinit_v1_aborted_joint_pretrain_gelu_dead_basin/`.
- **v2 TOKENS-ONLY gradient fit (renderer frozen) → GLACIAL** (measured Δl2 ~2e-5/update through the
  frozen random bank at n4) — injects ~nothing; scorer loop alive (gnorm 140) but the projection is
  empty. Micro-smoke custody `b_smoke_n4_v2/`.
- **v3 ANALYTIC chart projection (ADOPTED formulation):** tokens := area-mean downsample of GT frame_1
  at the render plane onto the lattice, split base = temporal mean (the 98.806% image-stationary static
  scene, op1) + per-frame delta residual. Deterministic, dead-basin-impossible, ~0-cost; measured
  projection stats base_absmax 0.989 / delta_rms 0.059. This is ALSO literally the stage-1
  static-base+delta factorization of the coordinator steer — one structure, two consumers.

**Pre-registered rule (sealed BEFORE results, ticket `ce5a74f312…` v2):** adopt solve-init iff
full-confirm n600 realized d_seg (EMA shadow, chunk≤120) at matched scorer-loop window end (ep39)
STRICTLY LOWER than the zero-init control **0.013833 / 534,597 B** (tb1 T2 lotto; control validity:
training-path code bit-identical to `17166ee9c4` — the HEAD diff is DUTY_TO_MEASURE metadata only,
verified). Bytes/lane-Betti-0/projection-cost are honesty axes, not verdict-changers.

**Gate trajectory (fd2 36-pair instrument; control from tb1 T2 telemetry, verified):**

| ep | 4 | 9 | 14 | 19 | 24 | 29 | 34 | 39 | full-confirm n600 | bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| zero-init (control) | .03343 | .02327 | .01940 | .01881 | .01908 | .01681 | .01564 | .01439 | **0.013833** | 534,597 |
| solve-init v3 | .02193 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

First gate: solve-init **0.02193 vs 0.03343 = −34.4% at ep4** (the projection does real work from step
zero). **RUN INCIDENTS (the reaper class, SHARPENED ×2 — three measured kills, one falsified
hypothesis, one measured survivor pattern):** the detached window was killed EXTERNALLY three times —
t_wall ~322s (ep8, sandboxed launch), ~355s (ep14, governed resume, sandboxed launch), ~286–320s
(ep17–18, governed resume, **sandbox-DISABLED launch — falsifying the sandbox hypothesis**) — all
SILENT (no error, no exit marker, no jetsam trace), all on a **~5–6-min timer from PROCESS start**,
with the PYTHON TRAINER as the launcher's DIRECT detached child. The measured survivor A/B: tb1's
25-min T2 windows on the SAME machine/launcher detached `bash tb1_t2_driver.sh` with python as the
DRIVER'S child (t2_detached manifests). **Kill #4 (t~359s, ep24 gate) then FALSIFIED the
driver-indirection hypothesis too** — the byte-identical tb1 pattern died in MY session while it
survived 2×25 min in tb1's ⇒ the reaper is SESSION-DEPENDENT and kills the whole detached tree on a
~5–6-min timer; sandbox mode and lineage are irrelevant. **The measured survivors IN this session are
harness-TRACKED shells (monitors 40+ min, background waiters 10+ min) — cure applied: the remaining
window ran as tracked `run_in_background` segments with wall-capped `--resume-from` ratchets** (and the
same form BINDS the long-burn fire from this session). Memory
`reaper_kills_sandboxed_detached_children_5min_delay_20260728.md` (v3; corrected twice in-session as
each hypothesis was falsified). Each resume ratcheted from the last intra checkpoint (P0
resumability: 4 kills cost ≤5 epochs each); Adam re-anchored fresh per #517/#518 (transient ep5 bump
7.26→15.5 then re-descent, as the law predicts). Secondary instrument note: the ep9 gate read 0.03481
on a NOT-YET-WARM resumed shadow (the #85 class, milder — projection-anchored); an instrument artifact,
not regression (ep_loss descending 7.98→5.04 across the same span); the ep14 gate was back ahead of
control (0.01822 vs 0.01940) even on the recovering shadow. The primary ep39 endpoint is unaffected
(shadow ~4τ warm by then, ≥ control's warmth).

**B VERDICT (the pre-registered rule applied AS WRITTEN): ADOPT SOLVE-INIT.** Full-confirm n600
realized d_seg at ep39 = **0.009839 vs control 0.013833 = −28.9%** (EMA shadow, chunk 32, 192 s) —
strictly better at matched scorer-loop epoch; the arm survived 3 external kills + 4 Adam re-anchor
resumes (each re-anchor is a known transient PENALTY ⇒ conservative direction). Gate trail (mixed
bases across resumes, recorded): .02193(live) → .03481(cold-shadow artifact) → .01822 → .01449 →
.01298 → .01168 → (ep34 n/a — segment boundary) → .00968. Honesty axes: **total counted bytes 706,376
vs 534,597 (+32.1%)** — the projected base+delta carries GT appearance detail the zlib temporal-delta
ledger codes poorly; the Phase-C stack (F1 factorization + margin-priced truncation + kt_prev1
entropy) is the NAMED cure, applied at the exporter on the burn's final tokens. **Lane topology channel
(the tb1 adjudication caveat): solve-init DOMINATES — realized lane Betti-0 393 (vs zero-init 164; vs
plain 264), GT-lane-components-erased 822 (vs 916/906)** — 94 more GT lane components alive: the
projection injects the appearance structure from which dashes nucleate. Advisory S-arithmetic:
Δ(100·d_seg) = −0.399 vs Δrate ≈ +0.114 (at 25/37.5M per byte, pre-coder) ⇒ net ≈ −0.29 advisory.
Scope: INSTANCE, single seed, no noise floor; [macOS-CPU/MLX advisory].

## C — LOTTO-CLASS HUNT: the token compression stack (CAE adjudicated FIRST)

Per the operator's CAE question + coordinator steer, raced in the PROVEN fc1/oc1 stage order
(FACTORIZE → DYNAMIC-QUANT → TRUNCATE → ENTROPY-CODE) on the **REAL tb1 T2 lotto payload** — the
decode-relevant description field `q = quantize(clip(base+delta), 16)`, shape (600,24,32,4), from the
sealed burn arm's final EMA checkpoint. Tool: `experiments/ddm_lv1_token_coder_race.py`; receipt:
`/Volumes/VertigoDataTier/pact/ddm_lv1_20260728/c_token_stack_race/receipt.json`. Closed-form KT rows
equal real adaptive-AC bytes to <0.01% (pp1 `roundtrip_proof` precedent); all contexts strictly causal
(symmetric decoder exists); decoder WIRING owed at the E4/WS1 exporter.

**Stage-1 FACTORIZATION (lossless, exact same decode field):**

| stream | best coder | bytes |
|---|---|---|
| F0 monolithic q | kt_prev1 | 423,407 |
| F0 monolithic q | kt_inter_cae (#574 ξ-identity ctx) | 424,624 |
| F0 monolithic q | brotli_q11_tdelta | 463,715 |
| **F1 static-base(mode) + mod-16 delta** | **kt_prev1** | **364,582 (+1,140 base)** |
| F1 delta | kt_inter_cae | 364,942 |
| F1 delta | cae_bitplane_inter (literal MPEG-4 CAE adaptation) | 368,205 |
| F1 delta | kt_o0 (≈ rANS-class order-0) | 372,641 |

- **Factorization pays −13.9% lossless** (423.4K → 365.7K incl. base); **−31% vs the 531,097 B tb1
  ledger anchor** (which coded quantize(base)+quantize(delta) separately — a COUNTED-ESTIMATE
  accounting; this race codes the field the exporter will actually ship).
- **CAE adjudication (the operator's question):** on THIS payload the ξ-advected temporal context
  (identity chart, co-located prev token) is the WHOLE signal — kt_prev1 ≈ kt_inter_cae ≈ CAE-bitplane
  within 1%, all ≈ −21% vs brotli and −14% vs spatial-context coding; the post-factorization stream is
  nearly memoryless (kt_o0 within 2.2% of the best context coder). INTER-CAE's mechanism (temporal
  template contexts) IS what pays; its spatial template adds ~nothing at D=16 lattice pitch (cells are
  16px apart — spatial correlation already absorbed by the lattice). Verdict: **CONSUMED as the
  temporal-context principle; the elaborate MPEG-4 template machinery is NOT worth its complexity on
  this payload** (INSTANCE scope, this checkpoint's tokens).
- Pools-law measured live at stage 2/3: quantization and truncation COMPETE for the same deep-margin
  redundancy pool — QT compose (134.7K) is WORSE than T1 alone (125.4K).

**Stage-2/3 LOSSY candidates (margin-slack-priced; eu1 R3 CELLBOX + rp1 slack + fc1 stages 4–5;
adopt gate = realized-flip validity, NEVER byte say-so — the fd2 lesson applied to coding):**

| variant | cells changed | F1+entropy bytes | realized d_seg (n600, EMA basis) — MEASURED |
|---|---|---|---|
| baseline q (lossless) | 0 | ~365,722 | **0.013833 — EXACT tb1 full-confirm reproduction (custody PROVEN)** |
| Q1 deep(margin>0.25)→L8 | 56.5% | ~319,184 | skipped-with-reason (same pool, byte-dominated by T1 which failed) |
| Q2 deep→L4 + mid→L8 | 78.2% | ~258,653 | skipped-with-reason (same) |
| T1 revert-to-base (|Δ|≤1 ∧ deep) | 37.2% | ~125,411 | **0.023111 — FAILS the realized gate** (+0.93 S d_seg for ~0.16 S bytes) |
| QT (Q2 + T1) | 39.9% | ~134,667 | skipped-with-reason (pools-law-dominated by T1 which failed) |

**§C-VALIDITY verdict (sandbox routing — the negative names its cures):** the uniform-lossy family
(margin-heuristic T1 here; uniform |g|-snap in §PN1-S2 — 4× against at q25) is DOMINATED at INSTANCE
scope on this checkpoint, exactly as the S2 curve predicted. **The ADOPTED C outcome is the LOSSLESS
stack: F1 factorization −31% (365.7KB incl. base) + zero-prior-byte adaptive context coding
(kt_prev1 ≈ INTER-CAE class)** — consumption point E4/WS1 exporter, re-raced on the burn's final
tokens. Named next rung: sensitivity-WEIGHTED (waterfilled) quantization — the S2 monotone decile
ordering (18.3→87.0 flips/quantum) is its allocation signal.

**Free-structure family (honesty-labeled; every row names prior art + falsifier + consumption point):**

| mechanism | honesty label | receipt / prior art | consumption point | falsifier |
|---|---|---|---|---|
| G1-LOTTO supermask (free PRNG bank, counted mask+mods+seed) | **CONSUMED — the sealed burn arm** (renderer stream 3,284 B = 6.2× under plain) | eu1 row 1 (LotteryCodec, PMLR v267); tb1 A2 adjudication | the burn itself | already adjudicated (Pareto, pre-registered) |
| Reservoir/ELM closed-form readouts | **BUILT, MEASURED-NEGATIVE on the witness vehicle (formulation scope)** — "built but not admitted as a default terminal initializer": semantic-head proxy improves while frozen-SegNet disagreement WORSENS pre-#341-polish | `frozen_segnet_gradient_elm_implementation_spec_20260712.md` + `frozen_segnet_necessity_optimality_alternatives_20260712.md` §4/§4.1 | $0 candidate: closed-form ridge solve of the tr1 head on frozen features at burn checkpoints (terminal-finisher family) | full-n600 realized d_seg not improved at equal counted bytes vs the trained head at a matched checkpoint |
| Multiresolution hash-grid token parametrization (free hash fn, counted table) | CONJECTURE (Müller et al. Instant-NGP lineage) | no in-tree receipt (NOT-FOUND, honest) | (D,c) waterfill round: race vs the D16/c4 lattice at MATCHED counted bytes | matched-bytes n600 realized d_seg not better than the lattice |
| Butterfly/monarch structured conv factorization | CONJECTURE (Dao et al. Monarch) | none in-tree | plain-arm renderer stream ONLY (races against LOTTO's 3,284 B — a high bar) | matched-bytes realized d_seg worse than LOTTO arm |
| Dithered/lattice token quantization | **HELD-NEVER-FIRED lever** (`--token-ste dither` wired at T0, raced flag; QDBS kinship = eu1 row 2) | tb1 DSL `lever_desc_level_roundtrip`; eu1 FD2-QDBS row | one bounded window on the burn baseline | round arm Pareto-dominates on (d_seg, bytes) |
| Seed-generated VQ codebooks (rule-118 free expansion from counted seed) | CONJECTURE; #461/#106 dedup lineage (steer-cited) | cc2 price-table receipts (`ddm_cc2_coder_races_20260725T035900Z.done`, `V5_TERMINAL_PROXY_WINS…RECEIVER_INTEGRATION_OWED`) | stage-1 factorization alternative at the E4/WS1 exporter (race vs F1+T1) | codebook+indices bytes > F1+T1 bytes at equal realized validity |
| Schmidhuber-consult additions (history compression / predictive coding / MDL) | ALREADY-BINDING principles, no new mechanism: the temporal-context principle IS the measured stage-4 winner (kt_prev1); train-least/Kolmogorov IS the solve-init doctrine | MEMORY train-least binding + this memo's C table | — | — |

## D — BURN PREP + FIRE

1. **Governed launcher for tr1 (the tb1-owed item): BUILT** — `tools/launch_tr1_run.py`: G1
   seal-freshness (recompile-from-DSL, REFUSES stale seals — measured refusing an out-dir drift),
   G2 import custody (shared-venv hijack guard), G3 anchor-based memory preflight (T2 measured
   12.8 GiB × 2 floor, reclaimable-aware), G4 scorer-slot ONE-n600, G5 detached+receipted
   (launch_receipt.json). Live-fired for the B v3 governed resume.
2. **Basin-entry handoff ENCODED (operator ×2 07-28 "train only to condition / solve only
   preferable always" — folded BEFORE reseal):** `basin_entry_fires` predicate in the tr1 trainer =
   the TerminalSolve §16.1 validity conditions ((a) quadratic crawl in BOTH smooth and realized
   channels, (b) lane-topology stability, (c) shadow-warm tau stage = no transitions remaining, plus
   a zero-alarm COUPLED_DESCENT window = the linearization fidelity that fd2 measured MISSING on the
   unconditioned pixel-lattice lift — conditioning creates solve-validity). Unit-tested 9/9 on the
   exact production predicate. On fire: `stage_basin_entry.npz` + `basin_handoff_receipt.json` naming
   the CONSUMED executors — `tools/quadratic_basin_finisher_probe.py` (#423 damped Newton-CG, head +
   full stages; per-pair token blocks separable given the frozen renderer) + the eg1 E3 crash-safe
   QDBS terminal rail (`cf7172e747`, all-49 closure `218ed874c7`) + #383 terminal pose — acceptance =
   v19 REALIZED joint ΔS<0 vs the handoff full-confirm baseline (which runs automatically at the
   handoff stop); #216/#475 saddle/grokking disambiguation = the recorded resume rule (stalled solve +
   still-descending training ⇒ saddle ⇒ resume, re-arm doubled window). Thresholds PROVISIONAL-derived
   (~10× separation from both the §16.1 measured crawl and the T2 measured active descent);
   rederivation trigger = the burn's own gate-delta distribution. DSL `lever_basin_handoff`; ON in the
   T3 reseal (SHORTENS the sealed wall-clock — train-least realized).
3. **Burn fire FORM (bound by the reaper v3 finding):** Monitor-supervised windows + wall-capped
   `--resume-from` ratchets from MAIN (in THIS session all detached orphans die at ~5–6 min; Monitor
   shells are the measured survivor — 45+ min). The sealed 480-min wall is consumed as resumable
   windows; continuation beyond this session = the governed launcher + ticket + resume command
   documented in §D-RECEIPT.
4. **Lane-pool race window (the tb1 adjudication's FIRST burn item; pre-registered rule applied
   AS WRITTEN):** `class_weight_lane 2.0` on the solve-init base, 25ep n600, matched-epoch rank vs
   the B-v3 ep24 reference on the fd2 36-pair instrument. MEASURED: (a) lane channel PASS — erased
   824 < 909, laneB0 288 vs 166; (b) bulk d_seg FAIL — 0.01503 > 1.05×0.01298 = 0.01363 (+15.8%).
   (a)∧(b) required ⇒ **weight 2.0 REJECTED; the burn keeps 1.0.** The pools-law cost structure is
   now MEASURED on this vehicle: lane nucleation is purchasable at a bulk-d_seg price the tolerance
   refuses. Full-confirm of the lane arm at its ep24 end: 0.014094 / 517,702 B (INSTANCE, single
   seed; comparability note: the reference ep24 came from a 3×-resumed run — asymmetry favoring the
   clean lane arm, i.e. the rejection is conservative). Weight 5.0 + finer sweep = owed race windows;
   lane Betti-0 stays a burn stage-exit facet. Custody `lane_w2_solveinit_lotto_n600/` (SSD).
5. **pn1 foldings (include-or-exclude-with-reason, before reseal):**
   - **S2+S4 fused ν measurement: CONSUMED — built + RUN this session** (`experiments/
     ddm_lv1_s2_nullspace_audit.py`, pn1 §3 protocol verbatim; probes deviation recorded: all-pairs
     sampling, strictly wider coverage than the fd2-36 restriction at equal cost). Results §PN1-S2
     below (decides the G4 token-budget feasibility + verifies-or-falsifies tb1's zero-init gauge
     claim BEFORE the burn spends its (D,c,levels) capacity, per the VOI ranking).
   - **row-2 adaptive-priors-are-FREE: CONSUMED BY CONSTRUCTION in the C race** — every KT row is a
     sequential-adaptive coder with ZERO counted prior bytes (the closed form IS the decode-time
     adaptive model; pp1 precedent); the context-MIXING upgrade (nncp-class logistic mixing over
     {prev1, spatial} predictors) is the named next rung at the exporter coder race (owed, reason:
     bounded session; the race re-runs on the burn's final tokens anyway).
   - **row-4 capacity waterfill: ELEVATED TO THE BINDING ALLOCATOR (operator "capacity should flow
     to where it buys the most")** — encoded in the reseal ticket as the allocator contract: the
     capacity split (token budget × D × c × mask density × conditioning width) is set by
     marginal-ΔS-per-byte KKT waterfill over MEASURED prices; the G3 (≤64KB renderer) and G4
     (≤130KB tokens) gates are CONSTRAINTS, not allocators. Pre-burn honesty: the initial split
     (D16/c4/w24/density 0.5) is the tb1 T2 raced Pareto winner (INSTANCE provenance), NOT
     waterfill-derived — so the ticket adds a REBALANCE EVENT class: at stage boundaries only
     (#312; basin-handoff / knee / 100ep gate-review), when measured marginal prices diverge
     ≥3× (PROVISIONAL threshold; rederivation = the first measured marginal pair), a
     capacity-rebalanced variant window RACES the incumbent at matched counted bytes (architecture
     cannot hot-swap mid-run — rebalance = stage-boundary raced windows, from-scratch where shapes
     change). Composition with ν: high ν ⇒ the waterfill drains token budget toward
     mask/conditioning (the 3,284B-of-64KB renderer imbalance is the motivating measured row).
   - **S1 Stage-A dress rehearsal: DEFERRED-WITH-REASON** — the scorer slot is the binding resource
     in this chain (B + lane + S2 + C validity + the burn's first gates); the rehearsal is
     independent of the burn config and runs at any later quiet slot (named owed item).
6. **§PN1-S2 — the ν measurement (RUN; receipt `s2_nullspace_audit/receipt.json`, SSD; verified
   line-by-line):** sensitivity map 34.9 s (frac_exact_zero **0.0**; |g| q10/q50/q90 = 1.76e-4 /
   1.23e-3 / 5.76e-3); 2,000 stratified hard-null probes, 1,230 s — **every decile realizes flips**
   (decile-0 mean 18.3 px/quantum → decile-9 mean 87.0, monotone; zero-flip fraction ≤0.005);
   null-snap curve: q25 → d_seg 0.014975 (+0.00114 = **+0.114 S for ~0.028 S of byte savings — 4×
   AGAINST**), q50 → 0.017439, q90 → 0.025575. **ν = 0.0 at the pre-registered +2e-4 tolerance —
   the G4 pivot band [0.55, 0.75] is decisively MISSED: the conditioned vehicle is ~fully
   scorer-visible per counted quantum.** Sandbox routing (the negative names its cures): (a) the
   token-LOTTO/free-nullspace lever is DEAD at INSTANCE scope (this T2 lotto tau-final checkpoint) —
   precondition-tagged, substrate-change trigger = a deeper-conditioned burn checkpoint reshaping
   |g|; never scoped wider. (b) tb1's zero-init gauge design is VERIFIED in the operative sense:
   there is no exploitable fiber for a coder to be paid from — the 530→130KB axis routes ENTIRELY
   through coding (the banked −31% lossless factorization stands; the coder race is the binding $0
   next) + RD-priced truncation with THIS curve as its prior (uniform low-|g| snapping is DOMINATED;
   sensitivity-WEIGHTED quantization is the named next rung — the monotone 18→87 decile ordering is
   its signal). (c) Waterfill consequence: NO capacity flows to nullspace exploitation; the
   allocator's ν-composition clause resolves to "token budget = coding gains + band-lemma
   re-pricing as d_seg descends." Protocol deviation recorded (all-pairs probe coverage, wider at
   equal cost). [macOS-CPU/MLX advisory]; score_claim=false.
7. Fold + RESEAL + FIRE + first gates: §D-RECEIPT below.

## Canonical-vs-unique decision per layer

- **ADOPT canonical:** tb1 trainer + DSL program module (extended, not forked) · pp1 closed-form KT
  machinery (generalized alphabet, same math) · `launch_detached_process.py` · serializer/review-gate
  · `ema_decay_run_geometry_v1` law (unchanged) · fc1/oc1 stage order (steer-bound).
- **UNIQUE (this arm):** the analytic solve-projection (v3) · the token-stack race harness
  (`ddm_lv1_token_coder_race.py`) · the tr1 governed launcher (the witness launcher hardcodes its
  trainer; principled-mismatch fork, reason recorded in its docstring).

## Observability surface

Per-layer: trainer telemetry.jsonl (projection stats, per-epoch loss/gnorm/liveness, typed gates,
alarms) · race receipt JSON (per-stream per-coder rows; per-variant cells-changed; validity rows with
basis/chunk/wall) · governed-launch receipts (gates + git head + dirty files). Decomposable: per-stream
per-coder per-variant. Diff-able: sealed ticket hashes; config_hash in every artifact. Queryable:
JSONL/JSON on SSD + this memo. Cite-able: commit shas below. Counterfactual: every lever a DSL factory;
race variants re-runnable from the same checkpoint.

## Wire-in / hooks (Catalog #125)

Sensitivity-map: N/A (no per-tensor rows; the race receipt seeds the token-stream axis). Pareto: the
C-table rows are typed (bytes, d_seg) planner-consumable points. Bit-allocator: the margin-priced
quant/truncation masks ARE the allocator primitive (consumption point: E4/WS1 exporter). Cathedral
autopilot: N/A (no paid dispatch). Continual-learning: this memo + DAG FEED. Probe-disambiguator: the
realized-flip validity stage IS the disambiguator for every lossy coding decision.

## Honest boundaries

- Nothing here is a score; no byte-closed archive exists from this arm; the E4/WS1 exporter + R6 chain
  remain the named owed route to any pointer move. Pointer UNMOVED at 0.1910828242 [contest-CPU].
- All d_seg rows are frozen-CPU-torch on macOS = advisory; single-seed, INSTANCE scope, no noise floor.
- KT rows are sequential-adaptive exact code lengths (real-AC-achievable, pp1-proven class), not yet a
  shipped decoder; container overhead not included.
- The C race payload is ONE checkpoint's tokens (T2 lotto ep39); coder rankings may shift on the burn's
  final tokens — the race harness re-runs from any checkpoint (one command).
- The B control was NOT re-run this session (tb1's T2 rows reused; justified by the verified
  training-path bit-identity to `17166ee9c4`).

<!-- SECTIONS BELOW FILLED AT WINDOW/VALIDITY/FIRE COMPLETION -->

## §A-TABLE — the 25-row leverage audit (Phase A fork, receipts grep-verified)

| # | item | verdict | receipt (verified) |
|---|---|---|---|
| 1 | v9 #500 optimal metric (categorical-Fisher law) | SUPERSEDED as standalone v9 lever; lessons consumed via the measured witness seg-form arsenal tb1 composes | `v9_missing_signal_constants_audit_20260715.md:96` ("#500 categorical-Fisher" law module); tb1 memo Design decision 2 (compose `make_loss_fn` measured forms op-for-op) |
| 2 | v9 #502 GENUINE curvelet/shearlet frames | SUPERSEDED (formulation-scoped): production default self-orient OFF; token-grid+conv vehicle carries NO Fourier/curvelet feature bank; re-entry only via matched n600 ON/OFF A/B | `curvelet_matched_bytes_ab_20260717.md:114` (`curvelet_equal_archive_transfer_v1` verdict, formulation-instance, `pointer_authorized=false`); CLAUDE.md 2026-07-27 supersession note for items 1–2; MEMORY `no_fourier_basis_DAG_FEED_20260715.md` ban-gate |
| 3 | v9 #503 recursive-fractal per-dimension representation + composition law | OWED→route: the surviving actionable form is the (D,c,levels) token-rate WATERFILL named in the T3 owed items; #503 itself holds NO_VERDICT | `per_stratum_recursive_fractal_optimal_20260721T191217Z.md:10` (**NO_VERDICT_RECEIVER_RATE_CUSTODY**) + `:36` (#503 diagnostic "not an admissible byte row"); tb1 memo T3 SEAL "the (D,c,levels) token-rate waterfill" owed |
| 4 | v9 #504 Bregman levers | OWED (never-fired lever family; no Bregman term exists in the tr1 loss) → duty ledger only; no burn-config route claimed | `bregman_all_surfaces_504_DAG_FEED_20260715.md:44` ("DSL: **OWED**, because no real trainer-consumed swept Bregman/centroid/sigma"); CLAUDE.md #351 (`GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED`, activation false) |
| 5 | v10 #539 power-diagram parametrization | CONSUMED (utility surface): trainer imports `power_diagram_witness.open_stored_npy_memmap` for GT custody; SUPERSEDED as carrier (token-grid renderer replaced the diagram parametrization) | `experiments/train_tr1_partition_renderer_mlx.py:684` (live import); `sub015_DAG…20260611.md:19854` (#539/#553 gauge fixtures line) |
| 6 | v10 #541 constructive inverse-solve | SUPERSEDED for rate (exact-plane family rate-dead); solve products survive as oracle/teacher only | `sub015_DAG…20260611.md:19706` ("#541 landed; the exact-plane family is rate-dead; preimage choice is an fp32 lever") |
| 7 | v10 #542 Cole–Hopf forms | SUPERSEDED by the realization-crux pivot (describe-line plane family rate-dead; no Cole–Hopf surface in tr1) | `SPEC_v10_capstone_cold_start_seeded_20260717.md:965` ("arm live" at spec time) superseded by `v10_description_pivot_budget_box_and_realization_crux_20260719.md` (MEMORY v10 spine: plane-family RATE-DEAD) |
| 8 | v10 #543 receiver/byte-close | OWED→burn: E4/WS1 exporter grammar wiring + numpy deploy-parity port is the NAMED tb1 T3 owed item; the only path to an exact row | tb1 memo "T3 SEAL STATUS: Named owed items… E4/WS1 exporter grammar wiring + numpy deploy-parity port (byte-close)"; `spec_v10_reconciliation_and_kkt_verify_20260719_fable.md:167` (#543 production receiver) |
| 9 | v10 #560 integer-plane spec | SUPERSEDED (same rate-dead family as #541/#560 ŷ-native line; no plane payload in tr1) | `optimal_start_card_366_refoundation_20260725.md:247` (#559/#560 ŷ-native integer-plane line listed as historical inventory); `sub015_DAG…:19706` rate-dead receipt |
| 10 | v10 #574 ξ-keyed INTER-CAE temporal coder | OWED→token-coder build (THE binding axis ~530KB): ξ-advected temporal contexts = identity-ξ in image chart per op1; XTDL1 was allocated ZERO bytes on the settled chart, so the mechanism is unconsumed | `xi_temporal_delta_coder_574_DAG_FEED_20260721T222234Z.md:24` ("allocate zero bytes to XTDL1…") + `:28` (`NO_VERDICT_RECEIVER_RATE_CUSTODY` wiring owed); tb1 memo T3 owed "entropy-coded learned-prior coder" |
| 11 | v10 #580 null compiler (ker(A) 80.67%) | CONSUMED as design force: tb1 zero-init token fields = no counted bytes on the gauge orbit by construction | `v10_seed_doctrine_closed_form_20260720.md:13` ("ker(A) 80.67% · gauge ~52%"); tb1 memo Design decision 3 (SYMMETRY force, ker(A) receives no gradient); `null_compiler_full_kernel_20260720T163500Z.json` |
| 12 | v10 #597 predict→project | OWED→lv1 Phase B (solve-init tokens IS the projection route: teacher/solve object as initialization oracle) | `v10_seed_doctrine_closed_form_20260720.md:23` ("Re-inverse-solve / predict→project (the ultimate form, operator GO) — the zero-distortion…") + `ddm_eu1…20260728.md` teacher-to-packet CONJECTURE ("existence witness and initialization oracle") |
| 13 | v10 #603 gap register G1–G8 | CONSUMED: tb1 adjudicates against the G1 rails {1e-3, 5e-4, 3e-4}, G3 renderer ≤64KB, G4 tokens ≤130KB throughout | tb1 memo T2 ("vs the G1 rails… the rails bind the LONG burn") + eu1 `EU1-G1-LOTTO-N600` gate spec (d_seg ≤1e-3 / 5e-4 / 3e-4, renderer ≤64 KiB) |
| 14 | g1 grammar (dv2/sdwl1 sentences) | OWED→token-coder build: whole-sentence causal sharing removed 452,675→(−25.5%) bytes on the typed sentence — the token stream is the new sentence to which the grammar lesson routes | `ddm_dv2_sdwl1_grammar_sentences_20260723_codex.md:56-58` ("whole-sentence causal sharing removes… 23,439 bytes (25.5041%)") |
| 15 | g2 solve-diff ledger | CONSUMED via the fd-line it seeded: fd2's `SEG_REALIZATION_GAP_AT_UINT8_DOMINANT` verdict is tb1's founding receipt (A1 gate = structural anti-fd2 instrument) | `ddm_g2_solve_diff_op_mining_canonical_equations_20260722.md:22` (E2 head/Fisher differential coordinates); tb1 memo header (fd2 verdict merged `e4bacb5d39`) |
| 16 | g3 score atlas + hard-pair registry | CONSUMED: tb1's n600 gate set IS the fd2 36-pair geometry (block 447–450 + 32 rng(0)) descending from the hard-pair registry | `ddm_g3_score_atlas_603_DAG_FEED_20260722T204813Z.md:15` ("top24/top64/control hard-pair registry"); tb1 memo A1 section (gate set at n600) |
| 17 | g4 spatial stationarity/context | OWED→token-coder build: zero-payload decoder-derived contexts are the CAE context source; boundary-gated code width $0 gate "feeds G4" | `ddm_g4_spatial_stationarity_603_DAG_FEED_20260722T212138Z.md:30` ("zero-payload aggregate-pixel and predictor-boundary context comparison"); trainer DUTY row `boundary_gated_token_code_width` ("feeds G4") |
| 18 | rd1 λ-frontier + duals | SUPERSEDED with the box retirement; blocking state unchanged: 0/162 finite same-object prices — nothing materializable to route | `codex_findings_ddm_mr1_independent_approver_merge_20260725_codex.md:89` ("RD1: 0/162 finite same-object prices"); MEMORY `box_retired_min_s_target_warp_family_closed…20260728.md` |
| 19 | dm/is1 solve line | SUPERSEDED as vehicle (BOX RETIRED); CONSUMED as oracle/lessons: solution-SET objective (is1 directive 6) is the lv1 Phase-B init doctrine; settled controls (C1 exact/BOX) are teacher custody | `ddm_is1_directive6_solution_set_objective_20260724.md`; ms2r_r3 settled controls (`stage_checkpoints/04_candidate/0.bin` 277.7M on SSD); MEMORY endgame §1-§10 |
| 20 | #700 oracle facade (of1 scorer-value oracle) | SUPERSEDED-for-tr1: tb1's A1/full-confirm compose the witness `cpu_verdict_d_seg_argmax_batch` helpers directly (bit-exact batched path); of1 facade remains the solve-line pricing surface | `codex_findings_ddm_of1_scorer_value_oracle_facade_20260724_codex.md` + `src/tac/tests/test_scorer_value_oracle.py` (exists); tb1 memo STORES CONSULTED (witness verdict helpers) |
| 21 | ct1/ev1 telemetry+joins | OWED→R6 chain: ct1 r6 rehearsal receipt is the byte-close→exact-eval rehearsal the burn's terminal export must consume; ev1 joins fed rd1 (now retired path) | `ddm_ct1_campaign_telemetry_encode_20260725T111500Z/r6_rehearsal_receipt.json` (14.2K, exists); `ddm_ev1_campaign_evidence_joins_DAG_FEED_20260724T191623Z.md:12` (V19/RD1 joins) |
| 22 | coder races #557/#558/cc2 | OWED→token-coder build: cc2 terminal verdict `V5_TERMINAL_PROXY_WINS_RACE2_RACE3_PRICE_TABLE_RECEIVER_INTEGRATION_OWED` (race2 Δaction −0.0019248, race3 −3,422 B, receiver INTEGRATION_OWED) — race the price-table winners on tb1's REAL token payload | `ddm_cc2_coder_races_20260725T035900Z.done` (verdict token verbatim; receipt sha `def7e527…`); `codex_findings_557_historical_anchor_reconciliation_20260722T044713Z_codex.md`; `autoencoder_describe_crosswalk_20260721T232351Z.md` (#558) |
| 23 | e4 Brotli packet | CONSUMED as proven bootstrap lesson + OWED→exporter: declared-dep Brotli 1.2.0 Q11 with deps `["torch","brotli"]`; enters at the E4/WS1 exporter and as a token-coder race baseline | `ddm_e4_brotli_declared_dep_DAG_FEED_20260724.md:13` ("Brotli 1.2.0 Q11 + dependencies `[\"torch\",\"brotli\"]`") + `:60` (rate recovery receipt path) |
| 24 | sn1 sided contract | CONSUMED + RACED: `class_weight_lane` IS the sn1 sided-asymmetry lever, wired in the trainer loss (per-GT-class Lane weight) and sealed in the ticket; the Lane-pool race (burn item 1) is its falsifier surface (pools law: COMPETE never stack) | sealed ticket lever `tr1_seg_ce` notes ("class_weight_lane = sn1 sided-asymmetry lever"); trainer `:759-764` (seg_pixel_w hook); tb1 memo caveat (b) Lane-pool race fires FIRST |
| 25 | at1 influence atlas | OWED (named blocker): atlas materialized but counted inert — 592 missing V19 receiver-closed joins; no burn/token-coder route claimed until joins close | `ddm_at1x_atlas_materialize_DAG_FEED_20260723.md:53` ("n600 gaze atlas | 592 missing V19 joins | counted inert") + `:55` (full-harness blocker: missing brotli in exact lock) |

## Headline summary

- Tally: **CONSUMED 7** (#539-utility, #580, #603, g2, g3, e4-lesson, sn1) · **RACED 1** (sn1's lane-pool falsifier, counted with its CONSUMED row) · **OWED 9** (#503→waterfill, #504-duty, #543→exporter, #574→token-coder, #597→solve-init, g1→token-coder, g4→token-coder, ct1→R6, cc2/#557/#558→token-coder, at1-blocked) · **SUPERSEDED 8** (#500, #502, #541, #542, #560, rd1, dm/is1-as-vehicle, #700-for-tr1).
- No silent omissions found: every charter row located a committed receipt; zero NOT-FOUND rows.
- Top-3 OWED by leverage:
  1. **#574 INTER-CAE × cc2/#557/#558 × g1 × g4 → the token-coder build** — four owed lines converge on the ONE binding rate axis (tb1 token stream ~530KB → G4 ≤130KB); cc2's price table + g4's zero-payload contexts + ξ-advected (identity-ξ) temporal contexts are the ready ingredients (= lv1 Phase C).
  2. **#543 receiver/byte-close → E4/WS1 exporter** — the only route to a pointer move (R6 exact row); every advisory result stays advisory until this lands (tb1 named owed item).
  3. **#597 predict→project → solve-init tokens** — the single highest-leverage unconsumed initialization mechanism (eu1 teacher-as-initialization-oracle receipt), measured by lv1 Phase B before the burn fires.
