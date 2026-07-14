# CONSOLIDATION — drain → cherry-pick → land (no loss/no trample) → prune to ONE SoT → permanent drift-gate (2026-07-14)

**Operator directive (verbatim):** "Let the queue drain and queue all follow ons but don't launch
and then you must cherry pick and land all signal on main no signal loss or trampling and then prune
all so we have a single source of truth and permanently prevent drift."

This ledger is the EXECUTION CHECKLIST fired when the drain-detector reports 0 RUNNING. Durable so it
survives compaction. NO signal loss, NO trampling, ONE source of truth, permanent drift prevention.

## Current state (2026-07-14T17:0x)
- **132-file uncommitted shared-worktree pile** (+7974/-1856) from concurrent codex arms — the drift.
- Live arms (drain first, do NOT kill): `genuine_curvelet_shearlet_build_measure`,
  `ripo_deep_warmstart_trust_region_500`, `warmstart_gauge_symmetry_homotopy`,
  `governor_measured_growth_fix`, `exp_linear_reparam_warmstart`, `witness_train_sweep_spec`.
- Dead + held_entangled (their slices intermingled in the pile): `provenance_canonicalize_fix_all_fakes`
  (owns canonical_equations/witness_dsl/preflight; DONE rc=0, 207-pass, receipt committed 7fb1b00eb1),
  `warmstart_organ_n1_rl` (finding banked to memory; code wire-ins held for provenance).
- Drain-detector: background job `boc8lb8gl` → notifies on 0 RUNNING. Monitor `bxdn0i47b` → DONE/fail lines.
- Pointer UNMOVED: 0.18804 borrowed / 0.19108 submittable.

## EXECUTION SEQUENCE (on drain, in order)
1. **Enumerate every arm's slice.** For each arm (live-now-drained + the 2 dead): read its final message
   (events log `:: <summary>` + `-o <label>.last.txt`) = its deliverable + surfaced follow-ons. Map its
   files via git diff. Build a per-arm file-ownership table (disjoint sets; flag any shared file for
   manual reconcile — that is where trample risk lives).
2. **Cherry-pick + land per-arm via the serializer, NO cross-arm mixing.** ONE
   `tools/subagent_commit_serializer.py --files <arm's disjoint set>` per arm, `--expected-content-sha256`
   POST-EDIT. Order least-entangled → most. A file touched by 2 arms = reconcile by hand (read both
   intents; combine; never let one commit clobber the other's edit). This IS "no trample."
3. **Held-wireins drain** (`.omx/research/held_wireins_for_provenance_drain_20260714.md`): after
   provenance's `canonical_equations/witness_dsl` slice lands, wire (a) Bregman squared-Hessian canonical-eq
   + DSL leg (`argmax_native_vjp_fidelity_v1` keeps the H^-1 solve, NO no-solve shortcut — memory
   dual_metric_no_solve_is_squared_hessian...), (b) Fable AMC `amc_perrow_tiered_code_bitalloc_v1` +
   `TieredCodeQATLever` DSL leg. Then DELETE that ledger.
4. **Governor fix — control-plane gauntlet BEFORE trust/land.** Verify against the 3 corner cases I
   injected (inbox `governor_measured_growth_fix.jsonl`): C1 pid-reuse identity-keying, C2 plateau-then-burst
   nonzero floor + MAX-over-subwindows, C3 relax ONLY where `_throttle_eligible` (own-leader) so the backstop
   covers every relaxed proc. Monotone-safety (only ever LOWER vs +25). Growing-proc REFUSE test uses a real
   trend fixture. If a corner case is unhandled → respawn/fix, do NOT land.
5. **Prune to ONE source of truth.** Dedupe: canonical_equations registry (no dup laws), witness_dsl levers
   (lever_registry.completeness — no orphan/dup), memory (MEMORY.md >17KB → cluster-compact via
   tools/cluster_summarize_memory_category.py, NOT hasty hand-prune), task-ledger re-disposition (500+ rows,
   mostly done — collapse). Drop stale/orphan artifacts with certify-or-block (provenance preserved).
6. **Full arxiv_scout sweep (operator-queued 2026-07-14).** Run `tools/arxiv_scout.py` across ALL 8
   cruxes (default flags; ~30s of API-politeness delays) → ranked discovery queue → surface to operator;
   each accepted row routes into PAPER_WARM_START on the clean post-consolidation tree. NO auto-launch.
7. **Permanent drift-gate (two-landing: fix + STRICT self-protect).** The drift class = concurrent arms
   pile uncommitted edits into the shared worktree → trample/signal-loss/non-reproducible-launch risk.
   Gate options (pick the structural one): (a) uncommitted-pile detector — Stop-hook/preflight WARNs when
   the working tree exceeds N files / M arms' worth without disentangle-commit; (b) strengthen the landing
   gate so a dead arm cannot be dispositioned `reviewed_committed` until its slice is actually committed;
   (c) prefer per-arm git-worktree isolation for future heavy multi-arm fan-out. Land the gate + tests +
   CLAUDE.md catalog row per "Bugs must be permanently fixed AND self-protected against."

## Follow-ons QUEUED — DO NOT LAUNCH (held until clean tree + governor fix + operator GO)
- **$0 n600 ISLAND-BIRTH DILATION-λ HYSTERESIS SWEEP (queued, from the Fork/continuation reframe).** The
  measurement-first gate that decides whether continuation earns its keep. Ramp dilation-λ (or seed-anneal)
  UP until the movable/lane island nucleates through frozen-SegNet at n600, then ramp BACK DOWN and record
  birth-fraction per class both directions. Outputs simultaneously: (a) the saddle-node critical λ*, (b)
  bistable-vs-monotone verdict (hysteresis loop present?), (c) whether the quasi-static assumption is even
  plausible. IF bistable → replace the #300/#323 GO-gate with a COMPUTED critical λ + principled
  fold-crossing (the Chan-Vese `λ_c=W_birth/(δ·A_GT_c)` balance law is the fixed-point; this pins the fold
  around it). IF monotone → continuation reframe has no operational value, say so + drop. NO model-build
  until this measures (the balance law + one operating point exist; the MISSING pieces are the transient/
  basin dynamics + this swept curve + time-scale-separation check). A MEANS: earns keep only via the run-
  config critical-λ it produces. Sisters: memory curriculum_is_continuation_instabilities_are_bifurcations,
  #300/#323 island-birth, #315 birth-completion ramp, Chan-Vese birth-balance canonical-eq, #318/#344.
- Witness training sweep (from `witness_train_sweep_spec` output) — dual-purpose (pointer candidates +
  costate-organ n=1 trajectory data).
- n600 pairkkt confirmation (banked; admits on correct accounting once governor lands).
- exp-linear weight reparam run (if its $0 2x2 smoke shows fewer-steps additive to Muon).
- The 4 other live arms' (curvelet/ripo/gauge/reparam/sweep) surfaced follow-ons — capture from their
  final messages into this list on drain.
- **PAPER (queued, not launched) arXiv 2602.18428 "The Geometry of Noise: Why Diffusion Models Don't Need
  Noise Conditioning" (Sahraee-Ardakan/Delbracio/Milanfar, Google, 2026-02).** SHARPEST hit of today's
  inputs for our live metric P0s. Mechanism: raw Marginal Energy E=-log p(u) has a **1/tᵖ singularity
  normal to the data manifold** (gradients diverge); the learned autonomous (time-invariant) field carries
  an implicit **local conformal metric that PERFECTLY CANCELS the singularity** → infinitely-deep well
  becomes a stable attractor; generation = Riemannian gradient flow on E. Plus a **Jensen-Gap / bounded-gain**
  result: velocity-based parameterizations are inherently stable, noise-prediction ones are high-gain
  unstable. FORK (PAPER_WARM_START): paper = diffusion on a data-manifold density w/ noise-marginalization;
  ours = coord-INR fitting a FROZEN-SCORER argmax partition (no diffusion/noise-conditioning). Transferable
  = the MATH not the application. 4 surfaces: (1) **#500 OPTIMAL-METRIC + #504 Bregman** — our d_seg lives
  on the codim-1 SEPARATRIX = a decision-landscape singularity where the margin field diverges; test whether
  a conformal factor cancels it and whether it's the SAME object #500/#504 derive ($0 derivation check).
  (2) **eikonal cure (#316/#318/#320)** — the ep110 divergence IS this boundary singularity; a
  conformal-metric flow could REPLACE adaptive-ε (singularity-free vs stabilized-singular). (3)
  **parameterization selection (#310/#497)** — bounded-gain → prefer SDF-velocity output over residual;
  explains hosc/fixed-β high-gain death. (4) **autonomous-flow-vs-SCHEDULE (#302, ties to Fork row below)**
  — "geometry encodes the conditioning, no schedule needed" → is the witness-native optimum an autonomous
  flow with the right metric, not a multi-stage τ/ε/λ schedule? (the autonomous-field concept the
  continuation reframe needs). A MEANS: earns keep via a measured eikonal-cure or a #500 metric-derivation
  row. Sisters: #500/#504/#316-#320/#302/#310/#497, Fork row, memory curriculum_is_continuation.
- **TOOL (queued, not launched) Fork Dynamics — forkdynamics.com (rendered via agent-browser; og:title
  verbatim "numerical bifurcation continuation tool for dynamical systems, web UI + CLI for maps and
  ODEs").** The AUTO/MatCont/BifurcationKit lineage: continue equilibria/limit-cycle branches as a
  parameter varies, detect+classify bifurcations (fold/Hopf/pitchfork/period-double). VERDICT — deepest
  conceptual fit of today's four (openresearch/weco/PTRM/Fork), but CONDITIONAL leverage, honest boundary:
  Fork eats LOW-dim explicit maps/ODEs; the witness is ~100K-param n=1 stochastic INR — you CANNOT feed it
  raw. It helps ONLY on a REDUCED-ORDER model we derive. The reframe it crystallizes (worth more than the
  tool): **our curriculum IS a continuation problem and our instabilities ARE bifurcations** — τ/ε/λ
  hand-offs = continuation parameters (#302); the ep110 eikonal re-entry = a stability-boundary crossing
  (#318 modified-eqn); **island-birth (movable/lane unborn, #300/#323) = a saddle-node/transcritical in the
  class-occupancy order parameter as dilation-λ/seed-anneal varies** → continuation would COMPUTE the exact
  critical λ the birth-lever must cross, replacing GO-gated trial-and-error with a bifurcation diagram; the
  costate/Pontryagin TPBVP (#247/#426) is an ODE system Fork could branch-analyze. NOT core infra (it's a
  MEANS, an analysis instrument like the dashboard — moves the pointer only via a better hand-off/birth
  point). Warm-start = derive the reduced order-parameter model (we have pieces: #318 DE, #344 linear-NCDE,
  #180 Morse-Smale) → run Fork's CLI on IT → get the bifurcation points. Sisters: #302/#315/#318/#344
  curriculum cluster, #300/#323 island-birth, #217 saddle-to-saddle, memory
  curriculum_is_continuation_instabilities_are_bifurcations_20260714.
- **PAPER (queued, not launched, per drain directive) arXiv 2607.11052 "Domain-Aware Scaling Laws
  Uncover Data Synergy" (Hamidieh/Mackey/Alvarez-Melis).** Warm-start (codex/opus, post-consolidation
  clean tree): data-synergy = which COMBINATIONS of trajectory "domains" give super-additive learning.
  Fork: paper = LLM pretraining / many domains / cross-model observational variation. Ours = the costate
  ORGAN (n=1 starved; organ memory already tracks 3 regimes: lane-erosion / mixed-Lane-Road /
  movable-island-unborn). CRUX: does domain-aware 2nd-order synergy predict which REGIME-MIX of witness
  trajectories cures the organ n=1 starvation super-additively → directly shapes the dual-purpose sweep's
  clip/lever/regime composition. Sisters: #434 synthetic-data SOTA, #481 continual-learning, #499
  n=1/low-data theory, the sweep_spec arm's dual-purpose mapping. (Witness-overfit is n=1 by design —
  this feeds the ORGAN data-curation, NOT the witness.)
- **PAPER (queued, not launched) arXiv 2607.07470 "A Theory of Contrastive Learning with Natural
  Images" (Torralba/Weiss).** Warm-start (codex/opus): the OPTIMAL contrastive representation = CNN with
  SINUSOIDAL first-layer filters whose FREQUENCIES come from a WATERFILLING algorithm on the dataset
  power spectrum + partial whitening. Fork: paper = contrastive SSL on stationary natural-image stats;
  ours = witness coord-INR fitting a frozen-SegNet argmax partition (n=1 clip, non-stationary). CRUX:
  does waterfill-on-the-BOUNDARY/along-tangent power-spectrum give the optimal witness Fourier-feature
  frequency allocation → directly attacks the measured 3.2× along-tangent deficit (#277/#497) and
  grounds curvelet/shearlet basis choice (#502). Waterfilling connects to our #157/#336 bit-alloc.
  Sisters: #25 BASIS-OVER-CREDITED, #497 alt-to-Fourier, #502 genuine curvelet/shearlet.
- **PAPER (queued, not launched) arXiv 2605.19943 PTRM — Probabilistic Tiny Recursive Model (Sghaier,
  Mila).** Mechanism: 7M TRM + SEEDED Gaussian noise at each deep-recursion step at inference → K
  parallel latent rollouts (diverse basins) → select via the model's Q-head (Sudoku-X 87.4→98.75%;
  91.2% vs 55.1% frontier at ~1e-4× cost). Warm-start fork, 3 surfaces: (1) STRONGEST → #396/#400
  MC-finisher: same shape (noise→parallel candidates→value-select) but WE select with the EXACT
  evaluator (no reward-model error); importable refinement = structured PER-DEPTH noise for basin
  diversity in the finisher proposal distribution. (2) costate organ n=1 ceiling: K seeded rollouts of
  the tiny organ = epistemic uncertainty (rollout spread = rec confidence) at ZERO added params —
  answers the measured bigger-nets-overfit wall. (3) decode-time K-rollout+stored-selection = already
  our selector-splice family (#48/R1/#399), validated not new; learned Q-head at inflate barred
  (no-scorers-at-inflate). Sisters: n1_organ_capacity memo, #426/#436, weco-AIDE² row above.
- **WECO AIDE² RSI (queued, not launched) — weco.ai/blog/first-evidence-of-recursive-self-improvement.**
  Verdict: their core mechanisms CONVERGE with ours (hidden-score-only selection = our exact-authority
  split, empirically dropping reward-hacking 63%→34% structurally; fixed $ budget = our ≤$20 cap;
  ~90% incumbent-rejection = our seal/A/B). IMPORTABLE DELTA: measured outer-loop selection for
  APPARATUS changes — build an INCIDENT-REPLAY BATTERY from harness_failure_ledger's 12 classes so
  harness edits get a measured accept/reject vs incumbent, not review-only (their 8-days-vs-2-years
  efficiency claim is the EV case). AIDE85 unreleased → watch-item. Sisters: #346 retrieval-first,
  #481 apparatus-as-CL-system, the codex CFL fix (today's apparatus edits = first battery candidates).

## Captured signal (no loss) — the click-polish run (#399 `_import` block-loop)
- STOPPED 2026-07-14 11:30 by **main stopping it under OPERATOR AUTHORIZATION** (after the safety
  classifier first denied the kill; operator then authorized) — **NOT jetsam/OOM** (my earlier
  forensic was a mis-attribution; corrected here so it is never re-forgotten). No crash marker in the
  log because it was a clean authorized stop.
- CONSEQUENCE: it is **NOT** a governor-OOM receipt — do not cite it as before/after evidence for the
  governor fix (that needs a genuinely memory-killed workload). Retracted.
- SUBSTANCE (intact, resumable): 52 blocks banked; best block-0 advisory **S=0.18791567** (−0.00013 vs
  the 0.18804 pointer, splice-verified seg_maxabs=0/pose_maxabs=0); sweep_state resumable at pass 129 /
  block 1; clicks_ledger intact; candidate staged **MODAL-HOLD**, NEVER exact-eval'd → pointer UNMOVED.
  Follow-on (operator-GO): resume the block-loop OR exact-eval the banked candidate. Capture the
  clicks_ledger + advisory candidate in the consolidation so a resumable sub-0.18804 advisory is not lost.

## APPARATUS BUGS diagnosed 2026-07-14 (operator "another bug or code smell") — OWED permanent fixes (two-landing)
Root cause of the stuck consolidation: the 2 "RUNNING" arms were pre-CFL `--sandbox workspace-write` arms
that STRUCTURALLY CANNOT COMMIT (blocks `.git/objects`) → doomed to strand regardless of runtime. Diagnosed:
1. **Retry inherits the pre-fix broken sandbox + NO RESUMABILITY** — ripo's codex died transient rc=1 after
   ~4h; the harness re-launched it FROM SCRATCH (pid 5481, 8min old under launcher 26184) STILL on
   workspace-write (up to 8×). FIX: retries must (a) use the current default sandbox (danger-full-access),
   (b) be resumable (not restart a 4h job from 0), OR (c) not auto-retry heavy long arms at all. + STRICT
   self-protect.
2. **In-flight pre-CFL arms not retrofitted or capped** — the CFL fix only helps NEW arms; nothing caps
   runtime or warns that a workspace-write arm will strand. FIX: a launch/liveness gate that flags/caps
   workspace-write arms as strand-doomed.
3. **Drain-detector TIMEOUT exits 0** — "DRAIN WAIT TIMEOUT: still 2 arms" surfaced as "completed (exit 0)"
   = a give-up looks like success (sibling of the _wt_test phantom + the codex_status-over-pgrep memory).
   FIX: timeout → nonzero exit + explicit TIMEOUT status. + reinforce: NEVER hand-roll pgrep for codex
   liveness (shared-pid false match — memory codex_fleet_liveness_use_status_tool_not_handrolled_pgrep);
   always tools/codex_status.py.
These are OWED (queued, not built now per consolidate-asap). Land at/after consolidation, two-landing each.

## ripo STOPPED (operator-GO 2026-07-14) + state
- `ripo_deep_warmstart_trust_region_500` STOPPED via killpg pgid 26184 (pgid-guarded, curvelet 85966
  untouched). Reason: doomed-to-strand (workspace-write) + no-resume retry loop wasting ~4h/cycle. Its
  written signal (log tail captured); its shared-tree code slice folds into the WHOLE-PILE consolidation
  (un-attributable per-arm — no manifest). Delegations + landing gate → terminal.
- Sibling `ripo_margin_fisher_seg_head_preconditioner_500` already REVIEWED rc=1 (errored, terminal).
- **Fleet now: 1 RUNNING = `genuine_curvelet_shearlet_build_measure` (#502 P0, ~6h).** Per operator: let it
  FINISH, then main-harvest its stranded diff (codex_harvest_commit --files — a stranded diff lands
  main-side, so finish≠loss). Consolidation (full sequence above) fires on curvelet DONE. Pointer 0.18804/0.19108.

## Invariants
NO signal loss (every arm final message + research memo + DAG FEED captured before any prune). NO trample
(per-arm serializer commits, shared files reconciled by hand). ONE SoT (DSL + canonical_equations + memory
+ DAG are the canonical four; prune duplicates INTO them). MPS never a score. Deterministic. Pointer honesty.

## Consolidation closure — 2026-07-14

The code/SoT drain landed in 18 reviewed serializer groups before the receipt commit:
`f43554bbf6`, `3c38aed572`, `4925036623`, `ae5b57525b`, `b70e51454e`, `7263015bd1`,
`b673d8fb85`, `4cf6be0bcd`, `d900c3cc1c`, `7be3499ca6`, `f8074d6e1c`, `84b5007a05`,
`5bacd6d5b1`, `cc7c02f78b`, `587ed7f98a`, `6bae3f73d3`, `cca5f1a1af`, and `0e9229ffd7`.

SoT reconciliation result: 701 canonical-equation events resolve to 332 unique current law IDs;
1,884 lane IDs validate with zero duplicates and all 24 additive lanes retained; 168 canonical task
rows pass strict validation with one registration per task. The current canonical-equations query and
lever registry load, with 80 unique lever factories and zero stale emitted flags.

Main-supervisor independently compared the landing with baseline `f8074d6e1c` and classified the
surfaced pytest failures as baseline debt. The only two pass-on-baseline flips are flaky PR85 real-smoke
probes whose subject code is byte-identical baseline-to-landing. Regression verdict: CLEAN versus that
baseline. This does not make the 12 baseline preflight-drift findings or the partial-suite failures green.

The three apparatus bugs above remain OWED: none was narrow enough for an honest `$0` strict
self-protect in this consolidation. The complete evidence and partial-suite counts are in
`.omx/research/consolidation_code_land_receipt_20260714.md`. Frontier pointer unchanged; no paid/GPU
launch occurred.

DAG FEED: `FEED-CONSOLIDATION-CODE-LAND-20260714` — 18 reviewed landings; equations/lane/task SoTs
reconciled; loaders verified; clean-versus-baseline regression verdict; baseline gate debt preserved;
three apparatus fixes OWED; pointer unchanged.
