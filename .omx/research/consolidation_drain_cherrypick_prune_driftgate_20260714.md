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
6. **Permanent drift-gate (two-landing: fix + STRICT self-protect).** The drift class = concurrent arms
   pile uncommitted edits into the shared worktree → trample/signal-loss/non-reproducible-launch risk.
   Gate options (pick the structural one): (a) uncommitted-pile detector — Stop-hook/preflight WARNs when
   the working tree exceeds N files / M arms' worth without disentangle-commit; (b) strengthen the landing
   gate so a dead arm cannot be dispositioned `reviewed_committed` until its slice is actually committed;
   (c) prefer per-arm git-worktree isolation for future heavy multi-arm fan-out. Land the gate + tests +
   CLAUDE.md catalog row per "Bugs must be permanently fixed AND self-protected against."

## Follow-ons QUEUED — DO NOT LAUNCH (held until clean tree + governor fix + operator GO)
- Witness training sweep (from `witness_train_sweep_spec` output) — dual-purpose (pointer candidates +
  costate-organ n=1 trajectory data).
- n600 pairkkt confirmation (banked; admits on correct accounting once governor lands).
- exp-linear weight reparam run (if its $0 2x2 smoke shows fewer-steps additive to Muon).
- The 4 other live arms' (curvelet/ripo/gauge/reparam/sweep) surfaced follow-ons — capture from their
  final messages into this list on drain.
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

## Invariants
NO signal loss (every arm final message + research memo + DAG FEED captured before any prune). NO trample
(per-arm serializer commits, shared files reconciled by hand). ONE SoT (DSL + canonical_equations + memory
+ DAG are the canonical four; prune duplicates INTO them). MPS never a score. Deterministic. Pointer honesty.
