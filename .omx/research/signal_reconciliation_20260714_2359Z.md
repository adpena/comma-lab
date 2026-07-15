# SIGNAL RECONCILIATION — durable capture of in-flight + chat-only routing (2026-07-14/15)

**Why:** operator directive "Ensure no signal loss / from the last four days." This ledger moves CHAT-ONLY
routing decisions + in-flight arm state into a durable, compaction-surviving artifact (operating-manual:
"a chat-only insight is a lost insight"). Pointer UNMOVED 0.18804 borrowed / 0.19108 submittable — all below
is MEANS.

## A. IN-FLIGHT (2 codex + 2 opus, disjoint domains, worktree-isolated) — resumable if this session dies
- **codex `apparatus_3bug_twoland_fixes`** (worktree branch `codexwt/apparatus_3bug...`): 3 apparatus-bug
  two-landing fixes — retry-inherits-broken-sandbox · in-flight-non-isolated-writer-cap · drain-timeout-
  false-stuck (each: fix + STRICT gate + tests). Deliverable = commits on its branch; MAIN reviews+merges.
- **codex `held_wireins_bregman_amc_v2`** (worktree): Bregman squared-Hessian canonical-eq + no-solve-fake
  guard on `argmax_native_vjp_fidelity_v1` (commit-1); AMC canonical-eq `amc_perrow_tiered_code_bitalloc_v1`
  advisory (commit-2); AMC `TieredCodeQATLever` kept OWED (needs non-existent trainer flags; pays only at a
  competitive checkpoint). v1 correctly BLOCKED (NO-FAKE) → respawned as v2.
- **opus waterfilling probe** (`aab46737b3d23ab66`): Torralba-Weiss Eq-4 boundary-spectrum $0 numpy probe →
  grounds P0 #502. Writes `waterfill_boundary_spectrum_curvelet_vs_fourier_probe_20260714.md`.
- **opus #499 organ n=1 theory** — LANDED (see §C).

## B. LEVER ROUTING from the 3 landed research memos (was chat-only → now durable)
- **Torralba-Weiss 2607.07470 → LEVER → #502/#497/#277/#25.** Theorem PROVES isotropic Fourier optimal only
  under global stationarity (our oriented non-stationary boundary violates it) → optimality points at
  curvelets/shearlets. Transferable = Eq-4 waterfilling on generalized eigenvalues. $0 probe RUNNING (§A).
- **Fork Dynamics (MIT, github.com/hinsley/Fork) → LEVER → bifurcation cluster (#302/#315/#318/#344/#300/
  #323/#217).** Confirms+extends the continuation reframe (memory `curriculum_is_continuation_...`). $0 probe
  = fit movable-occupancy normal form to an existing run, check fold-coincidence, continue in λ. VALUE GATED
  on deriving the reduced-order model, NOT the tool. → WAVE-2 (island-birth + Fork normal-form measurement).
- **Geometry of Noise 2602.18428 → DOMINATED-bookmark → #268** (metric-preconditioned boundary descent
  re-derives our measured margin=Fisher L1; no lever). memo f1da11541d.
- **data-synergy 2607.11052 → DOMINATED → #434/#499** (post-sweep analysis only; can't cure n=1).
- **PTRM 2605.19943 → DOMINATED → #247/curriculum** (already better-framed by bifurcation+costate).
- **WECO AIDE² RSI → DOMINATED → apparatus #247**; ONE adoptable idea = reward-hacking-rate observability.
- **GenCeption 2607.09024 + Looped-TTT → DOMINATED → organ-data-eff #434/#499 + costate-econ #454/#426/#342.**

## C. #499 ORGAN n=1 REFRAME (LANDED 63466c740b) — measurement-discipline CORRECTION, route now
- MEASURED: `U_hierarchical_physics_residual` already backtested (`warmstart_organ_n1_rl_backtest_20260714.json`):
  WINS aggregate MAE 0.002496 (vs persistence 0.002792 / ridge 0.003902) but LOSES **per-class** MAE 0.0559
  vs persistence **0.0108** (5.2× worse). GP-T aggregate 0.001852 also aggregate-only.
- **THE CORRECTION: the aggregate-MAE race is already won by physics-prior means; NO data-driven arm beats
  persistence's per-class 0.0108. The organ's job is the per-LEVER λ field → PER-CLASS, not aggregate, is
  the real gate.** The n=9 ceiling ("updates prior, can't identify a free field") = the per-class collapse.
  PAC-Bayes: admissible `KL ≲ 2n·ε²` ≈ a handful of scalar DOF at n=9.
- **3 $0 probes queued (all on the existing 9 intervals / 7 WF folds — must beat persistence per-class 0.0108):**
  1. per-lever partial-pooling disambiguation of U (block-precision partial pooling + P/Q ensemble).
  2. GP-with-physics-mean + per-lever coregionalized residual (tinygp derivative-obs kernel, MIT).
  3. PAC-Bayes/MDL capacity governor as an ADMISSION GATE (extincts the capacity-reflex).
  - Allergic flag: meta-learning / #434 distillation / GRU-DeepONet ALL need n≫10 — gated, not next.
- Memory `n1_organ_capacity_ceiling_shrinkage_physics_residual_measured_20260714` MUST be updated with the
  per-class-is-the-gate correction (aggregate win is banked but non-load-bearing for the organ's job).

## D. WAVE-2 QUEUE (durable; fires as codex/opus slots free)
1. **island-birth dilation-λ hysteresis sweep + Fork normal-form fit** (converged $0 n600 measurement) → codex slot.
2. **#499 organ $0 probes 1-3** (per-class partial-pooling / GP-physics-mean / PAC-Bayes gate) → opus/codex.
3. **#502 curvelet build** — GATED on the waterfilling probe's measured capacity-gain verdict.
4. Comprehensive last-4-days signal-loss audit (see the dispatched audit arm).

## E. HELD for operator-GO (CONTAINMENT): witness training sweep (heavier/potential-paid).
## F. OWED apparatus (being fixed by codex apparatus arm): the 3 bugs above.
