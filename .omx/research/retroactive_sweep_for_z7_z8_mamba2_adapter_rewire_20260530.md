# Retroactive sweep for Z7+Z8 Mamba-2 canonical SSD adapter rewire

## 2026-05-30 (timestamp: 20260530T201500Z)

Per Catalog #348 RETROACTIVE VERDICT-TAINT SWEEP discipline. This memo
addresses the 4-field contract for the canonical apparatus mutation
wave landed by lane
`lane_z7_z8_mamba2_adapter_canonical_helper_rewire_20260530`.

NOTE: Catalog #348 strictly applies to NEW STRICT preflight gates added
to `src/tac/preflight.py`. This wave does NOT add a new STRICT gate; it
extends an existing canonical equation (`mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1`)
with a new EmpiricalAnchor + rewires 3 source files + adds 30 new dedicated
tests. This retroactive sweep memo is the SISTER DISCIPLINE applied
voluntarily so future agents have the same verdict-taint audit surface
for the consumer-rewire wave.

## 1. Bug-class symptom signature

What evidence pattern WOULD have invalidated past KILL/DEFER/FALSIFY
verdicts on Z7-Mamba-2 / Z8 hierarchical predictive coding if the
canonical Mamba-2 SSD tri-backend helper had been available + actually
consumed earlier?

* **Wave 4 Dao-Gu fidelity audit** (2026-05-29, landed memo
  `wave_4_z7_mamba_2_dao_gu_fidelity_audit_landed_20260529.md`): the audit
  classified the existing `_ReferenceMamba2Cell` as Mamba-1 (S6) reference
  with **PARADIGM-LEVEL INTACT** + **IMPLEMENTATION-LEVEL DOCUMENTED ADAPTATION**
  per Catalog #307. The audit registered 4 reactivation criteria for
  upgrading to true Mamba-2 SSD reference. **This rewire satisfies
  reactivation criterion #2**: "Operator decides to register
  `mamba_2_ssd_vs_s6_reference_cell_at_contest_scale_v1` as a canonical
  equation per Catalog #344 + run paired-comparison smoke at d_state=128
  + headdim=64 → 2-head SSD vs 2048-entry S6 at the contest scale". The
  canonical equation IS registered at
  `mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1` per the
  b2936fb81 canonical helper landing; the rewire enables the
  paired-comparison.

* **Z8 M11/M12 Modal T4 dispatch path** (`z8_phase_2_build_milestones`):
  the existing path was paid Modal T4 because Z8 had NO MLX-LOCAL route.
  Per CLAUDE.md 8th MLX-first standing directive, this is a structural
  cost-class violation that has been blocking $0 macOS local research.
  **This rewire unlocks the MLX-LOCAL path structurally** — Z8 M12a can
  now route through `tac.substrates._shared.mamba2_ssd.mlx_backend` at
  $0 vs $5-15 per Modal T4 dispatch.

* **Z7-Mamba-2 L2 long-training $0** unlocked by the same structural
  pattern. Per `z7_mamba_2_l1_empirical_mlx_fair_shake_landed_20260526.md`
  the Z7-Mamba-2 L1 MLX-LOCAL smoke landed but L2 long-training required
  paid CUDA because the reference_torch backend (Mamba-1 S6) cannot
  parallelize at the SSD chunk-scan scale. **This rewire enables L2
  long-training via the canonical helper's MLX backend at $0** (operator
  can opt-in via `backend='ssd_reference'` in the Z7-Mamba-2 substrate
  config).

## 2. Pre-fix window

What period is covered by the sweep?

* **2026-05-18** original Z7-Mamba-2 substrate design memo
  (`z7_mamba2_substrate_design_memo_20260518.md`) → **2026-05-30** Z7+Z8
  canonical helper rewire landing (12 days of substrate-engineering work).

* **2026-05-26** Z8 Phase 2 binding-first methodology directive (`z8-hierarchical-predictive-coding-binding-first-active-build-target-yousfi-grounded-20260529`)
  → **2026-05-30** Z7+Z8 canonical helper rewire landing (4 days of Z8
  Phase 2 milestone build per
  `tac.substrates.z8_hierarchical_predictive_coding.build_progress.Z8_PHASE_2_BUILD_MILESTONES`).

* **2026-05-30** canonical Mamba-2 SSD tri-backend helper landing
  (commit `b2936fb81`) → **2026-05-30** Z7+Z8 rewire landing (~24h gap
  between canonical helper landing + actual consumer rewire). Per the
  Mamba-2 MLX port subagent's HIGH op-routables #1 + #2: the rewire was
  queued the moment the canonical helper landed.

* Cumulative paid Modal/Lightning/Vast.ai spend during this window on
  Z7-Mamba-2 + Z8 M11/M12 dispatches: per the Catalog #245 Modal call_id
  ledger, ~$5-15 per dispatch wave × multiple waves = TBD by ledger query.

## 3. Historical KILL/DEFER/FALSIFY search results

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" +
Catalog #307 paradigm-vs-implementation falsification classification:

* **No KILLED verdicts on Z7-Mamba-2 or Z8 hierarchical predictive coding**.
  Both are active substrates with multiple landed milestones; the rewire
  EXTENDS them via a sister opt-in backend, does NOT replace or kill.

* **Wave 4 Dao-Gu fidelity audit classified `_ReferenceMamba2Cell` as
  IMPLEMENTATION-LEVEL DOCUMENTED ADAPTATION** (Mamba-1 S6 not true
  Mamba-2 SSD) → reactivation-path-2 was registered. **This rewire
  EXECUTES reactivation-path-2** (canonical equation registered + new
  canonical SSD reference backend opt-in landed). Per Catalog #307: NOT
  a kill; rather, the documented adaptation is now sister-canonical with
  a true Mamba-2 SSD path that did not exist before.

* **No FALSIFICATION verdicts** on existing canonical equation
  anchors. The original anchor (`mamba2_ssd_tri_backend_byte_stable_first_anchor_20260530_l8`)
  + the new anchor (`mamba2_ssd_z7_z8_canonical_consumer_chain_empirically_verified_20260530`)
  are sister-anchors (both PROCEED-confirming the canonical equation's
  predictions; both residual=0.0).

* **No DEFER verdicts invalidated**. The rewire is purely additive
  (sister opt-in backend); no historical evidence is invalidated.

## 4. Per-finding RE-EVAL-priority assignment

For each historical finding affected by this wave:

1. **Z7-Mamba-2 substrate `backend='auto'` default behavior**
   — RE-EVAL-priority: **LOW**. Default behavior unchanged
   (continues to fall back to `reference_torch` Mamba-1 S6 on macOS).
   Existing canonical equation anchors continue to cite the existing
   backend. No backfill required.

2. **Z8 M11 L1 macOS-CPU smoke + M12 paired-CUDA sub-0.189 threshold**
   — RE-EVAL-priority: **HIGH**. The rewire unlocks the MLX-LOCAL path
   that was structurally absent before. Operator-routable next step:
   re-run Z8 M11 L1 smoke with `use_canonical_ssd=True` opt-in via
   the Z8 adapter; compare per-tensor activations vs the existing
   reference_torch path to characterize the S6-vs-SSD divergence at
   contest scale. Sister memo to be landed by the next Z8 M11/M12
   wave subagent.

3. **Z7-Mamba-2 L2 long-training paid CUDA cost** — RE-EVAL-priority:
   **MEDIUM**. The rewire enables L2 long-training via the canonical
   helper's MLX backend at $0. Operator-routable next step: schedule
   a Z7-Mamba-2 L2 long-training MLX smoke and compare wall-clock +
   gradient-norm trajectory vs the reference_torch path. Cost-benefit:
   $0 macOS wall-clock ~5-15h vs $5-15 paid CUDA wall-clock ~2-4h. If
   the gradient trajectory matches, the MLX path is the new canonical
   for L2 long-training research; if not, the reference_torch path
   remains canonical for paid CUDA dispatches.

4. **`tac.optimization.mamba2_predictor.Mamba2Predictor` test suite
   (102 tests across 6 files)** — RE-EVAL-priority: **LOW**. All 102
   tests continue to pass post-rewire (verified empirically). The
   rewire is purely additive (new `SSD_REFERENCE_BACKEND` constant +
   new `ssd_nheads`/`ssd_headdim` config fields + new
   `_CanonicalHelperSSDCell` class). No existing test required
   modification.

## Cross-references

* Canonical helper landing: `b2936fb81` — Mamba-2 SSD MLX port
  tri-backend helper (commit message)
* Canonical equation: `mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1`
  per Catalog #344
* Catalog #371: canonical equations auto-recalibrator (will fire on
  the new EmpiricalAnchor per its `when_3+_new_empirical_anchors_in_domain`
  trigger; currently 2 anchors so trigger not yet satisfied)
* Catalog #348: this retroactive sweep memo (THIS file)
* Wave 4 Dao-Gu fidelity audit: `wave_4_z7_mamba_2_dao_gu_fidelity_audit_landed_20260529.md`
* Z8 binding-first methodology: `z8-hierarchical-predictive-coding-binding-first-active-build-target-yousfi-grounded-20260529`
* CLAUDE.md "8th MLX-first" standing directive
* CLAUDE.md "Forbidden empirical-claim-without-evidence-tag"
* CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L8

## Verdict

**PROCEED**. No historical KILL/DEFER/FALSIFY verdicts were invalidated
by this wave. The rewire is purely additive — it executes Wave 4
Dao-Gu fidelity audit reactivation-path-2 (registering the canonical
equation + actually consuming the canonical helper) without invalidating
any existing milestone evidence. Future subagents should consider the
canonical SSD backend as the structural unlock for Z8 M12a MLX-LOCAL +
Z7-Mamba-2 L2 long-training $0 paths.
