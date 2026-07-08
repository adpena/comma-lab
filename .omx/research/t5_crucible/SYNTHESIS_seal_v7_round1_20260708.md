# SEAL v7 ROUND-1 SYNTHESIS (2026-07-08) — 4 lenses NOT_CLEAN, dispositions, round-2 plan

STORES CONSULTED: seal_v7_r1_bugs (cc66ce473) · seal_v7_r1_deepmath (bdb79ff9c) ·
seal_v7_r1_confound (06330c3b4) · seal_v7_r1_structure_blind (697dad238 + dd0026e36) ·
self_paced_tau_advance memo · trainer L1439/3909 (adaptive-ε build state verified) ·
curriculum_dsl:2427 (DirectionalBasisRebalance lever verified) · run-1 launch.sh (basis
allocation ground truth) · L25/L65 memory (3.2× along deficit, basis-before-capacity law) ·
#320 task state. review_status: synthesis by main orchestrator, fresh-eyes on all 4 reports.
Pointer 0.19110 UNMOVED.

## TALLY: NOT_CLEAN ×4 (counter = 0). Findings: 1 BLOCKER · 4 MAJOR · 8 MINOR · 7 REVISE.

## DISPOSITIONS
**IN-FLIGHT FIXES (3 fixers, all Opus):**
- BLOCKER (v7 unlaunchable: argparse + derive_named_config silent fall-through) + bugs-MAJOR
  (manifest protocol → b0.6 must VERIFY not warn) + bugs-MINOR (max-dwell governance row) →
  launch-path fixer. Acceptance = full dry-run gate chain passing on name-resolved v7.
- deep-math MAJOR-1 (stop_marginal_s 1e-4 vs derived s* = ν·forfeit = 6.897e-6) → ν·forfeit
  law REGISTERED + LawRef wired (provenance flips HARDCODED→DERIVED); MAJOR-2 (TAIL turnpike
  semantics) → honest dual-mode docs + dead-knob telemetry note. Bonus: MAJOR-2 as-built
  RESOLVES structure R-4 (no saw-tooth exists; TAIL = turnpike extension = the blind
  derivation's own preference).
- confound MAJOR-1 (event-muon fire-epoch not persisted → crash-resume break) → persistence
  fixer (+ dwell_at_cap MINOR folded).

**STRUCTURE REVISE dispositions:**
- R-1 (freq-along 4 STARVED; blind √32≈6, dash comb 26) → **PROPOSED CONFIG CHANGE, the round-2
  headline**: this is a triple-convergence — the measured 3.2× along-tangent deficit (L65) +
  the blind derivation's independent √32 + the crucible's OWN Arm-A mission
  (DirectionalBasisRebalance, DSL lever built at curriculum_dsl:2427, never fired). v7 still
  carries the starved across-32/along-4. PROPOSAL: enable DirectionalBasisRebalance
  (lane_offloaded regime) OR minimally rebalance along 4→6-8 within the bank budget —
  MEMORY-WATERFILLED first (bank size scales the cf cache; the #294 waterfill decides the
  exact allocation), reviewed in round 2. NOT enabling the crucible's own named basis lever
  would need a reasoned exclusion; none exists on record.
- R-2 (mod-dim 32 vs Whitney) → NO config change; the mod-dim telemetry (landed da2bd441e)
  now measures effective rank continuously; truncate-at-export D18 captures the rate side.
- R-3 (adaptive-ε "unbuilt") → FINDING CORRECTED: BUILT (#320, trainer 1439/3909,
  default-off). Enabling = check #320's A/B verdict first; round-2 item, not a blind flip.
- R-4 → RESOLVED (see deep-math MAJOR-2).
- R-5 (event vs clock) → OPERATOR DECISION, below.
- R-6 (β-end 10 vs blind 1→4) → KEEP 10 (v6-sealed measured anchor; the blind's 1→4 was the
  pre-anneal-fix era value; annealed-β divergence evidence supports the higher end under
  annealing). Provenance note added.
- R-7 (minor unbuilt finishers) → v7.1 ledger.

## THE MODE DECISION (to the operator, before launch)
THREE independent Opus recommendations now converge on **--tau-advance-mode clock for run-1**
(the τ-advance builder's memo · confound lens 4-step chain · structure R-5), all reasoning:
one unproven variable at a time — unify-L_τ is the load-bearing change; event-τ couples 3
schedules to a never-run sensor and confounds attribution; the flip to event for run-2 is one
token; lane/chroma sensor calibration accrues in BOTH modes via would-fire telemetry. The
operator's event directive is honored as SEQUENCING (run-2), not refused. Event-muon resume
(confound MAJOR-1) is being fixed regardless, so event mode is fully hardened for run-2
either way. RECOMMENDATION: clock run-1 · event run-2. Operator may override to event run-1;
MAJOR-1 fix + caps make it safe, just less attributable.

## ROUND-2 PLAN
Fixers land → basis decision (waterfill + round-2 review of the rebalance) → v7.3 compile
(name-resolved, dry-run gate chain green) → ROUND 2: same 4 lenses on the diff + regression
surfaces. Counter target: 3 consecutive clean. Then: operator mode decision surfaced with the
launch package → governed stop of run-1 → relaunch.
