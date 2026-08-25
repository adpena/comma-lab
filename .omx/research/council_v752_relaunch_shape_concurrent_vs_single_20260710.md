---
council_tier: T2
council_attendees: [Shannon, Dykstra, Contrarian, Assumption-Adversary, Yousfi, Fridrich, Boyd]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "smoke-verify the micro-batch twin is bit-identical at n600 BEFORE trusting the speedups in a pointer run; don't assume the batched path == the serial accum path."
council_assumption_adversary_verdict:
  - assumption: "the two top decision-table rungs (HorizonWeightedMargin 43.8%, StepNativeActivation 31.6%) HELP d_seg"
    classification: CARGO-CULTED
    rationale: "rel_sig is a floor-aware ESTIMATE (proxy), both rungs are NEVER-FIRED; the 'fold them in' framing inherits the table's proxy ranking as if measured. Their OWN fire criteria demand an A/B."
  - assumption: "DsegAwareTaper's EV is SETTLED (either the table's 73% OR #121's WEAK)"
    classification: CARGO-CULTED
    # MAGNITUDE_DISMISSAL_OK: NOT a dismissal — the OPPOSITE. The taper is ranked #1
    # (73% of the remaining 0.19108->0.15 descent, i.e. rel_sig already IS the relative-
    # significance number, ~0.030 S if realized). The Assumption-Adversary is flagging an
    # UNRECONCILED CONTRADICTION (table 73% vs #121 WEAK), not concluding it's small.
    # Neither verdict is trusted-until-MEASURED; the resolution is op-routable #4 (measure
    # the taper A/B), which is measure-don't-dismiss, not magnitude-based orphaning.
    rationale: "UNRECONCILED: table rel_sig 73% (~0.030 S of the 0.0411 remaining gap) CONTRADICTS #121's WEAK downgrade (frontier saliency 5.5x FLAT — a PROXY-saliency read, never a byte-closed n600 A/B). Neither is a measured through-R verdict → resolve by MEASUREMENT (op-routable #4), do not trust either estimate."
# Catalog #300 v2-frontmatter backfill 2026-08-25: council_decisions_recorded transcribed
# VERBATIM from this memo's own section "Verdict (the either/or dissolves)" items 1-4 +
# the recorded Contrarian revision. Frontmatter-only addition per the CLAUDE.md
# "Council hierarchy" backward-compatibility clause (NO body mutation).
council_decisions_recorded:
  - "First relaunch = CLEAN BASELINE arm — sealed v7.5.2 config + the FREE score-neutral speedups (micro-batch-pairs + verdict-batch bump: headroom spent on wall-clock only, bit-identical) + #408 telemetry. NOT blindly-folded rungs."
  - "Pass the readiness gate HONESTLY via its defer hatch: `# LAUNCH_READINESS_DEFER:HorizonWeightedMargin=terminal-band A/B fork per its converged-n600 fire criterion` + the same for StepNativeActivation. Deferred-with-reason, not skipped."
  - "MEASURE the two rungs as A/B warm-start forks from the converged baseline — SEQUENTIALLY now (pre-#173), promoted to CONCURRENT once #173's 2 HIGH clear (on the fork, not the from-scratch run)."
  - "Resolve the DsegAwareTaper 73%-vs-#121-WEAK contradiction (a #405 table-freshness bug) before trusting the table ranking further."
  - "Revision (Contrarian): bit-identity-smoke the micro-batch twin at n600 BEFORE it rides the pointer run."
---

# T2 — v7.5.2 relaunch shape: concurrent A/B arms vs single updated arm

**STORES CONSULTED:** `.omx/research/default_off_decision_table_20260710.jsonl` (the #405 table — fire-now
rungs + rel_sig + each rung's own fire criterion) · `.omx/research/n205_full_run_risk_register_watchlist_20260702.md`
(D9 concurrent-arm memory/scheduling risk) · `tools/witness_memory_preflight.py` (self-orient ON vs OFF
peak: 67.6 vs 24.5 GiB) · task #173 (fleet-scaling blockers) · #121 (taper WEAK downgrade) · #313/#261/#293
(micro-batch twin) · CLAUDE.md §Council conduct + §Council hierarchy T2 + the confound self-protection
non-negotiable · memory `curriculum_candidate_pool_p0_orphan_class_20260710` + `launch_readiness_gate_config_freshness_naive_launch_20260710`.

**Decision:** given the readiness gate now forces the relaunch config to consume the #405 table, and
self-orient OFF freed ~65 GiB, what shape is the relaunch? Options as posed: (A) single updated arm
(fold HorizonWeightedMargin + StepNativeActivation into one config); (B) concurrent A/B arms (exploit
headroom, measure each rung in parallel).

## Positions

- **Shannon (LEAD):** rel_sig is a PROXY for through-R n600 ΔS. Both top rungs' OWN table fire criteria
  are A/Bs ("converged n600 byte-close A/B"; "n600 adopt-verdict vs sealed hosc"). Folding blind = the
  surrogate-as-authority error (NO-FAKE #8). They must be MEASURED, not assumed. → against (A).
- **Dykstra (CO-LEAD):** concurrency is memory-feasible (2×24.5=49 < 89.6 ceiling) BUT the achievable set
  is {memory-feasible ∩ #173-cleared}, and #173's 2 HIGH + 4 MED blockers are UN-cleared → that set is
  currently EMPTY. Concurrent-A/B is INFEASIBLE-NOW; do not launch a 2nd arm into un-cleared blockers
  (the machine-crash class the governor exists to prevent). → (B) DEFERRED-pending-#173.
- **Contrarian:** the bottleneck is d_seg CONVERGENCE, not throughput. Don't let a shiny 65 GiB
  resource-unlock DRIVE the launch shape (resource-availability-as-strategy = lazy consensus). And (A)
  conflates: fold both → can't attribute which rung moved d_seg, learn nothing if they antagonize.
- **Assumption-Adversary:** VETO on any fold-without-measure (see frontmatter). The rungs' EV is
  cargo-culted until an A/B measures it; the taper #1-vs-#121 contradiction is a table-freshness bug.
- **Yousfi + Fridrich:** A/B = MATCHED CONTROL (confound-clean, 12-philosophies matchedctl). CRUX: the
  rungs' fire criteria are TERMINAL-BAND ("converged n600") — they are NOT from-scratch config knobs,
  they are decisions made ON a converged checkpoint. Natural + cheapest place to A/B them = warm-start
  forks from the converged baseline.
- **Boyd (grand):** seconds Dykstra — clear #173 before adding the concurrency dimension.

## Verdict (the either/or dissolves)

Both top rungs' own fire criteria say they are TERMINAL-BAND A/B decisions, not from-scratch config
choices. Therefore neither (A) nor (B)-now:

1. **First relaunch = CLEAN BASELINE arm** — sealed v7.5.2 config + the FREE score-neutral speedups
   (micro-batch-pairs + verdict-batch bump: headroom spent on wall-clock only, bit-identical) + #408
   telemetry. NOT blindly-folded rungs.
2. **Passes the readiness gate HONESTLY via its defer hatch:**
   `# LAUNCH_READINESS_DEFER:HorizonWeightedMargin=terminal-band A/B fork per its converged-n600 fire
   criterion` + same for StepNativeActivation. Deferred-with-reason, not skipped — exactly what the hatch
   is for.
3. **The two rungs are MEASURED as A/B warm-start forks from the converged baseline** — SEQUENTIALLY now
   (pre-#173), promoted to CONCURRENT once #173's 2 HIGH clear (headroom makes concurrent the eventual
   win, on the fork not the from-scratch run).
4. **Resolve the DsegAwareTaper 73%-vs-#121-WEAK contradiction** before trusting the table ranking
   further — a #405 table-freshness bug (feeds the "enforce updating" half of the operator directive).

Revisions (Contrarian): bit-identity-smoke the micro-batch twin at n600 before it rides the pointer run.

**Means:** pointer 0.19108282 [contest-CPU] UNMOVED; this is the launch-shape decision, not a row.
