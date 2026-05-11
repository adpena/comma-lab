# Lane registry stale-entry sweep — 2026-05-11

**Scope:** comprehensive sweep of `.omx/state/lane_registry.json` (348 lanes
post-add) beyond `lane_a_landed`'s alias-only audit landed in the 2026-05-11
punchlist cleanup. Per X audit follow-up + operator directive 2026-05-11
("keep pushing the compiler and wiring and integration and everything").

## Detection criteria

Four signal classes audited:

1. **F1**: L0 lanes with 0 gates marked (SKETCH per pre-registration discipline)
2. **F2**: stored `level` field disagrees with computed-from-gates level
3. **F3**: lanes whose notes carry "KILLED" / "FALSIFIED" / "DEFERRED" verdict
   without explicit `reactivation_criteria` (per CLAUDE.md "KILL/FALSIFIED
   memory verdicts" non-negotiable + "premature KILL without research
   exhaustion" forbidden pattern)
4. **F4**: lanes with `score_claim=true` text and no `[contest-CUDA]` or
   `[contest-CPU]` evidence in any gate string

## Results

| Finding | Count | Status |
|---|---|---|
| **F1** L0 lanes with 0 gates | 163 | EXPECTED — pre-registration discipline (L0 = SKETCH per CLAUDE.md "Lane maturity registry" lifecycle) |
| **F2** stored level vs computed mismatch | 0 | CLEAN — Check 90 strict preflight already enforces this |
| **F3** verdict-without-reactivation | 1 (after de-noise) | **FIXED in this landing** |
| **F4** score_claim without evidence path | 0 | CLEAN |

## F1 detail (163 L0 lanes)

These are CORRECT per CLAUDE.md "Lane maturity registry" non-negotiable:
"Pre-registration is mandatory. The moment a lane has a name and a council/
design verdict — even if it's only a sketch — it MUST be `add-lane`'d at
Level 0." 163 SKETCHes is consistent with active-substrate-portfolio scope
across NeRV/HNeRV/Cool-Chic/C3/wavelet/VQ-VAE/grayscale-LUT/SIREN/
coordinate-MLP/hyperprior/Ballé/SC++/MDL-FP4/foveation/RAFT-pose/LAPose
families enumerated by parallel subagents 2026-05-11.

**No action.** The L0 SKETCH pool is the lane-maturity discipline working
correctly.

## F2 detail (0 mismatches)

Check 90 (`check_lane_registry_consistent`) is STRICT @ 0 violations
already; the level invariant is enforced at commit time.

**No action.** STRICT preflight already covers this bug class.

## F3 detail (1 real finding after de-noise)

Initial keyword search caught 10 lanes with "KILL" / "DEFERRED" / "FALSIFIED"
in their notes. After per-lane inspection (because the keyword can appear
incidentally referencing OTHER lanes' state, not the lane's own verdict),
**only 1 lane was a true KILL-without-reactivation-criteria verdict**:

| Lane | Original notes | Reframed |
|---|---|---|
| `lane_gp_rerun` | "KILLED per Council #271. Gap: PROPER REPLACEMENT needed (B-spline / DCT / non-polynomial fit) at user's standard." | "DEFERRED-pending-research-with-proper-replacement (reframed 2026-05-11 per CLAUDE.md kill-as-last-resort non-negotiable; original verdict 'KILLED per Council #271' converted). Gap: a non-polynomial fit (B-spline / DCT / wavelet basis / Gaussian-process) at user's standard for residual modeling. Reactivation criteria: (a) a non-polynomial basis (B-spline / DCT / wavelet / GP) is implemented and (b) shows positive Δ score on a smoke dispatch with the GP rerun substrate. Council #271 reference preserved for forensic context." |

The other 9 false-positives (`lane_pfp16`, `lane_line_search_pose_refinement`,
`lane_comprehensive_adversarial_review_2026_05_09`,
`lane_check_125_backfill_and_production_hardening_polish`,
`lane_codex_round78_findings_fix`,
`lane_rust_packet_compiler_native_port_scaffold`,
`lane_pose_axis_telescopic_foveation_field_full_scaffold`,
`lane_pose_axis_raft_pose_stream_full_scaffold`,
`lane_pose_axis_lapose_motion_atom_allocator_full_scaffold`) all reference
"deferred" or "downgraded" in passing — either citing OTHER lanes' state OR
citing their OWN reactivation gate. Each was inspected; none required the
KILL-to-DEFERRED reframe pattern.

**Specifically `lane_pfp16`**: notes already include the explicit
reactivation gate ("until contest_cpu evidence is recorded") — false-positive
on the keyword search.

## F4 detail (0 findings)

No lane carries `score_claim=true` text in its notes without an authoritative
evidence path. Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag"
this is the canonical clean state.

**No action.**

## Comparison vs X's 2026-05-11 punchlist cleanup

X's `feedback_punchlist_cleanup_landed_20260511.md` re-tagged 19 lanes with
the explicit `score_claim=false / promotion_eligible=false /
ready_for_exact_eval_dispatch=false` trio AND reframed 3 historical
FALSIFIED lanes:

| Lane | X's reframe |
|---|---|
| `lane_owv3_0120_arith_masks` | DEFERRED-pending-research-with-AMRC-faithful-renderer |
| `lane_apogee_int7` | DEFERRED-pending-research-with-byte-aligned-int7-packer |
| `lane_pysr_cmaes_symbolic_regression` | DEFERRED-pending-research-with-different-surrogate-class-or-direct-empirical-search |

This sweep extends X's audit by catching the **single remaining KILL-verdict
lane** (`lane_gp_rerun`) that escaped X's enumeration — likely because the
verdict was Phase 1 work pre-dating X's 2026-05-11 sweep batch.

## Tooling verification

```
$ .venv/bin/python tools/lane_maturity.py validate
OK — 348 lane(s) validated cleanly.

$ .venv/bin/python tools/lane_maturity.py set-field lane_gp_rerun --field notes --value "DEFERRED-pending-research-with-proper-replacement ..."
OK — lane_gp_rerun.notes = "DEFERRED-pending-research-with-proper-replacement ..."
```

## Outstanding gaps (not action items for this sweep)

- **F1 SKETCH pool is large (163 lanes).** This is the lane-pre-registration
  discipline working as designed, BUT it makes the audit table noisy. Future
  improvement: a `--phase >= N` filter for `tools/lane_maturity.py audit`
  could narrow the table to in-flight + landed lanes only. Logged here for
  ledger continuity; NOT in scope for this sweep.

## CLAUDE.md compliance

- **KILL/FALSIFIED memory verdicts** non-negotiable: 1 KILL verdict (without
  reactivation criteria) reframed to DEFERRED-pending-research with
  reactivation criteria documented. 0 NEW KILLs issued.
- **Premature KILL without research exhaustion** forbidden pattern: closed
  via the reframe.
- **Forbidden empirical-claim-without-evidence-tag**: F4 confirms 0
  violations; the 19 re-tagged lanes from X's punchlist remain compliant.
- **Lane maturity registry** non-negotiable: 348 lanes validate via
  `tools/lane_maturity.py validate`; Check 90 STRICT covers the level
  invariant.

## Cross-references

- X's punchlist cleanup: `feedback_punchlist_cleanup_landed_20260511.md`
- Lane registry: `.omx/state/lane_registry.json` (348 lanes)
- Lane maturity audit log: `.omx/state/lane_maturity_audit.log` (JSONL)
- CLAUDE.md sections: "KILL/FALSIFIED memory verdicts", "Lane maturity
  registry", "Forbidden patterns" (kill-as-last-resort)
