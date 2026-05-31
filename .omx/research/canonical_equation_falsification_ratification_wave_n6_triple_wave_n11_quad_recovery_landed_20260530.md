# Canonical equation falsification ratification — Wave N+11 QUAD HALT anchor (LANDED 2026-05-30)

## Frontmatter

```yaml
lane_id: lane_canonical_equation_falsification_ratification_wave_n6_triple_wave_n11_quad_recovery_20260530
mission_predicted_contribution: apparatus_maintenance
horizon_class: frontier_pursuit
council_tier: T1
status: LANDED
wave: "N+6/N+11 ratification (recovery respawn)"
recovery: true
predecessor_subagent_id: a7d1b8762ba4b661a
dispatch_fired: false
paid_spend_usd: 0.0
canonical_equation_referenced: triple_substrate_composition_orthogonal_pose_axis_savings_v1
```

## HONEST scope correction (Catalog #229 premise verification + Catalog #287)

The recovery prompt's premise — that the Wave N+6 TRIPLE empirical FALSIFICATION
is a "deferred ratification step" — is **FALSIFIED-AT-PREMISE**. Premise
verification BEFORE editing (Catalog #229) found that the Wave N+6 TRIPLE
empirical FALSIFICATION was **ALREADY RATIFIED on 2026-05-28** by a sister
subagent as anchor
`wave_n6_triple_z6_v2_nscs06_v8_compound_c_paired_cuda_cpu_empirical_falsification_20260528`
(empirical 92.4795 paired CUDA / 92.4762 paired CPU; residual 92.320194; verdict
`IMPLEMENTATION_LEVEL_FALSIFICATION_COMPOUND_C_RENDERER_NOT_POSENET_RECOGNIZABLE`;
`CONTEST_ARCHIVE_MEMBER` provenance; `score_claim_valid=True`).

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: **the already-ratified
Wave N+6 falsification was NOT duplicated and NOT mutated.** This lane therefore
lands ONLY the genuinely-missing Wave N+11 QUAD HALT anchor + the canonical
documentation, tests, lane, probe outcome, and retroactive sweep.

This is the HONEST corrected execution of the deferred-ratification step named in
the Wave N+11 HALT landing memo (lines 75-77): the deferred portion that was NOT
yet present in the registry is the Wave N+11 QUAD HALT anchor itself.

## Recovery note (Catalog #206)

Predecessor `a7d1b8762ba4b661a` hit a session limit. Parent verified NO checkpoint
was written → predecessor completed no landing. Fresh respawn; no signal lost.

## What landed (registry BEFORE/AFTER)

| anchor | residual | empirical_verification_status | event_type | measurement_axis | score_claim_valid |
|---|---|---|---|---|---|
| wave_n6_..._empirical_falsification_20260528 (PRE-EXISTING) | 92.320194 | (null; pre-#363) | (pre-existing `anchor_appended`) | `[contest-CUDA]` | True |
| wave_n11_quad_halt_phantom_provenance_pre_check_failed (NEW) | 0.0 | ASSUMED_AWAITING_VERIFICATION | `anchor_appended` | `[macOS-MLX research-signal]` | False |

- Registry total equation count: 198 → 198 (the Wave N+11 row is an anchor event
  on an EXISTING equation, NOT a new equation registration).
- Target equation anchor count: 3 → 4.
- The Wave N+6 92.320194 falsification residual remains in
  `predicted_vs_empirical_residual` after the HALT append (the HALT's residual=0.0
  does NOT erase it).

The Wave N+11 HALT anchor used the canonical `update_equation_with_empirical_anchor`
helper (the registry's only anchor-append entry point per the verified API). Its
`provenance` is a real `tac.provenance.Provenance` object built via
`build_provenance_for_research_sidecar(...)` (NOT a dict) — `evidence_grade=RESEARCH_ONLY`,
`score_claim_valid=False`, `promotion_eligible=False`, `[macOS-MLX research-signal]`
axis. `empirical_output.contest_score=None` + `dispatch_fired=False` +
`paid_spend_usd=0.0` (no measurement; phantom-provenance gate refused dispatch).
`residual=0.0` because the dataclass refuses NaN/negative residual; the HALT signal
is carried by `empirical_verification_status="ASSUMED_AWAITING_VERIFICATION"`.

The append was performed by an idempotent self-guarding script (skips if the
anchor_id is already present) so re-running it is safe per APPEND-ONLY discipline.

## Catalog #307 paradigm-vs-implementation classification

**IMPLEMENTATION-LEVEL FALSIFICATION** (Compound C renderer-specific; inherited by
the QUAD from the already-ratified Wave N+6 TRIPLE). The multi-substrate
composition **paradigm is INTACT**:
- The Wave N+6 482× miss (predicted −0.036 → empirical 92.48) was caused by the
  Compound C renderer NOT producing PoseNet-recognizable frames (PoseNet=162.52 vs
  frontier ~0.01).
- The Wave N+11 QUAD adds Z7-Mamba-2 on top of 3-of-4 substrates that ARE the
  falsified Wave N+6 TRIPLE set, so the phantom-provenance pre-check (Catalog
  #321/#322) correctly HALTED before any paid dispatch.
- Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this is
  DEFER-pending-renderer-rescope, **NOT KILL**.

## Catalog #371 auto-recalibrator trigger status (HONEST)

The auto-recalibrator ran (`equations_checked=198`) but did **NOT** recalibrate
the target (`equations_recalibrated=0`; target NOT in the recalibrated set). HONEST
reason: the target's `predicted_vs_empirical_residual` summary already reflected the
Wave N+6 92.320194 falsification before this lane; the Wave N+11 HALT anchor's
`residual=0.0` did not change the dominant residual, so the recalibrator correctly
found nothing new to refit. No spurious recalibration was forced — that would be a
Catalog #371 orphan-stub anti-pattern.

## Catalog #321/#322/#323 phantom-provenance verification

The NEW Wave N+11 HALT anchor carries a canonical `Provenance` via
`build_provenance_for_research_sidecar(...)` with `score_claim_valid=False` +
`promotion_eligible=False` + `evidence_grade=RESEARCH_ONLY`. The production-registry
regression guard test asserts this. No phantom-provenance pollution.

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

Pinned in the HALT anchor's `reactivation_criteria` (stored in the Provenance
`rejection_reason` field per the canonical research-sidecar builder):

- **Path A**: standalone paired-CUDA RATIFICATION of each of the 4 QUAD substrates
  per Catalog #246 ($6-8 envelope) for real per-substrate anchors BEFORE re-composing.
- **Path B**: Wave N+12 BUILD re-routing Z6-v2 pose-axis through a PR101-class
  HNeRV-family validated renderer (NOT Compound C).
- **Path C**: defer the QUAD composition until ≥2 of the 4 substrates have standalone
  [contest-CUDA] anchors.

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map: N/A (ratification of a HALT anchor; no new per-axis
  sensitivity contribution)
- hook #2 Pareto constraint: N/A
- hook #3 bit-allocator: N/A
- hook #4 cathedral autopilot dispatch: ACTIVE — the canonical equation's residual
  (already 92.320194 from the Wave N+6 falsification) + the new HALT anchor flow to
  `tools/cathedral_autopilot_autonomous_loop` (a registered consumer) + the
  `canonical_equation_lookup_consumer`, so the ranker sees both the
  IMPLEMENTATION-LEVEL falsification and the QUAD HALT and discounts Compound C
  composition candidates.
- hook #5 continual-learning posterior: ACTIVE — the Wave N+11 HALT anchor IS a
  continual-learning anchor; Catalog #371 auto-recalibration was triggered (no
  refit needed, HONEST).
- hook #6 probe-disambiguator: ACTIVE — the canonical equation residual + the HALT
  anchor IS the disambiguator between "QUAD composition predicted [-0.04, -0.07]"
  (registration) and "QUAD composition HALTED at phantom-provenance, $0 spend"
  (this lane).

## Mission contribution (Catalog #300)

`apparatus_maintenance` — closes the genuinely-missing Wave N+11 QUAD HALT anchor
so the canonical equation registry stays coherent with the landed HALT evidence.
The already-ratified Wave N+6 falsification was honestly verified present (not
duplicated). Future agents inherit the QUAD HALT + reactivation paths rather than
re-discovering the composition trap via another paid dispatch.

## NO FAKE IMPLEMENTATIONS (Slot EEE 5 forbidden classes)

- Class 1 (markers-without-work): the Wave N+11 HALT anchor cites ACTUAL evidence
  (the Wave N+11 HALT memo; $0 dispatch; phantom-provenance pre-check) — not
  synthetic. The Wave N+6 falsification (92.48) was verified ALREADY-RATIFIED, not
  re-fabricated. ✓
- Class 2 (tests-verify-constants): the 17 dedicated tests verify the ACTUAL HALT
  anchor round-trip (via `update_equation_with_empirical_anchor` against an isolated
  tmp_path registry) + Catalog #371 recalibrator-runs-without-error + no-duplication
  + APPEND-ONLY + the production-registry regression guard asserts the real landed
  anchors — not constants. ✓
- Class 3 (placeholder-in-data-field): the HALT anchor `measurement_method` +
  `reactivation_criteria` are substantive per Catalog #287 — not placeholders. ✓
- Class 4 (synthetic-fixture): the HALT cites the real Wave N+11 HALT memo; the
  production-registry regression guard asserts the real landed anchors. The
  Provenance is a REAL `tac.provenance.Provenance` object (not a dict). ✓
- Class 5 (enum-padding): the paradigm-vs-implementation classification is HONEST
  IMPLEMENTATION-LEVEL per Catalog #307 — not a structurally-padded enum. ✓

## Sister-DISJOINT scope (Catalog #340)

This lane touched ONLY `.omx/state/canonical_equations_registry.jsonl` (via the
canonical helper) + the new test file + memos + lane registry + probe outcome.
DISJOINT vs concurrent recovery subagents: D3+D4 (CLAUDE.md), Wave N+12 Path B
(.omx/research + experiments/results), PR110-OPT-11 L0→L1
(src/tac/substrates/pr110_opt11).

## Cross-references

- `.omx/research/wave_n11_quad_composition_sub015_cascade_halt_phantom_provenance_pre_check_failed_landed_20260530.md` (the Wave N+11 HALT source)
- `.omx/research/wave_n6_triple_paired_cuda_ratification_corrected_archive_implementation_falsified_20260528.md` (the already-ratified Wave N+6 falsification)
- `.omx/research/retroactive_sweep_for_canonical_equation_falsification_ratification_wave_n6_triple_wave_n11_quad_recovery_20260530.md` (Catalog #348 sweep)
- canonical equation `triple_substrate_composition_orthogonal_pose_axis_savings_v1`
- Catalog #307 + #344 + #371 + #321/#322/#323 + #246 + #313 + #110/#113
