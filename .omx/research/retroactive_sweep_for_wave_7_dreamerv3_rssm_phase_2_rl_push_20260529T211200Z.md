# Retroactive sweep for Wave 7 DreamerV3 RSSM Phase 2 RL push (no new Catalog #)

Per Catalog #348 4-field contract for new-gate-or-canonical-mutation landings.
This wave does NOT land a new Catalog # gate; it lands a canonical equation
2nd anchor + canonical Provenance + council deliberation + probe outcome.
Per Catalog #348 sister discipline the canonical equation anchor landing is
treated like a gate landing for retroactive sweep purposes — the new evidence
may invalidate historical KILL/DEFER/FALSIFY verdicts that depended on the
prior 1-anchor state.

## Field 1 — Bug class symptom signature

The bug class this wave's anchor most directly extincts is:

**Symptom**: canonical equation `categorical_posterior_capacity_vs_continuous_gaussian_v1`
held only the closed-form math identity anchor (1 anchor) with no
trained-logits existence proof. Downstream consumers (cathedral autopilot
ranker, Dykstra Pareto solver per Catalog #372, canonical_equation_lookup_consumer
per Catalog #335) could only consume the closed-form prediction; the
prediction was NOT EMPIRICALLY VERIFIED to hold under gradient training.

A historical KILL/DEFER verdict relying on the proposition "the DreamerV3
RSSM categorical posterior cannot be trained" would now be falsified by
the Wave 7 trained-logits anchor (27.6x MSE reduction in 30 epochs on real
contest video).

## Field 2 — Pre-fix window

The pre-Wave-7 window is from 2026-05-20 (canonical equation derivation
landing per `.omx/research/dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z.md`)
through 2026-05-29 (Wave 3 audit landing + canonical first anchor) up to
this wave's commit boundary.

## Field 3 — Historical-KILL/DEFER/FALSIFY search results

Search executed:

```
grep -lE "(dreamer_v3_rssm|dreamerv3.rssm|categorical[-_]posterior).*(KILL|FALSIFIED|DEFER)" .omx/research/*.md
```

Result: 0 historical KILL or FALSIFIED verdicts on this substrate.
Existing DEFERRED verdicts:

- `.omx/research/wave_3_dreamerv3_rssm_math_audit_landed_20260529.md` Wave 3
  op-routable #5 (Path B2 PyTorch port + Modal smoke): **THIS WAVE CLOSES**
  the deferred op-routable via MLX-LOCAL surrogate per Catalog #1265
  contest-equivalence gate.
- `.omx/research/wave_3_dreamerv3_rssm_math_audit_landed_20260529.md` Wave 3
  symposium op-routable #1a ((G, K) sweep probe): **DEFERRED-PENDING-NEXT-WAVE**
  (queued per Wave 7 reactivation criterion #2; no priority change required).
- `.omx/research/wave_3_dreamerv3_rssm_math_audit_landed_20260529.md` Wave 3
  op-routable #6 (KL balancing + free bits at L1+): **UNCHANGED** (still
  N/A at L0; would require GRU + dynamics prior split).
- T3 grand-council symposium 2026-05-19 op-routable #6 (paired Modal CUDA +
  Linux x86_64 CPU on real archive bytes): **DEFERRED-PENDING-PYTORCH-PORT**
  (Wave 7 reactivation criterion #3 + #4 enumerate the canonical path).

## Field 4 — Per-finding RE-EVAL-priority assignment

| Historical finding | New evidence | RE-EVAL-priority |
|---|---|---|
| Wave 3 op-routable #5 (PyTorch port + Modal smoke for trained-logits anchor) | CLOSED via MLX-LOCAL surrogate per Catalog #1265 | RESOLVED (no re-eval needed; MLX-LOCAL surrogate IS the canonical close per CLAUDE.md "MLX portable-local-substrate authority") |
| Wave 3 symposium op-routable #1a ((G, K) sweep probe) | NEW TRAINED-LOGITS ANCHOR available as baseline for sweep comparison | MEDIUM (proceed when MLX-LOCAL bandwidth is available; ~$0 cost; ~10 min M5 Max wall-clock) |
| T3 symposium op-routable #6 (paired Modal CUDA + Linux x86_64 CPU) | EXISTENCE PROOF strengthens the PyTorch port readiness case; canonical equation now has 2 of 3 anchors needed for Catalog #371 auto-recalibration | HIGH (third anchor enables canonical equation auto-recalibration + cathedral autopilot ranking inclusion) |
| Wave 3 op-routable #6 (KL balancing + free bits at L1+) | UNCHANGED (still N/A at L0; requires GRU + dynamics prior split) | LOW (L1+ extension; not blocking) |
| All Catalog #229/#290/#294/#296/#303/#305/#325 design-memo discipline gates | per-substrate symposium memo landed for THIS wave per Catalog #325 | RESOLVED (canonical landing satisfies all gates) |

No historical KILL/FALSIFY verdicts require re-evaluation; the Wave 7 anchor
is additive (existence proof) and does not falsify any prior verdict.

## Mission contribution

`frontier_breaking_enabler`: the empirical existence proof unblocks the
PyTorch port + paired Modal dispatch reactivation path; without the existence
proof, the PyTorch port would be premature per CLAUDE.md "Substrate MUST be
at OPTIMAL FORM before paid empirical dispatch" + Catalog #315.

## Cross-references

- Wave 7 landing memo: `.omx/research/wave_7_dreamerv3_rssm_phase_2_rl_push_landed_20260529.md`
- Wave 7 per-substrate symposium: `.omx/research/dreamerv3_rssm_per_substrate_symposium_wave_7_phase_2_20260529.md`
- Wave 3 landing memo: `.omx/research/wave_3_dreamerv3_rssm_math_audit_landed_20260529.md`
- Canonical equation registry: `.omx/state/canonical_equations_registry.jsonl`
- Council deliberation posterior: `.omx/state/council_deliberation_posterior.jsonl`
- Probe outcomes ledger: `.omx/state/probe_outcomes.jsonl`
